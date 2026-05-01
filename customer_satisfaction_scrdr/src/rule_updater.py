"""
rule_updater.py
---------------
Incremental rule learning for the SCRDR engine.

Core SCRDR Principle
--------------------
When the engine misclassifies a record, we do NOT modify any existing rule.
Instead, we add a NEW exception rule under the last rule that fired.  The new
exception's conditions are derived from the misclassified record's features so
that it specifically covers this case and overrides the incorrect conclusion.

This preserves the integrity of the existing knowledge base while encoding
new, more refined knowledge as exceptions.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import copy

from scrdr_engine import Rule, evaluate_scrdr, match_condition


# ---------------------------------------------------------------------------
# Exception rule creation
# ---------------------------------------------------------------------------

def _generate_rule_id(parent: Rule, existing_exceptions: List[Rule]) -> str:
    """
    Generate a new rule ID that is a child of parent.
    E.g. if parent is "R1" and it already has "R1.1", "R1.2", return "R1.3".
    """
    n = len(existing_exceptions) + 1
    return f"{parent.rule_id}.{n}"


def build_exception_rule(
    parent_rule: Rule,
    input_record: Dict[str, str],
    correct_label: str,
    feature_keys: Optional[List[str]] = None,
) -> Rule:
    """
    Construct a new exception Rule to correct a misclassification.

    Strategy
    --------
    The new exception uses ALL features from the misclassified record as
    conditions (minus the label column).  This is the most specific rule that
    will exactly match the offending record.  In a production SCRDR system a
    domain expert would manually prune conditions; here we automate it.

    Parameters
    ----------
    parent_rule   : The last rule that fired (where the exception is appended).
    input_record  : The misclassified record (features only, no label).
    correct_label : The ground-truth label the new rule should produce.
    feature_keys  : Optional subset of features to include in conditions.
                    If None, all features in input_record are used.

    Returns
    -------
    A new Rule object (not yet attached to the tree).
    """
    rule_id = _generate_rule_id(parent_rule, parent_rule.exceptions)

    conditions: Dict[str, str] = {}
    keys = feature_keys if feature_keys else list(input_record.keys())
    for k in keys:
        if k in input_record:
            conditions[k] = input_record[k]

    justification = (
        f"Exception added by incremental learning: "
        f"correct label is '{correct_label}' for record with conditions "
        + ", ".join(f"{k}={v}" for k, v in conditions.items())
    )

    return Rule(
        rule_id=rule_id,
        conditions=conditions,
        conclusion=correct_label,
        justification=justification,
        exceptions=[],
    )


# ---------------------------------------------------------------------------
# Update: attach exception to the tree
# ---------------------------------------------------------------------------

def add_exception(
    parent_rule: Rule,
    new_exception: Rule,
) -> None:
    """
    Attach new_exception as the LAST child of parent_rule.exceptions.

    SCRDR constraint: exceptions are checked in order; later (more specific)
    exceptions are appended at the end so earlier general patterns still apply.

    NOTE: This mutates parent_rule in place.  Because Python objects are
    passed by reference, the original tree is updated automatically.
    """
    parent_rule.exceptions.append(new_exception)


# ---------------------------------------------------------------------------
# High-level: handle one misclassification
# ---------------------------------------------------------------------------

def correct_misclassification(
    input_record: Dict[str, str],
    correct_label: str,
    root_rule: Rule,
    verbose: bool = True,
) -> Optional[Rule]:
    """
    Detect and correct a single misclassification by adding an exception.

    Steps
    -----
    1. Run SCRDR to find the path of fired rules.
    2. Check whether the prediction is actually wrong.
    3. If wrong, build an exception under the LAST fired rule.
    4. Attach the exception (mutates the tree).
    5. Optionally print a log message.

    Parameters
    ----------
    input_record  : Feature dict (no label key).
    correct_label : The true class label.
    root_rule     : Root of the current SCRDR tree (mutated in place).
    verbose       : Print update messages.

    Returns
    -------
    The newly added Rule, or None if no correction was needed.
    """
    predicted, path = evaluate_scrdr(input_record, root_rule)

    if predicted == correct_label:
        if verbose:
            print(f"[rule_updater] Prediction '{predicted}' is already correct. No update needed.")
        return None

    last_fired_rule = path[-1]
    new_rule = build_exception_rule(
        parent_rule=last_fired_rule,
        input_record=input_record,
        correct_label=correct_label,
    )
    add_exception(last_fired_rule, new_rule)

    if verbose:
        print(
            f"[rule_updater] Misclassification corrected:\n"
            f"  Predicted : {predicted}\n"
            f"  Correct   : {correct_label}\n"
            f"  Last fired: {last_fired_rule.rule_id}\n"
            f"  New rule  : {new_rule.rule_id} added as exception under {last_fired_rule.rule_id}"
        )

    return new_rule


# ---------------------------------------------------------------------------
# Batch incremental learning pass
# ---------------------------------------------------------------------------

def incremental_learning_pass(
    records: List[Dict[str, str]],
    root_rule: Rule,
    label_key: str = "satisfaction",
    verbose: bool = True,
) -> Dict:
    """
    Perform one pass of incremental learning over a list of labelled records.

    For each record:
      - Predict using current rule tree.
      - If wrong, add an exception.

    Parameters
    ----------
    records   : List of dicts that include the label_key column.
    root_rule : SCRDR tree root (mutated in place).
    label_key : Name of the ground-truth column.
    verbose   : Print progress.

    Returns
    -------
    {
        "rules_added"  : int,
        "corrections"  : [{"record": ..., "was": ..., "now": ...}, ...],
    }
    """
    rules_added = 0
    corrections = []

    for record in records:
        input_features = {k: v for k, v in record.items() if k != label_key}
        correct_label = record.get(label_key)
        if not correct_label:
            continue

        new_rule = correct_misclassification(
            input_record=input_features,
            correct_label=correct_label,
            root_rule=root_rule,
            verbose=verbose,
        )

        if new_rule is not None:
            rules_added += 1
            predicted_before = evaluate_scrdr(input_features, root_rule)[0]
            corrections.append({
                "record": input_features,
                "was": predicted_before,  # after correction this re-evaluates to correct
                "now": correct_label,
                "new_rule_id": new_rule.rule_id,
            })

    return {
        "rules_added": rules_added,
        "corrections": corrections,
    }


# ---------------------------------------------------------------------------
# Serialisation helpers (shallow, for inspection)
# ---------------------------------------------------------------------------

def rule_tree_to_dict(rule: Rule) -> Dict:
    """
    Recursively convert a Rule tree to a plain dict (for JSON serialisation).
    """
    return {
        "rule_id": rule.rule_id,
        "conditions": rule.conditions,
        "conclusion": rule.conclusion,
        "justification": rule.justification,
        "exceptions": [rule_tree_to_dict(e) for e in rule.exceptions],
    }


def count_rules(rule: Rule) -> int:
    """Return total number of rules in the tree (including root)."""
    return 1 + sum(count_rules(e) for e in rule.exceptions)
