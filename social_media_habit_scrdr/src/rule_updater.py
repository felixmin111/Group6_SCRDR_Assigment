"""
rule_updater.py
---------------
Incremental Learning for SCRDR

When the SCRDR engine makes an incorrect prediction, a new exception rule
is added under the last fired rule. This is the core SCRDR learning mechanism:

  - No existing rules are ever MODIFIED
  - A new exception is APPENDED at the deepest matching point
  - The rule tree grows incrementally

This mirrors the original RDR (Ripple Down Rules) philosophy:
  "Fix the knowledge base by adding, not replacing."
"""

import json
import os
from typing import Optional
from scrdr_engine import Rule, load_rules_from_json, evaluate_scrdr, match_condition


def find_last_fired_rule(
    input_case: dict,
    rules: list[Rule]
) -> Optional[Rule]:
    """
    Find the deepest rule that fired (matched) during SCRDR evaluation.
    This is the rule under which the new exception will be attached.
    """
    for root_rule in rules:
        if match_condition(input_case, root_rule):
            last_matched = root_rule
            current_exception = root_rule.exception

            while current_exception is not None:
                if match_condition(input_case, current_exception):
                    last_matched = current_exception
                    current_exception = current_exception.exception
                else:
                    break

            return last_matched

    return rules[0]  # Fallback to root


def add_exception_rule(
    last_fired: Rule,
    new_conditions: dict,
    new_conclusion: str,
    new_justification: str,
    new_rule_id: Optional[str] = None
) -> Rule:
    """
    Add a new exception rule under the last fired rule.

    SCRDR Principle: The new rule is added as the exception child
    of the last rule that matched, so it only fires when the parent
    matches AND the new conditions also match.

    Args:
      last_fired       : The Rule object to attach the exception under
      new_conditions   : Conditions dict for the new exception
      new_conclusion   : The correct label for the exception case
      new_justification: Human-readable explanation for the rule
      new_rule_id      : Optional custom ID (auto-generated if None)

    Returns:
      The newly created Rule object
    """
    if new_rule_id is None:
        new_rule_id = f"{last_fired.id}_ex"

    new_rule = Rule(
        id=new_rule_id,
        conditions=new_conditions,
        conclusion=new_conclusion,
        justification=new_justification,
        exception=None
    )

    # If there is already an exception, chain the new rule AFTER it
    if last_fired.exception is None:
        last_fired.exception = new_rule
    else:
        # Find the end of the current exception chain and append there
        current = last_fired.exception
        while current.exception is not None:
            current = current.exception
        current.exception = new_rule

    print(f"[RuleUpdater] Added exception rule '{new_rule_id}' under '{last_fired.id}'")
    print(f"             Conclusion: {new_conclusion}")
    print(f"             Conditions: {new_conditions}")

    return new_rule


def save_rules_to_json(rules: list[Rule], filepath: str) -> None:
    """
    Persist the updated rule tree back to JSON.
    Only the first root rule (with nested exceptions) is saved.
    Multiple root rules are stored as exception2, exception3, etc.
    """
    root_dict = rules[0].to_dict()

    for i, rule in enumerate(rules[1:], start=2):
        root_dict[f"exception{i}"] = rule.to_dict()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(root_dict, f, indent=2)

    print(f"[RuleUpdater] Rules saved to {filepath}")


def correct_misclassification(
    input_case: dict,
    true_label: str,
    rules: list[Rule],
    rules_filepath: str,
    expert_conditions: Optional[dict] = None
) -> Rule:
    """
    Full incremental learning workflow:
      1. Identify the last fired rule
      2. Construct differentiating conditions
      3. Add exception rule
      4. Save updated rules

    Args:
      input_case        : The misclassified input
      true_label        : The correct classification
      rules             : Current list of root Rule objects
      rules_filepath    : Path to save updated JSON
      expert_conditions : Optional manually specified conditions.
                          If None, auto-generates from input features.

    Returns:
      The newly added Rule
    """
    last_fired = find_last_fired_rule(input_case, rules)
    new_rule_id = f"{last_fired.id}_ex{_count_exceptions(last_fired)+1}"

    if expert_conditions is None:
        expert_conditions = _auto_generate_conditions(input_case, last_fired)

    justification = (
        f"Exception added after misclassification: "
        f"When {_conditions_to_text(expert_conditions)}, "
        f"the correct label is '{true_label}', not '{last_fired.conclusion}'."
    )

    new_rule = add_exception_rule(
        last_fired=last_fired,
        new_conditions=expert_conditions,
        new_conclusion=true_label,
        new_justification=justification,
        new_rule_id=new_rule_id
    )

    save_rules_to_json(rules, rules_filepath)
    return new_rule


def _count_exceptions(rule: Rule) -> int:
    """Count how many exception levels already exist under this rule."""
    count = 0
    current = rule.exception
    while current is not None:
        count += 1
        current = current.exception
    return count


def _auto_generate_conditions(input_case: dict, last_fired: Rule) -> dict:
    """
    Auto-generate differentiating conditions from the misclassified case.
    Uses numeric features with exact thresholds and categorical equality.
    """
    conditions = {}
    numeric_features = ["daily_usage_hours"]

    for feat, val in input_case.items():
        if feat in ["id", "label"]:
            continue
        if feat in numeric_features:
            conditions[feat] = {"op": "<=", "value": float(val)}
        else:
            conditions[feat] = {"op": "==", "value": str(val)}

    return conditions


def _conditions_to_text(conditions: dict) -> str:
    """Convert conditions dict to human-readable string."""
    parts = []
    for feat, constraint in conditions.items():
        op = constraint["op"]
        val = constraint["value"]
        parts.append(f"{feat} {op} {val}")
    return " AND ".join(parts)
