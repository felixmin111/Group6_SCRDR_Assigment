"""
scrdr_engine.py
---------------
Core Single Classification Ripple Down Rules (SCRDR) Engine.

SCRDR Evaluation Semantics:
  - Rules form a tree. Each rule has zero or more exception children.
  - Evaluation begins at the root rule.
  - At any node:
      * Check whether the node's conditions match the input record.
      * If they match  → this rule "fires"; descend into its exception list.
      * If they don't  → stop; the PREVIOUS firing rule's conclusion stands.
  - Only ONE path through the tree is ever traversed (single-path guarantee).
  - Existing rules are NEVER modified; new knowledge is encoded as exceptions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """
    One node in the SCRDR tree.

    Attributes
    ----------
    rule_id     : Unique identifier (e.g. "R0", "R1.2").
    conditions  : Feature-value pairs that must ALL match for the rule to fire.
                  An empty dict means the rule always fires (root / default).
    conclusion  : The class label returned when this rule is the last to fire.
    justification: Human-readable explanation stored with the rule.
    exceptions  : Ordered list of child rules evaluated when THIS rule fires.
    """
    rule_id: str
    conditions: Dict[str, str]
    conclusion: str
    justification: str
    exceptions: List["Rule"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Condition matching
# ---------------------------------------------------------------------------

def match_condition(input_record: Dict[str, str], rule: Rule) -> bool:
    """
    Return True iff every feature-value pair in rule.conditions is present
    and equal in input_record.

    An empty conditions dict (root rule) always matches.

    Parameters
    ----------
    input_record : Dict mapping feature names → categorical values.
    rule         : The Rule whose conditions we test.

    Returns
    -------
    bool
    """
    for feature, required_value in rule.conditions.items():
        if input_record.get(feature) != required_value:
            return False
    return True


# ---------------------------------------------------------------------------
# SCRDR evaluation  (single-path traversal)
# ---------------------------------------------------------------------------

def evaluate_scrdr(
    input_record: Dict[str, str],
    root_rule: Rule,
) -> Tuple[str, List[Rule]]:
    """
    Traverse the SCRDR tree from root_rule and return the winning conclusion.

    Algorithm
    ---------
    1. Start at root_rule.
    2. If root_rule matches → it becomes the "last fired" rule and we look at
       its exception list.
    3. For each exception (in order):
         - If the exception matches → it becomes the new "last fired" rule and
           we descend into ITS exceptions (recursively / iteratively).
         - If the exception does NOT match → skip it and try the next sibling.
    4. When no exception matches (or there are none), return the conclusion of
       the last fired rule.

    Note: The root rule always matches (empty conditions), so there is always
    at least one "last fired" rule.

    Parameters
    ----------
    input_record : Feature dict for one customer record.
    root_rule    : The root of the SCRDR tree.

    Returns
    -------
    (conclusion, path)
        conclusion : Predicted label string.
        path       : List of Rule objects that fired, in order.
    """
    path: List[Rule] = []

    # Root must match (it is the default / catch-all)
    if not match_condition(input_record, root_rule):
        # Defensive: root with non-empty conditions — treat as no match
        return "Unknown", path

    current_rule: Rule = root_rule
    path.append(current_rule)

    # Walk down the single exception chain
    while current_rule.exceptions:
        fired_exception: Optional[Rule] = None
        for exc in current_rule.exceptions:
            if match_condition(input_record, exc):
                fired_exception = exc
                break  # first matching exception wins

        if fired_exception is None:
            break  # no exception fires → current_rule is final

        current_rule = fired_exception
        path.append(current_rule)

    return current_rule.conclusion, path


# ---------------------------------------------------------------------------
# Decision path explanation
# ---------------------------------------------------------------------------

def trace_decision_path(
    input_record: Dict[str, str],
    root_rule: Rule,
) -> Dict:
    """
    Run evaluate_scrdr and return a structured explanation dict.

    Returns
    -------
    {
        "prediction"  : str,
        "path_ids"    : [rule_id, ...],
        "explanation" : "narrative string",
        "steps"       : [{"rule_id": ..., "conclusion": ..., "justification": ...}, ...]
    }
    """
    conclusion, path = evaluate_scrdr(input_record, root_rule)

    steps = [
        {
            "rule_id": r.rule_id,
            "conclusion": r.conclusion,
            "justification": r.justification,
            "conditions_matched": r.conditions,
        }
        for r in path
    ]

    # Build a human-readable narrative
    narrative_parts = []
    for i, step in enumerate(steps):
        if i == 0:
            narrative_parts.append(
                f"[{step['rule_id']}] Default rule fired → '{step['conclusion']}'"
            )
        else:
            cond_str = ", ".join(
                f"{k}={v}" for k, v in step["conditions_matched"].items()
            )
            narrative_parts.append(
                f"[{step['rule_id']}] Exception overrides → '{step['conclusion']}' "
                f"because ({cond_str})"
            )

    explanation = " ➜ ".join(narrative_parts)

    return {
        "prediction": conclusion,
        "path_ids": [r.rule_id for r in path],
        "explanation": explanation,
        "steps": steps,
    }
