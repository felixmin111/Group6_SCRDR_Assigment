"""
utils.py
--------
Utility functions: JSON serialisation of the rule tree and pretty-printing.
"""

from __future__ import annotations
import json
from typing import Dict

from scrdr_engine import Rule


# ---------------------------------------------------------------------------
# JSON  ↔  Rule tree
# ---------------------------------------------------------------------------

def dict_to_rule(d: Dict) -> Rule:
    """
    Recursively convert a plain dict (loaded from JSON) into a Rule tree.

    Parameters
    ----------
    d : Dict with keys rule_id, conditions, conclusion, justification, exceptions.

    Returns
    -------
    Rule object.
    """
    return Rule(
        rule_id=d["rule_id"],
        conditions=d.get("conditions", {}),
        conclusion=d["conclusion"],
        justification=d.get("justification", ""),
        exceptions=[dict_to_rule(e) for e in d.get("exceptions", [])],
    )


def rule_to_dict(rule: Rule) -> Dict:
    """
    Recursively convert a Rule tree into a plain dict suitable for JSON.
    """
    return {
        "rule_id": rule.rule_id,
        "conditions": rule.conditions,
        "conclusion": rule.conclusion,
        "justification": rule.justification,
        "exceptions": [rule_to_dict(e) for e in rule.exceptions],
    }


def load_rules_from_json(filepath: str) -> Rule:
    """
    Load an SCRDR rule tree from a JSON file.

    Parameters
    ----------
    filepath : Path to the JSON file.

    Returns
    -------
    Root Rule object.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_rule(data)


def save_rules_to_json(root_rule: Rule, filepath: str) -> None:
    """
    Persist the current rule tree to a JSON file.

    Parameters
    ----------
    root_rule : Root of the SCRDR tree.
    filepath  : Destination path.
    """
    data = rule_to_dict(root_rule)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[utils] Rule tree saved to {filepath}")


# ---------------------------------------------------------------------------
# Tree visualisation
# ---------------------------------------------------------------------------

def print_rule_tree(rule: Rule, indent: int = 0) -> None:
    """
    Print a human-readable tree of all rules.

    Example output
    --------------
    R0  [default] → Neutral
      R1  [tone=positive, issue_resolved=yes] → Satisfied
        R1.1  [... , response_time=slow] → Neutral
          R1.1.1  [...] → Satisfied
      R2  [tone=negative, issue_resolved=no] → Dissatisfied
    """
    prefix = "  " * indent
    cond_str = (
        ", ".join(f"{k}={v}" for k, v in rule.conditions.items())
        if rule.conditions
        else "default"
    )
    print(f"{prefix}{rule.rule_id}  [{cond_str}] → {rule.conclusion}")
    for exc in rule.exceptions:
        print_rule_tree(exc, indent + 1)


def count_rules(rule: Rule) -> int:
    """Return the total number of rules in the tree."""
    return 1 + sum(count_rules(e) for e in rule.exceptions)


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def format_prediction(result: Dict) -> str:
    """
    Format a prediction result dict into a multi-line human-readable string.

    Parameters
    ----------
    result : Dict as returned by trace_decision_path / predict_one.

    Returns
    -------
    Formatted string.
    """
    lines = [
        f"Prediction : {result['prediction']}",
        f"Actual     : {result.get('actual', 'N/A')}",
        f"Correct    : {result.get('correct', 'N/A')}",
        f"Path       : {' → '.join(result['path_ids'])}",
        "Steps:",
    ]
    for step in result["steps"]:
        cond_str = (
            ", ".join(f"{k}={v}" for k, v in step["conditions_matched"].items())
            if step["conditions_matched"]
            else "always"
        )
        lines.append(
            f"  [{step['rule_id']}] conditions=({cond_str}) "
            f"→ conclusion={step['conclusion']}"
        )
        lines.append(f"    Justification: {step['justification']}")
    return "\n".join(lines)
