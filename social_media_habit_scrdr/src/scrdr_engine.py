"""
scrdr_engine.py
---------------
Single Classification Ripple Down Rules (SCRDR) Engine

SCRDR Traversal Logic:
  - Rules are organized in a priority list: specific rules before defaults
  - Iterate rules in priority order (most specific to least specific)
  - For the FIRST matching root rule: follow its exception chain
  - At each level: if exception fires, go deeper; else stop
  - Return the deepest matching rule's conclusion
  - Only ONE rule path is ever followed (strict single-path)

Rule Priority (highest to lowest):
  1. R2 — Unhealthy (most specific: requires 3 conditions)
  2. R1 — Healthy   (specific: requires 3 conditions)  
  3. R0 — Moderate  (default: always matches, empty conditions)
"""

import json
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class Rule:
    """Represents a single SCRDR rule node."""
    id: str
    conditions: dict
    conclusion: str
    justification: str
    exception: Optional["Rule"] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conditions": self.conditions,
            "conclusion": self.conclusion,
            "justification": self.justification,
            "exception": self.exception.to_dict() if self.exception else None
        }


def load_rules_from_json(filepath: str) -> list["Rule"]:
    """
    Load SCRDR rules from a JSON file.
    Returns a priority-ordered list of root Rule objects.
    More specific rules appear before the default (catch-all) rule.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    rules = []

    # Load sibling rules first (more specific — higher priority)
    for sibling_data in data.get("sibling_rules", []):
        rules.append(_parse_single_rule(sibling_data))

    # Load exception chain of root as a standalone high-priority rule
    if data.get("exception"):
        exc_rule = _parse_single_rule(data["exception"])
        rules.append(exc_rule)

    # Load the default rule (R0) last — it always matches, lowest priority
    default_rule = Rule(
        id=data["id"],
        conditions=data["conditions"],
        conclusion=data["conclusion"],
        justification=data["justification"],
        exception=None
    )
    rules.append(default_rule)

    return rules


def _parse_single_rule(data: dict) -> "Rule":
    """Recursively parse a rule dict into a Rule object."""
    exception = None
    if data.get("exception"):
        exception = _parse_single_rule(data["exception"])
    return Rule(
        id=data["id"],
        conditions=data["conditions"],
        conclusion=data["conclusion"],
        justification=data["justification"],
        exception=exception
    )


def match_condition(input_case: dict, rule: "Rule") -> bool:
    """
    Check whether all of a rule's conditions match the given input case.
    Supports operators: <=, >=, <, >, ==, in
    Returns True if ALL conditions satisfied (or conditions dict is empty).
    """
    if not rule.conditions:
        return True  # Default rule always matches

    for feature, constraint in rule.conditions.items():
        value = input_case.get(feature)
        if value is None:
            return False

        op = constraint["op"]
        threshold = constraint["value"]

        try:
            if op == "<=":
                if not (float(value) <= float(threshold)):
                    return False
            elif op == ">=":
                if not (float(value) >= float(threshold)):
                    return False
            elif op == "<":
                if not (float(value) < float(threshold)):
                    return False
            elif op == ">":
                if not (float(value) > float(threshold)):
                    return False
            elif op == "==":
                if str(value) != str(threshold):
                    return False
            elif op == "in":
                if str(value) not in [str(v) for v in threshold]:
                    return False
            else:
                raise ValueError(f"Unknown operator: {op}")
        except (TypeError, ValueError):
            return False

    return True


def evaluate_scrdr(input_case: dict, rules: list["Rule"]) -> dict:
    """
    Evaluate input case against SCRDR rule list.

    SCRDR Single-Path Logic:
      1. Iterate rules in priority order (specific -> general)
      2. First matching root rule triggers its exception chain
      3. Exception chain traversal: fire deeper if exception matches, else stop
      4. Return deepest matching rule's conclusion

    Returns dict with: label, rule_id, rule_path, explanation
    """
    for root_rule in rules:
        if match_condition(input_case, root_rule):
            path = [root_rule.id]
            current_conclusion = root_rule.conclusion
            current_rule_id = root_rule.id
            current_justification = root_rule.justification
            current_exception = root_rule.exception

            # Walk down the exception chain — single path only
            while current_exception is not None:
                if match_condition(input_case, current_exception):
                    path.append(current_exception.id)
                    current_conclusion = current_exception.conclusion
                    current_rule_id = current_exception.id
                    current_justification = current_exception.justification
                    current_exception = current_exception.exception
                else:
                    break  # Exception doesn't fire — stop here

            return {
                "label": current_conclusion,
                "rule_id": current_rule_id,
                "rule_path": path,
                "explanation": current_justification
            }

    # Fallback (should never reach here since R0 always matches)
    return {
        "label": "Moderate",
        "rule_id": "R0_fallback",
        "rule_path": ["R0_fallback"],
        "explanation": "Fallback default: no rules matched."
    }


def trace_path(input_case: dict, rules: list["Rule"]) -> list[str]:
    """Return only the fired rule IDs for debugging."""
    return evaluate_scrdr(input_case, rules)["rule_path"]
