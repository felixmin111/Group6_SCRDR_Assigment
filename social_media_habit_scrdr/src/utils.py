"""
utils.py
--------
Utility functions for the SCRDR Social Media Habit Analyzer.
"""

from typing import Any
from scrdr_engine import Rule


def format_input(
    daily_usage_hours: float,
    sleep_time: str,
    primary_usage_type: str,
    notification_check_frequency: str,
    purpose: str
) -> dict:
    """
    Helper to create a properly formatted input case dict.

    Args:
      daily_usage_hours           : Hours per day spent on social media
      sleep_time                  : 'early', 'normal', or 'late'
      primary_usage_type          : 'study', 'social', or 'entertainment'
      notification_check_frequency: 'low', 'medium', or 'high'
      purpose                     : 'learning', 'communication', or 'scrolling'

    Returns:
      dict suitable for passing to evaluate_scrdr()
    """
    valid_sleep = ["early", "normal", "late"]
    valid_usage = ["study", "social", "entertainment"]
    valid_freq = ["low", "medium", "high"]
    valid_purpose = ["learning", "communication", "scrolling"]

    if sleep_time not in valid_sleep:
        raise ValueError(f"sleep_time must be one of {valid_sleep}")
    if primary_usage_type not in valid_usage:
        raise ValueError(f"primary_usage_type must be one of {valid_usage}")
    if notification_check_frequency not in valid_freq:
        raise ValueError(f"notification_check_frequency must be one of {valid_freq}")
    if purpose not in valid_purpose:
        raise ValueError(f"purpose must be one of {valid_purpose}")

    return {
        "daily_usage_hours": float(daily_usage_hours),
        "sleep_time": sleep_time,
        "primary_usage_type": primary_usage_type,
        "notification_check_frequency": notification_check_frequency,
        "purpose": purpose
    }


def print_rule_tree(rule: Rule, depth: int = 0) -> None:
    """
    Recursively print the entire SCRDR rule tree in a readable format.

    Example output:
      [R0] DEFAULT → Moderate
        ↳ [R1] low hours + early/normal sleep + learning/comm → Healthy
            ↳ [R1a] high notif + social → Moderate
    """
    indent = "    " * depth
    arrow = "↳ " if depth > 0 else ""
    print(f"{indent}{arrow}[{rule.id}] → {rule.conclusion}")
    if rule.conditions:
        for feat, constraint in rule.conditions.items():
            op = constraint["op"]
            val = constraint["value"]
            print(f"{indent}   • {feat} {op} {val}")
    print(f"{indent}   ✎ {rule.justification}")
    if rule.exception:
        print_rule_tree(rule.exception, depth + 1)


def print_all_rules(rules: list[Rule]) -> None:
    """Print all root rule chains."""
    print("\n" + "=" * 60)
    print("SCRDR RULE TREE")
    print("=" * 60)
    for rule in rules:
        print_rule_tree(rule)
        print()


def compute_class_distribution(cases: list[dict], label_field: str = "label") -> dict:
    """Compute frequency distribution of labels in the dataset."""
    dist: dict[str, int] = {}
    for case in cases:
        label = case.get(label_field, "Unknown")
        dist[label] = dist.get(label, 0) + 1
    return dist


def confusion_matrix(
    results: list[dict],
    classes: list[str] = ["Healthy", "Moderate", "Unhealthy"]
) -> dict:
    """
    Compute a confusion matrix from evaluation results.

    Returns:
      dict mapping (true_label, predicted_label) → count
    """
    matrix: dict[tuple, int] = {}
    for c_true in classes:
        for c_pred in classes:
            matrix[(c_true, c_pred)] = 0

    for r in results:
        true = r["true_label"]
        pred = r["predicted"]
        if (true, pred) in matrix:
            matrix[(true, pred)] += 1

    return matrix


def print_confusion_matrix(
    matrix: dict,
    classes: list[str] = ["Healthy", "Moderate", "Unhealthy"]
) -> None:
    """Print confusion matrix in tabular format."""
    print("\nConfusion Matrix (True \\ Predicted):")
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("-" * (12 + 12 * len(classes)))
    for c_true in classes:
        row = f"{c_true:>12}"
        for c_pred in classes:
            row += f"{matrix.get((c_true, c_pred), 0):>12}"
        print(row)
    print()


def per_class_accuracy(
    matrix: dict,
    classes: list[str] = ["Healthy", "Moderate", "Unhealthy"]
) -> dict:
    """Compute per-class accuracy (recall) from confusion matrix."""
    result = {}
    for c in classes:
        total = sum(matrix.get((c, pred), 0) for pred in classes)
        correct = matrix.get((c, c), 0)
        result[c] = correct / total if total > 0 else 0.0
    return result
