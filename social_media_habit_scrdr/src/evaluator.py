"""
evaluator.py
------------
Batch evaluation of the SCRDR system on a dataset.
Computes accuracy, generates per-sample predictions, and formats output.
"""

import csv
from typing import Optional
from scrdr_engine import Rule, load_rules_from_json, evaluate_scrdr


def load_dataset(filepath: str) -> list[dict]:
    """Load CSV dataset as a list of dicts."""
    cases = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["daily_usage_hours"] = float(row["daily_usage_hours"])
            cases.append(row)
    return cases


def evaluate_dataset(
    cases: list[dict],
    rules: list[Rule],
    label_field: str = "label"
) -> dict:
    """
    Run SCRDR evaluation on all cases.

    Returns:
      {
        "results": list of per-sample result dicts,
        "accuracy": float,
        "correct": int,
        "total": int,
        "errors": list of error cases
      }
    """
    results = []
    correct = 0
    errors = []

    for case in cases:
        true_label = case.get(label_field, None)
        # Remove label from input before evaluation
        input_case = {k: v for k, v in case.items() if k != label_field}

        prediction = evaluate_scrdr(input_case, rules)
        predicted = prediction["label"]
        is_correct = (predicted == true_label)

        if is_correct:
            correct += 1
        else:
            errors.append({
                "id": case.get("id"),
                "input": input_case,
                "true_label": true_label,
                "predicted": predicted,
                "rule_path": prediction["rule_path"],
                "explanation": prediction["explanation"]
            })

        results.append({
            "id": case.get("id"),
            "input": input_case,
            "true_label": true_label,
            "predicted": predicted,
            "correct": is_correct,
            "rule_path": prediction["rule_path"],
            "explanation": prediction["explanation"]
        })

    total = len(cases)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "results": results,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors
    }


def print_prediction(result: dict, verbose: bool = True) -> None:
    """Pretty-print a single prediction result."""
    status = "✓" if result["correct"] else "✗"
    print(f"[{status}] ID: {result['id']}")
    print(f"    True Label : {result['true_label']}")
    print(f"    Predicted  : {result['predicted']}")
    print(f"    Rule Path  : {' → '.join(result['rule_path'])}")
    if verbose:
        print(f"    Explanation: {result['explanation']}")
    print()


def print_summary(eval_result: dict) -> None:
    """Print accuracy summary."""
    print("=" * 60)
    print("SCRDR EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples : {eval_result['total']}")
    print(f"Correct       : {eval_result['correct']}")
    print(f"Accuracy      : {eval_result['accuracy']*100:.1f}%")
    print(f"Errors        : {len(eval_result['errors'])}")
    print("=" * 60)


if __name__ == "__main__":
    import os

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules = load_rules_from_json(os.path.join(BASE, "rules", "scrdr_rules.json"))
    cases = load_dataset(os.path.join(BASE, "data", "social_media_dataset.csv"))

    eval_result = evaluate_dataset(cases, rules)

    print("\n--- SAMPLE PREDICTIONS (first 10) ---\n")
    for r in eval_result["results"][:10]:
        print_prediction(r)

    print("\n--- ERROR CASES ---\n")
    if eval_result["errors"]:
        for e in eval_result["errors"]:
            print(f"  ID {e['id']}: True={e['true_label']}, Predicted={e['predicted']}")
            print(f"  Rule Path: {' → '.join(e['rule_path'])}")
            print()
    else:
        print("  No errors found!\n")

    print_summary(eval_result)
