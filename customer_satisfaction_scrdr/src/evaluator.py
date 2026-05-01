"""
evaluator.py
------------
Batch evaluation of the SCRDR engine against a dataset.

Provides accuracy metrics, per-class statistics, and detailed prediction logs.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import csv

from scrdr_engine import Rule, trace_decision_path


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """
    Load a CSV file into a list of record dicts.

    The last column is expected to be 'satisfaction' (the ground-truth label).
    All values are kept as strings (categorical).

    Parameters
    ----------
    filepath : Path to the CSV file.

    Returns
    -------
    List of dicts, one per row.
    """
    records: List[Dict[str, str]] = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


# ---------------------------------------------------------------------------
# Single-record evaluation
# ---------------------------------------------------------------------------

def predict_one(record: Dict[str, str], root_rule: Rule) -> Dict:
    """
    Classify a single record and return a full explanation dict.

    Parameters
    ----------
    record    : Feature dict (may include 'satisfaction' key — ignored here).
    root_rule : Root of the SCRDR tree.

    Returns
    -------
    Explanation dict from trace_decision_path plus:
        "actual"  : ground-truth label (if present).
        "correct" : bool.
    """
    # Strip label from input features before classification
    input_features = {k: v for k, v in record.items() if k != "satisfaction"}
    actual = record.get("satisfaction", None)

    result = trace_decision_path(input_features, root_rule)
    result["actual"] = actual
    result["correct"] = (result["prediction"] == actual) if actual else None
    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    records: List[Dict[str, str]],
    root_rule: Rule,
    verbose: bool = False,
) -> Dict:
    """
    Run the SCRDR engine on every record in the dataset and collect metrics.

    Parameters
    ----------
    records   : List of record dicts (with ground-truth 'satisfaction' column).
    root_rule : Root of the SCRDR tree.
    verbose   : If True, print each prediction to stdout.

    Returns
    -------
    {
        "total"      : int,
        "correct"    : int,
        "accuracy"   : float,
        "per_class"  : {label: {tp, fp, fn, precision, recall, f1}},
        "predictions": [result_dict, ...]
    }
    """
    predictions = []
    labels = set()

    for record in records:
        result = predict_one(record, root_rule)
        predictions.append(result)
        if result["actual"]:
            labels.add(result["actual"])
        if verbose:
            status = "✓" if result["correct"] else "✗"
            print(
                f"[{status}] Actual={result['actual']:15s} "
                f"Predicted={result['prediction']:15s} "
                f"Path={result['path_ids']}"
            )

    # Aggregate metrics
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision / recall / F1
    per_class: Dict[str, Dict] = {}
    for lbl in sorted(labels):
        tp = sum(1 for p in predictions if p["actual"] == lbl and p["prediction"] == lbl)
        fp = sum(1 for p in predictions if p["actual"] != lbl and p["prediction"] == lbl)
        fn = sum(1 for p in predictions if p["actual"] == lbl and p["prediction"] != lbl)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[lbl] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_metrics(eval_result: Dict) -> None:
    """Print a formatted evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("SCRDR EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples  : {eval_result['total']}")
    print(f"Correct        : {eval_result['correct']}")
    print(f"Accuracy       : {eval_result['accuracy'] * 100:.2f}%")
    print("\nPer-class metrics:")
    print(f"  {'Label':15s} {'Prec':>6} {'Recall':>7} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 50)
    for lbl, m in eval_result["per_class"].items():
        print(
            f"  {lbl:15s} {m['precision']:6.3f} {m['recall']:7.3f} {m['f1']:6.3f} "
            f"{m['tp']:4d} {m['fp']:4d} {m['fn']:4d}"
        )
    print("=" * 60)


def print_misclassifications(eval_result: Dict) -> None:
    """Print records where SCRDR was wrong."""
    errors = [p for p in eval_result["predictions"] if p["correct"] is False]
    print(f"\nMisclassified samples ({len(errors)}):")
    for e in errors:
        print(f"  Actual={e['actual']} | Predicted={e['prediction']}")
        print(f"    Path: {' → '.join(e['path_ids'])}")
        print(f"    Explanation: {e['explanation']}")
        print()
