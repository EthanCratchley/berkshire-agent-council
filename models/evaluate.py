"""
Evaluate trained RF and KNN models on the held-out test set.

Loads the cached dataset, applies the same temporal 80/20 split used during
training, then reports per-horizon metrics for both classifiers side-by-side:
accuracy, per-class precision/recall/f1, confusion matrix, and a head-to-head
comparison summary.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.feature_engineering import FEATURE_ORDER
from shared.horizon import HORIZON_LABEL_CONFIG, VALID_HORIZONS

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cached_dataset.csv",
)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_ORDER = ["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"]


def load_test_split(horizon: str):
    """Load dataset and return (X_test, y_test) for a given horizon."""
    df = pd.read_csv(DATASET_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    label_col = f"label_{horizon}"
    hdf = df.dropna(subset=[label_col]).reset_index(drop=True)

    split_idx = int(len(hdf) * 0.8)
    test_df = hdf.iloc[split_idx:]

    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df[label_col].values
    return X_test, y_test, test_df


def _load_artifact(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _print_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n{title}")
    header = "".join(f"{l:>12}" for l in labels)
    print(f"{'Predicted ->':>14}{header}")
    for i, row_label in enumerate(labels):
        row_vals = "".join(f"{v:>12}" for v in cm[i])
        print(f"  {row_label:>12}{row_vals}")


def evaluate_horizon(horizon: str):
    """Evaluate RF and KNN for a single horizon. Returns dict of metrics."""
    cfg = HORIZON_LABEL_CONFIG[horizon]
    print(f"\n{'='*70}")
    print(f"  {horizon.upper()} HORIZON  ({cfg['forward_days']}-day forward return)")
    print(f"  Thresholds: {cfg['thresholds']}")
    print(f"{'='*70}")

    X_test, y_test, test_df = load_test_split(horizon)
    print(f"Test samples: {len(y_test)}")
    print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Test label distribution:")
    for label in CLASS_ORDER:
        count = (y_test == label).sum()
        print(f"  {label:<12} {count:>5}  ({count/len(y_test)*100:5.1f}%)")

    imputer = _load_artifact(f"imputer_{horizon}.pkl")
    if imputer is None:
        print(f"  Imputer not found for {horizon}. Skipping.")
        return None

    X_imputed = imputer.transform(X_test)

    results = {}

    # --- Random Forest ---
    rf = _load_artifact(f"rf_{horizon}.pkl")
    if rf is not None:
        rf_preds = rf.predict(X_imputed)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average="weighted")

        print(f"\n--- Random Forest ---")
        print(f"Accuracy:    {rf_acc:.4f}")
        print(f"Weighted F1: {rf_f1:.4f}")
        print(classification_report(y_test, rf_preds, labels=CLASS_ORDER, zero_division=0))
        _print_confusion_matrix(y_test, rf_preds, CLASS_ORDER, "RF Confusion Matrix")

        results["rf"] = {
            "accuracy": rf_acc,
            "weighted_f1": rf_f1,
            "predictions": rf_preds,
        }
    else:
        print("\n  RF model not found. Skipping.")

    # --- KNN ---
    knn = _load_artifact(f"knn_{horizon}.pkl")
    scaler = _load_artifact(f"scaler_{horizon}.pkl")
    if knn is not None and scaler is not None:
        X_scaled = scaler.transform(X_imputed)
        knn_preds = knn.predict(X_scaled)
        knn_acc = accuracy_score(y_test, knn_preds)
        knn_f1 = f1_score(y_test, knn_preds, average="weighted")

        print(f"\n--- KNN (k={knn.n_neighbors}) ---")
        print(f"Accuracy:    {knn_acc:.4f}")
        print(f"Weighted F1: {knn_f1:.4f}")
        print(classification_report(y_test, knn_preds, labels=CLASS_ORDER, zero_division=0))
        _print_confusion_matrix(y_test, knn_preds, CLASS_ORDER, "KNN Confusion Matrix")

        results["knn"] = {
            "accuracy": knn_acc,
            "weighted_f1": knn_f1,
            "predictions": knn_preds,
        }
    else:
        print("\n  KNN model or scaler not found. Skipping.")

    # --- Head-to-head ---
    if "rf" in results and "knn" in results:
        print(f"\n--- Head-to-Head: {horizon.upper()} ---")
        rf_acc = results["rf"]["accuracy"]
        knn_acc = results["knn"]["accuracy"]
        rf_f1 = results["rf"]["weighted_f1"]
        knn_f1 = results["knn"]["weighted_f1"]

        print(f"{'Metric':<20} {'RF':>10} {'KNN':>10} {'Winner':>10}")
        print(f"{'-'*50}")
        acc_winner = "RF" if rf_acc > knn_acc else ("KNN" if knn_acc > rf_acc else "Tie")
        f1_winner = "RF" if rf_f1 > knn_f1 else ("KNN" if knn_f1 > rf_f1 else "Tie")
        print(f"{'Accuracy':<20} {rf_acc:>10.4f} {knn_acc:>10.4f} {acc_winner:>10}")
        print(f"{'Weighted F1':<20} {rf_f1:>10.4f} {knn_f1:>10.4f} {f1_winner:>10}")

        # Agreement rate
        rf_p = results["rf"]["predictions"]
        knn_p = results["knn"]["predictions"]
        agreement = (rf_p == knn_p).mean()
        both_correct = ((rf_p == y_test) & (knn_p == y_test)).mean()
        either_correct = ((rf_p == y_test) | (knn_p == y_test)).mean()
        print(f"\n{'Agreement rate':<20} {agreement:>10.1%}")
        print(f"{'Both correct':<20} {both_correct:>10.1%}")
        print(f"{'Either correct':<20} {either_correct:>10.1%}")

        # Per-class winner
        print(f"\nPer-class F1 comparison:")
        print(f"{'Class':<14} {'RF F1':>8} {'KNN F1':>8} {'Winner':>8}")
        print(f"{'-'*40}")
        for label in CLASS_ORDER:
            mask = y_test == label
            if mask.sum() == 0:
                continue
            rf_class_f1 = f1_score(y_test == label, rf_p == label, zero_division=0)
            knn_class_f1 = f1_score(y_test == label, knn_p == label, zero_division=0)
            w = "RF" if rf_class_f1 > knn_class_f1 else ("KNN" if knn_class_f1 > rf_class_f1 else "Tie")
            print(f"{label:<14} {rf_class_f1:>8.3f} {knn_class_f1:>8.3f} {w:>8}")

    results["y_test"] = y_test
    return results


def evaluate(horizons: list[str] | None = None):
    """Run evaluation for specified horizons (default: all)."""
    if horizons is None:
        horizons = sorted(VALID_HORIZONS)

    all_results = {}
    for h in horizons:
        if h not in HORIZON_LABEL_CONFIG:
            print(f"Skipping unknown horizon: {h}")
            continue
        all_results[h] = evaluate_horizon(h)

    # --- Cross-horizon summary ---
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-HORIZON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Horizon':<10} {'RF Acc':>10} {'RF F1':>10} {'KNN Acc':>10} {'KNN F1':>10}")
        print(f"{'-'*50}")
        for h in sorted(all_results):
            r = all_results[h]
            if r is None:
                continue
            rf_acc = r.get("rf", {}).get("accuracy", float("nan"))
            rf_f1 = r.get("rf", {}).get("weighted_f1", float("nan"))
            knn_acc = r.get("knn", {}).get("accuracy", float("nan"))
            knn_f1 = r.get("knn", {}).get("weighted_f1", float("nan"))
            print(f"{h:<10} {rf_acc:>10.4f} {rf_f1:>10.4f} {knn_acc:>10.4f} {knn_f1:>10.4f}")

    return all_results


if __name__ == "__main__":
    import sys as _sys
    args = _sys.argv[1:]
    selected = [a for a in args if a in VALID_HORIZONS] or None
    evaluate(selected)
