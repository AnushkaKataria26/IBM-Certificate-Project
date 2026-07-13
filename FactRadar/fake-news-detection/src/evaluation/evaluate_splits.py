"""
evaluate_splits.py — Evaluate baseline model on all three test conditions.

Loads models/v0.1_baseline.joblib and evaluates on:
  (a) data/splits/test.csv                 — original random-split test set
  (b) data/splits/topic_disjoint_test.csv  — topic-disjoint holdout
  (c) data/splits/temporal_test.csv        — temporal holdout

Computes accuracy, precision, recall, F1 (macro), ROC-AUC, confusion matrix
for each split. Handles single-class edge cases gracefully.

Produces a three-way comparison table, F1 delta analysis, and saves results
to models/v0.1_evaluation_comparison.json.

Part of Phase 2: Rigorous Evaluation (Steps 4–6).
"""

import sys
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# Ensure project root is importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


def evaluate_single_split(
    pipeline, df: pd.DataFrame, split_name: str
) -> dict:
    """Evaluate a pipeline on a single data split.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        The trained model pipeline.
    df : pd.DataFrame
        The test data with 'clean_text' and 'label' columns.
    split_name : str
        Name of the split (for reporting).

    Returns
    -------
    dict
        Computed metrics (or None values where metrics are undefined).
    """

    X = df["clean_text"].fillna("")
    y_true = df["label"]
    n_classes = y_true.nunique()

    # Predict
    y_pred = pipeline.predict(X)

    # Base metrics (always computable)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    result = {
        "split_name": split_name,
        "n_rows": len(df),
        "n_classes_in_data": int(n_classes),
        "label_balance": y_true.value_counts().sort_index().to_dict(),
        "accuracy": round(float(acc), 6),
        "confusion_matrix": {
            "tn": int(cm[0][0]) if cm.shape == (2, 2) else None,
            "fp": int(cm[0][1]) if cm.shape == (2, 2) else None,
            "fn": int(cm[1][0]) if cm.shape == (2, 2) else None,
            "tp": int(cm[1][1]) if cm.shape == (2, 2) else None,
        },
    }

    # Metrics that require both classes
    if n_classes >= 2:
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        result["precision_macro"] = round(float(prec), 6)
        result["recall_macro"] = round(float(rec), 6)
        result["f1_macro"] = round(float(f1), 6)

        # ROC-AUC requires both classes AND predict_proba
        try:
            y_proba = pipeline.predict_proba(X)[:, 1]
            roc = roc_auc_score(y_true, y_proba)
            result["roc_auc"] = round(float(roc), 6)
        except (ValueError, IndexError):
            result["roc_auc"] = None
            result["roc_auc_skip_reason"] = (
                "ROC-AUC computation failed (likely single-class predictions)."
            )
    else:
        only_class = int(y_true.unique()[0])
        result["precision_macro"] = None
        result["recall_macro"] = None
        result["f1_macro"] = None
        result["roc_auc"] = None
        result["metrics_skip_reason"] = (
            f"F1-macro and ROC-AUC skipped: holdout contains only class "
            f"{only_class}. These metrics are undefined or misleading "
            f"with a single class present."
        )

    return result


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted three-way comparison table."""

    print(f"\n{'='*80}")
    print("THREE-WAY EVALUATION COMPARISON")
    print(f"{'='*80}")

    # Header
    names = [r["split_name"] for r in results]
    col_width = 22
    header = f"{'Metric':<25}" + "".join(f"{n:>{col_width}}" for n in names)
    print(header)
    print("-" * (25 + col_width * len(names)))

    # Metrics to display
    metrics = [
        ("Rows", "n_rows"),
        ("Classes in data", "n_classes_in_data"),
        ("Accuracy", "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1 (macro)", "f1_macro"),
        ("ROC-AUC", "roc_auc"),
    ]

    for display_name, key in metrics:
        row = f"{display_name:<25}"
        for r in results:
            val = r.get(key)
            if val is None:
                row += f"{'N/A':>{col_width}}"
            elif isinstance(val, float):
                row += f"{val:>{col_width}.6f}"
            else:
                row += f"{val:>{col_width}}"
        print(row)

    # Confusion matrices
    print(f"\n{'='*80}")
    print("CONFUSION MATRICES")
    print(f"{'='*80}")
    for r in results:
        cm = r["confusion_matrix"]
        print(f"\n  {r['split_name']}:")
        if cm["tn"] is not None:
            print(f"    {'':>15} Predicted 0  Predicted 1")
            print(f"    {'Actual 0':>15}    {cm['tn']:>6}       {cm['fp']:>6}")
            print(f"    {'Actual 1':>15}    {cm['fn']:>6}       {cm['tp']:>6}")
        else:
            print(f"    (confusion matrix not fully computable)")

        # Label balance
        balance = r.get("label_balance", {})
        print(f"    Label balance: {balance}")

        # Skip reasons
        if "metrics_skip_reason" in r:
            print(f"    ⚠️  {r['metrics_skip_reason']}")
        if "roc_auc_skip_reason" in r:
            print(f"    ⚠️  {r['roc_auc_skip_reason']}")


def print_f1_delta_analysis(results: list[dict]) -> None:
    """Print F1 delta analysis comparing alternative splits to random split."""

    print(f"\n{'='*80}")
    print("STEP 5: F1 DELTA ANALYSIS (GENERALIZATION GAP)")
    print(f"{'='*80}")

    # Find random-split result (always first)
    random_result = results[0]
    random_f1 = random_result.get("f1_macro")

    if random_f1 is None:
        print("  Cannot compute deltas: random-split F1 is unavailable.")
        return

    print(f"\n  Random-split test F1 (macro): {random_f1:.6f}")
    print()

    threshold_pp = 5.0
    deltas = {}

    for r in results[1:]:
        alt_f1 = r.get("f1_macro")
        name = r["split_name"]

        if alt_f1 is None:
            print(
                f"  {name}: F1 not available (skipped due to single-class "
                f"holdout). Delta cannot be computed."
            )
            deltas[name] = None
            continue

        delta = random_f1 - alt_f1
        delta_pp = delta * 100  # percentage points
        deltas[name] = round(delta_pp, 2)

        exceeds = abs(delta_pp) > threshold_pp
        indicator = "⚠️  EXCEEDS" if exceeds else "✅ WITHIN"

        print(
            f"  {name}:"
        )
        print(f"    F1 (macro):  {alt_f1:.6f}")
        print(f"    Delta:       {delta_pp:+.2f} pp")
        print(
            f"    {indicator} {threshold_pp}pp threshold"
        )

        if exceeds:
            print(
                f"    OBSERVATION: The {abs(delta_pp):.2f}pp gap suggests the "
                f"model may be relying on dataset-specific artifacts "
                f"(e.g., topic-correlated vocabulary, temporal writing style "
                f"shifts) rather than generalizable signal. This does not "
                f"necessarily mean the model is 'bad' — it means its reported "
                f"performance on the random split may overestimate real-world "
                f"generalization."
            )
        print()

    return deltas


def evaluate_all_splits(
    model_path: str = "models/v0.1_baseline.joblib",
    random_test_path: str = "data/splits/test.csv",
    topic_test_path: str = "data/splits/topic_disjoint_test.csv",
    temporal_test_path: str = "data/splits/temporal_test.csv",
    output_path: str = "models/v0.1_evaluation_comparison.json",
) -> dict:
    """Evaluate baseline model on all three test conditions.

    Returns
    -------
    dict
        Full comparison results including metrics and F1 deltas.
    """

    print("=" * 80)
    print("STEP 4: EVALUATE BASELINE MODEL ON ALL THREE TEST CONDITIONS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model from {model_path} ...")
    pipeline = joblib.load(model_path)
    print(f"  ✅ Model loaded: {type(pipeline).__name__}")

    # ------------------------------------------------------------------
    # 2. Load test sets
    # ------------------------------------------------------------------
    splits_config = [
        ("Random Split Test", random_test_path),
        ("Topic-Disjoint Test", topic_test_path),
        ("Temporal Test", temporal_test_path),
    ]

    results = []
    for name, path in splits_config:
        print(f"\nLoading {name} from {path} ...")
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}. Skipping.")
            continue

        df = pd.read_csv(path)
        print(f"  Loaded {len(df):,} rows")

        # Verify required columns
        if "clean_text" not in df.columns:
            print(f"  ❌ Missing 'clean_text' column. Skipping {name}.")
            continue
        if "label" not in df.columns:
            print(f"  ❌ Missing 'label' column. Skipping {name}.")
            continue

        # Handle empty clean_text
        empty_count = (df["clean_text"].fillna("").str.strip() == "").sum()
        if empty_count > 0:
            print(
                f"  ⚠️  {empty_count} rows with empty clean_text — "
                f"predictions on these will be based on zero-feature vectors."
            )

        print(f"  Evaluating {name} ...")
        result = evaluate_single_split(pipeline, df, name)
        results.append(result)
        print(f"  ✅ {name}: Accuracy={result['accuracy']:.4f}", end="")
        if result.get("f1_macro") is not None:
            print(f", F1={result['f1_macro']:.4f}", end="")
        if result.get("roc_auc") is not None:
            print(f", ROC-AUC={result['roc_auc']:.4f}", end="")
        print()

    # ------------------------------------------------------------------
    # 3. Print comparison table (Step 4)
    # ------------------------------------------------------------------
    print_comparison_table(results)

    # ------------------------------------------------------------------
    # 4. F1 delta analysis (Step 5)
    # ------------------------------------------------------------------
    deltas = print_f1_delta_analysis(results)

    # ------------------------------------------------------------------
    # 5. Step 6: Final verification report
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("STEP 6: FINAL VERIFICATION REPORT")
    print(f"{'='*80}")

    for r in results:
        name = r["split_name"]
        print(f"\n  {name}:")
        print(f"    Rows: {r['n_rows']:,}")
        print(f"    Classes in data: {r['n_classes_in_data']}")
        print(f"    Label balance: {r.get('label_balance', 'N/A')}")

        if r["n_classes_in_data"] < 2:
            print(f"    ⚠️  SINGLE-CLASS LIMITATION: {r.get('metrics_skip_reason', '')}")
        if r["n_rows"] < 500:
            print(f"    ⚠️  SMALL-SAMPLE LIMITATION: Only {r['n_rows']} rows.")

    print(f"\n  KNOWN DATASET LIMITATIONS:")
    print(
        f"    - True publisher-disjoint evaluation is NOT possible with this "
        f"dataset (source_file is collinear with label)."
    )
    print(
        f"    - Topic-disjoint split is a PROXY using the 'subject' column. "
        f"Each subject maps 100% to one label class, causing skewed holdout "
        f"label balance."
    )
    print(
        f"    - For genuine publisher-disjoint evaluation, use FakeNewsNet "
        f"or another dataset with source-domain metadata."
    )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "results": results,
        "f1_deltas_pp": deltas,
        "dataset_limitations": [
            "source_file is collinear with label — no publisher-disjoint split possible",
            "subject column used as proxy for topic-disjoint split; each subject is 100% one class",
            "Temporal split date parsing required multi-format handling; 1 row had a non-date string",
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved full comparison → {output_path}")

    print(f"\n{'='*80}")
    print("PHASE 2 EVALUATION COMPLETE")
    print(f"{'='*80}")

    return output


if __name__ == "__main__":
    evaluate_all_splits()
