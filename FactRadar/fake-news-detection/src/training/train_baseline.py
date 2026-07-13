"""
train_baseline.py — Train and evaluate the baseline Logistic Regression model.

Pipeline: TF-IDF(1,2-gram, 20k features) → LogisticRegression(balanced).
Fits on train split only, evaluates on val split and logs to MLflow.
Saves model locally and registers runs in MLflow.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.tfidf import build_tfidf_vectorizer
from src.evaluation.evaluate_splits import evaluate_single_split

# ---------------------------------------------------------------------------
# Project-wide reproducibility seed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

def compute_data_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for versioning."""
    if not os.path.exists(filepath):
        return "unknown"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def train_baseline(
    train_path: str = "data/splits/train.csv",
    val_path: str = "data/splits/val.csv",
    random_test_path: str = "data/splits/test.csv",
    topic_test_path: str = "data/splits/topic_disjoint_test.csv",
    temporal_test_path: str = "data/splits/temporal_test.csv",
    data_source_path: str = "data/processed/cleaned_dataset.csv",
    model_dir: str = "models",
) -> dict:
    """Train baseline model and evaluate, logging to MLflow."""

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sourcetrace")

    with mlflow.start_run(run_name="baseline_v0.1"):
        # Log environment info
        mlflow.set_tag("python_version", sys.version)
        mlflow.set_tag("hardware", "CPU")
        data_hash = compute_data_hash(data_source_path)
        mlflow.set_tag("data_hash", data_hash)

        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------
        print(f"Loading training data from {train_path} ...")
        df_train = pd.read_csv(train_path)
        print(f"Loading validation data from {val_path} ...")
        df_val = pd.read_csv(val_path)

        train_empty = (df_train["clean_text"].fillna("").str.strip() == "").sum()
        val_empty = (df_val["clean_text"].fillna("").str.strip() == "").sum()

        if train_empty > 0 or val_empty > 0:
            raise ValueError(
                f"Empty clean_text found after split! "
                f"Train empties: {train_empty}, Val empties: {val_empty}. "
                f"This indicates run_preprocessing.py's drop logic failed. "
            )

        X_train = df_train["clean_text"]
        y_train = df_train["label"]
        X_val = df_val["clean_text"]
        y_val = df_val["label"]

        # ------------------------------------------------------------------
        # 2. Build pipeline
        # ------------------------------------------------------------------
        vectorizer = build_tfidf_vectorizer()
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ])

        # ------------------------------------------------------------------
        # 3. Train
        # ------------------------------------------------------------------
        print("\nTraining pipeline (TF-IDF + LogisticRegression) ...")
        pipeline.fit(X_train, y_train)
        print("  ✅ Training complete.")

        # Log params
        tfidf_params = pipeline.named_steps["tfidf"].get_params()
        mlflow.log_params({
            "ngram_range": str(tfidf_params["ngram_range"]),
            "max_features": tfidf_params["max_features"],
            "min_df": tfidf_params["min_df"],
            "max_df": tfidf_params["max_df"],
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        })

        # ------------------------------------------------------------------
        # 4. Predict on validation set
        # ------------------------------------------------------------------
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        unique_preds = set(y_pred)
        if len(unique_preds) == 1:
            raise RuntimeError(
                f"DEGENERATE MODEL: Only class {unique_preds.pop()} predicted."
            )

        # ------------------------------------------------------------------
        # 5. Compute and log validation metrics
        # ------------------------------------------------------------------
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro")
        rec = recall_score(y_val, y_pred, average="macro")
        f1 = f1_score(y_val, y_pred, average="macro")
        roc_auc = roc_auc_score(y_val, y_proba)
        
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_roc_auc": roc_auc
        })

        # ------------------------------------------------------------------
        # 6. Evaluate on Test Splits
        # ------------------------------------------------------------------
        test_splits = [
            ("random", random_test_path),
            ("topic_disjoint", topic_test_path),
            ("temporal", temporal_test_path)
        ]

        metrics = {
            "val_accuracy": acc,
            "val_f1": f1
        }
        
        for split_prefix, path in test_splits:
            print(f"\nEvaluating on {split_prefix} test set from {path}...")
            if os.path.exists(path):
                df_test = pd.read_csv(path)
                res = evaluate_single_split(pipeline, df_test, split_prefix)
                
                # Log valid metrics
                for m_key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]:
                    val = res.get(m_key)
                    if val is not None:
                        mlflow.log_metric(f"{split_prefix}_{m_key}", float(val))
                        metrics[f"{split_prefix}_{m_key}"] = float(val)
                
                # Handle single-class edge case
                if res.get("n_classes_in_data", 0) < 2:
                    print(f"  ⚠️ SINGLE-CLASS LIMITATION DETECTED for {split_prefix}.")
                    mlflow.log_metric(f"{split_prefix}_single_class", 1.0)
                    metrics[f"{split_prefix}_single_class"] = 1.0
            else:
                print(f"  ❌ File not found: {path}")

        # ------------------------------------------------------------------
        # 7. Save model
        # ------------------------------------------------------------------
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "v0.1_baseline.joblib")
        joblib.dump(pipeline, model_path)
        
        mlflow.sklearn.log_model(pipeline, "model")
        print(f"\n  Saved model locally to {model_path} and logged to MLflow.")

        metrics_path = os.path.join(model_dir, "v0.1_baseline_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

if __name__ == "__main__":
    train_baseline()
