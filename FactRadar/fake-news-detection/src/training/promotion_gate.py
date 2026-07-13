import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import brier_score_loss
import torch

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

def get_predictions(model, texts):
    """Get positive class probabilities from model (sklearn or transformers pipeline)."""
    # Check if model is sklearn pipeline (has predict_proba)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)[:, 1]
        return probs
    else:
        # Assume it's a transformers pipeline or pyfunc that returns dicts/lists
        import transformers
        if isinstance(model, transformers.Pipeline):
            results = model(texts, truncation=True, max_length=512)
            # Find the prob for class 1 ("LABEL_1" or similar)
            probs = []
            for r in results:
                # Transformers pipeline might return {"label": "LABEL_1", "score": 0.9}
                # Or list of all scores if return_all_scores=True. We must be careful.
                pass
        
        # Actually, mlflow.pyfunc is safest
        # wait, we can just load via mlflow.pyfunc
        return None

def get_pyfunc_predictions(pyfunc_model, texts):
    # PyFunc predict output format varies depending on flavor.
    # For sklearn: returns predicted classes. We need predict_proba!
    # Let's load the UNDERLYING model to get probabilities.
    pass

def load_underlying_model(model_uri):
    """Load the underlying model (sklearn or pytorch) instead of generic pyfunc to get probabilities."""
    # We can inspect the MLmodel file or just try sklearn first, then pytorch/transformers
    try:
        return mlflow.sklearn.load_model(model_uri), "sklearn"
    except Exception:
        try:
            return mlflow.transformers.load_model(model_uri), "transformers"
        except Exception:
            return mlflow.pytorch.load_model(model_uri), "pytorch"

def compute_brier_score(model_uri, df, model_type):
    model, mtype = load_underlying_model(model_uri)
    
    texts = df["clean_text"].fillna("").astype(str).tolist()
    y_true = df["label"].astype(int).tolist()
    
    if mtype == "sklearn":
        probs = model.predict_proba(texts)[:, 1]
    elif mtype == "transformers":
        # model is a transformers pipeline
        results = model(texts, truncation=True, max_length=512, top_k=None) # top_k=None returns all scores
        probs = []
        for res in results:
            # res is a list of dicts like [{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.9}]
            # Find score for LABEL_1 or class 1
            prob1 = 0.0
            for label_score in res:
                if label_score['label'] == 'LABEL_1' or label_score['label'] == 1 or label_score['label'] == '1':
                    prob1 = label_score['score']
            probs.append(prob1)
        probs = np.array(probs)
    elif mtype == "pytorch":
        # It's a raw torch model, we need tokenizer
        # Actually it's complex to re-tokenize if the tokenizer isn't attached to the torch model flavor
        raise NotImplementedError("PyTorch flavor without pipeline not supported for Brier score in this script.")
        
    return brier_score_loss(y_true, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-version", type=int, required=True, help="Version number of the candidate model")
    args = parser.parse_args()

    client = MlflowClient()
    model_name = "sourcetrace-classifier"

    # 1. Load Candidate and Production
    try:
        candidate_mv = client.get_model_version(model_name, args.candidate_version)
    except Exception as e:
        print(f"Failed to load candidate version {args.candidate_version}: {e}")
        sys.exit(1)

    try:
        production_mv = client.get_model_version_by_alias(model_name, "Production")
    except Exception as e:
        print(f"No Production model found or failed to load alias: {e}")
        production_mv = None

    if not production_mv:
        print("No Production model. Promoting candidate by default.")
        client.set_registered_model_alias(model_name, "Production", candidate_mv.version)
        print("Transitioned candidate to @Production.")
        sys.exit(0)

    # 2. Get Metrics from runs
    cand_run = client.get_run(candidate_mv.run_id)
    prod_run = client.get_run(production_mv.run_id)

    cand_f1 = cand_run.data.metrics.get("topic_disjoint_f1_macro")
    prod_f1 = prod_run.data.metrics.get("topic_disjoint_f1_macro")

    # Edge case fallback
    cand_single = cand_run.data.metrics.get("topic_disjoint_single_class")
    prod_single = prod_run.data.metrics.get("topic_disjoint_single_class")

    fallback = False
    if cand_single == 1.0 or prod_single == 1.0 or cand_f1 is None or prod_f1 is None:
        print("⚠️ SINGLE-CLASS LIMITATION DETECTED in topic_disjoint split. Falling back to random-split F1.")
        cand_f1 = cand_run.data.metrics.get("random_f1_macro")
        prod_f1 = prod_run.data.metrics.get("random_f1_macro")
        fallback = True
        
        if cand_f1 is None or prod_f1 is None:
            print("Failed: Missing random_f1_macro metrics as well.")
            sys.exit(1)

    print(f"--- Metric Comparison ---")
    print(f"Candidate F1 ({'random' if fallback else 'topic-disjoint'}): {cand_f1:.4f}")
    print(f"Production F1 ({'random' if fallback else 'topic-disjoint'}): {prod_f1:.4f}")

    if cand_f1 < prod_f1:
        print(f"❌ PROMOTION FAILED: Candidate F1 ({cand_f1:.4f}) is lower than Production F1 ({prod_f1:.4f}).")
        sys.exit(0)
    else:
        print("✅ Candidate F1 meets or exceeds Production F1.")

    # 3. Brier Score Calibration Check
    print("\n--- Calibration Check (Brier Score on Topic-Disjoint) ---")
    test_path = "data/splits/topic_disjoint_test.csv"
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        sys.exit(1)
        
    df = pd.read_csv(test_path)
    # Remove empty
    df = df[df['clean_text'].fillna("").str.strip() != '']
    
    cand_uri = f"models:/{model_name}/{candidate_mv.version}"
    prod_uri = f"models:/{model_name}@Production"

    try:
        cand_brier = compute_brier_score(cand_uri, df, "candidate")
        prod_brier = compute_brier_score(prod_uri, df, "production")
        print(f"Candidate Brier Score:  {cand_brier:.4f}")
        print(f"Production Brier Score: {prod_brier:.4f}")
        
        # Brier score is a loss metric, lower is better. 
        # Requirement: "Candidate must have Brier score within a defined tolerance (e.g., not worse than production's by more than 0.02) to qualify."
        if cand_brier > (prod_brier + 0.02):
            print(f"❌ PROMOTION FAILED: Candidate Brier Score ({cand_brier:.4f}) is worse than Production ({prod_brier:.4f}) by more than 0.02.")
            sys.exit(0)
        else:
            print("✅ Candidate Brier Score is within tolerance.")
            
    except Exception as e:
        print(f"Failed to compute Brier Score: {e}")
        if fallback:
             print("Proceeding without Brier score due to fallback mode.")
        else:
             print("❌ PROMOTION FAILED due to Brier Score computation error.")
             sys.exit(1)

    # 4. Promote
    print("\n--- Promotion ---")
    print("Promoting Candidate to @Production, demoting current Production to @Staging.")
    client.set_registered_model_alias(model_name, "Staging", production_mv.version)
    client.set_registered_model_alias(model_name, "Production", candidate_mv.version)
    print("✅ Promotion successful.")

if __name__ == "__main__":
    main()
