import json
import time
import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def measure_latency_baseline(model, text, n_calls=100):
    latencies = []
    for _ in range(n_calls):
        start = time.perf_counter()
        _ = model.predict([text])
        latencies.append(time.perf_counter() - start)
    return np.mean(latencies) * 1000, np.percentile(latencies, 99) * 1000

def measure_latency_transformer(model, tokenizer, text, device, n_calls=100):
    latencies = []
    # Warmup
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)
        
    for _ in range(n_calls):
        start = time.perf_counter()
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            _ = model(**inputs)
        latencies.append(time.perf_counter() - start)
    return np.mean(latencies) * 1000, np.percentile(latencies, 99) * 1000

def main():
    print("Loading evaluation metrics...")
    try:
        with open("models/v0.1_evaluation_comparison.json", "r") as f:
            baseline_results = json.load(f)["results"]
    except FileNotFoundError:
        print("Error: Baseline evaluation comparison not found.")
        return
        
    try:
        with open("models/v0.1_transformer_evaluation_comparison.json", "r") as f:
            transformer_results = json.load(f)["results"]
    except FileNotFoundError:
        print("Error: Transformer evaluation comparison not found. Ensure Step 7 ran successfully.")
        return

    print("Measuring latency for baseline model...")
    baseline_model = joblib.load("models/v0.1_baseline.joblib")
    test_text = "This is a sample news article text for latency testing. It contains a reasonable amount of words to simulate a real query."
    baseline_mean, baseline_p99 = measure_latency_baseline(baseline_model, test_text)

    print("Measuring latency for transformer model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_path = "models/v0.1_transformer"
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    model.to(device)
    model.eval()
    transformer_mean, transformer_p99 = measure_latency_transformer(model, tokenizer, test_text, device)

    # Compile Table
    rows = []
    for br in baseline_results:
        rows.append({
            "Model": "Baseline (LogisticRegression)",
            "Test Condition": br["split_name"],
            "F1": f"{br['f1_macro']:.4f}" if br['f1_macro'] else "N/A",
            "Accuracy": f"{br['accuracy']:.4f}",
            "ROC-AUC": f"{br['roc_auc']:.4f}" if br['roc_auc'] else "N/A",
            "Latency Mean (ms)": f"{baseline_mean:.2f}",
            "Latency p99 (ms)": f"{baseline_p99:.2f}"
        })
        
    for tr in transformer_results:
        rows.append({
            "Model": "Transformer (DistilBERT)",
            "Test Condition": tr["split_name"],
            "F1": f"{tr['f1_macro']:.4f}" if tr['f1_macro'] else "N/A",
            "Accuracy": f"{tr['accuracy']:.4f}",
            "ROC-AUC": f"{tr['roc_auc']:.4f}" if tr['roc_auc'] else "N/A",
            "Latency Mean (ms)": f"{transformer_mean:.2f}",
            "Latency p99 (ms)": f"{transformer_p99:.2f}"
        })

    df_compare = pd.DataFrame(rows)
    print("\n--- MODEL COMPARISON ---")
    print(df_compare.to_markdown(index=False))
    
    # Calculate F1 Deltas
    print("\n--- COST / BENEFIT ANALYSIS ---")
    baseline_f1_temporal = next((r['f1_macro'] for r in baseline_results if r['split_name'] == 'Temporal Test' and r['f1_macro']), None)
    transformer_f1_temporal = next((r['f1_macro'] for r in transformer_results if r['split_name'] == 'Temporal Test' and r['f1_macro']), None)
    
    baseline_f1_topic = next((r['f1_macro'] for r in baseline_results if r['split_name'] == 'Topic-Disjoint Test' and r['f1_macro']), None)
    transformer_f1_topic = next((r['f1_macro'] for r in transformer_results if r['split_name'] == 'Topic-Disjoint Test' and r['f1_macro']), None)

    latency_increase_factor = transformer_mean / baseline_mean if baseline_mean > 0 else float('inf')
    
    print(f"Latency Cost: Transformer is {latency_increase_factor:.1f}x slower than Baseline on CPU.")
    
    if transformer_f1_temporal and baseline_f1_temporal:
        print(f"Temporal Test F1 Gain: {(transformer_f1_temporal - baseline_f1_temporal)*100:.2f}%")
    if transformer_f1_topic and baseline_f1_topic:
        print(f"Topic-Disjoint Test F1 Gain: {(transformer_f1_topic - baseline_f1_topic)*100:.2f}%")
        
    print("\nConclusion:")
    if transformer_f1_temporal and baseline_f1_temporal and transformer_f1_topic and baseline_f1_topic:
        gain_temp = transformer_f1_temporal - baseline_f1_temporal
        gain_topic = transformer_f1_topic - baseline_f1_topic
        avg_gain = (gain_temp + gain_topic) / 2
        
        if avg_gain > 0.05: # Arbitrary 5% threshold
            print(f"The Transformer provides a meaningful F1 gain ({avg_gain*100:.2f}% avg on hard splits). Depending on latency budgets, this gain MAY justify the {latency_increase_factor:.1f}x latency cost.")
        elif avg_gain > 0:
            print(f"The Transformer provides only a marginal F1 gain ({avg_gain*100:.2f}% avg on hard splits). This does NOT justify the {latency_increase_factor:.1f}x latency cost for a real-time serving layer, unless accuracy is the absolute only priority.")
        else:
            print(f"The Transformer degrades or shows zero F1 gain ({avg_gain*100:.2f}% avg on hard splits) while severely increasing latency ({latency_increase_factor:.1f}x slower). It is NOT recommended over the Baseline.")

if __name__ == "__main__":
    main()
