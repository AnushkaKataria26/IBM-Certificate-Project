import pandas as pd
import torch
import numpy as np
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def evaluate_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for evaluation.")
    
    model_path = "models/v0.1_transformer"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    splits = [
        ("Random Split Test", "data/splits/test.csv"),
        ("Topic-Disjoint Test", "data/splits/topic_disjoint_test.csv"),
        ("Temporal Test", "data/splits/temporal_test.csv")
    ]

    results = []
    
    for split_name, file_path in splits:
        print(f"Evaluating on {split_name}...")
        df = pd.read_csv(file_path)
        
        # Clean data as before
        df = df.dropna(subset=['clean_text'])
        df['clean_text'] = df['clean_text'].astype(str)
        df = df[df['clean_text'].str.strip() != '']
        
        # Convert labels to int
        if df['label'].dtype == object or df['label'].dtype.name == 'category':
            try:
                df['label'] = df['label'].astype(int)
            except ValueError:
                unique_labels = sorted(df['label'].unique().tolist())
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                df['label'] = df['label'].map(label_map)
        else:
            df['label'] = df['label'].astype(int)
            
        texts = df['clean_text'].tolist()
        labels = df['label'].tolist()
        
        n_rows = len(df)
        classes_present = np.unique(labels)
        n_classes = len(classes_present)
        label_balance = df['label'].value_counts().to_dict()
        
        # Batch inference
        batch_size = 32
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                outputs = model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                probs = probs[:, 1].cpu().numpy() # Probability for class 1
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                
        # Calculate metrics
        acc = accuracy_score(labels, all_preds)
        
        if n_classes == 2:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, all_preds, average='macro', zero_division=0)
            roc_auc = roc_auc_score(labels, all_probs)
            cm = confusion_matrix(labels, all_preds, labels=[0, 1])
            cm_dict = {"tn": int(cm[0][0]), "fp": int(cm[0][1]), "fn": int(cm[1][0]), "tp": int(cm[1][1])}
        else:
            print(f"WARNING: {split_name} has only {n_classes} class(es). Skipping F1 and ROC-AUC.")
            precision = None
            recall = None
            f1 = None
            roc_auc = None
            cm_dict = None
            
        split_result = {
            "split_name": split_name,
            "n_rows": n_rows,
            "n_classes_in_data": n_classes,
            "label_balance": label_balance,
            "accuracy": float(acc),
            "confusion_matrix": cm_dict,
            "precision_macro": float(precision) if precision is not None else None,
            "recall_macro": float(recall) if recall is not None else None,
            "f1_macro": float(f1) if f1 is not None else None,
            "roc_auc": float(roc_auc) if roc_auc is not None else None
        }
        
        results.append(split_result)
        
    output_data = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
        "model": model_path,
        "results": results
    }
    
    out_path = "models/v0.1_transformer_evaluation_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Evaluation saved to {out_path}")

if __name__ == "__main__":
    evaluate_transformer()
