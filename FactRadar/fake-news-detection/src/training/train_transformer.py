import time
import json
import hashlib
import sys
import os

import torch
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.transformer_tokenize import get_tokenizer
from src.training.prepare_transformer_dataset import prepare_transformer_datasets

def compute_data_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for versioning."""
    if not os.path.exists(filepath):
        return "unknown"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1': f1
    }

def evaluate_on_split(model, tokenizer, device, file_path, split_name):
    print(f"Evaluating on {split_name} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return {}
        
    df = pd.read_csv(file_path).head(50)
    df = df.dropna(subset=['clean_text'])
    df['clean_text'] = df['clean_text'].astype(str)
    df = df[df['clean_text'].str.strip() != '']
    
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
    
    n_classes = len(np.unique(labels))
    
    batch_size = 32
    all_preds = []
    all_probs = []
    
    model.eval()
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
            
    acc = accuracy_score(labels, all_preds)
    result = {"accuracy": float(acc), "n_classes_in_data": n_classes}
    
    if n_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, all_preds, average='macro', zero_division=0)
        roc_auc = roc_auc_score(labels, all_probs)
        result.update({
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "roc_auc": float(roc_auc)
        })
    else:
        print(f"  ⚠️ SINGLE-CLASS LIMITATION DETECTED for {split_name}.")
        
    return result

def train():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sourcetrace")

    with mlflow.start_run(run_name="transformer_v0.1"):
        mlflow.set_tag("python_version", sys.version)
        mlflow.set_tag("hardware", str(device).upper())
        data_hash = compute_data_hash("data/processed/cleaned_dataset.csv")
        mlflow.set_tag("data_hash", data_hash)

        train_dataset, val_dataset = prepare_transformer_datasets()
        train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(10, len(val_dataset))))
        tokenizer = get_tokenizer()
        
        model_name = "distilbert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        
        batch_size = 16
        learning_rate = 2e-5
        num_epochs = 3
        weight_decay = 0.01
        seed = 42

        # Log training parameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "seed": seed,
            "model_name": model_name
        })

        training_args = TrainingArguments(
            output_dir="models/v0.1_transformer_checkpoints",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed,
            report_to="none", # We will handle mlflow logging manually
            use_cpu=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        
        start_time = time.time()
        try:
            print(f"Starting training with batch size {batch_size}...")
            train_result = trainer.train()
        except torch.cuda.OutOfMemoryError:
            print("GPU OutOfMemoryError caught. Halving batch size and retrying once...")
            torch.cuda.empty_cache()
            batch_size = 8
            training_args.per_device_train_batch_size = batch_size
            training_args.per_device_eval_batch_size = batch_size
            mlflow.log_param("batch_size", batch_size) # Update batch size param
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )
            print(f"Restarting training with batch size {batch_size}...")
            train_result = trainer.train()
            
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        
        # Save Model locally
        save_path = "models/v0.1_transformer"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved locally to {save_path}")

        # Evaluate final model on val set
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": eval_metrics.get("eval_accuracy"),
            "val_precision": eval_metrics.get("eval_precision_macro"),
            "val_recall": eval_metrics.get("eval_recall_macro"),
            "val_f1": eval_metrics.get("eval_f1"),
        })

        # Evaluate on the 3 Test splits
        test_splits = [
            ("random", "data/splits/test.csv"),
            ("topic_disjoint", "data/splits/topic_disjoint_test.csv"),
            ("temporal", "data/splits/temporal_test.csv")
        ]

        metrics_data = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
            "model_name": model_name,
            "batch_size_used": batch_size,
            "training_time_seconds": training_time,
            "device": str(device)
        }

        for split_prefix, path in test_splits:
            res = evaluate_on_split(model, tokenizer, device, path, split_prefix)
            for m_key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]:
                val = res.get(m_key)
                if val is not None:
                    mlflow.log_metric(f"{split_prefix}_{m_key}", val)
                    metrics_data[f"{split_prefix}_{m_key}"] = val
            
            if res.get("n_classes_in_data", 0) < 2:
                mlflow.log_metric(f"{split_prefix}_single_class", 1.0)
                metrics_data[f"{split_prefix}_single_class"] = 1.0

        # Log to MLflow Registry
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        # Transformers flavor allows logging a pipeline or model + tokenizer directly
        try:
            import transformers
            task = "text-classification"
            transformers_pipeline = transformers.pipeline(
                task, model=model, tokenizer=tokenizer, device=model.device
            )
            mlflow.transformers.log_model(
                transformers_model=transformers_pipeline,
                artifact_path="model"
            )
            print("Logged transformer model to MLflow.")
        except Exception as e:
            print(f"Failed to log model to MLflow via transformers flavor: {e}")
            print("Trying generic pytorch logging as fallback...")
            mlflow.pytorch.log_model(model, "model")

        metrics_path = "models/v0.1_transformer_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()
