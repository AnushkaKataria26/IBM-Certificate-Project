import pandas as pd
from datasets import Dataset
import sys
import os

# Add src to path to import tokenize module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.transformer_tokenize import get_tokenizer, tokenize_batch

def load_and_prepare_split(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure no NaNs in clean_text
    df = df.dropna(subset=['clean_text'])
    df['clean_text'] = df['clean_text'].astype(str)
    df = df[df['clean_text'].str.strip() != '']
    
    # Check label encoding, ensure it's integer 0/1
    if 'label' in df.columns:
        if df['label'].dtype == object or df['label'].dtype.name == 'category':
            print("Label is string type, mapping to 0/1 integers...")
            # We assume classes are "fake" and "real" or similar, 
            # If the labels are already 0 and 1 but represented as strings, convert them:
            try:
                df['label'] = df['label'].astype(int)
            except ValueError:
                unique_labels = sorted(df['label'].unique().tolist())
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                print(f"Mapping labels: {label_map}")
                df['label'] = df['label'].map(label_map)
        else:
            df['label'] = df['label'].astype(int)
    
    # Create HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    return hf_dataset

def prepare_transformer_datasets(train_path="data/splits/train.csv", val_path="data/splits/val.csv"):
    train_dataset = load_and_prepare_split(train_path)
    val_dataset = load_and_prepare_split(val_path)
    
    tokenizer = get_tokenizer()
    
    # Define a wrapper for batched mapping
    def tokenize_function(examples):
        return tokenize_batch(examples, tokenizer)
    
    print("Tokenizing train dataset...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    # Put label column back
    train_tokenized = train_tokenized.add_column("label", train_dataset["label"])
    
    print("Tokenizing val dataset...")
    val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    # Put label column back
    val_tokenized = val_tokenized.add_column("label", val_dataset["label"])
    
    # Set format to PyTorch tensors
    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    print("Dataset preparation complete.")
    return train_tokenized, val_tokenized

if __name__ == "__main__":
    train_ds, val_ds = prepare_transformer_datasets()
    print("Train dataset sample:", train_ds[0])
    print("Train dataset length:", len(train_ds))
    print("Val dataset length:", len(val_ds))
