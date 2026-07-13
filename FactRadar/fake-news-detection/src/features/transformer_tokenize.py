import pandas as pd
from transformers import AutoTokenizer

def load_and_clean_data(file_path="data/processed/cleaned_dataset.csv"):
    """
    Loads dataset and reuses the Phase 1 drop logic to ensure no NaN/float values in clean_text.
    """
    df = pd.read_csv(file_path)
    initial_len = len(df)
    
    # Ensure clean_text is string and drop NaNs
    df = df.dropna(subset=['clean_text'])
    df['clean_text'] = df['clean_text'].astype(str)
    
    # Drop empty strings
    df = df[df['clean_text'].str.strip() != '']
    
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows due to NaN or empty 'clean_text'.")
        
    return df

def get_tokenizer(model_name="distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(examples, tokenizer):
    """
    Tokenizes a batch of examples.
    """
    return tokenizer(
        examples["clean_text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def check_truncation_stats(df, tokenizer):
    """
    Reports count and percentage of rows that will be truncated.
    """
    print("Checking truncation statistics...")
    # Tokenize without truncation to check actual lengths
    lengths = df['clean_text'].apply(lambda text: len(tokenizer.encode(text, truncation=False)))
    
    truncated_count = (lengths > 512).sum()
    total_count = len(df)
    truncation_pct = (truncated_count / total_count) * 100
    
    print(f"Total rows: {total_count}")
    print(f"Rows exceeding 512 tokens: {truncated_count} ({truncation_pct:.2f}%)")
    
    return truncated_count, truncation_pct

if __name__ == "__main__":
    df = load_and_clean_data()
    tokenizer = get_tokenizer()
    check_truncation_stats(df, tokenizer)
