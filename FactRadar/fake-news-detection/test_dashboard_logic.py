import pandas as pd
import requests

API_URL = "http://localhost:8000"

def process_batch(chunk):
    articles = chunk.to_dict(orient="records")
    response = requests.post(f"{API_URL}/predict/batch", json={"articles": articles})
    response.raise_for_status()
    return response.json()

# Load our test data
df = pd.read_csv("test_data/dashboard_test.csv")
original_count = len(df)

df['article_id'] = df['article_id'].astype(str)
df['title'] = df['title'].fillna("").astype(str)
df['text'] = df['text'].fillna("")
df['text'] = df['text'].astype(str)

valid_df = df[df['text'].str.strip() != ""]
skipped_count = original_count - len(valid_df)

print(f"Original count: {original_count}")
print(f"Skipped count (empty text): {skipped_count}")

# Simulate chunking
max_batch_size = 2 # Set to 2 to test chunking on 4 rows
all_results = []
total_processed = 0
total_failed = 0
warnings = set()

import math
num_chunks = math.ceil(len(valid_df) / max_batch_size)
print(f"Num chunks: {num_chunks}")

for i in range(num_chunks):
    chunk = valid_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
    resp_data = process_batch(chunk[['article_id', 'title', 'text']])
    all_results.extend(resp_data["results"])
    summary = resp_data["summary"]
    total_processed += summary["total_processed"]
    total_failed += summary["total_failed"]
    if summary.get("warnings"):
        for w in summary["warnings"]:
            warnings.add(w)

print(f"Total processed: {total_processed}")
print(f"Total failed: {total_failed}")
print(f"Warnings: {warnings}")
print(f"Results length: {len(all_results)}")
print(f"Results: {all_results}")
