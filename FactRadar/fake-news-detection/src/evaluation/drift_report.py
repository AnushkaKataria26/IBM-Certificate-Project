import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/monitoring.db")

def generate_report(n_requests=1000):
    if not DB_PATH.exists():
        print(f"Monitoring DB not found at {DB_PATH}. No inferences logged yet.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f'''
            SELECT * FROM inference_log 
            ORDER BY timestamp DESC 
            LIMIT {n_requests}
        ''', conn)
        
    if df.empty:
        print("No inference logs found.")
        return
        
    print(f"--- Drift Report (Last {len(df)} requests) ---")
    
    # Label Distribution
    print("\n1. Label Distribution:")
    label_counts = df['predicted_label'].value_counts(normalize=True) * 100
    for label, pct in label_counts.items():
        print(f"   {label}: {pct:.1f}%")
        
    # Confidence
    print("\n2. Confidence Scores:")
    print(f"   Mean: {df['confidence'].mean():.4f}")
    print(f"   Std:  {df['confidence'].std():.4f}")
    
    # Histogram buckets
    print("   Histogram:")
    bins = pd.cut(df['confidence'], bins=[0.0, 0.5, 0.7, 0.9, 1.0])
    hist = bins.value_counts().sort_index()
    for bin_val, count in hist.items():
        print(f"   {bin_val}: {count}")

    # Latency
    print("\n3. Latency Percentiles (ms):")
    lat_p50 = df['latency_ms'].quantile(0.50)
    lat_p95 = df['latency_ms'].quantile(0.95)
    lat_p99 = df['latency_ms'].quantile(0.99)
    print(f"   p50: {lat_p50:.2f}")
    print(f"   p95: {lat_p95:.2f}")
    print(f"   p99: {lat_p99:.2f}")

if __name__ == "__main__":
    generate_report()
