"""
load_dataset.py — Step 9: Build canonical schema from raw CSVs.

Loads Fake.csv and True.csv, adds:
  - label column: 1 = fake, 0 = real (integer convention used consistently)
  - source_file column: tracks origin file
  - article_id column: sequential integer starting from 0

Saves combined DataFrame to data/processed/combined_raw.csv.
"""

import sys
import os
import pandas as pd


def load_and_combine(data_dir: str = "data/raw",
                     output_dir: str = "data/processed") -> pd.DataFrame:
    """Load raw CSVs, add schema columns, concatenate, and save."""

    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("Loading raw CSVs...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # ------------------------------------------------------------------
    # 2. Add label column (1 = fake, 0 = real)
    # ------------------------------------------------------------------
    df_fake["label"] = 1
    df_true["label"] = 0

    # ------------------------------------------------------------------
    # 3. Add source_file column
    # ------------------------------------------------------------------
    df_fake["source_file"] = "Fake.csv"
    df_true["source_file"] = "True.csv"

    # ------------------------------------------------------------------
    # 4. Concatenate
    # ------------------------------------------------------------------
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Add article_id (sequential integer)
    # ------------------------------------------------------------------
    df_combined["article_id"] = range(len(df_combined))

    # ------------------------------------------------------------------
    # 6. Reorder columns for clarity
    # ------------------------------------------------------------------
    col_order = ["article_id", "label", "source_file", "title", "text", "subject", "date"]
    # Only reorder columns that exist (defensive)
    col_order = [c for c in col_order if c in df_combined.columns]
    remaining = [c for c in df_combined.columns if c not in col_order]
    df_combined = df_combined[col_order + remaining]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "combined_raw.csv")
    df_combined.to_csv(output_path, index=False)
    print(f"Saved combined dataset to: {output_path}")

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    print(f"\nFinal shape: {df_combined.shape}")
    print(f"\nLabel distribution:")
    label_counts = df_combined["label"].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "fake" if label_val == 1 else "real"
        pct = count / len(df_combined) * 100
        print(f"  {label_val} ({label_name}): {count:,} ({pct:.1f}%)")

    # Check for unexpected nulls introduced during merge
    null_counts = df_combined.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        print("\n✅ No unexpected nulls introduced during merge.")
    else:
        print(f"\n⚠️  Columns with nulls after merge:")
        for col, count in cols_with_nulls.items():
            print(f"    {col}: {count}")

    return df_combined


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    load_and_combine(data_dir, output_dir)
