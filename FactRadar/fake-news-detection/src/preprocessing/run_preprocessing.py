"""
run_preprocessing.py — Apply text cleaning to combined_raw.csv.

Produces data/processed/cleaned_dataset.csv with clean_title and clean_text
columns added. Drops rows with empty clean_text (unusable for training) and
exact duplicates on clean_text.

Does NOT drop rows solely for empty clean_title — title-only emptiness is
logged but retained since the body text is the primary training signal.
"""

import sys
import os
import logging

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.clean_text import clean_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing(
    input_path: str = "data/processed/combined_raw.csv",
    output_path: str = "data/processed/cleaned_dataset.csv",
) -> pd.DataFrame:
    """Load raw dataset, clean text columns, drop empties/duplicates, save."""

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"  Original row count: {original_count:,}")

    # ------------------------------------------------------------------
    # 2. Apply clean_text to title and text columns
    # ------------------------------------------------------------------
    print("\nCleaning 'title' column ...")
    df["clean_title"] = clean_series(df["title"], column_name="title")

    print("\nCleaning 'text' column ...")
    df["clean_text"] = clean_series(df["text"], column_name="text")

    # ------------------------------------------------------------------
    # 3. Report empty counts BEFORE dropping
    # ------------------------------------------------------------------
    empty_text_mask = df["clean_text"].str.strip() == ""
    empty_title_mask = df["clean_title"].str.strip() == ""

    empty_text_count = empty_text_mask.sum()
    empty_title_count = empty_title_mask.sum()

    print(f"\n--- Empty counts after cleaning (before drops) ---")
    print(f"  Rows with empty clean_text:  {empty_text_count:,}")
    print(f"  Rows with empty clean_title: {empty_title_count:,}")

    # ------------------------------------------------------------------
    # 4. Drop rows where clean_text is empty
    # ------------------------------------------------------------------
    df = df[~empty_text_mask].copy()
    after_empty_drop = len(df)
    print(f"\n  Rows after dropping empty clean_text: {after_empty_drop:,} "
          f"(dropped {empty_text_count:,})")

    # ------------------------------------------------------------------
    # 5. Drop exact duplicates on clean_text
    # ------------------------------------------------------------------
    dup_mask = df.duplicated(subset=["clean_text"], keep="first")
    dup_count = dup_mask.sum()
    df = df[~dup_mask].copy()
    after_dedup = len(df)
    print(f"  Rows after dropping duplicates on clean_text: {after_dedup:,} "
          f"(dropped {dup_count:,})")

    # ------------------------------------------------------------------
    # 6. Arithmetic audit
    # ------------------------------------------------------------------
    expected = original_count - empty_text_count - dup_count
    print(f"\n--- Arithmetic audit ---")
    print(f"  {original_count:,} (original) - {empty_text_count:,} (empty text) "
          f"- {dup_count:,} (duplicates) = {expected:,} (expected)")
    print(f"  Actual final count: {after_dedup:,}")
    assert after_dedup == expected, (
        f"Row count mismatch! Expected {expected}, got {after_dedup}. "
        "Investigate drop logic."
    )
    print("  ✅ Arithmetic checks out.")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    return df


if __name__ == "__main__":
    run_preprocessing()
