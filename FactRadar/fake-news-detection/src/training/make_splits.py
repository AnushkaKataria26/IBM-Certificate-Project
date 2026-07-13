"""
make_splits.py — Create stratified train/val/test splits.

Produces data/splits/train.csv, data/splits/val.csv, data/splits/test.csv
from data/processed/cleaned_dataset.csv using a 70/15/15 split with
stratification on the label column.

Uses random_state=42 consistently for reproducibility across the project.
"""

import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ---------------------------------------------------------------------------
# Project-wide reproducibility seed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42


def make_splits(
    input_path: str = "data/processed/cleaned_dataset.csv",
    output_dir: str = "data/splits",
) -> dict:
    """Create stratified train/val/test splits.

    Split ratios: 70% train, 15% val, 15% test.
    Verifies label distribution and article_id disjointness.

    Returns
    -------
    dict
        {"train": df_train, "val": df_val, "test": df_test}
    """

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"  Total rows: {total:,}")

    original_dist = df["label"].value_counts(normalize=True).sort_index()
    print(f"\n  Original label distribution:")
    for label, pct in original_dist.items():
        print(f"    Label {label}: {pct * 100:.2f}%")

    # ------------------------------------------------------------------
    # 2. First split: 70% train, 30% temp (which becomes 15+15)
    # ------------------------------------------------------------------
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 3. Second split: 50/50 of the 30% temp → 15% val, 15% test
    # ------------------------------------------------------------------
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["label"],
        random_state=RANDOM_STATE,
    )

    splits = {"train": df_train, "val": df_val, "test": df_test}

    # ------------------------------------------------------------------
    # 4. Verify label distribution within 1 percentage point
    # ------------------------------------------------------------------
    print("\n--- Label distribution per split ---")
    for name, split_df in splits.items():
        dist = split_df["label"].value_counts(normalize=True).sort_index()
        print(f"\n  {name} ({len(split_df):,} rows):")
        for label, pct in dist.items():
            orig_pct = original_dist[label]
            drift = abs(pct - orig_pct) * 100
            status = "✅" if drift <= 1.0 else "❌"
            print(f"    Label {label}: {pct * 100:.2f}% "
                  f"(original: {orig_pct * 100:.2f}%, drift: {drift:.2f}pp) {status}")
            if drift > 1.0:
                raise ValueError(
                    f"Label distribution drift exceeds 1pp in {name} split "
                    f"for label {label}: {drift:.2f}pp. "
                    "Check stratify parameter."
                )

    # ------------------------------------------------------------------
    # 5. Verify no article_id appears in more than one split
    # ------------------------------------------------------------------
    train_ids = set(df_train["article_id"])
    val_ids = set(df_val["article_id"])
    test_ids = set(df_test["article_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    assert len(train_val_overlap) == 0, f"Train/Val overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train/Test overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val/Test overlap: {val_test_overlap}"
    print("\n  ✅ No article_id overlap across splits.")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    for name, split_df in splits.items():
        path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"  Saved {name} → {path} ({len(split_df):,} rows)")

    # ------------------------------------------------------------------
    # 7. Row count audit
    # ------------------------------------------------------------------
    split_total = len(df_train) + len(df_val) + len(df_test)
    assert split_total == total, (
        f"Row count mismatch: splits sum to {split_total}, expected {total}"
    )
    print(f"\n  ✅ Total rows across splits: {split_total:,} == {total:,} original")

    return splits


if __name__ == "__main__":
    make_splits()
