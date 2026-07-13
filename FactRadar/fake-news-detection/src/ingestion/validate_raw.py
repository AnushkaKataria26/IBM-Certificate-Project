"""
validate_raw.py — Step 8: Validate raw Fake.csv and True.csv data files.

Checks:
  - Expected columns exist (title, text, subject, date)
  - Row counts per file
  - Null/empty text and title fields
  - Duplicate rows within each file (by exact text match)
  - Cross-file duplicates (same text in both Fake.csv and True.csv)
  - Class balance report

Does NOT drop any rows — only reports issues.
"""

import sys
import os
import pandas as pd


def validate_raw(data_dir: str = "data/raw") -> bool:
    """Validate Fake.csv and True.csv. Returns True if validation passes."""

    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    # ------------------------------------------------------------------
    # 1. Check files exist
    # ------------------------------------------------------------------
    for path in [fake_path, true_path]:
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            return False

    # ------------------------------------------------------------------
    # 2. Load CSVs
    # ------------------------------------------------------------------
    print("Loading CSVs...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    expected_columns = {"title", "text", "subject", "date"}

    # ------------------------------------------------------------------
    # 3. Confirm expected columns
    # ------------------------------------------------------------------
    for name, df in [("Fake.csv", df_fake), ("True.csv", df_true)]:
        actual = set(df.columns)
        if actual != expected_columns:
            missing = expected_columns - actual
            extra = actual - expected_columns
            print(f"WARNING: Column mismatch in {name}!")
            if missing:
                print(f"  Missing columns: {missing}")
            if extra:
                print(f"  Extra columns:   {extra}")
        else:
            print(f"  {name}: columns OK — {list(df.columns)}")

    # ------------------------------------------------------------------
    # 4. Row counts
    # ------------------------------------------------------------------
    print(f"\nRow counts:")
    print(f"  Fake.csv: {len(df_fake):,} rows")
    print(f"  True.csv: {len(df_true):,} rows")
    print(f"  Total:    {len(df_fake) + len(df_true):,} rows")

    # ------------------------------------------------------------------
    # 5. Null / empty checks
    # ------------------------------------------------------------------
    print("\nNull / empty field checks:")
    for name, df in [("Fake.csv", df_fake), ("True.csv", df_true)]:
        null_text = df["text"].isna().sum()
        empty_text = (df["text"].astype(str).str.strip() == "").sum()
        null_title = df["title"].isna().sum()
        empty_title = (df["title"].astype(str).str.strip() == "").sum()
        print(f"  {name}:")
        print(f"    null text:  {null_text}  |  empty text:  {empty_text}")
        print(f"    null title: {null_title}  |  empty title: {empty_title}")

    # ------------------------------------------------------------------
    # 6. Intra-file duplicates (by exact text match)
    # ------------------------------------------------------------------
    print("\nDuplicate rows (by exact text match):")
    for name, df in [("Fake.csv", df_fake), ("True.csv", df_true)]:
        dup_count = df.duplicated(subset=["text"], keep=False).sum()
        unique_dup_count = df.duplicated(subset=["text"], keep="first").sum()
        print(f"  {name}: {dup_count} rows involved in duplicates "
              f"({unique_dup_count} duplicate copies)")

    # ------------------------------------------------------------------
    # 7. Cross-file duplicates (same article in both files)
    # ------------------------------------------------------------------
    print("\nCross-file duplicates (same text in both Fake.csv and True.csv):")
    common_texts = set(df_fake["text"].dropna()) & set(df_true["text"].dropna())
    print(f"  {len(common_texts)} articles found in BOTH files (potential labeling errors)")

    # ------------------------------------------------------------------
    # 8. Class balance
    # ------------------------------------------------------------------
    print("\nClass balance:")
    total = len(df_fake) + len(df_true)
    print(f"  Fake: {len(df_fake):,} ({len(df_fake)/total*100:.1f}%)")
    print(f"  Real: {len(df_true):,} ({len(df_true)/total*100:.1f}%)")

    print("\n✅ Validation complete — no rows were dropped.")
    return True


if __name__ == "__main__":
    # Allow overriding data directory from command line
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    success = validate_raw(data_dir)
    sys.exit(0 if success else 1)
