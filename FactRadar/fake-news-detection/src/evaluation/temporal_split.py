"""
temporal_split.py — Create a chronological holdout split.

Parses the date column (with coercion for unparseable values), sorts
chronologically, and holds out the most recent 15% of rows.

Part of Phase 2: Rigorous Evaluation.
"""

import sys
import os

import pandas as pd
import numpy as np

# Ensure project root is importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


def temporal_split(
    input_path: str = "data/processed/cleaned_dataset.csv",
    output_dir: str = "data/splits",
    holdout_fraction: float = 0.15,
) -> dict:
    """Create a temporal holdout split using the date column.

    Drops rows with unparseable dates for this split only, sorts by date,
    and holds out the most recent `holdout_fraction` rows.

    Returns
    -------
    dict
        Summary of the split (date ranges, row counts, label balance).
    """

    print("=" * 70)
    print("STEP 3: TEMPORAL HOLDOUT SPLIT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"  Total rows: {total:,}")

    # ------------------------------------------------------------------
    # 2. Parse dates with coercion
    # ------------------------------------------------------------------
    print(f"\n--- Date parsing ---")

    # Strip whitespace from date column — many values have trailing spaces
    # that prevent format-based parsing
    df["date_stripped"] = df["date"].fillna("").str.strip()

    # Pass 1: Try default pandas parsing (handles 'December 31, 2017' etc.)
    df["parsed_date"] = pd.to_datetime(df["date_stripped"], errors="coerce")
    pass1_parsed = df["parsed_date"].notna().sum()
    print(f"  Pass 1 (default format): {pass1_parsed:,} parsed")

    # Pass 2: For remaining NaTs, try DD-Mon-YY format ('19-Feb-18')
    # which is used by Real/True articles in this dataset
    nat_mask = df["parsed_date"].isna()
    if nat_mask.any():
        df.loc[nat_mask, "parsed_date"] = pd.to_datetime(
            df.loc[nat_mask, "date_stripped"], format="%d-%b-%y", errors="coerce"
        )
        pass2_parsed = df["parsed_date"].notna().sum() - pass1_parsed
        print(f"  Pass 2 (DD-Mon-YY):      {pass2_parsed:,} additional parsed")

    # Pass 3: For still-remaining NaTs, try 'Mon DD, YYYY' abbreviated
    # format ('Dec 31, 2017') used by many Real/True articles
    nat_mask = df["parsed_date"].isna()
    if nat_mask.any():
        df.loc[nat_mask, "parsed_date"] = pd.to_datetime(
            df.loc[nat_mask, "date_stripped"], format="%b %d, %Y", errors="coerce"
        )
        pass3_parsed = (
            df["parsed_date"].notna().sum() - pass1_parsed - pass2_parsed
        )
        print(f"  Pass 3 (Mon DD, YYYY):   {pass3_parsed:,} additional parsed")

    n_parsed = df["parsed_date"].notna().sum()
    n_nat = df["parsed_date"].isna().sum()
    print(f"  Total parsed: {n_parsed:,}")
    print(f"  Still failed (NaT): {n_nat:,}")

    if n_nat > 0:
        # Show some examples of still-unparseable dates
        unparseable = df[df["parsed_date"].isna()]["date"]
        sample_unparseable = unparseable.head(10).tolist()
        print(f"  Sample still-unparseable values: {sample_unparseable}")
        print(
            f"\n  NOTE: {n_nat:,} rows with unparseable dates are DROPPED "
            f"for this temporal split only. They remain in the main dataset."
        )

    # ------------------------------------------------------------------
    # 3. Filter to parseable dates only
    # ------------------------------------------------------------------
    df_dated = df[df["parsed_date"].notna()].copy()
    print(f"\n  Rows with valid dates: {len(df_dated):,}")

    if len(df_dated) < 1000:
        print(
            f"\n  ⚠️  WARNING: Only {len(df_dated)} rows have valid dates. "
            f"Temporal split may not be meaningful."
        )

    # ------------------------------------------------------------------
    # 4. Sort chronologically and split
    # ------------------------------------------------------------------
    df_dated = df_dated.sort_values("parsed_date").reset_index(drop=True)

    cutoff_idx = int(len(df_dated) * (1 - holdout_fraction))
    df_train_temporal = df_dated.iloc[:cutoff_idx].copy()
    df_holdout = df_dated.iloc[cutoff_idx:].copy()

    # ------------------------------------------------------------------
    # 5. Report date ranges
    # ------------------------------------------------------------------
    print(f"\n--- Temporal Split Statistics ---")
    train_start = df_train_temporal["parsed_date"].min()
    train_end = df_train_temporal["parsed_date"].max()
    holdout_start = df_holdout["parsed_date"].min()
    holdout_end = df_holdout["parsed_date"].max()

    print(f"  Train period:   {train_start.date()} → {train_end.date()}")
    print(f"  Holdout period: {holdout_start.date()} → {holdout_end.date()}")
    print(f"  Cutoff date:    {holdout_start.date()}")

    print(f"\n  Train rows:   {len(df_train_temporal):>7,} ({len(df_train_temporal)/len(df_dated)*100:.1f}%)")
    print(f"  Holdout rows: {len(df_holdout):>7,} ({len(df_holdout)/len(df_dated)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Report label balance
    # ------------------------------------------------------------------
    print(f"\n  Holdout label balance:")
    holdout_label_counts = df_holdout["label"].value_counts().sort_index()
    holdout_label_pcts = df_holdout["label"].value_counts(normalize=True).sort_index()
    for label in sorted(holdout_label_counts.index):
        print(
            f"    Label {label}: {holdout_label_counts[label]:>6,} "
            f"({holdout_label_pcts[label]*100:.1f}%)"
        )

    print(f"\n  Train label balance:")
    train_label_counts = df_train_temporal["label"].value_counts().sort_index()
    train_label_pcts = df_train_temporal["label"].value_counts(normalize=True).sort_index()
    for label in sorted(train_label_counts.index):
        print(
            f"    Label {label}: {train_label_counts[label]:>6,} "
            f"({train_label_pcts[label]*100:.1f}%)"
        )

    # ------------------------------------------------------------------
    # 7. Check for degenerate holdout
    # ------------------------------------------------------------------
    n_classes = df_holdout["label"].nunique()
    holdout_rows = len(df_holdout)
    limitations = []

    if holdout_rows < 500:
        msg = (
            f"⚠️  DATA LIMITATION: Temporal holdout has only {holdout_rows} "
            f"rows (< 500). Metrics may be unreliable."
        )
        print(f"\n  {msg}")
        limitations.append(msg)

    if n_classes < 2:
        msg = (
            f"⚠️  DATA LIMITATION: Temporal holdout contains only one class "
            f"(label={df_holdout['label'].unique()[0]}). "
            f"ROC-AUC and F1-macro will be skipped."
        )
        print(f"\n  {msg}")
        limitations.append(msg)

    if not limitations:
        print(
            f"\n  ✅ Temporal holdout has {holdout_rows:,} rows with "
            f"{n_classes} classes — adequate for evaluation."
        )

    # ------------------------------------------------------------------
    # 8. Save (drop the helper parsed_date column)
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "temporal_test.csv")
    df_holdout_save = df_holdout.drop(columns=["parsed_date"])
    df_holdout_save.to_csv(output_path, index=False)
    print(f"\n  Saved holdout → {output_path} ({len(df_holdout):,} rows)")

    print(f"\n{'='*70}")
    print("END OF TEMPORAL SPLIT")
    print(f"{'='*70}")

    return {
        "parseable_rows": int(n_parsed),
        "unparseable_rows": int(n_nat),
        "train_date_range": f"{train_start.date()} to {train_end.date()}",
        "holdout_date_range": f"{holdout_start.date()} to {holdout_end.date()}",
        "holdout_rows": holdout_rows,
        "train_rows": len(df_train_temporal),
        "holdout_label_balance": holdout_label_counts.to_dict(),
        "limitations": limitations,
    }


if __name__ == "__main__":
    temporal_split()
