"""
inspect_metadata.py — Inspect dataset metadata for disjoint-split feasibility.

Checks whether source_file can serve as a publisher proxy (spoiler: it cannot
in the Kaggle Fake and Real News Dataset — it's collinear with label).
Reports subject column distribution as the alternative grouping variable.

Part of Phase 2: Rigorous Evaluation.
"""

import sys
import os

import pandas as pd

# Ensure project root is importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


def inspect_metadata(
    input_path: str = "data/processed/cleaned_dataset.csv",
) -> dict:
    """Inspect dataset columns for disjoint-split feasibility.

    Returns
    -------
    dict
        Summary of findings (source_file_usable, subject_values, etc.).
    """

    print("=" * 70)
    print("STEP 1: METADATA INSPECTION FOR DISJOINT SPLITTING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and report schema
    # ------------------------------------------------------------------
    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Check source_file for publisher-disjoint feasibility
    # ------------------------------------------------------------------
    print(f"\n--- source_file analysis ---")
    source_unique = df["source_file"].unique()
    print(f"  Unique values ({len(source_unique)}): {sorted(source_unique)}")

    source_label_crosstab = pd.crosstab(
        df["source_file"], df["label"], margins=True
    )
    print(f"\n  source_file × label cross-tabulation:")
    print(source_label_crosstab.to_string(index=True))

    # Determine if source_file is collinear with label
    source_file_usable = True
    if len(source_unique) == 2:
        # Check if each source_file value maps to exactly one label
        groups = df.groupby("source_file")["label"].nunique()
        if (groups == 1).all():
            source_file_usable = False
            print(
                f"\n  ⚠️  source_file has exactly 2 unique values and each "
                f"maps 1:1 with label."
            )
            print(
                f"     This column is COLLINEAR with label — it cannot be "
                f"used for publisher-disjoint splitting."
            )
            print(
                f"     A 'publisher-disjoint' split based on source_file "
                f"would be meaningless (it would just separate by class)."
            )
    elif len(source_unique) <= 2:
        source_file_usable = False
        print(f"\n  ⚠️  source_file has ≤2 unique values — not useful.")

    if source_file_usable:
        print(f"\n  ✅ source_file has >2 values and is not collinear with label.")
    else:
        print(
            f"\n  KNOWN LIMITATION: True publisher-disjoint evaluation is "
            f"NOT possible with this dataset."
        )
        print(
            f"  The Kaggle Fake and Real News Dataset does not contain a "
            f"per-article publisher/source-domain column."
        )
        print(
            f"  EXTENSION PATH: Use FakeNewsNet or another dataset with "
            f"actual source-domain metadata for genuine publisher-disjoint "
            f"evaluation."
        )

    # ------------------------------------------------------------------
    # 3. Report subject column as proxy grouping variable
    # ------------------------------------------------------------------
    print(f"\n--- subject column analysis ---")
    if "subject" not in df.columns:
        print("  ❌ No 'subject' column found. Cannot create topic-disjoint split.")
        return {
            "source_file_usable": source_file_usable,
            "subject_available": False,
        }

    subject_values = sorted(df["subject"].unique())
    print(f"  Unique subjects ({len(subject_values)}): {subject_values}")

    print(f"\n  subject × label distribution:")
    subject_label = pd.crosstab(
        df["subject"], df["label"], margins=True, margins_name="Total"
    )
    # Add percentage column
    subject_counts = df["subject"].value_counts().sort_index()
    print(subject_label.to_string(index=True))

    print(f"\n  Per-subject breakdown:")
    for subj in subject_values:
        subset = df[df["subject"] == subj]
        n = len(subset)
        label_dist = subset["label"].value_counts().sort_index()
        pcts = subset["label"].value_counts(normalize=True).sort_index()
        parts = []
        for lbl in sorted(label_dist.index):
            parts.append(f"label={lbl}: {label_dist[lbl]:,} ({pcts[lbl]*100:.1f}%)")
        print(f"    {subj:25s} → {n:>6,} rows  |  {', '.join(parts)}")

    # ------------------------------------------------------------------
    # 4. Date column quick check
    # ------------------------------------------------------------------
    print(f"\n--- date column quick check ---")
    if "date" in df.columns:
        non_null_dates = df["date"].notna().sum()
        null_dates = df["date"].isna().sum()
        print(f"  Non-null date values: {non_null_dates:,}")
        print(f"  Null date values: {null_dates:,}")
        print(f"  Sample values: {df['date'].dropna().head(5).tolist()}")
    else:
        print("  ❌ No 'date' column found.")

    print(f"\n{'='*70}")
    print("END OF METADATA INSPECTION")
    print(f"{'='*70}")

    return {
        "source_file_usable": source_file_usable,
        "subject_available": True,
        "subject_values": subject_values,
        "total_rows": len(df),
    }


if __name__ == "__main__":
    inspect_metadata()
