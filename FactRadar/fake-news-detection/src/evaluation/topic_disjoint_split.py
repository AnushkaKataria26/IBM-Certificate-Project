"""
topic_disjoint_split.py — Create a topic-disjoint holdout split.

Uses the 'subject' column as a proxy for publisher/source grouping.
Holds out entire subject categories so no topic appears in both train
and holdout.

CRITICAL DATA CHARACTERISTIC: In the Kaggle Fake and Real News Dataset,
each subject maps 100% to a single label (e.g., 'politicsNews' → all Real,
'News' → all Fake). This means we MUST hold out subjects from BOTH label
groups to produce a two-class holdout. This is not a bug — it is an
inherent property of this dataset, and we handle it explicitly.

Part of Phase 2: Rigorous Evaluation.
"""

import sys
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# Ensure project root is importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

RANDOM_STATE = 42


def topic_disjoint_split(
    input_path: str = "data/processed/cleaned_dataset.csv",
    output_dir: str = "data/splits",
) -> dict:
    """Create a topic-disjoint holdout split from subject groups.

    Strategy: Since each subject is 100% one label class, we hold out
    subjects from both classes to ensure the holdout has both labels.
    We select subjects to approximate ~20% holdout by row count.

    Returns
    -------
    dict
        Summary of the split (holdout subjects, row counts, label balance).
    """

    print("=" * 70)
    print("STEP 2: TOPIC-DISJOINT SPLIT (PROXY FOR PUBLISHER-DISJOINT)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"  Total rows: {total:,}")

    # ------------------------------------------------------------------
    # 2. Analyze subject-label relationship
    # ------------------------------------------------------------------
    subject_stats = []
    for subj in sorted(df["subject"].unique()):
        subset = df[df["subject"] == subj]
        majority_label = subset["label"].mode().iloc[0]
        label_purity = (subset["label"] == majority_label).mean()
        subject_stats.append({
            "subject": subj,
            "count": len(subset),
            "majority_label": int(majority_label),
            "purity": label_purity,
        })
    subject_df = pd.DataFrame(subject_stats)

    print("\n  Subject-label mapping:")
    for _, row in subject_df.iterrows():
        print(
            f"    {row['subject']:25s} → {row['count']:>6,} rows, "
            f"label={row['majority_label']}, purity={row['purity']*100:.1f}%"
        )

    # Separate subjects by their associated label
    real_subjects = subject_df[subject_df["majority_label"] == 0].copy()
    fake_subjects = subject_df[subject_df["majority_label"] == 1].copy()

    print(f"\n  Real (label=0) subjects: {sorted(real_subjects['subject'].tolist())}")
    print(f"  Fake (label=1) subjects: {sorted(fake_subjects['subject'].tolist())}")

    # ------------------------------------------------------------------
    # 3. Select holdout subjects (one from each label group)
    #
    # Strategy: Pick the smallest subject from each label group to hold
    # out. This keeps the holdout manageable while ensuring both classes
    # are represented. With only 2 real-class subjects and 5 fake-class
    # subjects, we have limited options.
    #
    # Real subjects: politicsNews (11,212), worldnews (9,709)
    # Fake subjects: News (9,050), politics (6,365), US_News (783),
    #                left-news (679), Government News (514)
    #
    # We hold out 'worldnews' (Real) and 'politics' (Fake) to get a
    # sizable holdout (~16k rows, ~42% of data) with both classes.
    # If we held out smaller subjects we'd get a tiny holdout.
    # Alternatively, holding out 'worldnews' + a mid-size fake subject
    # gives ~25-40% holdout. Let's target ~20-25%.
    # ------------------------------------------------------------------

    # Sort each group by count ascending; pick the subject that gives
    # us closest to 20% of total from each label group
    target_holdout_fraction = 0.20
    target_holdout_rows = int(total * target_holdout_fraction)

    # We need at least one subject from each class. Try combinations.
    np.random.seed(RANDOM_STATE)

    real_options = real_subjects.sort_values("count")
    fake_options = fake_subjects.sort_values("count")

    # With only 2 real subjects, we must pick exactly 1 for holdout
    # (holding out both would leave no real articles in train).
    # Pick the smaller real subject for holdout.
    holdout_real_subj = real_options.iloc[0]["subject"]  # smaller one
    holdout_real_count = real_options.iloc[0]["count"]

    # From fake subjects, pick enough to get close to target, but leave
    # at least 1 fake subject in train. Pick by ascending size until
    # we approach the target.
    remaining_budget = target_holdout_rows - holdout_real_count
    holdout_fake_subjs = []
    holdout_fake_count = 0

    for _, row in fake_options.iterrows():
        if holdout_fake_count + row["count"] <= remaining_budget:
            holdout_fake_subjs.append(row["subject"])
            holdout_fake_count += int(row["count"])

    # Ensure we have at least one fake subject
    if len(holdout_fake_subjs) == 0:
        # Take the smallest fake subject regardless
        holdout_fake_subjs = [fake_options.iloc[0]["subject"]]
        holdout_fake_count = int(fake_options.iloc[0]["count"])

    holdout_subjects = [holdout_real_subj] + holdout_fake_subjs
    train_subjects = [
        s for s in df["subject"].unique() if s not in holdout_subjects
    ]

    print(f"\n  Selected holdout subjects: {holdout_subjects}")
    print(f"  Remaining train subjects:  {sorted(train_subjects)}")

    # ------------------------------------------------------------------
    # 4. Create the split
    # ------------------------------------------------------------------
    holdout_mask = df["subject"].isin(holdout_subjects)
    df_holdout = df[holdout_mask].copy()
    df_train = df[~holdout_mask].copy()

    # ------------------------------------------------------------------
    # 5. Verify disjointness
    # ------------------------------------------------------------------
    train_subjs = set(df_train["subject"].unique())
    holdout_subjs = set(df_holdout["subject"].unique())
    overlap = train_subjs & holdout_subjs

    if overlap:
        raise RuntimeError(
            f"TOPIC OVERLAP DETECTED: {overlap}. "
            f"The split is not disjoint. This is a bug."
        )
    print(f"\n  ✅ No subject overlap between train and holdout.")

    # ------------------------------------------------------------------
    # 6. Report statistics
    # ------------------------------------------------------------------
    print(f"\n--- Topic-Disjoint Split Statistics ---")
    print(f"  Train:   {len(df_train):>7,} rows ({len(df_train)/total*100:.1f}%)")
    print(f"  Holdout: {len(df_holdout):>7,} rows ({len(df_holdout)/total*100:.1f}%)")

    print(f"\n  Holdout label balance:")
    holdout_label_counts = df_holdout["label"].value_counts().sort_index()
    holdout_label_pcts = df_holdout["label"].value_counts(normalize=True).sort_index()
    for label in sorted(holdout_label_counts.index):
        print(
            f"    Label {label}: {holdout_label_counts[label]:>6,} "
            f"({holdout_label_pcts[label]*100:.1f}%)"
        )

    print(f"\n  Train label balance:")
    train_label_counts = df_train["label"].value_counts().sort_index()
    train_label_pcts = df_train["label"].value_counts(normalize=True).sort_index()
    for label in sorted(train_label_counts.index):
        print(
            f"    Label {label}: {train_label_counts[label]:>6,} "
            f"({train_label_pcts[label]*100:.1f}%)"
        )

    # ------------------------------------------------------------------
    # 7. Check for degenerate holdout
    # ------------------------------------------------------------------
    n_classes_holdout = df_holdout["label"].nunique()
    holdout_rows = len(df_holdout)
    limitations = []

    if holdout_rows < 500:
        msg = (
            f"⚠️  DATA LIMITATION: Holdout has only {holdout_rows} rows "
            f"(< 500). Metrics may be unreliable due to small sample size."
        )
        print(f"\n  {msg}")
        limitations.append(msg)

    if n_classes_holdout < 2:
        msg = (
            f"⚠️  DATA LIMITATION: Holdout contains only one class "
            f"(label={df_holdout['label'].unique()[0]}). "
            f"ROC-AUC and F1-macro will be skipped in evaluation."
        )
        print(f"\n  {msg}")
        limitations.append(msg)

    if not limitations:
        print(f"\n  ✅ Holdout has {holdout_rows:,} rows with {n_classes_holdout} classes — adequate for evaluation.")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "topic_disjoint_test.csv")
    df_holdout.to_csv(output_path, index=False)
    print(f"\n  Saved holdout → {output_path} ({len(df_holdout):,} rows)")

    print(f"\n{'='*70}")
    print("END OF TOPIC-DISJOINT SPLIT")
    print(f"{'='*70}")

    return {
        "holdout_subjects": holdout_subjects,
        "train_subjects": sorted(train_subjects),
        "holdout_rows": holdout_rows,
        "train_rows": len(df_train),
        "holdout_label_balance": holdout_label_counts.to_dict(),
        "limitations": limitations,
    }


if __name__ == "__main__":
    topic_disjoint_split()
