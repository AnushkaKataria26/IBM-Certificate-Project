"""
tfidf.py — TF-IDF feature extraction for fake news detection.

Provides build_tfidf_vectorizer() returning a configured TfidfVectorizer.

CRITICAL DATA LEAKAGE NOTE:
    The vectorizer returned by this function must be fit ONLY on the
    training split's clean_text column. Fitting on validation or test data
    would leak information about the test vocabulary into the model,
    inflating evaluation metrics and producing unreliable results.

    In train_baseline.py, the vectorizer is embedded in an sklearn Pipeline
    that calls .fit() exactly once on the training data, then .transform()
    on val/test data — this is the correct pattern.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a configured TfidfVectorizer for the baseline model.

    Configuration rationale
    -----------------------
    ngram_range=(1, 2):
        Captures both unigrams and bigrams. Bigrams add phrase-level
        signal (e.g. "fake news", "breaking news") that unigrams miss.

    max_features=20000:
        Limits vocabulary size to the top 20k features by TF-IDF score.
        Balances expressiveness against memory/compute cost for a baseline.

    min_df=2:
        Excludes terms appearing in only one document. Such terms are
        typically noise, typos, or dataset-specific artifacts rather than
        generalizable signal. They add dimensionality without contributing
        to the model's ability to generalize.

    max_df=0.95:
        Excludes terms appearing in more than 95% of documents. These
        near-universal terms carry no discriminative value between fake
        and real news (analogous to corpus-level stopwords that TF-IDF's
        IDF component would already down-weight, but an explicit cutoff
        is cleaner).

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer ready for .fit() on training data.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.95,
    )
