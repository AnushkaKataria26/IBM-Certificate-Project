"""
clean_text.py — Text preprocessing for fake news detection.

Provides clean_text() for individual strings and clean_series() for
batch processing a pandas Series.

Apostrophe convention:
    We STRIP all apostrophes and non-alphanumeric characters (except spaces).
    Rationale: lemmatization works on base forms (e.g. "don't" → after
    apostrophe removal becomes "dont" → lemmatizer won't split it, but
    consistent removal avoids mixed representations like "don't" vs "dont"
    vs "don t" appearing as separate tokens). The small loss of contraction
    information is acceptable for a bag-of-words/TF-IDF baseline.
"""

import re
import logging
import math
from typing import Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-load NLTK resources (downloaded once, then cached)
# ---------------------------------------------------------------------------
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Compiled regex patterns (compile once, reuse)
# ---------------------------------------------------------------------------
# URLs: http(s)://... or www.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Wire-service prefixes: "CITY (Reuters) -" or "CITY (AP) -" etc.
_WIRE_PREFIX_RE = re.compile(
    r"^[A-Z][A-Za-z\s/]+\(Reuters\)\s*-?\s*", re.MULTILINE
)

# Bracketed source tags like [Reuters], [AP], [Source]
_BRACKETED_TAG_RE = re.compile(r"\[Reuters\]|\[AP\]|\[Source\]", re.IGNORECASE)

# Non-alphanumeric except spaces (apostrophes are removed too — see docstring)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

# Collapse multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text, remove_stopwords: bool = True) -> str:
    """Clean a single text string for NLP processing.

    Parameters
    ----------
    text : any
        Input text. Handles None, NaN (float), empty string, and
        non-string types defensively — all return "".
    remove_stopwords : bool, default True
        Whether to remove NLTK English stopwords. Setting to False
        preserves stylistic signal (e.g. pronoun usage patterns) that
        may be useful for some models.

    Returns
    -------
    str
        Cleaned, lowercased, lemmatized text. May be empty string if
        the input was entirely non-textual content.
    """
    # ------------------------------------------------------------------
    # Defensive input handling
    # ------------------------------------------------------------------
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    if text.strip() == "":
        return ""

    cleaned = text

    # ------------------------------------------------------------------
    # Strip wire-service / boilerplate artifacts
    # ------------------------------------------------------------------
    # These flags are used by clean_series() to aggregate counts
    cleaned = _WIRE_PREFIX_RE.sub("", cleaned)
    cleaned = _BRACKETED_TAG_RE.sub("", cleaned)

    # ------------------------------------------------------------------
    # Remove URLs
    # ------------------------------------------------------------------
    cleaned = _URL_RE.sub("", cleaned)

    # ------------------------------------------------------------------
    # Lowercase
    # ------------------------------------------------------------------
    cleaned = cleaned.lower()

    # ------------------------------------------------------------------
    # Remove punctuation / non-alphanumeric (except spaces)
    # ------------------------------------------------------------------
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)

    # ------------------------------------------------------------------
    # Collapse whitespace
    # ------------------------------------------------------------------
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return ""

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    tokens = word_tokenize(cleaned)

    # ------------------------------------------------------------------
    # Remove stopwords (configurable)
    # ------------------------------------------------------------------
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # ------------------------------------------------------------------
    # Lemmatize
    # ------------------------------------------------------------------
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------
    # Rejoin
    # ------------------------------------------------------------------
    result = " ".join(tokens)
    return result if result else ""


def clean_series(
    series: pd.Series,
    remove_stopwords: bool = True,
    column_name: str = "text",
) -> pd.Series:
    """Apply clean_text() to every element of a pandas Series.

    Logs counts of rows affected by boilerplate/artifact stripping so
    that data modifications are auditable rather than silent.

    Parameters
    ----------
    series : pd.Series
        Raw text series.
    remove_stopwords : bool, default True
        Passed through to clean_text().
    column_name : str, default "text"
        Used only in log messages to identify which column is being cleaned.

    Returns
    -------
    pd.Series
        Cleaned text series (same index as input).
    """
    # Count artifact-affected rows BEFORE cleaning
    wire_count = series.dropna().apply(
        lambda x: bool(_WIRE_PREFIX_RE.search(str(x)))
    ).sum()
    tag_count = series.dropna().apply(
        lambda x: bool(_BRACKETED_TAG_RE.search(str(x)))
    ).sum()
    url_count = series.dropna().apply(
        lambda x: bool(_URL_RE.search(str(x)))
    ).sum()

    logger.info(
        f"[{column_name}] Artifact counts before cleaning: "
        f"wire-service prefixes={wire_count}, "
        f"bracketed tags={tag_count}, "
        f"URLs={url_count}"
    )
    print(
        f"  [{column_name}] Artifact counts: "
        f"wire prefixes={wire_count}, "
        f"bracketed tags={tag_count}, "
        f"URLs={url_count}"
    )

    return series.apply(lambda x: clean_text(x, remove_stopwords=remove_stopwords))
