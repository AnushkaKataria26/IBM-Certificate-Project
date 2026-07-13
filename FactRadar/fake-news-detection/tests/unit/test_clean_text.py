"""
Unit tests for src/preprocessing/clean_text.py

Covers: normal text, empty string, None, NaN, pure punctuation,
embedded URLs, mixed case + extra whitespace.
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on sys.path so imports work when running pytest
# from any directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.clean_text import clean_text


class TestCleanTextNormal:
    """Test normal text inputs."""

    def test_basic_sentence(self):
        result = clean_text("The quick brown fox jumps over the lazy dog.")
        # Stopwords removed by default, lemmatized, lowercased
        assert isinstance(result, str)
        assert len(result) > 0
        # "the" is a stopword, should be removed
        assert "the" not in result.split()
        # "jumps" should be lemmatized to "jump"
        assert "jump" in result.split()

    def test_lowercase(self):
        result = clean_text("HELLO WORLD", remove_stopwords=False)
        assert result == "hello world"

    def test_stopwords_configurable_off(self):
        result = clean_text("The cat is on the mat", remove_stopwords=False)
        assert "the" in result.split()
        assert "is" in result.split()
        assert "on" in result.split()

    def test_stopwords_configurable_on(self):
        result = clean_text("The cat is on the mat", remove_stopwords=True)
        assert "the" not in result.split()
        assert "is" not in result.split()
        assert "cat" in result.split()
        assert "mat" in result.split()


class TestCleanTextEdgeCases:
    """Test defensive handling of non-string and edge-case inputs."""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_nan_input(self):
        assert clean_text(np.nan) == ""

    def test_float_nan(self):
        assert clean_text(float("nan")) == ""

    def test_whitespace_only(self):
        assert clean_text("   \t\n  ") == ""

    def test_pure_punctuation(self):
        result = clean_text("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result == ""

    def test_pure_punctuation_with_spaces(self):
        result = clean_text("!!! ??? ... --- +++")
        assert result == ""


class TestCleanTextURLs:
    """Test URL removal."""

    def test_http_url(self):
        result = clean_text("Visit http://example.com for more info")
        assert "http" not in result
        assert "example" not in result
        assert "visit" in result.split()

    def test_https_url(self):
        result = clean_text("Check https://www.example.com/page?q=1 now")
        assert "https" not in result
        assert "example" not in result

    def test_www_url(self):
        result = clean_text("Go to www.example.com today")
        assert "www" not in result
        assert "example" not in result

    def test_text_is_only_url(self):
        result = clean_text("https://example.com")
        assert result == ""


class TestCleanTextMixedCaseWhitespace:
    """Test mixed case and extra whitespace handling."""

    def test_mixed_case(self):
        result = clean_text("ThIs Is MiXeD CaSe TeXt", remove_stopwords=False)
        # Should be all lowercase
        assert result == result.lower()

    def test_extra_whitespace(self):
        result = clean_text("  hello    world   ", remove_stopwords=False)
        assert "  " not in result  # No double spaces
        assert result == "hello world"

    def test_tabs_and_newlines(self):
        result = clean_text("hello\t\tworld\n\nnew line", remove_stopwords=False)
        assert "\t" not in result
        assert "\n" not in result


class TestCleanTextArtifacts:
    """Test wire-service prefix and bracketed tag removal."""

    def test_reuters_prefix(self):
        result = clean_text(
            "WASHINGTON (Reuters) - The president spoke today",
            remove_stopwords=False,
        )
        assert "reuters" not in result
        assert "president" in result.split()

    def test_bracketed_reuters_tag(self):
        result = clean_text(
            "Breaking news [Reuters] about the election",
            remove_stopwords=False,
        )
        assert "reuters" not in result
        assert "breaking" in result.split()
