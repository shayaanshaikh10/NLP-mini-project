"""
features.py - turn cleaned text into numbers with TF-IDF.

Two kinds of features:
  - word n-grams (1-2 words): "bahut", "bahut accha", "NEG_accha"
  - character n-grams (2-5 letters inside word boundaries): "acc", "cch",
    "ccha"... These handle spelling chaos well: "accha", "acha" and "achha"
    share most of their character n-grams even if the normalization map
    misses a variant.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src import config

FEATURE_KINDS = ["word", "char", "both"]


def build_word_vectorizer():
    """TF-IDF over words and word pairs."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=config.WORD_NGRAM_RANGE,
        min_df=config.MIN_DF,
        sublinear_tf=True,      # use 1 + log(count) so repeated words don't dominate
        token_pattern=r"\S+",   # text is already tokenized by preprocess.py
        lowercase=False,        # already lowercase; keeps the NEG_ prefix readable
    )


def build_char_vectorizer():
    """TF-IDF over character n-grams taken inside word boundaries."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.CHAR_NGRAM_RANGE,
        min_df=config.MIN_DF,
        sublinear_tf=True,
        lowercase=False,
    )


def build_features(kind="both"):
    """Return the vectorizer for kind = "word", "char" or "both" (word + char side by side)."""
    if kind == "word":
        return build_word_vectorizer()
    if kind == "char":
        return build_char_vectorizer()
    if kind == "both":
        return FeatureUnion([
            ("word", build_word_vectorizer()),
            ("char", build_char_vectorizer()),
        ])
    raise ValueError(f"Unknown feature kind '{kind}', use one of {FEATURE_KINDS}")
