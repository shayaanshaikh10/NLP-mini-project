"""
Tests for HinglishSense.  Run from the project folder:  pytest
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.preprocess import clean_text, mark_negation  # noqa: E402
from src.utils import load_json  # noqa: E402

MODEL_READY = config.BEST_MODEL_PATH.exists()
DATASET_IS_REAL = not load_json(config.DATASET_INFO_PATH).get("is_demo", True)
needs_model = pytest.mark.skipif(not MODEL_READY, reason="run python -m src.train_baselines first")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_elongation_is_collapsed():
    assert clean_text("mastttttt") == "mast"         # 3+ letters -> 2 -> map -> mast
    assert clean_text("sooooo good") == "so good"


@pytest.mark.parametrize("variant, canonical", [
    ("acha", "accha"), ("achha", "accha"), ("bohot", "bahut"), ("bhot", "bahut"),
    ("nhi", "nahi"), ("nahin", "nahi"), ("bakwaas", "bakwas"), ("jabardast", "zabardast"),
    ("pyaar", "pyar"), ("thora", "thoda"), ("plz", "please"), ("gr8", "great"),
])
def test_spelling_variants_are_normalized(variant, canonical):
    assert clean_text(variant) == canonical


def test_multi_word_variant():
    assert clean_text("ek dum mast") == "ekdum mast"


def test_emoji_becomes_text_token():
    cleaned = clean_text("movie 😍")
    assert "emoji_smiling_face_with_heart_eyes" in cleaned
    assert "emo_pos" in cleaned
    assert "😍" not in cleaned


def test_urls_mentions_and_hashtags():
    cleaned = clean_text("@rahul dekho https://t.co/abc123 #WorldCup")
    assert "rahul" not in cleaned
    assert "http" not in cleaned and "t.co" not in cleaned
    assert cleaned == "dekho worldcup"


def test_html_entities_are_decoded():
    assert clean_text("tom &amp; jerry") == "tom & jerry"


def test_negation_marking():
    tokens = "film accha nahi tha".split()
    assert mark_negation(tokens, "after") == ["film", "accha", "nahi", "NEG_tha"]
    assert mark_negation(tokens, "both") == ["NEG_film", "NEG_accha", "nahi", "NEG_tha"]


def test_clean_text_handles_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@needs_model
def test_prediction_format():
    from src.predict import predict_sentiment
    result = predict_sentiment("movie theek thak thi")
    assert result["label"] in config.LABELS
    assert set(result["probabilities"]) == set(config.LABELS)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
    assert 0.0 <= result["confidence"] <= 1.0


@needs_model
@pytest.mark.parametrize("text", ["", "   ", None, "https://t.co/xyz", "a"])
def test_prediction_does_not_crash_on_odd_input(text):
    from src.predict import predict_sentiment
    result = predict_sentiment(text)
    assert result["label"] in config.LABELS
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


@needs_model
@pytest.mark.parametrize("model_name", ["naive_bayes", "logistic_regression", "linear_svm"])
def test_every_baseline_predicts(model_name):
    from src.predict import predict_sentiment
    assert predict_sentiment("bahut accha laga", model=model_name)["label"] in config.LABELS


@needs_model
@pytest.mark.skipif(not DATASET_IS_REAL, reason="only meaningful with a real dataset")
def test_obvious_positive_and_negative():
    from src.predict import predict_sentiment
    assert predict_sentiment("movie bahut mast thi")["label"] == "positive"
    assert predict_sentiment("ekdum bakwas film, paisa waste")["label"] == "negative"


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
def test_normalization_map_has_150_plus_variants():
    import pandas as pd
    table = pd.read_csv(config.NORMALIZATION_MAP_PATH)
    assert list(table.columns) == ["variant", "canonical"]
    assert len(table) >= 150
    assert table["variant"].is_unique


@pytest.mark.skipif(not config.TRAIN_PATH.exists(), reason="run python -m src.data_loader first")
def test_processed_splits_have_the_right_format():
    import pandas as pd
    for path in [config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH]:
        table = pd.read_csv(path)
        assert list(table.columns) == [config.TEXT_COL, config.LABEL_COL]
        assert set(table[config.LABEL_COL]) <= set(config.LABELS)
