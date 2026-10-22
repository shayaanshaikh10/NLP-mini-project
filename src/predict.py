"""
predict.py - predict the sentiment of any Hinglish sentence.

Example:
    from src.predict import predict_sentiment
    predict_sentiment("movie bahut mast thi")
    # {'label': 'positive', 'confidence': 0.83,
    #  'probabilities': {'negative': 0.05, 'neutral': 0.12, 'positive': 0.83},
    #  'cleaned_text': 'movie bahut mast thi', 'note': ''}

Run:  python -m src.predict "your sentence here"
"""

import sys

import joblib

from src import config
from src.preprocess import clean_text

# Which file each model name loads. "baseline" = the best model from training.
MODEL_FILES = {
    "baseline": config.BEST_MODEL_PATH,
    "naive_bayes": config.MODELS_DIR / "naive_bayes.joblib",
    "logistic_regression": config.MODELS_DIR / "logistic_regression.joblib",
    "linear_svm": config.MODELS_DIR / "linear_svm.joblib",
}
_loaded_models = {}  # simple cache so each model file is read only once


def transformer_is_available():
    """True if a fine-tuned transformer is saved in models/transformer/ and transformers is installed."""
    if not (config.TRANSFORMER_DIR / "config.json").exists():
        return False
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def available_models():
    """List the model names that can be used right now."""
    names = [name for name, path in MODEL_FILES.items() if path.exists()]
    if transformer_is_available():
        names.append("transformer")
    return names


def load_model(model="baseline"):
    """Load (and cache) a model by name. Raises a clear error if the file is missing."""
    if model in _loaded_models:
        return _loaded_models[model]
    if model == "transformer":
        if not transformer_is_available():
            raise FileNotFoundError("No transformer in models/transformer/ (or transformers not installed).")
        from transformers import pipeline
        loaded = pipeline("text-classification", model=str(config.TRANSFORMER_DIR), top_k=None)
    else:
        if model not in MODEL_FILES:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(MODEL_FILES)}")
        if not MODEL_FILES[model].exists():
            raise FileNotFoundError(f"{MODEL_FILES[model]} not found. "
                                    "Run 'python -m src.train_baselines' first.")
        loaded = joblib.load(MODEL_FILES[model])
    _loaded_models[model] = loaded
    return loaded


def empty_result(note):
    """Result returned when there is nothing to analyze (equal probabilities)."""
    share = round(1 / len(config.LABELS), 4)
    return {"label": "neutral", "confidence": share,
            "probabilities": {label: share for label in config.LABELS},
            "cleaned_text": "", "note": note}


def baseline_probabilities(pipeline, text):
    """Return {label: probability} from a scikit-learn pipeline."""
    probabilities = pipeline.predict_proba([text])[0]
    return {str(label): float(p) for label, p in zip(pipeline.classes_, probabilities)}


def transformer_probabilities(classifier, text):
    """Return {label: probability} from a Hugging Face text-classification pipeline."""
    scores = classifier([text])[0]
    return {item["label"]: float(item["score"]) for item in scores}


def predict_sentiment(text, model="baseline"):
    """Predict negative / neutral / positive for one text.

    Returns a dict with label, confidence, probabilities (sum to 1),
    the cleaned text the model saw, and a note for unusual input.
    """
    if text is None or not str(text).strip():
        return empty_result("Empty input - please type a sentence.")
    cleaned = clean_text(text)
    if not cleaned.strip():
        return empty_result("Nothing left after cleaning (only a link or mention?).")

    loaded = load_model(model)
    if model == "transformer":
        probabilities = transformer_probabilities(loaded, str(text))
    else:
        probabilities = baseline_probabilities(loaded, str(text))
    label = max(probabilities, key=probabilities.get)
    note = "Very short text - the prediction is less reliable." if len(cleaned.split()) <= 2 else ""
    return {"label": label, "confidence": round(probabilities[label], 4),
            "probabilities": {lab: round(probabilities.get(lab, 0.0), 4) for lab in config.LABELS},
            "cleaned_text": cleaned, "note": note}


def predict_many(texts, model="baseline"):
    """Predict a list of texts at once. Returns a list of (label, confidence) tuples."""
    return [(r["label"], r["confidence"]) for r in (predict_sentiment(t, model) for t in texts)]


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) or "movie ekdum mast thi, but climax thoda boring tha"
    print(sentence)
    print(predict_sentiment(sentence))
