"""
train_baselines.py - train and compare the classical models.

What it does:
  1. Ablation study (Logistic Regression, C=1): which features and which
     preprocessing work best? Chosen by VALIDATION macro-F1.
  2. Tune each model (Naive Bayes, Logistic Regression, Linear SVM) on the
     validation set using the best setup from step 1.
  3. Retrain every model as a full Pipeline (cleaner -> TF-IDF -> model)
     on the training set and save it to models/. The model with the best
     validation macro-F1 is also saved as models/best_baseline.joblib.

The test set is only used to REPORT ablation scores, never to choose anything.

Run:  python -m src.train_baselines
"""

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.features import build_features
from src.preprocess import TextCleaner, clean_text
from src.utils import Timer, log_step, save_json

MODEL_NAMES = ["naive_bayes", "logistic_regression", "linear_svm"]

# Ablation variants: (name, feature kind, use normalization map, negation scope)
ABLATION_VARIANTS = [
    ("word only", "word", True, None),
    ("word only, no normalization", "word", False, None),
    ("char only", "char", True, None),
    ("word + char", "both", True, None),
    ("word + char, no normalization", "both", False, None),
    ("word + char + NEG_ (after negator)", "both", True, "after"),
    ("word + char + NEG_ (both sides)", "both", True, "both"),
]


def load_split(path):
    """Read a processed CSV and return (texts, labels) as lists."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m src.data_loader' first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return list(df[config.TEXT_COL]), list(df[config.LABEL_COL])


def clean_all(texts, use_normalization=True, negation_scope=None):
    """Clean a list of texts with the given preprocessing settings."""
    return [clean_text(text, use_normalization, negation_scope) for text in texts]


def make_model(name, param):
    """Create an untrained classifier. param is alpha for NB and C for LR/SVM."""
    if name == "naive_bayes":
        return MultinomialNB(alpha=param)
    if name == "logistic_regression":
        return LogisticRegression(C=param, max_iter=config.LR_MAX_ITER,
                                  class_weight="balanced", random_state=config.RANDOM_STATE)
    if name == "linear_svm":
        return LinearSVC(C=param, class_weight="balanced", random_state=config.RANDOM_STATE)
    raise ValueError(f"Unknown model name: {name}")


def param_grid(name):
    """Return the small list of values to try for one model."""
    grids = {"naive_bayes": config.NB_ALPHAS,
             "logistic_regression": config.LR_CS,
             "linear_svm": config.SVM_CS}
    return grids[name]


def macro_f1(true_labels, predicted_labels):
    """Macro-F1: the F1 of each class averaged with equal weight (our main metric)."""
    return f1_score(true_labels, predicted_labels, average="macro")


# ---------------------------------------------------------------------------
# Step 1: ablation study
# ---------------------------------------------------------------------------
def run_one_ablation(variant, data):
    """Train LR (C=1) for one ablation variant and return a result row."""
    name, kind, use_norm, negation = variant
    train_x = clean_all(data["train_x"], use_norm, negation)
    vectorizer = build_features(kind)
    model = make_model("logistic_regression", 1.0)
    with Timer() as timer:
        model.fit(vectorizer.fit_transform(train_x), data["train_y"])
    val_pred = model.predict(vectorizer.transform(clean_all(data["val_x"], use_norm, negation)))
    test_pred = model.predict(vectorizer.transform(clean_all(data["test_x"], use_norm, negation)))
    return {"variant": name, "features": kind, "normalization": use_norm,
            "negation": negation or "off",
            "val_macro_f1": round(macro_f1(data["val_y"], val_pred), 4),
            "test_macro_f1": round(macro_f1(data["test_y"], test_pred), 4),
            "test_accuracy": round(accuracy_score(data["test_y"], test_pred), 4),
            "train_time_s": round(timer.seconds, 2)}


def run_ablation(data):
    """Run every ablation variant, save results/ablation.csv and return the table."""
    rows = []
    for variant in ABLATION_VARIANTS:
        row = run_one_ablation(variant, data)
        print(f"  {row['variant']:<38} val macro-F1 = {row['val_macro_f1']:.4f}")
        rows.append(row)
    table = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.ABLATION_PATH, index=False)
    return table


def choose_setup(ablation_table):
    """Pick the preprocessing + feature setup with the best VALIDATION score."""
    best = ablation_table.sort_values("val_macro_f1", ascending=False).iloc[0]
    negation = None if best["negation"] == "off" else best["negation"]
    return {"variant": best["variant"], "features": best["features"],
            "use_normalization": bool(best["normalization"]), "negation_scope": negation}


# ---------------------------------------------------------------------------
# Step 2: hyperparameter tuning on the validation set
# ---------------------------------------------------------------------------
def tune_model(name, train_matrix, train_y, val_matrix, val_y):
    """Try every value in the grid and return (best_param, best_val_macro_f1)."""
    best_param, best_score = None, -1.0
    for param in param_grid(name):
        model = make_model(name, param)
        model.fit(train_matrix, train_y)
        score = macro_f1(val_y, model.predict(val_matrix))
        print(f"  {name:<20} param={param:<5} val macro-F1 = {score:.4f}")
        if score > best_score:
            best_param, best_score = param, score
    return best_param, best_score


def tune_all_models(data, setup):
    """Vectorize once with the chosen setup, then tune each model. Returns a dict."""
    train_x = clean_all(data["train_x"], setup["use_normalization"], setup["negation_scope"])
    val_x = clean_all(data["val_x"], setup["use_normalization"], setup["negation_scope"])
    vectorizer = build_features(setup["features"])
    train_matrix = vectorizer.fit_transform(train_x)
    val_matrix = vectorizer.transform(val_x)
    results = {}
    for name in MODEL_NAMES:
        best_param, best_score = tune_model(name, train_matrix, data["train_y"],
                                            val_matrix, data["val_y"])
        results[name] = {"best_param": best_param, "val_macro_f1": round(best_score, 4)}
    return results


# ---------------------------------------------------------------------------
# Step 3: final pipelines
# ---------------------------------------------------------------------------
def build_pipeline(name, param, setup):
    """Full model: raw text -> TextCleaner -> TF-IDF -> classifier.

    LinearSVC has no predict_proba, so it is wrapped in CalibratedClassifierCV,
    which learns to turn its scores into probabilities (needed for confidence).
    """
    classifier = make_model(name, param)
    if name == "linear_svm":
        classifier = CalibratedClassifierCV(classifier, cv=3)
    return Pipeline([
        ("cleaner", TextCleaner(setup["use_normalization"], setup["negation_scope"])),
        ("features", build_features(setup["features"])),
        ("model", classifier),
    ])


def train_and_save(name, param, setup, data):
    """Fit one final pipeline on the training set, save it, return (seconds, path)."""
    pipeline = build_pipeline(name, param, setup)
    with Timer() as timer:
        pipeline.fit(data["train_x"], data["train_y"])
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return timer.seconds, path


def load_all_data():
    """Load train/val/test into one dict of lists."""
    train_x, train_y = load_split(config.TRAIN_PATH)
    val_x, val_y = load_split(config.VAL_PATH)
    test_x, test_y = load_split(config.TEST_PATH)
    return {"train_x": train_x, "train_y": train_y, "val_x": val_x,
            "val_y": val_y, "test_x": test_x, "test_y": test_y}


def main():
    """Ablation -> tuning -> final training -> save models and training_info.json."""
    data = load_all_data()
    print("Step 1: ablation study (Logistic Regression, C=1)")
    ablation_table = run_ablation(data)
    setup = choose_setup(ablation_table)
    print(f"Chosen setup (best on validation): {setup['variant']}")

    print("\nStep 2: tuning each model on the validation set")
    tuning = tune_all_models(data, setup)

    print("\nStep 3: training final pipelines on the training set")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        seconds, path = train_and_save(name, tuning[name]["best_param"], setup, data)
        tuning[name].update({"train_time_s": round(seconds, 2), "path": path.name})
        print(f"  {name:<20} trained in {seconds:.1f}s -> {path.name}")

    best_name = max(MODEL_NAMES, key=lambda name: tuning[name]["val_macro_f1"])
    joblib.dump(joblib.load(config.MODELS_DIR / f"{best_name}.joblib"), config.BEST_MODEL_PATH)
    save_json({"setup": setup, "models": tuning, "best_model": best_name},
              config.TRAINING_INFO_PATH)
    summary = ", ".join(f"{n}={tuning[n]['val_macro_f1']}" for n in MODEL_NAMES)
    log_step(f"Step 4 Baselines: setup '{setup['variant']}'; val macro-F1 {summary}; "
             f"best = {best_name} -> models/best_baseline.joblib")


if __name__ == "__main__":
    main()
