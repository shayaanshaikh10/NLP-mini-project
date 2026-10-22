"""
evaluate.py - test-set evaluation of every trained model.

Creates in results/:
  - comparison.csv            one row per model + one row per ablation variant
  - classification_report_<model>.txt
  - confusion_<model>.png     confusion-matrix heatmaps
  - error_analysis.csv        every test sample the best model got wrong
  - error_patterns.csv        how often typical failure patterns appear in the errors
  - top_features_lr.png       the 15 most positive / negative LR features
  - model_comparison.png, ablation.png

Run:  python -m src.evaluate
"""

import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from src import config
from src.preprocess import clean_text
from src.utils import Timer, load_json, log_step, save_json

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

# Colors: red = negative, gray = neutral, blue = positive (color-blind friendly pair)
LABEL_COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
BAR_COLOR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

# Words that hint at a failure pattern (used to describe the errors)
NEGATION_HINTS = set(config.NEGATION_WORDS)
CONTRAST_HINTS = {"but", "lekin", "par", "magar", "however", "though"}
SARCASM_HINTS = {"wah", "waah", "wow", "great", "slow_clap", "sure", "haha", "laugh_out_loud"}


def set_plot_style():
    """Quiet, readable chart style shared by every plot."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    })


def add_title(fig, title):
    """Left-aligned figure title that never gets cut off by long tick labels."""
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=12)


def load_test_set():
    """Return the test DataFrame (stops with a clear message if missing)."""
    if not config.TEST_PATH.exists():
        raise FileNotFoundError("data/processed/test.csv missing. Run 'python -m src.data_loader'.")
    return pd.read_csv(config.TEST_PATH, dtype=str, keep_default_na=False)


def load_trained_models():
    """Load every baseline pipeline listed in models/training_info.json."""
    info = load_json(config.TRAINING_INFO_PATH)
    if not info:
        raise FileNotFoundError("No trained models. Run 'python -m src.train_baselines' first.")
    models = {}
    for name, details in info["models"].items():
        models[name] = joblib.load(config.MODELS_DIR / details["path"])
    return models, info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate_model(name, pipeline, test_df, train_time):
    """Predict the test set with one model and return (metrics row, predictions, probs)."""
    texts = list(test_df[config.TEXT_COL])
    true_labels = list(test_df[config.LABEL_COL])
    with Timer() as timer:
        probabilities = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    predictions = [classes[index] for index in probabilities.argmax(axis=1)]
    row = {
        "model": name, "type": "model",
        "accuracy": round(accuracy_score(true_labels, predictions), 4),
        "macro_f1": round(f1_score(true_labels, predictions, average="macro"), 4),
        "weighted_f1": round(f1_score(true_labels, predictions, average="weighted"), 4),
        "train_time_s": train_time,
        "inference_ms_per_sample": round(1000 * timer.seconds / len(texts), 3),
        "model_size_mb": round(os.path.getsize(config.MODELS_DIR / f"{name}.joblib") / 1e6, 2),
    }
    return row, predictions, probabilities.max(axis=1)


def save_classification_report(name, true_labels, predictions):
    """Write sklearn's per-class precision/recall/F1 report to a text file and print it."""
    report = classification_report(true_labels, predictions, labels=config.LABELS, digits=4)
    print(f"\n=== {name} ===\n{report}")
    path = config.RESULTS_DIR / f"classification_report_{name}.txt"
    path.write_text(report, encoding="utf-8")


def ablation_rows():
    """Turn results/ablation.csv into rows for the comparison table."""
    if not config.ABLATION_PATH.exists():
        return []
    ablation = pd.read_csv(config.ABLATION_PATH)
    rows = []
    for _, item in ablation.iterrows():
        rows.append({"model": "ablation (LR): " + item["variant"], "type": "ablation",
                     "accuracy": item["test_accuracy"], "macro_f1": item["test_macro_f1"],
                     "weighted_f1": np.nan, "train_time_s": item["train_time_s"],
                     "inference_ms_per_sample": np.nan, "model_size_mb": np.nan})
    return rows


def transformer_row():
    """Read transformer test metrics if the optional model was trained (else None)."""
    metrics = load_json(config.TRANSFORMER_DIR / "metrics.json")
    if not metrics:
        return None
    return {"model": "xlm_roberta", "type": "model",
            "accuracy": round(metrics.get("test_accuracy", np.nan), 4),
            "macro_f1": round(metrics.get("test_macro_f1", np.nan), 4),
            "weighted_f1": np.nan, "train_time_s": metrics.get("train_time_s", np.nan),
            "inference_ms_per_sample": np.nan, "model_size_mb": np.nan}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(name, true_labels, predictions):
    """Heatmap of counts; each cell also shows the share of its true-label row."""
    matrix = confusion_matrix(true_labels, predictions, labels=config.LABELS)
    row_share = matrix / matrix.sum(axis=1, keepdims=True)
    annotations = [[f"{matrix[i, j]}\n{row_share[i, j]:.0%}" for j in range(3)] for i in range(3)]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(row_share, annot=annotations, fmt="", cmap=BLUE_RAMP, vmin=0, vmax=1,
                cbar=False, linewidths=2, linecolor="#fcfcfb", square=True,
                xticklabels=config.LABELS, yticklabels=config.LABELS, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    add_title(fig, f"Confusion matrix: {name.replace('_', ' ')} (test set)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def plot_horizontal_bars(labels, values, title, xlabel, path):
    """One-color horizontal bar chart with the value written at the end of each bar."""
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(labels) + 1.2))
    positions = range(len(labels))
    ax.barh(positions, values, color=BAR_COLOR, height=0.6)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()  # first item on top
    for position, value in zip(positions, values):
        ax.text(value + 0.005, position, f"{value:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel(xlabel)
    add_title(fig, title)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lr_polarity_scores(pipeline):
    """Return a Series: feature name -> (positive weight - negative weight) from LR."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["features"].get_feature_names_out()
    classes = list(model.classes_)
    weights = model.coef_[classes.index("positive")] - model.coef_[classes.index("negative")]
    # "word__bahut accha" -> "bahut accha" ; "char__ mast" -> "'_mast'" (char n-gram)
    readable = []
    for name in names:
        kind, _, feature = name.partition("__")
        readable.append(f"'{feature.replace(' ', '_')}' (char)" if kind == "char" else feature)
    return pd.Series(weights, index=readable)


def plot_top_features(pipeline, top_n=15):
    """Diverging bar chart of the most positive (blue) and most negative (red) LR features."""
    scores = lr_polarity_scores(pipeline).sort_values()
    top = pd.concat([scores.head(top_n), scores.tail(top_n)])
    colors = [LABEL_COLORS["negative"] if value < 0 else LABEL_COLORS["positive"] for value in top]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(top)), top.values, color=colors, height=0.65)
    ax.set_yticks(range(len(top)), top.index, fontsize=9)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_xlabel("Weight towards positive  (positive-class minus negative-class coefficient)")
    add_title(fig, f"Logistic Regression: top {top_n} positive (blue) and negative (red) features")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "top_features_lr.png", dpi=150)
    plt.close(fig)
    return top


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def word_vocabulary(pipeline):
    """Return the set of words the word-level TF-IDF learned (empty set if none)."""
    features = pipeline.named_steps["features"]
    if hasattr(features, "transformer_list"):  # FeatureUnion: look for the word part
        for name, vectorizer in features.transformer_list:
            if name == "word":
                return set(vectorizer.vocabulary_)
        return set()
    if features.analyzer == "word":
        return set(features.vocabulary_)
    return set()


def find_patterns(cleaned_text, true_label, predicted_label, vocabulary):
    """Return a dict of True/False flags for typical failure patterns."""
    tokens = cleaned_text.split()
    words = [token for token in tokens if token[0].isalnum()]
    unknown = [word for word in words if vocabulary and word not in vocabulary]
    return {
        "negation": any(token in NEGATION_HINTS for token in tokens),
        "mixed_sentiment": any(token in CONTRAST_HINTS for token in tokens),
        "possible_sarcasm": any(token in SARCASM_HINTS for token in tokens),
        "very_short_text": len(words) <= 4,
        "many_unknown_words": len(words) > 0 and len(unknown) / len(words) >= 0.3,
        "neutral_vs_positive": {true_label, predicted_label} == {"neutral", "positive"},
        "neutral_vs_negative": {true_label, predicted_label} == {"neutral", "negative"},
        "positive_vs_negative": {true_label, predicted_label} == {"positive", "negative"},
    }


def build_error_analysis(pipeline, test_df, predictions, confidences):
    """Save every misclassified test sample and a pattern summary. Returns the summary."""
    vocabulary = word_vocabulary(pipeline)
    rows = []
    for index, raw_text in enumerate(test_df[config.TEXT_COL]):
        true_label = test_df[config.LABEL_COL].iloc[index]
        cleaned = clean_text(raw_text)
        flags = find_patterns(cleaned, true_label, predictions[index], vocabulary)
        rows.append({"cleaned_text": cleaned, "raw_text": raw_text, "true_label": true_label,
                     "predicted_label": predictions[index],
                     "confidence": round(float(confidences[index]), 3), **flags})
    table = pd.DataFrame(rows)
    errors = table[table["true_label"] != table["predicted_label"]]
    errors.sort_values("confidence", ascending=False).to_csv(config.ERROR_ANALYSIS_PATH, index=False)
    return summarize_patterns(table, errors)


def summarize_patterns(all_rows, errors):
    """Compare how often each pattern appears in errors vs in the whole test set."""
    pattern_names = ["negation", "mixed_sentiment", "possible_sarcasm", "very_short_text",
                     "many_unknown_words"]
    summary = []
    for pattern in pattern_names:
        in_test = all_rows[pattern].mean()
        in_errors = errors[pattern].mean()
        error_rate = 1 - (all_rows[all_rows[pattern]]["true_label"]
                          == all_rows[all_rows[pattern]]["predicted_label"]).mean()
        summary.append({"pattern": pattern, "share_of_test_set": round(in_test, 3),
                        "share_of_errors": round(in_errors, 3),
                        "error_rate_when_present": round(error_rate, 3)})
    for pair in ["neutral_vs_positive", "neutral_vs_negative", "positive_vs_negative"]:
        summary.append({"pattern": "confusion: " + pair.replace("_vs_", " <-> "),
                        "share_of_test_set": np.nan,
                        "share_of_errors": round(errors[pair].mean(), 3),
                        "error_rate_when_present": np.nan})
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(config.RESULTS_DIR / "error_patterns.csv", index=False)
    return summary_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_all_models(models, info, test_df):
    """Evaluate every model, make its plots, and return (rows, predictions per model)."""
    rows, outputs = [], {}
    true_labels = list(test_df[config.LABEL_COL])
    for name, pipeline in models.items():
        train_time = info["models"][name].get("train_time_s", np.nan)
        row, predictions, confidences = evaluate_model(name, pipeline, test_df, train_time)
        save_classification_report(name, true_labels, predictions)
        plot_confusion_matrix(name, true_labels, predictions)
        rows.append(row)
        outputs[name] = (predictions, confidences)
    return rows, outputs


def plot_summaries(comparison):
    """Bar charts of test macro-F1 for the models and for the ablation variants."""
    models = comparison[comparison["type"] == "model"].sort_values("macro_f1", ascending=False)
    plot_horizontal_bars([name.replace("_", " ") for name in models["model"]],
                         list(models["macro_f1"]), "Test macro-F1 by model",
                         "Macro-F1 (higher is better)", config.RESULTS_DIR / "model_comparison.png")
    ablation = comparison[comparison["type"] == "ablation"]
    if len(ablation):
        labels = [name.replace("ablation (LR): ", "") for name in ablation["model"]]
        plot_horizontal_bars(labels, list(ablation["macro_f1"]),
                             "Ablation study: test macro-F1 (Logistic Regression, C=1)",
                             "Macro-F1 (higher is better)", config.RESULTS_DIR / "ablation.png")


def main():
    """Run the full evaluation and save every table and plot to results/."""
    set_plot_style()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_df = load_test_set()
    models, info = load_trained_models()
    rows, outputs = evaluate_all_models(models, info, test_df)
    extra = transformer_row()
    if extra:
        rows.append(extra)
    comparison = pd.DataFrame(rows + ablation_rows())
    comparison.to_csv(config.COMPARISON_PATH, index=False)
    print("\n", comparison.to_string(index=False))
    plot_summaries(comparison)

    best = info["best_model"]
    patterns = build_error_analysis(models[best], test_df, *outputs[best])
    print(f"\nError patterns for {best}:\n{patterns.to_string(index=False)}")
    if "logistic_regression" in models:
        plot_top_features(models["logistic_regression"])
    best_row = comparison[comparison["model"] == best].iloc[0]
    save_json({"best_model": best, "test_macro_f1": best_row["macro_f1"],
               "test_accuracy": best_row["accuracy"]}, config.RESULTS_DIR / "summary.json")
    scores = "; ".join(f"{r['model']} acc={r['accuracy']} macroF1={r['macro_f1']}" for r in rows)
    log_step(f"Step 6 Evaluation (test set): {scores}. Plots + error analysis in results/")


if __name__ == "__main__":
    main()
