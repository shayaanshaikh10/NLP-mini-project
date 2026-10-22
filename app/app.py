"""
HinglishSense - Streamlit demo.

Start it from the project folder with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Make "from src import ..." work when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.predict import available_models, load_model, predict_sentiment  # noqa: E402

# Badge color + icon for each label (the label text is always shown too)
LABEL_STYLE = {
    "positive": {"color": "green", "icon": ":material/sentiment_satisfied:", "hex": "#0ca30c"},
    "negative": {"color": "red", "icon": ":material/sentiment_dissatisfied:", "hex": "#d03b3b"},
    "neutral": {"color": "gray", "icon": ":material/sentiment_neutral:", "hex": "#898781"},
}
EXAMPLES = [
    "Movie ekdum mast thi yaar, must watch!",
    "Ekdum bakwas film, paisa waste 👎",
    "Kal college mein exam hai, 10 baje se",
    "Khana bilkul accha nahi tha, kabhi nahi jaunga",
    "India ne match jeet liya 😍🔥 proud moment",
    "Delivery late thi but product achha hai",
    "Bhai tu sach mein pagal hai kya, kuch bhi bolta hai",
    "Train 2 ghante late hai, station pe wait kar rahe hain",
]

st.set_page_config(page_title="HinglishSense", page_icon=":material/translate:")

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None


@st.cache_resource
def get_model(model_name):
    """Load a model once and keep it in memory for all users and reruns."""
    return load_model(model_name)


def use_example(example):
    """Button callback: put an example into the text box and analyze it."""
    st.session_state["input_text"] = example
    st.session_state["result"] = predict_sentiment(example, st.session_state["model_name"])


def analyze():
    """Button callback: analyze whatever is in the text box."""
    st.session_state["result"] = predict_sentiment(
        st.session_state["input_text"], st.session_state["model_name"])


def probability_chart(probabilities):
    """Bar chart of the three class probabilities, colored like the badge."""
    data = pd.DataFrame({"sentiment": list(probabilities),
                         "probability": list(probabilities.values())})
    colors = alt.Scale(domain=config.LABELS,
                       range=[LABEL_STYLE[label]["hex"] for label in config.LABELS])
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability",
                axis=alt.Axis(format="%")),
        y=alt.Y("sentiment:N", sort=config.LABELS, title=None),
        color=alt.Color("sentiment:N", scale=colors, legend=None),
        tooltip=["sentiment", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=130)


def show_result(result):
    """Render the badge, confidence, probability chart and cleaned text."""
    style = LABEL_STYLE[result["label"]]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(result["label"].capitalize(), icon=style["icon"], color=style["color"])
            st.markdown(f"**{result['confidence']:.0%}** confidence")
        st.altair_chart(probability_chart(result["probabilities"]))
        st.caption("What the model saw after Hinglish cleaning:")
        st.code(result["cleaned_text"] or "(empty)", language=None, wrap_lines=True)
        if result["note"]:
            st.warning(result["note"], icon=":material/info:")


def predict_csv(uploaded_file, model_name):
    """Batch mode: add predicted_label and confidence columns to an uploaded CSV."""
    table = pd.read_csv(uploaded_file)
    if config.TEXT_COL not in table.columns:
        st.error(f"The CSV needs a column named '{config.TEXT_COL}'. "
                 f"Found: {', '.join(table.columns)}")
        return None
    labels, confidences = [], []
    for text in table[config.TEXT_COL]:
        result = predict_sentiment(text, model_name)
        labels.append(result["label"])
        confidences.append(result["confidence"])
    table["predicted_label"] = labels
    table["confidence"] = confidences
    return table


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    models = available_models()
    if not models:
        st.error("No trained model found. Run `python -m src.train_baselines` first.")
        st.stop()
    st.selectbox("Model", models, key="model_name",
                 help="'baseline' is the best classical model chosen on the validation set.")
    get_model(st.session_state["model_name"])  # load once, cached

    with st.expander("Model comparison (test set)", icon=":material/leaderboard:"):
        if config.COMPARISON_PATH.exists():
            comparison = pd.read_csv(config.COMPARISON_PATH)
            columns = ["model", "accuracy", "macro_f1"]
            st.dataframe(comparison[columns], hide_index=True,
                         alt="Accuracy and macro-F1 of every model on the test set")
        else:
            st.caption("Run `python -m src.evaluate` to create the table.")

    st.subheader("About")
    st.caption("HinglishSense: an NLP mini project (B.E. AI & DS). Hinglish-aware "
               "cleaning, TF-IDF word + character n-grams, and Naive Bayes, Logistic "
               "Regression and Linear SVM classifiers trained on SentiMix "
               "(SemEval-2020 Task 9).")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("HinglishSense", anchor=False)
st.markdown("Sentiment analysis (positive, negative or neutral) for Hindi-English "
            "code-mixed text.")
st.info("Hinglish is Hindi written in the Roman (English) alphabet and mixed with "
        "English words, e.g. *\"movie ekdum mast thi, but climax thoda boring tha\"*. "
        "Spellings vary a lot (*acha / accha / achha*), so normal English tools struggle.",
        title="What is Hinglish?", icon=":material/translate:")

single_tab, batch_tab = st.tabs(["Single sentence", "Batch mode (CSV)"])

with single_tab:
    st.text_area("Type a Hinglish sentence", key="input_text", height=100,
                 placeholder="e.g. yaar ye phone bahut accha hai, battery bhi mast hai")
    st.button("Analyze", type="primary", icon=":material/psychology:", on_click=analyze)

    st.caption("Or try an example:")
    with st.container(horizontal=True):
        for example in EXAMPLES:
            st.button(example, on_click=use_example, args=(example,))

    if st.session_state["result"] is not None:
        show_result(st.session_state["result"])

with batch_tab:
    st.markdown(f"Upload a CSV with a **`{config.TEXT_COL}`** column. Each row gets a "
                "predicted label and a confidence.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        predictions = predict_csv(uploaded, st.session_state["model_name"])
        if predictions is not None:
            st.dataframe(predictions, hide_index=True, alt="Uploaded texts with predictions")
            st.download_button("Download predictions", predictions.to_csv(index=False),
                               file_name="hinglishsense_predictions.csv", mime="text/csv",
                               icon=":material/download:")
