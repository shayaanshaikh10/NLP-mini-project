# HinglishSense: Sentiment Analysis of Hindi-English Code-Mixed Text

> Project brief used for the autonomous build. (Project root = this folder.)

## 1. Context

- **Type:** college mini project for the subject **Natural Language Processing (NLP)**, final-year B.E. in AI & Data Science (Mumbai University).
- **Team:** 3 students, beginner to intermediate in Python. Code must be **simple, commented, and easy to explain in a viva**.
- **Working mode:** this will run **unattended, overnight**. Make decisions yourself, take the simplest option when unsure, never wait for confirmation, and log everything (Section 13).
- **Constraints:** no paid APIs or keys. Must run on a normal laptop (CPU only). Anything that needs a GPU is **optional** and must be skipped gracefully if no GPU is present.

## 2. Problem Statement

People in India commonly write social media posts, reviews and comments in **Hinglish**: Hindi words typed in the Roman script mixed with English (for example: *"movie ekdum mast thi, but climax thoda boring tha"*). Standard English sentiment tools fail on this text because of code-mixing, non-standard spellings (*accha / acha / achha*), slang and emojis. Build an NLP system that classifies Hinglish text as **positive, negative or neutral**, evaluate several approaches, and ship a simple web demo.

## 3. Objectives

1. Collect/prepare a labeled Hinglish sentiment dataset.
2. Build a **Hinglish-aware preprocessing pipeline** (normalization of spelling variants, emoji handling, elongation handling).
3. Train and compare **classical models**: Naive Bayes, Logistic Regression, Linear SVM, using TF-IDF word n-grams plus **character n-grams** (which handle spelling variation well).
4. Optionally fine-tune a **multilingual transformer** (for example `xlm-roberta-base` or a smaller multilingual model) if a GPU is available, and compare.
5. Evaluate properly: accuracy, macro-F1, per-class precision/recall, confusion matrix, error analysis.
6. Build a **Streamlit demo** that predicts sentiment of any typed Hinglish sentence.
7. Produce README, tests, and ready-to-paste report/PPT/viva content.

## 4. Tech Stack

- Python 3.10+
- pandas, NumPy, scikit-learn, joblib, regex, `emoji` library
- NLTK (only if needed for tokenization) or plain `re`/`str.split`
- Matplotlib, Seaborn (plots)
- Streamlit (demo UI)
- pytest
- **Optional (GPU only):** `torch`, `transformers`, `datasets`, `accelerate`
- **Do not use:** paid LLM APIs, LangChain, vector databases, or anything that needs an API key.

## 5. Project Structure

```
hinglishsense/
├── CLAUDE.md                     # this brief
├── README.md
├── BUILD_LOG.md                  # progress log written during the overnight run
├── requirements.txt
├── data/
│   ├── raw/                      # original downloaded dataset(s)
│   ├── processed/                # cleaned train.csv, val.csv, test.csv (text,label)
│   └── normalization_map.csv     # hinglish spelling variants -> canonical word
├── src/
│   ├── config.py                 # paths, labels, random seed, column names, model params
│   ├── data_loader.py            # find/download dataset, convert to text,label CSV, split
│   ├── preprocess.py             # clean_text() with all Hinglish rules
│   ├── features.py               # TF-IDF word + char vectorizers
│   ├── train_baselines.py        # trains NB, LR, SVM; saves models + metrics
│   ├── train_transformer.py      # OPTIONAL fine-tuning (skips cleanly with no GPU)
│   ├── evaluate.py               # metrics, confusion matrices, error analysis, comparison table
│   ├── predict.py                # predict_sentiment(text) -> label, confidence, probabilities
│   └── utils.py                  # logging, timing helpers
├── app/
│   └── app.py                    # Streamlit demo
├── tests/
│   └── test_pipeline.py
├── models/                       # saved models (.joblib, transformer folder if trained)
├── results/                      # plots, comparison.csv, error_analysis.csv, report_content.md
└── notebooks/
    └── hinglishsense_demo.ipynb  # end-to-end walkthrough for submission
```

## 6. Dataset (most important and riskiest step)

**Target format used by the whole project:** CSV files with exactly two columns, `text` and `label`, where label is one of `negative`, `neutral`, `positive`. Everything downstream depends only on this format, so write `data_loader.py` to convert any source into it.

**Try these sources in order (verify availability, do not assume):**

1. **SemEval-2020 Task 9 SentiMix (Hinglish)**: the standard code-mixed benchmark. Search GitHub and Hugging Face for a mirrored copy (its Hinglish subset has tokens with language tags and per-sentence sentiment; convert it to sentence-level `text,label`).
2. Hinglish / code-mixed sentiment datasets on **Hugging Face Hub** (use `datasets` or `huggingface_hub` search and pick one that has Roman-script Hindi-English text with sentiment labels).
3. Hinglish sentiment datasets on **Kaggle** (if the `kaggle` CLI and credentials are available; otherwise skip).
4. If a dataset file already exists in `data/raw/`, use it first. The user may drop one there.

**Rules:**
- Check the license/terms in a comment in `data_loader.py` and in the README.
- Inspect the label scheme, map it to the three labels, drop empty/duplicate rows, and print class distribution.
- Make a stratified split: **70% train / 15% validation / 15% test**, `random_state=42`. Never tune on the test set.
- If classes are heavily imbalanced, report it and use `class_weight="balanced"` for the models. Do not oversample the test set.

**Fallback (so the overnight run never ends empty-handed):** if no real dataset can be downloaded, create a small **clearly labeled demo dataset** of about 400 hand-written Hinglish sentences (balanced across the 3 labels, varied topics: movies, food, college, cricket, shopping, travel, with spelling variants and emojis), save it as `data/raw/demo_seed.csv`, run the whole pipeline on it, and put a **bold warning at the top of the README and in the final summary** that results come from a tiny demo dataset and the team must replace it with a real dataset in `data/raw/` and rerun (`python -m src.data_loader && python -m src.train_baselines && python -m src.evaluate`). Do not present demo-set numbers as real results.

## 7. Preprocessing (`preprocess.py`)

Implement `clean_text(text)` as a short sequence of small, documented steps:

1. Lowercase; strip URLs, @mentions, `#` symbols (keep the hashtag word), HTML entities, and extra whitespace.
2. **Emojis:** convert to text tokens with the `emoji` library (for example 😍 to `emoji_smiling_face_with_heart_eyes`) or map the common ones to `emo_pos` / `emo_neg`; keep them because they carry sentiment.
3. **Elongation:** collapse 3 or more repeated letters to 2 (`bahuttttt` to `bahutt`, `mastttt` to `mastt`), then use the normalization map to fix the result.
4. **Spelling normalization:** apply `data/normalization_map.csv` (variant to canonical). Create it with **at least 150 common Hinglish variants**, for example: `acha/accha/achha -> accha`, `bura/buraa -> bura`, `bahut/bohot/bahot/bhot -> bahut`, `nahi/nahin/nai/nhi/nahee -> nahi`, `mast/mastt -> mast`, `bekar/bekaar -> bekar`, `kya/kyaa/kia -> kya`, `hai/hain/h -> hai`, `bakwas/bakwaas -> bakwas`, `zabardast/jabardast -> zabardast`, `ekdum/ek dum -> ekdum`, `pyaar/pyar -> pyar`, `dil/dill -> dil`, `thoda/thora -> thoda`, `mujhe/mujhey -> mujhe`. Extend it with slang and shortforms (`ni`, `plz`, `thx`, `gr8`, `awsm`, `lol`, `omg`).
5. **Negation handling:** keep Hinglish negators (`nahi`, `na`, `mat`, `never`, `not`, `no`, `kabhi nahi`). Optionally add a `NEG_` prefix to the 2 words after a negator and document it as an experiment evaluated on validation data.
6. **Do NOT remove stopwords** (many Hinglish function words carry sentiment context) and do **not** stem or lemmatize (no good Hinglish tools). Explain this choice in a comment.
7. Return the cleaned string. Add a function `show_examples(df, n=10)` that prints raw vs cleaned text for a sanity check.

## 8. Models

### 8.1 Classical baselines (`train_baselines.py`, **required**)
- Features: **FeatureUnion of** word TF-IDF (`ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`) and **character TF-IDF** (`analyzer="char_wb"`, `ngram_range=(2,5)`, `min_df=2`, `sublinear_tf=True`).
- Models: `MultinomialNB` (use the word features only, or all non-negative features), `LogisticRegression(max_iter=1000, class_weight="balanced")`, `LinearSVC(class_weight="balanced")`.
- Tune a few key hyperparameters per model with `GridSearchCV` or a small manual loop on the **validation set** (for example `C` in `[0.1, 1, 10]`, NB `alpha` in `[0.1, 0.5, 1]`). Keep the grids small so everything finishes in minutes.
- Also run an **ablation**: (a) word features only, (b) char features only, (c) both, (d) with vs without the normalization map. Save the results as a table. This shows the value of the Hinglish-specific work.
- Save the best model as `models/best_baseline.joblib` (a full `Pipeline`, including the vectorizers).

### 8.2 Transformer fine-tuning (`train_transformer.py`, **optional**)
- Run **only if** `torch.cuda.is_available()` is True. Otherwise print a clear message, record "skipped (no GPU)" in `BUILD_LOG.md`, and move on. Never fail the build because of this.
- Model: `xlm-roberta-base` (or a smaller multilingual model if memory is limited). Use the Hugging Face `Trainer` with 3 epochs, learning rate 2e-5, batch size 16, max length 64, fp16 on GPU, and select the best checkpoint by validation macro-F1.
- Save to `models/transformer/` and add its metrics to the comparison table.
- Also generate `notebooks/colab_transformer_finetune.ipynb`, a self-contained Colab notebook that does this fine-tuning on the free T4 GPU (install deps, load `data/processed/*.csv` uploaded by the user, train, evaluate, download the model), so the team can run it later if the local machine has no GPU.

## 9. Evaluation (`evaluate.py`)

- For every trained model, on the **test set**: accuracy, **macro-F1** (the main metric), weighted-F1, per-class precision/recall/F1, and a `classification_report`.
- Confusion matrix heatmap per model (`results/confusion_<model>.png`).
- `results/comparison.csv` with one row per model (and per ablation variant): accuracy, macro-F1, training time, inference time per sample, model file size.
- `results/error_analysis.csv`: every misclassified test sample (cleaned text, true label, predicted label, confidence). Write a short analysis (in `report_content.md`) of the top failure patterns: sarcasm, mixed sentiment, negation, slang/unknown words, very short text, neutral vs positive confusion.
- Show the **top 15 most positive and most negative features** from the Logistic Regression model (words and char n-grams) as a bar chart, for interpretability.
- Be honest: report real numbers. Hinglish sentiment is hard, and a macro-F1 around 0.60 to 0.75 on SentiMix-style data is normal.

## 10. Prediction and Demo App

### `predict.py`
- `predict_sentiment(text, model="baseline")` returns `{label, confidence, probabilities}`; use calibrated probabilities (for `LinearSVC` wrap it in `CalibratedClassifierCV`, or use Logistic Regression for the demo). Handle empty or very short input gracefully.

### `app/app.py` (Streamlit)
- Title **HinglishSense**, one-line description, and a short note explaining what Hinglish is.
- Text area + "Analyze" button; show the predicted sentiment with a colored badge (green/red/grey), the confidence, and a **probability bar chart** for the 3 classes.
- Show the **cleaned/normalized text** under the result (so users see what the model saw).
- **Example sentences** as clickable buttons (about 8 varied Hinglish examples, including one with an emoji and one with negation).
- Sidebar: model selector (baseline, transformer if available), a "Model comparison" expander that displays `results/comparison.csv`, and a short "About" section.
- Optional: a **batch mode** tab where the user uploads a CSV with a `text` column and downloads predictions.
- Cache the model with `@st.cache_resource`. Must start with `streamlit run app/app.py`.

## 11. Tests (`tests/test_pipeline.py`)

- `clean_text` collapses elongation, normalizes at least 5 known variants, converts an emoji, removes URLs and mentions.
- `predict_sentiment` returns a valid label and probabilities that sum to about 1.
- Obvious cases: `"movie bahut mast thi"` predicts positive, `"ekdum bakwas film, paisa waste"` predicts negative (only assert these if the model was trained on real data; for the demo dataset, assert only the output format).
- Empty string does not crash.
- Run `pytest` and fix failures.

## 12. Coding Style Rules (important)

- **Beginner-friendly:** simple functions, clear names, no clever one-liners, no unnecessary abstractions.
- Short comments explaining what and why, plus a docstring on every function.
- Functions of about 30 lines or fewer. All constants in `config.py`. Relative paths only. Seeds set (`random_state=42`).
- Each script runnable on its own (`python -m src.train_baselines`) and importable.
- Clear error messages for a missing dataset, missing optional library, or missing model file.
- `requirements.txt` split into core and optional (transformer) sections with comments.

## 13. Execution Instructions for Claude Code (autonomous overnight build)

**Do not ask me questions and do not wait for confirmation. If anything is ambiguous, choose the simplest option that satisfies this brief, record the assumption in `BUILD_LOG.md`, and continue.**

- Keep `BUILD_LOG.md` updated: after each step, append a timestamped line with what was done, key numbers, and any problem plus how it was resolved.
- Initialize a git repo and **commit after each step** with a clear message (do not push anywhere).
- **Time-box** each step. If something is stuck or taking too long (for example a slow download or a long training run), fall back to the simpler path and note it. Never leave the project in a broken state.
- Never print or store secrets. Do not run anything destructive outside the project folder.

**Steps (in order):**

1. **Setup:** create the structure, a virtual environment, `requirements.txt`, and `config.py`. Install core dependencies. Install transformer dependencies only if a GPU is present.
2. **Dataset:** follow Section 6. Convert to `data/processed/{train,val,test}.csv`, print class distribution and 10 samples. Use the fallback demo dataset only if every real source fails, and then flag it prominently.
3. **Preprocessing:** build `normalization_map.csv` (150+ variants) and `preprocess.py`. Show raw vs cleaned examples.
4. **Baselines:** features, training, the ablation study, and saving the best model.
5. **Optional transformer:** only with a GPU. Always create the Colab notebook either way.
6. **Evaluation:** all metrics, plots, the comparison table, error analysis, and top-feature plots.
7. **Prediction module and Streamlit app.** Verify the app starts (headless check or import test) and fix any errors.
8. **Tests:** write them, run `pytest`, and fix failures.
9. **Docs:** `README.md` (overview, what Hinglish is, setup, how to run everything, how to swap the dataset, results table with real numbers, limitations, future scope) and `results/report_content.md` (see Section 14).
10. **Demo notebook:** `notebooks/hinglishsense_demo.ipynb` that runs the pipeline end to end on the processed data.
11. **Final summary** (maximum 25 lines) printed at the end and saved as `FINAL_SUMMARY.md`: what was built, dataset used (real or demo, with a warning if demo), the comparison table with real numbers, how to run the app, anything skipped and why, and the exact next steps for the team.

## 14. Report, PPT and Viva Content (`results/report_content.md`)

Ready-to-paste text for:

1. Abstract (about 150 words)
2. Introduction and motivation (code-mixing on Indian social media, why English tools fail)
3. Problem statement and objectives
4. Literature survey (3 to 5 short items on code-mixed sentiment analysis and SentiMix; **do not invent citations, mark each `[VERIFY REFERENCE]`**)
5. Dataset description (source, size, label distribution, preprocessing, split)
6. Methodology: the preprocessing pipeline (with a **Mermaid flowchart** of the full system), TF-IDF word and character n-grams, the models, and why stopwords/lemmatization were skipped
7. Implementation details (tools, hyperparameters)
8. Results and discussion with the real numbers, the ablation study, plots, and error analysis
9. Comparison of all models
10. Conclusion
11. Limitations (sarcasm, tiny or noisy data, Roman-script spelling chaos, Devanagari input not handled) and future scope (transliteration to Devanagari, Hinglish-specific embeddings, aspect-based sentiment, handling emojis and memes, deploying as an API or browser extension)
12. **20 viva questions with short answers** (what is code-mixing, what is TF-IDF, why character n-grams, why not remove stopwords, what is macro-F1 and why use it over accuracy, what is class imbalance, why a validation set, what is a transformer, why fine-tune, what are the limitations, and so on)
13. A 10-slide PPT outline with bullet points per slide.
