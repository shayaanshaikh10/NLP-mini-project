# HinglishSense

**Sentiment analysis of Hindi-English code-mixed (Hinglish) text: positive, negative or neutral.**

NLP mini project, final-year B.E. in AI & Data Science (Mumbai University).

> Results below come from the **real SentiMix dataset** (SemEval-2020 Task 9, about 19.7k Hinglish tweets), not a toy dataset.

---

## What is Hinglish?

Many Indians write Hindi in the Roman (English) alphabet and mix in English words:

> *"movie ekdum mast thi, but climax thoda boring tha"*

This is called **code-mixing**. Standard English sentiment tools struggle with it because:

- half the words are Hindi, which an English vocabulary doesn't know;
- there is no standard spelling (*acha / accha / achha*, *nahi / nhi / nai*, *bahut / bohot / bhot*);
- it is full of slang, elongations (*mastttt*), emojis and short forms (*plz*, *gr8*).

HinglishSense cleans and normalizes this text, turns it into word and character n-gram features, and compares three classical classifiers. A Streamlit web demo is included.

## Results (test set, 2,952 tweets)

| Model | Accuracy | **Macro-F1** | Weighted-F1 | Train time | Inference / sample | Size |
|---|---|---|---|---|---|---|
| **Linear SVM** (C=0.1, calibrated) | **0.637** | **0.640** | 0.636 | 5.9 s | 0.27 ms | 9.9 MB |
| Logistic Regression (C=1) | 0.626 | 0.629 | 0.624 | 8.7 s | 0.27 ms | 4.7 MB |
| Multinomial Naive Bayes (alpha=1) | 0.619 | 0.622 | 0.617 | 4.3 s | 0.27 ms | 7.3 MB |
| XLM-RoBERTa (optional) | not trained: no GPU on the build machine, use the Colab notebook | | | | | |

All three models use **word (1-2) + character (2-5) TF-IDF** on the cleaned text. Hyperparameters were tuned on the **validation** set; the test set was used only once, for this table. Linear SVM and Logistic Regression were practically tied on validation (0.6396 vs 0.6388 macro-F1). The SVM is saved as `models/best_baseline.joblib`.

Per-class F1 (Linear SVM): negative 0.667, **neutral 0.565**, positive 0.689. Neutral is the hardest class.

### Ablation study (Logistic Regression, C=1)

| Variant | Val macro-F1 | Test macro-F1 |
|---|---|---|
| word n-grams only | 0.624 | 0.615 |
| word only, **no** normalization map | 0.623 | 0.612 |
| char n-grams only | 0.638 | **0.636** |
| word + char (chosen) | **0.639** | 0.629 |
| word + char, **no** normalization map | 0.637 | 0.628 |
| word + char + NEG_ marking (after negator) | 0.636 | 0.631 |
| word + char + NEG_ marking (both sides) | 0.635 | 0.632 |

What the ablation shows:
- **Character n-grams matter most**: about +1.5 to +2 points over words alone, because they absorb spelling variation.
- The **normalization map** helps a little (+0.1 to +0.3), most of all for word-only features.
- **NEG_ marking** did not help on validation, so it is off in the final model. Note that Hindi puts the negator *after* the word ("accha **nahi**"), unlike English.

Differences under about 1 point are within noise for a test set of this size.

Plots are in [results/](results/): confusion matrices, model comparison, ablation, and the top Logistic Regression features.

## Setup

Requires Python 3.10+ (built and tested with 3.12). CPU only, no API keys.

```bash
# from the project folder
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## How to run everything

```bash
python -m src.data_loader       # download SentiMix, clean, 70/15/15 split -> data/processed/
python -m src.preprocess        # optional: shows raw vs cleaned examples
python -m src.train_baselines   # ablation + tuning + train NB / LR / SVM -> models/
python -m src.train_transformer # optional, needs a CUDA GPU (skips cleanly otherwise)
python -m src.evaluate          # metrics, plots, comparison.csv, error analysis -> results/
pytest                          # 33 tests

streamlit run app/app.py        # the web demo (http://localhost:8501)
```

Predict from Python or the command line:

```python
from src.predict import predict_sentiment
predict_sentiment("movie bahut mast thi")
# {'label': 'positive', 'confidence': 0.6, 'probabilities': {...}, 'cleaned_text': 'movie bahut mast thi', 'note': ''}
```
```bash
python -m src.predict "ekdum bakwas film, paisa waste"
```

**No GPU?** Open [notebooks/colab_transformer_finetune.ipynb](notebooks/colab_transformer_finetune.ipynb) in Google Colab (free T4 GPU) and upload `data/processed/{train,val,test}.csv`. Unzip the downloaded model into `models/transformer/`; the app then offers a "transformer" option. To add its numbers to the comparison table, `pip install torch transformers` locally and rerun `python -m src.evaluate`.

A guided walkthrough is in [notebooks/hinglishsense_demo.ipynb](notebooks/hinglishsense_demo.ipynb).

## How to use a different dataset

1. Put a CSV file directly in `data/raw/` with a text column (`text`, `sentence`, `tweet`, `comment`...) and a label column (`label`, `sentiment`...). Labels can be `negative/neutral/positive`, `neg/neu/pos`, or `0/1/2`.
2. Rerun: `python -m src.data_loader && python -m src.train_baselines && python -m src.evaluate`

A CSV in `data/raw/` always takes priority over SentiMix.

## How it works

```
raw text -> clean_text() -> TF-IDF (word 1-2 grams + char 2-5 grams) -> classifier -> label + probabilities
```

`clean_text()` in [src/preprocess.py](src/preprocess.py) runs these steps:
1. Lowercase; remove URLs and @mentions; drop `#` but keep the hashtag word; decode HTML entities.
2. Emojis become words (`😍` → `emoji_smiling_face_with_heart_eyes`), plus an `emo_pos` / `emo_neg` token for common ones.
3. Elongations are collapsed (`mastttt` → `mastt`).
4. Spellings are normalized with [data/normalization_map.csv](data/normalization_map.csv), which has 363 variants (`mastt` → `mast`, `bohot` → `bahut`, `nhi` → `nahi`, `plz` → `please`...).
5. Optionally, `NEG_` is added to words next to a negator (an experiment, off by default).

**No stopword removal and no stemming:** English stopword lists delete "not"/"no", Hindi function words (*hi, bhi, to, na*) change meaning, and English stemmers mangle Hindi words.

## Project structure

```
src/config.py            every constant: paths, labels, seed, grids
src/data_loader.py       dataset download/conversion + stratified split
src/preprocess.py        Hinglish cleaning (clean_text, TextCleaner)
src/features.py          TF-IDF word + char vectorizers
src/train_baselines.py   ablation, tuning, training, saving
src/train_transformer.py optional XLM-RoBERTa fine-tuning (GPU only)
src/evaluate.py          test metrics, plots, comparison, error analysis
src/predict.py           predict_sentiment()
app/app.py               Streamlit demo
tests/                   pytest suite (pipeline + app)
results/                 plots, CSV tables, report_content.md (report/PPT/viva text)
BUILD_LOG.md             timestamped log of the build
```

## Dataset and license

**SentiMix Hinglish** (SemEval-2020 Task 9, Patwa et al., 2020), downloaded from the Hugging Face mirror [`RTT1/SentiMix`](https://huggingface.co/datasets/RTT1/SentiMix), which is tagged `openrail`. The official train (14k), dev (3k) and labelled test (3k) parts were pooled, cleaned (mentions/URLs removed, broken emoji encodings repaired, empty and duplicate rows dropped) and re-split 70/15/15, stratified, with `random_state=42`. This gives 19,675 tweets: 37% neutral, 33% positive, 30% negative.

Because of the re-split, **our numbers are not directly comparable to the SemEval leaderboard**, which uses the official test set. The tweets come from Twitter/X: use them for academic work only and cite the task paper.

## Limitations

- **Domain:** SentiMix is mostly tweets about politics, cricket and celebrities. Product, food and movie reviews are out of domain; for example, *"khana accha nahi tha"* is predicted neutral.
- **Negation and sarcasm** are not understood. A bag of n-grams has no word order beyond pairs.
- **Neutral** is confused with both other classes; about 89% of errors involve neutral.
- **Noisy labels:** some of the most confident "errors" look mislabelled (e.g. *"jeet ki hardik badhai"* labelled neutral).
- Many tweets are **truncated** ("…"), and about 1% contain emoji bytes that could not be repaired.
- **Devanagari** input is tokenized correctly but is too rare in training to be learned; there is no transliteration.

## Future scope

- Fine-tune XLM-RoBERTa / MuRIL / IndicBERT (Colab notebook provided).
- Transliterate Roman Hindi to Devanagari, or the reverse, before classification.
- Hinglish-specific word embeddings (fastText trained on code-mixed text).
- Aspect-based sentiment (*"camera accha hai but battery bekar"* → camera: positive, battery: negative).
- Better handling of emojis, memes and sarcasm.
- Serve the model as a REST API or a browser extension.
