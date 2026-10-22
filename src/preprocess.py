"""
preprocess.py - Hinglish-aware text cleaning.

clean_text() runs these small steps in order:
  1. basic cleanup: lowercase, remove URLs / @mentions / '#', decode HTML
  2. emojis -> text tokens (😍 -> emoji_smiling_face_with_heart-eyes emo_pos)
  3. elongation: 3+ repeated letters -> 2 ("bahuttttt" -> "bahutt")
  4. spelling normalization with data/normalization_map.csv ("bahutt" -> "bahut")
  5. (optional) negation marking: add NEG_ to words next to "nahi", "not", ...

What we deliberately do NOT do:
  - No stopword removal: English stopword lists would delete words like
    "not"/"no", and Hindi function words ("hi", "bhi", "to", "na") change the
    feeling of a sentence ("accha hi tha" vs "accha nahi tha").
  - No stemming / lemmatization: tools such as the Porter stemmer only know
    English and would mangle Hindi words written in Roman script. Character
    n-grams (in features.py) already capture shared word parts.

Run:  python -m src.preprocess     (prints raw vs cleaned examples)
"""

import html
import re

import emoji
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")  # a character repeated 3+ times
# Split text into tokens: words (letters/digits/_/') or runs of punctuation ("!!")
TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]+")
NON_WORD_PATTERN = re.compile(r"\W+")

_normalization_cache = {}  # filled once by load_normalization_map()


def load_normalization_map(path=config.NORMALIZATION_MAP_PATH):
    """Read normalization_map.csv into two dicts: single words and phrases."""
    if path in _normalization_cache:
        return _normalization_cache[path]
    if not path.exists():
        raise FileNotFoundError(f"Normalization map not found: {path}")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    word_map = {}
    phrase_map = {}  # variants with a space, e.g. "ek dum" -> "ekdum"
    for variant, canonical in zip(table["variant"], table["canonical"]):
        if " " in variant:
            phrase_map[variant] = canonical
        else:
            word_map[variant] = canonical
    _normalization_cache[path] = (word_map, phrase_map)
    return word_map, phrase_map


def basic_cleanup(text):
    """Step 1: lowercase, decode HTML entities, drop URLs and @mentions, keep hashtag words."""
    text = html.unescape(str(text)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", " ")  # "#worldcup" -> "worldcup"
    return text


def emoji_to_token(emoji_character, emoji_data):
    """Turn one emoji into a single word, e.g. 😍 -> emoji_smiling_face_with_heart_eyes."""
    name = emoji_data["en"].strip(":")  # ":smiling_face_with_heart-eyes:"
    name = NON_WORD_PATTERN.sub("_", name).strip("_")
    return " emoji_" + name + " "


def convert_emojis(text):
    """Step 2: turn every emoji into a word token and add emo_pos / emo_neg for common ones."""
    extra_tokens = []
    for character in text:
        if character in config.POSITIVE_EMOJIS:
            extra_tokens.append("emo_pos")
        elif character in config.NEGATIVE_EMOJIS:
            extra_tokens.append("emo_neg")
    # 😂 -> " emoji_face_with_tears_of_joy "
    text = emoji.replace_emoji(text, replace=emoji_to_token)
    return text + " " + " ".join(extra_tokens)


def reduce_elongation(text):
    """Step 3: collapse letters repeated 3+ times to 2 ("mastttt" -> "mastt", "!!!!" -> "!!")."""
    return ELONGATION_PATTERN.sub(r"\1\1", text)


def normalize_spelling(tokens, word_map):
    """Step 4: replace each spelling variant with its canonical form."""
    normalized = []
    for token in tokens:
        normalized.append(word_map.get(token, token))
    return normalized


def replace_phrases(text, phrase_map):
    """Step 4 (part): replace multi-word variants such as "ek dum" -> "ekdum"."""
    for variant, canonical in phrase_map.items():
        text = re.sub(r"\b" + re.escape(variant) + r"\b", canonical, text)
    return text


def mark_negation(tokens, scope="after"):
    """Step 5 (optional): add NEG_ to the words around a negator.

    "film acchi nahi thi" with scope="both" -> "film NEG_acchi nahi NEG_thi"
    """
    marked = list(tokens)
    for index, token in enumerate(tokens):
        if token not in config.NEGATION_WORDS:
            continue
        positions = list(range(index + 1, index + 1 + config.NEGATION_WINDOW))
        if scope == "both":
            positions += list(range(index - config.NEGATION_WINDOW, index))
        for position in positions:
            in_range = 0 <= position < len(tokens)
            if in_range and tokens[position] not in config.NEGATION_WORDS \
                    and tokens[position][0].isalnum() and not marked[position].startswith("NEG_"):
                marked[position] = "NEG_" + tokens[position]
    return marked


def clean_text(text, use_normalization=True, negation_scope=None):
    """Clean one Hinglish text and return the cleaned string.

    use_normalization: apply the spelling map (switched off in the ablation study)
    negation_scope:    None (off), "after" or "both" (see mark_negation)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = basic_cleanup(text)
    text = convert_emojis(text)
    text = reduce_elongation(text)
    word_map, phrase_map = load_normalization_map()
    if use_normalization:
        text = replace_phrases(text, phrase_map)
    tokens = TOKEN_PATTERN.findall(text)
    if use_normalization:
        tokens = normalize_spelling(tokens, word_map)
    if negation_scope:
        tokens = mark_negation(tokens, negation_scope)
    return " ".join(tokens)


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn step that runs clean_text() on every text.

    Putting the cleaner inside the Pipeline means the saved model file
    contains the preprocessing too, so predict() works on raw text.
    """

    def __init__(self, use_normalization=True, negation_scope=None):
        self.use_normalization = use_normalization
        self.negation_scope = negation_scope

    def fit(self, texts, labels=None):
        """Nothing to learn - cleaning is rule-based."""
        return self

    def transform(self, texts):
        """Return a list with the cleaned version of every text."""
        cleaned = []
        for text in texts:
            cleaned.append(clean_text(text, self.use_normalization, self.negation_scope))
        return cleaned


def show_examples(df, n=10):
    """Print raw vs cleaned text for n random rows (a quick sanity check)."""
    sample = df.sample(min(n, len(df)), random_state=config.RANDOM_STATE)
    for raw_text in sample[config.TEXT_COL]:
        print("RAW    :", raw_text)
        print("CLEANED:", clean_text(raw_text))
        print()


if __name__ == "__main__":
    demo_sentences = [
        "Movie ekdum mastttt thi 😍😍 but climax thoda boring tha!!!",
        "Bohot bakwaas film yaar, paisa waste 👎 https://t.co/abc @friend",
        "Khana acha nahi tha, kabhi nahi jaunga #disappointed",
    ]
    for sentence in demo_sentences:
        print("RAW    :", sentence)
        print("CLEANED:", clean_text(sentence, negation_scope="both"))
        print()
    if config.TRAIN_PATH.exists():
        show_examples(pd.read_csv(config.TRAIN_PATH), n=10)
