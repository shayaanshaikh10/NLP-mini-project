"""
data_loader.py - get a labeled Hinglish dataset and turn it into
data/processed/{train,val,test}.csv with two columns: text,label.

Where the data comes from (tried in this order):
  1. Your own CSV dropped into data/raw/ (any .csv with a text column and a
     label column). This always wins, so swapping the dataset is easy.
  2. SemEval-2020 Task 9 "SentiMix" Hinglish data, downloaded from the
     Hugging Face mirror "RTT1/SentiMix" into data/raw/sentimix/.

LICENSE / TERMS:
  - SentiMix was released by the SemEval-2020 Task 9 organisers
    (Patwa et al., 2020) for research use. The Hugging Face mirror
    RTT1/SentiMix is tagged "openrail". The tweets come from Twitter/X, so
    use the data for academic work only and do not redistribute it
    commercially. Cite the task paper in your report.

Run:  python -m src.data_loader
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import log_step, save_json

SENTIMIX_REPO = "RTT1/SentiMix"
SENTIMIX_DIR = config.RAW_DIR / "sentimix"
SENTIMIX_FILES = {
    "train": "train_14k_split_conll.txt",
    "dev": "dev_3k_split_conll.txt",
    "test": "Hindi_test_unalbelled_conll_updated.txt",
    "test_labels": "test_labels_hinglish.txt",
}

# Map many possible label spellings onto our three labels
LABEL_MAP = {
    "negative": "negative", "neg": "negative", "0": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "1": "neutral",
    "positive": "positive", "pos": "positive", "2": "positive",
}
# Column names we accept in a user-provided CSV
TEXT_COLUMN_NAMES = ["text", "sentence", "tweet", "comment", "review", "content"]
LABEL_COLUMN_NAMES = ["label", "sentiment", "class", "polarity", "target"]


# ---------------------------------------------------------------------------
# Source 1: user CSV in data/raw/
# ---------------------------------------------------------------------------
def find_column(columns, candidates):
    """Return the first column whose lowercase name is in candidates, else None."""
    for column in columns:
        if column.strip().lower() in candidates:
            return column
    return None


def load_user_csv():
    """Load the first usable CSV found directly inside data/raw/ (or None)."""
    for csv_path in sorted(config.RAW_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        text_col = find_column(df.columns, TEXT_COLUMN_NAMES)
        label_col = find_column(df.columns, LABEL_COLUMN_NAMES)
        if text_col and label_col:
            print(f"Using user dataset: {csv_path.name}")
            df = df.rename(columns={text_col: config.TEXT_COL, label_col: config.LABEL_COL})
            return df[[config.TEXT_COL, config.LABEL_COL]], csv_path.name
        print(f"Skipping {csv_path.name}: no text/label columns found")
    return None, None


# ---------------------------------------------------------------------------
# Source 2: SentiMix (CoNLL format: "meta <id> <label>" then one token per line)
# ---------------------------------------------------------------------------
def download_sentimix():
    """Download the SentiMix files from Hugging Face if they are not already there."""
    from huggingface_hub import hf_hub_download

    for file_name in SENTIMIX_FILES.values():
        if not (SENTIMIX_DIR / file_name).exists():
            print(f"Downloading {file_name} ...")
            hf_hub_download(SENTIMIX_REPO, file_name, repo_type="dataset",
                            local_dir=str(SENTIMIX_DIR))


def is_underscore(token):
    """True for tokens made only of underscores, like '_' or '___'."""
    return len(token) > 0 and token.strip("_") == ""


def fix_mojibake(text):
    """Repair text that was saved with the wrong encoding.

    Some tweets contain 'ðŸ˜‚' instead of the emoji 😂. Re-encoding with
    cp1252 and decoding as UTF-8 fixes them. If that fails, keep the text.
    """
    if "ð" not in text and "â€" not in text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def tokens_to_sentence(tokens):
    """Join CoNLL tokens back into a sentence.

    The tokenizer split things like '@user' into '@','user' and URLs into
    'https','//','t','.','co','/','abc'. We drop mentions and URLs here
    (they carry no sentiment) and glue '#' back onto hashtag words.
    """
    words = []
    skip_next = False   # True right after '@' (the next token is a username)
    in_url = False      # True while we are inside a split-up URL
    seen_slash = False  # True once we passed the '/' before the URL id
    for token in tokens:
        if skip_next:
            # usernames with '_' were split: '@','RT','_','rajput' -> keep skipping
            skip_next = is_underscore(token)
        elif is_underscore(token) and not words:
            skip_next = True  # leftover piece of a username at the very start
        elif token == "@":
            skip_next = True
        elif token.lower() in ("http", "https"):
            in_url = True
            seen_slash = False
        elif in_url:
            if token == "/":
                seen_slash = True
            elif seen_slash or token == "…":
                in_url = False  # this was the random URL id -> URL is over
        else:
            words.append(fix_mojibake(token))
    sentence = " ".join(words)
    return sentence.replace("# ", "#").replace(" _ ", "_")


def parse_conll_file(path):
    """Parse a SentiMix CoNLL file into a list of (id, label, text) tuples."""
    rows = []
    with open(path, encoding="utf-8") as conll_file:
        blocks = conll_file.read().strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        header = lines[0].split("\t")
        if header[0] != "meta":
            continue  # malformed block, skip it
        tokens = [line.split("\t")[0] for line in lines[1:] if line.strip()]
        label = header[2] if len(header) > 2 else None
        rows.append((header[1], label, tokens_to_sentence(tokens)))
    return rows


def load_sentimix():
    """Load train + dev + labeled test of SentiMix into one DataFrame."""
    download_sentimix()
    train_rows = parse_conll_file(SENTIMIX_DIR / SENTIMIX_FILES["train"])
    dev_rows = parse_conll_file(SENTIMIX_DIR / SENTIMIX_FILES["dev"])
    test_rows = parse_conll_file(SENTIMIX_DIR / SENTIMIX_FILES["test"])

    # The test file has no labels inside it; they are in a separate CSV
    test_labels = pd.read_csv(SENTIMIX_DIR / SENTIMIX_FILES["test_labels"], dtype=str)
    label_by_id = dict(zip(test_labels["Uid"], test_labels["Sentiment"]))
    test_rows = [(uid, label_by_id.get(uid), text) for uid, _, text in test_rows]

    all_rows = train_rows + dev_rows + test_rows
    df = pd.DataFrame(all_rows, columns=["id", config.LABEL_COL, config.TEXT_COL])
    return df[[config.TEXT_COL, config.LABEL_COL]], "SentiMix (SemEval-2020 Task 9, Hinglish)"


# ---------------------------------------------------------------------------
# Cleaning, splitting and saving
# ---------------------------------------------------------------------------
def standardize_labels(df):
    """Map labels to negative/neutral/positive and drop rows we cannot map."""
    df = df.copy()
    df[config.LABEL_COL] = df[config.LABEL_COL].astype(str).str.strip().str.lower()
    df[config.LABEL_COL] = df[config.LABEL_COL].map(LABEL_MAP)
    unmapped = df[config.LABEL_COL].isna().sum()
    if unmapped:
        print(f"Dropped {unmapped} rows with an unknown label")
    return df.dropna(subset=[config.LABEL_COL])


def drop_bad_rows(df):
    """Remove empty texts and exact duplicate texts."""
    df = df.copy()
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str).str.strip()
    before = len(df)
    df = df[df[config.TEXT_COL].str.len() > 0]
    df = df.drop_duplicates(subset=[config.TEXT_COL])
    print(f"Removed {before - len(df)} empty/duplicate rows, {len(df)} rows left")
    return df.reset_index(drop=True)


def split_data(df):
    """Stratified 70/15/15 split into train, validation and test sets."""
    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=df[config.LABEL_COL], random_state=config.RANDOM_STATE)
    # Split the remaining 30% in half: 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
        stratify=temp_df[config.LABEL_COL], random_state=config.RANDOM_STATE)
    return train_df, val_df, test_df


def print_distribution(df, name):
    """Print how many examples of each label a DataFrame has."""
    counts = df[config.LABEL_COL].value_counts()
    print(f"\n{name}: {len(df)} rows")
    for label in config.LABELS:
        share = 100 * counts.get(label, 0) / len(df)
        print(f"  {label:<9} {counts.get(label, 0):>6}  ({share:.1f}%)")


def load_raw_dataset():
    """Try every data source in order and return (DataFrame, source_name)."""
    df, source = load_user_csv()
    if df is not None:
        return df, source
    try:
        return load_sentimix()
    except Exception as error:  # e.g. no internet connection
        raise RuntimeError(
            "No dataset found. Put a CSV with 'text' and 'label' columns in "
            f"data/raw/ and run again. (SentiMix download failed: {error})")


def main():
    """Build data/processed/{train,val,test}.csv from the best available source."""
    df, source = load_raw_dataset()
    df = drop_bad_rows(standardize_labels(df))
    print_distribution(df, "Full dataset")
    train_df, val_df, test_df = split_data(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    val_df.to_csv(config.VAL_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)
    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print_distribution(part, name)

    print("\n10 random training samples:")
    print(train_df.sample(10, random_state=config.RANDOM_STATE).to_string(index=False))

    counts = df[config.LABEL_COL].value_counts().to_dict()
    save_json({"source": source, "is_demo": False, "total_rows": len(df),
               "label_counts": counts, "train": len(train_df), "val": len(val_df),
               "test": len(test_df)}, config.DATASET_INFO_PATH)
    log_step(f"Dataset: {source}, {len(df)} rows after cleaning {counts}; "
             f"split train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")


if __name__ == "__main__":
    main()
