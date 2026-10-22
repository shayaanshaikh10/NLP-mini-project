"""
config.py - every constant of the project lives here.

Change a value here and every script picks it up. Paths are built relative
to the project folder, so the project works wherever it is copied.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root = the folder that contains src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NORMALIZATION_MAP_PATH = DATA_DIR / "normalization_map.csv"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
# Small JSON file that remembers where the data came from (real or demo)
DATASET_INFO_PATH = PROCESSED_DIR / "dataset_info.json"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_baseline.joblib"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
TRAINING_INFO_PATH = MODELS_DIR / "training_info.json"

RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_PATH = RESULTS_DIR / "comparison.csv"
ABLATION_PATH = RESULTS_DIR / "ablation.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "error_analysis.csv"

BUILD_LOG_PATH = PROJECT_ROOT / "BUILD_LOG.md"

# ---------------------------------------------------------------------------
# Data format
# ---------------------------------------------------------------------------
TEXT_COL = "text"
LABEL_COL = "label"
LABELS = ["negative", "neutral", "positive"]  # fixed order used everywhere

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------------
# Feature extraction (TF-IDF)
# ---------------------------------------------------------------------------
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
MIN_DF = 2

# ---------------------------------------------------------------------------
# Hyperparameter grids (tuned on the VALIDATION set, never on test)
# ---------------------------------------------------------------------------
NB_ALPHAS = [0.1, 0.5, 1.0]
LR_CS = [0.1, 1.0, 10.0]
SVM_CS = [0.1, 1.0, 10.0]
LR_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Words that flip the meaning of what follows them
NEGATION_WORDS = ["nahi", "na", "mat", "never", "not", "no", "kabhi"]
NEGATION_WINDOW = 2  # how many words after a negator get the NEG_ prefix

# ---------------------------------------------------------------------------
# Optional transformer settings (used only when a GPU is available)
# ---------------------------------------------------------------------------
TRANSFORMER_MODEL_NAME = "xlm-roberta-base"
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_MAX_LEN = 64
