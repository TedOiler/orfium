from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "dataset_100k_10dp.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_dataset.csv"

# Core column names
ID_COL = "VIDEO_ID"
TARGET_COL = "FUTURE_VIEWS"
SCRAPE_TIME_COL = "SCRAPE_TIME"
PUBLISHED_AT_COL = "VIDEO_PUBLISHED_AT"
TIME_IDX_COL = "TIME_IDX"

# Timestamp columns
TIMESTAMP_COLS = [SCRAPE_TIME_COL, PUBLISHED_AT_COL]

# Columns dropped before modelling
DROP_COLS = [
    "TITLE",
    "DESCRIPTION",
]

# Video-level anomaly rules
VIDEO_DURATION_COL = "VIDEO_DURATION"

# If a row has VIDEO_DURATION == 0 and any of these are positive,
# we treat the whole video as corrupted and drop it.
ZERO_DURATION_CHECK_COLS = [
    "VPH",
    "LIKES",
    "VIEWS",
    "COMMENTS",
    TARGET_COL,
]

# Log-transform columns
LOG_FEATURE_COLS = [
    "VIDEO_DURATION",
    "VPH",
    "LIKES",
    "VIEWS",
    "COMMENTS",
    "CHANNEL_SUBSCRIBERS",
]

LOG_TARGET = True
RANDOM_SEED = 42