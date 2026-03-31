import pandas as pd

from src.config import ID_COL, TARGET_COL
from typing import List, Optional

BASELINE_FEATURES = [
    "log_video_duration",
    "log_vph",
    "log_likes",
    "log_views",
    "log_comments",
    "log_channel_subscribers",
    "TIME_IDX",
    "time_since_prev_scrape_min",
    "time_since_first_scrape_min",
    "video_age_at_scrape_min",
]


def load_processed_data(path) -> pd.DataFrame:
    """Load processed dataset."""
    return pd.read_csv(path)


def get_available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Keep only features that are present in the dataframe."""
    return [col for col in feature_list if col in df.columns]


def build_model_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "log_future_views",
    group_col: str = ID_COL,
):
    """
    Build X, y, and groups for modelling.
    """
    if feature_cols is None:
        feature_cols = BASELINE_FEATURES

    feature_cols = get_available_features(df, feature_cols)

    missing_target = target_col not in df.columns
    if missing_target:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df[group_col].copy()

    return X, y, groups, feature_cols