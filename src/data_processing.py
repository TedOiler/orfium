import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    ID_COL,
    TARGET_COL,
    SCRAPE_TIME_COL,
    PUBLISHED_AT_COL,
    TIME_IDX_COL,
    TIMESTAMP_COLS,
    DROP_COLS,
    VIDEO_DURATION_COL,
    ZERO_DURATION_CHECK_COLS,
    LOG_FEATURE_COLS,
    LOG_TARGET,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(path)


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert configured timestamp columns to pandas datetime."""
    df = df.copy()
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort observations by video and scrape time."""
    return df.sort_values([ID_COL, SCRAPE_TIME_COL]).reset_index(drop=True)


def create_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Create within-video time index."""
    df = df.copy()
    df[TIME_IDX_COL] = df.groupby(ID_COL).cumcount()
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create elapsed-time features:
    - minutes since previous scrape
    - minutes since first scrape
    - optional age since publication
    """
    df = df.copy()

    prev_diff = df.groupby(ID_COL)[SCRAPE_TIME_COL].diff()
    df["time_since_prev_scrape_min"] = prev_diff.dt.total_seconds() / 60.0
    df["time_since_prev_scrape_min"] = df["time_since_prev_scrape_min"].fillna(0.0)

    first_scrape = df.groupby(ID_COL)[SCRAPE_TIME_COL].transform("min")
    since_first = df[SCRAPE_TIME_COL] - first_scrape
    df["time_since_first_scrape_min"] = since_first.dt.total_seconds() / 60.0

    if PUBLISHED_AT_COL in df.columns:
        pub_age = df[SCRAPE_TIME_COL] - df[PUBLISHED_AT_COL]
        df["video_age_at_scrape_min"] = pub_age.dt.total_seconds() / 60.0

    return df


def identify_bad_videos(df: pd.DataFrame) -> np.ndarray:
    """
    Identify videos with zero duration rows that also have positive
    engagement metrics. These videos are treated as corrupted.
    """
    zero_duration_mask = df[VIDEO_DURATION_COL] == 0
    inconsistent_mask = (df[ZERO_DURATION_CHECK_COLS] > 0).any(axis=1)

    bad_rows = df[zero_duration_mask & inconsistent_mask]
    bad_videos = bad_rows[ID_COL].unique()

    return bad_videos


def drop_bad_videos(df: pd.DataFrame, bad_videos: np.ndarray) -> pd.DataFrame:
    """Remove all rows belonging to corrupted videos."""
    return df.loc[~df[ID_COL].isin(bad_videos)].copy()


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-model columns chosen during EDA."""
    existing_drop_cols = [col for col in DROP_COLS if col in df.columns]
    return df.drop(columns=existing_drop_cols)


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log1p versions of configured features and target."""
    df = df.copy()

    for col in LOG_FEATURE_COLS:
        if col in df.columns:
            df[f"log_{col.lower()}"] = np.log1p(df[col])

    if LOG_TARGET and TARGET_COL in df.columns:
        df[f"log_{TARGET_COL.lower()}"] = np.log1p(df[TARGET_COL])

    return df


def validate_processed_data(df: pd.DataFrame) -> None:
    """Basic validation checks after processing."""
    if df[SCRAPE_TIME_COL].isna().any():
        raise ValueError("Missing SCRAPE_TIME values after parsing.")

    if TIME_IDX_COL not in df.columns:
        raise ValueError("TIME_IDX column was not created.")

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if (df.groupby(ID_COL)[SCRAPE_TIME_COL].diff().dropna() < pd.Timedelta(0)).any():
        raise ValueError("Detected time ordering issues after sorting.")


def process_data(save: bool = True) -> pd.DataFrame:
    """Full end-to-end data processing pipeline."""
    df = load_raw_data()
    print(f"Loaded raw data: {df.shape}")

    df = parse_timestamps(df)
    df = sort_panel(df)
    df = create_time_idx(df)

    bad_videos = identify_bad_videos(df)
    print(f"Identified corrupted videos to drop: {len(bad_videos)}")

    df = drop_bad_videos(df, bad_videos)
    print(f"Shape after dropping corrupted videos: {df.shape}")

    df = drop_unused_columns(df)
    df = create_time_features(df)
    df = add_log_features(df)

    validate_processed_data(df)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Saved processed data to: {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    processed_df = process_data(save=True)
    print(processed_df.head())