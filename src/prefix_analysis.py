import json
import pickle
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    PROCESSED_DATA_PATH,
    TABLES_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
)
from src.feature_engineering import load_processed_data, build_model_dataset
from src.split import make_grouped_train_val_test_split


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

import matplotlib.pyplot as plt


def plot_prefix_metrics(prefix_df: pd.DataFrame):
    """
    Plot RMSE_log vs number of observations (TIME_IDX + 1).
    """
    plt.figure()

    x = prefix_df["TIME_IDX"] + 1
    y = prefix_df["rmse_log"]

    plt.plot(x, y, marker="o")

    plt.xlabel("Number of observations available")
    plt.ylabel("RMSE (log scale)")
    plt.title("Model performance vs available observations (validation)")

    plt.grid()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "prefix_rmse_exact_val.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved prefix RMSE plot to: {out_path}")

def compute_prefix_metrics(df_val: pd.DataFrame) -> pd.DataFrame:
    """
    Compute validation error metrics grouped by exact TIME_IDX.
    Assumes df_val contains:
    - TIME_IDX
    - y_true_log
    - y_pred_log
    """
    results = []

    for time_idx, group in df_val.groupby("TIME_IDX", sort=True):
        y_true_log = group["y_true_log"]
        y_pred_log = group["y_pred_log"]

        results.append({
            "TIME_IDX": int(time_idx),
            "n_rows": int(len(group)),
            "n_videos": int(group["VIDEO_ID"].nunique()),
            "rmse_log": rmse(y_true_log, y_pred_log),
            "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        })

    results_df = pd.DataFrame(results).sort_values("TIME_IDX").reset_index(drop=True)
    return results_df


def load_selected_model_name():
    comparison_path = TABLES_DIR / "model_comparison_val.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Validation comparison file not found: {comparison_path}"
        )

    comparison_df = pd.read_csv(comparison_path)
    if "rmse_log" not in comparison_df.columns or "model" not in comparison_df.columns:
        raise ValueError("model_comparison_val.csv must contain 'model' and 'rmse_log' columns.")

    selected_model = comparison_df.sort_values("rmse_log").iloc[0]["model"]
    return selected_model


def load_model(model_name: str):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, model_path


def run_prefix_analysis():
    print("=" * 60)
    print("Prefix analysis on validation set")
    print("=" * 60)

    # Load selected model
    selected_model = load_selected_model_name()
    model, model_path = load_model(selected_model)

    print(f"Selected model from validation comparison: {selected_model}")
    print(f"Loaded model from: {model_path}")

    # Recreate exact same validation split
    df = load_processed_data(PROCESSED_DATA_PATH)
    X, y, groups, feature_cols = build_model_dataset(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
    ) = make_grouped_train_val_test_split(
        X,
        y,
        groups,
        val_size=0.15,
        test_size=0.15,
        random_state=RANDOM_SEED,
    )

    # Recover TIME_IDX for validation rows using aligned index
    val_meta = df.loc[X_val.index, ["VIDEO_ID", "TIME_IDX"]].copy()

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    val_predictions = pd.DataFrame({
        "VIDEO_ID": groups_val.values,
        "TIME_IDX": val_meta["TIME_IDX"].values,
        "y_true_log": y_val.values,
        "y_pred_log": y_val_pred,
        "abs_error_log": np.abs(y_val.values - y_val_pred),
    })

    # Basic consistency checks
    if len(val_predictions) != len(X_val):
        raise ValueError("Mismatch between validation predictions and validation rows.")

    if not np.array_equal(val_predictions["VIDEO_ID"].values, val_meta["VIDEO_ID"].values):
        raise ValueError("VIDEO_ID alignment check failed between validation split and metadata.")

    prefix_metrics = compute_prefix_metrics(val_predictions)

    # Save output
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "prefix_metrics_exact_val.csv"
    prefix_metrics.to_csv(out_path, index=False)

    # Print diagnostics
    print("\nValidation split recovered successfully.")
    print(f"Validation rows: {len(X_val)}")
    print(f"Validation videos: {groups_val.nunique()}")
    print(f"Unique TIME_IDX values: {sorted(prefix_metrics['TIME_IDX'].tolist())}")

    print("\nPrefix metrics (head):")
    print(prefix_metrics.head(10))

    first_rmse = prefix_metrics.iloc[0]["rmse_log"]
    last_rmse = prefix_metrics.iloc[-1]["rmse_log"]
    pct_change = 100 * (last_rmse - first_rmse) / first_rmse

    print("\nTrend summary:")
    print(f"First TIME_IDX rmse_log: {first_rmse:.4f}")
    print(f"Last TIME_IDX rmse_log:  {last_rmse:.4f}")
    print(f"Percentage change:       {pct_change:.2f}%")

    print(f"\nSaved exact prefix metrics to: {out_path}")

    plot_prefix_metrics(prefix_metrics)

    return prefix_metrics


if __name__ == "__main__":
    run_prefix_analysis()