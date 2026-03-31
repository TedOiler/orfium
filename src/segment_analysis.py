import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    PROCESSED_DATA_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.feature_engineering import load_processed_data, build_model_dataset
from src.split import make_grouped_train_val_test_split


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_selected_model_name():
    path = TABLES_DIR / "model_comparison_val.csv"
    df = pd.read_csv(path)
    return df.sort_values("rmse_log").iloc[0]["model"]


def load_model(model_name):
    path = MODELS_DIR / f"{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def assign_segments(y_true_log, n_bins=3):
    """
    Create quantile-based segments.
    """
    return pd.qcut(y_true_log, q=n_bins, labels=["low", "medium", "high"])


def compute_segment_metrics(df):
    results = []

    for segment, group in df.groupby("segment"):
        results.append({
            "segment": segment,
            "n_rows": len(group),
            "rmse_log": rmse(group["y_true_log"], group["y_pred_log"]),
            "mae_log": mean_absolute_error(group["y_true_log"], group["y_pred_log"]),
        })

    return pd.DataFrame(results).sort_values("segment")


def plot_segment_metrics(segment_df):
    plt.figure()

    x = segment_df["segment"]
    y = segment_df["rmse_log"]

    plt.bar(x, y)

    plt.xlabel("Video popularity segment")
    plt.ylabel("RMSE (log scale)")
    plt.title("Model performance by video popularity (validation)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "segment_rmse_val.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved segment plot to: {out_path}")


def run_segment_analysis():
    print("=" * 60)
    print("Segment analysis on validation set")
    print("=" * 60)

    # Load model
    model_name = load_selected_model_name()
    model = load_model(model_name)

    print(f"Using selected model: {model_name}")

    # Recreate split
    df = load_processed_data(PROCESSED_DATA_PATH)
    X, y, groups, _ = build_model_dataset(df)

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

    # Predict
    y_val_pred = model.predict(X_val)

    val_df = pd.DataFrame({
        "VIDEO_ID": groups_val.values,
        "y_true_log": y_val.values,
        "y_pred_log": y_val_pred,
    })

    # Assign segments
    val_df["segment"] = assign_segments(val_df["y_true_log"])

    # Compute metrics
    segment_metrics = compute_segment_metrics(val_df)

    # Save
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "segment_metrics_val.csv"
    segment_metrics.to_csv(out_path, index=False)

    print("\nSegment metrics:")
    print(segment_metrics)

    plot_segment_metrics(segment_metrics)

    print(f"\nSaved segment metrics to: {out_path}")

    return segment_metrics


if __name__ == "__main__":
    run_segment_analysis()