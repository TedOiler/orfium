import json
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    TABLES_DIR,
    RANDOM_SEED,
)
from src.feature_engineering import load_processed_data, build_model_dataset
from src.split import make_grouped_train_val_test_split


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true_log, y_pred_log):
    """
    Compute metrics on both log scale and original scale.
    Assumes y_true_log and y_pred_log are on log1p scale.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    metrics = {
        "rmse_log": rmse(y_true_log, y_pred_log),
        "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_original": rmse(y_true, y_pred),
        "mae_original": mean_absolute_error(y_true, y_pred),
        "r2_original": r2_score(y_true, y_pred),
    }
    return metrics


def train_baseline_model():
    # Load and prepare data
    df = load_processed_data(PROCESSED_DATA_PATH)
    X, y, groups, feature_cols = build_model_dataset(df)

    (X_train, X_val, X_test,
    y_train, y_val, y_test,
    groups_train, groups_val, groups_test,) = make_grouped_train_val_test_split(
        X,y,groups,
        val_size=0.15,
        test_size=0.15,
        random_state=RANDOM_SEED,)

    # Define model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0, solver="svd")),
    ])

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    metrics = {
        "model_name": "ridge_baseline",
        "n_train_rows": int(X_train.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_train_videos": int(groups_train.nunique()),
        "n_val_videos": int(groups_val.nunique()),
        "n_test_videos": int(groups_test.nunique()),
        "feature_cols": feature_cols,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    # Predictions dataframe
    val_predictions = pd.DataFrame({
        "y_true_log": y_val,
        "y_pred_log": y_val_pred,
        "y_true": np.expm1(y_val),
        "y_pred": np.expm1(y_val_pred),
        "VIDEO_ID": groups_val.values,
    })

    # Save outputs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "ridge_baseline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics_path = TABLES_DIR / "ridge_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    preds_path = TABLES_DIR / "ridge_baseline_val_predictions.csv"
    val_predictions.to_csv(preds_path, index=False)

    print("Baseline training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {preds_path}")
    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics, val_predictions


if __name__ == "__main__":
    train_baseline_model()