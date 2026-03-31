import json
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "rmse_log": rmse(y_true_log, y_pred_log),
        "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_original": rmse(y_true, y_pred),
        "mae_original": mean_absolute_error(y_true, y_pred),
        "r2_original": r2_score(y_true, y_pred),
    }


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_model_on_test(model, model_name, X_test, y_test, groups_test):
    y_test_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_test_pred)

    predictions = pd.DataFrame({
        "y_true_log": y_test,
        "y_pred_log": y_test_pred,
        "y_true": np.expm1(y_test),
        "y_pred": np.expm1(y_test_pred),
        "VIDEO_ID": groups_test.values,
        "model": model_name,
    })

    return metrics, predictions


def run_final_test_evaluation():
    print("=" * 60)
    print("Final held-out test evaluation")
    print("=" * 60)

    # Load processed data and recreate the same split
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

    print("\nTest split summary:")
    print(f"Rows: {len(X_test)}")
    print(f"Unique videos: {groups_test.nunique()}")

    # Load validation comparison to identify selected model
    val_comparison_path = TABLES_DIR / "model_comparison_val.csv"
    if not val_comparison_path.exists():
        raise FileNotFoundError(
            f"Validation comparison file not found: {val_comparison_path}"
        )

    val_comparison = pd.read_csv(val_comparison_path)
    selected_model = val_comparison.sort_values("rmse_log").iloc[0]["model"]

    print(f"\nModel selected on validation RMSE_log: {selected_model}")
    print("Test metrics below are for final held-out reporting only.")
    print("They were not used for model selection.")

    # Load models
    ridge_path = MODELS_DIR / "ridge_baseline.pkl"
    boosted_candidates = [
        ("lightgbm", MODELS_DIR / "lightgbm.pkl"),
        ("xgboost", MODELS_DIR / "xgboost.pkl"),
        ("hist_gradient_boosting", MODELS_DIR / "hist_gradient_boosting.pkl"),
    ]

    if not ridge_path.exists():
        raise FileNotFoundError(f"Ridge model not found: {ridge_path}")

    boosted_name = None
    boosted_path = None
    for model_name, path in boosted_candidates:
        if path.exists():
            boosted_name = model_name
            boosted_path = path
            break

    if boosted_path is None:
        raise FileNotFoundError("No boosted-model pickle found.")

    ridge_model = load_model(ridge_path)
    boosted_model = load_model(boosted_path)

    # Evaluate both models on held-out test
    ridge_metrics, ridge_predictions = evaluate_model_on_test(
        ridge_model, "ridge_baseline", X_test, y_test, groups_test
    )

    boosted_metrics, boosted_predictions = evaluate_model_on_test(
        boosted_model, boosted_name, X_test, y_test, groups_test
    )

    # Comparison table for transparency
    comparison_df = pd.DataFrame([
        {"model": "ridge_baseline", **ridge_metrics},
        {"model": boosted_name, **boosted_metrics},
    ]).sort_values("rmse_log").reset_index(drop=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = TABLES_DIR / "final_test_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\nFinal test comparison (for transparency only):")
    print(comparison_df)
    print(f"\nSaved final test comparison table to: {comparison_path}")

    # Save selected model's final outputs separately
    if selected_model == "ridge_baseline":
        selected_metrics = ridge_metrics
        selected_predictions = ridge_predictions
    else:
        selected_metrics = boosted_metrics
        selected_predictions = boosted_predictions

    selected_metrics_payload = {
        "selected_on": "validation_rmse_log",
        "selected_model": selected_model,
        "test_rows": int(X_test.shape[0]),
        "test_videos": int(groups_test.nunique()),
        "test_metrics": selected_metrics,
    }

    selected_metrics_path = TABLES_DIR / "selected_model_final_test_metrics.json"
    with open(selected_metrics_path, "w") as f:
        json.dump(selected_metrics_payload, f, indent=2)

    selected_predictions_path = TABLES_DIR / "selected_model_final_test_predictions.csv"
    selected_predictions.to_csv(selected_predictions_path, index=False)

    print(f"Saved selected model final test metrics to: {selected_metrics_path}")
    print(f"Saved selected model final test predictions to: {selected_predictions_path}")

    print(f"\nOfficial final model: {selected_model}")
    print("Official held-out test metrics:")
    for k, v in selected_metrics.items():
        print(f"{k}: {v:.4f}")

    return comparison_df, selected_metrics_payload, selected_predictions
        

if __name__ == "__main__":
    run_final_test_evaluation()