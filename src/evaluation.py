import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import TABLES_DIR, FIGURES_DIR


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_metrics_comparison():
    ridge_metrics_path = TABLES_DIR / "ridge_baseline_metrics.json"
    boosted_candidates = [
        TABLES_DIR / "lightgbm_metrics.json",
        TABLES_DIR / "xgboost_metrics.json",
        TABLES_DIR / "hist_gradient_boosting_metrics.json",
    ]

    ridge_metrics = load_json(ridge_metrics_path)

    boosted_metrics = None
    for path in boosted_candidates:
        if path.exists():
            boosted_metrics = load_json(path)
            break

    if boosted_metrics is None:
        raise FileNotFoundError("No boosted-model metrics JSON found.")

    comparison_df = pd.DataFrame([
        {
            "model": ridge_metrics["model_name"],
            **ridge_metrics["val_metrics"],
        },
        {
            "model": boosted_metrics["model_name"],
            **boosted_metrics["val_metrics"],
        },
    ])

    comparison_df = comparison_df.sort_values("rmse_log").reset_index(drop=True)
    best_model = comparison_df.loc[0, "model"]

    return comparison_df, best_model


def load_predictions():
    ridge_preds = pd.read_csv(TABLES_DIR / "ridge_baseline_val_predictions.csv")

    boosted_candidates = [
        ("lightgbm", TABLES_DIR / "lightgbm_val_predictions.csv"), # not used model on this version
        ("xgboost", TABLES_DIR / "xgboost_val_predictions.csv"), # not used model on this version
        ("hist_gradient_boosting", TABLES_DIR / "hist_gradient_boosting_val_predictions.csv"),
    ]

    for model_name, path in boosted_candidates:
        if path.exists():
            boosted_preds = pd.read_csv(path)
            return ridge_preds, boosted_preds, model_name

    raise FileNotFoundError("No boosted-model prediction CSV found.")


def save_metrics_table(comparison_df: pd.DataFrame):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "model_comparison_val.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"Saved validation metrics comparison table to: {out_path}")


def plot_actual_vs_predicted(preds_df: pd.DataFrame, model_name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(preds_df["y_true_log"], preds_df["y_pred_log"], alpha=0.15)
    plt.xlabel("Actual log(FUTURE_VIEWS)")
    plt.ylabel("Predicted log(FUTURE_VIEWS)")
    plt.title(f"Actual vs Predicted ({model_name}, validation)")

    min_val = min(preds_df["y_true_log"].min(), preds_df["y_pred_log"].min())
    max_val = max(preds_df["y_true_log"].max(), preds_df["y_pred_log"].max())
    plt.plot([min_val, max_val], [min_val, max_val])

    out_path = FIGURES_DIR / f"{model_name}_actual_vs_predicted_log_val.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")


def plot_residuals(preds_df: pd.DataFrame, model_name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = preds_df["y_true_log"] - preds_df["y_pred_log"]

    plt.figure(figsize=(6, 4))
    plt.scatter(preds_df["y_pred_log"], residuals, alpha=0.15)
    plt.xlabel("Predicted log(FUTURE_VIEWS)")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({model_name}, validation)")
    plt.axhline(0)

    out_path = FIGURES_DIR / f"{model_name}_residuals_log_val.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")


def run_evaluation():
    comparison_df, best_model = build_metrics_comparison()

    print("\nValidation model comparison:")
    print(comparison_df)
    print(f"\nSelected model based on validation RMSE_log: {best_model}")

    save_metrics_table(comparison_df)

    ridge_preds, boosted_preds, boosted_model_name = load_predictions()

    plot_actual_vs_predicted(ridge_preds, "ridge_baseline")
    plot_residuals(ridge_preds, "ridge_baseline")

    plot_actual_vs_predicted(boosted_preds, boosted_model_name)
    plot_residuals(boosted_preds, boosted_model_name)

    return comparison_df, best_model


if __name__ == "__main__":
    run_evaluation()