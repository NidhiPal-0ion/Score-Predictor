"""
src/evaluator.py
================
Model evaluation, metric computation, and output generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ── Evaluation of all models ───────────────────────────────────────────────────

def evaluate_all(
    trained: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Return:
        results_df  – DataFrame with one row per model and metric columns
        predictions – dict of model_name → y_pred array
    """
    rows = []
    predictions: Dict[str, np.ndarray] = {}

    for name, pipeline in trained.items():
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        metrics = compute_metrics(y_test, y_pred)
        rows.append({"Model": name, **metrics})
        logger.info("%s  →  %s", name, metrics)

    results_df = pd.DataFrame(rows).set_index("Model")
    return results_df, predictions


# ── Best model selection ───────────────────────────────────────────────────────

def select_best_model(results_df: pd.DataFrame, metric: str = "R2") -> str:
    """Return the model name with the best value for the given metric."""
    ascending = metric not in {"R2"}  # lower is better for MAE/MSE/RMSE
    return str(results_df[metric].sort_values(ascending=ascending).index[0])


# ── Output file writers ────────────────────────────────────────────────────────

def save_metrics(results_df: pd.DataFrame, path: Path = OUTPUTS_DIR / "metrics.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results_df.reset_index().to_dict(orient="records")
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Metrics saved → %s", path)


def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    path: Path = OUTPUTS_DIR / "test_predictions.csv",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = X_test.copy()
    df["actual_score"] = y_test.values
    for name, y_pred in predictions.items():
        safe = name.lower().replace(" ", "_")
        df[f"pred_{safe}"] = y_pred.round(2)
    df.to_csv(path, index=False)
    logger.info("Predictions saved → %s", path)


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    best_model: str,
    path: Path = OUTPUTS_DIR / "actual_vs_predicted.png",
) -> None:
    y_pred = predictions[best_model]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.55, edgecolors="white", linewidth=0.4, color="#4C72B0", s=45)
    lo, hi = min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.4, label="Perfect prediction")
    ax.set_xlabel("Actual Final Score", fontsize=12)
    ax.set_ylabel("Predicted Final Score", fontsize=12)
    ax.set_title(f"Actual vs Predicted  ·  {best_model}", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved → %s", path)


def plot_residuals(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    best_model: str,
    path: Path = OUTPUTS_DIR / "residuals.png",
) -> None:
    residuals = y_test.values - predictions[best_model]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(predictions[best_model], residuals, alpha=0.5, color="#DD8452", s=40)
    axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted Score")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")

    axes[1].hist(residuals, bins=25, color="#55A868", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis  ·  {best_model}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Residual plot saved → %s", path)


def plot_model_comparison(
    results_df: pd.DataFrame,
    path: Path = OUTPUTS_DIR / "model_comparison.png",
) -> None:
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colours = sns.color_palette("muted", n_colors=len(results_df))
    for ax, metric in zip(axes, metrics):
        values = results_df[metric]
        bars = ax.bar(values.index, values.values, color=colours, edgecolor="white")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticklabels(values.index, rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, values.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * abs(val),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison plot saved → %s", path)


def plot_feature_importance(
    trained: Dict[str, Pipeline],
    feature_names: list[str],
    path: Path = OUTPUTS_DIR / "feature_importance.png",
) -> None:
    """Plot feature importance for tree-based models and coefficients for linear ones."""
    importances: Dict[str, np.ndarray] = {}

    for name, pipeline in trained.items():
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances[name] = np.abs(model.coef_)

    if not importances:
        return

    n_plots = len(importances)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, imp) in zip(axes, importances.items()):
        sorted_idx = np.argsort(imp)
        ax.barh(
            [feature_names[i] for i in sorted_idx],
            imp[sorted_idx],
            color=sns.color_palette("muted")[0],
        )
        ax.set_title(f"{name}\nFeature Importance", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    path: Path = OUTPUTS_DIR / "correlation_heatmap.png",
) -> None:
    cols = feature_cols + ["final_score"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Correlation heatmap saved → %s", path)
