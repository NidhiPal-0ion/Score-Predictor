"""
src/eda.py
==========
Exploratory Data Analysis helpers.
Generates and saves visualisations for the EDA section of the dashboard
and for the outputs directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
sns.set_theme(style="whitegrid", palette="muted")

FEATURE_LABELS = {
    "study_hours_per_week": "Study Hours / Week",
    "attendance_percentage": "Attendance (%)",
    "previous_exam_score": "Previous Exam Score",
    "assignments_completed": "Assignments Completed",
    "sleep_hours": "Sleep Hours / Night",
    "parental_education": "Parental Education (0–4)",
    "tutoring_sessions": "Tutoring Sessions / Month",
    "stress_level": "Stress Level (1–10)",
    "final_score": "Final Score",
}


def plot_score_distribution(
    df: pd.DataFrame,
    path: Path = OUTPUTS_DIR / "score_distribution.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df["final_score"], bins=25, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("Final Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution")

    sns.boxplot(y=df["final_score"], ax=axes[1], color="#4C72B0", width=0.4)
    axes[1].set_ylabel("Final Score")
    axes[1].set_title("Score Box Plot")

    fig.suptitle("Final Score Overview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Score distribution plot saved → %s", path)


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    path: Path = OUTPUTS_DIR / "feature_vs_target.png",
) -> None:
    n = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        ax.scatter(df[col], df["final_score"], alpha=0.35, s=20, color="#4C72B0")
        # trend line
        valid = df[[col, "final_score"]].dropna()
        m, b = np.polyfit(valid[col], valid["final_score"], 1)
        x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="#C44E52", linewidth=1.5)
        ax.set_xlabel(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.set_ylabel("Final Score", fontsize=9)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9, fontweight="bold")

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature vs Final Score", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature vs target plot saved → %s", path)


def plot_pairplot(
    df: pd.DataFrame,
    cols: list[str],
    path: Path = OUTPUTS_DIR / "pairplot.png",
) -> None:
    subset = df[cols + ["final_score"]].dropna()
    # Use a sample to keep pairplot manageable
    sample = subset.sample(min(200, len(subset)), random_state=42)
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20})
    g.figure.suptitle("Pairplot (sample n=200)", y=1.01, fontsize=12, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=100)
    plt.close("all")
    logger.info("Pairplot saved → %s", path)
