"""
src/preprocessor.py
====================
Data loading, validation, and preprocessing utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Column definitions ─────────────────────────────────────────────────────────
TARGET_COL = "final_score"
DROP_COLS = ["student_id"]          # identifiers, not features
FEATURE_COLS = [
    "study_hours_per_week",
    "attendance_percentage",
    "previous_exam_score",
    "assignments_completed",
    "sleep_hours",
    "parental_education",
    "tutoring_sessions",
    "stress_level",
]
REQUIRED_COLS = FEATURE_COLS + [TARGET_COL]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV and perform basic structural validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df[TARGET_COL].isnull().all():
        raise ValueError(f"Target column '{TARGET_COL}' contains no valid values.")

    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Remove duplicates and rows where the target is missing."""
    stats: dict = {}

    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = before - len(df)
    logger.info("Removed %d duplicate rows.", stats["duplicates_removed"])

    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    stats["target_nulls_dropped"] = before - len(df)

    missing_feature_counts = df[FEATURE_COLS].isnull().sum().to_dict()
    stats["missing_features"] = {k: int(v) for k, v in missing_feature_counts.items() if v > 0}
    logger.info("Missing feature values per column: %s", stats["missing_features"])

    return df.reset_index(drop=True), stats


# ── Train / test split ─────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """Return a sklearn Pipeline that imputes then scales numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_feature_names() -> list[str]:
    return FEATURE_COLS
