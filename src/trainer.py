"""
src/trainer.py
==============
Model definitions, training, and persistence utilities.

Models trained:
    1. Linear Regression        – interpretable baseline
    2. Ridge Regression         – L2-regularised linear model
    3. Random Forest Regressor  – ensemble, captures non-linearities
    4. Gradient Boosting        – sequential boosting ensemble
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocessor import build_preprocessor

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model catalogue ────────────────────────────────────────────────────────────

def get_model_catalogue() -> Dict[str, object]:
    """Return a dict of name → unfitted estimator."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        ),
    }


# ── Pipeline builder ───────────────────────────────────────────────────────────

def build_pipeline(estimator) -> Pipeline:
    """Wrap a preprocessor + estimator into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Pipeline]:
    """Train all models and return a dict of name → fitted pipeline."""
    catalogue = get_model_catalogue()
    trained: Dict[str, Pipeline] = {}

    for name, estimator in catalogue.items():
        logger.info("Training: %s …", name)
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        logger.info("  ✓ %s trained.", name)

    return trained


# ── Persistence ────────────────────────────────────────────────────────────────

def save_models(trained: Dict[str, Pipeline], directory: Path = MODELS_DIR) -> None:
    """Serialise all trained pipelines to disk with joblib."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, pipeline in trained.items():
        safe_name = name.lower().replace(" ", "_")
        path = directory / f"{safe_name}.pkl"
        joblib.dump(pipeline, path)
        logger.info("Saved model '%s' → %s", name, path)


def load_model(name: str, directory: Path = MODELS_DIR) -> Pipeline:
    """Load a saved pipeline by model name."""
    safe_name = name.lower().replace(" ", "_")
    path = directory / f"{safe_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run main.py first.")
    return joblib.load(path)


def load_all_models(directory: Path = MODELS_DIR) -> Dict[str, Pipeline]:
    """Load every .pkl file in the models directory."""
    models = {}
    for pkl in sorted(directory.glob("*.pkl")):
        models[pkl.stem] = joblib.load(pkl)
        logger.info("Loaded model: %s", pkl.stem)
    return models
