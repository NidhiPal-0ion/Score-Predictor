"""
src/predictor.py
================
Single-student score prediction utilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessor import FEATURE_COLS

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")


def predict_score(
    pipeline: Pipeline,
    student_data: Dict[str, float],
) -> float:
    """
    Predict the final score for a single student.

    Parameters
    ----------
    pipeline     : Fitted sklearn Pipeline (preprocessor + model).
    student_data : Dict mapping feature names to their values.

    Returns
    -------
    Predicted final score (float, clipped to [0, 100]).
    """
    df = pd.DataFrame([student_data])[FEATURE_COLS]
    raw = float(pipeline.predict(df)[0])
    return float(np.clip(raw, 0.0, 100.0))


def save_example_prediction(
    student_data: Dict[str, float],
    prediction: float,
    model_name: str,
    path: Path = OUTPUTS_DIR / "example_prediction.json",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "input_features": student_data,
        "predicted_final_score": round(prediction, 2),
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Example prediction saved → %s", path)


# ── Grade helper ───────────────────────────────────────────────────────────────

def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A (Excellent)"
    if score >= 80:
        return "B (Good)"
    if score >= 70:
        return "C (Average)"
    if score >= 60:
        return "D (Below Average)"
    return "F (Fail)"


def score_to_percentile_desc(score: float) -> str:
    """Return a qualitative description of where the score falls."""
    if score >= 90:
        return "Top 10% — Outstanding performance!"
    if score >= 75:
        return "Upper quartile — Strong performance."
    if score >= 50:
        return "Middle range — Room for improvement."
    return "Lower range — Consider additional support."
