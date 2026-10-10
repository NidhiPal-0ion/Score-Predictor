"""
tests/test_pipeline.py
=======================
Automated tests for the Student Score Predictor pipeline.

Run with:  pytest -q
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path so src.* imports work without installing
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.generate_dataset import generate
from src.preprocessor import (
    FEATURE_COLS,
    TARGET_COL,
    clean_data,
    load_data,
    split_data,
    build_preprocessor,
)
from src.trainer import build_pipeline, get_model_catalogue, train_all
from src.evaluator import compute_metrics, evaluate_all, select_best_model
from src.predictor import predict_score, score_to_grade


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Generate a fresh synthetic dataset for testing."""
    return generate(n=200, seed=0)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    df, _ = clean_data(raw_df.copy())
    return df


@pytest.fixture(scope="module")
def split_data_fixture(clean_df):
    return split_data(clean_df)


@pytest.fixture(scope="module")
def trained_models(split_data_fixture):
    X_train, X_test, y_train, y_test = split_data_fixture
    return train_all(X_train, y_train)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Dataset generation
# ══════════════════════════════════════════════════════════════════════════════

class TestDataGeneration:
    def test_shape(self, raw_df):
        """Dataset must have expected columns and at least n rows."""
        assert len(raw_df) >= 200
        assert "final_score" in raw_df.columns
        for col in FEATURE_COLS:
            assert col in raw_df.columns

    def test_target_range(self, raw_df):
        """final_score should be in [0, 100]."""
        valid = raw_df["final_score"].dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_has_duplicates(self, raw_df):
        """Generator must inject at least some duplicate rows."""
        assert raw_df.duplicated().sum() >= 1

    def test_has_missing_values(self, raw_df):
        """Generator must inject missing feature values."""
        assert raw_df[FEATURE_COLS].isnull().sum().sum() >= 1

    def test_reproducibility(self):
        """Same seed → identical DataFrames."""
        df1 = generate(n=100, seed=7)
        df2 = generate(n=100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Data cleaning
# ══════════════════════════════════════════════════════════════════════════════

class TestCleaning:
    def test_duplicates_removed(self, raw_df, clean_df):
        """Clean dataset must have fewer or equal rows (duplicates gone)."""
        assert len(clean_df) <= len(raw_df)
        assert clean_df.duplicated().sum() == 0

    def test_no_target_nulls(self, clean_df):
        """There must be no missing values in the target column after cleaning."""
        assert clean_df[TARGET_COL].isnull().sum() == 0

    def test_stats_dict_keys(self, raw_df):
        _, stats = clean_data(raw_df.copy())
        assert "duplicates_removed" in stats
        assert "target_nulls_dropped" in stats
        assert "missing_features" in stats


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Train / test split
# ══════════════════════════════════════════════════════════════════════════════

class TestSplit:
    def test_split_sizes(self, clean_df, split_data_fixture):
        X_train, X_test, y_train, y_test = split_data_fixture
        total = len(clean_df)
        assert len(X_train) + len(X_test) == total
        assert abs(len(X_test) / total - 0.20) < 0.05  # roughly 20 %

    def test_no_leakage(self, split_data_fixture):
        """Train and test indices must not overlap."""
        X_train, X_test, y_train, y_test = split_data_fixture
        assert set(X_train.index).isdisjoint(set(X_test.index))

    def test_feature_columns(self, split_data_fixture):
        X_train, X_test, y_train, y_test = split_data_fixture
        assert list(X_train.columns) == FEATURE_COLS


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Preprocessing pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    def test_no_missing_after_transform(self, split_data_fixture):
        X_train, X_test, y_train, y_test = split_data_fixture
        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_train)
        X_te = prep.transform(X_test)
        assert not np.any(np.isnan(X_tr))
        assert not np.any(np.isnan(X_te))

    def test_output_shape(self, split_data_fixture):
        X_train, X_test, y_train, y_test = split_data_fixture
        prep = build_preprocessor()
        X_tr = prep.fit_transform(X_train)
        assert X_tr.shape == (len(X_train), len(FEATURE_COLS))


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Model training
# ══════════════════════════════════════════════════════════════════════════════

class TestTraining:
    def test_all_models_trained(self, trained_models):
        expected = set(get_model_catalogue().keys())
        assert expected == set(trained_models.keys())

    def test_pipelines_fitted(self, trained_models, split_data_fixture):
        X_train, X_test, y_train, y_test = split_data_fixture
        for name, pipe in trained_models.items():
            pred = pipe.predict(X_test)
            assert pred.shape == (len(X_test),), f"Bad output shape for {name}"

    def test_pipeline_output_clippable(self, trained_models, split_data_fixture):
        """Raw predictions may exceed [0,100] but should be close."""
        X_train, X_test, y_train, y_test = split_data_fixture
        for name, pipe in trained_models.items():
            pred = pipe.predict(X_test)
            # Allow ±10 headroom around [0, 100]
            assert pred.min() > -10, f"{name}: prediction too low"
            assert pred.max() < 110, f"{name}: prediction too high"


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    def test_compute_metrics_keys(self, split_data_fixture, trained_models):
        X_train, X_test, y_train, y_test = split_data_fixture
        first_pipeline = next(iter(trained_models.values()))
        y_pred = first_pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        for key in ("MAE", "MSE", "RMSE", "R2"):
            assert key in metrics

    def test_mae_non_negative(self, split_data_fixture, trained_models):
        X_train, X_test, y_train, y_test = split_data_fixture
        for name, pipe in trained_models.items():
            y_pred = pipe.predict(X_test)
            m = compute_metrics(y_test, y_pred)
            assert m["MAE"] >= 0, f"Negative MAE for {name}"

    def test_r2_reasonable(self, split_data_fixture, trained_models):
        """At least one model should achieve R² > 0.5 on synthetic data."""
        X_train, X_test, y_train, y_test = split_data_fixture
        r2_scores = []
        for name, pipe in trained_models.items():
            y_pred = pipe.predict(X_test)
            m = compute_metrics(y_test, y_pred)
            r2_scores.append(m["R2"])
        assert max(r2_scores) > 0.50, f"Best R²={max(r2_scores):.4f} is too low"

    def test_evaluate_all_returns_df(self, split_data_fixture, trained_models):
        X_train, X_test, y_train, y_test = split_data_fixture
        results_df, predictions = evaluate_all(trained_models, X_test, y_test)
        assert isinstance(results_df, pd.DataFrame)
        assert set(results_df.columns) == {"MAE", "MSE", "RMSE", "R2"}
        assert set(predictions.keys()) == set(trained_models.keys())

    def test_select_best_model(self, split_data_fixture, trained_models):
        X_train, X_test, y_train, y_test = split_data_fixture
        results_df, _ = evaluate_all(trained_models, X_test, y_test)
        best = select_best_model(results_df, "R2")
        assert best in trained_models


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestPrediction:
    EXAMPLE_STUDENT = {
        "study_hours_per_week": 20.0,
        "attendance_percentage": 88.0,
        "previous_exam_score": 74.0,
        "assignments_completed": 14.0,
        "sleep_hours": 7.5,
        "parental_education": 3.0,
        "tutoring_sessions": 4.0,
        "stress_level": 4.0,
    }

    def test_predict_returns_float(self, trained_models):
        pipe = next(iter(trained_models.values()))
        result = predict_score(pipe, self.EXAMPLE_STUDENT)
        assert isinstance(result, float)

    def test_predict_in_range(self, trained_models):
        """Clipped prediction must be in [0, 100]."""
        for name, pipe in trained_models.items():
            result = predict_score(pipe, self.EXAMPLE_STUDENT)
            assert 0.0 <= result <= 100.0, f"{name}: {result} out of range"

    def test_predict_missing_feature_handled(self, trained_models):
        """Pipeline should handle a missing value via imputer."""
        pipe = next(iter(trained_models.values()))
        student = self.EXAMPLE_STUDENT.copy()
        student["sleep_hours"] = float("nan")
        result = predict_score(pipe, student)
        assert 0.0 <= result <= 100.0

    def test_grade_labels(self):
        assert score_to_grade(95) == "A (Excellent)"
        assert score_to_grade(85) == "B (Good)"
        assert score_to_grade(72) == "C (Average)"
        assert score_to_grade(62) == "D (Below Average)"
        assert score_to_grade(45) == "F (Fail)"

    def test_grade_boundaries(self):
        assert score_to_grade(90) == "A (Excellent)"
        assert score_to_grade(89.9) == "B (Good)"
