"""
main.py
=======
Training and evaluation entry point.

Usage:
    python main.py

What it does:
    1. Generates the synthetic dataset (if not already present).
    2. Loads and validates the data.
    3. Cleans (removes duplicates, handles missing values).
    4. Splits into train / test sets (80 / 20).
    5. Trains Linear Regression, Ridge, Random Forest, and Gradient Boosting.
    6. Evaluates each model on the held-out test set.
    7. Selects the best model by R².
    8. Saves trained models, evaluation metrics, predictions, and plots.
    9. Runs an example prediction with sample student attributes.
"""

import json
import logging
import sys
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_scores.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Ensure outputs directories exist ──────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Student Score Predictor — Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Generate dataset if needed ────────────────────────────────────
    if not DATA_PATH.exists():
        logger.info("Dataset not found. Generating synthetic dataset …")
        from data.generate_dataset import generate
        df_gen = generate()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(DATA_PATH, index=False)
        logger.info("Dataset written to %s", DATA_PATH)
    else:
        logger.info("Dataset found at %s", DATA_PATH)

    # ── Step 2: Load & validate ────────────────────────────────────────────────
    from src.preprocessor import load_data, clean_data, split_data, FEATURE_COLS
    df = load_data(DATA_PATH)
    logger.info("Raw dataset shape: %s", df.shape)

    # ── Step 3: EDA plots ──────────────────────────────────────────────────────
    logger.info("Generating EDA visualisations …")
    from src.eda import plot_score_distribution, plot_feature_vs_target
    from src.evaluator import plot_correlation_heatmap
    plot_score_distribution(df, OUTPUTS_DIR / "score_distribution.png")
    plot_feature_vs_target(df, FEATURE_COLS, OUTPUTS_DIR / "feature_vs_target.png")
    plot_correlation_heatmap(df, FEATURE_COLS, OUTPUTS_DIR / "correlation_heatmap.png")

    # ── Step 4: Clean ──────────────────────────────────────────────────────────
    df, clean_stats = clean_data(df)
    logger.info("Clean stats: %s", clean_stats)
    logger.info("Clean dataset shape: %s", df.shape)

    # Save clean stats
    (OUTPUTS_DIR / "clean_stats.json").write_text(json.dumps(clean_stats, indent=2))

    # ── Step 5: Split ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # ── Step 6: Train all models ───────────────────────────────────────────────
    from src.trainer import train_all, save_models
    logger.info("Training models …")
    trained = train_all(X_train, y_train)
    save_models(trained, MODELS_DIR)

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    from src.evaluator import (
        evaluate_all,
        select_best_model,
        save_metrics,
        save_predictions,
        plot_actual_vs_predicted,
        plot_residuals,
        plot_model_comparison,
        plot_feature_importance,
    )
    results_df, predictions = evaluate_all(trained, X_test, y_test)

    logger.info("\n%s", results_df.to_string())

    best_model = select_best_model(results_df, metric="R2")
    logger.info("Best model (by R²): %s", best_model)

    # ── Step 8: Save outputs ───────────────────────────────────────────────────
    save_metrics(results_df, OUTPUTS_DIR / "metrics.json")
    save_predictions(X_test, y_test, predictions, OUTPUTS_DIR / "test_predictions.csv")
    plot_actual_vs_predicted(y_test, predictions, best_model, OUTPUTS_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test, predictions, best_model, OUTPUTS_DIR / "residuals.png")
    plot_model_comparison(results_df, OUTPUTS_DIR / "model_comparison.png")
    plot_feature_importance(trained, FEATURE_COLS, OUTPUTS_DIR / "feature_importance.png")

    # Save best model name for app.py to pick up
    (OUTPUTS_DIR / "best_model.txt").write_text(best_model)

    # ── Step 9: Example prediction ─────────────────────────────────────────────
    from src.predictor import predict_score, save_example_prediction, score_to_grade

    example_student = {
        "study_hours_per_week": 20.0,
        "attendance_percentage": 88.0,
        "previous_exam_score": 74.0,
        "assignments_completed": 14.0,
        "sleep_hours": 7.5,
        "parental_education": 3.0,
        "tutoring_sessions": 4.0,
        "stress_level": 4.0,
    }

    best_pipeline = trained[best_model]
    pred = predict_score(best_pipeline, example_student)
    grade = score_to_grade(pred)
    logger.info("Example prediction → %.2f  (%s)", pred, grade)
    save_example_prediction(example_student, pred, best_model, OUTPUTS_DIR / "example_prediction.json")

    logger.info("=" * 60)
    logger.info("  Training complete. All outputs saved to '%s'.", OUTPUTS_DIR)
    logger.info("  Launch dashboard:  streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
