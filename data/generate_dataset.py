"""
Synthetic Student Performance Dataset Generator
================================================
Generates a realistic synthetic dataset of 500 student records.

Features:
    - study_hours_per_week  : Average study hours per week (2–40)
    - attendance_percentage : Class attendance rate (40–100%)
    - previous_exam_score   : Score on the previous exam (30–100)
    - assignments_completed : Number of assignments submitted (0–20)
    - sleep_hours           : Average nightly sleep (4–10 hrs)
    - parental_education    : Highest parental education level (ordinal 0–4)
    - tutoring_sessions     : Extra tutoring sessions per month (0–10)
    - stress_level          : Self-reported stress level (1–10, lower is less)

Target:
    - final_score           : Final exam score (0–100, clipped)

Dataset is fully synthetic. No real student data is used.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 500


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(2, 40, n)
    attendance = rng.uniform(40, 100, n)
    prev_score = rng.uniform(30, 100, n)
    assignments = rng.integers(0, 21, n).astype(float)
    sleep_hours = rng.uniform(4, 10, n)
    parental_edu = rng.integers(0, 5, n).astype(float)   # 0=none … 4=postgrad
    tutoring = rng.integers(0, 11, n).astype(float)
    stress = rng.uniform(1, 10, n)

    # Build target with realistic, non-linear contributions + noise
    final_score = (
        0.70 * study_hours
        + 0.25 * attendance
        + 0.30 * prev_score
        + 1.20 * assignments
        + 1.50 * sleep_hours
        + 2.00 * parental_edu
        + 1.80 * tutoring
        - 1.00 * stress
        + rng.normal(0, 4, n)
    )

    # Normalise to a 0-100 range
    final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 100
    final_score = np.clip(final_score, 0, 100).round(2)

    df = pd.DataFrame(
        {
            "student_id": [f"S{i+1:04d}" for i in range(n)],
            "study_hours_per_week": study_hours.round(1),
            "attendance_percentage": attendance.round(1),
            "previous_exam_score": prev_score.round(1),
            "assignments_completed": assignments,
            "sleep_hours": sleep_hours.round(1),
            "parental_education": parental_edu,
            "tutoring_sessions": tutoring,
            "stress_level": stress.round(1),
            "final_score": final_score,
        }
    )

    # ── Inject deliberate imperfections for preprocessing practice ──
    # 10 missing values spread across feature columns
    feature_cols = [
        "study_hours_per_week",
        "attendance_percentage",
        "previous_exam_score",
        "assignments_completed",
        "sleep_hours",
    ]
    for col in feature_cols:
        missing_idx = rng.choice(n, size=2, replace=False)
        df.loc[missing_idx, col] = np.nan

    # 5 exact duplicate rows
    dup_rows = df.sample(5, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main() -> None:
    out_path = Path(__file__).parent / "student_scores.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}  ({len(df)} rows, {df.shape[1]} columns)")
    print(df.describe())


if __name__ == "__main__":
    main()
