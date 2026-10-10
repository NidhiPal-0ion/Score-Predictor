"""
app.py
======
Streamlit dashboard for the Student Score Predictor.

Sections:
    1. 🏠 Overview         — project summary and key stats
    2. 📊 Data Insights    — EDA charts and dataset stats
    3. 🤖 Model Evaluation — comparison table and plots
    4. 🎯 Predict Score    — interactive prediction form
    5. ℹ️  About           — methodology and limitations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_scores.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        color: white;
    }
    [data-testid="stSidebar"] .css-pkbazv { color: white; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #4C72B0;
    }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
        margin: 20px 0;
    }
    .pred-box h1 { font-size: 3.6rem; margin: 0; }
    .pred-box h3 { font-size: 1.3rem; margin: 4px 0 0 0; opacity: 0.9; }

    /* Grade badge */
    .grade-A  { background:#2ecc71; color:white; border-radius:8px; padding:6px 14px; font-weight:700; }
    .grade-B  { background:#3498db; color:white; border-radius:8px; padding:6px 14px; font-weight:700; }
    .grade-C  { background:#f39c12; color:white; border-radius:8px; padding:6px 14px; font-weight:700; }
    .grade-D  { background:#e67e22; color:white; border-radius:8px; padding:6px 14px; font-weight:700; }
    .grade-F  { background:#e74c3c; color:white; border-radius:8px; padding:6px 14px; font-weight:700; }

    /* Section header */
    .section-header {
        border-bottom: 2px solid #4C72B0;
        padding-bottom: 6px;
        margin-bottom: 18px;
        color: #1a1a2e;
    }

    /* Info banner */
    .info-banner {
        background: #e8f4fd;
        border-left: 4px solid #3498db;
        border-radius: 6px;
        padding: 14px 20px;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "metrics.json"
    if path.exists():
        records = json.loads(path.read_text())
        return pd.DataFrame(records).set_index("Model")
    return None


@st.cache_data
def load_predictions_csv() -> pd.DataFrame | None:
    path = OUTPUTS_DIR / "test_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_models():
    """Load all saved pipelines."""
    try:
        from src.trainer import load_all_models
        return load_all_models(MODELS_DIR)
    except Exception:
        return {}


def get_best_model_name() -> str | None:
    path = OUTPUTS_DIR / "best_model.txt"
    if path.exists():
        return path.read_text().strip()
    return None


def score_to_grade(score: float) -> tuple[str, str]:
    """Return (letter, css_class)."""
    if score >= 90:
        return "A — Excellent", "grade-A"
    if score >= 80:
        return "B — Good", "grade-B"
    if score >= 70:
        return "C — Average", "grade-C"
    if score >= 60:
        return "D — Below Average", "grade-D"
    return "F — Fail", "grade-F"


def models_are_trained() -> bool:
    return any(MODELS_DIR.glob("*.pkl"))


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h2 style='color:white;text-align:center;'>🎓 Score Predictor</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#ffffff30'>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Data Insights", "🤖 Model Evaluation", "🎯 Predict Score", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#ffffff30'>", unsafe_allow_html=True)

    if not models_are_trained():
        st.warning("⚠️ Models not trained yet.\nRun `python main.py` first.")
    else:
        st.success("✅ Models ready")

    best = get_best_model_name()
    if best:
        st.info(f"**Best model:** {best}")

    st.markdown(
        "<p style='color:#aaa;font-size:0.75rem;text-align:center;margin-top:auto'>Built with scikit-learn & Streamlit</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    st.markdown("<h1 style='color:#1a1a2e'>🎓 Student Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        **Welcome!** This application uses machine learning to predict a student's final exam score
        based on their study habits, attendance, previous performance, and lifestyle factors.
        It is a portfolio project demonstrating a complete ML workflow — from raw data to a live dashboard.
        """
    )

    df = load_dataset()
    metrics_df = load_metrics()

    # ── Key stats ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Dataset Rows", f"{len(df):,}" if df is not None else "—")
    with col2:
        st.metric("🔢 Features", "8")
    with col3:
        st.metric("🤖 Models Trained", len(list(MODELS_DIR.glob("*.pkl"))) if MODELS_DIR.exists() else 0)
    with col4:
        best_r2 = metrics_df["R2"].max() if metrics_df is not None else None
        st.metric("📈 Best R²", f"{best_r2:.4f}" if best_r2 else "—")

    st.divider()

    # ── Feature overview table ─────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("### 📌 Input Features")
        features_info = pd.DataFrame(
            {
                "Feature": [
                    "study_hours_per_week",
                    "attendance_percentage",
                    "previous_exam_score",
                    "assignments_completed",
                    "sleep_hours",
                    "parental_education",
                    "tutoring_sessions",
                    "stress_level",
                ],
                "Description": [
                    "Avg study hours per week",
                    "Class attendance rate (%)",
                    "Score on previous exam",
                    "Assignments submitted (0–20)",
                    "Avg nightly sleep (hrs)",
                    "Parental education level (0–4)",
                    "Extra tutoring sessions / month",
                    "Self-reported stress (1–10)",
                ],
                "Range": [
                    "2–40", "40–100", "30–100", "0–20", "4–10", "0–4", "0–10", "1–10"
                ],
            }
        )
        st.dataframe(features_info, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("### 🎯 Target Variable")
        st.markdown(
            """
            **`final_score`** — Student's final exam score (continuous, 0–100).

            #### Workflow Summary
            ```
            CSV Dataset
               ↓
            Validation & Cleaning
               ↓
            Train / Test Split (80/20)
               ↓
            Median Imputation + Scaling
               ↓
            Train 4 Regression Models
               ↓
            Evaluate (MAE, MSE, RMSE, R²)
               ↓
            Select Best Model
               ↓
            Interactive Dashboard
            ```
            """
        )

    # ── Quick-start ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🚀 Quick Start")
    st.code(
        "python -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1\n"
        "pip install -r requirements.txt\n"
        "python main.py\n"
        "streamlit run app.py",
        language="powershell",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Data Insights":
    st.markdown("<h1 class='section-header'>📊 Data Insights</h1>", unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Run `python main.py` to generate it.")
        st.stop()

    # ── Dataset preview ────────────────────────────────────────────────────────
    with st.expander("🔍 Dataset Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ── Summary stats ──────────────────────────────────────────────────────────
    with st.expander("📐 Descriptive Statistics", expanded=False):
        from src.preprocessor import FEATURE_COLS
        st.dataframe(df[FEATURE_COLS + ["final_score"]].describe().T.round(2), use_container_width=True)

    # ── Missing values ─────────────────────────────────────────────────────────
    st.markdown("### 🔍 Data Quality")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))
    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0]
    if not missing_series.empty:
        st.markdown("**Columns with missing values:**")
        st.dataframe(
            missing_series.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # ── EDA Charts ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Visualisations")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Score Distribution", "Feature vs Target", "Correlation Heatmap", "Feature Distributions"]
    )

    with tab1:
        img = OUTPUTS_DIR / "score_distribution.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info("Run `python main.py` to generate charts.")

    with tab2:
        img = OUTPUTS_DIR / "feature_vs_target.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info("Run `python main.py` to generate charts.")

    with tab3:
        img = OUTPUTS_DIR / "correlation_heatmap.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info("Run `python main.py` to generate charts.")

    with tab4:
        from src.preprocessor import FEATURE_COLS
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        selected_feat = st.selectbox("Select a feature:", FEATURE_COLS)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_feat].dropna(), kde=True, ax=ax, color="#4C72B0", bins=25)
        ax.set_xlabel(selected_feat)
        ax.set_title(f"Distribution of {selected_feat}")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Model Evaluation":
    st.markdown("<h1 class='section-header'>🤖 Model Evaluation</h1>", unsafe_allow_html=True)

    metrics_df = load_metrics()
    if metrics_df is None:
        st.warning("No evaluation results found. Please run `python main.py` first.")
        st.stop()

    best = get_best_model_name()

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance Metrics (Test Set)")

    styled = metrics_df.style.highlight_min(
        subset=["MAE", "MSE", "RMSE"], color="#d4edda"
    ).highlight_max(subset=["R2"], color="#d4edda").format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    if best:
        st.success(f"🏆 **Best Model:** {best}  (highest R² on the held-out test set)")

    st.markdown(
        """
        <div class='info-banner'>
        <b>Metric guide:</b><br>
        • <b>MAE</b> (Mean Absolute Error) — average absolute prediction error in score points.<br>
        • <b>MSE</b> (Mean Squared Error) — penalises large errors more heavily.<br>
        • <b>RMSE</b> (Root MSE) — same unit as the target; easier to interpret.<br>
        • <b>R²</b> — proportion of variance explained (1.0 = perfect fit).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Plots ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model Comparison", "Actual vs Predicted", "Residual Analysis", "Feature Importance"]
    )

    with tab1:
        img = OUTPUTS_DIR / "model_comparison.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    with tab2:
        img = OUTPUTS_DIR / "actual_vs_predicted.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    with tab3:
        img = OUTPUTS_DIR / "residuals.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    with tab4:
        img = OUTPUTS_DIR / "feature_importance.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info("Feature importance is shown for tree-based and linear models.")

    # ── Raw predictions ────────────────────────────────────────────────────────
    with st.expander("🗃️ View Test-Set Predictions CSV"):
        pred_df = load_predictions_csv()
        if pred_df is not None:
            st.dataframe(pred_df.head(50), use_container_width=True)
            st.download_button(
                "⬇️ Download full predictions",
                pred_df.to_csv(index=False),
                "test_predictions.csv",
                "text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICT SCORE
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🎯 Predict Score":
    st.markdown("<h1 class='section-header'>🎯 Predict a Student's Score</h1>", unsafe_allow_html=True)

    if not models_are_trained():
        st.error("⚠️ No trained models found. Please run `python main.py` first.")
        st.stop()

    trained = load_models()
    if not trained:
        st.error("Could not load models. Please run `python main.py` first.")
        st.stop()

    model_display_names = {
        "linear_regression": "Linear Regression",
        "ridge_regression": "Ridge Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    available = {model_display_names.get(k, k): v for k, v in trained.items()}
    best_name = get_best_model_name()
    default_idx = list(available.keys()).index(best_name) if best_name in available else 0

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📝 Student Attributes")
        selected_model = st.selectbox("Model to use:", list(available.keys()), index=default_idx)

        with st.form("prediction_form"):
            study_hours = st.slider("📚 Study Hours / Week", 2.0, 40.0, 20.0, 0.5)
            attendance = st.slider("🏫 Attendance (%)", 40.0, 100.0, 85.0, 1.0)
            prev_score = st.slider("📝 Previous Exam Score", 30.0, 100.0, 70.0, 1.0)
            assignments = st.slider("📋 Assignments Completed", 0, 20, 12, 1)
            sleep_hours = st.slider("😴 Sleep Hours / Night", 4.0, 10.0, 7.0, 0.5)

            col_a, col_b = st.columns(2)
            with col_a:
                parental_edu = st.selectbox(
                    "🎓 Parental Education",
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {
                        0: "0 — None",
                        1: "1 — High School",
                        2: "2 — Associate",
                        3: "3 — Bachelor's",
                        4: "4 — Postgraduate",
                    }[x],
                    index=2,
                )
            with col_b:
                tutoring = st.number_input("🧑‍🏫 Tutoring Sessions/Month", 0, 10, 3, 1)

            stress = st.slider("😰 Stress Level (1=low, 10=high)", 1.0, 10.0, 4.0, 0.5)

            submitted = st.form_submit_button("🔮 Predict Score", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Prediction Result")

        if submitted:
            student_data = {
                "study_hours_per_week": study_hours,
                "attendance_percentage": attendance,
                "previous_exam_score": prev_score,
                "assignments_completed": float(assignments),
                "sleep_hours": sleep_hours,
                "parental_education": float(parental_edu),
                "tutoring_sessions": float(tutoring),
                "stress_level": stress,
            }

            pipeline = available[selected_model]
            from src.predictor import predict_score, score_to_percentile_desc
            prediction = predict_score(pipeline, student_data)
            grade_text, grade_class = score_to_grade(prediction)
            percentile_desc = score_to_percentile_desc(prediction)

            # Score bar
            st.markdown(
                f"""
                <div class='pred-box'>
                    <h3>Predicted Final Score</h3>
                    <h1>{prediction:.1f} / 100</h1>
                    <p style='font-size:1.1rem;margin-top:8px'>{grade_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(int(prediction), text=f"{prediction:.1f}%")

            st.info(f"💡 {percentile_desc}")

            st.markdown("#### 📌 Input Summary")
            input_df = pd.DataFrame(
                {"Feature": list(student_data.keys()), "Value": list(student_data.values())}
            )
            st.dataframe(input_df, hide_index=True, use_container_width=True)

            # Comparison across all models
            st.markdown("#### 🔄 All Models Comparison")
            model_preds = []
            for mname, mpipe in available.items():
                p = predict_score(mpipe, student_data)
                model_preds.append({"Model": mname, "Predicted Score": round(p, 2)})
            st.dataframe(
                pd.DataFrame(model_preds).set_index("Model"),
                use_container_width=True,
            )

        else:
            st.markdown(
                """
                <div class='info-banner'>
                    👈 Fill in the student attributes on the left and click
                    <b>Predict Score</b> to see the result.
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Show example from outputs
            ex_path = OUTPUTS_DIR / "example_prediction.json"
            if ex_path.exists():
                ex = json.loads(ex_path.read_text())
                st.markdown("**Example prediction (from training run):**")
                st.json(ex)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "ℹ️ About":
    st.markdown("<h1 class='section-header'>ℹ️ About this Project</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        ## 🎓 Student Score Predictor

        This is a **portfolio machine learning project** built to demonstrate a complete, production-style
        regression workflow using Python and scikit-learn, presented through an interactive Streamlit dashboard.

        ---

        ### 🧪 Dataset

        The dataset is **fully synthetic** — generated programmatically using NumPy with no real student data.
        It contains **500 student records** (with intentional missing values and duplicates for preprocessing
        practice) and the following features:

        | Feature | Type | Description |
        |---|---|---|
        | `study_hours_per_week` | Numeric | Average hours studied per week |
        | `attendance_percentage` | Numeric | % of classes attended |
        | `previous_exam_score` | Numeric | Score on most recent exam |
        | `assignments_completed` | Numeric | Assignments submitted (0–20) |
        | `sleep_hours` | Numeric | Nightly sleep hours |
        | `parental_education` | Ordinal | Education level: 0 (none) to 4 (postgrad) |
        | `tutoring_sessions` | Numeric | Extra tutoring sessions per month |
        | `stress_level` | Numeric | Self-reported stress (1=low, 10=high) |

        The target (`final_score`) was synthesised as a weighted linear combination of these features
        plus Gaussian noise, then normalised to [0, 100].

        ---

        ### ⚙️ Methodology

        1. **Load & Validate** — Structural checks; reject invalid datasets.
        2. **Clean** — Remove duplicates; drop rows with missing target.
        3. **Split** — 80 / 20 train / test split (`random_state=42`).
        4. **Preprocess** — Median imputation (fitted on train only) + standard scaling, inside a sklearn `Pipeline`.
        5. **Train** — Four models: Linear Regression, Ridge, Random Forest, Gradient Boosting.
        6. **Evaluate** — MAE, MSE, RMSE, and R² on the held-out test set.
        7. **Select** — Best model chosen by R².
        8. **Save** — Fitted pipelines saved with joblib; metrics / predictions saved as JSON & CSV.

        ---

        ### ⚠️ Limitations

        - The dataset is **synthetic and small** (~500 rows). Metrics will not generalise to real students.
        - No cross-validation or hyperparameter tuning was performed beyond default settings.
        - The target variable's relationship to features is artificial (linear by construction).
        - **This tool should not be used for real academic decisions** (grading, admission, ranking).

        ---

        ### 🛠 Technology Stack

        | Tool | Purpose |
        |---|---|
        | Python 3.10+ | Core language |
        | pandas / NumPy | Data manipulation |
        | scikit-learn | ML models and pipelines |
        | Matplotlib / Seaborn | Static visualisations |
        | Streamlit | Interactive dashboard |
        | joblib | Model serialisation |
        | pytest | Automated testing |

        ---

        ### 📁 Project Structure

        ```
        score-predictor/
        ├── app.py                  ← Streamlit dashboard (this file)
        ├── main.py                 ← Training & evaluation pipeline
        ├── requirements.txt
        ├── .gitignore
        ├── README.md
        ├── data/
        │   ├── generate_dataset.py ← Synthetic data generator
        │   └── student_scores.csv  ← Generated dataset
        ├── src/
        │   ├── preprocessor.py     ← Load, validate, clean, split
        │   ├── trainer.py          ← Model definitions and training
        │   ├── evaluator.py        ← Metrics and plots
        │   ├── predictor.py        ← Single-student prediction
        │   └── eda.py              ← EDA visualisations
        ├── models/                 ← Saved model .pkl files
        ├── outputs/                ← Metrics, predictions, charts
        └── tests/                  ← pytest test suite
        ```
        """
    )
