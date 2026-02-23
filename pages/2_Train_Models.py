"""Model training and evaluation page."""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_from_upload, validate_dataset, get_summary_stats
from src.preprocessor import build_pipeline
from src.trainer import train_all, load_trained_models
from src.explainer import feature_importance_figure
from src.plots import roc_curves, confusion_matrix_fig, model_comparison_bar


def _models_ready() -> bool:
    return os.path.exists("models/best_model.pkl")


@st.cache_resource
def _get_trained_artifacts():
    try:
        all_models, results, best_name, best_model = load_trained_models()
        from src.preprocessor import load_artifacts
        scaler, feature_cols = load_artifacts()
        return all_models, results, best_name, best_model, scaler, feature_cols
    except Exception:
        return None, None, None, None, None, None


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">📊</span>
        <h1 style="margin:0;">Model Training & Evaluation</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Upload the dataset, configure training options, and compare model performance.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Step 1: Dataset ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span>📁  Step 1 — Dataset</span></div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload the UCI Heart Disease CSV",
    type=["csv"],
    help="Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset",
)

df = None

if uploaded:
    df = load_from_upload(uploaded)
    report = validate_dataset(df)
    if not report["ok"]:
        st.error(f"❌  Dataset issues: {report['issues']}")
        st.stop()
    for issue in report["issues"]:
        st.warning(f"⚠️  {issue}")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/heart.csv", index=False)
    st.success(f"✅  Loaded {df.shape[0]} patients, {df.shape[1]} columns.")

elif os.path.exists("data/heart.csv"):
    df = pd.read_csv("data/heart.csv")
    st.info(f"ℹ️  Using saved dataset — {df.shape[0]} patients.")

if df is not None:
    c1, c2, c3 = st.columns(3)
    stats = get_summary_stats(df)
    c1.metric("👥  Patients",      stats["n_patients"])
    c2.metric("❤️  Disease Cases", stats["class_distribution"].get(1, 0))
    c3.metric("📈  Disease %",     f"{stats['disease_pct']}%")

st.markdown("---")

# ── Step 2: Training options ──────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span>⚙️  Step 2 — Training Options</span></div>',
    unsafe_allow_html=True,
)

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    use_smote = st.toggle("⚖️  Apply SMOTE (balance classes)", value=True)
with col_opt2:
    calibrate = st.toggle("🎯  Calibrate probabilities", value=True)
with col_opt3:
    cv_folds = st.slider("🔁  Cross-validation folds", 3, 10, 5)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀  Train All Models", type="primary", disabled=(df is None)):
    progress = st.progress(0)
    status = st.empty()

    def on_progress(model_name, pct):
        progress.progress(pct)
        status.markdown(f"⏳  Training **{model_name}**...")

    with st.spinner("Preprocessing data..."):
        X_tr, X_te, y_tr, y_te, scaler, feature_cols = build_pipeline(
            df, apply_smote=use_smote
        )

    trained, results, best_name = train_all(
        X_tr, y_tr, X_te, y_te,
        calibrate=calibrate,
        cv_folds=cv_folds,
        progress_fn=on_progress,
    )

    progress.progress(100)
    status.empty()
    st.cache_resource.clear()
    st.success(f"✅  Training complete! Best model: **{best_name}**")

    np.save("models/X_test.npy", X_te)
    np.save("models/X_train.npy", X_tr)

# ── Results ───────────────────────────────────────────────────────────────────
if _models_ready():
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>📋  Results</span></div>',
        unsafe_allow_html=True,
    )

    all_models, results, best_name, _, _, _ = _get_trained_artifacts()

    if results:
        rows = []
        for name, m in results.items():
            rows.append({
                "Model":    name,
                "AUC":      m["roc_auc"],
                "Accuracy": m["accuracy"],
                "F1":       m["f1"],
                "CV AUC":   f"{m.get('cv_auc_mean', 0):.4f} ± {m.get('cv_auc_std', 0):.4f}",
                "Best":     "★" if name == best_name else "",
            })

        tbl = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        st.dataframe(
            tbl.style.highlight_max(subset=["AUC", "Accuracy", "F1"], color="#C6F6D5"),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📈  ROC Curves", "🔎  Per-Model Detail", "⚖️  Model Comparison"])

        with tab1:
            st.pyplot(roc_curves(results))

        with tab2:
            sel_name = st.selectbox("Select model", list(results.keys()))
            sel_res = results[sel_name]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯  AUC",               sel_res["roc_auc"])
            c2.metric("✅  Accuracy",           sel_res["accuracy"])
            c3.metric("📊  F1 Score",           sel_res["f1"])
            c4.metric("⚖️  Optimal Threshold",  f"{sel_res.get('optimal_threshold', 0.5):.3f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.pyplot(confusion_matrix_fig(sel_res["confusion_matrix"], sel_name))
            with col_b:
                if all_models and sel_name in all_models:
                    fig_imp = feature_importance_figure(
                        all_models[sel_name],
                        pickle.load(open("models/feature_cols.pkl", "rb")),
                        sel_name,
                    )
                    if fig_imp:
                        st.pyplot(fig_imp)

            if "cv_scores" in sel_res:
                st.info(
                    f"🔁  **Cross-validation AUC**: "
                    f"`{sel_res['cv_auc_mean']:.4f}` ± `{sel_res['cv_auc_std']:.4f}` "
                    f"across {len(sel_res['cv_scores'])} folds"
                )

        with tab3:
            metric_choice = st.selectbox("Metric", ["roc_auc", "accuracy", "f1"])
            st.pyplot(model_comparison_bar(results, metric=metric_choice))
