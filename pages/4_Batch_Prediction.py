"""Batch patient prediction page."""

import os
import pandas as pd
import streamlit as st

from src.data_loader import load_from_upload
from src.preprocessor import load_artifacts
from src.trainer import load_trained_models
from src.predictor import predict_batch
from src.plots import risk_score_histogram, radar_comparison


def _models_ready() -> bool:
    return os.path.exists("models/best_model.pkl")


@st.cache_resource
def _get_trained_artifacts():
    try:
        all_models, results, best_name, best_model = load_trained_models()
        scaler, feature_cols = load_artifacts()
        return all_models, results, best_name, best_model, scaler, feature_cols
    except Exception:
        return None, None, None, None, None, None


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">📂</span>
        <h1 style="margin:0;">Batch Prediction</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Upload a CSV containing multiple patients and score them all at once.
    </p>
    """,
    unsafe_allow_html=True,
)

if not _models_ready():
    st.warning("⚠️  Train models first — go to the **Train Models** page.")
    st.stop()

all_models, results, best_name, best_model, scaler, feature_cols = _get_trained_artifacts()

# ── Format guide ─────────────────────────────────────────────────────────────
with st.expander("📋  Expected CSV format", expanded=False):
    sample = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }])
    st.dataframe(sample, use_container_width=True)
    st.caption("All 13 feature columns required. Column names are case-insensitive.")

st.markdown("---")

uploaded = st.file_uploader("📁  Upload patient CSV", type=["csv"])

if not uploaded:
    st.markdown(
        """
        <div style="
            background:#F7FAFC; border:1px solid #E2E8F0; border-radius:12px;
            padding:1.6rem 2rem; text-align:center; color:#718096; margin-top:0.5rem;
        ">
            <div style="font-size:2rem; margin-bottom:0.4rem;">📂</div>
            <div style="font-weight:600; color:#4A5568; margin-bottom:0.2rem;">No file uploaded yet</div>
            <div style="font-size:0.88rem;">Upload a CSV above to score multiple patients at once.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded:
    df = load_from_upload(uploaded)
    st.info(f"ℹ️  Loaded **{len(df)}** patients")
    st.dataframe(df.head(5), use_container_width=True)

    model_choice = st.selectbox(
        "🤖  Model",
        list(all_models.keys()),
        index=list(all_models.keys()).index(best_name),
    )

    if st.button("🚀  Predict All", type="primary"):
        with st.spinner("Running predictions..."):
            preds = predict_batch(df, all_models[model_choice], scaler, feature_cols)

        pred_df = pd.DataFrame(preds)
        result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

        high = (pred_df["risk_label"] == "High Risk").sum()
        low  = (pred_df["risk_label"] == "Low Risk").sum()

        st.markdown("---")
        st.markdown(
            '<div class="section-header"><span>📊  Summary</span></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴  High Risk",  high)
        c2.metric("🟢  Low Risk",   low)
        c3.metric("📈  High Risk %", f"{high / len(pred_df) * 100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(risk_score_histogram(pred_df["probability"].tolist()))

        st.markdown("---")
        st.markdown(
            '<div class="section-header"><span>📋  All Predictions</span></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            result.style.apply(
                lambda row: [
                    "background-color: #FFF5F5"
                    if row["risk_label"] == "High Risk"
                    else "background-color: #F0FFF4"
                ] * len(row),
                axis=1,
            ),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "⬇️  Download Results CSV",
                data=result.to_csv(index=False),
                file_name="cardiosense_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if 2 <= len(df) <= 4:
            st.markdown("---")
            st.markdown(
                '<div class="section-header"><span>🕸️  Patient Profile Comparison</span></div>',
                unsafe_allow_html=True,
            )
            st.pyplot(radar_comparison(df.to_dict("records"), preds))
