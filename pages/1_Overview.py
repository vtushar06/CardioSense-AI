"""Overview / home page."""

import streamlit as st

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.25rem;">
        <span style="font-size:2rem;">🫀</span>
        <h1 style="margin:0; font-size:2rem;">CardioSense AI</h1>
    </div>
    <p style="color:#4A5568; font-size:1.05rem; margin-bottom:1.5rem;">
        Cardiovascular Risk Assessment &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Clinical Explainability
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖  ML Models",       "6 + DNN",     "Trained & compared")
c2.metric("📐  Features",         "13 + 4",      "Clinical + engineered")
c3.metric("🔍  Explainability",   "SHAP + Rules","Per-patient breakdown")
c4.metric("🗃️  Dataset",          "UCI Heart",   "303 clinical records")

st.markdown("---")

st.markdown(
    """
    <div class="section-header">
        <span>⚙️  How it works</span>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 1</div>
            <h4>🗂️  Train</h4>
            <p>Upload the UCI heart disease dataset. The pipeline preprocesses it,
            handles class imbalance with SMOTE, and trains 6 ML models with
            5-fold cross-validation and optional probability calibration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 2</div>
            <h4>🔬  Predict</h4>
            <p>Enter a patient's clinical values. The best model outputs a risk
            probability alongside a SHAP waterfall chart showing which features
            drove the score up or down.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 3</div>
            <h4>📋  Explain</h4>
            <p>Rule-based clinical flags independently check values against
            standard medical thresholds, giving a second layer of explanation
            alongside the model's SHAP output.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    """
    <div class="section-header">
        <span>📦  Tech Stack</span>
    </div>
    """,
    unsafe_allow_html=True,
)

t1, t2, t3, t4 = st.columns(4)
t1.info("**Models**\nscikit-learn · XGBoost · LightGBM · PyTorch")
t2.info("**Explainability**\nSHAP (Tree + Kernel)\nRule-based flags")
t3.info("**Preprocessing**\nSMOTE · StandardScaler\nFeature engineering")
t4.info("**UI**\nStreamlit · Matplotlib\nSeaborn · Plotly")

st.markdown("---")
st.warning(
    "⚕️  **Medical Disclaimer** — This system is built for educational and research purposes only. "
    "It has not been validated for clinical use and must not be used to make medical decisions. "
    "Always consult a qualified healthcare professional."
)
