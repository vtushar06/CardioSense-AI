"""Single patient risk prediction page."""

import os
import numpy as np
import streamlit as st

from src.preprocessor import load_artifacts, preprocess_single
from src.trainer import load_trained_models
from src.predictor import predict_single
from src.explainer import (
    make_explainer,
    shap_single_patient_figure,
    feature_importance_figure,
    risk_gauge_figure,
    get_rule_based_flags,
)


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


@st.cache_resource
def _get_explainer(_model, _X_train, model_name):
    return make_explainer(_model, _X_train, model_name)


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">🔬</span>
        <h1 style="margin:0;">Patient Risk Prediction</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Enter clinical values to estimate cardiovascular disease risk with SHAP-based explanation.
    </p>
    """,
    unsafe_allow_html=True,
)

if not _models_ready():
    st.warning("⚠️  Train models first — go to the **Train Models** page.")
    st.stop()

all_models, results, best_name, best_model, scaler, feature_cols = _get_trained_artifacts()

# ── Model selector ────────────────────────────────────────────────────────────
st.markdown("---")
col_ms, col_th = st.columns([2, 1])
with col_ms:
    model_choice = st.selectbox(
        "🤖  Model",
        list(all_models.keys()),
        index=list(all_models.keys()).index(best_name),
    )
with col_th:
    use_optimal = st.toggle("🎯  Use optimal threshold (Youden's J)", value=False)

chosen_model = all_models[model_choice]

# ── Patient input form ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="section-header"><span>🧑‍⚕️  Patient Clinical Data</span></div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**👤  Demographics & Vitals**")
    age      = st.slider("Age (years)", 20, 90, 55)
    sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 125)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 230)

with c2:
    st.markdown("**❤️  Heart Function**")
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    restecg  = st.selectbox(
        "Resting ECG",
        [0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
    )
    slope    = st.selectbox(
        "Slope of ST Segment",
        [0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
    )

with c3:
    st.markdown("**🩺  Symptoms & Tests**")
    cp    = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina", 1: "Atypical Angina",
            2: "Non-Anginal",    3: "Asymptomatic",
        }[x],
    )
    exang = st.selectbox("Exercise-Induced Angina", [0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No")
    fbs   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No")
    ca    = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal  = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal", 1: "Fixed Defect",
            2: "Reversable Defect", 3: "Unknown",
        }[x],
    )

patient = dict(
    age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
    fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
    oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
)

st.markdown("---")

if st.button("🚀  Run Prediction", type="primary"):
    pred = predict_single(
        patient, chosen_model, scaler, feature_cols,
        use_optimal_threshold=use_optimal,
        results=results, model_name=model_choice,
    )
    prob = pred["probability"]

    # ── Risk banner ──────────────────────────────────────────────────────────
    if prob >= 0.66:
        st.markdown(
            f'<div class="risk-high">'
            f'<h2>🔴  HIGH RISK</h2>'
            f'<p>Model estimates a <strong>{prob*100:.1f}%</strong> probability of cardiovascular disease.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif prob >= 0.33:
        st.markdown(
            f'<div class="risk-medium">'
            f'<h2>🟡  MEDIUM RISK</h2>'
            f'<p>Model estimates a <strong>{prob*100:.1f}%</strong> probability of cardiovascular disease.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-low">'
            f'<h2>🟢  LOW RISK</h2>'
            f'<p>Model estimates a <strong>{prob*100:.1f}%</strong> probability of cardiovascular disease.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_gauge, col_flags = st.columns([1, 2])

    with col_gauge:
        st.pyplot(risk_gauge_figure(prob))
        st.markdown(
            f"<div style='text-align:center;color:#4A5568;font-size:0.85rem'>"
            f"Model: <strong>{model_choice}</strong><br>"
            f"Threshold: <strong>{pred['threshold']:.3f}</strong></div>",
            unsafe_allow_html=True,
        )

    with col_flags:
        st.markdown(
            '<div class="section-header"><span>🚩  Clinical Flags</span></div>',
            unsafe_allow_html=True,
        )
        flags = get_rule_based_flags(patient)
        icon_map = {"danger": "🔴", "warning": "🟡", "ok": "🟢"}
        for level, msg in flags:
            st.markdown(
                f'<div class="flag-{level}">{icon_map.get(level, "")}  {msg}</div>',
                unsafe_allow_html=True,
            )

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>🧠  Why This Prediction? (SHAP)</span></div>',
        unsafe_allow_html=True,
    )

    if os.path.exists("models/X_train.npy"):
        X_train_np = np.load("models/X_train.npy")
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = _get_explainer(chosen_model, X_train_np, model_choice)
                patient_X = preprocess_single(patient, scaler, feature_cols)
                st.pyplot(shap_single_patient_figure(explainer, patient_X, feature_cols))
            except Exception as e:
                st.info(f"ℹ️  SHAP not available for this model type: {e}")
                fig_imp = feature_importance_figure(chosen_model, feature_cols, model_choice)
                if fig_imp:
                    st.pyplot(fig_imp)
    else:
        st.info("ℹ️  Retrain the model to enable SHAP explanations.")

    # ── Summary report ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>📄  Patient Summary Report</span></div>',
        unsafe_allow_html=True,
    )

    with st.expander("View full report", expanded=True):
        if prob >= 0.66:
            action = "Consider urgent cardiology referral."
        elif prob >= 0.33:
            action = "Schedule a routine cardiac review."
        else:
            action = "Maintain current lifestyle and routine monitoring."

        st.markdown(f"""
**Patient Snapshot**

| Field | Value |
|---|---|
| Age | {age} years |
| Sex | {"Male" if sex == 1 else "Female"} |
| Blood Pressure | {trestbps} mmHg |
| Cholesterol | {chol} mg/dl |
| Max Heart Rate | {thalach} bpm |

**Risk Assessment**

- Prediction: **{pred['risk_label']}** ({prob*100:.1f}% estimated probability)
- Model: {model_choice} · Threshold: {pred['threshold']:.3f}

**Suggested Action** *(educational only)*

- {action}
- Monitor BP and cholesterol at least annually.
- Report any chest pain or shortness of breath to a clinician immediately.

---
*This output is for educational purposes and does not constitute medical advice.*
""")
