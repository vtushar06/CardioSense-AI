"""Single patient risk prediction page — manual entry or CSV upload."""

import io
import os
import numpy as np
import pandas as pd
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

REQUIRED_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]


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


def _sample_csv() -> str:
    row = {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 220,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
    }
    return pd.DataFrame([row]).to_csv(index=False)


def _run_prediction_ui(patient, chosen_model, scaler, feature_cols,
                        results, model_choice, use_optimal):
    pred = predict_single(
        patient, chosen_model, scaler, feature_cols,
        use_optimal_threshold=use_optimal,
        results=results, model_name=model_choice,
    )
    prob = pred["probability"]
    flags = get_rule_based_flags(patient)

    st.session_state["last_patient"]    = patient
    st.session_state["last_prediction"] = pred
    st.session_state["last_flags"]      = flags

    # Risk banner
    if prob >= 0.66:
        st.markdown(
            f'<div class="risk-high"><h2>🔴  HIGH RISK</h2>'
            f'<p>Model estimates a <strong>{prob*100:.1f}%</strong> probability of cardiovascular disease.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif prob >= 0.33:
        st.markdown(
            f'<div class="risk-medium"><h2>🟡  MEDIUM RISK</h2>'
            f'<p>Model estimates a <strong>{prob*100:.1f}%</strong> probability of cardiovascular disease.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-low"><h2>🟢  LOW RISK</h2>'
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
        icon_map = {"danger": "🔴", "warning": "🟡", "ok": "🟢"}
        for level, msg in flags:
            st.markdown(
                f'<div class="flag-{level}">{icon_map.get(level, "")}  {msg}</div>',
                unsafe_allow_html=True,
            )

    # SHAP
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

    # Summary
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

        sex_disp = "Male" if patient.get("sex") == 1 else "Female"
        st.markdown(f"""
**Patient Snapshot**

| Field | Value |
|---|---|
| Age | {patient.get('age')} years |
| Sex | {sex_disp} |
| Blood Pressure | {patient.get('trestbps')} mmHg |
| Cholesterol | {patient.get('chol')} mg/dl |
| Max Heart Rate | {patient.get('thalach')} bpm |

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

    st.info("💡  Go to **AI Health Assistant** for a detailed LLM-generated health report based on this result.")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">🔬</span>
        <h1 style="margin:0;">Patient Risk Prediction</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Enter your clinical values manually or upload a CSV file — get your cardiovascular risk score with SHAP explanation.
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

st.markdown("---")

# ── Two input modes ───────────────────────────────────────────────────────────
tab_manual, tab_upload = st.tabs(["✏️  Manual Entry", "📤  Upload My Data"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Entry (existing form, unchanged)
# ════════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown(
        '<div class="section-header"><span>🧑‍⚕️  Patient Clinical Data</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👤  Demographics & Vitals**")
        age      = st.slider("Age (years)", 20, 90, 55, key="m_age")
        sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="m_sex")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 125, key="m_bp")
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 230, key="m_chol")

    with c2:
        st.markdown("**❤️  Heart Function**")
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150, key="m_hr")
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1, key="m_op")
        restecg  = st.selectbox(
            "Resting ECG", [0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
            key="m_ecg",
        )
        slope    = st.selectbox(
            "Slope of ST Segment", [0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
            key="m_slope",
        )

    with c3:
        st.markdown("**🩺  Symptoms & Tests**")
        cp    = st.selectbox(
            "Chest Pain Type", [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina", 1: "Atypical Angina",
                2: "Non-Anginal",    3: "Asymptomatic",
            }[x],
            key="m_cp",
        )
        exang = st.selectbox("Exercise-Induced Angina", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No", key="m_ex")
        fbs   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No", key="m_fbs")
        ca    = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3], key="m_ca")
        thal  = st.selectbox(
            "Thalassemia", [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal", 1: "Fixed Defect",
                2: "Reversable Defect", 3: "Unknown",
            }[x],
            key="m_thal",
        )

    patient_manual = dict(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
    )

    st.markdown("---")
    if st.button("🚀  Run Prediction", type="primary", key="btn_manual"):
        _run_prediction_ui(
            patient_manual, chosen_model, scaler,
            feature_cols, results, model_choice, use_optimal,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload My Data
# ════════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown(
        '<div class="section-header"><span>📤  Upload Your Personal Health Data</span></div>',
        unsafe_allow_html=True,
    )

    # Download sample
    col_info, col_dl = st.columns([3, 1])
    with col_info:
        st.markdown(
            "Upload a **CSV file with one row** containing your clinical values. "
            "Download the sample below to see the exact format required."
        )
    with col_dl:
        st.download_button(
            label="⬇️  Download sample CSV",
            data=_sample_csv(),
            file_name="my_health_data_sample.csv",
            mime="text/csv",
        )

    # Column reference
    with st.expander("📋  Column reference", expanded=False):
        ref = pd.DataFrame([
            {"Column": "age",      "Type": "integer",  "Example": "55",  "Description": "Age in years"},
            {"Column": "sex",      "Type": "0 or 1",   "Example": "1",   "Description": "1 = Male, 0 = Female"},
            {"Column": "cp",       "Type": "0–3",      "Example": "0",   "Description": "Chest pain: 0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic"},
            {"Column": "trestbps", "Type": "integer",  "Example": "130", "Description": "Resting blood pressure (mmHg)"},
            {"Column": "chol",     "Type": "integer",  "Example": "220", "Description": "Serum cholesterol (mg/dl)"},
            {"Column": "fbs",      "Type": "0 or 1",   "Example": "0",   "Description": "Fasting blood sugar >120 mg/dl: 1=Yes, 0=No"},
            {"Column": "restecg",  "Type": "0–2",      "Example": "0",   "Description": "Resting ECG: 0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy"},
            {"Column": "thalach",  "Type": "integer",  "Example": "150", "Description": "Maximum heart rate achieved (bpm)"},
            {"Column": "exang",    "Type": "0 or 1",   "Example": "0",   "Description": "Exercise-induced angina: 1=Yes, 0=No"},
            {"Column": "oldpeak",  "Type": "decimal",  "Example": "1.5", "Description": "ST depression during exercise"},
            {"Column": "slope",    "Type": "0–2",      "Example": "1",   "Description": "ST slope: 0=Upsloping, 1=Flat, 2=Downsloping"},
            {"Column": "ca",       "Type": "0–3",      "Example": "0",   "Description": "Number of major vessels coloured by fluoroscopy"},
            {"Column": "thal",     "Type": "0–3",      "Example": "2",   "Description": "Thalassemia: 0=Normal, 1=Fixed Defect, 2=Reversable Defect, 3=Unknown"},
        ])
        st.dataframe(ref, use_container_width=True, hide_index=True)

    uploaded_file = st.file_uploader(
        "📁  Upload your health data CSV",
        type=["csv"],
        key="user_data_upload",
    )

    if uploaded_file:
        try:
            df_user = pd.read_csv(uploaded_file)
            df_user.columns = [c.strip().lower() for c in df_user.columns]
        except Exception as e:
            st.error(f"❌  Could not read file: {e}")
            st.stop()

        # Validate columns
        missing_cols = [c for c in REQUIRED_COLS if c not in df_user.columns]
        if missing_cols:
            st.error(f"❌  Missing columns in your file: `{missing_cols}`")
            st.markdown("Download the sample CSV above to see the required format.")
            st.stop()

        if len(df_user) == 0:
            st.error("❌  The uploaded file has no data rows.")
            st.stop()

        # If multiple rows, let user pick one
        if len(df_user) > 1:
            st.info(f"ℹ️  Your file has **{len(df_user)}** rows. Select which patient record to analyse.")
            row_idx = st.selectbox(
                "Select patient row (0-indexed)",
                options=list(range(len(df_user))),
                format_func=lambda i: f"Row {i+1}",
                key="row_picker",
            )
            row = df_user.iloc[row_idx]
        else:
            row = df_user.iloc[0]

        # Build patient dict — fill missing optional fields with safe defaults
        patient_upload = {
            "age":      int(row.get("age", 55)),
            "sex":      int(row.get("sex", 1)),
            "cp":       int(row.get("cp", 0)),
            "trestbps": int(row.get("trestbps", 120)),
            "chol":     int(row.get("chol", 200)),
            "fbs":      int(row.get("fbs", 0)),
            "restecg":  int(row.get("restecg", 0)),
            "thalach":  int(row.get("thalach", 150)),
            "exang":    int(row.get("exang", 0)),
            "oldpeak":  float(row.get("oldpeak", 0.0)),
            "slope":    int(row.get("slope", 1)),
            "ca":       int(row.get("ca", 0)),
            "thal":     int(row.get("thal", 2)),
        }

        # Show what was read
        cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"}
        sex_label = "Male" if patient_upload["sex"] == 1 else "Female"
        st.success(f"✅  Loaded: Age {patient_upload['age']}, {sex_label}, "
                   f"BP {patient_upload['trestbps']} mmHg, "
                   f"Chol {patient_upload['chol']} mg/dl, "
                   f"Max HR {patient_upload['thalach']} bpm")

        with st.expander("📋  Full data preview", expanded=False):
            display = pd.DataFrame([patient_upload]).T.reset_index()
            display.columns = ["Field", "Value"]
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("🚀  Run Prediction on My Data", type="primary", key="btn_upload"):
            _run_prediction_ui(
                patient_upload, chosen_model, scaler,
                feature_cols, results, model_choice, use_optimal,
            )
