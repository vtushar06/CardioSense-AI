import os
import streamlit as st

from src.explainer import get_rule_based_flags
from src.llm_agent import run_agent, get_api_key
from src.pdf_export import generate_pdf_report


def _models_ready():
    return os.path.exists("models/best_model.pkl")


st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">🤖</span>
        <h1 style="margin:0;">AI Health Assistant</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Ask health questions about your risk results. The AI generates a structured,
        evidence-grounded health report — not a diagnosis.
    </p>
    """,
    unsafe_allow_html=True,
)

if not _models_ready():
    st.warning("⚠️  Train models first — go to the **Train Models** page.")
    st.stop()

st.markdown("---")

# ── Step 1: Get patient data ──────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span>🧑‍⚕️  Step 1 — Patient Data</span></div>',
    unsafe_allow_html=True,
)

has_session = (
    "last_patient" in st.session_state
    and "last_prediction" in st.session_state
    and "last_flags" in st.session_state
)

if has_session:
    st.success("✅  Patient data loaded from your last prediction.")
    patient    = st.session_state["last_patient"]
    prediction = st.session_state["last_prediction"]
    flags      = st.session_state["last_flags"]

    with st.expander("📋  View loaded patient data", expanded=False):
        import pandas as pd
        sex_label = "Male" if patient.get("sex") == 1 else "Female"
        summary = {
            "Age": f"{patient.get('age')} years",
            "Sex": sex_label,
            "Blood Pressure": f"{patient.get('trestbps')} mmHg",
            "Cholesterol": f"{patient.get('chol')} mg/dl",
            "Max Heart Rate": f"{patient.get('thalach')} bpm",
            "ST Depression": patient.get("oldpeak"),
            "Risk Level": prediction.get("risk_label"),
            "Risk Probability": f"{round(prediction.get('probability', 0) * 100, 1)}%",
        }
        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))
else:
    st.info(
        "ℹ️  No prediction found in this session. "
        "You can run a prediction on the **Patient Risk Prediction** page first, "
        "or enter patient values manually below."
    )

    with st.expander("✏️  Enter patient values manually", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.slider("Age",          20, 90, 55, key="m_age")
            sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="m_sex")
            trestbps = st.slider("Blood Pressure (mmHg)", 80, 200, 125, key="m_bp")
            chol     = st.slider("Cholesterol (mg/dl)",   100, 600, 230, key="m_chol")
        with c2:
            thalach  = st.slider("Max Heart Rate",    60, 220, 150, key="m_hr")
            oldpeak  = st.slider("ST Depression",     0.0, 6.0, 1.0, key="m_op")
            restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                                    format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
                                    key="m_ecg")
            slope    = st.selectbox("ST Slope", [0, 1, 2],
                                    format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
                                    key="m_slope")
        with c3:
            cp    = st.selectbox("Chest Pain", [0, 1, 2, 3],
                                 format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                                        2: "Non-Anginal", 3: "Asymptomatic"}[x],
                                 key="m_cp")
            exang = st.selectbox("Exercise Angina", [0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No", key="m_ex")
            fbs   = st.selectbox("Fasting Blood Sugar >120", [0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No", key="m_fbs")
            ca    = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3], key="m_ca")
            thal  = st.selectbox("Thalassemia", [0, 1, 2, 3],
                                 format_func=lambda x: {0: "Normal", 1: "Fixed Defect",
                                                        2: "Reversable Defect", 3: "Unknown"}[x],
                                 key="m_thal")

        patient = dict(
            age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
            fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
            oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
        )

        from src.preprocessor import load_artifacts
        from src.trainer import load_trained_models
        from src.predictor import predict_single

        try:
            all_models, results, best_name, best_model = load_trained_models()
            scaler, feature_cols = load_artifacts()
            prediction = predict_single(patient, best_model, scaler, feature_cols)
        except Exception:
            prediction = {"risk_label": "Unknown", "probability": 0.5, "threshold": 0.5}

        flags = get_rule_based_flags(patient)

st.markdown("---")

# ── Step 2: User query ────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span>💬  Step 2 — Ask a Health Question</span></div>',
    unsafe_allow_html=True,
)

suggested = [
    "What does my risk score mean and what should I do?",
    "What lifestyle changes can help reduce my cardiovascular risk?",
    "Explain my cholesterol and blood pressure readings.",
    "When should I see a doctor based on these results?",
]

col_q, col_s = st.columns([3, 1])
with col_s:
    suggestion_pick = st.selectbox("💡 Quick questions", ["Custom..."] + suggested, key="q_pick")

with col_q:
    if suggestion_pick == "Custom...":
        user_query = st.text_area(
            "Your question",
            value="What does my risk score mean and what should I do?",
            height=80,
            key="user_q",
        )
    else:
        user_query = st.text_area(
            "Your question",
            value=suggestion_pick,
            height=80,
            key="user_q2",
        )

if not get_api_key():
    st.warning(
        "⚠️  No Groq API key found — the AI will use a structured template instead of live LLM generation. "
        "Add your key to the `.env` file as `GROQ_API_KEY=your_key` and restart the app."
    )

st.markdown("---")

# ── Step 3: Generate report ───────────────────────────────────────────────────
if st.button("🚀  Generate Health Report", type="primary"):
    steps_placeholder = st.empty()

    with st.spinner("Agent is thinking..."):
        steps_placeholder.markdown(
            """
            <div style="background:#EBF4FF; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <strong>🔄 Agent Steps</strong><br>
                1️⃣  Collecting patient data and prediction results...<br>
                2️⃣  Calling AI model to generate structured report...<br>
                3️⃣  Parsing and formatting response...
            </div>
            """,
            unsafe_allow_html=True,
        )

        state = run_agent(patient, prediction, flags, user_query)

    steps_placeholder.empty()

    if state.get("error") and not state.get("used_llm"):
        st.warning(f"⚠️  AI call failed ({state['error']}). Showing template report instead.")

    sections = state["sections"]

    if state.get("used_llm"):
        st.success("✅  Report generated by Groq LLaMA-3 (llama3-8b-8192)")
    else:
        st.info("ℹ️  Report generated using structured template (no LLM call)")

    st.markdown("---")

    # ── Risk Summary ──────────────────────────────────────────────────────────
    prob = prediction.get("probability", 0)
    if prob >= 0.66:
        banner_cls = "risk-high"
        label_icon = "🔴  HIGH RISK"
    elif prob >= 0.33:
        banner_cls = "risk-medium"
        label_icon = "🟡  MEDIUM RISK"
    else:
        banner_cls = "risk-low"
        label_icon = "🟢  LOW RISK"

    st.markdown(
        f'<div class="{banner_cls}"><h2>{label_icon}</h2>'
        f'<p>Estimated probability: <strong>{round(prob * 100, 1)}%</strong></p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header"><span>📋  Patient Risk Summary</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(sections.get("risk_summary", ""), unsafe_allow_html=False)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            '<div class="section-header"><span>⚠️  Key Contributing Factors</span></div>',
            unsafe_allow_html=True,
        )
        raw = sections.get("contributing", "")
        for line in raw.split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ")
            if not line:
                continue
            if "danger" in line.lower() or "🔴" in line:
                st.markdown(f'<div class="flag-danger">🔴 {line}</div>', unsafe_allow_html=True)
            elif "warning" in line.lower() or "🟡" in line:
                st.markdown(f'<div class="flag-warning">🟡 {line}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-ok">• {line}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="section-header"><span>✅  Preventive Recommendations</span></div>',
            unsafe_allow_html=True,
        )
        recs = sections.get("recommendations", "")
        for line in recs.split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ")
            if line:
                st.markdown(f'<div class="flag-ok">✔ {line}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>🏥  When to Seek Medical Attention</span></div>',
        unsafe_allow_html=True,
    )
    seek = sections.get("seek_attention", "")
    for line in seek.split("\n"):
        line = line.strip().lstrip("- ").lstrip("* ")
        if line:
            st.markdown(f'<div class="flag-warning">→ {line}</div>', unsafe_allow_html=True)

    st.markdown("---")

    refs = sections.get("references", "")
    if refs:
        st.markdown(
            '<div class="section-header"><span>📚  References</span></div>',
            unsafe_allow_html=True,
        )
        for line in refs.split("\n"):
            line = line.strip().lstrip("- ").lstrip("* ")
            if line:
                st.markdown(f"- {line}")

    st.markdown("---")

    disclaimer = sections.get("disclaimer", "")
    if disclaimer:
        st.warning(f"⚕️ **Medical Disclaimer** — {disclaimer}")

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>📄  Export Report</span></div>',
        unsafe_allow_html=True,
    )

    try:
        pdf_bytes = generate_pdf_report(patient, prediction, sections)
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name="cardiosense_health_report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

    st.session_state["last_report_sections"] = sections
    st.session_state["last_report_patient"]   = patient
    st.session_state["last_report_prediction"] = prediction
