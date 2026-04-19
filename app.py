"""
CardioSense AI — Cardiovascular Risk Assessment Platform

Entry point for the Streamlit application. Page-specific logic
lives in the pages/ directory. This file only sets global config
and shared styling.

Run: streamlit run app.py
"""

# ── Thread-safety guards (must come before any ML library import) ────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import streamlit as st

st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0F2444 40%, #1A365D 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #EBF8FF !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label { color: #BEE3F8 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    [data-testid="stSidebar"] .stSuccess { background: rgba(56,161,105,0.18) !important; border-radius: 8px; border: 1px solid rgba(56,161,105,0.3) !important; }
    [data-testid="stSidebar"] .stWarning { background: rgba(214,158,46,0.18) !important; border-radius: 8px; border: 1px solid rgba(214,158,46,0.3) !important; }
    /* Sidebar nav links */
    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        padding: 6px 10px;
        transition: background 0.15s;
    }
    [data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.08) !important; }
    [data-testid="stSidebarNav"] [aria-selected="true"] { background: rgba(43,108,176,0.35) !important; }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Page header ── */
    h1 {
        color: #1A365D;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 0.25rem;
    }
    h2, h3 { color: #2D3748; font-weight: 600; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #2B6CB0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] label { color: #718096 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #1A365D !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ── Risk banners ── */
    .risk-high {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        border-left: 5px solid #E53E3E;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(229,62,62,0.12);
    }
    .risk-high h2 { color: #C53030 !important; margin: 0 0 0.4rem 0; }
    .risk-high p  { color: #742A2A; margin: 0; font-size: 1rem; }

    .risk-medium {
        background: linear-gradient(135deg, #FFFFF0 0%, #FEFCBF 100%);
        border-left: 5px solid #D69E2E;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(214,158,46,0.12);
    }
    .risk-medium h2 { color: #B7791F !important; margin: 0 0 0.4rem 0; }
    .risk-medium p  { color: #744210; margin: 0; font-size: 1rem; }

    .risk-low {
        background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
        border-left: 5px solid #38A169;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(56,161,105,0.12);
    }
    .risk-low h2 { color: #276749 !important; margin: 0 0 0.4rem 0; }
    .risk-low p  { color: #1C4532; margin: 0; font-size: 1rem; }

    /* ── Clinical flag cards ── */
    .flag-danger {
        background: #FFF5F5;
        border-left: 4px solid #E53E3E;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 5px 0;
        font-size: 0.9rem;
        color: #742A2A;
    }
    .flag-warning {
        background: #FFFFF0;
        border-left: 4px solid #D69E2E;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 5px 0;
        font-size: 0.9rem;
        color: #744210;
    }
    .flag-ok {
        background: #F0FFF4;
        border-left: 4px solid #38A169;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 5px 0;
        font-size: 0.9rem;
        color: #1C4532;
    }

    /* ── Step / feature cards ── */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .step-card:hover {
        box-shadow: 0 6px 18px rgba(43,108,176,0.1);
        transform: translateY(-2px);
    }
    .step-card .step-num {
        display: inline-block;
        background: linear-gradient(135deg, #2B6CB0, #1A365D);
        color: white;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.6rem;
        letter-spacing: 0.04em;
    }
    .step-card h4 { color: #1A365D; margin: 0.4rem 0 0.6rem 0; font-size: 1.05rem; }
    .step-card p  { color: #4A5568; font-size: 0.88rem; margin: 0; line-height: 1.6; }

    /* ── Tech stack cards ── */
    .tech-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tech-card .tc-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #2B6CB0;
        margin-bottom: 0.5rem;
    }
    .tech-card .tc-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1A365D;
        margin-bottom: 0.4rem;
    }
    .tech-card .tc-items {
        font-size: 0.85rem;
        color: #4A5568;
        line-height: 1.65;
    }

    /* ── Section header with accent line ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #EBF4FF;
    }
    .section-header span { font-size: 1.1rem; font-weight: 600; color: #2D3748; }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2B6CB0 0%, #1A365D 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.55rem 1.6rem;
        transition: opacity 0.2s, box-shadow 0.2s, transform 0.15s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.92;
        box-shadow: 0 4px 16px rgba(43,108,176,0.45);
        transform: translateY(-1px);
    }
    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border: 1px solid #CBD5E0 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stDownloadButton > button:hover {
        border-color: #2B6CB0 !important;
        box-shadow: 0 2px 8px rgba(43,108,176,0.15) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F7FAFC;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        color: #4A5568;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2B6CB0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        font-weight: 600 !important;
    }

    /* ── Alerts / info boxes ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 1px !important;
        font-size: 0.9rem;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #CBD5E0;
        border-radius: 12px;
        padding: 0.5rem;
        transition: border-color 0.2s;
        background: #FAFBFC;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2B6CB0;
        background: #EBF4FF;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-weight: 500;
        color: #2D3748;
        padding: 0.75rem 1rem;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2B6CB0, #4299E1) !important;
        border-radius: 4px;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #E2E8F0; margin: 1.8rem 0; }

    /* ── Hide Streamlit chrome ── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size:2.4rem;">🫀</div>
            <div style="font-size:1.2rem; font-weight:700; color:#EBF8FF; letter-spacing:-0.3px;">CardioSense AI</div>
            <div style="font-size:0.78rem; color:#90CDF4; margin-top:2px;">Cardiovascular Risk Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if os.path.exists("models/best_model.pkl"):
        st.success("✅  Models trained & ready")
    else:
        st.warning("⚠️  No models found.\nGo to **Train Models** first.")

    st.markdown("---")
    st.caption("⚕️ For educational use only. Not a clinical diagnostic tool.")

# ── Home page content ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #EBF4FF 0%, #F0FFF4 100%);
        border: 1px solid #BEE3F8;
        border-radius: 16px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
            <span style="font-size:2.2rem;">🫀</span>
            <h1 style="margin:0; font-size:2rem; color:#1A365D; letter-spacing:-0.5px;">CardioSense AI</h1>
        </div>
        <p style="color:#4A5568; font-size:1.05rem; margin:0 0 1.2rem 0; max-width:640px;">
            An end-to-end cardiovascular risk assessment platform powered by classical ML,
            deep learning, SHAP explainability, and an agentic AI health assistant.
        </p>
        <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
            <span style="background:#2B6CB0; color:white; font-size:0.78rem; font-weight:600; padding:4px 12px; border-radius:20px;">6 ML Models + DNN</span>
            <span style="background:#38A169; color:white; font-size:0.78rem; font-weight:600; padding:4px 12px; border-radius:20px;">SHAP Explainability</span>
            <span style="background:#805AD5; color:white; font-size:0.78rem; font-weight:600; padding:4px 12px; border-radius:20px;">LLaMA-3 AI Assistant</span>
            <span style="background:#D69E2E; color:white; font-size:0.78rem; font-weight:600; padding:4px 12px; border-radius:20px;">UCI Heart Dataset</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_sample_path = "data/sample_patients.csv"
if os.path.exists(_sample_path):
    with open(_sample_path, "rb") as _f:
        _sample_bytes = _f.read()
    col_dl, _ = st.columns([1, 4])
    with col_dl:
        st.download_button(
            "⬇️  Download Sample Patient CSV",
            data=_sample_bytes,
            file_name="sample_patients.csv",
            mime="text/csv",
            help="14 anonymised patient records — use this to test Batch Prediction or Patient Risk Prediction.",
            use_container_width=True,
        )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🤖  ML Models",       "6 + DNN",      "Trained & compared")
c2.metric("📐  Features",         "13 + 4",       "Clinical + engineered")
c3.metric("🔍  Explainability",   "SHAP + Rules", "Per-patient breakdown")
c4.metric("🗃️  Dataset",          "UCI Heart",    "303 clinical records")
c5.metric("🤖  AI Assistant",     "LLaMA-3",      "Free Groq LLM")

st.markdown("---")

st.markdown(
    '<div class="section-header"><span>⚙️  How it works</span></div>',
    unsafe_allow_html=True,
)

col_a, col_b, col_c, col_d, col_e = st.columns(5)

with col_a:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 1</div>
            <h4>🗂️  Train</h4>
            <p>Upload the UCI heart disease CSV. The pipeline cleans it,
            applies SMOTE for class balance, and trains 6 ML models with
            5-fold cross-validation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 2</div>
            <h4>📤  Upload / Enter</h4>
            <p>Enter patient values manually or upload your own health data
            CSV. The best model scores the patient instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 3</div>
            <h4>🔬  Predict</h4>
            <p>Get a risk probability with a visual gauge, clinical rule-based
            flags, and a SHAP waterfall chart explaining each factor.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 4</div>
            <h4>🤖  AI Report</h4>
            <p>The AI Health Assistant uses a Groq LLaMA-3 agent to generate
            a structured health report with recommendations and references.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_e:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">STEP 5</div>
            <h4>📄  Export PDF</h4>
            <p>Download a formatted PDF report containing the patient summary,
            AI-generated recommendations, and a medical disclaimer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    '<div class="section-header"><span>📦  Tech Stack</span></div>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("""
    <div class="tech-card">
        <div class="tc-label">Machine Learning</div>
        <div class="tc-title">🤖 ML Models</div>
        <div class="tc-items">scikit-learn · XGBoost<br>LightGBM · PyTorch<br>CardioNet DNN</div>
    </div>""", unsafe_allow_html=True)
with t2:
    st.markdown("""
    <div class="tech-card">
        <div class="tc-label">Explainability</div>
        <div class="tc-title">🔍 SHAP + Rules</div>
        <div class="tc-items">SHAP Tree + Kernel<br>Rule-based clinical flags<br>Per-patient waterfall</div>
    </div>""", unsafe_allow_html=True)
with t3:
    st.markdown("""
    <div class="tech-card">
        <div class="tc-label">Agentic AI</div>
        <div class="tc-title">🧠 LLM Agent</div>
        <div class="tc-items">Groq LLaMA-3.1<br>LangGraph · RAG<br>FAISS vector store</div>
    </div>""", unsafe_allow_html=True)
with t4:
    st.markdown("""
    <div class="tech-card">
        <div class="tc-label">Data Pipeline</div>
        <div class="tc-title">⚙️ Preprocessing</div>
        <div class="tc-items">SMOTE · StandardScaler<br>Feature engineering<br>Optuna · 5-fold CV</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        background: #FFFBEB;
        border: 1px solid #F6E05E;
        border-left: 4px solid #D69E2E;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 0.88rem;
        color: #744210;
    ">
        <strong>⚕️ Medical Disclaimer</strong> — This system is built for educational and research purposes only.
        It has not been validated for clinical use and must not be used to make clinical decisions.
        Always consult a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)

