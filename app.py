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
        background: linear-gradient(180deg, #0F2444 0%, #1A365D 50%, #2B6CB0 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #EBF8FF !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label { color: #BEE3F8 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
    [data-testid="stSidebar"] .stSuccess { background: rgba(56,161,105,0.2) !important; border-radius: 8px; }
    [data-testid="stSidebar"] .stWarning { background: rgba(214,158,46,0.2) !important; border-radius: 8px; }

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
        transition: box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    [data-testid="stMetric"] label { color: #718096 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
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

    /* ── Info / step cards ── */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .step-card .step-num {
        display: inline-block;
        background: #2B6CB0;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.6rem;
    }
    .step-card h4 { color: #1A365D; margin: 0.4rem 0 0.6rem 0; font-size: 1.05rem; }
    .step-card p  { color: #4A5568; font-size: 0.9rem; margin: 0; line-height: 1.55; }

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
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s, box-shadow 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        box-shadow: 0 4px 14px rgba(43,108,176,0.4);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F7FAFC;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        color: #4A5568;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2B6CB0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
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
