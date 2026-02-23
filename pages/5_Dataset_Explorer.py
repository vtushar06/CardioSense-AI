"""Dataset exploration and EDA page."""

import os
import pandas as pd
import streamlit as st

from src.data_loader import get_summary_stats
from src.plots import (
    class_distribution_fig,
    correlation_heatmap,
    feature_distribution_fig,
)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
        <span style="font-size:1.7rem;">📈</span>
        <h1 style="margin:0;">Dataset Explorer</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Explore distributions, correlations, and raw records in the loaded dataset.
    </p>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists("data/heart.csv"):
    st.warning("⚠️  Upload a dataset via the **Train Models** page first.")
    st.stop()

df = pd.read_csv("data/heart.csv")
stats = get_summary_stats(df)

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("👥  Total Patients",  stats["n_patients"])
c2.metric("📐  Features",        stats["n_features"])
c3.metric("❤️  Disease Cases",   stats["class_distribution"].get(1, 0))
c4.metric("❓  Missing Values",  f"{stats['missing_pct']:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Raw Data",
    "📊  Distributions",
    "🔗  Correlations",
    "📐  Statistics",
])

with tab1:
    n = st.slider("Rows to preview", 5, 50, 20)
    st.dataframe(df.head(n), use_container_width=True)

with tab2:
    st.markdown(
        '<div class="section-header"><span>⚖️  Class Balance</span></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(class_distribution_fig(df))

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>📊  Feature Distribution by Class</span></div>',
        unsafe_allow_html=True,
    )
    col = st.selectbox("Select feature", [c for c in df.columns if c != "target"])
    st.pyplot(feature_distribution_fig(df, col))

with tab3:
    st.markdown(
        '<div class="section-header"><span>🔗  Pearson Correlation Matrix</span></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(correlation_heatmap(df))

with tab4:
    st.markdown(
        '<div class="section-header"><span>📐  Descriptive Statistics</span></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df.describe().round(3), use_container_width=True)
