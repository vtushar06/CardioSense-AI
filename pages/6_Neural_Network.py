"""PyTorch CardioNet training page."""

import os
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessor import load_artifacts, build_pipeline
from src.trainer import load_trained_models
from src.plots import dnn_training_history, confusion_matrix_fig


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
        <span style="font-size:1.7rem;">🧠</span>
        <h1 style="margin:0;">CardioNet — Deep Neural Network</h1>
    </div>
    <p style="color:#718096; margin-bottom:1.5rem;">
        Train a PyTorch feed-forward network on the same preprocessed data as the classical models.
    </p>
    """,
    unsafe_allow_html=True,
)

if not _models_ready():
    st.warning("⚠️  Train classical models first — go to the **Train Models** page.")
    st.stop()

# ── Architecture overview ─────────────────────────────────────────────────────
with st.expander("🏗️  Architecture", expanded=False):
    st.markdown("""
| Layer | Details |
|---|---|
| Input | BatchNorm1d → 17 features |
| Hidden 1 | Linear(128) → BatchNorm → LeakyReLU(0.1) → Dropout |
| Hidden 2 | Linear(64) → BatchNorm → LeakyReLU(0.1) → Dropout |
| Hidden 3 | Linear(32) → LeakyReLU(0.1) |
| Output | Linear(1) — logit, converted via sigmoid at inference |

**Training:** AdamW · BCEWithLogitsLoss · CosineAnnealingLR · gradient clipping · early stopping
""")

st.markdown("---")
st.markdown(
    '<div class="section-header"><span>⚙️  Hyperparameters</span></div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    epochs     = st.slider("📅  Max epochs", 50, 300, 150)
    lr         = st.select_slider("📉  Learning rate", [0.0001, 0.001, 0.01], value=0.001)
with c2:
    batch_size = st.select_slider("📦  Batch size", [16, 32, 64], value=32)
    dropout    = st.slider("🎲  Dropout rate", 0.1, 0.5, 0.3, step=0.05)

if not os.path.exists("models/X_train.npy"):
    st.warning("⚠️  X_train.npy not found — retrain classical models first.")
    st.stop()

X_train_np = np.load("models/X_train.npy")
X_test_np  = np.load("models/X_test.npy")

if not os.path.exists("data/heart.csv"):
    st.error("❌  heart.csv missing — re-upload the dataset.")
    st.stop()

df_raw = pd.read_csv("data/heart.csv")
scaler, feature_cols = load_artifacts()
_, _, y_tr, y_te, _, _ = build_pipeline(df_raw, apply_smote=False)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀  Train CardioNet", type="primary"):
    from src.deep_model import train as dnn_train, evaluate_saved

    progress   = st.progress(0)
    status_txt = st.empty()
    chart_slot = st.empty()

    auc_log = {"train": [], "val": []}

    def epoch_cb(ep, total, tl, vl, va):
        progress.progress(int(ep / total * 100))
        status_txt.markdown(f"⏳  Epoch `{ep}/{total}` — Val AUC: `{va:.4f}`")
        auc_log["train"].append(float(1 - tl))
        auc_log["val"].append(float(va))
        chart_slot.line_chart(pd.DataFrame(auc_log, columns=["train", "val"]))

    model, history, device, best_auc = dnn_train(
        X_train_np, y_tr, X_test_np, y_te,
        epochs=epochs, lr=lr, batch_size=batch_size, dropout=dropout,
        on_epoch_end=epoch_cb,
    )

    progress.progress(100)
    status_txt.markdown(f"✅  Training complete — Best Val AUC: **{best_auc:.4f}**")

    dnn_res = evaluate_saved(X_test_np, y_te)

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span>📊  CardioNet Results</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯  AUC",      dnn_res["roc_auc"])
    c2.metric("✅  Accuracy", dnn_res["accuracy"])
    c3.metric("📊  F1",       dnn_res["f1"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(dnn_training_history(history))
    with col_b:
        st.pyplot(confusion_matrix_fig(dnn_res["confusion_matrix"], "CardioNet (DNN)"))

    _, all_results, cls_best_name, _, _, _ = _get_trained_artifacts()
    if all_results and cls_best_name in all_results:
        classical_auc = all_results[cls_best_name]["roc_auc"]
        st.markdown("---")
        st.markdown(
            '<div class="section-header"><span>⚖️  DNN vs Best Classical Model</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"| Model | AUC |\n|---|---|\n"
            f"| **{cls_best_name}** (classical best) | `{classical_auc}` |\n"
            f"| **CardioNet (DNN)** | `{dnn_res['roc_auc']}` |"
        )
