"""SHAP explanations and rule-based clinical flags."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from .data_loader import COLUMN_LABELS, NORMAL_RANGES


@st.cache_resource(show_spinner=False)
def make_explainer(model, X_train: np.ndarray, model_name: str):
    """Create a SHAP explainer appropriate for the model type.

    Cached with @st.cache_resource so the explainer is built once per app
    lifetime and reused across all user sessions — avoids expensive rebuilds
    on every prediction and reduces peak memory on the cloud container.
    """
    tree_based = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Decision Tree"]

    if any(name in model_name for name in tree_based):
        return shap.TreeExplainer(model)
    else:
        background = shap.sample(X_train, min(50, len(X_train)))
        return shap.KernelExplainer(model.predict_proba, background)


def shap_summary_figure(explainer, X_test: np.ndarray, feature_cols: list):
    """Generate a SHAP beeswarm summary plot for the test set."""
    sv = explainer.shap_values(X_test)

    # binary classifiers return a list [class0_vals, class1_vals]
    if isinstance(sv, list):
        sv = sv[1]

    display_names = [COLUMN_LABELS.get(c, c) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=display_names, show=False, plot_type="beeswarm")
    plt.tight_layout()
    return fig


def shap_single_patient_figure(explainer, patient_X: np.ndarray, feature_cols: list):
    """Bar chart of top SHAP contributions for a single patient."""
    sv = explainer.shap_values(patient_X)
    if isinstance(sv, list):
        sv = sv[1][0]
    else:
        sv = sv[0]

    display_names = [COLUMN_LABELS.get(c, c) for c in feature_cols]

    # Top 10 by absolute SHAP impact
    top_idx   = np.argsort(np.abs(sv))[-10:][::-1]
    top_names = [display_names[i] for i in top_idx]
    top_vals  = sv[top_idx]

    colors = ["#E53E3E" if v > 0 else "#38A169" for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value  (positive = increases risk)", fontsize=11)
    ax.set_title("Why This Prediction? — Top 10 Contributing Factors", fontsize=13, fontweight="bold")

    # value labels on bars
    for bar, val in zip(bars, top_vals[::-1]):
        x_pos = bar.get_width() + (0.003 if val >= 0 else -0.003)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9)

    red_patch   = mpatches.Patch(color="#E53E3E", label="Increases risk")
    green_patch = mpatches.Patch(color="#38A169", label="Decreases risk")
    ax.legend(handles=[red_patch, green_patch], loc="lower right")

    plt.tight_layout()
    return fig


def feature_importance_figure(model, feature_cols: list, model_name: str):
    """Model-native feature importance (tree importances or coefficient magnitudes)."""
    display_names = [COLUMN_LABELS.get(c, c) for c in feature_cols]

    importances = None
    label       = "Importance"

    # CalibratedClassifierCV wraps the estimator — need to unwrap
    estimator = model
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
        label       = "Coefficient Magnitude"

    if importances is None:
        return None

    sorted_idx  = np.argsort(importances)
    sorted_names = [display_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    colors = plt.cm.RdYlGn(sorted_vals / sorted_vals.max())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sorted_names, sorted_vals, color=colors)
    ax.set_xlabel(label)
    ax.set_title(f"{model_name} — Feature Importance", fontweight="bold")
    plt.tight_layout()
    return fig


def risk_gauge_figure(probability: float):
    """Semi-circular gauge visualisation for risk probability."""
    if probability < 0.33:
        color = "#38A169"
        label = "LOW RISK"
    elif probability < 0.66:
        color = "#D69E2E"
        label = "MEDIUM RISK"
    else:
        color = "#E53E3E"
        label = "HIGH RISK"

    fig, ax = plt.subplots(figsize=(4, 2.8), subplot_kw={"projection": "polar"})

    # Background arc
    theta_bg = np.linspace(0, np.pi, 200)
    ax.plot(theta_bg, np.ones(200) * 0.8, color="#E2E8F0", linewidth=14, solid_capstyle="round")

    # Filled arc proportional to probability
    theta_fill = np.linspace(0, probability * np.pi, 200)
    ax.plot(theta_fill, np.ones(200) * 0.8, color=color, linewidth=14, solid_capstyle="round")

    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.axis("off")

    ax.text(np.pi / 2, -0.15, f"{probability*100:.1f}%",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=color, transform=ax.transData)
    ax.text(np.pi / 2, -0.5, label,
            ha="center", va="center", fontsize=13, color=color,
            transform=ax.transData)

    plt.tight_layout()
    return fig


def get_rule_based_flags(patient: dict) -> list:
    """
    Check patient vitals against clinical thresholds.

    Returns:
        list of (level, message) tuples.
    """
    flags = []

    age      = patient.get("age", 0)
    chol     = patient.get("chol", 0)
    bp       = patient.get("trestbps", 0)
    hr       = patient.get("thalach", 0)
    fbs      = patient.get("fbs", 0)
    exang    = patient.get("exang", 0)
    cp       = patient.get("cp", 0)
    oldpeak  = patient.get("oldpeak", 0)

    if age >= 60:
        flags.append(("warning", f"Age {age} — risk increases significantly after 60."))
    if chol > 240:
        flags.append(("danger",  f"Cholesterol {chol} mg/dl — high (normal < 200 mg/dl)."))
    elif chol > 200:
        flags.append(("warning", f"Cholesterol {chol} mg/dl — borderline (normal < 200 mg/dl)."))
    if bp > 140:
        flags.append(("danger",  f"Blood pressure {bp} mmHg — Stage 2 hypertension (> 140)."))
    elif bp > 130:
        flags.append(("warning", f"Blood pressure {bp} mmHg — Stage 1 hypertension (> 130)."))
    if hr < 60:
        flags.append(("warning", f"Max heart rate {hr} bpm — below normal range."))
    if fbs == 1:
        flags.append(("warning", "Fasting blood sugar > 120 mg/dl — possible pre-diabetes."))
    if exang == 1:
        flags.append(("danger",  "Exercise-induced angina present — significant cardiac warning."))
    if cp in [1, 2, 3]:
        cp_map = {1: "atypical angina", 2: "non-anginal chest pain", 3: "asymptomatic"}
        flags.append(("warning", f"Chest pain type: {cp_map[cp]}."))
    if oldpeak > 2:
        flags.append(("danger",  f"ST depression {oldpeak} — indicates cardiac stress under exercise."))

    if not flags:
        flags.append(("ok", "No major clinical flags detected against standard thresholds."))

    return flags
