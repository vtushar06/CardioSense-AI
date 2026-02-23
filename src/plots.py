"""Matplotlib chart helpers for the dashboard."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = {
    "primary":   "#2B6CB0",
    "danger":    "#E53E3E",
    "warning":   "#D69E2E",
    "success":   "#38A169",
    "neutral":   "#718096",
    "bg":        "#F7FAFC",
}

MODEL_COLORS = [
    "#E53E3E", "#3182CE", "#38A169",
    "#9B59B6", "#D69E2E", "#2B6CB0"
]


def roc_curves(results: dict):
    """All model ROC curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC=0.50)")

    for (name, metrics), color in zip(results.items(), MODEL_COLORS):
        fpr = metrics.get("fpr", [])
        tpr = metrics.get("tpr", [])
        auc = metrics.get("roc_auc", 0)
        if fpr and tpr:
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{name}  (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_facecolor(PALETTE["bg"])
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def confusion_matrix_fig(cm: list, model_name: str):
    """Heatmap of a 2x2 confusion matrix."""
    cm_arr = np.array(cm)
    labels = ["No Disease", "Disease"]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax, cbar=False,
        annot_kws={"size": 16, "weight": "bold"}
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def model_comparison_bar(results: dict, metric="roc_auc"):
    """Horizontal bar chart comparing models on a single metric."""
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    # colour best bar differently
    best_idx = int(np.argmax(values))
    colors   = [PALETTE["primary"]] * len(names)
    colors[best_idx] = PALETTE["success"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors, edgecolor="white")

    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontweight="bold")
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    return fig


def class_distribution_fig(df: pd.DataFrame, target_col="target"):
    """Pie + bar side by side for class balance."""
    counts = df[target_col].value_counts()
    labels = ["No Disease (0)", "Disease (1)"]
    colors = [PALETTE["success"], PALETTE["danger"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[0].set_title("Class Balance", fontweight="bold")

    axes[1].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold")
    axes[1].set_ylabel("Patient Count")
    axes[1].set_title("Absolute Counts", fontweight="bold")
    axes[1].set_facecolor(PALETTE["bg"])

    fig.suptitle("Target Variable Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation heatmap for numeric columns."""
    corr = df.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        square=True, linewidths=0.5, annot=True,
        fmt=".2f", annot_kws={"size": 7}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def feature_distribution_fig(df: pd.DataFrame, col: str, target_col="target"):
    """Overlapping histogram split by target class for a single feature."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for cls, color, lbl in [(0, PALETTE["success"], "No Disease"),
                             (1, PALETTE["danger"],  "Disease")]:
        subset = df[df[target_col] == cls][col].dropna()
        ax.hist(subset, bins=20, alpha=0.55, color=color,
                label=lbl, edgecolor="white")

    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of '{col}' by Class", fontweight="bold")
    ax.legend()
    ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    return fig


def dnn_training_history(history: dict):
    """Loss and AUC curves from neural network training."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color=PALETTE["primary"])
    axes[0].plot(history["val_loss"],   label="Val Loss",   color=PALETTE["danger"], linestyle="--")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].set_facecolor(PALETTE["bg"])

    axes[1].plot(history["train_auc"], label="Train AUC", color=PALETTE["primary"])
    axes[1].plot(history["val_auc"],   label="Val AUC",   color=PALETTE["success"], linestyle="--")
    axes[1].set_title("AUC Over Training", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].set_facecolor(PALETTE["bg"])

    fig.suptitle("CardioNet Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def radar_comparison(patients: list, predictions: list):
    """
    Radar chart comparing up to 4 patients on key clinical features.
    patients: list of patient dicts
    predictions: list of prediction dicts
    """
    features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    ranges   = {
        "age":      (20, 90),
        "trestbps": (80, 200),
        "chol":     (100, 600),
        "thalach":  (60, 220),
        "oldpeak":  (0, 6),
    }

    N       = len(features)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors  = [PALETTE["danger"], PALETTE["primary"], PALETTE["success"], "#9B59B6"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for i, (pat, pred) in enumerate(zip(patients, predictions)):
        vals = []
        for feat in features:
            lo, hi = ranges[feat]
            raw    = pat.get(feat, (lo + hi) / 2)
            vals.append(float(np.clip((raw - lo) / (hi - lo), 0, 1)))
        vals += vals[:1]

        pct   = round(pred["probability"] * 100, 1)
        label = f"P{i+1}: {pred['risk_label']} ({pct}%)"
        ax.plot(angles, vals, "o-", linewidth=2, color=colors[i % 4], label=label)
        ax.fill(angles, vals, alpha=0.12, color=colors[i % 4])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=11)
    ax.set_title("Patient Clinical Profile Comparison", fontweight="bold", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    return fig


def risk_score_histogram(risk_scores: list):
    """Distribution of risk probabilities from a batch prediction."""
    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(risk_scores, bins=20, edgecolor="white", linewidth=0.5)

    # colour by risk zone
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.33:
            patch.set_facecolor(PALETTE["success"])
        elif left < 0.66:
            patch.set_facecolor(PALETTE["warning"])
        else:
            patch.set_facecolor(PALETTE["danger"])

    ax.axvline(0.33, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.66, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.15, ax.get_ylim()[1] * 0.92, "Low", color=PALETTE["success"], fontweight="bold")
    ax.text(0.45, ax.get_ylim()[1] * 0.92, "Med", color=PALETTE["warning"], fontweight="bold")
    ax.text(0.75, ax.get_ylim()[1] * 0.92, "High", color=PALETTE["danger"],  fontweight="bold")

    ax.set_xlabel("Risk Probability", fontsize=11)
    ax.set_ylabel("Number of Patients")
    ax.set_title("Batch Risk Score Distribution", fontweight="bold")
    ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    return fig
