"""Dataset loading and validation utilities."""

import os
import pandas as pd
import numpy as np


# Expected columns for the UCI Heart Disease dataset.
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Human-readable column labels for UI display.
COLUMN_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Result",
    "thalach":  "Maximum Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "No. of Major Vessels Coloured by Fluoroscopy",
    "thal":     "Thalassemia",
    "target":   "Heart Disease Present"
}

NORMAL_RANGES = {
    "trestbps": (90, 120, "mmHg"),
    "chol":     (0, 200, "mg/dl"),
    "thalach":  (60, 100, "bpm"),
    "oldpeak":  (0, 2,   ""),
}


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV and normalise column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run basic checks and return a validation report.

    Returns:
        dict with keys: ok, missing_cols, issues, shape
    """
    report = {
        "ok": True,
        "missing_cols": [],
        "issues": [],
        "shape": df.shape
    }

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        report["ok"] = False
        report["missing_cols"] = missing
        report["issues"].append(f"Missing columns: {missing}")

    if "target" in df.columns:
        class_counts = df["target"].value_counts()
        if len(class_counts) < 2:
            report["issues"].append("Target column has only one class — cannot train.")
            report["ok"] = False

        # warn if severely imbalanced (more than 4:1 ratio)
        if len(class_counts) == 2:
            ratio = class_counts.max() / class_counts.min()
            if ratio > 4:
                report["issues"].append(
                    f"Class imbalance detected (ratio {ratio:.1f}:1). "
                    "SMOTE will be applied during training."
                )

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        report["issues"].append(
            f"Null values found in: {cols_with_nulls.to_dict()}. "
            "These will be imputed with column medians."
        )

    return report


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return basic descriptive stats for the dashboard."""
    target_col = "target" if "target" in df.columns else None

    stats = {
        "n_patients": len(df),
        "n_features": len(df.columns) - (1 if target_col else 0),
        "missing_pct": df.isnull().mean().mean() * 100,
        "describe": df.describe().round(2).to_dict(),
    }

    if target_col:
        vc = df[target_col].value_counts()
        stats["class_distribution"] = vc.to_dict()
        stats["disease_pct"] = round(vc.get(1, 0) / len(df) * 100, 1)

    return stats


def load_from_upload(uploaded_file) -> pd.DataFrame:
    """Read a CSV from a Streamlit uploaded file object."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]
    return df
