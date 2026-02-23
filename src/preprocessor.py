"""Feature engineering, scaling, and train/test splitting."""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Features used for training.
BASE_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

TARGET = "target"


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill nulls with column medians."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw clinical data."""
    df = df.copy()

    # Age risk buckets
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 65, 120],
        labels=[0, 1, 2, 3]  # 0=young, 1=middle, 2=senior, 3=elderly
    ).astype(int)

    # High BP + High cholesterol together = compounded risk
    df["bp_chol_combined"] = (
        (df["trestbps"] > 130).astype(int) +
        (df["chol"] > 240).astype(int)
    )

    # Age-adjusted heart rate: lower ratio suggests reduced cardiac output
    df["hr_age_ratio"] = (df["thalach"] / df["age"]).round(4)

    # Chest pain type × exercise angina interaction
    df["cp_exang_interaction"] = df["cp"] * df["exang"]

    return df


def get_feature_list() -> list:
    """Return the full feature list (base + engineered)."""
    engineered = ["age_bucket", "bp_chol_combined", "hr_age_ratio", "cp_exang_interaction"]
    return BASE_FEATURES + engineered


def build_pipeline(df: pd.DataFrame, test_size=0.2, apply_smote=True):
    """
    Main preprocessing pipeline: impute, engineer features, split,
    optionally apply SMOTE, and scale.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    df = fill_missing(df)
    df = engineer_features(df)

    feature_cols = get_feature_list()
    X = df[feature_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    if apply_smote:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    _save_artifacts(scaler, feature_cols)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def preprocess_single(patient: dict, scaler: StandardScaler, feature_cols: list) -> np.ndarray:
    """Preprocess a single patient dict into a scaled array for inference."""
    row = pd.DataFrame([patient])
    row = fill_missing(row)
    row = engineer_features(row)

    # Fill missing engineered columns with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    X = row[feature_cols].values
    return scaler.transform(X)


def _save_artifacts(scaler, feature_cols):
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)


def load_artifacts():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return scaler, feature_cols
