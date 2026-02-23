"""
predictor.py
------------
Thin wrapper around model loading and inference.
Keeps app.py clean — the UI shouldn't care how models are stored.
"""

import numpy as np
from .preprocessor import preprocess_single, load_artifacts
from .trainer import load_trained_models


def _load_all():
    """Load all models and preprocessing artifacts."""
    all_models, results, best_name, best_model = load_trained_models()
    scaler, feature_cols = load_artifacts()
    return all_models, results, best_name, best_model, scaler, feature_cols


def predict_single(patient: dict, model=None, scaler=None, feature_cols=None,
                   use_optimal_threshold=False, results=None, model_name=None) -> dict:
    """
    Run inference for a single patient.

    Returns:
        dict with risk_label, probability, raw_pred, threshold.
    """
    if model is None or scaler is None or feature_cols is None:
        _, _, _, model, scaler, feature_cols = _load_all()

    X = preprocess_single(patient, scaler, feature_cols)

    threshold = 0.5
    if use_optimal_threshold and results and model_name and model_name in results:
        threshold = results[model_name].get("optimal_threshold", 0.5)

    prob     = float(model.predict_proba(X)[0][1])
    raw_pred = int(prob >= threshold)

    return {
        "risk_label":   "High Risk" if raw_pred == 1 else "Low Risk",
        "probability":  round(prob, 4),
        "raw_pred":     raw_pred,
        "threshold":    threshold
    }


def predict_batch(df, model, scaler, feature_cols) -> list:
    """Run predictions for all rows in a DataFrame. Returns list of result dicts."""
    from .preprocessor import fill_missing, engineer_features

    df = fill_missing(df)
    df = engineer_features(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X    = df[feature_cols].values
    X_sc = scaler.transform(X)

    probs = model.predict_proba(X_sc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return [
        {
            "risk_label":  "High Risk" if p == 1 else "Low Risk",
            "probability": round(float(pr), 4),
            "raw_pred":    int(p)
        }
        for pr, p in zip(probs, preds)
    ]
