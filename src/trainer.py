"""Model training, evaluation, and persistence for sklearn-based classifiers."""

import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


def get_models() -> dict:
    """Return dict of model name to configured estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=0.8,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
    }


def run_cross_validation(model, X_train, y_train, n_folds=5) -> dict:
    """Run stratified k-fold cross-validation and return AUC stats."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "cv_auc_mean": float(scores.mean()),
        "cv_auc_std":  float(scores.std()),
        "cv_scores":   scores.tolist()
    }


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute metrics with both default and optimal (Youden's J) thresholds.

    Returns dict with accuracy, AUC, F1, confusion matrices, and ROC data.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_idx   = int(np.argmax(tpr - fpr))
    best_thresh = float(thresholds[best_idx])

    y_pred_opt = (y_prob >= best_thresh).astype(int)

    return {
        "name":             model_name,
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "accuracy_opt":     round(accuracy_score(y_test, y_pred_opt), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "f1":               round(f1_score(y_test, y_pred), 4),
        "f1_opt":           round(f1_score(y_test, y_pred_opt), 4),
        "optimal_threshold":best_thresh,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "confusion_matrix_opt": confusion_matrix(y_test, y_pred_opt).tolist(),
        "class_report":     classification_report(y_test, y_pred, output_dict=True),
        "fpr":              fpr.tolist(),
        "tpr":              tpr.tolist(),
    }


def train_all(X_train, y_train, X_test, y_test,
              calibrate=True, cv_folds=5,
              progress_fn=None) -> tuple:
    """
    Train all models, optionally calibrate, cross-validate, and evaluate.

    Args:
        calibrate: wrap models in CalibratedClassifierCV (isotonic).
        progress_fn: optional callback(model_name, step_pct).

    Returns:
        trained_models, results, best_name
    """
    os.makedirs("models", exist_ok=True)

    models  = get_models()
    trained = {}
    results = {}
    total   = len(models)

    for i, (name, model) in enumerate(models.items()):
        if progress_fn:
            progress_fn(name, int((i / total) * 90))

        # Cross-validation first (on training data only)
        cv_stats = run_cross_validation(model, X_train, y_train, cv_folds)

        # Final fit on the full training set
        model.fit(X_train, y_train)

        # Calibrate probabilities
        if calibrate:
            cal_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
            cal_model.fit(X_train, y_train)
            trained[name] = cal_model
        else:
            trained[name] = model

        # Evaluate on held-out test set
        metrics = evaluate(trained[name], X_test, y_test, name)
        metrics.update(cv_stats)
        results[name] = metrics

    if progress_fn:
        progress_fn("Done", 100)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])

    _persist(trained, results, best_name)
    return trained, results, best_name


def _persist(trained, results, best_name):
    with open("models/all_models.pkl", "wb") as f:
        pickle.dump(trained, f)
    with open("models/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("models/best_name.pkl", "wb") as f:
        pickle.dump(best_name, f)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(trained[best_name], f)


def load_trained_models() -> tuple:
    """Load all persisted models and results."""
    with open("models/all_models.pkl", "rb") as f:
        all_models = pickle.load(f)
    with open("models/results.pkl", "rb") as f:
        results = pickle.load(f)
    with open("models/best_name.pkl", "rb") as f:
        best_name = pickle.load(f)
    with open("models/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    return all_models, results, best_name, best_model
