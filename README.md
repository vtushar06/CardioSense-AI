# CardioSense AI

Cardiovascular risk assessment platform built with Streamlit and scikit-learn.

Trains 6 ML models + a PyTorch neural network on clinical patient data,
compares their performance, and provides per-patient explainability
via SHAP values and rule-based clinical flags.

## Project Structure

```
CardioSense-AI/
  app.py                      Main Streamlit entry point (config + styling)
  requirements.txt            Python dependencies
  pages/
    1_Overview.py             Landing page
    2_Train_Models.py         Dataset upload, training, evaluation
    3_Predict_Patient.py      Single patient risk prediction + SHAP
    4_Batch_Prediction.py     CSV batch predictions
    5_Dataset_Explorer.py     EDA visualisations
    6_Neural_Network.py       PyTorch CardioNet training
  src/
    data_loader.py            CSV loading and validation
    preprocessor.py           Feature engineering, scaling, SMOTE
    trainer.py                6 classical ML models
    deep_model.py             PyTorch neural network
    explainer.py              SHAP + clinical rule flags
    predictor.py              Inference wrapper
    plots.py                  Matplotlib / Seaborn visualisations
  data/
    heart.csv                 UCI Heart Disease dataset (303 records)
  models/                     Auto-created after training
  notebooks/
    pipeline_testing.ipynb    End-to-end pipeline notebook
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/CardioSense-AI.git
cd CardioSense-AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS with Apple Silicon, LightGBM requires OpenMP:

```bash
brew install libomp
```

## Running

```bash
streamlit run app.py
```

Open http://localhost:8501. Navigate to **Train Models**, upload `heart.csv`,
and train. Then use **Predict Patient** or **Batch Prediction**.

## Dataset

Source: UCI Machine Learning Repository — Cleveland Heart Disease dataset
(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

303 patient records with 13 clinical features and a binary target
(0 = no disease, 1 = disease).

| Column   | Description                                  |
|----------|----------------------------------------------|
| age      | Age in years                                 |
| sex      | 1 = male, 0 = female                        |
| cp       | Chest pain type (0-3)                        |
| trestbps | Resting blood pressure (mmHg)                |
| chol     | Serum cholesterol (mg/dl)                    |
| fbs      | Fasting blood sugar > 120 mg/dl (1 = yes)   |
| restecg  | Resting ECG result (0, 1, 2)                |
| thalach  | Maximum heart rate achieved                  |
| exang    | Exercise-induced angina (1 = yes)            |
| oldpeak  | ST depression during exercise                |
| slope    | Slope of peak exercise ST segment            |
| ca       | Number of major vessels coloured (0-3)       |
| thal     | Thalassemia type                             |

## Models

| Model               | Type                  |
|----------------------|-----------------------|
| Logistic Regression  | Linear                |
| Decision Tree        | Tree                  |
| Random Forest        | Ensemble (bagging)    |
| Gradient Boosting    | Ensemble (boosting)   |
| XGBoost              | Gradient boosting     |
| LightGBM             | Gradient boosting     |
| CardioNet            | PyTorch feed-forward  |

All classical models support probability calibration (isotonic regression)
and 5-fold stratified cross-validation. SMOTE is applied to the training
set to handle class imbalance.

## Explainability

- **SHAP**: TreeExplainer for tree-based models, KernelExplainer for others.
  Produces per-patient waterfall charts showing feature contributions.
- **Clinical flags**: Deterministic checks against known medical thresholds
  (e.g. cholesterol > 240, BP > 140).

## Deployment

Push to GitHub, then deploy via:

- **Streamlit Community Cloud**: https://share.streamlit.io — set main file to `app.py`
- **Hugging Face Spaces**: Create a new Space with SDK = Streamlit

## Disclaimer

This system is built for academic purposes. It has not been validated for
clinical use and must not substitute advice from a qualified medical
professional.
