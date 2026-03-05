# Loan Sanction Prediction System

An **end-to-end production-grade ML system** that predicts loan approval eligibility and estimated sanction amount using a dual Random Forest model pipeline.

---

## Project Architecture

```
-Loan-Sanction-Prediction-System/
├── src/
│   ├── components/           # 5 modular ML pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all 5 stages
│   │   └── prediction_pipeline.py  # Serves predictions (PredictPipeline + CustomData)
│   ├── config/configuration.py     # ConfigurationManager
│   ├── entity/config_entity.py     # Typed dataclasses
│   ├── constants/__init__.py       # Central paths
│   ├── utils/common.py             # YAML, pickle, JSON helpers
│   ├── logger/__init__.py          # File + console logger
│   └── exception/__init__.py       # Custom exception with traceback
├── config/config.yaml              # All artifact paths
├── params.yaml                     # Model hyperparameters
├── schema.yaml                     # Feature schema
├── artifacts/                      # Auto-generated (gitignored)
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── templates/index.html            # Flask web UI
├── static/style.css                # Professional banking CSS
├── app.py                          # Flask REST API
├── Dockerfile                      # Docker container
├── render.yaml                     # Render deployment config
├── requirements.txt
└── setup.py
```

---

## 🤖 ML Models

| Task | Model | Notes |
|------|-------|-------|
| **Classification** | Random Forest (300 trees) | Predicts: Loan Approved Yes/No |
| **Regression** | Random Forest (200 trees) | Predicts: Sanctioned Amount (USD) |

**Preprocessing pipeline**:  
- IQR outlier capping → Missing value imputation → Log transform (skewed features) → OrdinalEncoder + OneHotEncoder + StandardScaler via `ColumnTransformer`

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd -Loan-Sanction-Prediction-System
pip install -r requirements.txt
pip install -e .
```

### 2. Train the models

```bash
python src/pipeline/training_pipeline.py
```

This runs all 5 stages and saves models to `artifacts/model_trainer/`.

### 3. Run the Flask app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 📦 Deployment on Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your repo — Render will auto-detect `render.yaml`
4. That's it! The `buildCommand` trains models; `startCommand` launches gunicorn.

### Docker (local)

```bash
docker build -t loan-prediction .
docker run -p 8080:8080 loan-prediction
```

Open `http://localhost:8080`.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Web UI form |
| `POST` | `/predict` | Submit form, get prediction |
| `GET`  | `/health` | Health check (`{"status": "ok"}`) |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | All artifact directory + file paths |
| `params.yaml` | RF hyperparameters (n_estimators, etc.) |
| `schema.yaml` | Feature names, types, encoding strategies |
| `src/pipeline/training_pipeline.py` | Run this to train from scratch |
| `artifacts/model_evaluation/metrics.json` | Model performance metrics |

---

## 🧪 Metrics (after training)

Check `artifacts/model_evaluation/metrics.json` for:
- **Classifier**: accuracy, precision, recall, F1
- **Regressor**: R², MAE, RMSE

---

## 📋 Requirements

- Python 3.9+
- scikit-learn, pandas, numpy, flask, gunicorn, dill, pyyaml, python-box
