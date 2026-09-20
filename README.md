# Customer Categorizer

A modular machine learning pipeline that segments customers into distinct groups using unsupervised clustering, built on a customer dataset of ~2,240 records and 22 features.

## Overview
The project moves from exploratory analysis (EDA notebooks) to a structured, reusable `src/`-based pipeline. Each stage is a separate component, making it easy to test, swap models, and extend.

## Pipeline Components
- **Data Ingestion:** loads and splits the raw customer data
- **Data Transformation:** cleaning, feature scaling, and PCA-based dimensionality reduction
- **Model Training:** trains Catboost clustering model, with config-driven (YAML) model selection and dynamic model loading via `importlib`
- **Evaluation:** silhouette scoring and plots to select the best model, with `eps` for DBSCAN tuned using `KneeLocator`
- **Prediction Pipeline:** loads the trained preprocessing object and model to assign segments to new customers

## Tech Stack
Python, scikit-learn, pandas, NumPy, YAML config

## Project Structure
```
src/
├── components/   # ingestion, transformation, training
├── pipeline/     # training and prediction pipelines
├── utils.py      # shared helpers (MainUtils)
└── ...
config/           # model and parameter YAML
notebooks/        # EDA
```

## Getting Started
```bash
git clone https://github.com/<your-username>/CUSTOMER-CATEGORIZER
cd CUSTOMER-CATEGORIZER
pip install -r requirements.txt
python src/pipeline/training_pipeline.py
```

## Key Learnings
- Structuring an ML project as a modular pipeline rather than a single notebook
- Why distance-based models (KMeans, Agglomerative) work on PCA-reduced data while DBSCAN is run on the full scaled feature space
- Evaluating unsupervised models with silhouette analysis

