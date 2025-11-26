# S&P 500 Predictive ML Project

This project predicts whether a stock will be in the top 50% or top 10% of next-day / 5-day returns using:
- Logistic Regression
- Random Forest
- XGBoost

Metrics:
- Accuracy
- ROC AUC
- PR AUC
- Precision@K
- SHAP values
- Feature importance

Dataset built using S&P 500 historical constituents to avoid survivorship bias.

Code:
- src/build_ml_dataset.py
- src/train_models.py
- src/data_loader.py

Models use a TimeSeriesSplit (n_splits=5) evaluation.
