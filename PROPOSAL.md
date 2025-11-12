# PROPOSAL.md

## Project Title  
**Predicting Tomorrow’s Stock Market Winners – A Machine Learning Approach to Daily Top Performer Classification**

## Category  
🧠 *Machine Learning & Data Science*  
📊 *Financial Data Analysis*  

---

## Problem Statement / Motivation  

Predicting short-term stock movements is one of the most challenging and fascinating problems in data science.  
This project aims to identify which stocks within a major index (such as the S&P 500 or CAC 40) are most likely to **outperform the market on the next trading day**.  

Instead of focusing on visualization or risk reporting, this work will emphasize **model estimation, feature engineering, and predictive performance** — the core elements of applied data science.  

The task will be formulated as a **binary classification problem**:
> Will this stock be among the top 10% performers tomorrow (1) or not (0)?

By developing and comparing machine learning models, the project seeks to understand how technical indicators and short-term market signals can help anticipate daily winners.

---

## Planned Approach and Technologies  

### 1. Data Collection  
- Historical price and volume data will be retrieved using `yfinance` for all components of the S&P 500 (or CAC 40).  
- The time range will cover multiple years (e.g., 2015–2025) to include different market conditions.  
- Data will be stored locally in `parquet` or `csv` format for reproducibility.

### 2. Feature Engineering  
Each stock-day observation will be transformed into a set of explanatory variables, including:  
- **Returns:** 1-day, 5-day, and lagged returns  
- **Momentum indicators:** Simple and Exponential Moving Averages (SMA, EMA), RSI, MACD  
- **Volatility measures:** Rolling standard deviation, realized volatility  
- **Volume-related features:** Volume z-scores, turnover ratios  
- **Calendar dummies:** Day-of-week, end-of-month effects  
- **Relative performance:** Return compared to index average  

The target variable will be binary:  
\[
y_t = 
\begin{cases} 
1 & \text{if next-day return is in top 10% of all stocks} \\
0 & \text{otherwise}
\end{cases}
\]

### 3. Modeling  
The following models will be implemented and compared:
- **Baseline:** Logistic Regression  
- **Tree-based models:** Random Forest, Gradient Boosting (XGBoost or LightGBM)  
- **Regularization and hyperparameter tuning:** GridSearchCV with `TimeSeriesSplit`  
- Optional stretch: simple neural network with `Keras`  

Each model will be trained in a **walk-forward (rolling) manner** to respect the temporal nature of financial data and avoid look-ahead bias.

### 4. Evaluation Metrics  
Performance will be assessed through:
- **Precision@K:** fraction of true top performers among the K predicted highest probabilities  
- **ROC-AUC & PR-AUC:** to measure general classification quality  
- **Cumulative gain analysis:** optional, to show potential practical impact of top-K strategy  

Feature importance and Shapley values will also be analyzed to interpret which indicators contribute most to the prediction.

---

## Expected Challenges  
- Managing **non-stationary and noisy financial data**.  
- Handling **class imbalance** (only ~10% of positive labels).  
- Preventing **data leakage** through proper time-aware validation.  
- Ensuring **model interpretability** and avoiding overfitting.  

---

## Success Criteria  
- Models achieve significantly higher Precision@K and ROC-AUC than random or naïve baselines.  
- The feature engineering process is clearly documented and reproducible.  
- The methodology demonstrates rigorous application of machine learning concepts (train/test splits, cross-validation, evaluation).  
- Results are interpretable and well presented in the final report.  

---

## Stretch Goals  
- Analyze how feature importance or model performance changes across market regimes (bull vs. bear).  
- Extend to multi-day horizons (e.g., Top 10% over 5-day returns).  
- Compare multiple indices (S&P 500 vs. CAC 40).  

---

## Technical Summary  
**Language:** Python 3.10+  
**Main Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `yfinance`, `ta`, `shap`, `pytest`  
**Estimated Code Length:** ~1,000 lines  
**Deliverables:**  
- Data preprocessing and feature engineering scripts  
- Machine learning models and evaluation notebooks  
- Analytical report and performance summary  

---

**Author:** Rafael Machado Cerqueira  
**Institution:** HEC Lausanne  
**Date:** November 2025
