# PROPOSAL.md  
## Predicting Tomorrow’s Stock Market Winners  
### A Machine Learning Approach for Cross-Sectional Return Prediction

---

## 1. Introduction

Predicting short-term stock movements is one of the most challenging tasks in financial economics. Daily returns are extremely noisy, markets are highly competitive, and most academic literature suggests very limited predictability in the short run.

This project aims to investigate whether machine learning models can identify, on any given trading day, the stocks within the S&P 500 universe that are most likely to outperform their peers on the following day.

The objective is not to forecast exact returns, but to rank stocks and classify them into relative performance groups (e.g., Top-50% or Top-10% next-day performers).

This work includes a complete pipeline for data processing, feature generation, model training, evaluation, and interpretability.

---

## 2. Research Question

**Can machine learning models predict which S&P 500 stocks will outperform the cross-section of the market on the next trading day?**

Key sub-questions:

- Can technical and fundamental features help classify stocks into future outperformers vs underperformers?
- Can simple ML models (Logistic Regression, Random Forest) beat naive baselines?
- Are more complex models (XGBoost) significantly better?
- Is performance stable across time using walk-forward validation?

---

## 3. Data

### 3.1 Market Data  
Daily OHLCV data for all S&P 500 stocks, including:

- Open, High, Low, Close, Adjusted Close  
- Volume  
- Daily returns  

### 3.2 Fundamental Data  
Fundamental features include:

- Price-to-Earnings (P/E)  
- Price-to-Book  
- Price-to-Sales  
- Dividend yield  
- Earnings per share  
- EBITDA  
- Market capitalization  

### 3.3 Historical Index Constituents  
The project uses datasets containing historical S&P 500 membership to ensure:

- No survivorship bias  
- Correct stock universe for each historical date  

---

## 4. Methodology

### 4.1 Data Pipeline

1. Load and clean daily price data  
2. Load point-in-time fundamentals and forward-fill  
3. Merge fundamentals with market data  
4. Generate technical indicators and feature set  
5. Build classification targets  
6. Export final ML dataset  

All operations respect chronological ordering to avoid look-ahead bias.

---

### 4.2 Target Construction

Two classification targets:

#### (1) **Top-50% next-day performer**  
Primary target:  
A stock receives label **1** if its next-day return exceeds the cross-sectional median.

#### (2) **Top-10% next-day performer**  
Optional robustness target.

---

### 4.3 Model Training

Trained models:

- Logistic Regression (with StandardScaler)  
- Random Forest  
- XGBoost (full run only)  

Training procedure:

- Walk-forward validation using **TimeSeriesSplit**  
- No shuffling (chronological integrity)  
- Evaluation repeated over multiple folds  

---

### 4.4 Evaluation Metrics

- Accuracy  
- ROC-AUC  
- PR-AUC  
- Precision@K (K = 10%, 20%)  
- Confusion matrices  
- Learning curves  
- Permutation feature importance  

Baselines:

- Random predictions  
- Naive momentum (return > 0)  
- Buy & Hold (always predict 1)

---

## 5. Expected Challenges

- High noise in daily returns  
- Limited short-term predictability  
- Aligning low-frequency fundamentals with daily data  
- Avoiding look-ahead bias  
- Variability in model performance across regimes  
- Class imbalance in Top-10% target  

---

## 6. Expected Contribution

The project aims to:

- Build a **fully reproducible financial ML pipeline**  
- Evaluate whether ML models can outperform simple baselines  
- Provide insights using feature importance and SHAP  
- Assess the economic relevance of short-term predictability  

---

## 7. Tools and Technologies

- Python 3.10+  
- pandas, numpy  
- scikit-learn  
- xgboost  
- shap  
- matplotlib  
- seaborn  
- yfinance  
- ta  
- pyarrow  


---

## 8. Expected Deliverables

- Complete ML-ready dataset  
- Full preprocessing and feature engineering scripts  
- Walk-forward ML models  
- Evaluation plots (ROC, PR, Precision@K, SHAP)  
- Analytical report  
- Clean and documented GitHub repository  

---

## 9. Conclusion

This project explores whether next-day stock outperformance is predictable using a combination of technical and fundamental features applied to historical S&P 500 data.

By building a rigorous ML pipeline, performing walk-forward validation, and leveraging interpretability tools, the project aims to contribute empirical evidence about the limits and potential of short-term predictive signals in equity markets.