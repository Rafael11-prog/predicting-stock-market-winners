# PROPOSAL.md

## Project Title
**Predicting Tomorrow’s Stock Market Winners – A Machine Learning Approach to Cross-Sectional Outperformance Classification**

## Category
📊 *Data Analysis & Machine Learning*
💼 *Business & Finance Tools*

---

## Problem Statement / Motivation

Predicting short-term stock movements is one of the most difficult and theoretically constrained tasks in empirical finance.

This project aims to determine which stocks among the **historical S&P 500 constituents** are most likely to outperform the cross-section of the index on the next trading day.

Following the professor’s feedback, this proposal now:
- Uses **historical S&P 500 constituents** to avoid survivorship bias.
- Includes **fundamental data** (P/E, P/B, ROE, D/E, margins, EPS growth, market cap).
- Predicts **Top-50% next-day performers** instead of Top-10%, making the target economically meaningful.
- Keeps Top-10% only as an optional robustness extension.

---

## Planned Approach and Technologies

### 1. Data Collection
- Daily OHLCV market data via `yfinance`
- **Historical S&P 500 membership data** (no survivorship bias)
- Time range: 2015–2025
- Fundamental variables:
  - P/E, P/B
  - ROE, Gross margin
  - Debt-to-Equity
  - EPS growth
  - Market capitalization

---

### 2. Feature Engineering

#### Technical Indicators
- 1-day, 5-day, 20-day returns  
- Rolling volatility (10-day, 20-day)  
- SMA, EMA, RSI, MACD  
- Volume indicators  
- Calendar effects  

#### Fundamental Indicators
- P/E, P/B  
- ROE  
- Debt-to-Equity  
- EPS growth  
- Profit margins  
- Market capitalization  

---

### 3. Target Variable

**Primary target:**  
Top-50% next-day performer  
*(return ≥ cross-sectional median)*

**Secondary (optional):**  
Top-10% next-day performer  
*(robustness check only)*

---

### 4. Models and Methods
- Logistic Regression  
- Random Forest  
- XGBoost  
- Rolling **TimeSeriesSplit** for training/validation  

---

### 5. Evaluation Metrics
- **Precision@K**
- **ROC-AUC**
- **PR-AUC**
- **Accuracy**
- Feature importance & SHAP values

---

## Expected Challenges
- Noisy and non-stationary financial data  
- Matching low-frequency fundamentals with daily prices  
- Maintaining correct historical index membership  
- Avoiding data leakage  
- Class imbalance (Top-10% variant)

---

## Success Criteria
- Models outperform naive baselines  
- Clean, reproducible feature engineering pipeline  
- Correct time-series ML methodology  
- Interpretability using SHAP or feature importance  

---

## Stretch Goals
- Regime-based analysis (bull vs bear)  
- Multi-day prediction horizon  
- Integration of macroeconomic variables  

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
