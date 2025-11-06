# PROPOSAL.md

## Project Title  
**Global Market Intelligence – Forecasting, Classification, and Risk Estimation on Equity Indices**

## Category  
📊 *Data Analysis & Visualization*  
🧠 *Statistical & Machine Learning Forecasting*  
📈 *Business & Finance Tools*

---

## Problem Statement / Motivation  

Financial markets are highly dynamic, and predicting short-term movements remains one of the biggest challenges in data science.  
The goal of this project is to build a **data-driven framework** capable of analyzing and forecasting the short-term performance and risk of major equity indices such as the **S&P 500** and **CAC 40**.  

More specifically, this project aims to:
1. **Forecast** 1-day and 5-day index returns using time series models (ARIMA, ETS) and benchmark them against naïve models.  
2. **Classify** stocks that are likely to be among the **top performers** the next day using historical features such as momentum, volatility, and volume anomalies.  
3. **Estimate risk** via a one-day **Value-at-Risk (VaR)** using different approaches (Historical Simulation, Variance-Covariance, Cornish-Fisher expansion) and validate them with **backtesting**.  

An interactive **Streamlit dashboard** will serve as a front-end interface to visualize results, forecasts, classification outcomes, and risk analytics in an intuitive way.

---

## Planned Approach and Technologies  

**Data Acquisition & Preparation:**  
- Download historical prices and volumes for S&P 500 and CAC 40 constituents using `yfinance`.  
- Clean and align data with `pandas` and handle missing values, splits, and outliers.

**Feature Engineering:**  
- Compute log returns, realized volatility, 5-day and 20-day momentum, normalized trading volume, SMA/EMA gaps, and calendar dummies (day of week, end-of-month).  
- Label future performance (Top-Decile returns) for supervised learning.

**Modeling Components:**  
- **Forecasting:** Use `statsmodels` ARIMA/ETS models and evaluate against a random-walk baseline.  
- **Classification:** Use `scikit-learn` models (Logistic Regression, Random Forest, Gradient Boosting) with walk-forward validation (`TimeSeriesSplit`).  
- **Risk Estimation:** Compute 1-day VaR using Historical, Variance-Covariance, and Cornish-Fisher methods, and assess accuracy through backtesting (exception rate vs. confidence level).

**Evaluation Metrics:**  
- Forecasting → RMSE, MAE, MAPE  
- Classification → Precision@K, ROC-AUC, PR-AUC, confusion matrix  
- Risk → Coverage rate (exceptions close to α), stability over time  

**Visualization / Dashboard:**  
- Implement a **Streamlit** dashboard with three main tabs:  
  1. *Forecasts* – ARIMA/ETS predictions with fan charts and performance table.  
  2. *Stock Selection* – Predicted vs. realized Top-K performers and Precision@K visualization.  
  3. *Risk Analytics* – VaR computation and backtesting results.  

---

## Expected Challenges  

- Preventing **data leakage** and maintaining strict chronological separation.  
- Managing **imbalanced labels** (few top performers).  
- Ensuring **robustness** of time series models in volatile periods.  
- Optimizing dashboard responsiveness and caching data efficiently.  

---

## Success Criteria  

- Forecasting models outperform the naïve baseline on MAE/RMSE.  
- Classifiers identify top movers with significantly better Precision@K than random selection.  
- VaR models maintain exception rates close to nominal α levels (e.g., 1% or 5%).  
- All code is modular, tested, and documented (PEP8, docstrings, pytest).  
- Interactive dashboard cleanly presents model outputs and insights.  

---

## Stretch Goals  

- Regime classification with volatility-based clustering (K-means).  
- Rolling CAPM estimation (dynamic betas) as an additional feature.  
- Backtest of a simple “Top-K strategy” with transaction costs.  
- Deployment on Streamlit Cloud or Docker for live demonstration.  

---

**Language:** Python 3.10+  
**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `yfinance`, `statsmodels`, `scikit-learn`, `scipy`, `pytest`, `streamlit`  
**Estimated Code Length:** ~1,000+ lines  
**Deliverables:** Forecasting, Classification, and Risk modules + Streamlit Dashboard  
