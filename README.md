# S&P 500 Predictive ML Project  
### *Cross-Sectional Stock Selection with Machine Learning*

---

## 📌 Overview

This project implements an **end-to-end machine-learning pipeline** to predict **which S&P 500 stocks are likely to outperform their peers cross-sectionally** over short horizons.

Rather than predicting absolute index returns, the task is framed as a **cross-sectional classification problem**:  
on each trading day, predict whether a stock belongs to the **top X% of forward returns among S&P 500 constituents**.

The pipeline combines:

- Daily S&P 500 price and volume data  
- **Point-in-time fundamentals** (time-indexed, forward-filled)  
- **Static cross-sectional fundamentals** (Kaggle snapshot)  
- Technical indicators and calendar effects  
- Time-series-valid cross-validation (`TimeSeriesSplit`)  

All stages are automated and reproducible via `main.py`.

---

## 📁 Pipeline Architecture

The pipeline is orchestrated by `main.py` and executed sequentially as follows:

Raw Prices & PIT Fundamentals
↓
Merged Panel (Ticker–Date)
↓
Cleaned Panel
↓
Merge Static Fundamentals
↓
ML Dataset (Features + Targets)
↓
Model Training & Evaluation
↓
Results (CSV + Plots)

### Pipeline Stages

1. **Build raw panel**  
   Merge daily OHLCV prices with point-in-time fundamentals  
   → `data/sp500_panel.parquet`

2. **Clean panel**  
   Remove merge artefacts, sort chronologically, forward-fill fundamentals  
   → `data/sp500_clean_panel.parquet`

3. **Merge static fundamentals**  
   Join cross-sectional accounting variables from Kaggle  
   → `data/sp500_panel_with_fundamentals.parquet`

4. **Build ML dataset**  
   Engineer features and construct classification targets  
   → `data/sp500_ml_dataset.parquet`

5. **Train & evaluate models**  
   Time-series cross-validation, metrics, plots, and backtests  
   → `results/`

---

## ▶️ How to Run

Install dependencies:

```bash```
pip install -r requirements.txt

Run the **full pipeline** (main resulats):

python main.py 

- Runs the entire pipeline
- Evaluates all four classification tasks
- Produces the main results used in the final report 
  (2-4 hours to run depend on the hardware)

Run only **a signle target(Top5'%)**

python main.py --top50

- Evaluates **only the next day Top 50% target** 
(1 hour to run)

Run **FAST development**

python main.py --fast

- just to check if the code is working
- 1 target with 2 folds only 
(10 minutes to run)
---

## 🎯 Objective

Predict whether a stock will be in the:

- **Top 50%** cross-section next day (main target)  
- **Top 10%** (robustness)  
- For **1-day** and **5-day** return horizons  

---

## 🧠 Models Implemented

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier with scaling |
| **Random Forest** | Non-linear model with feature importance |
| **XGBoost** | Gradient boosted model (optional in extended RUN) |
| **Random baseline** | Random probabilities |
| **Momentum baseline** | Predict based on yesterday’s return |
| **Buy & Hold baseline** | Always predict class 1 |

Training uses **TimeSeriesSplit**, ensuring no data leakage.

---

## 🧮 Features Used

### Technical Indicators
- Daily / Weekly returns  
- Rolling volatility  
- SMA, EMA  ratios
- RSI, MACD  
- Volume indicators  
- Calendar effects  

### Fundamental Indicators
- P/E  
- P/B  
- ROE  
- Debt-to-Equity  
- EPS growth  
- Profit margins  
- Market cap (log)

Total: ~40–50 engineered features.

---

## 🎯 Evaluation Metrics

- **Accuracy**  
- **ROC AUC**  
- **PR AUC**  
- **Precision@K** (10%, 20%)
- Confusion matrices   
- Permutation importance  
- Learning curves  

Outputs are saved in the `results/` directory.


## 📦 Project Structure

predicting-stock-market-winners/
│
├── main.py
├── scripts/
│ └── build_sp500_prices.py
├── src/
│ ├── data_loader.py
│ ├── prepare_dataset.py
│ ├── add_fundamentals.py
│ ├── build_ml_dataset.py
│ └── train_models.py
├── data/
├── results/ (ignored)
├── PROPOSAL.md
├── README.md
└── requirements.txt

---
## 👤 Author

**Rafael Machado Cerqueira**  
HEC Lausanne 
