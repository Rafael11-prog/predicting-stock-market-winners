# PROPOSAL.md

## Project Title  
**Global Market Dashboard – Real-Time Analysis and Visualization of Financial Markets**

## Category  
📊 *Data Analysis & Visualization*  
📈 *Business & Finance Tools*

---

## Problem Statement / Motivation  

Financial information is highly fragmented across different sources, APIs, and news platforms.  
As a result, understanding the real-time performance of global markets (such as the S&P 500, CAC 40, and NASDAQ) requires switching between multiple websites and manually interpreting raw data.  

This project aims to create a **Python-based financial dashboard** that provides a clean, consolidated, and interactive view of major stock indices and their top-performing assets.  
It will enable users — students, analysts, or investors — to quickly visualize market trends, identify outperformers, and monitor changes throughout the day.  

The motivation comes from a genuine interest in finance and technology: combining data analysis, programming, and financial understanding to build a tool that mirrors professional market terminals (like Bloomberg or Yahoo Finance) but remains lightweight and educational.

---

## Planned Approach and Technologies  

The system will consist of a modular Python application with the following components:

- **Data Collection:**  
  Fetch live and historical stock data from public APIs (such as YahooFinance or Alpha Vantage) using `yfinance` and `requests`.  

- **Data Processing:**  
  Clean, aggregate, and compute financial indicators with `pandas` and `numpy` (daily returns, volatility, and ranking of top gainers/losers).  

- **Visualization:**  
  Create dynamic visualizations using `matplotlib`, `seaborn`, and `plotly` for interactive charts (price evolution, distribution, and heatmaps).  

- **Dashboard Interface:**  
  Build an intuitive user interface using `streamlit`, allowing real-time exploration of markets, index summaries, and stock-specific pages.  

- **Code Quality & Testing:**  
  Follow PEP8 standards, type hints, and use `pytest` for unit and integration tests.  
  Implement caching and error handling to ensure stable performance under API rate limits.

The repository will include complete documentation, examples, and a test suite for reproducibility.

---

## Expected Challenges  

- API rate limits and incomplete data handling.  
- Maintaining near real-time updates without excessive API calls.  
- Designing efficient data structures for fast analysis and visualization.  
- Keeping the dashboard responsive and user-friendly while ensuring modular and testable code.  

---

## Success Criteria  

- The dashboard successfully displays up-to-date summaries for at least two indices (S&P 500 and CAC 40).  
- Users can view top gainers/losers and visualize price evolution for selected stocks.  
- All data manipulation and visualization logic are implemented in Python.  
- The code passes all tests, follows PEP8 standards, and is well documented.  

---

## Stretch Goals  

- Add a **correlation heatmap** between major stocks or sectors.  
- Implement **portfolio tracking** and performance comparison over time.  
- Introduce **WebSocket streaming** for true live price updates.  
- Deploy the dashboard online (e.g., via Streamlit Cloud or Docker).  

---

**Estimated Length:** ~1,000+ lines of Python code  
**Language:** Python 3.10+  
**Main Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `yfinance`, `pytest`

---
