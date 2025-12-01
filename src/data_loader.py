"""
Data loading & merging for the S&P 500 project.

This module:
- loads daily prices (sp500_data.csv)
- loads point-in-time fundamentals (sp500_pit_data.csv)
- cleans both datasets
- merges them into a single panel and saves it as a parquet file
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data")

PRICE_FILE = DATA_RAW / "sp500_data.csv"
FUND_FILE = DATA_RAW / "sp500_pit_data.csv"
PANEL_FILE = DATA_PROCESSED / "sp500_panel.parquet"


# -------------------------------------------------------------------
# Loading price data
# -------------------------------------------------------------------
def load_price_data(
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load daily price data, filter by date range and drop duplicates.

    Parameters
    ----------
    start_date : str
        First date to keep (YYYY-MM-DD).
    end_date : str
        Last date to keep (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Cleaned daily price data sorted by (Ticker, Date).
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["Date"])

    # Filter by date interval
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df = df.loc[mask].copy()

    # Sort & remove duplicates (same Ticker/Date)
    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    return df


# -------------------------------------------------------------------
# Loading point-in-time fundamentals
# -------------------------------------------------------------------
def load_fundamental_data() -> pd.DataFrame:
    """
    Load point-in-time fundamentals and keep only truly fundamental columns.

    Steps
    -----
    1. Read raw PIT file and sort by (Ticker, Date).
    2. Remove duplicated (Ticker, Date) rows.
    3. Keep only non-price columns as "fundamentals".
    4. Forward-fill fundamentals through time within each Ticker.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Ticker, <fundamental features>,
        forward-filled over time.
    """
    df = pd.read_csv(
        FUND_FILE,
        parse_dates=["Date"],
        on_bad_lines="skip",  # ignore malformed lines in the raw CSV
    )

    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    # Price-like columns that we do NOT treat as fundamentals
    price_like_cols = {
        "Date",
        "Ticker",
        "Adj Close",
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
    }

    # Everything that is not price-like is considered a fundamental variable
    fundamental_cols = [c for c in df.columns if c not in price_like_cols]

    # Keep Date, Ticker + fundamental columns
    df = df[["Date", "Ticker"] + fundamental_cols].copy()

    # Forward-fill fundamentals through time for each Ticker
    df = (
        df.set_index(["Ticker", "Date"])
        .groupby(level=0)
        .ffill()
        .reset_index()
    )

    return df


# -------------------------------------------------------------------
# Build the combined panel
# -------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    """
    Build the full price + fundamentals panel and save it as parquet.

    The merge is done as a left join on (Ticker, Date): we keep all
    price observations even if some fundamentals are missing.

    Returns
    -------
    pd.DataFrame
        Combined panel with prices and forward-filled fundamentals.
    """
    print("📥 Loading daily prices...")
    prices = load_price_data()
    print("   Prices shape:", prices.shape)

    print("📥 Loading point-in-time fundamentals...")
    funds = load_fundamental_data()
    print("   Fundamentals shape:", funds.shape)

    print("🔗 Merging prices and fundamentals on (Ticker, Date)...")
    panel = pd.merge(
        prices,
        funds,
        on=["Ticker", "Date"],
        how="left",  # keep all price rows even if fundamentals are missing
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_FILE, index=False)

    print("✅ Panel saved to:", PANEL_FILE)
    print("   Panel shape:", panel.shape)
    print(panel.head())

    return panel


if __name__ == "__main__":
    build_panel()
