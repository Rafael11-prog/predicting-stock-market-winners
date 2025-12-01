"""
Dataset cleaning utilities for the S&P 500 project.

This module takes the merged raw panel (prices + PIT fundamentals) and:
1. Removes duplicated suffix columns (e.g., *_y from merges)
2. Sorts observations by (Ticker, Date)
3. Forward-fills missing fundamentals within each Ticker
4. Saves a cleaned version of the panel for downstream pipelines
"""

from pathlib import Path
from typing import List

import pandas as pd

PANEL_IN = Path("data/sp500_panel.parquet")
PANEL_OUT = Path("data/sp500_clean_panel.parquet")


# -------------------------------------------------------------------
# Load raw merged panel
# -------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """
    Load the merged panel containing prices + fundamentals.

    Returns
    -------
    pd.DataFrame
        Full raw panel before cleaning.
    """
    df = pd.read_parquet(PANEL_IN)
    print(f"Panel loaded: {df.shape}")
    return df


# -------------------------------------------------------------------
# Clean dataset
# -------------------------------------------------------------------
def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw panel by:
    - removing duplicate columns (e.g., *_y)
    - sorting the dataset chronologically per Ticker
    - forward-filling fundamental data within each Ticker

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel built from price data and PIT fundamentals.

    Returns
    -------
    pd.DataFrame
        Cleaned and chronologically consistent panel.
    """

    # ---------------------------------------------------------------
    # 1) Remove merge duplicates: drop all columns ending with "_y"
    # ---------------------------------------------------------------
    df = df[[col for col in df.columns if not col.endswith("_y")]]

    # ---------------------------------------------------------------
    # 2) Sort by (Ticker, Date)
    # ---------------------------------------------------------------
    df = df.sort_values(["Ticker", "Date"])

    # ---------------------------------------------------------------
    # 3) Forward-fill fundamentals within each Ticker
    #    We do NOT fill price columns (Open, High, Low, Close, etc.)
    # ---------------------------------------------------------------
    price_cols = {
        "Date", "Ticker", "Open", "High", "Low",
        "Close", "Adj Close", "Volume"
    }

    fund_cols: List[str] = [c for c in df.columns if c not in price_cols]

    # Forward-fill fundamental variables
    df[fund_cols] = df.groupby("Ticker")[fund_cols].ffill()

    print(f"Panel cleaned: {df.shape}")
    return df


# -------------------------------------------------------------------
# Save cleaned panel
# -------------------------------------------------------------------
def save_clean(df: pd.DataFrame) -> None:
    """
    Save the cleaned panel to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned panel.
    """
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_OUT, index=False)
    print(f"✔️ Saved cleaned panel → {PANEL_OUT}")


# -------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = load_panel()
    df = clean_panel(df)
    save_clean(df)
