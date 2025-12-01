from pathlib import Path
from typing import List

import pandas as pd

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PANEL_FILE = Path("data/sp500_clean_panel.parquet")
FUND_FILE = Path("data/raw/constituents-financials_csv.csv")
OUT_FILE = Path("data/sp500_panel_with_fundamentals.parquet")


def load_panel() -> pd.DataFrame:
    """
    Load the cleaned price panel (one row per Ticker/Date).

    Returns
    -------
    df : pd.DataFrame
        Cleaned price panel with OHLCV prices.
    """
    df = pd.read_parquet(PANEL_FILE)
    print("Price panel loaded:", df.shape)
    return df


def load_fundamentals() -> pd.DataFrame:
    """
    Load static fundamentals (Kaggle file) and clean numeric columns.

    We:
    - rename 'Symbol' to 'Ticker' to match the price panel,
    - keep only a subset of useful columns,
    - strip '$' and ',' from numeric fields and convert to floats,
    - keep a single row per Ticker.
    """
    df = pd.read_csv(FUND_FILE)

    # Match column name with panel
    df = df.rename(columns={"Symbol": "Ticker"})

    cols_keep: List[str] = [
        "Ticker",
        "Sector",
        "Price/Earnings",
        "Dividend Yield",
        "Earnings/Share",
        "52 Week Low",
        "52 Week High",
        "Market Cap",
        "EBITDA",
        "Price/Sales",
        "Price/Book",
    ]
    df = df[cols_keep].copy()

    numeric_cols = [
        "Price/Earnings",
        "Dividend Yield",
        "Earnings/Share",
        "52 Week Low",
        "52 Week High",
        "Market Cap",
        "EBITDA",
        "Price/Sales",
        "Price/Book",
    ]

    # Remove '$' and ',' then convert to numeric
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One row per ticker
    df = df.drop_duplicates(subset=["Ticker"])

    print("Fundamentals loaded:", df.shape)
    return df


def merge_panel_and_fundamentals() -> pd.DataFrame:
    """
    Merge the clean price panel with static fundamentals on Ticker.

    Result is saved to OUT_FILE and also returned.
    """
    panel = load_panel()
    fund = load_fundamentals()

    print("🔗 Merging price panel + fundamentals...")
    merged = panel.merge(fund, on="Ticker", how="left")

    print("Merged panel shape:", merged.shape)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE, index=False)
    print("✔️ Saved merged panel to:", OUT_FILE)

    return merged


if __name__ == "__main__":
    merge_panel_and_fundamentals()
