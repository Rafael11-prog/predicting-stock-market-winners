from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PANEL_FILE = Path("data/sp500_panel_with_fundamentals.parquet")
OUT_DATA_FILE = Path("data/sp500_ml_dataset.parquet")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """
    Load the price + fundamentals panel and apply a date filter.

    Returns
    -------
    pd.DataFrame
        Panel sorted by (Ticker, Date) and restricted to the desired period.
    """
    df = pd.read_parquet(PANEL_FILE)
    print(f"[INFO] Panel with fundamentals loaded: {df.shape}")

    # Date filter (can be adapted if needed)
    df = df[(df["Date"] >= "2018-01-01") & (df["Date"] <= "2024-12-31")].copy()
    df = df.sort_values(["Ticker", "Date"])
    print(f"[INFO] Filtered panel: {df.shape}")
    return df


# -------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four classification targets based on forward returns.

    For each date, we build cross-sectional labels:
    - target_1d_top50 : ret_1d_ahead > cross-sectional median  (top 50%)
    - target_5d_top50 : ret_5d_ahead > cross-sectional median  (top 50%)
    - target_1d_top10 : ret_1d_ahead > cross-sectional 90th pct (top 10%)
    - target_5d_top10 : ret_5d_ahead > cross-sectional 90th pct (top 10%)

    Returns
    -------
    pd.DataFrame
        Input dataframe with new target and forward-return columns.
    """

    # ---------- 1-day ahead return ----------
    df["ret_1d_ahead"] = (
        df.groupby("Ticker")["Adj Close"]
        .pct_change(1)
        .shift(-1)
    )
    df["ret_1d_ahead"] = df["ret_1d_ahead"].replace([np.inf, -np.inf], np.nan)

    # Top 50% (median by date)
    median_1d = df.groupby("Date")["ret_1d_ahead"].transform("median")
    df["target_1d_top50"] = (df["ret_1d_ahead"] > median_1d).astype(int)

    # Top 10% (90th percentile by date)
    q90_1d = df.groupby("Date")["ret_1d_ahead"].transform(
        lambda s: s.quantile(0.90)
    )
    df["target_1d_top10"] = (df["ret_1d_ahead"] > q90_1d).astype(int)

    # ---------- 5-day ahead return ----------
    df["ret_5d_ahead"] = (
        df.groupby("Ticker")["Adj Close"]
        .pct_change(5)
        .shift(-5)
    )
    df["ret_5d_ahead"] = df["ret_5d_ahead"].replace([np.inf, -np.inf], np.nan)

    # Top 50% (median by date)
    median_5d = df.groupby("Date")["ret_5d_ahead"].transform("median")
    df["target_5d_top50"] = (df["ret_5d_ahead"] > median_5d).astype(int)

    # Top 10% (90th percentile by date)
    q90_5d = df.groupby("Date")["ret_5d_ahead"].transform(
        lambda s: s.quantile(0.90)
    )
    df["target_5d_top10"] = (df["ret_5d_ahead"] > q90_5d).astype(int)

    return df


# -------------------------------------------------------------------
# Price-based features
# -------------------------------------------------------------------
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based technical features:

    - Returns: 1d, 5d, 20d
    - Rolling volatility (20d, 60d) on 1d returns
    - Simple moving averages (10, 50) and price / MA ratios
    - Exponential moving averages (10, 20, 50)
    - RSI(14)
    - MACD (12-26-9)
    - Volume z-score per ticker
    """
    g = df.groupby("Ticker")

    # ---------- Returns ----------
    df["ret_1d"] = g["Adj Close"].pct_change(1)
    df["ret_5d"] = g["Adj Close"].pct_change(5)
    df["ret_20d"] = g["Adj Close"].pct_change(20)

    # ---------- Volatility (rolling std of 1d returns) ----------
    df["vol_20d"] = (
        g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["vol_60d"] = (
        g["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    )

    # ---------- Simple moving averages ----------
    ma10 = g["Adj Close"].rolling(10).mean().reset_index(level=0, drop=True)
    ma50 = g["Adj Close"].rolling(50).mean().reset_index(level=0, drop=True)
    df["ma10"] = ma10
    df["ma50"] = ma50
    df["price_vs_ma10"] = df["Adj Close"] / ma10
    df["price_vs_ma50"] = df["Adj Close"] / ma50

    # ---------- Exponential moving averages ----------
    df["ema10"] = g["Adj Close"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean()
    )
    df["ema20"] = g["Adj Close"].transform(
        lambda s: s.ewm(span=20, adjust=False).mean()
    )
    df["ema50"] = g["Adj Close"].transform(
        lambda s: s.ewm(span=50, adjust=False).mean()
    )

    # ---------- RSI(14) ----------
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val

    df["rsi_14"] = (
        df.groupby("Ticker")["Adj Close"]
        .apply(lambda s: rsi(s, window=14))
        .reset_index(level=0, drop=True)
    )

    # ---------- MACD (12-26-9) ----------
    ema12 = g["Adj Close"].transform(
        lambda s: s.ewm(span=12, adjust=False).mean()
    )
    ema26 = g["Adj Close"].transform(
        lambda s: s.ewm(span=26, adjust=False).mean()
    )
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = (
        df.groupby("Ticker")["macd_line"]
        .transform(lambda s: s.ewm(span=9, adjust=False).mean())
    )
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ---------- Volume z-score ----------
    # Note: column name is "Volume " (with a trailing space) in the raw data.
    vol_mean = g["Volume "].transform("mean")
    vol_std = g["Volume "].transform("std")
    df["volume_z"] = (df["Volume "] - vol_mean) / vol_std

    return df


# -------------------------------------------------------------------
# Fundamental features
# -------------------------------------------------------------------
def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fundamental features:

    - Size (log market cap)
    - PE, PB, PS, Dividend Yield
    - EBITDA / Market Cap
    - 52-week range position
    - Value signals: earnings_yield, book_to_price, sales_to_price
    - Proxies for ROE, Debt/Equity, EPS growth
    - Sector-level z-scores
    - Global ranks (size/value style signals)
    """

    # ---------- Basic columns ----------
    df["log_mktcap"] = np.log(df["Market Cap"].replace({0: np.nan}))

    df["pe"] = df["Price/Earnings"]
    df["pb"] = df["Price/Book"]
    df["ps"] = df["Price/Sales"]
    df["div_yield"] = df["Dividend Yield"]

    df["ebitda_to_mktcap"] = df["EBITDA"] / df["Market Cap"]

    df["52w_range"] = df["52 Week High"] - df["52 Week Low"]
    df["price_pos_in_52w"] = (
        (df["Adj Close"] - df["52 Week Low"]) / df["52w_range"]
    )

    # ---------- Value signals ----------
    df["earnings_yield"] = 1.0 / df["pe"]
    df["book_to_price"] = 1.0 / df["pb"]
    df["sales_to_price"] = 1.0 / df["ps"]

    for col in ["earnings_yield", "book_to_price", "sales_to_price"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # ---------- Proxies for ROE / D/E / EPS growth ----------
    df["eps"] = df["Earnings/Share"]

    # Approximate book value per share via PB
    df["book_value_per_share"] = df["Adj Close"] / df["pb"]
    df["roe_proxy"] = df["eps"] / df["book_value_per_share"]

    # Very crude proxy for D/E: large PB often means more leverage
    df["debt_to_equity_proxy"] = (df["pb"] - 1).clip(lower=0)

    # EPS growth proxy (time-series, per ticker)
    df["eps_growth_proxy"] = (
        df.groupby("Ticker")["eps"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    # ---------- Sector-level z-scores ----------
    sector_group = df.groupby("Sector")

    def sector_z(col_name: str) -> pd.Series:
        col = df[col_name]
        mean = sector_group[col_name].transform("mean")
        std = sector_group[col_name].transform("std")
        return (col - mean) / std

    df["log_mktcap_sector_z"] = sector_z("log_mktcap")
    df["pe_sector_z"] = sector_z("pe")
    df["pb_sector_z"] = sector_z("pb")
    df["ps_sector_z"] = sector_z("ps")
    df["ebitda_mktcap_sector_z"] = sector_z("ebitda_to_mktcap")

    # ---------- Global ranks ----------
    df["mktcap_rank"] = df["log_mktcap"].rank(method="average")
    # low PE = more value-like
    df["pe_rank"] = df["pe"].rank(ascending=True, method="average")
    df["pb_rank"] = df["pb"].rank(ascending=True, method="average")
    df["value_rank"] = df["earnings_yield"].rank(
        ascending=False,
        method="average",
    )

    return df


# -------------------------------------------------------------------
# Calendar features
# -------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar / seasonality features:

    - day_of_week (0=Monday, ..., 4=Friday)
    - month (1-12)
    - month-end and quarter-end dummy variables
    """
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    return df


# -------------------------------------------------------------------
# Dataset builder
# -------------------------------------------------------------------
def build_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the final ML dataset (features + 1d/5d targets).

    Returns
    -------
    df : pd.DataFrame
        Cleaned ML dataset with targets and features.
    feature_cols : list of str
        Names of feature columns (used later in the models).
    """
    df = load_panel()

    df = add_target(df)
    df = add_price_features(df)
    df = add_fundamental_features(df)
    df = add_calendar_features(df)

    # --- Feature columns ---
    feature_cols: List[str] = [
        # Price / technical
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "vol_60d",
        "price_vs_ma10", "price_vs_ma50",
        "ema10", "ema20", "ema50",
        "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "volume_z",

        # Simple fundamentals
        "log_mktcap", "pe", "pb", "ps", "div_yield",
        "ebitda_to_mktcap", "price_pos_in_52w",

        # Derived / value-oriented fundamentals
        "earnings_yield", "book_to_price", "sales_to_price",
        "roe_proxy", "debt_to_equity_proxy", "eps_growth_proxy",
        "log_mktcap_sector_z", "pe_sector_z", "pb_sector_z",
        "ps_sector_z", "ebitda_mktcap_sector_z",
        "mktcap_rank", "pe_rank", "pb_rank", "value_rank",

        # Calendar effects
        "day_of_week", "month", "is_month_end", "is_quarter_end",
    ]

    # Columns to keep in the final ML dataset
    all_cols: List[str] = [
        "Date", "Ticker",
        "ret_1d_ahead", "target_1d_top50", "target_1d_top10",
        "ret_5d_ahead", "target_5d_top50", "target_5d_top10",
    ] + feature_cols

    df = df[all_cols].copy()

    # Drop rows with NaNs in targets or features
    df = df.dropna(
        subset=[
            "ret_1d_ahead", "target_1d_top50", "target_1d_top10",
            "ret_5d_ahead", "target_5d_top50", "target_5d_top10",
        ] + feature_cols
    )

    OUT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DATA_FILE, index=False)

    print(f"[INFO] ML dataset saved to: {OUT_DATA_FILE}")
    print(f"[INFO] Final shape: {df.shape}")

    return df, feature_cols


if __name__ == "__main__":
    build_dataset()
