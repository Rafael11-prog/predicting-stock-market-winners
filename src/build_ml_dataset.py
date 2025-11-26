from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

PANEL_FILE = Path("data/sp500_panel_with_fundamentals.parquet")
OUT_DATA_FILE = Path("data/sp500_ml_dataset.parquet")


def load_panel():
    df = pd.read_parquet(PANEL_FILE)
    print("Panel avec fondamentaux :", df.shape)

    # Filtre période (à adapter si tu veux)
    df = df[(df["Date"] >= "2018-01-01") & (df["Date"] <= "2024-12-31")].copy()
    df = df.sort_values(["Ticker", "Date"])
    print("Panel filtré :", df.shape)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée quatre cibles de classification :

    - target_1d_top50 : ret_1d_ahead > médiane cross-sectionnelle (top 50%)
    - target_5d_top50 : ret_5d_ahead > médiane cross-sectionnelle (top 50%)

    - target_1d_top10 : ret_1d_ahead > 90e percentile cross-sectionnel (top 10%)
    - target_5d_top10 : ret_5d_ahead > 90e percentile cross-sectionnel (top 10%)
    """

    # ---------- Rendement 1 jour à l'avance ----------
    df["ret_1d_ahead"] = (
        df.groupby("Ticker")["Adj Close"]
          .pct_change(1)
          .shift(-1)
    )
    df["ret_1d_ahead"] = df["ret_1d_ahead"].replace([np.inf, -np.inf], np.nan)

    # Top 50% (médiane par date)
    median_1d = df.groupby("Date")["ret_1d_ahead"].transform("median")
    df["target_1d_top50"] = (df["ret_1d_ahead"] > median_1d).astype(int)

    # Top 10% (90e percentile par date)
    q90_1d = df.groupby("Date")["ret_1d_ahead"].transform(
        lambda s: s.quantile(0.90)
    )
    df["target_1d_top10"] = (df["ret_1d_ahead"] > q90_1d).astype(int)

    # ---------- Rendement 5 jours à l'avance ----------
    df["ret_5d_ahead"] = (
        df.groupby("Ticker")["Adj Close"]
          .pct_change(5)
          .shift(-5)
    )
    df["ret_5d_ahead"] = df["ret_5d_ahead"].replace([np.inf, -np.inf], np.nan)

    # Top 50% (médiane par date)
    median_5d = df.groupby("Date")["ret_5d_ahead"].transform("median")
    df["target_5d_top50"] = (df["ret_5d_ahead"] > median_5d).astype(int)

    # Top 10% (90e percentile par date)
    q90_5d = df.groupby("Date")["ret_5d_ahead"].transform(
        lambda s: s.quantile(0.90)
    )
    df["target_5d_top10"] = (df["ret_5d_ahead"] > q90_5d).astype(int)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features basées sur les prix :
    - rendements (1d, 5d, 20d)
    - volatilité rolling (20d, 60d)
    - moyennes mobiles / position vs MA
    - EMA (10, 20, 50)
    - RSI(14)
    - MACD (12-26-9)
    - volume z-score
    """

    g = df.groupby("Ticker")

    # ---------- rendements ----------
    df["ret_1d"] = g["Adj Close"].pct_change(1)
    df["ret_5d"] = g["Adj Close"].pct_change(5)
    df["ret_20d"] = g["Adj Close"].pct_change(20)

    # ---------- volatilité (rolling std des ret_1d) ----------
    df["vol_20d"] = (
        g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["vol_60d"] = (
        g["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    )

    # ---------- moyennes mobiles simples ----------
    ma10 = g["Adj Close"].rolling(10).mean().reset_index(level=0, drop=True)
    ma50 = g["Adj Close"].rolling(50).mean().reset_index(level=0, drop=True)
    df["ma10"] = ma10
    df["ma50"] = ma50
    df["price_vs_ma10"] = df["Adj Close"] / ma10
    df["price_vs_ma50"] = df["Adj Close"] / ma50

    # ---------- EMA (10 / 20 / 50) ----------
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

    # ---------- volume normalisé (z-score par titre) ----------
    vol_mean = g["Volume "].transform("mean")
    vol_std = g["Volume "].transform("std")
    df["volume_z"] = (df["Volume "] - vol_mean) / vol_std

    return df

def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features fondamentales :
    - size (log Market Cap)
    - PE, PB, PS, Dividend Yield
    - EBITDA / MarketCap
    - position dans le range 52 semaines
    - earnings_yield, book_to_price, sales_to_price
    - proxies de ROE, Debt/Equity, EPS growth
    - z-scores sectoriels
    - rangs globaux
    """

    # ---------- colonnes de base ----------
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

    # ---------- signaux "value" ----------
    df["earnings_yield"] = 1.0 / df["pe"]
    df["book_to_price"] = 1.0 / df["pb"]
    df["sales_to_price"] = 1.0 / df["ps"]

    for col in ["earnings_yield", "book_to_price", "sales_to_price"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # ---------- proxies ROE / Debt-to-Equity / EPS growth ----------
    df["eps"] = df["Earnings/Share"]

    # book value per share approximé via PB
    df["book_value_per_share"] = df["Adj Close"] / df["pb"]
    df["roe_proxy"] = df["eps"] / df["book_value_per_share"]

    # proxy simple de D/E : quand PB est largement >1, on suppose plus de levier
    df["debt_to_equity_proxy"] = (df["pb"] - 1).clip(lower=0)

    # proxy de croissance du bénéfice par action (EPS growth)
    df["eps_growth_proxy"] = (
        df.groupby("Ticker")["eps"].pct_change().replace([np.inf, -np.inf], np.nan)
    )

    # ---------- z-scores par secteur ----------
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

    # ---------- rangs globaux ----------
    df["mktcap_rank"] = df["log_mktcap"].rank(method="average")
    df["pe_rank"] = df["pe"].rank(ascending=True, method="average")   # low PE = value
    df["pb_rank"] = df["pb"].rank(ascending=True, method="average")
    df["value_rank"] = df["earnings_yield"].rank(ascending=False, method="average")

    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les effets calendaires :
    - day_of_week (0=lundi, ..., 4=vendredi)
    - month (1-12)
    - dummies fin de mois / fin de trimestre
    """

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)

    return df



def build_dataset() -> pd.DataFrame:
    """Construit le dataset ML (features + targets 1d et 5d)."""

    df = load_panel()

    df = add_target(df)
    df = add_price_features(df)
    df = add_fundamental_features(df)
    df = add_calendar_features(df)

    # -------- mêmes features qu'avant --------
    feature_cols = [
        # Prix / technique
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "vol_60d",
        "price_vs_ma10", "price_vs_ma50",
        "ema10", "ema20", "ema50",
        "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "volume_z",

        # Fondamentaux simples
        "log_mktcap", "pe", "pb", "ps", "div_yield",
        "ebitda_to_mktcap", "price_pos_in_52w",

        # Fondamentaux dérivés / value
        "earnings_yield", "book_to_price", "sales_to_price",
        "roe_proxy", "debt_to_equity_proxy", "eps_growth_proxy",
        "log_mktcap_sector_z", "pe_sector_z", "pb_sector_z",
        "ps_sector_z", "ebitda_mktcap_sector_z",
        "mktcap_rank", "pe_rank", "pb_rank", "value_rank",

        # Effets calendaires
        "day_of_week", "month", "is_month_end", "is_quarter_end",
    ]

    # Colonnes finales à garder
    all_cols = [
        "Date", "Ticker",
        "ret_1d_ahead", "target_1d_top50", "target_1d_top10",
        "ret_5d_ahead", "target_5d_top50", "target_5d_top10",
    ] + feature_cols


    df = df[all_cols].copy()

    # On enlève les lignes avec NaN dans les targets ou les features
    df = df.dropna(
        subset=[
            "ret_1d_ahead", "target_1d_top50", "target_1d_top10",
            "ret_5d_ahead", "target_5d_top50", "target_5d_top10",
        ] + feature_cols
    )


    OUT_DATA_FILE = Path("data/sp500_ml_dataset.parquet")
    OUT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DATA_FILE, index=False)

    print("✔️ Dataset ML sauvegardé :", OUT_DATA_FILE)
    print("Shape final :", df.shape)

    return df, feature_cols

if __name__ == "__main__":
    build_dataset()