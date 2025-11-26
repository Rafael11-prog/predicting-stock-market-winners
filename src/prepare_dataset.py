# src/prepare_dataset.py

import pandas as pd

def load_panel():
    df = pd.read_parquet("data/sp500_panel.parquet")
    print("Panel chargé :", df.shape)
    return df


def clean_panel(df):
    # 1. supprimer colonnes '_y'
    df = df[[col for col in df.columns if not col.endswith("_y")]]

    # 2. trier par ticker + date
    df = df.sort_values(["Ticker", "Date"])

    # 3. forward-fill des fondamentaux par ticker
    fund_cols = [c for c in df.columns if c not in ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df[fund_cols] = df.groupby("Ticker")[fund_cols].ffill()

    print("Panel nettoyé :", df.shape)
    return df


def save_clean(df):
    df.to_parquet("data/sp500_clean_panel.parquet")
    print("✔️ Sauvegardé : data/sp500_clean_panel.parquet")


if __name__ == "__main__":
    df = load_panel()
    df = clean_panel(df)
    save_clean(df)
