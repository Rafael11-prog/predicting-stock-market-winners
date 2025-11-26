from pathlib import Path
import pandas as pd

PANEL_FILE = Path("data/sp500_clean_panel.parquet")
FUND_FILE = Path("data/raw/constituents-financials_csv.csv")
OUT_FILE = Path("data/sp500_panel_with_fundamentals.parquet")


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_FILE)
    print("Panel prix chargé :", df.shape)
    return df

def load_fundamentals() -> pd.DataFrame:
    df = pd.read_csv(FUND_FILE)

    # On renomme 'Symbol' en 'Ticker' pour matcher le panel
    df = df.rename(columns={"Symbol": "Ticker"})

    # On garde les colonnes utiles
    cols_keep = [
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

    # Nettoyage numérique (remplacer les virgules, signes, etc.)
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

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Un seul enregistrement par ticker
    df = df.drop_duplicates(subset=["Ticker"])
    print("Fondamentaux chargés :", df.shape)
    return df


def merge_panel_and_fundamentals() -> pd.DataFrame:
    panel = load_panel()
    fund = load_fundamentals()

    print("🔗 Fusion panel + fondamentaux…")
    merged = panel.merge(fund, on="Ticker", how="left")

    print("Panel fusionné :", merged.shape)
    merged.to_parquet(OUT_FILE, index=False)
    print("✔️ Sauvegardé dans :", OUT_FILE)

    return merged


if __name__ == "__main__":
    merge_panel_and_fundamentals()
