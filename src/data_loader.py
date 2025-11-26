"""
Data loading & merging for S&P 500 project.

- Lit les prix journaliers (sp500_data.csv)
- Lit les données "point in time" (sp500_pit_data.csv)
- Nettoie un peu
- Fusionne en un seul panel et sauvegarde en parquet
"""

from pathlib import Path
import pandas as pd

# Répertoires et fichiers
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data")

PRICE_FILE = DATA_RAW / "sp500_data.csv"
FUND_FILE = DATA_RAW / "sp500_pit_data.csv"
PANEL_FILE = DATA_PROCESSED / "sp500_panel.parquet"


def load_price_data(
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Charge les prix quotidiens, filtre la période, enlève les doublons."""

    df = pd.read_csv(PRICE_FILE, parse_dates=["Date"])

    # Filtre sur la période (tu pourras ajuster les dates plus tard)
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df = df.loc[mask].copy()

    # Tri & suppression des doublons (même Date/Ticker)
    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    return df


def load_fundamental_data() -> pd.DataFrame:
    """
    Charge les données "point-in-time" et garde seulement les colonnes
    réellement fondamentales (tout ce qui n’est pas prix classique).
    Puis fait un forward-fill par Ticker.
    """

    df = pd.read_csv(
    FUND_FILE,
    parse_dates=["Date"],
    on_bad_lines="skip"  # ignore les lignes mal formées
)

    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    # Colonnes de prix qu’on ne considère pas comme "fundamentals"
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

    # Toutes les colonnes qui ne sont PAS des prix sont vues comme fondamentales
    fundamental_cols = [c for c in df.columns if c not in price_like_cols]

    # On garde Date, Ticker + fundamentals
    df = df[["Date", "Ticker"] + fundamental_cols].copy()

    # Forward-fill des fondamentaux dans le temps par Ticker
    df = (
        df.set_index(["Ticker", "Date"])
        .groupby(level=0)
        .ffill()
        .reset_index()
    )

    return df


def build_panel() -> pd.DataFrame:
    """Construit le gros panel prix + fondamentaux et le sauvegarde en parquet."""

    print("📥 Chargement des prix…")
    prices = load_price_data()
    print("   Prices shape:", prices.shape)

    print("📥 Chargement des fondamentaux…")
    funds = load_fundamental_data()
    print("   Fundamentals shape:", funds.shape)

    print("🔗 Fusion prix + fondamentaux…")
    panel = pd.merge(
        prices,
        funds,
        on=["Ticker", "Date"],
        how="left",  # on garde toutes les lignes de prix, même si pas de fondamentaux
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_FILE, index=False)

    print("✅ Panel sauvegardé dans:", PANEL_FILE)
    print("   Panel shape:", panel.shape)
    print(panel.head())

    return panel


if __name__ == "__main__":
    build_panel()
