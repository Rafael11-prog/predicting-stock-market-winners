"""
Script to prepare the raw S&P 500 price and point-in-time data.

This script is deliberately lightweight:

- It checks that the two key raw files used by the project exist:
    * data/raw/sp500_data.csv
    * data/raw/sp500_pit_data.csv

- If they are already present, it just prints a short summary (shape, head)
  so you can quickly verify that the data looks OK.

- If they are missing but a Kaggle archive is present
  (data/raw/archive.zip), it tries to extract the CSV files from there.

This keeps the project reproducible without hiding the fact that the
original data comes from an external source (Kaggle / yfinance).
"""

from __future__ import annotations

from pathlib import Path
import zipfile
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_RAW = Path("data/raw")
ARCHIVE = DATA_RAW / "archive.zip"

PRICE_FILE = DATA_RAW / "sp500_data.csv"
PIT_FILE = DATA_RAW / "sp500_pit_data.csv"


def _print_file_info(path: Path, name: str) -> None:
    """Print a small summary (shape + first lines) of a CSV file."""
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:  # pragma: no cover (defensive)
        print(f"⚠️ Could not read {name} at {path}: {exc}")
        return

    print(f"\n{name} found at: {path}")
    print(f"→ columns: {list(df.columns)}")
    print("→ first 5 rows:")
    print(df)


def _extract_from_archive() -> Optional[tuple[Path, Path]]:
    """
    Try to extract sp500_data.csv and sp500_pit_data.csv from archive.zip.

    Returns
    -------
    (price_path, pit_path) if successful, otherwise None.
    """
    if not ARCHIVE.exists():
        print(f"❌ archive.zip not found at {ARCHIVE}")
        print("   Please download the Kaggle archive and place it there.")
        return None

    print(f"📦 Found archive at: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        members = zf.namelist()
        print("   Files inside archive.zip:")
        for m in members:
            print("   -", m)

        # We assume the archive already contains the correctly named files.
        # If the names differ, you can adapt this logic.
        needed = {
            "sp500_data.csv": PRICE_FILE,
            "sp500_pit_data.csv": PIT_FILE,
        }

        for src_name, out_path in needed.items():
            matches = [m for m in members if m.endswith(src_name)]
            if not matches:
                print(f"❌ Could not find {src_name} inside archive.zip")
                return None

            member = matches[0]
            print(f"   → Extracting {member} → {out_path}")
            with zf.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())

    return PRICE_FILE, PIT_FILE


def ensure_raw_files() -> None:
    """
    Main helper:
    - if raw CSVs exist: print basic info
    - else: try to build them from archive.zip
    """
    price_exists = PRICE_FILE.exists()
    pit_exists = PIT_FILE.exists()

    if price_exists and pit_exists:
        print("✅ Raw files already present.")
        _print_file_info(PRICE_FILE, "Price data (sp500_data.csv)")
        _print_file_info(PIT_FILE, "Point-in-time data (sp500_pit_data.csv)")
        return

    print("⚠️ Raw CSV files not found in data/raw/")
    print(f"   Expected: {PRICE_FILE.name}, {PIT_FILE.name}")
    print("   Trying to rebuild them from archive.zip …")

    result = _extract_from_archive()
    if result is None:
        print("\n❌ Could not create the raw CSV files automatically.")
        print("   Please make sure that:")
        print("   - data/raw/archive.zip contains the Kaggle dataset, OR")
        print("   - data/raw/sp500_data.csv and data/raw/sp500_pit_data.csv")
        print("     are manually placed in the folder.")
        return

    print("\n✅ Successfully created raw CSV files from archive.zip.")
    _print_file_info(PRICE_FILE, "Price data (sp500_data.csv)")
    _print_file_info(PIT_FILE, "Point-in-time data (sp500_pit_data.csv)")


def main() -> None:
    """Entry point used by the scripts/ step in the pipeline."""
    ensure_raw_files()


if __name__ == "__main__":
    main()
