"""
Main entry point for the project.

This script orchestrates the full pipeline:

1. Build the raw panel of daily prices + point-in-time fundamentals
   -> src.data_loader.build_panel()

2. Clean the panel (drop *_y columns, forward-fill fundamentals)
   -> src.prepare_dataset.clean_panel()

3. Merge with cross-sectional fundamentals from Kaggle
   -> src.add_fundamentals.merge_panel_and_fundamentals()

4. Build the final ML dataset (features + targets)
   -> src.build_ml_dataset.build_dataset()

5. Train and evaluate ML models
   -> src.train_models.main()

Usage
-----
Main results (default: ALL targets):
    python main.py

Top50 only (single target):
    python main.py --top50

FAST development run (quick sanity check):
    python main.py --fast

FULL run + SHAP (slow):
    python main.py --shap

Notes
-----
- By default, `python main.py` evaluates all four targets:
  Top50/Top10 for 1-day and 5-day horizons.
- `--top50` restricts evaluation to the single next-day Top50 target.
- `--fast` always processes a single target and runs a lightweight configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import build_panel
from src.prepare_dataset import load_panel, clean_panel, save_clean
from src.add_fundamentals import merge_panel_and_fundamentals
from src.build_ml_dataset import build_dataset
import src.train_models as train_models_module


# ---------------------------------------------------------------------
# Helpers to (re)build each stage
# ---------------------------------------------------------------------
def run_build_panel() -> None:
    """Stage 1: build the raw price + PIT panel if needed."""
    panel_path = Path("data/sp500_panel.parquet")

    if panel_path.exists():
        print(f"✅ Panel already exists: {panel_path} (skipping rebuild)")
    else:
        print("▶ Building raw panel (prices + PIT fundamentals)...")
        build_panel()


def run_clean_panel() -> None:
    """Stage 2: clean the panel and forward-fill fundamentals."""
    clean_path = Path("data/sp500_clean_panel.parquet")

    if clean_path.exists():
        print(f"✅ Clean panel already exists: {clean_path} (skipping)")
        return

    print("▶ Cleaning panel...")
    df = load_panel()
    df = clean_panel(df)
    save_clean(df)


def run_add_fundamentals() -> None:
    """Stage 3: merge clean panel with Kaggle fundamentals."""
    out_path = Path("data/sp500_panel_with_fundamentals.parquet")

    if out_path.exists():
        print(f"✅ Panel with fundamentals already exists: {out_path} (skipping)")
        return

    print("▶ Merging panel with Kaggle fundamentals...")
    merge_panel_and_fundamentals()


def run_build_ml_dataset() -> None:
    """Stage 4: build the final ML dataset (features + targets)."""
    ml_path = Path("data/sp500_ml_dataset.parquet")

    if ml_path.exists():
        print(f"✅ ML dataset already exists: {ml_path} (skipping)")
        return

    print("▶ Building ML dataset (features + targets)...")
    build_dataset()


def run_training(fast_dev: bool, do_shap: bool, top50_only: bool) -> None:
    """
    Stage 5: train and evaluate models.

    We control the training behaviour by updating global config variables
    in src.train_models *before* calling train_models.main().
    """
    if fast_dev:
        print("⚙️  Running FAST mode (quick sanity check run).")
    else:
        print("⚙️  Running FULL mode (main results by default).")

    # Configure the training module programmatically
    train_models_module.FAST_DEV = fast_dev
    train_models_module.TOP50_ONLY = top50_only

    if fast_dev:
        # FAST: keep it light
        train_models_module.DO_SHAP = False
        train_models_module.DO_XGB = False
        train_models_module.N_SPLITS = 2
        train_models_module.MAX_ROWS = 150_000
    else:
        # FULL: 3 folds, XGB ON, SHAP OFF unless --shap
        train_models_module.DO_SHAP = do_shap
        train_models_module.DO_XGB = True
        train_models_module.N_SPLITS = 3
        train_models_module.MAX_ROWS = None

    train_models_module.main()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the S&P 500 ML pipeline (data → features → models)."
    )

    parser.add_argument(
        "--stage",
        choices=["all", "data", "clean", "fundamentals", "features", "train"],
        default="all",
        help=(
            "Which part of the pipeline to run:\n"
            "  all            = full pipeline (default)\n"
            "  data           = build raw price + PIT panel\n"
            "  clean          = clean panel and forward-fill fundamentals\n"
            "  fundamentals   = merge with Kaggle fundamentals\n"
            "  features       = build final ML dataset\n"
            "  train          = train and evaluate models only"
        ),
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run FAST mode (2 folds, subsample, no XGB, no SHAP; single target).",
    )

    parser.add_argument(
        "--shap",
        action="store_true",
        help="Enable SHAP (FULL mode only; slow).",
    )

    parser.add_argument(
        "--top50",
        action="store_true",
        help="Evaluate only the single next-day Top50 target (target_1d_top50).",
    )

    args = parser.parse_args()

    # Safety: in FAST mode, TOP50_ONLY doesn't matter because FAST_DEV already forces 1 target.
    if args.fast and args.shap:
        print("⚠️  Note: --shap is ignored in --fast mode.")

    return args


def main() -> None:
    args = parse_args()
    fast_dev = args.fast

    if args.stage in ("all", "data"):
        run_build_panel()

    if args.stage in ("all", "clean"):
        run_clean_panel()

    if args.stage in ("all", "fundamentals"):
        run_add_fundamentals()

    if args.stage in ("all", "features"):
        run_build_ml_dataset()

    if args.stage in ("all", "train"):
        run_training(
            fast_dev=fast_dev,
            do_shap=args.shap,
            top50_only=args.top50,
        )


if __name__ == "__main__":
    main()
