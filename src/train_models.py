"""
Model training & evaluation for the S&P 500 project.

This script:
- loads the pre-built ML dataset (features + targets),
- runs a TimeSeriesSplit cross-validation,
- trains three models: Logistic Regression, Random Forest, optional XGBoost,
- compares them to simple baselines:
    * Random scores,
    * Naive momentum (sign of 1-day return),
    * Buy & hold (always predict 1),
- computes and saves:
    * classification metrics (Accuracy, ROC AUC, PR AUC, Precision@K),
    * confusion matrices,
    * ROC & Precision-Recall curves,
    * Precision@K curves,
    * permutation importance (LR / RF),
    * learning curves (LR / RF),
    * XGBoost feature importance,
    * SHAP plots (optional, XGBoost only),
    * simple top-K backtest for XGBoost.

Use FAST_DEV=True during development to reduce runtime,
and set FAST_DEV=False for the final full run.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

shap.initjs()

# XGBoost is optional
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("⚠️ xgboost not installed: XGBoost model will be skipped.")


DATA_FILE = Path("data/sp500_ml_dataset.parquet")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
FAST_DEV: bool = True          # Set to False for the final full run
DO_SHAP: bool = not FAST_DEV   # SHAP is expensive: only in full run
DO_XGB: bool = not FAST_DEV    # XGBoost + SHAP are the most expensive
N_SPLITS: int = 3 if FAST_DEV else 5

# Subsample rows in development mode to speed up experiments
MAX_ROWS: Optional[int] = 150_000 if FAST_DEV else None


# -------------------------------------------------------------------
# Helpers: dataset loading
# -------------------------------------------------------------------
def load_ml_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the ML dataset and return:
    - full DataFrame
    - list of feature column names (excluding targets, date, ticker)

    Returns
    -------
    df : pd.DataFrame
        Full ML dataset.
    feature_cols : list of str
        Names of feature columns used as X.
    """
    df = pd.read_parquet(DATA_FILE)
    print("ML dataset loaded:", df.shape)

    # Ensure temporal ordering
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Subsample for development if requested
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS).reset_index(drop=True)
        print(f"[FAST_DEV] Subsampled to {len(df)} rows")

    # Columns excluded from features
    cols_excl = {
        "Date",
        "Ticker",
        "ret_1d_ahead",
        "ret_5d_ahead",
        "target_1d_top50",
        "target_5d_top50",
        "target_1d_top10",
        "target_5d_top10",
    }
    feature_cols = [c for c in df.columns if c not in cols_excl]

    print("Number of features:", len(feature_cols))
    return df, feature_cols


# -------------------------------------------------------------------
# Metrics & evaluation helpers
# -------------------------------------------------------------------
def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: float = 0.1) -> float:
    """
    Precision@K: fraction of true positives among the top k% scores.

    Parameters
    ----------
    y_true : array-like
        True binary labels in {0,1}.
    scores : array-like
        Continuous model scores (higher = more likely to be class 1).
    k : float, default=0.1
        Fraction of the sample to keep (0.1 = top 10%).

    Returns
    -------
    float
        Precision in the top k% of scores.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    n = len(scores)
    k_int = max(1, int(k * n))
    idx = np.argsort(scores)[::-1][:k_int]
    return float(y_true[idx].mean())


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics and print a short report.

    Metrics:
    - Accuracy
    - ROC AUC
    - PR AUC
    - Precision@10%
    - Precision@20%

    Returns a dict suitable for building a CV results DataFrame.
    """
    acc = accuracy_score(y_true, y_pred)

    try:
        roc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = np.nan

    p_at_10 = precision_at_k(y_true, y_scores, k=0.10)
    p_at_20 = precision_at_k(y_true, y_scores, k=0.20)

    print(f"\n===== {name} =====")
    print(f"Accuracy      : {acc:.3f}")
    print(f"ROC AUC       : {roc:.3f}")
    print(f"PR AUC        : {pr_auc:.3f}")
    print(f"Precision@10% : {p_at_10:.3f}")
    print(f"Precision@20% : {p_at_20:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    metrics = {
        "model": name,
        "accuracy": acc,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "precision_at_10": p_at_10,
        "precision_at_20": p_at_20,
    }
    return metrics


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    horizon_label: str,
) -> None:
    """
    Save a normalized confusion matrix (% per row) as PNG in results/.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_title(f"Confusion matrix - {model_name}\n{horizon_label}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    # Display values inside the cells
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if val < 0.5 else "white",
                fontsize=10,
            )

    fig.colorbar(im, fraction=0.046, pad=0.04)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"confusion_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"   ➜ Confusion matrix saved to: {fname}")


def save_xgb_feature_importance(
    model,
    feature_names: List[str],
    horizon_label: str,
    top_n: int = 20,
) -> None:
    """
    Save a horizontal bar plot of the top N XGBoost feature importances.
    """
    importances = model.feature_importances_

    idx = np.argsort(importances)[::-1][:top_n]
    imp_vals = importances[idx]
    imp_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(range(len(idx)), imp_vals[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(imp_names[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} features - XGBoost\n{horizon_label}")

    plt.tight_layout()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"xgb_feature_importance_{horizon_label.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"   ➜ XGBoost feature importance saved to: {fname}")


def save_shap_plots(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    horizon_label: str,
    max_samples: int = 20_000,
    top_dep: int = 3,
) -> None:
    """
    Generate and save SHAP plots for a tree-based model (XGBoost):

    - SHAP summary plot
    - SHAP bar plot (mean |SHAP|)
    - SHAP dependence plots for the top top_dep features
    """
    # Subsample to avoid memory explosion
    if X_test.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_test.shape[0], size=max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary models, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) SHAP summary plot
    fname1 = out_dir / f"shap_summary_{horizon_label.replace(' ', '_')}.png"
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ➜ SHAP summary plot saved to: {fname1}")

    # 2) SHAP bar plot (mean |SHAP|)
    fname2 = out_dir / f"shap_bar_{horizon_label.replace(' ', '_')}.png"
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ➜ SHAP bar plot saved to: {fname2}")

    # 3) SHAP dependence plots for top features
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_dep]

    for i in top_idx:
        feat_name = feature_names[i]
        fname_dep = out_dir / (
            f"shap_dependence_{feat_name}_{horizon_label.replace(' ', '_')}.png"
        )

        shap.dependence_plot(
            feat_name,
            shap_vals,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(fname_dep, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ➜ SHAP dependence plot for {feat_name} saved to: {fname_dep}")


def save_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    horizon_label: str,
) -> None:
    """
    Save a figure with:
    - ROC curve
    - Precision-Recall curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC - {model_name}\n{horizon_label}")
    axes[0].legend()

    # PR
    axes[1].plot(rec, prec, label="PR curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall - {model_name}\n{horizon_label}")
    axes[1].legend()

    plt.tight_layout()
    fname = out_dir / f"roc_pr_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   ➜ ROC & PR curves saved to: {fname}")


def save_precision_at_k_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    horizon_label: str,
) -> None:
    """
    Save Precision@K curve for K between 1% and 30%.
    """
    ks = np.linspace(0.01, 0.30, 30)
    precs: List[float] = []

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    n = len(scores)
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    for k in ks:
        k_int = max(1, int(k * n))
        precs.append(float(y_sorted[:k_int].mean()))

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks * 100, precs, marker="o")
    ax.set_xlabel("Top K% predictions")
    ax.set_ylabel("Precision@K")
    ax.set_title(f"Precision@K - {model_name}\n{horizon_label}")
    ax.grid(True)

    plt.tight_layout()
    fname = out_dir / f"precision_at_k_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   ➜ Precision@K curve saved to: {fname}")


def save_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    horizon_label: str,
    model_name: str,
    n_repeats: int = 5,
) -> None:
    """
    Compute and plot permutation importance for a fitted model
    (Random Forest or Logistic Regression) on the last fold.
    """
    print(f"\n   🔧 Permutation importance for {model_name}...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:20]
    imp_vals = importances[idx]
    imp_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(range(len(idx)), imp_vals[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(imp_names[::-1])
    ax.set_xlabel("Mean decrease in score")
    ax.set_title(f"Permutation importance - {model_name}\n{horizon_label}")

    plt.tight_layout()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"perm_importance_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   ➜ Permutation importance saved to: {fname}")


def save_learning_curve(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    horizon_label: str,
    model_name: str,
) -> None:
    """
    Compute and plot a learning curve (train_size vs ROC AUC)
    for Logistic Regression / Random Forest.
    """
    print(f"\n   🔧 Learning curve for {model_name}...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=3,
        shuffle=False,   # keep time ordering
        scoring="roc_auc",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, marker="o", label="Train ROC AUC")
    ax.plot(train_sizes, val_mean, marker="s", label="Validation ROC AUC")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Learning curve - {model_name}\n{horizon_label}")
    ax.legend()
    ax.grid(True)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"learning_curve_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   ➜ Learning curve saved to: {fname}")


# -------------------------------------------------------------------
# Simple backtest helper
# -------------------------------------------------------------------
def backtest_top_k(
    scores: np.ndarray,
    ret_ahead_test: np.ndarray,
    top_k: float = 0.10,
) -> Tuple[float, float]:
    """
    Very simple cross-sectional backtest:

    - sort all stocks by predicted score (descending),
    - buy the top_k fraction of names,
    - compute the mean forward return of:
        * the strategy (top_k),
        * the cross-sectional market average.

    Returns
    -------
    strat_ret : float
        Mean return of the top_k strategy.
    mkt_ret : float
        Mean return of the cross-sectional average.
    """
    scores = np.asarray(scores)
    ret_ahead_test = np.asarray(ret_ahead_test)

    order = np.argsort(scores)[::-1]
    k_int = max(1, int(top_k * len(order)))
    top_idx = order[:k_int]

    strat_ret = float(np.nanmean(ret_ahead_test[top_idx]))
    mkt_ret = float(np.nanmean(ret_ahead_test))

    return strat_ret, mkt_ret


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def main() -> None:
    df, feature_cols = load_ml_dataset()

    # For backtest choice of horizon
    ret_1d_all = df["ret_1d"].values
    ret_1d_ahead_all = df["ret_1d_ahead"].values
    ret_5d_ahead_all = df["ret_5d_ahead"].values
    dates = df["Date"].values

    # Targets (full list)
    targets_full: List[Tuple[str, str]] = [
        ("target_1d_top50", "HORIZON_1_DAY_Top50"),
        ("target_1d_top10", "HORIZON_1_DAY_Top10"),
        ("target_5d_top50", "HORIZON_5_DAYS_Top50"),
        ("target_5d_top10", "HORIZON_5_DAYS_Top10"),
    ]

    if FAST_DEV:
        targets = [targets_full[0]]  # e.g. only 1D Top 50% in dev
        print("[FAST_DEV] Only processing target:", targets)
    else:
        targets = targets_full
        print("[FULL RUN] All targets will be processed.")

    X_all = df[feature_cols].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for target_col, label in targets:
        print("\n" + "#" * 70)
        print(f"### {label}  ({target_col})")
        print("#" * 70 + "\n")

        y_all = df[target_col].values

        all_results: List[Dict] = []
        backtest_rows: List[Dict] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
            print(f"\n===== Fold {fold}/{tscv.n_splits} =====")
            print(
                "  Train :",
                pd.to_datetime(dates[train_idx]).min().date(),
                "->",
                pd.to_datetime(dates[train_idx]).max().date(),
            )
            print(
                "  Test  :",
                pd.to_datetime(dates[test_idx]).min().date(),
                "->",
                pd.to_datetime(dates[test_idx]).max().date(),
            )

            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            # Choose the correct forward return horizon for backtesting
            if "1d" in target_col:
                ret_ahead_test = ret_1d_ahead_all[test_idx]
            else:
                ret_ahead_test = ret_5d_ahead_all[test_idx]

            # ------------------------------------------------------
            # 1) Logistic Regression (with StandardScaler)
            # ------------------------------------------------------
            print("\n🔹 Training Logistic Regression...")
            logreg_clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            logreg_clf.fit(X_train, y_train)
            logreg_scores = logreg_clf.predict_proba(X_test)[:, 1]
            logreg_pred = (logreg_scores >= 0.5).astype(int)

            res_lr = evaluate_model("Logistic Regression", y_test, logreg_scores, logreg_pred)
            res_lr["fold"] = fold
            res_lr["target"] = label
            all_results.append(res_lr)

            # Confusion matrix + advanced plots only on last fold
            if fold == tscv.n_splits:
                save_confusion_matrix(y_test, logreg_pred, "Logistic_Regression", label)
                if not FAST_DEV:
                    save_learning_curve(logreg_clf, X_train, y_train, label, "Logistic_Regression")
                    save_permutation_importance(
                        logreg_clf, X_test, y_test, feature_cols, label, "Logistic_Regression"
                    )

            # ------------------------------------------------------
            # 2) Random Forest
            # ------------------------------------------------------
            print("\n🔹 Training Random Forest...")
            rf_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
            rf_clf.fit(X_train, y_train)
            rf_scores = rf_clf.predict_proba(X_test)[:, 1]
            rf_pred = (rf_scores >= 0.5).astype(int)

            res_rf = evaluate_model("Random Forest", y_test, rf_scores, rf_pred)
            res_rf["fold"] = fold
            res_rf["target"] = label
            all_results.append(res_rf)

            if fold == tscv.n_splits:
                save_confusion_matrix(y_test, rf_pred, "Random_Forest", label)
                if not FAST_DEV:
                    save_learning_curve(rf_clf, X_train, y_train, label, "Random_Forest")
                    save_permutation_importance(
                        rf_clf, X_test, y_test, feature_cols, label, "Random_Forest"
                    )

            # ------------------------------------------------------
            # 3) XGBoost (optional in dev)
            # ------------------------------------------------------
            xgb_scores = None
            xgb_clf = None

            if DO_XGB and (XGBClassifier is not None):
                print("\n🔹 Training XGBoost...")
                max_train = 300_000
                if X_train.shape[0] > max_train:
                    rng_sub = np.random.RandomState(42)
                    idx_sub = rng_sub.choice(X_train.shape[0], size=max_train, replace=False)
                    X_train_xgb = X_train[idx_sub]
                    y_train_xgb = y_train[idx_sub]
                    print(f"  (subsample for XGBoost: {X_train_xgb.shape})")
                else:
                    X_train_xgb, y_train_xgb = X_train, y_train

                xgb_clf = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                )
                xgb_clf.fit(X_train_xgb, y_train_xgb)
                xgb_scores = xgb_clf.predict_proba(X_test)[:, 1]
                xgb_pred = (xgb_scores >= 0.5).astype(int)

                res_xgb = evaluate_model("XGBoost", y_test, xgb_scores, xgb_pred)
                res_xgb["fold"] = fold
                res_xgb["target"] = label
                all_results.append(res_xgb)

                if fold == tscv.n_splits:
                    save_confusion_matrix(y_test, xgb_pred, "XGBoost", label)
                    save_xgb_feature_importance(xgb_clf, feature_cols, label, top_n=20)

                    if DO_SHAP:
                        print("\n   🔍 SHAP analysis (sample) on last fold...")
                        save_shap_plots(xgb_clf, X_test, feature_cols, label)
                        save_roc_pr_curves(y_test, xgb_scores, "XGBoost", label)
                        save_precision_at_k_curve(y_test, xgb_scores, "XGBoost", label)

                    # Simple top-10% backtest on last fold
                    strat_ret, mkt_ret = backtest_top_k(xgb_scores, ret_ahead_test, top_k=0.10)
                    backtest_rows.append(
                        {
                            "target": label,
                            "fold": fold,
                            "model": "XGBoost",
                            "top_k": 0.10,
                            "strategy_mean_return": strat_ret,
                            "market_mean_return": mkt_ret,
                        }
                    )
                    print(
                        f"\n   📈 Backtest XGBoost (Top 10% scores) - {label}: "
                        f"strat_ret={strat_ret:.4f}, mkt_ret={mkt_ret:.4f}"
                    )

            # ------------------------------------------------------
            # 4) Baseline 1: Random scores
            # ------------------------------------------------------
            print("\n🔹 Baseline: Random...")
            rng = np.random.RandomState(123 + fold)
            rand_scores = rng.rand(len(y_test))
            rand_pred = (rand_scores >= 0.5).astype(int)

            res_rand = evaluate_model("Random baseline", y_test, rand_scores, rand_pred)
            res_rand["fold"] = fold
            res_rand["target"] = label
            all_results.append(res_rand)

            # ------------------------------------------------------
            # 5) Baseline 2: Naive momentum (sign of 1-day return)
            # ------------------------------------------------------
            print("\n🔹 Baseline: Naive momentum (ret_1d > 0)...")
            mom_scores = ret_1d_all[test_idx]
            mom_pred = (mom_scores > 0).astype(int)

            res_mom = evaluate_model("Naive momentum (ret_1d>0)", y_test, mom_scores, mom_pred)
            res_mom["fold"] = fold
            res_mom["target"] = label
            all_results.append(res_mom)

            # ------------------------------------------------------
            # 6) Baseline 3: Buy & Hold (always long)
            # ------------------------------------------------------
            print("\n🔹 Baseline: Buy & Hold (always 1)...")
            bh_scores = np.ones(len(y_test))
            bh_pred = np.ones(len(y_test), dtype=int)

            res_bh = evaluate_model("Buy & Hold (always 1)", y_test, bh_scores, bh_pred)
            res_bh["fold"] = fold
            res_bh["target"] = label
            all_results.append(res_bh)

        # ==========================================================
        # Cross-validation summary: mean + std by model
        # ==========================================================
        results_df = pd.DataFrame(all_results)
        print("\n===== CV SUMMARY (TimeSeriesSplit) for", label, "=====")

        metrics_cols = ["accuracy", "roc_auc", "pr_auc", "precision_at_10", "precision_at_20"]
        summary = (
            results_df
            .groupby("model")[metrics_cols]
            .agg(["mean", "std"])
            .round(3)
        )
        print(summary)

        # Save raw CV results & aggregated summary
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / f"results_raw_{label}.csv"
        cv_path = out_dir / f"results_cv_summary_{label}.csv"
        results_df.to_csv(raw_path, index=False)
        summary.to_csv(cv_path)
        print(f"\n   💾 Raw results saved to: {raw_path}")
        print(f"   💾 CV summary saved to: {cv_path}")

        # Save backtest results if any
        if backtest_rows:
            backtest_df = pd.DataFrame(backtest_rows)
            bt_path = out_dir / f"backtest_{label}.csv"
            backtest_df.to_csv(bt_path, index=False)
            print(f"   💾 Backtest saved to: {bt_path}")


if __name__ == "__main__":
    main()