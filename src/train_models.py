import matplotlib.pyplot as plt
import shap
shap.initjs()
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("⚠️ xgboost non installé : le modèle XGBoost sera ignoré.")


DATA_FILE = Path("data/sp500_ml_dataset.parquet")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_ml_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    print("Dataset ML chargé :", df.shape)

    # On s'assure d'être bien trié temporellement
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # On exclut les colonnes qui ne sont pas des features
    cols_excl = {
        "Date", "Ticker",
        "ret_1d_ahead", "ret_5d_ahead",
        "target_1d_top50", "target_5d_top50",
        "target_1d_top10", "target_5d_top10",
    }
    feature_cols = [c for c in df.columns if c not in cols_excl]

    print("Nombre de features :", len(feature_cols))
    return df, feature_cols


def temporal_train_test_split(df: pd.DataFrame, feature_cols, target_col: str,
                              n_splits: int = 5):
    """
    Sépare train/test avec un TimeSeriesSplit :
    - on utilise un schéma "expanding window"
    - on garde le DERNIER split comme jeu de test final (le plus récent)

    Ça respecte le temps, tout en étant très proche de ce que tu faisais
    avec un simple cutoff sur la date.
    """

    # IMPORTANT : df déjà trié dans load_ml_dataset,
    # mais on sécurise :
    df_sorted = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    X = df_sorted[feature_cols].values
    y = df_sorted[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    last_train_idx, last_test_idx = None, None
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        last_train_idx, last_test_idx = train_idx, test_idx

    X_train, X_test = X[last_train_idx], X[last_test_idx]
    y_train, y_test = y[last_train_idx], y[last_test_idx]

    print(f"TimeSeriesSplit avec {n_splits} splits.")
    print(
        "  Train :",
        X_train.shape,
        "de",
        df_sorted.loc[last_train_idx, "Date"].min().date(),
        "à",
        df_sorted.loc[last_train_idx, "Date"].max().date(),
    )
    print(
        "  Test  :",
        X_test.shape,
        "de",
        df_sorted.loc[last_test_idx, "Date"].min().date(),
        "à",
        df_sorted.loc[last_test_idx, "Date"].max().date(),
    )

    return X_train, X_test, y_train, y_test

def precision_at_k(y_true, scores, k: float = 0.1) -> float:
    """
    Precision@k : proportion de vrais 1 dans le top k% des scores.
    k est un ratio (0.1 = top 10%).
    """
    n = len(scores)
    k_int = max(1, int(k * n))
    idx = np.argsort(scores)[::-1][:k_int]
    return y_true[idx].mean()


def evaluate_model(name, y_true, y_scores, y_pred):
    """Calcule les métriques demandées et les affiche."""

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    p_at_10 = precision_at_k(y_true, y_scores, k=0.10)
    p_at_20 = precision_at_k(y_true, y_scores, k=0.20)

    print(f"\n===== {name} =====")
    print(f"Accuracy      : {acc:.3f}")
    print(f"ROC AUC       : {roc:.3f}")
    print(f"PR AUC        : {pr_auc:.3f}")
    print(f"Precision@10% : {p_at_10:.3f}")
    print(f"Precision@20% : {p_at_20:.3f}")
    print("\nClassification report :")
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

def save_confusion_matrix(y_true, y_pred, model_name, horizon_label):
    """
    Sauvegarde une matrice de confusion normalisée (en %) dans results/confusion_*.png
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

    # Afficher les valeurs dans les cases
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
    fname = out_dir / f"confusion_{horizon_label.replace(' ', '_')}_{model_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"   ➜ Matrice de confusion sauvegardée dans: {fname}")

def save_xgb_feature_importance(model, feature_names, horizon_label, top_n=20):
    """
    Sauvegarde un bar plot des top features XGBoost dans results/xgb_feature_importance_*.png
    """
    importances = model.feature_importances_

    # Top N features
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
    print(f"   ➜ Feature importance XGBoost sauvegardée dans: {fname}")

def save_shap_plots(model, X_test, feature_names, horizon_label,
                    max_samples=20000, top_dep=3):
    """
    Génère :
    - SHAP summary plot
    - SHAP bar plot (mean |SHAP|)
    - SHAP dependence plots pour les top features

    Tout est sauvegardé dans results/
    """

    # Échantillonner pour ne pas exploser la RAM
    if X_test.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_test.shape[0], size=max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Pour les modèles binaires, shap_values peut être une liste [classe0, classe1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1) Summary plot ===
    fname1 = out_dir / f"shap_summary_{horizon_label.replace(' ', '_')}.png"
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ➜ SHAP summary plot sauvegardé dans: {fname1}")

    # === 2) Bar plot (mean |SHAP|) ===
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
    print(f"   ➜ SHAP bar plot sauvegardé dans: {fname2}")

    # === 3) Dependence plots sur les top features ===
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
        print(f"   ➜ SHAP dependence plot pour {feat_name} sauvegardé dans: {fname_dep}")

def save_roc_pr_curves(y_true, y_scores, model_name, horizon_label):
    """
    Sauvegarde une figure avec :
    - courbe ROC
    - courbe Precision-Recall
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

    print(f"   ➜ ROC & PR curves sauvegardées dans: {fname}")


def save_precision_at_k_curve(y_true, scores, model_name, horizon_label):
    """
    Sauvegarde la courbe Precision@K pour K de 1% à 30%.
    """

    ks = np.linspace(0.01, 0.30, 30)
    precs = []
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    n = len(scores)
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    for k in ks:
        k_int = max(1, int(k * n))
        precs.append(y_sorted[:k_int].mean())

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks * 100, precs, marker="o")
    ax.set_xlabel("Top K% des prédictions")
    ax.set_ylabel("Precision@K")
    ax.set_title(f"Precision@K - {model_name}\n{horizon_label}")
    ax.grid(True)

    plt.tight_layout()
    fname = out_dir / f"precision_at_k_{horizon_label.replace(' ', '_')}_{model_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   ➜ Precision@K curve sauvegardée dans: {fname}")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    df, feature_cols = load_ml_dataset()

    # On évalue les modèles pour deux horizons
    targets = [
        ("target_1d_top50", "HORIZON 1 JOUR (Top 50%)"),
        ("target_1d_top10", "HORIZON 1 JOUR (Top 10%)"),
        ("target_5d_top50", "HORIZON 5 JOURS (Top 50%)"),
        ("target_5d_top10", "HORIZON 5 JOURS (Top 10%)"),
    ]


    for target_col, label in targets:
        print("\n" + "#" * 60)
        print(f"### {label}  ({target_col})")
        print("#" * 60 + "\n")

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            df, feature_cols, target_col, n_splits=5
        )

        results = []

        # 1) Logistic Regression
        print("\n🔹 Entraînement Logistic Regression...")
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
        results.append(
            evaluate_model("Logistic Regression", y_test, logreg_scores, logreg_pred)
        )
        save_confusion_matrix(y_test, logreg_pred, "Logistic_Regression", label)
        # 2) Random Forest
        print("\n🔹 Entraînement Random Forest...")
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        rf_clf.fit(X_train, y_train)
        rf_scores = rf_clf.predict_proba(X_test)[:, 1]
        rf_pred = (rf_scores >= 0.5).astype(int)
        results.append(
            evaluate_model("Random Forest", y_test, rf_scores, rf_pred)
        )
        save_confusion_matrix(y_test, rf_pred, "Random_Forest", label)


        # 3) XGBoost (si dispo)
        if XGBClassifier is not None:
            print("\n🔹 Entraînement XGBoost...")
            max_train = 300_000
            if X_train.shape[0] > max_train:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_train.shape[0], size=max_train, replace=False)
                X_train_xgb = X_train[idx]
                y_train_xgb = y_train[idx]
                print(f"  (subsample pour XGBoost : {X_train_xgb.shape})")
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
            results.append(
                evaluate_model("XGBoost", y_test, xgb_scores, xgb_pred)
            )
            save_confusion_matrix(y_test, xgb_pred, "XGBoost", label)
            save_xgb_feature_importance(xgb_clf, feature_cols, label, top_n=20)
        print("\n   🔍 Calcul SHAP (échantillon)...")
        save_shap_plots(xgb_clf, X_test, feature_cols, label)
        # Courbes ROC/PR + Precision@K pour XGBoost
        save_roc_pr_curves(y_test, xgb_scores, "XGBoost", label)
        save_precision_at_k_curve(y_test, xgb_scores, "XGBoost", label)


        # Résumé pour cet horizon
        results_df = pd.DataFrame(results)
        print("\n===== RÉSUMÉ DES MODÈLES pour", label, "=====")
        print(results_df.set_index("model"))

if __name__ == "__main__":
    main()

