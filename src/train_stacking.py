"""Train stacking ensemble: RF + XGBoost + LightGBM -> LogisticRegression.

Pipeline:
  1. Load + engineer + validate + split.
  2. Optuna tune each base learner (30 trials, 5-fold ROC-AUC).
  3. Borderline-SMOTE on train fold (with physical-constraint filtering).
  4. Feature selection: mutual_info_classif + correlation filter.
  5. Fit StackingClassifier (cv=5) and evaluate on held-out test.
  6. Persist artefacts (model, scaler, selected_features, baseline_stats)
     and log everything to MLflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import run_pipeline  # noqa: E402

RANDOM_STATE = 42
N_TRIALS = 30
CV_FOLDS = 5
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "magic04.data"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ────────────────────────── Optuna objectives ──────────────────────────

def _cv_score(estimator, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def objective_rf(trial: optuna.Trial, X, y) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 6, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return _cv_score(RandomForestClassifier(**params), X, y)


def objective_xgb(trial: optuna.Trial, X, y) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    }
    return _cv_score(XGBClassifier(**params), X, y)


def objective_lgbm(trial: optuna.Trial, X, y) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return _cv_score(LGBMClassifier(**params), X, y)


def tune(name: str, objective, X, y) -> dict:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  {name}: best ROC-AUC = {study.best_value:.4f}")
    print(f"  {name}: best params  = {study.best_params}")
    return study.best_params


# ────────────────────────── Feature selection ──────────────────────────

def select_features(X: pd.DataFrame, y: pd.Series,
                    mi_drop_quantile: float = 0.10,
                    corr_threshold: float = 0.95) -> list[str]:
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=X.columns)
    mi_cutoff = mi_series.quantile(mi_drop_quantile)
    kept = mi_series[mi_series > mi_cutoff].sort_values(ascending=False)

    # Correlation filter: pairwise |r|>threshold → drop the one with lower MI
    cols = list(kept.index)
    corr = X[cols].corr().abs()
    to_drop: set[str] = set()
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] > corr_threshold:
                drop_target = c2 if kept[c1] >= kept[c2] else c1
                to_drop.add(drop_target)

    selected = [c for c in cols if c not in to_drop]
    print(f"Feature selection: {X.shape[1]} -> {len(selected)} features")
    print(f"  kept: {selected}")
    return selected


def shap_summary(model, X: pd.DataFrame, out_path: Path):
    explainer = shap.TreeExplainer(model)
    sample = X.sample(min(500, len(X)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ────────────────────────── Build stacking ──────────────────────────

def build_stack(rf_params: dict, xgb_params: dict, lgbm_params: dict) -> StackingClassifier:
    rf = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(**xgb_params, random_state=RANDOM_STATE,
                        eval_metric="logloss", verbosity=0, n_jobs=-1)
    lgbm = LGBMClassifier(**lgbm_params, random_state=RANDOM_STATE,
                          verbosity=-1, n_jobs=-1)
    return StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=CV_FOLDS,
        passthrough=False,
        n_jobs=-1,
    )


# ────────────────────────── Baseline stats for PSI ──────────────────────────

def compute_baseline_stats(X: pd.DataFrame, n_buckets: int = 10) -> dict:
    stats = {}
    for col in X.columns:
        values = X[col].values
        quantiles = np.quantile(values, np.linspace(0, 1, n_buckets + 1)).tolist()
        stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "quantiles": quantiles,
        }
    return stats


# ────────────────────────── Main ──────────────────────────

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment("magic_stacking")

    scaler_path = MODELS_DIR / "scaler.pkl"
    X_train, X_test, y_train, y_test, _ = run_pipeline(
        str(DATA_PATH),
        save_scaler_path=str(scaler_path),
        apply_smote=True,
    )

    with mlflow.start_run(run_name="stacking_full"):
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_features_initial", X_train.shape[1])

        # Feature selection BEFORE tuning so tuning sees final feature set
        selected = select_features(X_train, y_train)
        X_train = X_train[selected]
        X_test = X_test[selected]
        mlflow.log_param("n_features_selected", len(selected))
        mlflow.log_param("selected_features", ",".join(selected))

        print("\n[1/4] Tuning Random Forest…")
        rf_params = tune("RF", objective_rf, X_train, y_train)
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})

        print("\n[2/4] Tuning XGBoost…")
        xgb_params = tune("XGB", objective_xgb, X_train, y_train)
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

        print("\n[3/4] Tuning LightGBM…")
        lgbm_params = tune("LGBM", objective_lgbm, X_train, y_train)
        mlflow.log_params({f"lgbm_{k}": v for k, v in lgbm_params.items()})

        print("\n[4/4] Fitting StackingClassifier…")
        stack = build_stack(rf_params, xgb_params, lgbm_params)
        stack.fit(X_train, y_train)

        y_pred = stack.predict(X_test)
        y_proba = stack.predict_proba(X_test)[:, 1]
        metrics = {
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }
        print("\nTest metrics:")
        for k, v in metrics.items():
            print(f"  {k:8s} = {v:.4f}")
            mlflow.log_metric(k, v)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["Hadron", "Gamma"]))

        # SHAP on a base learner (RF) — full stack SHAP is heavy
        try:
            rf_base = stack.named_estimators_["rf"]
            shap_path = MODELS_DIR / "shap_summary.png"
            shap_summary(rf_base, X_train, shap_path)
            mlflow.log_artifact(str(shap_path))
            print(f"SHAP summary saved: {shap_path}")
        except Exception as e:
            print(f"SHAP step skipped: {e}")

        # Persist artefacts
        model_path = MODELS_DIR / "stacking_model.pkl"
        joblib.dump(stack, model_path)
        print(f"Model saved: {model_path}")

        # Re-fit scaler on selected features only so API can load it stand-alone
        # (the original scaler covers all 17 features; we keep it as-is, and the
        # API will project to selected features after scaling)
        selected_path = MODELS_DIR / "selected_features.json"
        with open(selected_path, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Selected features saved: {selected_path}")

        baseline = compute_baseline_stats(X_train)
        baseline_path = MODELS_DIR / "baseline_stats.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline stats saved: {baseline_path}")

        mlflow.sklearn.log_model(stack, "stacking_model")
        mlflow.log_artifact(str(selected_path))
        mlflow.log_artifact(str(baseline_path))

        print("\nDone.")


if __name__ == "__main__":
    main()
