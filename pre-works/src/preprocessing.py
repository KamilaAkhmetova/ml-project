import os
import warnings

import joblib
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .feature_engineering import (
        ALL_FEATURES,
        RAW_FEATURES,
        engineer_features,
        validate_physical_constraints,
    )
except ImportError:
    from feature_engineering import (
        ALL_FEATURES,
        RAW_FEATURES,
        engineer_features,
        validate_physical_constraints,
    )

COLUMNS = RAW_FEATURES + ["class"]
FEATURES = ALL_FEATURES
TARGET = "class"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, names=COLUMNS)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = df[TARGET].map({"g": 1, "h": 0})
    counts = df[TARGET].value_counts()
    print(f"Labels encoded: Gamma (1) = {counts[1]}, Hadron (0) = {counts[0]}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = engineer_features(df.drop(columns=[TARGET]))
    df_eng[TARGET] = df[TARGET].values
    return df_eng


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_borderline_smote(X_train: pd.DataFrame, y_train: pd.Series,
                           random_state: int = 42):
    """Borderline-SMOTE + drop synthetic rows that violate physical constraints."""
    original_size = len(X_train)
    smote = BorderlineSMOTE(random_state=random_state, k_neighbors=5)
    X_res_arr, y_res_arr = smote.fit_resample(X_train.values, y_train.values)

    X_res = pd.DataFrame(X_res_arr, columns=X_train.columns)
    y_res = pd.Series(y_res_arr, name=y_train.name)

    synthetic_mask = pd.Series([True] * len(X_res))
    synthetic_mask.iloc[:original_size] = False
    valid_mask = validate_physical_constraints(X_res[RAW_FEATURES])
    keep = (~synthetic_mask) | valid_mask
    dropped = int((~keep).sum())
    synthetic_count = int(synthetic_mask.sum())
    pct = (dropped / synthetic_count * 100) if synthetic_count else 0.0

    X_res = X_res.loc[keep].reset_index(drop=True)
    y_res = y_res.loc[keep].reset_index(drop=True)

    print(f"SMOTE: {original_size} -> {len(X_res)} (synthetic={synthetic_count}, dropped={dropped} = {pct:.1f}%)")
    if pct > 30:
        warnings.warn(f"BorderlineSMOTE dropped {pct:.1f}% of synthetic samples — fallback advised")
    return X_res, y_res


def scale_features(X_train, X_test, save_path: str | None = None):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    print(f"Scaled {X_train.shape[1]} features (mean=0, std=1)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved: {save_path}")
    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42,
                 save_scaler_path: str | None = None,
                 apply_smote: bool = True):
    print("=" * 60)
    print("  MAGIC Telescope — Preprocessing Pipeline (engineered + SMOTE)")
    print("=" * 60)

    df = load_data(filepath)
    df = encode_labels(df)

    raw_mask = validate_physical_constraints(df)
    dropped = int((~raw_mask).sum())
    if dropped:
        print(f"Dropped {dropped} raw rows violating physical constraints")
        df = df.loc[raw_mask].reset_index(drop=True)

    df = add_engineered_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    if apply_smote:
        X_train, y_train = apply_borderline_smote(X_train, y_train, random_state)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, save_path=save_scaler_path
    )

    print("\nPipeline complete.")
    print("=" * 60)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline("data/magic04.data", save_scaler_path="models/scaler.pkl")
