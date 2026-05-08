import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

COLUMNS = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
]

FEATURES = [c for c in COLUMNS if c != 'class']
TARGET = 'class'


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, names=COLUMNS)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = df[TARGET].map({'g': 1, 'h': 0})
    
    counts = df[TARGET].value_counts()
    print(f"✅ Labels encoded: Gamma (1) = {counts[1]}, Hadron (0) = {counts[0]}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          
    )
    
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set:  {X_test.shape[0]} samples")
    print(f"   Class ratio in train — Gamma: {y_train.mean():.1%}, Hadron: {1 - y_train.mean():.1%}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save_path: str = None):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)   
    X_test_scaled  = scaler.transform(X_test)         
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=FEATURES, index=X_test.index)
    
    print("✅ Features standardized (mean=0, std=1)")
    print(f"   Example — fAlpha mean after scaling: {X_train_scaled['fAlpha'].mean():.4f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"✅ Scaler saved to: {save_path}")
    
    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42,
                 save_scaler: bool = True):
    print("=" * 45)
    print("  MAGIC Telescope — Preprocessing Pipeline")
    print("=" * 45)
    
    df = load_data(filepath)
    df = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    
    scaler_path = '../models/scaler.pkl' if save_scaler else None
    X_train, X_test, scaler = scale_features(X_train, X_test, save_path=scaler_path)
    
    print("\n✅ Pipeline complete! Ready for modeling.")
    print("=" * 45)
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = run_pipeline('../data/magic04.data')
