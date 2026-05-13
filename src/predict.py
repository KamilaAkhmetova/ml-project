import argparse
import os
import sys

import joblib
import pandas as pd
from preprocessing import COLUMNS, FEATURES

CLASS_NAMES = {0: 'Hadron', 1: 'Gamma'}

MODEL_FILES = {
    'xgboost': 'xgboost.pkl',
    'random_forest': 'random_forest.pkl',
    'logistic_regression': 'logistic_regression.pkl',
    'mlp': 'mlp_neural_network.pkl',
}


def load_model(model_name, models_dir='../models'):
    if model_name not in MODEL_FILES:
        raise ValueError(f'Unknown model: {model_name}. Choose from {list(MODEL_FILES)}')

    path = os.path.join(models_dir, MODEL_FILES[model_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}. Run train_model.py first.')

    return joblib.load(path)


def load_scaler(models_dir='../models'):
    path = os.path.join(models_dir, 'scaler.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Scaler not found: {path}. Run train_model.py first.')
    return joblib.load(path)


def load_input(filepath):
    if filepath.endswith('.data'):
        df = pd.read_csv(filepath, names=COLUMNS)
        if 'class' in df.columns:
            df = df.drop(columns=['class'])
    else:
        df = pd.read_csv(filepath)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in input: {missing}')

    return df[FEATURES]


def predict(model_name, X, models_dir='../models'):
    scaler = load_scaler(models_dir)
    model = load_model(model_name, models_dir)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    out = pd.DataFrame({
        'prediction': y_pred,
        'label': [CLASS_NAMES[p] for p in y_pred],
        'gamma_probability': y_proba.round(4),
    })
    return out


def main():
    parser = argparse.ArgumentParser(description='Predict gamma/hadron from MAGIC features')
    parser.add_argument('input', help='Path to CSV/.data file with input features')
    parser.add_argument('--model', default='xgboost', choices=list(MODEL_FILES),
                        help='Which model to use (default: xgboost)')
    parser.add_argument('--models-dir', default='../models', help='Directory with .pkl files')
    parser.add_argument('--output', default=None, help='Optional path to save predictions CSV')
    args = parser.parse_args()

    X = load_input(args.input)
    print(f'Loaded {len(X)} samples from {args.input}')

    preds = predict(args.model, X, args.models_dir)

    print(f'\nUsing model: {args.model}')
    print(f'Predictions: Gamma = {(preds["prediction"] == 1).sum()}, '
          f'Hadron = {(preds["prediction"] == 0).sum()}')
    print('\nFirst 10 predictions:')
    print(preds.head(10).to_string(index=False))

    if args.output:
        preds.to_csv(args.output, index=False)
        print(f'\nSaved predictions to: {args.output}')


if __name__ == '__main__':
    sys.exit(main())
