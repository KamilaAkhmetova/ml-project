import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

from preprocessing import run_pipeline


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_proba),
    }


def save_processed(X_train, X_test, y_train, y_test, out_dir='../data/processed'):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)


def train_all(X_train, X_test, y_train, y_test, models_dir='../models'):
    os.makedirs(models_dir, exist_ok=True)

    results = []
    trained = {}

    for name, model in get_models().items():
        print(f'Training {name}...', end=' ')
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)

        filename = name.replace(' ', '_').lower() + '.pkl'
        joblib.dump(model, os.path.join(models_dir, filename))
        trained[name] = model
        print('done')

    results_df = pd.DataFrame(results).set_index('Model').sort_values('ROC-AUC', ascending=False)
    return trained, results_df


def main():
    X_train, X_test, y_train, y_test, _ = run_pipeline('../data/magic04.data')
    save_processed(X_train, X_test, y_train, y_test)

    trained, results_df = train_all(X_train, X_test, y_train, y_test)

    print('\n' + '=' * 55)
    print(results_df.round(4).to_string())
    print('=' * 55)

    best_name = results_df.index[0]
    print(f'\nBest model: {best_name}')
    print(f'ROC-AUC: {results_df.loc[best_name, "ROC-AUC"]:.4f}')

    y_pred = trained[best_name].predict(X_test)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Hadron (0)', 'Gamma (1)']))


if __name__ == '__main__':
    main()
