"""Iris logistic regression experiments tracked with MLflow.

Run from the project directory. Uses Kaggle's Iris.csv schema (or an equivalent
Iris.csv included for offline reproducibility, sourced from sklearn/UCI).
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
EXPERIMENT = 'TC5061_Iris_Clasificacion'


def parse_args():
    parser = argparse.ArgumentParser(description='Entrenamiento Iris con MLflow Tracking')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'Iris.csv')
    parser.add_argument('--C', type=float, default=1.0, help='Inverso de regularización (>0)')
    parser.add_argument('--solver', choices=['lbfgs', 'newton-cg', 'saga'], default='lbfgs')
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--experiment', type=str, default=EXPERIMENT)
    parser.add_argument('--tracking-uri', type=str, default=None,
                        help='Predeterminado: SQLite local ./mlflow_iris.db')
    args = parser.parse_args()
    if args.C <= 0 or args.max_iter <= 0 or args.tol <= 0 or not 0 < args.test_size < 1:
        parser.error('C, max-iter y tol deben ser positivos; test-size debe estar entre 0 y 1.')
    return args


def load_data(data_path):
    df = pd.read_csv(data_path)
    aliases = {
        'SepalLengthCm': 'sepal_length_cm',
        'SepalWidthCm': 'sepal_width_cm',
        'PetalLengthCm': 'petal_length_cm',
        'PetalWidthCm': 'petal_width_cm',
        'Species': 'species',
    }
    df = df.drop(columns=['Id'], errors='ignore').rename(columns=aliases)
    expected = FEATURES + ['species']
    if not set(expected).issubset(df):
        raise ValueError(f'Faltan columnas. Se requieren: {expected}')
    df = df[expected].copy()
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='raise')
    df['species'] = df['species'].astype(str).str.replace('Iris-', '', regex=False)
    if df['species'].isna().any() or df['species'].nunique() != 3:
        raise ValueError('Se esperaban tres especies sin etiquetas faltantes.')
    # Igual que el notebook: conservar registros con mediciones coincidentes.
    return df


def make_pipeline(args):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=args.C, solver=args.solver,
                                          max_iter=args.max_iter, tol=args.tol,
                                          random_state=args.seed)),
    ])


def scores(y_true, y_pred):
    return {
        'test_accuracy': float(accuracy_score(y_true, y_pred)),
        'test_balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'test_f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }


def execute(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df['species'], test_size=args.test_size,
        random_state=args.seed, stratify=df['species'],
    )
    model = make_pipeline(args)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = scores(y_test, predictions)
    labels = list(model.classes_)

    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'mlflow_iris.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f'lr_C{args.C:g}_{args.solver}_iter{args.max_iter}_seed{args.seed}'
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param('algorithm', 'LogisticRegression')
        mlflow.log_param('C', args.C)
        mlflow.log_param('solver', args.solver)
        mlflow.log_param('max_iter', args.max_iter)
        mlflow.log_param('tol', args.tol)
        mlflow.log_param('seed', args.seed)
        mlflow.log_param('test_size', args.test_size)
        mlflow.log_param('imputation', 'median')
        mlflow.log_param('scaling', 'StandardScaler')
        mlflow.log_param('split', 'stratified')
        mlflow.log_param('n_rows', len(df))
        mlflow.log_param('n_train', len(X_train))
        mlflow.log_param('n_test', len(X_test))
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'course': 'TC5061', 'dataset': 'Iris', 'task': 'multiclass classification'})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            matrix = confusion_matrix(y_test, predictions, labels=labels)
            disp = ConfusionMatrixDisplay(matrix, display_labels=labels)
            disp.plot(cmap='Blues', values_format='d', colorbar=False)
            plt.title(f'Matriz de confusión: {run_name}')
            plt.tight_layout()
            plot_file = output / 'confusion_matrix.png'
            plt.savefig(plot_file, dpi=160)
            plt.close('all')
            mlflow.log_artifact(str(plot_file), artifact_path='figures')
            report_file = output / 'classification_report.json'
            report_file.write_text(json.dumps(classification_report(y_test, predictions, output_dict=True, zero_division=0), indent=2), encoding='utf-8')
            mlflow.log_artifact(str(report_file), artifact_path='reports')
        mlflow.sklearn.log_model(model, name='iris_logistic_pipeline')
        run_id = run.info.run_id
    print(json.dumps({'run_id': run_id, 'run_name': run_name, 'tracking_uri': tracking_uri, **metrics}, indent=2))
    return metrics


if __name__ == '__main__':
    execute(parse_args())
