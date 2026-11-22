"""Training and prediction logic for the tuned credit-default random forest.

This module has no Gradio dependency, so the CI pipeline can test it quickly.
Hyperparameters are the best combination from the Module 8 randomised search.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DATA_FILE = "default_of_credit_card_clients.csv"
DATA_PATH = Path(__file__).parent / "data" / DATA_FILE
if not DATA_PATH.exists():  # in the browser (Gradio-Lite) the CSV sits beside this file
    DATA_PATH = Path(__file__).parent / DATA_FILE

# The browser version of Python (Pyodide) has no multiprocessing, so use one core there.
IN_BROWSER = sys.platform == "emscripten"
SEED = 20
THRESHOLD = 0.5  # the tuned threshold (0.52) added nothing in Module 8

FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
TARGET = "default"

BEST_PARAMS = dict(
    n_estimators=142,
    max_depth=20,
    min_samples_leaf=15,
    max_features="log2",
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=1 if IN_BROWSER else -1,
)


def load_data(path=DATA_PATH):
    """Load the UCI dataset from CSV and return (X, y).

    Accepts the usual CSV versions of the dataset: the target column may be
    called 'default payment next month' or 'default.payment.next.month'.
    """
    df = pd.read_csv(path)
    target_cols = [c for c in df.columns if str(c).lower().startswith("default")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected one target column starting with 'default', found {target_cols}")
    df = df.rename(columns={target_cols[0]: TARGET})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df[FEATURES], df[TARGET].astype(int)


def split(X, y):
    """The same 80/20 stratified split (seed 20) used in every module."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def train(X_train, y_train):
    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "n_test": int(len(y_test)),
    }


def build(path=DATA_PATH):
    """Train on the training split and evaluate on the held-out test split."""
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = split(X, y)
    model = train(X_train, y_train)
    return model, evaluate(model, X_test, y_test)


def predict_one(model, values: dict):
    """Return the probability of default for one client given as {feature: value}."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}], columns=FEATURES)
    return float(model.predict_proba(row)[0, 1])


if __name__ == "__main__":
    _, metrics = build()
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}" if isinstance(value, float) else f"{name:>10}: {value}")
