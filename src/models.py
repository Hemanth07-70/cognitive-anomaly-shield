"""Train, threshold-tune, and evaluate LR + XGBoost models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.features import FEATURE_COLS

NUMERIC_COLS = ["amount", "customer_age", "city_population", "distance_km"]


def split_train_val(train_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    return train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df["is_fraud"]
    )


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLS])
    return scaler


def scale(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return out


def train_lr(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    fraud_rate = y_train.mean()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def tune_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Find threshold that maximises F1 on the validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.999, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 3)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 3),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 3),
        "roc_auc":   round(roc_auc_score(y_test, probs), 3),
        "pr_auc":    round(average_precision_score(y_test, probs), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
        "probs": probs,
    }


def threshold_sweep(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Return a DataFrame of metrics at various thresholds."""
    probs = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in np.arange(0.01, 0.999, 0.01):
        preds = (probs >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f,
                     "frauds_caught": int(tp), "false_alarms": int(fp)})
    return pd.DataFrame(rows)
