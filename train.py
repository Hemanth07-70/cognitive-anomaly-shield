#!/usr/bin/env python3
"""Train models and save everything needed by the Streamlit app."""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_utils import load_data
from src.features import FEATURE_COLS
from src.models import (
    split_train_val, fit_scaler, scale,
    train_lr, train_xgb, tune_threshold, evaluate, threshold_sweep,
)
from src.explain import get_lr_explainer, get_xgb_explainer, global_importance

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Cognitive Anomaly Shield — training pipeline")
    print("=" * 60)

    # 1. Load data
    train_df, test_df = load_data()
    print(f"Train: {len(train_df):,} rows  ({train_df['is_fraud'].sum():,} frauds, "
          f"{train_df['is_fraud'].mean()*100:.2f}%)")
    print(f"Test : {len(test_df):,} rows  ({test_df['is_fraud'].sum():,} frauds, "
          f"{test_df['is_fraud'].mean()*100:.2f}%)")

    # 2. Train / validation split
    train_split, val_split = split_train_val(train_df)
    X_train = train_split[FEATURE_COLS]
    y_train = train_split["is_fraud"]
    X_val = val_split[FEATURE_COLS]
    y_val = val_split["is_fraud"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["is_fraud"]

    # 3. Scale
    scaler = fit_scaler(train_split)
    X_train_s = scale(X_train, scaler)
    X_val_s   = scale(X_val,   scaler)
    X_test_s  = scale(X_test,  scaler)

    # 4. Train models
    print("\nTraining Logistic Regression…")
    lr = train_lr(X_train_s, y_train)

    print("Training XGBoost…")
    xgb_model = train_xgb(X_train_s, y_train, X_val_s, y_val)

    # 5. Threshold tuning
    lr_thresh  = tune_threshold(lr,        X_val_s, y_val)
    xgb_thresh = tune_threshold(xgb_model, X_val_s, y_val)
    print(f"\nTuned thresholds  LR: {lr_thresh}  XGB: {xgb_thresh}")

    # 6. Evaluate
    lr_results  = evaluate(lr,        X_test_s, y_test, lr_thresh)
    xgb_results = evaluate(xgb_model, X_test_s, y_test, xgb_thresh)

    print("\n--- Logistic Regression ---")
    print(f"  Precision {lr_results['precision']}  Recall {lr_results['recall']}  "
          f"F1 {lr_results['f1']}  PR-AUC {lr_results['pr_auc']}")
    print(f"  TP {lr_results['tp']}  FP {lr_results['fp']}  "
          f"FN {lr_results['fn']}  TN {lr_results['tn']}")

    print("\n--- XGBoost ---")
    print(f"  Precision {xgb_results['precision']}  Recall {xgb_results['recall']}  "
          f"F1 {xgb_results['f1']}  PR-AUC {xgb_results['pr_auc']}")
    print(f"  TP {xgb_results['tp']}  FP {xgb_results['fp']}  "
          f"FN {xgb_results['fn']}  TN {xgb_results['tn']}")

    # 7. SHAP explainers + global importance
    print("\nComputing SHAP global importance…")
    sample_idx = X_test_s.sample(min(2000, len(X_test_s)), random_state=42).index
    X_shap = X_test_s.loc[sample_idx]

    lr_explainer  = get_lr_explainer(lr, X_train_s.sample(min(1000, len(X_train_s)), random_state=42))
    xgb_explainer = get_xgb_explainer(xgb_model)

    lr_importance  = global_importance(lr_explainer,  X_shap)
    xgb_importance = global_importance(xgb_explainer, X_shap)

    # 8. Threshold sweeps
    lr_sweep  = threshold_sweep(lr,        X_test_s, y_test)
    xgb_sweep = threshold_sweep(xgb_model, X_test_s, y_test)

    # 9. Save everything
    artifact = {
        "scaler":        scaler,
        "lr":            lr,
        "xgb":           xgb_model,
        "lr_thresh":     lr_thresh,
        "xgb_thresh":    xgb_thresh,
        "lr_results":    {k: v for k, v in lr_results.items()  if k != "probs"},
        "xgb_results":   {k: v for k, v in xgb_results.items() if k != "probs"},
        "lr_importance": lr_importance,
        "xgb_importance":xgb_importance,
        "lr_sweep":      lr_sweep,
        "xgb_sweep":     xgb_sweep,
        "n_test":        len(test_df),
        "n_frauds":      int(y_test.sum()),
        "fraud_rate":    float(y_test.mean()),
        "lr_explainer":  lr_explainer,
        "xgb_explainer": xgb_explainer,
        # keep a small test sample for the dashboard demo
        "X_test_sample": X_test_s.head(500).reset_index(drop=True),
        "y_test_sample": y_test.head(500).reset_index(drop=True),
    }

    save_path = os.path.join(MODELS_DIR, "artifact.joblib")
    joblib.dump(artifact, save_path)
    print(f"\nSaved to {save_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
