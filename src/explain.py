"""SHAP explanations for LR and XGBoost."""

import numpy as np
import pandas as pd
import shap
from src.features import FEATURE_COLS

FEATURE_LABELS = {
    "amount":          "Transaction amount ($)",
    "is_night":        "Night-time (22:00–03:59)",
    "is_weekend":      "Weekend",
    "customer_age":    "Cardholder age",
    "is_male":         "Cardholder is male",
    "city_population": "Home city population",
    "distance_km":     "Distance to merchant (km)",
    "amt_under_10":    "Amount band: under $10",
    "amt_10_25":       "Amount band: $10–25",
    "amt_25_200":      "Amount band: $25–200",
    "amt_200_500":     "Amount band: $200–500",
    "amt_500_800":     "Amount band: $500–800",
    "amt_800_1200":    "Amount band: $800–1,200",
    "amt_over_1200":   "Amount band: over $1,200",
    "cat_shopping_net": "Merchant: online shopping",
    "cat_shopping_pos": "Merchant: in-store shopping",
    "cat_misc_net":     "Merchant: online misc.",
    "cat_misc_pos":     "Merchant: in-store misc.",
    "cat_grocery_pos":  "Merchant: grocery (store)",
    "cat_grocery_net":  "Merchant: grocery (online)",
    "cat_gas_transport":"Merchant: gas & transport",
    "cat_food_dining":  "Merchant: food & dining",
    "cat_entertainment":"Merchant: entertainment",
    "cat_personal_care":"Merchant: personal care",
    "cat_health_fitness":"Merchant: health & fitness",
    "cat_travel":       "Merchant: travel",
    "cat_kids_pets":    "Merchant: kids & pets",
    "cat_home":         "Merchant: home",
}


def get_lr_explainer(model, X_background: pd.DataFrame):
    return shap.LinearExplainer(model, X_background, feature_perturbation="interventional")


def get_xgb_explainer(model):
    return shap.TreeExplainer(model)


def shap_values_single(explainer, X_single: pd.DataFrame) -> np.ndarray:
    sv = explainer.shap_values(X_single)
    if isinstance(sv, list):
        sv = sv[1]
    return sv[0] if sv.ndim == 2 else sv


def global_importance(explainer, X_sample: pd.DataFrame) -> pd.DataFrame:
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label":   [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
        "importance": mean_abs,
    }).sort_values("importance", ascending=False)
    return df


def plain_english_reasons(shap_vals: np.ndarray, row: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Return top_n plain-English reason strings (positive = toward fraud)."""
    pairs = sorted(zip(FEATURE_COLS, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feat, val in pairs[:top_n]:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "raised" if val > 0 else "lowered"
        reasons.append(
            f"**{label}** {direction} the fraud score by {abs(val):.2f} log-odds."
        )
    return reasons
