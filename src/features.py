"""Feature engineering: turns raw transaction columns into the 28 readable features."""

import pandas as pd
import numpy as np

AMOUNT_BAND_COLS = [
    "amt_under_10", "amt_10_25", "amt_25_200",
    "amt_200_500", "amt_500_800", "amt_800_1200", "amt_over_1200",
]

CATEGORY_COLS = [
    "cat_shopping_net", "cat_shopping_pos", "cat_misc_net", "cat_misc_pos",
    "cat_grocery_pos", "cat_grocery_net", "cat_gas_transport", "cat_food_dining",
    "cat_entertainment", "cat_personal_care", "cat_health_fitness",
    "cat_travel", "cat_kids_pets", "cat_home",
]

FEATURE_COLS = (
    ["amount", "is_night", "is_weekend", "customer_age", "is_male", "city_population", "distance_km"]
    + AMOUNT_BAND_COLS
    + CATEGORY_COLS
)

CATEGORY_MAP = {
    "shopping_net": "cat_shopping_net",
    "shopping_pos": "cat_shopping_pos",
    "misc_net": "cat_misc_net",
    "misc_pos": "cat_misc_pos",
    "grocery_pos": "cat_grocery_pos",
    "grocery_net": "cat_grocery_net",
    "gas_transport": "cat_gas_transport",
    "food_dining": "cat_food_dining",
    "entertainment": "cat_entertainment",
    "personal_care": "cat_personal_care",
    "health_fitness": "cat_health_fitness",
    "travel": "cat_travel",
    "kids_pets": "cat_kids_pets",
    "home": "cat_home",
}

CATEGORIES = list(CATEGORY_MAP.keys())


def _amount_bands(df: pd.DataFrame) -> pd.DataFrame:
    amt = df["amount"]
    df["amt_under_10"]  = (amt < 10).astype(int)
    df["amt_10_25"]     = ((amt >= 10)  & (amt < 25)).astype(int)
    df["amt_25_200"]    = ((amt >= 25)  & (amt < 200)).astype(int)
    df["amt_200_500"]   = ((amt >= 200) & (amt < 500)).astype(int)
    df["amt_500_800"]   = ((amt >= 500) & (amt < 800)).astype(int)
    df["amt_800_1200"]  = ((amt >= 800) & (amt < 1200)).astype(int)
    df["amt_over_1200"] = (amt >= 1200).astype(int)
    return df


def engineer_from_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw Kaggle CSV columns."""
    out = pd.DataFrame()
    out["amount"] = df["amt"].astype(float)

    # Time features
    trans_dt = pd.to_datetime(df["trans_date_trans_time"])
    hour = trans_dt.dt.hour
    out["is_night"] = ((hour >= 22) | (hour < 4)).astype(int)
    out["is_weekend"] = (trans_dt.dt.dayofweek >= 5).astype(int)

    # Cardholder
    dob = pd.to_datetime(df["dob"])
    trans_year = trans_dt.dt.year
    out["customer_age"] = (trans_year - dob.dt.year).astype(int)
    out["is_male"] = (df["gender"].str.upper() == "M").astype(int)
    out["city_population"] = df["city_pop"].astype(float)

    # Distance (Haversine simplified)
    lat1, lon1 = np.radians(df["lat"].astype(float)), np.radians(df["long"].astype(float))
    lat2, lon2 = np.radians(df["merch_lat"].astype(float)), np.radians(df["merch_long"].astype(float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    out["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    out = _amount_bands(out)

    for raw_cat, col in CATEGORY_MAP.items():
        out[col] = (df["category"] == raw_cat).astype(int)

    out["is_fraud"] = df["is_fraud"].astype(int)
    return out[FEATURE_COLS + ["is_fraud"]]


def engineer_from_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Process already-engineered synthetic data (pass-through with band computation)."""
    out = df.copy()
    out = _amount_bands(out)
    for col in CATEGORY_COLS:
        if col not in out.columns:
            out[col] = 0
    return out[FEATURE_COLS + ["is_fraud"]]


def build_single_transaction(
    amount: float,
    hour: int,
    is_weekend: int,
    category: str,
    customer_age: int,
    is_male: int,
    city_population: float,
    distance_km: float,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame from user inputs."""
    row = {
        "amount": amount,
        "is_night": int((hour >= 22) or (hour < 4)),
        "is_weekend": is_weekend,
        "customer_age": customer_age,
        "is_male": is_male,
        "city_population": city_population,
        "distance_km": distance_km,
    }
    for band in AMOUNT_BAND_COLS:
        row[band] = 0
    if amount < 10:
        row["amt_under_10"] = 1
    elif amount < 25:
        row["amt_10_25"] = 1
    elif amount < 200:
        row["amt_25_200"] = 1
    elif amount < 500:
        row["amt_200_500"] = 1
    elif amount < 800:
        row["amt_500_800"] = 1
    elif amount < 1200:
        row["amt_800_1200"] = 1
    else:
        row["amt_over_1200"] = 1

    for raw_cat, col in CATEGORY_MAP.items():
        row[col] = int(raw_cat == category)

    return pd.DataFrame([row])[FEATURE_COLS]
