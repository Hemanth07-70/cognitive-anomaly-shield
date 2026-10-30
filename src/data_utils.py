"""Load Kaggle data or generate synthetic data matching report statistics."""

import os
import numpy as np
import pandas as pd
from src.features import (
    CATEGORIES, CATEGORY_MAP, CATEGORY_COLS,
    engineer_from_kaggle, engineer_from_synthetic, FEATURE_COLS,
)

KAGGLE_DATASET = "kartik2112/fraud-detection"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_kaggle_data() -> bool:
    """Return True if kaggle download succeeded."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        return False
    try:
        import kaggle  # noqa: F401
        os.makedirs(DATA_DIR, exist_ok=True)
        os.system(
            f'kaggle datasets download -d {KAGGLE_DATASET} -p "{DATA_DIR}" --unzip'
        )
        return (
            os.path.exists(os.path.join(DATA_DIR, "fraudTrain.csv"))
            and os.path.exists(os.path.join(DATA_DIR, "fraudTest.csv"))
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synthetic data generation (matches report statistics)
# ---------------------------------------------------------------------------

# Fraud rates per amount band from the test-set table in the report
_BAND_FRAUD_RATES = {
    "under_10":  0.0010,
    "10_25":     0.0054,
    "25_200":    0.0003,
    "200_500":   0.0298,
    "500_800":   0.0718,
    "800_1200":  0.4541,
    "over_1200": 0.0073,
}

# Approximate category transaction mix (uniform-ish, with shopping_net heavier)
_CATEGORY_WEIGHTS = {
    "shopping_net":  0.085,
    "shopping_pos":  0.100,
    "misc_net":      0.055,
    "misc_pos":      0.070,
    "grocery_pos":   0.106,
    "grocery_net":   0.039,
    "gas_transport": 0.114,
    "food_dining":   0.079,
    "entertainment": 0.081,
    "personal_care": 0.079,
    "health_fitness": 0.074,
    "travel":        0.035,
    "kids_pets":     0.098,
    "home":          0.106,
}

_CATEGORY_FRAUD_RATES = {
    "shopping_net":  0.0121,
    "misc_net":      0.0098,
    "grocery_pos":   0.0092,
    "shopping_pos":  0.0043,
    "gas_transport": 0.0027,
    "travel":        0.0023,
    "grocery_net":   0.0021,
    "misc_pos":      0.0021,
    "personal_care": 0.0018,
    "entertainment": 0.0015,
    "food_dining":   0.0014,
    "health_fitness": 0.0014,
    "home":          0.0013,
    "kids_pets":     0.0013,
}


def _sample_amount_for_band(band: str, n: int, rng: np.random.Generator) -> np.ndarray:
    limits = {
        "under_10":  (1, 10),
        "10_25":     (10, 25),
        "25_200":    (25, 200),
        "200_500":   (200, 500),
        "500_800":   (500, 800),
        "800_1200":  (800, 1200),
        "over_1200": (1200, 2000),
    }
    lo, hi = limits[band]
    return rng.uniform(lo, hi, n)


def generate_synthetic(n_train: int = 300_000, n_test: int = 100_000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)

    def _make_split(n: int) -> pd.DataFrame:
        bands = list(_BAND_FRAUD_RATES.keys())
        band_weights = np.array([0.26, 0.10, 0.60, 0.035, 0.006, 0.003, 0.002])
        band_weights /= band_weights.sum()

        assigned_bands = rng.choice(bands, size=n, p=band_weights)
        amounts = np.zeros(n)
        for band in bands:
            mask = assigned_bands == band
            amounts[mask] = _sample_amount_for_band(band, int(mask.sum()), rng)

        # Night/day: 23% transactions are night-time
        is_night = rng.random(n) < 0.234
        is_weekend = rng.random(n) < 0.286

        # Cardholder
        customer_age = rng.integers(18, 85, n)
        is_male = rng.random(n) < 0.493
        city_population = rng.choice(
            [5000, 20000, 80000, 300000, 800000, 2000000],
            size=n,
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05],
        ).astype(float)
        city_population += rng.uniform(-1000, 1000, n)
        city_population = np.clip(city_population, 100, 5_000_000)
        distance_km = np.abs(rng.normal(78, 60, n))

        cats = list(_CATEGORY_WEIGHTS.keys())
        cat_p = np.array([_CATEGORY_WEIGHTS[c] for c in cats])
        cat_p /= cat_p.sum()
        assigned_cats = rng.choice(cats, size=n, p=cat_p)

        # Fraud label: combine band rate and category rate (take max to keep rate realistic)
        band_fraud_p = np.array([_BAND_FRAUD_RATES[b] for b in assigned_bands])
        cat_fraud_p = np.array([_CATEGORY_FRAUD_RATES[c] for c in assigned_cats])
        # Night multiplier ~17x
        night_mult = np.where(is_night, 6.0, 0.35)
        fraud_p = np.clip(band_fraud_p * night_mult + cat_fraud_p * night_mult * 0.3, 0, 0.95)
        is_fraud = (rng.random(n) < fraud_p).astype(int)

        df = pd.DataFrame({
            "amount": amounts,
            "is_night": is_night.astype(int),
            "is_weekend": is_weekend.astype(int),
            "customer_age": customer_age,
            "is_male": is_male.astype(int),
            "city_population": city_population,
            "distance_km": distance_km,
            "category": assigned_cats,
            "is_fraud": is_fraud,
        })
        return df

    def _expand(df: pd.DataFrame) -> pd.DataFrame:
        for cat_raw, col in CATEGORY_MAP.items():
            df[col] = (df["category"] == cat_raw).astype(int)
        df = df.drop(columns=["category"])
        from src.features import _amount_bands
        df = _amount_bands(df)
        return df[FEATURE_COLS + ["is_fraud"]]

    train_raw = _make_split(n_train)
    test_raw = _make_split(n_test)
    return _expand(train_raw), _expand(test_raw)


def load_data(force_synthetic: bool = False):
    """Return (train_df, test_df). Tries Kaggle first, falls back to synthetic."""
    train_path = os.path.join(DATA_DIR, "fraudTrain.csv")
    test_path = os.path.join(DATA_DIR, "fraudTest.csv")

    if not force_synthetic and os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading Kaggle data…")
        train = engineer_from_kaggle(pd.read_csv(train_path))
        test = engineer_from_kaggle(pd.read_csv(test_path))
        return train, test

    if not force_synthetic:
        print("Kaggle data not found. Trying download…")
        if download_kaggle_data():
            train = engineer_from_kaggle(pd.read_csv(train_path))
            test = engineer_from_kaggle(pd.read_csv(test_path))
            return train, test

    print("Generating synthetic data…")
    return generate_synthetic()
