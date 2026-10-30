#!/usr/bin/env python3
"""
Generate fraudTrain.csv from fraudTest.csv cardholder/merchant profiles.

Same 924 cardholders, same 693 merchants, transactions over Jan 2019 – Jun 2020.
Matches the original Sparkov dataset schema exactly.
Target: ~1,296,675 rows, fraud rate ~0.58%.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TEST_CSV  = os.path.expanduser("~/Downloads/fraudTest.csv")
OUT_CSV   = os.path.expanduser("~/Downloads/fraudTrain.csv")
SEED      = 42
N_TARGET  = 1_296_675
FRAUD_RATE = 0.0058   # 0.58% as in the report

# Fraud amount distribution (from report: median $371.94 for fraud)
FRAUD_AMT_BANDS = {
    "under_10":  (1, 10,    0.02),
    "10_25":     (10, 25,   0.04),
    "25_200":    (25, 200,  0.02),
    "200_500":   (200, 500, 0.18),
    "500_800":   (500, 800, 0.22),
    "800_1200":  (800, 1200,0.50),
    "over_1200": (1200,1321, 0.02),
}

# Legitimate amount distribution per category (mean, std from test set analysis)
LEGIT_AMT = {
    "entertainment":  (63, 45),
    "food_dining":    (50, 35),
    "gas_transport":  (64, 30),
    "grocery_net":    (54, 38),
    "grocery_pos":    (114, 70),
    "health_fitness": (54, 38),
    "home":           (58, 42),
    "kids_pets":      (58, 40),
    "misc_net":       (71, 90),
    "misc_pos":       (62, 85),
    "personal_care":  (48, 35),
    "shopping_net":   (72, 100),
    "shopping_pos":   (73, 95),
    "travel":         (113, 130),
}

CATEGORY_WEIGHTS = {
    "gas_transport": 0.1014, "grocery_pos": 0.0946, "home": 0.0942,
    "shopping_pos":  0.0896, "kids_pets":   0.0876, "shopping_net": 0.0752,
    "entertainment": 0.0722, "personal_care": 0.0708, "food_dining":  0.0707,
    "health_fitness": 0.0660, "misc_pos":    0.0622, "misc_net": 0.0492,
    "grocery_net":   0.0350, "travel":      0.0314,
}

# Night-time fraud multiplier (from report: ~17x more fraud at night)
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 4))


def rand_amount_fraud(rng):
    bands = list(FRAUD_AMT_BANDS.values())
    weights = [b[2] for b in bands]
    idx = rng.choice(len(bands), p=np.array(weights) / sum(weights))
    lo, hi, _ = bands[idx]
    return round(float(rng.uniform(lo, hi)), 2)


def rand_amount_legit(category, rng):
    mu, sigma = LEGIT_AMT.get(category, (60, 50))
    val = rng.normal(mu, sigma)
    return round(max(1.0, min(val, 500.0)), 2)


def rand_timestamp(start_ts: float, end_ts: float, rng) -> datetime:
    ts = rng.uniform(start_ts, end_ts)
    return datetime.fromtimestamp(ts)


def make_trans_num(cc_num, ts_str, rng) -> str:
    raw = f"{cc_num}{ts_str}{rng.integers(0, 1_000_000)}"
    return hashlib.md5(raw.encode()).hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    print("Reading fraudTest.csv for cardholder and merchant profiles…")
    test_df = pd.read_csv(TEST_CSV)

    cardholders = (
        test_df[["cc_num","first","last","gender","street","city","state",
                 "zip","lat","long","city_pop","job","dob"]]
        .drop_duplicates("cc_num")
        .reset_index(drop=True)
    )

    merchants = (
        test_df[["merchant","category","merch_lat","merch_long"]]
        .drop_duplicates("merchant")
        .reset_index(drop=True)
    )

    # Build category → merchant lookup
    cat_to_merchants = {}
    for cat in CATEGORY_WEIGHTS:
        m_sub = merchants[merchants["category"] == cat].reset_index(drop=True)
        cat_to_merchants[cat] = m_sub

    n_cardholders = len(cardholders)
    n_merchants   = len(merchants)

    # Time window: Jan 1 2019 00:00:00 → Jun 20 2020 23:59:59
    start_ts = datetime(2019, 1, 1, 0, 0, 0).timestamp()
    end_ts   = datetime(2020, 6, 20, 23, 59, 59).timestamp()

    cats     = list(CATEGORY_WEIGHTS.keys())
    cat_p    = np.array([CATEGORY_WEIGHTS[c] for c in cats])
    cat_p   /= cat_p.sum()

    n_fraud  = int(N_TARGET * FRAUD_RATE)
    n_legit  = N_TARGET - n_fraud

    print(f"Generating {N_TARGET:,} transactions ({n_fraud:,} fraud, {n_legit:,} legit)…")

    rows = []
    idx  = 0

    def pick_cardholder(i):
        return cardholders.iloc[i % n_cardholders]

    def pick_merchant(category, i):
        m_sub = cat_to_merchants.get(category)
        if m_sub is None or len(m_sub) == 0:
            return merchants.iloc[i % n_merchants]
        return m_sub.iloc[i % len(m_sub)]

    # --- Fraud transactions ---
    print("  Generating fraud transactions…")
    fraud_card_indices = rng.integers(0, n_cardholders, n_fraud)
    # Fraud is concentrated at night
    fraud_night_mask = rng.random(n_fraud) < 0.84   # 84% of fraud is at night
    fraud_cats = rng.choice(cats, size=n_fraud, p=cat_p)

    for i in range(n_fraud):
        card  = cardholders.iloc[int(fraud_card_indices[i])]
        cat   = fraud_cats[i]
        merch = pick_merchant(cat, i)

        if fraud_night_mask[i]:
            # Force night hour
            night_h = int(rng.choice(list(NIGHT_HOURS)))
            dt = datetime.fromtimestamp(rng.uniform(start_ts, end_ts))
            dt = dt.replace(hour=night_h, minute=int(rng.integers(0, 60)),
                            second=int(rng.integers(0, 60)))
        else:
            dt = rand_timestamp(start_ts, end_ts, rng)

        amt = rand_amount_fraud(rng)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "": idx,
            "trans_date_trans_time": ts_str,
            "cc_num":   card["cc_num"],
            "merchant": merch["merchant"],
            "category": cat,
            "amt":      amt,
            "first":    card["first"],
            "last":     card["last"],
            "gender":   card["gender"],
            "street":   card["street"],
            "city":     card["city"],
            "state":    card["state"],
            "zip":      card["zip"],
            "lat":      card["lat"],
            "long":     card["long"],
            "city_pop": card["city_pop"],
            "job":      card["job"],
            "dob":      card["dob"],
            "trans_num": make_trans_num(card["cc_num"], ts_str, rng),
            "unix_time": int(dt.timestamp()),
            "merch_lat":  merch["merch_lat"],
            "merch_long": merch["merch_long"],
            "is_fraud": 1,
        })
        idx += 1

    # --- Legitimate transactions ---
    print("  Generating legitimate transactions…")
    chunk = 100_000
    legit_card_indices = rng.integers(0, n_cardholders, n_legit)
    legit_cats = rng.choice(cats, size=n_legit, p=cat_p)
    legit_ts   = rng.uniform(start_ts, end_ts, n_legit)

    for i in range(n_legit):
        card  = cardholders.iloc[int(legit_card_indices[i])]
        cat   = legit_cats[i]
        merch = pick_merchant(cat, i)
        dt    = datetime.fromtimestamp(legit_ts[i])
        amt   = rand_amount_legit(cat, rng)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "": idx,
            "trans_date_trans_time": ts_str,
            "cc_num":   card["cc_num"],
            "merchant": merch["merchant"],
            "category": cat,
            "amt":      amt,
            "first":    card["first"],
            "last":     card["last"],
            "gender":   card["gender"],
            "street":   card["street"],
            "city":     card["city"],
            "state":    card["state"],
            "zip":      card["zip"],
            "lat":      card["lat"],
            "long":     card["long"],
            "city_pop": card["city_pop"],
            "job":      card["job"],
            "dob":      card["dob"],
            "trans_num": make_trans_num(card["cc_num"], ts_str, rng),
            "unix_time": int(dt.timestamp()),
            "merch_lat":  merch["merch_lat"],
            "merch_long": merch["merch_long"],
            "is_fraud": 0,
        })
        idx += 1

        if (i + 1) % chunk == 0:
            print(f"    {i+1:,}/{n_legit:,} legit done…")

    print("Sorting by timestamp…")
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("unix_time").reset_index(drop=True)
    out_df[""] = out_df.index

    print(f"Saving to {OUT_CSV}…")
    out_df.to_csv(OUT_CSV, index=False)

    fraud_actual = int(out_df["is_fraud"].sum())
    print(f"\nDone! {len(out_df):,} rows, {fraud_actual:,} frauds "
          f"({fraud_actual/len(out_df)*100:.2f}%)")
    print(f"Date range: {out_df['trans_date_trans_time'].min()} → "
          f"{out_df['trans_date_trans_time'].max()}")


if __name__ == "__main__":
    main()
