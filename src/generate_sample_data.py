"""
OPTIONAL: creates SYNTHETIC data in the same raw format as the Kaggle
"Retail Price Optimization" dataset (retail_price.csv), for testing the pipeline.
Use the real Kaggle file for your final portfolio.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
cats = {"bed_bath_table": 6, "garden_tools": 6, "consoles_games": 5, "health_beauty": 7,
        "cool_stuff": 6, "perfumery": 5, "computers_accessories": 7, "watches_gifts": 6,
        "furniture_decor": 6}
months = pd.date_range("2017-01-01", periods=20, freq="MS")

rows = []
for cat, n in cats.items():
    for i in range(1, n + 1):
        pid = f"{cat.split('_')[0]}{i}"
        base_price = rng.uniform(15, 300)
        elasticity = rng.uniform(-3.0, -0.4)       # true (hidden) elasticity
        base_qty = rng.uniform(8, 80)
        score = rng.uniform(3.6, 4.7)
        freight = base_price * rng.uniform(0.08, 0.3)
        comp_off = rng.uniform(0.85, 1.25, 3)
        prev_price = base_price
        for m in months:
            price = base_price * np.exp(rng.normal(0, 0.08))
            comps = base_price * comp_off * np.exp(rng.normal(0, 0.05, 3))
            holiday = int(rng.integers(0, 3))
            weekend = int(rng.integers(8, 10))
            seasonal = 1 + 0.25 * (m.month in (11, 12))
            mu = (base_qty * (price / base_price) ** elasticity
                  * (comps.mean() / base_price) ** 0.5 * seasonal * (1 + 0.05 * holiday))
            qty = max(1, int(rng.poisson(mu)))
            rows.append({
                "product_id": pid, "product_category_name": cat,
                "month_year": m.strftime("%d-%m-%Y"), "qty": qty,
                "total_price": round(qty * price, 2),
                "freight_price": round(freight * np.exp(rng.normal(0, 0.05)), 2),
                "unit_price": round(price, 2),
                "product_score": round(score, 1), "customers": int(qty * rng.uniform(1.5, 4)),
                "weekend": weekend, "holiday": holiday,
                "comp_1": round(comps[0], 2), "comp_2": round(comps[1], 2), "comp_3": round(comps[2], 2),
                "lag_price": round(prev_price, 2),
            })
            prev_price = price

df = pd.DataFrame(rows)
out = Path("data/raw/sample_retail_price.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Synthetic data saved to {out} ({len(df)} rows, {df.product_id.nunique()} products)")
