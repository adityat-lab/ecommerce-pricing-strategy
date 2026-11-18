"""Step 4: recommend prices that maximise profit, using the elasticity estimates.

Demand model:  q(p) = q0 * (p / p0) ** elasticity
Profit:        (p - unit_cost) * q(p),   unit_cost = COST_RATIO * current price  (ASSUMPTION)
The dataset has no product cost column, so change --cost-ratio to match your margins.
Prices are limited to +/-15% of the current price and to the range seen in the data.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cost-ratio", type=float, default=0.60, help="unit cost as a share of current price")
args = ap.parse_args()

df = pd.read_csv("data/processed/clean.csv", parse_dates=["date"]).sort_values("date")
prod = pd.read_csv("reports/elasticity_products.csv").set_index("product_id")
cat = pd.read_csv("reports/elasticity_categories.csv").set_index("category")

rows = []
for pid, g in df.groupby("product_id"):
    last = g.tail(3)
    p0, q0 = last.unit_price.mean(), last.qty.mean()
    cost = args.cost_ratio * p0
    category = g.category.iloc[0]

    e, source, conf = None, None, "Low"
    if pid in prod.index and prod.loc[pid, "elasticity"] < 0 and prod.loc[pid, "t_stat"] < -1:
        e, source = prod.loc[pid, "elasticity"], "product"
        conf = "High" if prod.loc[pid, "t_stat"] < -2 else "Medium"
    elif category in cat.index and cat.loc[category, "elasticity"] < 0:
        e, source = cat.loc[category, "elasticity"], "category"

    rec = {"product_id": pid, "category": category, "current_price": round(p0, 2),
           "avg_comp_price": round(last.comp_avg.mean(), 2) if last.comp_avg.notna().any() else np.nan}
    if e is None:
        rec.update(recommended_price=round(p0, 2), price_change_pct=0.0, elasticity=np.nan,
                   elasticity_source="none", confidence="Low", action="Hold (no reliable elasticity)",
                   qty_change_pct=0.0, revenue_change_pct=0.0, profit_change_pct=0.0)
        rows.append(rec)
        continue

    lo = max(0.85 * p0, 0.95 * g.unit_price.min())
    hi = min(1.15 * p0, 1.05 * g.unit_price.max())
    grid = np.append(np.linspace(lo, hi, 200), p0)
    q = q0 * (grid / p0) ** e
    profit = (grid - cost) * q
    best = int(np.argmax(profit))
    p1, q1 = grid[best], q[best]
    chg = (p1 / p0 - 1) * 100
    action = "Hold" if abs(chg) < 1 else ("Increase price" if chg > 0 else "Decrease price")
    base_profit = (p0 - cost) * q0
    rec.update(recommended_price=round(p1, 2), price_change_pct=round(chg, 1), elasticity=round(e, 2),
               elasticity_source=source, confidence=conf, action=action,
               qty_change_pct=round((q1 / q0 - 1) * 100, 1),
               revenue_change_pct=round((p1 * q1 / (p0 * q0) - 1) * 100, 1),
               profit_change_pct=round(((p1 - cost) * q1 / base_profit - 1) * 100, 1) if base_profit > 0 else np.nan)
    rows.append(rec)

out = pd.DataFrame(rows).sort_values("profit_change_pct", ascending=False)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/price_recommendations.csv", index=False)

solid = out[out.confidence.isin(["High", "Medium"])]
print(f"Cost ratio assumed: {args.cost_ratio:.0%} of current price")
print(f"Recommendations: {len(out)} products | High/Medium confidence: {len(solid)}")
print(out.action.value_counts().to_string())
print("\nTop 10 by expected profit uplift:")
print(out.head(10)[["product_id", "current_price", "recommended_price", "price_change_pct",
                    "elasticity", "confidence", "profit_change_pct"]].to_string(index=False))
if len(solid):
    print(f"\nAverage expected profit change on High/Medium confidence products: {solid.profit_change_pct.mean():+.1f}%")
    print("These are model estimates from historical data, not guaranteed results. Validate with A/B tests.")

plt.figure(figsize=(7, 4.5))
plt.hist(out.price_change_pct, bins=25, color="#12B886")
plt.title("Recommended Price Changes (%)"); plt.xlabel("% change vs current price")
plt.tight_layout(); plt.savefig("images/10_recommended_changes.png", dpi=130)
