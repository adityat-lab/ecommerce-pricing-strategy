"""Step 3: price elasticity of demand using log-log regression: ln(qty) = a + e * ln(price)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_OBS = 8           # minimum months of data per product
MIN_PRICE_VAR = 0.01  # price must actually change (std of log price)

df = pd.read_csv("data/processed/clean.csv", parse_dates=["date"])
df["lp"], df["lq"] = np.log(df.unit_price), np.log(df.qty)
Path("reports").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)


def label(e):
    if e <= -1:
        return "Elastic (price-sensitive)"
    if e < 0:
        return "Inelastic"
    return "Unreliable (positive)"


# ---- per product
rows = []
for pid, g in df.groupby("product_id"):
    n = len(g)
    if n < MIN_OBS or g.lp.std() < MIN_PRICE_VAR:
        continue
    X = np.column_stack([np.ones(n), g.lp.values])
    y = g.lq.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    ss_res, ss_tot = (res ** 2).sum(), ((y - y.mean()) ** 2).sum()
    se = np.sqrt(ss_res / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
    rows.append({"product_id": pid, "category": g.category.iloc[0], "n_obs": n,
                 "elasticity": beta[1], "std_err": se, "t_stat": beta[1] / se if se > 0 else np.nan,
                 "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0,
                 "avg_price": g.unit_price.mean(), "avg_qty": g.qty.mean()})
prod = pd.DataFrame(rows)
if prod.empty:
    raise SystemExit("Not enough price variation or observations per product to estimate elasticity.")
prod["type"] = prod.elasticity.apply(label)
prod = prod.sort_values("elasticity").round(3)
prod.to_csv("reports/elasticity_products.csv", index=False)

# ---- per category (pooled, with product fixed effects = demeaning inside each product)
crow = []
for cat, gc in df.groupby("category"):
    ok = [g for _, g in gc.groupby("product_id") if len(g) >= MIN_OBS and g.lp.std() >= MIN_PRICE_VAR]
    if not ok:
        continue
    x = np.concatenate([g.lp.values - g.lp.mean() for g in ok])
    y = np.concatenate([g.lq.values - g.lq.mean() for g in ok])
    sxx = (x ** 2).sum()
    b = (x * y).sum() / sxx
    dof = max(len(x) - len(ok) - 1, 1)
    se = np.sqrt(((y - b * x) ** 2).sum() / dof / sxx)
    crow.append({"category": cat, "n_products": len(ok), "n_obs": len(x),
                 "elasticity": b, "std_err": se, "t_stat": b / se})
cat = pd.DataFrame(crow).sort_values("elasticity").round(3)
cat["type"] = cat.elasticity.apply(label)
cat.to_csv("reports/elasticity_categories.csv", index=False)

plt.figure(figsize=(8, 4.5))
plt.barh(cat.category, cat.elasticity, xerr=1.96 * cat.std_err, color="#E8590C", capsize=3)
plt.axvline(-1, color="black", ls="--", lw=1)
plt.title("Price Elasticity by Category (dashed = -1)"); plt.xlabel("Elasticity")
plt.tight_layout(); plt.savefig("images/09_elasticity_by_category.png", dpi=130); plt.close()

print("\nCategory elasticity:\n", cat.to_string(index=False))
print(f"\nProducts with an estimate: {len(prod)} | elastic (< -1): {(prod.elasticity <= -1).sum()} "
      f"| inelastic: {((prod.elasticity > -1) & (prod.elasticity < 0)).sum()} "
      f"| positive/unreliable: {(prod.elasticity >= 0).sum()}")
print("\nMost price-sensitive products:\n", prod.head(5)[["product_id", "elasticity", "t_stat", "n_obs"]].to_string(index=False))
