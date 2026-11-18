"""Step 7: export product data for the interactive web page (docs/data.json).
Run after clean_data.py and elasticity.py."""
import json
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/clean.csv", parse_dates=["date"]).sort_values("date")
prod = pd.read_csv("reports/elasticity_products.csv").set_index("product_id")
cat = pd.read_csv("reports/elasticity_categories.csv").set_index("category")

out = []
for pid, g in df.groupby("product_id"):
    last, category = g.tail(3), g.category.iloc[0]
    e, conf = -1.0, "Low (default -1.0)"
    if pid in prod.index and prod.loc[pid, "elasticity"] < 0 and prod.loc[pid, "t_stat"] < -1:
        e = float(prod.loc[pid, "elasticity"])
        conf = "High" if prod.loc[pid, "t_stat"] < -2 else "Medium"
    elif category in cat.index and cat.loc[category, "elasticity"] < 0:
        e, conf = float(cat.loc[category, "elasticity"]), "Low (category estimate)"
    comps = [round(float(last[c].mean()), 2) for c in ["comp_1", "comp_2", "comp_3"] if last[c].notna().any()]
    out.append({"id": pid, "cat": category, "p0": round(float(last.unit_price.mean()), 2),
                "q0": round(float(last.qty.mean()), 1), "e": round(e, 2), "comp": comps, "conf": conf})

Path("docs").mkdir(exist_ok=True)
Path("docs/data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"Exported {len(out)} products to docs/data.json")
