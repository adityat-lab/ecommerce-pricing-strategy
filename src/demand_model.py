"""Step 5: demand forecasting model. Predict monthly quantity from price and market features.
Uses a time-based split (train on the earliest 80% of months, test on the latest 20%)."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["unit_price", "comp_1", "comp_2", "comp_3", "comp_avg", "price_gap_pct", "freight_price",
       "product_score", "customers", "weekend", "holiday", "month", "lag_price"]
CAT = ["category", "product_id"]

df = pd.read_csv("data/processed/clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
NUM = [c for c in NUM if df[c].notna().any()]
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
X_train, y_train, X_test, y_test = train[NUM + CAT], train.qty, test[NUM + CAT], test.qty


def pipe(model, scale=False):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler() if scale else "passthrough")])
    pre = ColumnTransformer([("num", num, NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return Pipeline([("prep", pre), ("model", model)])


models = {
    "Ridge Regression": pipe(Ridge(alpha=1.0), scale=True),
    "Random Forest": pipe(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    "Gradient Boosting": pipe(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)),
}

# baseline: each product's average quantity in the training period
base = test.product_id.map(train.groupby("product_id").qty.mean()).fillna(train.qty.mean())
metrics = {"Baseline (product average)": {"MAE": round(mean_absolute_error(y_test, base), 2),
                                          "R2": round(r2_score(y_test, base), 3)}}
preds = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds[name] = m.predict(X_test)
    metrics[name] = {"MAE": round(mean_absolute_error(y_test, preds[name]), 2),
                     "R2": round(r2_score(y_test, preds[name]), 3)}
for k, v in metrics.items():
    print(f"{k:28s} MAE={v['MAE']:<8} R2={v['R2']}")

best = min(models, key=lambda k: metrics[k]["MAE"])
Path("reports").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True); Path("images").mkdir(exist_ok=True)
Path("reports/demand_model_metrics.json").write_text(json.dumps(metrics, indent=2))
joblib.dump(models[best], "models/demand_model.joblib")
print(f"\nBest model: {best}. Saved to models/demand_model.joblib")

plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, preds[best], alpha=0.5, color="#4C6EF5")
lim = [0, max(y_test.max(), preds[best].max())]
plt.plot(lim, lim, "r--"); plt.xlabel("Actual qty"); plt.ylabel("Predicted qty")
plt.title(f"Actual vs Predicted ({best})")
plt.tight_layout(); plt.savefig("images/11_actual_vs_predicted.png", dpi=130); plt.close()

rf = models["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
plt.figure(figsize=(7, 5)); imp.plot.barh(color="#E8590C"); plt.title("Top Features (Random Forest)")
plt.tight_layout(); plt.savefig("images/12_feature_importance.png", dpi=130)
