"""Step 2: exploratory analysis. Charts go to images/, insights to reports/insights.md"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
Path("images").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
df = pd.read_csv("data/processed/clean.csv", parse_dates=["date"])
insights = []


def save(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}", dpi=130)
    plt.close()


# 1. revenue by category
rc = df.groupby("category")["revenue"].sum().sort_values()
plt.figure(figsize=(7, 4.5)); rc.plot.barh(color="#4C6EF5")
plt.title("Total Revenue by Category"); plt.xlabel("Revenue")
save("01_revenue_by_category.png")
top = rc.index[-1]
insights.append(f"**{top}** is the top category with {rc.iloc[-1] / rc.sum() * 100:.0f}% of total revenue.")

# 2. monthly revenue trend
mr = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
plt.figure(figsize=(9, 4)); plt.plot(mr.index.astype(str), mr.values, marker="o", color="#E8590C")
plt.xticks(rotation=60); plt.title("Monthly Revenue")
save("02_monthly_revenue.png")
if len(mr) > 1:
    insights.append(f"Peak revenue month is **{mr.idxmax()}** ({mr.max():,.0f}); weakest is **{mr.idxmin()}** ({mr.min():,.0f}).")

# 3. price vs quantity (log-log)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=np.log(df.unit_price), y=np.log(df.qty), hue="category", alpha=0.6, legend=False)
plt.xlabel("log(unit price)"); plt.ylabel("log(quantity)"); plt.title("Price vs Demand (log-log)")
save("03_price_vs_demand.png")

# 4. competitor price gap
plt.figure(figsize=(7, 4)); sns.histplot(df["price_gap_pct"].dropna(), bins=40, color="#12B886")
plt.axvline(0, color="red", ls="--"); plt.title("Our Price vs Competitor Average (%)")
save("04_price_gap_hist.png")
gap = df.groupby("category")["price_gap_pct"].mean().sort_values()
plt.figure(figsize=(7, 4.5)); gap.plot.barh(color="#845EF7")
plt.axvline(0, color="red", ls="--"); plt.title("Avg Price Gap vs Competitors by Category (%)")
save("05_gap_by_category.png")
if gap.notna().any():
    insights.append(f"Most expensive vs competitors: **{gap.index[-1]}** ({gap.iloc[-1]:+.1f}%); "
                    f"cheapest: **{gap.index[0]}** ({gap.iloc[0]:+.1f}%).")
    insights.append(f"We price above the competitor average in **{(df.price_gap_pct > 0).mean() * 100:.0f}%** of product-months.")

# 5. holiday / weekend effect
df["holiday_month"] = np.where(df.holiday > 0, "Has holiday", "No holiday")
hq = df.groupby("holiday_month")["qty"].mean()
plt.figure(figsize=(5, 4)); hq.plot.bar(color=["#FA5252", "#868E96"]); plt.title("Avg Monthly Qty: Holiday vs Not")
plt.xticks(rotation=0)
save("06_holiday_effect.png")

# 6. top products
tp = df.groupby("product_id")["revenue"].sum().sort_values().tail(10)
plt.figure(figsize=(7, 4.5)); tp.plot.barh(color="#FAB005"); plt.title("Top 10 Products by Revenue")
save("07_top_products.png")
share = df.groupby("product_id")["revenue"].sum().sort_values(ascending=False)
k = max(1, int(len(share) * 0.2))
insights.append(f"The top 20% of products ({k}) generate **{share.head(k).sum() / share.sum() * 100:.0f}%** of revenue.")

# 7. correlation
cols = ["qty", "unit_price", "comp_avg", "price_gap_pct", "freight_price", "product_score", "customers", "holiday"]
plt.figure(figsize=(7.5, 6)); sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
save("08_correlation.png")

md = "# Key Insights\n\n" + "\n".join(f"- {i}" for i in insights) + "\n"
Path("reports/insights.md").write_text(md, encoding="utf-8")
print(md)
