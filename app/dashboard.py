"""Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="E-commerce Pricing Strategy", page_icon="💰", layout="wide")


@st.cache_data
def load():
    d = pd.read_csv("data/processed/clean.csv", parse_dates=["date"])
    p = pd.read_csv("reports/elasticity_products.csv") if Path("reports/elasticity_products.csv").exists() else pd.DataFrame()
    c = pd.read_csv("reports/elasticity_categories.csv") if Path("reports/elasticity_categories.csv").exists() else pd.DataFrame()
    r = pd.read_csv("reports/price_recommendations.csv") if Path("reports/price_recommendations.csv").exists() else pd.DataFrame()
    return d, p, c, r


df, prod, cat, rec = load()
st.title("💰 E-commerce Pricing Strategy")

cats = st.sidebar.multiselect("Category", sorted(df.category.unique()), default=sorted(df.category.unique()))
d = df[df.category.isin(cats)]
if d.empty:
    st.warning("Select at least one category.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"{d.revenue.sum():,.0f}")
k2.metric("Units sold", f"{d.qty.sum():,.0f}")
k3.metric("Avg unit price", f"{d.unit_price.mean():,.2f}")
k4.metric("Avg gap vs competitors", f"{d.price_gap_pct.mean():+.1f}%")

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Competitor Pricing", "Elasticity", "Recommendations", "Price Simulator"])

with t1:
    a, b = st.columns(2)
    m = d.groupby(d.date.dt.to_period("M").astype(str)).revenue.sum().reset_index()
    a.plotly_chart(px.line(m, x="date", y="revenue", markers=True, title="Monthly Revenue"), use_container_width=True)
    rc = d.groupby("category").revenue.sum().sort_values().reset_index()
    b.plotly_chart(px.bar(rc, x="revenue", y="category", orientation="h", title="Revenue by Category"),
                   use_container_width=True)
    tp = d.groupby("product_id").revenue.sum().sort_values(ascending=False).head(10).reset_index()
    st.plotly_chart(px.bar(tp, x="product_id", y="revenue", title="Top 10 Products"), use_container_width=True)

with t2:
    a, b = st.columns(2)
    g = d.groupby("category").price_gap_pct.mean().sort_values().reset_index()
    a.plotly_chart(px.bar(g, x="price_gap_pct", y="category", orientation="h",
                          title="Avg price gap vs competitors (%)"), use_container_width=True)
    b.plotly_chart(px.histogram(d, x="price_gap_pct", nbins=40, title="Distribution of price gap (%)"),
                   use_container_width=True)
    st.plotly_chart(px.scatter(d, x="comp_avg", y="unit_price", color="category", opacity=0.6,
                               title="Our price vs competitor average"), use_container_width=True)

with t3:
    if cat.empty:
        st.info("Run `python src/elasticity.py` first.")
    else:
        st.caption("Elasticity below -1 means demand is price-sensitive: a 1% price rise cuts quantity by more than 1%.")
        fig = px.bar(cat, x="elasticity", y="category", orientation="h", error_x=1.96 * cat.std_err,
                     title="Price elasticity by category")
        fig.add_vline(x=-1, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(prod[prod.category.isin(cats)], use_container_width=True)

with t4:
    if rec.empty:
        st.info("Run `python src/price_optimization.py` first.")
    else:
        conf = st.multiselect("Confidence", ["High", "Medium", "Low"], default=["High", "Medium"])
        r = rec[rec.category.isin(cats) & rec.confidence.isin(conf)]
        st.caption("Estimates come from historical data and an assumed cost ratio. Validate with A/B tests before rollout.")
        st.dataframe(r, use_container_width=True)
        st.download_button("Download recommendations (CSV)", r.to_csv(index=False), "price_recommendations.csv")
        if not r.empty:
            st.plotly_chart(px.bar(r.sort_values("profit_change_pct").tail(15), x="profit_change_pct", y="product_id",
                                   orientation="h", color="action", title="Expected profit change (%)"),
                            use_container_width=True)

with t5:
    pid = st.selectbox("Product", sorted(d.product_id.unique()))
    g = df[df.product_id == pid].sort_values("date").tail(3)
    p0, q0 = g.unit_price.mean(), g.qty.mean()
    e, src = -1.0, "default (-1.0, no estimate)"
    if not prod.empty and pid in set(prod.product_id) and prod.set_index("product_id").loc[pid, "elasticity"] < 0:
        e, src = float(prod.set_index("product_id").loc[pid, "elasticity"]), "product estimate"
    elif not cat.empty and g.category.iloc[0] in set(cat.category) and cat.set_index("category").loc[g.category.iloc[0], "elasticity"] < 0:
        e, src = float(cat.set_index("category").loc[g.category.iloc[0], "elasticity"]), "category estimate"
    c1, c2 = st.columns(2)
    cost_ratio = c1.slider("Unit cost as % of current price", 20, 95, 60) / 100
    change = c2.slider("Price change (%)", -30, 30, 0)
    cost = cost_ratio * p0
    grid = np.linspace(0.7 * p0, 1.3 * p0, 120)
    qg = q0 * (grid / p0) ** e
    pnew = p0 * (1 + change / 100)
    qnew = q0 * (pnew / p0) ** e
    m1, m2, m3 = st.columns(3)
    m1.metric("New price", f"{pnew:,.2f}", f"{change:+d}%")
    m2.metric("Expected units / month", f"{qnew:,.1f}", f"{(qnew / q0 - 1) * 100:+.1f}%")
    base_profit = (p0 - cost) * q0
    m3.metric("Expected profit / month", f"{(pnew - cost) * qnew:,.0f}",
              f"{((pnew - cost) * qnew / base_profit - 1) * 100:+.1f}%" if base_profit > 0 else None)
    st.caption(f"Elasticity used: {e:.2f} ({src}).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=grid * qg, name="Revenue"))
    fig.add_trace(go.Scatter(x=grid, y=(grid - cost) * qg, name="Profit"))
    fig.add_vline(x=p0, line_dash="dash", annotation_text="current")
    fig.update_layout(title="Revenue and profit vs price", xaxis_title="Price", yaxis_title="Per month")
    st.plotly_chart(fig, use_container_width=True)
