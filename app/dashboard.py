"""Pricing Lab: Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pricing Lab", page_icon="💰", layout="wide")

PAL = ["#6c5ce7", "#ff6b6b", "#00b894", "#fdcb6e", "#0984e3", "#e17055", "#00cec9", "#d63031", "#a29bfe"]
RED, GREEN, BLUE, GREY, ORANGE = "#e5484d", "#00a37a", "#0984e3", "#8a8fa3", "#f08a24"

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
footer{visibility:hidden}
.hero{background:linear-gradient(120deg,#1b1f3b 0%,#5b3fd1 55%,#ff6b6b 125%);color:#fff;border-radius:20px;padding:26px 30px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.3rem;color:#fff;letter-spacing:-.02em}
.hero p{margin:6px 0 0;opacity:.92}
.kpi{border:1px solid rgba(127,127,127,.25);background:rgba(127,127,127,.07);border-radius:16px;padding:14px 16px}
.kpi .l{font-size:.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{font-size:1.85rem;font-weight:800;line-height:1.25}
.kpi .d{font-size:.84rem}
.story{border-left:5px solid #6c5ce7;background:rgba(108,92,231,.09);border-radius:12px;padding:12px 14px;margin:6px 0 10px}
.story b{display:block;font-size:1.02rem}
.story span{font-size:.92rem;opacity:.85}
.opp{border:1px solid rgba(127,127,127,.25);border-radius:16px;padding:16px;background:rgba(127,127,127,.06)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:700;color:#fff}
.inc{background:#0984e3}.dec{background:#e17055}.hold{background:#8a8fa3}
.arrow{font-size:1.5rem;font-weight:800;margin:.5rem 0}
.chip{display:inline-block;border:1px solid rgba(127,127,127,.35);border-radius:999px;padding:2px 10px;font-size:.8rem;margin:2px 2px 0 0}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = pd.read_csv("data/processed/clean.csv", parse_dates=["date"])

    def opt(p):
        return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()

    return (d, opt("reports/elasticity_products.csv"), opt("reports/elasticity_categories.csv"),
            opt("reports/price_recommendations.csv"))


df, prod, cat, rec = load()
ALL = sorted(df.category.unique())
CMAP = {c: PAL[i % len(PAL)] for i, c in enumerate(ALL)}


def money(x):
    return f"{x:,.0f}"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=55, b=10), title_font_size=16, legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,.2)")
    return fig


def show(fig, h=380):
    st.plotly_chart(style(fig, h))


def kpi(label, value, delta=None):
    d = ""
    if delta is not None:
        color = GREEN if delta >= 0 else RED
        d = f'<div class="d" style="color:{color}">{"▲" if delta >= 0 else "▼"} {abs(delta):.1f}% vs previous 3 months</div>'
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


def story(title, text):
    return f'<div class="story"><b>{title}</b><span>{text}</span></div>'


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    cats = st.multiselect("Categories", ALL, default=ALL)
    st.caption("Remove categories to focus the story.")
d = df[df.category.isin(cats)]
if d.empty:
    st.warning("Select at least one category in the sidebar.")
    st.stop()

# ---------- hero + KPIs ----------
st.markdown(
    f'<div class="hero"><h1>💰 Pricing Lab</h1><p>How do we compare with competitors, how price-sensitive are '
    f'customers, and which prices should move? &nbsp;·&nbsp; {d.product_id.nunique()} products &nbsp;·&nbsp; '
    f'{d.category.nunique()} categories &nbsp;·&nbsp; {d.date.min():%b %Y} to {d.date.max():%b %Y}</p></div>',
    unsafe_allow_html=True)

m = d.groupby(d.date.dt.to_period("M").astype(str)).revenue.sum().reset_index()
delta = None
if len(m) >= 6 and m.revenue.iloc[-6:-3].sum() > 0:
    delta = (m.revenue.tail(3).sum() / m.revenue.iloc[-6:-3].sum() - 1) * 100
gap_all = d.price_gap_pct.dropna()

k = st.columns(5)
k[0].markdown(kpi("Revenue", money(d.revenue.sum()), delta), unsafe_allow_html=True)
k[1].markdown(kpi("Units sold", money(d.qty.sum())), unsafe_allow_html=True)
k[2].markdown(kpi("Avg unit price", f"{d.unit_price.mean():,.2f}"), unsafe_allow_html=True)
k[3].markdown(kpi("Vs competitors", f"{gap_all.mean():+.1f}%" if len(gap_all) else "n/a"), unsafe_allow_html=True)
k[4].markdown(kpi("Products", d.product_id.nunique()), unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["🏠 Overview", "⚔️ Competitors", "📉 Elasticity", "🎯 Opportunities", "🧪 Simulator"])

# ---------- overview ----------
with tabs[0]:
    cat_rev = d.groupby("category").revenue.sum().sort_values(ascending=False)
    peak = m.loc[m.revenue.idxmax()]
    s = st.columns(4)
    s[0].markdown(story("🏆 Top category", f"{cat_rev.index[0]} brings {cat_rev.iloc[0] / cat_rev.sum() * 100:.0f}% of revenue."),
                  unsafe_allow_html=True)
    s[1].markdown(story("📈 Peak month", f"{peak.date} with {money(peak.revenue)} in revenue."), unsafe_allow_html=True)
    s[2].markdown(story("⚔️ Price position", f"Priced above competitors in {(gap_all > 0).mean() * 100:.0f}% of product-months."
                        if len(gap_all) else "No competitor prices in the data."), unsafe_allow_html=True)
    if not cat.empty and cat.category.isin(cats).any():
        sens = cat[cat.category.isin(cats)].sort_values("elasticity").iloc[0]
        s[3].markdown(story("📉 Most price-sensitive", f"{sens.category} (elasticity {sens.elasticity:.2f})."),
                      unsafe_allow_html=True)
    else:
        s[3].markdown(story("📉 Elasticity", "Run elasticity.py to see sensitivity."), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        fig = px.area(m, x="date", y="revenue", title="Monthly revenue", line_shape="spline")
        fig.update_traces(line_color=PAL[0], fillcolor="rgba(108,92,231,.18)")
        fig.add_annotation(x=peak.date, y=peak.revenue, text="Peak", showarrow=True, arrowhead=2)
        show(fig)
    with b:
        rc = d.groupby("category", as_index=False).revenue.sum().sort_values("revenue")
        fig = px.bar(rc, x="revenue", y="category", orientation="h", color="category", color_discrete_map=CMAP,
                     text_auto=".2s", title="Revenue by category")
        fig.update_layout(showlegend=False)
        show(fig)
    fig = px.treemap(d, path=[px.Constant("All products"), "category", "product_id"], values="revenue",
                     color="category", color_discrete_map={**CMAP, "(?)": "#d5d8e0"},
                     title="Where the revenue comes from (click a block to zoom in)")
    show(fig, 460)

# ---------- competitors ----------
with tabs[1]:
    if gap_all.empty:
        st.info("No competitor prices found in the data.")
    else:
        a, b = st.columns(2)
        g = d.groupby("category", as_index=False).price_gap_pct.mean().dropna().sort_values("price_gap_pct")
        g["Position"] = np.where(g.price_gap_pct > 0, "Above competitors", "Below competitors")
        with a:
            show(px.bar(g, x="price_gap_pct", y="category", orientation="h", color="Position",
                        color_discrete_map={"Above competitors": RED, "Below competitors": GREEN},
                        title="Average price gap vs competitors (%)"))
        with b:
            fig = px.histogram(d.dropna(subset=["price_gap_pct"]), x="price_gap_pct", nbins=40,
                               color_discrete_sequence=[PAL[0]], title="How big is the gap? (%)")
            fig.add_vline(x=0, line_dash="dash")
            show(fig)
        dd = d.dropna(subset=["comp_avg"])
        fig = px.scatter(dd, x="comp_avg", y="unit_price", color="category", color_discrete_map=CMAP, opacity=.7,
                         hover_name="product_id", title="Our price against the competitor average")
        lim = [float(min(dd.comp_avg.min(), dd.unit_price.min())), float(max(dd.comp_avg.max(), dd.unit_price.max()))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Same price as competitors",
                                 line=dict(dash="dash", color=GREY)))
        show(fig, 440)
        st.caption("Dots above the dashed line are priced higher than competitors, dots below are cheaper.")

# ---------- elasticity ----------
with tabs[2]:
    if cat.empty:
        st.info("Run `python src/elasticity.py`, then push the reports folder.")
    else:
        st.caption("Elasticity below -1 means customers are price-sensitive: a 1% price rise loses more than 1% of sales.")
        cc = cat[cat.category.isin(cats)].copy()
        cc["Group"] = np.where(cc.elasticity <= -1, "Price-sensitive (below -1)", "Less sensitive")
        a, b = st.columns(2)
        with a:
            if not cc.empty:
                fig = px.bar(cc, x="elasticity", y="category", orientation="h", color="Group", error_x=1.96 * cc.std_err,
                             color_discrete_map={"Price-sensitive (below -1)": RED, "Less sensitive": BLUE},
                             title="Elasticity by category (bars show uncertainty)")
                fig.add_vline(x=-1, line_dash="dash")
                show(fig, 420)
        pp = prod[prod.category.isin(cats)] if not prod.empty else prod
        with b:
            if not pp.empty:
                fig = px.scatter(pp, x="elasticity", y="avg_price", size="avg_qty", color="category",
                                 color_discrete_map=CMAP, hover_name="product_id", log_y=True, size_max=36,
                                 title="Product elasticity map (bubble size = units)")
                fig.add_vline(x=-1, line_dash="dash")
                show(fig, 420)
        if not pp.empty:
            st.dataframe(pp, hide_index=True)

# ---------- opportunities ----------
with tabs[3]:
    if rec.empty:
        st.info("Run `python src/price_optimization.py`, then push the reports folder.")
    else:
        conf = st.multiselect("Confidence level", ["High", "Medium", "Low"], default=["High", "Medium"])
        r = rec[rec.category.isin(cats) & rec.confidence.isin(conf)].dropna(subset=["profit_change_pct"])
        st.caption("Modelled estimates from historical data and an assumed cost ratio. Test with an A/B experiment before rollout.")
        if r.empty:
            st.info("No recommendations for this selection.")
        else:
            top = r.sort_values("profit_change_pct", ascending=False).head(3)
            cols = st.columns(len(top))
            for col, x in zip(cols, top.itertuples()):
                cls = {"Increase price": "inc", "Decrease price": "dec"}.get(x.action, "hold")
                col.markdown(
                    f'<div class="opp"><span class="badge {cls}">{x.action}</span>'
                    f'<h3 style="margin:.5rem 0 0">{x.product_id}</h3><div style="opacity:.7">{x.category}</div>'
                    f'<div class="arrow">{x.current_price:,.2f} → {x.recommended_price:,.2f}</div>'
                    f'<span class="chip">Profit {x.profit_change_pct:+.1f}%</span>'
                    f'<span class="chip">Units {x.qty_change_pct:+.1f}%</span>'
                    f'<span class="chip">Confidence: {x.confidence}</span></div>', unsafe_allow_html=True)
            st.write("")
            fig = px.bar(r.sort_values("profit_change_pct").tail(15), x="profit_change_pct", y="product_id",
                         orientation="h", color="action",
                         color_discrete_map={"Increase price": BLUE, "Decrease price": "#e17055", "Hold": GREY,
                                             "Hold (no reliable elasticity)": GREY},
                         title="Expected profit change by product (%)")
            show(fig, 460)
            st.dataframe(r, hide_index=True)
            st.download_button("⬇️ Download recommendations (CSV)", r.to_csv(index=False), "price_recommendations.csv")

# ---------- simulator ----------
with tabs[4]:
    pid = st.selectbox("Pick a product", sorted(d.product_id.unique()))
    g = df[df.product_id == pid].sort_values("date").tail(3)
    p0, q0, category = g.unit_price.mean(), g.qty.mean(), g.category.iloc[0]
    e, src = -1.0, "default of -1.0 (no estimate)"
    if not prod.empty and pid in set(prod.product_id) and prod.set_index("product_id").loc[pid, "elasticity"] < 0:
        e, src = float(prod.set_index("product_id").loc[pid, "elasticity"]), "this product"
    elif not cat.empty and category in set(cat.category) and cat.set_index("category").loc[category, "elasticity"] < 0:
        e, src = float(cat.set_index("category").loc[category, "elasticity"]), "its category"

    c1, c2 = st.columns(2)
    cost_ratio = c1.slider("Unit cost as % of current price", 20, 95, 60) / 100
    cost = cost_ratio * p0
    grid = np.linspace(0.7 * p0, 1.3 * p0, 160)
    qg = q0 * (grid / p0) ** e
    pg = (grid - cost) * qg
    bi = int(np.argmax(np.where((grid >= .85 * p0) & (grid <= 1.15 * p0), pg, -np.inf)))
    best_pct = int(round((grid[bi] / p0 - 1) * 100))
    key = f"chg_{pid}"
    st.session_state.setdefault(key, 0)
    if c2.button("⭐ Jump to the best price"):
        st.session_state[key] = best_pct
    change = c2.slider("Price change (%)", -30, 30, key=key)

    pnew = p0 * (1 + change / 100)
    qnew = q0 * (pnew / p0) ** e
    base = (p0 - cost) * q0
    m1, m2, m3 = st.columns(3)
    m1.metric("New price", f"{pnew:,.2f}", f"{change:+d}%")
    m2.metric("Units per month", f"{qnew:,.1f}", f"{(qnew / q0 - 1) * 100:+.1f}%")
    m3.metric("Profit per month", f"{(pnew - cost) * qnew:,.0f}",
              f"{((pnew - cost) * qnew / base - 1) * 100:+.1f}%" if base > 0 else None)
    st.caption(f"Elasticity used: {e:.2f}, taken from {src}. Best price within ±15% is about {grid[bi]:,.2f} ({best_pct:+d}%).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=grid * qg, name="Revenue", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=grid, y=pg, name="Profit", line=dict(color=ORANGE, width=3), fill="tozeroy",
                             fillcolor="rgba(240,138,36,.12)"))
    fig.add_vline(x=p0, line_dash="dash", annotation_text="today")
    fig.add_trace(go.Scatter(x=[grid[bi]], y=[pg[bi]], mode="markers+text", text=["⭐ best"], textposition="top center",
                             marker=dict(size=14, color=ORANGE), name="Best price"))
    fig.add_trace(go.Scatter(x=[pnew], y=[(pnew - cost) * qnew], mode="markers",
                             marker=dict(size=14, color=PAL[0], symbol="diamond"), name="Your price"))
    fig.update_layout(title="Revenue and profit against price", xaxis_title="Price", yaxis_title="Per month")
    show(fig, 440)