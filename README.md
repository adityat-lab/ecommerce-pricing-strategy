# 💰 E-commerce Pricing Strategy

Price elasticity analysis, competitor price benchmarking, demand forecasting and
profit-based price recommendations, with an interactive Streamlit dashboard.

## Business Questions
1. How sensitive is demand to price (price elasticity) by product and category?
2. How do our prices compare with competitors, and does the gap affect sales?
3. Which products should increase or decrease price to maximise profit?
4. Can we forecast monthly demand from price and market features?

## Dataset
[Retail Price Optimization (Kaggle, suddharshan)](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization):
monthly product-level sales (`qty`, `unit_price`, `total_price`), freight price, product score,
holiday/weekend counts and three competitor prices. About 676 rows across 52 products.
Download `retail_price.csv` and place it in `data/raw/`.

## Method
- **Elasticity:** log-log regression, ln(qty) = a + e * ln(price), per product and per category
  (category estimate pools its products with product fixed effects).
- **Optimisation:** demand curve q(p) = q0 * (p/p0)^e, profit = (p - cost) * q(p).
  Prices are limited to +/-15% of the current price and to the range seen in the data.
- **Assumption:** the dataset has no product cost, so unit cost = 60% of price (change with `--cost-ratio`).
- **Demand model:** Ridge, Random Forest and Gradient Boosting with a time-based train/test split.

## Project Structure
```
├── data/raw/            raw CSV (not committed)
├── data/processed/      cleaned data
├── src/                 clean_data.py, eda.py, elasticity.py, price_optimization.py,
│                        demand_model.py, sql_analysis.py, generate_sample_data.py
├── sql/                 analysis_queries.sql
├── app/dashboard.py     Streamlit dashboard
├── images/              charts and screenshots
└── reports/             insights, elasticity and recommendation tables
```

## How to Run
```bash
pip install -r requirements.txt
python src/clean_data.py
python src/eda.py
python src/elasticity.py
python src/price_optimization.py --cost-ratio 0.6
python src/demand_model.py
python src/sql_analysis.py
streamlit run app/dashboard.py
```

## Key Insights
> Paste `reports/insights.md` and the printed elasticity results here.

## Model Results
> Paste `reports/demand_model_metrics.json` here.

## Limitations
- Elasticity comes from observational data: prices were not randomly assigned, so other factors can bias it.
- Products whose price barely changed give weak estimates; those are flagged Low confidence.
- Profit uses an assumed cost ratio. Validate recommendations with A/B tests before rollout.

## Future Improvements
- Add real unit costs and compute true margin
- Add cross-price elasticity versus competitor prices
- A/B test simulation, deploy on Streamlit Community Cloud
