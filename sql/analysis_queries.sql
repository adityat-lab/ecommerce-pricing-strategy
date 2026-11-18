-- Q1: Revenue and average price by category
SELECT category, COUNT(DISTINCT product_id) AS products, ROUND(SUM(revenue), 0) AS revenue,
       ROUND(AVG(unit_price), 2) AS avg_price, SUM(qty) AS units
FROM sales
GROUP BY category
ORDER BY revenue DESC;

-- Q2: Monthly revenue with month-over-month growth (window function LAG)
WITH m AS (
    SELECT substr(date, 1, 7) AS ym, SUM(revenue) AS revenue
    FROM sales
    GROUP BY ym
)
SELECT ym, ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY ym)) / LAG(revenue) OVER (ORDER BY ym), 1) AS mom_pct
FROM m
ORDER BY ym;

-- Q3: Pareto - cumulative revenue share of top products
WITH p AS (
    SELECT product_id, SUM(revenue) AS rev
    FROM sales
    GROUP BY product_id
)
SELECT product_id, ROUND(rev, 0) AS revenue,
       ROUND(100.0 * SUM(rev) OVER (ORDER BY rev DESC) / SUM(rev) OVER (), 1) AS cum_share_pct
FROM p
ORDER BY rev DESC
LIMIT 15;

-- Q4: Price position vs competitors by category
SELECT category, ROUND(AVG(price_gap_pct), 1) AS avg_gap_pct,
       ROUND(100.0 * AVG(CASE WHEN price_gap_pct > 0 THEN 1 ELSE 0 END), 1) AS pct_months_above_competitors
FROM sales
GROUP BY category
ORDER BY avg_gap_pct DESC;

-- Q5: What happens to quantity when a product price moves from one month to the next
WITH chg AS (
    SELECT product_id, qty, unit_price,
           LAG(unit_price) OVER (PARTITION BY product_id ORDER BY date) AS prev_price,
           LAG(qty) OVER (PARTITION BY product_id ORDER BY date) AS prev_qty
    FROM sales
)
SELECT CASE WHEN unit_price > prev_price * 1.02 THEN 'Price up more than 2%'
            WHEN unit_price < prev_price * 0.98 THEN 'Price down more than 2%'
            ELSE 'Stable' END AS price_move,
       COUNT(*) AS cases,
       ROUND(AVG(100.0 * (qty - prev_qty) / prev_qty), 1) AS avg_qty_change_pct
FROM chg
WHERE prev_price IS NOT NULL AND prev_qty > 0
GROUP BY price_move;

-- Q6: Rank products inside each category by revenue
SELECT category, product_id, ROUND(SUM(revenue), 0) AS revenue,
       RANK() OVER (PARTITION BY category ORDER BY SUM(revenue) DESC) AS rank_in_category
FROM sales
GROUP BY category, product_id
ORDER BY category, rank_in_category
LIMIT 20;

-- Q7: Holiday months vs normal months
SELECT CASE WHEN holiday > 0 THEN 'Holiday month' ELSE 'No holiday' END AS month_type,
       COUNT(*) AS rows_count, ROUND(AVG(qty), 1) AS avg_qty, ROUND(AVG(unit_price), 2) AS avg_price
FROM sales
GROUP BY month_type;
