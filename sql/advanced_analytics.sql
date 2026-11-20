
-- QUERY: monthly_revenue
-- 1. Monthly Revenue Analysis

SELECT
    d.year,
    d.month,
    substr(d.full_date, 1, 7) AS month,
    ROUND(SUM(CASE WHEN f.quantity > 0
        THEN f.revenue ELSE 0 END), 2) AS gross_sales,
    ROUND(SUM(CASE WHEN f.quantity < 0
        THEN ABS(f.revenue) ELSE 0 END), 2) AS returned_value,
    ROUND(SUM(f.revenue), 2) AS net_revenue,
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT f.invoice) AS total_invoices
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month, month
ORDER BY d.year, d.month;


-- QUERY: yoy_growth
-- 2. Year-over-Year Revenue Growth

WITH monthly AS (
    SELECT
        substr(d.full_date, 1, 7) AS month,
        SUM(f.revenue) AS net_revenue
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY month
),
growth AS (
    SELECT
        month,
        net_revenue,
        LAG(net_revenue, 12) OVER (
            ORDER BY month
        ) AS previous_year_revenue
    FROM monthly
)
SELECT
    month,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(previous_year_revenue, 2)
        AS previous_year_revenue,
    ROUND(
        100.0 * (net_revenue - previous_year_revenue)
        / NULLIF(ABS(previous_year_revenue), 0),
        2
    ) AS yoy_growth_percent
FROM growth
ORDER BY month;


-- QUERY: top_products
-- 3. Top 20 Products by Gross Sales

WITH product_metrics AS (
    SELECT
        p.stock_code,
        p.description,
        SUM(CASE WHEN f.quantity > 0
            THEN f.revenue ELSE 0 END) AS gross_sales,
        SUM(CASE WHEN f.quantity < 0
            THEN ABS(f.revenue) ELSE 0 END) AS return_value,
        SUM(f.revenue) AS net_revenue,
        SUM(CASE WHEN f.quantity > 0
            THEN f.quantity ELSE 0 END) AS units_sold,
        COUNT(DISTINCT f.invoice) AS total_orders
    FROM fact_sales f
    JOIN dim_product p
        ON f.product_key = p.product_key
    GROUP BY p.stock_code, p.description
)
SELECT
    RANK() OVER (ORDER BY gross_sales DESC) AS product_rank,
    stock_code,
    description,
    ROUND(gross_sales, 2) AS gross_sales,
    ROUND(return_value, 2) AS return_value,
    ROUND(net_revenue, 2) AS net_revenue,
    units_sold,
    total_orders
FROM product_metrics
ORDER BY gross_sales DESC
LIMIT 20;


-- QUERY: country_performance
-- 4. Country-wise Business Performance

SELECT
    co.country_name,
    COUNT(DISTINCT f.invoice) AS total_orders,
    COUNT(DISTINCT CASE
        WHEN c.customer_id <> '__UNKNOWN__'
        THEN c.customer_id END
    ) AS unique_customers,
    ROUND(SUM(CASE WHEN f.quantity > 0
        THEN f.revenue ELSE 0 END), 2) AS gross_sales,
    ROUND(SUM(f.revenue), 2) AS net_revenue,
    SUM(CASE WHEN f.quantity > 0
        THEN f.quantity ELSE 0 END) AS units_sold
FROM fact_sales f
JOIN dim_country co
    ON f.country_key = co.country_key
JOIN dim_customer c
    ON f.customer_key = c.customer_key
GROUP BY co.country_name
ORDER BY net_revenue DESC;


-- QUERY: customer_ranking
-- 5. Customer Purchase Ranking

WITH customer_metrics AS (
    SELECT
        c.customer_id,
        COUNT(DISTINCT f.invoice) AS total_orders,
        SUM(CASE WHEN f.quantity > 0
            THEN f.revenue ELSE 0 END) AS gross_spend,
        SUM(f.revenue) AS net_revenue,
        SUM(CASE WHEN f.quantity > 0
            THEN f.quantity ELSE 0 END) AS total_units
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    WHERE c.customer_id <> '__UNKNOWN__'
    GROUP BY c.customer_id
)
SELECT
    DENSE_RANK() OVER (
        ORDER BY net_revenue DESC
    ) AS customer_rank,
    customer_id,
    total_orders,
    ROUND(gross_spend, 2) AS gross_spend,
    ROUND(net_revenue, 2) AS net_revenue,
    total_units
FROM customer_metrics
ORDER BY net_revenue DESC
LIMIT 100;


-- QUERY: repeat_customers
-- 6. Repeat Customer Analysis

WITH customer_orders AS (
    SELECT
        c.customer_id,
        COUNT(DISTINCT f.invoice) AS order_count
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    WHERE c.customer_id <> '__UNKNOWN__'
    GROUP BY c.customer_id
),
segments AS (
    SELECT
        CASE
            WHEN order_count = 1 THEN 'One-time Customer'
            ELSE 'Repeat Customer'
        END AS customer_segment,
        COUNT(*) AS customer_count,
        SUM(order_count) AS total_orders
    FROM customer_orders
    GROUP BY customer_segment
)
SELECT
    customer_segment,
    customer_count,
    total_orders,
    ROUND(
        100.0 * customer_count /
        SUM(customer_count) OVER (),
        2
    ) AS customer_percentage
FROM segments
ORDER BY customer_count DESC;


-- QUERY: cohort_analysis
-- 7. Monthly Customer Cohort Retention

WITH customer_months AS (
    SELECT DISTINCT
        c.customer_id,
        substr(d.full_date, 1, 7) AS order_month
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    JOIN dim_date d
        ON f.date_key = d.date_key
    WHERE
        c.customer_id <> '__UNKNOWN__'
        AND f.quantity > 0
        AND f.revenue > 0
),
first_month AS (
    SELECT
        customer_id,
        MIN(order_month) AS cohort_month
    FROM customer_months
    GROUP BY customer_id
),
cohort_activity AS (
    SELECT
        fm.cohort_month,
        cm.order_month,
        cm.customer_id,
        (
            (
                CAST(substr(cm.order_month, 1, 4) AS INTEGER)
                - CAST(substr(fm.cohort_month, 1, 4) AS INTEGER)
            ) * 12
            +
            CAST(substr(cm.order_month, 6, 2) AS INTEGER)
            - CAST(substr(fm.cohort_month, 6, 2) AS INTEGER)
        ) AS month_number
    FROM customer_months cm
    JOIN first_month fm
        ON cm.customer_id = fm.customer_id
),
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(*) AS cohort_size
    FROM first_month
    GROUP BY cohort_month
),
retention AS (
    SELECT
        cohort_month,
        month_number,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM cohort_activity
    GROUP BY cohort_month, month_number
)
SELECT
    r.cohort_month,
    r.month_number,
    s.cohort_size,
    r.active_customers,
    ROUND(
        100.0 * r.active_customers / s.cohort_size,
        2
    ) AS retention_percentage
FROM retention r
JOIN cohort_sizes s
    ON r.cohort_month = s.cohort_month
ORDER BY r.cohort_month, r.month_number;


-- QUERY: revenue_trends
-- 8. Running Revenue and Three-Month Moving Average

WITH monthly AS (
    SELECT
        substr(d.full_date, 1, 7) AS month,
        SUM(f.revenue) AS net_revenue
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_key = d.date_key
    GROUP BY month
)
SELECT
    month,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(
        SUM(net_revenue) OVER (
            ORDER BY month
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ), 2
    ) AS cumulative_revenue,
    ROUND(
        AVG(net_revenue) OVER (
            ORDER BY month
            ROWS BETWEEN 2 PRECEDING
            AND CURRENT ROW
        ), 2
    ) AS moving_average_3_month
FROM monthly
ORDER BY month;
