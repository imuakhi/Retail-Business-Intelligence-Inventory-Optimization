
import pandas as pd
import sqlite3
from pathlib import Path

# ==================================
# 1. LOAD CLEAN DATA
# ==================================

df = pd.read_csv("data/processed/clean_sales.csv")

print("Dataset Loaded:", df.shape)

# ==================================
# 2. CREATE DATABASE
# ==================================

db_dir = Path("data/database")
db_dir.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(
    db_dir / "retail_database.db"
)

# Store data in SQLite
df.to_sql(
    "sales",
    conn,
    if_exists="replace",
    index=False
)

print("Database Created Successfully!")

# ==================================
# 3. MONTHLY REVENUE
# ==================================

query_1 = """
SELECT
    strftime('%Y-%m', InvoiceDate) AS month,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM sales
GROUP BY month
ORDER BY month;
"""

monthly = pd.read_sql_query(query_1, conn)

print("\n1. MONTHLY REVENUE")
print(monthly.to_string(index=False))

# ==================================
# 4. TOP 10 PRODUCTS
# ==================================

query_2 = """
SELECT
    Description,
    SUM(Quantity) AS total_quantity,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM sales
GROUP BY Description
ORDER BY total_quantity DESC
LIMIT 10;
"""

products = pd.read_sql_query(query_2, conn)

print("\n2. TOP 10 PRODUCTS")
print(products.to_string(index=False))

# ==================================
# 5. COUNTRY-WISE REVENUE
# ==================================

query_3 = """
SELECT
    Country,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM sales
GROUP BY Country
ORDER BY total_revenue DESC
LIMIT 10;
"""

countries = pd.read_sql_query(query_3, conn)

print("\n3. TOP 10 COUNTRIES")
print(countries.to_string(index=False))

# ==================================
# 6. TOP CUSTOMERS
# ==================================

query_4 = """
SELECT
    "Customer ID",
    ROUND(SUM(Revenue), 2) AS total_spending
FROM sales
WHERE "Customer ID" IS NOT NULL
GROUP BY "Customer ID"
ORDER BY total_spending DESC
LIMIT 10;
"""

customers = pd.read_sql_query(query_4, conn)

print("\n4. TOP 10 CUSTOMERS")
print(customers.to_string(index=False))

# ==================================
# 7. AVERAGE ORDER VALUE
# ==================================

query_5 = """
SELECT
    ROUND(AVG(order_revenue), 2) AS average_order_value
FROM (
    SELECT
        Invoice,
        SUM(Revenue) AS order_revenue
    FROM sales
    GROUP BY Invoice
);
"""

aov = pd.read_sql_query(query_5, conn)

print("\n5. AVERAGE ORDER VALUE")
print(aov)

# ==================================
# 8. REPEAT CUSTOMERS
# ==================================

query_6 = """
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END)
        AS repeat_customers
FROM (
    SELECT
        "Customer ID",
        COUNT(DISTINCT Invoice) AS order_count
    FROM sales
    WHERE "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
);
"""

repeat = pd.read_sql_query(query_6, conn)

print("\n6. REPEAT CUSTOMER ANALYSIS")
print(repeat)

# ==================================
# 9. SAVE SQL RESULTS
# ==================================

output_dir = Path("reports/sql_results")
output_dir.mkdir(parents=True, exist_ok=True)

monthly.to_csv(output_dir / "monthly_revenue.csv", index=False)
products.to_csv(output_dir / "top_products.csv", index=False)
countries.to_csv(output_dir / "country_revenue.csv", index=False)
customers.to_csv(output_dir / "top_customers.csv", index=False)

# Close database
conn.close()

print("\nSQL BUSINESS ANALYSIS COMPLETED!")
