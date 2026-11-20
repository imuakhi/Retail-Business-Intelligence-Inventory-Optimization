
from src.database.database import get_connection

connection = get_connection()

tables = [
    "dim_customer",
    "dim_product",
    "dim_date",
    "dim_country",
    "fact_sales"
]

print("\nWAREHOUSE TABLE SUMMARY")

for table in tables:
    count = connection.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"{table}: {count:,} rows")

print("\nMONTHLY REVENUE")

query = """
SELECT
    d.year,
    d.month,
    ROUND(SUM(f.revenue), 2) AS revenue
FROM fact_sales f
JOIN dim_date d
    ON f.date_key = d.date_key
GROUP BY d.year, d.month
ORDER BY d.year, d.month
LIMIT 12;
"""

for row in connection.execute(query):
    print(row)

connection.close()
