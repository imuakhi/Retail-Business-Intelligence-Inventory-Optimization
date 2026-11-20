
from pathlib import Path
import sqlite3
import pandas as pd
from streamlit import connection

from src.database.database import (
    get_connection,
    initialize_database
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_sales.csv"

CHUNK_SIZE = 50000


def normalize_customer(value):
    if pd.isna(value) or str(value).strip() == "":
        return "__UNKNOWN__"

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def load_warehouse():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    initialize_database()
    connection = get_connection()
    

    total_loaded = 0
    total_invalid = 0
    first_chunk = True

    try:
        for df in pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            required = [
                "Invoice",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "Price",
                "Customer ID",
                "Country"
            ]

            missing = set(required) - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing columns: {missing}"
                )

            df = df[required].copy()

            # Clean identifiers
            df["Invoice"] = df["Invoice"].fillna("").astype(str)
            df["StockCode"] = (
                df["StockCode"].fillna("__UNKNOWN__").astype(str)
            )
            df["Description"] = df["Description"].fillna("")
            df["Country"] = df["Country"].fillna(
                "__UNKNOWN__"
            ).astype(str)

            df["Customer ID"] = df["Customer ID"].apply(
                normalize_customer
            )

            # Convert numeric columns
            df["Quantity"] = pd.to_numeric(
                df["Quantity"], errors="coerce"
            )

            df["Price"] = pd.to_numeric(
                df["Price"], errors="coerce"
            )

            # Convert date
            dates = pd.to_datetime(
                df["InvoiceDate"], errors="coerce"
            )

            df["InvoiceDate"] = dates.dt.strftime("%Y-%m-%d")

            # Remove rows with unusable essential values
            valid = (
                df["Quantity"].notna()
                & df["Price"].notna()
                & df["InvoiceDate"].notna()
            )

            total_invalid += int((~valid).sum())
            df = df.loc[valid].copy()

            if df.empty:
                continue

            # Rename for warehouse staging table
            stage = df.rename(columns={
                "Invoice": "invoice",
                "StockCode": "stock_code",
                "Description": "description",
                "Quantity": "quantity",
                "InvoiceDate": "full_date",
                "Price": "unit_price",
                "Customer ID": "customer_id",
                "Country": "country_name"
            })

            stage = stage[[
                "invoice",
                "stock_code",
                "description",
                "quantity",
                "full_date",
                "unit_price",
                "customer_id",
                "country_name"
            ]]

            # Temporary staging data
            stage.to_sql(
                "stg_sales",
                connection,
                if_exists="replace",
                index=False
            )

            # Customer dimension
            connection.execute("""
                INSERT OR IGNORE INTO dim_customer(customer_id)
                SELECT DISTINCT customer_id
                FROM stg_sales
            """)

            # Product dimension
            connection.execute("""
                INSERT OR IGNORE INTO dim_product(
                    stock_code, description
                )
                SELECT stock_code, MAX(description)
                FROM stg_sales
                GROUP BY stock_code
            """)

            # Update descriptions if new information exists
            connection.execute("""
                UPDATE dim_product
                SET description = (
                    SELECT MAX(s.description)
                    FROM stg_sales s
                    WHERE s.stock_code = dim_product.stock_code
                )
                WHERE EXISTS (
                    SELECT 1
                    FROM stg_sales s
                    WHERE s.stock_code = dim_product.stock_code
                    AND s.description <> ''
                )
            """)

            # Country dimension
            connection.execute("""
                INSERT OR IGNORE INTO dim_country(country_name)
                SELECT DISTINCT country_name
                FROM stg_sales
            """)

            # Date dimension
            connection.execute("""
                INSERT OR IGNORE INTO dim_date(
                    date_key,
                    full_date,
                    year,
                    quarter,
                    month,
                    month_name,
                    week,
                    day,
                    day_of_week
                )
                SELECT DISTINCT
                    CAST(strftime('%Y%m%d', full_date) AS INTEGER),
                    full_date,
                    CAST(strftime('%Y', full_date) AS INTEGER),
                    ((CAST(strftime('%m', full_date) AS INTEGER) - 1) / 3) + 1,
                    CAST(strftime('%m', full_date) AS INTEGER),
                    strftime('%m', full_date),
                    CAST(strftime('%W', full_date) AS INTEGER),
                    CAST(strftime('%d', full_date) AS INTEGER),
                    CAST(strftime('%w', full_date) AS INTEGER)
                FROM stg_sales
            """)

            # Load fact table
            connection.execute("""
                INSERT INTO fact_sales(
                    invoice,
                    customer_key,
                    product_key,
                    date_key,
                    country_key,
                    quantity,
                    unit_price,
                    revenue,
                    is_return
                )
                SELECT
                    s.invoice,
                    c.customer_key,
                    p.product_key,
                    d.date_key,
                    co.country_key,
                    s.quantity,
                    s.unit_price,
                    s.quantity * s.unit_price,
                    CASE WHEN s.quantity < 0 THEN 1 ELSE 0 END
                FROM stg_sales s
                JOIN dim_customer c
                    ON c.customer_id = s.customer_id
                JOIN dim_product p
                    ON p.stock_code = s.stock_code
                JOIN dim_date d
                    ON d.full_date = s.full_date
                JOIN dim_country co
                    ON co.country_name = s.country_name
            """)

            connection.commit()

            loaded = len(stage)
            total_loaded += loaded

            print(
                f"Loaded {total_loaded:,} rows | "
                f"Invalid rows skipped: {total_invalid:,}"
            )

            first_chunk = False

        connection.execute("DROP TABLE IF EXISTS stg_sales")
        connection.commit()

        print("\nWAREHOUSE LOADING COMPLETED")
        print(f"Total loaded rows: {total_loaded:,}")
        print(f"Invalid rows skipped: {total_invalid:,}")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    load_warehouse()


# ==========================================================
# SAFE WAREHOUSE REBUILD
# Clear previous records before a complete reload.
# Child fact table must be cleared before dimension tables.
# ==========================================================

print("Clearing previous warehouse records...")

connection.execute("DELETE FROM fact_sales")
connection.execute("DELETE FROM dim_customer")
connection.execute("DELETE FROM dim_product")
connection.execute("DELETE FROM dim_date")
connection.execute("DELETE FROM dim_country")

connection.execute("DROP TABLE IF EXISTS stg_sales")

connection.commit()

print("Previous warehouse records cleared.")
