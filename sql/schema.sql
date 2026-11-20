
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key INTEGER PRIMARY KEY,
    customer_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_key INTEGER PRIMARY KEY,
    stock_code TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date TEXT NOT NULL UNIQUE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    month_name TEXT,
    week INTEGER,
    day INTEGER,
    day_of_week INTEGER
);

CREATE TABLE IF NOT EXISTS dim_country (
    country_key INTEGER PRIMARY KEY,
    country_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS fact_sales (
    sales_key INTEGER PRIMARY KEY,
    invoice TEXT,
    customer_key INTEGER,
    product_key INTEGER,
    date_key INTEGER,
    country_key INTEGER,
    quantity REAL,
    unit_price REAL,
    revenue REAL,
    is_return INTEGER,

    FOREIGN KEY (customer_key)
        REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key)
        REFERENCES dim_product(product_key),
    FOREIGN KEY (date_key)
        REFERENCES dim_date(date_key),
    FOREIGN KEY (country_key)
        REFERENCES dim_country(country_key)
);

CREATE INDEX IF NOT EXISTS idx_sales_date
ON fact_sales(date_key);

CREATE INDEX IF NOT EXISTS idx_sales_product
ON fact_sales(product_key);

CREATE INDEX IF NOT EXISTS idx_sales_customer
ON fact_sales(customer_key);

CREATE INDEX IF NOT EXISTS idx_sales_country
ON fact_sales(country_key);
