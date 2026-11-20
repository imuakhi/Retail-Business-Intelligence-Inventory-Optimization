
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================
# 1. LOAD CLEAN DATA
# =====================================

df = pd.read_csv("data/processed/clean_sales.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())

# Create output folder
output_dir = Path("reports/eda_charts")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# =====================================
# 2. MONTHLY SALES ANALYSIS
# =====================================

df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("YearMonth")["Revenue"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=monthly_sales,
    x="YearMonth",
    y="Revenue",
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(output_dir / "monthly_sales.png")
plt.show()

# =====================================
# 3. TOP 10 PRODUCTS
# =====================================

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_products,
    x="Quantity",
    y="Description"
)

plt.title("Top 10 Best Selling Products")
plt.tight_layout()

plt.savefig(output_dir / "top_products.png")
plt.show()

# =====================================
# 4. COUNTRY-WISE REVENUE
# =====================================

country_sales = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=country_sales,
    x="Revenue",
    y="Country"
)

plt.title("Top 10 Countries by Revenue")
plt.tight_layout()

plt.savefig(output_dir / "country_revenue.png")
plt.show()

# =====================================
# 5. CUSTOMER ANALYSIS
# =====================================

customer_df = df.dropna(subset=["Customer ID"])

top_customers = (
    customer_df.groupby("Customer ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Customers by Revenue:")
print(top_customers)

# =====================================
# 6. REVENUE DISTRIBUTION
# =====================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Revenue",
    bins=50,
    kde=True
)

plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.tight_layout()

plt.savefig(output_dir / "revenue_distribution.png")
plt.show()

# =====================================
# 7. BUSINESS SUMMARY
# =====================================

print("\n========== BUSINESS SUMMARY ==========")

print("Total Revenue:", round(df["Revenue"].sum(), 2))

print("Total Quantity Sold:", df["Quantity"].sum())

print("Unique Products:", df["StockCode"].nunique())

print("Unique Customers:", df["Customer ID"].nunique())

print("Total Transactions:", df["Invoice"].nunique())

print("\nBest Performing Country:")
print(country_sales.iloc[0])

print("\nBest Selling Product:")
print(top_products.iloc[0])

print("\nEDA Analysis Completed Successfully!")
