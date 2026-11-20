
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "clean_sales.csv"
OUTPUT_DIR = ROOT / "reports" / "abc_xyz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XYZ_X_LIMIT = 0.50
XYZ_Y_LIMIT = 1.00

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\n[1/7] Loading sales data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required = ["StockCode", "Quantity", "InvoiceDate", "Price"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["StockCode"] = df["StockCode"].astype("string").str.strip()

df = df.dropna(
    subset=["StockCode", "InvoiceDate", "Quantity", "Price"]
)

df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0) &
    (df["StockCode"] != "")
].copy()

if "Revenue" not in df.columns:
    df["Revenue"] = df["Quantity"] * df["Price"]

df = df[df["Revenue"] > 0].copy()

print("Valid transaction rows:", len(df))

# ==========================================
# 3. PRODUCT REVENUE AND ABC CLASSIFICATION
# ==========================================

print("\n[2/7] Performing ABC analysis...")

product = (
    df.groupby("StockCode")
    .agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Average_Selling_Price=("Price", "mean"),
        Transaction_Count=("InvoiceDate", "count")
    )
    .reset_index()
)

product = product.sort_values(
    "Total_Revenue", ascending=False
).reset_index(drop=True)

total_revenue = product["Total_Revenue"].sum()

if total_revenue <= 0:
    raise ValueError("Total revenue must be positive.")

product["Revenue_Share"] = (
    product["Total_Revenue"] / total_revenue
)

product["Cumulative_Revenue_Share"] = (
    product["Revenue_Share"].cumsum()
)

# Assign category based on cumulative contribution
previous_cumulative = (
    product["Cumulative_Revenue_Share"].shift(1).fillna(0)
)

product["ABC_Category"] = np.select(
    [
        previous_cumulative < 0.80,
        previous_cumulative < 0.95
    ],
    ["A", "B"],
    default="C"
)

# ==========================================
# 4. WEEKLY DEMAND AND XYZ CLASSIFICATION
# ==========================================

print("\n[3/7] Performing XYZ analysis...")

df["Week"] = (
    df["InvoiceDate"]
    .dt.to_period("W-SUN")
    .dt.start_time
)

weekly = (
    df.groupby(["StockCode", "Week"])["Quantity"]
    .sum()
    .reset_index()
)

all_weeks = pd.date_range(
    weekly["Week"].min(),
    weekly["Week"].max(),
    freq="W-MON"
)

all_products = product["StockCode"].tolist()

full_index = pd.MultiIndex.from_product(
    [all_products, all_weeks],
    names=["StockCode", "Week"]
)

weekly = (
    weekly.set_index(["StockCode", "Week"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

demand_stats = (
    weekly.groupby("StockCode")["Quantity"]
    .agg(
        Average_Weekly_Demand="mean",
        Demand_Std="std",
        Active_Weeks=lambda x: int((x > 0).sum()),
        Total_Weeks="count"
    )
    .reset_index()
)

demand_stats["Demand_Std"] = (
    demand_stats["Demand_Std"].fillna(0)
)

demand_stats["Demand_CV"] = np.where(
    demand_stats["Average_Weekly_Demand"] > 0,
    demand_stats["Demand_Std"] /
    demand_stats["Average_Weekly_Demand"],
    np.nan
)

demand_stats["XYZ_Category"] = np.select(
    [
        demand_stats["Demand_CV"] <= XYZ_X_LIMIT,
        demand_stats["Demand_CV"] <= XYZ_Y_LIMIT
    ],
    ["X", "Y"],
    default="Z"
)

# ==========================================
# 5. COMBINE ABC + XYZ
# ==========================================

print("\n[4/7] Combining ABC and XYZ...")

result = product.merge(
    demand_stats,
    on="StockCode",
    how="left"
)

result["ABC_XYZ"] = (
    result["ABC_Category"] +
    result["XYZ_Category"]
)

result = result.sort_values(
    ["ABC_Category", "XYZ_Category", "Total_Revenue"],
    ascending=[True, True, False]
)

result.to_csv(
    OUTPUT_DIR / "abc_xyz_product_classification.csv",
    index=False
)

# Matrix
matrix = pd.crosstab(
    result["ABC_Category"],
    result["XYZ_Category"]
).reindex(
    index=["A", "B", "C"],
    columns=["X", "Y", "Z"],
    fill_value=0
)

matrix.to_csv(
    OUTPUT_DIR / "abc_xyz_matrix.csv"
)

# ==========================================
# 6. PRODUCT RECOMMENDATIONS
# ==========================================

print("\n[5/7] Creating product recommendations...")

recommendations = {
    "AX": "High priority: stable demand. Maintain reliable replenishment.",
    "AY": "High priority: monitor variability and maintain safety stock.",
    "AZ": "High value but unpredictable. Use frequent review and close monitoring.",
    "BX": "Moderate priority: standard replenishment policy.",
    "BY": "Moderate priority: review demand trends regularly.",
    "BZ": "Use cautious replenishment and monitor demand changes.",
    "CX": "Low value and stable. Consider simple periodic replenishment.",
    "CY": "Low value with variable demand. Avoid excessive stock.",
    "CZ": "Low value and highly unpredictable. Review stocking necessity."
}

result["Business_Recommendation"] = (
    result["ABC_XYZ"].map(recommendations)
)

result.to_csv(
    OUTPUT_DIR / "abc_xyz_product_classification.csv",
    index=False
)

# ==========================================
# 7. VISUALIZATIONS
# ==========================================

print("\n[6/7] Creating charts...")

plt.figure(figsize=(12, 6))
plt.plot(
    np.arange(1, len(product) + 1),
    product["Cumulative_Revenue_Share"] * 100
)
plt.axhline(80, linestyle="--", label="80%")
plt.axhline(95, linestyle="--", label="95%")
plt.title("ABC Pareto Analysis")
plt.xlabel("Products ranked by revenue")
plt.ylabel("Cumulative revenue contribution (%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "abc_pareto_chart.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("ABC-XYZ Product Classification Matrix")
plt.xlabel("XYZ Demand Variability")
plt.ylabel("ABC Revenue Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "abc_xyz_heatmap.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
result["ABC_XYZ"].value_counts().sort_index().plot(
    kind="bar"
)
plt.title("Product Count by ABC-XYZ Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "abc_xyz_segment_count.png", dpi=300)
plt.close()

# ==========================================
# 8. SUMMARY
# ==========================================

print("\n[7/7] ABC-XYZ analysis completed.")

print("\nABC distribution:")
print(result["ABC_Category"].value_counts())

print("\nXYZ distribution:")
print(result["XYZ_Category"].value_counts())

print("\nABC-XYZ matrix:")
print(matrix)

print("\nOutput files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
