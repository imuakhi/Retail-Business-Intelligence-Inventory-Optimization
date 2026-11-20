
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

ABC_PATH = ROOT / "reports" / "abc_xyz" / "abc_xyz_product_classification.csv"
FORECAST_PATH = ROOT / "reports" / "advanced_forecasting" / "future_demand_forecast.csv"

OUTPUT_DIR = ROOT / "reports" / "inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSUMPTIONS_PATH = (
    OUTPUT_DIR / "inventory_assumptions.csv"
)

RANDOM_SEED = 42
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645
REVIEW_PERIOD_WEEKS = 1

# ==========================================
# 2. LOAD INPUTS
# ==========================================

print("\n[1/7] Loading ABC-XYZ and forecast data...")

for path in [ABC_PATH, FORECAST_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

abc = pd.read_csv(ABC_PATH)
forecast = pd.read_csv(FORECAST_PATH)

forecast["Week"] = pd.to_datetime(
    forecast["Week"], errors="coerce"
)

forecast["Predicted_Demand"] = pd.to_numeric(
    forecast["Predicted_Demand"], errors="coerce"
)

forecast = forecast.dropna(
    subset=["StockCode", "Week", "Predicted_Demand"]
)

forecast = forecast[
    forecast["Predicted_Demand"] >= 0
].copy()

# ==========================================
# 3. CREATE INVENTORY ASSUMPTIONS
# ==========================================

print("\n[2/7] Preparing inventory assumptions...")

products = abc[["StockCode"]].drop_duplicates().copy()

if not ASSUMPTIONS_PATH.exists():

    rng = np.random.default_rng(RANDOM_SEED)

    assumptions = products.copy()

    assumptions["Lead_Time_Weeks"] = rng.integers(
        1, 7, size=len(assumptions)
    )

    assumptions["Current_Stock"] = np.nan

    assumptions["Ordering_Cost"] = rng.uniform(
        300, 1000, size=len(assumptions)
    ).round(2)

    assumptions["Holding_Cost_Rate"] = 0.25

    assumptions["Assumption_Type"] = "SIMULATED"

    assumptions.to_csv(
        ASSUMPTIONS_PATH,
        index=False
    )

    print("Created simulated assumptions file:")
    print(ASSUMPTIONS_PATH)

assumptions = pd.read_csv(ASSUMPTIONS_PATH)

required_assumptions = [
    "StockCode",
    "Lead_Time_Weeks",
    "Current_Stock",
    "Ordering_Cost",
    "Holding_Cost_Rate"
]

missing = [
    c for c in required_assumptions
    if c not in assumptions.columns
]

if missing:
    raise ValueError(
        f"Missing assumption columns: {missing}"
    )

# ==========================================
# 4. CALCULATE DEMAND STATISTICS
# ==========================================

print("\n[3/7] Calculating demand statistics...")

demand = (
    abc[[
        "StockCode",
        "Average_Weekly_Demand",
        "Demand_Std",
        "ABC_Category",
        "XYZ_Category",
        "ABC_XYZ",
        "Average_Selling_Price"
    ]]
    .copy()
)

# Forecast mean is used as an additional forward-looking signal.
forecast_stats = (
    forecast.groupby("StockCode")["Predicted_Demand"]
    .agg(
        Average_Forecast_Weekly="mean",
        Total_Forecast_4_Weeks="sum"
    )
    .reset_index()
)

demand = demand.merge(
    forecast_stats,
    on="StockCode",
    how="left"
)

demand["Average_Forecast_Weekly"] = (
    demand["Average_Forecast_Weekly"]
    .fillna(demand["Average_Weekly_Demand"])
)

# Blend historical demand with model forecast.
demand["Planning_Weekly_Demand"] = (
    0.5 * demand["Average_Weekly_Demand"] +
    0.5 * demand["Average_Forecast_Weekly"]
)

# ==========================================
# 5. MERGE ASSUMPTIONS
# ==========================================

print("\n[4/7] Merging simulated assumptions...")

inventory = demand.merge(
    assumptions,
    on="StockCode",
    how="left"
)

if inventory["Lead_Time_Weeks"].isna().any():
    raise ValueError("Some products have missing lead times.")

numeric_columns = [
    "Lead_Time_Weeks",
    "Current_Stock",
    "Ordering_Cost",
    "Holding_Cost_Rate"
]

for col in numeric_columns:
    inventory[col] = pd.to_numeric(
        inventory[col], errors="coerce"
    )

if inventory["Ordering_Cost"].isna().any():
    raise ValueError("Ordering cost contains missing values.")

if inventory["Holding_Cost_Rate"].isna().any():
    raise ValueError("Holding cost rate contains missing values.")

if (inventory["Lead_Time_Weeks"] <= 0).any():
    raise ValueError("Lead time must be positive.")

if (inventory["Ordering_Cost"] <= 0).any():
    raise ValueError("Ordering cost must be positive.")

if (inventory["Holding_Cost_Rate"] <= 0).any():
    raise ValueError("Holding cost rate must be positive.")

# Create simulated stock only where user has not supplied one.
rng = np.random.default_rng(RANDOM_SEED)

missing_stock = inventory["Current_Stock"].isna()

inventory.loc[missing_stock, "Current_Stock"] = (
    inventory.loc[missing_stock, "Planning_Weekly_Demand"]
    * rng.uniform(0.5, 5.0, size=missing_stock.sum())
)

if (inventory["Current_Stock"] < 0).any():
    raise ValueError("Current stock cannot be negative.")

# ==========================================
# 6. CALCULATE SAFETY STOCK, ROP AND EOQ
# ==========================================

print("\n[5/7] Calculating inventory policies...")

inventory["Lead_Time_Demand"] = (
    inventory["Planning_Weekly_Demand"] *
    inventory["Lead_Time_Weeks"]
)

inventory["Safety_Stock"] = (
    Z_SCORE *
    inventory["Demand_Std"].fillna(0) *
    np.sqrt(inventory["Lead_Time_Weeks"])
)

inventory["Reorder_Point"] = (
    inventory["Lead_Time_Demand"] +
    inventory["Safety_Stock"]
)

# Approximate annual demand from weekly demand.
inventory["Annual_Demand"] = (
    inventory["Planning_Weekly_Demand"] * 52
)

# Cost assumptions:
# Average selling price is NOT actual purchase cost.
# 60% of selling price is a simulated unit acquisition cost.
inventory["Estimated_Unit_Cost"] = (
    inventory["Average_Selling_Price"] * 0.60
)

inventory["Annual_Holding_Cost_Per_Unit"] = (
    inventory["Estimated_Unit_Cost"] *
    inventory["Holding_Cost_Rate"]
)

inventory["EOQ"] = np.sqrt(
    (
        2 *
        inventory["Annual_Demand"] *
        inventory["Ordering_Cost"]
    ) /
    inventory["Annual_Holding_Cost_Per_Unit"].replace(0, np.nan)
)

inventory["EOQ"] = (
    inventory["EOQ"].replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

inventory["Target_Stock_Level"] = (
    inventory["Planning_Weekly_Demand"] *
    (
        inventory["Lead_Time_Weeks"] +
        REVIEW_PERIOD_WEEKS
    ) +
    inventory["Safety_Stock"]
)

inventory["Inventory_Status"] = np.where(
    inventory["Current_Stock"] <= inventory["Reorder_Point"],
    "REORDER NOW",
    "SUFFICIENT STOCK"
)

inventory["Suggested_Order_Qty"] = np.where(
    inventory["Current_Stock"] <= inventory["Reorder_Point"],
    np.maximum(
        inventory["EOQ"],
        inventory["Target_Stock_Level"] -
        inventory["Current_Stock"]
    ),
    0
)

inventory["Suggested_Order_Qty"] = np.ceil(
    inventory["Suggested_Order_Qty"]
).astype(int)

# ==========================================
# 7. EXPORT REPORTS
# ==========================================

print("\n[6/7] Exporting inventory reports...")

inventory = inventory.sort_values(
    ["Inventory_Status", "ABC_Category"],
    ascending=[True, True]
)

inventory.to_csv(
    OUTPUT_DIR / "inventory_optimization_report.csv",
    index=False
)

reorder_list = inventory[
    inventory["Suggested_Order_Qty"] > 0
].copy()

reorder_list.to_csv(
    OUTPUT_DIR / "reorder_recommendations.csv",
    index=False
)

summary = (
    inventory.groupby("ABC_XYZ")
    .agg(
        Product_Count=("StockCode", "count"),
        Average_Weekly_Demand=("Planning_Weekly_Demand", "mean"),
        Average_Safety_Stock=("Safety_Stock", "mean"),
        Reorder_Product_Count=(
            "Suggested_Order_Qty",
            lambda x: int((x > 0).sum())
        )
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "inventory_segment_summary.csv",
    index=False
)

# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n[7/7] Inventory optimization completed.")

print("\nProducts:", len(inventory))

print(
    "Products requiring reorder:",
    len(reorder_list)
)

print(
    "Total suggested order quantity:",
    inventory["Suggested_Order_Qty"].sum()
)

print("\nInventory status:")
print(inventory["Inventory_Status"].value_counts())

print("\nOutput files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nIMPORTANT:")
print("Inventory assumptions are simulated.")
print("Estimated unit cost is not actual purchase cost.")
