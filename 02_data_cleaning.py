
import pandas as pd
from pathlib import Path

# ==================================
# 1. LOAD DATA
# ==================================

file_path = "online_retail_II.xlsx"

df_2009 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

# Combine both sheets
df = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

print("Original Dataset:", df.shape)

# ==================================
# 2. BASIC PREPROCESSING
# ==================================

df["Invoice"] = df["Invoice"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Description"] = df["Description"].str.strip()

# ==================================
# 3. IDENTIFY DATA ISSUES
# ==================================

df["Is_Cancelled"] = df["Invoice"].str.startswith("C")

df["Is_Return"] = df["Quantity"] < 0

df["Is_Invalid_Price"] = df["Price"] <= 0

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nCancelled Invoices:", df["Is_Cancelled"].sum())

print("\nNegative Quantity:", df["Is_Return"].sum())

print("\nInvalid Prices:", df["Is_Invalid_Price"].sum())

# ==================================
# 4. SEPARATE RETURNS
# ==================================

returns_df = df[
    (df["Is_Cancelled"]) |
    (df["Is_Return"])
].copy()

# ==================================
# 5. CREATE CLEAN SALES DATA
# ==================================

clean_df = df[
    (~df["Is_Cancelled"]) &
    (~df["Is_Return"]) &
    (~df["Is_Invalid_Price"]) &
    (df["Quantity"] > 0) &
    (df["InvoiceDate"].notna()) &
    (df["StockCode"].notna())
].copy()

# Remove exact duplicate records
clean_df = clean_df.drop_duplicates()

# ==================================
# 6. CALCULATE REVENUE
# ==================================

clean_df["Revenue"] = (
    clean_df["Quantity"] * clean_df["Price"]
)

# ==================================
# 7. SAVE PROCESSED DATA
# ==================================

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

clean_df.to_csv(
    output_dir / "clean_sales.csv",
    index=False
)

returns_df.to_csv(
    output_dir / "returns_transactions.csv",
    index=False
)

# ==================================
# 8. FINAL SUMMARY
# ==================================

print("\n========== CLEANING SUMMARY ==========")

print("Original Rows:", len(df))

print("Clean Sales Rows:", len(clean_df))

print("Returns Rows:", len(returns_df))

print("Total Revenue:", round(clean_df["Revenue"].sum(), 2))

print("\nClean Dataset Shape:", clean_df.shape)

print("\nMissing Customer IDs:", clean_df["Customer ID"].isnull().sum())

print("\nData Cleaning Completed Successfully!")
