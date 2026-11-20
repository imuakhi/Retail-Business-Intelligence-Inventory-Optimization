
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================
# 1. LOAD CLEAN DATA
# =====================================

df = pd.read_csv("data/processed/clean_sales.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove customers with missing ID
df = df.dropna(subset=["Customer ID"]).copy()

# =====================================
# 2. CREATE RFM TABLE
# =====================================

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({

    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,

    "Invoice": "nunique",

    "Revenue": "sum"

})

rfm.rename(columns={

    "InvoiceDate": "Recency",
    "Invoice": "Frequency",
    "Revenue": "Monetary"

}, inplace=True)

print("\nRFM TABLE:")
print(rfm.head())

# =====================================
# 3. CALCULATE RFM SCORES
# =====================================

# Rank first to avoid duplicate quantile boundaries
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

rfm["RFM_Total"] = (
    rfm["R_Score"] +
    rfm["F_Score"] +
    rfm["M_Score"]
)

# =====================================
# 4. CUSTOMER SEGMENTATION
# =====================================

def segment_customer(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f >= 2:
        return "Potential Loyalists"

    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2:
        return "Hibernating"

    else:
        return "Others"


rfm["Segment"] = rfm.apply(
    segment_customer,
    axis=1
)

# =====================================
# 5. SEGMENT SUMMARY
# =====================================

segment_summary = rfm.groupby("Segment").agg(

    Customer_Count=("Recency", "count"),

    Average_Recency=("Recency", "mean"),

    Average_Frequency=("Frequency", "mean"),

    Average_Monetary=("Monetary", "mean")

).round(2)

print("\nCUSTOMER SEGMENT SUMMARY:")
print(segment_summary)

# =====================================
# 6. VISUALIZE SEGMENTS
# =====================================

output_dir = Path("reports/rfm")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

segment_counts = rfm["Segment"].value_counts()

sns.barplot(
    x=segment_counts.values,
    y=segment_counts.index
)

plt.title("Customer Segmentation Distribution")
plt.xlabel("Number of Customers")
plt.ylabel("Customer Segment")

plt.tight_layout()

plt.savefig(output_dir / "customer_segments.png")
plt.show()

# =====================================
# 7. SAVE RFM RESULTS
# =====================================

rfm.to_csv(
    output_dir / "rfm_customer_data.csv"
)

segment_summary.to_csv(
    output_dir / "rfm_segment_summary.csv"
)

print("\nTotal Customers:", len(rfm))

print("\nRFM ANALYSIS COMPLETED SUCCESSFULLY!")
