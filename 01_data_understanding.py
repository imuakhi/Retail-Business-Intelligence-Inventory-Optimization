
import pandas as pd
import numpy as np

# Load dataset
file_path = "online_retail_II.xlsx"

# Read both sheets
df_2009 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

# Combine datasets
df = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

# Dataset inspection
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

