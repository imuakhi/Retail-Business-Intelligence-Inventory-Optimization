
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/clean_sales.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset Loaded:", df.shape)

# ==========================================
# 2. SELECT TOP 20 PRODUCTS
# ==========================================

top_products = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .nlargest(20)
    .index
)

df = df[df["StockCode"].isin(top_products)].copy()

# ==========================================
# 3. CREATE WEEKLY DEMAND
# ==========================================

df["Week"] = (
    df["InvoiceDate"]
    .dt.to_period("W")
    .dt.start_time
)

weekly = (
    df.groupby(["StockCode", "Week"])["Quantity"]
    .sum()
    .reset_index()
)

# Create complete weekly timeline
all_weeks = pd.date_range(
    weekly["Week"].min(),
    weekly["Week"].max(),
    freq="W-MON"
)

# Align original weeks to Monday
weekly["Week"] = (
    weekly["Week"] -
    pd.to_timedelta(weekly["Week"].dt.weekday, unit="D")
)

products = weekly["StockCode"].unique()

full_index = pd.MultiIndex.from_product(
    [products, all_weeks],
    names=["StockCode", "Week"]
)

weekly = (
    weekly.set_index(["StockCode", "Week"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

weekly = weekly.sort_values(
    ["StockCode", "Week"]
).reset_index(drop=True)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_features(data):

    data = data.copy()

    data["Lag_1"] = (
        data.groupby("StockCode")["Quantity"]
        .shift(1)
    )

    data["Lag_2"] = (
        data.groupby("StockCode")["Quantity"]
        .shift(2)
    )

    data["Lag_4"] = (
        data.groupby("StockCode")["Quantity"]
        .shift(4)
    )

    data["Rolling_Mean_4"] = (
        data.groupby("StockCode")["Quantity"]
        .transform(
            lambda x: x.shift(1).rolling(4).mean()
        )
    )

    data["Month"] = data["Week"].dt.month

    data["Week_Number"] = (
        data["Week"].dt.isocalendar().week.astype(int)
    )

    return data


weekly = create_features(weekly)

model_data = weekly.dropna().copy()

features = [
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Rolling_Mean_4",
    "Month",
    "Week_Number"
]

target = "Quantity"

# ==========================================
# 5. TIME-BASED TRAIN TEST SPLIT
# ==========================================

last_week = model_data["Week"].max()

test_start = last_week - pd.Timedelta(weeks=7)

train_data = model_data[
    model_data["Week"] < test_start
]

test_data = model_data[
    model_data["Week"] >= test_start
]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nTraining Rows:", len(train_data))
print("Testing Rows:", len(test_data))

# ==========================================
# 6. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 7. MODEL PREDICTION
# ==========================================

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)

# ==========================================
# 8. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\n========== MODEL EVALUATION ==========")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# ==========================================
# 9. COMPARE ACTUAL VS PREDICTED
# ==========================================

comparison = test_data[
    ["StockCode", "Week", "Quantity"]
].copy()

comparison["Predicted_Quantity"] = predictions

print("\nActual vs Predicted:")
print(comparison.head(15))

# ==========================================
# 10. FORECAST NEXT 4 WEEKS
# ==========================================

forecast_results = []

for product in products:

    history = weekly[
        weekly["StockCode"] == product
    ][["Week", "Quantity"]].copy()

    for step in range(1, 5):

        next_week = (
            history["Week"].max() +
            pd.Timedelta(weeks=1)
        )

        values = history["Quantity"].values

        row = pd.DataFrame([{
            "StockCode": product,
            "Week": next_week,
            "Lag_1": values[-1],
            "Lag_2": values[-2],
            "Lag_4": values[-4],
            "Rolling_Mean_4": np.mean(values[-4:]),
            "Month": next_week.month,
            "Week_Number": int(next_week.isocalendar().week)
        }])

        predicted = max(
            0,
            model.predict(row[features])[0]
        )

        predicted = round(predicted)

        forecast_results.append({
            "StockCode": product,
            "Week": next_week,
            "Predicted_Demand": predicted
        })

        history = pd.concat([
            history,
            pd.DataFrame([{
                "Week": next_week,
                "Quantity": predicted
            }])
        ], ignore_index=True)

forecast_df = pd.DataFrame(forecast_results)

# ==========================================
# 11. SAVE RESULTS
# ==========================================

output_dir = Path("reports/forecasting")
output_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_dir / "model_evaluation.csv",
    index=False
)

forecast_df.to_csv(
    output_dir / "future_demand_forecast.csv",
    index=False
)

# ==========================================
# 12. VISUALIZATION
# ==========================================

example_product = products[0]

actual_product = comparison[
    comparison["StockCode"] == example_product
]

plt.figure(figsize=(13, 6))

plt.plot(
    actual_product["Week"],
    actual_product["Quantity"],
    label="Actual Demand",
    marker="o"
)

plt.plot(
    actual_product["Week"],
    actual_product["Predicted_Quantity"],
    label="Predicted Demand",
    marker="o"
)

plt.title(f"Actual vs Predicted Demand: {example_product}")
plt.xlabel("Week")
plt.ylabel("Quantity Sold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    output_dir / "actual_vs_predicted.png"
)

plt.show()

print("\n========== FORECAST SUMMARY ==========")

print(forecast_df.head(20))

print("\nDEMAND FORECASTING COMPLETED!")
