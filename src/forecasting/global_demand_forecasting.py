
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_sales.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "advanced_ml_forecasting"
MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"

HORIZON = 12
TEST_WEEKS = 12
MAX_TRAIN_ROWS = 250_000
RANDOM_STATE = 42

LAGS = [1, 2, 3, 4, 8, 12]
ROLLING_WINDOWS = [4, 8, 12]

FEATURE_COLUMNS = [
    "product_code",
    "year",
    "month",
    "week_of_year",
    "time_index",
    "week_sin",
    "week_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_8",
    "lag_12",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_12",
    "rolling_std_4",
    "rolling_std_8",
    "rolling_std_12",
    "zero_rate_8",
]


# ==========================================================
# 1. LOAD AND AGGREGATE WEEKLY DEMAND
# ==========================================================

def load_weekly_demand():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    required_columns = [
        "StockCode",
        "Quantity",
        "InvoiceDate"
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        dtype={"StockCode": "string"},
        low_memory=False
    )

    df["StockCode"] = df["StockCode"].str.strip()
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(
        subset=["StockCode", "Quantity", "InvoiceDate"]
    )

    # Positive sold quantities represent product demand.
    # Returns are not counted as positive demand.
    df = df[
        (df["StockCode"] != "")
        & (df["Quantity"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No positive sales demand found.")

    # Monday is the beginning of each week.
    df["Week"] = (
        df["InvoiceDate"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    weekly = (
        df.groupby(["StockCode", "Week"], as_index=False)
        ["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Demand"})
    )

    # Create a continuous weekly panel for every product.
    products = sorted(weekly["StockCode"].unique())

    all_weeks = pd.date_range(
        start=weekly["Week"].min(),
        end=weekly["Week"].max(),
        freq="W-MON"
    )

    panel = pd.MultiIndex.from_product(
        [products, all_weeks],
        names=["StockCode", "Week"]
    ).to_frame(index=False)

    panel = panel.merge(
        weekly,
        on=["StockCode", "Week"],
        how="left"
    )

    panel["Demand"] = panel["Demand"].fillna(0.0)

    product_map = {
        stock_code: index
        for index, stock_code in enumerate(products)
    }

    panel["product_code"] = (
        panel["StockCode"]
        .map(product_map)
        .astype("int32")
    )

    panel = panel.sort_values(
        ["StockCode", "Week"]
    ).reset_index(drop=True)

    print(f"Products: {len(products):,}")
    print(f"Weeks: {len(all_weeks):,}")
    print(f"Weekly panel rows: {len(panel):,}")

    return panel, product_map, all_weeks


# ==========================================================
# 2. FEATURE ENGINEERING
# ==========================================================

def create_features(panel):

    panel = panel.copy()

    panel["Week"] = pd.to_datetime(panel["Week"])

    panel["year"] = panel["Week"].dt.year
    panel["month"] = panel["Week"].dt.month

    panel["week_of_year"] = (
        panel["Week"].dt.isocalendar().week.astype(int)
    )

    panel["time_index"] = (
        (panel["Week"] - panel["Week"].min()).dt.days // 7
    ).astype(int)

    panel["week_sin"] = np.sin(
        2 * np.pi * panel["week_of_year"] / 52.18
    )

    panel["week_cos"] = np.cos(
        2 * np.pi * panel["week_of_year"] / 52.18
    )

    grouped = panel.groupby(
        "StockCode", sort=False
    )["Demand"]

    for lag in LAGS:
        panel[f"lag_{lag}"] = grouped.shift(lag)

    shifted = panel.groupby(
        "StockCode", sort=False
    )["Demand"].shift(1)

    for window in ROLLING_WINDOWS:

        panel[f"rolling_mean_{window}"] = (
            shifted.groupby(panel["StockCode"])
            .transform(
                lambda values: values.rolling(
                    window, min_periods=1
                ).mean()
            )
        )

        panel[f"rolling_std_{window}"] = (
            shifted.groupby(panel["StockCode"])
            .transform(
                lambda values: values.rolling(
                    window, min_periods=2
                ).std()
            )
        )

    panel["zero_rate_8"] = (
        shifted.eq(0)
        .groupby(panel["StockCode"])
        .transform(
            lambda values: values.rolling(
                8, min_periods=1
            ).mean()
        )
    )

    return panel


# ==========================================================
# 3. EVALUATION METRICS
# ==========================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum()
        / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_pct": wape
    }


# ==========================================================
# 4. MODEL TRAINING
# ==========================================================

def train_models(train):

    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = np.log1p(train["Demand"].clip(lower=0))

    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "LightGBM": LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        print(f"{name} training completed.")

    return models


# ==========================================================
# 5. MODEL COMPARISON
# ==========================================================

def evaluate_models(models, test):

    X_test = test[FEATURE_COLUMNS]
    actual = test["Demand"].to_numpy()

    predictions = test[
        ["StockCode", "Week", "Demand", "lag_1"]
    ].copy()

    predictions = predictions.rename(
        columns={
            "Demand": "Actual_Demand",
            "lag_1": "Previous_Week_Demand"
        }
    )

    results = []

    # Baseline: last week's demand.
    baseline = np.maximum(
        test["lag_1"].fillna(0).to_numpy(),
        0
    )

    baseline_metrics = calculate_metrics(
        actual, baseline
    )

    results.append({
        "Model": "Baseline_Lag1",
        **baseline_metrics
    })

    for name, model in models.items():

        prediction = np.maximum(
            np.expm1(model.predict(X_test)),
            0
        )

        predictions[f"{name}_Prediction"] = prediction

        metrics = calculate_metrics(
            actual, prediction
        )

        results.append({
            "Model": name,
            **metrics
        })

    comparison = pd.DataFrame(results).sort_values(
        "WAPE_pct",
        na_position="last"
    )

    return comparison, predictions


# ==========================================================
# 6. RECURSIVE FUTURE FORECASTING
# ==========================================================

def forecast_future(
    panel,
    model,
    product_map,
    all_weeks,
    horizon
):

    history = {}

    for stock_code, group in panel.groupby("StockCode"):
        history[stock_code] = (
            group.sort_values("Week")["Demand"]
            .astype(float)
            .tolist()
        )

    last_week = pd.Timestamp(all_weeks.max())
    first_week = pd.Timestamp(all_weeks.min())

    forecast_rows = []

    for step in range(1, horizon + 1):

        future_week = last_week + pd.Timedelta(
            weeks=step
        )

        week_number = int(
            future_week.isocalendar().week
        )

        feature_rows = []
        stock_codes = list(history.keys())

        for stock_code in stock_codes:

            values = history[stock_code]

            row = {
                "product_code": product_map[stock_code],
                "year": future_week.year,
                "month": future_week.month,
                "week_of_year": week_number,
                "time_index": int(
                    (future_week - first_week).days // 7
                ),
                "week_sin": np.sin(
                    2 * np.pi * week_number / 52.18
                ),
                "week_cos": np.cos(
                    2 * np.pi * week_number / 52.18
                )
            }

            for lag in LAGS:
                row[f"lag_{lag}"] = (
                    values[-lag]
                    if len(values) >= lag
                    else np.nan
                )

            for window in ROLLING_WINDOWS:

                recent = np.asarray(
                    values[-window:], dtype=float
                )

                row[f"rolling_mean_{window}"] = (
                    float(np.mean(recent))
                    if len(recent) > 0
                    else np.nan
                )

                row[f"rolling_std_{window}"] = (
                    float(np.std(recent, ddof=1))
                    if len(recent) > 1
                    else np.nan
                )

            recent_8 = np.asarray(
                values[-8:], dtype=float
            )

            row["zero_rate_8"] = (
                float(np.mean(recent_8 == 0))
                if len(recent_8) > 0
                else np.nan
            )

            feature_rows.append(row)

        X_future = pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLUMNS
        )

        predicted = np.maximum(
            np.expm1(model.predict(X_future)),
            0
        )

        for stock_code, demand in zip(
            stock_codes, predicted
        ):

            demand = float(demand)

            history[stock_code].append(demand)

            forecast_rows.append({
                "StockCode": stock_code,
                "Week": future_week.strftime("%Y-%m-%d"),
                "Predicted_Demand": round(demand, 3)
            })

        print(
            f"Forecast completed: "
            f"{future_week.strftime('%Y-%m-%d')}"
        )

    return pd.DataFrame(forecast_rows)


# ==========================================================
# 7. MAIN PIPELINE
# ==========================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ADVANCED GLOBAL DEMAND FORECASTING")
    print("=" * 60)

    # Load data
    panel, product_map, all_weeks = load_weekly_demand()

    if len(all_weeks) <= TEST_WEEKS + 12:
        raise ValueError(
            "Insufficient weekly history for this validation setup."
        )

    # Feature engineering
    print("\nCreating time-series features...")

    featured = create_features(panel)

    featured = featured.dropna(
        subset=["lag_1"]
    ).reset_index(drop=True)

    test_start = pd.Timestamp(
        all_weeks[-TEST_WEEKS]
    )

    train = featured[
        featured["Week"] < test_start
    ].copy()

    test = featured[
        featured["Week"] >= test_start
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Test begins: {test_start.date()}")

    # Train models
    models = train_models(train)

    # Evaluate
    comparison, test_predictions = evaluate_models(
        models, test
    )

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    test_predictions.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    # Select best ML model, excluding the baseline.
    ml_comparison = comparison[
        comparison["Model"].isin(models.keys())
    ].dropna(subset=["WAPE_pct"])

    if ml_comparison.empty:
        raise ValueError(
            "Model evaluation failed; no valid WAPE score."
        )

    best_name = ml_comparison.iloc[0]["Model"]
    best_model = models[best_name]

    print(f"\nSelected model: {best_name}")

    # Save the selected model and feature list.
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURE_COLUMNS,
        "product_map": product_map
    }

    joblib.dump(
        model_bundle,
        MODEL_DIR / "best_demand_model.joblib"
    )

    # Feature importance
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": best_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    # Future forecast
    future = forecast_future(
        panel=panel,
        model=best_model,
        product_map=product_map,
        all_weeks=all_weeks,
        horizon=HORIZON
    )

    future.to_csv(
        REPORT_DIR / "future_demand_forecast_ml.csv",
        index=False
    )

    # Summary
    best_metrics = ml_comparison[
        ml_comparison["Model"] == best_name
    ].iloc[0]

    summary = pd.DataFrame([{
        "Selected_Model": best_name,
        "Products_Forecasted": future["StockCode"].nunique(),
        "Forecast_Horizon_Weeks": HORIZON,
        "Training_Rows_Used": min(
            len(train), MAX_TRAIN_ROWS
        ),
        "Testing_Rows": len(test),
        "Test_Start_Week": test_start.date(),
        "MAE": best_metrics["MAE"],
        "RMSE": best_metrics["RMSE"],
        "WAPE_pct": best_metrics["WAPE_pct"]
    }])

    summary.to_csv(
        REPORT_DIR / "forecast_summary.csv",
        index=False
    )

    print("\nFORECASTING PIPELINE COMPLETED")
    print(summary.to_string(index=False))

    print(f"\nReports saved in: {REPORT_DIR}")
    print(f"Model saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
