
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "clean_sales.csv"
)

ABC_PATH = (
    PROJECT_ROOT
    / "reports"
    / "abc_xyz"
    / "abc_xyz_product_classification.csv"
)

INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "inventory"
    / "inventory_optimization_report.csv"
)

FORECAST_CANDIDATES = [
    PROJECT_ROOT
    / "reports"
    / "optuna_forecasting"
    / "tuned_future_forecast.csv",

    PROJECT_ROOT
    / "reports"
    / "advanced_ml_forecasting"
    / "future_demand_forecast_ml.csv",

    PROJECT_ROOT
    / "reports"
    / "advanced_forecasting"
    / "future_demand_forecast.csv"
]

OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "monte_carlo_inventory"
)

SIMULATIONS = 500
SIMULATION_WEEKS = 26
REVIEW_PERIOD_WEEKS = 1
MAX_PRODUCTS = 500
RANDOM_STATE = 42

# Assumptions used only when actual fields are unavailable.
DEFAULT_LEAD_TIME_WEEKS = 2
DEFAULT_ORDERING_COST = 50.0
DEFAULT_HOLDING_RATE = 0.25
DEFAULT_UNIT_COST_FACTOR = 0.60

# Assumed penalty for each unit of unmet demand.
STOCKOUT_PENALTY_MULTIPLIER = 2.0

# Service-level targets by ABC category.
SERVICE_LEVELS = {
    "A": 0.98,
    "B": 0.95,
    "C": 0.90,
    "DEFAULT": 0.95
}


# ==========================================================
# 2. GENERAL HELPERS
# ==========================================================

def normalize_column_name(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def find_column(df, aliases):
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    alias_names = [
        normalize_column_name(alias)
        for alias in aliases
    ]

    # Prefer exact matches.
    for alias in alias_names:
        if alias in normalized:
            return normalized[alias]

    # Then allow suffix/prefix matches for merged columns.
    for alias in alias_names:
        for normalized_name, original_name in normalized.items():
            if (
                normalized_name.startswith(alias)
                or normalized_name.endswith(alias)
            ):
                return original_name

    return None


def numeric_series(df, aliases, default):
    column = find_column(df, aliases)

    if column is None:
        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def clean_stock_codes(series):
    return series.astype("string").str.strip()


# ==========================================================
# 3. LOAD SALES AND CREATE WEEKLY DEMAND HISTORY
# ==========================================================

def load_weekly_history():

    if not SALES_PATH.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {SALES_PATH}"
        )

    required = [
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price"
    ]

    df = pd.read_csv(
        SALES_PATH,
        usecols=required,
        dtype={"StockCode": "string"},
        low_memory=False
    )

    df["StockCode"] = clean_stock_codes(
        df["StockCode"]
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "StockCode",
            "Quantity",
            "InvoiceDate",
            "Price"
        ]
    )

    # Positive quantities represent units sold.
    sales = df[
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (df["StockCode"] != "")
    ].copy()

    if sales.empty:
        raise ValueError(
            "No positive sales demand found."
        )

    sales["Week"] = (
        sales["InvoiceDate"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    weekly = (
        sales.groupby(
            ["StockCode", "Week"],
            as_index=False
        )["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Demand"})
    )

    all_weeks = pd.date_range(
        weekly["Week"].min(),
        weekly["Week"].max(),
        freq="W-MON"
    )

    # Build a zero-filled demand history per product.
    history = {}

    for stock_code, group in weekly.groupby("StockCode"):

        series = (
            group.set_index("Week")["Demand"]
            .reindex(all_weeks, fill_value=0)
            .astype(float)
        )

        history[str(stock_code)] = series.to_numpy()

    average_price = (
        sales.groupby("StockCode")["Price"]
        .mean()
        .to_dict()
    )

    historical_metrics = []

    for stock_code, values in history.items():

        historical_metrics.append({
            "StockCode": stock_code,
            "Historical_Weekly_Mean": float(
                np.mean(values)
            ),
            "Historical_Weekly_Std": float(
                np.std(values, ddof=1)
                if len(values) > 1 else 0
            ),
            "Historical_Active_Weeks": int(
                np.count_nonzero(values)
            ),
            "Historical_Weeks": len(values)
        })

    metrics = pd.DataFrame(historical_metrics)

    print(f"Products in historical demand: {len(history):,}")
    print(f"Historical weeks: {len(all_weeks):,}")

    return history, metrics, average_price


# ==========================================================
# 4. LOAD ABC-XYZ AND INVENTORY ASSUMPTIONS
# ==========================================================

def load_inventory_master(
    historical_metrics,
    average_price
):

    if not ABC_PATH.exists():
        raise FileNotFoundError(
            f"ABC-XYZ report not found: {ABC_PATH}"
        )

    abc = pd.read_csv(
        ABC_PATH,
        dtype={"StockCode": "string"},
        low_memory=False
    )

    stock_column = find_column(
        abc,
        ["StockCode"]
    )

    if stock_column is None:
        raise ValueError(
            "StockCode column missing from ABC-XYZ report."
        )

    abc["StockCode"] = clean_stock_codes(
        abc[stock_column]
    )

    abc = abc.drop_duplicates(
        subset=["StockCode"]
    )

    master = abc.copy()

    # Inventory optimization report is optional.
    # Existing values are reused when available.
    if INVENTORY_PATH.exists():

        inventory = pd.read_csv(
            INVENTORY_PATH,
            dtype={"StockCode": "string"},
            low_memory=False
        )

        inv_stock_col = find_column(
            inventory,
            ["StockCode"]
        )

        if inv_stock_col is not None:

            inventory["StockCode"] = clean_stock_codes(
                inventory[inv_stock_col]
            )

            inventory = inventory.drop_duplicates(
                subset=["StockCode"]
            )

            master = master.merge(
                inventory,
                on="StockCode",
                how="left",
                suffixes=("", "_inventory")
            )

            print("Existing inventory report loaded.")

    master = master.merge(
        historical_metrics,
        on="StockCode",
        how="left"
    )

    master["Average_Selling_Price"] = (
        master["StockCode"]
        .map(average_price)
    )

    master["Historical_Weekly_Mean"] = (
        master["Historical_Weekly_Mean"].fillna(0)
    )

    master["Historical_Weekly_Std"] = (
        master["Historical_Weekly_Std"].fillna(0)
    )

    # Reuse the existing average demand field if the new
    # historical measure is unavailable.
    historical_fallback = numeric_series(
        master,
        ["Average_Weekly_Demand"],
        0
    ).fillna(0)

    master["Historical_Weekly_Mean"] = (
        master["Historical_Weekly_Mean"]
        .where(
            master["Historical_Weekly_Mean"] > 0,
            historical_fallback
        )
    )

    # Resolve inventory fields with explicit fallbacks.
    master["Lead_Time_Weeks"] = numeric_series(
        master,
        [
            "Lead_Time_Weeks",
            "Lead_Time",
            "Supplier_Lead_Time_Weeks"
        ],
        DEFAULT_LEAD_TIME_WEEKS
    )

    master["Lead_Time_Weeks"] = (
        master["Lead_Time_Weeks"]
        .fillna(DEFAULT_LEAD_TIME_WEEKS)
        .clip(lower=1, upper=12)
        .round()
        .astype(int)
    )

    master["Ordering_Cost"] = numeric_series(
        master,
        ["Ordering_Cost", "Order_Cost"],
        DEFAULT_ORDERING_COST
    )

    master["Ordering_Cost"] = (
        master["Ordering_Cost"]
        .fillna(DEFAULT_ORDERING_COST)
        .clip(lower=0)
    )

    master["Holding_Cost_Rate"] = numeric_series(
        master,
        ["Holding_Cost_Rate", "Holding_Rate"],
        DEFAULT_HOLDING_RATE
    )

    master["Holding_Cost_Rate"] = (
        master["Holding_Cost_Rate"]
        .fillna(DEFAULT_HOLDING_RATE)
        .clip(lower=0)
    )

    existing_unit_cost = numeric_series(
        master,
        [
            "Unit_Cost",
            "Estimated_Unit_Cost",
            "Average_Unit_Cost"
        ],
        np.nan
    )

    proxy_unit_cost = (
        master["Average_Selling_Price"]
        * DEFAULT_UNIT_COST_FACTOR
    )

    master["Unit_Cost"] = (
        existing_unit_cost
        .where(existing_unit_cost.notna(), proxy_unit_cost)
        .fillna(1.0)
        .clip(lower=0.01)
    )

    existing_stock = numeric_series(
        master,
        [
            "Current_Stock",
            "Opening_Stock",
            "Initial_Stock",
            "Simulated_Current_Stock"
        ],
        np.nan
    )

    # Synthetic current stock fallback: approximately
    # three forecast weeks of stock.
    master["Current_Stock"] = existing_stock

    master = master.merge(
        historical_metrics[
            ["StockCode", "Historical_Weekly_Mean"]
        ].rename(
            columns={
                "Historical_Weekly_Mean": "_Fallback_Weekly_Mean"
            }
        ),
        on="StockCode",
        how="left",
        suffixes=("", "_duplicate")
    )

    fallback_stock = (
        master["_Fallback_Weekly_Mean"].fillna(0) * 3
    )

    master["Current_Stock"] = (
        master["Current_Stock"]
        .where(
            master["Current_Stock"].notna(),
            fallback_stock
        )
        .fillna(0)
        .clip(lower=0)
    )

    master = master.drop(
        columns=["_Fallback_Weekly_Mean"],
        errors="ignore"
    )

    # Determine ABC category.
    abc_column = find_column(
        master,
        ["ABC_Category", "ABC"]
    )

    xyz_column = find_column(
        master,
        ["XYZ_Category", "XYZ"]
    )

    if abc_column:
        master["ABC_Category_Clean"] = (
            master[abc_column]
            .astype(str)
            .str.upper()
            .str.strip()
            .str[0]
        )
    elif xyz_column and find_column(master, ["ABC_XYZ"]):
        combined_column = find_column(
            master,
            ["ABC_XYZ"]
        )

        master["ABC_Category_Clean"] = (
            master[combined_column]
            .astype(str)
            .str.upper()
            .str[0]
        )
    else:
        master["ABC_Category_Clean"] = "DEFAULT"

    master["Inventory_Assumptions_Used"] = (
        "Existing fields reused where available; "
        "missing fields use documented synthetic defaults"
    )

    return master


# ==========================================================
# 5. LOAD THE LATEST FUTURE FORECAST
# ==========================================================

def load_forecast():

    selected_path = next(
        (
            path
            for path in FORECAST_CANDIDATES
            if path.exists()
        ),
        None
    )

    if selected_path is None:
        print(
            "No future forecast file found. "
            "Historical weekly mean will be used."
        )

        return {}, "Historical weekly mean fallback"

    forecast = pd.read_csv(
        selected_path,
        dtype={"StockCode": "string"},
        low_memory=False
    )

    stock_col = find_column(
        forecast,
        ["StockCode"]
    )

    demand_col = find_column(
        forecast,
        ["Predicted_Demand", "Forecast_Demand"]
    )

    week_col = find_column(
        forecast,
        ["Week", "Forecast_Week", "Date"]
    )

    if stock_col is None or demand_col is None:
        raise ValueError(
            "Forecast file must contain StockCode "
            "and Predicted_Demand or Forecast_Demand."
        )

    forecast["StockCode"] = clean_stock_codes(
        forecast[stock_col]
    )

    forecast["Forecast_Demand"] = pd.to_numeric(
        forecast[demand_col],
        errors="coerce"
    ).clip(lower=0)

    if week_col is not None:

        forecast["Forecast_Week"] = pd.to_datetime(
            forecast[week_col],
            errors="coerce"
        )

        forecast = forecast.dropna(
            subset=["Forecast_Week", "Forecast_Demand"]
        )

        forecast = forecast.sort_values(
            ["StockCode", "Forecast_Week"]
        )

    else:

        forecast = forecast.dropna(
            subset=["Forecast_Demand"]
        )

    forecast_map = {}

    for stock_code, group in forecast.groupby("StockCode"):

        forecast_map[str(stock_code)] = (
            group["Forecast_Demand"]
            .astype(float)
            .to_numpy()
        )

    print(f"Forecast source: {selected_path}")
    print(f"Forecasted products: {len(forecast_map):,}")

    return forecast_map, str(selected_path.relative_to(PROJECT_ROOT))


# ==========================================================
# 6. GENERATE MONTE CARLO DEMAND SAMPLES
# ==========================================================

def generate_demand_samples(
    historical_values,
    target_weekly_means,
    rng
):

    historical_values = np.asarray(
        historical_values,
        dtype=float
    )

    target_weekly_means = np.asarray(
        target_weekly_means,
        dtype=float
    )

    runs = SIMULATIONS
    weeks = len(target_weekly_means)

    historical_mean = (
        float(np.mean(historical_values))
        if len(historical_values) > 0
        else 0.0
    )

    if historical_mean > 0:

        # Bootstrap historical demand to preserve
        # zero-demand weeks and historical variability.
        sampled = rng.choice(
            historical_values,
            size=(runs, weeks),
            replace=True
        )

        scale = (
            target_weekly_means
            / historical_mean
        )

        demand = sampled * scale.reshape(1, -1)

    else:

        # Fallback where there is no positive historical demand.
        demand = np.column_stack([
            rng.poisson(
                max(float(mean), 0),
                size=runs
            )
            for mean in target_weekly_means
        ])

    return np.maximum(
        np.rint(demand),
        0
    ).astype(float)


# ==========================================================
# 7. SIMULATE ONE INVENTORY REPLENISHMENT POLICY
# ==========================================================

def simulate_inventory_policy(
    demand_samples,
    opening_stock,
    lead_time,
    reorder_point,
    target_stock,
    unit_cost,
    ordering_cost,
    holding_cost_rate
):

    runs, total_weeks = demand_samples.shape

    horizon = SIMULATION_WEEKS

    on_hand = np.full(
        runs,
        max(float(opening_stock), 0)
    )

    pipeline = np.zeros(
        (runs, lead_time)
    )

    holding_cost_total = np.zeros(runs)
    ordering_cost_total = np.zeros(runs)
    stockout_cost_total = np.zeros(runs)

    total_demand = np.zeros(runs)
    total_lost_units = np.zeros(runs)

    stockout_weeks = np.zeros(runs)
    orders_placed = np.zeros(runs)

    inventory_sum = np.zeros(runs)

    for week in range(horizon):

        # Receive previously placed orders.
        receipts = pipeline[:, 0].copy()

        on_hand += receipts

        # Move pending orders one week closer to arrival.
        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = 0

        demand = demand_samples[:, week]

        fulfilled = np.minimum(
            on_hand,
            demand
        )

        lost = np.maximum(
            demand - fulfilled,
            0
        )

        on_hand -= fulfilled

        total_demand += demand
        total_lost_units += lost

        stockout_weeks += (lost > 0).astype(float)

        inventory_sum += on_hand

        # Weekly holding-cost approximation.
        holding_cost_total += (
            on_hand
            * unit_cost
            * holding_cost_rate
            / 52
        )

        stockout_cost_total += (
            lost
            * unit_cost
            * STOCKOUT_PENALTY_MULTIPLIER
        )

        # Inventory position includes pending replenishment.
        inventory_position = (
            on_hand + pipeline.sum(axis=1)
        )

        order_quantity = np.where(
            inventory_position <= reorder_point,
            np.maximum(
                target_stock - inventory_position,
                0
            ),
            0
        )

        placed = order_quantity > 0

        orders_placed += placed.astype(float)

        ordering_cost_total += (
            placed.astype(float) * ordering_cost
        )

        # Order arrives after the specified lead time.
        pipeline[:, lead_time - 1] += order_quantity

    total_cost = (
        holding_cost_total
        + ordering_cost_total
        + stockout_cost_total
    )

    demand_total = float(total_demand.sum())
    lost_total = float(total_lost_units.sum())

    return {
        "Avg_Total_Cost": float(total_cost.mean()),
        "Avg_Holding_Cost": float(
            holding_cost_total.mean()
        ),
        "Avg_Ordering_Cost": float(
            ordering_cost_total.mean()
        ),
        "Avg_Stockout_Cost": float(
            stockout_cost_total.mean()
        ),
        "Stockout_Probability": float(
            np.mean(total_lost_units > 0)
        ),
        "Fill_Rate": (
            1 - lost_total / demand_total
            if demand_total > 0
            else 1.0
        ),
        "Avg_Stockout_Weeks": float(
            stockout_weeks.mean()
        ),
        "Avg_Orders": float(
            orders_placed.mean()
        ),
        "Avg_On_Hand_Inventory": float(
            (inventory_sum / horizon).mean()
        ),
        "Avg_Lost_Units": float(
            total_lost_units.mean()
        )
    }


# ==========================================================
# 8. DETERMINE SERVICE LEVEL AND FORECAST MEAN
# ==========================================================

def get_service_level(category):

    category = str(category).upper().strip()

    if category in SERVICE_LEVELS:
        return SERVICE_LEVELS[category]

    return SERVICE_LEVELS["DEFAULT"]


def get_future_weekly_means(
    stock_code,
    forecast_map,
    historical_mean,
    total_weeks
):

    values = forecast_map.get(stock_code)

    if values is None or len(values) == 0:

        return np.full(
            total_weeks,
            max(float(historical_mean), 0)
        )

    values = np.asarray(
        values,
        dtype=float
    )

    values = np.maximum(
        np.nan_to_num(values, nan=0.0),
        0
    )

    if len(values) >= total_weeks:
        return values[:total_weeks]

    # Carry the final available forecast forward for
    # the portion of the 26-week simulation not covered.
    extension = np.full(
        total_weeks - len(values),
        values[-1]
    )

    return np.concatenate([
        values,
        extension
    ])


# ==========================================================
# 9. RUN PRODUCT-LEVEL MONTE CARLO OPTIMIZATION
# ==========================================================

def run_monte_carlo():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("MONTE CARLO INVENTORY OPTIMIZATION")
    print("=" * 65)

    # Load source data.
    history, historical_metrics, average_price = (
        load_weekly_history()
    )

    master = load_inventory_master(
        historical_metrics,
        average_price
    )

    forecast_map, forecast_source = load_forecast()

    # Limit the initial run to top products by historical revenue.
    revenue_column = find_column(
        master,
        ["Total_Revenue", "Revenue"]
    )

    if revenue_column:
        master["_Sort_Revenue"] = pd.to_numeric(
            master[revenue_column],
            errors="coerce"
        ).fillna(0)
    else:
        master["_Sort_Revenue"] = (
            master["Historical_Weekly_Mean"]
        )

    master = master.sort_values(
        "_Sort_Revenue",
        ascending=False
    )

    if MAX_PRODUCTS is not None:
        master = master.head(MAX_PRODUCTS).copy()

    if master.empty:
        raise ValueError(
            "No products available for simulation."
        )

    print(f"Products selected: {len(master):,}")
    print(f"Monte Carlo runs per policy: {SIMULATIONS:,}")
    print(f"Simulation horizon: {SIMULATION_WEEKS} weeks")

    parameter_rows = []
    detail_rows = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    total_products = len(master)

    for product_number, (_, row) in enumerate(
        master.iterrows(),
        start=1
    ):

        stock_code = str(row["StockCode"])

        if stock_code not in history:
            continue

        historical_values = history[stock_code]

        historical_mean = float(
            row["Historical_Weekly_Mean"]
        )

        future_means = get_future_weekly_means(
            stock_code,
            forecast_map,
            historical_mean,
            SIMULATION_WEEKS
            + int(row["Lead_Time_Weeks"])
            + REVIEW_PERIOD_WEEKS
        )

        forecast_mean = float(
            np.mean(future_means[:SIMULATION_WEEKS])
        )

        lead_time = int(
            row["Lead_Time_Weeks"]
        )

        category = row["ABC_Category_Clean"]

        service_level = get_service_level(
            category
        )

        opening_stock = float(
            row["Current_Stock"]
        )

        unit_cost = float(
            row["Unit_Cost"]
        )

        ordering_cost = float(
            row["Ordering_Cost"]
        )

        holding_rate = float(
            row["Holding_Cost_Rate"]
        )

        # Generate the same demand samples for both policies.
        # This makes the policy comparison fairer.
        demand_samples = generate_demand_samples(
            historical_values,
            future_means,
            rng
        )

        # Demand during supplier lead time.
        lead_time_demand = (
            demand_samples[:, :lead_time].sum(axis=1)
        )

        # Demand during lead time plus review period.
        target_window = (
            lead_time + REVIEW_PERIOD_WEEKS
        )

        protection_period_demand = (
            demand_samples[:, :target_window]
            .sum(axis=1)
        )

        # Dynamic service-level reorder point.
        optimized_rop = float(
            np.quantile(
                lead_time_demand,
                service_level
            )
        )

        mean_lead_time_demand = float(
            np.mean(lead_time_demand)
        )

        optimized_safety_stock = max(
            optimized_rop - mean_lead_time_demand,
            0
        )

        optimized_target = float(
            np.quantile(
                protection_period_demand,
                service_level
            )
        )

        optimized_target = max(
            optimized_target,
            optimized_rop
        )

        # Traditional policy follows the earlier project
        # baseline: ROP = 2 weeks of demand, target = 4 weeks.
        traditional_rop = (
            forecast_mean * 2
        )

        traditional_target = (
            forecast_mean * 4
        )

        traditional_target = max(
            traditional_target,
            traditional_rop
        )

        # Policy simulation.
        traditional_results = simulate_inventory_policy(
            demand_samples=demand_samples,
            opening_stock=opening_stock,
            lead_time=lead_time,
            reorder_point=traditional_rop,
            target_stock=traditional_target,
            unit_cost=unit_cost,
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_rate
        )

        optimized_results = simulate_inventory_policy(
            demand_samples=demand_samples,
            opening_stock=opening_stock,
            lead_time=lead_time,
            reorder_point=optimized_rop,
            target_stock=optimized_target,
            unit_cost=unit_cost,
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_rate
        )

        for policy_name, results in [
            ("Traditional", traditional_results),
            ("Monte_Carlo_Optimized", optimized_results)
        ]:

            detail_rows.append({
                "StockCode": stock_code,
                "ABC_Category": category,
                "Policy": policy_name,
                "Lead_Time_Weeks": lead_time,
                "Service_Level_Target": service_level,
                **results
            })

        parameter_rows.append({
            "StockCode": stock_code,
            "ABC_Category": category,
            "Historical_Weekly_Mean": historical_mean,
            "Forecast_Weekly_Mean": forecast_mean,
            "Historical_Weekly_Std": float(
                np.std(historical_values, ddof=1)
                if len(historical_values) > 1 else 0
            ),
            "Lead_Time_Weeks": lead_time,
            "Current_Stock": opening_stock,
            "Service_Level_Target": service_level,
            "Traditional_ROP": traditional_rop,
            "Traditional_Target_Stock": traditional_target,
            "Optimized_ROP": optimized_rop,
            "Optimized_Safety_Stock": optimized_safety_stock,
            "Optimized_Target_Stock": optimized_target,
            "Unit_Cost_Assumption": unit_cost,
            "Ordering_Cost_Assumption": ordering_cost,
            "Holding_Cost_Rate_Assumption": holding_rate
        })

        if (
            product_number % 50 == 0
            or product_number == total_products
        ):
            print(
                f"Processed {product_number:,} "
                f"of {total_products:,} products"
            )

    if not detail_rows:
        raise ValueError(
            "No products were simulated. "
            "Check StockCode consistency."
        )

    details = pd.DataFrame(detail_rows)
    parameters = pd.DataFrame(parameter_rows)

    # ------------------------------------------------------
    # Save detailed policy results
    # ------------------------------------------------------

    details.to_csv(
        OUTPUT_DIR / "monte_carlo_policy_details.csv",
        index=False
    )

    parameters.to_csv(
        OUTPUT_DIR / "dynamic_reorder_parameters.csv",
        index=False
    )

    # ------------------------------------------------------
    # Create product-level policy comparison
    # ------------------------------------------------------

    comparison = details.pivot(
        index="StockCode",
        columns="Policy",
        values=[
            "Avg_Total_Cost",
            "Stockout_Probability",
            "Fill_Rate",
            "Avg_On_Hand_Inventory",
            "Avg_Orders"
        ]
    )

    comparison.columns = [
        f"{metric}_{policy}"
        for metric, policy in comparison.columns
    ]

    comparison = comparison.reset_index()

    comparison["Cost_Saving"] = (
        comparison["Avg_Total_Cost_Traditional"]
        - comparison["Avg_Total_Cost_Monte_Carlo_Optimized"]
    )

    comparison["Cost_Saving_Percent"] = np.where(
        comparison["Avg_Total_Cost_Traditional"] > 0,
        100 * comparison["Cost_Saving"]
        / comparison["Avg_Total_Cost_Traditional"],
        np.nan
    )

    comparison["Stockout_Risk_Reduction"] = (
        comparison["Stockout_Probability_Traditional"]
        - comparison["Stockout_Probability_Monte_Carlo_Optimized"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "inventory_policy_comparison.csv",
        index=False
    )

    # ------------------------------------------------------
    # Aggregate policy summary
    # ------------------------------------------------------

    summary = (
        details.groupby("Policy")
        .agg(
            Products=("StockCode", "nunique"),
            Average_Total_Cost=("Avg_Total_Cost", "mean"),
            Average_Holding_Cost=("Avg_Holding_Cost", "mean"),
            Average_Ordering_Cost=("Avg_Ordering_Cost", "mean"),
            Average_Stockout_Cost=("Avg_Stockout_Cost", "mean"),
            Average_Stockout_Probability=("Stockout_Probability", "mean"),
            Average_Fill_Rate=("Fill_Rate", "mean"),
            Average_Inventory=("Avg_On_Hand_Inventory", "mean"),
            Average_Orders=("Avg_Orders", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "inventory_policy_summary.csv",
        index=False
    )

    # ------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------

    risk = comparison[
        [
            "StockCode",
            "Stockout_Probability_Monte_Carlo_Optimized",
            "Fill_Rate_Monte_Carlo_Optimized",
            "Cost_Saving_Percent"
        ]
    ].copy()

    risk["Inventory_Risk"] = np.select(
        [
            risk[
                "Stockout_Probability_Monte_Carlo_Optimized"
            ] >= 0.75,

            risk[
                "Stockout_Probability_Monte_Carlo_Optimized"
            ] >= 0.40
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    risk.to_csv(
        OUTPUT_DIR / "inventory_risk_report.csv",
        index=False
    )

    # ------------------------------------------------------
    # Scenario assumptions
    # ------------------------------------------------------

    assumptions = {
        "simulations_per_product": SIMULATIONS,
        "simulation_horizon_weeks": SIMULATION_WEEKS,
        "review_period_weeks": REVIEW_PERIOD_WEEKS,
        "max_products": MAX_PRODUCTS,
        "random_seed": RANDOM_STATE,
        "service_levels_by_abc": SERVICE_LEVELS,
        "default_lead_time_weeks": DEFAULT_LEAD_TIME_WEEKS,
        "default_ordering_cost": DEFAULT_ORDERING_COST,
        "default_holding_cost_rate": DEFAULT_HOLDING_RATE,
        "unit_cost_factor_of_average_selling_price": (
            DEFAULT_UNIT_COST_FACTOR
        ),
        "stockout_penalty_multiple_of_unit_cost": (
            STOCKOUT_PENALTY_MULTIPLIER
        ),
        "forecast_source": forecast_source,
        "demand_sampling": (
            "Bootstrap historical weekly demand and scale "
            "to future weekly forecast mean"
        ),
        "current_stock_fallback": (
            "Three weeks of forecast demand when no usable "
            "current-stock field is available"
        ),
        "limitations": [
            "Current stock and supplier lead time may be synthetic.",
            "Unit cost may be estimated as 60% of average selling price.",
            "Ordering and holding costs may be assumptions.",
            "Stockout penalty is a scenario assumption.",
            "The simulation is not a verified real-company outcome."
        ]
    }

    with open(
        OUTPUT_DIR / "simulation_assumptions.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            assumptions,
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Charts
    # ------------------------------------------------------

    plt.figure(figsize=(9, 6))

    summary.set_index("Policy")[
        [
            "Average_Holding_Cost",
            "Average_Ordering_Cost",
            "Average_Stockout_Cost"
        ]
    ].plot(
        kind="bar",
        stacked=True
    )

    plt.title("Inventory Cost Breakdown by Policy")
    plt.xlabel("Inventory Policy")
    plt.ylabel("Average Simulated Cost")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "inventory_cost_comparison.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["Policy"],
        summary["Average_Stockout_Probability"]
    )

    plt.title("Average Stockout Probability")
    plt.ylabel("Probability")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "stockout_probability_comparison.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["Policy"],
        summary["Average_Fill_Rate"]
    )

    plt.title("Average Inventory Fill Rate")
    plt.ylabel("Fill Rate")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "inventory_fill_rate_comparison.png",
        dpi=200
    )

    plt.close()

    # ------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------

    traditional = summary[
        summary["Policy"] == "Traditional"
    ].iloc[0]

    optimized = summary[
        summary["Policy"] == "Monte_Carlo_Optimized"
    ].iloc[0]

    cost_saving = (
        traditional["Average_Total_Cost"]
        - optimized["Average_Total_Cost"]
    )

    percentage_saving = (
        cost_saving / traditional["Average_Total_Cost"] * 100
        if traditional["Average_Total_Cost"] > 0
        else None
    )

    executive_summary = {
        "products_simulated": int(
            comparison["StockCode"].nunique()
        ),
        "simulations_per_product": SIMULATIONS,
        "simulation_horizon_weeks": SIMULATION_WEEKS,
        "traditional_average_cost": float(
            traditional["Average_Total_Cost"]
        ),
        "optimized_average_cost": float(
            optimized["Average_Total_Cost"]
        ),
        "estimated_cost_saving": float(cost_saving),
        "estimated_cost_saving_percent": (
            float(percentage_saving)
            if percentage_saving is not None
            else None
        ),
        "traditional_average_fill_rate": float(
            traditional["Average_Fill_Rate"]
        ),
        "optimized_average_fill_rate": float(
            optimized["Average_Fill_Rate"]
        ),
        "traditional_average_stockout_probability": float(
            traditional["Average_Stockout_Probability"]
        ),
        "optimized_average_stockout_probability": float(
            optimized["Average_Stockout_Probability"]
        ),
        "interpretation": (
            "Scenario simulation results using historical demand "
            "bootstrap and available or assumed inventory inputs."
        )
    }

    with open(
        OUTPUT_DIR / "inventory_executive_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            executive_summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("MONTE CARLO INVENTORY PIPELINE COMPLETED")
    print("=" * 65)

    print("\nPOLICY SUMMARY")
    print(summary.to_string(index=False))

    print("\nEXECUTIVE SUMMARY")
    print(json.dumps(executive_summary, indent=4))

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_monte_carlo()
