
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

INVENTORY_PATH = (
    ROOT / "reports" / "inventory" /
    "inventory_optimization_report.csv"
)

OUTPUT_DIR = ROOT / "reports" / "simulation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIMULATION_WEEKS = 26
RANDOM_SEED = 42

SERVICE_LEVEL_Z = 1.645
REVIEW_PERIOD_WEEKS = 1

# Simulated stockout penalty as a fraction
# of estimated acquisition cost.
STOCKOUT_PENALTY_RATE = 0.50

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\n[1/5] Loading inventory recommendations...")

if not INVENTORY_PATH.exists():
    raise FileNotFoundError(
        f"Inventory report not found: {INVENTORY_PATH}"
    )

df = pd.read_csv(INVENTORY_PATH)

required = [
    "StockCode",
    "Planning_Weekly_Demand",
    "Demand_Std",
    "Lead_Time_Weeks",
    "Current_Stock",
    "EOQ",
    "Estimated_Unit_Cost",
    "Holding_Cost_Rate"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in required[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("No valid inventory products available.")

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================

print("\n[2/5] Defining simulation engine...")

def simulate_policy(
    product,
    weekly_demand,
    policy_name,
    initial_stock,
    avg_demand,
    demand_std,
    lead_time,
    eoq,
    unit_cost,
    holding_rate
):

    stock = float(initial_stock)

    # Pending replenishment orders:
    # (arrival_week, quantity)
    pipeline = []

    total_demand = 0
    total_fulfilled = 0
    total_stockout_units = 0
    stockout_weeks = 0

    total_ordering_cost = 0
    total_holding_cost = 0
    total_stockout_cost = 0

    inventory_sum = 0
    order_count = 0

    records = []

    safety_stock = (
        SERVICE_LEVEL_Z *
        demand_std *
        np.sqrt(lead_time)
    )

    if policy_name == "Traditional":

        reorder_point = avg_demand * 2
        target_level = avg_demand * 4

    else:

        reorder_point = (
            avg_demand * lead_time +
            safety_stock
        )

        target_level = (
            avg_demand *
            (lead_time + REVIEW_PERIOD_WEEKS) +
            safety_stock
        )

    for week in range(1, SIMULATION_WEEKS + 1):

        # ----------------------------------
        # A. RECEIVE ARRIVING ORDERS
        # ----------------------------------

        arriving = [
            order for order in pipeline
            if order[0] <= week
        ]

        for arrival_week, quantity in arriving:
            stock += quantity

        pipeline = [
            order for order in pipeline
            if order[0] > week
        ]

        # ----------------------------------
        # B. FULFIL CUSTOMER DEMAND
        # ----------------------------------

        demand = float(weekly_demand[week - 1])

        fulfilled = min(stock, demand)

        stock -= fulfilled

        shortage = demand - fulfilled

        total_demand += demand
        total_fulfilled += fulfilled
        total_stockout_units += shortage

        if shortage > 0:
            stockout_weeks += 1

        # ----------------------------------
        # C. INVENTORY COSTS
        # ----------------------------------

        average_inventory_proxy = stock

        weekly_holding_rate = (
            unit_cost * holding_rate / 52
        )

        holding_cost = (
            average_inventory_proxy *
            weekly_holding_rate
        )

        stockout_cost = (
            shortage *
            unit_cost *
            STOCKOUT_PENALTY_RATE
        )

        total_holding_cost += holding_cost
        total_stockout_cost += stockout_cost

        # ----------------------------------
        # D. CHECK INVENTORY POSITION
        # ----------------------------------

        on_order = sum(
            quantity for arrival_week, quantity in pipeline
        )

        inventory_position = stock + on_order

        order_quantity = 0

        if inventory_position <= reorder_point:

            if policy_name == "Traditional":

                order_quantity = max(
                    0,
                    np.ceil(target_level - inventory_position)
                )

            else:

                order_quantity = max(
                    0,
                    np.ceil(
                        max(
                            eoq,
                            target_level - inventory_position
                        )
                    )
                )

            if order_quantity > 0:

                arrival_week = week + int(lead_time)

                pipeline.append(
                    (arrival_week, order_quantity)
                )

                total_ordering_cost += (
                    float(product["Ordering_Cost"])
                )

                order_count += 1

        inventory_sum += stock

        records.append({
            "StockCode": product["StockCode"],
            "Policy": policy_name,
            "Week": week,
            "Demand": demand,
            "Fulfilled": fulfilled,
            "Stockout_Units": shortage,
            "Ending_Inventory": stock,
            "Order_Quantity": order_quantity,
            "Inventory_Position": inventory_position
        })

    # --------------------------------------
    # E. PERFORMANCE METRICS
    # --------------------------------------

    fill_rate = (
        total_fulfilled / total_demand * 100
        if total_demand > 0 else 100
    )

    total_cost = (
        total_ordering_cost +
        total_holding_cost +
        total_stockout_cost
    )

    summary = {
        "StockCode": product["StockCode"],
        "Policy": policy_name,
        "Total_Demand": total_demand,
        "Fulfilled_Demand": total_fulfilled,
        "Stockout_Units": total_stockout_units,
        "Stockout_Weeks": stockout_weeks,
        "Fill_Rate_Percent": fill_rate,
        "Average_Inventory": (
            inventory_sum / SIMULATION_WEEKS
        ),
        "Orders_Placed": order_count,
        "Ordering_Cost": total_ordering_cost,
        "Holding_Cost": total_holding_cost,
        "Stockout_Cost": total_stockout_cost,
        "Total_Cost": total_cost
    }

    return summary, records


# ==========================================
# 4. RUN BOTH POLICIES
# ==========================================

print("\n[3/5] Running inventory simulations...")

rng = np.random.default_rng(RANDOM_SEED)

all_summaries = []
all_records = []

for _, row in df.iterrows():

    avg_demand = max(
        0,
        float(row["Planning_Weekly_Demand"])
    )

    demand_std = max(
        0,
        float(row["Demand_Std"])
    )

    lead_time = max(
        1,
        int(np.ceil(row["Lead_Time_Weeks"]))
    )

    initial_stock = max(
        0,
        float(row["Current_Stock"])
    )

    eoq = max(0, float(row["EOQ"]))

    unit_cost = max(
        0.01,
        float(row["Estimated_Unit_Cost"])
    )

    holding_rate = max(
        0,
        float(row["Holding_Cost_Rate"])
    )

    # Generate ONE demand sequence per product.
    # Both policies receive the same sequence for fair comparison.
    simulated_demand = rng.normal(
        loc=avg_demand,
        scale=demand_std,
        size=SIMULATION_WEEKS
    )

    simulated_demand = np.maximum(
        0,
        np.round(simulated_demand)
    )

    for policy in ["Traditional", "Optimized"]:

        summary, records = simulate_policy(
            product=row,
            weekly_demand=simulated_demand,
            policy_name=policy,
            initial_stock=initial_stock,
            avg_demand=avg_demand,
            demand_std=demand_std,
            lead_time=lead_time,
            eoq=eoq,
            unit_cost=unit_cost,
            holding_rate=holding_rate
        )

        all_summaries.append(summary)
        all_records.extend(records)

    print("Completed:", row["StockCode"])

# ==========================================
# 5. EXPORT AND COMPARE RESULTS
# ==========================================

print("\n[4/5] Creating comparison reports...")

product_results = pd.DataFrame(all_summaries)

weekly_results = pd.DataFrame(all_records)

product_results.to_csv(
    OUTPUT_DIR / "product_simulation_results.csv",
    index=False
)

weekly_results.to_csv(
    OUTPUT_DIR / "weekly_simulation_details.csv",
    index=False
)

comparison = (
    product_results.groupby("Policy")
    .agg(
        Total_Demand=("Total_Demand", "sum"),
        Fulfilled_Demand=("Fulfilled_Demand", "sum"),
        Stockout_Units=("Stockout_Units", "sum"),
        Stockout_Weeks=("Stockout_Weeks", "sum"),
        Average_Inventory=("Average_Inventory", "mean"),
        Total_Orders=("Orders_Placed", "sum"),
        Ordering_Cost=("Ordering_Cost", "sum"),
        Holding_Cost=("Holding_Cost", "sum"),
        Stockout_Cost=("Stockout_Cost", "sum"),
        Total_Cost=("Total_Cost", "sum")
    )
    .reset_index()
)

comparison["Fill_Rate_Percent"] = (
    comparison["Fulfilled_Demand"] /
    comparison["Total_Demand"].replace(0, np.nan) * 100
)

comparison.to_csv(
    OUTPUT_DIR / "policy_comparison.csv",
    index=False
)

# ==========================================
# 6. CALCULATE IMPROVEMENT
# ==========================================

traditional = comparison[
    comparison["Policy"] == "Traditional"
].iloc[0]

optimized = comparison[
    comparison["Policy"] == "Optimized"
].iloc[0]

def percentage_reduction(old, new):
    if old == 0:
        return np.nan

    return (old - new) / old * 100


improvement = pd.DataFrame([{
    "Metric": "Stockout Units",
    "Traditional": traditional["Stockout_Units"],
    "Optimized": optimized["Stockout_Units"],
    "Reduction_Percent": percentage_reduction(
        traditional["Stockout_Units"],
        optimized["Stockout_Units"]
    )
}, {
    "Metric": "Total Simulated Cost",
    "Traditional": traditional["Total_Cost"],
    "Optimized": optimized["Total_Cost"],
    "Reduction_Percent": percentage_reduction(
        traditional["Total_Cost"],
        optimized["Total_Cost"]
    )
}, {
    "Metric": "Holding Cost",
    "Traditional": traditional["Holding_Cost"],
    "Optimized": optimized["Holding_Cost"],
    "Reduction_Percent": percentage_reduction(
        traditional["Holding_Cost"],
        optimized["Holding_Cost"]
    )
}])

improvement.to_csv(
    OUTPUT_DIR / "policy_improvement_report.csv",
    index=False
)

# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n[5/5] Simulation completed.")

print("\nPOLICY COMPARISON")
print(comparison.to_string(index=False))

print("\nIMPROVEMENT REPORT")
print(improvement.to_string(index=False))

print("\nOutput files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nIMPORTANT:")
print("All simulation demand and cost assumptions are synthetic.")
print("Results are scenario estimates, not actual company savings.")
