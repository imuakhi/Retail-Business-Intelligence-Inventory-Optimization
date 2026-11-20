
from pathlib import Path
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_PATHS = [
    "reports/optuna_forecasting/tuned_future_forecast.csv",
    "reports/advanced_ml_forecasting/future_demand_forecast_ml.csv",
    "reports/advanced_forecasting/future_demand_forecast.csv"
]

INVENTORY_PATHS = [
    "reports/monte_carlo_inventory/dynamic_reorder_parameters.csv",
    "reports/inventory/inventory_optimization_report.csv",
    "reports/inventory/reorder_recommendations.csv"
]


def normalize_id(value):
    if value is None or pd.isna(value):
        return ""

    result = str(value).strip()

    if result.endswith(".0"):
        result = result[:-2]

    return result


def json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def read_csv(relative_path, **kwargs):

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            low_memory=False,
            **kwargs
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(relative_path):

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def first_existing(paths):

    for relative_path in paths:

        if (PROJECT_ROOT / relative_path).exists():
            return relative_path

    return None


def get_artifact_status():

    required_files = {
        "database": "data/warehouse/retail_analytics.db",
        "forecast_model": "models/forecasting/best_tuned_demand_model.joblib",
        "tuned_forecast": "reports/optuna_forecasting/tuned_future_forecast.csv",
        "customer_rfm": "reports/customer_analytics/rfm_customer_segments.csv",
        "churn_model": "models/churn/churn_model.joblib",
        "inventory_report": "reports/inventory/inventory_optimization_report.csv",
        "monte_carlo_report": "reports/monte_carlo_inventory/inventory_policy_summary.csv"
    }

    return {
        name: (PROJECT_ROOT / relative_path).exists()
        for name, relative_path in required_files.items()
    }


# ==========================================================
# BUSINESS OVERVIEW
# ==========================================================

def get_overview():

    monthly = read_csv(
        "reports/sql_analytics/monthly_revenue.csv"
    )

    customers = read_json(
        "reports/customer_analytics/customer_analytics_summary.json"
    )

    inventory = read_json(
        "reports/monte_carlo_inventory/inventory_executive_summary.json"
    )

    forecast_path = first_existing([
        "reports/optuna_forecasting/tuning_summary.csv",
        "reports/advanced_ml_forecasting/forecast_summary.csv"
    ])

    forecast = (
        read_csv(forecast_path)
        if forecast_path
        else pd.DataFrame()
    )

    total_revenue = 0.0
    total_invoices = 0

    if not monthly.empty:

        if "net_revenue" in monthly.columns:
            total_revenue = pd.to_numeric(
                monthly["net_revenue"],
                errors="coerce"
            ).fillna(0).sum()

        if "total_invoices" in monthly.columns:
            total_invoices = int(
                pd.to_numeric(
                    monthly["total_invoices"],
                    errors="coerce"
                ).fillna(0).sum()
            )

    forecast_summary = {}

    if not forecast.empty:
        forecast_summary = json_safe(
            forecast.iloc[0].to_dict()
        )

    result = {
        "total_net_revenue": float(total_revenue),
        "monthly_invoice_count_sum": total_invoices,
        "total_customers": customers.get("total_customers"),
        "rfm_segments": customers.get("rfm_segments"),
        "selected_churn_model": customers.get(
            "selected_churn_model"
        ),
        "inventory_summary": inventory,
        "forecast_summary": forecast_summary,
        "available_artifacts": get_artifact_status()
    }

    return json_safe(result)


# ==========================================================
# DEMAND FORECASTING SERVICE
# ==========================================================

def get_forecasts(stock_code=None, limit=100):

    selected_path = first_existing(FORECAST_PATHS)

    if not selected_path:
        return None

    df = read_csv(selected_path)

    required = [
        "StockCode",
        "Week",
        "Predicted_Demand"
    ]

    if not set(required).issubset(df.columns):
        return None

    df["StockCode"] = df["StockCode"].map(normalize_id)

    df["Predicted_Demand"] = pd.to_numeric(
        df["Predicted_Demand"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Predicted_Demand", "Week"]
    )

    if stock_code:
        df = df[
            df["StockCode"] == normalize_id(stock_code)
        ]

    df = df.sort_values(
        ["StockCode", "Week"]
    ).head(limit)

    records = []

    for _, row in df.iterrows():

        records.append({
            "stock_code": str(row["StockCode"]),
            "week": str(row["Week"]),
            "predicted_demand": float(
                row["Predicted_Demand"]
            )
        })

    return {
        "count": len(records),
        "source": selected_path,
        "records": records
    }


# ==========================================================
# INVENTORY RECOMMENDATION SERVICE
# ==========================================================

def get_inventory_recommendation(stock_code):

    selected_path = first_existing(INVENTORY_PATHS)

    if not selected_path:
        return None

    df = read_csv(selected_path)

    if "StockCode" not in df.columns:
        return None

    df["StockCode"] = df["StockCode"].map(normalize_id)

    matching = df[
        df["StockCode"] == normalize_id(stock_code)
    ]

    if matching.empty:
        return None

    return {
        "stock_code": normalize_id(stock_code),
        "source": selected_path,
        "recommendation": json_safe(
            matching.iloc[0].to_dict()
        )
    }


# ==========================================================
# CUSTOMER PROFILE SERVICE
# ==========================================================

def get_customer_profile(customer_id):

    customer_id = normalize_id(customer_id)

    rfm = read_csv(
        "reports/customer_analytics/rfm_customer_segments.csv"
    )

    clv = read_csv(
        "reports/customer_analytics/customer_clv_scenarios.csv"
    )

    churn_features = read_csv(
        "reports/customer_analytics/customer_churn_features.csv"
    )

    result = {
        "customer_id": customer_id,
        "rfm": None,
        "clv": None,
        "historical_churn_features_available": False
    }

    if not rfm.empty and "Customer_ID" in rfm.columns:

        rfm["Customer_ID"] = rfm["Customer_ID"].map(
            normalize_id
        )

        match = rfm[
            rfm["Customer_ID"] == customer_id
        ]

        if not match.empty:
            result["rfm"] = json_safe(
                match.iloc[0].to_dict()
            )

    if not clv.empty and "Customer_ID" in clv.columns:

        clv["Customer_ID"] = clv["Customer_ID"].map(
            normalize_id
        )

        match = clv[
            clv["Customer_ID"] == customer_id
        ]

        if not match.empty:
            result["clv"] = json_safe(
                match.iloc[0].to_dict()
            )

    if (
        not churn_features.empty
        and "Customer_ID" in churn_features.columns
    ):

        result["historical_churn_features_available"] = (
            customer_id
            in set(
                churn_features["Customer_ID"].map(normalize_id)
            )
        )

    if result["rfm"] is None and result["clv"] is None:
        return None

    return result


# ==========================================================
# CHURN RISK INFERENCE
# ==========================================================

@lru_cache(maxsize=1)
def load_churn_bundle():

    model_path = (
        PROJECT_ROOT
        / "models"
        / "churn"
        / "churn_model.joblib"
    )

    if not model_path.exists():
        return None

    return joblib.load(model_path)


def get_churn_risk(customer_id):

    customer_id = normalize_id(customer_id)

    features_path = (
        "reports/customer_analytics/customer_churn_features.csv"
    )

    features_df = read_csv(features_path)

    if features_df.empty or "Customer_ID" not in features_df.columns:
        return None

    features_df["Customer_ID"] = (
        features_df["Customer_ID"].map(normalize_id)
    )

    customer_row = features_df[
        features_df["Customer_ID"] == customer_id
    ]

    if customer_row.empty:
        return None

    bundle = load_churn_bundle()

    if bundle is None:
        return None

    model = bundle["model"]
    feature_columns = bundle["features"]
    metadata = bundle.get("metadata", {})

    missing = [
        column
        for column in feature_columns
        if column not in customer_row.columns
    ]

    if missing:
        return None

    X = customer_row[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        return None

    churn_index = classes.index(1)

    probability = float(
        probabilities[churn_index]
    )

    # These are illustrative display bands, not
    # calibrated operational intervention thresholds.
    if probability >= 0.66:
        risk_level = "High"
    elif probability >= 0.33:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "customer_id": customer_id,
        "predicted_no_purchase_probability": probability,
        "risk_level": risk_level,
        "target_definition": metadata.get(
            "target_definition",
            "Historical 90-day no-purchase proxy"
        ),
        "observation_cutoff": metadata.get(
            "observation_cutoff"
        ),
        "warning": (
            "Historical inactivity-model score; not a guarantee "
            "of future churn. Probability calibration has not "
            "been established."
        )
    }
