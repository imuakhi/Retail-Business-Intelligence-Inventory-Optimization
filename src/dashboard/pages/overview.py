
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    load_json,
    format_number,
    month_labels,
    show_missing
)


def render():

    st.title("Executive Business Overview")

    st.caption(
        "Retail Business Intelligence & Inventory "
        "Optimization System"
    )

    monthly = load_csv(
        "reports/sql_analytics/monthly_revenue.csv"
    )

    customers = load_json(
        "reports/customer_analytics/customer_analytics_summary.json"
    )

    forecast = load_csv(
        "reports/optuna_forecasting/tuning_summary.csv"
    )

    inventory = load_json(
        "reports/monte_carlo_inventory/inventory_executive_summary.json"
    )

    # ------------------------------------------------------
    # KPI CALCULATIONS
    # ------------------------------------------------------

    total_revenue = 0
    total_invoices = 0
    total_customers = customers.get("total_customers", 0)

    if not monthly.empty:

        if "net_revenue" in monthly.columns:
            total_revenue = pd.to_numeric(
                monthly["net_revenue"],
                errors="coerce"
            ).fillna(0).sum()

        if "total_invoices" in monthly.columns:
            total_invoices = pd.to_numeric(
                monthly["total_invoices"],
                errors="coerce"
            ).fillna(0).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Net Revenue",
        format_number(total_revenue)
    )

    col2.metric(
        "Total Customer Records",
        format_number(total_customers, 0)
    )

    col3.metric(
        "Invoice Count (Monthly Sum)",
        format_number(total_invoices, 0)
    )

    col4.metric(
        "Products Simulated",
        format_number(
            inventory.get("products_simulated", 0),
            0
        )
    )

    st.divider()

    # ------------------------------------------------------
    # REVENUE TREND
    # ------------------------------------------------------

    st.subheader("Monthly Revenue Trend")

    if not monthly.empty and "net_revenue" in monthly.columns:

        chart_data = month_labels(monthly)

        if "Period" in chart_data.columns:

            chart_data["net_revenue"] = pd.to_numeric(
                chart_data["net_revenue"],
                errors="coerce"
            )

            st.line_chart(
                chart_data.set_index("Period")[
                    ["net_revenue"]
                ]
            )

        else:
            st.dataframe(monthly.head(20), width="stretch")

    else:
        show_missing("Monthly Revenue")

    st.divider()

    # ------------------------------------------------------
    # CUSTOMER AND INVENTORY INSIGHTS
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Intelligence")

        churn_rate = customers.get(
            "historical_90_day_no_purchase_rate_pct",
            None
        )

        if churn_rate is not None:
            st.metric(
                "Historical 90-Day No-Purchase Rate",
                f"{churn_rate:.2f}%"
            )

        st.write(
            "Selected K-Means Clusters:",
            customers.get(
                "selected_kmeans_clusters",
                "N/A"
            )
        )

        st.write(
            "Customer Segments:",
            customers.get(
                "rfm_segments",
                "N/A"
            )
        )

    with right:

        st.subheader("Inventory Intelligence")

        if inventory:

            saving = inventory.get(
                "estimated_cost_saving_percent"
            )

            stockout = inventory.get(
                "optimized_average_stockout_probability"
            )

            fill_rate = inventory.get(
                "optimized_average_fill_rate"
            )

            if saving is not None:
                st.metric(
                    "Estimated Scenario Cost Saving",
                    f"{saving:.2f}%"
                )

            if stockout is not None:
                st.metric(
                    "Optimized Stockout Probability",
                    f"{stockout * 100:.2f}%"
                )

            if fill_rate is not None:
                st.metric(
                    "Optimized Fill Rate",
                    f"{fill_rate * 100:.2f}%"
                )

        else:
            show_missing("Monte Carlo Inventory")

    st.divider()

    # ------------------------------------------------------
    # FORECAST MODEL SUMMARY
    # ------------------------------------------------------

    st.subheader("Demand Forecasting Model")

    if not forecast.empty:

        row = forecast.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Selected Model",
            str(row.get("Selected_Model", "N/A"))
        )

        c2.metric(
            "Final Test WAPE",
            f"{float(row.get('Final_Test_WAPE', 0)):.2f}%"
        )

        c3.metric(
            "Forecast Horizon",
            f"{int(row.get('Future_Forecast_Weeks', 0))} weeks"
        )

    else:

        # Use the earlier model summary if Optuna summary
        # has not been generated.
        old_summary = load_csv(
            "reports/advanced_ml_forecasting/forecast_summary.csv"
        )

        if not old_summary.empty:
            st.dataframe(
                old_summary,
                width="stretch"
            )
        else:
            show_missing("Forecast Summary")

    st.caption(
        "Dashboard values are based on available project reports. "
        "Simulated inventory financial metrics are scenario estimates."
    )
