
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    load_json,
    format_number,
    show_missing
)


def render():

    st.title("Monte Carlo Inventory Simulation")

    summary = load_csv(
        "reports/monte_carlo_inventory/inventory_policy_summary.csv"
    )

    executive = load_json(
        "reports/monte_carlo_inventory/inventory_executive_summary.json"
    )

    comparison = load_csv(
        "reports/monte_carlo_inventory/inventory_policy_comparison.csv"
    )

    risk = load_csv(
        "reports/monte_carlo_inventory/inventory_risk_report.csv"
    )

    # ------------------------------------------------------
    # EXECUTIVE SIMULATION KPIs
    # ------------------------------------------------------

    st.subheader("Simulation Performance")

    if executive:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Products Simulated",
            f"{executive.get('products_simulated', 0):,}"
        )

        saving = executive.get(
            "estimated_cost_saving_percent"
        )

        c2.metric(
            "Scenario Cost Saving",
            (
                f"{saving:.2f}%"
                if saving is not None
                else "N/A"
            )
        )

        c3.metric(
            "Optimized Fill Rate",
            f"{executive.get('optimized_average_fill_rate', 0) * 100:.2f}%"
        )

        c4.metric(
            "Optimized Stockout Risk",
            f"{executive.get('optimized_average_stockout_probability', 0) * 100:.2f}%"
        )

    else:
        show_missing("Monte Carlo Executive Summary")

    st.divider()

    # ------------------------------------------------------
    # POLICY COMPARISON
    # ------------------------------------------------------

    st.subheader("Traditional vs Optimized Policy")

    if not summary.empty:

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True
        )

        if "Average_Total_Cost" in summary.columns:

            st.bar_chart(
                summary.set_index("Policy")[
                    ["Average_Total_Cost"]
                ]
            )

        if "Average_Stockout_Probability" in summary.columns:

            st.bar_chart(
                summary.set_index("Policy")[
                    ["Average_Stockout_Probability"]
                ]
            )

        if "Average_Fill_Rate" in summary.columns:

            st.bar_chart(
                summary.set_index("Policy")[
                    ["Average_Fill_Rate"]
                ]
            )

    else:
        show_missing("Inventory Policy Summary")

    st.divider()

    # ------------------------------------------------------
    # PRODUCT LEVEL SIMULATION
    # ------------------------------------------------------

    st.subheader("Product-level Policy Comparison")

    if not comparison.empty:

        sort_column = (
            "Cost_Saving_Percent"
            if "Cost_Saving_Percent" in comparison.columns
            else comparison.columns[0]
        )

        display = comparison.sort_values(
            sort_column,
            ascending=False
        )

        st.dataframe(
            display.head(100),
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Inventory Policy Comparison")

    st.divider()

    # ------------------------------------------------------
    # STOCKOUT RISK
    # ------------------------------------------------------

    st.subheader("Inventory Risk Monitoring")

    if not risk.empty:

        if "Inventory_Risk" in risk.columns:

            risk_counts = (
                risk["Inventory_Risk"]
                .value_counts()
                .rename_axis("Risk Level")
                .reset_index(name="Products")
            )

            st.bar_chart(
                risk_counts.set_index("Risk Level")[
                    ["Products"]
                ]
            )

        st.dataframe(
            risk.head(100),
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Inventory Risk Report")

    # ------------------------------------------------------
    # ASSUMPTIONS AND LIMITATIONS
    # ------------------------------------------------------

    assumptions = load_json(
        "reports/monte_carlo_inventory/simulation_assumptions.json"
    )

    if assumptions:

        with st.expander("Simulation Assumptions & Limitations"):

            st.json(assumptions)
