
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    first_existing,
    show_missing
)


def render():

    st.title("Inventory Optimization")

    # ------------------------------------------------------
    # ABC-XYZ PRODUCT CLASSIFICATION
    # ------------------------------------------------------

    st.subheader("ABC-XYZ Product Classification")

    abc = load_csv(
        "reports/abc_xyz/abc_xyz_product_classification.csv"
    )

    if not abc.empty:

        segment_col = (
            "ABC_XYZ"
            if "ABC_XYZ" in abc.columns
            else "ABC_Category"
        )

        if segment_col in abc.columns:

            distribution = (
                abc[segment_col]
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Products")
            )

            st.bar_chart(
                distribution.set_index("Segment")[
                    ["Products"]
                ]
            )

        st.dataframe(
            abc.head(100),
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("ABC-XYZ Classification")

    st.divider()

    # ------------------------------------------------------
    # INVENTORY PARAMETERS
    # ------------------------------------------------------

    st.subheader("Reorder Point & Safety Stock")

    inventory_path = first_existing([
        "reports/monte_carlo_inventory/dynamic_reorder_parameters.csv",
        "reports/inventory/inventory_optimization_report.csv"
    ])

    if inventory_path:

        inventory = load_csv(
            inventory_path
        )

        st.dataframe(
            inventory.head(100),
            width="stretch",
            hide_index=True
        )

        numeric_columns = [
            "Traditional_ROP",
            "Optimized_ROP",
            "Optimized_Safety_Stock"
        ]

        available = [
            column for column in numeric_columns
            if column in inventory.columns
        ]

        if available and "StockCode" in inventory.columns:

            chart = inventory[
                ["StockCode"] + available
            ].copy()

            chart = chart.head(20).set_index(
                "StockCode"
            )

            st.subheader("Product-wise Reorder Parameters")

            st.bar_chart(chart)

    else:
        show_missing("Inventory Optimization Parameters")

    st.divider()

    # ------------------------------------------------------
    # REORDER RECOMMENDATIONS
    # ------------------------------------------------------

    st.subheader("Reorder Recommendations")

    recommendations = load_csv(
        "reports/inventory/reorder_recommendations.csv"
    )

    if not recommendations.empty:

        st.dataframe(
            recommendations.head(100),
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Reorder Recommendations",
            data=recommendations.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="reorder_recommendations.csv",
            mime="text/csv"
        )

    else:
        show_missing("Reorder Recommendations")
