
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    month_labels,
    format_number,
    show_missing
)


def render():

    st.title("Sales & Revenue Analytics")

    monthly = load_csv(
        "reports/sql_analytics/monthly_revenue.csv"
    )

    products = load_csv(
        "reports/sql_analytics/top_products.csv"
    )

    countries = load_csv(
        "reports/sql_analytics/country_performance.csv"
    )

    # ------------------------------------------------------
    # MONTHLY REVENUE
    # ------------------------------------------------------

    st.subheader("Monthly Revenue Performance")

    if not monthly.empty:

        chart_data = month_labels(monthly)

        metric_cols = st.columns(3)

        for col, column, label in zip(
            metric_cols,
            ["gross_sales", "returned_value", "net_revenue"],
            ["Gross Sales", "Returned Value", "Net Revenue"]
        ):

            if column in monthly.columns:

                value = pd.to_numeric(
                    monthly[column],
                    errors="coerce"
                ).fillna(0).sum()

                col.metric(
                    label,
                    format_number(value)
                )

        if (
            "Period" in chart_data.columns
            and "net_revenue" in chart_data.columns
        ):

            st.line_chart(
                chart_data.set_index("Period")[
                    ["net_revenue"]
                ]
            )

        if (
            "Period" in chart_data.columns
            and "gross_sales" in chart_data.columns
        ):

            st.bar_chart(
                chart_data.set_index("Period")[
                    ["gross_sales"]
                ]
            )

        st.dataframe(
            monthly,
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Monthly Revenue SQL")

    st.divider()

    # ------------------------------------------------------
    # TOP PRODUCTS
    # ------------------------------------------------------

    st.subheader("Top 20 Products")

    if not products.empty:

        revenue_col = (
            "gross_sales"
            if "gross_sales" in products.columns
            else "net_revenue"
        )

        if revenue_col in products.columns:

            chart = products.head(20).copy()

            label_col = (
                "description"
                if "description" in chart.columns
                else "stock_code"
            )

            chart[revenue_col] = pd.to_numeric(
                chart[revenue_col],
                errors="coerce"
            )

            chart = chart.dropna(
                subset=[revenue_col]
            )

            chart = chart.sort_values(
                revenue_col,
                ascending=False
            )

            chart = chart.set_index(label_col)

            st.bar_chart(
                chart[[revenue_col]]
            )

        st.dataframe(
            products.head(20),
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Top Products")

    st.divider()

    # ------------------------------------------------------
    # COUNTRY ANALYSIS
    # ------------------------------------------------------

    st.subheader("Country-wise Revenue Performance")

    if not countries.empty:

        if "net_revenue" in countries.columns:

            countries["net_revenue"] = pd.to_numeric(
                countries["net_revenue"],
                errors="coerce"
            )

            top_countries = (
                countries.sort_values(
                    "net_revenue",
                    ascending=False
                )
                .head(15)
            )

            country_column = (
                "country_name"
                if "country_name" in top_countries.columns
                else top_countries.columns[0]
            )

            st.bar_chart(
                top_countries.set_index(country_column)[
                    ["net_revenue"]
                ]
            )

        st.dataframe(
            countries,
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Country Performance")
