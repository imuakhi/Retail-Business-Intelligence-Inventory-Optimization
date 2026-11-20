
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    first_existing,
    show_missing
)


def render():

    st.title("Advanced Demand Forecasting")

    # ------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------

    comparison_path = first_existing([
        "reports/optuna_forecasting/tuned_model_comparison.csv",
        "reports/advanced_ml_forecasting/model_comparison.csv",
        "reports/advanced_forecasting/model_comparison.csv"
    ])

    if comparison_path:

        comparison = load_csv(
            comparison_path
        )

        st.subheader("Forecast Model Comparison")

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True
        )

        error_col = (
            "WAPE_pct"
            if "WAPE_pct" in comparison.columns
            else None
        )

        if error_col:

            chart = comparison.copy()

            chart[error_col] = pd.to_numeric(
                chart[error_col],
                errors="coerce"
            )

            st.bar_chart(
                chart.set_index("Model")[
                    [error_col]
                ]
            )

    else:
        show_missing("Forecast Model Comparison")

    st.divider()

    # ------------------------------------------------------
    # FUTURE DEMAND
    # ------------------------------------------------------

    st.subheader("Future Weekly Demand Forecast")

    forecast_path = first_existing([
        "reports/optuna_forecasting/tuned_future_forecast.csv",
        "reports/advanced_ml_forecasting/future_demand_forecast_ml.csv",
        "reports/advanced_forecasting/future_demand_forecast.csv"
    ])

    if forecast_path:

        forecast = load_csv(
            forecast_path
        )

        required = [
            "StockCode",
            "Week",
            "Predicted_Demand"
        ]

        if set(required).issubset(forecast.columns):

            forecast["Predicted_Demand"] = pd.to_numeric(
                forecast["Predicted_Demand"],
                errors="coerce"
            )

            forecast["Week"] = pd.to_datetime(
                forecast["Week"],
                errors="coerce"
            )

            forecast = forecast.dropna(
                subset=["Week", "Predicted_Demand"]
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Products Forecasted",
                f"{forecast['StockCode'].nunique():,}"
            )

            col2.metric(
                "Forecast Records",
                f"{len(forecast):,}"
            )

            weekly = (
                forecast.groupby("Week")[
                    "Predicted_Demand"
                ]
                .sum()
                .reset_index()
            )

            st.line_chart(
                weekly.set_index("Week")[
                    ["Predicted_Demand"]
                ]
            )

            st.subheader("Product-wise Forecast")

            selected_products = sorted(
                forecast["StockCode"].dropna().unique()
            )

            selected_product = st.selectbox(
                "Select Product",
                selected_products
            )

            product_forecast = forecast[
                forecast["StockCode"] == selected_product
            ].sort_values("Week")

            st.line_chart(
                product_forecast.set_index("Week")[
                    ["Predicted_Demand"]
                ]
            )

            st.dataframe(
                product_forecast,
                width="stretch",
                hide_index=True
            )

        else:
            st.error(
                "Forecast file has an unexpected column structure."
            )

    else:
        show_missing("Future Demand Forecast")

    st.divider()

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    st.subheader("Forecast Feature Importance")

    importance_path = first_existing([
        "reports/optuna_forecasting/tuned_feature_importance.csv",
        "reports/advanced_ml_forecasting/feature_importance.csv"
    ])

    if importance_path:

        importance = load_csv(
            importance_path
        )

        if "Feature" in importance.columns:

            importance = importance.head(15)

            value_col = (
                "Importance"
                if "Importance" in importance.columns
                else "Mean_Absolute_SHAP"
            )

            if value_col in importance.columns:

                st.bar_chart(
                    importance.set_index("Feature")[
                        [value_col]
                    ]
                )

            st.dataframe(
                importance,
                width="stretch",
                hide_index=True
            )

    else:
        show_missing("Feature Importance")
