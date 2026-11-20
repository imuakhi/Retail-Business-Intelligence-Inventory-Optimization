
import streamlit as st
import pandas as pd

from src.dashboard.data_loader import (
    load_csv,
    format_number,
    show_missing
)


def render():

    st.title("Customer Intelligence")

    # ------------------------------------------------------
    # RFM SEGMENTATION
    # ------------------------------------------------------

    st.subheader("RFM Customer Segmentation")

    rfm = load_csv(
        "reports/customer_analytics/rfm_segment_summary.csv"
    )

    if not rfm.empty:

        c1, c2 = st.columns(2)

        if "Customers" in rfm.columns:

            with c1:
                st.bar_chart(
                    rfm.set_index("RFM_Segment")[
                        ["Customers"]
                    ]
                )

        if "Total_Monetary" in rfm.columns:

            with c2:
                st.bar_chart(
                    rfm.set_index("RFM_Segment")[
                        ["Total_Monetary"]
                    ]
                )

        st.dataframe(
            rfm,
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("RFM Segmentation")

    st.divider()

    # ------------------------------------------------------
    # K-MEANS CUSTOMER CLUSTERS
    # ------------------------------------------------------

    st.subheader("K-Means Customer Clusters")

    clusters = load_csv(
        "reports/customer_analytics/kmeans_cluster_profiles.csv"
    )

    if not clusters.empty:

        if "KMeans_Cluster" in clusters.columns:

            st.bar_chart(
                clusters.set_index("KMeans_Cluster")[
                    ["Customer_Count"]
                ]
            )

        st.dataframe(
            clusters,
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("K-Means Clustering")

    st.divider()

    # ------------------------------------------------------
    # CUSTOMER LIFETIME VALUE
    # ------------------------------------------------------

    st.subheader("Customer Lifetime Value Scenarios")

    clv = load_csv(
        "reports/customer_analytics/customer_clv_scenarios.csv"
    )

    if not clv.empty:

        columns = [
            "Profit_CLV_Conservative",
            "Profit_CLV_Base",
            "Profit_CLV_Optimistic"
        ]

        available = [
            column for column in columns
            if column in clv.columns
        ]

        if available:

            averages = (
                clv[available]
                .apply(pd.to_numeric, errors="coerce")
                .mean()
            )

            c1, c2, c3 = st.columns(3)

            for col, name in zip(
                [c1, c2, c3],
                available
            ):

                col.metric(
                    name.replace("Profit_CLV_", ""),
                    format_number(averages[name])
                )

        st.dataframe(
            clv.head(100),
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Customer CLV")

    st.divider()

    # ------------------------------------------------------
    # CHURN MODEL PERFORMANCE
    # ------------------------------------------------------

    st.subheader("Churn Prediction Model Comparison")

    churn = load_csv(
        "reports/customer_analytics/churn_model_comparison.csv"
    )

    if not churn.empty:

        if "PR_AUC" in churn.columns:

            st.bar_chart(
                churn.set_index("Model")[["PR_AUC"]]
            )

        st.dataframe(
            churn,
            width="stretch",
            hide_index=True
        )

    else:
        show_missing("Churn Prediction")
