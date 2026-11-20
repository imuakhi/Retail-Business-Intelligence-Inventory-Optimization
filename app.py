
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Retail BI & Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricLabel"] p {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700;
}

h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DATA LOADER
# =====================================================

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


sales = load_csv(
    BASE_DIR / "data/processed/clean_sales.csv"
)

forecast = load_csv(
    REPORTS_DIR / "advanced_forecasting/future_demand_forecast.csv"
)

abc = load_csv(
    REPORTS_DIR / "abc_xyz/abc_xyz_product_classification.csv"
)

inventory = load_csv(
    REPORTS_DIR / "inventory/inventory_optimization_report.csv"
)

simulation = load_csv(
    REPORTS_DIR / "simulation/policy_comparison.csv"
)

# =====================================================
# DATA PREPARATION
# =====================================================

if not sales.empty:
    sales["InvoiceDate"] = pd.to_datetime(
        sales["InvoiceDate"], errors="coerce"
    )

    sales["Quantity"] = pd.to_numeric(
        sales["Quantity"], errors="coerce"
    )

    sales["Price"] = pd.to_numeric(
        sales["Price"], errors="coerce"
    )

    sales["Revenue"] = sales["Quantity"] * sales["Price"]

    sales = sales.dropna(
        subset=["InvoiceDate", "Revenue"]
    )

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏬 Retail BI System")
st.sidebar.caption("Advanced Analytics & Inventory Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Sales Analytics",
        "Product Intelligence",
        "Demand Forecasting",
        "Inventory Optimization",
        "Simulation Analysis"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Retail BI & Inventory Optimization System")
st.sidebar.caption("Data Science Portfolio Project")


# =====================================================
# HEADER
# =====================================================

st.title("Retail Business Intelligence & Inventory Optimization")
st.markdown(
    "An integrated analytics platform for sales intelligence, "
    "demand forecasting and inventory decision support."
)

st.divider()


# =====================================================
# EXECUTIVE OVERVIEW
# =====================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    if sales.empty:
        st.error("Sales dataset not found or empty.")
        st.stop()

    total_revenue = sales["Revenue"].sum()
    total_orders = sales["Invoice"].nunique()
    total_products = sales["StockCode"].nunique()
    total_customers = sales["Customer ID"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"{total_revenue:,.2f}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Products",
        f"{total_products:,}"
    )

    col4.metric(
        "Customers",
        f"{total_customers:,}"
    )

    st.subheader("Revenue Trend")

    monthly = (
        sales.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        country_sales = (
            sales.groupby("Country")["Revenue"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig = px.bar(
            country_sales,
            x="Revenue",
            y="Country",
            orientation="h",
            title="Top 10 Countries by Revenue"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_products = (
            sales.groupby("StockCode")["Revenue"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="StockCode",
            y="Revenue",
            title="Top 10 Products by Revenue"
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SALES ANALYTICS
# =====================================================

elif page == "Sales Analytics":

    st.header("Sales Analytics")

    if sales.empty:
        st.warning("Sales dataset unavailable.")
        st.stop()

    min_date = sales["InvoiceDate"].min().date()
    max_date = sales["InvoiceDate"].max().date()

    date_range = st.date_input(
        "Select analysis period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range

        filtered = sales[
            sales["InvoiceDate"].dt.date.between(
                start_date, end_date
            )
        ]
    else:
        filtered = sales.copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("Revenue", f"{filtered['Revenue'].sum():,.2f}")
    c2.metric("Orders", f"{filtered['Invoice'].nunique():,}")
    c3.metric("Units Sold", f"{filtered['Quantity'].sum():,.0f}")

    monthly = (
        filtered.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        title="Sales Performance Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    country = (
        filtered.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        country,
        x="Country",
        y="Revenue",
        title="Country-wise Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales Transaction Data")
    st.dataframe(filtered.head(1000), use_container_width=True)


# =====================================================
# PRODUCT INTELLIGENCE
# =====================================================

elif page == "Product Intelligence":

    st.header("ABC-XYZ Product Intelligence")

    if abc.empty:
        st.warning("ABC-XYZ report not found.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        abc_counts = (
            abc["ABC_Category"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Products")
        )

        fig = px.pie(
            abc_counts,
            names="Category",
            values="Products",
            title="ABC Product Classification"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        xyz_counts = (
            abc["XYZ_Category"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Products")
        )

        fig = px.pie(
            xyz_counts,
            names="Category",
            values="Products",
            title="XYZ Demand Classification"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ABC-XYZ Product Matrix")

    if {"ABC_Category", "XYZ_Category"}.issubset(abc.columns):
        matrix = pd.crosstab(
            abc["ABC_Category"],
            abc["XYZ_Category"]
        )

        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            title="ABC-XYZ Classification Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Classification Report")
    st.dataframe(abc, use_container_width=True)


# =====================================================
# DEMAND FORECASTING
# =====================================================

elif page == "Demand Forecasting":

    st.header("AI Demand Forecasting")

    if forecast.empty:
        st.warning("Forecast report not found.")
        st.stop()

    forecast["Week"] = pd.to_datetime(
        forecast["Week"], errors="coerce"
    )

    st.metric(
        "Forecast Records",
        f"{len(forecast):,}"
    )

    product_list = sorted(
        forecast["StockCode"].dropna().astype(str).unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        product_list
    )

    product_forecast = forecast[
        forecast["StockCode"].astype(str) == selected_product
    ].sort_values("Week")

    fig = px.line(
        product_forecast,
        x="Week",
        y="Predicted_Demand",
        markers=True,
        title=f"Future Demand Forecast: {selected_product}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Data")
    st.dataframe(product_forecast, use_container_width=True)


# =====================================================
# INVENTORY OPTIMIZATION
# =====================================================

elif page == "Inventory Optimization":

    st.header("Inventory Optimization")

    if inventory.empty:
        st.warning("Inventory optimization report not found.")
        st.stop()

    st.metric(
        "Products Analyzed",
        f"{len(inventory):,}"
    )

    st.subheader("Inventory Optimization Report")

    st.dataframe(
        inventory,
        use_container_width=True
    )

    numeric_cols = inventory.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        selected_metric = st.selectbox(
            "Select inventory metric",
            numeric_cols
        )

        fig = px.histogram(
            inventory,
            x=selected_metric,
            title=f"Distribution of {selected_metric}"
        )

        st.plotly_chart(fig, use_container_width=True)

    if "Suggested_Order_Quantity" in inventory.columns:
        recommendations = inventory[
            inventory["Suggested_Order_Quantity"] > 0
        ]

        st.subheader("Reorder Recommendations")

        st.dataframe(
            recommendations,
            use_container_width=True
        )


# =====================================================
# SIMULATION ANALYSIS
# =====================================================

elif page == "Simulation Analysis":

    st.header("Inventory Policy Simulation")

    if simulation.empty:
        st.warning("Simulation policy comparison report not found.")
        st.stop()

    st.subheader("Traditional vs Optimized Policy")

    st.dataframe(
        simulation,
        use_container_width=True
    )

    numeric_cols = simulation.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        metric = st.selectbox(
            "Select comparison metric",
            numeric_cols
        )

        fig = px.bar(
            simulation,
            x=simulation.columns[0],
            y=metric,
            title=f"Policy Comparison: {metric}",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Retail Business Intelligence & Inventory Optimization System | "
    "Advanced Data Analytics & Machine Learning Portfolio"
)
