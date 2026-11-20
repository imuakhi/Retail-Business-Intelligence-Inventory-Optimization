
import streamlit as st

from src.dashboard.pages import (
    overview,
    sales,
    customers,
    forecasting,
    inventory,
    simulation
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Retail BI & Inventory Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM STYLING
# ==========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px;
        border-radius: 10px;
    }

    h1, h2, h3 {
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("Retail BI System")

st.sidebar.caption(
    "Business Intelligence & Inventory Optimization"
)

page_options = {
    "Executive Overview": overview.render,
    "Sales Analytics": sales.render,
    "Customer Intelligence": customers.render,
    "Demand Forecasting": forecasting.render,
    "Inventory Optimization": inventory.render,
    "Monte Carlo Simulation": simulation.render
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(page_options.keys())
)

st.sidebar.divider()

st.sidebar.caption(
    "Advanced Data Analytics | Machine Learning | BI"
)

if st.sidebar.button("Refresh Dashboard Data"):

    st.cache_data.clear()

    st.rerun()

# ==========================================================
# PAGE RENDERING
# ==========================================================

page_options[selected_page]()

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Retail Business Intelligence & Inventory Optimization System "
    "| Portfolio Project | Built with Python, SQL and Streamlit"
)
