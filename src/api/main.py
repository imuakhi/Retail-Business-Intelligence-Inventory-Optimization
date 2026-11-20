
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    ForecastResponse
)

from src.api.services import (
    get_artifact_status,
    get_overview,
    get_forecasts,
    get_inventory_recommendation,
    get_customer_profile,
    get_churn_risk
)

# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Retail Business Intelligence API",
    description=(
        "REST API for Retail Sales Analytics, "
        "Demand Forecasting, Customer Intelligence "
        "and Inventory Optimization."
    ),
    version="1.0.0",
    contact={
        "name": "Retail BI Portfolio Project"
    }
)

# Local dashboard integration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"]
)


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "message": "Retail Business Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/api/v1/health"
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get(
    "/api/v1/health",
    response_model=HealthResponse
)
def health_check():

    return {
        "status": "healthy",
        "service": "Retail BI API",
        "artifacts": get_artifact_status()
    }


# ==========================================================
# BUSINESS OVERVIEW
# ==========================================================

@app.get("/api/v1/overview")
def business_overview():

    return get_overview()


# ==========================================================
# FORECAST ENDPOINT
# ==========================================================

@app.get(
    "/api/v1/forecast",
    response_model=ForecastResponse
)
def demand_forecast(
    stock_code: str | None = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=5000
    )
):

    result = get_forecasts(
        stock_code=stock_code,
        limit=limit
    )

    if result is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast report is unavailable or "
                "has an unexpected column structure. "
                "Run the forecasting pipeline first."
            )
        )

    return result


# ==========================================================
# INVENTORY ENDPOINT
# ==========================================================

@app.get("/api/v1/inventory/{stock_code}")
def inventory_recommendation(stock_code: str):

    result = get_inventory_recommendation(
        stock_code
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Inventory record not found. "
                "Check StockCode and inventory reports."
            )
        )

    return result


# ==========================================================
# CUSTOMER PROFILE ENDPOINT
# ==========================================================

@app.get("/api/v1/customers/{customer_id}")
def customer_profile(customer_id: str):

    result = get_customer_profile(
        customer_id
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Customer profile not found."
        )

    return result


# ==========================================================
# CUSTOMER CHURN RISK ENDPOINT
# ==========================================================

@app.get(
    "/api/v1/customers/{customer_id}/churn-risk"
)
def customer_churn_risk(customer_id: str):

    result = get_churn_risk(
        customer_id
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Churn score unavailable. "
                "Check the customer feature report and saved model."
            )
        )

    return result
