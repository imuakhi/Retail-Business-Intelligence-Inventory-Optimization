
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "documentation" in data


def test_health_endpoint():

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "artifacts" in data


def test_overview_endpoint():

    response = client.get("/api/v1/overview")

    assert response.status_code == 200

    assert "available_artifacts" in response.json()


def test_forecast_endpoint_validation():

    response = client.get(
        "/api/v1/forecast?limit=0"
    )

    assert response.status_code == 422


def test_forecast_endpoint_limit_validation():

    response = client.get(
        "/api/v1/forecast?limit=6000"
    )

    assert response.status_code == 422


def test_unknown_customer():

    response = client.get(
        "/api/v1/customers/UNKNOWN_CUSTOMER_999999"
    )

    assert response.status_code == 404
