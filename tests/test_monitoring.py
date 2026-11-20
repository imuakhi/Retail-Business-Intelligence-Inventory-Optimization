
import numpy as np

from src.monitoring.monitoring import (
    calculate_psi,
    calculate_forecast_metrics
)


def test_psi_is_low_for_identical_data():

    data = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ])

    psi = calculate_psi(data, data)

    assert psi < 0.001


def test_psi_detects_distribution_shift():

    reference = np.random.default_rng(
        42
    ).normal(10, 2, 1000)

    current = np.random.default_rng(
        43
    ).normal(30, 2, 1000)

    psi = calculate_psi(
        reference,
        current
    )

    assert psi > 0.10


def test_forecast_metrics():

    actual = np.array([10, 20, 30, 40])

    predicted = np.array([12, 18, 30, 44])

    metrics = calculate_forecast_metrics(
        actual,
        predicted
    )

    assert metrics["MAE"] == 2.0
    assert np.isclose(metrics["RMSE"], np.sqrt(6))
    assert metrics["Observations"] == 4
   
    
    



def test_forecast_metrics_ignores_invalid_values():

    actual = np.array([10, 20, np.nan, 40])

    predicted = np.array([10, 25, 30, np.inf])

    metrics = calculate_forecast_metrics(
        actual,
        predicted
    )

    assert metrics["Observations"] == 2
    assert metrics["MAE"] == 2.5


def test_psi_rejects_empty_input():

    psi = calculate_psi([], [1, 2, 3])

    assert np.isnan(psi)
