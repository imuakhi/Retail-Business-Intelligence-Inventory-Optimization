
import pandas as pd
import pytest

from src.data_engineering.data_quality import (
    run_data_quality_checks
)


@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "Invoice": ["10001", "10002", "10003", "10004"],
        "StockCode": ["A1", "A2", "A3", "A4"],
        "Description": ["Product A", "Product B", "Product C", "Product D"],
        "Quantity": [10, 5, -2, 8],
        "InvoiceDate": [
            "2011-01-01",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04"
        ],
        "Price": [10.5, 20.0, 15.0, 8.5],
        "Customer ID": [101, 102, 103, 104],
        "Country": ["UK", "UK", "France", "Germany"]
    })


def test_quality_engine_returns_report(sample_data):

    report, summary = run_data_quality_checks(sample_data)

    assert not report.empty
    assert "Check" in report.columns
    assert "Severity" in report.columns
    assert "Affected_Rows" in report.columns

    assert summary["total_rows"] == 4


def test_negative_quantity_is_detected(sample_data):

    report, summary = run_data_quality_checks(sample_data)

    negative_check = report[
        report["Check"] == "Negative Quantity / Returns"
    ]

    assert len(negative_check) == 1
    assert negative_check.iloc[0]["Affected_Rows"] == 1


def test_missing_required_column_raises_error(sample_data):

    sample_data = sample_data.drop(columns=["Price"])

    with pytest.raises(ValueError):
        run_data_quality_checks(sample_data)


def test_quality_score_is_valid(sample_data):

    report, summary = run_data_quality_checks(sample_data)

    assert 0 <= summary["quality_score_percent"] <= 100


def test_duplicate_detection(sample_data):

    sample_data = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True
    )

    report, summary = run_data_quality_checks(sample_data)

    duplicate_check = report[
        report["Check"] == "Duplicate Rows"
    ]

    assert duplicate_check.iloc[0]["Affected_Rows"] == 1
