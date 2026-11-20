
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def calculate_psi(reference, current, bins=10):

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    quantiles = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)[1:-1]
    )

    internal_edges = np.unique(quantiles)

    if len(internal_edges) == 0:
        center = float(np.median(reference))
        spread = max(abs(center) * 0.01, 0.5)

        internal_edges = np.array([
            center - spread,
            center + spread
        ])

    edges = np.concatenate([
        [-np.inf],
        internal_edges,
        [np.inf]
    ])

    reference_counts = np.histogram(
        reference,
        bins=edges
    )[0]

    current_counts = np.histogram(
        current,
        bins=edges
    )[0]

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct, epsilon, None
    )

    current_pct = np.clip(
        current_pct, epsilon, None
    )

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def calculate_forecast_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError(
            "No valid forecast observations."
        )

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum()
        / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "Observations": int(len(actual)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE_pct": float(wape)
    }


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "clean_sales.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "monitoring"
)

BASELINE_PATH = OUTPUT_DIR / "baseline_profile.json"
CURRENT_PATH = OUTPUT_DIR / "current_profile.json"
DRIFT_PATH = OUTPUT_DIR / "data_drift_report.csv"

FORECAST_CANDIDATES = [
    "reports/optuna_forecasting/tuned_test_predictions.csv",
    "reports/advanced_ml_forecasting/test_predictions.csv"
]

PREVIOUS_METRICS_PATH = (
    OUTPUT_DIR / "previous_forecast_metrics.json"
)

LATEST_METRICS_PATH = (
    OUTPUT_DIR / "latest_forecast_metrics.json"
)

FEATURES = ["Quantity", "Price"]
N_BINS = 10
RANDOM_STATE = 42


# ==========================================================
# 1. PURE PSI FUNCTION
# ==========================================================

def calculate_forecast_metrics(reference, current, bins=10):

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    quantiles = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)[1:-1]
    )

    internal_edges = np.unique(quantiles)

    if len(internal_edges) == 0:

        center = float(np.median(reference))
        spread = max(abs(center) * 0.01, 0.5)

        internal_edges = np.array([
            center - spread,
            center + spread
        ])

    edges = np.concatenate([
        [-np.inf],
        internal_edges,
        [np.inf]
    ])

    reference_counts = np.histogram(
        reference,
        bins=edges
    )[0]

    current_counts = np.histogram(
        current,
        bins=edges
    )[0]

    reference_pct = (
        reference_counts / len(reference)
    )

    current_pct = (
        current_counts / len(current)
    )

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ==========================================================
# 2. FORECAST PERFORMANCE METRICS
# ==========================================================

def calculate_forecast_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid forecast observations.")

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum()
        / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "Observations": int(len(actual)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE_pct": float(wape)
    }


# ==========================================================
# 3. PROFILE CREATION
# ==========================================================

def create_profile(df):

    profile = {
        "rows": int(len(df)),
        "features": {}
    }

    for feature in FEATURES:

        if feature not in df.columns:
            continue

        original = df[feature]

        values = pd.to_numeric(
            original,
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        quantiles = np.quantile(
            values,
            np.linspace(0, 1, N_BINS + 1)[1:-1]
        )

        internal_edges = np.unique(quantiles)

        if len(internal_edges) == 0:

            center = float(np.median(values))
            spread = max(abs(center) * 0.01, 0.5)

            internal_edges = np.array([
                center - spread,
                center + spread
            ])

        edges = np.concatenate([
            [-np.inf],
            internal_edges,
            [np.inf]
        ])

        counts = np.histogram(
            values,
            bins=edges
        )[0]

        proportions = (
            counts / len(values)
        )

        profile["features"][feature] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "missing_rate": float(
                original.isna().mean()
            ),
            "edges": [
                None if not np.isfinite(edge)
                else float(edge)
                for edge in edges
            ],
            "bin_proportions": [
                float(value)
                for value in proportions
            ]
        }

    return profile


# ==========================================================
# 4. PSI FROM SAVED BASELINE PROFILE
# ==========================================================

def psi_from_profile(baseline_feature, current_series):

    values = pd.to_numeric(
        current_series,
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    edges = np.array([
        -np.inf if value is None else value
        for value in baseline_feature["edges"]
    ], dtype=float)

    expected = np.asarray(
        baseline_feature["bin_proportions"],
        dtype=float
    )

    counts = np.histogram(
        values,
        bins=edges
    )[0]

    actual = counts / len(values)

    epsilon = 1e-6

    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)

    return float(
        np.sum(
            (actual - expected)
            * np.log(actual / expected)
        )
    )


# ==========================================================
# 5. DATA DRIFT MONITORING
# ==========================================================

def run_data_drift_monitoring(df):

    current_profile = create_profile(df)

    if not BASELINE_PATH.exists():

        with open(
            BASELINE_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                current_profile,
                file,
                indent=4
            )

        with open(
            CURRENT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                current_profile,
                file,
                indent=4
            )

        rows = []

        for feature in current_profile["features"]:

            rows.append({
                "Feature": feature,
                "PSI": None,
                "Missing_Rate_Change": None,
                "Drift_Status": "BASELINE_CREATED"
            })

        report = pd.DataFrame(rows)

        report.to_csv(
            DRIFT_PATH,
            index=False
        )

        print(
            "Baseline created. Run monitoring again "
            "to compare future data against it."
        )

        return report

    with open(
        BASELINE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        baseline = json.load(file)

    rows = []

    for feature in FEATURES:

        if (
            feature not in baseline["features"]
            or feature not in df.columns
        ):
            continue

        baseline_feature = baseline[
            "features"
        ][feature]

        psi = psi_from_profile(
            baseline_feature,
            df[feature]
        )

        current_missing_rate = float(
            df[feature].isna().mean()
        )

        missing_change = (
            current_missing_rate
            - baseline_feature["missing_rate"]
        )

        if not np.isfinite(psi):
            status = "INSUFFICIENT_DATA"
        elif psi < 0.10:
            status = "LOW_DRIFT"
        elif psi < 0.25:
            status = "MODERATE_DRIFT"
        else:
            status = "HIGH_DRIFT"

        rows.append({
            "Feature": feature,
            "PSI": psi,
            "Baseline_Missing_Rate": (
                baseline_feature["missing_rate"]
            ),
            "Current_Missing_Rate": current_missing_rate,
            "Missing_Rate_Change": missing_change,
            "Drift_Status": status
        })

    with open(
        CURRENT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            current_profile,
            file,
            indent=4
        )

    report = pd.DataFrame(rows)

    report.to_csv(
        DRIFT_PATH,
        index=False
    )

    print("\nDATA DRIFT REPORT")
    print(report.to_string(index=False))

    return report


# ==========================================================
# 6. FORECAST PERFORMANCE MONITORING
# ==========================================================

def run_forecast_monitoring():

    selected_path = next(
        (
            PROJECT_ROOT / path
            for path in FORECAST_CANDIDATES
            if (PROJECT_ROOT / path).exists()
        ),
        None
    )

    if selected_path is None:
        print("Forecast prediction file not found.")
        return pd.DataFrame()

    df = pd.read_csv(
        selected_path,
        low_memory=False
    )

    required = [
        "Actual_Demand",
        "Predicted_Demand"
    ]

    rows = []

    if set(required).issubset(df.columns):

        # Phase 16 tuned holdout predictions.
        model_groups = (
            df.groupby("Model")
            if "Model" in df.columns
            else [("Unknown", df)]
        )

        for model_name, group in model_groups:

            metrics = calculate_forecast_metrics(
                group["Actual_Demand"],
                group["Predicted_Demand"]
            )

            rows.append({
                "Model": str(model_name),
                **metrics
            })

    else:

        # Phase 15 output contains one prediction column
        # for each trained ML model.
        if "Actual_Demand" not in df.columns:

            print("Actual demand column missing.")
            return pd.DataFrame()

        for column in df.columns:

            if not column.endswith("_Prediction"):
                continue

            model_name = column.replace(
                "_Prediction", ""
            )

            metrics = calculate_forecast_metrics(
                df["Actual_Demand"],
                df[column]
            )

            rows.append({
                "Model": model_name,
                **metrics
            })

    report = pd.DataFrame(rows)

    if report.empty:
        print("No valid forecast metrics found.")
        return report

    report = report.sort_values(
        "WAPE_pct",
        na_position="last"
    )

    report.to_csv(
        OUTPUT_DIR / "forecast_performance.csv",
        index=False
    )

    # Compare with the previous monitoring run.
    previous = {}

    if PREVIOUS_METRICS_PATH.exists():

        with open(
            PREVIOUS_METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            previous = json.load(file)

    changes = []

    for _, row in report.iterrows():

        model = row["Model"]
        current_wape = row["WAPE_pct"]

        old_wape = previous.get(model)

        if old_wape is not None:

            increase = (
                current_wape - old_wape
            )

            relative_change = (
                increase / abs(old_wape) * 100
                if old_wape != 0
                else None
            )

        else:
            increase = None
            relative_change = None

        changes.append({
            "Model": model,
            "Previous_WAPE_pct": old_wape,
            "Current_WAPE_pct": current_wape,
            "Change_WAPE_points": increase,
            "Relative_Change_pct": relative_change,
            "Status": (
                "WORSE_THAN_PREVIOUS_RUN"
                if relative_change is not None
                and relative_change >= 10
                else "NO_COMPARABLE_PREVIOUS_RUN"
                if relative_change is None
                else "STABLE_OR_IMPROVED"
            )
        })

    pd.DataFrame(changes).to_csv(
        OUTPUT_DIR / "forecast_performance_comparison.csv",
        index=False
    )

    current_metrics = {
        str(row["Model"]): float(row["WAPE_pct"])
        for _, row in report.iterrows()
        if np.isfinite(row["WAPE_pct"])
    }

    with open(
        PREVIOUS_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            current_metrics,
            file,
            indent=4
        )

    with open(
        LATEST_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "source": str(
                    selected_path.relative_to(PROJECT_ROOT)
                ),
                "metrics": current_metrics
            },
            file,
            indent=4
        )

    print("\nFORECAST PERFORMANCE")
    print(report.to_string(index=False))

    return report


# ==========================================================
# 7. MAIN MONITORING PIPELINE
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("DATA & MODEL MONITORING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda column: column in FEATURES,
        low_memory=False
    )

    print(f"Monitoring rows: {len(df):,}")

    drift_report = run_data_drift_monitoring(df)

    performance_report = run_forecast_monitoring()

    summary = {
        "data_rows_monitored": int(len(df)),
        "drift_features_checked": int(len(drift_report)),
        "forecast_models_checked": int(
            len(performance_report)
        ),
        "baseline_exists": BASELINE_PATH.exists(),
        "current_profile_exists": CURRENT_PATH.exists(),
        "note": (
            "Drift thresholds are PSI heuristics. "
            "A baseline created from the same historical dataset "
            "does not prove real-world data stability. "
            "Forecast comparison requires comparable evaluation "
            "windows and datasets."
        )
    }

    with open(
        OUTPUT_DIR / "monitoring_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nMONITORING COMPLETED")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
