
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.forecasting.global_demand_forecasting import (
    load_weekly_demand,
    create_features,
    FEATURE_COLUMNS,
    TEST_WEEKS
)

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "best_tuned_demand_model.joblib"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "optuna_forecasting"
    / "tuned_test_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "optuna_forecasting"
    / "tuning_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainable_ai"
)

SHAP_SAMPLE_SIZE = 2000
MAX_PLOT_ROWS = 5000
RANDOM_STATE = 42


# ==========================================================
# 1. LOAD TRAINED MODEL
# ==========================================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run Phase 16 Optuna pipeline first."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "model_name",
        "features"
    ]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(
                f"Model bundle is missing: {key}"
            )

    model = bundle["model"]
    model_name = bundle["model_name"]

    print(f"Loaded model: {model_name}")

    return model, model_name


# ==========================================================
# 2. PREPARE FEATURE DATA
# ==========================================================

def prepare_explanation_data():

    print("\nLoading historical demand data...")

    panel, product_map, all_weeks = load_weekly_demand()

    featured = create_features(panel)

    featured = featured.dropna(
        subset=["lag_1"]
    ).reset_index(drop=True)

    if len(all_weeks) <= TEST_WEEKS:
        raise ValueError(
            "Insufficient weekly history."
        )

    test_start = pd.Timestamp(
        all_weeks[-TEST_WEEKS]
    )

    explanation_data = featured[
        featured["Week"] >= test_start
    ].copy()

    if explanation_data.empty:
        raise ValueError(
            "No explanation data available."
        )

    print(
        f"Explanation rows: {len(explanation_data):,}"
    )

    return explanation_data


# ==========================================================
# 3. GENERATE SHAP VALUES
# ==========================================================

def calculate_shap_values(model, data):

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(data)
    )

    sample = data.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).copy()

    X = sample[FEATURE_COLUMNS].copy()

    print("\nCalculating SHAP values...")
    print(f"Sample size: {len(X):,}")

    # TreeExplainer supports XGBoost and LightGBM.
    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(
            X,
            check_additivity=False
        )
    except TypeError:
        shap_values = explainer.shap_values(X)

    # Compatibility for models returning a list.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.shape != X.shape:
        raise ValueError(
            "Unexpected SHAP output dimensions: "
            f"{shap_values.shape}"
        )

    print("SHAP calculation completed.")

    return sample, X, explainer, shap_values


# ==========================================================
# 4. GLOBAL SHAP FEATURE IMPORTANCE
# ==========================================================

def save_global_explanations(
    X,
    shap_values
):

    print("\nCreating global feature importance...")

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Mean_Absolute_SHAP": np.abs(
            shap_values
        ).mean(axis=0)
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    # Beeswarm summary plot
    shap.summary_plot(
        shap_values,
        X,
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Summary - Demand Forecasting"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary_beeswarm.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Global feature importance bar plot
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=15,
        show=False
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_global_importance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Dependence plot for the most influential feature
    top_feature = importance.iloc[0]["Feature"]

    shap.dependence_plot(
        top_feature,
        shap_values,
        X,
        interaction_index=None,
        show=False
    )

    plt.title(
        f"SHAP Dependence: {top_feature}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_top_feature_dependence.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Global SHAP reports saved.")

    return importance


# ==========================================================
# 5. LOCAL PREDICTION EXPLANATION
# ==========================================================

def save_local_explanation(
    model,
    explainer,
    X,
    shap_values,
    sample
):

    print("\nCreating local prediction explanation...")

    # Explain the first sampled product-week.
    local_X = X.iloc[[0]]

    local_values = shap_values[0]

    predicted_log = float(
        model.predict(local_X)[0]
    )

    predicted_demand = max(
        float(np.expm1(predicted_log)),
        0
    )

    base_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)[0]

    base_value = float(base_value)

    local_report = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Feature_Value": local_X.iloc[0].values,
        "SHAP_Contribution_Log_Demand": local_values
    })

    local_report["Absolute_Contribution"] = (
        local_report[
            "SHAP_Contribution_Log_Demand"
        ].abs()
    )

    local_report = local_report.sort_values(
        "Absolute_Contribution",
        ascending=False
    )

    local_report.to_csv(
        OUTPUT_DIR / "local_prediction_explanation.csv",
        index=False
    )

    local_metadata = {
        "StockCode": str(sample.iloc[0]["StockCode"]),
        "Week": str(sample.iloc[0]["Week"].date()),
        "Model_Predicted_Demand": predicted_demand,
        "Predicted_Log_Demand": predicted_log,
        "SHAP_Base_Value_Log_Scale": base_value,
        "Sum_SHAP_Contributions": float(
            np.sum(local_values)
        ),
        "Reconstructed_Log_Prediction": float(
            base_value + np.sum(local_values)
        ),
        "Explanation_Scale": (
            "SHAP contributions explain log1p(Demand), "
            "not raw demand units."
        )
    }

    with open(
        OUTPUT_DIR / "local_prediction_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            local_metadata,
            file,
            indent=4
        )

    print("Local explanation saved.")
    print(
        f"Example product: "
        f"{local_metadata['StockCode']}"
    )
    print(
        f"Predicted demand: "
        f"{predicted_demand:.2f}"
    )


# ==========================================================
# 6. LOAD HOLDOUT PREDICTIONS
# ==========================================================

def load_holdout_predictions(model_name):

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    required = [
        "StockCode",
        "Week",
        "Actual_Demand",
        "Predicted_Demand",
        "Model"
    ]

    missing = set(required) - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    predictions = predictions[
        predictions["Model"] == model_name
    ].copy()

    if predictions.empty:
        raise ValueError(
            f"No holdout predictions found for {model_name}"
        )

    predictions["Week"] = pd.to_datetime(
        predictions["Week"]
    )

    predictions["Actual_Demand"] = pd.to_numeric(
        predictions["Actual_Demand"],
        errors="coerce"
    )

    predictions["Predicted_Demand"] = pd.to_numeric(
        predictions["Predicted_Demand"],
        errors="coerce"
    )

    predictions = predictions.dropna(
        subset=[
            "Actual_Demand",
            "Predicted_Demand",
            "Week"
        ]
    )

    predictions["Residual"] = (
        predictions["Actual_Demand"]
        - predictions["Predicted_Demand"]
    )

    predictions["Absolute_Error"] = (
        predictions["Residual"].abs()
    )

    return predictions


# ==========================================================
# 7. FORECAST ERROR ANALYSIS
# ==========================================================

def save_error_analysis(predictions):

    print("\nGenerating forecast error analysis...")

    # Product-wise error summary
    product_error = (
        predictions.groupby("StockCode")
        .agg(
            Actual_Demand=("Actual_Demand", "sum"),
            Predicted_Demand=("Predicted_Demand", "sum"),
            MAE=("Absolute_Error", "mean"),
            Total_Absolute_Error=("Absolute_Error", "sum")
        )
        .reset_index()
    )

    product_error["WAPE_pct"] = np.where(
        product_error["Actual_Demand"] != 0,
        100 * product_error["Total_Absolute_Error"]
        / product_error["Actual_Demand"].abs(),
        np.nan
    )

    product_error = product_error.sort_values(
        "WAPE_pct",
        ascending=False,
        na_position="last"
    )

    product_error.to_csv(
        OUTPUT_DIR / "product_forecast_error.csv",
        index=False
    )

    # Week-wise error summary
    weekly_error = (
        predictions.groupby("Week")
        .agg(
            Actual_Demand=("Actual_Demand", "sum"),
            Predicted_Demand=("Predicted_Demand", "sum"),
            Total_Absolute_Error=("Absolute_Error", "sum")
        )
        .reset_index()
    )

    weekly_error["WAPE_pct"] = np.where(
        weekly_error["Actual_Demand"] != 0,
        100 * weekly_error["Total_Absolute_Error"]
        / weekly_error["Actual_Demand"].abs(),
        np.nan
    )

    weekly_error.to_csv(
        OUTPUT_DIR / "weekly_forecast_error.csv",
        index=False
    )

    # Overall metrics
    actual = predictions["Actual_Demand"].to_numpy()
    predicted = predictions["Predicted_Demand"].to_numpy()

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

    metrics = {
        "Model": str(predictions["Model"].iloc[0]),
        "Test_Rows": int(len(predictions)),
        "Products": int(
            predictions["StockCode"].nunique()
        ),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE_pct": float(wape)
    }

    with open(
        OUTPUT_DIR / "forecast_error_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Save combined holdout predictions and errors
    predictions.to_csv(
        OUTPUT_DIR / "holdout_error_details.csv",
        index=False
    )

    return metrics


# ==========================================================
# 8. VISUALIZATIONS
# ==========================================================

def create_error_visualizations(predictions):

    print("\nCreating forecast visualizations...")

    plot_data = predictions

    if len(plot_data) > MAX_PLOT_ROWS:
        plot_data = plot_data.sample(
            MAX_PLOT_ROWS,
            random_state=RANDOM_STATE
        )

    # Actual vs Predicted
    plt.figure(figsize=(9, 7))

    plt.scatter(
        plot_data["Actual_Demand"],
        plot_data["Predicted_Demand"],
        alpha=0.35
    )

    minimum = min(
        plot_data["Actual_Demand"].min(),
        plot_data["Predicted_Demand"].min()
    )

    maximum = max(
        plot_data["Actual_Demand"].max(),
        plot_data["Predicted_Demand"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs Predicted Demand")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=200
    )

    plt.close()

    # Residual distribution
    plt.figure(figsize=(9, 6))

    plt.hist(
        plot_data["Residual"],
        bins=50,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Forecast Residual Distribution")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "residual_distribution.png",
        dpi=200
    )

    plt.close()

    # Weekly WAPE
    weekly = (
        predictions.groupby("Week")
        .agg(
            Actual=("Actual_Demand", "sum"),
            Absolute_Error=("Absolute_Error", "sum")
        )
        .reset_index()
    )

    weekly["WAPE_pct"] = np.where(
        weekly["Actual"] != 0,
        100 * weekly["Absolute_Error"]
        / weekly["Actual"].abs(),
        np.nan
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        weekly["Week"],
        weekly["WAPE_pct"],
        marker="o"
    )

    plt.xlabel("Week")
    plt.ylabel("WAPE (%)")
    plt.title("Weekly Forecast Error Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "weekly_forecast_error.png",
        dpi=200
    )

    plt.close()

    print("Forecast visualizations saved.")


# ==========================================================
# 9. MAIN PIPELINE
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("EXPLAINABLE AI & FORECAST ERROR ANALYSIS")
    print("=" * 65)

    # Load production model
    model, model_name = load_trained_model()

    # Prepare data for SHAP
    explanation_data = prepare_explanation_data()

    # SHAP analysis
    sample, X, explainer, shap_values = (
        calculate_shap_values(
            model,
            explanation_data
        )
    )

    importance = save_global_explanations(
        X,
        shap_values
    )

    save_local_explanation(
        model,
        explainer,
        X,
        shap_values,
        sample
    )

    # Historical holdout error analysis
    predictions = load_holdout_predictions(
        model_name
    )

    metrics = save_error_analysis(
        predictions
    )

    create_error_visualizations(
        predictions
    )

    print("\n" + "=" * 65)
    print("EXPLAINABLE AI PIPELINE COMPLETED")
    print("=" * 65)

    print("\nSELECTED MODEL:")
    print(model_name)

    print("\nFORECAST METRICS:")
    print(json.dumps(metrics, indent=4))

    print("\nTOP 10 SHAP FEATURES:")
    print(importance.head(10).to_string(index=False))

    print(f"\nAll reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
