
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.forecasting.global_demand_forecasting import (
    load_weekly_demand,
    create_features,
    calculate_metrics,
    forecast_future,
    FEATURE_COLUMNS,
    HORIZON,
    TEST_WEEKS,
    MAX_TRAIN_ROWS,
    RANDOM_STATE
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "optuna_forecasting"
)

MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"

# Adjust these to control execution time.
N_TRIALS = 10
MAX_OPTUNA_ROWS = 120_000
FOLDS = 3
VALIDATION_WEEKS = 4


def sample_training_data(data, maximum=MAX_OPTUNA_ROWS):
    if len(data) > maximum:
        return data.sample(
            n=maximum,
            random_state=RANDOM_STATE
        )

    return data


def build_model(model_name, trial):

    if model_name == "XGBoost":

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 150, 450, step=50
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 9
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.15, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 10
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 10, log=True
            )
        }

        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params
        )

    if model_name == "LightGBM":

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 150, 450, step=50
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.15, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 4, 12
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 60
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.01, 5, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 10, log=True
            )
        }

        return LGBMRegressor(
            objective="regression",
            subsample_freq=1,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params
        )

    raise ValueError(f"Unknown model: {model_name}")


def create_walk_forward_folds(data, test_start):

    tuning_start = (
        test_start - pd.Timedelta(
            weeks=FOLDS * VALIDATION_WEEKS
        )
    )

    folds = []

    for fold_number in range(FOLDS):

        validation_start = (
            tuning_start
            + pd.Timedelta(
                weeks=fold_number * VALIDATION_WEEKS
            )
        )

        validation_end = (
            validation_start
            + pd.Timedelta(weeks=VALIDATION_WEEKS)
        )

        train = data[
            data["Week"] < validation_start
        ].copy()

        validation = data[
            (data["Week"] >= validation_start)
            & (data["Week"] < validation_end)
        ].copy()

        if train.empty or validation.empty:
            raise ValueError(
                f"Empty time-series fold: {fold_number + 1}"
            )

        folds.append((train, validation))

    return folds


def objective_factory(model_name, folds):

    def objective(trial):

        fold_scores = []

        for fold_number, (train, validation) in enumerate(
            folds, start=1
        ):

            train_sample = sample_training_data(train)

            X_train = train_sample[FEATURE_COLUMNS]
            y_train = np.log1p(
                train_sample["Demand"].clip(lower=0)
            )

            X_valid = validation[FEATURE_COLUMNS]
            actual = validation["Demand"].to_numpy()

            model = build_model(model_name, trial)

            model.fit(X_train, y_train)

            prediction = np.maximum(
                np.expm1(model.predict(X_valid)),
                0
            )

            metrics = calculate_metrics(
                actual, prediction
            )

            score = metrics["WAPE_pct"]

            if not np.isfinite(score):
                raise ValueError(
                    "Invalid WAPE score in validation fold."
                )

            fold_scores.append(score)

        return float(np.mean(fold_scores))

    return objective


def fit_model(model_name, params, train):

    train_sample = sample_training_data(
        train, MAX_TRAIN_ROWS
    )

    X_train = train_sample[FEATURE_COLUMNS]
    y_train = np.log1p(
        train_sample["Demand"].clip(lower=0)
    )

    model = (
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params
        )
        if model_name == "XGBoost"
        else LGBMRegressor(
            objective="regression",
            subsample_freq=1,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params
        )
    )

    model.fit(X_train, y_train)

    return model


def evaluate_final_model(model_name, model, test):

    prediction = np.maximum(
        np.expm1(
            model.predict(test[FEATURE_COLUMNS])
        ),
        0
    )

    metrics = calculate_metrics(
        test["Demand"].to_numpy(),
        prediction
    )

    output = test[
        ["StockCode", "Week", "Demand"]
    ].copy()

    output["Model"] = model_name
    output["Predicted_Demand"] = prediction

    output = output.rename(
        columns={"Demand": "Actual_Demand"}
    )

    return metrics, output


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    panel, product_map, all_weeks = load_weekly_demand()

    if len(all_weeks) <= (
        TEST_WEEKS + FOLDS * VALIDATION_WEEKS + 12
    ):
        raise ValueError(
            "Not enough weekly history for walk-forward validation."
        )

    featured = create_features(panel)

    featured = featured.dropna(
        subset=["lag_1"]
    ).reset_index(drop=True)

    test_start = pd.Timestamp(
        all_weeks[-TEST_WEEKS]
    )

    development = featured[
        featured["Week"] < test_start
    ].copy()

    final_test = featured[
        featured["Week"] >= test_start
    ].copy()

    if development.empty or final_test.empty:
        raise ValueError(
            "Development or final testing dataset is empty."
        )

    folds = create_walk_forward_folds(
        development, test_start
    )

    print(f"Walk-forward folds: {len(folds)}")
    print(f"Final test begins: {test_start.date()}")

    all_results = []
    all_predictions = []
    best_candidates = {}

    for model_name in ["XGBoost", "LightGBM"]:

        print(f"\nOptimizing {model_name}...")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE
            )
        )

        study.optimize(
            objective_factory(model_name, folds),
            n_trials=N_TRIALS,
            show_progress_bar=False
        )

        print(f"Best validation WAPE: {study.best_value:.4f}%")
        print(f"Best parameters: {study.best_params}")

        # Save all optimization trials.
        trial_file = (
            REPORT_DIR
            / f"{model_name.lower()}_optuna_trials.csv"
        )

        study.trials_dataframe().to_csv(
            trial_file, index=False
        )

        # Save the best parameter set.
        best_candidates[model_name] = {
            "validation_wape": study.best_value,
            "parameters": study.best_params
        }

        with open(
            REPORT_DIR
            / f"{model_name.lower()}_best_params.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                best_candidates[model_name],
                file,
                indent=4
            )

        # Train on the development period and test
        # against the final untouched 12-week period.
        development_model = fit_model(
            model_name,
            study.best_params,
            development
        )

        metrics, predictions = evaluate_final_model(
            model_name,
            development_model,
            final_test
        )

        all_results.append({
            "Model": model_name,
            "Walk_Forward_WAPE": study.best_value,
            **metrics
        })

        all_predictions.append(predictions)

    comparison = pd.DataFrame(
        all_results
    ).sort_values("WAPE_pct")

    comparison.to_csv(
        REPORT_DIR / "tuned_model_comparison.csv",
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df.to_csv(
        REPORT_DIR / "tuned_test_predictions.csv",
        index=False
    )

    print("\nFINAL HOLDOUT COMPARISON")
    print(comparison.to_string(index=False))

    # Select model based on the final holdout WAPE.
    best_name = comparison.iloc[0]["Model"]
    best_params = best_candidates[best_name]["parameters"]

    print(f"\nSelected tuned model: {best_name}")

    # Refit selected model on all available historical
    # features after the holdout evaluation.
    final_model = fit_model(
        best_name,
        best_params,
        featured
    )

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "features": FEATURE_COLUMNS,
        "parameters": best_params,
        "product_map": product_map
    }

    joblib.dump(
        bundle,
        MODEL_DIR / "best_tuned_demand_model.joblib"
    )

    # Feature importance
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": final_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR / "tuned_feature_importance.csv",
        index=False
    )

    # Recursive 12-week future forecast.
    future = forecast_future(
        panel=panel,
        model=final_model,
        product_map=product_map,
        all_weeks=all_weeks,
        horizon=HORIZON
    )

    future.to_csv(
        REPORT_DIR / "tuned_future_forecast.csv",
        index=False
    )

    summary = pd.DataFrame([{
        "Selected_Model": best_name,
        "Final_Test_WAPE": comparison.iloc[0]["WAPE_pct"],
        "Final_Test_MAE": comparison.iloc[0]["MAE"],
        "Final_Test_RMSE": comparison.iloc[0]["RMSE"],
        "Walk_Forward_Folds": FOLDS,
        "Trials_Per_Model": N_TRIALS,
        "Future_Forecast_Weeks": HORIZON,
        "Forecasted_Products": future["StockCode"].nunique()
    }])

    summary.to_csv(
        REPORT_DIR / "tuning_summary.csv",
        index=False
    )

    print("\nOPTUNA PIPELINE COMPLETED")
    print(summary.to_string(index=False))
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
