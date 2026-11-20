
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_sales.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "customer_analytics"
)

CLUSTER_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "clustering"
)

CHURN_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "churn"
)

RANDOM_STATE = 42
CHURN_HORIZON_DAYS = 90
KMEANS_MAX_CLUSTERS = 8

# These are illustrative assumptions, not actual company margins.
CLV_SCENARIOS = {
    "Conservative": {
        "gross_margin": 0.10,
        "lifespan_years": 1.0
    },
    "Base": {
        "gross_margin": 0.20,
        "lifespan_years": 2.0
    },
    "Optimistic": {
        "gross_margin": 0.30,
        "lifespan_years": 3.0
    }
}

CHURN_FEATURES = [
    "Recency_Days",
    "Frequency",
    "Gross_Revenue",
    "Avg_Order_Value",
    "Unique_Products",
    "Active_Days",
    "Tenure_Days",
    "Orders_Per_30_Days",
    "Return_Value",
    "Return_Rate",
    "Mean_Interpurchase_Days"
]


# ==========================================================
# 2. DATA LOADING AND CLEANING
# ==========================================================

def normalize_customer_id(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() in ["", "nan", "none", "null"]:
        return np.nan

    if value.endswith(".0"):
        value = value[:-2]

    return value


def load_transactions():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    required_columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID"
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Customer ID": "string"
        },
        low_memory=False
    )

    df["Customer_ID"] = (
        df["Customer ID"].map(normalize_customer_id)
    )

    df["Invoice"] = df["Invoice"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Customer_ID",
            "Invoice",
            "InvoiceDate",
            "Quantity",
            "Price"
        ]
    ).copy()

    df = df[
        (df["Invoice"] != "")
        & (df["StockCode"].notna())
        & (df["StockCode"] != "")
    ].copy()

    df["Revenue"] = (
        df["Quantity"] * df["Price"]
    )

    if df.empty:
        raise ValueError(
            "No valid customer transaction records found."
        )

    print(f"Valid customer transactions: {len(df):,}")
    print(
        f"Unique known customers: "
        f"{df['Customer_ID'].nunique():,}"
    )

    return df


# ==========================================================
# 3. RFM SCORING
# ==========================================================

def quantile_score(series, higher_is_better=True):

    values = pd.to_numeric(
        series, errors="coerce"
    ).fillna(0)

    if len(values) < 4:
        percentile = values.rank(
            method="average",
            pct=True
        )

        score = np.ceil(
            percentile * 4
        ).clip(1, 4).astype(int)

        if not higher_is_better:
            score = 5 - score

        return score

    ranks = values.rank(
        method="first",
        ascending=True
    )

    if higher_is_better:
        labels = [1, 2, 3, 4]
    else:
        labels = [4, 3, 2, 1]

    return pd.qcut(
        ranks,
        q=4,
        labels=labels
    ).astype(int)


# ==========================================================
# 4. RFM CUSTOMER ANALYSIS
# ==========================================================

def build_rfm_analysis(df):

    print("\nBuilding RFM customer metrics...")

    # Positive quantity and positive price represent sales.
    sales = df[
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
    ].copy()

    if sales.empty:
        raise ValueError("No positive purchase transactions found.")

    # One order can contain several product lines.
    orders = (
        sales.groupby(
            ["Customer_ID", "Invoice"],
            as_index=False
        )
        .agg(
            Order_Date=("InvoiceDate", "max"),
            Order_Revenue=("Revenue", "sum"),
            Order_Units=("Quantity", "sum")
        )
    )

    reference_date = (
        sales["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        orders.groupby("Customer_ID")
        .agg(
            First_Purchase=("Order_Date", "min"),
            Last_Purchase=("Order_Date", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("Order_Revenue", "sum"),
            Total_Units=("Order_Units", "sum"),
            Active_Days=("Order_Date", "nunique")
        )
        .reset_index()
    )

    product_counts = (
        sales.groupby("Customer_ID")["StockCode"]
        .nunique()
        .rename("Unique_Products")
    )

    rfm = rfm.merge(
        product_counts,
        on="Customer_ID",
        how="left"
    )

    rfm["Recency_Days"] = (
        reference_date - rfm["Last_Purchase"]
    ).dt.days.clip(lower=0)

    rfm["Avg_Order_Value"] = (
        rfm["Monetary"]
        / rfm["Frequency"].clip(lower=1)
    )

    # Return transactions are reported separately.
    returns = df[
        (df["Quantity"] < 0)
        | (df["Revenue"] < 0)
    ].copy()

    if not returns.empty:

        returns["Return_Value"] = (
            returns["Revenue"].abs()
        )

        return_metrics = (
            returns.groupby("Customer_ID")
            .agg(
                Return_Value=("Return_Value", "sum"),
                Return_Lines=("Invoice", "size"),
                Return_Invoices=("Invoice", "nunique")
            )
            .reset_index()
        )

        rfm = rfm.merge(
            return_metrics,
            on="Customer_ID",
            how="left"
        )

    else:

        rfm["Return_Value"] = 0.0
        rfm["Return_Lines"] = 0
        rfm["Return_Invoices"] = 0

    rfm[
        ["Return_Value", "Return_Lines", "Return_Invoices"]
    ] = rfm[
        ["Return_Value", "Return_Lines", "Return_Invoices"]
    ].fillna(0)

    rfm["Net_Revenue_After_Returns"] = (
        rfm["Monetary"] - rfm["Return_Value"]
    )

    rfm["Return_Rate"] = (
        rfm["Return_Value"]
        / rfm["Monetary"].replace(0, np.nan)
    ).fillna(0)

    # RFM scores: higher is better for all final scores.
    rfm["R_Score"] = quantile_score(
        rfm["Recency_Days"],
        higher_is_better=False
    )

    rfm["F_Score"] = quantile_score(
        rfm["Frequency"],
        higher_is_better=True
    )

    rfm["M_Score"] = quantile_score(
        rfm["Monetary"],
        higher_is_better=True
    )

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    rfm["RFM_Total_Score"] = (
        rfm["R_Score"]
        + rfm["F_Score"]
        + rfm["M_Score"]
    )

    conditions = [
        (
            (rfm["R_Score"] >= 4)
            & (rfm["F_Score"] >= 4)
            & (rfm["M_Score"] >= 4)
        ),
        (
            (rfm["F_Score"] >= 3)
            & (rfm["R_Score"] >= 2)
        ),
        (
            (rfm["R_Score"] >= 3)
            & (rfm["F_Score"] >= 2)
        ),
        (
            (rfm["R_Score"] >= 3)
            & (rfm["F_Score"] <= 2)
        ),
        (
            (rfm["R_Score"] <= 2)
            & (rfm["F_Score"] >= 3)
        ),
        (
            (rfm["R_Score"] <= 2)
            & (rfm["F_Score"] <= 2)
        )
    ]

    segment_names = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Recent Customers",
        "At Risk",
        "Hibernating"
    ]

    rfm["RFM_Segment"] = np.select(
        conditions,
        segment_names,
        default="Needs Attention"
    )

    rfm = rfm.sort_values(
        "Monetary",
        ascending=False
    )

    rfm.to_csv(
        OUTPUT_DIR / "rfm_customer_segments.csv",
        index=False
    )

    segment_summary = (
        rfm.groupby("RFM_Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Avg_Recency=("Recency_Days", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Total_Monetary=("Monetary", "sum"),
            Avg_Monetary=("Monetary", "mean")
        )
        .reset_index()
    )

    segment_summary.to_csv(
        OUTPUT_DIR / "rfm_segment_summary.csv",
        index=False
    )

    # RFM segment chart
    plt.figure(figsize=(10, 6))

    segment_counts = (
        rfm["RFM_Segment"]
        .value_counts()
        .sort_values(ascending=False)
    )

    segment_counts.plot(kind="bar")

    plt.title("RFM Customer Segment Distribution")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rfm_segment_distribution.png",
        dpi=200
    )

    plt.close()

    print("RFM analysis completed.")

    return rfm, sales


# ==========================================================
# 5. K-MEANS CUSTOMER CLUSTERING
# ==========================================================

def run_kmeans_clustering(rfm):

    print("\nRunning K-Means clustering...")

    cluster_features = [
        "Recency_Days",
        "Frequency",
        "Monetary"
    ]

    X = rfm[cluster_features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X = np.log1p(
        X.clip(lower=0)
    )

    if len(X) < 5:
        raise ValueError(
            "At least 5 customers are required for clustering."
        )

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    maximum_k = min(
        KMEANS_MAX_CLUSTERS,
        len(X) - 1
    )

    evaluation_rows = []

    for k in range(2, maximum_k + 1):

        print(f"Evaluating K = {k}")

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(10000, len(X_scaled)),
            random_state=RANDOM_STATE
        )

        evaluation_rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette_Score": score
        })

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    evaluation.to_csv(
        OUTPUT_DIR / "kmeans_cluster_evaluation.csv",
        index=False
    )

    best_k = int(
        evaluation.sort_values(
            "Silhouette_Score",
            ascending=False
        ).iloc[0]["K"]
    )

    print(f"Selected number of clusters: {best_k}")

    final_model = KMeans(
        n_clusters=best_k,
        n_init=20,
        random_state=RANDOM_STATE
    )

    clusters = final_model.fit_predict(
        X_scaled
    )

    cluster_data = rfm.copy()

    cluster_data["KMeans_Cluster"] = clusters

    cluster_data.to_csv(
        OUTPUT_DIR / "kmeans_customer_segments.csv",
        index=False
    )

    cluster_profiles = (
        cluster_data.groupby("KMeans_Cluster")
        .agg(
            Customer_Count=("Customer_ID", "count"),
            Avg_Recency=("Recency_Days", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Return_Rate=("Return_Rate", "mean")
        )
        .reset_index()
    )

    cluster_profiles.to_csv(
        OUTPUT_DIR / "kmeans_cluster_profiles.csv",
        index=False
    )

    # Inertia and silhouette selection chart
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        evaluation["K"],
        evaluation["Inertia"],
        marker="o"
    )

    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-Means Cluster Selection")

    ax2 = ax1.twinx()

    ax2.plot(
        evaluation["K"],
        evaluation["Silhouette_Score"],
        marker="s",
        linestyle="--"
    )

    ax2.set_ylabel("Silhouette Score")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "kmeans_cluster_selection.png",
        dpi=200
    )

    plt.close(fig)

    joblib.dump(
        {
            "model": final_model,
            "scaler": scaler,
            "features": cluster_features,
            "selected_k": best_k
        },
        CLUSTER_MODEL_DIR / "customer_kmeans_model.joblib"
    )

    print("K-Means clustering completed.")

    return cluster_data, evaluation


# ==========================================================
# 6. CUSTOMER LIFETIME VALUE SCENARIOS
# ==========================================================

def calculate_clv_scenarios(rfm, sales):

    print("\nCalculating customer lifetime value scenarios...")

    first_date = sales["InvoiceDate"].min()
    last_date = sales["InvoiceDate"].max()

    observation_years = max(
        (last_date - first_date).days / 365.25,
        1 / 12
    )

    clv = rfm.copy()

    clv["Observation_Period_Years"] = observation_years

    clv["Annualized_Revenue"] = (
        clv["Monetary"] / observation_years
    )

    clv["Annualized_Purchase_Frequency"] = (
        clv["Frequency"] / observation_years
    )

    clv["Annualized_Revenue"] = (
        clv["Annualized_Revenue"].clip(lower=0)
    )

    clv["Revenue_CLV_Conservative"] = (
        clv["Annualized_Revenue"]
        * CLV_SCENARIOS["Conservative"]["lifespan_years"]
    )

    clv["Profit_CLV_Conservative"] = (
        clv["Revenue_CLV_Conservative"]
        * CLV_SCENARIOS["Conservative"]["gross_margin"]
    )

    clv["Revenue_CLV_Base"] = (
        clv["Annualized_Revenue"]
        * CLV_SCENARIOS["Base"]["lifespan_years"]
    )

    clv["Profit_CLV_Base"] = (
        clv["Revenue_CLV_Base"]
        * CLV_SCENARIOS["Base"]["gross_margin"]
    )

    clv["Revenue_CLV_Optimistic"] = (
        clv["Annualized_Revenue"]
        * CLV_SCENARIOS["Optimistic"]["lifespan_years"]
    )

    clv["Profit_CLV_Optimistic"] = (
        clv["Revenue_CLV_Optimistic"]
        * CLV_SCENARIOS["Optimistic"]["gross_margin"]
    )

    clv.to_csv(
        OUTPUT_DIR / "customer_clv_scenarios.csv",
        index=False
    )

    assumptions = pd.DataFrame([
        {
            "Scenario": scenario,
            "Gross_Margin_Assumption": values["gross_margin"],
            "Expected_Lifespan_Years": values["lifespan_years"],
            "Basis": "Illustrative assumption; replace with business evidence"
        }
        for scenario, values in CLV_SCENARIOS.items()
    ])

    assumptions.to_csv(
        OUTPUT_DIR / "clv_scenario_assumptions.csv",
        index=False
    )

    print(
        f"Observation period: {observation_years:.2f} years"
    )

    print("CLV scenario analysis completed.")

    return clv


# ==========================================================
# 7. BUILD HISTORICAL CHURN DATASET
# ==========================================================

def build_churn_dataset(df):

    print("\nBuilding 90-day inactivity prediction dataset...")

    maximum_date = df["InvoiceDate"].max()

    observation_cutoff = (
        maximum_date
        - pd.Timedelta(days=CHURN_HORIZON_DAYS)
    )

    # Information available on or before the snapshot.
    historical = df[
        df["InvoiceDate"] <= observation_cutoff
    ].copy()

    # Purchases in the following 90-day outcome window.
    outcome_period = df[
        (df["InvoiceDate"] > observation_cutoff)
        & (df["InvoiceDate"] <= maximum_date)
    ].copy()

    historical_sales = historical[
        (historical["Quantity"] > 0)
        & (historical["Price"] > 0)
    ].copy()

    future_sales = outcome_period[
        (outcome_period["Quantity"] > 0)
        & (outcome_period["Price"] > 0)
    ].copy()

    if historical_sales.empty:
        raise ValueError(
            "No historical sales before churn observation cutoff."
        )

    # Build one row per customer order.
    orders = (
        historical_sales.groupby(
            ["Customer_ID", "Invoice"],
            as_index=False
        )
        .agg(
            Order_Date=("InvoiceDate", "max"),
            Order_Revenue=("Revenue", "sum")
        )
    )

    customers = (
        orders.groupby("Customer_ID")
        .agg(
            First_Purchase=("Order_Date", "min"),
            Last_Purchase=("Order_Date", "max"),
            Frequency=("Invoice", "nunique"),
            Gross_Revenue=("Order_Revenue", "sum"),
            Active_Days=("Order_Date", "nunique"),
            Avg_Order_Value=("Order_Revenue", "mean")
        )
    )

    product_counts = (
        historical_sales.groupby("Customer_ID")["StockCode"]
        .nunique()
        .rename("Unique_Products")
    )

    customers = customers.join(
        product_counts,
        how="left"
    )

    customers["Recency_Days"] = (
        observation_cutoff - customers["Last_Purchase"]
    ).dt.days.clip(lower=0)

    customers["Tenure_Days"] = (
        observation_cutoff - customers["First_Purchase"]
    ).dt.days.clip(lower=0)

    customers["Orders_Per_30_Days"] = (
        customers["Frequency"]
        / (customers["Tenure_Days"].clip(lower=30) / 30)
    )

    # Average days between orders.
    ordered = orders.sort_values(
        ["Customer_ID", "Order_Date"]
    ).copy()

    ordered["Purchase_Gap_Days"] = (
        ordered.groupby("Customer_ID")["Order_Date"]
        .diff()
        .dt.days
    )

    gap_features = (
        ordered.groupby("Customer_ID")["Purchase_Gap_Days"]
        .mean()
        .rename("Mean_Interpurchase_Days")
    )

    customers = customers.join(
        gap_features,
        how="left"
    )

    customers["Mean_Interpurchase_Days"] = (
        customers["Mean_Interpurchase_Days"].fillna(0)
    )

    # Returns up to the observation cutoff.
    historical_returns = historical[
        (historical["Quantity"] < 0)
        | (historical["Revenue"] < 0)
    ].copy()

    historical_returns["Return_Value"] = (
        historical_returns["Revenue"].abs()
    )

    return_values = (
        historical_returns.groupby("Customer_ID")["Return_Value"]
        .sum()
        .rename("Return_Value")
    )

    customers = customers.join(
        return_values,
        how="left"
    )

    customers["Return_Value"] = (
        customers["Return_Value"].fillna(0)
    )

    customers["Return_Rate"] = (
        customers["Return_Value"]
        / customers["Gross_Revenue"].replace(0, np.nan)
    ).fillna(0)

    # Label = 1 means no purchase in the next 90 days.
    future_customer_ids = set(
        future_sales["Customer_ID"].unique()
    )

    customers["Churn_90d"] = (
        ~customers.index.isin(future_customer_ids)
    ).astype(int)

    customers = customers.replace(
        [np.inf, -np.inf],
        np.nan
    )

    customers.to_csv(
        OUTPUT_DIR / "customer_churn_features.csv",
        index_label="Customer_ID"
    )

    churn_rate = (
        customers["Churn_90d"].mean() * 100
    )

    print(f"Observation cutoff: {observation_cutoff}")
    print(f"Outcome end: {maximum_date}")
    print(f"Churn proxy rate: {churn_rate:.2f}%")
    print(f"Churn dataset customers: {len(customers):,}")

    return customers, observation_cutoff, maximum_date


# ==========================================================
# 8. CHURN MODEL TRAINING AND EVALUATION
# ==========================================================

def run_churn_prediction(customers, cutoff, maximum_date):

    print("\nTraining customer inactivity prediction models...")

    X = customers[CHURN_FEATURES].copy()
    y = customers["Churn_90d"].astype(int)

    class_counts = y.value_counts()

    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(
            "Churn dataset needs at least two examples "
            "from each target class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model_pipelines = {
        "LogisticRegression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            )
        ])
    }

    comparison_rows = []
    fitted_models = {}
    predictions_by_model = {}

    for model_name, pipeline in model_pipelines.items():

        print(f"\nTraining {model_name}...")

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        predicted_labels = (
            probabilities >= 0.5
        ).astype(int)

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(
                y_test, predicted_labels
            ),
            "Precision": precision_score(
                y_test,
                predicted_labels,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predicted_labels,
                zero_division=0
            ),
            "F1": f1_score(
                y_test,
                predicted_labels,
                zero_division=0
            ),
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        }

        comparison_rows.append(metrics)

        fitted_models[model_name] = pipeline

        predictions_by_model[model_name] = {
            "probabilities": probabilities,
            "labels": predicted_labels
        }

        print(
            f"{model_name}: "
            f"PR-AUC = {pr_auc:.4f}, "
            f"ROC-AUC = {roc_auc:.4f}"
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        ["PR_AUC", "ROC_AUC"],
        ascending=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "churn_model_comparison.csv",
        index=False
    )

    best_name = str(
        comparison.iloc[0]["Model"]
    )

    best_model = fitted_models[best_name]

    best_predictions = predictions_by_model[best_name]

    y_pred = best_predictions["labels"]
    probabilities = best_predictions["probabilities"]

    # Out-of-sample test predictions.
    test_predictions = pd.DataFrame({
        "Customer_ID": X_test.index,
        "Actual_Churn_90d": y_test.to_numpy(),
        "Churn_Probability": probabilities,
        "Predicted_Churn_90d": y_pred
    })

    test_predictions.to_csv(
        OUTPUT_DIR / "churn_test_predictions.csv",
        index=False
    )

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Purchased_Again",
            "No_Purchase_90d"
        ],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "churn_classification_report.csv"
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual_Purchased_Again",
            "Actual_No_Purchase_90d"
        ],
        columns=[
            "Predicted_Purchased_Again",
            "Predicted_No_Purchase_90d"
        ]
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "churn_confusion_matrix.csv"
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Purchased Again",
        "No Purchase"
    ])

    ax.set_yticklabels([
        "Purchased Again",
        "No Purchase"
    ])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Customer Inactivity Prediction - Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "churn_confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)

    # Feature importance / coefficient analysis
    classifier = best_model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):

        importance_values = (
            classifier.feature_importances_
        )

    elif hasattr(classifier, "coef_"):

        importance_values = (
            np.abs(classifier.coef_[0])
        )

    else:

        importance_values = np.zeros(
            len(CHURN_FEATURES)
        )

    importance = pd.DataFrame({
        "Feature": CHURN_FEATURES,
        "Importance": importance_values
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "churn_feature_importance.csv",
        index=False
    )

    # Refit selected estimator on all labelled customers.
    final_model = clone(best_model)

    final_model.fit(X, y)

    metadata = {
        "selected_model": best_name,
        "target_definition": (
            "1 = no positive purchase during the next "
            "90 days after the observation cutoff"
        ),
        "observation_cutoff": str(cutoff),
        "outcome_end_date": str(maximum_date),
        "feature_columns": CHURN_FEATURES,
        "classification_threshold": 0.5,
        "note": (
            "Historical inactivity proxy; not a verified "
            "future churn prediction."
        )
    }

    joblib.dump(
        {
            "model": final_model,
            "features": CHURN_FEATURES,
            "metadata": metadata
        },
        CHURN_MODEL_DIR / "churn_model.joblib"
    )

    with open(
        OUTPUT_DIR / "churn_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"\nSelected churn model: {best_name}")

    return comparison, test_predictions, importance


# ==========================================================
# 9. EXECUTIVE SUMMARY
# ==========================================================

def create_customer_summary(
    rfm,
    cluster_data,
    churn_data,
    cluster_evaluation,
    churn_comparison
):

    best_cluster = int(
        cluster_evaluation.sort_values(
            "Silhouette_Score",
            ascending=False
        ).iloc[0]["K"]
    )

    best_churn_model = str(
        churn_comparison.iloc[0]["Model"]
    )

    summary = {
        "total_customers": int(
            rfm["Customer_ID"].nunique()
        ),
        "rfm_segments": int(
            rfm["RFM_Segment"].nunique()
        ),
        "selected_kmeans_clusters": best_cluster,
        "customers_in_churn_dataset": int(
            len(churn_data)
        ),
        "historical_90_day_no_purchase_rate_pct": round(
            float(churn_data["Churn_90d"].mean() * 100),
            2
        ),
        "selected_churn_model": best_churn_model,
        "clv_scenario_names": list(
            CLV_SCENARIOS.keys()
        ),
        "clv_assumptions": CLV_SCENARIOS
    }

    with open(
        OUTPUT_DIR / "customer_analytics_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nCUSTOMER ANALYTICS SUMMARY")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )


# ==========================================================
# 10. MAIN PIPELINE
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CLUSTER_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHURN_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("ADVANCED CUSTOMER INTELLIGENCE SYSTEM")
    print("=" * 65)

    df = load_transactions()

    # RFM analysis
    rfm, sales = build_rfm_analysis(df)

    # K-Means segmentation
    cluster_data, cluster_evaluation = (
        run_kmeans_clustering(rfm)
    )

    # CLV scenario analysis
    calculate_clv_scenarios(
        rfm,
        sales
    )

    # Historical churn dataset
    churn_data, cutoff, maximum_date = (
        build_churn_dataset(df)
    )

    # Churn model training and evaluation
    churn_comparison, churn_predictions, churn_importance = (
        run_churn_prediction(
            churn_data,
            cutoff,
            maximum_date
        )
    )

    # Executive summary
    create_customer_summary(
        rfm,
        cluster_data,
        churn_data,
        cluster_evaluation,
        churn_comparison
    )

    print("\n" + "=" * 65)
    print("CUSTOMER INTELLIGENCE PIPELINE COMPLETED")
    print("=" * 65)

    print(f"\nReports directory: {OUTPUT_DIR}")
    print(f"Clustering model directory: {CLUSTER_MODEL_DIR}")
    print(f"Churn model directory: {CHURN_MODEL_DIR}")


if __name__ == "__main__":
    main()
