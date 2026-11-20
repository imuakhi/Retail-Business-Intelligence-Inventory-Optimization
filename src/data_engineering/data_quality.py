from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data/processed/clean_sales.csv"
OUTPUT_DIR = BASE_DIR / "reports/data_quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]


# =========================================================
# DATA LOADING
# =========================================================

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


# =========================================================
# DATA QUALITY ENGINE
# =========================================================

def run_data_quality_checks(df):

    report = []

    def add_check(
        check_name,
        category,
        affected_rows,
        severity,
        details
    ):

        affected_rows = int(affected_rows)

        report.append({
            "Check": check_name,
            "Category": category,
            "Affected_Rows": affected_rows,
            "Severity": severity,
            "Details": details
        })

    total_rows = len(df)

    # -----------------------------------------------------
    # 1. SCHEMA VALIDATION
    # -----------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    add_check(
        "Required Columns",
        "Schema",
        len(missing_columns),
        "CRITICAL" if missing_columns else "PASS",
        (
            f"Missing columns: {missing_columns}"
            if missing_columns
            else "All required columns are present."
        )
    )

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # -----------------------------------------------------
    # 2. MISSING VALUES
    # -----------------------------------------------------

    missing_values = df.isnull().sum()

    for column, count in missing_values.items():

        if count > 0:

            add_check(
                f"Missing Values - {column}",
                "Completeness",
                count,
                "WARNING",
                f"{count} missing values in {column}."
            )

    if missing_values.sum() == 0:

        add_check(
            "Missing Values",
            "Completeness",
            0,
            "PASS",
            "No missing values detected."
        )

    # -----------------------------------------------------
    # 3. DUPLICATE ROWS
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    add_check(
        "Duplicate Rows",
        "Uniqueness",
        duplicate_count,
        "WARNING" if duplicate_count else "PASS",
        f"Detected {duplicate_count} duplicate rows."
    )

    # -----------------------------------------------------
    # 4. NUMERIC VALIDATION
    # -----------------------------------------------------

    numeric_columns = ["Quantity", "Price"]

    for column in numeric_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = (
            df[column].notna() & converted.isna()
        ).sum()

        add_check(
            f"Invalid Numeric Values - {column}",
            "Validity",
            invalid_count,
            "CRITICAL" if invalid_count else "PASS",
            f"{invalid_count} non-numeric values in {column}."
        )

        df[column] = converted

    # -----------------------------------------------------
    # 5. INVALID QUANTITY
    # -----------------------------------------------------

    invalid_quantity = (
        df["Quantity"].notna() &
        (df["Quantity"] == 0)
    ).sum()

    negative_quantity = (
        df["Quantity"].notna() &
        (df["Quantity"] < 0)
    ).sum()

    add_check(
        "Zero Quantity",
        "Business Rules",
        invalid_quantity,
        "WARNING" if invalid_quantity else "PASS",
        "Zero-quantity transactions require review."
    )

    add_check(
        "Negative Quantity / Returns",
        "Business Rules",
        negative_quantity,
        "INFO" if negative_quantity else "PASS",
        "Negative quantities may represent legitimate returns."
    )

    # -----------------------------------------------------
    # 6. INVALID PRICE
    # -----------------------------------------------------

    invalid_price = (
        df["Price"].notna() &
        (df["Price"] < 0)
    ).sum()

    zero_price = (
        df["Price"].notna() &
        (df["Price"] == 0)
    ).sum()

    add_check(
        "Negative Price",
        "Business Rules",
        invalid_price,
        "CRITICAL" if invalid_price else "PASS",
        "Negative unit prices require investigation."
    )

    add_check(
        "Zero Price",
        "Business Rules",
        zero_price,
        "WARNING" if zero_price else "PASS",
        "Zero-priced products require business review."
    )

    # -----------------------------------------------------
    # 7. DATE VALIDATION
    # -----------------------------------------------------

    parsed_dates = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    invalid_dates = (
        df["InvoiceDate"].notna() &
        parsed_dates.isna()
    ).sum()

    add_check(
        "Invalid Invoice Dates",
        "Validity",
        invalid_dates,
        "CRITICAL" if invalid_dates else "PASS",
        f"{invalid_dates} dates could not be parsed."
    )

    # -----------------------------------------------------
    # 8. OUTLIER DETECTION - IQR
    # -----------------------------------------------------

    for column in ["Quantity", "Price"]:

        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = (
            (values < lower) |
            (values > upper)
        ).sum()

        add_check(
            f"Outliers - {column}",
            "Statistical",
            outlier_count,
            "INFO" if outlier_count else "PASS",
            (
                f"IQR lower bound: {lower:.3f}, "
                f"upper bound: {upper:.3f}."
            )
        )

    # -----------------------------------------------------
    # 9. DATA QUALITY SCORE
    # -----------------------------------------------------

    report_df = pd.DataFrame(report)

    critical_checks = report_df[
        report_df["Severity"] == "CRITICAL"
    ]

    warning_checks = report_df[
        report_df["Severity"] == "WARNING"
    ]

    critical_failures = len(critical_checks)
    warning_failures = len(warning_checks)

    total_checks = len(report_df)

    passed_checks = (
        report_df["Severity"] == "PASS"
    ).sum()

    quality_score = (
        passed_checks / total_checks * 100
        if total_checks else 0
    )

    summary = {
        "total_rows": int(total_rows),
        "total_columns": int(len(df.columns)),
        "total_checks": int(total_checks),
        "passed_checks": int(passed_checks),
        "critical_failures": int(critical_failures),
        "warning_checks": int(warning_failures),
        "quality_score_percent": round(
            float(quality_score), 2
        ),
        "dataset": INPUT_FILE.name,
        "note": (
            "Quality score is a simple check-pass ratio, "
            "not a statistical guarantee of data correctness."
        )
    }

    return report_df, summary


# =========================================================
# REPORT GENERATION
# =========================================================

def save_reports(report_df, summary):

    csv_path = OUTPUT_DIR / "data_quality_report.csv"

    json_path = OUTPUT_DIR / "data_quality_summary.json"

    report_df.to_csv(
        csv_path,
        index=False
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(f"CSV report saved: {csv_path}")
    print(f"JSON summary saved: {json_path}")


# =========================================================
# VISUALIZATION
# =========================================================

def create_quality_chart(report_df):

    chart_data = (
        report_df.groupby("Severity")
        ["Affected_Rows"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        chart_data["Severity"],
        chart_data["Affected_Rows"]
    )

    ax.set_title("Data Quality Issues by Severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Affected Rows")

    fig.tight_layout()

    chart_path = OUTPUT_DIR / "data_quality_dashboard.png"

    fig.savefig(chart_path, dpi=300)

    plt.close(fig)

    print(f"Chart saved: {chart_path}")


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    print("\n[1/4] Loading cleaned retail dataset...")

    df = load_data(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n[2/4] Running data quality checks...")

    report_df, summary = run_data_quality_checks(df)

    print("\n[3/4] Saving quality reports...")

    save_reports(report_df, summary)

    print("\n[4/4] Generating quality visualization...")

    create_quality_chart(report_df)

    print("\n========== DATA QUALITY SUMMARY ==========")

    print(json.dumps(summary, indent=4))

    print("\nDATA QUALITY ENGINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()