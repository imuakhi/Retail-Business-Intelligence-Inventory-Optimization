
from pathlib import Path
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "automation"

LOG_PATH = LOG_DIR / "automation_pipeline.log"
SUMMARY_PATH = OUTPUT_DIR / "pipeline_run_summary.json"


def setup_logger():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger("retail_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)

    return logger


def build_pipeline(args):

    steps = [
        (
            "Data Quality and ETL",
            "src.data_engineering.etl_pipeline"
        )
    ]

    # Warehouse rebuilding is destructive and therefore
    # requires an explicit command-line flag.
    if args.rebuild_warehouse:

        steps.extend([
            (
                "Database Warehouse Rebuild",
                "src.database.load_warehouse"
            ),
            (
                "Advanced SQL Analytics",
                "src.database.run_queries"
            )
        ])

    else:

        print(
            "\nSkipping database rebuild and SQL execution. "
            "Existing warehouse and SQL reports will be reused."
        )

    if not args.skip_model_training:

        steps.extend([
            (
                "Global Demand Forecasting",
                "src.forecasting.global_demand_forecasting"
            ),
            (
                "Optuna Hyperparameter Optimization",
                "src.forecasting.optuna_tuning"
            ),
            (
                "Explainable AI",
                "src.forecasting.explain_forecasting"
            )
        ])

    else:

        print(
            "\nSkipping model training and explainability. "
            "Existing model reports will be reused."
        )

    steps.extend([
        (
            "Customer Intelligence",
            "src.customer_intelligence.customer_intelligence"
        ),
        (
            "Monte Carlo Inventory Simulation",
            "src.inventory.monte_carlo_inventory"
        ),
        (
            "Data and Model Monitoring",
            "src.monitoring.monitoring"
        )
    ])

    return steps


def execute_step(name, module, logger):

    command = [
        sys.executable,
        "-m",
        module
    ]

    print("\n" + "=" * 70)
    print(f"STARTING: {name}")
    print(f"COMMAND: {' '.join(command)}")
    print("=" * 70)

    logger.info(
        "STARTING %s | %s",
        name,
        module
    )

    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    for line in process.stdout:

        print(line, end="")

        logger.info(
            "[%s] %s",
            name,
            line.rstrip()
        )

    return_code = process.wait()

    elapsed = time.perf_counter() - start_time

    status = (
        "SUCCESS"
        if return_code == 0
        else "FAILED"
    )

    logger.info(
        "FINISHED %s | %s | %.2f seconds",
        name,
        status,
        elapsed
    )

    print(
        f"\n{name}: {status} "
        f"({elapsed:.2f} seconds)"
    )

    return {
        "stage": name,
        "module": module,
        "status": status,
        "return_code": return_code,
        "duration_seconds": round(elapsed, 2)
    }


def save_summary(results, started_at):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    failed = [
        result
        for result in results
        if result["status"] != "SUCCESS"
    ]

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "overall_status": (
            "FAILED" if failed else "SUCCESS"
        ),
        "total_stages": len(results),
        "successful_stages": sum(
            result["status"] == "SUCCESS"
            for result in results
        ),
        "failed_stages": len(failed),
        "stages": results
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    return summary


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automated Retail BI and Inventory "
            "Analytics Pipeline"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["full", "monitor"],
        default="full",
        help="Run full pipeline or monitoring only."
    )

    parser.add_argument(
        "--rebuild-warehouse",
        action="store_true",
        help=(
            "Clear and completely reload the SQLite warehouse "
            "before running SQL analytics."
        )
    )

    parser.add_argument(
        "--skip-model-training",
        action="store_true",
        help=(
            "Reuse existing forecasting model reports "
            "instead of retraining."
        )
    )

    args = parser.parse_args()

    logger = setup_logger()

    started_at = datetime.now().isoformat()

    if args.mode == "monitor":

        steps = [
            (
                "Data and Model Monitoring",
                "src.monitoring.monitoring"
            )
        ]

    else:

        steps = build_pipeline(args)

    results = []

    print("\nRETAIL BI AUTOMATION PIPELINE")
    print(f"Mode: {args.mode}")
    print(f"Total planned stages: {len(steps)}")

    for name, module in steps:

        result = execute_step(
            name,
            module,
            logger
        )

        results.append(result)

        if result["return_code"] != 0:

            print(
                "\nPipeline stopped because a stage failed."
            )

            break

    summary = save_summary(
        results,
        started_at
    )

    logger.info(
        "PIPELINE OVERALL STATUS: %s",
        summary["overall_status"]
    )

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print(f"\nLog file: {LOG_PATH}")
    print(f"Summary file: {SUMMARY_PATH}")

    if summary["overall_status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
