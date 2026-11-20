
from pathlib import Path
import logging
import sys

import pandas as pd

from src.data_engineering.data_quality import (
    load_data,
    run_data_quality_checks,
    save_reports,
    create_quality_chart
)


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data/processed/clean_sales.csv"

OUTPUT_DIR = BASE_DIR / "data/processed"

LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "etl_pipeline.log",
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# ETL PIPELINE
# --------------------------------------------------

def run_pipeline():

    logger.info("ETL pipeline started")

    try:

        # 1. EXTRACT
        logger.info("Extracting source data")

        df = load_data(INPUT_FILE)

        logger.info(
            "Loaded %s rows and %s columns",
            len(df),
            len(df.columns)
        )

        # 2. TRANSFORM
        logger.info("Running data quality validation")

        report, summary = run_data_quality_checks(df)

        # 3. LOAD REPORTS
        logger.info("Saving data quality reports")

        save_reports(report, summary)

        create_quality_chart(report)

        # Save validated dataset separately.
        # Do not silently delete returns or unusual transactions.

        validated_path = (
            OUTPUT_DIR / "quality_checked_sales.csv"
        )

        df.to_csv(validated_path, index=False)

        logger.info(
            "Validated dataset saved: %s",
            validated_path
        )

        logger.info(
            "Quality score: %s%%",
            summary["quality_score_percent"]
        )

        logger.info("ETL pipeline completed successfully")

    except Exception:

        logger.exception("ETL pipeline failed")

        raise


if __name__ == "__main__":
    run_pipeline()
