
from pathlib import Path
import re
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "retail_analytics.db"
SQL_PATH = PROJECT_ROOT / "sql" / "advanced_analytics.sql"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "sql_analytics"


def load_queries():
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    content = SQL_PATH.read_text(encoding="utf-8")

    pattern = re.compile(
        r"(?m)^-- QUERY:\s*([a-zA-Z0-9_]+)\s*$"
    )

    matches = list(pattern.finditer(content))

    if not matches:
        raise ValueError("No named queries found in SQL file.")

    queries = {}

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        query = content[start:end].strip().rstrip(";").strip()

        if not query:
            raise ValueError(
                f"Empty SQL query: {match.group(1)}"
            )

        queries[match.group(1)] = query

    return queries


def run_analytics():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_queries()
    summary = []

    connection = sqlite3.connect(DB_PATH)

    try:
        for query_name, sql in queries.items():
            print(f"\nRunning: {query_name}")

            result = pd.read_sql_query(sql, connection)

            output_path = OUTPUT_DIR / f"{query_name}.csv"
            result.to_csv(output_path, index=False)

            summary.append({
                "query_name": query_name,
                "rows_returned": len(result),
                "columns": len(result.columns),
                "status": "SUCCESS",
                "output_file": str(output_path.relative_to(PROJECT_ROOT))
            })

            print(f"Rows: {len(result):,}")
            print(f"Saved: {output_path}")

        summary_df = pd.DataFrame(summary)

        summary_path = OUTPUT_DIR / "execution_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nALL SQL ANALYTICS COMPLETED")
        print(summary_df.to_string(index=False))
        print(f"\nReports directory: {OUTPUT_DIR}")

    finally:
        connection.close()


if __name__ == "__main__":
    run_analytics()
