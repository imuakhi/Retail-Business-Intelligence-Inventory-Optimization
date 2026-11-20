
import sqlite3
from pathlib import Path


def test_database_schema_creates_all_tables(tmp_path):

    project_root = Path(__file__).resolve().parents[1]

    schema_path = project_root / "sql" / "schema.sql"

    assert schema_path.exists()

    database_path = tmp_path / "test_warehouse.db"

    connection = sqlite3.connect(database_path)

    try:

        connection.execute("PRAGMA foreign_keys = ON")

        schema = schema_path.read_text(
            encoding="utf-8"
        )

        connection.executescript(schema)

        result = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        table_names = {
            row[0]
            for row in result
        }

        expected_tables = {
            "dim_customer",
            "dim_product",
            "dim_date",
            "dim_country",
            "fact_sales"
        }

        assert expected_tables.issubset(table_names)

    finally:
        connection.close()


def test_fact_sales_foreign_keys_are_enforced(tmp_path):

    project_root = Path(__file__).resolve().parents[1]

    schema_path = project_root / "sql" / "schema.sql"

    database_path = tmp_path / "test_foreign_keys.db"

    connection = sqlite3.connect(database_path)

    try:

        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript(
            schema_path.read_text(encoding="utf-8")
        )

        try:

            connection.execute("""
                INSERT INTO fact_sales(
                    invoice,
                    customer_key,
                    product_key,
                    date_key,
                    country_key,
                    quantity,
                    unit_price,
                    revenue,
                    is_return
                )
                VALUES (
                    'TEST-INVOICE',
                    999,
                    999,
                    20260101,
                    999,
                    1,
                    10,
                    10,
                    0
                )
            """)

            connection.commit()

            assert False, "Foreign key constraint was not enforced."

        except sqlite3.IntegrityError:
            connection.rollback()

    finally:
        connection.close()
