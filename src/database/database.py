
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "retail_analytics.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def initialize_database():
    connection = get_connection()

    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        connection.executescript(schema)
        connection.commit()
        print("Database tables created successfully!")
        print(f"Database location: {DB_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
