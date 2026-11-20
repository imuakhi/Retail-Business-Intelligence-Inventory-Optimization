
from pathlib import Path
import json

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(ttl=300, show_spinner=False)
def load_csv(relative_path):

    file_path = PROJECT_ROOT / relative_path

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            file_path,
            low_memory=False
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_json(relative_path):

    file_path = PROJECT_ROOT / relative_path

    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def first_existing(paths):

    for path in paths:
        if (PROJECT_ROOT / path).exists():
            return path

    return None


def format_number(value, decimals=2):

    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_money(value):

    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def show_missing(label):

    st.info(
        f"{label} report পাওয়া যায়নি। "
        "সংশ্লিষ্ট previous analysis pipeline run করো।"
    )


def month_labels(df):

    df = df.copy()

    if df.empty:
        return df

    if "month.1" in df.columns:
        df["Period"] = df["month.1"].astype(str)

    elif "month" in df.columns:
        month = df["month"]

        # Handle YYYY-MM string column.
        if month.astype(str).str.contains("-").any():
            df["Period"] = month.astype(str)

        elif "year" in df.columns:
            df["Period"] = (
                df["year"].astype(str)
                + "-"
                + month.astype(str).str.zfill(2)
            )

    if "Period" in df.columns:
        df = df.sort_values("Period")

    return df
