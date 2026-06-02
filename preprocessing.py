from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    REGION_NAME,
    TARGET_COLUMN,
    ACTUAL_COLUMN,
    EIA_FORECAST_COLUMN,
    TRAINING_TARGET_COLUMN,
    LOAD_TABLE_COLUMNS,
    LOCAL_TIMEZONE,
)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_binary_mapping(series):
    return series.map({
        "No": 0,
        "Yes": 1,
        "False": 0,
        "True": 1,
        "false": 0,
        "true": 1,
        0: 0,
        1: 1,
        0.0: 0,
        1.0: 1,
        False: 0,
        True: 1,
    }).fillna(series).astype("int8")


def add_calendar_features(df):
    df = df.copy()

    df["timestamp_local"] = (
        df["timestamp_utc"]
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    df["data_date"] = pd.to_datetime(df["timestamp_utc"].dt.date)
    df["local_date"] = pd.to_datetime(df["timestamp_local"].dt.date)

    df["local_year"] = df["timestamp_local"].dt.year
    df["local_month"] = df["timestamp_local"].dt.month
    df["local_day"] = df["timestamp_local"].dt.day
    df["local_hour"] = df["timestamp_local"].dt.hour
    df["local_day_of_week"] = df["timestamp_local"].dt.dayofweek
    df["local_day_of_year"] = df["timestamp_local"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["local_day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["local_day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["local_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["local_month"] / 12)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["local_day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["local_day_of_year"] / 365)

    return df


def add_simple_holiday_features(df):
    """
    Lightweight holiday feature creation.

    For now this uses weekend only and keeps holiday-related columns available.
    Later we can upgrade this to use the `holidays` package exactly like training.
    """
    df = df.copy()

    df["is_weekend"] = (df["local_day_of_week"] >= 5).astype("int8")

    for col in [
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "is_holiday_period",
    ]:
        if col not in df.columns:
            df[col] = 0

        df[col] = safe_binary_mapping(df[col])

    return df


def prepare_datetime_for_sqlite(df):
    df = df.copy()

    for col in [
        "timestamp_utc",
        "timestamp_local",
        "data_date",
        "local_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype(str)

    return df


def preprocess_eia_region_data(
    raw_df,
    historical_start_timestamp,
):
    """
    Convert raw EIA region-data API response into load_hourly-compatible rows.

    Expected EIA raw columns include:
    - period
    - respondent
    - type
    - value

    Type mapping:
    - D  -> region_demand_mw
    - DF -> region_demand_forecast_mw
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=LOAD_TABLE_COLUMNS)

    df = raw_df.copy()

    required_cols = [
        "period",
        "respondent",
        "type",
        "value",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns from EIA API response: "
            + ", ".join(missing_cols)
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["period"],
        utc=True,
        errors="coerce"
    )

    df["value"] = safe_numeric(df["value"])

    df = df.dropna(subset=["timestamp_utc"]).copy()

    df = df[df["respondent"].astype(str) == REGION_NAME].copy()

    if df.empty:
        return pd.DataFrame(columns=LOAD_TABLE_COLUMNS)

    pivot_df = (
        df.pivot_table(
            index=["timestamp_utc", "respondent"],
            columns="type",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    pivot_df = pivot_df.rename(columns={
        "respondent": "Region",
        "D": ACTUAL_COLUMN,
        "DF": EIA_FORECAST_COLUMN,
    })

    if ACTUAL_COLUMN not in pivot_df.columns:
        pivot_df[ACTUAL_COLUMN] = np.nan

    if EIA_FORECAST_COLUMN not in pivot_df.columns:
        pivot_df[EIA_FORECAST_COLUMN] = np.nan

    pivot_df = pivot_df.sort_values("timestamp_utc").reset_index(drop=True)

    historical_start_timestamp = pd.to_datetime(
        historical_start_timestamp,
        utc=True,
        errors="coerce"
    )

    pivot_df["time_idx"] = (
        (pivot_df["timestamp_utc"] - historical_start_timestamp)
        .dt.total_seconds() // 3600
    ).astype("int64")

    pivot_df = add_calendar_features(pivot_df)
    pivot_df = add_simple_holiday_features(pivot_df)

    pivot_df[TRAINING_TARGET_COLUMN] = pivot_df[ACTUAL_COLUMN]
    pivot_df[TARGET_COLUMN] = pivot_df[ACTUAL_COLUMN]

    now = datetime.now(timezone.utc).isoformat()

    pivot_df["source"] = "eia_api"
    pivot_df["created_at"] = now
    pivot_df["updated_at"] = now

    for col in LOAD_TABLE_COLUMNS:
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan

    pivot_df = pivot_df[LOAD_TABLE_COLUMNS].copy()

    pivot_df = prepare_datetime_for_sqlite(pivot_df)

    pivot_df = pivot_df.where(pd.notnull(pivot_df), None)

    return pivot_df