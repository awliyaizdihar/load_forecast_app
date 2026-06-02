import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    CSV_PATH,
    REGION_NAME,
    TRAIN_END_DATE,
    TARGET_COLUMN,
    ACTUAL_COLUMN,
    EIA_FORECAST_COLUMN,
    TRAINING_TARGET_COLUMN,
    LOAD_TABLE_COLUMNS,
)
from db_utils import (
    get_connection,
    create_tables,
    clear_tables,
    upsert_load_hourly,
    get_table_counts,
    get_database_summary,
)


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


def ensure_column(df, column_name, default_value=np.nan):
    if column_name not in df.columns:
        df[column_name] = default_value
    return df


def prepare_csv_for_database(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Please place all_features_all_timerange.csv inside the data/ folder."
        )

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns):,}")

    required_columns = [
        "timestamp_utc",
        "Region",
        ACTUAL_COLUMN,
        TRAINING_TARGET_COLUMN,
    ]

    missing_required = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns in CSV:\n"
            + "\n".join(missing_required)
        )

    # Datetime handling
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp_utc"]).copy()

    # Keep target region only
    df["Region"] = df["Region"].astype(str)
    df = df[df["Region"] == REGION_NAME].copy()

    print(f"Rows after filtering Region={REGION_NAME}: {len(df):,}")

    # Sort before time_idx construction
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Derive local timestamp safely from UTC
    df["timestamp_local"] = (
        df["timestamp_utc"]
        .dt.tz_convert("America/Los_Angeles")
    )

    # Date columns
    if "data_date" in df.columns:
        df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce")
    else:
        df["data_date"] = df["timestamp_utc"].dt.date
        df["data_date"] = pd.to_datetime(df["data_date"])

    if "local_date" in df.columns:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    else:
        df["local_date"] = df["timestamp_local"].dt.date
        df["local_date"] = pd.to_datetime(df["local_date"])

    # Rebuild continuous time_idx to match TFT app logic
    df["time_idx"] = (
        (df["timestamp_utc"] - df["timestamp_utc"].min())
        .dt.total_seconds() // 3600
    ).astype(int)

    # Ensure local calendar columns exist
    df["local_year"] = df["timestamp_local"].dt.year
    df["local_month"] = df["timestamp_local"].dt.month
    df["local_day"] = df["timestamp_local"].dt.day
    df["local_hour"] = df["timestamp_local"].dt.hour
    df["local_day_of_week"] = df["timestamp_local"].dt.dayofweek
    df["local_day_of_year"] = df["timestamp_local"].dt.dayofyear

    # Ensure cyclical columns exist
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24)
    if "hour_cos" not in df.columns:
        df["hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24)

    if "day_of_week_sin" not in df.columns:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["local_day_of_week"] / 7)
    if "day_of_week_cos" not in df.columns:
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["local_day_of_week"] / 7)

    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["local_month"] / 12)
    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * df["local_month"] / 12)

    if "day_of_year_sin" not in df.columns:
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["local_day_of_year"] / 365)
    if "day_of_year_cos" not in df.columns:
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["local_day_of_year"] / 365)

    # Ensure holiday columns exist
    holiday_cols = [
        "is_weekend",
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "is_holiday_period",
    ]

    for col in holiday_cols:
        df = ensure_column(df, col, 0)
        df[col] = safe_binary_mapping(df[col])

    # EIA forecast column may be absent depending on preprocessing version
    df = ensure_column(df, EIA_FORECAST_COLUMN, np.nan)

    # Build model target exactly like TFT app/training logic
    df[TARGET_COLUMN] = np.where(
        df["data_date"] < TRAIN_END_DATE,
        df[TRAINING_TARGET_COLUMN],
        df[ACTUAL_COLUMN]
    )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Metadata
    now = datetime.now(timezone.utc).isoformat()

    df["source"] = "historical_csv"
    df["created_at"] = now
    df["updated_at"] = now

    # Keep only database columns
    for col in LOAD_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[LOAD_TABLE_COLUMNS].copy()

    # Convert datetime columns to ISO strings for SQLite
    datetime_cols = [
        "timestamp_utc",
        "timestamp_local",
        "data_date",
        "local_date",
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").astype(str)

    # Replace NaN with None for SQLite
    df = df.where(pd.notnull(df), None)

    print(f"Prepared rows for database: {len(df):,}")

    return df


def main(reset=False):
    conn = get_connection()

    print("Creating database tables...")
    create_tables(conn)

    if reset:
        print("Reset flag detected. Clearing existing tables...")
        clear_tables(conn)

    df = prepare_csv_for_database(CSV_PATH)

    print("Upserting rows into load_hourly table...")
    inserted_rows = upsert_load_hourly(conn, df)

    print(f"Upserted rows: {inserted_rows:,}")

    counts = get_table_counts(conn)
    print("\nTable counts:")
    for table_name, count in counts.items():
        print(f"- {table_name}: {count:,}")

    summary = get_database_summary(conn)
    print("\nDatabase summary:")
    print(summary.to_string(index=False))

    conn.close()

    print("\nDone. Database seeding completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed SQLite database from historical load forecasting CSV."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing tables before seeding."
    )

    args = parser.parse_args()

    main(reset=args.reset)