import sqlite3

import pandas as pd

from config import (
    DB_PATH,
    DATABASE_DIR,
    LOAD_TABLE,
    PREDICTION_TABLE,
)


def ensure_database_dir():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(db_path=DB_PATH):
    ensure_database_dir()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables(conn):
    create_load_hourly_table(conn)
    create_model_predictions_table(conn)
    conn.commit()


def create_load_hourly_table(conn):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LOAD_TABLE} (
            timestamp_utc TEXT NOT NULL,
            timestamp_local TEXT,
            data_date TEXT,
            local_date TEXT,
            Region TEXT NOT NULL,

            region_demand_mw REAL,
            region_demand_forecast_mw REAL,
            target_region_demand_mw REAL,
            model_target_mw REAL,

            time_idx INTEGER,

            local_year INTEGER,
            local_month INTEGER,
            local_day INTEGER,
            local_hour INTEGER,
            local_day_of_week INTEGER,
            local_day_of_year INTEGER,

            hour_sin REAL,
            hour_cos REAL,
            day_of_week_sin REAL,
            day_of_week_cos REAL,
            month_sin REAL,
            month_cos REAL,
            day_of_year_sin REAL,
            day_of_year_cos REAL,

            is_weekend INTEGER,
            is_holiday INTEGER,
            is_day_before_holiday INTEGER,
            is_day_after_holiday INTEGER,
            is_holiday_period INTEGER,

            source TEXT,
            created_at TEXT,
            updated_at TEXT,

            PRIMARY KEY (timestamp_utc, Region)
        );
        """
    )


def create_model_predictions_table(conn):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
            timestamp_utc TEXT NOT NULL,
            timestamp_local TEXT,
            Region TEXT NOT NULL,

            model_name TEXT NOT NULL,
            horizon_hours INTEGER NOT NULL,

            prediction_mw REAL,
            actual_mw REAL,
            absolute_error_mw REAL,
            ape REAL,

            prediction_created_at TEXT,
            source TEXT,

            PRIMARY KEY (
                timestamp_utc,
                Region,
                model_name,
                horizon_hours
            )
        );
        """
    )


def clear_tables(conn):
    conn.execute(f"DELETE FROM {PREDICTION_TABLE};")
    conn.execute(f"DELETE FROM {LOAD_TABLE};")
    conn.commit()


def upsert_load_hourly(conn, df):
    if df.empty:
        return 0

    rows = df.to_dict(orient="records")

    sql = f"""
        INSERT INTO {LOAD_TABLE} (
            timestamp_utc,
            timestamp_local,
            data_date,
            local_date,
            Region,

            region_demand_mw,
            region_demand_forecast_mw,
            target_region_demand_mw,
            model_target_mw,

            time_idx,

            local_year,
            local_month,
            local_day,
            local_hour,
            local_day_of_week,
            local_day_of_year,

            hour_sin,
            hour_cos,
            day_of_week_sin,
            day_of_week_cos,
            month_sin,
            month_cos,
            day_of_year_sin,
            day_of_year_cos,

            is_weekend,
            is_holiday,
            is_day_before_holiday,
            is_day_after_holiday,
            is_holiday_period,

            source,
            created_at,
            updated_at
        )
        VALUES (
            :timestamp_utc,
            :timestamp_local,
            :data_date,
            :local_date,
            :Region,

            :region_demand_mw,
            :region_demand_forecast_mw,
            :target_region_demand_mw,
            :model_target_mw,

            :time_idx,

            :local_year,
            :local_month,
            :local_day,
            :local_hour,
            :local_day_of_week,
            :local_day_of_year,

            :hour_sin,
            :hour_cos,
            :day_of_week_sin,
            :day_of_week_cos,
            :month_sin,
            :month_cos,
            :day_of_year_sin,
            :day_of_year_cos,

            :is_weekend,
            :is_holiday,
            :is_day_before_holiday,
            :is_day_after_holiday,
            :is_holiday_period,

            :source,
            :created_at,
            :updated_at
        )
        ON CONFLICT(timestamp_utc, Region) DO UPDATE SET
            timestamp_local = excluded.timestamp_local,
            data_date = excluded.data_date,
            local_date = excluded.local_date,

            region_demand_mw = excluded.region_demand_mw,
            region_demand_forecast_mw = excluded.region_demand_forecast_mw,
            target_region_demand_mw = excluded.target_region_demand_mw,
            model_target_mw = excluded.model_target_mw,

            time_idx = excluded.time_idx,

            local_year = excluded.local_year,
            local_month = excluded.local_month,
            local_day = excluded.local_day,
            local_hour = excluded.local_hour,
            local_day_of_week = excluded.local_day_of_week,
            local_day_of_year = excluded.local_day_of_year,

            hour_sin = excluded.hour_sin,
            hour_cos = excluded.hour_cos,
            day_of_week_sin = excluded.day_of_week_sin,
            day_of_week_cos = excluded.day_of_week_cos,
            month_sin = excluded.month_sin,
            month_cos = excluded.month_cos,
            day_of_year_sin = excluded.day_of_year_sin,
            day_of_year_cos = excluded.day_of_year_cos,

            is_weekend = excluded.is_weekend,
            is_holiday = excluded.is_holiday,
            is_day_before_holiday = excluded.is_day_before_holiday,
            is_day_after_holiday = excluded.is_day_after_holiday,
            is_holiday_period = excluded.is_holiday_period,

            source = excluded.source,
            updated_at = excluded.updated_at;
    """

    conn.executemany(sql, rows)
    conn.commit()

    return len(rows)


def upsert_model_predictions(conn, df):
    if df.empty:
        return 0

    rows = df.to_dict(orient="records")

    sql = f"""
        INSERT INTO {PREDICTION_TABLE} (
            timestamp_utc,
            timestamp_local,
            Region,

            model_name,
            horizon_hours,

            prediction_mw,
            actual_mw,
            absolute_error_mw,
            ape,

            prediction_created_at,
            source
        )
        VALUES (
            :timestamp_utc,
            :timestamp_local,
            :Region,

            :model_name,
            :horizon_hours,

            :prediction_mw,
            :actual_mw,
            :absolute_error_mw,
            :ape,

            :prediction_created_at,
            :source
        )
        ON CONFLICT(
            timestamp_utc,
            Region,
            model_name,
            horizon_hours
        ) DO UPDATE SET
            timestamp_local = excluded.timestamp_local,
            prediction_mw = excluded.prediction_mw,
            actual_mw = excluded.actual_mw,
            absolute_error_mw = excluded.absolute_error_mw,
            ape = excluded.ape,
            prediction_created_at = excluded.prediction_created_at,
            source = excluded.source;
    """

    conn.executemany(sql, rows)
    conn.commit()

    return len(rows)


def read_load_hourly(conn, region=None):
    query = f"SELECT * FROM {LOAD_TABLE}"

    params = []

    if region is not None:
        query += " WHERE Region = ?"
        params.append(region)

    query += " ORDER BY timestamp_utc"

    df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce"
        )

        # Rebuild local timestamp from UTC to avoid mixed timezone parsing issues.
        df["timestamp_local"] = (
            df["timestamp_utc"]
            .dt.tz_convert("America/Los_Angeles")
        )

        df["data_date"] = pd.to_datetime(
            df["data_date"],
            errors="coerce"
        )

        df["local_date"] = pd.to_datetime(
            df["local_date"],
            errors="coerce"
        )

    return df


def read_model_predictions(conn, region=None, model_name=None):
    query = f"SELECT * FROM {PREDICTION_TABLE}"

    conditions = []
    params = []

    if region is not None:
        conditions.append("Region = ?")
        params.append(region)

    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp_utc"

    df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce"
        )

        df["timestamp_local"] = (
            df["timestamp_utc"]
            .dt.tz_convert("America/Los_Angeles")
        )

        df["prediction_created_at"] = pd.to_datetime(
            df["prediction_created_at"],
            errors="coerce"
        )

    return df


def get_table_counts(conn):
    load_count = conn.execute(
        f"SELECT COUNT(*) FROM {LOAD_TABLE};"
    ).fetchone()[0]

    prediction_count = conn.execute(
        f"SELECT COUNT(*) FROM {PREDICTION_TABLE};"
    ).fetchone()[0]

    return {
        LOAD_TABLE: load_count,
        PREDICTION_TABLE: prediction_count,
    }


def get_database_summary(conn):
    query = f"""
        SELECT
            MIN(timestamp_utc) AS min_timestamp_utc,
            MAX(timestamp_utc) AS max_timestamp_utc,
            COUNT(*) AS row_count
        FROM {LOAD_TABLE};
    """

    return pd.read_sql_query(query, conn)