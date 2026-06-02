from pathlib import Path
import pandas as pd


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
MODEL_DIR = BASE_DIR / "models"

CSV_PATH = DATA_DIR / "all_features_all_timerange.csv"
DB_PATH = DATABASE_DIR / "load_forecast.db"
MODEL_PATH = MODEL_DIR / "TFT_No_Weather.ckpt"


# ============================================================
# Region and Model Configuration
# ============================================================

REGION_NAME = "CAL"

TARGET_COLUMN = "model_target_mw"
ACTUAL_COLUMN = "region_demand_mw"
EIA_FORECAST_COLUMN = "region_demand_forecast_mw"
TRAINING_TARGET_COLUMN = "target_region_demand_mw"

MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 1

TRAIN_END_DATE = pd.Timestamp("2024-01-01")
VALID_END_DATE = pd.Timestamp("2025-01-01")
TEST_END_DATE = pd.Timestamp("2026-05-07")

GROUP_IDS = ["Region"]
STATIC_CATEGORICALS = ["Region"]

BATCH_SIZE = 128
NUM_WORKERS = 0


# ============================================================
# TFT No-Weather Feature Configuration
# ============================================================

BASE_FEATURES = [
    "time_idx",
]

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

HOLIDAY_FEATURES = [
    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_period",
]

NO_WEATHER_KNOWN_REALS = (
    BASE_FEATURES
    + CALENDAR_FEATURES
    + HOLIDAY_FEATURES
)


# ============================================================
# Database Table Names
# ============================================================

LOAD_TABLE = "load_hourly"
PREDICTION_TABLE = "model_predictions"


# ============================================================
# Columns to Store in load_hourly
# ============================================================

LOAD_TABLE_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "data_date",
    "local_date",
    "Region",

    "region_demand_mw",
    "region_demand_forecast_mw",
    "target_region_demand_mw",
    "model_target_mw",

    "time_idx",

    "local_year",
    "local_month",
    "local_day",
    "local_hour",
    "local_day_of_week",
    "local_day_of_year",

    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",

    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_period",

    "source",
    "created_at",
    "updated_at",
]

# ============================================================
# EIA API Configuration
# ============================================================

EIA_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

EIA_RESPONDENT = "CAL"

EIA_TYPES = [
    "D",   # Demand
    "DF",  # Demand forecast
]

EIA_DEFAULT_LENGTH = 5000

LOCAL_TIMEZONE = "America/Los_Angeles"