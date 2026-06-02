import warnings
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import (
    DB_PATH,
    MODEL_PATH,
    REGION_NAME,
    TARGET_COLUMN,
    ACTUAL_COLUMN,
    EIA_FORECAST_COLUMN,
    TRAINING_TARGET_COLUMN,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    TRAIN_END_DATE,
    VALID_END_DATE,
    TEST_END_DATE,
    GROUP_IDS,
    STATIC_CATEGORICALS,
    BATCH_SIZE,
    NUM_WORKERS,
    NO_WEATHER_KNOWN_REALS,
    EIA_RESPONDENT,
)

from db_utils import (
    get_connection,
    create_tables,
    read_load_hourly,
    read_model_predictions,
    get_table_counts,
    upsert_model_predictions,
    upsert_load_hourly,
)

from eia_client import fetch_eia_region_data
from preprocessing import preprocess_eia_region_data

warnings.filterwarnings("ignore")


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="CAL Load Forecasting Dashboard",
    layout="wide"
)


# ============================================================
# Utility Functions
# ============================================================

def check_required_files():
    missing_items = []

    if not DB_PATH.exists():
        missing_items.append(str(DB_PATH))

    if not MODEL_PATH.exists():
        missing_items.append(str(MODEL_PATH))

    return missing_items


@st.cache_data
def load_and_prepare_data():
    conn = get_connection()
    create_tables(conn)

    df = read_load_hourly(
        conn=conn,
        region=REGION_NAME
    )

    conn.close()

    if df.empty:
        st.error(
            "Database is empty. Please run `python seed_database.py` first."
        )
        st.stop()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

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

    df = df[df["Region"].astype(str) == REGION_NAME].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    numeric_cols = [
        ACTUAL_COLUMN,
        EIA_FORECAST_COLUMN,
        TRAINING_TARGET_COLUMN,
        TARGET_COLUMN,
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
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    return df


def get_existing_known_reals(df):
    return [col for col in NO_WEATHER_KNOWN_REALS if col in df.columns]


def historical_average_baseline(df, selected_timestamp):
    selected_timestamp = pd.Timestamp(selected_timestamp)

    selected_month = selected_timestamp.month
    selected_hour = selected_timestamp.hour

    matched_data = df[
        (df["local_month"] == selected_month)
        & (df["local_hour"] == selected_hour)
    ].copy()

    if matched_data.empty:
        matched_data = df[df["local_hour"] == selected_hour].copy()

    if matched_data.empty:
        return df[TARGET_COLUMN].mean()

    return matched_data[TARGET_COLUMN].mean()


def evaluate_prediction(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    nonzero_mask = y_true != 0

    if nonzero_mask.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask])
                    / y_true[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot != 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

def build_prediction_records(
    result_df,
    model_name="TFT_No_Weather",
    horizon_hours=1,
    source="streamlit_app",
):
    """
    Convert TFT prediction result dataframe into records
    compatible with the model_predictions database table.
    """
    if result_df is None or result_df.empty:
        return pd.DataFrame()

    prediction_df = result_df.copy()

    required_cols = [
        "timestamp_utc",
        "timestamp_local",
        "Region",
        "tft_prediction_mw",
        "actual_mw",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in prediction_df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns for prediction records: "
            + ", ".join(missing_cols)
        )

    prediction_records = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            prediction_df["timestamp_utc"],
            utc=True,
            errors="coerce"
        ).astype(str),

        "timestamp_local": pd.to_datetime(
            prediction_df["timestamp_local"],
            errors="coerce"
        ).astype(str),

        "Region": prediction_df["Region"].astype(str),

        "model_name": model_name,
        "horizon_hours": horizon_hours,

        "prediction_mw": pd.to_numeric(
            prediction_df["tft_prediction_mw"],
            errors="coerce"
        ),

        "actual_mw": pd.to_numeric(
            prediction_df["actual_mw"],
            errors="coerce"
        ),
    })

    prediction_records["absolute_error_mw"] = (
        prediction_records["actual_mw"]
        - prediction_records["prediction_mw"]
    ).abs()

    prediction_records["ape"] = np.where(
        prediction_records["actual_mw"] != 0,
        prediction_records["absolute_error_mw"]
        / prediction_records["actual_mw"]
        * 100,
        np.nan
    )

    prediction_records["prediction_created_at"] = (
        pd.Timestamp.utcnow().isoformat()
    )

    prediction_records["source"] = source

    prediction_records = prediction_records.where(
        pd.notnull(prediction_records),
        None
    )

    return prediction_records

def save_prediction_records_to_db(prediction_records):
    """
    Save prediction records to the model_predictions table.
    """
    if prediction_records is None or prediction_records.empty:
        return 0

    conn = get_connection()
    create_tables(conn)

    saved_rows = upsert_model_predictions(
        conn=conn,
        df=prediction_records
    )

    conn.close()

    # Clear Streamlit cache so database status can refresh
    load_and_prepare_data.clear()

    return saved_rows

def load_predictions_from_db():
    conn = get_connection()
    create_tables(conn)

    predictions_df = read_model_predictions(
        conn=conn,
        region=REGION_NAME,
        model_name="TFT_No_Weather"
    )

    conn.close()

    if predictions_df.empty:
        return predictions_df

    predictions_df["timestamp_utc"] = pd.to_datetime(
        predictions_df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    predictions_df["timestamp_local"] = (
        predictions_df["timestamp_utc"]
        .dt.tz_convert("America/Los_Angeles")
    )

    numeric_cols = [
        "prediction_mw",
        "actual_mw",
        "absolute_error_mw",
        "ape",
        "horizon_hours",
    ]

    for col in numeric_cols:
        if col in predictions_df.columns:
            predictions_df[col] = pd.to_numeric(
                predictions_df[col],
                errors="coerce"
            )

    return predictions_df

def build_monitoring_dataframe(load_df, predictions_df):
    if predictions_df.empty:
        monitoring_df = load_df.copy()
        monitoring_df["tft_prediction_mw"] = np.nan
        monitoring_df["tft_absolute_error_mw"] = np.nan
        monitoring_df["tft_ape"] = np.nan
        return monitoring_df

    pred_cols = [
        "timestamp_utc",
        "Region",
        "prediction_mw",
        "absolute_error_mw",
        "ape",
    ]

    pred_df = predictions_df[pred_cols].copy()

    pred_df = pred_df.rename(columns={
        "prediction_mw": "tft_prediction_mw",
        "absolute_error_mw": "tft_absolute_error_mw",
        "ape": "tft_ape",
    })

    monitoring_df = load_df.merge(
        pred_df,
        on=["timestamp_utc", "Region"],
        how="left"
    )

    return monitoring_df

def calculate_monitoring_metrics(monitoring_df):
    eval_df = monitoring_df.dropna(
        subset=[ACTUAL_COLUMN, "tft_prediction_mw"]
    ).copy()

    if eval_df.empty:
        return None

    return evaluate_prediction(
        eval_df[ACTUAL_COLUMN],
        eval_df["tft_prediction_mw"]
    )

def get_app_eia_api_key():
    """
    Read EIA API key from Streamlit secrets or environment variable.
    """
    import os

    try:
        if "EIA_API_KEY" in st.secrets:
            return st.secrets["EIA_API_KEY"]
    except Exception:
        pass

    return os.getenv("EIA_API_KEY")

def fetch_and_store_latest_eia_data(load_df, lookback_days=7):
    """
    Fetch latest EIA data, preprocess it, and upsert it into load_hourly.
    """
    api_key = get_app_eia_api_key()

    if not api_key:
        raise ValueError(
            "EIA_API_KEY is missing. Set it as an environment variable or Streamlit secret."
        )

    historical_start = load_df["timestamp_utc"].min()

    latest_timestamp = load_df["timestamp_utc"].max()

    api_start_timestamp = latest_timestamp - pd.Timedelta(days=lookback_days)

    api_start = api_start_timestamp.strftime("%Y-%m-%dT%H")

    raw_api_df = fetch_eia_region_data(
        api_key=api_key,
        respondent=EIA_RESPONDENT,
        start=api_start,
        length=5000,
    )

    processed_api_df = preprocess_eia_region_data(
        raw_df=raw_api_df,
        historical_start_timestamp=historical_start,
    )

    if processed_api_df.empty:
        return {
            "raw_rows": len(raw_api_df),
            "processed_rows": 0,
            "saved_rows": 0,
        }

    conn = get_connection()
    create_tables(conn)

    saved_rows = upsert_load_hourly(
        conn=conn,
        df=processed_api_df
    )

    conn.close()

    load_and_prepare_data.clear()

    return {
        "raw_rows": len(raw_api_df),
        "processed_rows": len(processed_api_df),
        "saved_rows": saved_rows,
    }

def extract_actuals_from_dataloader(dataloader):
    import torch

    actuals = []

    for x, y in dataloader:
        if isinstance(y, tuple):
            y = y[0]
        actuals.append(y.detach().cpu())

    return torch.cat(actuals, dim=0)

def get_latest_unpredicted_prediction_range(
    load_df,
    predictions_df,
    max_rows=24,
    model_name="TFT_No_Weather",
):
    """
    Select the latest rows that have enough encoder context and do not yet
    have saved TFT predictions in the model_predictions table.

    Returns:
        start_time_idx, end_time_idx, selected_df
    """
    df_sorted = load_df.sort_values("time_idx").copy()

    if df_sorted.empty:
        return None, None, pd.DataFrame()

    min_allowed_time_idx = df_sorted["time_idx"].min() + MAX_ENCODER_LENGTH

    eligible_df = df_sorted[
        df_sorted["time_idx"] >= min_allowed_time_idx
    ].copy()

    if eligible_df.empty:
        return None, None, pd.DataFrame()

    eligible_df["timestamp_utc"] = pd.to_datetime(
        eligible_df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    # Only compare against saved predictions from the same model and horizon.
    if predictions_df is not None and not predictions_df.empty:
        pred_df = predictions_df.copy()

        if "model_name" in pred_df.columns:
            pred_df = pred_df[pred_df["model_name"] == model_name].copy()

        if "horizon_hours" in pred_df.columns:
            pred_df = pred_df[pd.to_numeric(pred_df["horizon_hours"], errors="coerce") == 1].copy()

        pred_df["timestamp_utc"] = pd.to_datetime(
            pred_df["timestamp_utc"],
            utc=True,
            errors="coerce"
        )

        predicted_timestamps = set(pred_df["timestamp_utc"].dropna())
    else:
        predicted_timestamps = set()

    unpredicted_df = eligible_df[
        ~eligible_df["timestamp_utc"].isin(predicted_timestamps)
    ].copy()

    if unpredicted_df.empty:
        return None, None, pd.DataFrame()

    selected_df = unpredicted_df.tail(max_rows).copy()

    start_time_idx = int(selected_df["time_idx"].min())
    end_time_idx = int(selected_df["time_idx"].max())

    return start_time_idx, end_time_idx, selected_df


def plot_tft_prediction_segments(
    ax,
    tft_chart_df,
    time_col="timestamp_local",
    prediction_col="tft_prediction_mw",
    max_gap_hours=1.5,
):
    """
    Plot TFT predictions as line segments without connecting points across
    large time gaps. This prevents misleading diagonal lines when predictions
    are sparse.
    """
    if tft_chart_df is None or tft_chart_df.empty:
        return

    plot_df = tft_chart_df.sort_values(time_col).copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col, prediction_col]).copy()

    if plot_df.empty:
        return

    time_diff_hours = plot_df[time_col].diff().dt.total_seconds().div(3600)
    plot_df["segment_id"] = (time_diff_hours > max_gap_hours).cumsum()

    first_segment = True

    for _, segment_df in plot_df.groupby("segment_id"):
        label = "TFT Prediction" if first_segment else None

        if len(segment_df) == 1:
            ax.scatter(
                segment_df[time_col],
                segment_df[prediction_col],
                marker="x",
                s=60,
                label=label,
                zorder=5
            )
        else:
            ax.plot(
                segment_df[time_col],
                segment_df[prediction_col],
                linestyle="-",
                label=label
            )

        first_segment = False

# ============================================================
# TFT Functions
# ============================================================

@st.cache_resource
def load_tft_dependencies_and_model(model_path_str):
    try:
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer

        model = TemporalFusionTransformer.load_from_checkpoint(model_path_str)
        model.eval()

        return {
            "TimeSeriesDataSet": TimeSeriesDataSet,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "GroupNormalizer": GroupNormalizer,
            "model": model,
        }

    except ImportError as e:
        st.error(
            "TFT dependencies are not installed. Install PyTorch and "
            "PyTorch Forecasting before using the TFT model."
        )
        st.exception(e)
        st.stop()

    except Exception as e:
        st.error("Failed to load TFT checkpoint.")
        st.exception(e)
        st.stop()


@st.cache_resource
def build_training_dataset(df_serialized, known_reals_tuple):
    deps = load_tft_dependencies_and_model(str(MODEL_PATH))

    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]

    known_reals = list(known_reals_tuple)

    df = pd.read_json(StringIO(df_serialized), orient="split")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["data_date"] = pd.to_datetime(df["data_date"])
    df["Region"] = df["Region"].astype(str)

    training_df = df[df["data_date"] < TRAIN_END_DATE].copy()

    training = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=GROUP_IDS,

        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        static_categoricals=STATIC_CATEGORICALS,

        time_varying_known_reals=known_reals,

        time_varying_unknown_reals=[
            TARGET_COLUMN
        ],

        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation="softplus"
        ),

        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training


def predict_tft_for_range(df, training_dataset, start_time_idx, end_time_idx):
    deps = load_tft_dependencies_and_model(str(MODEL_PATH))
    model = deps["model"]

    prediction_df = df[
        df["time_idx"] <= end_time_idx
    ].copy()

    prediction_dataset = type(training_dataset).from_dataset(
        training_dataset,
        prediction_df,
        min_prediction_idx=start_time_idx,
        stop_randomization=True
    )

    prediction_dataloader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    predictions = model.predict(prediction_dataloader)

    if hasattr(predictions, "detach"):
        predictions_np = predictions.detach().cpu().numpy().reshape(-1)
    else:
        predictions_np = np.asarray(predictions).reshape(-1)

    actuals = extract_actuals_from_dataloader(prediction_dataloader)

    if hasattr(actuals, "detach"):
        actuals_np = actuals.detach().cpu().numpy().reshape(-1)
    else:
        actuals_np = np.asarray(actuals).reshape(-1)

    target_rows = df[
        (df["time_idx"] >= start_time_idx)
        & (df["time_idx"] <= end_time_idx)
    ].copy()

    min_len = min(len(target_rows), len(predictions_np), len(actuals_np))

    result = target_rows.iloc[:min_len].copy()
    result["tft_prediction_mw"] = predictions_np[:min_len]
    result["actual_mw"] = actuals_np[:min_len]
    result["tft_error_mw"] = result["actual_mw"] - result["tft_prediction_mw"]
    result["tft_absolute_error_mw"] = result["tft_error_mw"].abs()

    return result


# ============================================================
# Main App
# ============================================================

st.title("CAL Electricity Load Forecasting Dashboard")

st.write(
    "This dashboard follows the final TFT No-Weather experiment setup for "
    "CAL electricity load forecasting. It compares actual load, a historical "
    "average baseline, and the trained TFT No-Weather model when the checkpoint "
    "is available."
)

missing_items = check_required_files()

if str(DB_PATH) in missing_items:
    st.error(
        f"Database file not found: `{DB_PATH}`. "
        "Please run `python seed_database.py` first."
    )
    st.stop()

df = load_and_prepare_data()

known_reals = get_existing_known_reals(df)

missing_feature_cols = [
    col for col in NO_WEATHER_KNOWN_REALS
    if col not in df.columns
]

with st.expander("Database Status", expanded=False):
    conn = get_connection()
    create_tables(conn)

    counts = get_table_counts(conn)
    predictions_df_preview = read_model_predictions(
        conn=conn,
        region=REGION_NAME
    )

    conn.close()

    st.write("**Database path:**")
    st.code(str(DB_PATH))

    st.write("**Table row counts:**")
    st.json(counts)

    st.write("**Prediction rows loaded:**", len(predictions_df_preview))

    if not predictions_df_preview.empty:
        st.write("**Prediction models in database:**")
        st.dataframe(
            predictions_df_preview["model_name"]
            .value_counts()
            .reset_index()
            .rename(columns={
                "model_name": "model_name",
                "count": "row_count"
            }),
            use_container_width=True
        )

with st.expander("Model Configuration", expanded=False):
    st.write("**Region:**", REGION_NAME)
    st.write("**Target column:**", TARGET_COLUMN)
    st.write("**Max encoder length:**", MAX_ENCODER_LENGTH)
    st.write("**Max prediction length:**", MAX_PREDICTION_LENGTH)
    st.write("**Known real features used:**")
    st.code("\n".join(known_reals))

    if missing_feature_cols:
        st.warning(
            "Some expected no-weather features are missing from the database. "
            "The app will use only the available features."
        )
        st.code("\n".join(missing_feature_cols))


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Dashboard Settings")

mode = st.sidebar.radio(
    "Prediction Mode",
    [
        "Live Monitoring",
        "Single Historical Prediction",
        "Recent Test Backtest",
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("Model setup: **TFT No-Weather**")
st.sidebar.write("Region: **CAL**")
st.sidebar.write("Data source: **SQLite Database**")

model_available = MODEL_PATH.exists()

if model_available:
    st.sidebar.success("TFT checkpoint found")
else:
    st.sidebar.warning("TFT checkpoint not found")


# ============================================================
# Top Dataset Summary
# ============================================================

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Region", REGION_NAME)
col2.metric("Rows", f"{len(df):,}")
col3.metric("Start Date", str(df["timestamp_local"].min().date()))
col4.metric("End Date", str(df["timestamp_local"].max().date()))

latest_actual = df[ACTUAL_COLUMN].dropna().iloc[-1]

st.metric(
    "Latest Actual Load",
    f"{latest_actual:,.2f} MW"
)


# ============================================================
# Single Historical Prediction Mode
# ============================================================

# ============================================================
# Live Monitoring Mode
# ============================================================

if mode == "Live Monitoring":
    latest_prediction_rows = st.sidebar.selectbox(
        "TFT Prediction Update Rows",
        [24, 48, 72, 168],
        index=0
    )
        
    st.subheader("Live Monitoring")

    st.write(
        "This page monitors CAL electricity load using the database as the main data source. "
        "It combines actual demand data with TFT No-Weather predictions that have already "
        "been saved into the database."
    )

    st.markdown("### EIA API Update")

    update_col1, update_col2 = st.columns([1, 2])

    with update_col1:
        fetch_button = st.button(
            "Fetch Latest EIA Data",
            use_container_width=True
        )

    with update_col2:
        st.caption(
            "This button manually retrieves the latest CAL actual demand and "
            "EIA demand forecast from the EIA API, then stores the processed data "
            "into the SQLite database."
        )

    if fetch_button:
        with st.spinner("Fetching latest EIA data and updating database..."):
            try:
                update_result = fetch_and_store_latest_eia_data(
                    load_df=df,
                    lookback_days=7
                )

                st.success(
                    "EIA data update completed. "
                    f"Raw rows: {update_result['raw_rows']:,}, "
                    f"processed rows: {update_result['processed_rows']:,}, "
                    f"saved rows: {update_result['saved_rows']:,}."
                )

                st.rerun()

            except Exception as e:
                st.error("Failed to fetch or store EIA data.")
                st.exception(e)

    st.markdown("### TFT Prediction Update")

    prediction_col1, prediction_col2 = st.columns([1, 2])

    with prediction_col1:
        run_latest_prediction_button = st.button(
            "Run TFT Prediction for Latest Data",
            use_container_width=True
        )
    
    if run_latest_prediction_button:
        if not model_available:
            st.error(
                f"TFT checkpoint not found at `{MODEL_PATH}`. "
                "Cannot run latest TFT prediction."
            )
        else:
            current_predictions_df = load_predictions_from_db()

            start_time_idx, end_time_idx, selected_prediction_rows_df = (
                get_latest_unpredicted_prediction_range(
                    load_df=df,
                    predictions_df=current_predictions_df,
                    max_rows=latest_prediction_rows,
                    model_name="TFT_No_Weather",
                )
            )

            if start_time_idx is None:
                st.info(
                    "The latest eligible rows already have saved TFT predictions. "
                    "Fetch newer EIA data first or increase the monitoring window if needed."
                )
            else:
                selected_start = selected_prediction_rows_df["timestamp_local"].min()
                selected_end = selected_prediction_rows_df["timestamp_local"].max()

                st.info(
                    "Running TFT prediction for unpredicted latest rows from "
                    f"{selected_start} to {selected_end}."
                )

                with st.spinner("Running TFT prediction for latest unpredicted database rows..."):
                    df_serialized = df.to_json(orient="split", date_format="iso")

                    training_dataset = build_training_dataset(
                        df_serialized,
                        tuple(known_reals)
                    )

                    latest_result_df = predict_tft_for_range(
                        df=df,
                        training_dataset=training_dataset,
                        start_time_idx=start_time_idx,
                        end_time_idx=end_time_idx
                    )

                    prediction_records = build_prediction_records(
                        result_df=latest_result_df,
                        model_name="TFT_No_Weather",
                        horizon_hours=1,
                        source="live_monitoring_latest_prediction"
                    )

                    saved_rows = save_prediction_records_to_db(prediction_records)

                st.success(
                    f"Saved {saved_rows:,} latest TFT prediction record(s) to the database."
                )

                st.rerun()

    with prediction_col2:
        st.caption(
            "This button runs the TFT No-Weather model for the latest available rows "
            "in the database and stores the predictions into the model_predictions table."
        )

    predictions_df = load_predictions_from_db()

    monitoring_df = build_monitoring_dataframe(
        load_df=df,
        predictions_df=predictions_df
    )

    latest_row = monitoring_df.sort_values("timestamp_utc").iloc[-1]

    latest_actual = latest_row[ACTUAL_COLUMN]

    latest_eia_forecast = (
        latest_row[EIA_FORECAST_COLUMN]
        if EIA_FORECAST_COLUMN in monitoring_df.columns
        else np.nan
    )

    latest_prediction_df = predictions_df.dropna(
        subset=["prediction_mw"]
    ).sort_values("timestamp_utc") if not predictions_df.empty else pd.DataFrame()

    if not latest_prediction_df.empty:
        latest_prediction_row = latest_prediction_df.iloc[-1]
        latest_tft_prediction = latest_prediction_row["prediction_mw"]
        latest_prediction_time = latest_prediction_row["timestamp_local"]
    else:
        latest_tft_prediction = np.nan
        latest_prediction_time = None

    st.markdown("### Latest Status")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Actual Load",
        f"{latest_actual:,.2f} MW"
        if pd.notnull(latest_actual)
        else "N/A"
    )

    col2.metric(
        "Latest EIA Forecast",
        f"{latest_eia_forecast:,.2f} MW"
        if pd.notnull(latest_eia_forecast)
        else "N/A"
    )

    col3.metric(
        "Latest Saved TFT Prediction",
        f"{latest_tft_prediction:,.2f} MW"
        if pd.notnull(latest_tft_prediction)
        else "N/A"
    )

    col4.metric(
        "Last Data Timestamp",
        str(latest_row["timestamp_local"])
    )

    latest_data_time = latest_row["timestamp_local"]

    if latest_prediction_time is not None:
        st.caption(f"Latest data timestamp: {latest_data_time}")
        st.caption(f"Latest saved TFT prediction timestamp: {latest_prediction_time}")

        if latest_prediction_time < latest_data_time:
            st.warning(
                "Latest load/EIA data is newer than the saved TFT predictions. "
                "Run TFT Prediction for Latest Data to synchronize the monitoring dashboard."
            )
    else:
        st.info(
            "No TFT prediction has been saved yet. "
            "Run Single Historical Prediction, Recent Test Backtest, or Run TFT Prediction for Latest Data first."
        )

    st.markdown("### Evaluation Metrics from Saved TFT Predictions")

    metrics = calculate_monitoring_metrics(monitoring_df)

    if metrics is None:
        st.warning(
            "No saved TFT predictions with matching actual values are available yet."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("MAE", f"{metrics['MAE']:,.2f} MW")
        m2.metric("RMSE", f"{metrics['RMSE']:,.2f} MW")
        m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        m4.metric("R²", f"{metrics['R2']:.4f}")

    st.markdown("### Main Monitoring Chart")

    window_options = {
        "Last 24 hours": 24,
        "Last 3 days": 72,
        "Last 7 days": 168,
        "Last 14 days": 336,
        "Last 30 days": 720,
    }

    selected_window_label = st.sidebar.selectbox(
        "Monitoring Window",
        list(window_options.keys()),
        index=2
    )

    selected_window = window_options[selected_window_label]

    chart_df = monitoring_df.sort_values("timestamp_utc").tail(selected_window).copy()

    tft_points_in_window = int(chart_df["tft_prediction_mw"].notna().sum())
    total_points_in_window = int(len(chart_df))
    tft_coverage_pct = (
        tft_points_in_window / total_points_in_window * 100
        if total_points_in_window > 0
        else 0
    )

    coverage_col1, coverage_col2 = st.columns(2)

    coverage_col1.metric(
        "TFT Prediction Coverage in Window",
        f"{tft_points_in_window:,} / {total_points_in_window:,} rows"
    )

    coverage_col2.metric(
        "TFT Coverage Percentage",
        f"{tft_coverage_pct:.1f}%"
    )

    if tft_points_in_window == 0:
        st.warning(
            "No saved TFT predictions are available within the selected monitoring window. "
            "Run TFT Prediction for Latest Data or select a longer monitoring window."
        )
    elif tft_coverage_pct < 50:
        st.info(
            "TFT predictions cover only part of the selected monitoring window. "
            "The chart will avoid connecting prediction segments across large time gaps."
        )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        chart_df["timestamp_local"],
        chart_df[ACTUAL_COLUMN],
        label="Actual Load"
    )

    if EIA_FORECAST_COLUMN in chart_df.columns:
        eia_chart_df = chart_df.dropna(subset=[EIA_FORECAST_COLUMN])

        if not eia_chart_df.empty:
            ax.plot(
                eia_chart_df["timestamp_local"],
                eia_chart_df[EIA_FORECAST_COLUMN],
                linestyle="--",
                label="EIA Forecast"
            )

    tft_chart_df = chart_df.dropna(subset=["tft_prediction_mw"]).copy()

    if not tft_chart_df.empty:
        plot_tft_prediction_segments(
            ax=ax,
            tft_chart_df=tft_chart_df,
            time_col="timestamp_local",
            prediction_col="tft_prediction_mw",
            max_gap_hours=1.5,
        )

    ax.set_title("CAL Actual Load, EIA Forecast, and Saved TFT Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.legend()

    st.pyplot(fig)

    st.markdown("### Saved Prediction Records")

    if predictions_df.empty:
        st.info("No prediction records are currently stored in the database.")
    else:
        display_prediction_df = predictions_df.sort_values(
            "timestamp_utc",
            ascending=False
        ).copy()

        display_cols = [
            "timestamp_local",
            "model_name",
            "horizon_hours",
            "prediction_mw",
            "actual_mw",
            "absolute_error_mw",
            "ape",
            "source",
            "prediction_created_at",
        ]

        display_cols = [
            col for col in display_cols
            if col in display_prediction_df.columns
        ]

        st.dataframe(
            display_prediction_df[display_cols].head(100),
            use_container_width=True
        )

elif mode == "Single Historical Prediction":
    st.subheader("Single Historical Prediction")

    st.write(
        "Select one timestamp from the available dataset range. "
        "The dashboard will compare actual load, historical average baseline, "
        "and TFT No-Weather prediction if the model checkpoint is available."
    )

    min_date = df["timestamp_local"].min().date()
    max_date = df["timestamp_local"].max().date()

    selected_date = st.sidebar.date_input(
        "Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    time_options = {
        "12 AM": 0,
        "1 AM": 1,
        "2 AM": 2,
        "3 AM": 3,
        "4 AM": 4,
        "5 AM": 5,
        "6 AM": 6,
        "7 AM": 7,
        "8 AM": 8,
        "9 AM": 9,
        "10 AM": 10,
        "11 AM": 11,
        "12 PM": 12,
        "1 PM": 13,
        "2 PM": 14,
        "3 PM": 15,
        "4 PM": 16,
        "5 PM": 17,
        "6 PM": 18,
        "7 PM": 19,
        "8 PM": 20,
        "9 PM": 21,
        "10 PM": 22,
        "11 PM": 23,
    }

    selected_time_label = st.sidebar.selectbox(
        "Hour",
        list(time_options.keys()),
        index=12
    )

    selected_hour = time_options[selected_time_label]

    candidate_rows = df[
        (df["timestamp_local"].dt.date == selected_date)
        & (df["timestamp_local"].dt.hour == selected_hour)
    ].copy()

    if candidate_rows.empty:
        st.warning(
            "Selected timestamp is not available in the database. "
            "Please choose another date or hour."
        )
    else:
        selected_row = candidate_rows.iloc[0]
        selected_time_idx = int(selected_row["time_idx"])

        if selected_time_idx < df["time_idx"].min() + MAX_ENCODER_LENGTH:
            st.warning(
                "The selected timestamp does not have enough historical encoder context "
                f"({MAX_ENCODER_LENGTH} hours). Please select a later timestamp."
            )
        else:
            actual_value = float(selected_row[ACTUAL_COLUMN])

            baseline_prediction = historical_average_baseline(
                df,
                selected_row["timestamp_local"]
            )

            tft_pred_value = None

            col1, col2, col3 = st.columns(3)

            col1.metric("Actual Load", f"{actual_value:,.2f} MW")
            col2.metric("Historical Baseline", f"{baseline_prediction:,.2f} MW")
            col3.metric(
                "Baseline Absolute Error",
                f"{abs(actual_value - baseline_prediction):,.2f} MW"
            )

            tft_result = None

            if model_available:
                if st.button("Run TFT Prediction", use_container_width=True):
                    with st.spinner("Running TFT No-Weather prediction..."):
                        df_serialized = df.to_json(orient="split", date_format="iso")

                        training_dataset = build_training_dataset(
                            df_serialized,
                            tuple(known_reals)
                        )

                        tft_result = predict_tft_for_range(
                            df=df,
                            training_dataset=training_dataset,
                            start_time_idx=selected_time_idx,
                            end_time_idx=selected_time_idx
                        )

                    if tft_result is not None and not tft_result.empty:
                        pred_value = float(tft_result["tft_prediction_mw"].iloc[0])
                        tft_pred_value = pred_value
                        tft_error = actual_value - pred_value

                        st.subheader("TFT No-Weather Result")

                        c1, c2, c3 = st.columns(3)

                        c1.metric("TFT Prediction", f"{pred_value:,.2f} MW")
                        c2.metric("TFT Error", f"{tft_error:,.2f} MW")
                        c3.metric("TFT Absolute Error", f"{abs(tft_error):,.2f} MW")

                        comparison_df = pd.DataFrame([
                            {
                                "Method": "Actual",
                                "Load_MW": actual_value,
                                "Absolute_Error_MW": 0.0,
                            },
                            {
                                "Method": "Historical Average Baseline",
                                "Load_MW": baseline_prediction,
                                "Absolute_Error_MW": abs(actual_value - baseline_prediction),
                            },
                            {
                                "Method": "TFT No-Weather",
                                "Load_MW": pred_value,
                                "Absolute_Error_MW": abs(tft_error),
                            },
                        ])

                        st.dataframe(comparison_df, use_container_width=True)

                        prediction_records = build_prediction_records(
                            result_df=tft_result,
                            model_name="TFT_No_Weather",
                            horizon_hours=1,
                            source="single_historical_prediction"
                        )

                        saved_rows = save_prediction_records_to_db(prediction_records)

                        st.success(
                            f"Saved {saved_rows:,} TFT prediction record(s) to the database."
                        )

                    else:
                        st.error("TFT prediction returned no result.")

            else:
                st.warning(
                    f"TFT checkpoint not found at `{MODEL_PATH}`. "
                    "The dashboard can still show the historical average baseline, "
                    "but TFT prediction is disabled."
                )

            st.subheader("Generated No-Weather Features for Selected Timestamp")

            feature_cols_to_show = [
                "timestamp_local",
                "time_idx",
                "Region",
                ACTUAL_COLUMN,
                TARGET_COLUMN,
            ] + known_reals

            feature_cols_to_show = list(dict.fromkeys(feature_cols_to_show))

            feature_cols_to_show = [
                col for col in feature_cols_to_show
                if col in df.columns
            ]

            selected_feature_df = selected_row[feature_cols_to_show].to_frame().T

            st.dataframe(
                selected_feature_df,
                use_container_width=True
            )

            st.subheader("Recent Actual Load Around Selected Timestamp")

            context_df = df[
                (df["time_idx"] >= selected_time_idx - 168)
                & (df["time_idx"] <= selected_time_idx + 24)
            ].copy()

            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(
                context_df["timestamp_local"],
                context_df[ACTUAL_COLUMN]
            )

            ax.axvline(
                selected_row["timestamp_local"],
                linestyle="--"
            )

            ax.axhline(
                baseline_prediction,
                linestyle=":"
            )

            ax.scatter(
                selected_row["timestamp_local"],
                actual_value,
                s=80,
                zorder=5
            )

            if tft_pred_value is not None:
                ax.scatter(
                    selected_row["timestamp_local"],
                    tft_pred_value,
                    s=100,
                    marker="X",
                    zorder=6
                )

                ax.plot(
                    [selected_row["timestamp_local"], selected_row["timestamp_local"]],
                    [actual_value, tft_pred_value],
                    linestyle="--",
                    alpha=0.7
                )

                ax.annotate(
                    f"TFT: {tft_pred_value:,.0f} MW",
                    xy=(selected_row["timestamp_local"], tft_pred_value),
                    xytext=(10, 10),
                    textcoords="offset points"
                )

            ax.annotate(
                f"Actual: {actual_value:,.0f} MW",
                xy=(selected_row["timestamp_local"], actual_value),
                xytext=(10, -15),
                textcoords="offset points"
            )

            ax.set_title("Actual Load Around Selected Timestamp")
            ax.set_xlabel("Time")
            ax.set_ylabel("Load (MW)")

            st.pyplot(fig)

            st.markdown("#### Chart Legend")

            legend_col1, legend_col2, legend_col3 = st.columns(3)

            with legend_col1:
                st.markdown("**━ Actual Load**  \nHistorical actual electricity load.")
                st.markdown("**╏ Selected Timestamp**  \nDashed vertical line showing selected date and hour.")

            with legend_col2:
                st.markdown("**⋯ Historical Average Baseline**  \nAverage load for the selected month and hour.")
                st.markdown("**🔵 Actual Point**  \nActual load at selected timestamp.")

            with legend_col3:
                st.markdown("**❌ TFT Prediction**  \nTFT No-Weather model output.")


# ============================================================
# Recent Test Backtest Mode
# ============================================================

elif mode == "Recent Test Backtest":
    st.subheader("Recent Test Backtest")

    st.write(
        "This mode runs TFT No-Weather prediction over recent test-period rows "
        "and compares the results with actual load."
    )

    if not model_available:
        st.warning(
            f"TFT checkpoint not found at `{MODEL_PATH}`. "
            "Please place the trained model checkpoint in the `models/` folder."
        )

        st.subheader("Fallback: Recent Actual Load Trend")

        recent_df = df.sort_values("timestamp_local").tail(168)

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            recent_df["timestamp_local"],
            recent_df[ACTUAL_COLUMN],
            label="Actual Load"
        )

        ax.set_title("Recent Actual Load Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (MW)")
        ax.legend()

        st.pyplot(fig)

    else:
        test_df = df[
            (df["data_date"] >= VALID_END_DATE)
            & (df["data_date"] < TEST_END_DATE)
        ].copy()

        if test_df.empty:
            st.error("No test-period rows found in the database.")
        else:
            max_rows = st.sidebar.slider(
                "Backtest rows",
                min_value=24,
                max_value=min(336, len(test_df)),
                value=min(168, len(test_df)),
                step=24
            )

            selected_test_df = test_df.tail(max_rows).copy()

            start_time_idx = int(selected_test_df["time_idx"].min())
            end_time_idx = int(selected_test_df["time_idx"].max())

            if st.button("Run Recent TFT Backtest", use_container_width=True):
                with st.spinner("Running recent TFT backtest..."):
                    df_serialized = df.to_json(orient="split", date_format="iso")

                    training_dataset = build_training_dataset(
                        df_serialized,
                        tuple(known_reals)
                    )

                    result_df = predict_tft_for_range(
                        df=df,
                        training_dataset=training_dataset,
                        start_time_idx=start_time_idx,
                        end_time_idx=end_time_idx
                    )

                    prediction_records = build_prediction_records(
                        result_df=result_df,
                        model_name="TFT_No_Weather",
                        horizon_hours=1,
                        source="recent_test_backtest"
                    )

                    saved_rows = save_prediction_records_to_db(prediction_records)

                metrics = evaluate_prediction(
                    result_df["actual_mw"],
                    result_df["tft_prediction_mw"]
                )

                st.subheader("TFT No-Weather Backtest Metrics")

                st.success(
                    f"Saved {saved_rows:,} TFT backtest prediction record(s) to the database."
                )

                if metrics is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("MAE", f"{metrics['MAE']:,.2f} MW")
                    c2.metric("RMSE", f"{metrics['RMSE']:,.2f} MW")
                    c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    c4.metric("R²", f"{metrics['R2']:.4f}")

                st.subheader("Actual vs TFT Prediction")

                fig, ax = plt.subplots(figsize=(12, 4))

                ax.plot(
                    result_df["timestamp_local"],
                    result_df["actual_mw"],
                    label="Actual Load"
                )

                ax.plot(
                    result_df["timestamp_local"],
                    result_df["tft_prediction_mw"],
                    label="TFT No-Weather Prediction"
                )

                ax.set_title("Recent Test Backtest: Actual vs TFT Prediction")
                ax.set_xlabel("Time")
                ax.set_ylabel("Load (MW)")
                ax.legend()

                st.pyplot(fig)

                st.subheader("Prediction Table")

                display_cols = [
                    "timestamp_local",
                    "actual_mw",
                    "tft_prediction_mw",
                    "tft_error_mw",
                    "tft_absolute_error_mw",
                ]

                st.dataframe(
                    result_df[display_cols],
                    use_container_width=True
                )

                st.download_button(
                    label="Download TFT Backtest Results as CSV",
                    data=result_df[display_cols].to_csv(index=False),
                    file_name="tft_no_weather_backtest_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Dashboard version: SQLite database integration. "
    "The model setup follows the TFT No-Weather configuration: "
    "time_idx + calendar features + holiday features, without weather variables."
)