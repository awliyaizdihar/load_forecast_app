import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from textwrap import dedent


warnings.filterwarnings("ignore")


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="CAL Load Forecasting Dashboard",
    layout="wide"
)


# ============================================================
# App Constants
# ============================================================

DATA_PATH = "all_features_all_timerange.csv"
MODEL_PATH = "models/TFT_No_Weather.ckpt"

REGION_NAME = "CAL"

TARGET_COLUMN = "model_target_mw"
ACTUAL_COLUMN = "region_demand_mw"
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
# Feature Configuration Based on TFT Ablation Notebook
# ============================================================

BASE_FEATURES = [
    "time_idx"
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
# Utility Functions
# ============================================================

def check_required_files():
    missing_items = []

    if not os.path.exists(DATA_PATH):
        missing_items.append(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        missing_items.append(MODEL_PATH)

    return missing_items


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


@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Datetime handling
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    if "data_date" in df.columns:
        df["data_date"] = pd.to_datetime(
            df["data_date"],
            errors="coerce"
        )
    else:
        df["data_date"] = df["timestamp_utc"].dt.date
        df["data_date"] = pd.to_datetime(df["data_date"])

    # Avoid mixed timezone problem by deriving local timestamp from UTC
    df["timestamp_local"] = (
        df["timestamp_utc"]
        .dt.tz_convert("America/Los_Angeles")
    )

    if "local_date" in df.columns:
        df["local_date"] = pd.to_datetime(
            df["local_date"],
            errors="coerce"
        )
    else:
        df["local_date"] = df["timestamp_local"].dt.date
        df["local_date"] = pd.to_datetime(df["local_date"])

    # Keep CAL only
    df = df[df["Region"].astype(str) == REGION_NAME].copy()

    # Sort before time_idx construction
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Continuous hourly time index, matched with TFT notebook
    df["time_idx"] = (
        (df["timestamp_utc"] - df["timestamp_utc"].min())
        .dt.total_seconds() // 3600
    ).astype(int)

    df["Region"] = df["Region"].astype(str)

    # Binary columns
    binary_cols = [
        "is_weekend",
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "is_holiday_period",
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = safe_binary_mapping(df[col])

    # Build TFT target exactly like ablation notebook
    if TRAINING_TARGET_COLUMN not in df.columns:
        st.error(f"Missing required column: {TRAINING_TARGET_COLUMN}")
        st.stop()

    if ACTUAL_COLUMN not in df.columns:
        st.error(f"Missing required column: {ACTUAL_COLUMN}")
        st.stop()

    df[TARGET_COLUMN] = np.where(
        df["data_date"] < TRAIN_END_DATE,
        df[TRAINING_TARGET_COLUMN],
        df[ACTUAL_COLUMN]
    )

    # Drop rows without model target
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


def extract_actuals_from_dataloader(dataloader):
    import torch

    actuals = []

    for x, y in dataloader:
        if isinstance(y, tuple):
            y = y[0]
        actuals.append(y.detach().cpu())

    return torch.cat(actuals, dim=0)


# ============================================================
# TFT Functions
# ============================================================

@st.cache_resource
def load_tft_dependencies_and_model(model_path):
    try:
        import torch
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer

        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        model.eval()

        return {
            "torch": torch,
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
def build_training_dataset(df_serialized, known_reals):
    """
    Streamlit cache_resource cannot directly cache mutable dataframe reliably,
    so this function receives a serialized dataframe.
    """
    from io import StringIO

    deps = load_tft_dependencies_and_model(MODEL_PATH)

    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]

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
    deps = load_tft_dependencies_and_model(MODEL_PATH)
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

if DATA_PATH in missing_items:
    st.error(
        f"Dataset file not found: `{DATA_PATH}`. "
        "Please place the CSV file in the same folder as app.py."
    )
    st.stop()

df = load_and_prepare_data()

known_reals = get_existing_known_reals(df)

missing_feature_cols = [
    col for col in NO_WEATHER_KNOWN_REALS
    if col not in df.columns
]

with st.expander("Model Configuration", expanded=False):
    st.write("**Region:**", REGION_NAME)
    st.write("**Target column:**", TARGET_COLUMN)
    st.write("**Max encoder length:**", MAX_ENCODER_LENGTH)
    st.write("**Max prediction length:**", MAX_PREDICTION_LENGTH)
    st.write("**Known real features used:**")
    st.code("\n".join(known_reals))

    if missing_feature_cols:
        st.warning(
            "Some expected no-weather features are missing from the CSV. "
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
        "Single Historical Prediction",
        "Recent Test Backtest"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("Model setup: **TFT No-Weather**")
st.sidebar.write("Region: **CAL**")

model_available = os.path.exists(MODEL_PATH)


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

if mode == "Single Historical Prediction":
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

    selected_timestamp_naive = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)

    candidate_rows = df[
        (df["timestamp_local"].dt.date == selected_date)
        & (df["timestamp_local"].dt.hour == selected_hour)
    ].copy()

    if candidate_rows.empty:
        st.warning(
            "Selected timestamp is not available in the dataset. "
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
                            known_reals
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

            st.dataframe(
                selected_row[feature_cols_to_show].to_frame().T,
                use_container_width=True
            )

            st.subheader("Recent Actual Load Around Selected Timestamp")

            context_df = df[
                (df["time_idx"] >= selected_time_idx - 168)
                & (df["time_idx"] <= selected_time_idx + 24)
            ].copy()

            fig, ax = plt.subplots(figsize=(12, 4))

            # Actual load line
            ax.plot(
                context_df["timestamp_local"],
                context_df[ACTUAL_COLUMN],
                label="Actual Load"
            )

            # Selected timestamp vertical line
            ax.axvline(
                selected_row["timestamp_local"],
                linestyle="--",
                label="Selected Timestamp"
            )

            # Historical baseline horizontal line
            ax.axhline(
                baseline_prediction,
                linestyle=":",
                label="Historical Average Baseline"
            )

            # Actual point at selected timestamp
            ax.scatter(
                selected_row["timestamp_local"],
                actual_value,
                s=80,
                label="Actual at Selected Timestamp",
                zorder=5
            )

            # TFT prediction point, only if available
            if tft_pred_value is not None:
                ax.scatter(
                    selected_row["timestamp_local"],
                    tft_pred_value,
                    s=100,
                    marker="X",
                    label="TFT Prediction",
                    zorder=6
                )

                # Optional: connect actual and TFT prediction with a vertical segment
                ax.plot(
                    [selected_row["timestamp_local"], selected_row["timestamp_local"]],
                    [actual_value, tft_pred_value],
                    linestyle="--",
                    alpha=0.7
                )

                # Optional: annotate TFT prediction value
                ax.annotate(
                    f"TFT: {tft_pred_value:,.0f} MW",
                    xy=(selected_row["timestamp_local"], tft_pred_value),
                    xytext=(10, 10),
                    textcoords="offset points"
                )

            # Optional: annotate actual value
            ax.annotate(
                f"Actual: {actual_value:,.0f} MW",
                xy=(selected_row["timestamp_local"], actual_value),
                xytext=(10, -15),
                textcoords="offset points"
            )

            ax.set_title("Actual Load Around Selected Timestamp")
            ax.set_xlabel("Time")
            ax.set_ylabel("Load (MW)")
            # ax.legend()

            st.pyplot(fig)

            st.markdown("#### Chart Legend")

            legend_col1, legend_col2, legend_col3 = st.columns(3)

            with legend_col1:
                st.markdown("**🔵 Actual Load**  \nHistorical actual electricity load.")
                st.markdown("**╏ Selected Timestamp**  \nDashed vertical line showing selected date and hour.")

            with legend_col2:
                st.markdown("**⋯ Historical Average Baseline**  \nDotted horizontal line showing average load for selected month and hour.")
                st.markdown("**🔵 Actual Point**  \nFilled circle showing actual load at selected timestamp.")

            with legend_col3:
                st.markdown("**❌ TFT Prediction**  \nX marker showing TFT No-Weather model output.")


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
            st.error("No test-period rows found in the dataset.")
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
                        known_reals
                    )

                    result_df = predict_tft_for_range(
                        df=df,
                        training_dataset=training_dataset,
                        start_time_idx=start_time_idx,
                        end_time_idx=end_time_idx
                    )

                metrics = evaluate_prediction(
                    result_df["actual_mw"],
                    result_df["tft_prediction_mw"]
                )

                st.subheader("TFT No-Weather Backtest Metrics")

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
    "Dashboard version: TFT No-Weather integration. "
    "The model setup follows the ablation study configuration: "
    "time_idx + calendar features + holiday features, without weather variables."
)