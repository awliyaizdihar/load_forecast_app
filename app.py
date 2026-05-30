import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CAL Load Forecasting App",
    layout="wide"
)

st.title("CAL Electricity Load Forecasting App")
st.write(
    "This app predicts electricity load for the CAL region. "
    "Currently, the prediction uses a placeholder baseline model."
)

@st.cache_data
def load_data():
    df = pd.read_csv("all_features_all_timerange.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    df["local_date"] = pd.to_datetime(df["local_date"])
    return df

df = load_data()

# Since final project uses CAL only
df_cal = df[df["Region"] == "CAL"].copy()

st.sidebar.header("Forecast Input")

forecast_date = st.sidebar.date_input("Date")

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

selected_time = st.sidebar.selectbox("Time", list(time_options.keys()))
forecast_hour = time_options[selected_time]

st.sidebar.markdown("### Weather Input")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=20.0,
    step=1.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    value=60.0,
    step=1.0,
    min_value=0.0,
    max_value=100.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (km/h)",
    value=5.0,
    step=1.0,
    min_value=0.0
)

precipitation = st.sidebar.number_input(
    "Precipitation (mm)",
    value=0.0,
    step=1.0,
    min_value=0.0
)

def placeholder_predict(df_cal, forecast_date, forecast_hour):
    forecast_month = pd.Timestamp(forecast_date).month

    matched_data = df_cal[
        (df_cal["local_month"] == forecast_month) &
        (df_cal["local_hour"] == forecast_hour)
    ]

    if matched_data.empty:
        matched_data = df_cal[df_cal["local_hour"] == forecast_hour]

    if matched_data.empty:
        return df_cal["target_region_demand_mw"].mean()

    return matched_data["target_region_demand_mw"].mean()

if st.sidebar.button("Generate Forecast", use_container_width=True):
    prediction = placeholder_predict(df_cal, forecast_date, forecast_hour)

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted CAL Load",
        value=f"{prediction:,.2f} MW"
    )

    st.info(
        "Note: This is currently a placeholder prediction using the average "
        "historical load for the selected hour. It will be replaced with the final ML model later."
    )

    generated_features = pd.DataFrame([{
        "Region": "CAL",
        "Date": forecast_date,
        "Time": selected_time,
        "Hour Value": forecast_hour,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Precipitation (mm)": precipitation
    }])

    st.subheader("Input / Generated Features")
    st.dataframe(generated_features)

    st.subheader("Recent Actual Load Trend")

    recent_df = df_cal.sort_values("timestamp_local").tail(168)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        recent_df["timestamp_local"],
        recent_df["target_region_demand_mw"],
        label="Actual Load"
    )
    ax.axhline(
        prediction,
        linestyle="--",
        label="Placeholder Prediction"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.legend()

    st.pyplot(fig)