import pandas as pd


def historical_average_baseline(df_cal, forecast_date, forecast_hour):
    """
    Simple baseline forecast.

    The prediction is calculated as the historical average demand for the
    selected month and hour. If no matching month-hour data is available,
    the function falls back to the average for the selected hour. If that is
    also unavailable, it falls back to the overall average demand.
    """
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


def evaluate_baseline_for_hour(df_cal, prediction, forecast_hour):
    """
    Calculate simple historical MAE and MAPE for the selected hour.
    This is not a true future test evaluation, but it provides a quick
    dashboard-level diagnostic for the baseline forecast.
    """
    same_hour_data = df_cal[df_cal["local_hour"] == forecast_hour].copy()

    if same_hour_data.empty:
        return None, None

    same_hour_data["absolute_error"] = (
        same_hour_data["target_region_demand_mw"] - prediction
    ).abs()

    mae = same_hour_data["absolute_error"].mean()

    valid_mape_data = same_hour_data[
        same_hour_data["target_region_demand_mw"] != 0
    ].copy()

    if valid_mape_data.empty:
        mape = None
    else:
        mape = (
            valid_mape_data["absolute_error"]
            / valid_mape_data["target_region_demand_mw"]
        ).mean() * 100

    return mae, mape