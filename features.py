import pandas as pd
import numpy as np
import holidays


def create_calendar_features(forecast_date, forecast_hour):
    """
    Create time-based calendar features for a selected forecast date and hour.
    These features follow the no-weather experiment setup.
    """
    timestamp = pd.Timestamp(forecast_date) + pd.Timedelta(hours=int(forecast_hour))

    features = {
        "local_year": timestamp.year,
        "local_month": timestamp.month,
        "local_day": timestamp.day,
        "local_hour": timestamp.hour,
        "local_day_of_week": timestamp.dayofweek,
        "local_day_of_year": timestamp.dayofyear,

        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),

        "day_of_week_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7),

        "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
        "month_cos": np.cos(2 * np.pi * timestamp.month / 12),

        "day_of_year_sin": np.sin(2 * np.pi * timestamp.dayofyear / 365),
        "day_of_year_cos": np.cos(2 * np.pi * timestamp.dayofyear / 365),
    }

    return features


def create_holiday_features(forecast_date, country="US"):
    """
    Create US holiday-related features.
    These features are retained because the ablation study showed that
    holiday information is useful for the no-weather setup.
    """
    date = pd.Timestamp(forecast_date).date()

    us_holidays = holidays.country_holidays(country)

    previous_date = date - pd.Timedelta(days=1)
    next_date = date + pd.Timedelta(days=1)

    is_holiday = int(date in us_holidays)
    is_day_before_holiday = int(next_date in us_holidays)
    is_day_after_holiday = int(previous_date in us_holidays)

    features = {
        "is_weekend": int(pd.Timestamp(date).dayofweek >= 5),
        "is_holiday": is_holiday,
        "is_day_before_holiday": is_day_before_holiday,
        "is_day_after_holiday": is_day_after_holiday,
        "is_holiday_period": int(
            is_holiday or is_day_before_holiday or is_day_after_holiday
        ),
        "holiday_name": us_holidays.get(date, "None"),
    }

    return features


def create_no_weather_features(forecast_date, forecast_hour):
    """
    Combine calendar and holiday features for the final no-weather dashboard.
    """
    features = {}

    features.update(
        create_calendar_features(
            forecast_date=forecast_date,
            forecast_hour=forecast_hour
        )
    )

    features.update(
        create_holiday_features(
            forecast_date=forecast_date
        )
    )

    return features