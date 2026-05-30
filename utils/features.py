import pandas as pd
import holidays

def create_calendar_features(forecast_date, forecast_hour):
    timestamp = pd.Timestamp(forecast_date) + pd.Timedelta(hours=int(forecast_hour))

    features = {
        "year": timestamp.year,
        "month": timestamp.month,
        "day": timestamp.day,
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
    }

    return features

def create_holiday_features(forecast_date, country="US"):
    us_holidays = holidays.country_holidays(country)
    is_holiday = int(forecast_date in us_holidays)

    return {
        "is_holiday": is_holiday,
        "holiday_name": us_holidays.get(forecast_date, "None")
    }
