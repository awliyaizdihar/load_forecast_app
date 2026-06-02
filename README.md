# CAL Electricity Load Forecasting Dashboard

A Streamlit dashboard for CAL electricity load monitoring and TFT No-Weather forecasting. The app uses a local SQLite database seeded from the historical preprocessed dataset, with optional manual updates from the EIA API.

## Project Structure

```text
load_forecast_app/
├── app.py
├── config.py
├── db_utils.py
├── seed_database.py
├── eia_client.py
├── preprocessing.py
├── requirements.txt
├── data/
│   └── all_features_all_timerange.csv
├── database/
│   └── load_forecast.db
├── models/
│   └── TFT_No_Weather.ckpt
└── .venv/
```

## Setup

Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies.

```powershell
pip install -r requirements.txt
```

Seed the SQLite database from the historical CSV.

```powershell
python seed_database.py
```

To reset and reseed the database:

```powershell
python seed_database.py --reset
```

Run the dashboard.

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Optional EIA API Setup

For manual EIA API updates, set the API key before running Streamlit.

```powershell
$env:EIA_API_KEY="YOUR_EIA_API_KEY"
streamlit run app.py
```

## Dashboard Pages

### Live Monitoring

Main operational page. It displays latest CAL actual load, EIA forecast, saved TFT predictions, evaluation metrics, monitoring chart, and saved prediction records. It also includes buttons to fetch latest EIA data and run TFT predictions for recent database rows.

### Single Historical Prediction

Runs TFT prediction for one selected timestamp from the database and compares actual load, historical average baseline, and TFT No-Weather prediction.

### Recent Test Backtest

Runs TFT No-Weather prediction on the fixed historical test period used in the experiment. This page is intended for model evaluation, not for the latest EIA API data.

## Notes

- The final model setup is **TFT No-Weather**.
- The database table `load_hourly` stores actual load, EIA forecast, and no-weather features.
- The database table `model_predictions` stores saved TFT prediction results.
- EIA updates are manual through the dashboard button, not scheduled automatically.

## Common Issues

If Streamlit still shows old results, clear cache:

```powershell
streamlit cache clear
streamlit run app.py
```

If the database is missing, run:

```powershell
python seed_database.py
```

If EIA update fails, check that `EIA_API_KEY` is set correctly.
