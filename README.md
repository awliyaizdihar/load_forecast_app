# CAL Electricity Load Forecasting App

This project is a Streamlit-based dashboard for electricity load forecasting in the CAL region. The current version is a proof-of-concept dashboard that reads historical feature data from a CSV file and generates a placeholder forecast based on historical average load.

The dashboard is intended to be updated later with the final machine learning model, especially the best-performing no-weather model from the experiment results.

## Project Overview

The app currently provides:

- Forecast input through the Streamlit sidebar.
- Date and hour selection for prediction.
- Placeholder load prediction using historical average demand.
- Display of generated input features.
- Recent actual load trend visualization.

The current placeholder model should later be replaced with the final trained model.

## Project Structure

A typical structure for this project is:

```text
load_forecast_app/
│
├── app.py
├── features.py
├── requirements.txt
├── README.md
└── all_features_all_timerange.csv
```

### File Description

| File                             | Description                                                   |
| -------------------------------- | ------------------------------------------------------------- |
| `app.py`                         | Main Streamlit application file.                              |
| `features.py`                    | Helper functions for calendar and holiday feature generation. |
| `requirements.txt`               | Python package dependencies.                                  |
| `all_features_all_timerange.csv` | Dataset used by the current dashboard version.                |
| `README.md`                      | Project documentation and running instructions.               |

## Requirements

Recommended setup:

- Python 3.10 or 3.11
- Windows PowerShell or VS Code terminal
- Virtual environment (`.venv`)

## How to Run the Project

### 1. Open the project folder

Open PowerShell or the VS Code terminal, then go to the project folder:

```powershell
cd "C:\Users\surya\Downloads\Rapi\NTUST\Semester 2\MI5125701_Machine Learning and Big Data Analytics\load_forecast_app"
```

Adjust the path if the project is located in a different folder.

### 2. Create a virtual environment

If `.venv` does not exist yet, create it using:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\activate
```

After activation, the terminal should show `(.venv)` at the beginning of the line, for example:

```text
(.venv) PS C:\Users\surya\...\load_forecast_app>
```

### 4. Install dependencies

```powershell
pip install -r .\requirements.txt
```

If installation fails, check the troubleshooting section below.

### 5. Run the Streamlit app

```powershell
streamlit run app.py
```

If Streamlit does not open automatically, copy the local URL shown in the terminal and open it in the browser. Usually, the URL is:

```text
http://localhost:8501
```

## Current App Workflow

1. The app loads `all_features_all_timerange.csv`.
2. The data is filtered to the CAL region.
3. The user selects a forecast date and hour from the sidebar.
4. The app generates a placeholder prediction using historical average load for the selected month and hour.
5. The app displays:
   - Predicted CAL load
   - Input/generated features
   - Recent actual load trend

## Notes for Future Development

Based on the latest experiment results, the final dashboard should be aligned with the no-weather model configuration. Therefore, the weather input section in the current proof-of-concept version may be removed later.

Recommended next updates:

- Replace the placeholder prediction with the final trained model.
- Remove manual weather inputs if the final model uses the no-weather configuration.
- Add EIA API integration to fetch actual demand data.
- Add EIA forecast data as a comparison baseline.
- Add model evaluation metrics such as MAE, RMSE, MAPE, and R².
- Add comparison chart between actual demand, EIA forecast, and model forecast.

## Troubleshooting / Common Problems

### 1. `streamlit is not recognized`

This usually means the virtual environment is not activated or Streamlit is not installed.

Try:

```powershell
.\.venv\Scripts\activate
python -m streamlit run app.py
```

If it still does not work, reinstall the dependencies:

```powershell
pip install -r .\requirements.txt
```

### 2. Error when installing `fonttools`

Example error:

```text
ERROR: Wheel 'fonttools' ... is invalid.
```

This is usually caused by a corrupted or incomplete package download.

Try:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip cache purge
pip install --no-cache-dir -r .\requirements.txt
```

If the same error still appears, install `fonttools` manually:

```powershell
pip install --no-cache-dir --force-reinstall fonttools
pip install --no-cache-dir -r .\requirements.txt
```

### 3. `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'pandas'
```

This means the required package has not been installed in the active environment.

Make sure the virtual environment is active:

```powershell
.\.venv\Scripts\activate
```

Then install the requirements again:

```powershell
pip install -r .\requirements.txt
```

### 4. CSV file not found

Example:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'all_features_all_timerange.csv'
```

Make sure `all_features_all_timerange.csv` is located in the same folder as `app.py`.

The folder should look like this:

```text
load_forecast_app/
├── app.py
├── features.py
├── requirements.txt
└── all_features_all_timerange.csv
```

If the CSV file is inside a `data/` folder, update the path inside `app.py`, for example:

```python
df = pd.read_csv("data/all_features_all_timerange.csv")
```

### 5. PowerShell cannot activate `.venv`

If this command fails:

```powershell
.\.venv\Scripts\activate
```

and shows an execution policy error, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then try activating the virtual environment again:

```powershell
.\.venv\Scripts\activate
```

### 6. App runs but browser does not open

This is normal sometimes. Check the terminal output and open the local URL manually:

```text
http://localhost:8501
```

### 7. App shows old code after editing

Stop the running app using `Ctrl + C` in the terminal, then run it again:

```powershell
streamlit run app.py
```

You can also click **Rerun** in the Streamlit browser page.

## Suggested Stable `requirements.txt`

If dependency installation causes problems, use a more stable version set:

```text
streamlit==1.40.2
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.2
scikit-learn==1.5.2
joblib==1.4.2
requests==2.32.3
holidays==0.59
```

Then reinstall using:

```powershell
pip install --no-cache-dir -r .\requirements.txt
```

## Important Note

The current app still uses a placeholder prediction method. The prediction result should not be interpreted as the final machine learning model output. The placeholder should be replaced with the final trained model before final deployment or presentation.
