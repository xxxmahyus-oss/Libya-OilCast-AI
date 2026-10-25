# Libya OilCast AI

A Streamlit research prototype for machine-learning-based oil production forecasting.

## Requirements
- Python 3.10 or newer
- Internet is only needed to install dependencies; the app runs locally after installation.

## Run on your computer
1. Extract this ZIP file.
2. Open Terminal / Command Prompt in the extracted `libya_oilcast_ai` folder.
3. Create and activate a virtual environment (optional but recommended).

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Run using Google Colab
This app is designed to run as a local Streamlit web app. It can also be run in Colab, but exposing the app URL requires an additional tunneling setup. For the easiest first run, use a computer with Python installed.

## Included features
- Synthetic multi-well data generator (clearly labelled synthetic)
- Dashboard and production analysis
- Chronological holdout evaluation
- Linear Regression, Random Forest, Extra Trees, Hist Gradient Boosting, and persistence baseline
- MAE, RMSE and R²
- Iterative forecast for 7–90 days
- CSV import and CSV exports
- PDF report generation

## CSV format
Required:
- `date`: date, e.g. `2025-01-31`
- `oil_production_bopd`: oil production rate in barrels per day

Recommended:
- `well`
- `field`
- `water_cut_pct`
- `pressure_psi`
- `choke_pct`
- `uptime_pct`

The app expects daily-ish observations. It groups duplicate field/well/date rows by mean. If your source has multiple records per day that should be summed (for example, separate valid volume totals), adapt the aggregation logic first.

## Research caveats
- Synthetic data is for software testing only and is not evidence of performance in Libyan fields.
- Long-horizon forecasts are recursive and uncertainty compounds.
- Forecast quality depends on correct data, units, operational context, and validation design.
- This prototype does not establish scientific novelty or patentability.
