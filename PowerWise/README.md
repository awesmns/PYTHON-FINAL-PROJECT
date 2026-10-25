# PowerWise - Electricity Usage Predictor

Python capstone project (Python, Pandas, Scikit-learn, Matplotlib + Streamlit web app).

PowerWise analyses historical electricity consumption and forecasts future usage.
Features: **usage prediction, monthly comparisons, peak-consumption alerts, savings suggestions**,
wrapped in a web app with a **login / register page and a dashboard**.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 and log in with the demo account:

| Username | Password |
|---|---|
| `demo` | `demo123` |

(or use the *Create account* tab to register your own account).

### Command-line version (no web app)
```bash
python run_analysis.py                                  # sample data -> outputs/
python run_analysis.py --csv my_data.csv --tariff 8.5   # your own data
```

### Tests
```bash
python -m unittest discover -s tests -v                 # 16 tests
```

## Folder structure
```
PowerWise/
|-- app.py                  # Streamlit web app: login + dashboard
|-- run_analysis.py         # command-line analysis, saves report + charts to outputs/
|-- requirements.txt
|-- README.md  REPORT.md    # REPORT.md = full project explanation for submission
|-- powerwise/
|   |-- data_gen.py         # creates the synthetic sample data
|   |-- model.py            # loading, features, scikit-learn models, cross-validation, forecast
|   |-- insights.py         # monthly comparison, peak alerts, savings suggestions
|   |-- plots.py            # Matplotlib charts
|   |-- pipeline.py         # runs everything in one call
|   `-- auth.py             # accounts: SQLite + hashed passwords + lock-out
|-- data/sample_hourly.csv  # 722 days of hourly readings (timestamp, kwh, temperature)
|-- outputs/                # report.txt, CSVs and 8 charts from run_analysis.py
`-- tests/test_powerwise.py
```

## Using your own data
In the app open **My data** and upload a CSV (each user's file is kept separately):
* hourly: `timestamp, kwh` and optionally `temperature`  -> all features
* daily : `date, kwh` (optionally `temperature`)        -> forecast, monthly comparison, savings (no hourly peak alerts)

At least 180 days of data are needed. The included sample data is **synthetic** (made by
`python -m powerwise.data_gen`); replace it with real meter readings for real use.
The user database `data/powerwise.db` is created automatically the first time the app starts.
