# PowerWise - Electricity Usage Predictor
**Capstone project report (Python, Pandas, Scikit-learn, Matplotlib)**

## 1. Problem and aim
People usually learn how much electricity they use only when the monthly bill arrives. PowerWise studies a
household's historical consumption, **forecasts the next 30 days**, **compares months**, raises **peak
consumption alerts** and gives **savings suggestions** with estimated rupee savings. A web app with
login lets each user keep and analyse their own data.

## 2. Technology
| Tool | Used for |
|---|---|
| Python 3 | whole project |
| Pandas / NumPy | loading, cleaning, resampling hourly -> daily -> monthly, feature engineering |
| Scikit-learn | Ridge Regression, Random Forest, Gradient Boosting, metrics |
| Matplotlib | all charts |
| Streamlit | web app (login page + dashboard) |
| SQLite + hashlib | user accounts with salted, hashed passwords |

## 3. Data
`data/sample_hourly.csv`: 722 days (1 Nov 2024 - 23 Oct 2026) of hourly `kwh` and `temperature`.
It is **synthetic**: it has a daily routine (morning and evening peaks), weekend effect, seasonal temperature,
air-conditioning load on hot days, 4% yearly growth, a holiday week away and a few injected events
(evening geyser/oven spikes, an AC left on overnight). Real meter readings can be uploaded instead.
Cleaning: duplicates removed, missing hours interpolated, days with fewer than 20 valid hours dropped.

## 4. Methodology
### 4.1 Usage prediction (machine learning)
1. Hourly data is summed to **daily kWh**.
2. **Features** (all known when the forecast is made):
   * calendar: day of week (sin/cos), weekend flag, day of year (sin/cos)
   * typical temperature for that calendar day and "degrees above 25 C" (from history; no weather API needed)
   * recent usage level: 28-day and 7-day averages and the same weekday 5 weeks earlier - **all measured at least 30 days before the day being predicted**.
   This means a 30-day forecast never needs guessed values and **no future data leaks** into training (there is a unit test for this).
3. **Models:** Ridge Regression, Random Forest, Gradient Boosting, compared with a naive baseline ("same weekday 5 weeks ago").
4. **Evaluation:** rolling-origin back-test - four consecutive 30-day test windows at the end of the data; each model is trained only on the days before each window (a random split would be wrong for time series).
5. The model with the lowest MAE is re-trained on all data and predicts the next 30 days. The **prediction range** (95%) comes from the model's errors in the back-test.

**Results (sample data, pooled over 120 test days):**

| Model | MAE (kWh/day) | RMSE | MAPE % | R2 |
|---|---|---|---|---|
| **Random Forest (selected)** | **0.95** | 1.64 | 5.7 | 0.20 |
| Gradient Boosting | 0.96 | 1.73 | 5.8 | 0.12 |
| Ridge Regression | 1.03 | 1.73 | 6.1 | 0.11 |
| Naive baseline | 1.47 | 2.13 | 9.4 | -0.34 |

The best model is about **35% more accurate than the naive baseline** (average error about 6% of daily usage).
R2 is modest because the test windows contain unpredictable one-off spikes and weather noise; MAE/MAPE are the
more meaningful measures here. Forecast for the next 30 days: about 422 kWh (range 327-517), roughly Rs.2,955 at Rs.7/kWh.
The most important features are the typical temperature, degrees above 25 C and the recent 4-week level.

### 4.2 Monthly comparison
Hourly/daily data is aggregated per month. Each month shows kWh, bill, average kWh/day, change versus the previous month and
versus the same month last year. Comparison uses **kWh per day**, so a month still in progress is fairly comparable.
Charts: year-by-year grouped bars and monthly bill bars. Example: October 2026 is 2.4% higher per day than September and 5.0% higher than October 2025.

### 4.3 Peak consumption alerts
* **Hourly spikes** (last 30 days): an hour is flagged when it is far above the usual for that hour and day type
  (robust z-score from median/MAD >= 4 and at least 0.8 kWh extra). Example found: 14 Oct 2026, 19:00-21:59, 4.5 kWh in one hour vs 1.4 usual.
* **Peak window:** the busiest four hours of the day and their share of total energy (18:00-21:59 = 33%).
* **Forecast alerts:** days predicted above the user's top-10% level, and a warning if next month is expected to rise by more than 10%.
* **Record demand:** highest hourly use this month compared with history.

### 4.4 Savings suggestions (with estimated Rs. per year)
Base/stand-by load (1-5 am usage), cooling sensitivity to temperature (regression of kWh on degrees above 25 C; rule of thumb 6% per 1 C on the AC setting),
shifting 15% of peak-window use under a time-of-day tariff (assumed 1.3x, adjustable), weekend usage, and year-on-year growth.
Every tip states its assumptions; the numbers are estimates, not guarantees.

## 5. Web application
* **Login / Create account** page. Passwords are never stored: each user has a random salt and a PBKDF2-HMAC-SHA256 hash (200,000 iterations).
  Usernames are validated, passwords need 6+ characters with letters and numbers, the same error is shown for unknown user and wrong password,
  and an account is **locked for 5 minutes after 5 failed attempts**. Demo account: `demo` / `demo123`.
* **Dashboard pages:** Dashboard (summary cards, forecast chart, latest alerts), Forecast (30-day chart, pick any date range, download),
  Monthly comparison, Peak alerts, Savings, Model performance (comparison, back-test, feature importance, temperature vs usage), My data (upload own CSV), Account (change password), Logout.
* Each user can upload their own CSV; it is validated first and bad files give a clear message.

## 6. Testing
`python -m unittest discover -s tests -v` runs 16 tests: registration/login/lock-out/password hashing, no-leakage check of the features,
forecast shape and sanity, ML model beats the baseline, friendly errors for bad files, monthly totals match, the injected spike is detected, savings are positive.

## 7. Limitations and future scope
* Sample data is synthetic; accuracy on real homes will differ.
* Future weather is approximated by the typical temperature for that date (a weather API forecast would improve results).
* Flat tariff (plus an optional peak multiplier); real slab tariffs could be added.
* Single-day one-off spikes cannot be predicted - they are what the alerts are for.
* Future work: weather API, appliance-level data, SMS/email alerts, LSTM/Prophet models, multi-household admin view, cloud database.

## 8. Possible viva questions
* **Why not a random train/test split?** Time-series data: the model must be tested on the future of what it trained on.
* **Why compare with a naive baseline?** A model is only useful if it beats a trivial guess.
* **What is data leakage and how is it avoided?** Using information from the future; features use only data >= 30 days older than the predicted day, and a unit test checks it.
* **How are passwords protected?** Salted PBKDF2 hashes, constant-time comparison, lock-out after repeated failures.
* **Why Random Forest?** It captures non-linear effects (e.g. usage rises sharply once it is hot) and is robust; it won the back-test.
