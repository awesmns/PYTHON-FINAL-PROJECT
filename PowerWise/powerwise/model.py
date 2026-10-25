"""Data loading and machine-learning forecasting for PowerWise.

Approach
--------
* Hourly (or daily) readings are aggregated to DAILY kWh.
* Features use only information that is really available when the forecast is made:
    - calendar: day-of-week, weekend flag, day-of-year (cyclic sin/cos)
    - climatological temperature for that calendar day (no weather API needed)
    - recent usage level measured >= 30 days earlier (so a 30-day forecast needs no
      recursive guessing and there is no data leakage)
* Three scikit-learn models (Ridge, Random Forest, Gradient Boosting) and a naive
  baseline are compared with rolling-origin cross-validation (4 consecutive
  30-day test windows, always training on the past only).
* The best model (lowest average MAE) is re-trained on all data and used to
  forecast the next 30 days with a prediction range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 30          # forecast length in days (also the minimum feature lag)
CV_FOLDS = 4          # number of rolling test windows
MIN_DAYS = 180        # minimum history needed for a meaningful model
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class DataError(ValueError):
    """Raised with a user-friendly message when an uploaded file cannot be used."""


def load_usage(source) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Read a CSV (path or file-like). Returns (hourly or None, daily).

    Accepted formats
      hourly: timestamp, kwh [, temperature]
      daily : date, kwh      [, temperature]
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # unreadable / not a CSV
        raise DataError(f"Could not read the file as CSV ({exc}).") from exc
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "kwh" not in raw.columns:
        raise DataError("The file needs a 'kwh' column (electricity used).")
    tcol = "timestamp" if "timestamp" in raw.columns else ("date" if "date" in raw.columns else None)
    if tcol is None:
        raise DataError("The file needs a 'timestamp' (hourly) or 'date' (daily) column.")
    raw[tcol] = pd.to_datetime(raw[tcol], errors="coerce")
    raw["kwh"] = pd.to_numeric(raw["kwh"], errors="coerce")
    raw = raw.dropna(subset=[tcol, "kwh"])
    raw = raw[raw["kwh"] >= 0].drop_duplicates(tcol).sort_values(tcol)
    has_temp = "temperature" in raw.columns
    if has_temp:
        raw["temperature"] = pd.to_numeric(raw["temperature"], errors="coerce")

    if tcol == "timestamp":
        hourly = raw.set_index("timestamp")[["kwh"] + (["temperature"] if has_temp else [])]
        hourly = hourly.asfreq("h")
        hourly["kwh"] = hourly["kwh"].interpolate(limit=6)
        if has_temp:
            hourly["temperature"] = hourly["temperature"].interpolate(limit=6)
        agg = {"kwh": "sum"}
        if has_temp:
            agg.update({"temperature": "mean"})
        counts = hourly["kwh"].resample("D").count()
        daily = hourly.resample("D").agg(agg)
        daily = daily[counts >= 20]              # drop days with too many missing hours
        if has_temp:
            daily["temp_max"] = hourly["temperature"].resample("D").max()
    else:
        hourly = None
        daily = raw.set_index("date")[["kwh"] + (["temperature"] if has_temp else [])]

    daily = daily.asfreq("D")
    daily["kwh"] = daily["kwh"].interpolate(limit=3)
    if has_temp:
        daily["temperature"] = daily["temperature"].interpolate(limit=3)
    daily = daily.dropna(subset=["kwh"])
    daily = daily.asfreq("D")
    daily["kwh"] = daily["kwh"].interpolate()
    if has_temp:
        daily["temperature"] = daily["temperature"].interpolate().bfill().ffill()
    daily.index.name = "date"
    if len(daily) < MIN_DAYS:
        raise DataError(f"Need at least {MIN_DAYS} days of data for forecasting; the file has {len(daily)}.")
    return hourly, daily


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def climatology(daily: pd.DataFrame) -> pd.Series | None:
    """Typical temperature for each day of the year (smoothed), from history only."""
    if "temperature" not in daily.columns:
        return None
    g = daily["temperature"].groupby(daily.index.dayofyear).mean().reindex(range(1, 367))
    g = g.interpolate(limit_direction="both")
    ext = pd.concat([g.iloc[-15:], g, g.iloc[:15]])          # wrap around the year end
    sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
    sm.index = range(1, 367)
    return sm


def make_features(series: pd.Series, clim: pd.Series | None) -> pd.DataFrame:
    """Feature table for every date in `series` (may include future NaN dates).

    Every usage-derived feature is shifted by HORIZON days, so the row for date t
    only uses data up to t-30. This is what makes a 30-day forecast leak-free.
    """
    idx = series.index
    f = pd.DataFrame(index=idx)
    dow, doy = idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["is_weekend"] = (dow >= 5).astype(int)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if clim is not None:
        t = clim.reindex(doy).to_numpy()
        f["temp_clim"] = t
        f["hot_degrees"] = np.clip(t - 25.0, 0, None)
    f["level_28d"] = series.rolling(28, min_periods=20).mean().shift(HORIZON)
    f["level_7d"] = series.rolling(7, min_periods=5).mean().shift(HORIZON)
    f["same_dow_5w"] = series.shift(35)                      # same weekday, 5 weeks earlier
    return f


def _models() -> dict:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                       subsample=0.8, random_state=RANDOM_STATE),
    }


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE %": float(np.mean(np.abs(y - p) / np.maximum(y, 1e-6)) * 100),
            "R2": r2_score(y, p)}


# --------------------------------------------------------------------------- #
# Training, evaluation, forecasting
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    forecast: pd.DataFrame                     # date, forecast, low, high
    best_model: str
    scores: pd.DataFrame                       # model comparison (pooled over all CV windows)
    fold_scores: pd.DataFrame                  # per-fold detail
    last_fold: pd.DataFrame                    # actual vs predicted for the newest test window
    importances: pd.Series | None
    resid_std: float
    features: list = field(default_factory=list)


def run_forecast(daily: pd.DataFrame, horizon: int = HORIZON) -> ForecastResult:
    """Compare models by rolling-origin CV, fit the best on all data, forecast `horizon` days."""
    y = daily["kwh"]
    n = len(y)
    models = _models()
    fold_rows, preds_by_model = [], {m: [] for m in list(models) + ["Naive (same weekday 5 weeks ago)"]}
    last_fold = None

    for k in range(CV_FOLDS, 0, -1):
        test_end = n - (k - 1) * horizon
        test_start = test_end - horizon
        train_y = y.iloc[:test_start]
        if len(train_y) < 120:
            continue
        clim = climatology(daily.iloc[:test_start])
        feats = make_features(y.iloc[:test_end], clim)          # nothing after the window is used
        train_X = feats.iloc[:test_start].dropna()
        train_t = y.loc[train_X.index]
        test_X = feats.iloc[test_start:test_end]
        test_t = y.iloc[test_start:test_end]
        fold_pred = {}
        for name, mdl in models.items():
            mdl.fit(train_X, train_t)
            fold_pred[name] = mdl.predict(test_X.fillna(train_X.mean()))
        fold_pred["Naive (same weekday 5 weeks ago)"] = test_X["same_dow_5w"].fillna(train_t.mean()).to_numpy()
        for name, p in fold_pred.items():
            m = _metrics(test_t, p)
            fold_rows.append({"fold": CV_FOLDS - k + 1, "test_from": test_t.index[0].date(),
                              "test_to": test_t.index[-1].date(), "model": name, **m})
            preds_by_model[name].append(pd.Series(p, index=test_t.index))
        last_fold = (test_t, fold_pred)

    fold_scores = pd.DataFrame(fold_rows)
    # pooled metrics over all test windows (R2 inside one 30-day window is not meaningful)
    rows = []
    for name, parts in preds_by_model.items():
        p = pd.concat(parts)
        rows.append({"model": name, **_metrics(y.reindex(p.index), p)})
    scores = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    ml_scores = scores[scores["model"].isin(models)]
    best = ml_scores.iloc[0]["model"]

    # residual spread of the best model across all CV windows -> prediction range
    resid = pd.concat(preds_by_model[best]) - y.reindex(pd.concat(preds_by_model[best]).index)
    resid_std = float(resid.std())

    # fit best model on ALL data and forecast the future
    clim_all = climatology(daily)
    future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    extended = pd.concat([y, pd.Series(np.nan, index=future_idx)])
    feats_all = make_features(extended, clim_all)
    X_hist = feats_all.loc[y.index].dropna()
    final = _models()[best]
    final.fit(X_hist, y.loc[X_hist.index])
    X_fut = feats_all.loc[future_idx].fillna(X_hist.mean())
    pred = np.clip(final.predict(X_fut), 0, None)
    steps = np.arange(1, horizon + 1)
    band = 1.96 * resid_std * np.sqrt(1 + 0.3 * steps / horizon)
    fc = pd.DataFrame({"date": future_idx, "forecast": pred,
                       "low": np.clip(pred - band, 0, None), "high": pred + band})

    importances = None
    if hasattr(final, "feature_importances_"):
        importances = pd.Series(final.feature_importances_, index=X_hist.columns).sort_values(ascending=False)

    lf = pd.DataFrame({"actual": last_fold[0], "predicted": last_fold[1][best],
                       "naive": last_fold[1]["Naive (same weekday 5 weeks ago)"]})
    return ForecastResult(fc, best, scores, fold_scores, lf, importances, resid_std, list(X_hist.columns))
