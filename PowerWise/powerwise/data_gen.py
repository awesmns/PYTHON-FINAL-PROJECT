"""Generate a realistic synthetic HOURLY household electricity dataset.

Run:  python -m powerwise.data_gen
Creates data/sample_hourly.csv with columns: timestamp, kwh, temperature

The data has a daily routine (morning and evening peaks), a weekend effect,
Indian-climate-style seasonal temperature, air-conditioning load on hot days,
random noise and a few real-life events (a holiday week away, some evening
spikes, a night-time AC left on) so that alerts have something to find.
"""
import os

import numpy as np
import pandas as pd

HOUR_PROFILE = np.array([0.35, 0.30, 0.28, 0.28, 0.30, 0.40, 0.70, 0.90, 0.70, 0.50, 0.45, 0.45,
                         0.50, 0.50, 0.50, 0.55, 0.65, 0.90, 1.30, 1.50, 1.40, 1.10, 0.70, 0.45]) * 0.8


def generate(path="data/sample_hourly.csv", start="2024-11-01", end="2026-10-23", seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, f"{end} 23:00", freq="h")
    n_days = len(idx) // 24
    days = pd.date_range(start, periods=n_days, freq="D")

    # daily mean temperature: hottest around late April, coolest around January, + weather noise
    doy = days.dayofyear.to_numpy()
    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.9)
    daily_temp = 25.0 + 4.5 * np.sin(2 * np.pi * (doy - 24) / 365) + noise
    hour = idx.hour.to_numpy()
    temp = np.repeat(daily_temp, 24) + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.3, len(idx))

    dow = idx.dayofweek.to_numpy()
    weekend = dow >= 5
    load = HOUR_PROFILE[hour].copy()
    load[weekend] *= 1.12
    load[weekend & (hour >= 10) & (hour <= 16)] *= 1.15

    # air-conditioning / cooling: afternoon-to-night, grows with temperature above 25 C
    ac_hours = ((hour >= 12) & (hour <= 23)) | (hour <= 1)
    load += np.where(ac_hours, 0.06 * np.clip(temp - 25.0, 0, None), 0.0)

    # slow yearly growth (~4%) and random noise
    growth = 1.0 + 0.04 * (np.arange(len(idx)) / (365 * 24))
    load = load * growth * rng.lognormal(0, 0.10, len(idx))

    s = pd.Series(load, index=idx)
    s["2025-12-20":"2025-12-27"] *= 0.40                       # family away for a week
    for day, h0, h1, extra in [("2025-05-14", 19, 22, 2.8), ("2025-10-02", 18, 21, 2.2),
                               ("2026-03-18", 19, 22, 3.0), ("2026-10-14", 19, 21, 3.2)]:
        m = (s.index.date == pd.Timestamp(day).date()) & (s.index.hour >= h0) & (s.index.hour <= h1)
        s[m] += extra                                           # geyser / oven / iron left on
    m = (s.index.date == pd.Timestamp("2026-09-02").date()) & (s.index.hour >= 1) & (s.index.hour <= 5)
    s[m] += 1.6                                                 # AC running all night

    df = pd.DataFrame({"timestamp": idx, "kwh": s.round(3).to_numpy(), "temperature": temp.round(1)})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print(f"Wrote data/sample_hourly.csv: {len(d)} hourly rows, {len(d) // 24} days")
