"""Monthly comparisons, peak-consumption alerts and savings suggestions."""
from __future__ import annotations

import numpy as np
import pandas as pd

CURRENCY = "Rs."


# --------------------------------------------------------------------------- #
# Monthly comparison
# --------------------------------------------------------------------------- #
def monthly_table(daily: pd.DataFrame, tariff: float) -> pd.DataFrame:
    """One row per calendar month with usage, cost and month-over-month / year-over-year change."""
    g = daily["kwh"].resample("MS").agg(["sum", "count"])
    g.columns = ["kwh", "days"]
    g["days_in_month"] = g.index.days_in_month
    g["complete"] = g["days"] >= g["days_in_month"]
    g["avg_per_day"] = g["kwh"] / g["days"]
    g["cost"] = g["kwh"] * tariff
    # compare using average per day so a partial month is still comparable
    g["mom_pct"] = g["avg_per_day"].pct_change() * 100
    g["yoy_pct"] = (g["avg_per_day"] / g["avg_per_day"].shift(12) - 1) * 100
    return g.reset_index().rename(columns={"date": "month"})


def monthly_summary(mt: pd.DataFrame) -> list[str]:
    """Plain-English comparison of the latest month with the previous month and last year."""
    lines = []
    cur = mt.iloc[-1]
    label = cur["month"].strftime("%B %Y")
    part = "" if cur["complete"] else f" (so far, {int(cur['days'])} days)"
    lines.append(f"{label}{part}: {cur['kwh']:.0f} kWh, about {CURRENCY}{cur['cost']:,.0f} "
                 f"({cur['avg_per_day']:.1f} kWh/day).")
    if len(mt) > 1 and pd.notna(cur["mom_pct"]):
        prev = mt.iloc[-2]
        word = "higher" if cur["mom_pct"] > 0 else "lower"
        lines.append(f"Compared with {prev['month']:%B %Y}: {abs(cur['mom_pct']):.1f}% {word} per day "
                     f"({prev['avg_per_day']:.1f} -> {cur['avg_per_day']:.1f} kWh/day).")
    if pd.notna(cur["yoy_pct"]):
        ly = mt.iloc[-13]
        word = "higher" if cur["yoy_pct"] > 0 else "lower"
        lines.append(f"Compared with {ly['month']:%B %Y} (last year): {abs(cur['yoy_pct']):.1f}% {word} per day.")
    full = mt[mt["complete"]]
    if len(full) >= 3:
        hi, lo = full.loc[full["kwh"].idxmax()], full.loc[full["kwh"].idxmin()]
        lines.append(f"Highest month on record: {hi['month']:%B %Y} ({hi['kwh']:.0f} kWh). "
                     f"Lowest: {lo['month']:%B %Y} ({lo['kwh']:.0f} kWh).")
    return lines


# --------------------------------------------------------------------------- #
# Hourly profile and peak alerts
# --------------------------------------------------------------------------- #
def peak_window(hourly: pd.DataFrame, width: int = 4) -> tuple[int, int]:
    """Start/end hour of the busiest `width` consecutive hours of the day."""
    prof = hourly["kwh"].groupby(hourly.index.hour).mean().reindex(range(24)).fillna(0)
    sums = [prof.iloc[h:h + width].sum() for h in range(24 - width + 1)]
    start = int(np.argmax(sums))
    return start, start + width - 1


def hourly_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average kWh for each hour of the day, split into weekday / weekend."""
    d = hourly.copy()
    d["hour"] = d.index.hour
    d["type"] = np.where(d.index.dayofweek >= 5, "Weekend", "Weekday")
    return d.pivot_table(index="hour", columns="type", values="kwh", aggfunc="mean")


def hourly_spikes(hourly: pd.DataFrame, recent_days: int = 30, z_min: float = 4.0, min_excess: float = 0.8) -> pd.DataFrame:
    """Hours in the last `recent_days` that were far above the usual for that hour and day type."""
    cutoff = hourly.index[-1] - pd.Timedelta(days=recent_days)
    h = hourly.copy()
    h["hour"], h["wk"] = h.index.hour, h.index.dayofweek >= 5
    hist = h[h.index <= cutoff]
    grp = hist.groupby(["wk", "hour"])["kwh"]
    med = grp.median().rename("usual")
    mad = grp.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))).rename("spread")
    stats = pd.concat([med, mad], axis=1).reset_index()
    rec = h[h.index > cutoff].reset_index().merge(stats, on=["wk", "hour"], how="left")
    rec["spread"] = np.maximum(rec["spread"], 0.15 * rec["usual"])
    rec["excess"] = rec["kwh"] - rec["usual"]
    rec["z"] = rec["excess"] / rec["spread"]
    out = rec[(rec["z"] >= z_min) & (rec["excess"] >= min_excess)].copy()
    return out.rename(columns={rec.columns[0]: "timestamp"})


def _runs(hours: list[int]) -> str:
    """[19,20,21,23] -> '19:00-21:59, 23:00-23:59'."""
    hours = sorted(hours)
    parts, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h != prev + 1:
            parts.append((start, prev)); start = h
        prev = h
    parts.append((start, prev))
    return ", ".join(f"{a:02d}:00-{b:02d}:59" for a, b in parts)


def peak_alerts(hourly: pd.DataFrame | None, daily: pd.DataFrame, forecast: pd.DataFrame, tariff: float) -> list[dict]:
    """Return alerts as dicts: level (HIGH/MEDIUM/INFO), title, detail."""
    alerts = []

    # 1. Heavy-usage days predicted for the next 30 days
    last_year = daily["kwh"].tail(365)
    p90 = float(last_year.quantile(0.90))
    heavy = forecast[forecast["forecast"] > p90]
    if len(heavy):
        days = ", ".join(f"{d:%d %b}" for d in heavy["date"].head(8))
        more = f" (+{len(heavy) - 8} more)" if len(heavy) > 8 else ""
        alerts.append({"level": "MEDIUM", "title": f"{len(heavy)} high-usage day(s) predicted in the next 30 days",
                       "detail": f"Forecast exceeds {p90:.1f} kWh/day (your top-10% level over the last year) on: {days}{more}. "
                                 f"Plan heavy appliance use (washing, ironing, geyser) for lighter days."})
    nxt = float(forecast["forecast"].sum())
    cur = float(daily["kwh"].tail(30).sum())
    if nxt > cur * 1.10:
        alerts.append({"level": "MEDIUM", "title": "Next month's usage is expected to rise",
                       "detail": f"Forecast {nxt:.0f} kWh vs {cur:.0f} kWh in the last 30 days (+{(nxt / cur - 1) * 100:.0f}%), "
                                 f"roughly {CURRENCY}{(nxt - cur) * tariff:,.0f} more on the bill."})

    if hourly is None:
        alerts.append({"level": "INFO", "title": "Hourly alerts need hourly data",
                       "detail": "Your file has daily readings only. Upload hourly data (timestamp, kwh) to unlock peak-hour alerts."})
        return alerts

    # 2. Unusual hourly spikes in the last 30 days
    sp = hourly_spikes(hourly)
    if len(sp):
        sp["day"] = sp["timestamp"].dt.date
        for day, g in sorted(sp.groupby("day"), reverse=True)[:5]:
            peak = g.loc[g["kwh"].idxmax()]
            extra = g["excess"].sum()
            night = all(h <= 5 for h in g["hour"])
            cause = ("A heavy appliance (AC, heater, geyser) may have been left running overnight."
                     if night else "Likely a heavy appliance (geyser, oven, iron, AC) used at the same time as other loads.")
            alerts.append({"level": "HIGH" if peak["z"] >= 6 else "MEDIUM",
                           "title": f"Peak spike on {day:%d %b %Y} ({_runs(g['hour'].tolist())})",
                           "detail": f"Reached {peak['kwh']:.1f} kWh in one hour at {int(peak['hour']):02d}:00 vs {peak['usual']:.1f} kWh usual "
                                     f"({peak['z']:.1f} standard deviations above normal). Extra energy that day: {extra:.1f} kWh "
                                     f"(about {CURRENCY}{extra * tariff:,.0f}). {cause}"})

    # 3. Share of energy in the busiest window
    a, b = peak_window(hourly)
    share = hourly[(hourly.index.hour >= a) & (hourly.index.hour <= b)]["kwh"].sum() / hourly["kwh"].sum() * 100
    level = "MEDIUM" if share > 35 else "INFO"
    alerts.append({"level": level, "title": f"Peak window is {a:02d}:00-{b:02d}:59",
                   "detail": f"{share:.0f}% of your electricity is used in these 4 hours ({share / (4 / 24) / 100:.1f}x the average hour). "
                             f"Moving some usage out of this window lowers peak demand and, with time-of-day tariffs, your bill."})

    # 4. Highest hourly demand this month vs history
    rec_max = hourly["kwh"].tail(30 * 24).max()
    p99 = hourly["kwh"].iloc[:-30 * 24].quantile(0.99)
    if rec_max > p99 * 1.15:
        alerts.append({"level": "MEDIUM", "title": "Record-level hourly demand this month",
                       "detail": f"Highest hour in the last 30 days: {rec_max:.1f} kWh, above your usual top 1% ({p99:.1f} kWh)."})
    return alerts


# --------------------------------------------------------------------------- #
# Savings suggestions
# --------------------------------------------------------------------------- #
def savings_suggestions(hourly: pd.DataFrame | None, daily: pd.DataFrame, tariff: float,
                        peak_multiplier: float = 1.3, shift_share: float = 0.15) -> list[dict]:
    """Each suggestion: title, detail, saving (Rs. per year, or None)."""
    tips = []
    yearly_kwh = daily["kwh"].tail(365).mean() * 365

    # (a) base load / stand-by
    if hourly is not None:
        night = hourly[(hourly.index.hour >= 1) & (hourly.index.hour <= 4)]["kwh"].tail(365 * 24)
        base = float(night.median())
        base_year = base * 24 * 365
        save_kwh = base_year * 0.15
        tips.append({"title": "Cut always-on (stand-by) load",
                     "detail": f"Between 1 am and 5 am you still use about {base:.2f} kWh every hour (fridge, router, TV/set-top box, chargers). "
                               f"That is {base_year:,.0f} kWh a year ({base_year / yearly_kwh * 100:.0f}% of your total). Switching off at the plug and "
                               f"replacing an old fridge could trim about 15% of it ({save_kwh:,.0f} kWh).",
                     "saving": save_kwh * tariff})
    else:
        base = float(daily["kwh"].quantile(0.05)) / 24
        tips.append({"title": "Cut always-on (stand-by) load",
                     "detail": f"Even on your quietest days you use about {base * 24:.1f} kWh. Switching off devices at the plug could trim about 15%.",
                     "saving": base * 24 * 365 * 0.15 * tariff})

    # (b) cooling / temperature sensitivity
    if "temperature" in daily.columns and daily["temperature"].std() > 1:
        excess = np.clip(daily["temperature"] - 25.0, 0, None)
        if excess.std() > 0.3:
            slope, intercept = np.polyfit(excess, daily["kwh"], 1)
            if slope > 0.1:
                cooling_year = slope * excess.tail(365).sum()
                save_kwh = cooling_year * 0.06 * 2          # rule of thumb: ~6% per 1 degC, setting AC 2 degC higher
                tips.append({"title": "Set the AC 2 degC higher",
                             "detail": f"Your usage rises by about {slope:.2f} kWh per day for every degC above 25 degC. Cooling-related use is roughly "
                                       f"{cooling_year:,.0f} kWh a year. As a rule of thumb each 1 degC higher AC setting saves about 6%; "
                                       f"also use ceiling fans and keep filters clean.",
                             "saving": save_kwh * tariff})

    # (c) peak-hour shifting (needs hourly data and a time-of-day tariff)
    if hourly is not None:
        a, b = peak_window(hourly)
        peak_kwh = hourly[(hourly.index.hour >= a) & (hourly.index.hour <= b)]["kwh"].tail(365 * 24).sum()
        saving = peak_kwh * shift_share * tariff * (peak_multiplier - 1)
        tips.append({"title": f"Shift some usage out of {a:02d}:00-{b:02d}:59",
                     "detail": f"If your utility charges more in peak hours (assumed {peak_multiplier:.1f}x the normal rate), moving {shift_share * 100:.0f}% of "
                               f"peak-window use (washing machine, dishwasher, water pump, EV/phone charging) to off-peak hours saves about "
                               f"{CURRENCY}{saving:,.0f} a year. Even without such a tariff it lowers peak load on the grid.",
                     "saving": saving})

    # (d) weekend effect
    wk = daily["kwh"].groupby(daily.index.dayofweek >= 5).mean()
    if len(wk) == 2 and wk[True] > wk[False] * 1.08:
        diff = wk[True] - wk[False]
        tips.append({"title": "Look at weekend usage",
                     "detail": f"Weekends use {diff:.1f} kWh/day more than weekdays ({wk[True]:.1f} vs {wk[False]:.1f}). Run full loads only, "
                               f"avoid leaving lights, fans and TVs on in empty rooms, and spread laundry across the week.",
                     "saving": diff * 0.10 * 104 * tariff})

    # (e) trend vs last year
    if len(daily) > 400:
        now, before = daily["kwh"].iloc[-90:].mean(), daily["kwh"].iloc[-455:-365].mean()
        if now > before * 1.08:
            tips.append({"title": "Usage is higher than a year ago",
                         "detail": f"The last 90 days average {now:.1f} kWh/day vs {before:.1f} kWh/day in the same period last year "
                                   f"(+{(now / before - 1) * 100:.0f}%). Check for new appliances, ageing equipment or changed habits.",
                         "saving": (now - before) * 365 * tariff * 0.5})
    return tips
