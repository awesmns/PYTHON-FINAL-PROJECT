"""Matplotlib charts for PowerWise. Every function returns a Figure (caller shows/saves it)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_BLUE, C_ORANGE, C_GREY, C_BAND, C_GRID, C_TEXT = "#2a78d6", "#eb6834", "#6b6a66", "#c3c2b7", "#e6e5e0", "#0b0b0b"
C_GREEN = "#1baf7a"


def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=C_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9992")
    ax.tick_params(colors="#52514e")


def _legend_below(ax, ncol=4, y=-0.18):
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def plot_history_forecast(daily: pd.DataFrame, fc: pd.DataFrame, days_back: int = 90) -> plt.Figure:
    hist = daily["kwh"].tail(days_back)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(hist.index, hist.values, color=C_BLUE, linewidth=1.5, label=f"Actual (last {days_back} days)")
    ax.fill_between(fc["date"], fc["low"], fc["high"], color=C_BAND, alpha=0.5, linewidth=0, label="95% range")
    ax.plot(fc["date"], fc["forecast"], color=C_ORANGE, linewidth=1.8, linestyle="--", label="Forecast (next 30 days)")
    ax.axvline(hist.index[-1], color="#9a9992", linewidth=0.8)
    _style(ax, "Daily electricity usage: history and forecast", "kWh per day")
    _legend_below(ax, 3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_monthly_comparison(mt: pd.DataFrame) -> plt.Figure:
    """Grouped bars: each calendar month, one bar per year (avg kWh/day)."""
    d = mt.copy()
    d["year"], d["mon"] = d["month"].dt.year, d["month"].dt.month
    piv = d.pivot(index="mon", columns="year", values="avg_per_day").reindex(range(1, 13))
    years = list(piv.columns)
    colors = ["#c3c2b7", C_BLUE, C_ORANGE, C_GREEN][-len(years):] if len(years) <= 4 else None
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(years)
    x = np.arange(12)
    for i, y in enumerate(years):
        ax.bar(x + i * w - 0.4 + w / 2, piv[y].values, width=w, label=str(y), color=colors[i] if colors else None)
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    _style(ax, "Monthly comparison: average usage per day, year by year", "kWh per day")
    _legend_below(ax, len(years), -0.12)
    fig.tight_layout()
    return fig


def plot_monthly_cost(mt: pd.DataFrame, months: int = 12) -> plt.Figure:
    d = mt.tail(months)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = [C_BLUE if c else "#9bbfec" for c in d["complete"]]
    bars = ax.bar(d["month"].dt.strftime("%b %y"), d["kwh"], color=colors, width=0.7)
    ax.bar_label(bars, labels=[f"Rs.{c:,.0f}" for c in d["cost"]], fontsize=8, color="#52514e", padding=2)
    _style(ax, "Monthly usage and estimated bill (light bar = month in progress)", "kWh per month")
    fig.tight_layout()
    return fig


def plot_hourly_profile(profile: pd.DataFrame, window: tuple[int, int] | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for col, color in zip(profile.columns, [C_BLUE, C_ORANGE]):
        ax.plot(profile.index, profile[col], color=color, linewidth=2, marker="o", markersize=4, label=col)
    if window:
        ax.axvspan(window[0] - 0.5, window[1] + 0.5, color=C_BAND, alpha=0.45, label="Peak window")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day", color="#52514e")
    _style(ax, "Your typical day: average usage by hour", "kWh per hour")
    _legend_below(ax, 3, -0.22)
    fig.tight_layout()
    return fig


def plot_model_comparison(scores: pd.DataFrame) -> plt.Figure:
    d = scores.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [C_ORANGE if "Naive" in m else C_BLUE for m in d["model"]]
    bars = ax.barh(d["model"], d["MAE"], color=colors, height=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, color="#52514e")
    ax.grid(axis="x", color=C_GRID)
    _style(ax, "Model comparison: average error (lower is better)")
    ax.set_xlabel("MAE, kWh per day", color="#52514e")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(lf: pd.DataFrame, model: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(lf.index, lf["actual"], color=C_BLUE, linewidth=1.6, label="Actual")
    ax.plot(lf.index, lf["predicted"], color=C_ORANGE, linewidth=1.6, linestyle="--", label=f"{model} (predicted)")
    ax.plot(lf.index, lf["naive"], color="#9a9992", linewidth=1.0, label="Naive baseline")
    _style(ax, "Back-test: predictions vs what really happened (latest 30 days)", "kWh per day")
    _legend_below(ax, 3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_importances(imp: pd.Series) -> plt.Figure:
    labels = {"temp_clim": "Typical temperature", "hot_degrees": "Degrees above 25 C", "level_28d": "Recent 4-week level",
              "level_7d": "Recent 1-week level", "dow_sin": "Weekday (sin)", "dow_cos": "Weekday (cos)",
              "is_weekend": "Weekend", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)", "same_dow_5w": "Same weekday 5 weeks ago"}
    d = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([labels.get(i, i) for i in d.index], d.values, color=C_BLUE, height=0.6)
    ax.grid(axis="x", color=C_GRID)
    _style(ax, "What drives the prediction (feature importance)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_temperature_usage(daily: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.scatter(daily["temperature"], daily["kwh"], s=14, color=C_BLUE, alpha=0.55, linewidth=0)
    x = daily["temperature"].to_numpy()
    m, c = np.polyfit(x, daily["kwh"].to_numpy(), 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + c, color=C_ORANGE, linewidth=2, label=f"Trend: {m:+.2f} kWh per degC")
    ax.set_xlabel("Average daily temperature (degC)", color="#52514e")
    _style(ax, "Hotter days use more electricity", "kWh per day")
    _legend_below(ax, 1, -0.2)
    fig.tight_layout()
    return fig
