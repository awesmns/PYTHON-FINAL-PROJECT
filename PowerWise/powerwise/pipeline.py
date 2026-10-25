"""One call that runs the whole PowerWise analysis (used by both the app and the CLI)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import insights as ins
from .model import ForecastResult, load_usage, run_forecast


@dataclass
class Analysis:
    hourly: pd.DataFrame | None
    daily: pd.DataFrame
    result: ForecastResult
    monthly: pd.DataFrame
    summary: list
    alerts: list
    tips: list
    tariff: float

    @property
    def next30_kwh(self) -> float:
        return float(self.result.forecast["forecast"].sum())

    @property
    def last30_kwh(self) -> float:
        return float(self.daily["kwh"].tail(30).sum())


def analyze(source, tariff: float = 7.0, peak_multiplier: float = 1.3) -> Analysis:
    hourly, daily = load_usage(source)
    result = run_forecast(daily)
    monthly = ins.monthly_table(daily, tariff)
    return Analysis(
        hourly=hourly, daily=daily, result=result, monthly=monthly,
        summary=ins.monthly_summary(monthly),
        alerts=ins.peak_alerts(hourly, daily, result.forecast, tariff),
        tips=ins.savings_suggestions(hourly, daily, tariff, peak_multiplier=peak_multiplier),
        tariff=tariff,
    )
