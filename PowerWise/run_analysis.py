"""Command-line version of PowerWise: runs the analysis and saves charts + a text report.

    python run_analysis.py                              # uses data/sample_hourly.csv
    python run_analysis.py --csv my_data.csv --tariff 8  # your own data
"""
import argparse
import os

from powerwise import insights as ins
from powerwise import plots
from powerwise.pipeline import analyze


def main():
    ap = argparse.ArgumentParser(description="PowerWise - electricity usage predictor")
    ap.add_argument("--csv", default="data/sample_hourly.csv")
    ap.add_argument("--tariff", type=float, default=7.0, help="Rs. per kWh")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    an = analyze(a.csv, a.tariff)
    r, fc = an.result, an.result.forecast
    L = []
    L += ["=" * 72, "POWERWISE REPORT", "=" * 72,
          f"Data: {an.daily.index[0]:%d %b %Y} to {an.daily.index[-1]:%d %b %Y} ({len(an.daily)} days"
          f"{', hourly readings' if an.hourly is not None else ', daily readings'})",
          f"Average usage: {an.daily['kwh'].mean():.1f} kWh/day   Tariff: Rs.{a.tariff}/kWh", "",
          "-" * 72, "1. USAGE PREDICTION (next 30 days)", "-" * 72,
          f"Best model         : {r.best_model}",
          f"Expected usage     : {an.next30_kwh:,.0f} kWh  (range {fc['low'].sum():,.0f} - {fc['high'].sum():,.0f})",
          f"Estimated bill     : Rs.{an.next30_kwh * a.tariff:,.0f}   (last 30 days: Rs.{an.last30_kwh * a.tariff:,.0f}, "
          f"{(an.next30_kwh / an.last30_kwh - 1) * 100:+.1f}%)", "",
          "Model comparison (rolling back-test over the last 120 days, 4 windows of 30 days):",
          r.scores.round(3).to_string(index=False), "",
          "-" * 72, "2. MONTHLY COMPARISON", "-" * 72, *an.summary, "",
          "-" * 72, "3. PEAK CONSUMPTION ALERTS", "-" * 72]
    for al in an.alerts:
        L += [f"[{al['level']}] {al['title']}", f"    {al['detail']}"]
    L += ["", "-" * 72, "4. SAVINGS SUGGESTIONS", "-" * 72]
    for i, t in enumerate(an.tips, 1):
        sv = f"  (about Rs.{t['saving']:,.0f} per year)" if t.get("saving") else ""
        L += [f"{i}. {t['title']}{sv}", f"    {t['detail']}"]
    text = "\n".join(L)
    print(text)

    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write(text + "\n")
    fc.round({c: 2 for c in ["forecast", "low", "high"]}).to_csv(os.path.join(a.out, "forecast.csv"), index=False)
    r.scores.round(3).to_csv(os.path.join(a.out, "model_scores.csv"), index=False)
    an.monthly.round({c: 2 for c in an.monthly.select_dtypes("float").columns}).to_csv(os.path.join(a.out, "monthly_comparison.csv"), index=False)

    figs = {"1_forecast.png": plots.plot_history_forecast(an.daily, fc),
            "2_monthly_comparison.png": plots.plot_monthly_comparison(an.monthly),
            "3_monthly_cost.png": plots.plot_monthly_cost(an.monthly),
            "4_model_comparison.png": plots.plot_model_comparison(r.scores),
            "5_backtest.png": plots.plot_actual_vs_predicted(r.last_fold, r.best_model)}
    if r.importances is not None:
        figs["6_feature_importance.png"] = plots.plot_importances(r.importances)
    if an.hourly is not None:
        figs["7_hourly_profile.png"] = plots.plot_hourly_profile(ins.hourly_profile(an.hourly), ins.peak_window(an.hourly))
    if "temperature" in an.daily.columns:
        figs["8_temperature_vs_usage.png"] = plots.plot_temperature_usage(an.daily)
    for name, fig in figs.items():
        fig.savefig(os.path.join(a.out, name), dpi=150)
    print(f"\nSaved report, CSV files and {len(figs)} charts in '{a.out}/'")


if __name__ == "__main__":
    main()
