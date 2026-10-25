"""
PowerWise - Electricity Usage Predictor (web app)
=================================================
Login / register page + dashboard.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Demo account (created automatically on first run):  username: demo    password: demo123
"""
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from powerwise import auth
from powerwise import insights as ins
from powerwise import plots
from powerwise.model import DataError, load_usage
from powerwise.pipeline import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "data", "sample_hourly.csv")
USER_DIR = os.path.join(HERE, "data", "user_data")
TEMPLATE = "timestamp,kwh,temperature\n2026-01-01 00:00,0.31,19.5\n2026-01-01 01:00,0.28,19.1\n2026-01-01 02:00,0.27,18.8\n"

st.set_page_config(page_title="PowerWise", page_icon="⚡", layout="wide")
auth.init_db()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def user_file(username):
    return os.path.join(USER_DIR, f"{username}.csv")


def data_path(username):
    p = user_file(username)
    return p if os.path.exists(p) else SAMPLE


@st.cache_data(show_spinner="Training the forecasting models...")
def cached_analysis(path, mtime, tariff, peak_mult):
    """`mtime` is only here so the cache refreshes when the data file changes."""
    return analyze(path, tariff, peak_mult)


def alert_box(a):
    text = f"**{a['title']}**  \n{a['detail']}"
    if a["level"] == "HIGH":
        st.error("🔴 " + text)
    elif a["level"] == "MEDIUM":
        st.warning("🟠 " + text)
    else:
        st.info("🔵 " + text)


# --------------------------------------------------------------------------- #
# Login / register page
# --------------------------------------------------------------------------- #
def login_page():
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<h1 style='text-align:center'>⚡ PowerWise</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:gray'>Electricity usage predictor - "
                    "forecasts, monthly comparisons, peak alerts and savings tips.</p>", unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Login", "Create account"])

        with tab_in:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                go = st.form_submit_button("Log in")
            if go:
                try:
                    st.session_state["user"] = auth.login(u, p)
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))
            st.caption("Demo account: username **demo**, password **demo123**")

        with tab_up:
            with st.form("register_form"):
                name = st.text_input("Full name")
                u2 = st.text_input("Choose a username", help="3-20 characters: letters, numbers, underscore")
                p1 = st.text_input("Password", type="password", help="At least 6 characters with letters and numbers")
                p2 = st.text_input("Confirm password", type="password")
                reg = st.form_submit_button("Create account")
            if reg:
                if p1 != p2:
                    st.error("The two passwords do not match.")
                else:
                    try:
                        auth.register(u2, name, p1)
                        st.success("Account created. You can log in now.")
                    except auth.AuthError as e:
                        st.error(str(e))


# --------------------------------------------------------------------------- #
# Dashboard pages
# --------------------------------------------------------------------------- #
def page_overview(an, tariff):
    r, fc = an.result, an.result.forecast
    best = r.scores[r.scores["model"] == r.best_model].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average per day", f"{an.daily['kwh'].mean():.1f} kWh")
    c2.metric("Last 30 days", f"{an.last30_kwh:,.0f} kWh", f"Rs.{an.last30_kwh * tariff:,.0f}", delta_color="off")
    chg = (an.next30_kwh / an.last30_kwh - 1) * 100
    c3.metric("Next 30 days (forecast)", f"{an.next30_kwh:,.0f} kWh", f"{chg:+.1f}% vs last 30 days", delta_color="inverse")
    c4.metric("Forecast accuracy", f"{100 - best['MAPE %']:.1f}%", f"avg error {best['MAE']:.2f} kWh/day", delta_color="off")

    show(plots.plot_history_forecast(an.daily, fc))
    st.caption(f"Predicted by **{r.best_model}**, chosen as the most accurate of 3 models in a back-test. "
               f"Estimated bill for the next 30 days: **Rs.{an.next30_kwh * tariff:,.0f}**.")

    left, right = st.columns(2)
    with left:
        st.subheader("Latest alerts")
        top = sorted(an.alerts, key=lambda a: {"HIGH": 0, "MEDIUM": 1, "INFO": 2}[a["level"]])[:3]
        for a in top:
            alert_box(a)
        if not top:
            st.success("No alerts - usage looks normal.")
    with right:
        st.subheader("This month")
        for line in an.summary:
            st.write("• " + line)
        total = sum(t["saving"] for t in an.tips if t.get("saving"))
        if total > 0:
            st.success(f"💡 Savings tips could be worth about **Rs.{total:,.0f} a year**. See the *Savings* page.")


def page_forecast(an, tariff):
    fc = an.result.forecast
    st.subheader("Usage prediction - next 30 days")
    show(plots.plot_history_forecast(an.daily, fc, days_back=120))

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected usage", f"{an.next30_kwh:,.0f} kWh", f"range {fc['low'].sum():,.0f}-{fc['high'].sum():,.0f}", delta_color="off")
    c2.metric("Estimated bill", f"Rs.{an.next30_kwh * tariff:,.0f}")
    c3.metric("Highest day", f"{fc['forecast'].max():.1f} kWh", f"{fc.loc[fc['forecast'].idxmax(), 'date']:%d %b}", delta_color="off")

    st.markdown("##### Predict a particular day or period")
    a, b = st.columns(2)
    start = a.date_input("From", value=fc["date"].iloc[0].date(), min_value=fc["date"].iloc[0].date(), max_value=fc["date"].iloc[-1].date())
    end = b.date_input("To", value=fc["date"].iloc[6].date(), min_value=fc["date"].iloc[0].date(), max_value=fc["date"].iloc[-1].date())
    if end < start:
        st.error("'To' date must be on or after 'From' date.")
    else:
        sel = fc[(fc["date"].dt.date >= start) & (fc["date"].dt.date <= end)]
        st.write(f"**{len(sel)} day(s)**: about **{sel['forecast'].sum():.0f} kWh** "
                 f"(range {sel['low'].sum():.0f}-{sel['high'].sum():.0f} kWh), roughly **Rs.{sel['forecast'].sum() * tariff:,.0f}**.")

    with st.expander("Daily forecast table"):
        t = fc.copy()
        t["date"] = t["date"].dt.strftime("%a %d %b %Y")
        st.dataframe(t.rename(columns={"date": "Date", "forecast": "Forecast kWh", "low": "Low", "high": "High"})
                     .round({"Forecast kWh": 2, "Low": 2, "High": 2}), hide_index=True)
    st.download_button("Download forecast (CSV)", fc.to_csv(index=False), file_name="forecast.csv", mime="text/csv")


def page_monthly(an, tariff):
    st.subheader("Monthly comparison")
    for line in an.summary:
        st.write("• " + line)
    show(plots.plot_monthly_comparison(an.monthly))
    show(plots.plot_monthly_cost(an.monthly))
    t = an.monthly.copy()
    t["Month"] = t["month"].dt.strftime("%b %Y") + t["complete"].map({True: "", False: " (so far)"})
    t = t[["Month", "kwh", "avg_per_day", "cost", "mom_pct", "yoy_pct"]].rename(columns={
        "kwh": "kWh", "avg_per_day": "kWh/day", "cost": "Bill (Rs.)", "mom_pct": "vs prev. month %", "yoy_pct": "vs last year %"})
    t = t.iloc[::-1].round({"kWh": 0, "kWh/day": 1, "Bill (Rs.)": 0, "vs prev. month %": 1, "vs last year %": 1})
    st.dataframe(t, hide_index=True)
    st.download_button("Download monthly table (CSV)", t.to_csv(index=False), file_name="monthly_comparison.csv", mime="text/csv")


def page_peaks(an):
    st.subheader("Peak consumption alerts")
    for a in an.alerts:
        alert_box(a)
    if an.hourly is not None:
        win = ins.peak_window(an.hourly)
        st.markdown(f"##### Your typical day - busiest window {win[0]:02d}:00-{win[1]:02d}:59")
        show(plots.plot_hourly_profile(ins.hourly_profile(an.hourly), win))


def page_savings(an, tariff):
    st.subheader("Savings suggestions")
    total = sum(t["saving"] for t in an.tips if t.get("saving"))
    st.metric("Estimated potential saving", f"Rs.{total:,.0f} per year", f"at Rs.{tariff}/kWh", delta_color="off")
    for t in an.tips:
        with st.container(border=True):
            st.markdown(f"**{t['title']}**")
            st.write(t["detail"])
            if t.get("saving"):
                st.success(f"≈ Rs.{t['saving']:,.0f} per year")
    st.caption("Savings are estimates based on your own usage pattern and the assumptions stated in each tip.")


def page_model(an):
    r = an.result
    st.subheader("How good is the prediction?")
    st.write("Three scikit-learn models were trained on your history and tested on four consecutive 30-day windows "
             "they had **never seen** (always trained on the past only). A simple 'same weekday five weeks ago' guess is "
             "included as a baseline. The most accurate model is used for the forecast.")
    sc = r.scores.rename(columns={"model": "Model", "MAE": "MAE (kWh/day)", "RMSE": "RMSE (kWh/day)"}).round(3)
    st.dataframe(sc, hide_index=True)
    st.caption("MAE = average error per day, RMSE = like MAE but punishes big misses, MAPE = error as % of usage, "
               "R2 = share of day-to-day variation explained (modest here because single-day spikes and weather noise cannot be predicted).")
    left, right = st.columns(2)
    with left:
        show(plots.plot_model_comparison(r.scores))
    with right:
        if r.importances is not None:
            show(plots.plot_importances(r.importances))
    show(plots.plot_actual_vs_predicted(r.last_fold, r.best_model))
    if "temperature" in an.daily.columns:
        show(plots.plot_temperature_usage(an.daily))
    with st.expander("Results of every test window"):
        st.dataframe(r.fold_scores.round(3), hide_index=True)


def page_data(username):
    st.subheader("My data")
    mine = os.path.exists(user_file(username))
    st.info("You are using **your uploaded data**." if mine else "You are using the built-in **sample data** (synthetic).")
    st.write("Upload a CSV with **hourly** readings (`timestamp, kwh` and optionally `temperature`) or **daily** readings "
             "(`date, kwh`). At least 180 days are needed for forecasting. Hourly data unlocks peak-hour alerts.")
    st.download_button("Download hourly CSV template", TEMPLATE, file_name="template_hourly.csv", mime="text/csv")
    up = st.file_uploader("Upload your electricity usage CSV", type=["csv"])
    if up is not None and st.button("Use this file"):
        raw = up.getvalue()
        try:
            load_usage(io.BytesIO(raw))
        except DataError as e:
            st.error(str(e))
        else:
            os.makedirs(USER_DIR, exist_ok=True)
            with open(user_file(username), "wb") as f:
                f.write(raw)
            st.success("Data saved. Open the Dashboard to see your forecast.")
            st.rerun()
    if mine and st.button("Go back to the sample data"):
        os.remove(user_file(username))
        st.rerun()


def page_account(user):
    st.subheader("Account")
    st.write(f"Name: **{user['full_name']}**  \nUsername: **{user['username']}**")
    with st.form("pw_form"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        new2 = st.text_input("Confirm new password", type="password")
        ok = st.form_submit_button("Change password")
    if ok:
        if new != new2:
            st.error("The new passwords do not match.")
        else:
            try:
                auth.change_password(user["username"], old, new)
                st.success("Password changed.")
            except auth.AuthError as e:
                st.error(str(e))


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def dashboard(user):
    PAGES = ["📊 Dashboard", "🔮 Forecast", "📅 Monthly comparison", "🚨 Peak alerts",
             "💡 Savings", "🧠 Model performance", "🗂 My data", "👤 Account"]
    with st.sidebar:
        st.markdown(f"### ⚡ PowerWise\nHello, **{user['full_name']}**")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.divider()
        tariff = st.number_input("Price per unit (Rs./kWh)", min_value=0.5, max_value=100.0, value=7.0, step=0.5)
        peak_mult = st.number_input("Peak-hour price multiplier", min_value=1.0, max_value=3.0, value=1.3, step=0.1,
                                    help="Used only for the peak-shifting savings tip (time-of-day tariff).")
        st.divider()
        if st.button("Log out"):
            st.session_state.pop("user", None)
            st.rerun()

    if page == PAGES[6]:
        page_data(user["username"])
        return
    if page == PAGES[7]:
        page_account(user)
        return

    path = data_path(user["username"])
    try:
        an = cached_analysis(path, os.path.getmtime(path), tariff, peak_mult)
    except DataError as e:
        st.error(f"Your data could not be analysed: {e}")
        st.info("Open **My data** to upload a different file or return to the sample data.")
        return

    st.title(page[2:])
    if page == PAGES[0]:
        page_overview(an, tariff)
    elif page == PAGES[1]:
        page_forecast(an, tariff)
    elif page == PAGES[2]:
        page_monthly(an, tariff)
    elif page == PAGES[3]:
        page_peaks(an)
    elif page == PAGES[4]:
        page_savings(an, tariff)
    elif page == PAGES[5]:
        page_model(an)


if "user" not in st.session_state:
    login_page()
else:
    dashboard(st.session_state["user"])
