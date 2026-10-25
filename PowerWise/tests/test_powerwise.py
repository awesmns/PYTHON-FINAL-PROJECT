"""Unit tests.  Run from the project folder:   python -m unittest discover -s tests -v"""
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powerwise import auth, insights
from powerwise.model import HORIZON, DataError, climatology, load_usage, make_features, run_forecast

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_hourly.csv")


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "t.db")
        auth.init_db(self.db, seed_demo=False)

    def test_register_and_login(self):
        auth.register("alice_1", "Alice", "secret12", self.db)
        u = auth.login("ALICE_1", "secret12", self.db)          # username is case-insensitive
        self.assertEqual(u["full_name"], "Alice")

    def test_wrong_password(self):
        auth.register("bob123", "Bob", "secret12", self.db)
        with self.assertRaises(auth.AuthError):
            auth.login("bob123", "wrong", self.db)

    def test_unknown_user_gives_same_message(self):
        auth.register("bob123", "Bob", "secret12", self.db)
        msgs = set()
        for u, p in [("bob123", "bad"), ("nobody", "bad")]:
            try:
                auth.login(u, p, self.db)
            except auth.AuthError as e:
                msgs.add(str(e))
        self.assertEqual(len(msgs), 1)

    def test_password_not_stored_in_plain_text(self):
        auth.register("carol", "Carol", "plain1234", self.db)
        with open(self.db, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"plain1234", raw)

    def test_duplicate_and_validation(self):
        auth.register("dave_9", "Dave", "secret12", self.db)
        for args in [("dave_9", "Dave", "secret12"), ("x", "X", "secret12"), ("ok_user", "N", "short"),
                     ("ok_user", "N", "onlyletters"), ("bad name!", "N", "secret12")]:
            with self.assertRaises(auth.AuthError):
                auth.register(*args, self.db)

    def test_lockout_after_five_failures(self):
        auth.register("eve_1", "Eve", "secret12", self.db)
        for _ in range(auth.MAX_ATTEMPTS):
            with self.assertRaises(auth.AuthError):
                auth.login("eve_1", "bad", self.db)
        with self.assertRaises(auth.AuthError) as ctx:          # even the right password is refused
            auth.login("eve_1", "secret12", self.db)
        self.assertIn("Too many", str(ctx.exception))

    def test_change_password(self):
        auth.register("fay_1", "Fay", "secret12", self.db)
        auth.change_password("fay_1", "secret12", "newpass99", self.db)
        self.assertEqual(auth.login("fay_1", "newpass99", self.db)["username"], "fay_1")
        with self.assertRaises(auth.AuthError):
            auth.login("fay_1", "secret12", self.db)

    def test_demo_account_seeded(self):
        db = os.path.join(tempfile.mkdtemp(), "d.db")
        auth.init_db(db)
        self.assertEqual(auth.login("demo", "demo123", db)["username"], "demo")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hourly, cls.daily = load_usage(SAMPLE)
        cls.result = run_forecast(cls.daily)

    def test_loading(self):
        self.assertEqual(len(self.hourly), len(self.daily) * 24)
        self.assertTrue((self.daily["kwh"] > 0).all())

    def test_features_do_not_leak_future_data(self):
        """Changing data newer than t-30 days must not change the features for day t."""
        y = self.daily["kwh"]
        clim = climatology(self.daily)
        base = make_features(y, clim)
        t = 400
        y2 = y.copy()
        y2.iloc[t - HORIZON + 1:] = 999.0                        # corrupt everything after t-30
        changed = make_features(y2, clim)
        pd.testing.assert_series_equal(base.iloc[t], changed.iloc[t])

    def test_forecast_shape_and_sanity(self):
        fc = self.result.forecast
        self.assertEqual(len(fc), HORIZON)
        self.assertTrue((fc["forecast"] >= 0).all())
        self.assertTrue((fc["low"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["high"]).all())
        self.assertEqual(fc["date"].iloc[0], self.daily.index[-1] + pd.Timedelta(days=1))

    def test_ml_beats_naive_baseline(self):
        s = self.result.scores.set_index("model")["MAE"]
        self.assertLess(s[self.result.best_model], s["Naive (same weekday 5 weeks ago)"])

    def test_bad_files_raise_friendly_errors(self):
        for text in ["a,b\n1,2\n", "date,kwh\n2026-01-01,5\n", "not a csv \x00\x01"]:
            with self.assertRaises(DataError):
                load_usage(io.StringIO(text))


class InsightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hourly, cls.daily = load_usage(SAMPLE)
        cls.fc = run_forecast(cls.daily).forecast

    def test_monthly_table(self):
        mt = insights.monthly_table(self.daily, 7.0)
        self.assertAlmostEqual(mt["kwh"].sum(), self.daily["kwh"].sum(), places=3)
        self.assertFalse(mt.iloc[-1]["complete"])               # October 2026 is partial in the sample

    def test_peak_alert_finds_injected_spike(self):
        titles = " ".join(a["title"] for a in insights.peak_alerts(self.hourly, self.daily, self.fc, 7.0))
        self.assertIn("14 Oct 2026", titles)

    def test_savings_have_positive_values(self):
        tips = insights.savings_suggestions(self.hourly, self.daily, 7.0)
        self.assertGreaterEqual(len(tips), 3)
        self.assertTrue(all(t["saving"] > 0 for t in tips))


if __name__ == "__main__":
    unittest.main()
