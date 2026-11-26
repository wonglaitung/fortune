"""event_radar：宏观事件抓取/过滤/归并/时区/渲染回归。"""
import os
import sys
import unittest
from datetime import date, datetime
from unittest.mock import patch

from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.event_radar as er  # noqa: E402


class _FakeDate(date):
    @classmethod
    def today(cls):
        return date(2026, 11, 26)


def _row(country, event, gmt, consensus="", actual="", previous=""):
    return {"country": country, "eventName": event, "gmt": gmt,
            "consensus": consensus, "actual": actual, "previous": previous}


class TestTimezone(unittest.TestCase):
    def test_winter_and_summer(self):
        self.assertEqual(er.event_time(date(2026, 11, 26), "08:30").strftime("%m-%d %H:%M"), "11-26 21:30")
        self.assertEqual(er.event_time(date(2026, 7, 15), "08:30").strftime("%m-%d %H:%M"), "07-15 20:30")

    def test_invalid_gmt(self):
        self.assertIsNone(er.event_time(date(2026, 11, 26), "24H"))


class TestCollectMacroEvents(unittest.TestCase):
    def setUp(self):
        self.rows = {
            date(2026, 11, 26): [
                _row("United States", "Core PCE Price Index", "08:30", "0.3%", "0.3%", "0.3%"),
                _row("United States", "Core PCE Prices", "08:30", "2.50%", "2.50%", "2.50%"),
                _row("United States", "GDP", "08:30", "3.4%", "4.3%", "2.2%"),
                _row("United States", "GDP Price Index", "08:30", "2.2%", "1.8%", "2.2%"),
                _row("United States", "Crude Oil Imports", "10:30", "-2.155M", "-2.155M", "-2.155M"),
                _row("China", "Chinese Industrial profit YTD", "20:30", "", "15.5%", "15.5%"),
            ],
            date(2026, 11, 27): [
                _row("Hong Kong", "Exports", "03:30", "", "64.9%", "64.9%"),
            ],
            date(2026, 12, 1): [
                _row("United States", "Nonfarm Payrolls", "08:30", "100K", "200K", "80K"),
            ],
        }
        self.now = datetime(2026, 11, 26, 23, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))

    def _collect(self):
        with patch.object(er, "date", _FakeDate), \
             patch.object(er, "fetch_rows", side_effect=lambda d: self.rows.get(d, [])):
            return er.collect_macro_events(days=5, now=self.now)

    def test_grouping_and_filtering(self):
        items = self._collect()
        events = [e["event"] for e in items]
        self.assertNotIn("GDP Price Index", events)
        self.assertNotIn("Crude Oil Imports", events)
        self.assertIn("GDP", events)
        self.assertIn("Chinese Industrial profit YTD", events)
        self.assertIn("Exports", events)
        core = [e for e in items if e["fam"] == "Core PCE"]
        self.assertEqual(len(core), 1)

    def test_actual_only_when_released(self):
        items = self._collect()
        by = {e["event"]: e for e in items}
        self.assertEqual(by["GDP"]["actual"], "4.3%")
        self.assertIn("+", er.surprise("4.3%", "3.4%"))
        self.assertEqual(by["Nonfarm Payrolls"]["actual"], "")

    def test_order_by_importance(self):
        items = self._collect()
        scores = [e["score"] for e in items]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(items[0]["score"], items[-1]["score"])

    def test_all_empty_days_returns_empty(self):
        with patch.object(er, "date", _FakeDate), patch.object(er, "fetch_rows", return_value=[]):
            self.assertEqual(er.collect_macro_events(days=5), [])

    def test_all_fetch_fail_raises(self):
        with patch.object(er, "date", _FakeDate), patch.object(er, "fetch_rows", return_value=None):
            with self.assertRaises(RuntimeError):
                er.collect_macro_events(days=5)


class TestRenderBrief(unittest.TestCase):
    def test_columns_and_placeholder(self):
        rows = [{"hkt": "11-26 21:30", "country": "United States", "event": "GDP",
                 "consensus": "3.4%", "previous": "2.2%", "actual": "4.3%",
                 "score": 9.0, "why": "增长 → 盈利/风险偏好"}]
        text = er.render_brief(rows, 5)
        self.assertIn("实际(超预期)", text)
        self.assertIn("4.3%", text)
        self.assertIn("不构成交易建议", text)
        empty = er.render_brief([], 5)
        self.assertIn("无符合条件的已排期事件", empty)


if __name__ == "__main__":
    unittest.main()
