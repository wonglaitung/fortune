from datetime import date, timedelta

import pytest

from scripts.dca_eval import xirr


def test_xirr_simple_10pct():
    d0 = date(2020, 1, 1)
    v = xirr([(d0, -100.0), (d0 + timedelta(days=365), 110.0)])
    assert v == pytest.approx(0.10, abs=1e-4)


def test_xirr_multiple_contributions():
    d0 = date(2020, 1, 1)
    cf = [(d0, -100.0), (d0 + timedelta(days=365), -100.0),
          (d0 + timedelta(days=730), 220.0)]
    v = xirr(cf)
    assert 0.05 < v < 0.15
