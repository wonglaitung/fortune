import math

import pytest

from scripts.stock_fundamentals import justified_pb


def test_justified_pb_basic():
    v = justified_pb(0.093, 0.32, coe=0.11)
    assert v == pytest.approx((0.093 - 0.05) / (0.11 - 0.05))


def test_justified_pb_full_payout():
    assert justified_pb(0.09, 1.0, coe=0.11) == pytest.approx(0.09 / 0.11)


def test_justified_pb_invalid():
    assert math.isnan(justified_pb(0.01, 0.0, coe=0.11, g_cap=0.05))
