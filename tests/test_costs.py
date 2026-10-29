import numpy as np
import pandas as pd

from ml_services.costs import (COST_PER_SIDE, COST_ROUND_TRIP,
                               cost_from_turnover, cost_from_weight_change)
from ml_services.strategy_eval import daily_mark


def test_round_trip_and_per_side():
    assert COST_PER_SIDE == COST_ROUND_TRIP / 2
    assert cost_from_turnover(1.0) == COST_ROUND_TRIP
    assert cost_from_turnover(0.5) == COST_ROUND_TRIP / 2


def test_weight_change_is_half_of_round_trip():
    # 0→1（全买）再 1→0（全卖），合计 |Δw|=2 ⇒ 正好一次往返
    assert cost_from_weight_change(1.0) == COST_ROUND_TRIP / 2
    assert cost_from_weight_change(2.0) == COST_ROUND_TRIP


def test_daily_mark_full_round_trip_cost():
    ret = pd.Series([0.01] * 4, index=pd.date_range('2024-01-01', periods=4))
    w = pd.Series([1.0, 1.0, 0.0, 0.0], index=ret.index)
    net = daily_mark(ret, w, round_trip_cost=COST_ROUND_TRIP, lag=0)
    gross = float((w * ret).sum())
    # 建仓 0.5×1 + 平仓 0.5×1 = 1.0 次往返 ⇒ 总成本 = COST_ROUND_TRIP
    assert float(net.sum()) == pytest_approx(gross - COST_ROUND_TRIP)


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-12)
