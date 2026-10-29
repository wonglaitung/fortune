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


def test_side_cost_model_components():
    from ml_services.costs import SideCostModel
    m = SideCostModel(broker_rate=0.001, min_fee=50.0, impact_k=1.0)
    c = m.per_side(50000, adv=1e8, sigma=0.02, spread=0.002)
    assert c['stamp'] == pytest_approx(0.001)
    assert c['commission'] == pytest_approx(0.001)
    assert c['half_spread'] == pytest_approx(0.001)
    assert 0 < c['impact'] < 0.05
    # 最低佣金：单笔 5000 → 佣金 1%
    small = m.per_side(5000, adv=1e8, sigma=0.02, spread=0.002)
    assert small['commission'] == pytest_approx(0.01)
    # ETF 免印花税
    etf = m.per_side(50000, adv=1e8, sigma=0.02, spread=0.002, is_etf=True)
    assert etf['stamp'] == 0.0


def test_hk_tick_size_bands():
    from ml_services.costs import hk_tick_size, tick_spread
    assert hk_tick_size(4.0) == 0.01
    assert hk_tick_size(50) == 0.05
    assert hk_tick_size(387) == 0.2
    assert tick_spread(387, floor=0.0) == pytest_approx(0.2 / 387)
    assert tick_spread(1.0) == pytest_approx(0.01)  # 1% 但上限 2% 内
