"""策略评估单一实现（ml_services/strategy_eval.py）—— D45。"""

import numpy as np
import pandas as pd

from ml_services.strategy_eval import (
    daily_mark, static_matched, metrics_daily, mde_ir, has_timing_increment)


def test_daily_mark_no_lookahead():
    idx = pd.bdate_range('2020-01-01', periods=5)
    ret = pd.Series([0.0, 0.10, 0.0, 0.0, 0.0], index=idx)   # 第 2 日大涨
    w = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=idx)      # 第 2 日决定持
    net = daily_mark(ret, w, round_trip_cost=0.0, lag=1)
    assert net.iloc[1] == 0.0            # T 决定、T+1 生效 ⇒ 不吃第 2 日大涨
    assert net.iloc[2] == 0.0            # 第 3 日无收益


def test_static_matched_is_constant():
    idx = pd.bdate_range('2020-01-01', periods=4)
    ret = pd.Series([0.01, -0.02, 0.03, 0.0], index=idx)
    st = static_matched(ret, 0.5)
    assert np.allclose(st.values, 0.5 * ret.values)


def test_mde_formula():
    assert abs(mde_ir(39, 20) - 2.8 * (252 / 780) ** 0.5) < 1e-9
    assert mde_ir(156, 20) < mde_ir(39, 20)
    assert np.isnan(mde_ir(0, 20))


def test_metrics_daily_signs():
    net = pd.Series([0.01, -0.05, 0.02], index=pd.bdate_range('2020-01-01', periods=3))
    m = metrics_daily(net)
    assert m['maxdd'] < 0


def test_has_timing_increment_true_for_shallow_dd():
    idx = pd.bdate_range('2020-01-01', periods=8)
    ret = pd.Series([0.01, 0.01, -0.30, 0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    w = pd.Series([1, 0, 0, 1, 1, 1, 1, 1], index=idx)      # T-1 决定、崩盘日空仓（lag=1）
    sig = daily_mark(ret, w, round_trip_cost=0.0)
    st = static_matched(ret, float(w.mean()))
    assert has_timing_increment(sig, st) is True
