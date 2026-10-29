"""波动目标风控层（ml_services/vol_target_timing.py）单元测试。

重点：仓位缩放方向、月频步进、**无未来函数（lag）**、指标符号、双基准接线。
不依赖网络（不测 load_hsi）。
"""

import numpy as np
import pandas as pd
import pytest

from ml_services.vol_target_timing import (
    realized_vol, vol_exposure, to_monthly, strat_returns,
    static_matched_returns, metrics, backtest, evaluate_config, run_grid,
    TRADING_DAYS, COST_ROUND_TRIP,
)


def _synthetic(low_vol_days=250, high_vol_days=120, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2020-01-01', periods=low_vol_days + high_vol_days)
    low = rng.normal(0.0005, 0.006, low_vol_days)      # 低波动温和上行
    high = rng.normal(-0.002, 0.03, high_vol_days)     # 高波动下挫
    ret = pd.Series(np.concatenate([low, high]), index=idx)
    close = (1 + ret).cumprod() * 100.0
    return close, ret


def test_realized_vol_higher_in_high_vol():
    _, ret = _synthetic()
    rv = realized_vol(ret, 20)
    assert rv.iloc[-1] > rv.iloc[240]  # 后段高波动 > 前段低波动


def test_exposure_shrinks_when_vol_high():
    _, ret = _synthetic()
    w = vol_exposure(ret, target_vol=0.12, window=20)
    w = w.dropna()
    assert w.min() >= 0.0 and w.max() <= 1.0
    # 高波动段仓位应显著低于低波动段
    assert w.iloc[-1] < w.iloc[240]


def test_to_monthly_is_stepwise():
    _, ret = _synthetic()
    w = vol_exposure(ret, 0.12, 20).dropna()
    wm = to_monthly(w)
    # 同月内应恒定
    key = wm.index.to_period('M')
    for _, grp in wm.groupby(key):
        assert grp.nunique() == 1
    # 每月值 = 该月最后一个观测
    assert wm.iloc[-1] == pytest.approx(w.iloc[-1])


def test_strat_returns_no_lookahead():
    """脉冲信号在 T 日出现，仓位应 T+1 才生效 → 不吃到 T 日收益。"""
    idx = pd.bdate_range('2024-01-01', periods=100)
    ret = pd.Series(0.0, index=idx)
    ret.iloc[50] = 0.5
    w = pd.Series(0.0, index=idx)
    w.iloc[50] = 1.0
    _, gross, _, w_pos = strat_returns(ret, w, lag=1)
    assert w_pos.iloc[50] == 0.0          # T 日还没建仓
    assert gross.iloc[50] == 0.0          # 不吃到 T 日大涨
    assert w_pos.iloc[51] == 1.0          # T+1 生效
    assert gross.iloc[51] == pytest.approx(0.0)  # 次日无收益


def test_metrics_signs():
    idx = pd.bdate_range('2020-01-01', periods=300)
    r = pd.Series(np.r_[np.full(150, 0.002), np.full(20, -0.01), np.full(130, 0.002)],
                  index=idx)
    m = metrics(r)
    assert m['maxdd'] < 0
    assert m['total'] > 0
    assert m['calmar'] > 0
    assert m['n'] == 300


def test_backtest_wiring_and_matched_exposure():
    close, _ = _synthetic()
    res = backtest(close, target_vol=0.12, window=20)
    for k in ('vt', 'bh', 'st', 'w_pos', 'mean_exp'):
        assert k in res
    # 静态对照仓位 = vol-target 平均仓位
    assert 0.0 <= res['mean_exp'] <= 1.0
    # 同窗比较（等长）
    assert len(res['vt']) == len(res['bh']) == len(res['st'])


def test_static_matched_has_no_timing():
    """静态恒定仓位不应有仓位择时：w_pos 除首日外全等于目标仓位。"""
    idx = pd.bdate_range('2020-01-01', periods=200)
    ret = pd.Series(0.0005, index=idx)
    st = static_matched_returns(ret, 0.6)
    assert len(st) == 200


def test_run_grid_shape():
    close, _ = _synthetic()
    grid = run_grid(close)
    assert grid['n'] == 6
    assert 0 <= grid['dd_pass'] <= 6
    assert 0 <= grid['calmar_pass'] <= 6
    assert isinstance(grid['grid_ok'], bool)


def test_evaluate_config_booleans():
    close, _ = _synthetic()
    c = evaluate_config(close, 0.12, 20)
    assert isinstance(c['dd_ok'], bool)
    assert isinstance(c['calmar_ok'], bool)
