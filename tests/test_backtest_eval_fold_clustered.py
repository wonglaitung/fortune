# -*- coding: utf-8 -*-
"""backtest_eval 折聚类显著性测试：

行级 binomial p（p_value_vs_baseline）把每行当独立观测，忽略折间相关
（同折共享模型、同日横截面相关），有效独立观测 ≈ 折数而非行数，
实测行级 p=0.0044 → 折聚类 p=0.167（伪显著）。见 lessons.md 三.45。

本文件锁住：折聚类 p 必须被计算并输出，且数值与 scipy 单样本 t 一致。
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from ml_services.backtest_eval import (
    MIN_FOLDS_FOR_TEST, _fold_clustered_lift, evaluate_win_rate,
)


def _synth(n_folds, rows_per_fold=200, seed=42, lift=0.0, n_trade=100):
    """构造带 fold / _trade / _win 的合成数据（满足每折行数门槛）"""
    rng = np.random.default_rng(seed)
    out = []
    for f in range(1, n_folds + 1):
        trade = np.zeros(rows_per_fold, dtype=bool)
        trade[:n_trade] = True
        # 基准组胜率 0.5，交易组 0.5+lift（clip 保证合法）
        p_win = np.where(trade, min(0.5 + lift, 0.999), 0.5)
        win = rng.random(rows_per_fold) < p_win
        out.append({
            'fold': f,
            'prob': np.where(trade, 0.7, 0.3),
            'actual_return': np.where(win, 0.02, -0.02),
            '_trade': trade,
            '_win': win,
        })
    return pd.concat([pd.DataFrame(o) for o in out], ignore_index=True)


def test_insufficient_folds_returns_none():
    assert _fold_clustered_lift(_synth(MIN_FOLDS_FOR_TEST - 1)) is None


def test_missing_fold_column_returns_none():
    assert _fold_clustered_lift(_synth(10).drop(columns=['fold'])) is None


def test_matches_scipy_one_sample_t():
    d = _synth(19, lift=0.15)
    fc = _fold_clustered_lift(d)
    assert fc is not None and fc['p_value'] is not None

    lifts = [
        g.loc[g['_trade'], '_win'].mean() - g['_win'].mean()
        for _, g in d.groupby('fold')
    ]
    arr = np.asarray(lifts, dtype=float)
    t, p = sp_stats.ttest_1samp(arr, 0.0)
    assert fc['n_folds'] == len(arr)
    assert fc['lift'] == pytest.approx(arr.mean())
    assert fc['t'] == pytest.approx(t)
    assert fc['p_value'] == pytest.approx(p)


def test_rows_below_threshold_filtered():
    """每折行数 < MIN_FOLD_ROWS 时该折不计入"""
    d = _synth(19, rows_per_fold=20, n_trade=10)
    assert _fold_clustered_lift(d) is None


def test_zero_variance_flagged_insufficient():
    d = pd.DataFrame({
        'fold': np.repeat(np.arange(1, 11), 80),
        'prob': 0.7,
        'actual_return': 0.02,
    })
    d['_trade'] = d['prob'] >= 0.5
    d['_win'] = d['actual_return'] > 0.005
    fc = _fold_clustered_lift(d)
    assert fc is not None
    assert fc['p_value'] is None
    assert fc['reliability'] == 'insufficient'


def test_evaluate_exposes_fold_clustered():
    wr = evaluate_win_rate(_synth(19, lift=0.15), horizon=20)
    assert wr is not None
    assert 'fold_clustered' in wr
    fc = wr['fold_clustered']
    assert fc is not None
    assert fc['n_folds'] >= MIN_FOLDS_FOR_TEST
    assert fc['p_value'] is not None
    assert fc['ci_low'] <= fc['lift'] <= fc['ci_high']


def test_row_level_and_fold_p_are_different_objects():
    """行级 p 与折聚类 p 必须作为两个独立字段存在，不可互相覆盖"""
    wr = evaluate_win_rate(_synth(19, lift=0.15), horizon=20)
    assert wr['p_value_vs_baseline'] is not None
    assert wr['fold_clustered']['p_value'] is not None
    assert wr['lift'] is not None
    assert wr['fold_clustered']['lift'] is not None
