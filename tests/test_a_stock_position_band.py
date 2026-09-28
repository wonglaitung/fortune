# -*- coding: utf-8 -*-
"""P4.3 仓位口径 C 对齐（决策点 4 唯一口径）：
概率→单票仓位区间 + LLM 三档仓位钳制 + 信号阈值与港股一致"""
import pytest

from a_stock_recommendation_generator import (
    AStockRecommendationGenerator,
    clamp_positions_to_band,
    position_band,
)


@pytest.mark.parametrize('prob,band', [
    (0.65, (4, 6)),
    (0.60, (4, 6)),      # 边界含
    (0.59, (2, 3)),
    (0.55, (2, 3)),      # 边界含
    (0.54, (0, 2)),
    (0.51, (0, 2)),
    (0.50, (0, 0)),      # ≤0.50 禁买
    (0.42, (0, 0)),
])
def test_position_band(prob, band):
    assert position_band(prob) == band


def test_clamp_keeps_order_and_bounds():
    # LLM 给出越界且乱序 → 钳进 [2,6] 且 保守≤适度≤激进
    c, m, a = clamp_positions_to_band(10, 5, 3, 2, 6)
    assert 2 <= c <= m <= a <= 6
    # 弱信号档 [0,2]：激进也不能超 2%
    c, m, a = clamp_positions_to_band(8, 8, 8, 0, 2)
    assert (c, m, a) == (2, 2, 2)
    # 禁买档 [0,0]：全归零
    assert clamp_positions_to_band(5, 5, 5, 0, 0) == (0, 0, 0)
    # 非法输入
    assert clamp_positions_to_band(None, 'x', -3, 2, 6) == (2, 2, 2)


def _gen(prob_20d, short='买入', mid='买入'):
    g = AStockRecommendationGenerator()
    ml_pred = {'predictions': {1: {'probability': 0.5}, 5: {'probability': 0.5},
                               20: {'probability': prob_20d}}}
    llm_rec = {'short_term': short, 'mid_term': mid,
               'position_conservative': 9, 'position_moderate': 9, 'position_aggressive': 9}
    return g._generate_single_recommendation(
        '000001', {'current_price': 10.0, 'change_percent': 1.0}, ml_pred, llm_rec, {})


def test_signal_thresholds_match_gestalt_c():
    assert _gen(0.62)['signal_type'] == 'strong_buy'
    assert _gen(0.57)['signal_type'] == 'buy'
    # 0.50-0.55 弱信号 → 观望（不得列买入），与港股 5095 规则一致
    assert _gen(0.53)['signal_type'] == 'hold'
    assert _gen(0.49)['signal_type'] == 'hold'


def test_positions_constrained_by_probability_band():
    for prob, band in ((0.62, (4, 6)), (0.57, (2, 3)), (0.53, (0, 2)), (0.49, (0, 0))):
        rec = _gen(prob)
        lo, hi = band
        for key in ('position_conservative', 'position_moderate',
                    'position_aggressive', 'position_pct'):
            assert lo <= rec[key] <= hi, (prob, key, rec[key])
        assert (rec['position_conservative'] <= rec['position_moderate']
                <= rec['position_aggressive'])
