# -*- coding: utf-8 -*-
"""A股排序分数 = 交叉参考：分数只给仓位上限，不作为买卖否决项"""
import pytest

from a_stock_recommendation_generator import (
    AStockRecommendationGenerator,
    clamp_positions_to_band,
    position_cap,
)


@pytest.mark.parametrize('score,cap', [
    (0.65, 6),
    (0.60, 6),
    (0.59, 3),
    (0.55, 3),
    (0.54, 2),
    (0.51, 2),
    (0.50, 2),      # 弱分不归零：仍允许 ≤2%
    (0.42, 2),
])
def test_position_cap(score, cap):
    assert position_cap(score) == cap


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


def test_signal_thresholds_cross_reference():
    assert _gen(0.62)['signal_type'] == 'strong_buy'
    assert _gen(0.57)['signal_type'] == 'buy'
    # 分数中性/偏弱不否决主判据：仍列买入，只是仓位收紧
    assert _gen(0.53)['signal_type'] == 'buy'
    assert _gen(0.49)['signal_type'] == 'buy'


def test_weak_score_does_not_veto_buy():
    rec = _gen(0.49)
    assert rec['position_pct'] <= 2
    assert '分歧' in rec['reason']


def test_positions_constrained_by_score_cap():
    for score, cap in ((0.62, 6), (0.57, 3), (0.53, 2), (0.49, 2)):
        rec = _gen(score)
        for key in ('position_conservative', 'position_moderate',
                    'position_aggressive', 'position_pct'):
            assert 0 <= rec[key] <= cap, (score, key, rec[key])
        assert (rec['position_conservative'] <= rec['position_moderate']
                <= rec['position_aggressive'])
