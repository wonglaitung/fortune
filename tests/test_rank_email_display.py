# -*- coding: utf-8 -*-
"""D59 rank 口径展示层契约（HK 个股邮件）

rank 口径：三周期 = {5,10,20}，输出为 0-100 相对强弱分（无概率语义、无校准、无置信度）。
"""
import comprehensive_analysis as ca


def test_horizons_is_5_10_20():
    assert ca.HORIZONS == (5, 10, 20)


def test_format_hcell_rank_shows_bare_score():
    out = ca._format_hcell({'direction': '↑', 'probability': 0.73})
    assert out == '73'
    assert '<span' not in out
    assert '置信' not in out


def test_format_hcell_none_returns_dash():
    assert ca._format_hcell(None) == '-'


def test_rebuild_patterns_uses_5_10_20_keys():
    th = {'0700.HK': {'predictions': {
        5: {'prediction': 1}, 10: {'prediction': 0}, 20: {'prediction': 1}}}}
    n = ca.rebuild_three_horizon_patterns(th)
    assert n == 1
    assert th['0700.HK']['pattern'] == '101'
