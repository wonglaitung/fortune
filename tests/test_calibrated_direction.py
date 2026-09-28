# -*- coding: utf-8 -*-
"""校准后方向/模式同口径 测试

背景：邮件三周期列表曾出现「↑ 0.49 (置信52%)」矛盾——箭头按原始概率判、
数字按 Isotonic 校准概率显示（apply_to_results 只改 probability 不改 direction），
模式(111)也仍按原始方向拼，与概率列矛盾。
修复：apply_to_results 同步重判 prediction/direction（校准后 ≥0.5 判涨），
校准后 rebuild_three_horizon_patterns 重算模式/交易建议/胜率。
"""
import pytest

import comprehensive_analysis as ca
from ml_services.daily_confidence import DailyConfidence


def _dc_with_stubs(calib_map):
    """绕过 __init__（避免加载/拟合 pkl），注入可控的校准映射"""
    dc = DailyConfidence.__new__(DailyConfidence)
    dc.prob_cal = {}
    dc.conf_cal = {}
    dc.calibrate = lambda p, h: calib_map.get(h, {}).get(p, p)
    dc.confidence = lambda p, h: 0.52
    return dc


def _make_results(probs):
    """probs: {horizon: raw_prob} → 三周期结果（raw 方向按 >0.5 判）"""
    preds = {}
    for h, p in probs.items():
        preds[h] = {'prediction': 1 if p > 0.5 else 0,
                    'probability': p,
                    'direction': '↑' if p > 0.5 else '↓'}
    pattern = ''.join('1' if preds[h]['prediction'] == 1 else '0' for h in (1, 5, 20))
    return {'code': '2800.HK', 'predictions': preds,
            'pattern': pattern, 'pattern_info': ca.get_pattern_action(pattern)}


def test_direction_follows_calibrated_probability():
    # 用户实例：2800.HK raw 全部 >0.5（原始判涨），校准后 0.49/0.43/0.57
    results = {'2800.HK': _make_results({1: 0.55, 5: 0.56, 20: 0.60})}
    dc = _dc_with_stubs({1: {0.55: 0.49}, 5: {0.56: 0.43}, 20: {0.60: 0.57}})
    dc.apply_to_results(results)

    p = results['2800.HK']['predictions']
    assert p[1]['probability'] == pytest.approx(0.49)
    assert p[1]['direction'] == '↓'          # 不再出现 ↑ 0.49
    assert p[1]['prediction'] == 0
    assert p[5]['direction'] == '↓'
    assert p[20]['direction'] == '↑'          # 0.57 ≥ 0.5
    assert p[1]['confidence'] == pytest.approx(0.52)


def test_calibrated_pattern_rebuilt():
    # 重建前 pattern=111（raw 方向），重建后应随校准方向变 001
    results = {'2800.HK': _make_results({1: 0.55, 5: 0.56, 20: 0.60})}
    assert results['2800.HK']['pattern'] == '111'

    dc = _dc_with_stubs({1: {0.55: 0.49}, 5: {0.56: 0.43}, 20: {0.60: 0.57}})
    dc.apply_to_results(results)
    n = ca.rebuild_three_horizon_patterns(results)

    res = results['2800.HK']
    assert res['pattern'] == '001'
    assert n == 1
    assert res['pattern_info'] == ca.get_pattern_action('001')


def test_pattern_rebuild_no_change_when_direction_unaffected():
    # 校准后方向不变 → 模式不变（rebuilt 计数 0）
    results = {'0700.HK': _make_results({1: 0.70, 5: 0.72, 20: 0.75})}
    original = results['0700.HK']['pattern']
    dc = _dc_with_stubs({1: {0.70: 0.66}, 5: {0.72: 0.68}, 20: {0.75: 0.71}})
    dc.apply_to_results(results)
    ca.rebuild_three_horizon_patterns(results)
    assert results['0700.HK']['pattern'] == original == '111'


def test_boundary_exactly_half_is_up():
    # 校准概率恰为 0.50 → 判涨（与邮件颜色说明「↑ 50-60%」及市场调整列 ≥0.50 一致）
    results = {'0005.HK': _make_results({1: 0.48, 5: 0.48, 20: 0.48})}
    dc = _dc_with_stubs({1: {0.48: 0.50}, 5: {0.48: 0.50}, 20: {0.48: 0.50}})
    dc.apply_to_results(results)
    for h in (1, 5, 20):
        assert results['0005.HK']['predictions'][h]['direction'] == '↑'


def test_missing_probability_left_untouched():
    # 某周期概率缺失（None）→ 跳过，不误判
    results = {'9999.HK': {'code': '9999.HK',
                           'predictions': {1: {'prediction': 1, 'probability': None, 'direction': '↑'},
                                           5: {'prediction': 1, 'probability': 0.6, 'direction': '↑'},
                                           20: {'prediction': 1, 'probability': 0.7, 'direction': '↑'}},
                           'pattern': '111', 'pattern_info': ca.get_pattern_action('111')}}
    dc = _dc_with_stubs({})
    dc.apply_to_results(results)
    p = results['9999.HK']['predictions']
    assert p[1]['probability'] is None
    assert p[1]['direction'] == '↑'  # 未被误改
    assert p[5]['probability'] == 0.6  # 无校准器 → 保持原值
    assert p[5]['direction'] == '↑'  # raw 0.6 ≥0.5 → 方向不变
