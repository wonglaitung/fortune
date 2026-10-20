# -*- coding: utf-8 -*-
"""概率口径测试（D26：probability 保持原始 P(上涨)，confidence 承载 P(方向判对)）

历史：曾把 `probability`（P(上涨)）覆写为校准值并按其 ≥0.5 重判方向——
① 20d 校准器在 p<0.741 为平台（≈0.4933）→ 32 只邮件全显示「上涨概率 0.49」，
   三重确认 0.55/0.60 门槛结构性不可达；② 原始 0.74 的强涨股被判成「↓」；
   ③ A 闸跑未校准概率、生产阈值跑校准后概率（同一性缺口）。
现口径：probability/prediction/direction 一律用模型原始输出；confidence 单独承载校准信息，
   且当某周期 confidence 当日横截面无区分度时撤销并告警（退化守卫）。
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
    # confidence 随原始概率变化（真实校准器是阶梯函数，但不应恒为常数——
    # 否则会触发 D26 退化守卫，测试就不测「正常路径」了）
    dc.confidence = lambda p, h: round(min(0.95, max(0.05, 0.30 + 0.5 * p)), 4)
    return dc


def _make_results(probs):
    """probs: {horizon: raw_prob} → 三周期结果。

    方向阈值按生产同源口径 `>= 0.5` 判涨（ml_trading_model.py:5225 `proba[1] >= 0.5`），
    D26 后 apply_to_results 不再改写方向，故夹具必须与生产者一致。
    """
    preds = {}
    for h, p in probs.items():
        preds[h] = {'prediction': 1 if p >= 0.5 else 0,
                    'probability': p,
                    'direction': '↑' if p >= 0.5 else '↓'}
    pattern = ''.join('1' if preds[h]['prediction'] == 1 else '0' for h in (5, 10, 20))
    return {'code': '2800.HK', 'predictions': preds,
            'pattern': pattern, 'pattern_info': ca.get_pattern_action(pattern)}


def test_probability_stays_raw_and_confidence_carries_calibration():
    # 2800.HK raw 0.55/0.56/0.60：probability 保持原值、方向按原始概率、confidence 另有其值
    results = {'2800.HK': _make_results({5: 0.55, 10: 0.56, 20: 0.60})}
    dc = _dc_with_stubs({5: {0.55: 0.49}, 10: {0.56: 0.43}, 20: {0.60: 0.57}})
    dc.apply_to_results(results)

    p = results['2800.HK']['predictions']
    assert p[5]['probability'] == pytest.approx(0.55)   # 不再被覆写为 0.49
    assert p[10]['probability'] == pytest.approx(0.56)
    assert p[20]['probability'] == pytest.approx(0.60)
    assert p[5]['direction'] == '↑' and p[5]['prediction'] == 1
    assert p[20]['direction'] == '↑'
    assert p[5]['confidence'] == pytest.approx(0.30 + 0.5 * 0.55)


def test_pattern_unchanged_since_direction_follows_raw():
    # D26 后方向不再被校准值改写 → pattern 保持 111，rebuild 计数 0（幂等）
    results = {'2800.HK': _make_results({5: 0.55, 10: 0.56, 20: 0.60})}
    assert results['2800.HK']['pattern'] == '111'

    dc = _dc_with_stubs({5: {0.55: 0.49}, 10: {0.56: 0.43}, 20: {0.60: 0.57}})
    dc.apply_to_results(results)
    n = ca.rebuild_three_horizon_patterns(results)

    res = results['2800.HK']
    assert res['pattern'] == '111'
    assert n == 0
    assert res['pattern_info'] == ca.get_pattern_action('111')


def test_strong_raw_signal_not_flipped_by_calibration():
    # 原始 0.74（强涨）不得因校准平台值被判成 ↓——这是 20d 邮件全 0.49 的直接教训
    results = {'0961.HK': _make_results({5: 0.74, 10: 0.74, 20: 0.74})}
    dc = _dc_with_stubs({})
    dc.confidence = lambda p, h: 0.4933          # 20d 平台值
    dc.apply_to_results(results)
    for h in (5, 10, 20):
        assert results['0961.HK']['predictions'][h]['direction'] == '↑'
        assert results['0961.HK']['predictions'][h]['probability'] == pytest.approx(0.74)


def test_degenerate_confidence_guard_removes_confidence():
    # 校准器对所有股票给同一值 → confidence 无区分度 → 撤销并告警，probability 不受影响
    results = {c: _make_results({5: 0.50, 10: 0.50, 20: 0.50}) for c in ('0005.HK', '0700.HK', '2318.HK', '0001.HK', '0388.HK')}
    dc = _dc_with_stubs({})
    dc.confidence = lambda p, h: 0.4933
    dc.apply_to_results(results)
    for res in results.values():
        for h in (5, 10, 20):
            assert 'confidence' not in res['predictions'][h]
            assert res['predictions'][h]['probability'] == pytest.approx(0.50)


def test_pattern_rebuild_no_change_when_direction_unaffected():
    # 校准后方向不变 → 模式不变（rebuilt 计数 0）
    results = {'0700.HK': _make_results({5: 0.70, 10: 0.72, 20: 0.75})}
    original = results['0700.HK']['pattern']
    dc = _dc_with_stubs({5: {0.70: 0.66}, 10: {0.72: 0.68}, 20: {0.75: 0.71}})
    dc.apply_to_results(results)
    ca.rebuild_three_horizon_patterns(results)
    assert results['0700.HK']['pattern'] == original == '111'


def test_boundary_exactly_half_is_up():
    # 原始概率恰为 0.50 → 判涨（与邮件颜色说明「↑ 50-60%」及市场调整列 ≥0.50 一致）
    results = {'0005.HK': _make_results({5: 0.50, 10: 0.50, 20: 0.50})}
    dc = _dc_with_stubs({})
    dc.apply_to_results(results)
    for h in (5, 10, 20):
        assert results['0005.HK']['predictions'][h]['direction'] == '↑'


def test_missing_probability_left_untouched():
    # 某周期概率缺失（None）→ 跳过，不误判
    results = {'9999.HK': {'code': '9999.HK',
                           'predictions': {5: {'prediction': 1, 'probability': None, 'direction': '↑'},
                                           10: {'prediction': 1, 'probability': 0.6, 'direction': '↑'},
                                           20: {'prediction': 1, 'probability': 0.7, 'direction': '↑'}},
                           'pattern': '111', 'pattern_info': ca.get_pattern_action('111')}}
    dc = _dc_with_stubs({})
    dc.apply_to_results(results)
    p = results['9999.HK']['predictions']
    assert p[5]['probability'] is None
    assert p[5]['direction'] == '↑'  # 未被误改
    assert p[10]['probability'] == 0.6  # 无校准器 → 保持原值
    assert p[10]['direction'] == '↑'  # raw 0.6 ≥0.5 → 方向不变
