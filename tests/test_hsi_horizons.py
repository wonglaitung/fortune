# -*- coding: utf-8 -*-
"""D59 §十八 恒指三周期契约：周期由 1/5/20 改为 5/10/20（标签仍为绝对二分类）。

恒指是单标的，无横截面 ⇒ 不用 rank 标签；仅改预测周期。
"""
import hsi_prediction as hp


def test_horizons_is_5_10_20():
    assert hp.HORIZONS == (5, 10, 20)


def test_no_legacy_horizon_literal():
    src = open(hp.__file__, encoding='utf-8').read()
    assert '[1, 5, 20]' not in src
    assert 'pred_1d' not in src


def test_three_horizon_pattern_stats_keys_documented():
    assert hasattr(hp.HSI_Predictor, '_calculate_three_horizon_pattern_stats')
    assert hasattr(hp.HSI_Predictor, '_check_hsi_transmission_mode')
