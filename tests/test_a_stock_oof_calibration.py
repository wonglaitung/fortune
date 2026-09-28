# -*- coding: utf-8 -*-
"""A股 OOF 概率校准（P3.1 / 决策点2）

背景：A股 prediction_history 仅 132 条且 2026-09-27 停回写，生产校准器（MIN_SAMPLES=200）
拟不出来 → 决策点2 拍板改用 walk-forward OOF 预测拟合 Isotonic（PIT、样本足）。
方向必须与校准概率同口径（lessons 三.19，避免「↑ 0.49」自相矛盾）。
"""
import json
import os

import pandas as pd
import pytest

from ml_services.daily_confidence import DailyConfidence
import a_stock_comprehensive_analysis as asca


def _write_oof_csv(path, n=400, flip=0.0):
    """合成 OOF CSV（新导出格式：Predict_Prob/Actual_Direction/Is_Correct）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for i in range(n):
        p = 0.3 + 0.4 * (i % 5) / 4.0
        up = int(p > 0.5) if flip == 0 else 1 - int(p > 0.5)
        rows.append({
            'Fold': i % 19 + 1,
            'Date': f'2025-{(i % 12) + 1:02d}-02 00:00:00+00:00',
            'Stock_Code': f'{600000 + i:06d}',
            'Predict_Prob': p,
            'Predict_Direction': 'UP' if p >= 0.5 else 'DOWN',
            'Actual_Return': 0.01 if up else -0.01,
            'Actual_Direction': 'UP' if up else 'DOWN',
            'Is_Correct': bool(up == (p >= 0.5)),
            'Market_Layer': 'normal',
            'Dynamic_Threshold': 0.5,
            'Market_Up_Ratio': 0.5,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_oof_reads_new_format_and_fits(tmp_path):
    csv = tmp_path / 'x_a_stock_catboost_20d' / 'prediction_analysis.csv'
    _write_oof_csv(str(csv))

    dc = DailyConfidence(cal_prefix='a_stock_',
                         oof_glob=str(tmp_path / '*_a_stock_catboost_{h}d' / 'prediction_analysis.csv'),
                         min_samples=100, cal_dir=str(tmp_path / 'cal'))

    df = dc._oof(20)
    assert len(df) == 400
    assert {'prob', 'up', 'correct'} <= set(df.columns)
    assert dc.prob_cal[20] is not None and dc.conf_cal[20] is not None

    # 校准映射单调且落在 [0,1]，高概率 → 高校准值
    lo, hi = dc.calibrate(0.3, 20), dc.calibrate(0.7, 20)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi >= lo

    # 快照元数据落盘（写明数据源，重跑 walk-forward 须再拟合）
    meta = json.loads((tmp_path / 'cal' / 'a_stock_cal_meta_20.json').read_text(encoding='utf-8'))
    assert meta['n_samples'] == 400
    assert meta['source_file'].endswith('prediction_analysis.csv')


def test_missing_horizon_passthrough(tmp_path):
    """1d/5d CSV 未产出时该周期不拟合 → calibrate 透传原值"""
    csv = tmp_path / 'x_a_stock_catboost_20d' / 'prediction_analysis.csv'
    _write_oof_csv(str(csv))
    dc = DailyConfidence(cal_prefix='a_stock_',
                         oof_glob=str(tmp_path / '*_a_stock_catboost_{h}d' / 'prediction_analysis.csv'),
                         min_samples=100, cal_dir=str(tmp_path / 'cal'))
    assert dc.prob_cal.get(1) is None
    assert dc.calibrate(0.62, 1) == 0.62   # 透传


def test_calibrate_prefix_isolates_from_hk(tmp_path):
    """A股校准器文件名带 a_stock_ 前缀，不覆盖港股校准器"""
    csv = tmp_path / 'x_a_stock_catboost_20d' / 'prediction_analysis.csv'
    _write_oof_csv(str(csv))
    cal_dir = tmp_path / 'cal'
    cal_dir.mkdir()
    (cal_dir / 'prob_cal_20.pkl').write_bytes(b'HK_SENTINEL')

    DailyConfidence(cal_prefix='a_stock_',
                    oof_glob=str(tmp_path / '*_a_stock_catboost_{h}d' / 'prediction_analysis.csv'),
                    min_samples=100, cal_dir=str(cal_dir))
    # 港股 20d 校准器未被改写
    assert (cal_dir / 'prob_cal_20.pkl').read_bytes() == b'HK_SENTINEL'
    assert (cal_dir / 'a_stock_prob_cal_20.pkl').exists()


def test_direction_follows_calibrated_probability():
    """方向按校准后概率重判（三.19）：raw 0.55 → 校准 0.49 必须显示 ↓"""
    class _Stub:
        def calibrate(self, p, h):
            return {20: 0.49}.get(h, p)

    prob, direction = asca.calibrate_probability(_Stub(), 0.55, 20)
    assert prob == pytest.approx(0.49)
    assert direction == '↓'

    prob, direction = asca.calibrate_probability(_Stub(), 0.55, 5)   # 5d 无映射
    assert prob == pytest.approx(0.55) and direction == '↑'

    prob, direction = asca.calibrate_probability(None, 0.40, 20)     # 无校准器
    assert prob == pytest.approx(0.40) and direction == '↓'
