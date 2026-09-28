# -*- coding: utf-8 -*-
"""P3.2 A股分位门槛（D8 同款）：bear/weak = 校准概率 P92/P90 分位 + 回退链"""
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression

from ml_services.a_stock_gates import (
    A_GATE_FALLBACK, A_GATE_SNAPSHOT, compute_a_gate_thresholds, get_a_gates,
)


def test_runtime_gates_shape_and_order():
    gates = get_a_gates()
    assert set(gates) == {'bear', 'weak'}
    # bear(P92) 门槛必然 ≥ weak(P90)，且都在 (0.5, 1.0) 内
    assert gates['bear'] >= gates['weak']
    assert all(0.5 < v <= 1.0 for v in gates.values())
    # 回退值不能原样冒充分位（分位与绝对值应不同）
    assert gates['bear'] != A_GATE_FALLBACK['bear'] or gates['weak'] != A_GATE_FALLBACK['weak']


def test_compute_quantiles_on_synthetic_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    raw = np.clip(rng.beta(2, 3, n), 0.01, 0.99)
    # 单调校准器（无需真数据）：prob -> prob^0.8 用 isotonic 拟合 identity 即可
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(raw, raw)
    cal_file = tmp_path / 'cal.pkl'
    joblib.dump(iso, cal_file)

    csv = tmp_path / 'prediction_analysis.csv'
    pd.DataFrame({
        'Date': ['2026-01-%02d' % (i % 28 + 1) for i in range(n)],
        'Predict_Prob': raw,
    }).to_csv(csv, index=False)

    gates = compute_a_gate_thresholds(
        source_csv=str(csv), calibrator_file=str(cal_file),
        use_snapshot_fallback=False, min_samples=100)

    cal = iso.predict(raw.reshape(-1, 1))
    assert abs(gates['bear'] - np.percentile(cal, 92)) < 1e-9
    assert abs(gates['weak'] - np.percentile(cal, 90)) < 1e-9
    assert gates['bear'] >= gates['weak']


def test_fallback_chain(tmp_path):
    # 源缺失 → 内嵌快照（不回落到绝对值）
    got = compute_a_gate_thresholds(source_csv=str(tmp_path / 'nope.csv'),
                                    calibrator_file=str(tmp_path / 'nope.pkl'))
    assert got == A_GATE_SNAPSHOT

    # 样本不足 + 关闭快照回退 → 绝对值
    raw = np.full(10, 0.6)
    csv = tmp_path / 'small.csv'
    pd.DataFrame({'Date': ['2026-01-01'] * 10, 'Predict_Prob': raw}).to_csv(csv, index=False)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(raw, raw)
    cal_file = tmp_path / 'cal.pkl'
    joblib.dump(iso, cal_file)
    got = compute_a_gate_thresholds(source_csv=str(csv), calibrator_file=str(cal_file),
                                    use_snapshot_fallback=False, min_samples=100)
    assert got == A_GATE_FALLBACK
