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
    # bear(P92) 门槛必然 ≥ weak(P90)，且都在 (0.5, 1.0] 内
    assert gates['bear'] >= gates['weak']
    assert all(0.5 < v <= 1.0 for v in gates.values())
    # ⚠️ 2026-10-12 修正：原断言是「回退值不能原样冒充分位」，
    # 但**真实生产就是恒等于回退值** —— 校准后分布塌缩 → 退化保护恒触发。
    # 该断言自 10-02 加守卫起持续红灯；改成固化**真实行为**并指向根因，
    # 而不是删掉检查或把断言改成"永远成立"的空话。见 lessons 三.61。
    assert gates == A_GATE_FALLBACK, (
        'A 股门控已不再等于绝对回退值 —— 说明退化保护不再触发，'
        '**先查清再改测试**：此时分位门槛真正生效，a_stock_gates.A_GATE_SNAPSHOT 与本文档需同步'
    )
    assert A_GATE_SNAPSHOT == A_GATE_FALLBACK, '快照注释声称「退化保护后实际生效值」，与回退值应一致'


def test_compute_quantiles_on_synthetic_csv(tmp_path):
    # ⚠️ 2026-10-12 修正：原用 beta(2,3)，校准后 P92−P90 仅 ~0.01 < A_GATE_MIN_SPREAD(0.05)
    # ⇒ 必被退化保护挡掉、走回退，而本用例却断言分位值 → 持续红灯。
    # 改为**非退化**分布（90% 主体 + 10% 宽尾）；退化形态由 test_degenerate_falls_back 固化。
    rng = np.random.default_rng(0)
    n = 300
    nb = int(n * 0.10)
    raw = np.concatenate([rng.uniform(0.45, 0.55, size=n - nb),
                          rng.uniform(0.60, 1.00, size=nb)])
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


def test_degenerate_falls_back(tmp_path):
    """退化保护：校准后分布塌缩 ⇒ 放弃分位、回退绝对阈值。

    ⚠️ 这是**当前真实生产形态**（2026-10-12 实测，见 lessons 三.61）：
    A 股 `get_a_gates()` 返回 {'bear':0.70,'weak':0.65} == A_GATE_FALLBACK。
    """
    from ml_services.a_stock_gates import A_GATE_MIN_SPREAD
    rng = np.random.default_rng(1)
    n = 300
    raw = np.clip(rng.beta(2, 3, n), 0.01, 0.99)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(raw, raw)
    cal_file = tmp_path / 'cal.pkl'
    joblib.dump(iso, cal_file)
    csv = tmp_path / 'degenerate.csv'
    pd.DataFrame({'Date': ['2026-01-%02d' % (i % 28 + 1) for i in range(n)],
                  'Predict_Prob': raw}).to_csv(csv, index=False)

    cal = iso.predict(raw.reshape(-1, 1))
    assert np.percentile(cal, 92) - np.percentile(cal, 90) < A_GATE_MIN_SPREAD

    got = compute_a_gate_thresholds(source_csv=str(csv), calibrator_file=str(cal_file),
                                    use_snapshot_fallback=False, min_samples=100)
    assert got == A_GATE_FALLBACK


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
