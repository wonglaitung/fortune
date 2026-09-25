# -*- coding: utf-8 -*-
"""分位门槛（market_regime b 方案）测试：
- bear/weak 门槛按回测校准概率分布分位动态计算（P92/P90）
- PIT：只用 Date <= as_of 的记录
- 样本不足/数据缺失回退绝对值 0.70/0.65
- MarketSentimentFilter.prepare 集成：bear 层拿到分位值而非写死的 0.70
- 数据源必须是 walk-forward 回测 CSV（prediction_history 右尾过窄已实测证伪）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_services.market_regime import (
    GATE_QUANTILES, GATE_FALLBACK, GATE_MIN_SAMPLES, GATE_SNAPSHOT,
    compute_gate_thresholds, MarketSentimentFilter,
)


def _make_walkforward_csv(path: Path, n=400, start="2025-01-01"):
    """构造 output/*_catboost_20d/prediction_analysis.csv 结构"""
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, periods=n)
    probs = np.linspace(0.40, 0.90, n)
    pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Predict_Prob": probs,
    }).to_csv(path, index=False)
    return [d.strftime("%Y-%m-%d") for d in dates], probs


def _make_identity_calibrator(path: Path):
    from sklearn.isotonic import IsotonicRegression
    import joblib
    x = np.linspace(0, 1, 11)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(x, x)
    joblib.dump(iso, path)


def _csv_in(tmp_path: Path) -> Path:
    return tmp_path / "output" / "20250101_000000_catboost_20d" / "prediction_analysis.csv"


def test_gate_quantiles_config():
    # 配置与 prompt 文案（前约8%/前约10%）一致性防漂移
    assert GATE_QUANTILES == {"bear": 0.92, "weak": 0.90}
    assert GATE_FALLBACK == {"bear": 0.70, "weak": 0.65}


def test_compute_gates_quantile_and_pit(tmp_path):
    csv = _csv_in(tmp_path)
    iso_path = tmp_path / "iso.pkl"
    dates, probs = _make_walkforward_csv(csv, n=400)
    _make_identity_calibrator(iso_path)

    # 全量：分位值 = 线性概率分布的 P92/P90
    gates = compute_gate_thresholds(as_of=None, source_csv=str(csv),
                                    calibrator_file=str(iso_path))
    assert gates["bear"] == pytest.approx(np.percentile(probs, 92), abs=1e-6)
    assert gates["weak"] == pytest.approx(np.percentile(probs, 90), abs=1e-6)
    assert gates["bear"] > gates["weak"] > 0.50

    # PIT：as_of 中段 → 分位随可用样本收缩
    gates_mid = compute_gate_thresholds(as_of=dates[199], source_csv=str(csv),
                                        calibrator_file=str(iso_path))
    assert gates_mid["bear"] < gates["bear"]

    # PIT：as_of 太早 → 样本不足 → 回退链到内嵌快照（CI 同值）
    gates_early = compute_gate_thresholds(as_of=dates[99], source_csv=str(csv),
                                          calibrator_file=str(iso_path))
    assert gates_early == pytest.approx(GATE_SNAPSHOT, abs=1e-9)

    # 关闭快照回退 → 绝对值
    gates_abs = compute_gate_thresholds(as_of=dates[99], source_csv=str(csv),
                                        calibrator_file=str(iso_path),
                                        use_snapshot_fallback=False)
    assert gates_abs == GATE_FALLBACK


def test_compute_gates_fallback_chain(tmp_path):
    # 无 CSV → 内嵌快照（CI 行为）
    gates = compute_gate_thresholds(source_csv=str(tmp_path / "nope.csv"),
                                    calibrator_file=str(tmp_path / "nope.pkl"))
    assert gates == pytest.approx(GATE_SNAPSHOT, abs=1e-9)
    # 快照被清空 → 最终回退绝对值
    import ml_services.market_regime as mr
    saved, mr.GATE_SNAPSHOT = mr.GATE_SNAPSHOT, {}
    try:
        assert compute_gate_thresholds(source_csv=str(tmp_path / "nope.csv"),
                                       calibrator_file=str(tmp_path / "nope.pkl")) == GATE_FALLBACK
    finally:
        mr.GATE_SNAPSHOT = saved


def test_latest_source_excludes_a_stock(tmp_path):
    """CI 视角：a_stock 目录虽被 glob 命中，但必须被正则排除，选中港股最新目录"""
    import ml_services.market_regime as mr
    hk = _csv_in(tmp_path)  # output/20250101_000000_catboost_20d/...
    _make_walkforward_csv(hk, n=400)
    a_share = (tmp_path / "output" / "20260722_181211_a_stock_catboost_20d"
               / "prediction_analysis.csv")
    _make_walkforward_csv(a_share, n=400, start="2024-01-01")
    monkey = tmp_path / "output" / "*" / "prediction_analysis.csv"
    saved = mr._GATE_QUANTILE_GLOB
    mr._GATE_QUANTILE_GLOB = str(monkey)
    try:
        picked = mr._latest_gate_source_csv()
        assert picked is not None and "_a_stock_" not in picked
        assert picked.endswith(str(hk.relative_to(tmp_path))) or picked == str(hk)
    finally:
        mr._GATE_QUANTILE_GLOB = saved


def test_filter_prepare_uses_quantile_gate(tmp_path, monkeypatch):
    from ml_services import market_regime as mr

    csv = _csv_in(tmp_path)
    iso_path = tmp_path / "iso.pkl"
    _make_walkforward_csv(csv, n=400)
    _make_identity_calibrator(iso_path)
    monkeypatch.setattr(mr, "_GATE_QUANTILE_GLOB", str(tmp_path / "output" / "*" / "prediction_analysis.csv"))
    monkeypatch.setattr(mr, "_GATE_CALIBRATOR_FILE", str(iso_path))

    # 构造 bear 市场（上涨比例 0.25，位于 20-30% 区间）
    idx = pd.bdate_range("2026-06-01", periods=30)
    rows = []
    for d in idx:
        for i in range(8):
            rows.append({"Date": d, "Return_1d": 0.01 if i < 2 else -0.01})
    returns_df = pd.DataFrame(rows)

    f = MarketSentimentFilter(lookback_days=0)
    f.prepare_market_schedule(returns_df)
    thr, layer, ratio = f.get_threshold("2026-06-15")
    assert layer == "bear"
    assert ratio == pytest.approx(0.25)
    # 门槛应等于同一 as_of 下的 PIT 分位值（而非写死的 0.70）
    expected = compute_gate_thresholds(as_of="2026-06-15", source_csv=str(csv),
                                       calibrator_file=str(iso_path))["bear"]
    assert thr == pytest.approx(expected, abs=1e-9)
    assert thr != pytest.approx(0.70)

    # 关闭分位开关 → 回退 DEFAULT_LAYERS 绝对值
    f2 = MarketSentimentFilter(lookback_days=0, use_quantile_gates=False)
    f2.prepare_market_schedule(returns_df)
    thr2, layer2, _ = f2.get_threshold("2026-06-15")
    assert layer2 == "bear"
    assert thr2 == pytest.approx(0.70)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
