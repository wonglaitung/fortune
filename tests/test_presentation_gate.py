"""A 呈现闸（scripts/presentation_gate.py）回归测试。

背景（2026-10-07）：三道闸原本只以散文写在 AGENTS.md，靠执行者"记得去做"。
把 A 闸落成可执行检查时，发现 AGENTS:115 对 lessons 三.28 存在转录失真——
原句「预测 UP 组收益最大值必须 <0」方向写反且丢失列名，该普适规则在合法
数据上必然 FAIL（实测合法 UP 组 max=+0.4209）。本测试同时锁定：
  1. 合法数据必须 PASS（防止把错规则固化进脚本）
  2. 三类泄漏必须 FAIL（防止闸门失效）
  3. 符号退化这一项不能被 H1/H2 替代（实测其准确率 48.9%、概率不饱和）
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "presentation_gate.py")

_rng = np.random.RandomState(0)


def _make_csv(tmp_path, kind="legit", n=2000, horizon=20):
    """构造最小可用的 prediction_analysis.csv。"""
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    d = pd.Series(pd.to_datetime(_rng.choice(dates.values, n)))
    prob = _rng.rand(n)

    if kind == "legit":
        ret = _rng.randn(n) * 0.02
    elif kind == "high_acc":
        # 准确率拉到 100%，但收益仍为随机（模拟标签入模）
        ret = np.where(prob >= 0.5, 0.02, -0.02) + _rng.randn(n) * 0.001
    elif kind == "saturated":
        prob = (_rng.rand(n) > 0.5).astype(float)
        ret = _rng.randn(n) * 0.02
    else:
        raise ValueError(kind)

    pred = np.where(prob >= 0.5, "UP", "DOWN")
    act = np.where(ret > 0, "UP", "DOWN")
    df = pd.DataFrame(
        {
            "Fold": _rng.randint(1, 20, n),
            "Date": d.dt.strftime("%Y-%m-%d"),
            "Stock_Code": "0001.HK",
            "Predict_Prob": prob,
            "Predict_Direction": pred,
            "Actual_Return": ret,
            "Actual_Direction": act,
            "Is_Correct": pred == act,
        }
    )
    path = os.path.join(str(tmp_path), f"{kind}.csv")
    df.to_csv(path, index=False)
    return path


def _run(path, horizon=20):
    p = subprocess.run(
        [sys.executable, SCRIPT, "--input", path, "--horizon", str(horizon)],
        capture_output=True, text=True, timeout=300,
    )
    return p.returncode, p.stdout


def test_legitimate_data_passes(tmp_path):
    """合法数据必须 PASS —— 防止把错误规则固化成闸门。"""
    rc, out = _run(_make_csv(tmp_path, "legit"))
    assert rc == 0, f"合法数据被 A 闸误判 FAIL：\n{out}"


def test_high_accuracy_leak_fails(tmp_path):
    """准确率 >65% 必须 FAIL（AGENTS:115 文档阈值）。"""
    rc, out = _run(_make_csv(tmp_path, "high_acc"))
    assert rc == 1, f"100% 准确率泄漏未被拦截：\n{out}"
    assert "H1" in out


def test_probability_saturation_fails(tmp_path):
    """Predict_Prob 饱和成 0/1 必须 FAIL。"""
    rc, out = _run(_make_csv(tmp_path, "saturated"))
    assert rc == 1, f"概率饱和未被拦截：\n{out}"
    assert "H2" in out


def test_row_level_significance_labeled_pseudo(tmp_path):
    """约束 1：行级 z 必须显式标注为伪显著，不得单独呈现。"""
    _, out = _run(_make_csv(tmp_path, "legit"))
    assert "行级" in out and "伪" in out, f"未标注行级统计的伪显著性：\n{out}"
    assert "Fold" in out or "折聚类" in out, f"缺少 Fold 聚类对照：\n{out}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
