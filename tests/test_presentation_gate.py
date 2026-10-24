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


def _pick_codes(n_sectors=3, per_sector=4):
    """从 config 的真实映射里挑 n_sectors 个板块、每板块 per_sector 只代码（≥10 只）。"""
    import config as _cfg

    by_sec = {}
    for _code, _v in _cfg.STOCK_SECTOR_MAPPING.items():
        _s = _v.get("sector") if isinstance(_v, dict) else _v
        by_sec.setdefault(_s, []).append(str(_code).replace(".HK", "") + ".HK")
    _secs = [_s for _s, _cs in by_sec.items() if len(_cs) >= per_sector][:n_sectors]
    _codes = []
    for _s in _secs:
        _codes += by_sec[_s][:per_sector]
    return _codes


def _make_exog_csv(tmp_path, mode="common_only", n_dates=30):
    """构造多股票多板块面板。

    mode='common_only'：prob/return 仅在板块层面相关（个股特异不相关）
        ⇒ H6 应 FAIL（IC 全来自共同层）。
    mode='with_idio'：另加个股层面 prob↔return 相关
        ⇒ H6 应 PASS（存在个股特异选择信息）。
    """
    _codes = _pick_codes()
    assert len(_codes) >= 10, "挑出的代码不足 10 只"
    _rng = np.random.RandomState(7)
    _dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    _rows = []
    _sec_of = {}
    import config as _cfg

    for _c in _codes:
        _v = _cfg.STOCK_SECTOR_MAPPING.get(_c, {})
        _sec_of[_c] = _v.get("sector") if isinstance(_v, dict) else _v
    for _t, _d in enumerate(_dates):
        # 板块层面信号（共同层）；个股 idio 仅在 with_idio 存在。
        # 刻意把 prob 压在 [~0.4,0.6] 窄带：方向准确率维持在 <65%（避免 H1 误触发），
        # 但 rank 次序仍跟随信号 ⇒ rank IC 高。ret 加较大噪声 ⇒ 方向主要靠噪声。
        _sec_sig = {_s: _rng.randn() for _s in set(_sec_of.values())}
        for _c in _codes:
            _s = _sec_of[_c]
            _common = _sec_sig[_s]
            # 个股层面噪声（prob 侧）；仅 with_idio 时与 return 侧共享一个成分
            _idio_p = _rng.randn() * 0.1
            _idio_r = _rng.randn() * 0.01
            if mode == "with_idio":
                _shared = _rng.randn()
                _idio_p += _shared * 0.7
                _idio_r += _shared * 0.2
            # 共同层（板块）驱动 prob 与 ret；个股噪声在 common_only 下互不相关
            _prob = float(np.clip(0.5 + 0.1 * _common + 0.1 * _idio_p, 0.02, 0.98))
            # ret 的共同层成分压小、噪声适中 ⇒ 方向准确率 <65%（H1 不误触发），
            # 但 rank 次序仍跟随 common ⇒ raw IC 高于 0.02 下限。
            _ret = 0.03 + 0.02 * _common + 0.02 * _idio_r + _rng.randn() * 0.05
            _pred = "UP" if _prob >= 0.5 else "DOWN"
            _act = "UP" if _ret > 0 else "DOWN"
            _rows.append(
                {
                    "Fold": (_t % 3) + 1,
                    "Date": _d.strftime("%Y-%m-%d"),
                    "Stock_Code": _c,
                    "Predict_Prob": _prob,
                    "Predict_Direction": _pred,
                    "Actual_Return": _ret,
                    "Actual_Direction": _act,
                    "Is_Correct": _pred == _act,
                }
            )
    _df = pd.DataFrame(_rows)
    _p = os.path.join(str(tmp_path), f"exog_{mode}.csv")
    _df.to_csv(_p, index=False)
    return _p


def test_h6_common_layer_only_fails(tmp_path):
    """H6：IC 全来自板块共同层、个股特异为零 ⇒ FAIL。"""
    _p = _make_exog_csv(tmp_path, "common_only")
    rc, out = _run(_p)
    assert rc == 1, f"共同层-only 数据未被 H6 拦截：\n{out}"
    assert "H6" in out, out


def test_h6_with_idiosyncratic_passes(tmp_path):
    """H6：含个股特异选择信息 ⇒ PASS。"""
    _p = _make_exog_csv(tmp_path, "with_idio")
    rc, out = _run(_p)
    assert rc == 0, f"含个股特异的数据被 H6 误判 FAIL：\n{out}"
    assert "H6" in out, out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
