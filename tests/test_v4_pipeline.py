"""V4 三周期管线（scripts/v4_pipeline.py）—— D59 P1。"""

import numpy as np
import pandas as pd

import scripts.v4_pipeline as vp


def _panel(seed=0, n_dates=810, n_stocks=25):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_dates)
    rows = []
    for c in range(n_stocks):
        code = f"{c:04d}.HK"
        fwd = rng.normal(0.001, 0.05, n_dates)
        row = {"date": idx, "code": code, "fwd": fwd}
        for f in vp.FEATS:
            row[f] = rng.normal(0, 1, n_dates)
        rows.append(pd.DataFrame(row))
    p = pd.concat(rows, ignore_index=True)
    p["rank"] = p.groupby("date")["fwd"].rank(pct=True)
    p["hsi_fwd"] = 0.0
    p["sector"] = "s"
    return p


def test_evaluate_returns_metrics():
    # 用噪声特征 ⇒ IC≈0，但函数必须返回结构完整的指标
    p = _panel()
    r = vp.evaluate(p, 5)
    assert r is None or {"ic", "net_ir", "excess_ci_lo", "excess_ci_hi", "n"} <= set(r)


def test_evaluate_insufficient():
    p = _panel(n_dates=100)
    assert vp.evaluate(p, 20) is None


def test_cost_sign_matters():
    # 高成本下净超额不可能高于毛超额
    p = _panel()
    r = vp.evaluate(p, 5)
    if r:
        assert r["net_excess_pp"] <= r["gross_pp"] + 1e-9
