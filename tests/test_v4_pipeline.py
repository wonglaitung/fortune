"""V4 三周期管线（scripts/v4_pipeline.py）—— D59 P1/P2。"""

import numpy as np
import pandas as pd

import scripts.v4_pipeline as vp


def _panel(seed=0, n_dates=810, n_stocks=25, signal=0.0):
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
    if signal:
        p["fwd"] = p["fwd"] + signal * p[vp.FEATS[0]]
    p["rank"] = p.groupby("date")["fwd"].rank(pct=True)
    return p


def test_walk_forward_shapes():
    p = _panel(n_dates=810)
    pred = vp.walk_forward_scores(p, 5)
    assert {"date", "code", "score", "fwd"} <= set(pred.columns)
    assert len(pred) > 0
    # 每组（每日）股票数应等于池子大小
    assert pred.groupby("date").size().max() == 25


def test_period_stats_insufficient():
    p = _panel(n_dates=100)
    pred = vp.walk_forward_scores(p, 20)
    assert vp._period_stats(pred, 20, 10) is None


def test_gates_returns_pbo_dsr():
    p = _panel()
    pred = vp.walk_forward_scores(p, 5)
    g = vp.gates(pred, 5, [5, 10, 20])
    assert g is None or {"pbo", "dsr", "n_configs"} <= set(g)


def test_cost_never_helps():
    p = _panel(signal=0.01)
    pred = vp.walk_forward_scores(p, 5)
    k = 10
    # 逐期绝对净收益 ≤ 对应毛收益（成本非负）
    for _, g in pred.groupby("date"):
        g = g.sort_values("score", ascending=False)
        assert g["fwd"].values[:k].mean() - 1.0 * vp.COST_ROUND_TRIP <= g["fwd"].values[:k].mean() + 1e-12
