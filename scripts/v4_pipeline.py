#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 排序标签 · 三周期（5/10/20d）核心管线（D59 P1+P2）。

**旁路**（不拆旧生产）：验证"重写"能否产生可交易结果，并接上**三道闸**（PBO/DSR/超额CI）。

- 标签：V4 = 当日 `fwd` 的**横截面百分位**（回归目标）；
- 模型：LightGBM **回归**（出连续 score）；
- 周期：**{5,10,20}**；
- 评估：IC（跨块 t）· TopK **净**超额 IR + 95%CI · **PBO（CSCV）· DSR（收缩夏普）** · 换手。

用法：python3 scripts/v4_pipeline.py [--horizons 5,10,20] [--topks 5,10,20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.label_ablation import FEATS, _fit_predict, load_panel  # noqa: E402

COST_ROUND_TRIP = 0.005  # 往返 0.5%（单边 0.25%，D97）
LAG = 1
MIN_TRAIN_DATES = 600


def walk_forward_scores(panel, h):
    """展开窗 + embargo：一次训练 → 每个（非重叠）调仓日的 score。返回 [date,code,score,fwd]。"""
    panel = panel.dropna(subset=FEATS + ["rank", "fwd"]).copy()
    dates = sorted(panel["date"].unique())
    test_dates = dates[MIN_TRAIN_DATES::h]
    rows = []
    for t in test_dates:
        cutoff = np.datetime64(t) - np.timedelta64(h + LAG, "D")
        tr = panel[panel["date"] <= cutoff]
        te = panel[panel["date"] == t]
        if len(tr) < 500 or len(te) < 20:
            continue
        sc = _fit_predict(tr[FEATS].values, tr["rank"].values, te[FEATS].values, "reg")
        rows.append(pd.DataFrame({"date": t, "code": te["code"].values, "score": sc, "fwd": te["fwd"].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "code", "score", "fwd"])


def _period_stats(pred, h, topk):
    """逐期（非重叠）IC 与 净收益序列。返回 dict。"""
    ics, net, excess, prev = [], [], [], set()
    for _, g in pred.groupby("date", sort=True):
        g = g.sort_values("score", ascending=False)
        fwd = g["fwd"].values
        if len(fwd) < topk + 1:
            continue
        ics.append(spearmanr(g["score"].values, fwd).correlation)
        sel = g["code"].values[:topk]
        turn = len(set(sel) - prev) / topk if prev else 1.0
        prev = set(sel)
        tn = fwd[:topk].mean() - turn * COST_ROUND_TRIP
        bench = fwd.mean()
        net.append(tn)
        excess.append(tn - bench)
    ics = np.asarray(ics); net = np.asarray(net); excess = np.asarray(excess)
    if len(ics) < 5:
        return None
    ir_ex = excess.mean() / excess.std(ddof=1) * np.sqrt(252.0 / h) if excess.std(ddof=1) > 0 else np.nan
    se = excess.std(ddof=1) / np.sqrt(len(excess))
    t_ic, p_ic = ttest_1samp(ics, 0.0)
    return {"n": len(ics), "ic": float(ics.mean()), "t_ic": float(t_ic), "p_ic": float(p_ic),
            "net_ir": float(ir_ex), "excess_pp": float(excess.mean() * 100),
            "ci_lo": float((excess.mean() - 1.96 * se) * 100), "ci_hi": float((excess.mean() + 1.96 * se) * 100),
            "net": net}


def gates(pred, h, topks):
    """PBO/DSR：配置族 = TopK × {5,10,20} 的逐期**净**收益矩阵。"""
    from ml_services.eval_overfit import cscv_pbo, deflated_sharpe
    from scipy.stats import skew, kurtosis
    cols = {}
    by_date = {d: g for d, g in pred.groupby("date", sort=True)}
    for k in topks:
        s = _period_stats(pred, h, k)
        if s:
            cols[f"top{k}"] = pd.Series(s["net"], index=list(by_date.keys())[:len(s["net"])])
    if len(cols) < 2:
        return None
    M = pd.DataFrame(cols).dropna()
    if len(M) < 16:
        return None
    pbo, _ = cscv_pbo(M.values)
    mu = M.mean(); sd = M.std(ddof=1); sr = mu / sd
    best = sr.idxmax()
    dsr = deflated_sharpe(float(sr[best]), len(M), M.shape[1], float(sr.var(ddof=1)),
                          float(skew(M[best].values)), float(kurtosis(M[best].values, fisher=False)))
    return {"pbo": float(pbo), "dsr": float(dsr), "n_periods": len(M), "n_configs": M.shape[1], "best": best}


def latest_scores(panel, h):
    """训练到 (最新日 − (h+lag)) → 对最新日横截面打分（供展示层）。"""
    panel = panel.dropna(subset=FEATS + ["rank", "fwd"]).copy()
    dates = sorted(panel["date"].unique())
    last = dates[-1]
    cutoff = np.datetime64(last) - np.timedelta64(h + LAG, "D")
    tr = panel[panel["date"] <= cutoff]
    te = panel[panel["date"] == last]
    if len(tr) < 500 or len(te) < 20:
        return None
    sc = _fit_predict(tr[FEATS].values, tr["rank"].values, te[FEATS].values, "reg")
    return pd.DataFrame({"code": te["code"].values, "score": sc})


def main():
    ap = argparse.ArgumentParser(description="V4 排序标签·三周期管线（D59 P1+P2）")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--topks", default="5,10,20")
    ap.add_argument("--topk", type=int, default=10, help="主口径 TopK（报指标用）")
    ap.add_argument("--show-latest", action="store_true", help="额外打印最新横截面的相对强弱（展示层预览）")
    args = ap.parse_args()
    hs = [int(x) for x in args.horizons.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    print("V4 排序标签 · 三周期管线（D59 P1+P2）—— 旁路，不改生产")
    print("=" * 96)
    print(f"{'H':>3}{'期数':>6}{'IC':>9}{'t(IC)':>8}{'净超额':>9}{'净超额95%CI':>20}{'净IR':>7}{'PBO':>7}{'DSR':>7}")
    for h in hs:
        panel = load_panel(h)
        panel["rank"] = panel.groupby("date")["fwd"].rank(pct=True)
        pred = walk_forward_scores(panel, h)
        s = _period_stats(pred, h, args.topk) if len(pred) else None
        g = gates(pred, h, topks) if len(pred) else None
        if not s:
            print(f"{h:>3}  样本不足")
            continue
        pbo = f"{g['pbo']:.2f}" if g else "—"
        dsr = f"{g['dsr']:.3f}" if g else "—"
        print(f"{h:>3}{s['n']:>6}{s['ic']:>+9.3f}{s['t_ic']:>+8.2f}{s['excess_pp']:>+8.2f}%"
              f"[{s['ci_lo']:>+6.2f}%,{s['ci_hi']:>+6.2f}%]{s['net_ir']:>+7.2f}{pbo:>7}{dsr:>7}")
    print("-" * 96)
    print("判据(D59 §三): 净IR / PBO<0.5 / DSR≥0.95 / 超额CI下界>0（同 D2）；先验偏 null。")
    print("⚠️ 旁路结果**不**改生产/总账。")
    if args.show_latest:
        from ml_services.rank_display import render_text
        h0 = hs[0]
        panel = load_panel(h0)
        panel["rank"] = panel.groupby("date")["fwd"].rank(pct=True)
        sl = latest_scores(panel, h0)
        print("=" * 96)
        if sl is None:
            print(f"最新横截面样本不足（H={h0}）")
        else:
            from ml_services.rank_display import build_rows
            rows = build_rows(sl["score"].values, sl["code"].values)
            print(render_text(rows, h0, top=12))
    return 0


if __name__ == "__main__":
    sys.exit(main())
