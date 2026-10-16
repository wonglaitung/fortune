#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 排序标签 · 三周期（5/10/20d）核心管线（D59 P1）。

**旁路**（不拆旧生产）：验证"重写"能否产生可交易结果。
- 标签：V4 = 当日 `fwd` 的**横截面百分位**（回归目标）；
- 模型：LightGBM **回归**（出连续 score）；
- 周期：**{5,10,20}**；
- 评估：IC（跨块 t）· TopK=10 **净**超额 IR · 超额 95%CI（bootstrap）· 换手。

用法：python3 scripts/v4_pipeline.py [--horizons 5,10,20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.label_ablation import FEATS, _fit_predict, load_panel  # noqa: E402

TOPK = 10
COST = 0.005          # 双边成本（与 portfolio_backtest 一致）
LAG = 1
MIN_TRAIN_DATES = 600


def evaluate(panel, h):
    """展开窗 + embargo：训练只用 `date ≤ t−(h+lag)` 的行（标签已实现）。返回指标。"""
    panel = panel.dropna(subset=FEATS + ["rank", "fwd"]).copy()
    dates = sorted(panel["date"].unique())
    test_dates = dates[MIN_TRAIN_DATES::h]          # 非重叠调仓
    ics, nets, gross_ex = [], [], []
    prev = set()
    for t in test_dates:
        cutoff = np.datetime64(t) - np.timedelta64(h + LAG, "D")
        tr = panel[panel["date"] <= cutoff]
        te = panel[panel["date"] == t]
        if len(tr) < 500 or len(te) < 20:
            continue
        sc = _fit_predict(tr[FEATS].values, tr["rank"].values, te[FEATS].values, "reg")
        fwd = te["fwd"].values
        ic = spearmanr(sc, fwd).correlation
        if np.isfinite(ic):
            ics.append(ic)
        order = np.argsort(-sc)
        sel = te["code"].values[order[:TOPK]]
        top = fwd[order[:TOPK]]
        bench = fwd.mean()
        turn = len(set(sel) - prev) / TOPK if prev else 1.0
        prev = set(sel)
        nets.append(float(top.mean() - turn * COST - bench))   # 相对等权的**净**超额
        gross_ex.append(float(top.mean() - bench))
    if len(ics) < 5:
        return None
    ics = np.asarray(ics)
    nets = np.asarray(nets)
    se = nets.std(ddof=1) / np.sqrt(len(nets))
    ir = nets.mean() / nets.std(ddof=1) * np.sqrt(252.0 / h) if nets.std(ddof=1) > 0 else np.nan
    t_ic, p_ic = ttest_1samp(ics, 0.0)
    return {"n": len(ics), "ic": float(ics.mean()), "t_ic": float(t_ic), "p_ic": float(p_ic),
            "net_excess_pp": float(nets.mean() * 100), "excess_ci_lo": float((nets.mean() - 1.96 * se) * 100),
            "excess_ci_hi": float((nets.mean() + 1.96 * se) * 100), "net_ir": float(ir),
            "gross_pp": float(np.mean(gross_ex) * 100)}


def main():
    ap = argparse.ArgumentParser(description="V4 排序标签·三周期管线（D59 P1）")
    ap.add_argument("--horizons", default="5,10,20")
    args = ap.parse_args()
    hs = [int(x) for x in args.horizons.split(",")]
    print("V4 排序标签 · 三周期管线（D59 P1）—— 旁路，不改生产")
    print("=" * 84)
    print(f"{'H':>3}{'块数':>6}{'IC':>9}{'t(IC)':>8}{'毛超额':>9}{'净超额':>9}{'净超额95%CI':>20}{'净IR':>7}")
    for h in hs:
        panel = load_panel(h)
        panel["rank"] = panel.groupby("date")["fwd"].rank(pct=True)
        r = evaluate(panel, h)
        if not r:
            print(f"{h:>3}  样本不足")
            continue
        print(f"{h:>3}{r['n']:>6}{r['ic']:>+9.3f}{r['t_ic']:>+8.2f}{r['gross_pp']:>+8.2f}%"
              f"{r['net_excess_pp']:>+8.2f}%[{r['excess_ci_lo']:>+6.2f}%,{r['excess_ci_hi']:>+6.2f}%]"
              f"{r['net_ir']:>+7.2f}")
    print("-" * 84)
    print("判据(D59 §三): 净IR / PBO / DSR / 超额CI下界>0（同 D2）；先验偏 null。")
    print("⚠️ 旁路结果**不**改生产/总账；PBO/DSR 待 P2。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
