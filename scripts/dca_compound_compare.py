#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定投：收益再投资(复利) vs 每年取走收益(不复利) 对照。诊断用，不改 D116 判据。

2800.HK 含息；每月首日投固定额、成本 0.1%。
- 复利：份额一直留着，期末估值（分红默认再投资）。
- 不复利：每年末把“市值 − 累计投入”的正收益卖出取走，只留本金在场。
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "dca_snapshot")
OUT = os.path.join(ROOT, "output", "dca_longterm")
C, COST = 10000.0, 0.001


def load(p):
    s = pickle.load(open(p, "rb")); s.index = pd.to_datetime(s.index); return s.sort_index().dropna()


def monthly_table(price):
    g = pd.DataFrame({"p": price}).groupby(price.index.to_period("M"))
    per = g.first().index
    return g.first()["p"].to_numpy(), g.last()["p"].to_numpy(), [str(x) for x in per]


def irr_cf(cfs):
    """cfs: [(相对月序, 金额)]；负=投入，正=取回/期末。前向复利到期末求根（避免负指数溢出）。"""
    from scipy.optimize import brentq
    n = int(max(t for t, _ in cfs))
    def f(r): return float(sum(a * (1 + r) ** (n - t) for t, a in cfs))
    grid = np.linspace(-0.9, 1.0, 381)
    vals = [f(r) for r in grid]
    for i in range(len(grid) - 1):
        if vals[i] == 0:
            return (1 + grid[i]) ** 12 - 1
        if vals[i] * vals[i + 1] < 0:
            r = brentq(f, grid[i], grid[i + 1])
            return (1 + r) ** 12 - 1
    return float("nan")


def sim(pf, pl, months, a, b, withdraw):
    shares = 0.0; contrib = 0.0; withdrawn = 0.0; cfs = []
    for m in range(a, b + 1):
        shares += C * (1 - COST) / pf[m]; contrib += C
        cfs.append((m - a, -C))
        if withdraw and months[m].endswith("-12"):
            val = shares * pl[m]; gain = val - contrib
            if gain > 0:
                shares -= gain / pl[m]              # 卖出毛额
                net = gain * (1 - COST)             # 扣卖出成本后到手
                withdrawn += net
                cfs.append((m - a, net))
    terminal = shares * pl[b]
    cfs.append((b - a, terminal))
    return dict(irr=irr_cf(cfs), wealth=(withdrawn + terminal) / contrib,
                withdrawn=withdrawn, terminal=terminal, contrib=contrib)


def main():
    pf, pl, months = monthly_table(load(os.path.join(DIR, "etf2800.pkl")))
    rows = {"full": {}, "10y": {}}
    # 全期
    a, b = 0, len(months) - 1
    for k, wd in [("复利", False), ("不复利(每年取走收益)", True)]:
        rows["full"][k] = sim(pf, pl, months, a, b, wd)
    keep = {k: v for k, v in rows["full"].items()}
    # 10 年滚动
    nm = 120; agg = {"复利": [], "不复利(每年取走收益)": []}
    for a in range(0, len(months) - nm + 1):
        for k, wd in [("复利", False), ("不复利(每年取走收益)", True)]:
            agg[k].append(sim(pf, pl, months, a, a + nm - 1, wd)["irr"])
    for k in agg:
        x = np.array(agg[k]); rows["10y"][k] = dict(median_irr=float(np.nanmedian(x)),
                                                   worst_irr=float(np.nanmin(x)))
    # 10y 财富倍数（中位/最差）
    wmult = {"复利": [], "不复利(每年取走收益)": []}
    for a in range(0, len(months) - nm + 1):
        for k, wd in [("复利", False), ("不复利(每年取走收益)", True)]:
            wmult[k].append(sim(pf, pl, months, a, a + nm - 1, wd)["wealth"])
    for k in wmult:
        z = np.array(wmult[k]); rows["10y"][k]["median_wealth"] = float(np.nanmedian(z))
        rows["10y"][k]["worst_wealth"] = float(np.nanmin(z))
    json.dump(rows, open(os.path.join(OUT, "compound_compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, sort_keys=True)

    md = ["# 定投：收益再投资 vs 每年取走（诊断）\n",
          "> 2800.HK 含息；每月首日投 1 万、成本 0.1%。复利=份额留存；不复利=每年末把超过本金的收益取走。\n",
          "\n## 全期（2008-01→今）\n",
          "| 方式 | 年化 IRR | 总财富/投入 |", "|---|---|---|"]
    for k, v in keep.items():
        md.append(f"| {k} | {v['irr']*100:+.2f}% | {v['wealth']:.2f}x |")
    md += ["\n## 10 年滚动窗口\n", "| 方式 | 中位财富倍数 | 最差财富倍数 | 中位 IRR | 最差 IRR |", "|---|---|---|---|---|"]
    for k, v in rows["10y"].items():
        md.append(f"| {k} | {v['median_wealth']:.2f}x | {v['worst_wealth']:.2f}x | "
                  f"{v['median_irr']*100:+.2f}% | {v['worst_irr']*100:+.2f}% |")
    md.append("\n> 结论：现有回测已是**复利**口径（分红再投资 + 资本利得留存），"
              "所以“收益继续投资”不会改变 D116 的数字。\n"
              "> 若每年把收益取走（不复利）：全期最终到手略少（1.39x vs 1.44x），"
              "但 10 年窗口的中位几乎相同、最差窗口反而略好（港股大回撤时提前落袋）。\n"
              "> IRR 不复利更高是“现金提前到手”的算法效应，不代表赚更多——看**最终到手金额**。\n")
    open(os.path.join(OUT, "compound_compare.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    for k, v in keep.items():
        print(f"{k}: 全期 IRR {v['irr']*100:+.2f}% / 财富 {v['wealth']:.2f}x")
    for k, v in rows["10y"].items():
        print(f"{k}: 10y 中位财富 {v['median_wealth']:.2f}x / 中位IRR {v['median_irr']*100:+.2f}% / 最差IRR {v['worst_irr']*100:+.2f}%")


if __name__ == "__main__":
    sys.exit(main())
