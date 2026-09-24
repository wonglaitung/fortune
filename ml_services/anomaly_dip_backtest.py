#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格复核"价格异常 + 当日下跌 → 抄底"策略（DECISIONS 之后第 2 步）

定义（显式）：信号日 = 该股当日收益的 30 日 z-score ≤ 阈值（=异常大跌，即"价格异常+当日下跌"）。
交易：收盘买入，持有 5 日；扣双边成本 0.5%。
对照：无条件买入（所有样本）的 5 日胜率/收益。
输出：信号样本数 / 胜率 vs 基准 / 净均收益 / bootstrap CI / 净IR / 逐年。

用法：python3 ml_services/anomaly_dip_backtest.py
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = 'data/feature_cache'
COST = 0.005
WINDOW = 30
HORIZON = 5
THRESHOLDS = [-2.5, -3.0, -4.0]


def load():
    files = {}
    for p in glob.glob(os.path.join(CACHE_DIR, '*_shift.pkl')):
        code = os.path.basename(p).split('_')[0]
        if code not in files or p > files[code]:
            files[code] = p
    frames = []
    for code, p in files.items():
        try:
            d = pd.read_pickle(p)['data']['stock_df']
        except Exception:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None)
        c = d['Close']
        ret = c.pct_change()
        z = (ret - ret.rolling(WINDOW, min_periods=WINDOW).mean()) / ret.rolling(WINDOW, min_periods=WINDOW).std()
        f = pd.DataFrame({'ret': ret, 'z': z, 'fwd5': c.shift(-HORIZON) / c - 1, 'code': code})
        frames.append(f.dropna(subset=['z', 'fwd5']))
    panel = pd.concat(frames)
    panel = panel[(panel.index >= '2021-10-01') & (panel.index < '2026-08-01')]
    panel = panel.reset_index()
    date_col = 'Date' if 'Date' in panel.columns else panel.columns[0]
    panel['date'] = pd.to_datetime(panel[date_col])
    panel['year'] = panel['date'].dt.year
    return panel


def boot_mean(x, n=2000, seed=42):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    rng = np.random.RandomState(seed); idx = np.arange(len(x))
    b = [x[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def eval_strategy(sig, base):
    n = len(sig)
    net = sig['fwd5'] - COST
    win = (net > 0).mean()
    mean = net.mean()
    std = net.std(ddof=1)
    ir = mean / std * np.sqrt(252.0 / HORIZON) if std > 0 else np.nan
    lo, hi = boot_mean(net)
    base_win = (base['fwd5'] > COST).mean()
    return dict(n=n, win=win, base_win=base_win, lift=win - base_win, mean=mean,
                lo=lo, hi=hi, ir=ir)


def main(out_md=None):
    panel = load()
    base = panel  # 无条件买入基准 = 全样本
    print(f"样本: {len(panel)} 日期: {panel['date'].nunique()} 股票: {panel['code'].nunique()}")

    L = []
    L.append(f"# 异常大跌抄底策略复核（{HORIZON}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 触发: 当日收益 30日z-score ≤ 阈值（价格异常+当日下跌）；买入持 {HORIZON} 日，成本 {COST:.3f}")
    L.append(f"- 基准: 无条件买入（全样本）\n")
    L.append("| 阈值(z≤) | 信号数 | 信号胜率 | 基准胜率 | 超额lift | 净均收益/期 | 净IR | 净IR 95%CI |")
    L.append("|---------|--------|---------|---------|---------|------------|------|-----------|")
    res = {}
    for thr in THRESHOLDS:
        sig = panel[panel['z'] <= thr]
        r = eval_strategy(sig, base)
        res[thr] = r
        L.append(f"| {thr} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | {r['lift']*100:+.1f}pp | "
                 f"{r['mean']*100:+.2f}% | {r['ir']:.2f} | [{r['lo']*100:+.1f}%,{r['hi']*100:+.1f}%] |")
    L.append("")
    L.append("## 逐年（z≤-3）\n")
    sig3 = panel[panel['z'] <= -3.0]
    L.append("| 年份 | 信号数 | 信号胜率 | 基准胜率 | 净均收益/期 |")
    L.append("|------|--------|---------|---------|------------|")
    for y, g in sig3.groupby('year'):
        r = eval_strategy(g, base[base['year'] == y])
        L.append(f"| {int(y)} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | {r['mean']*100:+.2f}% |")
    L.append("")
    L.append("## 结论\n")
    r = res[-3.0]
    verdict = "✅ 显著跑赢基准" if (r['lift'] > 0.05 and r['lo'] > 0) else ("⚠️ 微弱/不稳健" if r['mean'] > 0 else "❌ 无超额（甚至为负）")
    L.append(f"- 原文档口径（2026-05）：价格异常+当日下跌 → 5日 +4.12% / 胜率 72%")
    L.append(f"- 严格复核（z≤−3，2021–2026，{r['n']} 个信号）：胜率 {r['win']*100:.1f}% vs 基准 {r['base_win']*100:.1f}%"
             f"，净均 {r['mean']*100:+.2f}%，净IR {r['ir']:.2f}，CI [{r['lo']*100:+.1f}%,{r['hi']*100:+.1f}%]")
    L.append(f"- 判定: {verdict}")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    main('output/anomaly_dip_backtest.md')