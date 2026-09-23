#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part B：在更大港股池（hk_universe_cache，~124 只）上测 Alpha158 复合 IC，
与 59 只（feature_cache）对比，验证"横截面越大 → IC 检测力越强"。

用法：python3 ml_services/alpha158_universe.py
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.alpha158_baseline import alpha_factors

UNI_CACHE = 'data/hk_universe_cache'
FEAT_CACHE = 'data/feature_cache'
TEST_START = '2021-10-01'
TEST_END = '2026-08-01'


def load_ohlcv_from(dirp, prefix_filter=None):
    files = {}
    for p in glob.glob(os.path.join(dirp, '*.pkl')):
        code = os.path.basename(p).split('_')[0]
        if code not in files or p > files[code]:
            files[code] = p
    out = {}
    for code, p in files.items():
        try:
            d = pd.read_pickle(p)
            if 'data' in d:
                d = d['data']['stock_df']
            if 'Date' in d.columns:
                d = d.set_index('Date')
            d.index = pd.to_datetime(d.index).tz_localize(None)
            out[code] = d
        except Exception:
            continue
    return out


def panel_from(ohlcv, horizon=20):
    frames = []
    for code, d in ohlcv.items():
        f = alpha_factors(d)
        f['fwd_ret'] = d['Close'].shift(-horizon) / d['Close'] - 1
        f['code'] = code
        frames.append(f.dropna(subset=['fwd_ret']))
    panel = pd.concat(frames)
    panel = panel[(panel.index >= TEST_START) & (panel.index < TEST_END)]
    panel = panel.reset_index()
    panel = panel.rename(columns={panel.columns[0]: 'date'})
    panel['date'] = pd.to_datetime(panel['date'])
    feat = [c for c in alpha_factors(next(iter(ohlcv.values()))).columns if c in panel.columns]
    gm = panel.groupby('date')[feat].transform('mean')
    gs = panel.groupby('date')[feat].transform('std')
    panel[feat] = ((panel[feat] - gm) / (gs + 1e-9)).replace([np.inf, -np.inf], np.nan)
    panel['score'] = panel[feat].mean(axis=1)
    return panel


def composite_stats(panel):
    ics = []
    for d, g in panel.groupby('date'):
        g = g.dropna(subset=['score', 'fwd_ret'])
        if len(g) < 15:
            continue
        ic = spearmanr(g['score'], g['fwd_ret']).correlation
        if np.isfinite(ic):
            ics.append((ic, len(g)))
    vals = np.array([x[0] for x in ics]); ns = np.array([x[1] for x in ics])
    if len(vals) < 30:
        return dict(n=len(vals), stocks=ns.mean(), ic=np.nan, icir=np.nan, std=np.nan)
    return dict(n=len(vals), stocks=ns.mean(), ic=vals.mean(),
                icir=vals.mean() / vals.std(ddof=1), std=vals.std(ddof=1))


def run(out_md=None):
    uni = load_ohlcv_from(UNI_CACHE)
    feat = load_ohlcv_from(FEAT_CACHE)
    p_uni = panel_from(uni)
    p_feat = panel_from(feat)
    s_uni = composite_stats(p_uni)
    s_feat = composite_stats(p_feat)

    L = []
    L.append("# Part B：更大港股池的 Alpha158 复合 IC\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 窗口: {TEST_START} ~ {TEST_END}\n")
    L.append("| 池 | 股票数(日均) | 期数 | 复合IC | ICIR | 日IC std |")
    L.append("|----|------------|------|--------|------|---------|")
    for name, s in (("59只(feature_cache)", s_feat), ("~124只(hk_universe_cache)", s_uni)):
        L.append(f"| {name} | {s['stocks']:.0f} | {s['n']} | {s['ic']:+.4f} | {s['icir']:.3f} | {s['std']:.4f} |")
    L.append("")
    L.append("## 结论\n")
    L.append(f"- 日度 IC std：59只 {s_feat['std']:.3f} → 124只 {s_uni['std']:.3f}"
             f"（理论 1/√(m−3)：59→0.134、124→0.091）。")
    L.append(f"- 复合 IC：59只 {s_feat['ic']:+.4f} → 124只 {s_uni['ic']:+.4f}；ICIR 对比 {s_feat['icir']:.2f} → {s_uni['icir']:.2f}。")
    L.append("- 若扩池后 std 下降、IC/ICIR 上升 → 支持'横截面是瓶颈'；若 IC 反而下降 → 说明新增股票稀释了信号。")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    run('output/alpha158_universe_20d.md')