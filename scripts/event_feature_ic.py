#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2b：事件特征单变量 IC（30 只；月度横截面）。"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_services.feature_engineering.event_features import FEATURES, build_event_features

HSI = 'data/hk_universe_cache/HSI.pkl'


def load_px(code):
    p = f'data/hk_broad/snapshot/{str(code).zfill(5)}.pkl'
    if not os.path.exists(p):
        return None
    s = pd.read_pickle(p)['Close'].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def nw_t(x, lag=4):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    mu = x.mean(); var = np.var(x, ddof=1)
    for l in range(1, lag + 1):
        g = np.sum((x[:-l] - mu) * (x[l:] - mu)) / (n - 1)
        var += 2 * (1 - l / (lag + 1)) * g
    return mu / np.sqrt(var / n) if var > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='data/events/events.csv')
    ap.add_argument('--magnitudes', default='data/events/event_magnitudes.csv')
    ap.add_argument('--out', default='output/_logs/event_feature_ic.md')
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default='2026-08-31')
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    mg = pd.read_csv(args.magnitudes) if os.path.exists(args.magnitudes) else None
    hsi = pd.read_pickle(HSI)['Close'].dropna()
    hsi.index = pd.to_datetime(hsi.index).tz_localize(None).normalize()

    codes = sorted(ev.code.astype(str).str.zfill(5).unique())
    feats, labels, month_dates = {}, {}, []
    for code in codes:
        px = load_px(code)
        if px is None:
            continue
        px = px[(px.index >= args.start) & (px.index <= args.end)]
        if px.empty:
            continue
        f = build_event_features(code, px.index, ev, mg)
        feats[code] = f
        me = px.resample('ME').last().index
        me = [d for d in me if d in px.index]
        month_dates.extend(me)
    month_dates = sorted(set(month_dates))

    rows = []
    for t in month_dates:
        for code, f in feats.items():
            if t not in f.index:
                continue
            px = load_px(code)
            pos = px.index.searchsorted(t)
            if pos + 21 >= len(px):
                continue
            d0, d1 = px.index[pos + 1], px.index[pos + 21]
            hi0 = hsi.index.searchsorted(d0)
            hi1 = hsi.index.searchsorted(d1)
            if hi1 >= len(hsi):
                continue
            lab = (px.loc[d1] / px.loc[d0] - 1) - (hsi.iloc[hi1] / hsi.iloc[hi0] - 1)
            rows.append({'month': t, 'code': code, 'label': lab,
                         **{c: f.loc[t, c] for c in FEATURES}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('无样本')

    lines = ['# 事件特征 IC（P2b，30 只）', '',
             f'- 样本：{df.code.nunique()} 只，{df.month.nunique()} 个月，{len(df)} 行',
             '- 标签：未来 20 日个股 − HSI（T+1 入场）；IC 为月度横截面 Spearman。',
             '', '| 特征 | 有效月 | 均值 IC | NW t | MDE(t=2) | 活跃月占比 |',
             '|---|---:|---:|---:|---:|---:|']
    for c in FEATURES:
        ics, active = [], 0
        for _, g in df.groupby('month'):
            if g[c].nunique() < 2:
                continue
            ic, _ = spearmanr(g[c], g['label'])
            if np.isfinite(ic):
                ics.append(ic)
            if (g[c] != 0).any():
                active += 1
        if not ics:
            continue
        mean = float(np.mean(ics)); t = nw_t(ics)
        se = mean / t if t and np.isfinite(t) and t != 0 else np.nan
        lines.append(f'| {c} | {len(ics)} | {mean:+.4f} | {t:+.2f} | {abs(2*se):.4f} | {active}/{df.month.nunique()} |')
    lines += ['', '> 稀疏特征只做“有没有方向”的初筛；功率不足时不据此下结论。']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
