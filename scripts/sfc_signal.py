#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFC 卖空数据面板：周频空头强度 → 前瞻收益（描述性研究）。

时点（PIT）：
- SFC 文件日期 = 该周最后一个交易日；公布有滞后，默认 **+10 个交易日** 才可用于交易
- 强度 = short_hk$ / ADV20（HKD，20 日均额），避免依赖股数/市值
- 前瞻收益 = 生效日收盘 → +fwd 个交易日收盘

用法：
  python3 scripts/sfc_signal.py --fwd 20 --lag 10
  python3 scripts/sfc_signal.py --fwd 20 --lag 10 --out output/_logs/sfc_signal.md
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _norm_idx(idx):
    idx = pd.to_datetime(idx)
    idx = idx.tz_convert(None) if idx.tz is not None else idx
    return idx.normalize()


def load_prices(snapshot_dir, codes):
    close, vol = {}, {}
    for c in codes:
        p = os.path.join(snapshot_dir, c + '.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df.index = _norm_idx(df.index)
        close[c] = df['Close'].astype(float)
        vol[c] = df['Volume'].astype(float)
    return pd.DataFrame(close).sort_index(), pd.DataFrame(vol).sort_index()


def nw_t(x, lags):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3:
        return np.nan
    mu = x.mean()
    e = x - mu
    s = (e @ e) / len(x)
    for lag in range(1, min(lags, len(x) - 1) + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / len(x)
    return float(mu / np.sqrt(max(s, 1e-18) / len(x)))


def build(sfc_csv, snapshot_dir, universe_json, fwd, lag):
    uni = json.load(open(universe_json))
    # 统一成 5 位数字键（universe 里可能是 0700.HK，SFC 是 700）
    codes = sorted({c.split('.')[0].zfill(5) for c in uni if c.endswith('.HK')})
    close, vol = load_prices(snapshot_dir, codes)
    cal = close.index[close.notna().sum(axis=1) >= 200]
    close, vol = close.reindex(cal), vol.reindex(cal)

    adv20 = (close * vol).rolling(20, min_periods=10).mean()

    sfc = pd.read_csv(sfc_csv, dtype={'stock_code': str})
    sfc['date'] = pd.to_datetime(sfc['date'].astype(str), format='%Y%m%d')
    sfc['code'] = sfc['stock_code'].str.split('.').str[0].str.zfill(5)
    sfc = sfc[sfc['code'].isin(codes)]
    short = sfc.pivot_table(index='date', columns='code', values='short_hk$', aggfunc='sum')

    rows = []
    for d, row in short.iterrows():
        pos = cal.searchsorted(d, side='right') - 1
        eff = pos + lag
        if eff + fwd >= len(cal):
            continue
        t0, t1 = cal[eff], cal[eff + fwd]
        adv = adv20.loc[cal[pos]] if pos >= 0 else None
        fwd_ret = close.loc[t1] / close.loc[t0] - 1
        for code, sv in row.dropna().items():
            a = adv.get(code) if adv is not None else np.nan
            if not np.isfinite(a) or a <= 0:
                continue
            rows.append({
                'week': d, 'eff_date': t0, 'code': code,
                'short_hkd': float(sv), 'adv20': float(a),
                'intensity': float(sv) / float(a),
                'fwd_ret': float(fwd_ret.get(code, np.nan)),
            })
    panel = pd.DataFrame(rows).dropna(subset=['fwd_ret'])
    panel['year'] = panel['eff_date'].dt.year
    return panel


def evaluate(panel, fwd, lag, out):
    def ics(g):
        g = g.dropna(subset=['intensity', 'fwd_ret'])
        if len(g) < 20 or g['intensity'].nunique() < 3:
            return np.nan
        return stats.spearmanr(g['intensity'], g['fwd_ret']).correlation

    weekly = panel.groupby('eff_date').apply(ics).dropna()
    lines = ['# SFC 卖空强度 → 前瞻收益（描述性）', '',
             f'- 样本：{panel["eff_date"].min().date()} → {panel["eff_date"].max().date()}，'
             f'{panel["week"].nunique()} 周，{panel["code"].nunique()} 只',
             f'- 滞后：{lag} 交易日；前瞻：{fwd} 交易日',
             f'- 周频 rank IC：均值 {weekly.mean():.4f}，NW t={nw_t(weekly.values, max(1, fwd // 5)):.2f}，n={len(weekly)}', '']

    for lo, hi, name in ((2012, 2018, '2012–2018'), (2019, 2026, '2019–2026')):
        sub = panel[(panel['year'] >= lo) & (panel['year'] <= hi)]
        w = sub.groupby('eff_date').apply(ics).dropna()
        lines.append(f'- {name}：IC {w.mean():.4f}（t={nw_t(w.values, max(1, fwd // 5)):.2f}, n={len(w)}）')

    # 分位价差：强度最低 vs 最高（高做空 → 预期跑输）
    panel = panel.copy()
    panel['q'] = panel.groupby('eff_date')['intensity'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        if s.notna().sum() >= 25 else np.nan)
    spread = panel[panel['q'].isin([1, 5])].pivot_table(
        index='eff_date', columns='q', values='fwd_ret', aggfunc='mean').dropna()
    if len(spread):
        s = spread[1] - spread[5]
        lines += ['', f'- Q1−Q5 周均收益（低空头−高空头）：{s.mean():+.4%}，'
                      f'NW t={nw_t(s.values, max(1, fwd // 5)):.2f}；'
                      f'胜率 {float((s > 0).mean()):.0%}']

    # 流动性分层（按 ADV 中位数）
    panel['liq'] = panel.groupby('eff_date')['adv20'].transform(
        lambda s: np.where(s >= s.median(), 'high', 'low'))
    for name, sub in panel.groupby('liq'):
        w = sub.groupby('eff_date').apply(ics).dropna()
        lines.append(f'- 流动性 {name}：IC {w.mean():.4f}（t={nw_t(w.values, max(1, fwd // 5)):.2f}, n={len(w)}）')

    text = '\n'.join(lines)
    print(text)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            f.write(text + '\n')
        print(f'\n报告: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sfc', default='data/sfc_short/sfc_short_long.csv')
    ap.add_argument('--snapshot', default='data/hk_broad/snapshot')
    ap.add_argument('--universe', default='data/hk_broad/universe_310.json')
    ap.add_argument('--fwd', type=int, default=20)
    ap.add_argument('--lag', type=int, default=10)
    ap.add_argument('--out', default='output/_logs/sfc_signal.md')
    ap.add_argument('--save-panel', default=None)
    args = ap.parse_args()

    panel = build(args.sfc, args.snapshot, args.universe, args.fwd, args.lag)
    if args.save_panel:
        panel.to_csv(args.save_panel, index=False)
        print(f'面板已保存: {args.save_panel}（{len(panel)} 行）')
    evaluate(panel, args.fwd, args.lag, args.out)


if __name__ == '__main__':
    main()
