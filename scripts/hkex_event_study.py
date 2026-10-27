#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HKEX 公告事件研究：标题分类 → 次日收盘入场 → N 日 CAR。

口径（预登记见 output/_logs/prereg_hkex_events.txt）：
- 入场 = 公告日**次一交易日**收盘； CAR = 个股收益 − 310 等权收益 − 0.5%
- 显著性：按入场月份聚类（同月事件相关）
"""
import argparse
import html
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


POST_EVENT = ('RESULTS OF', 'COMPLETION OF', 'CLOSING OF', 'CLOSE OF', 'DESPATCH OF')


def classify(title):
    """只取「首次公告」；排除结果/完成类后续公告（预登记口径）。"""
    t = html.unescape(str(title)).upper()
    if any(k in t for k in POST_EVENT):
        return None
    if 'POSITIVE PROFIT ALERT' in t:
        return 'positive_alert'
    if 'PROFIT WARNING' in t:
        return 'profit_warning'
    if 'RIGHTS ISSUE' in t or 'OPEN OFFER' in t:
        return 'rights'
    if 'PLACING' in t or 'PLACEMENT' in t:
        return 'placement'
    return None


def load_prices(codes, snapshot='data/hk_broad/snapshot'):
    close = {}
    for c in codes:
        p = os.path.join(snapshot, c + '.pkl')
        if os.path.exists(p):
            df = pd.read_pickle(p)
            idx = pd.to_datetime(df.index)
            idx = idx.tz_convert(None) if idx.tz is not None else idx
            close[c] = pd.Series(df['Close'].values, index=idx.normalize())
    return pd.DataFrame(close).sort_index()


def cluster_t(df, col='car', by='month'):
    g = df.groupby(by)[col].mean().dropna()
    if len(g) < 3:
        return np.nan, len(g)
    return float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='data/hkex_announcements/all.csv')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--cost', type=float, default=0.005)
    ap.add_argument('--out', default='output/_logs/hkex_event_study.md')
    args = ap.parse_args()

    df = pd.read_csv(args.events, dtype={'code': str})
    df['code'] = df['code'].str.zfill(5)
    df['dt'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['type'] = df['title'].map(classify)
    ev = df[df['type'].notna()].copy()
    ev['ann_date'] = ev['dt'].dt.normalize()
    ev = ev.sort_values('dt').drop_duplicates(subset=['code', 'type', 'ann_date'])

    close = load_prices(sorted(ev['code'].unique()))
    cal = close.index
    lines = ['# HKEX 公告事件研究', '',
             f'- 事件: {len(ev)} 条；股票: {ev["code"].nunique()}；'
             f'区间: {ev["ann_date"].min().date()} → {ev["ann_date"].max().date()}',
             f'- 入场: 公告次日收盘；持有 {args.horizon} 日；成本 {args.cost:.1%}（CAR 已扣）', '']
    if ev.empty:
        print('\n'.join(lines)); return

    # 入场日
    pos = cal.searchsorted(ev['ann_date'].values, side='right')
    pos = np.clip(pos, 0, len(cal) - 1)
    ev['entry_pos'] = pos
    ev = ev[ev['entry_pos'] + args.horizon < len(cal)].copy()
    ev['entry'] = cal[ev['entry_pos'].values]
    ev['exit'] = cal[ev['entry_pos'].values + args.horizon]

    # 等权基准（按入场/退出日）
    def bench(row):
        r = close.loc[row['exit']] / close.loc[row['entry']] - 1
        return float(r.mean())
    ev['bench'] = ev.apply(bench, axis=1)
    ev['ret'] = [
        float(close.loc[e, c] / close.loc[s, c] - 1)
        if c in close.columns and pd.notna(close.loc[e, c]) and pd.notna(close.loc[s, c]) else np.nan
        for s, e, c in zip(ev['entry'], ev['exit'], ev['code'])
    ]
    ev = ev.dropna(subset=['ret'])
    ev['car'] = ev['ret'] - ev['bench'] - args.cost
    ev['month'] = ev['entry'].dt.to_period('M')
    ev['era'] = np.where(ev['entry'].dt.year <= 2018, '2010–2018', '2019–2026')

    lines += ['| 类型 | 事件数 | 月数 | 平均 CAR | 月聚类 t | 2010–2018 | 2019–2026 |',
              '|---|---|---|---|---|---|---|']
    for typ, g in ev.groupby('type'):
        t, k = cluster_t(g)
        a = g[g['era'] == '2010–2018']['car'].mean()
        b = g[g['era'] == '2019–2026']['car'].mean()
        lines.append(f"| {typ} | {len(g)} | {k} | {g['car'].mean():+.3%} | {t:.2f} | "
                     f"{a:+.3%} | {b:+.3%} |")
    text = '\n'.join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write(text + '\n')
    print(f'报告: {args.out}')


if __name__ == '__main__':
    main()
