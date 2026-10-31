#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 事件研究：公告后 T+1 入场，计算 1/5/20/60 日超额收益（vs HSI）。

用法：python3 scripts/event_study.py --events data/events/events_310.csv
输出：output/_logs/event_car.csv + event_study.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HSI = 'data/hk_universe_cache/HSI.pkl'
SNAP = 'data/hk_broad/snapshot'


def load_px(code):
    p = f'{SNAP}/{code}.pkl'
    if not os.path.exists(p):
        return None
    s = pd.read_pickle(p)['Close'].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def cluster_t(df, col):
    """按月份聚类：先月内平均，再对月均值做 t。"""
    if df.empty:
        return np.nan, 0
    m = df.assign(month=pd.to_datetime(df.date).dt.to_period('M')).groupby('month')[col].mean()
    if len(m) < 2:
        return np.nan, len(m)
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='data/events/events_310.csv')
    ap.add_argument('--out', default='output/_logs/event_study.md')
    ap.add_argument('--horizons', default='1,5,20,60')
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(',')]

    ev = pd.read_csv(args.events)
    ev['code'] = ev.code.astype(str).str.zfill(5)
    ev = ev[ev.stage == 'announcement'].copy()
    ev['announce_ts'] = pd.to_datetime(ev.announce_ts)
    hsi = pd.read_pickle(HSI)['Close'].dropna()
    hsi.index = pd.to_datetime(hsi.index).tz_localize(None).normalize()
    hsi = hsi.sort_index()

    recs, px_cache = [], {}
    for _, r in ev.iterrows():
        code = r.code
        if code not in px_cache:
            px_cache[code] = load_px(code)
        s = px_cache[code]
        if s is None or s.empty:
            continue
        # T+1 入场：第一个 > announce_ts 的交易日
        pos = s.index.searchsorted(r.announce_ts.normalize(), side='right')
        if pos >= len(s):
            continue
        d0 = s.index[pos]
        hpos = hsi.index.searchsorted(d0, side='left')
        if hpos >= len(hsi.index):
            continue
        d0h = hsi.index[hpos]
        for h in horizons:
            if pos + h >= len(s):
                continue
            d1 = s.index[pos + h]
            h1pos = hsi.index.searchsorted(d1, side='left')
            if h1pos >= len(hsi.index):
                continue
            d1h = hsi.index[h1pos]
            sr = s.loc[d1] / s.loc[d0] - 1
            hr = hsi.loc[d1h] / hsi.loc[d0h] - 1
            recs.append({'code': code, 'name': r['name'], 'event_type': r.event_type,
                         'direction': r.direction, 'date': d0, 'horizon': h,
                         'car': sr - hr})

    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv('output/_logs/event_car.csv', index=False)

    lines = ['# P1 事件研究（T+1 入场，超额 vs HSI）', '',
             f'- 事件源：`{args.events}`；仅 `stage=announcement`',
             f'- 样本：{len(df)} 条事件-周期；事件 {df[["code","event_type","date"]].drop_duplicates().shape[0] if len(df) else 0} 个',
             '', '| 事件 | 周期 | N | 均值 | 中位 | 负占比 | 月聚类 t | 月数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for typ in ['PROFIT_WARNING', 'POSITIVE_PROFIT_ALERT', 'PLACEMENT_COMPANY',
                'RIGHTS_ISSUE', 'CB_ISSUANCE', 'PRIVATISATION', 'SCHEME',
                'PLACEMENT_SHAREHOLDER', 'LOCKUP']:
        sub = df[df.event_type == typ]
        if sub.empty:
            continue
        for h in horizons:
            x = sub[sub.horizon == h]
            if x.empty:
                continue
            t, nm = cluster_t(x, 'car')
            lines.append(f"| {typ} | {h}d | {len(x)} | {x.car.mean():+.2%} | {x.car.median():+.2%} | "
                         f"{(x.car < 0).mean():.0%} | {t:+.2f} | {nm} |")
    lines += ['', '> t 为按月份聚类的 t（先月内平均，再对月均值）；重叠事件会高估独立观测。',
              '> 事件研究是描述性，不等于可交易；P2 才做增量 IC。']
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
