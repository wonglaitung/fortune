#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DI（权益披露）事件研究：大股东买入/卖出 → 前瞻超额收益。

口径：
- 只用原因码 1101（买入）与 1201（卖出）；其余（期权/信托/借贷/其他）排除；
- PIT：以「公告日 filing_date」为可用日，T+1 入场；
- 超额 = 个股 − HSI；按月份聚类 t。
"""
import argparse
import os

import numpy as np
import pandas as pd

SNAP = 'data/hk_broad/snapshot'
HSI = 'data/hk_universe_cache/HSI.pkl'


def load_px(code5):
    p = f'{SNAP}/{str(code5).zfill(5)}.pkl'
    if not os.path.exists(p):
        return None
    s = pd.read_pickle(p)['Close'].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def nw_t(x, lag=4):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3:
        return np.nan
    mu = x.mean(); e = x - mu; var = (e @ e) / len(x)
    for l in range(1, min(lag, len(x) - 1) + 1):
        var += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / len(x)
    return float(mu / np.sqrt(max(var, 1e-18) / len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--di', default='data/di/di_notices.csv')
    ap.add_argument('--horizons', default='5,20,60')
    ap.add_argument('--out', default='output/_logs/di_signal.md')
    args = ap.parse_args()
    hs = [int(x) for x in args.horizons.split(',')]

    d = pd.read_csv(args.di, dtype={'code': str})
    d['code'] = d.code.str.zfill(5)
    d['filing_date'] = pd.to_datetime(d.filing_date, errors='coerce')
    d['event_date'] = pd.to_datetime(d.event_date, errors='coerce')
    d['shares_long'] = pd.to_numeric(d.shares_long, errors='coerce')
    d['avg_price'] = pd.to_numeric(d.avg_price, errors='coerce')
    d = d.dropna(subset=['filing_date', 'shares_long'])
    d = d[d.shares_long > 0]
    d['side'] = np.where(d.reason.astype(str).str.startswith('1101'), 'BUY',
                  np.where(d.reason.astype(str).str.startswith('1201'), 'SELL', None))
    d = d[d.side.notna()]
    d = d.sort_values('filing_date').drop_duplicates(
        subset=['code', 'side', 'event_date', 'name', 'shares_long'], keep='last')

    hsi = pd.read_pickle(HSI)['Close'].dropna()
    hsi.index = pd.to_datetime(hsi.index).tz_localize(None).normalize()
    hsi = hsi.sort_index()
    px = {}
    recs = []
    for _, r in d.iterrows():
        c = r.code
        if c not in px:
            px[c] = load_px(c)
        s = px[c]
        if s is None or s.empty:
            continue
        pos = s.index.searchsorted(r.filing_date.normalize(), side='right')
        if pos >= len(s):
            continue
        d0 = s.index[pos]
        h0 = hsi.index.searchsorted(d0, side='left')
        if h0 >= len(hsi):
            continue
        for h in hs:
            if pos + h >= len(s):
                continue
            d1 = s.index[pos + h]
            h1 = hsi.index.searchsorted(d1, side='left')
            if h1 >= len(hsi):
                continue
            car = (s.loc[d1] / s.loc[d0] - 1) - (hsi.iloc[h1] / hsi.iloc[h0] - 1)
            recs.append({'code': c, 'side': r.side, 'filing_date': r.filing_date,
                         'event_date': r.event_date, 'horizon': h, 'car': car,
                         'shares': r.shares_long, 'name': r['name']})
    ev = pd.DataFrame(recs)
    lines = ['# DI 大股东买入/卖出 → 前瞻超额（T+1）', '',
             f'- 样本：{d.code.nunique()} 只；{d.filing_date.min().date()} → {d.filing_date.max().date()}',
             '- 仅 1101 买入 / 1201 卖出；超额 vs HSI；t 按公告月聚类', '',
             '| 方向 | 周期 | N | 均值 | 中位 | 正占比 | 月聚类 t | 月数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    if ev.empty:
        lines.append('| — | — | 0 | — | — | — | — | — |')
    else:
        for side in ['BUY', 'SELL']:
            for h in hs:
                x = ev[(ev.side == side) & (ev.horizon == h)]
                if x.empty:
                    continue
                m = x.assign(month=x.filing_date.dt.to_period('M')).groupby('month').car.mean()
                lines.append(f'| {side} | {h}d | {len(x)} | {x.car.mean():+.2%} | {x.car.median():+.2%} | '
                             f'{(x.car > 0).mean():.0%} | {nw_t(m.values):+.2f} | {len(m)} |')
        ev.to_csv('output/_logs/di_events.csv', index=False)
    lines += ['', '> 事件研究是描述性；DI 是 5% 以上大股东/董事披露，样本稀疏，不能直接当信号。']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
