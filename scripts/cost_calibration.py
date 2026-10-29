#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本模型校准：按 ADV 分位/佣金档/单笔金额输出有效往返成本。"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.costs import SideCostModel
from ml_services.exposures import load_exposures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--out', default='output/_logs/cost_calibration.md')
    args = ap.parse_args()

    df = pd.read_csv(args.pred, usecols=['Date', 'Stock_Code'])
    df['date'] = pd.to_datetime(df['Date']).dt.normalize()
    df['code'] = df['Stock_Code']
    dates = sorted(df['date'].unique())[::args.horizon]
    codes = sorted(df['code'].unique())
    adv, sig, spr = load_exposures(codes)

    rows = []
    for d in dates:
        names = df.loc[df['date'] == d, 'code'].tolist()
        if len(names) < 20:
            continue
        a = adv.loc[d, names] if d in adv.index else pd.Series(dtype=float)
        s = sig.loc[d, names] if d in sig.index else pd.Series(dtype=float)
        p = spr.loc[d, names] if d in spr.index else pd.Series(dtype=float)
        q = pd.qcut(a.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        for code in names:
            if code not in a.index or not np.isfinite(a.get(code, np.nan)):
                continue
            rows.append({'date': d, 'code': code, 'adv': a[code],
                         'sigma': s.get(code, np.nan), 'spread': p.get(code, np.nan),
                         'q': q.get(code)})
    base = pd.DataFrame(rows).dropna(subset=['adv'])

    lines = ['# 成本模型校准（往返 %）', '',
             f'- 输入: `{args.pred}`；调仓日样本 {base["date"].nunique()} 个；股票 {base["code"].nunique()} 只', '']

    lines += ['## 按 ADV 五分位（佣金 0.10%，单笔 5 万，k=1）', '',
              '| ADV 分位 | 中位往返 | 均值往返 | 中位 ADV(HKD) |', '|---|---|---|---|']
    m = SideCostModel(broker_rate=0.001, min_fee=50.0, impact_k=1.0)
    tmp = base.copy()
    tmp['rt'] = [2 * m.per_side(50000, r.adv, r.sigma, r.spread)['total'] for r in tmp.itertuples()]
    for q, g in tmp.groupby('q', observed=True):
        lines.append(f"| {q} | {g['rt'].median():.2%} | {g['rt'].mean():.2%} | {g['adv'].median():,.0f} |")
    lines += ['', f"整体中位往返：**{tmp['rt'].median():.2%}**（flat 基准 0.50%）", '']

    lines += ['## 佣金档 × 单笔金额（整体中位往返）', '',
              '| 佣金 | 5 万 | 20 万 | 100 万 |', '|---|---|---|---|']
    for rate in (0.0003, 0.001, 0.0025):
        mm = SideCostModel(broker_rate=rate, min_fee=50.0, impact_k=1.0)
        vals = []
        for tv in (50000, 200000, 1000000):
            r = np.median([2 * mm.per_side(tv, x.adv, x.sigma, x.spread)['total']
                           for x in base.itertuples()])
            vals.append(f'{r:.2%}')
        lines.append(f"| {rate:.2%} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ['', '## 冲击系数敏感性（佣金 0.10%，单笔 5 万）', '',
              '| k | 整体中位往返 |', '|---|---|']
    for k in (0.5, 1.0, 2.0):
        mk = SideCostModel(broker_rate=0.001, min_fee=50.0, impact_k=k)
        r = np.median([2 * mk.per_side(50000, x.adv, x.sigma, x.spread)['total']
                       for x in base.itertuples()])
        lines.append(f'| {k} | {r:.2%} |')

    text = '\n'.join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write(text + '\n')
    print(f'\n报告: {args.out}')


if __name__ == '__main__':
    main()
