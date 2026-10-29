#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方案 A：raw 打分 + 板块持仓上限（预注册扫描）。

预注册（跑之前写定）：
  - 成功 = 至少一组「超额 IR 95%CI 下界 > 0」且相邻板块上限（±1 档）符号一致
  - 否则判 null
网格：TopK ∈ {10,20,30} × 板块上限 ∈ {2,3,5,None} × dropout ∈ {0,2}
成本：双边 0.5%（与 walk_forward TOTAL_COST 一致）
打分：raw Predict_Prob（不中性）

用法：
    python3 scripts/sector_cap_backtest.py --pred output/<dir>/prediction_analysis.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_services.overlap_stats import block_bootstrap_se  # noqa: E402
from ml_services.portfolio_backtest import load_panel  # noqa: E402

from ml_services.costs import COST_ROUND_TRIP


def run_config(df, horizon, topk, cap, dropout, costs=COST_ROUND_TRIP):
    dates = sorted(df['date'].unique())
    rb = dates[::horizon]
    rows, prev, held = [], set(), []
    for d in rb:
        g = df[df['date'] == d].copy()
        if len(g) < 20:
            continue
        g = g.sort_values('prob', ascending=False, kind='mergesort')
        bench = g['ret'].mean()

        if dropout and held:
            pool = g[g['code'].isin(held)]
            keep = []
            for c, s in zip(pool['code'], pool['sector']):
                if sum(1 for _, ks in keep if ks == s) < (cap or 10 ** 6):
                    keep.append((c, s))
                if len(keep) >= topk - dropout:
                    break
            sel, cnt = [c for c, _ in keep], {}
            for _, s in keep:
                cnt[s] = cnt.get(s, 0) + 1
            for c, s in zip(g['code'], g['sector']):
                if len(sel) >= topk:
                    break
                if c in sel:
                    continue
                if cnt.get(s, 0) < (cap or 10 ** 6):
                    sel.append(c); cnt[s] = cnt.get(s, 0) + 1
        else:
            sel, cnt = [], {}
            for c, s in zip(g['code'], g['sector']):
                if len(sel) >= topk:
                    break
                if cnt.get(s, 0) < (cap or 10 ** 6):
                    sel.append(c); cnt[s] = cnt.get(s, 0) + 1
        k = max(len(sel), 1)
        long = g[g['code'].isin(sel)]
        cur = set(sel)
        turn = len(cur - prev) / k if prev else 1.0
        prev, held = cur, sel
        top_gross = long['ret'].mean()
        rows.append(dict(date=d, bench=bench, top_gross=top_gross,
                         top_net=top_gross - turn * costs, turnover=turn, n=k))
    bt = pd.DataFrame(rows)
    ex = bt['top_net'] - bt['bench']
    ann = np.sqrt(252 / horizon)
    se = block_bootstrap_se(ex.values, max(1, 252 // horizon))
    return dict(
        ir=float(bt['top_net'].mean() / bt['top_net'].std(ddof=1) * ann),
        ex_mean=float(ex.mean() * 100), ex_ir=float(ex.mean() / ex.std(ddof=1) * ann),
        lo=float((ex.mean() - 1.96 * se) * 100), hi=float((ex.mean() + 1.96 * se) * 100),
        wr=float((bt['top_net'] > 0).mean()), turn=float(bt['turnover'].mean()),
        n=len(bt), bt=bt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--horizon', type=int, default=20)
    args = ap.parse_args()

    df = load_panel(args.pred)
    print(f'样本 {len(df)} 行 / {df.date.nunique()} 日 / {df.code.nunique()} 只 / '
          f'板块 {df.sector.nunique()} 个')

    rows = []
    for topk in (10, 20, 30):
        for cap in (2, 3, 5, None):
            for dropout in (0, 2):
                r = run_config(df, args.horizon, topk, cap, dropout)
                rows.append(dict(TopK=topk, 板块上限=cap or '∞', dropout=dropout,
                                 净IR=r['ir'], 超额IR=r['ex_ir'],
                                 CI下=r['lo'], CI上=r['hi'],
                                 净超额pp=r['ex_mean'], 换手=r['turn'], 胜率=r['wr']))
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    print('\n=== 预注册判据 ===')
    ok = out[(out['CI下'] > 0)]
    if ok.empty:
        print('❌ 无任何组合的 CI 下界 > 0 ⇒ 判 null')
    else:
        for _, r in ok.iterrows():
            print(f"  ✓ TopK={r['TopK']} cap={r['板块上限']} dropout={r['dropout']}: "
                  f"超额IR {r['超额IR']:.2f} CI[{r['CI下']:+.3f},{r['CI上']:+.3f}]")

    print('\n=== 逐年净超额（pp/期）===')
    for topk in (10, 20, 30):
        for cap in (3, None):
            r = run_config(df, args.horizon, topk, cap, 0)
            bt = r['bt'].copy(); bt['y'] = pd.to_datetime(bt['date']).dt.year
            ex = (bt['top_net'] - bt['bench']).groupby(bt['y']).mean() * 100
            print(f'TopK={topk} cap={cap or "∞"}: ' +
                  ' '.join(f'{y}:{v:+.2f}' for y, v in ex.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
