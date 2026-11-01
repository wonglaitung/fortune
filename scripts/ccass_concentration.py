#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCASS 集中度快照（使用 FF301 官方股本作分母）。

用法：
  python3 scripts/ccass_concentration.py --date latest --top 20
输出：output/_logs/ccass_concentration.md + data/ccass_concentration.csv
"""
import argparse
import glob
import os

import pandas as pd

FF301 = 'data/hkex_monthly_shares.csv'


def latest_dir(root):
    ds = [d for d in glob.glob(os.path.join(root, '20*')) if os.path.isdir(d)]
    return max(ds) if ds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ccass')
    ap.add_argument('--date', default='latest')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--out', default='output/_logs/ccass_concentration.md')
    args = ap.parse_args()
    d = latest_dir(args.root) if args.date == 'latest' else os.path.join(args.root, args.date)
    if not d or not os.path.isdir(d):
        raise SystemExit('CCASS 目录不存在')

    ff = pd.read_csv(FF301, dtype={'code': str})
    ff['code'] = ff.code.str.zfill(5)
    ff = ff.set_index('code')[['issued', 'total']].to_dict('index')

    rows = []
    for p in glob.glob(os.path.join(d, '*.csv')):
        code = os.path.basename(p)[:-4]
        if code.startswith('_'):
            continue
        try:
            x = pd.read_csv(p, dtype={'participant_id': str})
        except Exception:
            continue
        if x.empty or 'pct' not in x:
            continue
        pct = pd.to_numeric(x.pct, errors='coerce').dropna().sort_values(ascending=False)
        if pct.empty:
            continue
        shares = pd.to_numeric(x.shares, errors='coerce').fillna(0).sum()
        iss = ff.get(code, {})
        issued = iss.get('issued') or iss.get('total')
        rows.append({
            'code': code, 'date': os.path.basename(d),
            'n_participants': len(pct),
            'top1_pct': pct.head(1).sum(), 'top5_pct': pct.head(5).sum(),
            'top10_pct': pct.head(10).sum(),
            'hhi': float(((pct / 100.0) ** 2).sum()),
            'ccass_shares': shares,
            'issued_ff301': issued,
            'ccass_coverage': shares / issued if issued else None,
        })
    df = pd.DataFrame(rows).sort_values('top5_pct', ascending=False)
    df.to_csv('data/ccass_concentration.csv', index=False)
    top = df.head(args.top)
    lines = ['# CCASS 集中度（FF301 股本）', '',
             f'- 日期：{os.path.basename(d)}；股票数：{len(df)}',
             '- top5/top10 = CCASS 参与者持股占比；HHI = Σ(pct/100)²；coverage = CCASS 股数/官方股本',
             '', '| 代码 | 参与者 | Top1 | Top5 | Top10 | HHI | CCASS覆盖率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for _, r in top.iterrows():
        cov = f"{r.ccass_coverage:.0%}" if pd.notna(r.ccass_coverage) else '—'
        lines.append(f"| {r.code} | {r.n_participants} | {r.top1_pct:.1f}% | {r.top5_pct:.1f}% | "
                     f"{r.top10_pct:.1f}% | {r.hhi:.3f} | {cov} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
