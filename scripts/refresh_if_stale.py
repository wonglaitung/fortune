#!/usr/bin/env python3
"""过期数据自动刷新：财务三表 / 价格快照。新鲜则跳过，不重复抓取。"""
import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FUND = os.path.join(ROOT, 'data', 'fund_cache')
SNAP = os.path.join(ROOT, 'data', 'hk_broad', 'snapshot')
UNIV = os.path.join(ROOT, 'data', 'hk_broad', 'universe_310.json')


def age_days(path):
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 86400


def refresh_fund(code, max_age, dry):
    from ml_services.feature_engineering.f1_fundamental_features import (
        fetch_one_statement, STATEMENTS, INDICATORS,
    )
    out = []
    for sym, _ in STATEMENTS:
        for ind, _ in INDICATORS:
            path = os.path.join(FUND, f'{code}_{sym}_{ind}.pkl')
            age = age_days(path)
            if age is not None and age <= max_age:
                out.append((f'{sym}_{ind}', 'SKIP', f'fresh {age:.1f}d'))
                continue
            if dry:
                out.append((f'{sym}_{ind}', 'WOULD_REFRESH',
                            'missing' if age is None else f'stale {age:.1f}d'))
                continue
            stale = path + '.stale'
            if os.path.exists(path):
                os.replace(path, stale)
            try:
                df = fetch_one_statement(code, sym, ind)
            except Exception as exc:
                df = pd.DataFrame()
                out.append((f'{sym}_{ind}', 'ERROR', str(exc)[:80]))
            if os.path.exists(stale):
                if df is None or df.empty:
                    os.replace(stale, path)
                else:
                    os.remove(stale)
            last = df.index[-1].date() if df is not None and len(df) else 'EMPTY'
            out.append((f'{sym}_{ind}', 'REFRESH', f'as_of={last}'))
    return out


def refresh_price(code, max_age, dry):
    from data_services.tencent_finance import get_hk_stock_data_tencent
    c5 = code.split('.')[0].zfill(5)
    path = os.path.join(SNAP, c5 + '.pkl')
    age = age_days(path)
    if age is not None and age <= max_age:
        return ('SKIP', f'fresh {age:.1f}d')
    if dry:
        return ('WOULD_REFRESH', 'missing' if age is None else f'stale {age:.1f}d')
    try:
        d = get_hk_stock_data_tencent(c5, 30)
    except Exception as exc:
        return ('ERROR', str(exc)[:80])
    if d is None or d.empty:
        return ('ERROR', 'empty quote')
    d = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).tz_localize('UTC')
    if os.path.exists(path):
        old = pd.read_pickle(path)
        new = d[d.index > old.index.max()]
        merged = pd.concat([old, new]).sort_index()
        added = len(new)
    else:
        merged = d
        added = len(d)
    tmp = path + '.tmp'
    pd.to_pickle(merged, tmp)
    os.replace(tmp, path)
    return ('REFRESH', f'as_of={merged.index.max().date()} rows+={added}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default='', help='comma codes, e.g. 0700.HK,9988.HK')
    ap.add_argument('--all-fund', action='store_true', help='all 310 pool codes')
    ap.add_argument('--sources', default='fund_cache,prices')
    ap.add_argument('--fund-max-age-days', type=float, default=7)
    ap.add_argument('--price-max-age-days', type=float, default=5)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    codes = [c for c in args.codes.split(',') if c]
    if args.all_fund:
        codes += list(json.load(open(UNIV)).keys())
    codes = list(dict.fromkeys(codes))
    if not codes:
        ap.error('need --codes or --all-fund')
    src = {s for s in args.sources.split(',') if s}
    for code in codes:
        print(f'### {code}')
        if 'fund_cache' in src:
            for name, action, detail in refresh_fund(code, args.fund_max_age_days, args.dry_run):
                print(f'  fund_cache {name:24s} {action:14s} {detail}')
        if 'prices' in src:
            action, detail = refresh_price(code, args.price_max_age_days, args.dry_run)
            print(f'  prices     {"snapshot":24s} {action:14s} {detail}')


if __name__ == '__main__':
    main()
