#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史股本（yfinance get_shares_full；FF301 最新月为官方校验）。

输出：data/shares_history.csv（code,date,shares,source）
"""
import argparse
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WATCHLIST

OUT = 'data/shares_history.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=None)
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default=None)
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(',')] if args.codes else list(WATCHLIST)
    rows = []
    for cfg in codes:
        c5 = cfg.split('.')[0].zfill(5)
        try:
            s = yf.Ticker(cfg).get_shares_full(start=args.start, end=args.end)
            if s is None or s.empty:
                print(cfg, 'empty'); continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            for d, v in s.items():
                if pd.notna(v) and v > 0:
                    rows.append({'code': c5, 'date': d.date().isoformat(),
                                 'shares': int(v), 'source': 'yfinance'})
            print(cfg, len(s))
        except Exception as e:
            print(cfg, 'ERR', str(e)[:80])
    df = pd.DataFrame(rows).drop_duplicates(subset=['code', 'date'], keep='last')
    df.to_csv(OUT, index=False)
    print(f'{len(df)} 行 / {df.code.nunique() if len(df) else 0} 只 → {OUT}')


if __name__ == '__main__':
    main()
