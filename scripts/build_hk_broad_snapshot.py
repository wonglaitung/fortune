#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 data/hk_broad 的宽表行情导出为个股快照 pkl（供 HK_MARKET_SNAPSHOT_DIR 使用）。

用途：扩池回测时，pipeline 的腾讯接口在 >1500 天会静默截断到 640 行，
       必须走冻结快照。本脚本零网络、确定性输出。

用法：
    python3 scripts/build_hk_broad_snapshot.py --universe data/hk_broad/universe_2020-06-30.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'hk_broad')
FIELDS = ['open', 'high', 'low', 'close', 'volume']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', required=True)
    ap.add_argument('--out', default=os.path.join(SRC, 'snapshot'))
    args = ap.parse_args()

    uni = json.load(open(args.universe, encoding='utf-8'))
    codes = [c.replace('.HK', '').zfill(5) for c in uni]
    os.makedirs(args.out, exist_ok=True)

    panels = {f: pd.read_parquet(os.path.join(SRC, f'panel_{f}.parquet')) for f in FIELDS}
    ok = nodata = 0
    for code in codes:
        cols = {f.capitalize(): panels[f][code] for f in FIELDS if code in panels[f].columns}
        if not cols or cols['Close'].dropna().empty:
            open(os.path.join(args.out, f'{code}.nodata'), 'w').close()
            nodata += 1
            continue
        df = pd.DataFrame(cols).dropna(subset=['Close']).sort_index()
        df.index = pd.to_datetime(df.index).tz_localize('UTC')
        df.index.name = 'Date'
        with open(os.path.join(args.out, f'{code}.pkl'), 'wb') as f:
            pickle.dump(df, f)
        ok += 1
    print(f'快照写入 {args.out}: {ok} 只 / nodata {nodata}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
