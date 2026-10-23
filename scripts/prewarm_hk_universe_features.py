#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预热宽基池特征缓存（B′ 实验用）。

只调用 prepare_data（逐股写特征缓存），不训练。供随后 WF 命中缓存用。

用法：
    HK_UNIVERSE_FILE=data/hk_broad/universe_2020-06-30.json \
    HK_MARKET_SNAPSHOT_DIR=data/hk_broad/snapshot \
    python3 scripts/prewarm_hk_universe_features.py --start 2019-06-01 --end 2026-10-20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2019-06-01')
    ap.add_argument('--end', default='2026-10-20')
    ap.add_argument('--horizon', type=int, default=20)
    args = ap.parse_args()

    import config
    from ml_services.ml_trading_model import CatBoostModel
    from ml_services.network_ids import resolve_community_ids

    codes = list(config.TRAINING_STOCKS)
    print(f'池 {len(codes)} 只，区间 {args.start} → {args.end}', flush=True)
    t0 = time.time()
    m = CatBoostModel()
    df = m.prepare_data(codes, start_date=args.start, end_date=args.end,
                        horizon=args.horizon, for_backtest=False,
                        label_mode='rank', mode='production',
                        community_ids=resolve_community_ids())
    print(f'✅ 完成: {len(df)} 行 × {df.shape[1]} 列，耗时 {time.time() - t0:.0f}s', flush=True)
    print(f'索引 {df.index.min()} ~ {df.index.max()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
