#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为宽基池生成 STOCK_SECTOR_MAPPING 兼容元数据（B′ 方案）。

输入：data/hk_broad/membership.csv（PIT 月度前 N）
输出：data/hk_broad/universe_<month>.json（code -> {sector,name,type,defensive,...}）

行业：yfinance `.info['sector'/'industry']` → 项目 sector/type（关键词映射）
评分：同 sector 的现有 56 只中位数；无同类则用全局中位数
缓存：data/hk_broad/yf_meta_cache.json（避免重复请求）

用法：
    python3 scripts/build_hk_broad_meta.py --month 2020-06-30 --topn 300
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'data', 'hk_broad')
CACHE = os.path.join(OUT, 'yf_meta_cache.json')
SCORE_KEYS = ['defensive', 'growth', 'cyclical', 'liquidity', 'risk']


def map_sector(yf_sector: str | None, yf_industry: str | None) -> tuple[str, str]:
    s = (yf_sector or '').lower()
    ind = (yf_industry or '').lower()
    if 'financial' in s:
        if 'bank' in ind:
            return 'bank', 'bank'
        if 'insurance' in ind:
            return 'insurance', 'insurance'
        return 'financial', 'financial'
    if 'technology' in s:
        if 'semiconductor' in ind:
            return 'semiconductor', 'semiconductor'
        return 'tech', 'tech'
    if 'communication' in s:
        return 'tech', 'tech'
    if 'consumer cyclical' in s:
        if 'auto' in ind:
            return 'auto', 'auto'
        return 'consumer', 'consumer'
    if 'consumer defensive' in s:
        return 'consumer', 'consumer'
    if 'healthcare' in s:
        return 'biotech', 'biotech'
    if 'industrials' in s:
        if 'shipping' in ind or 'marine' in ind:
            return 'shipping', 'shipping'
        return 'industrial', 'industrial'
    if 'energy' in s:
        return 'energy', 'energy'
    if 'basic materials' in s:
        return 'materials', 'materials'
    if 'utilities' in s:
        if 'gas' in ind:
            return 'gas', 'gas'
        return 'utility', 'utility'
    if 'real estate' in s:
        return 'real_estate', 'real_estate'
    return 'other', 'other'


def fetch_meta(codes: list[str]) -> dict:
    cache = json.load(open(CACHE, encoding='utf-8')) if os.path.exists(CACHE) else {}
    todo = [c for c in codes if c not in cache]
    print(f'元数据：已缓存 {len(cache)}，待抓 {len(todo)}', flush=True)
    if todo:
        import yfinance as yf
        for i, code in enumerate(todo, 1):
            tk = f'{int(code):04d}.HK'
            info = {}
            for attempt in range(2):
                try:
                    info = yf.Ticker(tk).info or {}
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 1:
                        print(f'  [{i}/{len(todo)}] {tk} 失败 {str(e)[:50]}', flush=True)
                    time.sleep(1.0)
            cache[code] = {'sector': info.get('sector'), 'industry': info.get('industry'),
                           'name': info.get('longName') or info.get('shortName') or code}
            if i % 20 == 0:
                json.dump(cache, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)
                print(f'  {i}/{len(todo)} done', flush=True)
            time.sleep(0.15)
        json.dump(cache, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)
    return cache


def median_scores() -> tuple[dict, dict]:
    import config
    sm = config.STOCK_SECTOR_MAPPING
    df = pd.DataFrame([{'sector': v['sector'], **{k: v[k] for k in SCORE_KEYS}}
                       for v in sm.values()])
    by = df.groupby('sector')[SCORE_KEYS].median().to_dict('index')
    glob = df[SCORE_KEYS].median().to_dict()
    return by, glob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='2020-06-30')
    ap.add_argument('--topn', type=int, default=300)
    args = ap.parse_args()

    m = pd.read_csv(os.path.join(OUT, 'membership.csv'), dtype={'code': str})
    m['code'] = m.code.str.zfill(5)
    sub = m[m['month'] <= args.month]
    mon = sub['month'].max()
    u = sub[sub['month'] == mon].sort_values('rank').head(args.topn)
    codes = u['code'].tolist()
    print(f'冻结月 {mon}，池 {len(codes)} 只', flush=True)

    meta = fetch_meta(codes)
    by, glob = median_scores()

    out = {}
    for code in codes:
        info = meta.get(code, {})
        sector, typ = map_sector(info.get('sector'), info.get('industry'))
        scores = by.get(sector, glob)
        key = f'{int(code):04d}.HK' if int(code) < 10000 else f'{code}.HK'
        out[key] = {
            'sector': sector, 'type': typ,
            'name': info.get('name') or code,
            **{k: float(scores[k]) for k in SCORE_KEYS},
        }
    path = os.path.join(OUT, f'universe_{mon}.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    import collections
    print('sector 分布:', dict(collections.Counter(v['sector'] for v in out.values())), flush=True)
    print('写入', path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
