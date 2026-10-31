#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据新鲜度检查（P1）：只看“最后日期/年龄”，不做质量判断。

用法：
  python3 scripts/data_freshness.py
  python3 scripts/data_freshness.py --strict   # 有 ❌ 时非零退出
"""
import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _age(d):
    return (pd.Timestamp.today().normalize() - pd.Timestamp(d).normalize()).days


def snapshot_date():
    fs = glob.glob(os.path.join(ROOT, 'data/hk_broad/snapshot/*.pkl'))
    best = None
    for f in fs[:80]:
        try:
            idx = pd.read_pickle(f).index
            if len(idx):
                v = pd.Timestamp(idx.max()).tz_localize(None).normalize()
                best = v if best is None or v > best else best
        except Exception:
            continue
    return best


def announcements_date():
    p = os.path.join(ROOT, 'data/hkex_announcements/all.csv')
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, usecols=['date_time'])
            return pd.to_datetime(df.date_time, dayfirst=True, errors='coerce').max()
        except Exception:
            pass
    fs = glob.glob(os.path.join(ROOT, 'data/hkex_announcements/*.json'))
    return datetime.fromtimestamp(max(os.path.getmtime(f) for f in fs)) if fs else None


def events_date():
    p = os.path.join(ROOT, 'data/events/events_310.csv')
    if not os.path.exists(p):
        return None
    return pd.to_datetime(pd.read_csv(p).announce_ts, errors='coerce').max()


def sfc_date():
    p = os.path.join(ROOT, 'data/sfc_short/sfc_short_long.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=['date'])
    return pd.to_datetime(df.date.astype(str), format='%Y%m%d', errors='coerce').max()


def ccass_date():
    ds = [os.path.basename(d) for d in glob.glob(os.path.join(ROOT, 'data/ccass/20*')) if os.path.isdir(d)]
    return pd.to_datetime(max(ds), format='%Y%m%d') if ds else None


def model_date():
    fs = glob.glob(os.path.join(ROOT, 'data/ml_trading_model_lgbm_*d.pkl'))
    return datetime.fromtimestamp(max(os.path.getmtime(f) for f in fs)) if fs else None


def hsi_pit_date():
    p = os.path.join(ROOT, 'data/hsi_constituents_pit.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    for c in ('date', 'effective_date', 'month'):
        if c in df.columns:
            return pd.to_datetime(df[c], errors='coerce').max()
    return None


def fund_cache_date():
    fs = glob.glob(os.path.join(ROOT, 'data/fund_cache/*.pkl'))
    return datetime.fromtimestamp(max(os.path.getmtime(f) for f in fs)) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--skip-local', action='store_true', help='跳过仅本地存在的快照/CCASS/fund_cache')
    args = ap.parse_args()

    checks = [
        ('行情快照', snapshot_date(), 3, 7),
        ('公告', announcements_date(), 7, 14),
        ('事件表', events_date(), 7, 14),
        ('SFC 卖空', sfc_date(), 10, 21),
        ('CCASS', ccass_date(), 3, 7),
        ('港股模型 pkl', model_date(), 35, 45),
        ('恒指 PIT 成分', hsi_pit_date(), 45, 90),
        ('财报表 fund_cache', fund_cache_date(), 200, 365),
    ]
    lines = ['# 数据新鲜度', '', '| 数据 | 最后日期 | 年龄（天） | 状态 |', '|---|---:|---:|---|']
    bad = False
    local_only = {'行情快照', 'CCASS', '财报表 fund_cache'}
    for name, d, warn, fail in checks:
        if args.skip_local and name in local_only:
            lines.append(f'| {name} | — | — | ⏭️ 跳过（Action） |')
            continue
        if d is None or pd.isna(d):
            lines.append(f'| {name} | — | — | ⚠️ 缺失 |')
            continue
        age = _age(d)
        if age > fail:
            st = '❌ 过期'; bad = True
        elif age > warn:
            st = '⚠️ 偏旧'
        else:
            st = '✅'
        lines.append(f'| {name} | {pd.Timestamp(d).date()} | {age} | {st} |')
    lines += ['', '> 阈值：快照/CCASS 3/7 天；公告/事件 7/14；SFC 10/21；模型 35/45；恒指 PIT 45/90；财报 200/365。']
    text = '\n'.join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, 'w').write(text + '\n')
    if args.strict and bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
