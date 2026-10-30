#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定投 vs 一次性：终值 + IRR（资金加权）比较（D105）。

为什么不用终值单独判断：定投的钱在场时间短，终值天然低；IRR 才回答
「每一块钱在场上每年赚多少」。两者要同时报。

用法：
  python3 scripts/dca_eval.py --code 0700.HK --start 2016-10-17 --end 2026-10-16
  python3 scripts/dca_eval.py --index hsi --start 2016-10-17 --end 2026-10-16
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_services.tencent_finance import get_hsi_data_tencent, get_hk_stock_data_tencent


def _norm(idx):
    idx = pd.to_datetime(idx)
    return idx.tz_convert(None) if idx.tz is not None else idx


def load_prices(code=None, index=None, start=None, end=None):
    if index == 'hsi':
        df = get_hsi_data_tencent(period_days=5000)
        if df is None or df.empty:
            p = 'HSI.pkl'
            if not os.path.exists(p):
                raise SystemExit('HSI 数据不可用')
            df = pd.read_pickle(p)
    else:
        c = code.split('.')[0]
        p = f"data/hk_broad/snapshot/{c.zfill(5)}.pkl"
        if os.path.exists(p):
            df = pd.read_pickle(p)
        else:
            df = get_hk_stock_data_tencent(c, period_days=5000)
    if df is None or df.empty:
        raise SystemExit('价格数据不可用')
    df.index = _norm(df.index).normalize()
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df['Close'].dropna()


def xirr(cashflows):
    """cashflows: [(date, amount)]；负数=投入，正数=收回。返回年化 IRR。"""
    if len(cashflows) < 2:
        return np.nan
    t0 = cashflows[0][0]

    def npv(r):
        return sum(a / (1 + r) ** ((d - t0).days / 365.0) for d, a in cashflows)

    try:
        return brentq(npv, -0.99, 10.0)
    except Exception:
        return np.nan


def max_drawdown(px):
    cum = px / px.iloc[0]
    return float((cum / cum.cummax() - 1).min())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--code')
    g.add_argument('--index', choices=['hsi'])
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--amount', type=float, default=1.0, help='每期投入（默认 1 单位）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    px = load_prices(args.code, args.index, args.start, args.end)
    monthly = px.resample('ME').last().dropna()
    if len(monthly) < 6:
        raise SystemExit('样本不足（<6 个月）')

    n = len(monthly)
    total = n * args.amount
    units = (args.amount / monthly).sum()
    dca_terminal = units * monthly.iloc[-1]
    lump_units = total / monthly.iloc[0]
    lump_terminal = lump_units * monthly.iloc[-1]

    dca_cf = [(d, -args.amount) for d in monthly.index]
    dca_cf.append((monthly.index[-1], dca_terminal))
    lump_cf = [(monthly.index[0], -total), (monthly.index[-1], lump_terminal)]

    dca_irr = xirr(dca_cf)
    lump_irr = xirr(lump_cf)

    lines = ['# 定投 vs 一次性', '',
             f'- 标的：{args.index or args.code}；{monthly.index[0].date()} → {monthly.index[-1].date()}（{n} 个月）',
             f'- 总投入：{total:.2f}；标的区间最大回撤：{max_drawdown(px):.1%}', '',
             '| 方式 | 终值 | 盈亏 | IRR（年化） |', '|---|---|---|---|',
             f'| 月定投 | {dca_terminal:.2f} | {dca_terminal-total:+.2f} | {dca_irr:+.2%} |',
             f'| 期初一次性 | {lump_terminal:.2f} | {lump_terminal-total:+.2f} | {lump_irr:+.2%} |',
             '',
             '> 读法：终值回答「最后谁的钱多」；IRR 回答「每一块钱在场上每年赚多少」。',
             '> 定投终值低不一定是差，可能只是钱在场时间短；两者必须同时报。']
    text = '\n'.join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, 'w').write(text + '\n')
        print(f'\n报告: {args.out}')


if __name__ == '__main__':
    main()
