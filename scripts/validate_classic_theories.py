#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用项目数据验证 CLASSIC_TRADING_THEORIES.md 中可检验的命题（D102）。"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_services.tencent_finance import get_hsi_data_tencent

COST = 0.005
SNAP = 'data/hk_broad/snapshot'
START, END = pd.Timestamp('2016-10-17'), pd.Timestamp('2026-10-16')


def norm(i):
    i = pd.to_datetime(i)
    return i.tz_convert(None) if i.tz is not None else i


def load_panel():
    uni = json.load(open('data/hk_broad/universe_310.json'))
    close, high, low, op = {}, {}, {}, {}
    for code in uni:
        p = f"{SNAP}/{code.split('.')[0].zfill(5)}.pkl"
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df.index = norm(df.index).normalize()
        df = df[(df.index >= START) & (df.index <= END)].sort_index()
        if len(df) < 300:
            continue
        close[code] = df['Close']; high[code] = df['High']
        low[code] = df['Low']; op[code] = df['Open']
    return (pd.DataFrame(close).sort_index(), pd.DataFrame(high).sort_index(),
            pd.DataFrame(low).sort_index(), pd.DataFrame(op).sort_index())


def nw_t(x, lags=1):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3:
        return np.nan
    mu = x.mean(); e = x - mu; s = (e @ e) / len(x)
    for l in range(1, min(lags, len(x) - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / len(x)
    return float(mu / np.sqrt(max(s, 1e-18) / len(x)))


def dca(close, months=120):
    s = close.dropna().resample('ME').last().dropna().tail(months)
    if len(s) < 24:
        return None
    units = (1.0 / s).sum(); terminal = units * s.iloc[-1]
    lump = len(s) * (s.iloc[-1] / s.iloc[0])
    return {'months': len(s), 'terminal': terminal, 'lump': lump,
            'ratio': terminal / lump - 1, 'dca_cagr': (terminal / len(s)) ** (12 / len(s)) - 1}


def patterns(close, high, low, op):
    body = (close - op).abs(); rng = (high - low).replace(0, np.nan)
    upper = high - np.maximum(close, op); lower = np.minimum(close, op) - low
    hammer = (lower > 2 * body) & (upper < body) & (body / rng < 0.4)
    prev_bear = (close.shift(1) < op.shift(1))
    bull_engulf = prev_bear & (close > op) & (op <= close.shift(1)) & (close >= op.shift(1))
    d1 = close.shift(2) < op.shift(2)
    d2 = body.shift(1) / rng.shift(1) < 0.3
    d3 = close > op
    mid1 = (op.shift(2) + close.shift(2)) / 2
    morning = d1 & d2 & d3 & (close > mid1)
    return {'hammer': hammer, 'bullish_engulfing': bull_engulf, 'morning_star': morning}


def candlestick(close, high, low, op, horizons=(5, 10, 20)):
    pats = patterns(close, high, low, op)
    out = {}
    for name, mat in pats.items():
        for h in horizons:
            cars, months = [], []
            for code in mat.columns:
                idx = np.where(mat[code].fillna(False).values)[0]
                for i in idx:
                    if i + 1 + h >= len(close):
                        continue
                    entry = close[code].iloc[i + 1]
                    exitp = close[code].iloc[i + 1 + h]
                    if not (entry > 0 and exitp > 0):
                        continue
                    bench = (close.iloc[i + 1 + h] / close.iloc[i + 1] - 1).mean()
                    cars.append(exitp / entry - 1 - bench - COST)
                    months.append(close.index[i + 1].to_period('M'))
            if cars:
                dfm = pd.DataFrame({'car': cars, 'm': months})
                mt = dfm.groupby('m').car.mean()
                out[(name, h)] = {'n': len(dfm), 'mean': dfm.car.mean(),
                                  't': nw_t(mt.values, 1)}
    return out


def main():
    close, high, low, op = load_panel()
    print(f"面板: {close.shape[1]} 只 / {len(close)} 日 / {close.index[0].date()}→{close.index[-1].date()}")
    hsi = get_hsi_data_tencent(period_days=4200)
    hsi.index = norm(hsi.index).normalize(); hsi = hsi.sort_index()
    hsi = hsi[(hsi.index >= START) & (hsi.index <= END)]

    d = dca(hsi['Close'])
    print(f"\nDCA vs 一次性: {d}")
    # 动量：过去 12-1（跳过最近 1 月）
    dates = close.index
    # 自定义：12-1 与 1 月反转
    pos = {dt: i for i, dt in enumerate(dates)}
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).last().values
    def run(sig_fn, side):
        rows, prev = [], set()
        for dt in me:
            i = pos[dt]
            if i - 252 < 0 or i + 21 >= len(dates):
                continue
            sig = sig_fn(i); fwd = close.iloc[i + 21] / close.iloc[i] - 1
            ok = sig.notna() & fwd.notna()
            if ok.sum() < 30: continue
            s, f = sig[ok], fwd[ok]; n = max(1, int(len(s) * 0.1))
            sel = s.nsmallest(n).index if side == 'bottom' else s.nlargest(n).index
            port = f[sel].mean(); bench = f.mean()
            to = 1.0 if not prev else len(set(sel) - prev) / len(sel); prev = set(sel)
            rows.append((port - bench - to * COST, to))
        r = pd.DataFrame(rows, columns=['ex', 'to'])
        ir = r.ex.mean() / r.ex.std(ddof=1) * np.sqrt(12) if len(r) > 2 else np.nan
        return len(r), r.ex.mean(), ir, nw_t(r.ex.values, 1), r.to.mean()
    print('动量 12-1:', run(lambda i: close.iloc[i - 21] / close.iloc[i - 252] - 1, 'top'))
    print('反转 1 月:', run(lambda i: close.iloc[i] / close.iloc[i - 21] - 1, 'bottom'))
    cs = candlestick(close, high, low, op)
    print('\n蜡烛图:')
    for k, v in sorted(cs.items()):
        print(k, v)


if __name__ == '__main__':
    main()
