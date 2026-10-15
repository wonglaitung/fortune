#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""经典简单择时规则验证（D43）：恒指 Long/Flat 能否"少亏/真会躲"。

- R1 Faber 10 月均线（月频）；R2 200 日线；R3 12 月动量。全 PIT，决策 ≤T、T+1 生效。
- 每日 mark-to-market；换手计成本（单边 0.1%）。
- 判据（D43）：① maxDD(规则)<maxDD(持有)；② maxDD(规则)<maxDD(**静态等仓位 w̄**)（否则只是少投）；③ Sharpe≥持有−0.05。

用法：python3 scripts/classic_timing_backtest.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.001
RULES = ['faber10m', 'ma150', 'ma200', 'ma250', 'mom12']


def load_hsi():
    try:
        import yfinance as yf
        s = yf.Ticker('^HSI').history(period='max', interval='1d')['Close'].copy()
    except Exception:
        s = pd.Series(dtype=float)
    if s is None or s.empty:
        d = pickle.load(open(os.path.join(ROOT, 'data', 'hk_universe_cache', 'HSI.pkl'), 'rb'))
        s = d['Close' if 'Close' in d.columns else 'close'].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, 'tz', None) is not None:
        s.index = s.index.tz_convert(None)
    return s[~s.index.duplicated(keep='last')].sort_index()


def position(close, rule):
    if rule.startswith('ma') and rule[2:].isdigit():
        return (close > close.rolling(int(rule[2:])).mean()).astype(float)
    if rule == 'mom12':
        return (close.pct_change(252) > 0).astype(float)
    # faber10m：月收盘 > 10 月均线；信号用于"下一个月"
    key = close.index.to_period('M')
    m = close.groupby(key).last()
    sig = (m > m.rolling(10).mean()).astype(float)
    sig_next = sig.copy()
    sig_next.index = sig.index + 1
    return pd.Series(sig_next.reindex(key).to_numpy(), index=close.index).fillna(0.0)


def stats(r):
    r = np.asarray(r, float)
    cum = np.cumprod(1 + r)
    cagr = cum[-1] ** (252.0 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    sharpe = r.mean() * 252 / vol if vol > 0 else float('nan')
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return cagr, vol, sharpe, dd, (cagr / abs(dd) if dd else float('nan'))


def main():
    close = load_hsi()
    dr = close.pct_change().fillna(0.0)
    print(f"恒指 {close.index.min().date()} ~ {close.index.max().date()}（{len(close)} 日）")
    bc, bv, bs, bdd, bc2 = stats(dr.values)
    print(f"{'口径':<12}{'CAGR':>8}{'波动':>7}{'Sharpe':>8}{'maxDD':>8}{'Calmar':>8}{'在场':>6}")
    print(f"{'买入持有':<12}{bc*100:>7.1f}%{bv*100:>6.1f}%{bs:>8.2f}{bdd*100:>7.1f}%{bc2:>8.2f}{100:>5.0f}%")
    print("=" * 72)
    n_pass = 0
    for rule in RULES:
        pos = position(close, rule)
        pos_lag = pos.shift(1).fillna(0.0)
        turn = (pos - pos_lag).abs()
        r = (pos_lag * dr - turn * COST).values
        c, v, s, dd, c2 = stats(r)
        wbar = float(pos.mean())
        r_static = (wbar * dr).values
        _, _, _, dd_static, _ = stats(r_static)
        ok = (dd > bdd) and (dd > dd_static) and (s >= bs - 0.05)
        n_pass += int(ok)
        print(f"{rule:<12}{c*100:>7.1f}%{v*100:>6.1f}%{s:>8.2f}{dd*100:>7.1f}%{c2:>8.2f}{wbar*100:>5.0f}%"
              f"  [持有 maxDD {bdd*100:.0f}% | 静态w̄ maxDD {dd_static*100:.0f}%] {'✅' if ok else '❌'}")
    print("=" * 72)
    print(f"通过 {n_pass}/{len(RULES)}（判据：①少亏 且 ②真会躲[浅于静态w̄] 且 ③不劣于持有）")
    print(f"判定：{'✅ 简单择时有用' if n_pass >= 1 else '❌ 简单择时也不管用（至多少投）'}")


if __name__ == '__main__':
    main()
