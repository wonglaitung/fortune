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
sys.path.insert(0, ROOT)
from ml_services.strategy_eval import (  # noqa: E402
    daily_mark, static_matched, metrics_daily, has_timing_increment)

COST_ROUND_TRIP = 0.002  # 往返 0.2%（指数 ETF，无印花税；单边 0.1%，D97）
RULES = ['faber10m', 'ma150', 'ma200', 'ma250', 'mom12']


def load_hsi():
    try:
        import yfinance as yf
        s = yf.Ticker('^HSI').history(period='max', interval='1d', timeout=15)['Close'].copy()
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


def main():
    close = load_hsi()
    dr = close.pct_change().fillna(0.0)
    print(f"恒指 {close.index.min().date()} ~ {close.index.max().date()}（{len(close)} 日）")
    # 评估口径统一走 ml_services.strategy_eval（逐日 mark + 静态对照，D45）
    bh = metrics_daily(dr)
    print(f"{'口径':<12}{'CAGR':>8}{'波动':>7}{'Sharpe':>8}{'maxDD':>8}{'Calmar':>8}{'在场':>6}")
    print(f"{'买入持有':<12}{bh['cagr']*100:>7.1f}%{bh['vol']*100:>6.1f}%{bh['sharpe']:>8.2f}"
          f"{bh['maxdd']*100:>7.1f}%{bh['calmar']:>8.2f}{100:>5.0f}%")
    print("=" * 72)
    n_pass = 0
    for rule in RULES:
        pos = position(close, rule).reindex(dr.index).fillna(0.0)
        wbar = float(pos.mean())
        net = daily_mark(dr, pos, round_trip_cost=COST_ROUND_TRIP)     # 逐日 mark（含 T+1 + 换手成本）
        st = static_matched(dr, wbar)                          # 静态等仓位对照
        m, ms = metrics_daily(net), metrics_daily(st)
        ok = (m['maxdd'] > bh['maxdd']) and (m['maxdd'] > ms['maxdd']) and (m['sharpe'] >= bh['sharpe'] - 0.05)
        n_pass += int(ok)
        print(f"{rule:<12}{m['cagr']*100:>7.1f}%{m['vol']*100:>6.1f}%{m['sharpe']:>8.2f}"
              f"{m['maxdd']*100:>7.1f}%{m['calmar']:>8.2f}{wbar*100:>5.0f}%"
              f"  [持有 maxDD {bh['maxdd']*100:.0f}% | 静态w̄ maxDD {ms['maxdd']*100:.0f}%]"
              f" {'✅' if ok else '❌'}")
    print("=" * 72)
    print(f"通过 {n_pass}/{len(RULES)}（判据：①少亏 且 ②真会躲[浅于静态w̄] 且 ③不劣于持有）")
    print(f"判定：{'✅ 简单择时有用' if n_pass >= 1 else '❌ 简单择时也不管用（至多少投）'}")


if __name__ == '__main__':
    main()
