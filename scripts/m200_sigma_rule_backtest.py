#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户规则回测：收盘"升穿" MA200 买入；自买入价"升或跌超过 1 个标准差"即卖（先触及者）。

"1 个标准差"两版（歧义，都测）：
  A = 买入时**滚动 20 日收益率**标准差（≈日波动 σ_d）；出场带 = **±1σ_d**；
  B = σ_d × √20（≈20 日累计波动）；出场带 = **±1σ_20d**。
最长持有 MAXH 日强制平仓；成本单边 0.1%。

回答："这个规则的胜率如何"。⚠️ 对称 ±1σ 出场 ⇒ 胜率≈50% 多为**构造使然**（先触及哪边）。
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_PER_SIDE = 0.001  # 单边 0.1%（往返 0.2%）
MAXH = 250


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


def simulate(close, sigma_scale, label):
    ret = close.pct_change()
    ma = close.rolling(200).mean()
    vol = ret.rolling(20).std()
    c = close.values
    m = ma.values
    v = vol.values
    entries = [i for i in range(201, len(c) - 1)
               if not np.isnan(m[i - 1]) and c[i] > m[i] and c[i - 1] <= m[i - 1]]
    trades = []
    for i in entries:
        if np.isnan(v[i]) or v[i] <= 0:
            continue
        entry = c[i]
        band = v[i] * sigma_scale
        exit_i, direction = None, None
        for j in range(i + 1, min(i + 1 + MAXH, len(c))):
            up = c[j] / entry - 1
            if up >= band:
                exit_i, direction = j, 'up'
                break
            if up <= -band:
                exit_i, direction = j, 'down'
                break
        if exit_i is None:
            exit_i, direction = min(i + MAXH, len(c) - 1), 'timeout'
        gross = c[exit_i] / entry - 1
        trades.append((pd.Timestamp(close.index[i]).date(), direction, gross, gross - 2 * COST_PER_SIDE,
                       exit_i - i))
    df = pd.DataFrame(trades, columns=['entry_date', 'exit_dir', 'gross', 'net', 'hold_days'])
    if df.empty:
        print(f"[{label}] 无交易")
        return
    n = len(df)
    win = (df['net'] > 0).mean()
    up_exit = (df['exit_dir'] == 'up').mean()
    aw = df.loc[df['net'] > 0, 'net'].mean()
    al = df.loc[df['net'] <= 0, 'net'].mean()
    print(f"[{label}] 交易 {n} 次 | **胜率(净>0) {win*100:.1f}%** | 上侧出场 {up_exit*100:.1f}% | "
          f"均持 {df['hold_days'].mean():.0f} 日")
    print(f"        均盈 {aw*100:+.2f}% | 均亏 {al*100:+.2f}% | **每笔净 {df['net'].mean()*100:+.2f}%** | "
          f"毛/笔 {df['gross'].mean()*100:+.2f}% | 超时出场 {(df['exit_dir']=='timeout').mean()*100:.0f}%")


def main():
    close = load_hsi()
    print(f"恒指 {close.index.min().date()} ~ {close.index.max().date()}（{len(close)} 日）")
    print("规则：收盘升穿 MA200 买入；自买入价 升/跌 超过 1σ 即卖（先触及者）；成本单边 0.1%")
    print("=" * 90)
    simulate(close, 1.0, 'A  σ=日波动(20d)')
    simulate(close, np.sqrt(20), 'B  σ=日波动×√20 (≈20日)')


if __name__ == '__main__':
    main()
