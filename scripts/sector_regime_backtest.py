#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大市 × 板块 交叉验证（D38）：按大盘状态切换"周期/防御"板块，能否跑赢"全板块等权"。

- 板块收益：复用 D36（数据/hk_market_snapshot，成员≥3）。
- 大盘状态：v1 = HSI 20 日收益 >0 记"牛"；v2 = HSI 收盘 > MA200 记"牛"（HSI.pkl，PIT）。
- 规则：牛 → 持周期组等权；熊 → 持防御组等权（或现金）。基准 = 全板块等权。
- 关键对照：还须优于"一直持周期组"（否则只是周期 beta）。
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sector_rotation_backtest import load_sector_returns, _mean_ci, _ir, H  # noqa: E402

CYCLICAL = {'tech', 'semiconductor', 'ai', 'new_energy', 'energy', 'shipping', 'consumer', 'auto'}
DEFENSIVE = {'bank', 'utility', 'insurance', 'real_estate', 'gas', 'materials'}


def load_hsi():
    d = pickle.load(open(os.path.join(ROOT, 'data', 'hk_universe_cache', 'HSI.pkl'), 'rb'))
    col = 'close' if 'close' in d.columns else 'Close'
    s = d[col].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, 'tz', None) is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep='last')].sort_index()


def make_state(hsi, kind):
    if kind == 'ret20':
        return (hsi / hsi.shift(20) - 1) > 0
    return hsi > hsi.rolling(200).mean()


def rotation(sret, state, cyclical, defensive, bear_cash):
    idx = sret.index
    rows = []
    for i in range(H, len(idx) - H - 1, H):
        d = idx[i]
        if d not in state.index or pd.isna(state.loc[d]):
            continue
        bull = bool(state.loc[d])
        fwd = sret.iloc[i + 1:i + 1 + H].sum()
        strat = fwd[cyclical].mean() if bull else (0.0 if bear_cash else fwd[defensive].mean())
        rows.append((d, strat, fwd.mean(), fwd[cyclical].mean()))
    return pd.DataFrame(rows, columns=['date', 'strat', 'bench', 'cycl'])


def main():
    sret, _ = load_sector_returns('2020-01-01')
    hsi = load_hsi().reindex(sret.index, method='ffill')
    cyc = [c for c in sret.columns if c in CYCLICAL]
    dfn = [c for c in sret.columns if c in DEFENSIVE]
    print(f"板块 {len(sret.columns)} | 周期组 {cyc} | 防御组 {dfn}")
    print(f"窗口 {sret.index.min().date()} ~ {sret.index.max().date()}")
    print("=" * 96)
    print(f"{'状态定义':<10}{'熊市动作':<10}{'期数':>5}{'超额均/期':>10}{'超额IR':>8}{'超额95%CI':>18}{'vs周期组IR':>11}{'判定':>5}")
    n_pass = 0
    configs = [('ret20', False), ('ret20', True), ('ma200', False), ('ma200', True)]
    for kind, bear_cash in configs:
        state = make_state(hsi, kind)
        r = rotation(sret, state, cyc, dfn, bear_cash)
        ex = (r['strat'] - r['bench']).values
        m, lo, hi = _mean_ci(ex)
        ok = lo > 0
        n_pass += int(ok)
        print(f"{kind:<10}{('现金' if bear_cash else '防御组'):<8}{len(r):>5} {m*100:>9.2f}% {_ir(ex):>8.2f} "
              f"[{lo*100:>+6.2f}%,{hi*100:>+6.2f}%] {_ir((r['strat']-r['cycl']).values):>11.2f} "
              f"{'✅' if ok else '❌':>5}")
    print("-" * 96)
    print(f"通过 {n_pass}/4（预注册：状态定义 v1/v2 至少一版成立，且须优于'一直持周期组'）")
    print(f"判定：{'✅ 交互有 alpha' if n_pass >= 1 else '❌ 交互无 alpha（切换不跑赢全板块等权）'}")


if __name__ == '__main__':
    main()
