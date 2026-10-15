#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格因子层验证（D39）：价格型风格（动量/低波动/短期反转）能否跑赢"全池等权"。

- 池：现池 56 只（`config.STOCK_SECTOR_MAPPING`）；数据：`data/hk_market_snapshot`（PIT）。
- 每 H=20 交易日调仓：买风格分 top-K=10 只（等权）；基准 = 全池等权。
- 价值/规模/质量无 PIT ⇒ 本条不测（D39 §一）。
- 判据：超额 IR 95%CI 下界>0；动量三窗{20,60,120}≥2 成立。

用法：python3 scripts/style_factor_backtest.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import STOCK_SECTOR_MAPPING  # noqa: E402

SNAP = os.path.join(ROOT, 'data', 'hk_market_snapshot')
START, H, K = '2020-01-01', 20, 10
STYLES = ['mom_20', 'mom_60', 'mom_120', 'lowvol_60', 'rev_5']


def load_panel():
    closes = {}
    for code in STOCK_SECTOR_MAPPING:
        fp = os.path.join(SNAP, code.replace('.HK', '').zfill(5) + '.pkl')
        if not os.path.exists(fp):
            continue
        df = pickle.load(open(fp, 'rb'))
        s = df['Close'].copy()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, 'tz', None) is not None:
            s.index = s.index.tz_localize(None)
        closes[code] = s
    panel = pd.DataFrame(closes).sort_index()
    return panel[panel.index >= START].pct_change()


def score(rets, i, name):
    L = int(name.split('_')[1])
    if i - L + 1 < 0:
        return None
    win = rets.iloc[i - L + 1:i + 1]
    if name.startswith('mom'):
        return win.sum()          # 动量：越高越好
    if name.startswith('lowvol'):
        return -win.std()         # 低波：波动越小分越高
    return -win.sum()             # 反转：过去越跌分越高


def run(rets, name):
    rows = []
    for i in range(120, len(rets) - H - 1, H):
        sc = score(rets, i, name)
        if sc is None:
            continue
        sc = sc.dropna()
        if len(sc) < K + 1:
            continue
        top = sc.sort_values(ascending=False).head(K).index
        fwd = rets.iloc[i + 1:i + 1 + H].sum()
        rows.append((rets.index[i], fwd[top].mean(), fwd.mean()))
    return pd.DataFrame(rows, columns=['date', 'strat', 'bench'])


def _ci(x, n=2000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    b = [x[rng.choice(len(x), len(x), replace=True)].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _ir(x):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(252.0 / H) if s > 0 else float('nan')


def main():
    rets = load_panel()
    print(f"池 {rets.shape[1]} 只 | 窗口 {rets.index.min().date()} ~ {rets.index.max().date()} | top-K={K} H={H}")
    print("=" * 88)
    print(f"{'风格':<12}{'期数':>5}{'超额均/期':>10}{'超额IR':>8}{'超额95%CI':>18}{'胜率':>7}{'基准IR':>7}{'判定':>5}")
    mom_pass = 0
    for name in STYLES:
        r = run(rets, name)
        ex = (r['strat'] - r['bench']).values
        lo, hi = _ci(ex)
        ok = lo > 0
        if name.startswith('mom'):
            mom_pass += int(ok)
        print(f"{name:<12}{len(r):>5}{ex.mean()*100:>9.2f}%{_ir(ex):>8.2f}"
              f"[{lo*100:>+6.2f}%,{hi*100:>+6.2f}%]{float((ex>0).mean())*100:>6.0f}%{_ir(r['bench'].values):>7.2f}"
              f"{'✅' if ok else '❌':>5}")
    print("-" * 88)
    print(f"动量三窗通过 {mom_pass}/3（判据 ≥2）；判定：{'✅ 有风格 alpha' if mom_pass >= 2 else '❌ 风格层无 alpha（价格型）'}")


if __name__ == '__main__':
    main()
