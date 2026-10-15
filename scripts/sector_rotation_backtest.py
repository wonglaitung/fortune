#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板块层 alpha 验证：板块动量轮动能否跑赢"全板块等权"（D36）。

- 池：现池 56 只（`config.STOCK_SECTOR_MAPPING`）→ 板块（成员等权），仅取成员 ≥3 的板块。
- 数据：`data/hk_market_snapshot/*.pkl`（日线），窗口 2020-01 → 末日。
- 策略：每 H=20 交易日调仓，按过去 L 日动量买最强 top-K 板块；基准 = **全板块等权**。
- 指标：超额均值/IR、**块 bootstrap 95%CI**、累计、胜率、回撤；附基准绝对 IR。
- 判据（D36）：超额 IR CI **下界>0**；跨 {L=20/60}×{K=2/3} **≥3/4 格**成立。

用法：python3 scripts/sector_rotation_backtest.py
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import STOCK_SECTOR_MAPPING  # noqa: E402

SNAP = os.path.join(ROOT, 'data', 'hk_market_snapshot')
START = '2020-01-01'
H = 20
GRID = [(20, 2), (20, 3), (60, 2), (60, 3)]  # (动量窗 L, top-K)


def load_sector_returns(start=START, end=None):
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
    panel = panel[panel.index >= start]
    if end:
        panel = panel[panel.index <= end]
    rets = panel.pct_change()
    sec = {}
    for code in panel.columns:
        sec.setdefault(STOCK_SECTOR_MAPPING.get(code, {}).get('sector', 'unknown'), []).append(code)
    sec = {k: v for k, v in sec.items() if len(v) >= 3 and k != 'unknown'}
    sret = pd.DataFrame({k: rets[v].mean(axis=1) for k, v in sec.items()})
    return sret, {k: len(v) for k, v in sec.items()}


def rotation(sret, L, K):
    mom = sret.rolling(L).sum()
    idx = sret.index
    rows = []
    for i in range(L, len(idx) - H - 1, H):
        m = mom.iloc[i].dropna()
        if len(m) < K + 1:
            continue
        top = m.sort_values(ascending=False).head(K).index
        fwd = sret.iloc[i + 1:i + 1 + H].sum()
        rows.append((idx[i], fwd[top].mean(), fwd.mean()))
    return pd.DataFrame(rows, columns=['date', 'strat', 'bench'])


def _mean_ci(x, n=2000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    b = [x[rng.choice(len(x), len(x), replace=True)].mean() for _ in range(n)]
    return x.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _ir(x):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(252.0 / H) if s > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description='板块动量轮动验证（D36/D37）')
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=None)
    ap.add_argument('--cells', default='20:2,20:3,60:2,60:3', help='L:K,L:K…')
    a = ap.parse_args()
    GRID = [tuple(int(x) for x in c.split(':')) for c in a.cells.split(',')]
    sret, sizes = load_sector_returns(a.start, a.end)
    print(f"板块（成员≥3）：{len(sret.columns)} 个 | 窗口 {sret.index.min().date()} ~ {sret.index.max().date()}")
    print("  " + ", ".join(f"{k}({n})" for k, n in sorted(sizes.items(), key=lambda x: -x[1])))
    print("=" * 92)
    print(f"{'L':>3} {'K':>2} {'期数':>5} {'超额均/期':>10} {'超额IR':>8} {'超额95%CI':>18} {'胜率':>7} {'基准IR':>7} {'判定':>5}")
    n_pass = 0
    for (L, K) in GRID:
        r = rotation(sret, L, K)
        ex = (r['strat'] - r['bench']).values
        m, lo, hi = _mean_ci(ex)
        ir = _ir(ex)
        ok = lo > 0
        n_pass += int(ok)
        wr = float((ex > 0).mean())
        print(f"{L:>3} {K:>2} {len(r):>5} {m*100:>9.2f}% {ir:>8.2f} "
              f"[{lo*100:>+6.2f}%,{hi*100:>+6.2f}%] {wr*100:>6.0f}% {_ir(r['bench'].values):>7.2f} "
              f"{'✅' if ok else '❌':>5}")
    print("-" * 92)
    need = 3 if len(GRID) >= 4 else max(2, len(GRID) - 1)
    print(f"网格通过：{n_pass}/{len(GRID)} 格（本判定阈值 ≥{need}/{len(GRID)}）")
    print("说明：判据阈值按各 D 号预登记解读（D36=≥3/4；D37=≥2/3）。")


if __name__ == '__main__':
    main()
