#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D54：换宇宙再测风格因子 —— 56 池 vs 宽池(125)，**同 D39 口径，只换池**。

- 56 池：`config.STOCK_SECTOR_MAPPING` + `data/hk_market_snapshot`（PIT，D39 原口径）。
- 宽池：`data/hk_universe_cache/`（125 只**当前**大中盘，2020-10→）——⚠️ **含幸存者偏差**。
- H=20、TopK=10、等权；超额 = 风格组合 − 全池等权；CI=bootstrap 2000。

⚠️ 宽池=当前名单 ⇒ **偏差偏向"有 alpha"** ⇒ 宽池仍 null = **更强 null**；宽池有戏 ⇒ 疑幸存者伪影。

用法：python3 scripts/style_factor_broad.py
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import STOCK_SECTOR_MAPPING  # noqa: E402

SNAP56 = os.path.join(ROOT, 'data', 'hk_market_snapshot')
SNAP_BROAD = os.path.join(ROOT, 'data', 'hk_universe_cache')
START, H, K = '2020-01-01', 20, 10
STYLES = ['mom_20', 'mom_60', 'mom_120', 'lowvol_60', 'rev_5']
N_STYLES = len(STYLES)
CLIP = 0.5   # 数据卫生：单日 |收益| > 50% 视为脏数据（港股无涨跌停，但 50%+ 多为拆分/错价）


def _read_close(fp):
    df = pickle.load(open(fp, 'rb'))
    s = df['Close'].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, 'tz', None) is not None:
        s.index = s.index.tz_localize(None)
    # 数据卫生：价格须为正且有限（宽池有毛刺 → pct_change 会出 ±inf）
    s = pd.to_numeric(s, errors='coerce')
    s = s.where(np.isfinite(s) & (s > 0))
    return s


def _panel(closes):
    p = pd.DataFrame(closes).sort_index()
    p = p[p.index >= START]
    rets = p.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    # 数据卫生：裁剪极端单日收益（脏数据 / 未复权拆分 / 错价）
    rets = rets.clip(lower=-CLIP, upper=CLIP)
    return rets


def load_pool56():
    closes = {}
    for code in STOCK_SECTOR_MAPPING:
        fp = os.path.join(SNAP56, code.replace('.HK', '').zfill(5) + '.pkl')
        if os.path.exists(fp):
            closes[code] = _read_close(fp)
    return _panel(closes)


def load_broad():
    closes = {}
    for fp in glob.glob(os.path.join(SNAP_BROAD, '*.pkl')):
        code = os.path.basename(fp).replace('.pkl', '') + '.HK'
        try:
            s = _read_close(fp)
        except Exception:  # noqa: BLE001
            continue
        if len(s) > 500:
            closes[code] = s
    return _panel(closes)


def score(rets, i, name):
    L = int(name.split('_')[1])
    if i - L + 1 < 0:
        return None
    win = rets.iloc[i - L + 1:i + 1]
    if name.startswith('mom'):
        return win.sum()
    if name.startswith('lowvol'):
        return -win.std()
    return -win.sum()


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


def _tbl(tag, rets):
    print(f"\n### {tag}  （池 {rets.shape[1]} | {rets.index.min().date()} ~ {rets.index.max().date()} | K={K} H={H}）")
    print(f"{'风格':<12}{'期数':>5}{'超额均/期':>10}{'超额IR':>8}{'超额95%CI':>20}{'胜率':>7}{'基准IR':>7}{'判定':>5}")
    mom_pass, any_pass = 0, 0
    for name in STYLES:
        r = run(rets, name)
        if len(r) < 5:
            print(f"{name:<12}  期数不足")
            continue
        ex = (r['strat'] - r['bench']).values
        lo, hi = _ci(ex)
        # Bonferroni over 5 风格：CI 用 99%/5 近似 → 用 1.96→2.81 的宽松界 (粗判)
        ok = lo > 0
        if name.startswith('mom'):
            mom_pass += int(ok)
        any_pass += int(ok)
        print(f"{name:<12}{len(r):>5}{ex.mean()*100:>9.2f}%{_ir(ex):>8.2f}"
              f"[{lo*100:>+6.2f}%,{hi*100:>+6.2f}%]{float((ex > 0).mean())*100:>6.0f}%"
              f"{_ir(r['bench'].values):>7.2f}{'✅' if ok else '❌':>5}")
    return mom_pass, any_pass


def main():
    print("D54 换宇宙再测风格因子（同 D39 口径，只换池）")
    print("=" * 96)
    p56 = load_pool56()
    broad = load_broad()
    m56, a56 = _tbl("56 池（hk_market_snapshot，PIT）", p56)
    mb, ab = _tbl("宽池 125（hk_universe_cache，⚠️ 当前名单=幸存者偏差）", broad)
    print("\n" + "-" * 96)
    print(f"56 池：动量 {m56}/3 过；宽池：动量 {mb}/3 过")
    print("判据：超额 CI 下界>0（Bonferroni over 5 风格更严 ⇒ 单看 CI 已是宽松界）")
    print("⚠️ 宽池含幸存者偏差（偏向阳性）⇒ 宽池仍 null = 更强 null；宽池阳性 = 疑伪影，不得据此宣称 alpha")


if __name__ == '__main__':
    main()
