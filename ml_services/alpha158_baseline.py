#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha158 风格特征基线（HK 池，仅 OHLCV+VWAP，Qlib 未装则自实现核心子集）

目的：用**公开、可复现**的因子集在港股池上建立 IC/ICIR 基线，
对照我们 CatBoost 模型的 Rank IC（20d 全期 ≈ 0.02）。
业界参照：Qlib Alpha158 上 LightGBM Rank IC 0.048 / CatBoost 0.042。

实现：~60 个经典技术/量价因子（KDJ/RSI/ROC/BOLL/MA/位置/量比/CORR/VR/CCI/WR/OBV/VWAP…）
      → 按日横截面 z-score → 逐日 Rank IC → 单因子与复合因子 IC/ICIR。

用法：
  python3 ml_services/alpha158_baseline.py --horizon 20
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = 'data/feature_cache'
TEST_START = '2021-10-01'
TEST_END = '2026-08-01'


def _llv(x, n): return x.rolling(n, min_periods=n).min()
def _hhv(x, n): return x.rolling(n, min_periods=n).max()
def _ma(x, n): return x.rolling(n, min_periods=n).mean()
def _std(x, n): return x.rolling(n, min_periods=n).std()


def alpha_factors(d):
    """输入含 Open/High/Low/Close/Volume/VWAP 的 DataFrame，输出 ~60 个因子"""
    C, H, L, V = d['Close'], d['High'], d['Low'], d['Volume']
    f = pd.DataFrame(index=d.index)

    # KDJ
    rsv = (C - _llv(L, 9)) / (_hhv(H, 9) - _llv(L, 9) + 1e-9) * 100
    K = rsv.ewm(alpha=1 / 3, min_periods=3).mean()
    D = K.ewm(alpha=1 / 3, min_periods=3).mean()
    J = 3 * K - 2 * D
    f['KDJ_K'], f['KDJ_D'], f['KDJ_J'] = K, D, J

    # RSI
    up = (C.diff()).clip(lower=0); dn = (-C.diff()).clip(lower=0)
    for n in (6, 12, 24):
        au = up.rolling(n, min_periods=n).mean(); ad = dn.rolling(n, min_periods=n).mean()
        f[f'RSI_{n}'] = 100 - 100 / (1 + au / (ad + 1e-9))

    # ROC / MOM / CMO
    for n in (5, 10, 20):
        f[f'ROC_{n}'] = C / C.shift(n) - 1
        f[f'MOM_{n}'] = C - C.shift(n)
        ud = (C.diff() > 0).astype(float); dd = (C.diff() < 0).astype(float)
        su = (C.diff() * ud).rolling(n, min_periods=n).sum()
        sd = (-C.diff() * dd).rolling(n, min_periods=n).sum()
        f[f'CMO_{n}'] = (su - sd) / (su + sd + 1e-9) * 100

    # Price position / BOLL / MA
    for n in (10, 20, 60):
        f[f'PricePos_{n}'] = (C - _llv(L, n)) / (_hhv(H, n) - _llv(L, n) + 1e-9)
        f[f'MA_Ratio_{n}'] = C / _ma(C, n) - 1
    for n in (5, 20):
        f[f'MA_Slope_{n}'] = (_ma(C, n) - _ma(C, n).shift(5)) / _ma(C, n)
    for n in (5, 10, 20, 60):
        f[f'Dist_MA_{n}'] = (C - _ma(C, n)) / _ma(C, n)
    mid = _ma(C, 20); sd = _std(C, 20)
    f['BOLL_pos'] = (C - mid) / (sd + 1e-9)
    f['BOLL_upper'] = (C - (mid + 2 * sd)) / (mid + 2 * sd + 1e-9)
    f['BOLL_lower'] = (C - (mid - 2 * sd)) / (mid - 2 * sd + 1e-9)

    # 日内位置（KBAR 风格）
    rng = H - L + 1e-9
    temp = (C - L) / rng
    f['TEMP'] = temp
    for n in (10, 20):
        f[f'TEMP_MA_{n}'] = _ma(temp, n)

    # 量能
    for n in (5, 20):
        f[f'Vol_Ratio_{n}'] = V / (_ma(V, n) + 1e-9)
    f['Vol_Z_20'] = (V - _ma(V, 20)) / (_std(V, 20) + 1e-9)
    f['Vol_Slope'] = _ma(V, 5) / (_ma(V, 20) + 1e-9) - 1
    for n in (10, 20):
        f[f'CORR_CV_{n}'] = C.rolling(n, min_periods=n).corr(V)
    # VR：上涨日量 / 下跌日量
    for n in (10, 20):
        upv = (V * ud).rolling(n, min_periods=n).sum()
        dnv = (V * dd).rolling(n, min_periods=n).sum()
        f[f'VR_{n}'] = upv / (dnv + 1e-9)

    # VWAP / 金额代理
    tp = (H + L + C) / 3
    vwap = (V * tp).rolling(5, min_periods=5).sum() / (V.rolling(5, min_periods=5).sum() + 1e-9)
    f['VWAP_Ratio'] = C / vwap - 1
    amt = V * C
    f['AMT_Ratio_20'] = amt / (_ma(amt, 20) + 1e-9) - 1

    # OBV 趋势
    obv = (np.sign(C.diff()).fillna(0) * V).cumsum()
    f['OBV_Slope_20'] = (obv - obv.shift(20)) / (_ma(V, 20) * 20 + 1e-9)

    # WR / CCI
    for n in (10, 14):
        f[f'WR_{n}'] = (_hhv(H, n) - C) / (_hhv(H, n) - _llv(L, n) + 1e-9) * 100
    tp_m = _ma(tp, 14)
    md = tp.rolling(14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    f['CCI_14'] = (tp - tp_m) / (0.015 * md + 1e-9)

    # ATR
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    f['ATR_Ratio_14'] = _ma(tr, 14) / C

    return f


def load_ohlcv():
    files = {}
    for p in glob.glob(os.path.join(CACHE_DIR, '*_shift.pkl')):
        code = os.path.basename(p).split('_')[0]
        if code not in files or p > files[code]:
            files[code] = p
    out = {}
    for code, p in files.items():
        try:
            d = pd.read_pickle(p)['data']['stock_df']
        except Exception:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None)
        out[code] = d
    return out


def run(horizon, out_md):
    ohlcv = load_ohlcv()
    frames = []
    for code, d in ohlcv.items():
        f = alpha_factors(d)
        f['fwd_ret'] = d['Close'].shift(-horizon) / d['Close'] - 1
        f['code'] = code
        f = f.dropna(subset=['fwd_ret'])
        frames.append(f)
    panel = pd.concat(frames)
    panel = panel[(panel.index >= TEST_START) & (panel.index < TEST_END)]
    panel = panel.reset_index().rename(columns={panel.index.name or 'index': 'date'})
    if 'Date' in panel.columns:
        panel = panel.rename(columns={'Date': 'date'})
    panel['date'] = pd.to_datetime(panel['date'])
    feat_cols = [c for c in alpha_factors(ohlcv[next(iter(ohlcv))]).columns]
    feat_cols = [c for c in feat_cols if c in panel.columns]
    print(f"panel {panel.shape}  因子 {len(feat_cols)}  股票 {panel['code'].nunique()}  日期 {panel['date'].nunique()}")

    # 按日横截面 z-score
    gm = panel.groupby('date')[feat_cols].transform('mean')
    gs = panel.groupby('date')[feat_cols].transform('std')
    panel[feat_cols] = ((panel[feat_cols] - gm) / (gs + 1e-9)).replace([np.inf, -np.inf], np.nan)

    # 逐因子 Rank IC
    rows = []
    for col in feat_cols:
        ics = []
        for d, g in panel.groupby('date'):
            if g[col].notna().sum() < 10:
                continue
            ic = spearmanr(g[col], g['fwd_ret']).correlation
            if np.isfinite(ic):
                ics.append(ic)
        if len(ics) < 30:
            continue
        ics = np.array(ics)
        rows.append((col, ics.mean(), ics.mean() / ics.std(ddof=1), (ics > 0).mean(), len(ics)))
    rows.sort(key=lambda r: -abs(r[1]))
    top = rows[:10]

    # 复合：全部因子等权 / top10(按ICIR, 样本内选择，注明)
    def composite(cols):
        s = panel[cols].mean(axis=1) if len(cols) > 1 else panel[cols[0]]
        ics = []
        for d, g in panel.assign(_c=s).groupby('date'):
            if g['_c'].notna().sum() < 10:
                continue
            ic = spearmanr(g['_c'], g['fwd_ret']).correlation
            if np.isfinite(ic):
                ics.append(ic)
        ics = np.array(ics)
        return ics.mean(), ics.mean() / ics.std(ddof=1), len(ics)

    allf = [r[0] for r in rows]
    top10_by_icir = sorted(rows, key=lambda r: -abs(r[2]))[:10]
    c_all = composite(allf)
    c_top = composite([r[0] for r in top10_by_icir])

    L = []
    L.append(f"# Alpha158 风格特征基线（{horizon}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 窗口: {TEST_START} ~ {TEST_END}　股票: {panel['code'].nunique()}　因子: {len(feat_cols)}")
    L.append(f"- 说明: 仅 OHLCV+VWAP（Qlib 未装，自实现 Alpha158 核心子集）\n")
    L.append("## 单因子 Rank IC TOP10\n")
    L.append("| 因子 | 平均IC | ICIR | 正比例 | 期数 |")
    L.append("|------|--------|------|--------|------|")
    for c, ic, icir, pos, n in top:
        L.append(f"| {c} | {ic:.4f} | {icir:.3f} | {pos*100:.0f}% | {n} |")
    L.append("")
    L.append("## 复合因子\n")
    L.append("| 复合 | 平均IC | ICIR | 期数 |")
    L.append("|------|--------|------|------|")
    L.append(f"| 全部因子等权 | {c_all[0]:.4f} | {c_all[1]:.3f} | {c_all[2]} |")
    L.append(f"| Top10(按ICIR, 样本内) | {c_top[0]:.4f} | {c_top[1]:.3f} | {c_top[2]} |")
    L.append("")
    L.append("## 对照\n")
    L.append("- 本模型 CatBoost 20d Rank IC ≈ 0.02（全期）；Qlib Alpha158 基准：LightGBM RankIC 0.048 / CatBoost 0.042")
    L.append(f"- 单因子最强：{top[0][0]}（IC {top[0][1]:.4f}）")
    L.append("")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def main():
    ap = argparse.ArgumentParser(description='Alpha158 风格特征基线')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()
    out = args.output or f"output/alpha158_baseline_{args.horizon}d.md"
    run(args.horizon, out)


if __name__ == '__main__':
    main()