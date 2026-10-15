#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""价值/质量风格层（PIT 近似）验证（D41）。

- 基本面：akshare `stock_financial_hk_analysis_indicator_em`（多期，带 REPORT_DATE），现池 56 只。
- **PIT 近似**：某期财报在 REPORT_DATE + LAG_DAYS（默认 90）后才"可用"；t 时点用最近已生效的一期。
  ⚠️ 无公告日 ⇒ 近似（真 PIT 须 HKEX 公告日 / 付费源）。
- 风格分（越高越好）：价值 E/P=EPS/价、B/P=BPS/价；质量 ROE、毛利率。
- 每 H=20 交易日调仓，买 top-K=10 等权；基准=全池等权；非重叠 + 块 bootstrap。

用法：python3 scripts/style_fundamental_backtest.py
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import STOCK_SECTOR_MAPPING  # noqa: E402

SNAP = os.path.join(ROOT, 'data', 'hk_market_snapshot')
CACHE = os.path.join(ROOT, 'data', 'fundamental_multi_cache.pkl')
START, H, K, LAG = '2020-01-01', 20, 10, 90
STYLES = ['ep', 'bp', 'roe', 'gpm']


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def fetch_fund(codes):
    import akshare as ak
    out = {}
    for code in codes:
        sym = code.replace('.HK', '').zfill(5)
        try:
            df = ak.stock_financial_hk_analysis_indicator_em(symbol=sym, indicator='年度')
        except Exception as e:
            print(f"  ⚠️ {sym} 取数失败: {str(e)[:50]}")
            continue
        df = df.rename(columns={'REPORT_DATE': 'report_date', 'BASIC_EPS': 'eps', 'BPS': 'bps',
                                'ROE_AVG': 'roe', 'GROSS_PROFIT_RATIO': 'gpm', 'NET_PROFIT_RATIO': 'npm'})
        cols = [c for c in ['report_date', 'eps', 'bps', 'roe', 'gpm', 'npm'] if c in df.columns]
        d = df[cols].copy()
        d['report_date'] = pd.to_datetime(d['report_date'], errors='coerce')
        for c in ['eps', 'bps', 'roe', 'gpm', 'npm']:
            if c in d.columns:
                d[c] = _num(d[c])
        d = d.dropna(subset=['report_date']).sort_values('report_date')
        d['avail_date'] = d['report_date'] + pd.Timedelta(days=LAG)
        out[code] = d.set_index('avail_date')[[c for c in ['eps', 'bps', 'roe', 'gpm', 'npm'] if c in d.columns]].sort_index()
        time.sleep(0.15)
    return out


def load_prices(start=START, end=None):
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
    return panel


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
    ap = argparse.ArgumentParser(description='价值/质量风格层（PIT 近似，D41/D42）')
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=None)
    ap.add_argument('--lag', type=int, default=LAG, help='报告滞后天数（缓存基于 90d ⇒ 按 lag-90 平移）')
    ap.add_argument('--styles', default=','.join(STYLES))
    a = ap.parse_args()

    codes = list(STOCK_SECTOR_MAPPING.keys())
    if os.path.exists(CACHE):
        fund = pickle.load(open(CACHE, 'rb'))
        print(f"📦 用缓存基本面（{len(fund)} 只）")
    else:
        print("📊 抓多期港股财务（akshare）...")
        fund = fetch_fund(codes)
        pickle.dump(fund, open(CACHE, 'wb'))
    fund = {c: f for c, f in fund.items() if len(f) > 0}
    # 缓存 avail_date = report_date+90；按 lag-90 平移得到目标滞后
    shift = pd.Timedelta(days=a.lag - 90)
    if shift != pd.Timedelta(0):
        fund = {c: f.set_axis(f.index + shift) for c, f in fund.items()}
    print(f"基本面 {len(fund)} 只 | LAG={a.lag}d")

    prices = load_prices(a.start, a.end)
    fund_pit = {c: f.reindex(prices.index, method='ffill') for c, f in fund.items()}

    print(f"池 {prices.shape[1]} 只 | 窗口 {prices.index.min().date()} ~ {prices.index.max().date()} | top-K={K} H={H} LAG={a.lag}d")

    def run(style):
        idx = prices.index
        rows = []
        for i in range(0, len(idx) - H - 1, H):
            t = idx[i]
            px = prices.iloc[i]
            sc = {}
            for code, f in fund_pit.items():
                if code not in px.index:
                    continue
                p = px[code]
                if not np.isfinite(p) or p <= 0 or t not in f.index:
                    continue
                row = f.loc[t]
                if style == 'ep' and pd.notna(row['eps']):
                    sc[code] = row['eps'] / p
                elif style == 'bp' and pd.notna(row['bps']):
                    sc[code] = row['bps'] / p
                elif style == 'roe' and pd.notna(row['roe']):
                    sc[code] = row['roe']
                elif style == 'gpm' and pd.notna(row['gpm']):
                    sc[code] = row['gpm']
            if len(sc) < K + 1:
                continue
            top = pd.Series(sc).sort_values(ascending=False).head(K).index
            fwd = (prices.iloc[i + H] / prices.iloc[i] - 1).dropna()
            if len(fwd) < K + 1:
                continue
            rows.append((t, fwd[top].mean(), fwd.mean()))
        return pd.DataFrame(rows, columns=['date', 'strat', 'bench'])

    print("=" * 88)
    print(f"{'风格':<8}{'期数':>5}{'超额均/期':>10}{'超额IR':>8}{'超额95%CI':>18}{'胜率':>7}{'基准IR':>7}{'判定':>5}")
    vpass = qpass = 0
    for style in a.styles.split(','):
        r = run(style)
        ex = (r['strat'] - r['bench']).values
        lo, hi = _ci(ex)
        ok = lo > 0
        if style in ('ep', 'bp'):
            vpass += int(ok)
        else:
            qpass += int(ok)
        print(f"{style:<8}{len(r):>5}{ex.mean()*100:>9.2f}%{_ir(ex):>8.2f}"
              f"[{lo*100:>+6.2f}%,{hi*100:>+6.2f}%]{float((ex>0).mean())*100:>6.0f}%{_ir(r['bench'].values):>7.2f}"
              f"{'✅' if ok else '❌':>5}")
    print("-" * 88)
    print(f"价值通过 {vpass}/2，质量通过 {qpass}/2（判据：价值≥1 且 质量≥1）")
    print(f"判定：{'✅ 价值/质量有 alpha' if (vpass >= 1 and qpass >= 1) else '❌ 价值/质量无 alpha（PIT 近似）'}")


if __name__ == '__main__':
    main()
