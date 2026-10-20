#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格复核"价格异常 + 当日下跌 → 抄底"策略（DECISIONS 之后第 2 步）

定义（显式）：信号日 = 该股当日收益的 30 日 z-score ≤ 阈值（=异常大跌，即"价格异常+当日下跌"）。
交易：收盘买入，持有 5 / 10 / 20 日三档；扣双边成本 0.5%。
对照：无条件买入（所有样本）的同周期胜率/收益。
输出：信号样本数 / 胜率 vs 基准 / 净均收益 / bootstrap CI / 净IR / 逐年。

用法：python3 ml_services/anomaly_dip_backtest.py
     python3 ml_services/anomaly_dip_backtest.py --horizons 5,10,20
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = 'data/feature_cache'
COST = 0.005
WINDOW = 30
HORIZONS = [5, 10, 20]
THRESHOLDS = [-2.5, -3.0, -4.0]


def _latest_cache_files(cache_dir):
    """每股最新特征缓存路径（按 mtime 取最新）。

    兼容两种命名：旧 `{code}_{YYYYMMDD}_shift.pkl`；新
    `{code}_{YYYYMMDD}_{tminus1|tclose}_{f1|base}_{manifest_fp}.pkl`（O4/D27 缓存键重构）。
    仅取 OHLC，故不同 manifest 指纹/口径均可（取最新即可）。
    """
    best = {}
    for pat in ('*_shift.pkl', '*_tclose_*.pkl', '*_tminus1_*.pkl'):
        for p in glob.glob(os.path.join(cache_dir, pat)):
            code = os.path.basename(p).split('_')[0]
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            prev = best.get(code)
            if prev is None or mtime > prev[0]:
                best[code] = (mtime, p)
    return {c: v[1] for c, v in best.items()}


def load():
    files = _latest_cache_files(CACHE_DIR)
    frames = []
    for code, p in files.items():
        try:
            d = pd.read_pickle(p)['data']['stock_df']
        except Exception:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None)
        c = d['Close']
        ret = c.pct_change()
        z = (ret - ret.rolling(WINDOW, min_periods=WINDOW).mean()) / ret.rolling(WINDOW, min_periods=WINDOW).std()
        cols = {'ret': ret, 'z': z, 'code': code}
        for h in HORIZONS:
            cols[f'fwd{h}'] = c.shift(-h) / c - 1
        frames.append(pd.DataFrame(cols).dropna(subset=['z']))
    panel = pd.concat(frames)
    panel = panel[(panel.index >= '2021-10-01') & (panel.index < '2026-08-01')]
    panel = panel.reset_index()
    date_col = 'Date' if 'Date' in panel.columns else panel.columns[0]
    panel['date'] = pd.to_datetime(panel[date_col])
    panel['year'] = panel['date'].dt.year
    return panel


def boot_mean(x, n=2000, seed=42):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    rng = np.random.RandomState(seed); idx = np.arange(len(x))
    b = [x[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def eval_strategy(sig, base, horizon):
    col = f'fwd{horizon}'
    sig = sig.dropna(subset=[col])
    base = base.dropna(subset=[col])
    n = len(sig)
    if n == 0:
        return dict(n=0, win=np.nan, base_win=np.nan, lift=np.nan, mean=np.nan,
                    lo=np.nan, hi=np.nan, ir=np.nan)
    net = sig[col] - COST
    win = (net > 0).mean()
    mean = net.mean()
    std = net.std(ddof=1)
    ir = mean / std * np.sqrt(252.0 / horizon) if std > 0 else np.nan
    lo, hi = boot_mean(net)
    base_win = (base[col] > COST).mean()
    return dict(n=n, win=win, base_win=base_win, lift=win - base_win, mean=mean,
                lo=lo, hi=hi, ir=ir)


def run_horizon(panel, horizon, L):
    base = panel  # 无条件买入基准 = 全样本
    L.append(f"## 持有 {horizon} 日\n")
    L.append(f"- 触发: 当日收益 30日z-score ≤ 阈值（价格异常+当日下跌）；买入持 {horizon} 日，成本 {COST:.3f}")
    L.append(f"- 基准: 无条件买入（全样本）\n")
    L.append("| 阈值(z≤) | 信号数 | 信号胜率 | 基准胜率 | 超额lift | 净均收益/期 | 净IR | 净IR 95%CI |")
    L.append("|---------|--------|---------|---------|---------|------------|------|-----------|")
    res = {}
    for thr in THRESHOLDS:
        sig = panel[panel['z'] <= thr]
        r = eval_strategy(sig, base, horizon)
        res[thr] = r
        L.append(f"| {thr} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | {r['lift']*100:+.1f}pp | "
                 f"{r['mean']*100:+.2f}% | {r['ir']:.2f} | [{r['lo']*100:+.1f}%,{r['hi']*100:+.1f}%] |")
    L.append("")
    L.append(f"### 逐年（{horizon}d，z≤-3）\n")
    sig3 = panel[panel['z'] <= -3.0]
    L.append("| 年份 | 信号数 | 信号胜率 | 基准胜率 | 净均收益/期 |")
    L.append("|------|--------|---------|---------|------------|")
    for y, g in sig3.groupby('year'):
        r = eval_strategy(g, base[base['year'] == y], horizon)
        L.append(f"| {int(y)} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | {r['mean']*100:+.2f}% |")
    L.append("")
    r = res[-3.0]
    verdict = "✅ 显著跑赢基准" if (r['lift'] > 0.05 and r['lo'] > 0) else ("⚠️ 微弱/不稳健" if r['mean'] > 0 else "❌ 无超额（甚至为负）")
    L.append(f"**{horizon}d 判定（z≤−3，2021–2026，{r['n']} 个信号）**: 胜率 {r['win']*100:.1f}% vs 基准 {r['base_win']*100:.1f}%"
             f"，净均 {r['mean']*100:+.2f}%，净IR {r['ir']:.2f}，CI [{r['lo']*100:+.1f}%,{r['hi']*100:+.1f}%] ⇒ {verdict}\n")
    return res


def main(out_md=None, horizons=None):
    horizons = horizons or HORIZONS
    panel = load()
    print(f"样本: {len(panel)} 日期: {panel['date'].nunique()} 股票: {panel['code'].nunique()}")

    L = []
    L.append(f"# 异常大跌抄底策略复核（{'/'.join(str(h) for h in horizons)}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 原文档口径（2026-05）：价格异常+当日下跌 → 5日 +4.12% / 胜率 72%")
    L.append(f"- 口径：收盘买入、扣双边成本 {COST:.3f}、30日 z-score 触发的非重叠持有\n")
    for h in horizons:
        run_horizon(panel, h, L)
    L.append("## 说明\n")
    L.append("- 结论**仅适用于本股票池 / 本数据窗口**；单周期显著不等于可交易，须逐年与跨窗口稳健。")
    L.append("- 行级胜率为样本内统计；显著性须按 fold / 时间聚类（本项目教训 lessons 三.29）。")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='异常大跌抄底策略复核')
    ap.add_argument('--horizons', type=str, default=','.join(str(h) for h in HORIZONS),
                    help='持有周期，逗号分隔（默认 5,10,20）')
    ap.add_argument('--output', type=str, default='output/anomaly_dip_backtest.md')
    args = ap.parse_args()
    hs = [int(x) for x in args.horizons.split(',') if x.strip()]
    main(args.output, hs)
