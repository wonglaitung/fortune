#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒指信号交易严格复核（成本 + bootstrap）

输入：data/hsi_walk_forward/hsi_prediction_analysis_*.csv（fold,date,prob,pred,actual_return）
策略：非重叠调仓（每 horizon 交易日一次）；pred=1 做多 / 0 空仓（long-flat），
      或 2*pred-1（long-short）。入场成本 0.5%（round-trip 在入场时计一次）。
基准：恒指买入持有（always long）。
输出：净均收益 / 胜率 vs 基准 / 净IR / bootstrap CI。

用法：python3 ml_services/hsi_signal_backtest.py --horizon 20
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST = 0.005
DEFAULT_CSV = {
    1: 'data/hsi_walk_forward/hsi_prediction_analysis_20260923_134140.csv',
    5: 'data/hsi_walk_forward/hsi_prediction_analysis_20260923_134145.csv',
    20: 'data/hsi_walk_forward/hsi_prediction_analysis_20260923_134150.csv',
}


def boot(x, n=2000, seed=42):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    rng = np.random.RandomState(seed); idx = np.arange(len(x))
    b = [x[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def run(horizon, csv, out_md):
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    rb = df.iloc[::horizon].reset_index(drop=True)  # 非重叠
    rb['pos_lf'] = rb['pred'].astype(int)
    rb['pos_ls'] = 2 * rb['pred'].astype(int) - 1
    # 入场成本：位置变化时计一次 0.5%
    entry_lf = rb['pos_lf'].diff().fillna(rb['pos_lf']).abs().clip(upper=1).values
    entry_ls = rb['pos_ls'].diff().fillna(rb['pos_ls']).abs().clip(upper=1).values
    ret_lf = rb['pos_lf'] * rb['actual_return'] - entry_lf * COST
    ret_ls = rb['pos_ls'] * rb['actual_return'] - entry_ls * COST
    base = rb['actual_return'].values  # 买入持有

    def stats(x):
        m = x.mean(); s = x.std(ddof=1)
        return m, (m / s * np.sqrt(252.0 / horizon) if s > 0 else np.nan), (x > 0).mean()

    m_lf, ir_lf, wr_lf = stats(ret_lf)
    m_ls, ir_ls, wr_ls = stats(ret_ls)
    m_b, ir_b, wr_b = stats(base)
    lo_lf, hi_lf = boot(ret_lf)
    lo_ls, hi_ls = boot(ret_ls)

    L = []
    L.append(f"# 恒指信号交易复核（{horizon}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 数据: `{csv}`　非重叠期数: {len(rb)}　入场成本 {COST:.3f}\n")
    L.append("| 策略 | 净均收益/期 | 胜率 | 净IR | 净均收益 95%CI |")
    L.append("|------|------------|------|------|----------------|")
    L.append(f"| 恒指买入持有（基准） | {m_b*100:+.2f}% | {wr_b*100:.1f}% | {ir_b:.2f} | — |")
    L.append(f"| 信号 Long/Flat | {m_lf*100:+.2f}% | {wr_lf*100:.1f}% | {ir_lf:.2f} | [{lo_lf*100:+.2f}%,{hi_lf*100:+.2f}%] |")
    L.append(f"| 信号 Long/Short | {m_ls*100:+.2f}% | {wr_ls*100:.1f}% | {ir_ls:.2f} | [{lo_ls*100:+.2f}%,{hi_ls*100:+.2f}%] |")
    L.append("")
    L.append("## 结论\n")
    verdict = "✅ 显著优于买入持有" if (ir_lf > 0.5 and lo_lf > 0) else ("⚠️ 优于持有但不显著" if m_lf > m_b else "❌ 不优于持有")
    L.append(f"- 买入持有基准：净IR {ir_b:.2f}；信号 Long/Flat 净IR {ir_lf:.2f}，"
             f"差 {ir_lf-ir_b:+.2f}，CI [{lo_lf*100:+.2f}%,{hi_lf*100:+.2f}%]")
    L.append(f"- 判定: {verdict}")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def main():
    ap = argparse.ArgumentParser(description='恒指信号交易复核')
    ap.add_argument('--horizon', type=int, choices=[1, 5, 20], default=20)
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()
    csv = args.csv or DEFAULT_CSV[args.horizon]
    out = args.output or f"output/hsi_signal_backtest_{args.horizon}d.md"
    run(args.horizon, csv, out)


if __name__ == '__main__':
    main()