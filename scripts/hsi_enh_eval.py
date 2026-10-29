#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSI 方向增强 A/B 评估（D35 判据）。

输入：data/hsi_walk_forward/hsi_prediction_analysis_*.csv（fold,date,prob,pred,actual_return）
口径：非重叠调仓（每 horizon 期一次）；pred=1 做多 / 0 空仓；入场成本 0.5%（仓位变化时）。
输出：信号 Long/Flat vs 买入持有——净均收益、净IR、**超额均收益 95%CI**、**折聚类 t**、**maxDD**。

判据（D35 §四）：
  主：净IR(信号) ≥ 净IR(持有) 且 超额 CI 下界 > 0；
  方向技能：折聚类 t>1.96；风险：maxDD(信号) ≤ maxDD(持有)。
  （PBO/DSR 针对"配置选择"，单一方向信号无配置集 ⇒ 不适用，N/A。）
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.costs import COST_ROUND_TRIP


def _stats(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float('nan'), float('nan')
    m = x.mean()
    s = x.std(ddof=1)
    return m, (m / s if s > 0 else float('nan'))


def _boot_ci(x, n=2000, seed=42):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    b = [x[rng.choice(len(x), len(x), replace=True)].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _maxdd(ret):
    cum = (1 + np.asarray(ret, float)).cumprod()
    return float((cum / np.maximum.accumulate(cum) - 1).min())


def evaluate(csv, horizon):
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    rb = df.iloc[::horizon].reset_index(drop=True)
    pos = rb['pred'].astype(int).values
    entry = np.abs(np.diff(np.r_[pos[0], pos])).clip(max=1)
    ret_sig = pos * rb['actual_return'].values - 0.5 * entry * COST_ROUND_TRIP
    base = rb['actual_return'].values
    excess = ret_sig - base
    ir_sig = _stats(ret_sig)[1] * np.sqrt(252.0 / horizon)
    ir_base = _stats(base)[1] * np.sqrt(252.0 / horizon)
    ex_lo, ex_hi = _boot_ci(excess)
    # 折聚类 t（按 fold 分组求均值再 t）
    if 'fold' in rb.columns:
        fm = rb.assign(ex=excess).groupby('fold')['ex'].mean().values
        ft = float(fm.mean() / (fm.std(ddof=1) / np.sqrt(len(fm)))) if len(fm) > 1 and fm.std(ddof=1) > 0 else float('nan')
    else:
        ft = float('nan')
    return dict(
        n=len(rb), ir_sig=ir_sig, ir_base=ir_base,
        ex_mean=excess.mean(), ex_lo=ex_lo, ex_hi=ex_hi,
        fold_t=ft, maxdd_sig=_maxdd(ret_sig), maxdd_base=_maxdd(base),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--csv', nargs='+', required=True, help='label=csv（多个）')
    args = ap.parse_args()

    print(f"{'变体':<12} {'期数':>4} {'净IR_信号':>9} {'净IR_持有':>9} {'超额均/期':>9} {'超额95%CI':>18} {'折聚类t':>7} {'maxDD_信号':>10} {'maxDD_持有':>10} {'判定':>4}")
    for item in args.csv:
        label, csv = item.split('=', 1)
        r = evaluate(csv, args.horizon)
        main_ok = (r['ir_sig'] >= r['ir_base']) and (r['ex_lo'] > 0)
        skill_ok = (r['fold_t'] == r['fold_t']) and (r['fold_t'] > 1.96)  # not nan
        dd_ok = r['maxdd_sig'] >= r['maxdd_base']  # 负数比较：信号回撤更浅(更接近0)则>=成立
        verdict = 'PASS' if (main_ok and skill_ok and dd_ok) else 'FAIL'
        print(f"{label:<12} {r['n']:>4} {r['ir_sig']:>9.2f} {r['ir_base']:>9.2f} "
              f"{r['ex_mean']*100:>8.2f}% [{r['ex_lo']*100:>6.2f}%,{r['ex_hi']*100:>6.2f}%] "
              f"{r['fold_t']:>7.2f} {r['maxdd_sig']*100:>9.1f}% {r['maxdd_base']*100:>9.1f}% {verdict:>4}")


if __name__ == '__main__':
    main()
