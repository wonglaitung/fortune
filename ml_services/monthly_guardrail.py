#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月度护栏：对最新 Walk-forward 预测一键复核（DECISIONS D2）

对给定/最新 prediction_analysis.csv 输出：
  - 20d 行业中性 TopK 组合：净IR / 95%CI / P(IR>0.5) / 累计 / 换手
  - PBO（CSCV）与 DSR（Deflated Sharpe）
  - 信号超额 lift（胜率 − 无条件买入基准）
判定（docs/DECISIONS.md D2）：
  - 通过：净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 → 可升级
  - 保留：未达升级但 IR>0 → 低配辅助
  - 停用：IR≤0

用法：
  python3 ml_services/monthly_guardrail.py                          # 自动找最新 20d 预测
  python3 ml_services/monthly_guardrail.py --horizon 20 --pred <csv> --output output/mg.md
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.portfolio_backtest import load_panel, backtest, bootstrap, _stats, COST
from ml_services.eval_overfit import cscv_pbo, deflated_sharpe


def latest_pred(horizon):
    cands = glob.glob(f"output/*_catboost_{horizon}d/prediction_analysis.csv")
    if not cands:
        cands = glob.glob(f"output/walk_forward_catboost_{horizon}d_*.csv")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def signal_lift(df):
    """信号胜率 − 无条件买入基准胜率（净收益>0.5%）"""
    df = df.copy()
    for a, b in [('Predict_Prob', 'prob'), ('Actual_Return', 'ret'), ('Dynamic_Threshold', 'dyn')]:
        df[b] = pd.to_numeric(df[a], errors='coerce')
    df['dyn'] = df['dyn'].fillna(0.5)
    df = df.dropna(subset=['ret', 'prob'])
    trade = df['prob'] >= df['dyn']
    win = df['ret'] > COST
    base = win.mean()
    wr = win[trade].mean() if trade.sum() else np.nan
    return wr, base, (wr - base) if wr is not None else np.nan


def run(horizon, pred_csv, topk, cost, out_md):
    df = load_panel(pred_csv)
    bt = backtest(df, horizon, topk, True, cost=cost)  # 行业中性
    st = _stats(bt['top_net'].values, horizon)
    c = bootstrap(bt['top_net'].values, horizon)
    turnover = float(bt['turnover'].mean())
    wr, base, lift = signal_lift(df)

    from ml_services.eval_overfit import build_matrix
    M = build_matrix(horizon, pred_csv)
    pbo, _ = cscv_pbo(M.values)
    mu = M.mean(); sd = M.std(ddof=1); sr = mu / sd
    best = sr.idxmax()
    from scipy.stats import skew, kurtosis
    dsr = deflated_sharpe(sr[best], len(M), M.shape[1], float(sr.var(ddof=1)),
                          float(skew(M[best].values)), float(kurtosis(M[best].values, fisher=False)))

    passed = st['ir'] >= 0.7 and pbo < 0.5 and (dsr is not None and dsr >= 0.95)
    if st['ir'] <= 0:
        verdict = "🔴 停用（IR≤0）"
    elif passed:
        verdict = "🟢 通过 → 可升级"
    else:
        verdict = "🟡 保留低配（未达升级门槛）"

    L = []
    L.append(f"# 月度护栏复核（{horizon}d）\n")
    L.append(f"- 日期: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 数据: `{pred_csv}`")
    L.append(f"- TopK={topk} 行业中性，成本 {cost:.3f}\n")
    L.append("| 指标 | 值 | 门槛 |")
    L.append("|------|----|------|")
    L.append(f"| 净IR | **{st['ir']:.2f}** [{c['ir_lo']:.2f},{c['ir_hi']:.2f}] | ≥0.7 且 CI 下限>0 更佳 |")
    L.append(f"| P(IR>0.5) | {c['p_ir']*100:.0f}% | — |")
    L.append(f"| 净均收益/期 | {st['mean']*100:+.2f}% | >0 |")
    L.append(f"| 累计净收益 | {st['cum']*100:+.1f}% | — |")
    L.append(f"| 换手 | {turnover*100:.0f}% | 越低越好 |")
    L.append(f"| **PBO** | **{pbo:.2f}** | <0.5 |")
    L.append(f"| **DSR**（最优 {best}） | **{dsr:.3f}** | ≥0.95 |")
    L.append(f"| 信号胜率 / 基准胜率 | {wr*100:.1f}% / {base*100:.1f}% | — |")
    L.append(f"| **超额 lift** | **{lift*100:+.1f}pp** | >0 |")
    L.append("")
    L.append(f"## 判定：{verdict}")
    L.append("")
    L.append("> 判定规则：净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 → 升级；IR>0 但未达 → 保留低配；IR≤0 → 停用。")
    L.append("> （docs/DECISIONS.md D2）\n")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")
    return passed


def main():
    ap = argparse.ArgumentParser(description='月度护栏（净IR/PBO/DSR/lift）')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--cost', type=float, default=COST)
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    pred = args.pred or latest_pred(args.horizon)
    if not pred:
        print("❌ 未找到 prediction_analysis.csv，请用 --pred 指定")
        sys.exit(1)
    out = args.output or f"output/monthly_guardrail_{datetime.now():%Y%m%d}.md"
    run(args.horizon, pred, args.topk, args.cost, out)


if __name__ == '__main__':
    main()