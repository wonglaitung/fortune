#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测过拟合护栏：PBO（CSCV）+ Deflated Sharpe Ratio（DSR）

- PBO：Bailey et al. (2014) 的组合对称交叉验证（CSCV），
  估计"样本内最优配置在样本外落到中位数以下"的概率。
  判读：PBO < 0.5 未过拟合；< 0.25 强证据。
- DSR：Bailey & López de Prado (2014)，按试验次数 N 与 Sharpe 方差对 Sharpe 做收缩。

输入矩阵 M：形状 (T 期, N 配置) 的逐期收益。
用法（对 Phase 3 配置族做检验）：
  python3 ml_services/eval_overfit.py --horizon 20
"""

import os
import sys
import argparse
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.portfolio_backtest import load_panel, backtest, DEFAULT_PRED

EULER = 0.5772156649015329


def _sharpe_cols(M):
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    sd[sd == 0] = np.nan
    return mu / sd


def cscv_pbo(M, n_splits=8):
    """CSCV 估计 PBO；返回 (pbo, logits)"""
    T, N = M.shape
    S = n_splits
    while T // S < 2 and S > 4:
        S -= 2
    bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [M[bounds[i]:bounds[i + 1]] for i in range(S)]
    logits = []
    for comb in combinations(range(S), S // 2):
        comp = [i for i in range(S) if i not in comb]
        IS = np.vstack([blocks[i] for i in comb])
        OOS = np.vstack([blocks[i] for i in comp])
        is_p = _sharpe_cols(IS)
        oos_p = _sharpe_cols(OOS)
        if np.all(np.isnan(is_p)):
            continue
        nstar = int(np.nanargmax(is_p))
        ranks = rankdata(oos_p, method='average')
        w = ranks[nstar] / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    pbo = float((logits <= 0).mean()) if len(logits) else np.nan
    return pbo, logits


def deflated_sharpe(sr, T, n_trials, var_sr, skewness, kurt):
    """DSR = P(真实 Sharpe > 0)，对多重试验与非正态做收缩"""
    if T < 3 or n_trials < 2 or var_sr <= 0:
        return np.nan
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)
    denom = np.sqrt(max(1 - skewness * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-9))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def build_matrix(horizon):
    """构建配置族 (topk × 中性) 的逐期收益矩阵"""
    df = load_panel(DEFAULT_PRED[horizon])
    cols = {}
    dates = None
    for topk in (5, 10, 20):
        for neutral in (False, True):
            bt = backtest(df, horizon, topk, neutral)
            name = f"top{topk}-{'neutral' if neutral else 'raw'}"
            cols[name] = bt.set_index('date')['top_net']
    M = pd.DataFrame(cols).dropna()
    return M


def run(horizon, out_md):
    M = build_matrix(horizon)
    T, N = M.shape
    pbo, logits = cscv_pbo(M.values, n_splits=8)
    # 各配置指标
    mu = M.mean(); sd = M.std(ddof=1)
    sr = (mu / sd)
    ann = np.sqrt(252.0 / horizon)
    best = sr.idxmax()
    b = M[best]
    sr_p = sr[best]
    dsr = deflated_sharpe(sr_p, T, N, float(sr.var(ddof=1)),
                          float(skew(b.values)), float(kurtosis(b.values, fisher=False)))

    L = []
    L.append(f"# 过拟合护栏（{horizon}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 配置矩阵: T={T} 期 × N={N} 配置（topk∈{{5,10,20}} × raw/行业中性）\n")
    L.append("## 各配置（净，未年化 Sharpe）\n")
    L.append("| 配置 | 均收益/期 | Sharpe/期 | 年化IR |")
    L.append("|------|----------|-----------|--------|")
    for c in M.columns:
        L.append(f"| {c} | {mu[c]*100:+.2f}% | {sr[c]:.3f} | {sr[c]*ann:.2f} |")
    L.append("")
    L.append("## 护栏结果\n")
    L.append(f"- **PBO = {pbo:.2f}** → " +
             ("✅ <0.5 未显示过拟合" if pbo < 0.5 else "❌ ≥0.5 疑似过拟合") +
             ("（<0.25 强证据）" if pbo < 0.25 else ""))
    L.append(f"- 样本内最优配置: **{best}**（年化IR {sr[best]*ann:.2f}）")
    L.append(f"- **DSR = {dsr:.3f}**（N={N} 次试验，计入 Sharpe 方差与非正态）→ " +
             ("✅ 通过（>0.95）" if dsr > 0.95 else "⚠️ 未达 0.95，存在选择偏差风险"))
    L.append("")
    L.append("> 说明：PBO 判读 <0.5 未过拟合；DSR>0.95 表示经多重试验收缩后仍显著。")
    L.append(f"> ⚠️ {horizon}d 仅 {T} 期、配置族仅 {N} 个，检验功效有限，结论仅供参考。\n")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def main():
    ap = argparse.ArgumentParser(description='PBO + DSR 过拟合护栏')
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 20])
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()
    out = args.output or f"output/overfit_guard_{args.horizon}d.md"
    run(args.horizon, out)


if __name__ == '__main__':
    main()
