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


def latest_pred(horizon, market='hk'):
    """最新 prediction_analysis.csv；默认港股（排除 A股目录，避免误取）"""
    if str(market).lower().startswith('a'):
        # 覆盖 catboost 与 lightgbm（P4.1 A/B）两种目录
        cands = glob.glob(f"output/*_a_stock_*_{horizon}d/prediction_analysis.csv")
    else:
        cands = [f for f in glob.glob(f"output/*_catboost_{horizon}d/prediction_analysis.csv")
                 if '_a_stock_' not in f]
        if not cands:
            cands = glob.glob(f"output/walk_forward_catboost_{horizon}d_*.csv")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def signal_lift(df):
    """信号胜率 − 无条件买入基准胜率（净收益>0.5%）"""
    df = df.copy()
    for a, b in [('Predict_Prob', 'prob'), ('Actual_Return', 'ret'), ('Dynamic_Threshold', 'dyn')]:
        # 旧A股CSV缺 Dynamic_Threshold → 阈值回退 0.5（与 backtest_eval 一致）
        df[b] = pd.to_numeric(df[a], errors='coerce') if a in df.columns else np.nan
    df['dyn'] = df['dyn'].fillna(0.5)
    df = df.dropna(subset=['ret', 'prob'])
    trade = df['prob'] >= df['dyn']
    win = df['ret'] > COST
    base = win.mean()
    wr = win[trade].mean() if trade.sum() else np.nan
    return wr, base, (wr - base) if wr is not None else np.nan


def pbo_null_band(M, n_perm=10, seed=0):
    """PBO 的零假设分布：把时间轴整行重排后重算 PBO，得到「完全没有时间可预测性」时的取值范围。

    为什么必须有这个（lessons 三.63）：CSCV 在样本内挑夏普最高的配置、看它样本外排第几。
    当几个配置彼此不可分时，样本内 argmax 就是抛硬币 ⇒ PBO 必然趋近 1。
    实测 T=39×6 时零假设带为 0.73–0.94，真实数据 0.871 落在带内 ⇒ 与「无信息」不可区分。
    单看一个 PBO 数字会把它误读成「过拟合」，故此函数把噪声带一并算出。

    ⚠️ 必须**整行重排**，不能逐列独立打乱：
    逐列打乱会把「同一期各配置面对同一个市场」这个真实相关性也毁掉，
    六个配置变成互不相干 ⇒ CSCV 的 IS argmax 更随机 ⇒ PBO 被系统性推到 1，
    零假设带会假性收窄到 [0.97, 1.00]，反而掩盖真实数据的读数。
    整行重排保留跨配置相关性、只破坏时间可预测性，才是这里该问的零假设。
    """
    rng = np.random.default_rng(seed)
    vals = []
    X = M.values
    for _ in range(n_perm):
        vals.append(cscv_pbo(X[rng.permutation(X.shape[0]), :])[0])
    v = np.asarray(vals)
    return float(np.percentile(v, 5)), float(np.percentile(v, 95)), len(v)


def config_spread(M):
    """各配置全期夏普的极差。极差过小 ⇒ 配置互不可分 ⇒ PBO 必然高（lessons 三.63 准则 2）。"""
    sr = (M.mean() / M.std(ddof=1))
    return float(sr.max() - sr.min()), sr


def run(horizon, pred_csv, topk, cost, out_md, market='hk', null_perm=10):
    df = load_panel(pred_csv, market=market)
    bt = backtest(df, horizon, topk, True, cost=cost)  # 行业中性
    st = _stats(bt['top_net'].values, horizon)
    c = bootstrap(bt['top_net'].values, horizon)
    turnover = float(bt['turnover'].mean())
    wr, base, lift = signal_lift(df)

    from ml_services.eval_overfit import build_matrix
    M = build_matrix(horizon, pred_csv, market=market)
    pbo, _ = cscv_pbo(M.values)
    mu = M.mean(); sd = M.std(ddof=1); sr = mu / sd
    best = sr.idxmax()
    from scipy.stats import skew, kurtosis
    dsr = deflated_sharpe(sr[best], len(M), M.shape[1], float(sr.var(ddof=1)),
                          float(skew(M[best].values)), float(kurtosis(M[best].values, fisher=False)))

    spread, _ = config_spread(M)
    null_lo = null_hi = None
    if null_perm and null_perm > 0:
        null_lo, null_hi, _ = pbo_null_band(M, n_perm=null_perm)

    if null_lo is not None:
        in_band = null_lo <= pbo <= null_hi
        if in_band:
            pbo_read = (f"PBO {pbo:.2f} 落在零假设带 [{null_lo:.2f}, {null_hi:.2f}] 内 "
                        f"⇒ 与「无信息」不可区分，**不可读作过拟合**")
        elif pbo < null_lo:
            pbo_read = (f"PBO {pbo:.2f} **低于**零假设带 [{null_lo:.2f}, {null_hi:.2f}] 下界 "
                        f"⇒ 本窗口的偶然，不可采信（lessons 三.29）")
        else:
            pbo_read = (f"PBO {pbo:.2f} **高于**零假设带 [{null_lo:.2f}, {null_hi:.2f}] 上界 "
                        f"⇒ 该过拟合读数在本尺寸下才站得住")
    else:
        pbo_read = f"PBO {pbo:.2f}（未算零假设带，--null-perm=0）"

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
    L.append(f"| 净IR | **{st['ir']:.2f}** [{c['ir_lo']:.2f},{c['ir_hi']:.2f}] | ≥0.7 且 CI 下界>0 更佳 |")
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
    L.append(f"### PBO 读法（lessons 三.63，{len(M)} 期 × {M.shape[1]} 配置）")
    L.append("")
    L.append(f"- {pbo_read}")
    L.append(f"- 六个配置全期夏普极差 **{spread:.3f}**"
             + ("（过小 ⇒ 配置互不可分 ⇒ PBO 必然偏高，先看别的指标）" if spread < 0.2 else ""))
    L.append("- **PBO 必须与零假设带同报**；单看一个数字会把噪声读成过拟合，"
             "也会把本窗口的偶然读成改善。")

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
    ap.add_argument('--market', type=str, default='hk', choices=['hk', 'a'],
                    help='市场：hk=港股（默认），a=A股（A_STOCK_REFORM_PLAN P2.1）')
    ap.add_argument('--null-perm', type=int, default=10,
                    help='PBO 零假设带的重排次数；0=不算（会退回「单看一个 PBO 数字」的坏用法）')
    args = ap.parse_args()

    pred = args.pred or latest_pred(args.horizon, market=args.market)
    if not pred:
        print("❌ 未找到 prediction_analysis.csv，请用 --pred 指定")
        sys.exit(1)
    suffix = '_a' if args.market == 'a' else ''
    out = (args.output
           or f"output/monthly_guardrail_{args.horizon}d_{datetime.now():%Y%m%d}{suffix}.md")
    run(args.horizon, pred, args.topk, args.cost, out, market=args.market,
        null_perm=args.null_perm)


if __name__ == '__main__':
    main()