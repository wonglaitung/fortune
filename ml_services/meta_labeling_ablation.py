#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元标签消融实验：在主模型有 edge 的子集（如 5d 2025）+ Kelly 仓位下复测

问题：之前的原型只测了"过滤"（precision），且在全期（主模型 lift≈0）上做，
      未测 de Prado 方法的另一半——按 P(correct) 做 Kelly 仓位。
本脚本：复用 meta_labeling_prototype 的管线，对指定年份子集比较三种方案：
  - Baseline：全部信号，等权
  - Meta-过滤：meta_prob >= 阈值，等权
  - Meta+Kelly：按 0.5*clip(2p-1,0,1) 定仓位（p=meta_prob）
指标：交易数/保留率/Precision/净均收益/净 IR（按信号日分批次）。

用法：
  python3 ml_services/meta_labeling_ablation.py --horizon 5  --year 2025
  python3 ml_services/meta_labeling_ablation.py --horizon 20 --year 2025
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.meta_labeling_prototype import (
    build_dataset, attach_labels, meta_feature_frame, purged_walk_forward,
    COST,
)

DEFAULT_PRED = {
    5: 'output/20260922_212530_catboost_5d/prediction_analysis.csv',
    20: 'output/20260922_162806_catboost_20d/prediction_analysis.csv',
}


def _batch_ir(df, weights, horizon):
    """按信号日分批次，计算等风险组合的净均收益与年化 IR"""
    r = df['net_ret'].values
    w = np.asarray(weights, dtype=float)
    g = pd.DataFrame({'d': df['date'].values, 'wr': w * r, 'w': w})
    g = g.groupby('d').sum()
    g = g[g['w'] > 0]
    if len(g) < 2:
        return np.nan, np.nan
    batch = (g['wr'] / g['w']).values
    mean = float(np.mean(batch))
    std = float(np.std(batch, ddof=1))
    ir = mean / std * np.sqrt(252.0 / horizon) if std > 0 else np.nan
    return mean, ir


def _scheme(df, weights, horizon):
    w = np.asarray(weights, dtype=float)
    keep = w > 0
    d = df[keep]
    n = len(d)
    if n == 0:
        return dict(trades=0, retention=0.0, precision=np.nan, mean=np.nan, ir=np.nan)
    precision = float(d['net_win'].mean())
    wr = w[keep]
    mean = float(np.sum(wr * d['net_ret'].values) / np.sum(wr))
    _, ir = _batch_ir(d, wr, horizon)
    return dict(trades=n, retention=n / len(df), precision=precision, mean=mean, ir=ir)


def _batch_returns(df, weights, horizon):
    """按信号日的组合批次收益（Series，索引=日期）"""
    w = np.asarray(weights, dtype=float)
    keep = w > 0
    d = df[keep]
    w = w[keep]
    g = pd.DataFrame({'d': d['date'].values, 'wr': w * d['net_ret'].values, 'w': w}).groupby('d').sum()
    g = g[g['w'] > 0]
    return g['wr'] / g['w']


def _ir(batch, horizon):
    if len(batch) < 2:
        return np.nan
    m = float(np.mean(batch)); s = float(np.std(batch, ddof=1))
    return m / s * np.sqrt(252.0 / horizon) if s > 0 else np.nan


def _bootstrap_ir_delta(df, w_base, w_scheme, horizon, n_boot=1000, seed=42):
    """对信号日做 block bootstrap，返回 (delta_ir均值, 2.5%, 97.5%, P(delta>0))"""
    b0 = _batch_returns(df, w_base, horizon)
    b1 = _batch_returns(df, w_scheme, horizon)
    common = b0.index.intersection(b1.index)
    b0 = b0.loc[common]; b1 = b1.loc[common]
    if len(common) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(common))
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = _ir(b1.values[s], horizon) - _ir(b0.values[s], horizon)
        if np.isfinite(d):
            deltas.append(d)
    if not deltas:
        return (np.nan, np.nan, np.nan, np.nan)
    deltas = np.array(deltas)
    return (float(deltas.mean()), float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)), float((deltas > 0).mean()))


def run(horizon, year, pred_csv, cache_dir, out_md):
    print(f"\n{'='*70}\n元标签消融  horizon={horizon}  year={year}\n{'='*70}")
    pred, prices = build_dataset(pred_csv, cache_dir)
    sig = pred[pred['Predict_Prob'] >= pred['Dynamic_Threshold']].copy()
    sig = attach_labels(sig, prices, horizon).dropna(subset=['tb_ret']).reset_index(drop=True)
    X = meta_feature_frame(sig, prices).reset_index(drop=True)
    meta_prob, cal_prob, thresholds = purged_walk_forward(sig, X, horizon, return_calibrated=True)
    sig['_thr'] = sig['fold'].map(thresholds) if 'fold' in sig.columns else 0.5
    sig['meta_prob'] = meta_prob.values
    sig['meta_prob_cal'] = cal_prob.values

    # 选择评估子集
    if year:
        sig = sig[sig['date'].dt.year == year].copy()
    sig = sig[sig['meta_prob'].notna()].copy()
    sig['net_ret'] = sig['tb_ret'] - COST
    sig['net_win'] = (sig['net_ret'] > 0).astype(int)
    print(f"评估样本: {len(sig)}  年份={year or '全期'}")

    if len(sig) == 0:
        print("无样本，退出")
        return

    thr = sig['_thr'].fillna(0.5).values
    w_base = np.ones(len(sig))
    w_metaf = (sig['meta_prob'] >= thr).astype(float).values
    w_kelly = 0.5 * np.clip(2 * sig['meta_prob'].values - 1, 0, 1)
    w_cal = 0.5 * np.clip(2 * sig['meta_prob_cal'].values - 1, 0, 1)
    w_primary = 0.5 * np.clip(2 * sig['Predict_Prob'].values - 1, 0, 1)
    w_vol = np.clip(1.0 - pd.Series(sig['daily_vol'].values).rank(pct=True).values, 0, 1)
    w_rank = pd.Series(sig['meta_prob'].values).rank(pct=True).values

    schemes = [
        ("Baseline（全部信号，等权）", w_base),
        ("Meta-过滤（等权）", w_metaf),
        ("Meta+Kelly（元概率定仓）", w_kelly),
        ("Meta+Kelly‑Cal（Isotonic 校准后）", w_cal),
        ("Meta-Rank（元概率排名定仓）", w_rank),
        ("Primary+Kelly（主概率定仓，对照）", w_primary),
        ("VolRule（低波动加仓，对照）", w_vol),
    ]
    results = {name: _scheme(sig, w, horizon) for name, w in schemes}
    # block bootstrap：各方案相对 Baseline 的净IR增量
    ci = {}
    for name, w in schemes:
        if name.startswith("Baseline"):
            continue
        ci[name] = _bootstrap_ir_delta(sig, w_base, w, horizon)
    ci_meta_vs_primary = _bootstrap_ir_delta(sig, w_primary, w_kelly, horizon)

    def _row(name, m):
        d, lo, hi, ppos = ci.get(name, (np.nan,) * 4)
        cir = '—' if np.isnan(d) else f"{d:+.2f} [{lo:+.2f},{hi:+.2f}] ({ppos*100:.0f}%)"
        return (f"| {name} | {m['trades']} | {_p(m['retention'])} | {_p(m['precision'])} | "
                f"{_p(m['mean'])} | {_f(m['ir'])} | {cir} |")

    L = []
    L.append(f"# 元标签消融（{horizon}d，{year or '全期'}）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 评估样本: {len(sig)}　成本: {COST:.3f}　Kelly=0.5×clip(2p−1,0,1)")
    L.append(f"- 目的: 主模型有 edge 子集上，检验元标签**过滤 vs 仓位**，并给 Bootstrap 置信区间")
    L.append(f"- 「ΔIR」= 相对 Baseline 的净IR增量 block bootstrap（1000 次，中括号 95%CI，括号内 P(Δ>0)）\n")
    L.append("| 方案 | 交易数 | 保留率 | Precision | 净均收益 | 净IR | ΔIR vs Baseline |")
    L.append("|------|--------|--------|-----------|---------|------|-----------------|")
    for name, _ in schemes:
        L.append(_row(name, results[name]))
    L.append("")
    base = results["Baseline（全部信号，等权）"]

    meta_f = results["Meta-过滤（等权）"]
    meta_k = results["Meta+Kelly（元概率定仓）"]
    primary_k = results["Primary+Kelly（主概率定仓，对照）"]
    cal_k = results["Meta+Kelly‑Cal（Isotonic 校准后）"]
    d_mp, lo_mp, hi_mp, p_mp = ci_meta_vs_primary
    L.append("### 元模型增量（相对 Primary+Kelly 的 block bootstrap）")
    L.append(f"- ΔIR(元−主) = {d_mp:+.2f} [{lo_mp:+.2f}, {hi_mp:+.2f}]，P(Δ>0)={p_mp*100:.0f}%")
    L.append(f"- 校准前后：Meta-Kelly {meta_k['ir']:.2f} → Meta-Kelly-Cal {cal_k['ir']:.2f}\n")

    prec_lift = meta_f['precision'] - base['precision']
    ir_filter = meta_f['ir'] - base['ir']
    ir_kelly = meta_k['ir'] - base['ir']
    L.append("## 结论\n")
    L.append(f"- 过滤：Precision {prec_lift*100:+.1f}pp（保留率 {meta_f['retention']*100:.0f}%），"
             f"净IR {ir_filter:+.2f}")
    L.append(f"- 仓位：净IR {ir_kelly:+.2f}（{base['ir']:.2f} → {meta_k['ir']:.2f}），"
             f"交易数 {meta_k['trades']}")
    L.append(f"- **增量对照**：用主概率定仓净IR = {primary_k['ir']:.2f}；"
             f"元模型相对主概率的增量 = {meta_k['ir'] - primary_k['ir']:+.2f}")

    improved_kelly = meta_k['ir'] > base['ir'] + 0.05
    if base['ir'] <= 0:
        if improved_kelly:
            verdict = "⚠️ Kelly 仅**减轻亏损**；主模型无正 edge，仍未转正"
        else:
            verdict = "❌ 无改善"
    elif improved_kelly and meta_k['ir'] >= 0.5:
        verdict = "✅ 主模型有正 edge + Kelly 提升净IR（方法生效）"
    elif improved_kelly:
        verdict = "⚠️ 有改善但不显著"
    else:
        verdict = "❌ 无改善"
    L.append(f"- 判定: {verdict}")
    L.append(f"- 注: **过滤**几乎无提升；价值主要在 **Kelly 仓位**。"
             f"但 Kelly 只能放大/保护既有 edge，**不能把负 edge 转正**（见全期/2024 对比）。")
    L.append("")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def _p(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"


def _f(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description='元标签消融（年份子集 + Kelly）')
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 20])
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--cache-dir', type=str, default='data/feature_cache')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    pred_csv = args.pred or DEFAULT_PRED[args.horizon]
    suffix = f"{args.year}" if args.year else "all"
    out = args.output or f"output/meta_labeling_ablation_{args.horizon}d_{suffix}.md"
    run(args.horizon, args.year, pred_csv, args.cache_dir, out)


if __name__ == '__main__':
    main()
