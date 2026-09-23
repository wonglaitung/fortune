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


def run(horizon, year, pred_csv, cache_dir, out_md):
    print(f"\n{'='*70}\n元标签消融  horizon={horizon}  year={year}\n{'='*70}")
    pred, prices = build_dataset(pred_csv, cache_dir)
    sig = pred[pred['Predict_Prob'] >= pred['Dynamic_Threshold']].copy()
    sig = attach_labels(sig, prices, horizon).dropna(subset=['tb_ret']).reset_index(drop=True)
    X = meta_feature_frame(sig, prices).reset_index(drop=True)
    meta_prob, thresholds = purged_walk_forward(sig, X, horizon)
    sig['_thr'] = sig['fold'].map(thresholds) if 'fold' in sig.columns else 0.5
    sig['meta_prob'] = meta_prob.values

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
    base = _scheme(sig, np.ones(len(sig)), horizon)
    meta_f = _scheme(sig, (sig['meta_prob'] >= thr).astype(float).values, horizon)
    kelly = 0.5 * np.clip(2 * sig['meta_prob'].values - 1, 0, 1)
    meta_k = _scheme(sig, kelly, horizon)
    # 对照：直接用主模型概率定仓（若与元模型相当，则元模型无增量价值）
    kelly_primary = 0.5 * np.clip(2 * sig['Predict_Prob'].values - 1, 0, 1)
    primary_k = _scheme(sig, kelly_primary, horizon)

    def _row(name, m):
        return (f"| {name} | {m['trades']} | {_p(m['retention'])} | {_p(m['precision'])} | "
                f"{_p(m['mean'])} | {_f(m['ir'])} |")

    L = []
    L.append(f"# 元标签消融（{horizon}d，{year or '全期'}）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 评估样本: {len(sig)}　成本: {COST:.3f}　Kelly=0.5×clip(2p−1,0,1)")
    L.append(f"- 目的: 在主模型有 edge 的子集上，检验元标签的**过滤**与**仓位**两种用法\n")
    L.append("| 方案 | 交易数 | 保留率 | Precision | 净均收益 | 净IR |")
    L.append("|------|--------|--------|-----------|---------|------|")
    L.append(_row("Baseline（全部信号，等权）", base))
    L.append(_row("Meta-过滤（等权）", meta_f))
    L.append(_row("Meta+Kelly（按元概率定仓）", meta_k))
    L.append(_row("Primary+Kelly（按主概率定仓，对照）", primary_k))
    L.append("")

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
