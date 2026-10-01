#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相对 alpha 专用校验（预注册实验 (a)，见 lessons 十 / progress.txt）

用途：共享评估链（backtest_eval / monthly_guardrail / portfolio_backtest）按**绝对**收益口径设计，
在 label_mode='relative_hsi' 的产物上会低估"剥离 Beta 后的特异信号"。本脚本**后处理**（不改共享脚本），
用 Relative_Return（股票未来收益 − HSI 未来收益）做三件事：
  1) 方向技能：预测 Relative_Return>0 的准确率，基准 = 实际跑赢恒指的比例
  2) Rank IC：Predict_Prob 与 Relative_Return 的秩相关（相对收益上的信息量）
  3) 市场中性 P&L：只对"做多"信号计 Relative_Return 收益 → 净 IR / 胜率（剥离大盘后的可交易性）

HSI 未来收益由本脚本重算（腾讯 qfq 恒指按日期对齐 + shift(-horizon)），
不改数据源/评估链，确保绝对口径基线不被污染。

用法：
    python3 scripts/rel_alpha_check.py --input output/<dir>/prediction_analysis.csv --horizon 20
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_hsi_forward_return(horizon: int) -> pd.Series:
    """重算 HSI qfq 未来收益，索引归一为 naive 日期（与个股 Date 对齐）。"""
    from data_services.tencent_finance import get_hsi_data_tencent
    from ml_services.ml_trading_model import _normalize_ohlcv_cols
    # 取足够长的历史（覆盖 CSV 区间 + horizon）
    hsi = get_hsi_data_tencent(period_days=5000)
    if hsi is None or hsi.empty:
        raise RuntimeError("无法获取恒生指数数据")
    hsi = _normalize_ohlcv_cols(hsi)
    idx = pd.to_datetime(hsi.index)
    idx = idx.tz_convert(None) if idx.tz is not None else idx
    hsi.index = idx.normalize()
    close = hsi['Close']
    fwd = close.shift(-horizon) / close - 1
    return fwd


def main():
    ap = argparse.ArgumentParser(description='相对 alpha 专用校验（Relative_Return 口径）')
    ap.add_argument('--input', required=True, help='prediction_analysis.csv 路径')
    ap.add_argument('--horizon', type=int, required=True, help='预测周期（天）')
    ap.add_argument('--gate', type=float, default=0.5, help='做多阈值（默认 0.5）')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if 'Date' not in df.columns or 'Actual_Return' not in df.columns:
        raise RuntimeError(f"CSV 缺少必要列: 需要 Date/Actual_Return，实际 {list(df.columns)}")

    df['Date'] = pd.to_datetime(df['Date'])
    # 优先用 CSV 已有的相对列（CSV 列白名单未含 Relative_Return，故通常需重算）
    if 'Relative_Return' in df.columns and df['Relative_Return'].notna().any():
        rel = df['Relative_Return']
        align_rate = 1.0
    else:
        hsi_fwd = build_hsi_forward_return(args.horizon)
        _d = df['Date']
        if getattr(_d.dt, 'tz', None) is not None:
            _d = _d.dt.tz_convert(None)
        df['HSI_Future_Return'] = _d.dt.normalize().map(hsi_fwd)
        align_rate = float(df['HSI_Future_Return'].notna().mean())
        rel = df['Actual_Return'] - df['HSI_Future_Return']

    n0 = len(df)
    df = df.assign(Relative_Return=rel)
    df = df.dropna(subset=['Relative_Return', 'Predict_Prob']).copy()
    if df.empty:
        raise RuntimeError('对齐后无有效样本（HSI 覆盖不足？）')
    print(f"HSI 对齐命中率: {align_rate:.1%} | 有效样本: {len(df)}/{n0}（未对齐已丢弃）")

    # 基准：实际跑赢恒指的比例
    base_rate = (df['Relative_Return'] > 0).mean()

    # 1) 方向技能（Predict_Direction 可能是 UP/DOWN 字符串或 0/1）
    if 'Predict_Direction' in df.columns and df['Predict_Direction'].dtype == object:
        pred_dir = df['Predict_Direction'].astype(str).str.upper().map({'UP': 1.0, 'DOWN': 0.0})
    elif 'Predict_Direction' in df.columns:
        pred_dir = pd.to_numeric(df['Predict_Direction'], errors='coerce')
    else:
        pred_dir = (df['Predict_Prob'] > args.gate).astype(float)
    df['Pred_Rel_Dir'] = pred_dir.fillna((df['Predict_Prob'] > args.gate).astype(float))
    acc = (df['Pred_Rel_Dir'] == (df['Relative_Return'] > 0).astype(float)).mean()
    skill = acc - base_rate

    # 1b) ⚠️ 聚类修正：行级 z 值是伪显著，必须按 Fold 聚类重算。
    # 同一折共享同一模型与同一大盘未来收益，跨股票/跨日高度相关，
    # 有效独立观测数≈折数（~50）而非行数（~4 万）。教训见 lessons 三.28。
    naive_se = np.sqrt(base_rate * (1 - base_rate) / max(len(df), 1))
    naive_z = skill / naive_se if naive_se > 0 else float('nan')
    fold_lift, fold_t, fold_p, n_folds = (float('nan'),) * 4
    if 'Fold' in df.columns and df['Fold'].nunique() >= 5:
        per_fold = df.groupby('Fold').apply(
            lambda x: ((x['Pred_Rel_Dir'] == (x['Relative_Return'] > 0).astype(float)).mean()
                       - (x['Relative_Return'] > 0).mean()),
            include_groups=False)
        n_folds = int(len(per_fold))
        fold_lift = float(per_fold.mean())
        fold_se = float(per_fold.std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else float('nan')
        if fold_se > 0:
            fold_t = fold_lift / fold_se
            try:
                from scipy import stats as _st
                fold_p = float(2 * (1 - _st.t.cdf(abs(fold_t), n_folds - 1)))
            except Exception:
                fold_p = float('nan')

    # 1c) 信号对大盘方向的暴露（区分"特异 alpha"与"押大盘 beta"）
    mkt_corr = (float(df['Predict_Prob'].corr(df['HSI_Future_Return']))
                if 'HSI_Future_Return' in df.columns else float('nan'))

    # 2) Rank IC（信息量）
    rank_ic = df['Predict_Prob'].rank().corr(df['Relative_Return'].rank())
    pearson_ic = df['Predict_Prob'].corr(df['Relative_Return'])

    # 3) 市场中性 P&L（只对做多信号计相对收益）
    longs = df[df['Pred_Rel_Dir'] > 0.5]
    if len(longs) == 0:
        net_ir = float('nan'); net_win = float('nan'); n_trades = 0
    else:
        rets = longs['Relative_Return'].values
        n_trades = len(rets)
        net_win = (rets > 0).mean()
        # 按日聚合 → 净 IR（年化，horizon 缩放）
        daily = longs.groupby('Date')['Relative_Return'].mean()
        net_ir = daily.mean() / daily.std() * np.sqrt(252 / max(args.horizon, 1)) if daily.std() > 0 else float('nan')

    print("=" * 66)
    print("相对 alpha 专用校验（Relative_Return 口径）")
    print("=" * 66)
    print(f"输入: {args.input}")
    print(f"horizon: {args.horizon}d | 有效样本: {len(df)}")
    print(f"基准（跑赢恒指比例）: {base_rate:.2%}")
    print(f"方向准确率: {acc:.2%} | 方向技能(lift): {skill:+.2%}")
    print("-" * 66)
    print("⚠️ 显著性（行级 z 是伪显著，必须看聚类结果）")
    print(f"  行级(伪)  : z={naive_z:.1f}   ← 忽略折间相关，不可用于判读")
    if n_folds == n_folds:  # 非 NaN
        verdict = "显著" if (fold_p == fold_p and fold_p < 0.05) else "不显著"
        print(f"  Fold聚类  : {n_folds} 折  lift={fold_lift:+.4f}  t={fold_t:.2f}  p={fold_p:.3f} → {verdict}")
    else:
        print("  Fold聚类  : CSV 无 Fold 列，无法聚类")
    print(f"  corr(Pred_Prob, HSI未来收益) = {mkt_corr:+.4f}  ← 大盘暴露（≈0 才非押大盘）")
    print("-" * 66)
    print(f"Rank IC: {rank_ic:.4f} | Pearson IC: {pearson_ic:.4f}")
    print(f"做多信号数: {n_trades} | 中性净胜率: {net_win:.2%} | 中性净 IR(年化): {net_ir:.3f}")
    print("=" * 66)
    print("判读：聚类 p≥0.05 且 Rank IC≈0 且中性净 IR≤0")
    print("      → 相对口径亦无可辨识 alpha（D1 成立），且不可归因于'信号存在但被成本吃掉'")
    print("=" * 66)


if __name__ == '__main__':
    main()
