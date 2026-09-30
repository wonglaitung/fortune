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
    # 对齐 HSI 未来收益（按日期）
    hsi_fwd = build_hsi_forward_return(args.horizon)
    df['HSI_Future_Return'] = df['Date'].dt.tz_convert(None).dt.normalize().map(hsi_fwd)
    df['Relative_Return'] = df['Actual_Return'] - df['HSI_Future_Return']

    df = df.dropna(subset=['Relative_Return', 'Predict_Prob']).copy()
    if df.empty:
        raise RuntimeError('对齐后无有效样本')

    # 基准：实际跑赢恒指的比例
    base_rate = (df['Relative_Return'] > 0).mean()

    # 1) 方向技能
    if 'Predict_Direction' in df.columns:
        pred_dir = df['Predict_Direction'].astype(float)
    else:
        pred_dir = (df['Predict_Prob'] > args.gate).astype(float)
    df['Pred_Rel_Dir'] = pred_dir
    acc = (df['Pred_Rel_Dir'] == (df['Relative_Return'] > 0).astype(float)).mean()
    skill = acc - base_rate

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

    print("=" * 60)
    print("相对 alpha 专用校验（Relative_Return 口径）")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"horizon: {args.horizon}d | 有效样本: {len(df)}")
    print(f"基准（跑赢恒指比例）: {base_rate:.2%}")
    print(f"方向准确率: {acc:.2%} | 方向技能(lift): {skill:+.2%}")
    print(f"Rank IC: {rank_ic:.4f} | Pearson IC: {pearson_ic:.4f}")
    print(f"做多信号数: {n_trades} | 中性净胜率: {net_win:.2%} | 中性净 IR(年化): {net_ir:.3f}")
    print("=" * 60)
    print("判读：若 Rank IC≈0 且中性净 IR≤0 → 相对口径也无 edge（D1 在相对标签下成立）")
    print("=" * 60)


if __name__ == '__main__':
    main()
