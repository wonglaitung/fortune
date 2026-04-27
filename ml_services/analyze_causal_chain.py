#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果链分析：1d→5d 和 5d→20d 的传导关系

分析内容：
1. 连续五天1d预测的"积分"与5d预测的因果关系
2. 连续四次5d预测的"积分"与20d预测的因果关系

核心问题：
- 5个连续1d预测的共识能否预测5d方向？准确率如何？
- 4个连续5d预测的共识能否预测20d方向？准确率如何？
- 预测一致性（全涨/全跌/混合）如何影响准确率？
- 1d/5d预测实际验证结果对后续预测的传导效应

创建时间：2026-04-26
"""

import warnings
import os
import sys
import json
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_services.calendar_features import CalendarFeatureCalculator
from data_services.volatility_model import GARCHVolatilityModel
from data_services.regime_detector import RegimeDetector

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
OUTPUT_DIR = "output"


def fetch_and_prepare_data():
    """获取数据并准备三周期预测（集成新特征）"""
    print("📊 正在获取数据...")
    hsi = yf.Ticker("^HSI")
    df = hsi.history(period="5y", interval="1d")

    print(f"  原始数据: {len(df)} 条")

    # ========== 基础技术指标 ==========
    for window in [20, 60, 120, 250]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

    df['Return_1d'] = df['Close'].pct_change().shift(1)
    df['Return_5d'] = df['Close'].pct_change(5).shift(1)
    df['Return_20d'] = df['Close'].pct_change(20).shift(1)
    df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
    df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema12 - ema26).shift(1)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 布林带
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Middle'] + 2 * bb_std - (df['BB_Middle'] - 2 * bb_std)) / df['BB_Middle']

    # 趋势强度
    df['MA250_Slope'] = df['MA250'].diff()
    df['MA_Bullish_Alignment'] = ((df['MA20'] > df['MA60']) & (df['MA60'] > df['MA250'])).astype(int)
    df['MA_Bearish_Alignment'] = ((df['MA20'] < df['MA60']) & (df['MA60'] < df['MA250'])).astype(int)

    # ========== 宏观数据 ==========
    print("  获取宏观数据...")
    us_df = yf.Ticker("^TNX").history(period="5y", interval="1d")
    vix_df = yf.Ticker("^VIX").history(period="5y", interval="1d")

    if not us_df.empty:
        us_aligned = us_df.reindex(df.index, method='ffill')
        df['US_10Y_Yield'] = us_aligned['Close'] / 10
    if not vix_df.empty:
        vix_aligned = vix_df.reindex(df.index, method='ffill')
        df['VIX'] = vix_aligned['Close']

    # ========== 新增特征 ==========
    print("  计算日历效应特征...")
    calendar_calc = CalendarFeatureCalculator()
    df = calendar_calc.calculate_features(df)

    print("  计算 GARCH 波动率特征...")
    garch_model = GARCHVolatilityModel()
    df = garch_model.calculate_features(df)

    print("  计算市场状态特征...")
    regime_detector = RegimeDetector()
    df = regime_detector.calculate_features(df)

    print(f"  ✅ 特征计算完成: {len(df.columns)} 列")
    return df


def get_feature_list():
    """获取特征列表（基础 + 新增）"""
    features = [
        'MA20', 'MA60', 'MA120', 'MA250',
        'Return_1d', 'Return_5d', 'Return_20d',
        'Volatility_20d', 'Volatility_60d',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Width',
        'MA250_Slope', 'MA_Bullish_Alignment', 'MA_Bearish_Alignment',
        'US_10Y_Yield', 'VIX',
        'Day_of_Week', 'Month_of_Year', 'Days_to_Options_Expiry',
        'Days_to_Holiday', 'Is_Pre_Holiday', 'Is_Typhoon_Season',
        'Month_Sin', 'Month_Cos',
        'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
        'Market_Regime', 'Regime_Prob_0', 'Regime_Prob_1', 'Regime_Prob_2',
    ]
    return features


def run_walkforward_predictions(df):
    """运行Walk-forward获取三周期预测（step=5确保连续日覆盖）"""
    print("\n🔬 运行Walk-forward预测...")

    horizons = [1, 5, 20]
    available_features = [f for f in get_feature_list() if f in df.columns]
    print(f"  可用特征数: {len(available_features)}")

    # 创建目标
    for h in horizons:
        df[f'Target_{h}d'] = (df['Close'].pct_change(h).shift(-h) > 0).astype(int)
        df[f'Future_Return_{h}d'] = df['Close'].pct_change(h).shift(-h)

    # Walk-forward
    train_days = 252
    step_days = 5
    total_days = len(df)
    num_folds = (total_days - train_days) // step_days
    print(f"  总样本: {total_days}, Fold数: {num_folds}")

    all_predictions = {h: [] for h in horizons}

    for fold in range(num_folds):
        train_start = fold * step_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = min(test_start + step_days, total_days)

        if test_start >= total_days:
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        for horizon in horizons:
            cols_needed = available_features + [f'Target_{horizon}d', f'Future_Return_{horizon}d', 'Close']
            train_clean = train_df[cols_needed].dropna()
            test_clean = test_df[cols_needed].dropna()

            if len(train_clean) < 50 or len(test_clean) < 1:
                continue

            X_train = train_clean[available_features]
            y_train = train_clean[f'Target_{horizon}d']

            try:
                model = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=3,
                    random_seed=RANDOM_SEED, auto_class_weights='Balanced', verbose=0
                )
                model.fit(X_train, y_train)

                for idx, row in test_clean.iterrows():
                    X = row[available_features].values.reshape(1, -1)
                    prob = model.predict_proba(X)[0, 1]
                    pred = 1 if prob > 0.5 else 0

                    all_predictions[horizon].append({
                        'date': idx,
                        'prediction': pred,
                        'probability': prob,
                        'actual_direction': row[f'Target_{horizon}d'],
                        'actual_return': row[f'Future_Return_{horizon}d'],
                        'close': row['Close']
                    })
            except Exception:
                continue

        if (fold + 1) % 50 == 0:
            print(f"  已完成 {fold + 1}/{num_folds} folds...")

    return all_predictions


def analyze_1d_to_5d_chain(all_predictions):
    """
    分析 1d→5d 因果链

    对每个有5d预测的日期T，收集T,T+1,...,T+4的1d预测，
    分析5个连续1d预测与5d预测/实际的因果关系。
    """
    print("\n" + "=" * 80)
    print("📊 分析 1d→5d 因果链")
    print("=" * 80)

    df_1d = pd.DataFrame(all_predictions[1])
    df_5d = pd.DataFrame(all_predictions[5])

    if len(df_1d) == 0 or len(df_5d) == 0:
        print("❌ 数据不足")
        return None

    df_1d.set_index('date', inplace=True)
    df_5d.set_index('date', inplace=True)

    # 去重（同一日期可能有多个fold的预测，取最后一个）
    df_1d = df_1d[~df_1d.index.duplicated(keep='last')]
    df_5d = df_5d[~df_5d.index.duplicated(keep='last')]

    df_1d.sort_index(inplace=True)
    df_5d.sort_index(inplace=True)

    results = []
    skipped = 0

    for date in df_5d.index:
        # 找到 date 在 df_1d 中的位置
        if date not in df_1d.index:
            skipped += 1
            continue

        date_idx = df_1d.index.get_loc(date)

        # 需要 T, T+1, T+2, T+3, T+4 的 1d 预测
        if date_idx + 4 >= len(df_1d.index):
            skipped += 1
            continue

        dates_needed = [df_1d.index[date_idx + i] for i in range(5)]

        # 验证日期间距合理（排除节假日导致的非连续）
        gaps = [(dates_needed[i+1] - dates_needed[i]).days for i in range(4)]
        if any(g > 5 for g in gaps):  # 间隔超过5天说明有长假
            skipped += 1
            continue

        # 获取 5d 预测和实际
        pred_5d = int(df_5d.loc[date, 'prediction'])
        actual_5d = int(df_5d.loc[date, 'actual_direction'])
        prob_5d = float(df_5d.loc[date, 'probability'])

        # 获取 5 个连续 1d 预测
        preds_1d = [int(df_1d.loc[d, 'prediction']) for d in dates_needed]
        probs_1d = [float(df_1d.loc[d, 'probability']) for d in dates_needed]
        actuals_1d = [int(df_1d.loc[d, 'actual_direction']) for d in dates_needed]

        # 共识指标
        consensus_count = sum(preds_1d)  # 0-5，多少个预测"涨"
        majority_vote = 1 if consensus_count >= 3 else 0
        pattern = ''.join(str(p) for p in preds_1d)

        # 实际验证
        actual_consensus_count = sum(actuals_1d)
        actual_majority = 1 if actual_consensus_count >= 3 else 0
        correct_1d_count = sum(1 for p, a in zip(preds_1d, actuals_1d) if p == a)

        # 1d预测概率的均值
        avg_prob_1d = np.mean(probs_1d)

        # 1d预测概率的方差（分歧度）
        prob_variance = np.var(probs_1d)

        results.append({
            'date': str(date),
            'pred_5d': pred_5d,
            'actual_5d': actual_5d,
            'prob_5d': prob_5d,
            'preds_1d': preds_1d,
            'probs_1d': probs_1d,
            'actuals_1d': actuals_1d,
            'consensus_count': consensus_count,
            'majority_vote': majority_vote,
            'pattern': pattern,
            'actual_consensus_count': actual_consensus_count,
            'actual_majority': actual_majority,
            'correct_1d_count': correct_1d_count,
            'avg_prob_1d': avg_prob_1d,
            'prob_variance': prob_variance,
            '5d_correct': pred_5d == actual_5d,
            'majority_correct': majority_vote == actual_5d,
        })

    print(f"  有效样本: {len(results)} (跳过: {skipped})")

    if not results:
        return None

    return pd.DataFrame(results)


def analyze_5d_to_20d_chain(all_predictions):
    """
    分析 5d→20d 因果链

    对每个有20d预测的日期T，收集T,T+5,T+10,T+15的5d预测，
    分析4个连续5d预测与20d预测/实际的因果关系。
    """
    print("\n" + "=" * 80)
    print("📊 分析 5d→20d 因果链")
    print("=" * 80)

    df_5d = pd.DataFrame(all_predictions[5])
    df_20d = pd.DataFrame(all_predictions[20])

    if len(df_5d) == 0 or len(df_20d) == 0:
        print("❌ 数据不足")
        return None

    df_5d.set_index('date', inplace=True)
    df_20d.set_index('date', inplace=True)

    df_5d = df_5d[~df_5d.index.duplicated(keep='last')]
    df_20d = df_20d[~df_20d.index.duplicated(keep='last')]

    df_5d.sort_index(inplace=True)
    df_20d.sort_index(inplace=True)

    results = []
    skipped = 0

    for date in df_20d.index:
        if date not in df_5d.index:
            skipped += 1
            continue

        date_idx = df_5d.index.get_loc(date)

        # 需要 T, T+5, T+10, T+15 的 5d 预测
        # 在 DataFrame 中，T+5 对应 index + 5
        required_indices = [date_idx + i * 5 for i in range(4)]
        if any(idx >= len(df_5d.index) for idx in required_indices):
            skipped += 1
            continue

        dates_needed = [df_5d.index[idx] for idx in required_indices]

        # 验证日期间距
        gaps = [(dates_needed[i+1] - dates_needed[i]).days for i in range(3)]
        if any(g > 10 for g in gaps):
            skipped += 1
            continue

        # 获取 20d 预测和实际
        pred_20d = int(df_20d.loc[date, 'prediction'])
        actual_20d = int(df_20d.loc[date, 'actual_direction'])
        prob_20d = float(df_20d.loc[date, 'probability'])

        # 获取 4 个连续 5d 预测
        preds_5d = [int(df_5d.loc[d, 'prediction']) for d in dates_needed]
        probs_5d = [float(df_5d.loc[d, 'probability']) for d in dates_needed]
        actuals_5d = [int(df_5d.loc[d, 'actual_direction']) for d in dates_needed]

        # 共识指标
        consensus_count = sum(preds_5d)
        majority_vote = 1 if consensus_count >= 3 else 0
        pattern = ''.join(str(p) for p in preds_5d)

        # 实际验证
        correct_5d_count = sum(1 for p, a in zip(preds_5d, actuals_5d) if p == a)

        # 5d 预测概率均值和方差
        avg_prob_5d = np.mean(probs_5d)
        prob_variance = np.var(probs_5d)

        results.append({
            'date': str(date),
            'pred_20d': pred_20d,
            'actual_20d': actual_20d,
            'prob_20d': prob_20d,
            'preds_5d': preds_5d,
            'probs_5d': probs_5d,
            'actuals_5d': actuals_5d,
            'consensus_count': consensus_count,
            'majority_vote': majority_vote,
            'pattern': pattern,
            'correct_5d_count': correct_5d_count,
            'avg_prob_5d': avg_prob_5d,
            'prob_variance': prob_variance,
            '20d_correct': pred_20d == actual_20d,
            'majority_correct': majority_vote == actual_20d,
        })

    print(f"  有效样本: {len(results)} (跳过: {skipped})")

    if not results:
        return None

    return pd.DataFrame(results)


def print_1d_5d_analysis(df):
    """打印 1d→5d 因果链分析结果"""
    print("\n" + "=" * 80)
    print("📊 1d→5d 因果链分析结果")
    print("=" * 80)

    n = len(df)
    print(f"\n总样本数: {n}")

    # ========== 1. 基础准确率 ==========
    pred_5d_acc = df['5d_correct'].mean() * 100
    majority_acc = df['majority_correct'].mean() * 100
    print(f"\n【基础准确率】")
    print(f"  5d模型直接预测准确率: {pred_5d_acc:.2f}%")
    print(f"  1d多数投票准确率:      {majority_acc:.2f}%")
    print(f"  差异: {majority_acc - pred_5d_acc:+.2f}%")

    # ========== 2. 按共识一致性分析 ==========
    print(f"\n【按1d预测一致性分析】")
    print(f"  {'一致性':<12} {'样本数':<8} {'占比':<10} {'5d模型准确率':<14} {'1d投票准确率':<14} {'5d实际上涨率':<14}")
    print("  " + "-" * 72)

    for count in range(6):
        mask = df['consensus_count'] == count
        n_group = mask.sum()
        if n_group == 0:
            continue
        pct = n_group / n * 100
        model_acc = df.loc[mask, '5d_correct'].mean() * 100
        vote_acc = df.loc[mask, 'majority_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_5d'].mean() * 100
        label = f"{'↑' * count}{'↓' * (5-count)}"
        print(f"  {label:<12} {n_group:<8} {pct:.1f}%     {model_acc:.2f}%        {vote_acc:.2f}%        {actual_up:.2f}%")

    # ========== 3. 全涨全跌 vs 混合 ==========
    print(f"\n【一致性分组对比】")
    all_up = df['consensus_count'] == 5
    all_down = df['consensus_count'] == 0
    mixed = (df['consensus_count'] >= 1) & (df['consensus_count'] <= 4)
    strong = (df['consensus_count'] >= 4) | (df['consensus_count'] <= 1)

    for label, mask in [('全涨(11111)', all_up), ('全跌(00000)', all_down),
                         ('混合(1-4个涨)', mixed), ('强信号(≤1或≥4)', strong)]:
        n_group = mask.sum()
        if n_group == 0:
            continue
        model_acc = df.loc[mask, '5d_correct'].mean() * 100
        vote_acc = df.loc[mask, 'majority_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_5d'].mean() * 100
        print(f"  {label:<20} n={n_group:<5} 模型:{model_acc:.2f}%  投票:{vote_acc:.2f}%  实际上涨率:{actual_up:.2f}%")

    # ========== 4. 1d预测实际验证的传导效应 ==========
    print(f"\n【1d预测实际验证的传导效应】")
    print(f"  {'1d验证正确数':<15} {'样本数':<8} {'5d模型准确率':<14} {'5d实际上涨率':<14}")
    print("  " + "-" * 52)

    for correct_count in range(6):
        mask = df['correct_1d_count'] == correct_count
        n_group = mask.sum()
        if n_group == 0:
            continue
        model_acc = df.loc[mask, '5d_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_5d'].mean() * 100
        print(f"  {correct_count}/5 正确       {n_group:<8} {model_acc:.2f}%        {actual_up:.2f}%")

    # ========== 5. 概率相关性 ==========
    print(f"\n【概率相关性】")
    corr_avg_prob = df['avg_prob_1d'].corr(df['prob_5d'])
    corr_avg_actual = df['avg_prob_1d'].corr(df['actual_5d'].astype(float))
    corr_5d_actual = df['prob_5d'].corr(df['actual_5d'].astype(float))
    print(f"  1d平均概率 vs 5d模型概率: r = {corr_avg_prob:.4f}")
    print(f"  1d平均概率 vs 5d实际方向: r = {corr_avg_actual:.4f}")
    print(f"  5d模型概率 vs 5d实际方向: r = {corr_5d_actual:.4f}")

    # ========== 6. 关键模式分析 ==========
    print(f"\n【关键模式分析（Top 10 频率）】")
    pattern_stats = []
    for pattern, group in df.groupby('pattern'):
        n_pattern = len(group)
        model_acc = group['5d_correct'].mean() * 100
        actual_up = group['actual_5d'].mean() * 100
        pattern_stats.append((pattern, n_pattern, model_acc, actual_up))

    pattern_stats.sort(key=lambda x: -x[1])

    print(f"  {'模式':<10} {'样本数':<8} {'占比':<8} {'5d模型准确率':<14} {'5d实际上涨率':<14}")
    print("  " + "-" * 56)
    for pattern, n_pattern, model_acc, actual_up in pattern_stats[:10]:
        pct = n_pattern / n * 100
        print(f"  {pattern:<10} {n_pattern:<8} {pct:.1f}%  {model_acc:.2f}%        {actual_up:.2f}%")


def print_5d_20d_analysis(df):
    """打印 5d→20d 因果链分析结果"""
    print("\n" + "=" * 80)
    print("📊 5d→20d 因果链分析结果")
    print("=" * 80)

    n = len(df)
    print(f"\n总样本数: {n}")

    # ========== 1. 基础准确率 ==========
    pred_20d_acc = df['20d_correct'].mean() * 100
    majority_acc = df['majority_correct'].mean() * 100
    print(f"\n【基础准确率】")
    print(f"  20d模型直接预测准确率: {pred_20d_acc:.2f}%")
    print(f"  5d多数投票准确率:       {majority_acc:.2f}%")
    print(f"  差异: {majority_acc - pred_20d_acc:+.2f}%")

    # ========== 2. 按共识一致性分析 ==========
    print(f"\n【按5d预测一致性分析】")
    print(f"  {'一致性':<12} {'样本数':<8} {'占比':<10} {'20d模型准确率':<14} {'5d投票准确率':<14} {'20d实际上涨率':<14}")
    print("  " + "-" * 72)

    for count in range(5):
        mask = df['consensus_count'] == count
        n_group = mask.sum()
        if n_group == 0:
            continue
        pct = n_group / n * 100
        model_acc = df.loc[mask, '20d_correct'].mean() * 100
        vote_acc = df.loc[mask, 'majority_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_20d'].mean() * 100
        label = f"{'↑' * count}{'↓' * (4-count)}"
        print(f"  {label:<12} {n_group:<8} {pct:.1f}%     {model_acc:.2f}%        {vote_acc:.2f}%        {actual_up:.2f}%")

    # ========== 3. 全涨全跌 vs 混合 ==========
    print(f"\n【一致性分组对比】")
    all_up = df['consensus_count'] == 4
    all_down = df['consensus_count'] == 0
    mixed = (df['consensus_count'] >= 1) & (df['consensus_count'] <= 3)
    strong = (df['consensus_count'] >= 3) | (df['consensus_count'] <= 1)

    for label, mask in [('全涨(1111)', all_up), ('全跌(0000)', all_down),
                         ('混合(1-3个涨)', mixed), ('强信号(≤1或≥3)', strong)]:
        n_group = mask.sum()
        if n_group == 0:
            continue
        model_acc = df.loc[mask, '20d_correct'].mean() * 100
        vote_acc = df.loc[mask, 'majority_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_20d'].mean() * 100
        print(f"  {label:<20} n={n_group:<5} 模型:{model_acc:.2f}%  投票:{vote_acc:.2f}%  实际上涨率:{actual_up:.2f}%")

    # ========== 4. 5d预测实际验证的传导效应 ==========
    print(f"\n【5d预测实际验证的传导效应】")
    print(f"  {'5d验证正确数':<15} {'样本数':<8} {'20d模型准确率':<14} {'20d实际上涨率':<14}")
    print("  " + "-" * 52)

    for correct_count in range(5):
        mask = df['correct_5d_count'] == correct_count
        n_group = mask.sum()
        if n_group == 0:
            continue
        model_acc = df.loc[mask, '20d_correct'].mean() * 100
        actual_up = df.loc[mask, 'actual_20d'].mean() * 100
        print(f"  {correct_count}/4 正确       {n_group:<8} {model_acc:.2f}%        {actual_up:.2f}%")

    # ========== 5. 概率相关性 ==========
    print(f"\n【概率相关性】")
    corr_avg_prob = df['avg_prob_5d'].corr(df['prob_20d'])
    corr_avg_actual = df['avg_prob_5d'].corr(df['actual_20d'].astype(float))
    corr_20d_actual = df['prob_20d'].corr(df['actual_20d'].astype(float))
    print(f"  5d平均概率 vs 20d模型概率: r = {corr_avg_prob:.4f}")
    print(f"  5d平均概率 vs 20d实际方向: r = {corr_avg_actual:.4f}")
    print(f"  20d模型概率 vs 20d实际方向: r = {corr_20d_actual:.4f}")

    # ========== 6. 关键模式分析 ==========
    print(f"\n【关键模式分析（全部16种模式）】")
    pattern_stats = []
    for pattern, group in df.groupby('pattern'):
        n_pattern = len(group)
        model_acc = group['20d_correct'].mean() * 100
        vote_acc = group['majority_correct'].mean() * 100
        actual_up = group['actual_20d'].mean() * 100
        pattern_stats.append((pattern, n_pattern, model_acc, vote_acc, actual_up))

    pattern_stats.sort(key=lambda x: -x[1])

    print(f"  {'模式':<10} {'样本数':<8} {'占比':<8} {'20d模型准确率':<14} {'5d投票准确率':<14} {'20d实际上涨率':<14}")
    print("  " + "-" * 68)
    for pattern, n_pattern, model_acc, vote_acc, actual_up in pattern_stats:
        pct = n_pattern / n * 100
        print(f"  {pattern:<10} {n_pattern:<8} {pct:.1f}%  {model_acc:.2f}%        {vote_acc:.2f}%        {actual_up:.2f}%")


def print_comparison(df_1d_5d, df_5d_20d):
    """打印对比总结"""
    print("\n" + "=" * 80)
    print("📊 因果链对比总结")
    print("=" * 80)

    print(f"\n{'指标':<30} {'1d→5d 链':<20} {'5d→20d 链':<20}")
    print("-" * 70)

    # 样本数
    print(f"{'样本数':<30} {len(df_1d_5d):<20} {len(df_5d_20d):<20}")

    # 模型直接准确率
    model_acc_1d5d = df_1d_5d['5d_correct'].mean() * 100
    model_acc_5d20d = df_5d_20d['20d_correct'].mean() * 100
    print(f"{'模型直接预测准确率':<30} {model_acc_1d5d:.2f}%{'':<14} {model_acc_5d20d:.2f}%")

    # 多数投票准确率
    vote_acc_1d5d = df_1d_5d['majority_correct'].mean() * 100
    vote_acc_5d20d = df_5d_20d['majority_correct'].mean() * 100
    print(f"{'多数投票准确率':<30} {vote_acc_1d5d:.2f}%{'':<14} {vote_acc_5d20d:.2f}%")

    # 一致信号准确率
    if df_1d_5d['consensus_count'].eq(5).sum() > 0:
        all_up_1d5d = df_1d_5d.loc[df_1d_5d['consensus_count'] == 5, '5d_correct'].mean() * 100
    else:
        all_up_1d5d = 0
    if df_5d_20d['consensus_count'].eq(4).sum() > 0:
        all_up_5d20d = df_5d_20d.loc[df_5d_20d['consensus_count'] == 4, '20d_correct'].mean() * 100
    else:
        all_up_5d20d = 0
    print(f"{'全涨一致准确率':<30} {all_up_1d5d:.2f}%{'':<14} {all_up_5d20d:.2f}%")

    if df_1d_5d['consensus_count'].eq(0).sum() > 0:
        all_down_1d5d = df_1d_5d.loc[df_1d_5d['consensus_count'] == 0, '5d_correct'].mean() * 100
    else:
        all_down_1d5d = 0
    if df_5d_20d['consensus_count'].eq(0).sum() > 0:
        all_down_5d20d = df_5d_20d.loc[df_5d_20d['consensus_count'] == 0, '20d_correct'].mean() * 100
    else:
        all_down_5d20d = 0
    print(f"{'全跌一致准确率':<30} {all_down_1d5d:.2f}%{'':<14} {all_down_5d20d:.2f}%")

    # 传导效应：子预测正确数对目标准确率的影响
    print(f"\n{'传导效应（子预测正确→目标准确率）':<30}")
    print(f"{'-'*70}")

    # 1d→5d
    corr_1d5d = df_1d_5d['correct_1d_count'].corr(df_1d_5d['5d_correct'].astype(float))
    corr_5d20d = df_5d_20d['correct_5d_count'].corr(df_5d_20d['20d_correct'].astype(float))
    print(f"{'子预测正确数 vs 目标准确率 r':<30} {corr_1d5d:.4f}{'':<15} {corr_5d20d:.4f}")

    # 概率传导
    corr_prob_1d5d = df_1d_5d['avg_prob_1d'].corr(df_1d_5d['prob_5d'])
    corr_prob_5d20d = df_5d_20d['avg_prob_5d'].corr(df_5d_20d['prob_20d'])
    print(f"{'子预测概率均值 vs 目标概率 r':<30} {corr_prob_1d5d:.4f}{'':<15} {corr_prob_5d20d:.4f}")

    # ========== 核心结论 ==========
    print(f"\n{'='*80}")
    print("💡 核心结论")
    print(f"{'='*80}")

    # 结论 1：哪条链的传导更强
    if abs(corr_5d20d) > abs(corr_1d5d):
        print(f"\n  1. 5d→20d 传导链更强（r={corr_5d20d:.4f} > r={corr_1d5d:.4f}）")
        print(f"     → 4个5d预测的积分对20d预测更有因果解释力")
        print(f"     → 1d预测噪声太大，5个1d预测的积分对5d预测解释力弱")
    else:
        print(f"\n  1. 1d→5d 传导链更强（r={corr_1d5d:.4f} > r={corr_5d20d:.4f}）")

    # 结论 2：一致信号的强度
    strong_1d5d = df_1d_5d[(df_1d_5d['consensus_count'] >= 4) | (df_1d_5d['consensus_count'] <= 1)]
    mixed_1d5d = df_1d_5d[(df_1d_5d['consensus_count'] >= 2) & (df_1d_5d['consensus_count'] <= 3)]
    if len(strong_1d5d) > 0 and len(mixed_1d5d) > 0:
        strong_acc = strong_1d5d['5d_correct'].mean() * 100
        mixed_acc = mixed_1d5d['5d_correct'].mean() * 100
        print(f"\n  2. 1d→5d链: 强信号准确率 {strong_acc:.2f}% vs 混合信号 {mixed_acc:.2f}%")
        if strong_acc > mixed_acc:
            print(f"     → 1d预测一致性强时，5d预测更可靠")

    strong_5d20d = df_5d_20d[(df_5d_20d['consensus_count'] >= 3) | (df_5d_20d['consensus_count'] <= 1)]
    mixed_5d20d = df_5d_20d[(df_5d_20d['consensus_count'] == 2)]
    if len(strong_5d20d) > 0 and len(mixed_5d20d) > 0:
        strong_acc = strong_5d20d['20d_correct'].mean() * 100
        mixed_acc = mixed_5d20d['20d_correct'].mean() * 100
        print(f"\n  3. 5d→20d链: 强信号准确率 {strong_acc:.2f}% vs 混合信号 {mixed_acc:.2f}%")
        if strong_acc > mixed_acc:
            print(f"     → 5d预测一致性强时，20d预测更可靠")

    # 结论 3：多数投票 vs 模型直接预测
    print(f"\n  4. 多数投票 vs 模型直接预测:")
    print(f"     1d→5d: 投票 {vote_acc_1d5d:.2f}% vs 模型 {model_acc_1d5d:.2f}%")
    print(f"     5d→20d: 投票 {vote_acc_5d20d:.2f}% vs 模型 {model_acc_5d20d:.2f}%")
    if model_acc_1d5d > vote_acc_1d5d:
        print(f"     → 模型直接预测优于子预测积分")
    else:
        print(f"     → 子预测积分优于模型直接预测（子预测包含额外信息）")


def save_results(df_1d_5d, df_5d_20d):
    """保存分析结果到JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_type': 'causal_chain_1d5d_5d20d',
    }

    # 1d→5d 汇总
    if df_1d_5d is not None:
        results['1d_to_5d'] = {
            'total_samples': len(df_1d_5d),
            'model_accuracy': float(df_1d_5d['5d_correct'].mean() * 100),
            'majority_vote_accuracy': float(df_1d_5d['majority_correct'].mean() * 100),
            'consistency_analysis': {},
            'transmission_analysis': {},
            'pattern_analysis': {},
        }

        for count in range(6):
            mask = df_1d_5d['consensus_count'] == count
            if mask.sum() > 0:
                results['1d_to_5d']['consistency_analysis'][str(count)] = {
                    'count': int(mask.sum()),
                    'model_accuracy': float(df_1d_5d.loc[mask, '5d_correct'].mean() * 100),
                    'actual_up_rate': float(df_1d_5d.loc[mask, 'actual_5d'].mean() * 100),
                }

        for correct in range(6):
            mask = df_1d_5d['correct_1d_count'] == correct
            if mask.sum() > 0:
                results['1d_to_5d']['transmission_analysis'][str(correct)] = {
                    'count': int(mask.sum()),
                    '5d_model_accuracy': float(df_1d_5d.loc[mask, '5d_correct'].mean() * 100),
                }

        for pattern, group in df_1d_5d.groupby('pattern'):
            results['1d_to_5d']['pattern_analysis'][pattern] = {
                'count': int(len(group)),
                '5d_model_accuracy': float(group['5d_correct'].mean() * 100),
                'actual_up_rate': float(group['actual_5d'].mean() * 100),
            }

    # 5d→20d 汇总
    if df_5d_20d is not None:
        results['5d_to_20d'] = {
            'total_samples': len(df_5d_20d),
            'model_accuracy': float(df_5d_20d['20d_correct'].mean() * 100),
            'majority_vote_accuracy': float(df_5d_20d['majority_correct'].mean() * 100),
            'consistency_analysis': {},
            'transmission_analysis': {},
            'pattern_analysis': {},
        }

        for count in range(5):
            mask = df_5d_20d['consensus_count'] == count
            if mask.sum() > 0:
                results['5d_to_20d']['consistency_analysis'][str(count)] = {
                    'count': int(mask.sum()),
                    'model_accuracy': float(df_5d_20d.loc[mask, '20d_correct'].mean() * 100),
                    'actual_up_rate': float(df_5d_20d.loc[mask, 'actual_20d'].mean() * 100),
                }

        for correct in range(5):
            mask = df_5d_20d['correct_5d_count'] == correct
            if mask.sum() > 0:
                results['5d_to_20d']['transmission_analysis'][str(correct)] = {
                    'count': int(mask.sum()),
                    '20d_model_accuracy': float(df_5d_20d.loc[mask, '20d_correct'].mean() * 100),
                }

        for pattern, group in df_5d_20d.groupby('pattern'):
            results['5d_to_20d']['pattern_analysis'][pattern] = {
                'count': int(len(group)),
                '20d_model_accuracy': float(group['20d_correct'].mean() * 100),
                'actual_up_rate': float(group['actual_20d'].mean() * 100),
            }

    output_path = os.path.join(OUTPUT_DIR, 'causal_chain_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("因果链分析：1d→5d 和 5d→20d 的传导关系")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取数据
    df = fetch_and_prepare_data()

    # 2. 运行Walk-forward预测
    all_predictions = run_walkforward_predictions(df)

    # 3. 分析 1d→5d 因果链
    df_1d_5d = analyze_1d_to_5d_chain(all_predictions)

    # 4. 分析 5d→20d 因果链
    df_5d_20d = analyze_5d_to_20d_chain(all_predictions)

    # 5. 打印结果
    if df_1d_5d is not None:
        print_1d_5d_analysis(df_1d_5d)

    if df_5d_20d is not None:
        print_5d_20d_analysis(df_5d_20d)

    if df_1d_5d is not None and df_5d_20d is not None:
        print_comparison(df_1d_5d, df_5d_20d)

    # 6. 保存结果
    save_results(df_1d_5d, df_5d_20d)

    print("\n" + "=" * 80)
    print(f"分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
