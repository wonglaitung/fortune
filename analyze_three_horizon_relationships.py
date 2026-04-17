#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三周期预测关系分析

分析1天、5天、20天三个预测周期之间的：
1. 时间序列关系（先后顺序）
2. 因果关系（一个周期预测如何影响其他周期）
3. 规律模式（一致性、传导性、均值回归等）
"""

import warnings
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
RANDOM_SEED = 42


def fetch_and_prepare_data():
    """获取数据并准备三周期预测"""
    print("📊 正在获取数据...")
    hsi = yf.Ticker("^HSI")
    df = hsi.history(period="max", interval="1d")

    # 计算特征
    for window in [20, 60, 120, 250]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

    df['Return_1d'] = df['Close'].pct_change().shift(1)
    df['Return_5d'] = df['Close'].pct_change(5).shift(1)
    df['Return_20d'] = df['Close'].pct_change(20).shift(1)
    df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
    df['RSI'] = df['Close'].rolling(14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean())))
    ).shift(1)

    # 获取宏观数据
    us_df = yf.Ticker("^TNX").history(period="max", interval="1d")
    vix_df = yf.Ticker("^VIX").history(period="max", interval="1d")

    if not us_df.empty:
        us_aligned = us_df.reindex(df.index, method='ffill')
        df['US_10Y_Yield'] = us_aligned['Close'] / 10
    if not vix_df.empty:
        vix_aligned = vix_df.reindex(df.index, method='ffill')
        df['VIX'] = vix_aligned['Close']

    # 筛选2020-2024年
    df = df[(df.index >= '2020-01-01') & (df.index <= '2024-12-31')]
    df = df.iloc[:-20]  # 预留未来数据

    return df


def run_walkforward_predictions(df):
    """运行Walk-forward获取三周期预测"""
    print("\n🔬 运行Walk-forward预测...")

    horizons = [1, 5, 20]
    features = ['MA20', 'MA60', 'MA120', 'MA250',
                'Return_1d', 'Return_5d', 'Return_20d',
                'Volatility_20d', 'RSI']
    if 'US_10Y_Yield' in df.columns:
        features.extend(['US_10Y_Yield', 'VIX'])

    available_features = [f for f in features if f in df.columns]

    # 创建目标
    for h in horizons:
        df[f'Target_{h}d'] = (df['Close'].pct_change(h).shift(-h) > 0).astype(int)
        df[f'Future_Return_{h}d'] = df['Close'].pct_change(h).shift(-h)

    # Walk-forward
    train_days = 252
    step_days = 5
    total_days = len(df)
    num_folds = (total_days - train_days) // step_days

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
            train_clean = train_df[available_features + [f'Target_{horizon}d', f'Future_Return_{horizon}d', 'Close']].dropna()
            test_clean = test_df[available_features + [f'Target_{horizon}d', f'Future_Return_{horizon}d', 'Close']].dropna()

            if len(train_clean) < 30 or len(test_clean) < 1:
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
            except:
                continue

    return all_predictions


def analyze_temporal_relationships(all_predictions):
    """分析时间关系"""
    print("\n" + "=" * 80)
    print("⏰ 一、时间关系分析")
    print("=" * 80)

    # 构建DataFrame
    df_1d = pd.DataFrame(all_predictions[1])
    df_5d = pd.DataFrame(all_predictions[5])
    df_20d = pd.DataFrame(all_predictions[20])

    if len(df_1d) == 0 or len(df_5d) == 0 or len(df_20d) == 0:
        print("数据不足")
        return

    df_1d.set_index('date', inplace=True)
    df_5d.set_index('date', inplace=True)
    df_20d.set_index('date', inplace=True)

    print("\n1. 时间范围关系")
    print(f"   1天预测覆盖: {df_1d.index[0].strftime('%Y-%m-%d')} ~ {df_1d.index[-1].strftime('%Y-%m-%d')}")
    print(f"   5天预测覆盖: {df_5d.index[0].strftime('%Y-%m-%d')} ~ {df_5d.index[-1].strftime('%Y-%m-%d')}")
    print(f"   20天预测覆盖: {df_20d.index[0].strftime('%Y-%m-%d')} ~ {df_20d.index[-1].strftime('%Y-%m-%d')}")

    print("\n2. 持有期重叠分析")
    print("   假设第T日发出预测:")
    print("   - 1天周期: 预测 T+1 日涨跌, 持有期 [T, T+1]")
    print("   - 5天周期: 预测 T+5 日涨跌, 持有期 [T, T+5]")
    print("   - 20天周期: 预测 T+20 日涨跌, 持有期 [T, T+20]")
    print("\n   时间重叠:")
    print("   - 1天与5天: [T, T+1] 完全包含于 [T, T+5] (前20%)")
    print("   - 1天与20天: [T, T+1] 完全包含于 [T, T+20] (前5%)")
    print("   - 5天与20天: [T, T+5] 完全包含于 [T, T+20] (前25%)")

    return df_1d, df_5d, df_20d


def analyze_causal_relationships(df_1d, df_5d, df_20d):
    """分析因果关系"""
    print("\n" + "=" * 80)
    print("🔗 二、因果关系分析")
    print("=" * 80)

    # 对齐数据
    common_dates = df_1d.index.intersection(df_5d.index).intersection(df_20d.index)

    print(f"\n共同样本数: {len(common_dates)}")

    # 提取预测和实际
    pred_1d = df_1d.loc[common_dates, 'prediction']
    pred_5d = df_5d.loc[common_dates, 'prediction']
    pred_20d = df_20d.loc[common_dates, 'prediction']

    actual_1d = df_1d.loc[common_dates, 'actual_direction']
    actual_5d = df_5d.loc[common_dates, 'actual_direction']
    actual_20d = df_20d.loc[common_dates, 'actual_direction']

    print("\n1. 预测方向的领先-滞后关系")

    # 分析：1天预测准确时，5天和20天的表现
    correct_1d = actual_1d == pred_1d
    print(f"\n   当1天预测正确时 (n={correct_1d.sum()}):")
    print(f"   - 5天预测准确率: {(pred_5d[correct_1d] == actual_5d[correct_1d]).mean()*100:.2f}%")
    print(f"   - 20天预测准确率: {(pred_20d[correct_1d] == actual_20d[correct_1d]).mean()*100:.2f}%")

    incorrect_1d = ~correct_1d
    print(f"\n   当1天预测错误时 (n={incorrect_1d.sum()}):")
    print(f"   - 5天预测准确率: {(pred_5d[incorrect_1d] == actual_5d[incorrect_1d]).mean()*100:.2f}%")
    print(f"   - 20天预测准确率: {(pred_20d[incorrect_1d] == actual_20d[incorrect_1d]).mean()*100:.2f}%")

    # 分析：5天预测准确时
    correct_5d = actual_5d == pred_5d
    print(f"\n   当5天预测正确时 (n={correct_5d.sum()}):")
    print(f"   - 20天预测准确率: {(pred_20d[correct_5d] == actual_20d[correct_5d]).mean()*100:.2f}%")

    print("\n2. 信号传导分析")

    # 三周期一致时的后续表现
    consensus_bull = (pred_1d == 1) & (pred_5d == 1) & (pred_20d == 1)
    consensus_bear = (pred_1d == 0) & (pred_5d == 0) & (pred_20d == 0)

    print(f"\n   一致看涨后 (n={consensus_bull.sum()}):")
    print(f"   - 1天实际上涨率: {actual_1d[consensus_bull].mean()*100:.2f}%")
    print(f"   - 5天实际上涨率: {actual_5d[consensus_bull].mean()*100:.2f}%")
    print(f"   - 20天实际上涨率: {actual_20d[consensus_bull].mean()*100:.2f}%")

    print(f"\n   一致看跌后 (n={consensus_bear.sum()}):")
    print(f"   - 1天实际下跌率: {(1-actual_1d[consensus_bear]).mean()*100:.2f}%")
    print(f"   - 5天实际下跌率: {(1-actual_5d[consensus_bear]).mean()*100:.2f}%")
    print(f"   - 20天实际下跌率: {(1-actual_20d[consensus_bear]).mean()*100:.2f}%")

    print("\n3. 概率值的因果关系")

    # 分析概率值的相关性
    prob_1d = df_1d.loc[common_dates, 'probability']
    prob_5d = df_5d.loc[common_dates, 'probability']
    prob_20d = df_20d.loc[common_dates, 'probability']

    print(f"\n   概率值相关性:")
    print(f"   - 1天 vs 5天: r = {prob_1d.corr(prob_5d):.4f}")
    print(f"   - 1天 vs 20天: r = {prob_1d.corr(prob_20d):.4f}")
    print(f"   - 5天 vs 20天: r = {prob_5d.corr(prob_20d):.4f}")


def analyze_patterns(df_1d, df_5d, df_20d):
    """分析规律模式"""
    print("\n" + "=" * 80)
    print("📊 三、规律模式分析")
    print("=" * 80)

    common_dates = df_1d.index.intersection(df_5d.index).intersection(df_20d.index)

    pred_1d = df_1d.loc[common_dates, 'prediction']
    pred_5d = df_5d.loc[common_dates, 'prediction']
    pred_20d = df_20d.loc[common_dates, 'prediction']

    actual_1d = df_1d.loc[common_dates, 'actual_direction']
    actual_5d = df_5d.loc[common_dates, 'actual_direction']
    actual_20d = df_20d.loc[common_dates, 'actual_direction']

    return_1d = df_1d.loc[common_dates, 'actual_return']
    return_5d = df_5d.loc[common_dates, 'actual_return']
    return_20d = df_20d.loc[common_dates, 'actual_return']

    print("\n1. 一致性模式分析")

    # 所有可能的组合
    patterns = {
        '涨-涨-涨 (111)': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 1)),
        '涨-涨-跌 (110)': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 0)),
        '涨-跌-涨 (101)': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 1)),
        '涨-跌-跌 (100)': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 0)),
        '跌-涨-涨 (011)': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 1)),
        '跌-涨-跌 (010)': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 0)),
        '跌-跌-涨 (001)': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 1)),
        '跌-跌-跌 (000)': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 0)),
    }

    print(f"\n   {'模式':<20} {'样本数':<10} {'占比':<10} {'1天胜率':<10} {'5天胜率':<10} {'20天胜率':<10}")
    print("   " + "-" * 70)

    for pattern_name, mask in patterns.items():
        count = mask.sum()
        if count == 0:
            continue
        pct = count / len(common_dates) * 100
        acc_1d = (pred_1d[mask] == actual_1d[mask]).mean() * 100
        acc_5d = (pred_5d[mask] == actual_5d[mask]).mean() * 100
        acc_20d = (pred_20d[mask] == actual_20d[mask]).mean() * 100
        print(f"   {pattern_name:<20} {count:<10} {pct:<10.2f}% {acc_1d:<10.2f}% {acc_5d:<10.2f}% {acc_20d:<10.2f}%")

    print("\n2. 均值回归与趋势延续")

    # 短期预测 vs 长期实际
    short_bull_long_bull = ((pred_1d == 1) & (actual_20d == 1)).sum()
    short_bull_long_bear = ((pred_1d == 1) & (actual_20d == 0)).sum()
    short_bear_long_bull = ((pred_1d == 0) & (actual_20d == 1)).sum()
    short_bear_long_bear = ((pred_1d == 0) & (actual_20d == 0)).sum()

    print(f"\n   短期看涨 -> 长期实际上涨: {short_bull_long_bull} ({short_bull_long_bull/(short_bull_long_bull+short_bull_long_bear)*100:.1f}%)")
    print(f"   短期看涨 -> 长期实际下跌: {short_bull_long_bear} ({short_bull_long_bear/(short_bull_long_bull+short_bull_long_bear)*100:.1f}%)")
    print(f"   短期看跌 -> 长期实际上涨: {short_bear_long_bull} ({short_bear_long_bull/(short_bear_long_bull+short_bear_long_bear)*100:.1f}%)")
    print(f"   短期看跌 -> 长期实际下跌: {short_bear_long_bear} ({short_bear_long_bear/(short_bear_long_bull+short_bear_long_bear)*100:.1f}%)")

    print("\n3. 收益分布规律")

    print(f"\n   {'周期':<10} {'平均收益':<12} {'收益标准差':<12} {'最大收益':<12} {'最小收益':<12}")
    print("   " + "-" * 60)
    for horizon, ret in [('1天', return_1d), ('5天', return_5d), ('20天', return_20d)]:
        print(f"   {horizon:<10} {ret.mean()*100:<12.3f}% {ret.std()*100:<12.3f}% {ret.max()*100:<12.3f}% {ret.min()*100:<12.3f}%")

    print("\n   一致看涨时的收益:")
    mask = (pred_1d == 1) & (pred_5d == 1) & (pred_20d == 1)
    if mask.sum() > 0:
        print(f"   - 1天平均收益: {return_1d[mask].mean()*100:.3f}% (标准差: {return_1d[mask].std()*100:.3f}%)")
        print(f"   - 5天平均收益: {return_5d[mask].mean()*100:.3f}% (标准差: {return_5d[mask].std()*100:.3f}%)")
        print(f"   - 20天平均收益: {return_20d[mask].mean()*100:.3f}% (标准差: {return_20d[mask].std()*100:.3f}%)")

    print("\n   一致看跌时的收益:")
    mask = (pred_1d == 0) & (pred_5d == 0) & (pred_20d == 0)
    if mask.sum() > 0:
        print(f"   - 1天平均收益: {return_1d[mask].mean()*100:.3f}% (标准差: {return_1d[mask].std()*100:.3f}%)")
        print(f"   - 5天平均收益: {return_5d[mask].mean()*100:.3f}% (标准差: {return_5d[mask].std()*100:.3f}%)")
        print(f"   - 20天平均收益: {return_20d[mask].mean()*100:.3f}% (标准差: {return_20d[mask].std()*100:.3f}%)")


def analyze_sequence_patterns(df_1d, df_5d, df_20d):
    """分析序列模式 - 一个周期正确后其他周期的表现"""
    print("\n" + "=" * 80)
    print("🔄 四、序列动态分析")
    print("=" * 80)

    common_dates = df_1d.index.intersection(df_5d.index).intersection(df_20d.index)

    pred_1d = df_1d.loc[common_dates, 'prediction']
    pred_5d = df_5d.loc[common_dates, 'prediction']
    pred_20d = df_20d.loc[common_dates, 'prediction']

    actual_1d = df_1d.loc[common_dates, 'actual_direction']
    actual_5d = df_5d.loc[common_dates, 'actual_direction']
    actual_20d = df_20d.loc[common_dates, 'actual_direction']

    print("\n1. 预测正确序列")

    # 1天正确 -> 5天表现
    correct_1d = pred_1d == actual_1d
    print(f"\n   1天预测正确后，5天预测:")
    print(f"   - 5天也正确: {(correct_1d & (pred_5d == actual_5d)).sum()} ({(correct_1d & (pred_5d == actual_5d)).sum()/correct_1d.sum()*100:.1f}%)")
    print(f"   - 5天错误: {(correct_1d & (pred_5d != actual_5d)).sum()} ({(correct_1d & (pred_5d != actual_5d)).sum()/correct_1d.sum()*100:.1f}%)")

    # 5天正确 -> 20天表现
    correct_5d = pred_5d == actual_5d
    print(f"\n   5天预测正确后，20天预测:")
    print(f"   - 20天也正确: {(correct_5d & (pred_20d == actual_20d)).sum()} ({(correct_5d & (pred_20d == actual_20d)).sum()/correct_5d.sum()*100:.1f}%)")
    print(f"   - 20天错误: {(correct_5d & (pred_20d != actual_20d)).sum()} ({(correct_5d & (pred_20d != actual_20d)).sum()/correct_5d.sum()*100:.1f}%)")

    print("\n2. 连续正确链")

    # 三周期全对
    all_correct = (pred_1d == actual_1d) & (pred_5d == actual_5d) & (pred_20d == actual_20d)
    print(f"\n   三周期全对: {all_correct.sum()}次 ({all_correct.sum()/len(common_dates)*100:.2f}%)")

    # 至少一对
    at_least_one = (pred_1d == actual_1d) | (pred_5d == actual_5d) | (pred_20d == actual_20d)
    print(f"   至少一对: {at_least_one.sum()}次 ({at_least_one.sum()/len(common_dates)*100:.2f}%)")

    print("\n3. 周期间一致性动态")

    # 1天和5天一致时的20天表现
    agree_1d_5d = pred_1d == pred_5d
    print(f"\n   1天与5天预测一致时 (n={agree_1d_5d.sum()}):")
    print(f"   - 20天预测与它们一致: {(agree_1d_5d & (pred_20d == pred_1d)).sum()} ({(agree_1d_5d & (pred_20d == pred_1d)).sum()/agree_1d_5d.sum()*100:.1f}%)")
    print(f"   - 20天预测与它们相反: {(agree_1d_5d & (pred_20d != pred_1d)).sum()} ({(agree_1d_5d & (pred_20d != pred_1d)).sum()/agree_1d_5d.sum()*100:.1f}%)")

    # 当1天和5天一致且正确时
    agree_correct = agree_1d_5d & (pred_1d == actual_1d) & (pred_5d == actual_5d)
    if agree_correct.sum() > 0:
        print(f"\n   1天与5天一致且都正确时 (n={agree_correct.sum()}):")
        print(f"   - 20天也正确: {(agree_correct & (pred_20d == actual_20d)).sum()} ({(agree_correct & (pred_20d == actual_20d)).sum()/agree_correct.sum()*100:.1f}%)")


def main():
    """主函数"""
    print("=" * 80)
    print("三周期预测关系深度分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取数据
    df = fetch_and_prepare_data()

    # 2. 运行预测
    all_predictions = run_walkforward_predictions(df)

    # 3. 分析时间关系
    df_1d, df_5d, df_20d = analyze_temporal_relationships(all_predictions)

    # 4. 分析因果关系
    analyze_causal_relationships(df_1d, df_5d, df_20d)

    # 5. 分析规律模式
    analyze_patterns(df_1d, df_5d, df_20d)

    # 6. 分析序列模式
    analyze_sequence_patterns(df_1d, df_5d, df_20d)

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
