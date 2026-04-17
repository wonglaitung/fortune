#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证"三周期一致预测策略"的有效性 - V2

使用与生产环境一致的Walk-forward验证方法
特征集与hsi_walk_forward.py保持一致

验证目标：
- 一致看涨占比约17.4%，至少一周期正确率约92%
- 一致看跌占比约24.0%，至少一周期正确率约87%
"""

import warnings
import os
import sys
import argparse
from datetime import datetime
import json
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf

# ========== 配置 ==========
HSI_SYMBOL = "^HSI"
OUTPUT_DIR = "output"
RANDOM_SEED = 42

# 特征配置（与hsi_walk_forward.py保持一致）
FEATURE_CONFIG = {
    'macro_features': ['US_10Y_Yield', 'US_10Y_Yield_Change_5d', 'VIX', 'VIX_Change_5d'],
    'ma_features': ['MA20', 'MA60', 'MA120', 'MA250', 'Volume_MA250'],
    'volatility_features': ['Volatility_120d', 'Volatility_20d', 'Volatility_60d'],
    'rsi_features': ['RSI'],
    'return_features': ['Return_1d', 'Return_5d', 'Return_20d'],
}


def fetch_data():
    """获取数据"""
    print("📊 正在获取数据...")

    # 获取恒指数据（使用更长的历史数据以匹配lessons.md中的样本量）
    hsi = yf.Ticker(HSI_SYMBOL)
    hsi_df = hsi.history(period="max", interval="1d")

    if hsi_df.empty:
        raise ValueError("恒指数据获取失败")

    # 获取美债收益率
    us_yield = yf.Ticker("^TNX")
    us_df = us_yield.history(period="max", interval="1d")

    # 获取VIX
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(period="max", interval="1d")

    print(f"  ✅ 获取到 {len(hsi_df)} 条恒指数据")
    print(f"     时间范围：{hsi_df.index[0].strftime('%Y-%m-%d')} ~ {hsi_df.index[-1].strftime('%Y-%m-%d')}")

    return hsi_df, us_df, vix_df


def calculate_features(hsi_df, us_df, vix_df):
    """计算特征"""
    print("🔧 正在计算特征...")

    df = hsi_df.copy()

    # 移动平均线
    for window in [20, 60, 120, 250]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()

    # 收益率（使用昨日值避免数据泄漏）
    df['Return_1d'] = df['Close'].pct_change().shift(1)
    df['Return_5d'] = df['Close'].pct_change(5).shift(1)
    df['Return_20d'] = df['Close'].pct_change(20).shift(1)

    # 波动率
    df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
    df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()
    df['Volatility_120d'] = df['Return_1d'].rolling(window=120).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)

    # 宏观因子
    if not us_df.empty:
        us_aligned = us_df.reindex(df.index, method='ffill')
        df['US_10Y_Yield'] = us_aligned['Close'] / 10
        df['US_10Y_Yield_Change_5d'] = df['US_10Y_Yield'].pct_change(5)

    if not vix_df.empty:
        vix_aligned = vix_df.reindex(df.index, method='ffill')
        df['VIX'] = vix_aligned['Close']
        df['VIX_Change_5d'] = df['VIX'].pct_change(5)

    print(f"  ✅ 特征计算完成")
    return df


def create_target(df, horizon):
    """创建预测目标"""
    df = df.copy()
    df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(horizon).shift(-horizon)
    df[f'Target_{horizon}d'] = (df[f'Future_Return_{horizon}d'] > 0).astype(int)
    return df


def run_walkforward_multi_horizon(df, start_date='2019-01-01', end_date='2024-12-31'):
    """
    运行多周期Walk-forward验证

    参数:
    - df: 特征数据
    - start_date: 回测开始日期
    - end_date: 回测结束日期

    返回:
    - predictions_df: 预测结果DataFrame
    """
    print("\n" + "=" * 80)
    print("📊 开始Walk-forward多周期验证")
    print("=" * 80)

    horizons = [1, 5, 20]
    all_features = []
    for features in FEATURE_CONFIG.values():
        all_features.extend(features)

    # 筛选可用特征
    available_features = [f for f in all_features if f in df.columns]
    print(f"特征数量: {len(available_features)}")

    # 筛选回测日期范围
    df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    print(f"回测日期范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日数: {len(df)}")

    # 生成月份列表
    all_months = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    while current <= end:
        all_months.append(current.strftime('%Y-%m'))
        current = current + pd.DateOffset(months=1)

    # 配置：12个月训练，1个月测试，1个月步长
    train_window_months = 12
    test_window_months = 1
    step_window_months = 1

    total_months = len(all_months)
    num_folds = (total_months - train_window_months - test_window_months) // step_window_months + 1

    print(f"\nFold数量: {num_folds}")
    print("=" * 80)

    # 存储所有预测结果
    all_predictions = {h: [] for h in horizons}

    for fold in range(num_folds):
        # 计算训练和测试期间
        train_start_idx = fold * step_window_months
        train_end_idx = train_start_idx + train_window_months
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_window_months

        if test_end_idx > total_months:
            break

        train_months = all_months[train_start_idx:train_end_idx]
        test_months = all_months[test_start_idx:test_end_idx]

        train_start = pd.to_datetime(train_months[0] + '-01').tz_localize('UTC')
        train_end = (pd.to_datetime(train_months[-1] + '-01') +
                    pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')
        test_start = pd.to_datetime(test_months[0] + '-01').tz_localize('UTC')
        test_end = (pd.to_datetime(test_months[-1] + '-01') +
                   pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')

        print(f"\nFold {fold + 1}/{num_folds}")
        print(f"  训练: {train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')}")
        print(f"  测试: {test_start.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')}")

        # 筛选数据
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        # 对每个周期训练和预测
        for horizon in horizons:
            # 创建目标
            train_df_h = create_target(train_df.copy(), horizon)
            test_df_h = create_target(test_df.copy(), horizon)

            # 准备数据
            train_clean = train_df_h[available_features + [f'Target_{horizon}d', f'Future_Return_{horizon}d', 'Close']].dropna()
            test_clean = test_df_h[available_features + [f'Target_{horizon}d', f'Future_Return_{horizon}d', 'Close']].dropna()

            if len(train_clean) < 30 or len(test_clean) < 1:
                continue

            X_train = train_clean[available_features]
            y_train = train_clean[f'Target_{horizon}d']
            X_test = test_clean[available_features]

            # 训练模型
            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=5,
                random_seed=RANDOM_SEED,
                auto_class_weights='Balanced',
                verbose=0
            )

            try:
                model.fit(X_train, y_train)

                # 预测
                for idx, row in test_clean.iterrows():
                    X = row[available_features].values.reshape(1, -1)
                    prob = model.predict_proba(X)[0, 1]
                    pred = 1 if prob > 0.5 else 0

                    all_predictions[horizon].append({
                        'date': idx,
                        'horizon': horizon,
                        'prediction': pred,  # 0=下跌, 1=上涨
                        'probability': prob,
                        'actual_direction': row[f'Target_{horizon}d'],
                        'actual_return': row[f'Future_Return_{horizon}d'],
                        'close': row['Close']
                    })
            except Exception as e:
                print(f"    ⚠️ 周期{horizon}训练失败: {e}")
                continue

    print(f"\n{'='*80}")
    print(f"预测完成：")
    for h in horizons:
        print(f"  {h}天周期: {len(all_predictions[h])} 个预测")

    return all_predictions


def analyze_consensus(all_predictions):
    """分析三周期一致预测的效果"""
    print("\n" + "=" * 80)
    print("📊 三周期一致预测分析")
    print("=" * 80)

    horizons = [1, 5, 20]

    # 按日期分组
    daily_predictions = {}
    for horizon in horizons:
        for pred in all_predictions[horizon]:
            date = pred['date']
            if date not in daily_predictions:
                daily_predictions[date] = {}
            daily_predictions[date][horizon] = pred

    dates = sorted(daily_predictions.keys())

    consensus_bull = []
    consensus_bear = []
    mixed = []

    for date in dates:
        preds = daily_predictions[date]

        # 检查是否三个周期都有预测
        if len(preds) < 3:
            continue

        # 获取各周期预测方向
        pred_1d = preds[1]['prediction']
        pred_5d = preds[5]['prediction']
        pred_20d = preds[20]['prediction']

        # 获取各周期实际方向
        actual_1d = preds[1]['actual_direction']
        actual_5d = preds[5]['actual_direction']
        actual_20d = preds[20]['actual_direction']

        # 获取各周期实际收益
        return_1d = preds[1]['actual_return']
        return_5d = preds[5]['actual_return']
        return_20d = preds[20]['actual_return']

        entry = {
            'date': date,
            'pred_1d': pred_1d,
            'pred_5d': pred_5d,
            'pred_20d': pred_20d,
            'actual_1d': actual_1d,
            'actual_5d': actual_5d,
            'actual_20d': actual_20d,
            'return_1d': return_1d,
            'return_5d': return_5d,
            'return_20d': return_20d,
            'close': preds[1]['close']
        }

        # 判断一致性
        if pred_1d == 1 and pred_5d == 1 and pred_20d == 1:
            entry['consensus'] = 'bull'
            entry['at_least_one_correct'] = (actual_1d == 1) or (actual_5d == 1) or (actual_20d == 1)
            entry['all_correct'] = (actual_1d == 1) and (actual_5d == 1) and (actual_20d == 1)
            consensus_bull.append(entry)
        elif pred_1d == 0 and pred_5d == 0 and pred_20d == 0:
            entry['consensus'] = 'bear'
            entry['at_least_one_correct'] = (actual_1d == 0) or (actual_5d == 0) or (actual_20d == 0)
            entry['all_correct'] = (actual_1d == 0) and (actual_5d == 0) and (actual_20d == 0)
            consensus_bear.append(entry)
        else:
            entry['consensus'] = 'mixed'
            mixed.append(entry)

    total_samples = len(consensus_bull) + len(consensus_bear) + len(mixed)

    print(f"\n📈 总体统计：")
    print(f"  总样本数：{total_samples}")
    print(f"  一致看涨：{len(consensus_bull)} ({len(consensus_bull)/total_samples*100:.2f}%)")
    print(f"  一致看跌：{len(consensus_bear)} ({len(consensus_bear)/total_samples*100:.2f}%)")
    print(f"  方向不一致：{len(mixed)} ({len(mixed)/total_samples*100:.2f}%)")

    # 分析一致看涨
    bull_stats = {}
    if consensus_bull:
        print(f"\n🟢 一致看涨分析（{len(consensus_bull)}个样本）：")

        correct_1d = sum(1 for x in consensus_bull if x['actual_1d'] == 1)
        correct_5d = sum(1 for x in consensus_bull if x['actual_5d'] == 1)
        correct_20d = sum(1 for x in consensus_bull if x['actual_20d'] == 1)

        print(f"  1天准确率：{correct_1d}/{len(consensus_bull)} = {correct_1d/len(consensus_bull)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(consensus_bull)} = {correct_5d/len(consensus_bull)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(consensus_bull)} = {correct_20d/len(consensus_bull)*100:.2f}%")

        at_least_one = sum(1 for x in consensus_bull if x['at_least_one_correct'])
        print(f"  至少一周期正确：{at_least_one}/{len(consensus_bull)} = {at_least_one/len(consensus_bull)*100:.2f}%")

        all_correct = sum(1 for x in consensus_bull if x['all_correct'])
        print(f"  三周期全部正确：{all_correct}/{len(consensus_bull)} = {all_correct/len(consensus_bull)*100:.2f}%")

        avg_return_1d = np.mean([x['return_1d'] for x in consensus_bull])
        avg_return_5d = np.mean([x['return_5d'] for x in consensus_bull])
        avg_return_20d = np.mean([x['return_20d'] for x in consensus_bull])

        print(f"  平均1天收益：{avg_return_1d*100:.3f}%")
        print(f"  平均5天收益：{avg_return_5d*100:.3f}%")
        print(f"  平均20天收益：{avg_return_20d*100:.3f}%")

        bull_stats = {
            'count': len(consensus_bull),
            'percentage': len(consensus_bull)/total_samples*100,
            'accuracy_1d': correct_1d/len(consensus_bull)*100,
            'accuracy_5d': correct_5d/len(consensus_bull)*100,
            'accuracy_20d': correct_20d/len(consensus_bull)*100,
            'at_least_one_correct': at_least_one/len(consensus_bull)*100,
            'all_correct': all_correct/len(consensus_bull)*100,
            'avg_return_1d': avg_return_1d*100,
            'avg_return_5d': avg_return_5d*100,
            'avg_return_20d': avg_return_20d*100,
        }

    # 分析一致看跌
    bear_stats = {}
    if consensus_bear:
        print(f"\n🔴 一致看跌分析（{len(consensus_bear)}个样本）：")

        correct_1d = sum(1 for x in consensus_bear if x['actual_1d'] == 0)
        correct_5d = sum(1 for x in consensus_bear if x['actual_5d'] == 0)
        correct_20d = sum(1 for x in consensus_bear if x['actual_20d'] == 0)

        print(f"  1天准确率：{correct_1d}/{len(consensus_bear)} = {correct_1d/len(consensus_bear)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(consensus_bear)} = {correct_5d/len(consensus_bear)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(consensus_bear)} = {correct_20d/len(consensus_bear)*100:.2f}%")

        at_least_one = sum(1 for x in consensus_bear if x['at_least_one_correct'])
        print(f"  至少一周期正确：{at_least_one}/{len(consensus_bear)} = {at_least_one/len(consensus_bear)*100:.2f}%")

        all_correct = sum(1 for x in consensus_bear if x['all_correct'])
        print(f"  三周期全部正确：{all_correct}/{len(consensus_bear)} = {all_correct/len(consensus_bear)*100:.2f}%")

        avg_return_1d = np.mean([x['return_1d'] for x in consensus_bear])
        avg_return_5d = np.mean([x['return_5d'] for x in consensus_bear])
        avg_return_20d = np.mean([x['return_20d'] for x in consensus_bear])

        print(f"  平均1天收益：{avg_return_1d*100:.3f}%")
        print(f"  平均5天收益：{avg_return_5d*100:.3f}%")
        print(f"  平均20天收益：{avg_return_20d*100:.3f}%")

        bear_stats = {
            'count': len(consensus_bear),
            'percentage': len(consensus_bear)/total_samples*100,
            'accuracy_1d': correct_1d/len(consensus_bear)*100,
            'accuracy_5d': correct_5d/len(consensus_bear)*100,
            'accuracy_20d': correct_20d/len(consensus_bear)*100,
            'at_least_one_correct': at_least_one/len(consensus_bear)*100,
            'all_correct': all_correct/len(consensus_bear)*100,
            'avg_return_1d': avg_return_1d*100,
            'avg_return_5d': avg_return_5d*100,
            'avg_return_20d': avg_return_20d*100,
        }

    # 对比：方向不一致
    if mixed:
        print(f"\n⚠️ 方向不一致分析（{len(mixed)}个样本）：")

        correct_1d = sum(1 for x in mixed if x['actual_1d'] == x['pred_1d'])
        correct_5d = sum(1 for x in mixed if x['actual_5d'] == x['pred_5d'])
        correct_20d = sum(1 for x in mixed if x['actual_20d'] == x['pred_20d'])

        print(f"  1天准确率：{correct_1d}/{len(mixed)} = {correct_1d/len(mixed)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(mixed)} = {correct_5d/len(mixed)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(mixed)} = {correct_20d/len(mixed)*100:.2f}%")

    return {
        'total_samples': total_samples,
        'consensus_bull': bull_stats,
        'consensus_bear': bear_stats,
        'mixed_count': len(mixed),
        'mixed_percentage': len(mixed)/total_samples*100 if total_samples > 0 else 0,
    }


def print_validation_results(stats):
    """打印验证结论"""
    print("\n" + "=" * 80)
    print("📋 验证结论")
    print("=" * 80)

    print("\n【待验证的声明】")
    print("1. 一致看涨占比约17.4%，至少一周期正确率约92%")
    print("2. 一致看跌占比约24.0%，至少一周期正确率约87%")

    print("\n【实际回测结果】")

    # 验证一致看涨
    if stats['consensus_bull']:
        bull_pct = stats['consensus_bull']['percentage']
        bull_acc = stats['consensus_bull']['at_least_one_correct']

        pct_check = "✅" if 15 <= bull_pct <= 22 else "❌"
        acc_check = "✅" if bull_acc >= 85 else "❌"

        print(f"1. 一致看涨：")
        print(f"   占比 {bull_pct:.2f}% {pct_check} (预期17.4%)")
        print(f"   至少一周期正确率 {bull_acc:.2f}% {acc_check} (预期92%)")

    # 验证一致看跌
    if stats['consensus_bear']:
        bear_pct = stats['consensus_bear']['percentage']
        bear_acc = stats['consensus_bear']['at_least_one_correct']

        pct_check = "✅" if 20 <= bear_pct <= 28 else "❌"
        acc_check = "✅" if bear_acc >= 80 else "❌"

        print(f"2. 一致看跌：")
        print(f"   占比 {bear_pct:.2f}% {pct_check} (预期24.0%)")
        print(f"   至少一周期正确率 {bear_acc:.2f}% {acc_check} (预期87%)")

    # 总结
    print("\n【总结】")
    bull_ok = stats['consensus_bull'] and 15 <= stats['consensus_bull']['percentage'] <= 22 and stats['consensus_bull']['at_least_one_correct'] >= 85
    bear_ok = stats['consensus_bear'] and 20 <= stats['consensus_bear']['percentage'] <= 28 and stats['consensus_bear']['at_least_one_correct'] >= 80

    if bull_ok and bear_ok:
        print("✅ 验证通过：三周期一致预测策略的统计数据基本正确")
    else:
        print("❌ 验证失败：回测结果与声明存在显著差异")
        print("\n可能原因：")
        print("- 声明的数据可能来自不同的回测参数或时间段")
        print("- 模型参数或特征集可能有所不同")
        print("- 需要检查lessons.md中数据的来源和具体实现")


def main():
    print("=" * 80)
    print("三周期一致预测策略验证 V2")
    print("=" * 80)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取数据
    hsi_df, us_df, vix_df = fetch_data()

    # 2. 计算特征
    df = calculate_features(hsi_df, us_df, vix_df)

    # 3. 运行Walk-forward验证
    # 使用2020-2024年数据，匹配lessons.md中的样本量
    all_predictions = run_walkforward_multi_horizon(df, start_date='2020-01-01', end_date='2024-12-31')

    # 4. 分析一致预测
    stats = analyze_consensus(all_predictions)

    # 5. 打印验证结论
    print_validation_results(stats)

    # 6. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'consensus_validation_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存至：{output_file}")


if __name__ == "__main__":
    main()
