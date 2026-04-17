#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证"三周期一致预测策略"的有效性

回测逻辑：
1. 在历史数据上训练1天、5天、20天三个周期的CatBoost模型
2. 对每个交易日，使用三个模型分别预测
3. 记录三周期一致看涨/看跌的情况
4. 验证实际收益率和准确率

验证目标：
- 一致看涨样本占比约17.4%，至少一周期正确率约92%
- 一致看跌样本占比约24.0%，至少一周期正确率约87%
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# 配置
HSI_SYMBOL = "^HSI"
DATA_DIR = "data"
OUTPUT_DIR = "output"

# 固定随机种子确保可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def fetch_hsi_data(period="5y"):
    """获取恒指历史数据"""
    print("📊 正在获取恒指数据...")
    hsi = yf.Ticker(HSI_SYMBOL)
    df = hsi.history(period=period, interval="1d")

    if df.empty:
        raise ValueError("恒指数据获取失败")

    print(f"  ✅ 获取到 {len(df)} 条数据，时间范围：{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    return df


def calculate_features(df):
    """计算技术指标特征"""
    print("🔧 正在计算特征...")

    # 移动平均线
    for window in [20, 60, 120, 250]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

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
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)

    # 成交量
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    print(f"  ✅ 特征计算完成")
    return df


def prepare_data(df, horizon):
    """为指定周期准备训练和测试数据"""
    # 创建目标变量：horizon天后是否上涨
    df[f'Target_{horizon}d'] = (df['Close'].pct_change(horizon).shift(-horizon) > 0).astype(int)

    # 选择特征
    features = ['MA20', 'MA60', 'MA120', 'MA250',
                'Return_1d', 'Return_5d', 'Return_20d',
                'Volatility_20d', 'Volatility_60d', 'Volatility_120d',
                'RSI', 'Volume_MA20']

    # 删除缺失值
    df_clean = df[features + [f'Target_{horizon}d', 'Close']].dropna()

    return df_clean, features


def train_model(X_train, y_train, X_test, y_test):
    """训练CatBoost模型"""
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=3,
        auto_class_weights='Balanced',
        random_seed=RANDOM_SEED,
        verbose=0
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    return model


def walk_forward_prediction(df, features, horizon, train_size=0.7):
    """
    使用Walk-forward方法进行多周期预测

    参数:
    - df: 数据DataFrame
    - features: 特征列表
    - horizon: 预测周期（1, 5, 20）
    - train_size: 训练集比例

    返回:
    - predictions: 预测结果列表
    """
    predictions = []

    # 计算分割点
    n_samples = len(df)
    train_end = int(n_samples * train_size)

    # 确保有足够的数据进行训练
    if train_end < 100:
        print(f"  ⚠️ 数据量不足，跳过周期 {horizon}")
        return predictions

    # 使用前70%数据训练模型
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]

    X_train = train_df[features]
    y_train = train_df[f'Target_{horizon}d']
    X_test = test_df[features]
    y_test = test_df[f'Target_{horizon}d']

    # 训练模型
    model = train_model(X_train, y_train, X_test, y_test)

    # 对测试集每个时间点进行预测
    for i, (idx, row) in enumerate(test_df.iterrows()):
        X = row[features].values.reshape(1, -1)
        prob = model.predict_proba(X)[0, 1]
        pred = 1 if prob > 0.5 else 0

        # 获取实际未来收益
        future_return = df['Close'].pct_change(horizon).shift(-horizon).loc[idx]
        actual_direction = 1 if future_return > 0 else 0

        predictions.append({
            'date': idx,
            'horizon': horizon,
            'prediction': pred,  # 0=下跌, 1=上涨
            'probability': prob,
            'actual_direction': actual_direction,
            'actual_return': future_return,
            'is_correct': pred == actual_direction,
            'close': row['Close']
        })

    return predictions


def analyze_consensus(daily_predictions, horizons=[1, 5, 20]):
    """
    分析三周期一致预测的效果

    参数:
    - daily_predictions: 按日期分组的预测结果
    - horizons: 周期列表

    返回:
    - consensus_stats: 一致预测统计
    """
    print("\n" + "=" * 80)
    print("📊 三周期一致预测分析")
    print("=" * 80)

    # 按日期分组
    dates = sorted(daily_predictions.keys())

    consensus_bull = []  # 一致看涨
    consensus_bear = []  # 一致看跌
    mixed = []           # 方向不一致

    for date in dates:
        preds = daily_predictions[date]

        # 检查是否三个周期都有预测
        if len(preds) < 3:
            continue

        # 获取各周期预测方向
        pred_1d = preds.get(1, {}).get('prediction', None)
        pred_5d = preds.get(5, {}).get('prediction', None)
        pred_20d = preds.get(20, {}).get('prediction', None)

        if pred_1d is None or pred_5d is None or pred_20d is None:
            continue

        # 获取各周期实际方向
        actual_1d = preds.get(1, {}).get('actual_direction', None)
        actual_5d = preds.get(5, {}).get('actual_direction', None)
        actual_20d = preds.get(20, {}).get('actual_direction', None)

        # 获取各周期实际收益
        return_1d = preds.get(1, {}).get('actual_return', 0)
        return_5d = preds.get(5, {}).get('actual_return', 0)
        return_20d = preds.get(20, {}).get('actual_return', 0)

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
            'close': preds.get(1, {}).get('close', 0)
        }

        # 判断一致性
        if pred_1d == 1 and pred_5d == 1 and pred_20d == 1:
            # 三周期一致看涨
            entry['consensus'] = 'bull'
            entry['at_least_one_correct'] = (actual_1d == 1) or (actual_5d == 1) or (actual_20d == 1)
            entry['all_correct'] = (actual_1d == 1) and (actual_5d == 1) and (actual_20d == 1)
            consensus_bull.append(entry)
        elif pred_1d == 0 and pred_5d == 0 and pred_20d == 0:
            # 三周期一致看跌
            entry['consensus'] = 'bear'
            entry['at_least_one_correct'] = (actual_1d == 0) or (actual_5d == 0) or (actual_20d == 0)
            entry['all_correct'] = (actual_1d == 0) and (actual_5d == 0) and (actual_20d == 0)
            consensus_bear.append(entry)
        else:
            # 方向不一致
            entry['consensus'] = 'mixed'
            mixed.append(entry)

    total_samples = len(consensus_bull) + len(consensus_bear) + len(mixed)

    print(f"\n📈 总体统计：")
    print(f"  总样本数：{total_samples}")
    print(f"  一致看涨：{len(consensus_bull)} ({len(consensus_bull)/total_samples*100:.2f}%)")
    print(f"  一致看跌：{len(consensus_bear)} ({len(consensus_bear)/total_samples*100:.2f}%)")
    print(f"  方向不一致：{len(mixed)} ({len(mixed)/total_samples*100:.2f}%)")

    # 分析一致看涨
    if consensus_bull:
        print(f"\n🟢 一致看涨分析（{len(consensus_bull)}个样本）：")

        # 各周期准确率
        correct_1d = sum(1 for x in consensus_bull if x['actual_1d'] == 1)
        correct_5d = sum(1 for x in consensus_bull if x['actual_5d'] == 1)
        correct_20d = sum(1 for x in consensus_bull if x['actual_20d'] == 1)

        print(f"  1天准确率：{correct_1d}/{len(consensus_bull)} = {correct_1d/len(consensus_bull)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(consensus_bull)} = {correct_5d/len(consensus_bull)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(consensus_bull)} = {correct_20d/len(consensus_bull)*100:.2f}%")

        # 至少一个正确
        at_least_one = sum(1 for x in consensus_bull if x['at_least_one_correct'])
        print(f"  至少一周期正确：{at_least_one}/{len(consensus_bull)} = {at_least_one/len(consensus_bull)*100:.2f}%")

        # 全部正确
        all_correct = sum(1 for x in consensus_bull if x['all_correct'])
        print(f"  三周期全部正确：{all_correct}/{len(consensus_bull)} = {all_correct/len(consensus_bull)*100:.2f}%")

        # 平均收益率
        avg_return_1d = np.mean([x['return_1d'] for x in consensus_bull])
        avg_return_5d = np.mean([x['return_5d'] for x in consensus_bull])
        avg_return_20d = np.mean([x['return_20d'] for x in consensus_bull])

        print(f"  平均1天收益：{avg_return_1d*100:.3f}%")
        print(f"  平均5天收益：{avg_return_5d*100:.3f}%")
        print(f"  平均20天收益：{avg_return_20d*100:.3f}%")

    # 分析一致看跌
    if consensus_bear:
        print(f"\n🔴 一致看跌分析（{len(consensus_bear)}个样本）：")

        # 各周期准确率（预测下跌正确 = 实际下跌）
        correct_1d = sum(1 for x in consensus_bear if x['actual_1d'] == 0)
        correct_5d = sum(1 for x in consensus_bear if x['actual_5d'] == 0)
        correct_20d = sum(1 for x in consensus_bear if x['actual_20d'] == 0)

        print(f"  1天准确率：{correct_1d}/{len(consensus_bear)} = {correct_1d/len(consensus_bear)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(consensus_bear)} = {correct_5d/len(consensus_bear)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(consensus_bear)} = {correct_20d/len(consensus_bear)*100:.2f}%")

        # 至少一个正确
        at_least_one = sum(1 for x in consensus_bear if x['at_least_one_correct'])
        print(f"  至少一周期正确：{at_least_one}/{len(consensus_bear)} = {at_least_one/len(consensus_bear)*100:.2f}%")

        # 全部正确
        all_correct = sum(1 for x in consensus_bear if x['all_correct'])
        print(f"  三周期全部正确：{all_correct}/{len(consensus_bear)} = {all_correct/len(consensus_bear)*100:.2f}%")

        # 平均收益率（注意：看跌预测正确时收益率为负）
        avg_return_1d = np.mean([x['return_1d'] for x in consensus_bear])
        avg_return_5d = np.mean([x['return_5d'] for x in consensus_bear])
        avg_return_20d = np.mean([x['return_20d'] for x in consensus_bear])

        print(f"  平均1天收益：{avg_return_1d*100:.3f}%")
        print(f"  平均5天收益：{avg_return_5d*100:.3f}%")
        print(f"  平均20天收益：{avg_return_20d*100:.3f}%")

    # 对比：方向不一致
    if mixed:
        print(f"\n⚠️ 方向不一致分析（{len(mixed)}个样本）：")

        correct_1d = sum(1 for x in mixed if x['actual_1d'] == x['pred_1d'])
        correct_5d = sum(1 for x in mixed if x['actual_5d'] == x['pred_5d'])
        correct_20d = sum(1 for x in mixed if x['actual_20d'] == x['pred_20d'])

        print(f"  1天准确率：{correct_1d}/{len(mixed)} = {correct_1d/len(mixed)*100:.2f}%")
        print(f"  5天准确率：{correct_5d}/{len(mixed)} = {correct_5d/len(mixed)*100:.2f}%")
        print(f"  20天准确率：{correct_20d}/{len(mixed)} = {correct_20d/len(mixed)*100:.2f}%")

    # 返回统计结果
    return {
        'total_samples': total_samples,
        'consensus_bull': {
            'count': len(consensus_bull),
            'percentage': len(consensus_bull)/total_samples*100 if total_samples > 0 else 0,
            'accuracy_1d': correct_1d/len(consensus_bull)*100 if consensus_bull else 0,
            'accuracy_5d': correct_5d/len(consensus_bull)*100 if consensus_bull else 0,
            'accuracy_20d': correct_20d/len(consensus_bull)*100 if consensus_bull else 0,
            'at_least_one_correct': at_least_one/len(consensus_bull)*100 if consensus_bull else 0,
        },
        'consensus_bear': {
            'count': len(consensus_bear),
            'percentage': len(consensus_bear)/total_samples*100 if total_samples > 0 else 0,
            'accuracy_1d': correct_1d/len(consensus_bear)*100 if consensus_bear else 0,
            'accuracy_5d': correct_5d/len(consensus_bear)*100 if consensus_bear else 0,
            'accuracy_20d': correct_20d/len(consensus_bear)*100 if consensus_bear else 0,
            'at_least_one_correct': at_least_one/len(consensus_bear)*100 if consensus_bear else 0,
        },
        'mixed': {
            'count': len(mixed),
            'percentage': len(mixed)/total_samples*100 if total_samples > 0 else 0,
        }
    }


def run_backtest():
    """运行回测验证"""
    print("=" * 80)
    print("三周期一致预测策略验证")
    print("=" * 80)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取数据
    df = fetch_hsi_data(period="5y")

    # 2. 计算特征
    df = calculate_features(df)

    # 3. 对三个周期分别进行walk-forward预测
    horizons = [1, 5, 20]
    all_predictions = {h: [] for h in horizons}

    for horizon in horizons:
        print(f"\n📊 正在处理 {horizon}天周期...")

        # 准备数据
        df_clean, features = prepare_data(df.copy(), horizon)

        if len(df_clean) < 200:
            print(f"  ⚠️ 数据不足，跳过")
            continue

        # Walk-forward预测
        predictions = walk_forward_prediction(df_clean, features, horizon)
        all_predictions[horizon] = predictions

        print(f"  ✅ 生成 {len(predictions)} 个预测")

    # 4. 按日期分组
    daily_predictions = {}
    for horizon in horizons:
        for pred in all_predictions[horizon]:
            date = pred['date']
            if date not in daily_predictions:
                daily_predictions[date] = {}
            daily_predictions[date][horizon] = pred

    # 5. 分析一致预测
    stats = analyze_consensus(daily_predictions)

    # 6. 输出验证结论
    print("\n" + "=" * 80)
    print("📋 验证结论")
    print("=" * 80)

    print("\n【待验证的声明】")
    print("1. 一致看涨占比约17.4%，至少一周期正确率约92%")
    print("2. 一致看跌占比约24.0%，至少一周期正确率约87%")

    print("\n【实际回测结果】")
    if stats['consensus_bull']['count'] > 0:
        bull_pct = stats['consensus_bull']['percentage']
        bull_acc = stats['consensus_bull'].get('at_least_one_correct', 0)
        print(f"1. 一致看涨：占比 {bull_pct:.2f}% ({'✅' if 15 <= bull_pct <= 20 else '❌'}), "
              f"至少一周期正确率 {bull_acc:.2f}% ({'✅' if bull_acc >= 85 else '❌'})")

    if stats['consensus_bear']['count'] > 0:
        bear_pct = stats['consensus_bear']['percentage']
        bear_acc = stats['consensus_bear'].get('at_least_one_correct', 0)
        print(f"2. 一致看跌：占比 {bear_pct:.2f}% ({'✅' if 22 <= bear_pct <= 26 else '❌'}), "
              f"至少一周期正确率 {bear_acc:.2f}% ({'✅' if bear_acc >= 80 else '❌'})")

    # 7. 保存详细结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'multi_horizon_consensus_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'data_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(df)
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存至：{output_file}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证三周期一致预测策略')
    parser.add_argument('--period', type=str, default='5y', help='数据周期 (默认: 5y)')
    args = parser.parse_args()

    run_backtest()
