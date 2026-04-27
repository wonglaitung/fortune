#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三周期预测关系分析（增强版）

集成新增特征：日历效应、GARCH波动率、市场状态检测

分析1天、5天、20天三个预测周期之间的：
1. 时间序列关系（先后顺序）
2. 因果关系（一个周期预测如何影响其他周期）
3. 规律模式（一致性、传导性、均值回归等）

更新时间：2026-04-26
"""

import warnings
import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新增特征模块
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

    # ========== 新增特征：日历效应 ==========
    print("  计算日历效应特征...")
    calendar_calc = CalendarFeatureCalculator()
    df = calendar_calc.calculate_features(df)

    # ========== 新增特征：GARCH 波动率 ==========
    print("  计算 GARCH 波动率特征...")
    garch_model = GARCHVolatilityModel()
    df = garch_model.calculate_features(df)

    # ========== 新增特征：市场状态检测 ==========
    print("  计算市场状态特征...")
    regime_detector = RegimeDetector()
    df = regime_detector.calculate_features(df)

    print(f"  ✅ 特征计算完成: {len(df.columns)} 列")

    return df


def get_feature_list():
    """获取特征列表（基础 + 新增）"""
    features = [
        # 基础技术指标
        'MA20', 'MA60', 'MA120', 'MA250',
        'Return_1d', 'Return_5d', 'Return_20d',
        'Volatility_20d', 'Volatility_60d',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Width',
        'MA250_Slope', 'MA_Bullish_Alignment', 'MA_Bearish_Alignment',
        # 宏观因子
        'US_10Y_Yield', 'VIX',
    ]
    return features


def run_walkforward_predictions(df):
    """运行Walk-forward获取三周期预测"""
    print("\n🔬 运行Walk-forward预测...")

    horizons = [1, 5, 20]

    # 基础特征 + 新增特征
    base_features = get_feature_list()

    # 新增特征（日历、GARCH、Regime）
    new_features = [
        # 日历效应（选取重要特征）
        'Day_of_Week', 'Month_of_Year', 'Days_to_Options_Expiry',
        'Days_to_Holiday', 'Is_Pre_Holiday', 'Is_Typhoon_Season',
        'Month_Sin', 'Month_Cos',
        # GARCH
        'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
        # 市场状态
        'Market_Regime', 'Regime_Prob_0', 'Regime_Prob_1', 'Regime_Prob_2',
    ]

    all_features = base_features + new_features
    available_features = [f for f in all_features if f in df.columns]

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
            except Exception as e:
                continue

        if (fold + 1) % 50 == 0:
            print(f"  已完成 {fold + 1}/{num_folds} folds...")

    return all_predictions


def analyze_and_save_results(all_predictions):
    """分析结果并保存到JSON"""
    print("\n📊 分析结果...")

    # 构建DataFrame
    df_1d = pd.DataFrame(all_predictions[1])
    df_5d = pd.DataFrame(all_predictions[5])
    df_20d = pd.DataFrame(all_predictions[20])

    if len(df_1d) == 0 or len(df_5d) == 0 or len(df_20d) == 0:
        print("❌ 数据不足，无法分析")
        return None

    df_1d.set_index('date', inplace=True)
    df_5d.set_index('date', inplace=True)
    df_20d.set_index('date', inplace=True)

    # 对齐数据
    common_dates = df_1d.index.intersection(df_5d.index).intersection(df_20d.index)
    print(f"  共同样本数: {len(common_dates)}")

    pred_1d = df_1d.loc[common_dates, 'prediction']
    pred_5d = df_5d.loc[common_dates, 'prediction']
    pred_20d = df_20d.loc[common_dates, 'prediction']

    actual_1d = df_1d.loc[common_dates, 'actual_direction']
    actual_5d = df_5d.loc[common_dates, 'actual_direction']
    actual_20d = df_20d.loc[common_dates, 'actual_direction']

    # ========== 1. 独立准确率 ==========
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'enhanced (with calendar/GARCH/regime features)',
        'total_samples': len(common_dates),
        'independent_accuracy': {
            '1d': {
                'accuracy': float((pred_1d == actual_1d).mean()),
                'samples': len(common_dates)
            },
            '5d': {
                'accuracy': float((pred_5d == actual_5d).mean()),
                'samples': len(common_dates)
            },
            '20d': {
                'accuracy': float((pred_20d == actual_20d).mean()),
                'samples': len(common_dates)
            }
        }
    }

    # ========== 2. 因果分析 ==========
    correct_1d = actual_1d == pred_1d
    correct_5d = actual_5d == pred_5d

    results['causal_analysis'] = {
        '1d_correct_then_20d_accuracy': float((pred_20d[correct_1d] == actual_20d[correct_1d]).mean()) if correct_1d.sum() > 0 else 0,
        '1d_wrong_then_20d_accuracy': float((pred_20d[~correct_1d] == actual_20d[~correct_1d]).mean()) if (~correct_1d).sum() > 0 else 0,
        '5d_correct_then_20d_accuracy': float((pred_20d[correct_5d] == actual_20d[correct_5d]).mean()) if correct_5d.sum() > 0 else 0,
        '1d_5d_correct_then_20d_accuracy': float((pred_20d[correct_1d & correct_5d] == actual_20d[correct_1d & correct_5d]).mean()) if (correct_1d & correct_5d).sum() > 0 else 0,
    }

    # ========== 3. 八大模式分析 ==========
    patterns = {
        '111': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 1)),
        '110': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 0)),
        '101': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 1)),
        '100': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 0)),
        '011': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 1)),
        '010': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 0)),
        '001': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 1)),
        '000': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 0)),
    }

    results['patterns'] = {}
    for pattern, mask in patterns.items():
        count = mask.sum()
        if count > 0:
            results['patterns'][pattern] = {
                'count': int(count),
                'pct': float(count / len(common_dates) * 100),
                '20d_accuracy': float((pred_20d[mask] == actual_20d[mask]).mean() * 100),
                '1d_accuracy': float((pred_1d[mask] == actual_1d[mask]).mean() * 100),
                '5d_accuracy': float((pred_5d[mask] == actual_5d[mask]).mean() * 100),
            }

    # ========== 4. 一致信号分析 ==========
    consensus_bull = patterns['111']
    consensus_bear = patterns['000']

    results['consensus_analysis'] = {
        'bull': {
            'count': int(consensus_bull.sum()),
            '1d_actual_up_rate': float(actual_1d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
            '5d_actual_up_rate': float(actual_5d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
            '20d_actual_up_rate': float(actual_20d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
        },
        'bear': {
            'count': int(consensus_bear.sum()),
            '1d_actual_down_rate': float((1 - actual_1d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
            '5d_actual_down_rate': float((1 - actual_5d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
            '20d_actual_down_rate': float((1 - actual_20d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
        }
    }

    # ========== 5. 概率相关性 ==========
    prob_1d = df_1d.loc[common_dates, 'probability']
    prob_5d = df_5d.loc[common_dates, 'probability']
    prob_20d = df_20d.loc[common_dates, 'probability']

    results['probability_correlation'] = {
        '1d_vs_5d': float(prob_1d.corr(prob_5d)),
        '1d_vs_20d': float(prob_1d.corr(prob_20d)),
        '5d_vs_20d': float(prob_5d.corr(prob_20d)),
    }

    # ========== 6. 序列分析 ==========
    all_correct = (pred_1d == actual_1d) & (pred_5d == actual_5d) & (pred_20d == actual_20d)
    at_least_one = (pred_1d == actual_1d) | (pred_5d == actual_5d) | (pred_20d == actual_20d)

    results['sequence_analysis'] = {
        'all_correct_count': int(all_correct.sum()),
        'all_correct_pct': float(all_correct.sum() / len(common_dates) * 100),
        'at_least_one_correct_count': int(at_least_one.sum()),
        'at_least_one_correct_pct': float(at_least_one.sum() / len(common_dates) * 100),
    }

    # ========== 保存结果 ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'three_horizon_validation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存到: {output_path}")

    return results


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 80)
    print("📊 三周期预测分析结果")
    print("=" * 80)

    # 独立准确率
    print("\n【独立准确率】")
    for h in ['1d', '5d', '20d']:
        acc = results['independent_accuracy'][h]['accuracy'] * 100
        print(f"  {h}: {acc:.2f}%")

    # 因果分析
    print("\n【因果分析】")
    print(f"  1天正确 → 20天准确率: {results['causal_analysis']['1d_correct_then_20d_accuracy']*100:.2f}%")
    print(f"  1天错误 → 20天准确率: {results['causal_analysis']['1d_wrong_then_20d_accuracy']*100:.2f}%")
    print(f"  1+5天正确 → 20天准确率: {results['causal_analysis']['1d_5d_correct_then_20d_accuracy']*100:.2f}%")

    # 八大模式
    print("\n【八大模式（按20天准确率排序）】")
    sorted_patterns = sorted(results['patterns'].items(), key=lambda x: x[1]['20d_accuracy'], reverse=True)
    print(f"  {'模式':<6} {'样本数':<8} {'占比':<10} {'20天准确率':<12}")
    print("  " + "-" * 40)
    for pattern, data in sorted_patterns:
        print(f"  {pattern:<6} {data['count']:<8} {data['pct']:.2f}%    {data['20d_accuracy']:.2f}%")

    # 一致信号
    print("\n【一致信号分析】")
    bull = results['consensus_analysis']['bull']
    bear = results['consensus_analysis']['bear']
    print(f"  一致看涨(111): {bull['count']}次, 20天实际上涨率: {bull['20d_actual_up_rate']:.2f}%")
    print(f"  一致看跌(000): {bear['count']}次, 20天实际下跌率: {bear['20d_actual_down_rate']:.2f}%")


def main():
    """主函数"""
    print("=" * 80)
    print("三周期预测关系深度分析（增强版）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取数据（含新特征）
    df = fetch_and_prepare_data()

    # 2. 运行预测
    all_predictions = run_walkforward_predictions(df)

    # 3. 分析并保存结果
    results = analyze_and_save_results(all_predictions)

    if results:
        # 4. 打印摘要
        print_summary(results)

    print("\n" + "=" * 80)
    print(f"分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
