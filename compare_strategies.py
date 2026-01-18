#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略对比脚本 - 对比机器学习模型和手工信号的预测准确性
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 导入项目模块
from ml_trading_model import MLTradingModel, WATCHLIST
from hk_smart_money_tracker import analyze_stock
from tencent_finance import get_hk_stock_data_tencent


def get_ml_model_prediction(model, code):
    """获取机器学习模型的预测"""
    try:
        result = model.predict(code)
        if result:
            return {
                'code': code,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'date': result['date']
            }
    except Exception as e:
        print(f"获取ML预测失败 {code}: {e}")
    return None


def get_smart_money_signal(code):
    """获取手工信号"""
    try:
        result = analyze_stock(code, code)
        if result and isinstance(result, dict):
            return {
                'code': code,
                'has_buildup': result.get('has_buildup', False),
                'has_distribution': result.get('has_distribution', False)
            }
    except Exception as e:
        print(f"获取手工信号失败 {code}: {e}")
    return None


def compare_predictions(codes):
    """对比两种策略的预测"""
    print("=" * 80)
    print("策略对比分析 - 预测准确性对比")
    print("=" * 80)

    # 加载机器学习模型
    ml_model = MLTradingModel()
    try:
        ml_model.load_model('data/ml_trading_model.pkl')
    except:
        print("警告: 未找到机器学习模型，请先运行训练模式")
        return

    ml_predictions = []
    smart_signals = []
    comparison_results = []

    for code in codes:
        try:
            print(f"\n分析股票: {code}")

            # 获取ML预测
            ml_pred = get_ml_model_prediction(ml_model, code)
            if ml_pred:
                ml_predictions.append(ml_pred)

            # 获取手工信号
            smart_signal = get_smart_money_signal(code)
            if smart_signal:
                smart_signals.append(smart_signal)

            # 对比
            if ml_pred and smart_signal:
                comparison = {
                    'code': code,
                    'ml_prediction': ml_pred['prediction'],
                    'ml_probability': ml_pred['probability'],
                    'smart_has_buildup': smart_signal['has_buildup'],
                    'smart_has_distribution': smart_signal['has_distribution'],
                    'signal_match': (ml_pred['prediction'] == 1 and smart_signal['has_buildup']) or
                                   (ml_pred['prediction'] == 0 and not smart_signal['has_buildup'])
                }
                comparison_results.append(comparison)

        except Exception as e:
            print(f"分析股票 {code} 失败: {e}")
            continue

    # 汇总结果
    print("\n" + "=" * 80)
    print("机器学习模型预测汇总")
    print("=" * 80)

    if ml_predictions:
        ml_df = pd.DataFrame(ml_predictions)
        print(f"预测股票数: {len(ml_df)}")
        print(f"预测上涨: {len(ml_df[ml_df['prediction'] == 1])}")
        print(f"预测下跌: {len(ml_df[ml_df['prediction'] == 0])}")
        print(f"平均上涨概率: {ml_df[ml_df['prediction'] == 1]['probability'].mean():.4f}")
        print(f"平均下跌概率: {(1 - ml_df[ml_df['prediction'] == 0]['probability']).mean():.4f}")

        print("\n预测详情:")
        print("-" * 80)
        print(f"{'代码':<10} {'预测':<8} {'概率':<10} {'日期':<15}")
        print("-" * 80)
        for _, row in ml_df.iterrows():
            pred_label = "上涨" if row['prediction'] == 1 else "下跌"
            print(f"{row['code']:<10} {pred_label:<8} {row['probability']:.4f}    {row['date'].strftime('%Y-%m-%d')}")

    print("\n" + "=" * 80)
    print("手工信号汇总")
    print("=" * 80)

    if smart_signals:
        smart_df = pd.DataFrame(smart_signals)
        print(f"分析股票数: {len(smart_df)}")
        print(f"建仓信号: {len(smart_df[smart_df['has_buildup'] == True])}")
        print(f"出货信号: {len(smart_df[smart_df['has_distribution'] == True])}")

        print("\n信号详情:")
        print("-" * 80)
        print(f"{'代码':<10} {'建仓':<8} {'出货':<8}")
        print("-" * 80)
        for _, row in smart_df.iterrows():
            buildup = "是" if row['has_buildup'] else "否"
            distribution = "是" if row['has_distribution'] else "否"
            print(f"{row['code']:<10} {buildup:<8} {distribution:<8}")

    print("\n" + "=" * 80)
    print("策略对比")
    print("=" * 80)

    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)
        print(f"{'代码':<10} {'ML预测':<8} {'ML概率':<10} {'建仓信号':<10} {'信号一致':<10}")
        print("-" * 80)

        match_count = 0
        mismatch_count = 0

        for _, row in comp_df.iterrows():
            ml_pred_label = "上涨" if row['ml_prediction'] == 1 else "下跌"
            smart_label = "是" if row['smart_has_buildup'] else "否"
            match_label = "✓" if row['signal_match'] else "✗"

            if row['signal_match']:
                match_count += 1
            else:
                mismatch_count += 1

            print(f"{row['code']:<10} {ml_pred_label:<8} {row['ml_probability']:.4f}    {smart_label:<10} {match_label:<10}")

        print("-" * 80)
        print(f"信号一致: {match_count} 只股票")
        print(f"信号不一致: {mismatch_count} 只股票")
        print(f"一致性: {match_count / len(comp_df) * 100:.2f}%")

        # 保存对比结果
        comp_path = 'data/strategy_comparison.csv'
        comp_df.to_csv(comp_path, index=False)
        print(f"\n对比结果已保存到 {comp_path}")


def main():
    parser = argparse.ArgumentParser(description='策略对比分析')
    parser.add_argument('--start-date', type=str, default=None,
                       help='分析开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='分析结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    compare_predictions(WATCHLIST)


if __name__ == '__main__':
    main()
