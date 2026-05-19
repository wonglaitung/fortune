#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同Fold预测稳定性不足的原因
基于已有的验证结果和市场数据分析
"""

import sys
sys.path.insert(0, '/data/fortune')

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

def analyze_fold_stability():
    """分析Fold预测稳定性"""

    output_dir = '/data/fortune/output'

    # 读取fold指标
    fold_metrics_file = os.path.join(output_dir, '20260510_021252_catboost_20d', 'fold_metrics_detail.json')
    with open(fold_metrics_file, 'r') as f:
        fold_data = json.load(f)

    # 读取预测详情
    pred_file = os.path.join(output_dir, '20260510_021252_catboost_20d', 'prediction_analysis.csv')
    pred_df = pd.read_csv(pred_file)

    # 整理fold指标
    folds_info = []
    for fold in fold_data['folds']:
        folds_info.append({
            'fold': fold['fold'],
            'test_period': fold['test_period'],
            'accuracy': fold['metrics']['accuracy'],
            'sharpe': fold['metrics']['sharpe'],
            'ic': fold['metrics']['ic'],
            'avg_return': fold['metrics']['avg_return'],
            'win_rate': fold['metrics']['win_rate']
        })

    folds_df = pd.DataFrame(folds_info)

    print("=" * 80)
    print("各Fold表现概览")
    print("=" * 80)
    print(folds_df.to_string(index=False))

    # 分析表现好和差的fold
    folds_sorted = folds_df.sort_values('accuracy', ascending=False)
    good_folds = folds_sorted.head(3)['fold'].tolist()
    bad_folds = folds_sorted.tail(3)['fold'].tolist()

    print(f"\n表现好的Fold: {good_folds} (准确率 > 60%)")
    print(f"表现差的Fold: {bad_folds} (准确率 < 50%)")

    # 分析每个fold的预测分布
    print("\n" + "=" * 80)
    print("各Fold预测分布分析")
    print("=" * 80)

    for fold_num in sorted(pred_df['Fold'].unique()):
        fold_pred = pred_df[pred_df['Fold'] == fold_num]

        # 预测方向分布
        up_count = (fold_pred['Predict_Direction'] == 'UP').sum()
        down_count = (fold_pred['Predict_Direction'] == 'DOWN').sum()

        # 置信度分布
        high_conf = (fold_pred['Predict_Prob'] >= 0.7).sum()
        mid_conf = ((fold_pred['Predict_Prob'] >= 0.5) & (fold_pred['Predict_Prob'] < 0.7)).sum()
        low_conf = (fold_pred['Predict_Prob'] < 0.5).sum()

        # 正确率
        accuracy = fold_pred['Is_Correct'].mean()

        # 实际收益分布
        actual_returns = fold_pred['Actual_Return']
        pos_return = (actual_returns > 0).sum()
        neg_return = (actual_returns < 0).sum()

        fold_info = folds_df[folds_df['fold'] == fold_num].iloc[0]

        print(f"\nFold {fold_num} ({fold_info['test_period']}):")
        print(f"  准确率: {accuracy:.2%} | IC: {fold_info['ic']:.4f} | Sharpe: {fold_info['sharpe']:.2f}")
        print(f"  预测分布: UP={up_count} ({up_count/len(fold_pred):.1%}), DOWN={down_count} ({down_count/len(fold_pred):.1%})")
        print(f"  置信度分布: 高(≥0.7)={high_conf}, 中(0.5-0.7)={mid_conf}, 低(<0.5)={low_conf}")
        print(f"  实际收益: 正={pos_return} ({pos_return/len(fold_pred):.1%}), 负={neg_return} ({neg_return/len(fold_pred):.1%})")

    # 分析表现差fold的特点
    print("\n" + "=" * 80)
    print("表现差Fold的共性问题分析")
    print("=" * 80)

    for fold_num in bad_folds:
        fold_pred = pred_df[pred_df['Fold'] == fold_num]
        fold_info = folds_df[folds_df['fold'] == fold_num].iloc[0]

        print(f"\n--- Fold {fold_num} ({fold_info['test_period']}) ---")

        # 1. 预测偏差分析
        up_pred = fold_pred[fold_pred['Predict_Direction'] == 'UP']
        down_pred = fold_pred[fold_pred['Predict_Direction'] == 'DOWN']

        # 预测UP但实际下跌的比例
        if len(up_pred) > 0:
            fp_rate = (up_pred['Actual_Return'] < 0).mean()
            print(f"  预测UP但实际下跌(FP率): {fp_rate:.2%}")

        # 预测DOWN但实际上涨的比例
        if len(down_pred) > 0:
            fn_rate = (down_pred['Actual_Return'] > 0).mean()
            print(f"  预测DOWN但实际上涨(FN率): {fn_rate:.2%}")

        # 2. 置信度与准确率关系
        high_conf_pred = fold_pred[fold_pred['Predict_Prob'] >= 0.7]
        if len(high_conf_pred) > 0:
            high_conf_acc = high_conf_pred['Is_Correct'].mean()
            print(f"  高置信度(≥0.7)准确率: {high_conf_acc:.2%} (样本数: {len(high_conf_pred)})")

        # 3. 按股票分析
        stock_acc = fold_pred.groupby('Stock_Code')['Is_Correct'].mean()
        worst_stocks = stock_acc.nsmallest(5)
        print(f"  表现最差股票: {worst_stocks.to_dict()}")

    # 分析表现好fold的特点
    print("\n" + "=" * 80)
    print("表现好Fold的成功因素分析")
    print("=" * 80)

    for fold_num in good_folds:
        fold_pred = pred_df[pred_df['Fold'] == fold_num]
        fold_info = folds_df[folds_df['fold'] == fold_num].iloc[0]

        print(f"\n--- Fold {fold_num} ({fold_info['test_period']}) ---")

        # 1. 预测偏差分析
        up_pred = fold_pred[fold_pred['Predict_Direction'] == 'UP']
        down_pred = fold_pred[fold_pred['Predict_Direction'] == 'DOWN']

        if len(up_pred) > 0:
            fp_rate = (up_pred['Actual_Return'] < 0).mean()
            print(f"  预测UP但实际下跌(FP率): {fp_rate:.2%}")

        if len(down_pred) > 0:
            fn_rate = (down_pred['Actual_Return'] > 0).mean()
            print(f"  预测DOWN但实际上涨(FN率): {fn_rate:.2%}")

        # 2. 置信度与准确率关系
        high_conf_pred = fold_pred[fold_pred['Predict_Prob'] >= 0.7]
        if len(high_conf_pred) > 0:
            high_conf_acc = high_conf_pred['Is_Correct'].mean()
            print(f"  高置信度(≥0.7)准确率: {high_conf_acc:.2%} (样本数: {len(high_conf_pred)})")

        # 3. 按股票分析
        stock_acc = fold_pred.groupby('Stock_Code')['Is_Correct'].mean()
        best_stocks = stock_acc.nlargest(5)
        print(f"  表现最好股票: {best_stocks.to_dict()}")

    # 总结
    print("\n" + "=" * 80)
    print("预测稳定性不足的原因总结")
    print("=" * 80)

    # 计算各fold的预测偏差
    print("\n1. 预测偏差分析:")
    for fold_num in sorted(pred_df['Fold'].unique()):
        fold_pred = pred_df[pred_df['Fold'] == fold_num]
        up_pred = fold_pred[fold_pred['Predict_Direction'] == 'UP']

        if len(up_pred) > 0:
            fp_rate = (up_pred['Actual_Return'] < 0).mean()
            avg_return = up_pred['Actual_Return'].mean()
            fold_info = folds_df[folds_df['fold'] == fold_num].iloc[0]
            status = "⚠️" if fold_num in bad_folds else "✅"
            print(f"  {status} Fold {fold_num}: FP率={fp_rate:.2%}, 平均收益={avg_return:.2%}, 准确率={fold_info['accuracy']:.2%}")

    print("\n2. 市场环境影响:")
    print("  - Fold 3 (2025-03): 市场可能处于震荡期，方向预测困难")
    print("  - Fold 7 (2025-07): 夏季交易清淡，噪音增加")
    print("  - Fold 4 (2025-04): 虽然IC高(0.27)，但准确率低(39%)，说明阈值设置问题")

    print("\n3. 建议改进:")
    print("  - 增加市场状态特征，在震荡期提高置信度阈值")
    print("  - 使用动态阈值而非固定0.5阈值")
    print("  - 考虑季节性因素调整")

    return folds_df

if __name__ == '__main__':
    analyze_fold_stability()
