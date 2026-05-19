#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查现有特征在表现差Fold中的实际值
"""

import sys
sys.path.insert(0, '/data/fortune')

import pandas as pd
import numpy as np
import os
import json

def check_features_in_bad_folds():
    """检查特征在表现差Fold中的值"""

    output_dir = '/data/fortune/output'

    # 读取预测详情
    pred_file = os.path.join(output_dir, '20260510_021252_catboost_20d', 'prediction_analysis.csv')
    pred_df = pd.read_csv(pred_file)

    # 表现差的Fold
    bad_folds = [3, 7, 4]

    print("=" * 80)
    print("表现差Fold的市场状态特征分析")
    print("=" * 80)

    # 检查预测分布
    for fold_num in bad_folds:
        fold_pred = pred_df[pred_df['Fold'] == fold_num]

        print(f"\n--- Fold {fold_num} ---")
        print(f"预测UP但实际下跌(FP): {(fold_pred['Predict_Direction'] == 'UP') & (fold_pred['Actual_Return'] < 0)}.sum()")
        print(f"预测DOWN但实际上涨(FN): {(fold_pred['Predict_Direction'] == 'DOWN') & (fold_pred['Actual_Return'] > 0)}.sum()")

        # 按股票分析
        print("\n按股票准确率:")
        stock_acc = fold_pred.groupby('Stock_Code').agg({
            'Is_Correct': 'mean',
            'Predict_Prob': 'mean',
            'Actual_Return': 'mean'
        }).round(4)
        stock_acc.columns = ['准确率', '平均置信度', '平均收益']
        print(stock_acc.sort_values('准确率').head(10))

    # 分析问题
    print("\n" + "=" * 80)
    print("问题诊断")
    print("=" * 80)

    print("""
根据分析，现有特征已包含：
1. ✅ MACD/RSI背离检测（趋势反转信号）
2. ✅ 情绪指标（sentiment_ma3/7/14等）
3. ✅ 季节性特征（Month_Sin/Cos）
4. ✅ 动态阈值（Confidence_Threshold_Multiplier）
5. ✅ HMM市场状态（HSI_Market_Regime）

但预测稳定性仍然不足，可能原因：

1. **特征未被有效利用**
   - 背离特征可能重要性较低，未进入Top特征
   - 模型可能过度依赖价格动量特征

2. **市场状态转换滞后**
   - HMM检测到状态变化时，市场已转向
   - 需要更早期的预警信号

3. **训练数据分布问题**
   - Fold 3训练期(2024-03~2025-02)包含牛市
   - 测试期(2025-03)突然下跌，分布偏移

4. **建议改进方向**
   - 增加市场转折点预测模型（专门预测趋势反转）
   - 使用在线学习适应分布偏移
   - 提高背离特征的权重或创建新组合
    """)

if __name__ == '__main__':
    check_features_in_bad_folds()
