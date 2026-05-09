#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测难度分析
按置信度阈值（0.60、0.65、0.70）统计每只股票的预测表现
"""

import sys
sys.path.insert(0, '/data/fortune')

import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import STOCK_SECTOR_MAPPING

def analyze_prediction_difficulty():
    """分析股票预测难度"""

    # 找到最新的预测分析文件
    output_dir = '/data/fortune/output'
    prediction_files = []
    for f in os.listdir(output_dir):
        if f.endswith('_catboost_20d') and os.path.isdir(os.path.join(output_dir, f)):
            pred_file = os.path.join(output_dir, f, 'prediction_analysis.csv')
            if os.path.exists(pred_file):
                prediction_files.append(pred_file)

    if not prediction_files:
        print("未找到预测分析文件")
        return

    # 使用最新的文件
    latest_file = max(prediction_files, key=lambda x: os.path.getmtime(x))
    print(f"使用预测文件: {latest_file}")

    # 读取预测数据
    df = pd.read_csv(latest_file)
    print(f"总预测样本数: {len(df)}")

    # 获取股票名称映射
    stock_names = {}
    for code, info in STOCK_SECTOR_MAPPING.items():
        stock_names[code] = info.get('name', code)

    # 置信度阈值列表
    thresholds = [0.60, 0.65, 0.70]

    results = []

    for threshold in thresholds:
        print(f"\n分析置信度阈值 >= {threshold}")

        # 筛选满足置信度阈值的预测
        high_conf = df[df['Predict_Prob'] >= threshold].copy()

        # 按股票分组统计
        for stock_code in df['Stock_Code'].unique():
            stock_data = high_conf[high_conf['Stock_Code'] == stock_code]

            if len(stock_data) < 10:  # 样本太少不统计
                continue

            # 计算统计指标
            n_samples = len(stock_data)
            accuracy = stock_data['Is_Correct'].mean()

            # 分离盈利和亏损交易
            # 预测方向为UP时，实际收益为正才是盈利
            # 预测方向为DOWN时，实际收益为负才是盈利（正确预测下跌）
            correct_preds = stock_data[stock_data['Is_Correct'] == True]
            wrong_preds = stock_data[stock_data['Is_Correct'] == False]

            # 盈利交易（正确预测）
            if len(correct_preds) > 0:
                # 对于正确预测上涨的，收益为正
                # 对于正确预测下跌的，收益为负（但我们计算的是"避免损失"）
                # 这里简化处理：用实际收益的绝对值来衡量
                profits = correct_preds['Actual_Return'].abs()
                avg_profit = profits.mean()
                max_profit = profits.max()
            else:
                avg_profit = 0
                max_profit = 0

            # 亏损交易（错误预测）
            if len(wrong_preds) > 0:
                losses = wrong_preds['Actual_Return'].abs()
                avg_loss = losses.mean()
                max_loss = losses.max()
            else:
                avg_loss = 0
                max_loss = 0

            # 计算平均收益率（所有交易的收益平均）
            # 注意：这里需要考虑预测方向
            # 预测UP时，如果正确，收益为正；如果错误，收益为负
            # 预测DOWN时，如果正确，收益为负（避免损失）；如果错误，收益为正（错过上涨）
            # 简化：用实际收益计算，预测正确时收益为正方向，错误时为负方向

            # 更准确的计算：策略收益
            # 预测UP时买入，实际收益 = Actual_Return
            # 预测DOWN时不买入（或卖出），策略收益 = -Actual_Return（如果实际涨了就亏）
            # 这里简化：只统计预测UP的交易（因为实际策略是预测UP才买入）

            up_preds = stock_data[stock_data['Predict_Direction'] == 'UP']
            if len(up_preds) > 0:
                avg_return = up_preds['Actual_Return'].mean()
                max_return = up_preds['Actual_Return'].max()
                # 亏损：预测UP但实际下跌
                wrong_up = up_preds[up_preds['Actual_Return'] < 0]
                if len(wrong_up) > 0:
                    avg_loss_up = wrong_up['Actual_Return'].mean()  # 负数
                    max_loss_up = wrong_up['Actual_Return'].min()   # 最负的
                else:
                    avg_loss_up = 0
                    max_loss_up = 0
            else:
                avg_return = 0
                max_return = 0
                avg_loss_up = 0
                max_loss_up = 0

            # 盈亏比
            if avg_loss_up != 0:
                profit_loss_ratio = abs(avg_return / avg_loss_up)
            else:
                profit_loss_ratio = 999  # 无亏损

            # 预测难度评估
            if accuracy >= 0.75:
                difficulty = '易预测'
            elif accuracy >= 0.60:
                difficulty = '中等'
            else:
                difficulty = '难预测'

            stock_name = stock_names.get(stock_code, stock_code)

            results.append({
                '股票代码': stock_code,
                '股票名称': stock_name,
                '置信度阈值': f'>= {threshold:.2f}',
                '样本数': n_samples,
                '准确率': round(accuracy, 4),
                '平均收益率': round(avg_return, 4),
                '平均盈利': round(avg_return, 4) if avg_return > 0 else 0,
                '最大收益': round(max_return, 4),
                '平均损失': round(avg_loss_up, 4) if avg_loss_up < 0 else 0,
                '最大损失': round(max_loss_up, 4) if max_loss_up < 0 else 0,
                '盈亏比': round(profit_loss_ratio, 2) if profit_loss_ratio < 999 else 'N/A',
                '预测难度': difficulty
            })

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 按置信度阈值和准确率排序
    result_df = result_df.sort_values(['置信度阈值', '准确率'], ascending=[True, False])

    # 保存结果
    output_file = os.path.join(output_dir, 'stock_prediction_difficulty_analysis.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n结果已保存到: {output_file}")
    print(f"总记录数: {len(result_df)}")

    # 打印摘要
    for threshold in thresholds:
        subset = result_df[result_df['置信度阈值'] == f'>= {threshold:.2f}']
        print(f"\n置信度 >= {threshold}:")
        print(f"  股票数: {len(subset)}")
        print(f"  平均准确率: {subset['准确率'].mean():.2%}")
        print(f"  易预测: {len(subset[subset['预测难度'] == '易预测'])}")
        print(f"  中等: {len(subset[subset['预测难度'] == '中等'])}")
        print(f"  难预测: {len(subset[subset['预测难度'] == '难预测'])}")

    return result_df

if __name__ == '__main__':
    analyze_prediction_difficulty()