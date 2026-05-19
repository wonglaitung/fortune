#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同Fold的特征重要性差异
找出预测稳定性不足的原因
"""

import sys
sys.path.insert(0, '/data/fortune')

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from config import TRAINING_STOCKS
from ml_services.ml_trading_model import CatBoostModel

def analyze_fold_feature_importance():
    """分析不同Fold的特征重要性"""

    output_dir = '/data/fortune/output'

    # 读取fold指标
    fold_metrics_file = os.path.join(output_dir, '20260510_021252_catboost_20d', 'fold_metrics_detail.json')
    with open(fold_metrics_file, 'r') as f:
        fold_data = json.load(f)

    # 找出表现好和表现差的fold
    folds_info = []
    for fold in fold_data['folds']:
        folds_info.append({
            'fold': fold['fold'],
            'test_period': fold['test_period'],
            'accuracy': fold['metrics']['accuracy'],
            'sharpe': fold['metrics']['sharpe'],
            'ic': fold['metrics']['ic'],
            'avg_return': fold['metrics']['avg_return']
        })

    folds_df = pd.DataFrame(folds_info)
    print("=" * 80)
    print("各Fold表现概览")
    print("=" * 80)
    print(folds_df.to_string(index=False))

    # 按准确率排序
    folds_df = folds_df.sort_values('accuracy', ascending=False)

    # 选出表现最好和最差的fold
    good_folds = folds_df.head(3)['fold'].tolist()
    bad_folds = folds_df.tail(3)['fold'].tolist()

    print(f"\n表现好的Fold: {good_folds}")
    print(f"表现差的Fold: {bad_folds}")

    # 训练每个fold并获取特征重要性
    stock_list = list(TRAINING_STOCKS.keys())

    # Walk-forward 参数
    train_window_months = 12
    test_window_months = 1
    start_date = '2024-01-01'

    all_feature_importance = []

    for fold_num in range(1, 13):
        print(f"\n{'='*80}")
        print(f"训练 Fold {fold_num}")
        print("=" * 80)

        # 计算训练和测试日期
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, '%Y-%m-%d')

        # 训练窗口起始
        train_start = start + timedelta(days=(fold_num - 1) * 30)
        train_end = train_start + timedelta(days=365)  # 约12个月

        # 测试窗口
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=30)

        print(f"训练期间: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"测试期间: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

        try:
            # 创建模型
            model = CatBoostModel()

            # 训练模型
            model.train(
                stock_list,
                start_date=train_start.strftime('%Y-%m-%d'),
                end_date=train_end.strftime('%Y-%m-%d'),
                horizon=20,
                use_feature_selection=False
            )

            # 获取特征重要性
            if hasattr(model, 'catboost_model') and model.catboost_model is not None:
                feat_imp = pd.DataFrame({
                    'Feature': model.feature_columns,
                    'Importance': model.catboost_model.feature_importances_
                })
                feat_imp = feat_imp.sort_values('Importance', ascending=False)
                feat_imp['Fold'] = fold_num
                feat_imp['Rank'] = range(1, len(feat_imp) + 1)

                all_feature_importance.append(feat_imp)

                # 打印Top 10特征
                print(f"\nFold {fold_num} Top 10 特征:")
                print(feat_imp.head(10).to_string(index=False))

        except Exception as e:
            print(f"Fold {fold_num} 训练失败: {e}")
            continue

    if not all_feature_importance:
        print("没有成功训练任何Fold")
        return

    # 合并所有特征重要性
    all_feat_df = pd.concat(all_feature_importance, ignore_index=True)

    # 分析特征重要性稳定性
    print("\n" + "=" * 80)
    print("特征重要性稳定性分析")
    print("=" * 80)

    # 计算每个特征在各fold中的排名变化
    feature_rank_pivot = all_feat_df.pivot_table(
        index='Feature',
        columns='Fold',
        values='Rank'
    )

    # 计算排名标准差
    feature_rank_pivot['Rank_Std'] = feature_rank_pivot.std(axis=1)
    feature_rank_pivot['Rank_Mean'] = feature_rank_pivot.mean(axis=1)
    feature_rank_pivot = feature_rank_pivot.sort_values('Rank_Std', ascending=False)

    print("\n排名变化最大的特征（不稳定）:")
    print(feature_rank_pivot.head(20)[['Rank_Mean', 'Rank_Std']].to_string())

    print("\n排名变化最小的特征（稳定）:")
    print(feature_rank_pivot.tail(20)[['Rank_Mean', 'Rank_Std']].to_string())

    # 分析表现好和差的fold的特征重要性差异
    print("\n" + "=" * 80)
    print("表现好 vs 表现差 Fold 的特征重要性差异")
    print("=" * 80)

    # 计算好fold和差fold的平均特征重要性
    good_fold_imp = all_feat_df[all_feat_df['Fold'].isin(good_folds)].groupby('Feature')['Importance'].mean()
    bad_fold_imp = all_feat_df[all_feat_df['Fold'].isin(bad_folds)].groupby('Feature')['Importance'].mean()

    diff_df = pd.DataFrame({
        'Feature': good_fold_imp.index,
        'Good_Fold_Imp': good_fold_imp.values,
        'Bad_Fold_Imp': bad_fold_imp.values
    })
    diff_df['Diff'] = diff_df['Good_Fold_Imp'] - diff_df['Bad_Fold_Imp']
    diff_df['Diff_Pct'] = (diff_df['Diff'] / diff_df['Bad_Fold_Imp'] * 100).round(2)
    diff_df = diff_df.sort_values('Diff', ascending=False)

    print("\n好Fold更看重的特征:")
    print(diff_df.head(15).to_string(index=False))

    print("\n差Fold更看重的特征:")
    print(diff_df.tail(15).to_string(index=False))

    # 保存结果
    output_file = os.path.join(output_dir, 'fold_feature_importance_analysis.csv')
    all_feat_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n特征重要性已保存到: {output_file}")

    # 保存稳定性分析
    stability_file = os.path.join(output_dir, 'feature_stability_analysis.csv')
    feature_rank_pivot.to_csv(stability_file, encoding='utf-8-sig')
    print(f"稳定性分析已保存到: {stability_file}")

    return all_feat_df, feature_rank_pivot

if __name__ == '__main__':
    analyze_fold_feature_importance()
