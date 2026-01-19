#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两种模型的预测结果
"""

import pandas as pd

# 读取两种模型的预测结果
lgbm_pred = pd.read_csv('data/lgbm_model_predictions.csv')
gbdt_lr_pred = pd.read_csv('data/gbdt_lr_model_predictions.csv')

# 合并数据
comparison = lgbm_pred.merge(
    gbdt_lr_pred,
    on='code',
    suffixes=('_lgbm', '_gbdt_lr')
)

# 重命名列
comparison.columns = ['代码', '名称_LGBM', '预测_LGBM', '概率_LGBM', '价格', '日期_LGBM', '目标_LGBM',
                     '名称_GBDT_LR', '预测_GBDT_LR', '概率_GBDT_LR', '价格_GBDT_LR', '日期_GBDT_LR', '目标_GBDT_LR']

# 计算预测一致性
comparison['预测一致'] = comparison['预测_LGBM'] == comparison['预测_GBDT_LR']

# 计算概率差异
comparison['概率差异'] = abs(comparison['概率_LGBM'] - comparison['概率_GBDT_LR'])

# 排序
comparison = comparison.sort_values('概率差异', ascending=False)

# 显示对比结果
print("=" * 120)
print("两种模型预测结果对比")
print("=" * 120)
print(f"\n{'代码':<10} {'股票名称':<12} {'LGBM预测':<10} {'LGBM概率':<10} {'GBDT+LR预测':<12} {'GBDT+LR概率':<12} {'是否一致':<8} {'概率差异':<10}")
print("-" * 120)

for _, row in comparison.iterrows():
    lgbm_pred_label = "上涨" if row['预测_LGBM'] == 1 else "下跌"
    gbdt_lr_pred_label = "上涨" if row['预测_GBDT_LR'] == 1 else "下跌"
    consistent = "✓" if row['预测一致'] else "✗"

    print(f"{row['代码']:<10} {row['名称_LGBM']:<12} {lgbm_pred_label:<10} {row['概率_LGBM']:<10.4f} {gbdt_lr_pred_label:<12} {row['概率_GBDT_LR']:<12.4f} {consistent:<8} {row['概率差异']:<10.4f}")

# 统计
print("\n" + "=" * 120)
print("统计摘要")
print("=" * 120)

consistent_count = comparison['预测一致'].sum()
total_count = len(comparison)
print(f"\n预测一致性: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)")

lgbm_up = (comparison['预测_LGBM'] == 1).sum()
lgbm_down = (comparison['预测_LGBM'] == 0).sum()
print(f"\nLGBM 模型: 上涨 {lgbm_up} 只, 下跌 {lgbm_down} 只")

gbdt_lr_up = (comparison['预测_GBDT_LR'] == 1).sum()
gbdt_lr_down = (comparison['预测_GBDT_LR'] == 0).sum()
print(f"GBDT+LR 模型: 上涨 {gbdt_lr_up} 只, 下跌 {gbdt_lr_down} 只")

avg_prob_diff = comparison['概率差异'].mean()
print(f"\n平均概率差异: {avg_prob_diff:.4f}")

# 显示不一致的预测
inconsistent = comparison[~comparison['预测一致']]
if len(inconsistent) > 0:
    print("\n" + "=" * 120)
    print("预测不一致的股票")
    print("=" * 120)
    for _, row in inconsistent.iterrows():
        lgbm_pred_label = "上涨" if row['预测_LGBM'] == 1 else "下跌"
        gbdt_lr_pred_label = "上涨" if row['预测_GBDT_LR'] == 1 else "下跌"
        print(f"{row['代码']:<10} {row['名称_LGBM']:<12} LGBM: {lgbm_pred_label} ({row['概率_LGBM']:.4f})  vs  GBDT+LR: {gbdt_lr_pred_label} ({row['概率_GBDT_LR']:.4f})")