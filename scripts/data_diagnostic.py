#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据诊断脚本 - 分析特征数据问题

功能：
1. 分析特征的 NaN 分布
2. 识别高 NaN 比例的特征
3. 建议数据处理策略
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# 导入项目模块
from config import WATCHLIST as STOCK_LIST
from ml_services.ml_trading_model import MLTradingModel
from ml_services.logger_config import get_logger

logger = get_logger('data_diagnostic')


def analyze_data_quality(model, codes, horizon):
    """
    分析数据质量

    参数:
    - model: 模型实例
    - codes: 股票代码列表
    - horizon: 预测周期

    返回:
    - analysis: 分析结果字典
    """
    logger.info("=" * 70)
    logger.info("数据质量分析")
    logger.info("=" * 70)

    # 准备数据
    logger.info(f"准备数据（{len(codes)} 只股票，周期 {horizon} 天）...")
    df = model.prepare_data(codes, horizon=horizon)

    logger.info(f"原始数据形状: {df.shape}")

    # 分析 1: 特征列分布
    feature_columns = model.get_feature_columns(df)
    logger.info(f"\n特征数量: {len(feature_columns)}")

    # 分析 2: 每个特征的 NaN 比例
    logger.info("\n分析特征 NaN 分布...")
    feature_nan_ratio = {}
    for col in feature_columns:
        if col in df.columns:
            nan_ratio = df[col].isnull().sum() / len(df)
            feature_nan_ratio[col] = nan_ratio

    # 统计
    nan_ratios = pd.Series(feature_nan_ratio)

    logger.info(f"\nNaN 分布统计:")
    logger.info(f"  无 NaN 的特征: {(nan_ratios == 0).sum()}")
    logger.info(f"  NaN < 10% 的特征: {((nan_ratios > 0) & (nan_ratios < 0.1)).sum()}")
    logger.info(f"  NaN 10-50% 的特征: {((nan_ratios >= 0.1) & (nan_ratios < 0.5)).sum()}")
    logger.info(f"  NaN >= 50% 的特征: {(nan_ratios >= 0.5).sum()}")

    # 找出高 NaN 比例的特征
    high_nan_features = nan_ratios[nan_ratios >= 0.5].sort_values(ascending=False)

    if len(high_nan_features) > 0:
        logger.info(f"\nNaN >= 50% 的特征（Top 20）:")
        for i, (feat, ratio) in enumerate(high_nan_features.head(20).items(), 1):
            logger.info(f"  {i:2d}. {feat:<50} {ratio*100:6.2f}%")

    # 分析 3: 行级别的 NaN 分布
    logger.info("\n分析行级别 NaN 分布...")
    row_nan_counts = df[feature_columns].isnull().sum(axis=1)

    logger.info(f"\n行 NaN 统计:")
    logger.info(f"  无 NaN 的行: {(row_nan_counts == 0).sum()} ({(row_nan_counts == 0).sum()/len(df)*100:.1f}%)")
    logger.info(f"  NaN < 10% 的行: {((row_nan_counts > 0) & (row_nan_counts < len(feature_columns)*0.1)).sum()}")
    logger.info(f"  NaN 10-50% 的行: {((row_nan_counts >= len(feature_columns)*0.1) & (row_nan_counts < len(feature_columns)*0.5)).sum()}")
    logger.info(f"  NaN >= 50% 的行: {(row_nan_counts >= len(feature_columns)*0.5).sum()}")

    # 分析 4: 标签分布
    logger.info(f"\n标签分布:")
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        logger.info(f"  上涨 (Label=1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        logger.info(f"  下跌 (Label=0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")

        # 检查标签是否在 NaN 行中
        nan_label_count = df['Label'].isnull().sum()
        logger.info(f"  标签 NaN 数量: {nan_label_count}")

    # 分析 5: dropna 的影响
    logger.info("\n分析 dropna 影响...")
    df_after_dropna = df.dropna(subset=feature_columns + ['Label'])
    logger.info(f"  dropna 后样本数: {len(df_after_dropna)}")
    logger.info(f"  dropna 损失率: {(len(df) - len(df_after_dropna))/len(df)*100:.1f}%")

    if len(df_after_dropna) < 50:
        logger.warning("\n⚠️  dropna 后样本严重不足！")

        # 尝试不同的 dropna 策略
        logger.info("\n尝试不同的数据清理策略:")

        # 策略 1: 只移除全 NaN 的行
        all_nan_rows = df[feature_columns].isnull().all(axis=1)
        df_strategy1 = df[~all_nan_rows].dropna(subset=['Label'])
        logger.info(f"  策略1 (移除全NaN行): {len(df_strategy1)} 样本")

        # 策略 2: 移除高 NaN 特征
        low_nan_features = nan_ratios[nan_ratios < 0.3].index.tolist()
        df_strategy2 = df.dropna(subset=low_nan_features + ['Label'])
        logger.info(f"  策略2 (移除高NaN特征): {len(low_nan_features)} 特征, {len(df_strategy2)} 样本")

        # 策略 3: 组合
        df_strategy3 = df[~all_nan_rows].dropna(subset=low_nan_features + ['Label'])
        logger.info(f"  策略3 (组合): {len(low_nan_features)} 特征, {len(df_strategy3)} 样本")

    # 返回分析结果
    analysis = {
        'total_features': len(feature_columns),
        'high_nan_features': high_nan_features,
        'sample_count_before': len(df),
        'sample_count_after': len(df_after_dropna),
        'dropna_loss_rate': (len(df) - len(df_after_dropna))/len(df),
        'recommended_strategy': 'use_low_nan_features' if len(df_after_dropna) < 50 else 'standard'
    }

    return analysis


def suggest_solutions(analysis):
    """建议解决方案"""
    logger.info("\n" + "=" * 70)
    logger.info("解决方案建议")
    logger.info("=" * 70)

    if analysis['sample_count_after'] >= 100:
        logger.info("\n✅ 数据量充足，可以使用标准 dropna 策略")
        return

    logger.info("\n⚠️  数据量不足，建议采用以下方案:")

    logger.info("\n方案 1: 移除高 NaN 比例的特征")
    logger.info("  - 移除 NaN >= 30% 的特征")
    logger.info("  - 保留低 NaN 特征")
    logger.info("  - 预期效果: 减少特征数量，增加有效样本")

    logger.info("\n方案 2: 使用更少的股票，更短的周期")
    logger.info("  - 减少股票数量到 5-10 只")
    logger.info("  - 使用更短的预测周期（1天或5天）")
    logger.info("  - 预期效果: 数据质量更高，特征计算更准确")

    logger.info("\n方案 3: 调整数据准备逻辑")
    logger.info("  - 在特征计算阶段填充缺失值")
    logger.info("  - 使用插值方法补全缺失值")
    logger.info("  - 预期效果: 减少数据损失")

    logger.info("\n方案 4: 增加历史数据范围")
    logger.info("  - 使用更多历史数据（3年或5年）")
    logger.info("  - 增加数据密度")
    logger.info("  - 预期效果: 更多的有效样本")


def main():
    """主函数"""
    logger.info("开始数据诊断...")

    # 创建模型实例
    model = MLTradingModel()

    # 测试不同的配置
    configurations = [
        {'codes': list(STOCK_LIST.keys())[:5], 'horizon': 1, 'name': '5只股票, 1天周期'},
        {'codes': list(STOCK_LIST.keys())[:5], 'horizon': 5, 'name': '5只股票, 5天周期'},
        {'codes': list(STOCK_LIST.keys())[:5], 'horizon': 20, 'name': '5只股票, 20天周期'},
        {'codes': list(STOCK_LIST.keys())[:10], 'horizon': 1, 'name': '10只股票, 1天周期'},
        {'codes': list(STOCK_LIST.keys())[:10], 'horizon': 5, 'name': '10只股票, 5天周期'},
    ]

    results = []

    for config in configurations:
        logger.info(f"\n{'='*70}")
        logger.info(f"配置: {config['name']}")
        logger.info(f"{'='*70}")

        try:
            analysis = analyze_data_quality(
                model,
                config['codes'],
                config['horizon']
            )
            results.append({
                'config': config['name'],
                'analysis': analysis
            })
        except Exception as e:
            logger.error(f"分析失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("诊断总结")
    logger.info("=" * 70)

    logger.info("\n各配置数据质量对比:")
    logger.info(f"{'配置':<20} {'特征数':<10} {'dropna前':<12} {'dropna后':<12} {'损失率':<10}")
    logger.info("-" * 70)

    best_config = None
    max_samples = 0

    for result in results:
        config_name = result['config']
        analysis = result['analysis']

        logger.info(
            f"{config_name:<20} "
            f"{analysis['total_features']:<10} "
            f"{analysis['sample_count_before']:<12} "
            f"{analysis['sample_count_after']:<12} "
            f"{analysis['dropna_loss_rate']*100:<9.1f}%"
        )

        if analysis['sample_count_after'] > max_samples:
            max_samples = analysis['sample_count_after']
            best_config = config_name

    if best_config:
        logger.info(f"\n✅ 推荐配置: {best_config}")
        logger.info(f"   有效样本数: {max_samples}")

    # 建议解决方案
    if results:
        suggest_solutions(results[-1]['analysis'])

    logger.info("\n" + "=" * 70)
    logger.info("诊断完成")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
