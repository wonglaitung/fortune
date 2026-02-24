#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择使用示例脚本

演示如何使用特征选择评估结果来训练和评估模型
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from ml_services.ml_trading_model import MLTradingModel
from config import TRAINING_STOCKS
from ml_services.logger_config import get_logger

logger = get_logger('feature_selection_example')


def load_feature_list(csv_path, n_features=100):
    """
    从特征重要性 CSV 加载特征列表

    参数:
    - csv_path: CSV 文件路径
    - n_features: 选择的特征数量

    返回:
    - features: 特征名称列表
    """
    df = pd.read_csv(csv_path)
    features = df.head(n_features)['Feature'].tolist()
    logger.info(f"从 {csv_path} 加载了 {len(features)} 个特征")
    return features


def train_with_feature_selection(feature_list_path, n_features=100, horizon=5):
    """
    使用特征选择训练模型

    参数:
    - feature_list_path: 特征重要性 CSV 路径
    - n_features: 选择的特征数量
    - horizon: 预测周期
    """
    logger.info("=" * 70)
    logger.info("使用特征选择训练模型")
    logger.info("=" * 70)

    # 加载特征列表
    feature_list = load_feature_list(feature_list_path, n_features)

    # 创建模型实例
    model = MLTradingModel()
    model.horizon = horizon
    model.use_feature_selection = True
    model.selected_features = feature_list

    # 获取训练股票
    codes = TRAINING_STOCKS[:10]  # 使用前10只股票作为示例
    logger.info(f"使用 {len(codes)} 只股票进行训练")

    # 训练模型
    logger.info("\n开始训练...")
    model.train(codes, horizon=horizon)

    # 获取验证准确率
    if hasattr(model, 'validation_accuracy'):
        logger.info(f"\n验证准确率: {model.validation_accuracy:.4f}")

    return model


def compare_feature_sets(feature_list_path, horizon=5):
    """
    对比不同特征集的性能

    参数:
    - feature_list_path: 特征重要性 CSV 路径
    - horizon: 预测周期
    """
    logger.info("=" * 70)
    logger.info("对比不同特征集的性能")
    logger.info("=" * 70)

    # 定义不同的特征集
    feature_configs = [
        {'name': '全部特征', 'n_features': None},
        {'name': 'Top 50', 'n_features': 50},
        {'name': 'Top 100', 'n_features': 100},
        {'name': 'Top 200', 'n_features': 200},
    ]

    results = []
    codes = TRAINING_STOCKS[:10]

    for config in feature_configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"训练模型: {config['name']}")
        logger.info(f"{'='*70}")

        model = MLTradingModel()
        model.horizon = horizon

        if config['n_features']:
            feature_list = load_feature_list(feature_list_path, config['n_features'])
            model.use_feature_selection = True
            model.selected_features = feature_list

        # 训练模型
        try:
            model.train(codes, horizon=horizon)

            # 获取结果
            accuracy = getattr(model, 'validation_accuracy', 0)
            results.append({
                '特征集': config['name'],
                '特征数量': config['n_features'] if config['n_features'] else 3972,
                '验证准确率': accuracy
            })

            logger.info(f"验证准确率: {accuracy:.4f}")

        except Exception as e:
            logger.error(f"训练失败: {e}")
            results.append({
                '特征集': config['name'],
                '特征数量': config['n_features'] if config['n_features'] else 3972,
                '验证准确率': 0
            })

    # 打印对比结果
    logger.info("\n" + "=" * 70)
    logger.info("性能对比结果")
    logger.info("=" * 70)

    results_df = pd.DataFrame(results)
    logger.info(f"\n{results_df.to_string(index=False)}")

    # 保存结果
    output_path = 'output/feature_comparison_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n对比结果已保存: {output_path}")

    return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征选择使用示例')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['train', 'compare', 'analyze'],
                       help='运行模式: train(训练), compare(对比), analyze(分析)')
    parser.add_argument('--feature_csv', type=str,
                       default='output/feature_eval_v2/feature_importance.csv',
                       help='特征重要性 CSV 路径')
    parser.add_argument('--n_features', type=int, default=100,
                       help='选择的特征数量')
    parser.add_argument('--horizon', type=int, default=5,
                       help='预测周期')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练模式：使用特征选择训练单个模型
        logger.info("运行模式: 训练")
        model = train_with_feature_selection(
            args.feature_csv,
            args.n_features,
            args.horizon
        )

        # 保存模型
        model_path = f'output/model_with_top{args.n_features}.pkl'
        model.save_model(model_path)
        logger.info(f"\n模型已保存: {model_path}")

    elif args.mode == 'compare':
        # 对比模式：对比不同特征集的性能
        logger.info("运行模式: 对比")
        compare_feature_sets(args.feature_csv, args.horizon)

    elif args.mode == 'analyze':
        # 分析模式：分析特征重要性
        logger.info("运行模式: 分析")

        df = pd.read_csv(args.feature_csv)

        logger.info(f"\n特征总数: {len(df)}")

        logger.info(f"\nTop 10 特征:")
        for i, row in df.head(10).iterrows():
            logger.info(f"  {i+1:2d}. {row['Feature']:<40} {row['Mean_Importance']:>8.2f} (CV={row['CV_Importance']:.4f})")

        # 统计分析
        logger.info(f"\n特征重要性分布:")
        logger.info(f"  平均值: {df['Mean_Importance'].mean():.2f}")
        logger.info(f"  中位数: {df['Mean_Importance'].median():.2f}")
        logger.info(f"  标准差: {df['Mean_Importance'].std():.2f}")
        logger.info(f"  最大值: {df['Mean_Importance'].max():.2f}")
        logger.info(f"  最小值: {df['Mean_Importance'].min():.2f}")

        # 变异系数分析
        logger.info(f"\n变异系数 (CV) 分布:")
        logger.info(f"  平均值: {df['CV_Importance'].mean():.4f}")
        logger.info(f"  中位数: {df['CV_Importance'].median():.4f}")
        logger.info(f"  最小值: {df['CV_Importance'].min():.4f}")
        logger.info(f"  最大值: {df['CV_Importance'].max():.4f}")

        # 特征数量建议
        logger.info(f"\n特征数量建议:")
        for n in [50, 100, 200, 500]:
            threshold = df.iloc[n-1]['Mean_Importance'] if len(df) >= n else 0
            logger.info(f"  Top {n}: 阈值 >= {threshold:.2f}")

    logger.info("\n✅ 完成！")


if __name__ == '__main__':
    main()
