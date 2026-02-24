#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的特征评估脚本 - 修复数据问题

改进：
1. 移除高 NaN 比例的特征（NaN >= 30%）
2. 只使用无 NaN 的行
3. 支持更灵活的配置
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
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

# 导入项目模块
from config import WATCHLIST as STOCK_LIST
from ml_services.ml_trading_model import MLTradingModel
from ml_services.logger_config import get_logger

logger = get_logger('feature_eval_v2')


def evaluate_feature_importance_improved(
    model_path='output/ml_trading_model_lightgbm_20d.pkl',
    horizon=5,  # 默认使用5天周期
    num_stocks=10,  # 默认10只股票
    max_nan_ratio=0.3,  # 移除NaN >= 30%的特征
    output_dir='output/feature_eval_v2'
):
    """
    改进的特征重要性评估

    参数:
    - model_path: 模型文件路径
    - horizon: 预测周期
    - num_stocks: 股票数量
    - max_nan_ratio: 最大允许的NaN比例
    - output_dir: 输出目录
    """
    logger.info("=" * 70)
    logger.info("改进的特征重要性评估")
    logger.info("=" * 70)
    logger.info(f"配置: {num_stocks}只股票, {horizon}天周期")
    logger.info(f"特征过滤: NaN < {max_nan_ratio*100:.0f}%")

    os.makedirs(output_dir, exist_ok=True)

    # 创建模型实例
    model = MLTradingModel()
    model.horizon = horizon

    # 准备数据
    logger.info("\n步骤 1: 准备数据...")
    codes = list(STOCK_LIST.keys())[:num_stocks]
    logger.info(f"加载 {len(codes)} 只股票的数据...")

    df = model.prepare_data(codes, horizon=horizon)
    logger.info(f"原始数据形状: {df.shape}")

    # 获取特征列
    feature_columns = model.get_feature_columns(df)
    logger.info(f"原始特征数量: {len(feature_columns)}")

    # 步骤 2: 处理分类特征
    logger.info("\n步骤 2: 处理分类特征...")
    categorical_features = []
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                logger.debug(f"  编码分类特征: {col}")
                categorical_features.append(col)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # 步骤 3: 分析特征 NaN 分布
    logger.info("\n步骤 3: 分析特征 NaN 分布...")
    feature_nan_ratio = {}
    for col in feature_columns:
        if col in df.columns:
            nan_ratio = df[col].isnull().sum() / len(df)
            feature_nan_ratio[col] = nan_ratio

    nan_ratios = pd.Series(feature_nan_ratio)

    logger.info(f"特征 NaN 分布:")
    logger.info(f"  无 NaN 的特征: {(nan_ratios == 0).sum()}")
    logger.info(f"  NaN < {max_nan_ratio*100:.0f}% 的特征: {((nan_ratios > 0) & (nan_ratios < max_nan_ratio)).sum()}")
    logger.info(f"  NaN >= {max_nan_ratio*100:.0f}% 的特征: {(nan_ratios >= max_nan_ratio).sum()}")

    # 步骤 4: 过滤特征
    logger.info("\n步骤 4: 过滤高 NaN 特征...")
    valid_features = nan_ratios[nan_ratios < max_nan_ratio].index.tolist()
    logger.info(f"保留特征数量: {len(valid_features)}")

    if len(valid_features) < 100:
        logger.warning(f"特征数量过少（{len(valid_features)}），增加 max_nan_ratio...")
        max_nan_ratio = 0.8
        valid_features = nan_ratios[nan_ratios < max_nan_ratio].index.tolist()
        logger.info(f"调整后保留特征数量: {len(valid_features)}")

    # 步骤 5: 准备训练数据
    logger.info("\n步骤 5: 准备训练数据...")

    # 只保留有效特征
    X = df[valid_features].copy()

    # 检查标签列
    if 'Label' not in df.columns:
        logger.error("标签列 'Label' 不存在")
        return None

    y = df['Label']

    # 移除标签为 NaN 的行
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"移除标签NaN后样本数: {len(X)}")

    # 移除全为 NaN 的行
    all_nan_rows = X.isnull().all(axis=1)
    X = X[~all_nan_rows]
    y = y[~all_nan_rows]

    logger.info(f"移除全NaN行后样本数: {len(X)}")

    # 填充剩余的 NaN（使用中位数）
    logger.info(f"填充剩余 NaN 值...")
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            logger.debug(f"  填充特征 {col}: {X[col].isnull().sum()} 个 NaN -> 中位数 {median_val:.4f}")

    # 最终检查
    logger.info(f"\n最终数据集:")
    logger.info(f"  样本数量: {len(X)}")
    logger.info(f"  特征数量: {len(X.columns)}")
    logger.info(f"  标签分布: {y.value_counts().to_dict()}")

    if len(X) < 100:
        logger.error(f"样本数量不足（{len(X)} < 100），无法进行评估")
        return None

    # 步骤 6: 交叉验证特征重要性
    logger.info("\n" + "=" * 70)
    logger.info("步骤 6: 交叉验证特征重要性")
    logger.info("=" * 70)

    tscv = TimeSeriesSplit(n_splits=3)
    feature_importance_dict = {name: [] for name in X.columns}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"Fold {fold + 1}/3...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 训练模型
        fold_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.03,
            verbose=-1,
            random_state=42 + fold
        )
        fold_model.fit(X_train, y_train)

        # 记录特征重要性
        importance = fold_model.feature_importances_
        for i, name in enumerate(X.columns):
            feature_importance_dict[name].append(importance[i])

        # 验证准确率
        y_pred = fold_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logger.info(f"  验证准确率: {acc:.4f}")

    # 步骤 7: 计算统计信息
    logger.info("\n步骤 7: 计算特征重要性统计...")
    importance_stats = []

    for name, scores in feature_importance_dict.items():
        importance_stats.append({
            'Feature': name,
            'Mean_Importance': np.mean(scores),
            'Std_Importance': np.std(scores),
            'CV_Importance': np.std(scores) / (np.mean(scores) + 1e-6),
            'Min_Importance': np.min(scores),
            'Max_Importance': np.max(scores)
        })

    importance_df = pd.DataFrame(importance_stats)
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)

    # 保存结果
    output_csv = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"\n特征重要性已保存: {output_csv}")

    # 生成报告
    logger.info("\n步骤 8: 生成评估报告...")
    report_path = os.path.join(output_dir, 'feature_evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("特征重要性评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置: {num_stocks}只股票, {horizon}天周期\n")
        f.write(f"特征过滤: NaN < {max_nan_ratio*100:.0f}%\n\n")

        f.write(f"数据集统计:\n")
        f.write(f"  样本数量: {len(X)}\n")
        f.write(f"  特征数量: {len(X.columns)}\n")
        f.write(f"  原始特征数: {len(feature_columns)}\n")
        f.write(f"  过滤特征数: {len(feature_columns) - len(X.columns)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Top 50 特征（按平均重要性排序）\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'排名':<6} {'特征名称':<50} {'平均重要性':<15} {'变异系数':<15}\n")
        f.write("-" * 80 + "\n")

        for i, row in importance_df.head(50).iterrows():
            rank = i + 1
            f.write(f"{rank:<6} {row['Feature']:<50} {row['Mean_Importance']:<15.2f} {row['CV_Importance']:<15.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("特征选择建议\n")
        f.write("=" * 80 + "\n\n")

        # 计算建议的阈值
        top_100_threshold = importance_df.iloc[99]['Mean_Importance'] if len(importance_df) >= 100 else 0
        top_200_threshold = importance_df.iloc[199]['Mean_Importance'] if len(importance_df) >= 200 else 0

        f.write(f"1. 保留 Top 100 特征:\n")
        f.write(f"   - 阈值: 重要性 >= {top_100_threshold:.2f}\n")
        f.write(f"   - 数量: 100 个\n\n")

        f.write(f"2. 保留 Top 200 特征:\n")
        f.write(f"   - 阈值: 重要性 >= {top_200_threshold:.2f}\n")
        f.write(f"   - 数量: 200 个\n\n")

        # 找出最稳定的特征
        stable_features = importance_df.nsmallest(20, 'CV_Importance')
        f.write(f"3. 最稳定的 Top 20 特征（最低变异系数）:\n")
        for i, row in stable_features.iterrows():
            f.write(f"   {i+1:2d}. {row['Feature']:<50} CV={row['CV_Importance']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("详细分析\n")
        f.write("=" * 80 + "\n\n")

        # 计算特征重要性分布
        f.write(f"特征重要性分布:\n")
        f.write(f"  - 平均值: {importance_df['Mean_Importance'].mean():.2f}\n")
        f.write(f"  - 中位数: {importance_df['Mean_Importance'].median():.2f}\n")
        f.write(f"  - 标准差: {importance_df['Mean_Importance'].std():.2f}\n")
        f.write(f"  - 最大值: {importance_df['Mean_Importance'].max():.2f}\n")
        f.write(f"  - 最小值: {importance_df['Mean_Importance'].min():.2f}\n\n")

    logger.info(f"评估报告已保存: {report_path}")

    # 打印 Top 20
    logger.info("\n" + "=" * 70)
    logger.info("Top 20 特征:")
    logger.info("=" * 70)
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"{i+1:3d}. {row['Feature']:<50} {row['Mean_Importance']:>10.2f} (CV={row['CV_Importance']:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info("特征重要性评估完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return importance_df


def main():
    """主函数"""
    logger.info("开始改进的特征重要性评估...")

    # 运行评估
    importance_df = evaluate_feature_importance_improved(
        horizon=5,  # 使用5天周期
        num_stocks=10,  # 使用10只股票
        max_nan_ratio=0.3,  # 移除NaN >= 30%的特征
        output_dir='output/feature_eval_v2'
    )

    if importance_df is not None:
        logger.info("\n✅ 评估成功完成！")
        logger.info("可以查看以下文件:")
        logger.info("  - output/feature_eval_v2/feature_importance.csv")
        logger.info("  - output/feature_eval_v2/feature_evaluation_report.txt")
    else:
        logger.error("\n❌ 评估失败，请检查日志")


if __name__ == '__main__':
    main()
