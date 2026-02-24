#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的特征评估脚本 - 不依赖 SHAP

功能：
1. 使用 LightGBM 内置特征重要性
2. 交叉验证评估特征重要性稳定性
3. 生成特征评估报告
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

logger = get_logger('simple_feature_eval')


def evaluate_feature_importance_simple(model_path, horizon=20, output_dir='output/feature_eval'):
    """
    简化的特征重要性评估

    参数:
    - model_path: 模型文件路径
    - horizon: 预测周期
    - output_dir: 输出目录
    """
    logger.info("=" * 70)
    logger.info("简化特征重要性评估")
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 创建模型实例
    model = MLTradingModel()
    model.horizon = horizon

    # 准备数据
    logger.info("准备训练数据...")
    codes = list(STOCK_LIST.keys())[:5]  # 使用5只股票作为示例
    df = model.prepare_data(codes, horizon=horizon)

    df = df.dropna()
    df = df.sort_index()

    # 获取特征列
    feature_columns = model.get_feature_columns(df)
    logger.info(f"特征数量: {len(feature_columns)}")

    # 处理分类特征
    for col in feature_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[feature_columns]
    y = df['Label']

    logger.info(f"原始样本数量: {len(X)}")

    # 检查数据量
    if len(X) < 100:
        logger.warning(f"数据量不足（{len(X)} 条），增加股票数量...")
        codes = list(STOCK_LIST.keys())[:20]  # 增加到20只股票
        df = model.prepare_data(codes, horizon=horizon)
        df = df.dropna()
        df = df.sort_index()

        # 获取特征列
        feature_columns = model.get_feature_columns(df)

        # 处理分类特征
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        X = df[feature_columns]
        y = df['Label']
        logger.info(f"增加后样本数量: {len(X)}")

    if len(X) < 50:
        logger.error(f"数据量仍然不足（{len(X)} 条），跳过评估")
        return None

    # 交叉验证评估特征重要性
    logger.info("\n" + "=" * 70)
    logger.info("交叉验证特征重要性")
    logger.info("=" * 70)

    tscv = TimeSeriesSplit(n_splits=3)
    feature_importance_dict = {name: [] for name in feature_columns}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"Fold {fold + 1}/3...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 训练模型
        fold_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.03,
            verbose=-1
        )
        fold_model.fit(X_train, y_train)

        # 记录特征重要性
        importance = fold_model.feature_importances_
        for i, name in enumerate(feature_columns):
            feature_importance_dict[name].append(importance[i])

        # 验证准确率
        y_pred = fold_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logger.info(f"  验证准确率: {acc:.4f}")

    # 计算统计信息
    logger.info("\n计算特征重要性统计...")
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
    output_csv = os.path.join(output_dir, 'feature_importance_simple.csv')
    importance_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"\n特征重要性已保存: {output_csv}")

    # 生成报告
    report_path = os.path.join(output_dir, 'feature_evaluation_simple_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("简化特征重要性评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"特征总数: {len(importance_df)}\n")
        f.write(f"样本数量: {len(X)}\n")
        f.write(f"交叉验证折数: 3\n\n")

        f.write("=" * 80 + "\n")
        f.write("Top 30 特征（按平均重要性排序）\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'排名':<6} {'特征名称':<50} {'平均重要性':<15} {'变异系数':<15}\n")
        f.write("-" * 80 + "\n")

        for i, row in importance_df.head(30).iterrows():
            f.write(f"{i+1:<6} {row['Feature']:<50} {row['Mean_Importance']:<15.2f} {row['CV_Importance']:<15.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("特征选择建议\n")
        f.write("=" * 80 + "\n\n")

        # 计算重要性阈值
        top_50_threshold = importance_df.iloc[49]['Mean_Importance'] if len(importance_df) >= 50 else 0
        top_100_threshold = importance_df.iloc[99]['Mean_Importance'] if len(importance_df) >= 100 else 0

        f.write(f"1. 保留 Top 50 特征:\n")
        f.write(f"   - 阈值: 重要性 >= {top_50_threshold:.2f}\n")
        f.write(f"   - 数量: 50 个\n\n")

        f.write(f"2. 保留 Top 100 特征:\n")
        f.write(f"   - 阈值: 重要性 >= {top_100_threshold:.2f}\n")
        f.write(f"   - 数量: 100 个\n\n")

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

        # 高变异系数特征
        high_cv_features = importance_df.nlargest(10, 'CV_Importance')
        f.write(f"4. 变异系数最高的 10 个特征（最不稳定）:\n")
        for i, row in high_cv_features.iterrows():
            f.write(f"   {i+1:2d}. {row['Feature']:<50} CV={row['CV_Importance']:.4f}\n")

    logger.info(f"\n评估报告已保存: {report_path}")

    # 打印 Top 20
    logger.info("\n" + "=" * 70)
    logger.info("Top 20 特征:")
    logger.info("=" * 70)
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"{i+1:3d}. {row['Feature']:<50} {row['Mean_Importance']:>10.2f} (CV={row['CV_Importance']:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info("评估完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return importance_df


if __name__ == '__main__':
    logger.info("开始简化特征重要性评估...")

    importance_df = evaluate_feature_importance_simple(
        model_path='output/ml_trading_model_lightgbm_20d.pkl',
        horizon=20,
        output_dir='output/feature_eval_simple'
    )

    logger.info("\n提示: 如需使用 SHAP 进行更深入的分析，请运行:")
    logger.info("  pip install shap")
    logger.info("  python scripts/feature_evaluation.py")
