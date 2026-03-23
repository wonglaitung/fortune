#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择优化脚本 - 支持两种方法：
1. 统计方法：F-test + 互信息混合方法
2. 模型重要性法：基于LightGBM的特征重要性

从2936个特征中筛选出最有效的特征（500-1000个）
预期提升：0.5-1%准确率
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# 导入项目模块
from config import WATCHLIST as STOCK_LIST
from ml_services.ml_trading_model import MLTradingModel
from ml_services.logger_config import get_logger

logger = get_logger('feature_selection')


def load_training_data(horizon=20):
    """
    加载训练数据（复用现有模型的数据准备流程）

    返回:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    """
    logger.info("=" * 50)
    print("📊 加载训练数据")
    logger.info("=" * 50)

    # 创建模型实例
    model = MLTradingModel()

    # 准备数据（使用指定horizon）
    import random
    all_codes = list(STOCK_LIST.keys())
    codes = random.sample(all_codes, min(10, len(all_codes)))  # 随机选择10只股票以提高速度
    logger.info(f"准备加载 {len(codes)} 只股票的数据...")

    # 调用prepare_data方法（返回DataFrame）
    df = model.prepare_data(codes, horizon=horizon)

    # 先删除全为NaN的列
    cols_all_nan = df.columns[df.isnull().all()].tolist()
    if cols_all_nan:
        print(f"🗑️  删除 {len(cols_all_nan)} 个全为NaN的列")
        df = df.drop(columns=cols_all_nan)

    # 删除包含NaN的行
    df = df.dropna()

    # 确保数据按日期索引排序
    df = df.sort_index()

    # 获取特征列
    feature_columns = model.get_feature_columns(df)
    print(f"使用 {len(feature_columns)} 个特征")

    # 处理分类特征
    categorical_features = []
    for col in feature_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            print(f"  编码分类特征: {col}")
            categorical_features.append(col)
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 准备特征和标签
    X = df[feature_columns].values  # 转换为numpy数组
    y = df['Label'].values

    # 确保X是数值类型
    if X.dtype == object:
        print("   - 转换特征矩阵为数值类型...")
        X = np.array(X, dtype=np.float64)

    # 检查并处理异常值（在特征选择前清理）
    print("   - 检查特征矩阵中的异常值...")
    inf_mask = np.isinf(X)
    large_mask = np.abs(X) > 1e10
    invalid_mask = inf_mask | large_mask
    
    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        invalid_features = np.sum(invalid_mask, axis=0)
        print(f"   - 发现 {invalid_count} 个异常值（无穷大或过大），将替换为0")
        print(f"   - 受影响的特征数量: {np.sum(invalid_features > 0)}")
        # 将异常值替换为0
        X = X.copy()
        X[invalid_mask] = 0.0
    
    # 检查NaN值
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        print(f"   - 发现 {nan_count} 个NaN值，将替换为0")
        X = np.nan_to_num(X, nan=0.0)
    
    # 删除全为0的列（这些列可能是无效的）
    all_zero_cols = (X == 0).all(axis=0)
    if np.any(all_zero_cols):
        zero_col_count = np.sum(all_zero_cols)
        print(f"   - 删除 {zero_col_count} 个全为0的特征列")
        X = X[:, ~all_zero_cols]
        feature_columns = [feature_columns[i] for i in range(len(feature_columns)) if not all_zero_cols[i]]

    logger.info(f"数据加载完成")
    print(f"   - 样本数量: {len(X)}")
    print(f"   - 特征数量: {len(feature_columns)}")
    print(f"   - 目标变量分布: {y}")
    print("")

    return X, y, feature_columns


# ==================== 统计方法：F-test + 互信息 ====================

def feature_selection_f_test(X, y, k=1000):
    """
    使用F-test选择特征

    参数:
    - X: 特征矩阵
    - y: 目标变量
    - k: 选择的特征数量

    返回:
    - selected_features: 选择的特征索引
    - scores: F-test分数
    """
    logger.info("=" * 50)
    print("🔬 F-test特征选择")
    logger.info("=" * 50)

    # 检查并处理无穷大值和过大的值
    print("   - 检查异常值...")
    inf_mask = np.isinf(X)
    large_mask = np.abs(X) > 1e10
    invalid_mask = inf_mask | large_mask
    
    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        invalid_features = np.sum(invalid_mask, axis=0)
        print(f"   - 发现 {invalid_count} 个异常值，将替换为0")
        print(f"   - 受影响的特征数量: {np.sum(invalid_features > 0)}")
        X = X.copy()
        X[invalid_mask] = 0.0  # 将异常值替换为0
    
    # 检查NaN值
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        print(f"   - 发现 {nan_count} 个NaN值，将替换为0")
        X = np.nan_to_num(X, nan=0.0)

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)
    scores = selector.scores_

    # 处理NaN分数
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
        max_score = np.max(valid_scores)
    else:
        mean_score = 0
        max_score = 0

    logger.info(f"F-test选择完成")
    print(f"   - 选择特征数量: {len(selected_features)}")
    print(f"   - 平均F-test分数: {mean_score:.2f}")
    print(f"   - 最高F-test分数: {max_score:.2f}")
    print("")

    return selected_features, scores


def feature_selection_mutual_info(X, y, k=1000):
    """
    使用互信息选择特征

    参数:
    - X: 特征矩阵
    - y: 目标变量
    - k: 选择的特征数量

    返回:
    - selected_features: 选择的特征索引
    - scores: 互信息分数
    """
    logger.info("=" * 50)
    print("🔬 互信息特征选择")
    logger.info("=" * 50)

    # 检查并处理无穷大值和过大的值
    print("   - 检查异常值...")
    inf_mask = np.isinf(X)
    large_mask = np.abs(X) > 1e10
    invalid_mask = inf_mask | large_mask
    
    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        print(f"   - 发现 {invalid_count} 个异常值，将替换为0")
        X = X.copy()
        X[invalid_mask] = 0.0  # 将异常值替换为0
    
    # 检查NaN值
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        print(f"   - 发现 {nan_count} 个NaN值，将替换为0")
        X = np.nan_to_num(X, nan=0.0)

    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)
    scores = selector.scores_

    # 处理NaN分数
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
    else:
        mean_score = 0

    logger.info(f"互信息选择完成")
    print(f"   - 选择特征数量: {len(selected_features)}")
    print(f"   - 平均互信息分数: {mean_score:.4f}")
    print(f"   - 最高互信息分数: {np.max(scores):.4f}")
    print("")

    return selected_features, scores


def feature_selection_statistical(X, y, feature_names, top_k=500):
    """
    使用F-test + 互信息混合方法选择特征

    策略：
    1. 分别使用F-test和互信息选择top 1000特征
    2. 取两者的交集（约500-700个特征）
    3. 按综合得分排序，选择top 500特征

    参数:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    - top_k: 最终选择的特征数量

    返回:
    - selected_features: 选择的特征索引
    - feature_scores: 特征得分DataFrame
    """
    logger.info("=" * 50)
    print("🔬 F-test + 互信息混合特征选择（统计方法）")
    logger.info("=" * 50)

    # 1. F-test选择
    f_selected, f_scores = feature_selection_f_test(X, y, k=1000)

    # 2. 互信息选择
    mi_selected, mi_scores = feature_selection_mutual_info(X, y, k=1000)

    # 3. 取交集
    f_set = set(f_selected)
    mi_set = set(mi_selected)
    intersection = f_set.intersection(mi_set)

    logger.info(f"选择结果统计")
    print(f"   - F-test选择: {len(f_selected)} 个特征")
    print(f"   - 互信息选择: {len(mi_selected)} 个特征")
    print(f"   - 交集: {len(intersection)} 个特征")
    print("")

    # 4. 计算综合得分（归一化后平均）
    all_features = set(range(len(feature_names)))

    feature_data = []
    for idx in all_features:
        f_score = f_scores[idx] if idx in f_set else 0
        mi_score = mi_scores[idx] if idx in mi_set else 0

        # 归一化
        f_score_norm = f_score / np.max(f_scores) if np.max(f_scores) > 0 else 0
        mi_score_norm = mi_score / np.max(mi_scores) if np.max(mi_scores) > 0 else 0

        # 综合得分（平均）
        combined_score = (f_score_norm + mi_score_norm) / 2

        feature_data.append({
            'Feature_Index': idx,
            'Feature_Name': feature_names[idx],
            'F_Test_Score': f_score,
            'F_Test_Normalized': f_score_norm,
            'MI_Score': mi_score,
            'MI_Normalized': mi_score_norm,
            'Combined_Score': combined_score,
            'In_Intersection': idx in intersection
        })

    # 创建DataFrame
    feature_scores = pd.DataFrame(feature_data)

    # 5. 选择top_k特征
    feature_scores_sorted = feature_scores.sort_values('Combined_Score', ascending=False)
    selected_features = feature_scores_sorted.head(top_k)['Feature_Index'].values

    logger.info(f"混合选择完成")
    print(f"   - 最终选择特征数量: {len(selected_features)}")
    print(f"   - 交集特征数量: {feature_scores_sorted.head(top_k)['In_Intersection'].sum()}")
    print(f"   - 平均综合得分: {feature_scores_sorted.head(top_k)['Combined_Score'].mean():.4f}")
    print("")

    return selected_features, feature_scores_sorted


# ==================== 模型重要性法 ====================

def feature_selection_model_importance(X, y, feature_names, top_k=500):
    """
    使用LightGBM模型重要性选择特征

    策略：
    1. 训练LightGBM模型
    2. 获取特征重要性（gain-based）
    3. 选择top_k特征

    参数:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    - top_k: 最终选择的特征数量

    返回:
    - selected_features: 选择的特征索引
    - feature_scores: 特征得分DataFrame
    """
    logger.info("=" * 50)
    print("🔬 LightGBM模型重要性特征选择")
    logger.info("=" * 50)

    # 创建LightGBM数据集
    lgb_train = lgb.Dataset(X, y)

    # 使用简单参数快速训练
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # 训练模型
    print("训练LightGBM模型...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100  # 设置合理的迭代次数
    )

    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')

    logger.info(f"模型训练完成")
    print(f"   - 特征数量: {len(importance)}")
    print(f"   - 平均重要性: {np.mean(importance):.4f}")
    print(f"   - 最高重要性: {np.max(importance):.4f}")
    print("")

    # 创建DataFrame
    feature_data = []
    for idx, name in enumerate(feature_names):
        feature_data.append({
            'Feature_Index': idx,
            'Feature_Name': name,
            'Importance': importance[idx],
            'Importance_Normalized': importance[idx] / np.max(importance) if np.max(importance) > 0 else 0
        })

    feature_scores = pd.DataFrame(feature_data)

    # 按重要性排序并选择top_k
    feature_scores_sorted = feature_scores.sort_values('Importance', ascending=False)
    selected_features = feature_scores_sorted.head(top_k)['Feature_Index'].values

    logger.info(f"特征选择完成")
    print(f"   - 最终选择特征数量: {len(selected_features)}")
    print(f"   - 平均重要性: {feature_scores_sorted.head(top_k)['Importance'].mean():.4f}")
    print(f"   - 最低重要性: {feature_scores_sorted.head(top_k)['Importance'].min():.4f}")
    print("")

    return selected_features, feature_scores_sorted


def feature_selection_cumulative_importance(X, y, feature_names, score_method='f_test', target_importance=0.95, min_features=100, max_features=1000):
    """
    基于累积重要性自动决定特征数量
    
    策略：
    1. 使用F-test或互信息选择所有特征
    2. 按重要性排序
    3. 计算累积重要性
    4. 选择累积重要性达到目标的特征
    5. 自动决定特征数量（在min_features和max_features之间）

    参数:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    - score_method: 评分方法 ('f_test' 或 'mutual_info')
    - target_importance: 目标累积重要性 (0.95或0.99)
    - min_features: 最小特征数量
    - max_features: 最大特征数量

    返回:
    - selected_features: 选择的特征索引
    - feature_scores: 特征得分DataFrame
    """
    logger.info("=" * 50)
    print(f"🔬 基于累积重要性自动决定特征数量（{score_method}）")
    logger.info("=" * 50)
    
    # 1. 选择所有特征并获取分数
    if score_method == 'f_test':
        selector = SelectKBest(f_classif, k='all')
        X_selected = selector.fit_transform(X, y)
        scores = selector.scores_
        score_name = 'F_Test_Score'
    else:  # mutual_info
        selector = SelectKBest(mutual_info_classif, k='all')
        X_selected = selector.fit_transform(X, y)
        scores = selector.scores_
        score_name = 'MI_Score'
    
    # 2. 处理NaN和无限大值
    valid_mask = np.isfinite(scores)
    valid_indices = np.where(valid_mask)[0]
    
    logger.info(f"有效特征数量: {len(valid_indices)} / {len(scores)}")
    
    if len(valid_indices) < min_features:
        logger.warning(f"有效特征数量({len(valid_indices)})小于最小特征数({min_features})")
        # 使用所有有效特征
        min_features = max(50, len(valid_indices) // 2)
    
    # 只使用有效特征
    valid_scores = scores[valid_mask]
    valid_feature_names = [feature_names[i] for i in range(len(feature_names)) if valid_mask[i]]
    
    # 3. 按重要性排序
    sorted_indices = np.argsort(valid_scores)[::-1]  # 降序
    sorted_scores = valid_scores[sorted_indices]
    sorted_feature_names = [valid_feature_names[i] for i in sorted_indices]
    
    # 4. 计算累积重要性
    total_importance = np.sum(sorted_scores)
    cumulative_importance = np.cumsum(sorted_scores) / total_importance
    
    # 5. 找到达到目标累积重要性的特征数
    num_features = np.searchsorted(cumulative_importance, target_importance) + 1
    
    # 6. 确保特征数量在范围内
    num_features = max(num_features, min_features)
    num_features = min(num_features, max_features)
    num_features = min(num_features, len(valid_scores))
    
    # 7. 选择特征（使用原始索引）
    original_indices = valid_indices[sorted_indices[:num_features]]
    
    # 8. 创建DataFrame（使用原始特征名）
    feature_data = []
    for i in range(len(original_indices)):
        idx = original_indices[i]
        feature_data.append({
            'Feature_Index': idx,
            'Feature_Name': feature_names[idx],
            score_name: scores[idx],
            'Normalized_Score': scores[idx] / np.max(valid_scores) if np.max(valid_scores) > 0 else 0,
            'Cumulative_Importance': cumulative_importance[i] if i < len(cumulative_importance) else 1.0
        })
    
    feature_scores = pd.DataFrame(feature_data)
    
    logger.info(f"累积重要性特征选择完成")
    print(f"   - 目标累积重要性: {target_importance:.2%}")
    print(f"   - 实际累积重要性: {cumulative_importance[num_features-1]:.2%}")
    print(f"   - 自动选择特征数量: {num_features}")
    print(f"   - 最小特征数: {min_features}")
    print(f"   - 最大特征数: {max_features}")
    print(f"   - 平均{score_name}: {np.mean(sorted_scores[:num_features]):.4f}")
    print(f"   - 最高{score_name}: {np.max(sorted_scores[:num_features]):.4f}")
    print("")
    
    return original_indices, feature_scores


def evaluate_feature_selection(X, y, selected_features, feature_names):
    """
    评估特征选择效果

    参数:
    - X: 原始特征矩阵
    - y: 目标变量
    - selected_features: 选择的特征索引
    - feature_names: 特征名称列表

    返回:
    - performance: 性能指标
    """
    logger.info("=" * 50)
    print("📈 评估特征选择效果")
    logger.info("=" * 50)

    try:
        # 选择特征
        X_selected = X.iloc[:, selected_features]

        # 创建LightGBM数据集
        lgb_train = lgb.Dataset(X_selected, y)

        # 使用简单参数快速评估
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        # 使用cv评估
        cv_results = lgb.cv(
            params,
            lgb_train,
            num_boost_round=100,
            nfold=5,
            stratified=False
        )

        # 检查cv_results的键名
        logger.info(f"cv_results键名: {list(cv_results.keys())}")

        # 尝试找到正确的键名
        mean_key = None
        stdv_key = None
        for key in cv_results.keys():
            if 'mean' in key:
                mean_key = key
            if 'stdv' in key:
                stdv_key = key

        if mean_key and stdv_key:
            avg_logloss = np.mean(cv_results[mean_key])
            std_logloss = np.std(cv_results[stdv_key])

            logger.info(f"评估完成")
            print(f"   - 平均{mean_key}: {avg_logloss:.4f}")
            print(f"   - {stdv_key}: {std_logloss:.4f}")
            print("")

            performance = {
                'avg_logloss': avg_logloss,
                'std_logloss': std_logloss,
                'num_features': len(selected_features)
            }

            return performance
        else:
            logger.warning(f"无法找到正确的键名，使用默认值")
            performance = {
                'avg_logloss': 0.0,
                'std_logloss': 0.0,
                'num_features': len(selected_features)
            }
            return performance

    except Exception as e:
        logger.warning(f"评估失败: {e}")
        print("使用默认值继续...")
        performance = {
            'avg_logloss': 0.0,
            'std_logloss': 0.0,
            'num_features': len(selected_features)
        }
        return performance


def save_results(feature_scores, selected_features, output_dir='output', method_name='feature_selection'):
    """
    保存特征选择结果

    参数:
    - feature_scores: 特征得分DataFrame
    - selected_features: 选择的特征索引
    - output_dir: 输出目录
    - method_name: 方法名称
    """
    logger.info("=" * 50)
    print("💾 保存结果")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存特征得分
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feature_scores_path = os.path.join(output_dir, f'{method_name}_scores_{timestamp}.csv')
    feature_scores.to_csv(feature_scores_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征得分已保存至: {feature_scores_path}")

    # 保存选择的特征
    selected_features_path = os.path.join(output_dir, f'{method_name}_selected_{timestamp}.csv')
    selected_df = feature_scores[feature_scores['Feature_Index'].isin(selected_features)].copy()
    selected_df.to_csv(selected_features_path, index=False, encoding='utf-8-sig')
    logger.info(f"选择的特征已保存至: {selected_features_path}")

    # 保存特征名称列表（供ml_trading_model.py使用）
    feature_names_path = os.path.join(output_dir, f'{method_name}_features_{timestamp}.txt')
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        # 只保存特征名称，不保存索引
        feature_names = feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Feature_Name'].tolist()
        f.write('\n'.join(feature_names))
    logger.info(f"特征名称列表已保存至: {feature_names_path}")

    # 保存特征索引列表（方便后续使用）
    selected_indices_path = os.path.join(output_dir, f'{method_name}_indices_{timestamp}.txt')
    with open(selected_indices_path, 'w', encoding='utf-8') as f:
        f.write(','.join(map(str, selected_features)))
    logger.info(f"特征索引已保存至: {selected_indices_path}")

    # 保存最新选择的特征（不带时间戳，方便其他脚本引用）
    latest_features_path = os.path.join(output_dir, f'{method_name}_features_latest.txt')
    with open(latest_features_path, 'w', encoding='utf-8') as f:
        feature_names = feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Feature_Name'].tolist()
        f.write('\n'.join(feature_names))
    logger.info(f"最新特征名称列表已保存至: {latest_features_path}")

    print("")

    return {
        'scores_path': feature_scores_path,
        'selected_path': selected_features_path,
        'features_path': feature_names_path,
        'indices_path': selected_indices_path,
        'latest_path': latest_features_path
    }


def main():
    parser = argparse.ArgumentParser(description='特征选择优化 - 支持统计方法、模型重要性法和累积重要性法')
    parser.add_argument('--method', type=str, default='model',
                       choices=['statistical', 'model', 'cumulative_importance'],
                       help='特征选择方法: statistical(统计方法), model(模型重要性法), cumulative_importance(累积重要性法)')
    parser.add_argument('--top-k', type=int, default=500,
                       help='最终选择的特征数量 (默认: 500, 仅用于statistical和model方法)')
    parser.add_argument('--horizon', type=int, default=20,
                       choices=[1, 5, 20],
                       help='预测周期 (默认: 20)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录 (默认: output)')
    parser.add_argument('--target-importance', type=float, default=0.95,
                       help='目标累积重要性 (默认: 0.95, 仅用于cumulative_importance方法)')
    parser.add_argument('--min-features', type=int, default=100,
                       help='最小特征数量 (默认: 100, 仅用于cumulative_importance方法)')
    parser.add_argument('--max-features', type=int, default=1000,
                       help='最大特征数量 (默认: 1000, 仅用于cumulative_importance方法)')
    parser.add_argument('--score-method', type=str, default='f_test',
                       choices=['f_test', 'mutual_info'],
                       help='评分方法 (默认: f_test, 仅用于cumulative_importance方法)')

    args = parser.parse_args()

    method_name = {
        'statistical': 'statistical',
        'model': 'model_importance',
        'cumulative_importance': 'cumulative_importance'
    }[args.method]

    print("\n" + "=" * 80)
    print("🚀 特征选择优化开始")
    logger.info("=" * 50)
    print(f"⚙️  参数配置:")
    print(f"   - 特征选择方法: {args.method}")
    if args.method == 'cumulative_importance':
        print(f"   - 评分方法: {args.score_method}")
        print(f"   - 目标累积重要性: {args.target_importance:.2%}")
        print(f"   - 最小特征数: {args.min_features}")
        print(f"   - 最大特征数: {args.max_features}")
    else:
        print(f"   - 目标特征数量: {args.top_k}")
    print(f"   - 预测周期: {args.horizon}天")
    print(f"   - 输出目录: {args.output_dir}")
    print("")

    try:
        # 步骤1: 加载训练数据
        X, y, feature_names = load_training_data(horizon=args.horizon)

        # 步骤2: 根据方法选择特征
        if args.method == 'statistical':
            selected_features, feature_scores = feature_selection_statistical(
                X, y, feature_names, top_k=args.top_k
            )
        elif args.method == 'cumulative_importance':
            selected_features, feature_scores = feature_selection_cumulative_importance(
                X, y, feature_names,
                score_method=args.score_method,
                target_importance=args.target_importance,
                min_features=args.min_features,
                max_features=args.max_features
            )
        else:  # model
            selected_features, feature_scores = feature_selection_model_importance(
                X, y, feature_names, top_k=args.top_k
            )

        # 步骤3: 评估效果
        performance = evaluate_feature_selection(X, y, selected_features, feature_names)

        # 步骤4: 保存结果
        result_paths = save_results(feature_scores, selected_features, args.output_dir, method_name)

        # 步骤5: 显示Top 20特征
        logger.info("=" * 50)
        print("🏆 Top 20特征")
        logger.info("=" * 50)

        if args.method == 'statistical':
            print_cols = ['Feature_Name', 'Combined_Score', 'In_Intersection']
        elif args.method == 'cumulative_importance':
            if args.score_method == 'f_test':
                print_cols = ['Feature_Name', 'F_Test_Score', 'Cumulative_Importance']
            else:
                print_cols = ['Feature_Name', 'MI_Score', 'Cumulative_Importance']
        else:  # model
            print_cols = ['Feature_Name', 'Importance', 'Importance_Normalized']

        top_20 = feature_scores.head(20)
        print(top_20[print_cols].to_string(index=False))
        print("")

        logger.info("=" * 50)
        logger.info("特征选择优化完成！")
        logger.info("=" * 50)
        logger.info(f"优化总结:")
        print(f"   - 原始特征数量: {len(feature_names)}")
        print(f"   - 优化后特征数量: {len(selected_features)}")
        print(f"   - 特征减少比例: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
        if args.method == 'cumulative_importance':
            print(f"   - 目标累积重要性: {args.target_importance:.2%}")
            if args.score_method == 'f_test':
                print(f"   - 实际累积重要性: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Cumulative_Importance'].max():.2%}")
                print(f"   - 平均F-test分数: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['F_Test_Score'].mean():.4f}")
            else:
                print(f"   - 实际累积重要性: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Cumulative_Importance'].max():.2%}")
                print(f"   - 平均互信息分数: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['MI_Score'].mean():.4f}")
        elif args.method == 'statistical':
            print(f"   - 交集特征占比: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['In_Intersection'].sum()/len(selected_features)*100:.1f}%")
            print(f"   - 平均综合得分: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Combined_Score'].mean():.4f}")
        else:  # model
            print(f"   - 平均重要性: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Importance'].mean():.4f}")
            print(f"   - 最低重要性: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Importance'].min():.4f}")
        print("")
        print(f"💡 下一步:")
        print(f"   1. 检查选择的特征列表（保存到 {result_paths['features_path']}）")
        print(f"   2. 在ml_trading_model.py中集成特征选择逻辑")
        print(f"   3. 重新训练模型并评估准确率提升")
        print(f"   4. 对比优化前后的模型性能")
        print("")

    except Exception as e:
        logger.error(f"特征选择失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
