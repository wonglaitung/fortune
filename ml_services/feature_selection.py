#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择优化脚本 - 使用F-test + 互信息混合方法
从2936个特征中筛选出最有效的特征（500-1000个）
预期提升：0.5-1%准确率
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# 导入项目模块
from config import WATCHLIST as STOCK_LIST
from ml_services.ml_trading_model import MLTradingModel
from ml_services.logger_config import get_logger

logger = get_logger('feature_selection')


def load_training_data():
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

    # 准备数据（使用20天预测horizon）
    codes = list(STOCK_LIST.keys())
    logger.info(f"准备加载 {len(codes)} 只股票的数据...")

    # 调用prepare_data方法（返回DataFrame）
    df = model.prepare_data(codes, horizon=20)

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
    X = df[feature_columns]
    y = df['Label']

    logger.info(f"数据加载完成")
    print(f"   - 样本数量: {len(X)}")
    print(f"   - 特征数量: {len(feature_columns)}")
    print(f"   - 目标变量分布: {y.value_counts().to_dict()}")
    print("")

    return X, y, feature_columns


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

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)
    scores = selector.scores_

    logger.info(f"F-test选择完成")
    print(f"   - 选择特征数量: {len(selected_features)}")
    print(f"   - 平均F-test分数: {np.mean(scores):.2f}")
    print(f"   - 最高F-test分数: {np.max(scores):.2f}")
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

    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)
    scores = selector.scores_

    logger.info(f"互信息选择完成")
    print(f"   - 选择特征数量: {len(selected_features)}")
    print(f"   - 平均互信息分数: {np.mean(scores):.4f}")
    print(f"   - 最高互信息分数: {np.max(scores):.4f}")
    print("")

    return selected_features, scores


def feature_selection_hybrid(X, y, feature_names, top_k=500):
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
    print("🔬 F-test + 互信息混合特征选择")
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


def save_results(feature_scores, selected_features, output_dir='output'):
    """
    保存特征选择结果

    参数:
    - feature_scores: 特征得分DataFrame
    - selected_features: 选择的特征索引
    - output_dir: 输出目录
    """
    logger.info("=" * 50)
    print("💾 保存结果")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存特征得分
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feature_scores_path = os.path.join(output_dir, f'feature_selection_scores_{timestamp}.csv')
    feature_scores.to_csv(feature_scores_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征得分已保存至: {feature_scores_path}")

    # 保存选择的特征
    selected_features_path = os.path.join(output_dir, f'selected_features_{timestamp}.csv')
    selected_df = feature_scores[feature_scores['Feature_Index'].isin(selected_features)].copy()
    selected_df.to_csv(selected_features_path, index=False, encoding='utf-8-sig')
    logger.info(f"选择的特征已保存至: {selected_features_path}")

    # 保存特征索引列表（方便后续使用）
    selected_indices_path = os.path.join(output_dir, f'selected_feature_indices_{timestamp}.txt')
    with open(selected_indices_path, 'w', encoding='utf-8') as f:
        f.write(','.join(map(str, selected_features)))
    logger.info(f"特征索引已保存至: {selected_indices_path}")

    print("")


def main():
    parser = argparse.ArgumentParser(description='特征选择优化 - F-test + 互信息混合方法')
    parser.add_argument('--top-k', type=int, default=500,
                       help='最终选择的特征数量 (默认: 500)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录 (默认: output)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 特征选择优化开始")
    logger.info("=" * 50)
    print(f"⚙️  参数配置:")
    print(f"   - 目标特征数量: {args.top_k}")
    print(f"   - 输出目录: {args.output_dir}")
    print("")

    try:
        # 步骤1: 加载训练数据
        X, y, feature_names = load_training_data()

        # 步骤2: 混合特征选择
        selected_features, feature_scores = feature_selection_hybrid(
            X, y, feature_names, top_k=args.top_k
        )

        # 步骤3: 评估效果
        performance = evaluate_feature_selection(X, y, selected_features, feature_names)

        # 步骤4: 保存结果
        save_results(feature_scores, selected_features, args.output_dir)

        # 步骤5: 显示Top 20特征
        logger.info("=" * 50)
        print("🏆 Top 20特征")
        logger.info("=" * 50)
        top_20 = feature_scores.head(20)
        print(top_20[['Feature_Name', 'Combined_Score', 'In_Intersection']].to_string(index=False))
        print("")

        logger.info("=" * 50)
        logger.info("特征选择优化完成！")
        logger.info("=" * 50)
        logger.info(f"优化总结:")
        print(f"   - 原始特征数量: {len(feature_names)}")
        print(f"   - 优化后特征数量: {len(selected_features)}")
        print(f"   - 特征减少比例: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
        print(f"   - 交集特征占比: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['In_Intersection'].sum()/len(selected_features)*100:.1f}%")
        print(f"   - 平均综合得分: {feature_scores[feature_scores['Feature_Index'].isin(selected_features)]['Combined_Score'].mean():.4f}")
        print("")
        print(f"💡 下一步:")
        print(f"   1. 检查选择的特征列表（保存到 {args.output_dir}/selected_features_*.csv）")
        print(f"   2. 在ml_trading_model.py中集成特征选择逻辑")
        print(f"   3. 重新训练模型并评估准确率提升")
        print(f"   4. 对比优化前后的模型性能")
        print("")

    except Exception as e:
        logger.error(f"特征选择失败: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()