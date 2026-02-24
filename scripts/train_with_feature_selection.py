#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用特征选择结果训练模型

基于特征评估结果，使用不同特征集训练模型并对比性能
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
from ml_services.ml_trading_model import MLTradingModel
from config import WATCHLIST
from ml_services.logger_config import get_logger

logger = get_logger('train_with_feature_selection')


def load_features_from_csv(csv_path, n_features=None, min_importance=None):
    """
    从特征重要性 CSV 加载特征列表

    参数:
    - csv_path: CSV 文件路径
    - n_features: 选择前 N 个特征
    - min_importance: 最小重要性阈值

    返回:
    - features: 特征名称列表
    """
    logger.info(f"加载特征列表: {csv_path}")

    df = pd.read_csv(csv_path)

    # 应用过滤条件
    filtered_df = df.copy()

    if min_importance is not None:
        filtered_df = filtered_df[filtered_df['Mean_Importance'] >= min_importance]
        logger.info(f"  重要性阈值: {min_importance} -> {len(filtered_df)} 个特征")

    if n_features is not None:
        filtered_df = filtered_df.head(n_features)
        logger.info(f"  Top N: {n_features} -> {len(filtered_df)} 个特征")

    features = filtered_df['Feature'].tolist()
    logger.info(f"  最终选择: {len(features)} 个特征")

    return features


def train_model_with_features(
    model_type='lightgbm',
    horizon=5,
    num_stocks=10,
    feature_set_name='all_features',
    features=None,
    output_dir='output/models_with_feature_selection'
):
    """
    使用指定特征集训练模型

    参数:
    - model_type: 模型类型
    - horizon: 预测周期
    - num_stocks: 训练股票数量
    - feature_set_name: 特征集名称（用于标识）
    - features: 特征列表（None 表示使用全部特征）
    - output_dir: 输出目录

    返回:
    - result: 训练结果字典
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"训练模型: {model_type}, 特征集: {feature_set_name}")
    logger.info("=" * 70)

    # 创建模型实例
    model = MLTradingModel()
    model.horizon = horizon

    # 设置特征选择
    if features is not None:
        model.use_feature_selection = True
        model.selected_features = features
        logger.info(f"使用特征选择: {len(features)} 个特征")
    else:
        model.use_feature_selection = False
        logger.info("使用全部特征")

    # 获取训练股票
    codes = list(WATCHLIST.keys())[:num_stocks]
    logger.info(f"训练股票数量: {len(codes)}")

    # 训练模型
    try:
        start_time = datetime.now()
        logger.info(f"\n开始训练...")

        # 调用 train 方法
        model.train(codes, horizon=horizon, use_feature_selection=model.use_feature_selection)

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # 获取验证准确率（从 model_accuracy.json 读取）
        validation_accuracy = None
        validation_std = None

        try:
            accuracy_file = 'data/model_accuracy.json'
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    accuracy_data = json.load(f)

                # 根据模型类型和horizon查找准确率
                key = f'{model_type}_{horizon}d'
                if key in accuracy_data:
                    validation_accuracy = accuracy_data[key]['accuracy']
                    validation_std = accuracy_data[key]['std']
                    logger.info(f"从 {accuracy_file} 读取到准确率: {validation_accuracy:.4f}")
                else:
                    logger.warning(f"未找到 {key} 的准确率数据")
        except Exception as e:
            logger.warning(f"读取准确率文件失败: {e}")

        logger.info(f"\n训练完成！")
        logger.info(f"训练时间: {training_time:.2f} 秒")

        if validation_accuracy:
            logger.info(f"验证准确率: {validation_accuracy:.4f} (+/- {validation_std:.4f})")

        # 保存模型
        model_filename = f"{model_type}_{feature_set_name}_h{horizon}.pkl"
        model_path = os.path.join(output_dir, model_filename)

        model.save_model(model_path)
        logger.info(f"模型已保存: {model_path}")

        result = {
            'feature_set': feature_set_name,
            'num_features': len(features) if features else 'all',
            'validation_accuracy': validation_accuracy,
            'validation_std': validation_std,
            'training_time': training_time,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        return {
            'feature_set': feature_set_name,
            'num_features': len(features) if features else 'all',
            'validation_accuracy': None,
            'validation_std': None,
            'training_time': None,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def compare_feature_sets(
    model_type='lightgbm',
    horizon=5,
    num_stocks=10,
    output_dir='output/models_with_feature_selection'
):
    """
    对比不同特征集的模型性能

    参数:
    - model_type: 模型类型
    - horizon: 预测周期
    - num_stocks: 训练股票数量
    - output_dir: 输出目录

    返回:
    - comparison_df: 对比结果 DataFrame
    """
    logger.info("=" * 70)
    logger.info("对比不同特征集的性能")
    logger.info("=" * 70)

    # 加载特征重要性
    feature_csv = 'output/feature_eval_v2/feature_importance.csv'

    if not os.path.exists(feature_csv):
        logger.error(f"特征重要性文件不存在: {feature_csv}")
        logger.info("请先运行: python scripts/feature_eval_v2.py")
        return None

    # 定义不同的特征集
    feature_sets = [
        {
            'name': 'all_features',
            'description': '全部特征（3972个）',
            'features': None
        },
        {
            'name': 'top_50',
            'description': 'Top 50 特征',
            'features': load_features_from_csv(feature_csv, n_features=50)
        },
        {
            'name': 'top_100',
            'description': 'Top 100 特征',
            'features': load_features_from_csv(feature_csv, n_features=100)
        },
        {
            'name': 'top_200',
            'description': 'Top 200 特征',
            'features': load_features_from_csv(feature_csv, n_features=200)
        },
    ]

    # 可选：高重要性且稳定的特征
    try:
        df = pd.read_csv(feature_csv)
        stable_high = df[
            (df['Mean_Importance'] > 5) &  # 高重要性
            (df['CV_Importance'] < 0.6)     # 高稳定性
        ].head(100)
        feature_sets.append({
            'name': 'stable_high_100',
            'description': '高重要性且稳定的100个特征',
            'features': stable_high['Feature'].tolist()
        })
    except Exception as e:
        logger.warning(f"创建稳定特征集失败: {e}")

    # 训练所有模型
    results = []

    for feature_set in feature_sets:
        logger.info(f"\n{'='*70}")
        logger.info(f"特征集: {feature_set['name']} - {feature_set['description']}")
        logger.info(f"{'='*70}")

        result = train_model_with_features(
            model_type=model_type,
            horizon=horizon,
            num_stocks=num_stocks,
            feature_set_name=feature_set['name'],
            features=feature_set['features'],
            output_dir=output_dir
        )

        result['description'] = feature_set['description']
        results.append(result)

    # 创建对比 DataFrame
    comparison_df = pd.DataFrame(results)

    # 打印对比结果
    logger.info("\n" + "=" * 70)
    logger.info("性能对比结果")
    logger.info("=" * 70)

    comparison_df_display = comparison_df[['feature_set', 'description', 'num_features', 'validation_accuracy', 'validation_std', 'training_time']].copy()

    # 格式化准确率和标准差
    comparison_df_display['accuracy_display'] = comparison_df_display.apply(
        lambda row: f"{row['validation_accuracy']:.4f} (+/- {row['validation_std']:.4f})"
        if row['validation_accuracy'] is not None else "N/A", axis=1
    )
    comparison_df_display = comparison_df_display.drop(['validation_accuracy', 'validation_std'], axis=1)
    comparison_df_display = comparison_df_display.rename(columns={'accuracy_display': 'validation_accuracy'})

    # 计算相对于全部特征的性能提升
    if len(comparison_df) > 0 and comparison_df.iloc[0]['feature_set'] == 'all_features':
        baseline_acc = comparison_df.iloc[0]['validation_accuracy']
        baseline_time = comparison_df.iloc[0]['training_time']

        if baseline_acc:
            comparison_df_display['accuracy_change'] = comparison_df['validation_accuracy'].apply(
                lambda x: f"{(x - baseline_acc)*100:+.2f}%" if x is not None else "N/A"
            )
        if baseline_time:
            comparison_df_display['time_improvement'] = comparison_df['training_time'].apply(
                lambda x: f"{(1 - x/baseline_time)*100:+.1f}%" if x is not None else "N/A"
            )

    logger.info(f"\n{comparison_df_display.to_string(index=False)}")

    # 保存对比结果
    comparison_path = os.path.join(output_dir, f'comparison_{model_type}_h{horizon}.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
    logger.info(f"\n对比结果已保存: {comparison_path}")

    # 生成详细报告
    report_path = os.path.join(output_dir, f'comparison_report_{model_type}_h{horizon}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("特征选择对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"预测周期: {horizon} 天\n")
        f.write(f"训练股票数: {num_stocks}\n\n")

        f.write("=" * 80 + "\n")
        f.write("详细结果\n")
        f.write("=" * 80 + "\n\n")

        for i, row in comparison_df.iterrows():
            f.write(f"{i+1}. {row['feature_set']}\n")
            f.write(f"   描述: {row['description']}\n")
            f.write(f"   特征数量: {row['num_features']}\n")

            if row['validation_accuracy'] is not None:
                f.write(f"   验证准确率: {row['validation_accuracy']:.4f} (+/- {row['validation_std']:.4f})\n")

            if row['training_time']:
                f.write(f"   训练时间: {row['training_time']:.2f} 秒\n")

            if 'error' in row:
                f.write(f"   错误: {row['error']}\n")

            f.write("\n")

        # 分析和推荐
        f.write("=" * 80 + "\n")
        f.write("分析和推荐\n")
        f.write("=" * 80 + "\n\n")

        if len(comparison_df) > 0:
            # 过滤出有准确率数据的模型
            valid_models = comparison_df[comparison_df['validation_accuracy'].notna()]

            if len(valid_models) > 0:
                # 找出最好的模型
                best_idx = valid_models['validation_accuracy'].idxmax()
                best_model = comparison_df.iloc[best_idx]

                f.write("📊 性能分析:\n")
                f.write(f"   最佳模型: {best_model['feature_set']}\n")
                f.write(f"   准确率: {best_model['validation_accuracy']:.4f} (+/- {best_model['validation_std']:.4f})\n")
                f.write(f"   特征数量: {best_model['num_features']}\n")
                if best_model['training_time']:
                    f.write(f"   训练时间: {best_model['training_time']:.2f} 秒\n")

                # 计算速度提升
                baseline = comparison_df[comparison_df['feature_set'] == 'all_features']
                if len(baseline) > 0 and baseline.iloc[0]['training_time']:
                    baseline_time = baseline.iloc[0]['training_time']
                    if best_model['training_time']:
                        speedup = baseline_time / best_model['training_time']
                        f.write(f"   速度提升: {speedup:.2f}x\n")

                f.write("\n💡 推荐建议:\n")
                if best_model['feature_set'] != 'all_features':
                    f.write(f"   1. 使用 {best_model['feature_set']} 特征集\n")
                    f.write(f"   2. 该特征集在准确率和速度之间取得了最佳平衡\n")
                else:
                    f.write(f"   1. 使用全部特征（但可以考虑减少特征以提升速度）\n")

                f.write(f"   2. 可以尝试不同的特征数量来优化性能\n")
            else:
                f.write("⚠️ 警告: 没有有效的准确率数据可供分析\n")

    logger.info(f"详细报告已保存: {report_path}")

    return comparison_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用特征选择结果训练模型')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'train', 'analyze'],
                       help='运行模式: compare(对比), train(训练单个), analyze(仅分析）')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                       choices=['lightgbm', 'gbdt', 'catboost'],
                       help='模型类型')
    parser.add_argument('--horizon', type=int, default=5,
                       choices=[1, 5, 20],
                       help='预测周期')
    parser.add_argument('--num_stocks', type=int, default=10,
                       help='训练股票数量')
    parser.add_argument('--feature_set', type=str, default='top_100',
                       choices=['top_50', 'top_100', 'top_200', 'all_features', 'stable_high_100'],
                       help='特征集名称（train 模式）')
    parser.add_argument('--output_dir', type=str,
                       default='output/models_with_feature_selection',
                       help='输出目录')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("使用特征选择结果训练模型")
    logger.info("=" * 70)
    logger.info(f"模式: {args.mode}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"预测周期: {args.horizon} 天")
    logger.info(f"股票数量: {args.num_stocks}")

    if args.mode == 'compare':
        # 对比模式：训练并对比所有特征集
        comparison_df = compare_feature_sets(
            model_type=args.model_type,
            horizon=args.horizon,
            num_stocks=args.num_stocks,
            output_dir=args.output_dir
        )

        if comparison_df is not None:
            logger.info("\n✅ 对比完成！")
            logger.info(f"结果保存在: {args.output_dir}")

    elif args.mode == 'train':
        # 训练模式：训练单个模型
        logger.info(f"\n训练单个模型: {args.feature_set}")

        feature_csv = 'output/feature_eval_v2/feature_importance.csv'

        # 根据特征集名称加载特征
        if args.feature_set == 'all_features':
            features = None
        elif args.feature_set == 'stable_high_100':
            df = pd.read_csv(feature_csv)
            features = df[
                (df['Mean_Importance'] > 5) &
                (df['CV_Importance'] < 0.6)
            ].head(100)['Feature'].tolist()
        else:
            n_features = int(args.feature_set.replace('top_', ''))
            features = load_features_from_csv(feature_csv, n_features=n_features)

        result = train_model_with_features(
            model_type=args.model_type,
            horizon=args.horizon,
            num_stocks=args.num_stocks,
            feature_set_name=args.feature_set,
            features=features,
            output_dir=args.output_dir
        )

        if result.get('validation_accuracy'):
            logger.info(f"\n✅ 训练成功！")
            logger.info(f"验证准确率: {result['validation_accuracy']:.4f}")
            logger.info(f"模型路径: {result['model_path']}")
        else:
            logger.error(f"\n❌ 训练失败: {result.get('error', 'Unknown')}")

    elif args.mode == 'analyze':
        # 分析模式：仅分析特征选择结果
        logger.info(f"\n分析特征选择结果")

        feature_csv = 'output/feature_eval_v2/feature_importance.csv'
        df = pd.read_csv(feature_csv)

        logger.info(f"\n特征总数: {len(df)}")
        logger.info(f"\nTop 10 特征:")
        for i, row in df.head(10).iterrows():
            logger.info(f"  {i+1:2d}. {row['Feature']:<40} {row['Mean_Importance']:>8.2f} (CV={row['CV_Importance']:.4f})")

        logger.info(f"\n特征重要性统计:")
        logger.info(f"  平均值: {df['Mean_Importance'].mean():.2f}")
        logger.info(f"  中位数: {df['Mean_Importance'].median():.2f}")
        logger.info(f"  最大值: {df['Mean_Importance'].max():.2f}")

        logger.info(f"\n✅ 分析完成！")

    logger.info("\n" + "=" * 70)
    logger.info("完成！")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
