#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征评估脚本 - 使用 SHAP 值进行特征重要性分析

功能：
1. 使用 SHAP 值分析特征贡献度
2. 交叉验证评估特征重要性稳定性
3. 对比不同特征集的模型性能
4. 生成特征评估报告
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

# 尝试导入 SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# 导入项目模块
from config import WATCHLIST as STOCK_LIST
from ml_services.ml_trading_model import MLTradingModel
from ml_services.logger_config import get_logger

logger = get_logger('feature_evaluation')


class FeatureEvaluator:
    """特征评估器"""

    def __init__(self, output_dir='output/feature_evaluation'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.shap_values = None
        self.feature_importance = None

    def load_model_and_data(self, model_path, horizon=20):
        """
        加载已训练的模型和数据

        参数:
        - model_path: 模型文件路径
        - horizon: 预测周期

        返回:
        - model: 训练好的模型
        - X: 特征矩阵
        - y: 标签向量
        - feature_names: 特征名称列表
        """
        logger.info("=" * 70)
        logger.info("步骤 1: 加载模型和数据")
        logger.info("=" * 70)

        # 创建模型实例
        model = MLTradingModel()
        model.horizon = horizon

        # 加载模型
        if os.path.exists(model_path):
            logger.info(f"加载模型: {model_path}")
            model.load_model(model_path)
        else:
            logger.error(f"模型文件不存在: {model_path}")
            return None, None, None, None

        # 准备数据
        logger.info("准备训练数据...")
        codes = list(STOCK_LIST.keys())[:10]  # 使用前10只股票作为示例
        df = model.prepare_data(codes, horizon=horizon)

        # 删除包含NaN的行
        df = df.dropna()
        df = df.sort_index()

        # 获取特征列
        feature_columns = model.get_feature_columns(df)
        logger.info(f"特征数量: {len(feature_columns)}")

        # 处理分类特征
        categorical_features = []
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                logger.debug(f"编码分类特征: {col}")
                categorical_features.append(col)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # 准备特征和标签
        X = df[feature_columns]
        y = df['Label']

        logger.info(f"样本数量: {len(X)}")
        logger.info(f"标签分布: {y.value_counts().to_dict()}")

        return model, X, y, feature_columns

    def calculate_shap_values(self, model, X, background_sample=100):
        """
        计算 SHAP 值

        参数:
        - model: 训练好的模型
        - X: 特征矩阵
        - background_sample: 背景样本数量（用于加速计算）

        返回:
        - shap_values: SHAP 值矩阵
        """
        if not HAS_SHAP:
            logger.error("SHAP 库未安装，请运行: pip install shap")
            return None

        logger.info("=" * 70)
        logger.info("步骤 2: 计算 SHAP 值")
        logger.info("=" * 70)

        try:
            # 使用部分数据作为背景数据集
            background = shap.sample(X, min(background_sample, len(X)))

            # 创建 SHAP 解释器
            logger.info("创建 SHAP 解释器...")
            explainer = shap.TreeExplainer(model.model, background)

            # 计算 SHAP 值
            logger.info(f"计算 SHAP 值（样本数: {len(X)}）...")
            self.shap_values = explainer.shap_values(X)

            # 如果是多分类，取第一类的值
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]

            logger.info(f"SHAP 值形状: {self.shap_values.shape}")

            return self.shap_values

        except Exception as e:
            logger.error(f"计算 SHAP 值失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def analyze_feature_importance(self, X, feature_names, top_k=50):
        """
        分析特征重要性

        参数:
        - X: 特征矩阵
        - feature_names: 特征名称列表
        - top_k: 显示前 k 个特征

        返回:
        - feature_importance: 特征重要性 DataFrame
        """
        if self.shap_values is None:
            logger.error("SHAP 值未计算")
            return None

        logger.info("=" * 70)
        logger.info("步骤 3: 分析特征重要性")
        logger.info("=" * 70)

        # 计算平均绝对 SHAP 值
        mean_shap = np.abs(self.shap_values).mean(axis=0)

        # 创建特征重要性 DataFrame
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_shap,
            'SHAP_Std': np.abs(self.shap_values).std(axis=0),
            'Max_SHAP': np.abs(self.shap_values).max(axis=0),
            'Min_SHAP': np.abs(self.shap_values).min(axis=0)
        })

        # 排序
        self.feature_importance = self.feature_importance.sort_values(
            'Mean_Abs_SHAP', ascending=False
        )

        logger.info(f"特征重要性计算完成")
        logger.info(f"Top 10 特征:")
        for i, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['Feature']:<40} {row['Mean_Abs_SHAP']:.6f}")

        return self.feature_importance

    def plot_shap_summary(self, X, feature_names, save_path=None):
        """
        绘制 SHAP 汇总图

        参数:
        - X: 特征矩阵
        - feature_names: 特征名称列表
        - save_path: 保存路径
        """
        if self.shap_values is None:
            logger.error("SHAP 值未计算")
            return

        logger.info("绘制 SHAP 汇总图...")

        try:
            plt.figure(figsize=(12, 8))

            # 创建 DataFrame
            shap_df = pd.DataFrame(self.shap_values, columns=feature_names)

            # 选择 top 20 特征
            if self.feature_importance is not None:
                top_features = self.feature_importance.head(20)['Feature'].tolist()
                shap_df = shap_df[top_features]

            # 绘制箱线图
            shap_df.boxplot(figsize=(14, 8))
            plt.title('SHAP Value Distribution by Feature (Top 20)', fontsize=14)
            plt.ylabel('SHAP Value', fontsize=12)
            plt.xlabel('Feature', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"SHAP 汇总图已保存: {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"绘制 SHAP 汇总图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def plot_feature_importance(self, save_path=None, top_k=30):
        """
        绘制特征重要性图

        参数:
        - save_path: 保存路径
        - top_k: 显示前 k 个特征
        """
        if self.feature_importance is None:
            logger.error("特征重要性未计算")
            return

        logger.info("绘制特征重要性图...")

        try:
            # 选择 top k 特征
            top_features = self.feature_importance.head(top_k).copy()

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 子图1: 平均 SHAP 值
            ax1.barh(range(len(top_features)), top_features['Mean_Abs_SHAP'])
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['Feature'], fontsize=9)
            ax1.invert_yaxis()
            ax1.set_xlabel('Mean Absolute SHAP Value')
            ax1.set_title(f'Top {top_k} Features by Mean SHAP Value')
            ax1.grid(axis='x', alpha=0.3)

            # 子图2: SHAP 值分布
            ax2.boxplot([self.shap_values[:, i] for i in range(len(top_features))])
            ax2.set_xticks(range(1, len(top_features) + 1))
            ax2.set_xticklabels(top_features['Feature'], rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('SHAP Value')
            ax2.set_title('SHAP Value Distribution')
            ax2.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"特征重要性图已保存: {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"绘制特征重要性图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def evaluate_feature_stability(self, model, X, y, feature_names, n_splits=5):
        """
        交叉验证评估特征重要性稳定性

        参数:
        - model: 模型实例
        - X: 特征矩阵
        - y: 标签向量
        - feature_names: 特征名称列表
        - n_splits: 交叉验证折数

        返回:
        - stability_df: 特征稳定性 DataFrame
        """
        logger.info("=" * 70)
        logger.info("步骤 4: 评估特征重要性稳定性")
        logger.info("=" * 70)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        feature_scores = {name: [] for name in feature_names}

        logger.info(f"进行 {n_splits} 折交叉验证...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"处理 Fold {fold + 1}/{n_splits}...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 训练模型
            fold_model = lgb.LGBMClassifier(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.02,
                verbose=-1
            )
            fold_model.fit(X_train, y_train)

            # 获取特征重要性
            importance = fold_model.feature_importances_

            # 记录每个特征的重要性
            for i, name in enumerate(feature_names):
                feature_scores[name].append(importance[i])

        # 计算稳定性指标
        stability_data = []
        for name, scores in feature_scores.items():
            stability_data.append({
                'Feature': name,
                'Mean_Importance': np.mean(scores),
                'Std_Importance': np.std(scores),
                'CV_Importance': np.std(scores) / (np.mean(scores) + 1e-6),  # 变异系数
                'Min_Importance': np.min(scores),
                'Max_Importance': np.max(scores)
            })

        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('Mean_Importance', ascending=False)

        logger.info(f"特征稳定性评估完成")
        logger.info(f"最稳定的前10个特征 (最低 CV):")
        for i, row in stability_df.nsmallest(10, 'CV_Importance').iterrows():
            logger.info(f"  {row['Feature']:<40} CV={row['CV_Importance']:.4f}")

        return stability_df

    def generate_report(self, model_path, stability_df=None, top_k=50):
        """
        生成特征评估报告

        参数:
        - model_path: 模型路径
        - stability_df: 特征稳定性 DataFrame
        - top_k: 报告中显示的特征数量
        """
        logger.info("=" * 70)
        logger.info("步骤 5: 生成评估报告")
        logger.info("=" * 70)

        report_path = os.path.join(self.output_dir, 'feature_evaluation_report.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("特征评估报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"特征总数: {len(self.feature_importance)}\n\n")

            # SHAP 重要性排名
            f.write("=" * 80 + "\n")
            f.write("1. SHAP 特征重要性排名\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'排名':<6} {'特征名称':<50} {'平均SHAP值':<15} {'标准差':<15}\n")
            f.write("-" * 80 + "\n")

            for i, row in self.feature_importance.head(top_k).iterrows():
                rank = i + 1
                f.write(f"{rank:<6} {row['Feature']:<50} {row['Mean_Abs_SHAP']:<15.6f} {row['SHAP_Std']:<15.6f}\n")

            # 特征稳定性（如果有）
            if stability_df is not None:
                f.write("\n" + "=" * 80 + "\n")
                f.write("2. 特征重要性稳定性 (交叉验证)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"{'排名':<6} {'特征名称':<50} {'平均重要性':<15} {'变异系数':<15}\n")
                f.write("-" * 80 + "\n")

                for i, row in stability_df.head(top_k).iterrows():
                    rank = i + 1
                    f.write(f"{rank:<6} {row['Feature']:<50} {row['Mean_Importance']:<15.6f} {row['CV_Importance']:<15.6f}\n")

            # 建议
            f.write("\n" + "=" * 80 + "\n")
            f.write("3. 特征选择建议\n")
            f.write("=" * 80 + "\n\n")

            top_features = self.feature_importance.head(top_k)['Feature'].tolist()
            f.write(f"- 推荐保留 {top_k} 个重要特征:\n")
            for i, feat in enumerate(top_features[:10], 1):
                f.write(f"  {i}. {feat}\n")
            f.write(f"  ... (共 {top_k} 个)\n\n")

            f.write("- 移除低重要性特征:\n")
            f.write(f"  - 建议: 移除 SHAP 重要性低于 {self.feature_importance.iloc[top_k]['Mean_Abs_SHAP']:.6f} 的特征\n")
            f.write(f"  - 数量: {len(self.feature_importance) - top_k} 个特征\n\n")

            f.write("- 特征稳定性:\n")
            if stability_df is not None:
                stable_features = stability_df.nsmallest(20, 'CV_Importance')['Feature'].tolist()
                f.write(f"  - 最稳定的前20个特征:\n")
                for i, feat in enumerate(stable_features[:5], 1):
                    f.write(f"    {i}. {feat}\n")
                f.write(f"    ... (共 {len(stable_features)} 个)\n\n")

        logger.info(f"评估报告已保存: {report_path}")

        # 同时保存 CSV 文件
        csv_path = os.path.join(self.output_dir, 'feature_importance_shap.csv')
        self.feature_importance.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"特征重要性 CSV 已保存: {csv_path}")

        if stability_df is not None:
            stability_path = os.path.join(self.output_dir, 'feature_stability.csv')
            stability_df.to_csv(stability_path, index=False, encoding='utf-8')
            logger.info(f"特征稳定性 CSV 已保存: {stability_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征评估脚本')
    parser.add_argument('--model_path', type=str,
                       default='output/ml_trading_model_lightgbm_20d.pkl',
                       help='模型文件路径')
    parser.add_argument('--horizon', type=int, default=20,
                       choices=[1, 5, 20],
                       help='预测周期')
    parser.add_argument('--top_k', type=int, default=50,
                       help='显示前 k 个特征')
    parser.add_argument('--output_dir', type=str,
                       default='output/feature_evaluation',
                       help='输出目录')
    parser.add_argument('--skip_shap', action='store_true',
                       help='跳过 SHAP 计算（如果未安装 shap）')

    args = parser.parse_args()

    if not HAS_SHAP and not args.skip_shap:
        logger.error("SHAP 库未安装，请运行: pip install shap")
        logger.error("或使用 --skip_shap 参数跳过 SHAP 计算")
        return

    # 创建评估器
    evaluator = FeatureEvaluator(output_dir=args.output_dir)

    # 加载模型和数据
    model, X, y, feature_names = evaluator.load_model_and_data(
        args.model_path, args.horizon
    )
    if model is None:
        return

    # 计算 SHAP 值
    if not args.skip_shap:
        shap_values = evaluator.calculate_shap_values(model.model, X)
        if shap_values is not None:
            # 分析特征重要性
            evaluator.analyze_feature_importance(X, feature_names, args.top_k)

            # 绘制图表
            evaluator.plot_shap_summary(
                X, feature_names,
                save_path=os.path.join(args.output_dir, 'shap_summary.png')
            )
            evaluator.plot_feature_importance(
                save_path=os.path.join(args.output_dir, 'feature_importance.png'),
                top_k=args.top_k
            )

    # 评估特征稳定性
    stability_df = evaluator.evaluate_feature_stability(
        model, X, y, feature_names, n_splits=5
    )

    # 生成报告
    evaluator.generate_report(
        args.model_path,
        stability_df=stability_df,
        top_k=args.top_k
    )

    logger.info("\n" + "=" * 70)
    logger.info("特征评估完成！")
    logger.info("=" * 70)
    logger.info(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
