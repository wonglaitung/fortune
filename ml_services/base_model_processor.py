#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Model Processor - 提供模型训练和分析的基础功能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
from ml_services.logger_config import get_logger

logger = get_logger('base_model_processor')


class BaseModelProcessor:
    """模型处理器基类"""

    def __init__(self):
        self.continuous_features = []
        self.category_features = []
        self.output_dir = 'output'

    def load_feature_config(self, config_path='config/feature_config.json'):
        """
        加载特征配置文件
        返回: bool 是否加载成功
        """
        if not os.path.exists(config_path):
            # 如果配置文件不存在，使用默认配置
            logger.warning(f"配置文件不存在: {config_path}")
            print("ℹ️  将使用默认配置（所有特征视为连续特征）")
            return True

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.continuous_features = config.get('continuous_features', [])
            self.category_features = config.get('category_features', [])

            logger.info(f"成功加载特征配置:")
            print(f"   - 连续特征: {len(self.continuous_features)} 个")
            print(f"   - 类别特征: {len(self.category_features)} 个")

            return True

        except Exception as e:
            logger.error(f"加载特征配置失败: {e}")
            return False

    def analyze_feature_importance(self, booster, feature_names):
        """
        分析特征重要性
        返回: DataFrame 包含特征重要性信息
        """
        # 获取特征重要性
        importance_gain = booster.feature_importance(importance_type='gain')
        importance_split = booster.feature_importance(importance_type='split')

        # 创建DataFrame
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Gain_Importance': importance_gain,
            'Split_Importance': importance_split
        })

        # 归一化
        feat_imp['Gain_Importance'] = feat_imp['Gain_Importance'] / feat_imp['Gain_Importance'].sum()
        feat_imp['Split_Importance'] = feat_imp['Split_Importance'] / feat_imp['Split_Importance'].sum()

        # 按 Gain 重要性排序
        feat_imp = feat_imp.sort_values('Gain_Importance', ascending=False).reset_index(drop=True)

        return feat_imp

    def calculate_ks_statistic(self, y_true, y_pred_proba):
        """
        计算 KS 统计量
        返回: float KS 值
        """
        from scipy import stats

        # 计算正负样本的预测概率
        pos = y_pred_proba[y_true == 1]
        neg = y_pred_proba[y_true == 0]

        # 计算 KS 统计量
        ks_statistic, p_value = stats.ks_2samp(pos, neg)

        return ks_statistic

    def plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """
        绘制 ROC 曲线
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC 曲线已保存至 {save_path}")

    def get_leaf_path_enhanced(self, booster, tree_index, leaf_index, feature_names, category_prefixes=None):
        """
        解析叶子节点的决策路径（增强版）
        返回: list 决策规则列表
        """
        try:
            tree_df = booster.trees_to_dataframe()
            
            # 筛选指定树
            tree_data = tree_df[tree_df['tree_index'] == tree_index].copy()
            
            if tree_data.empty:
                return None
            
            # 构建叶子节点的node_index：{tree_index}-L{leaf_index}
            target_node_id = f"{tree_index}-L{leaf_index}"
            
            # 查找叶子节点
            leaf_node = tree_data[tree_data['node_index'] == target_node_id]
            
            if leaf_node.empty:
                return None
            
            # 向上追溯父节点
            current_node_id = target_node_id
            path = []
            
            while True:
                node_info = tree_data[tree_data['node_index'] == current_node_id]
                if node_info.empty:
                    break
                    
                parent_id_str = node_info['parent_index'].values[0]
                
                # 检查是否到达根节点
                if pd.isna(parent_id_str) or parent_id_str == '':
                    break
                    
                # 获取父节点信息
                parent_info = tree_data[tree_data['node_index'] == parent_id_str]
                if parent_info.empty:
                    break
                
                # 检查父节点是否有分裂信息
                if pd.notna(parent_info['split_feature'].values[0]):
                    feature_str = parent_info['split_feature'].values[0]
                    threshold = parent_info['threshold'].values[0]
                    left_child = parent_info['left_child'].values[0]
                    right_child = parent_info['right_child'].values[0]
                    
                    # 解析特征索引（从 "Column_X" 格式中提取）
                    if isinstance(feature_str, str) and feature_str.startswith('Column_'):
                        try:
                            feature_idx = int(feature_str.split('_')[1])
                        except (ValueError, IndexError):
                            feature_idx = -1
                    else:
                        feature_idx = -1
                    
                    # 获取特征名称
                    if 0 <= feature_idx < len(feature_names):
                        feature_name = feature_names[feature_idx]
                    else:
                        feature_name = f"Feature_{feature_str}"
                    
                    # 判断当前节点是左子节点还是右子节点
                    if pd.notna(left_child) and current_node_id == left_child:
                        operator = "<="
                    elif pd.notna(right_child) and current_node_id == right_child:
                        operator = ">"
                    else:
                        # 无法确定，跳过
                        current_node_id = parent_id_str
                        continue
                    
                    # 格式化规则
                    if category_prefixes and any(feature_name.startswith(prefix) for prefix in category_prefixes):
                        # 类别特征
                        rule = f"{feature_name} 是 {int(threshold)} 类"
                    else:
                        # 连续特征
                        rule = f"{feature_name} {operator} {threshold:.4f}"
                    
                    path.append(rule)
                
                # 移动到父节点
                current_node_id = parent_id_str
            
            # 反转路径，从根到叶子
            path.reverse()
            return path if path else None
            
        except Exception as e:
            import traceback
            logger.warning(f"解析叶子路径失败: {e}")
            traceback.print_exc()
            return None

    def save_models(self, gbdt_model, lr_model, category_features, continuous_features):
        """
        保存模型和配置
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存 GBDT 模型
        gbdt_model_path = os.path.join(self.output_dir, 'gbdt_model.txt')
        gbdt_model.booster_.save_model(gbdt_model_path)
        logger.info(f"GBDT 模型已保存至 {gbdt_model_path}")

        # 保存 LR 模型
        import pickle
        lr_model_path = os.path.join(self.output_dir, 'lr_model.pkl')
        with open(lr_model_path, 'wb') as f:
            pickle.dump(lr_model, f)
        logger.info(f"LR 模型已保存至 {lr_model_path}")

        # 保存实际训练的树数量
        actual_n_estimators = gbdt_model.best_iteration_
        with open(os.path.join(self.output_dir, 'actual_n_estimators.csv'), 'w') as f:
            f.write(f"actual_n_estimators,{actual_n_estimators}\n")
        logger.info(f"实际树数量已保存至 {self.output_dir}/actual_n_estimators.csv")

        # 保存特征配置
        with open(os.path.join(self.output_dir, 'category_features.csv'), 'w') as f:
            f.write(','.join(category_features))
        logger.info(f"类别特征已保存至 {self.output_dir}/category_features.csv")

        with open(os.path.join(self.output_dir, 'continuous_features.csv'), 'w') as f:
            f.write(','.join(continuous_features))
        logger.info(f"连续特征已保存至 {self.output_dir}/continuous_features.csv")

    def show_model_interpretation_prompt(self):
        """
        显示模型解读提示
        """
        print("\n" + "="*70)
        logger.info("模型可解释性分析提示")
        print("="*70)
        print("训练完成后，将生成以下可解释性报告：")
        print("1. gbdt_feature_importance.csv - GBDT 特征重要性（含影响方向）")
        print("2. lr_leaf_coefficients.csv - LR 模型的叶子节点系数")
        print("3. roc_curve.png - ROC 曲线图")
        print("4. 叶子规则解析 - 高权重叶子节点的决策路径")
        print("\n💡 使用建议：")
        print("- 关注 Gain 重要性高的特征，它们对模型贡献最大")
        print("- 正向影响（Positive）表示特征值越大，预测为正类的概率越高")
        print("- 负向影响（Negative）表示特征值越大，预测为正类的概率越低")
        print("- LR 系数绝对值大的叶子节点表示重要的决策规则")
        print("="*70 + "\n")
