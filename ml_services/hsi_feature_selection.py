#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP 动态特征选择模块

基于 SHAP (SHapley Additive exPlanations) 分析特征重要性：
1. 全局特征重要性排序
2. 按市场状态（Regime）分组的特征重要性差异
3. 动态特征选择：根据当前市场状态选择最优特征子集

使用方式：
    selector = DynamicFeatureSelector(model, X_train, y_train)
    selector.analyze()  # 分析特征重要性
    selected = selector.select_features(X, regime=2)  # 按市场状态选择特征

依赖：shap 库（pip install shap）
"""

import os
import sys
import warnings
import json
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 缓存目录
SHAP_CACHE_DIR = 'data/shap_cache'


class DynamicFeatureSelector:
    """基于 SHAP 的动态特征选择器"""

    def __init__(self, model=None, X_train=None, y_train=None,
                 regime_col='Market_Regime', top_k_ratio=0.7):
        """
        参数:
        - model: 训练好的 CatBoost 模型
        - X_train: 训练特征
        - y_train: 训练标签
        - regime_col: 市场状态列名
        - top_k_ratio: 保留的特征比例（0.7 = 保留 70% 最重要的特征）
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.regime_col = regime_col
        self.top_k_ratio = top_k_ratio

        self.global_importance = None
        self.regime_importance = {}
        self.feature_names = None
        self.shap_values = None

        os.makedirs(SHAP_CACHE_DIR, exist_ok=True)

    def analyze(self, regime_series=None):
        """
        分析特征重要性

        参数:
        - regime_series: 市场状态序列（与 X_train 对齐），用于分组分析

        返回:
        - self
        """
        if self.model is None or self.X_train is None:
            raise ValueError("需要提供模型和训练数据")

        print("  🔍 分析 SHAP 特征重要性...")

        import shap

        self.feature_names = (self.X_train.columns.tolist()
                              if isinstance(self.X_train, pd.DataFrame)
                              else [f'f{i}' for i in range(self.X_train.shape[1])])

        # 计算 SHAP 值
        # CatBoost 使用 TreeExplorer
        try:
            explainer = shap.TreeExplainer(self.model)
            X_sample = self.X_train
            # 大样本时采样以加速（SHAP 计算开销大）
            if len(self.X_train) > 500:
                sample_idx = np.random.RandomState(42).choice(
                    len(self.X_train), 500, replace=False
                )
                X_sample = self.X_train.iloc[sample_idx] if isinstance(self.X_train, pd.DataFrame) \
                    else self.X_train[sample_idx]

            self.shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"  ⚠️ SHAP 计算失败: {e}")
            return self

        # 1. 全局特征重要性（平均绝对 SHAP 值）
        if isinstance(self.shap_values, list):
            # 二分类返回两个数组的列表
            shap_abs = np.abs(self.shap_values[1]) if len(self.shap_values) == 2 \
                else np.abs(self.shap_values[0])
        else:
            shap_abs = np.abs(self.shap_values)

        self.global_importance = pd.Series(
            shap_abs.mean(axis=0),
            index=self.feature_names
        ).sort_values(ascending=False)

        # 2. 按市场状态分组分析特征重要性
        if regime_series is not None and self.regime_col in regime_series.name if hasattr(regime_series, 'name') else False:
            self._analyze_by_regime(regime_series, X_sample)

        # 保存分析结果
        self._save_results()

        print(f"  ✅ SHAP 分析完成（Top 5 特征: {self.global_importance.head(5).index.tolist()}）")

        return self

    def _analyze_by_regime(self, regime_series, X_sample):
        """按市场状态分组分析特征重要性"""
        # 确保索引对齐
        if isinstance(X_sample, pd.DataFrame):
            sample_idx = X_sample.index
        else:
            sample_idx = range(len(X_sample))

        # regime_series 需要与 sample_idx 对齐
        if isinstance(regime_series, pd.Series):
            aligned_regime = regime_series.reindex(sample_idx)
        else:
            aligned_regime = pd.Series(regime_series, index=sample_idx)

        # SHAP 值的绝对值
        if isinstance(self.shap_values, list):
            shap_abs = np.abs(self.shap_values[1]) if len(self.shap_values) == 2 \
                else np.abs(self.shap_values[0])
        else:
            shap_abs = np.abs(self.shap_values)

        # 按状态分组
        for regime in [0, 1, 2]:  # 震荡、上涨、下跌
            mask = aligned_regime.values == regime
            if mask.sum() < 10:
                continue

            regime_shap = shap_abs[mask]
            self.regime_importance[regime] = pd.Series(
                regime_shap.mean(axis=0),
                index=self.feature_names
            ).sort_values(ascending=False)

    def select_features(self, X, regime=None):
        """
        动态选择特征

        参数:
        - X: 待选择的特征数据
        - regime: 当前市场状态（None 时使用全局重要性）

        返回:
        - DataFrame/Series: 选择后的特征数据
        - list: 选择的特征名列表
        """
        if self.global_importance is None:
            # 未分析时返回全部特征
            return X, (X.columns.tolist() if isinstance(X, pd.DataFrame)
                       else [f'f{i}' for i in range(X.shape[1])])

        # 确定使用哪个重要性排序
        if regime is not None and regime in self.regime_importance:
            importance = self.regime_importance[regime]
        else:
            importance = self.global_importance

        # 选择 Top-K 特征
        top_k = max(10, int(len(importance) * self.top_k_ratio))
        selected_features = importance.head(top_k).index.tolist()

        # 只保留存在的特征
        if isinstance(X, pd.DataFrame):
            existing = [f for f in selected_features if f in X.columns]
            return X[existing], existing
        else:
            # numpy array
            if self.feature_names:
                indices = [self.feature_names.index(f) for f in selected_features
                          if f in self.feature_names]
                return X[:, indices], [self.feature_names[i] for i in indices]
            return X, list(range(X.shape[1]))

    def get_importance_diff(self, feature_name=None):
        """
        获取特征在不同市场状态下的重要性差异

        参数:
        - feature_name: 特征名（None 时返回所有特征）

        返回:
        - DataFrame: 各状态下的特征重要性
        """
        if not self.regime_importance:
            return pd.DataFrame()

        data = {}
        for regime, importance in self.regime_importance.items():
            label = {0: '震荡', 1: '上涨', 2: '下跌'}.get(regime, f'状态{regime}')
            data[label] = importance

        result = pd.DataFrame(data).fillna(0)

        if feature_name:
            if feature_name in result.index:
                return result.loc[[feature_name]]
            return pd.DataFrame()

        return result

    def get_regime_specific_features(self, top_k=5):
        """
        获取各市场状态下最独特的特征（与全局重要性差异最大的特征）

        参数:
        - top_k: 每个状态返回的独特特征数量

        返回:
        - dict: {状态: [独特特征列表]}
        """
        if not self.regime_importance or self.global_importance is None:
            return {}

        result = {}
        for regime, importance in self.regime_importance.items():
            label = {0: '震荡', 1: '上涨', 2: '下跌'}.get(regime, f'状态{regime}')
            # 差异 = 状态重要性 - 全局重要性
            diff = importance - self.global_importance.reindex(importance.index).fillna(0)
            # 排序：差异最大的在前（该状态下相对更重要的特征）
            unique_features = diff.sort_values(ascending=False).head(top_k).index.tolist()
            result[label] = unique_features

        return result

    def _save_results(self):
        """保存分析结果"""
        results = {}

        if self.global_importance is not None:
            results['global_importance'] = self.global_importance.to_dict()

        for regime, importance in self.regime_importance.items():
            results[f'regime_{regime}_importance'] = importance.to_dict()

        cache_file = os.path.join(SHAP_CACHE_DIR, 'shap_analysis.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def load_results(self):
        """加载缓存的分析结果"""
        cache_file = os.path.join(SHAP_CACHE_DIR, 'shap_analysis.json')
        if not os.path.exists(cache_file):
            return False

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if 'global_importance' in results:
                self.global_importance = pd.Series(results['global_importance'])

            for regime in [0, 1, 2]:
                key = f'regime_{regime}_importance'
                if key in results:
                    self.regime_importance[regime] = pd.Series(results[key])

            return True
        except Exception:
            return False

    def print_summary(self):
        """打印特征重要性摘要"""
        if self.global_importance is None:
            print("  ⚠️ 尚未进行 SHAP 分析")
            return

        print("\n" + "=" * 60)
        print("📊 SHAP 特征重要性分析")
        print("=" * 60)

        print("\n全局 Top 15 特征:")
        for i, (feat, imp) in enumerate(self.global_importance.head(15).items(), 1):
            print(f"  {i:2d}. {feat:<35s} {imp:.6f}")

        if self.regime_importance:
            print("\n市场状态特异性特征:")
            regime_features = self.get_regime_specific_features(top_k=3)
            for regime_name, features in regime_features.items():
                print(f"  {regime_name}: {', '.join(features)}")
