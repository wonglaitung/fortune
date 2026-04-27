#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息衰减分析模块

计算每个特征在不同 Lag（1,5,10,20）下与目标变量的互信息（Mutual Information），
识别"快变量"（短期有效）和"慢变量"（长期有效）。

新增特征：
- MI_Fast_Signal_Count: 快变量中正值特征的数量（短期动量共识强度）
- MI_Slow_Signal_Count: 慢变量中正值特征的数量（长期趋势共识强度）
- MI_Fast_Slow_Divergence: 快慢信号背离=潜在反转
- MI_Decay_Rate_RSI: RSI 在各 lag 的 MI 归一化衰减速率
- MI_Decay_Rate_MACD: MACD_Hist 的 MI 归一化衰减速率

两阶段实现：
- 阶段 1：离线 MI 分析（不纳入生产流程，生成 mi_lag_assignments.json）
- 阶段 2：运行时特征生成（读取 mi_lag_assignments.json，计算聚合特征）

防泄漏关键规则：
- MI 分析结果在 Walk-forward 的每个 fold 开始时基于训练数据重新计算
- lag 分配在 fold 内保持冻结
- 所有特征保持 shift(1) 最小偏移

依赖：sklearn.feature_selection.mutual_info_classif
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

# 配置
CACHE_DIR = 'data/mi_cache'
MI_ASSIGNMENTS_FILE = os.path.join(CACHE_DIR, 'mi_lag_assignments.json')

# Lag 定义
LAGS = [1, 5, 10, 20]

# 快变量定义（lag <= 5）
FAST_LAG_THRESHOLD = 5


class InfoDecayAnalyzer:
    """信息衰减分析器"""

    def __init__(self, cache_dir=CACHE_DIR):
        """
        参数:
        - cache_dir: MI 分析结果缓存目录
        """
        self.cache_dir = cache_dir
        self.mi_assignments = None

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

    def analyze_mutual_information(self, df, feature_cols, target_col='Target',
                                   lags=LAGS, save_to_file=True):
        """
        阶段 1：离线 MI 分析

        计算每个特征在不同 Lag 下的互信息，识别最优 Lag 和衰减速率。

        参数:
        - df: 包含特征和目标的 DataFrame
        - feature_cols: 特征列名列表
        - target_col: 目标列名
        - lags: 要分析的 Lag 列表
        - save_to_file: 是否保存结果到文件

        返回:
        - dict: 每个特征的最优 Lag 和衰减速率
        """
        from sklearn.feature_selection import mutual_info_classif

        print("  📊 计算互信息衰减分析...")

        results = {}

        for feature in feature_cols:
            if feature not in df.columns:
                continue

            mi_values = []
            for lag in lags:
                # 对特征应用 lag
                lagged_feature = df[feature].shift(lag)

                # 创建有效数据掩码（特征和目标都非 NaN）
                valid_mask = lagged_feature.notna() & df[target_col].notna()

                if valid_mask.sum() < 50:
                    mi_values.append(0.0)
                    continue

                X = lagged_feature[valid_mask].values.reshape(-1, 1)
                y = df.loc[valid_mask, target_col].values

                # 计算互信息
                try:
                    mi = mutual_info_classif(X, y, random_state=42)[0]
                    mi_values.append(mi)
                except Exception:
                    mi_values.append(0.0)

            # 找到最优 Lag（MI 最大的 Lag）
            optimal_lag = lags[np.argmax(mi_values)] if mi_values else 1

            # 计算衰减速率（归一化）
            # 衰减速率 = (MI_lag1 - MI_lag20) / (MI_lag1 + 1e-10)
            if len(mi_values) >= 4 and mi_values[0] > 0:
                decay_rate = (mi_values[0] - mi_values[-1]) / (mi_values[0] + 1e-10)
            else:
                decay_rate = 0.0

            results[feature] = {
                'optimal_lag': int(optimal_lag),
                'decay_rate': float(decay_rate),
                'mi_values': [float(v) for v in mi_values],
                'is_fast': optimal_lag <= FAST_LAG_THRESHOLD,
            }

        # 保存结果
        if save_to_file:
            with open(MI_ASSIGNMENTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✅ MI 分析结果已保存到 {MI_ASSIGNMENTS_FILE}")

        self.mi_assignments = results
        return results

    def load_mi_assignments(self):
        """加载 MI 分析结果"""
        if os.path.exists(MI_ASSIGNMENTS_FILE):
            with open(MI_ASSIGNMENTS_FILE, 'r') as f:
                self.mi_assignments = json.load(f)
            return True
        return False

    def calculate_features(self, df, feature_cols=None):
        """
        阶段 2：运行时特征生成

        读取 mi_lag_assignments.json，计算快/慢信号聚合特征和衰减速率特征。

        参数:
        - df: 包含特征的 DataFrame
        - feature_cols: 要分析的特征列名列表（如果为 None，使用所有已知特征）

        返回:
        - DataFrame: 添加了信息衰减特征的 DataFrame
        """
        print("  📈 计算信息衰减特征...")

        # 尝试加载 MI 分析结果
        if not self.load_mi_assignments():
            print("  ⚠️ MI 分析结果不存在，使用默认值")
            self._fill_default_features(df)
            return df

        if feature_cols is None:
            feature_cols = list(self.mi_assignments.keys())

        # 计算快/慢信号计数
        fast_count = 0
        slow_count = 0

        for feature in feature_cols:
            if feature not in self.mi_assignments:
                continue

            assignment = self.mi_assignments[feature]
            optimal_lag = assignment['optimal_lag']

            if feature not in df.columns:
                continue

            # 获取特征在最优 Lag 下的值
            lagged_value = df[feature].shift(optimal_lag)

            # 判断信号方向
            if assignment['is_fast']:
                # 快变量：短期有效，正值表示看涨
                if lagged_value.iloc[-1] > 0 if len(lagged_value) > 0 else False:
                    fast_count += 1
            else:
                # 慢变量：长期有效，正值表示看涨
                if lagged_value.iloc[-1] > 0 if len(lagged_value) > 0 else False:
                    slow_count += 1

        # 计算快/慢信号计数（滚动方式）
        fast_signal_counts = []
        slow_signal_counts = []

        for i in range(len(df)):
            fast_c = 0
            slow_c = 0

            for feature in feature_cols:
                if feature not in self.mi_assignments or feature not in df.columns:
                    continue

                assignment = self.mi_assignments[feature]
                optimal_lag = assignment['optimal_lag']
                is_fast = assignment['is_fast']

                # 获取特征在最优 Lag 下的值
                idx = i - optimal_lag
                if idx < 0:
                    continue

                value = df[feature].iloc[idx]

                if pd.isna(value):
                    continue

                if is_fast:
                    if value > 0:
                        fast_c += 1
                else:
                    if value > 0:
                        slow_c += 1

            fast_signal_counts.append(fast_c)
            slow_signal_counts.append(slow_c)

        df['MI_Fast_Signal_Count'] = fast_signal_counts
        df['MI_Slow_Signal_Count'] = slow_signal_counts
        df['MI_Fast_Slow_Divergence'] = df['MI_Fast_Signal_Count'] - df['MI_Slow_Signal_Count']

        # 计算特定特征的衰减速率
        # RSI 衰减速率
        if 'RSI' in self.mi_assignments:
            df['MI_Decay_Rate_RSI'] = self.mi_assignments['RSI']['decay_rate']
        else:
            df['MI_Decay_Rate_RSI'] = 0.0

        # MACD_Hist 衰减速率
        if 'MACD_Hist' in self.mi_assignments:
            df['MI_Decay_Rate_MACD'] = self.mi_assignments['MACD_Hist']['decay_rate']
        else:
            df['MI_Decay_Rate_MACD'] = 0.0

        # ⚠️ 数据泄漏防护：对计数特征应用 shift(1)
        for col in ['MI_Fast_Signal_Count', 'MI_Slow_Signal_Count', 'MI_Fast_Slow_Divergence']:
            df[col] = df[col].shift(1)

        feature_count = len([c for c in df.columns if c in self.get_feature_names()])
        print(f"  ✅ 信息衰减特征计算完成（{feature_count} 个特征）")

        return df

    def _fill_default_features(self, df):
        """填充默认特征值"""
        for col in self.get_feature_names():
            df[col] = 0.0

    @staticmethod
    def get_feature_names():
        """返回所有信息衰减特征名"""
        return [
            'MI_Fast_Signal_Count',
            'MI_Slow_Signal_Count',
            'MI_Fast_Slow_Divergence',
            'MI_Decay_Rate_RSI',
            'MI_Decay_Rate_MACD',
        ]


# 信息衰减特征配置（用于 FEATURE_CONFIG）
INFO_DECAY_FEATURE_CONFIG = {
    'info_decay_features': InfoDecayAnalyzer.get_feature_names()
}
