#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨尺度关联特征模块

计算多时间尺度之间的关联特征，识别趋势转折信号：
- Vol_of_Vol_20d: 波动率的波动率（vol-of-vol 飙升预示趋势转折）
- Vol_Ratio_5d_20d: 短期/长期波动率比率（>1 表示短期波动加剧）
- Return_1d_5d_Correlation: 1d/5d 收益率的滚动相关性（跨尺度耦合强度）
- Vol_Cluster_Signal: 波动率聚集信号（趋势转折前兆）
- Momentum_Consistency: 三周期动量方向一致性 [-3,+3]

核心洞察：
- 短期波动加剧（即使均值没变）往往预示中长期趋势的转折
- Momentum_Consistency 将三周期一致/背离模式编码为连续特征
- Vol_of_Vol_20d 是经典的"波动率体制转换"前导指标

依赖：pandas, numpy
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')


class MultiscaleFeatureCalculator:
    """跨尺度关联特征计算器"""

    def __init__(self, lookback=252):
        """
        参数:
        - lookback: 回看窗口（交易日），用于分位数计算
        """
        self.lookback = lookback

    def calculate_features(self, df):
        """
        计算跨尺度关联特征

        参数:
        - df: 包含 Close 列的 DataFrame

        返回:
        - DataFrame: 添加了跨尺度特征的 DataFrame
        """
        print("  📊 计算跨尺度关联特征...")

        # ========== 1. Vol_of_Vol_20d - 波动率的波动率 ==========
        # 使用已有的 Volatility_20d（已经 shift(1)）
        if 'Volatility_20d' in df.columns:
            vol_20d = df['Volatility_20d']
        else:
            # 回退：重新计算
            returns = df['Close'].pct_change()
            vol_20d = returns.rolling(window=20).std().shift(1)

        # 波动率的 20 日滚动标准差
        df['Vol_of_Vol_20d'] = vol_20d.rolling(window=20).std()

        # ========== 2. Vol_Ratio_5d_20d - 短期/长期波动率比率 ==========
        # 使用已有的 Volatility_20d
        if 'Volatility_20d' in df.columns:
            vol_20d = df['Volatility_20d']
        else:
            vol_20d = df['Close'].pct_change().rolling(window=20).std().shift(1)

        # 计算 5 日波动率
        if 'Return_1d' in df.columns:
            # Return_1d 已经是 shift(1) 后的
            vol_5d = df['Close'].pct_change().rolling(window=5).std().shift(1)
        else:
            vol_5d = df['Close'].pct_change().rolling(window=5).std().shift(1)

        # 短期波动率 / 长期波动率
        df['Vol_Ratio_5d_20d'] = (vol_5d / (vol_20d + 1e-10))

        # ========== 3. Return_1d_5d_Correlation - 跨尺度收益相关性 ==========
        # 使用已有的 Return_1d 和 Return_5d
        if 'Return_1d' in df.columns and 'Return_5d' in df.columns:
            r_1d = df['Return_1d']
            r_5d = df['Return_5d']
        else:
            # 回退：重新计算
            r_1d = df['Close'].pct_change().shift(1)
            r_5d = df['Close'].pct_change(5).shift(1)

        # 滚动相关性（20 日窗口）
        df['Return_1d_5d_Correlation'] = r_1d.rolling(window=20).corr(r_5d)

        # ========== 4. Vol_Cluster_Signal - 波动率聚集信号 ==========
        # 当 vol-of-vol 超过 60 日分位数的 90% 时，标记为波动率聚集
        vol_of_vol = df['Vol_of_Vol_20d']
        vol_of_vol_quantile_90 = vol_of_vol.rolling(window=60, min_periods=30).quantile(0.9)

        df['Vol_Cluster_Signal'] = (vol_of_vol > vol_of_vol_quantile_90).astype(int)

        # ========== 5. Momentum_Consistency - 三周期动量一致性 ==========
        # 使用已有的 Return 特征
        if all(col in df.columns for col in ['Return_1d', 'Return_5d', 'Return_20d']):
            r_1d = df['Return_1d']
            r_5d = df['Return_5d']
            r_20d = df['Return_20d']
        else:
            # 回退：重新计算
            r_1d = df['Close'].pct_change().shift(1)
            r_5d = df['Close'].pct_change(5).shift(1)
            r_20d = df['Close'].pct_change(20).shift(1)

        df['Momentum_Consistency'] = (
            np.sign(r_1d) + np.sign(r_5d) + np.sign(r_20d)
        )

        feature_count = len([c for c in df.columns if c in self.get_feature_names()])
        print(f"  ✅ 跨尺度关联特征计算完成（{feature_count} 个特征）")

        return df

    @staticmethod
    def get_feature_names():
        """返回所有跨尺度特征名"""
        return [
            'Vol_of_Vol_20d',
            'Vol_Ratio_5d_20d',
            'Return_1d_5d_Correlation',
            'Vol_Cluster_Signal',
            'Momentum_Consistency',
        ]


# 跨尺度特征配置（用于 FEATURE_CONFIG）
MULTISCALE_FEATURE_CONFIG = {
    'multiscale_features': MultiscaleFeatureCalculator.get_feature_names()
}
