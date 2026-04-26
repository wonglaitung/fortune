#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GARCH 波动率建模模块

对恒生指数收益率拟合 GARCH(1,1) 模型，提取条件波动率特征：
- GARCH_Conditional_Vol: 条件波动率（模型预测的当日波动率）
- GARCH_Vol_Ratio: 条件波动率与历史波动率的比率（>1表示波动率可能上升）
- GARCH_Persistence: 波动率持续性（alpha1 + beta1，越接近1持续性越强）
- GARCH_Vol_Change_5d: 条件波动率5日变化率

依赖：arch 库（pip install arch）
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')


class GARCHVolatilityModel:
    """GARCH(1,1) 波动率模型"""

    def __init__(self, vol_type='GARCH', p=1, q=1, dist='normal'):
        """
        参数:
        - vol_type: 波动率模型类型 ('GARCH', 'EGARCH', 'GJR-GARCH')
        - p: GARCH 阶数
        - q: ARCH 阶数
        - dist: 误差分布 ('normal', 't', 'skewt')
        """
        self.vol_type = vol_type
        self.p = p
        self.q = q
        self.dist = dist
        self.model_result = None
        self.conditional_volatility = None

    def fit(self, returns):
        """
        拟合 GARCH 模型

        参数:
        - returns: 收益率序列（百分比形式，如 0.01 表示 1%）

        返回:
        - self
        """
        from arch import arch_model

        # 去除 NaN
        clean_returns = returns.dropna()

        # 将收益率放大100倍，从比例转为百分比（GARCH拟合更稳定）
        scaled_returns = clean_returns * 100

        # 创建 GARCH 模型
        model = arch_model(
            scaled_returns,
            vol=self.vol_type,
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean='Zero',  # 零均值（收益率已去均值）
            rescale=False
        )

        # 拟合模型（静默模式）
        self.model_result = model.fit(disp='off', show_warning=False)
        self.conditional_volatility = self.model_result.conditional_volatility / 100  # 缩放回原比例

        return self

    def get_persistence(self):
        """
        获取波动率持续性参数（alpha + beta）

        返回:
        - float: 波动率持续性（0-1之间，越接近1持续性越强）
        """
        if self.model_result is None:
            return np.nan

        params = self.model_result.params

        if self.vol_type == 'GARCH':
            # alpha[1] + beta[1]
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            return alpha + beta
        elif self.vol_type == 'EGARCH':
            # EGARCH 持续性 = beta[1]
            return params.get('beta[1]', 0)
        else:
            return np.nan

    def calculate_features(self, df, return_col='Return_1d'):
        """
        计算 GARCH 波动率特征

        参数:
        - df: 包含收益率列的 DataFrame
        - return_col: 收益率列名

        返回:
        - DataFrame: 添加了 GARCH 特征的 DataFrame
        """
        print("  📈 计算 GARCH 波动率特征...")

        # 使用原始收益率（不用 shift，因为 GARCH 本身就是条件模型）
        # 但 GARCH 条件波动率是基于截至 t-1 的信息预测 t 时刻的波动率
        # 所以天然避免了数据泄漏

        # 计算原始收益率（如果 Return_1d 是 shift(1) 后的，需要用未 shift 的）
        if return_col in df.columns:
            raw_returns = df[return_col]
        else:
            # 回退：直接计算
            raw_returns = df['Close'].pct_change()

        try:
            # 拟合 GARCH 模型
            self.fit(raw_returns)

            if self.conditional_volatility is not None:
                # 对齐索引
                cond_vol = pd.Series(
                    self.conditional_volatility,
                    index=self.model_result.conditional_volatility.index,
                    name='GARCH_Conditional_Vol'
                )

                # 将 GARCH 结果合并到原始 DataFrame
                # GARCH 的索引是原始收益率的非 NaN 索引
                garch_df = pd.DataFrame(index=cond_vol.index)
                garch_df['GARCH_Conditional_Vol'] = cond_vol.values

                # 计算历史波动率（20日）
                garch_df['Hist_Vol_20d'] = raw_returns.rolling(window=20).std()

                # GARCH 条件波动率 / 历史波动率 比率
                # > 1: 模型认为当前波动率高于历史平均，风险增加
                # < 1: 模型认为当前波动率低于历史平均，相对平静
                garch_df['GARCH_Vol_Ratio'] = (
                    garch_df['GARCH_Conditional_Vol'] /
                    (garch_df['Hist_Vol_20d'] + 1e-10)
                )

                # GARCH 波动率5日变化率
                garch_df['GARCH_Vol_Change_5d'] = (
                    garch_df['GARCH_Conditional_Vol'].pct_change(5)
                )

                # 波动率持续性
                persistence = self.get_persistence()
                garch_df['GARCH_Persistence'] = persistence

                # 合并到主 DataFrame（按索引对齐）
                for col in ['GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
                           'GARCH_Vol_Change_5d', 'GARCH_Persistence']:
                    df[col] = np.nan
                    # 只填充有 GARCH 结果的行
                    common_idx = df.index.intersection(garch_df.index)
                    df.loc[common_idx, col] = garch_df.loc[common_idx, col].values

                # ⚠️ 数据泄漏防护：GARCH 条件波动率是基于 t-1 之前的信息
                # 但为安全起见，对 GARCH 特征也做 shift(1)
                for col in ['GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
                           'GARCH_Vol_Change_5d']:
                    df[col] = df[col].shift(1)

                feature_count = len([c for c in df.columns if c in self.get_feature_names()])
                print(f"  ✅ GARCH 波动率特征计算完成（{feature_count} 个特征，持续性={persistence:.4f}）")
            else:
                self._fill_default_features(df)

        except Exception as e:
            print(f"  ⚠️ GARCH 模型拟合失败: {e}，使用默认值")
            self._fill_default_features(df)

        return df

    def _fill_default_features(self, df):
        """GARCH 失败时填充默认值"""
        for col in self.get_feature_names():
            if col == 'GARCH_Persistence':
                df[col] = 0.8  # 典型的持续性值
            else:
                df[col] = 0.0

    @staticmethod
    def get_feature_names():
        """返回所有 GARCH 特征名"""
        return [
            'GARCH_Conditional_Vol',
            'GARCH_Vol_Ratio',
            'GARCH_Vol_Change_5d',
            'GARCH_Persistence',
        ]


# GARCH 特征配置（用于 FEATURE_CONFIG）
GARCH_FEATURE_CONFIG = {
    'garch_features': GARCHVolatilityModel.get_feature_names()
}
