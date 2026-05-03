#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征残差化模块 - 剔除宏观因子对微观特征的影响

功能：
- 对动量、成交量等微观特征，剔除美债利率、VIX等宏观因子贡献
- 保留宏观特征作为独立输入（用于择时判断）
- 降低特征之间的共线性，提高模型选股能力

使用方法：
    from data_services.feature_residualizer import FeatureResidualizer

    residualizer = FeatureResidualizer()
    df_residual = residualizer.residualize(df)

理论基础：
- 港股是离岸市场，定价权在美债利率（分母端）和内地基本面（分子端）之间拉锯
- 不剔除宏观因子贡献，模型会变成"美债利率预测器"，导致"全涨全跌"
- 残差化后，模型被迫学习个股特异性特征，而非依赖宏观信号

参考文档：
- docs/FEATURE_ENGINEERING.md - 全局特征与标签设计章节
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


class FeatureResidualizer:
    """特征残差化：剔除宏观因子对微观特征的影响"""

    # 宏观特征列表（全局特征，对所有股票"一视同仁"）
    # 注意：实际数据中有 130 个宏观相关特征，这里列出核心的原始宏观特征
    # 交叉特征（如 10d_Trend_HSI_Return_5d）由这些原始特征衍生
    MACRO_FEATURES = [
        # 美债利率
        'US_10Y_Yield',          # 美国10年期国债收益率
        'US_10Y_Yield_Change',   # 美债利率变化

        # VIX 波动率
        'VIX_Level',             # VIX波动率水平
        'VIX_Change',            # VIX变化
        'VIX_Ratio_MA20',        # VIX相对MA20比率

        # 美股指数
        'SP500_Return',          # 标普500收益率
        'SP500_Return_5d',       # 标普500 5日收益率
        'SP500_Return_20d',      # 标普500 20日收益率
        'NASDAQ_Return',         # 纳斯达克收益率
        'NASDAQ_Return_5d',      # 纳斯达克 5日收益率
        'NASDAQ_Return_20d',     # 纳斯达克 20日收益率

        # 恒生指数
        'HSI_Return_1d',         # 恒指1日收益率
        'HSI_Return_5d',         # 恒指5日收益率
        'HSI_Return_20d',        # 恒指20日收益率
        'HSI_Return_60d',        # 恒指60日收益率

        # HSI 市场状态（HMM）
        'HSI_Regime_Prob_0',     # 震荡市概率
        'HSI_Regime_Prob_1',     # 牛市概率
        'HSI_Regime_Prob_2',     # 熊市概率
        'HSI_Regime_Duration',   # 当前状态持续时间
        'HSI_Regime_Transition_Prob',  # 状态转换概率
    ]

    # 微观特征列表（个股特异性特征）
    # 注意：实际数据中有 357 个微观特征，这里列出核心的原始微观特征
    MICRO_FEATURES = [
        # 动量特征
        'Momentum_Accel_5d',     # 5日动量加速
        'Momentum_Accel_10d',    # 10日动量加速
        'Price_Pct_20d',         # 20日价格百分位
        'Close_Position',        # 收盘价位置

        # 成交量特征
        'Volume_Ratio_5d',       # 5日成交量比率
        'Volume_Volatility',     # 成交量波动率
        'OBV',                   # 能量潮
        'CMF',                   # Chaikin资金流量

        # 技术指标 - RSI
        'RSI',                   # RSI（实际数据中的名称）
        'RSI_Deviation',         # RSI偏离度
        'RSI_ROC',               # RSI变化率

        # 技术指标 - MACD
        'MACD',                  # MACD
        'MACD_signal',           # MACD信号线（实际数据中的名称）
        'MACD_histogram',        # MACD柱（实际数据中的名称）
        'MACD_Hist',             # MACD柱（另一个名称）

        # 技术指标 - 布林带
        'BB_Position',           # 布林带位置
        'BB_Width',              # 布林带宽度
        'BB_Width_Normalized',   # 标准化布林带宽度

        # 技术指标 - ATR
        'ATR',                   # ATR
        'ATR_Ratio',             # ATR比率
        'ATR_Risk_Score',        # ATR风险评分

        # 技术指标 - ADX
        'ADX',                   # ADX
        '+DI',                   # +DI
        '-DI',                   # -DI

        # 波动率特征
        'Volatility_30pct',      # 30日波动率

        # 均线偏离
        'MA5_Deviation_Std',     # MA5偏离度（标准化）
        'MA20_Deviation_Std',    # MA20偏离度（标准化）
        'MA120_Deviation',       # MA120偏离度
        'MA250_Deviation',       # MA250偏离度

        # 趋势特征
        '10d_Trend',             # 10日趋势
        '20d_Trend',             # 20日趋势
        '60d_Trend',             # 60日趋势

        # 异常检测特征
        'Anomaly_Severity_Score',  # 异常严重程度
        'Anomaly_Buy_Signal',      # 异常买入信号
        'Anomaly_Wait_Signal',     # 异常等待信号
    ]

    # 交叉特征前缀（这些特征包含宏观成分，但结构复杂，暂不处理）
    CROSS_FEATURE_PREFIXES = [
        '10d_Trend_',
        '20d_Trend_',
        '3d_Trend_',
        '5d_Trend_',
        '60d_Trend_',
        'Outperforms_HSI_',
        'Strong_Volume_Up_',
        'Weak_Volume_Down_',
    ]

    def __init__(
        self,
        macro_features: Optional[List[str]] = None,
        micro_features: Optional[List[str]] = None,
        min_samples: int = 100
    ):
        """
        初始化特征残差化器

        Args:
            macro_features: 宏观特征列表（默认使用 MACRO_FEATURES）
            micro_features: 微观特征列表（默认使用 MICRO_FEATURES）
            min_samples: 最小样本数（用于拟合回归模型）
        """
        self.macro_features = macro_features or self.MACRO_FEATURES
        self.micro_features = micro_features or self.MICRO_FEATURES
        self.min_samples = min_samples
        self.models: Dict[str, LinearRegression] = {}  # 存储每个微观特征的回归模型

    def residualize(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
        keep_original: bool = False
    ) -> pd.DataFrame:
        """
        对每个微观特征，剔除宏观因子贡献

        Args:
            df: 包含特征的 DataFrame
            inplace: 是否原地修改（默认 False，返回新 DataFrame）
            keep_original: 是否保留原始特征（默认 False，残差直接替换原始特征）

        Returns:
            DataFrame: 包含残差特征的 DataFrame
        """
        if not inplace:
            df_residual = df.copy()
        else:
            df_residual = df

        # 检查宏观特征是否存在
        available_macro = [f for f in self.macro_features if f in df.columns]
        if not available_macro:
            warnings.warn("没有可用的宏观特征，无法进行残差化")
            return df_residual

        # 对每个微观特征进行残差化
        residualized_count = 0
        for micro_feat in self.micro_features:
            if micro_feat not in df.columns:
                continue

            # 检查有效样本数（排除 NaN 和无穷大值）
            required_cols = available_macro + [micro_feat]
            valid_mask = df_residual[required_cols].notna().all(axis=1)
            # 额外检查无穷大值
            for col in required_cols:
                if df_residual[col].dtype in ['float64', 'float32']:
                    valid_mask &= np.isfinite(df_residual[col])

            valid_count = valid_mask.sum()

            if valid_count < self.min_samples:
                continue  # 静默跳过，避免过多警告

            try:
                # 拟合回归模型：微观特征 = β * 宏观特征 + 残差
                model = LinearRegression()
                X_train = df_residual.loc[valid_mask, available_macro].values
                y_train = df_residual.loc[valid_mask, micro_feat].values

                model.fit(X_train, y_train)

                # 存储模型（用于后续预测）
                self.models[micro_feat] = model

                # 计算残差：残差 = 实际值 - 预测值
                residual_name = f'{micro_feat}_Residual'
                # 预测时填充 NaN 为 0
                X_pred = df_residual[available_macro].fillna(0).values
                predicted_values = model.predict(X_pred)
                df_residual[residual_name] = df_residual[micro_feat].fillna(0) - predicted_values

                # 如果不保留原始特征，用残差替换
                if not keep_original:
                    df_residual[micro_feat] = df_residual[residual_name]
                    df_residual.drop(columns=[residual_name], inplace=True)

                residualized_count += 1

            except Exception as e:
                # 静默跳过失败的特征
                continue

        return df_residual

    def get_residual_features(self) -> List[str]:
        """获取残差特征名称列表"""
        return [f'{f}_Residual' for f in self.micro_features if f in self.models]

    def get_model_coefficients(self) -> pd.DataFrame:
        """
        获取回归模型系数（用于分析宏观因子对微观特征的影响）

        Returns:
            DataFrame: 每个微观特征对宏观因子的回归系数
        """
        if not self.models:
            return pd.DataFrame()

        coef_data = {}
        for micro_feat, model in self.models.items():
            coef_data[micro_feat] = dict(zip(self.macro_features, model.coef_))

        return pd.DataFrame(coef_data).T

    def analyze_macro_impact(self) -> Dict[str, float]:
        """
        分析宏观因子对微观特征的总体影响程度（R²）

        Returns:
            Dict: 每个微观特征的 R² 值（宏观因子解释的比例）
        """
        if not self.models:
            return {}

        impact = {}
        for micro_feat, model in self.models.items():
            # R² 表示宏观因子解释的方差比例
            impact[micro_feat] = model.score(
                self._get_training_data(micro_feat)['macro'],
                self._get_training_data(micro_feat)['micro']
            )

        return impact

    def _get_training_data(self, micro_feat: str) -> Dict:
        """获取训练数据（内部方法）"""
        # 此方法需要在外部调用 residualize() 后才能使用
        # 返回空字典作为占位符
        return {'macro': None, 'micro': None}


def residualize_features(
    df: pd.DataFrame,
    macro_features: Optional[List[str]] = None,
    micro_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    快捷函数：对 DataFrame 进行特征残差化

    Args:
        df: 包含特征的 DataFrame
        macro_features: 宏观特征列表
        micro_features: 微观特征列表

    Returns:
        DataFrame: 包含残差特征的 DataFrame
    """
    residualizer = FeatureResidualizer(
        macro_features=macro_features,
        micro_features=micro_features
    )
    return residualizer.residualize(df)


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
    n = len(dates)

    # 模拟数据
    df = pd.DataFrame({
        'US_10Y_Yield': np.random.normal(4.5, 0.5, n),
        'VIX_Level': np.random.normal(20, 5, n),
        'Momentum_20d': np.random.normal(0, 0.1, n) + np.random.normal(4.5, 0.5, n) * 0.02,  # 受利率影响
        'RSI_14': np.random.normal(50, 15, n),
        'Volume_Ratio_5d': np.random.normal(1, 0.3, n),
    }, index=dates)

    # 残差化
    residualizer = FeatureResidualizer()
    df_residual = residualizer.residualize(df)

    print("原始特征:")
    print(df.head())
    print("\n残差特征:")
    print(df_residual[['Momentum_20d_Residual', 'RSI_14_Residual', 'Volume_Ratio_5d_Residual']].head())

    # 分析宏观因子影响
    print("\n宏观因子对微观特征的回归系数:")
    print(residualizer.get_model_coefficients())