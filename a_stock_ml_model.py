#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股机器学习交易模型

继承港股模型结构，适配A股特有特征：
- 涨跌停限制
- 北向资金
- 融资融券
- 龙虎榜
- 样本权重训练（核心股3.0，扩展股1.0）
- 双指数市场特征（中证1000 + 创业板指）
- 滚动网络特征（防止时序泄漏）

版本：v2.0 (2026-07-17)
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入港股模型
from ml_services.ml_trading_model import CatBoostModel, FeatureEngineer, ABSOLUTE_PRICE_FEATURES, logger

# 导入A股配置和数据服务
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_TRAINING_LIST,
    A_STOCK_SECTOR_MAPPING,
    A_STOCK_WEIGHTS,
    is_core_holding,
    get_limit_rate,
    get_market_code,
    get_sample_weight,
    A_STOCK_MARKET_INDEX,
    A_STOCK_NETWORK_FEATURES_DIR,
)

# A股模型保存路径
A_STOCK_MODELS_DIR = 'data/a_stock_models'
from data_services.a_stock_data import get_a_stock_data, get_index_data

# ========== A股市场级特征列表 ==========
# 所有股票同值的市场级特征，需与网络特征交叉后使用
# 替代港股的 MARKET_LEVEL_FEATURES（HSI_Return、HSI_Market_Regime等）

A_STOCK_MARKET_LEVEL_FEATURES = [
    # 中证1000收益（中小盘Beta信号）
    'CSI1000_Return_1d', 'CSI1000_Return_5d', 'CSI1000_Return_20d',
    # 创业板指收益（成长股风险偏好）
    'CYB_Return_1d', 'CYB_Return_5d', 'CYB_Return_20d',
    # 强弱对比因子（个股相对于中证1000）
    'RS_CSI1000_1d', 'RS_CSI1000_5d',
    # 市场情绪互动（创业板量能/中证1000量能）
    'Sentiment_Ratio', 'Sentiment_Ratio_Change_5d',
    # A股市场状态（HMM检测，使用中证1000）
    'AStock_Market_Regime', 'AStock_Regime_Prob_0', 'AStock_Regime_Prob_1', 'AStock_Regime_Prob_2',
    'AStock_Regime_Duration', 'AStock_Regime_Transition_Prob',
    # 美股特征（保留，对A股有参考价值）
    'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
    'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
    'VIX', 'VIX_Change_5d', 'VIX_Level', 'VIX_Change', 'VIX_Ratio_MA20',
    # 利率特征（保留）
    'US_2Y_Yield', 'US_10Y_Yield', 'US_30Y_Yield',
    'CN_10Y_Yield', 'CN_US_10Y_Spread',
]

# 标记是否已应用 monkey-patch
_patched = False


def _calculate_a_stock_regime_features(csi1000_df, use_shift=True):
    """
    计算A股市场状态特征（使用HMM + 中证1000）

    复用港股的 RegimeDetector 类，但输入中证1000数据，
    输出特征添加 AStock_ 前缀避免与港股 HSI_ 特征冲突。

    Args:
        csi1000_df: 中证1000指数数据
        use_shift: 是否使用滞后数据（Walk-forward用True，收市后预测用False）

    Returns:
        DataFrame: A股市场状态特征（AStock_Market_Regime等）
    """
    if csi1000_df is None or csi1000_df.empty:
        return None

    from data_services.regime_detector import RegimeDetector

    try:
        detector = RegimeDetector(n_states=3, lookback=252)
        csi1000_with_regime = detector.calculate_features(csi1000_df.copy(), use_shift=use_shift)

        # 重命名列（添加 AStock_ 前缀，避免与港股 HSI_ 特征冲突）
        feature_names = RegimeDetector.get_feature_names()
        rename_map = {c: f'AStock_{c}' for c in feature_names}
        regime_df = csi1000_with_regime[feature_names].rename(columns=rename_map)

        logger.info(f"A股市场状态特征计算完成: {list(regime_df.columns)}")
        return regime_df

    except Exception as e:
        logger.warning(f"A股市场状态特征计算失败: {e}")
        return None


def _apply_a_stock_patch():
    """应用 A股数据源替换（延迟执行，只在需要时调用）"""
    global _patched
    if _patched:
        return

    import ml_services.ml_trading_model as ml_module

    # 替换数据源函数
    def _get_a_stock_data_wrapper(stock_code, period_days=500):
        code = stock_code.replace('.HK', '')
        return get_a_stock_data(code, period_days=period_days, use_cache=True)

    def _get_index_data_wrapper(period_days=500):
        return get_index_data('sh', period_days=period_days)

    ml_module.get_hk_stock_data_tencent = _get_a_stock_data_wrapper
    ml_module.get_hsi_data_tencent = _get_index_data_wrapper

    _patched = True
    logger.debug("已应用A股数据源替换")


class AStockFeatureEngineer(FeatureEngineer):
    """A股特征工程类 - 继承港股特征工程，添加A股特有特征"""

    def __init__(self):
        super().__init__()
        self.market = 'a_stock'

    def create_fundamental_features(self, code):
        """
        创建A股基本面特征（重写父类方法）

        使用腾讯财经 API 获取 A 股实时基本面数据，替代港股接口。

        Args:
            code: 股票代码（如 '300440'）

        Returns:
            dict: 基本面特征字典
        """
        try:
            import requests

            # 确定市场代码
            market = get_market_code(code)
            url = f'http://qt.gtimg.cn/q={market}{code}'

            response = requests.get(url, timeout=10)
            data = response.text

            if data.startswith('v_'):
                parts = data.split('~')
                if len(parts) > 46:
                    result = {
                        'PE': np.nan,
                        'PB': np.nan,
                        'Market_Cap': np.nan,
                        'ROE': np.nan,
                        'ROA': np.nan,
                        'Dividend_Yield': np.nan,
                        'EPS': np.nan,
                        'Net_Margin': np.nan,
                        'Gross_Margin': np.nan,
                    }

                    # 解析腾讯财经数据
                    # 参考：http://qt.gtimg.cn/q=sz300440
                    # 格式：v_sz300440="51~运达科技~300440~..."

                    try:
                        # 市盈率（动态）- 索引39
                        if len(parts) > 39 and parts[39]:
                            result['PE'] = float(parts[39])
                    except (ValueError, TypeError):
                        pass

                    try:
                        # 市净率 - 索引46
                        if len(parts) > 46 and parts[46]:
                            result['PB'] = float(parts[46])
                    except (ValueError, TypeError):
                        pass

                    try:
                        # 总市值（亿元）- 索引45
                        if len(parts) > 45 and parts[45]:
                            result['Market_Cap'] = float(parts[45])
                    except (ValueError, TypeError):
                        pass

                    logger.debug(f"获取 {code} 基本面数据成功: PE={result['PE']:.2f}, PB={result['PB']:.2f}")
                    return result

        except Exception as e:
            logger.warning(f"获取 A 股基本面数据失败 {code}: {e}")

        return {}

    def create_stock_type_features(self, code, df):
        """
        创建股票类型特征（重写父类方法）

        使用 A_STOCK_SECTOR_MAPPING 本地配置，替代港股接口。

        Args:
            code: 股票代码
            df: 股票数据 DataFrame（未使用，保留参数兼容性）

        Returns:
            dict: 股票类型特征字典
        """
        stock_info = A_STOCK_SECTOR_MAPPING.get(code, {})

        if stock_info:
            return {
                'stock_type_sector': hash(stock_info.get('sector', 'unknown')) % 100,  # 板块哈希编码
                'defensive_score': stock_info.get('defensive', 50),
                'growth_score': stock_info.get('growth', 50),
                'cyclical_score': stock_info.get('cyclical', 50),
                'liquidity_score': stock_info.get('liquidity', 50),
                'risk_score': stock_info.get('risk', 50),
                'is_core': 1 if stock_info.get('is_core', False) else 0,
            }

        logger.warning(f"未找到股票 {code} 的板块配置")
        return {
            'stock_type_sector': 0,
            'defensive_score': 50,
            'growth_score': 50,
            'cyclical_score': 50,
            'liquidity_score': 50,
            'risk_score': 50,
            'is_core': 0,
        }

    def create_event_driven_features(self, code, df):
        """
        创建事件驱动特征（重写父类方法）

        A股暂不使用港股的财报日期接口，返回默认值。
        后续可接入 A股专用数据源（如东方财富）。

        Args:
            code: 股票代码
            df: 股票数据DataFrame

        Returns:
            df: 添加默认事件驱动特征的DataFrame
        """
        # A股暂不实现财报日期特征，直接返回原始DataFrame
        # 避免调用港股接口（如 yfinance 的 300440.HK）
        logger.debug(f"A股 {code} 跳过事件驱动特征（暂不支持）")
        return df

    def add_a_stock_features(self, df, stock_code):
        """
        添加A股特有特征

        Args:
            df: 股票数据DataFrame
            stock_code: 股票代码

        Returns:
            df: 添加A股特征后的DataFrame
        """
        # 1. 涨跌停特征
        df = self._add_limit_features(df, stock_code)

        # 2. 北向资金特征
        df = self._add_northbound_features(df)

        return df

    def _add_limit_features(self, df, stock_code):
        """添加涨跌停特征"""
        limit_rate = get_limit_rate(stock_code)

        # 计算涨停价和跌停价
        df['High_Limit'] = df['Close'].shift(1) * (1 + limit_rate)
        df['Low_Limit'] = df['Close'].shift(1) * (1 - limit_rate)

        # 涨跌停状态
        df['Limit_Up'] = (df['Close'] >= df['High_Limit'] * 0.995).astype(int)
        df['Limit_Down'] = (df['Close'] <= df['Low_Limit'] * 1.005).astype(int)

        # 连续涨跌停天数
        df['Consecutive_Limit_Up'] = (df['Limit_Up']
            .groupby((df['Limit_Up'] == 0).cumsum())
            .cumsum())
        df['Consecutive_Limit_Down'] = (df['Limit_Down']
            .groupby((df['Limit_Down'] == 0).cumsum())
            .cumsum())

        # 距离涨跌停空间
        df['Space_To_Limit_Up'] = (df['High_Limit'] - df['Close']) / df['Close']
        df['Space_To_Limit_Down'] = (df['Close'] - df['Low_Limit']) / df['Close']

        return df

    def _add_northbound_features(self, df):
        """添加北向资金特征"""
        from data_services.northbound_data import NorthboundDataService
        service = NorthboundDataService()
        northbound_df = service.fetch_history()

        if northbound_df is None or northbound_df.empty:
            df['Northbound_Net_Buy'] = 0
            df['Northbound_Net_Inflow'] = 0
            return df

        northbound_df = northbound_df.copy()
        northbound_df.index = pd.to_datetime(northbound_df.index).tz_localize(None)

        df_temp = df.copy()
        if df_temp.index.tz is not None:
            df_temp.index = df_temp.index.tz_localize(None)

        for col in ['net_buy', 'net_inflow']:
            if col in northbound_df.columns:
                df_temp[f'Northbound_{col.title()}'] = df_temp.index.map(
                    lambda x: northbound_df.loc[:x, col].iloc[-1] if (northbound_df.index <= x).any() else 0
                )

        for col in df_temp.columns:
            if col.startswith('Northbound_'):
                df[col] = df_temp[col]

        return df

    def calculate_relative_strength_a_stock(self, stock_df, csi1000_df, cyb_df):
        """
        计算A股相对强度指标（相对于中证1000和创业板指）

        Args:
            stock_df: 股票数据DataFrame
            csi1000_df: 中证1000指数数据
            cyb_df: 创业板指数据

        Returns:
            DataFrame: 添加相对强度特征后的数据
        """
        if stock_df.empty or csi1000_df is None or cyb_df is None:
            return stock_df

        stock_df = stock_df.copy()

        # 统一时区处理：移除所有时区信息
        if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
            stock_df.index = stock_df.index.tz_localize(None)

        # 确保索引对齐
        csi1000_df = csi1000_df.copy()
        cyb_df = cyb_df.copy()

        # 处理时区
        if hasattr(csi1000_df.index, 'tz') and csi1000_df.index.tz is not None:
            csi1000_df.index = csi1000_df.index.tz_localize(None)
        if hasattr(cyb_df.index, 'tz') and cyb_df.index.tz is not None:
            cyb_df.index = cyb_df.index.tz_localize(None)

        # 计算中证1000收益率
        csi1000_df['CSI1000_Return_1d'] = csi1000_df['Close'].pct_change()
        csi1000_df['CSI1000_Return_5d'] = csi1000_df['Close'].pct_change(5)
        csi1000_df['CSI1000_Return_20d'] = csi1000_df['Close'].pct_change(20)

        # 计算创业板指收益率
        cyb_df['CYB_Return_1d'] = cyb_df['Close'].pct_change()
        cyb_df['CYB_Return_5d'] = cyb_df['Close'].pct_change(5)
        cyb_df['CYB_Return_20d'] = cyb_df['Close'].pct_change(20)

        # 合并到股票数据
        csi1000_cols = ['CSI1000_Return_1d', 'CSI1000_Return_5d', 'CSI1000_Return_20d']
        cyb_cols = ['CYB_Return_1d', 'CYB_Return_5d', 'CYB_Return_20d']

        # 合并中证1000收益
        stock_df = stock_df.merge(
            csi1000_df[csi1000_cols],
            left_index=True, right_index=True, how='left'
        )

        # 合并创业板收益
        stock_df = stock_df.merge(
            cyb_df[cyb_cols],
            left_index=True, right_index=True, how='left'
        )

        # 计算相对强度（个股收益 - 中证1000收益）
        if 'Return_5d' in stock_df.columns and 'CSI1000_Return_5d' in stock_df.columns:
            stock_df['RS_CSI1000_5d'] = stock_df['Return_5d'] - stock_df['CSI1000_Return_5d']
        if 'Return_1d' in stock_df.columns and 'CSI1000_Return_1d' in stock_df.columns:
            stock_df['RS_CSI1000_1d'] = stock_df['Return_1d'] - stock_df['CSI1000_Return_1d']

        return stock_df

    def create_a_stock_market_environment_features(self, stock_df, csi1000_df, cyb_df, us_market_df=None, use_shift=True):
        """
        创建A股市场环境特征

        Args:
            stock_df: 股票数据DataFrame
            csi1000_df: 中证1000指数数据
            cyb_df: 创业板指数据
            us_market_df: 美股市场数据（可选）
            use_shift: 是否使用滞后数据（Walk-forward用True，收市后预测用False）

        Returns:
            DataFrame: 添加市场环境特征后的数据
        """
        if stock_df.empty:
            return stock_df

        stock_df = stock_df.copy()
        shift_val = 1 if use_shift else 0

        # 统一时区处理：移除所有时区信息
        if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
            stock_df.index = stock_df.index.tz_localize(None)

        # 1. A股指数特征（相对强度）
        stock_df = self.calculate_relative_strength_a_stock(stock_df, csi1000_df, cyb_df)

        # 2. 市场情绪互动（创业板量能/中证1000量能）
        if csi1000_df is not None and 'Volume' in csi1000_df.columns:
            csi1000_df = csi1000_df.copy()
            csi1000_df['Volume_5d_mean'] = csi1000_df['Volume'].rolling(5).mean()
            # 统一时区
            if hasattr(csi1000_df.index, 'tz') and csi1000_df.index.tz is not None:
                csi1000_df.index = csi1000_df.index.tz_localize(None)

        if cyb_df is not None and 'Volume' in cyb_df.columns:
            cyb_df = cyb_df.copy()
            cyb_df['Volume_5d_mean'] = cyb_df['Volume'].rolling(5).mean()
            # 统一时区
            if hasattr(cyb_df.index, 'tz') and cyb_df.index.tz is not None:
                cyb_df.index = cyb_df.index.tz_localize(None)

        if csi1000_df is not None and cyb_df is not None:
            # 计算情绪比例
            sentiment_df = pd.DataFrame(index=csi1000_df.index)
            sentiment_df['Sentiment_Ratio'] = cyb_df['Volume_5d_mean'] / (csi1000_df['Volume_5d_mean'] + 1e-10)
            sentiment_df['Sentiment_Ratio_Change_5d'] = sentiment_df['Sentiment_Ratio'].pct_change(5)

            stock_df = stock_df.merge(
                sentiment_df[['Sentiment_Ratio', 'Sentiment_Ratio_Change_5d']],
                left_index=True, right_index=True, how='left'
            )

        # 3. 美股特征（保留，对A股有参考价值）
        if us_market_df is not None and not us_market_df.empty:
            # 统一时区
            us_market_df = us_market_df.copy()
            if hasattr(us_market_df.index, 'tz') and us_market_df.index.tz is not None:
                us_market_df.index = us_market_df.index.tz_localize(None)

            us_features = [
                'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
                'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
                'VIX', 'VIX_Change_5d', 'VIX_Level', 'VIX_Change', 'VIX_Ratio_MA20',
                'US_2Y_Yield', 'US_10Y_Yield', 'US_30Y_Yield',
                'CN_10Y_Yield', 'CN_US_10Y_Spread',
            ]
            existing_us_features = [f for f in us_features if f in us_market_df.columns]
            if existing_us_features:
                # 对美股特征进行 shift，确保不包含未来信息
                us_market_df_shifted = us_market_df[existing_us_features].shift(shift_val)
                stock_df = stock_df.merge(
                    us_market_df_shifted,
                    left_index=True, right_index=True, how='left'
                )

        return stock_df


class AStockTradingModel(CatBoostModel):
    """A股交易模型 - 继承港股CatBoost模型

    关键改进：
    1. 样本权重训练（核心股3.0，扩展股1.0）
    2. 网络特征路径：data/a_stock_network_features/
    3. 市场特征：双指数（中证1000 + 创业板指）
    4. 标签标准化（除以滚动波动率）
    """

    def __init__(self, horizon=20):
        # 应用 A股数据源替换（延迟执行）
        _apply_a_stock_patch()

        # 调用父类初始化
        super().__init__()
        self.horizon = horizon
        self.market = 'a_stock'
        self.feature_engineer = AStockFeatureEngineer()
        self.stock_list = list(A_STOCK_TRAINING_LIST.keys())
        self.stock_names = A_STOCK_TRAINING_LIST
        self.stock_sector_mapping = A_STOCK_SECTOR_MAPPING

        # A股专用配置
        self.network_features_file = os.path.join(A_STOCK_NETWORK_FEATURES_DIR, 'network_features_for_ml.json')
        self.community_ids_file = os.path.join(A_STOCK_NETWORK_FEATURES_DIR, 'community_ids.json')
        self.community_ids = None

        # 加载社区ID列表
        self._load_community_ids()

    def _load_community_ids(self):
        """加载预计算的社区ID列表"""
        if os.path.exists(self.community_ids_file):
            try:
                with open(self.community_ids_file, 'r') as f:
                    self.community_ids = json.load(f)
                logger.info(f"加载社区ID列表: {self.community_ids}")
            except Exception as e:
                logger.warning(f"加载社区ID失败: {e}")
                self.community_ids = None

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=None,
                     for_backtest=False, min_return_threshold=0.0,
                     use_feature_cache=True, community_ids=None, mode='backtest'):
        """
        准备A股训练/预测数据 - A股专用实现

        不调用父类 prepare_data()，而是复制港股流程并替换港股特有部分：
        - 使用中证1000/创业板指代替恒生指数
        - 添加涨跌停、北向资金等A股特有特征
        - 使用A股网络特征路径

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期
            for_backtest: 是否为回测准备数据
            min_return_threshold: 最小收益阈值
            use_feature_cache: 是否使用特征缓存
            community_ids: 预定义的社区ID列表
            mode: 数据模式
                - 'backtest': Walk-forward验证（默认），使用T-1数据
                - 'production': 收市后预测，使用当日数据
        """
        self.horizon = horizon if horizon is not None else self.horizon
        self.min_return_threshold = min_return_threshold

        # 根据 mode 确定 use_shift
        use_shift = (mode == 'backtest')
        all_data = []

        # ========== 1. 获取市场数据 ==========
        print("获取A股市场数据...")

        # 1.1 获取美股市场数据（保留，对A股有参考价值）
        from ml_services.us_market_data import us_market_data
        us_market_df = us_market_data.get_all_us_market_data(period_days=1460)
        if us_market_df is not None:
            print(f"  ✅ 美股市场数据: {len(us_market_df)} 天")
        else:
            print("  ⚠️ 无法获取美股市场数据")

        # 1.2 获取A股指数数据（中证1000 + 创业板指）
        csi1000_df = get_index_data('csi1000', period_days=1460)
        cyb_df = get_index_data('cyb', period_days=1460)

        if csi1000_df is not None:
            print(f"  ✅ 中证1000: {len(csi1000_df)} 天")
        else:
            print("  ⚠️ 无法获取中证1000数据")

        if cyb_df is not None:
            print(f"  ✅ 创业板指: {len(cyb_df)} 天")
        else:
            print("  ⚠️ 无法获取创业板指数据")

        # 1.3 计算A股市场状态特征（HMM + 中证1000）
        a_stock_regime_df = None
        if csi1000_df is not None and not csi1000_df.empty:
            print("  计算A股市场状态特征...")
            a_stock_regime_df = _calculate_a_stock_regime_features(csi1000_df, use_shift=use_shift)
            if a_stock_regime_df is not None:
                print("  ✅ A股市场状态特征计算完成")

        # ========== 2. 加载A股网络特征 ==========
        network_features_data = None
        if os.path.exists(self.network_features_file):
            try:
                with open(self.network_features_file, 'r') as f:
                    network_features_data = json.load(f)
                print(f"  ✅ 网络特征加载完成（{len(network_features_data)} 只股票）")
            except Exception as e:
                print(f"  ⚠️ 网络特征加载失败: {e}")

        # 使用预加载的社区ID
        if community_ids is None:
            community_ids = self.community_ids

        # ========== 3. 每只股票特征计算 ==========
        cache_hits = 0
        cache_misses = 0

        for code in codes:
            try:
                print(f"处理股票: {code}")

                # 获取股票数据（4年约1460天）
                stock_df = get_a_stock_data(code, period_days=1460, use_cache=True)
                if stock_df is None or stock_df.empty:
                    print(f"  ⚠️ 无法获取股票 {code} 数据")
                    continue

                # ========== 3.1 通用特征（复用父类方法）==========
                # 技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df, use_shift=use_shift, code=code)

                # 多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df, use_shift=use_shift)

                # 基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value

                # 板块特征
                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                # 事件驱动特征
                stock_df = self.feature_engineer.create_event_driven_features(code, stock_df)

                # 技术指标与基本面交互特征
                stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

                # ========== 3.2 A股特有特征（关键新增！）==========
                # 涨跌停特征 + 北向资金特征
                stock_df = self.feature_engineer.add_a_stock_features(stock_df, code)
                print(f"  ✅ A股特有特征已添加（涨跌停、北向资金）")

                # ========== 3.3 A股市场环境特征 ==========
                # 中证1000收益 + 创业板指收益 + 美股特征
                stock_df = self.feature_engineer.create_a_stock_market_environment_features(
                    stock_df, csi1000_df, cyb_df, us_market_df, use_shift=use_shift)

                # ========== 3.4 合并A股市场状态特征 ==========
                if a_stock_regime_df is not None:
                    # 处理时区：统一移除时区信息
                    regime_df_temp = a_stock_regime_df.copy()
                    stock_df_temp = stock_df.copy()

                    if hasattr(regime_df_temp.index, 'tz') and regime_df_temp.index.tz is not None:
                        regime_df_temp.index = regime_df_temp.index.tz_localize(None)
                    if hasattr(stock_df_temp.index, 'tz') and stock_df_temp.index.tz is not None:
                        stock_df_temp.index = stock_df_temp.index.tz_localize(None)

                    # Reindex 并 forward-fill
                    regime_aligned = regime_df_temp.reindex(stock_df_temp.index, method='ffill')

                    for col in regime_aligned.columns:
                        stock_df[col] = regime_aligned[col].values

                # ========== 3.5 网络特征（A股路径）==========
                if network_features_data is not None and code in network_features_data:
                    net_features = network_features_data[code]
                    for key, value in net_features.items():
                        stock_df[key] = value
                else:
                    # 为缺失网络特征的股票提供默认值
                    default_net_features = {
                        'net_degree_centrality': 0.0,
                        'net_betweenness_centrality': 0.0,
                        'net_eigenvector_centrality': 0.0,
                        'net_closeness_centrality': 0.0,
                        'net_composite_centrality': 0.0,
                        'net_community_id': -1,
                        'net_community_size': 0,
                        'net_community_centrality_rank': -1,
                        'net_sector_cohesion': 0.0,
                        'net_mst_degree': 0,
                        'net_mst_neighbor_sectors': 0,
                        'net_inter_community_ratio': 0.0,
                        'net_constraint': 1.0,
                        'net_effective_size': 0.0,
                        'net_local_clustering': 0.0,
                    }
                    for key, value in default_net_features.items():
                        stock_df[key] = value
                    logger.debug(f"股票 {code} 使用默认网络特征")

                # ========== 3.6 交叉特征 ==========
                stock_df = self.feature_engineer.create_interaction_features(stock_df)

                # ========== 3.7 市场网络交叉特征 ==========
                stock_df = self.feature_engineer.create_market_network_interaction_features(
                    stock_df, community_ids=community_ids)

                # ========== 3.8 异常检测特征 ==========
                stock_df = self.feature_engineer.create_anomaly_features(stock_df, use_shift=use_shift)

                # ========== 3.9 创建标签 ==========
                stock_df = self.feature_engineer.create_label(
                    stock_df, horizon=self.horizon,
                    for_backtest=for_backtest,
                    min_return_threshold=min_return_threshold)

                # 添加股票代码
                stock_df['Stock_Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                logger.warning(f"处理股票 {code} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_data) == 0:
            raise ValueError("没有可用的数据")

        # ========== 4. 合并数据 ==========
        df = pd.concat(all_data, ignore_index=False)

        # 按日期索引排序（确保时间序列正确，避免股票泄漏）
        df = df.sort_index()

        # 转换索引为 datetime（统一为UTC时区）
        df.index = pd.to_datetime(df.index, utc=True)

        # 过滤日期范围（如果指定）
        if start_date:
            start_date = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_date]

        # ========== 5. 添加样本权重 ==========
        if 'Stock_Code' in df.columns:
            df['sample_weight'] = df['Stock_Code'].apply(get_sample_weight)
            df['is_core'] = df['Stock_Code'].apply(lambda x: 1 if is_core_holding(x) else 0)

        # ========== 6. 标签标准化（除以滚动波动率）==========
        # 解决主板10%和创业板20%涨跌停的非对称问题
        if 'Label' in df.columns and 'volatility_20d' in df.columns:
            mask = df['Label'].notna() & (df['volatility_20d'] > 0)
            df.loc[mask, 'Label_Normalized'] = df.loc[mask, 'Label'] / df.loc[mask, 'volatility_20d']

        logger.info(f"A股数据准备完成，共 {len(df)} 条记录")

        return df

    def get_feature_columns(self, df, dedup_threshold=None):
        """
        获取特征列 - A股版本

        排除：
        - 基础列（Open, High, Low, Close等）
        - 绝对价格特征（跨股票不可比）
        - A股市场级特征（所有股票同值，需与网络特征交叉后使用）

        Args:
            df: 数据DataFrame
            dedup_threshold: Pearson相关性去冗余阈值（默认None不启用）

        Returns:
            list: 特征列名列表
        """
        # 基础排除列
        base_exclude = ['Code', 'Stock_Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                       'Vol_MA20', '+DM', '-DM', '+DI', '-DI']

        # 已删除的冗余特征
        deprecated_features = ['Returns', 'Volatility', 'MA5_Deviation', 'MA10_Deviation',
                              'BB_Breakout', 'High_Position_20d', 'MA_Bullish_Resonance',
                              'Momentum_5d', 'Momentum_10d', 'Momentum_120d', 'Momentum_250d',
                              'Consecutive_Ranging_Days', 'Confidence_Threshold_Multiplier',
                              'Price_Return_Std_30d']

        exclude_columns = base_exclude + ABSOLUTE_PRICE_FEATURES + deprecated_features

        # 排除A股市场级特征（所有股票同值，无法区分个股）
        # 这些特征已通过与网络社区特征交叉保留信息
        market_exclude = set(A_STOCK_MARKET_LEVEL_FEATURES)

        feature_columns = [col for col in df.columns
                          if col not in exclude_columns
                          and col not in market_exclude
                          and not col.startswith('Label')]

        # 可选：Pearson去冗余
        if dedup_threshold and len(feature_columns) > 0:
            numeric_cols = [c for c in feature_columns if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            if len(numeric_cols) > 1:
                sample_df = df[numeric_cols].tail(500)
                corr_matrix = sample_df.corr()
                to_remove = set()
                for i in range(len(numeric_cols)):
                    if numeric_cols[i] in to_remove:
                        continue
                    for j in range(i+1, len(numeric_cols)):
                        if numeric_cols[j] in to_remove:
                            continue
                        if abs(corr_matrix.iloc[i, j]) > dedup_threshold:
                            to_remove.add(numeric_cols[j])
                feature_columns = [c for c in feature_columns if c not in to_remove]
                if to_remove:
                    logger.info(f"Pearson去冗余（阈值={dedup_threshold}）：删除 {len(to_remove)} 个高相关特征")

        return feature_columns

    def train_with_weights(self, X, y, sample_weights=None, cat_features=None, horizon=20):
        """
        使用样本权重训练模型（带时间序列交叉验证）

        注意：特征必须已经编码为数值型（由 train() 方法处理）

        Args:
            X: 特征矩阵（已编码为数值）
            y: 标签
            sample_weights: 样本权重（核心股3.0，扩展股1.0）
            cat_features: 分类特征索引（已编码为数值，应为None）
            horizon: 预测周期
        """
        from catboost import CatBoostClassifier, Pool
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, f1_score

        # 如果没有提供样本权重，使用默认权重
        if sample_weights is None:
            sample_weights = np.ones(len(y))

        # 验证特征是否为数值型
        if X.dtype == object:
            logger.error("特征矩阵包含字符串，请确保分类特征已编码")
            # 尝试转换
            try:
                X = X.astype(float)
                logger.warning("已自动转换为数值型")
            except Exception as e:
                logger.error(f"无法转换特征: {e}")
                raise ValueError("特征必须为数值型，请检查分类特征编码")

        # 创建 CatBoost 模型（不使用 auto_class_weights，避免与样本权重冲突）
        catboost_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Accuracy',
            'depth': 8,
            'learning_rate': 0.06,
            'n_estimators': 400,
            'l2_leaf_reg': 2,
            'subsample': 0.75,
            'colsample_bylevel': 0.8,
            'random_seed': 2020,
            'verbose': False,
        }

        self.catboost_model = CatBoostClassifier(**catboost_params)

        # 使用时间序列交叉验证
        # 数据已按日期排序，训练集是早期所有股票，验证集是后期所有股票
        # 这样验证集的日期完全是新的，避免市场特征泄漏
        tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
        cv_scores = []
        cv_f1_scores = []

        logger.info("开始时间序列交叉验证...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            weights_train_fold = sample_weights[train_idx]

            # 创建新的模型实例（避免累积训练）
            fold_model = CatBoostClassifier(**catboost_params)

            train_pool = Pool(
                data=X_train_fold,
                label=y_train_fold,
                weight=weights_train_fold,
                cat_features=None
            )
            val_pool = Pool(data=X_val_fold, label=y_val_fold)

            # 不使用 eval_set 进行早停（避免过拟合验证集）
            fold_model.fit(train_pool, verbose=False)
            y_pred_fold = fold_model.predict(X_val_fold)

            score = accuracy_score(y_val_fold, y_pred_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
            cv_scores.append(score)
            cv_f1_scores.append(f1)
            logger.info(f"   Fold {fold} 验证准确率: {score:.4f}, F1分数: {f1:.4f}")

        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        mean_f1 = np.mean(cv_f1_scores)
        std_f1 = np.std(cv_f1_scores)

        logger.info(f"\n✅ 交叉验证完成")
        logger.info(f"   平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        logger.info(f"   平均验证F1分数: {mean_f1:.4f} (+/- {std_f1:.4f})")

        # 使用全部数据重新训练（带样本权重）
        full_pool = Pool(
            data=X,
            label=y,
            weight=sample_weights,
            cat_features=None
        )
        self.catboost_model.fit(full_pool, verbose=100)

        # 更新实际树数量
        self.actual_n_estimators = self.catboost_model.tree_count_

        logger.info(f"\n✅ 模型训练完成（样本加权：核心股3.0，扩展股1.0）")
        logger.info(f"   实际训练树数量: {self.actual_n_estimators}")

        # 保存准确率到文件（供综合分析使用）
        self._save_accuracy(mean_accuracy, std_accuracy, mean_f1, std_f1, horizon)

        return {
            'accuracy': mean_accuracy,
            'std': std_accuracy,
            'f1': mean_f1,
            'f1_std': std_f1
        }

    def _save_accuracy(self, accuracy, std, f1, f1_std, horizon):
        """保存模型准确率到文件"""
        import json
        accuracy_info = {
            'model_type': 'a_stock_catboost',
            'horizon': horizon,
            'accuracy': float(accuracy),
            'std': float(std),
            'f1_score': float(f1),
            'f1_std': float(f1_std),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        accuracy_file = 'data/model_accuracy.json'
        try:
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}

            key = f'a_stock_catboost_{horizon}d'
            existing_data[key] = accuracy_info

            with open(accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(f"准确率已保存到 {accuracy_file}")
        except Exception as e:
            logger.warning(f"保存准确率失败: {e}")

    def train(self, codes=None, start_date=None, end_date=None, horizon=None, use_feature_selection=False, min_return_threshold=0.0, use_sample_weights=True):
        """
        训练A股模型

        Args:
            codes: 股票代码列表（可选，默认使用配置）
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期
            use_feature_selection: 是否使用特征选择
            min_return_threshold: 最小收益阈值
            use_sample_weights: 是否使用样本权重训练
        """
        if codes is None:
            codes = self.stock_list
        if horizon is None:
            horizon = self.horizon

        logger.info("=" * 60)
        logger.info(f"开始训练A股模型（预测周期: {horizon}天）")
        logger.info(f"股票数量: {len(codes)}")
        logger.info(f"样本权重: {'启用' if use_sample_weights else '禁用'}")
        logger.info("=" * 60)

        # 准备数据
        df = self.prepare_data(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            use_feature_cache=True,
            mode='backtest'
        )

        if df is None or df.empty:
            logger.error("数据准备失败")
            return None

        # 准备特征和标签
        feature_cols = [c for c in df.columns if c not in
                        ['Label', 'Label_Normalized', 'Stock_Code', 'Date', 'sample_weight', 'is_core']]

        # 处理分类特征（与父类 CatBoostModel 一致）
        from sklearn.preprocessing import LabelEncoder
        categorical_features = []
        self.categorical_encoders = {}

        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                logger.info(f"  检测到分类特征: {col}")
                df[col] = df[col].fillna('unknown').astype(str)
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.categorical_encoders[col] = encoder
                categorical_features.append(col)

        if categorical_features:
            logger.info(f"已编码 {len(categorical_features)} 个分类特征")

        X = df[feature_cols].values
        y = df['Label'].values
        sample_weights = df['sample_weight'].values if 'sample_weight' in df.columns else None

        # 确保特征矩阵为数值型（转换 object 类型为 float）
        if X.dtype == object:
            logger.warning("特征矩阵包含 object 类型，转换为 float")
            X = X.astype(float)

        # 保存特征列名（用于预测时对齐）
        self.feature_columns = feature_cols

        # 使用样本权重训练
        if use_sample_weights and sample_weights is not None:
            self.train_with_weights(X, y, sample_weights, horizon=horizon)
        else:
            # 调用父类训练方法
            super().train(
                codes=codes,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                use_feature_selection=use_feature_selection,
                min_return_threshold=min_return_threshold
            )

        # 返回特征重要性
        return self.get_feature_importance()

    def get_feature_importance(self, top_n=50):
        """
        获取特征重要性

        Args:
            top_n: 返回前 N 个重要特征

        Returns:
            DataFrame: 特征重要性排序
        """
        if self.catboost_model is None:
            logger.warning("模型未训练，无法获取特征重要性")
            return None

        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            logger.warning("特征列未设置")
            return None

        # 获取特征重要性
        import pandas as pd
        feat_imp = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.catboost_model.feature_importances_
        })
        feat_imp = feat_imp.sort_values('Importance', ascending=False)

        logger.info(f"特征重要性计算完成（共 {len(feat_imp)} 个特征）")
        if top_n:
            return feat_imp.head(top_n)
        return feat_imp

    def predict(self, code=None, predict_date=None, horizon=None, use_feature_cache=True, mode='production'):
        """
        生成A股预测

        Args:
            code: 股票代码（可选，默认预测所有自选股）
            predict_date: 预测日期
            horizon: 预测周期
            use_feature_cache: 是否使用特征缓存
            mode: 预测模式
        """
        if horizon is None:
            horizon = self.horizon

        logger.info("=" * 60)
        logger.info(f"开始生成A股预测（预测周期: {horizon}天）")
        logger.info("=" * 60)

        # 如果没有指定股票代码，预测所有自选股
        if code is None:
            results = []
            for stock_code in self.stock_list:
                try:
                    result = super().predict(
                        code=stock_code,
                        predict_date=predict_date,
                        horizon=horizon,
                        use_feature_cache=use_feature_cache,
                        mode=mode
                    )
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"预测 {stock_code} 失败: {e}")
            return results
        else:
            return super().predict(
                code=code,
                predict_date=predict_date,
                horizon=horizon,
                use_feature_cache=use_feature_cache,
                mode=mode
            )

    def get_stock_data(self, stock_code, period_days=500):
        """获取A股股票数据"""
        return get_a_stock_data(stock_code, period_days=period_days, use_cache=True)

    def get_index_data_for_market(self, period_days=500):
        """获取A股指数数据"""
        return get_index_data('sh', period_days=period_days)


# ========== 命令行接口 ==========

# A股模型保存路径
A_STOCK_MODEL_DIR = 'data/a_stock_models'

def main():
    parser = argparse.ArgumentParser(description='A股机器学习交易模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                       help='运行模式: train=训练, predict=预测')
    parser.add_argument('--horizon', type=int, default=20, choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月（默认）')
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择')
    parser.add_argument('--start-date', type=str, default=None,
                       help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--predict-date', type=str, default=None,
                       help='预测日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', type=str, default=None,
                       help='股票代码列表，逗号分隔')

    args = parser.parse_args()

    # 解析股票列表
    codes = None
    if args.stocks:
        codes = args.stocks.split(',')

    # 初始化模型
    model = AStockTradingModel(horizon=args.horizon)

    # 模型保存路径
    os.makedirs(A_STOCK_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(A_STOCK_MODEL_DIR, f'trading_model_catboost_{args.horizon}d.pkl')

    if args.mode == 'train':
        # 训练模型
        feature_importance = model.train(
            codes=codes,
            start_date=args.start_date,
            end_date=args.end_date,
            use_feature_selection=args.use_feature_selection
        )
        # 保存模型
        model.save_model(model_path)
        logger.info(f"A股模型已保存到 {model_path}")

        # 保存特征重要性
        if feature_importance is not None:
            importance_path = model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"特征重要性已保存到 {importance_path}")

    elif args.mode == 'predict':
        # 加载模型
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            logger.error(f"请先运行训练模式: python3 a_stock_ml_model.py --mode train --horizon {args.horizon}")
            return
        model.load_model(model_path)
        logger.info(f"A股模型已从 {model_path} 加载")

        # 生成预测
        predictions = model.predict(
            predict_date=args.predict_date,
            use_feature_cache=True,
            mode='production'
        )

        # 保存预测结果
        if predictions:
            from datetime import datetime, timedelta
            from a_stock_config import A_STOCK_WATCHLIST

            # 计算 target_date
            def get_target_date(start_date, horizon):
                """计算目标日期（简化版，实际交易日计算更复杂）"""
                target = start_date + timedelta(days=horizon)
                return target.strftime('%Y-%m-%d')

            # 构建预测 DataFrame
            pred_data = []
            for pred in predictions:
                if pred:
                    data_date = pred['date'].strftime('%Y-%m-%d') if hasattr(pred['date'], 'strftime') else str(pred['date'])
                    target_date = get_target_date(pred['date'], args.horizon) if hasattr(pred['date'], 'timedelta') else data_date

                    pred_data.append({
                        'Stock_Code': pred['code'],
                        'Stock_Name': A_STOCK_WATCHLIST.get(pred['code'], pred['code']),
                        'Prediction': pred['prediction'],
                        'Prediction_Proba': pred['probability'],
                        'Current_Price': pred['current_price'],
                        'Data_Date': data_date,
                        'Target_Date': target_date
                    })

            if pred_data:
                import pandas as pd
                pred_df = pd.DataFrame(pred_data)
                pred_file = os.path.join(A_STOCK_MODEL_DIR, f'ml_predictions_{args.horizon}d.csv')
                pred_df.to_csv(pred_file, index=False)
                logger.info(f"A股预测结果已保存到 {pred_file}")

                # 打印预测结果摘要
                print("\n" + "=" * 60)
                print(f"📊 A股预测结果（{args.horizon}天周期）")
                print("=" * 60)
                for _, row in pred_df.iterrows():
                    pred_label = '上涨' if row['Prediction'] == 1 else '下跌'
                    confidence = row['Prediction_Proba'] if row['Prediction'] == 1 else 1 - row['Prediction_Proba']
                    print(f"  {row['Stock_Name']:<10} {pred_label} (置信度: {confidence:.1%}, 概率: {row['Prediction_Proba']:.4f})")
                print("=" * 60)


if __name__ == '__main__':
    main()
