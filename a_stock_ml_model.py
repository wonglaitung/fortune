#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股机器学习交易模型

继承港股模型结构，适配A股特有特征：
- 涨跌停限制
- 北向资金
- 融资融券
- 龙虎榜
"""

import os
import sys
import warnings
import argparse
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入港股模型
from ml_services.ml_trading_model import (
    BaseTradingModel,
    CatBoostModel,
    FeatureEngineer,
    ABSOLUTE_PRICE_FEATURES,
    logger,
)

# 导入A股配置和数据服务
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_TRAINING_LIST,
    A_STOCK_SECTOR_MAPPING,
    get_limit_rate,
    get_market_code,
    A_STOCK_INDEX,
    A_STOCK_CACHE_DIR,
    A_STOCK_FEATURE_CACHE_DIR,
)
from data_services.a_stock_data import get_a_stock_data, get_index_data, get_a_stock_info_tencent
from data_services.northbound_data import get_northbound_features
from data_services.technical_analysis import TechnicalAnalyzer


class AStockFeatureEngineer(FeatureEngineer):
    """A股特征工程类 - 继承港股特征工程，添加A股特有特征"""

    def __init__(self):
        super().__init__()
        self.market = 'a_stock'

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
        """
        添加涨跌停特征

        Args:
            df: 股票数据DataFrame
            stock_code: 股票代码

        Returns:
            df: 添加涨跌停特征后的DataFrame
        """
        limit_rate = get_limit_rate(stock_code)

        # 计算涨停价和跌停价（使用前一日收盘价）
        df['High_Limit'] = df['Close'].shift(1) * (1 + limit_rate)
        df['Low_Limit'] = df['Close'].shift(1) * (1 - limit_rate)

        # 涨跌停状态（允许0.5%误差，因为实际价格可能四舍五入）
        df['Limit_Up'] = (df['Close'] >= df['High_Limit'] * 0.995).astype(int)
        df['Limit_Down'] = (df['Close'] <= df['Low_Limit'] * 1.005).astype(int)

        # 连续涨停天数
        df['Consecutive_Limit_Up'] = (df['Limit_Up']
            .groupby((df['Limit_Up'] == 0).cumsum())
            .cumsum())

        # 连续跌停天数
        df['Consecutive_Limit_Down'] = (df['Limit_Down']
            .groupby((df['Limit_Down'] == 0).cumsum())
            .cumsum())

        # 距离涨停的空间
        df['Space_To_Limit_Up'] = (df['High_Limit'] - df['Close']) / df['Close']

        # 距离跌停的空间
        df['Space_To_Limit_Down'] = (df['Close'] - df['Low_Limit']) / df['Close']

        return df

    def _add_northbound_features(self, df):
        """
        添加北向资金特征

        Args:
            df: 股票数据DataFrame

        Returns:
            df: 添加北向资金特征后的DataFrame
        """
        # 获取北向资金历史数据
        from data_services.northbound_data import NorthboundDataService
        service = NorthboundDataService()
        northbound_df = service.fetch_history()

        if northbound_df is None or northbound_df.empty:
            # 添加默认值
            df['Northbound_Net_Buy'] = 0
            df['Northbound_Net_Inflow'] = 0
            df['Northbound_SH_Net_Buy'] = 0
            df['Northbound_SZ_Net_Buy'] = 0
            return df

        # 合并北向资金数据
        northbound_df = northbound_df.copy()
        northbound_df.index = pd.to_datetime(northbound_df.index).tz_localize(None)

        # 重置索引以便合并
        df_temp = df.copy()
        if df_temp.index.tz is not None:
            df_temp.index = df_temp.index.tz_localize(None)

        # 对齐日期
        for col in ['net_buy', 'net_inflow', 'sh_net_buy', 'sz_net_buy']:
            if col in northbound_df.columns:
                df_temp[f'Northbound_{col.title()}'] = df_temp.index.map(
                    lambda x: northbound_df.loc[:x, col].iloc[-1] if (northbound_df.index <= x).any() else 0
                )

        # 复制回原DataFrame
        for col in df_temp.columns:
            if col.startswith('Northbound_'):
                df[col] = df_temp[col]

        return df


class AStockTradingModel(CatBoostModel):
    """A股交易模型 - 继承港股CatBoost模型"""

    def __init__(self, horizon=20):
        # 调用父类初始化
        super().__init__()
        self.horizon = horizon
        self.market = 'a_stock'
        self.feature_engineer = AStockFeatureEngineer()
        self.stock_list = list(A_STOCK_TRAINING_LIST.keys())
        self.stock_names = A_STOCK_TRAINING_LIST
        self.stock_sector_mapping = A_STOCK_SECTOR_MAPPING

    def get_stock_data(self, stock_code, period_days=500):
        """
        获取A股股票数据

        Args:
            stock_code: 股票代码
            period_days: 数据天数

        Returns:
            DataFrame: 股票数据
        """
        return get_a_stock_data(stock_code, period_days=period_days, use_cache=True)

    def get_index_data(self, period_days=500):
        """
        获取A股指数数据

        Args:
            period_days: 数据天数

        Returns:
            DataFrame: 指数数据
        """
        return get_index_data('sh', period_days=period_days)

    def prepare_features(self, df, stock_code, mode='production'):
        """
        准备特征（覆盖父类方法，添加A股特有特征）

        Args:
            df: 股票数据DataFrame
            stock_code: 股票代码
            mode: 'production' 或 'backtest'

        Returns:
            DataFrame: 特征DataFrame
        """
        # 调用父类方法准备基础特征
        df = super().prepare_features(df, stock_code, mode=mode)

        # 添加A股特有特征
        df = self.feature_engineer.add_a_stock_features(df, stock_code)

        return df

    def _build_market_features(self, df, hsi_df=None):
        """
        构建市场级特征（覆盖父类方法，使用上证指数）

        Args:
            df: 股票数据DataFrame
            hsi_df: 指数数据（可选）

        Returns:
            DataFrame: 添加市场特征后的DataFrame
        """
        # 获取上证指数数据
        if hsi_df is None:
            hsi_df = self.get_index_data()

        if hsi_df is None or hsi_df.empty:
            logger.warning(f"无法获取上证指数数据，跳过市场特征")
            return df

        # 确保索引是datetime
        if not isinstance(hsi_df.index, pd.DatetimeIndex):
            hsi_df.index = pd.to_datetime(hsi_df.index)

        # 移除时区信息
        if hsi_df.index.tz is not None:
            hsi_df.index = hsi_df.index.tz_localize(None)
        if df.index.tz is not None:
            df_temp = df.copy()
            df_temp.index = df_temp.index.tz_localize(None)
        else:
            df_temp = df

        # 计算上证指数收益率
        hsi_df['SH_Return_1d'] = hsi_df['Close'].pct_change()
        hsi_df['SH_Return_3d'] = hsi_df['Close'].pct_change(3)
        hsi_df['SH_Return_5d'] = hsi_df['Close'].pct_change(5)
        hsi_df['SH_Return_10d'] = hsi_df['Close'].pct_change(10)
        hsi_df['SH_Return_20d'] = hsi_df['Close'].pct_change(20)
        hsi_df['SH_Return_60d'] = hsi_df['Close'].pct_change(60)

        # 上证指数波动率
        hsi_df['SH_Volatility_20d'] = hsi_df['SH_Return_1d'].rolling(20).std()

        # 上证指数均线
        hsi_df['SH_MA5'] = hsi_df['Close'].rolling(5).mean()
        hsi_df['SH_MA20'] = hsi_df['Close'].rolling(20).mean()
        hsi_df['SH_MA60'] = hsi_df['Close'].rolling(60).mean()

        # 上证指数相对位置
        hsi_df['SH_Ratio_MA5'] = hsi_df['Close'] / hsi_df['SH_MA5'] - 1
        hsi_df['SH_Ratio_MA20'] = hsi_df['Close'] / hsi_df['SH_MA20'] - 1
        hsi_df['SH_Ratio_MA60'] = hsi_df['Close'] / hsi_df['SH_MA60'] - 1

        # 合并到股票数据
        market_cols = [
            'SH_Return_1d', 'SH_Return_3d', 'SH_Return_5d',
            'SH_Return_10d', 'SH_Return_20d', 'SH_Return_60d',
            'SH_Volatility_20d',
            'SH_Ratio_MA5', 'SH_Ratio_MA20', 'SH_Ratio_MA60',
        ]

        for col in market_cols:
            if col in hsi_df.columns:
                df_temp[col] = df_temp.index.map(
                    lambda x: hsi_df.loc[:x, col].iloc[-1] if (hsi_df.index <= x).any() else np.nan
                )

        # 复制回原DataFrame
        for col in market_cols:
            if col in df_temp.columns:
                df[col] = df_temp[col]

        return df

    def train(self, start_date=None, end_date=None, use_feature_selection=False):
        """
        训练A股模型

        Args:
            start_date: 训练开始日期
            end_date: 训练结束日期
            use_feature_selection: 是否使用特征选择
        """
        logger.info("=" * 60)
        logger.info(f"开始训练A股模型（预测周期: {self.horizon}天）")
        logger.info("=" * 60)

        # 调用父类训练方法
        return super().train(
            start_date=start_date,
            end_date=end_date,
            use_feature_selection=use_feature_selection
        )

    def predict(self, predict_date=None, use_feature_selection=False, mode='production'):
        """
        生成A股预测

        Args:
            predict_date: 预测日期
            use_feature_selection: 是否使用特征选择
            mode: 预测模式
        """
        logger.info("=" * 60)
        logger.info(f"开始生成A股预测（预测周期: {self.horizon}天）")
        logger.info("=" * 60)

        # 调用父类预测方法
        return super().predict(
            predict_date=predict_date,
            use_feature_selection=use_feature_selection,
            mode=mode
        )


# ========== 命令行接口 ==========

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

    args = parser.parse_args()

    # 初始化模型
    model = AStockTradingModel(horizon=args.horizon)

    if args.mode == 'train':
        model.train(
            start_date=args.start_date,
            end_date=args.end_date,
            use_feature_selection=args.use_feature_selection
        )
    elif args.mode == 'predict':
        model.predict(
            predict_date=args.predict_date,
            use_feature_selection=args.use_feature_selection,
            mode='production'
        )


if __name__ == '__main__':
    main()
