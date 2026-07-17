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
from ml_services.ml_trading_model import CatBoostModel, FeatureEngineer, ABSOLUTE_PRICE_FEATURES, logger

# 导入A股配置和数据服务
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_TRAINING_LIST,
    A_STOCK_SECTOR_MAPPING,
    get_limit_rate,
    get_market_code,
    A_STOCK_INDEX,
)
from data_services.a_stock_data import get_a_stock_data, get_index_data

# 标记是否已应用 monkey-patch
_patched = False


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


class AStockTradingModel(CatBoostModel):
    """A股交易模型 - 继承港股CatBoost模型"""

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

    def train(self, codes=None, start_date=None, end_date=None, horizon=None, use_feature_selection=False, min_return_threshold=0.0):
        """
        训练A股模型

        Args:
            codes: 股票代码列表（可选，默认使用配置）
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期
            use_feature_selection: 是否使用特征选择
            min_return_threshold: 最小收益阈值
        """
        if codes is None:
            codes = self.stock_list
        if horizon is None:
            horizon = self.horizon

        logger.info("=" * 60)
        logger.info(f"开始训练A股模型（预测周期: {horizon}天）")
        logger.info(f"股票数量: {len(codes)}")
        logger.info("=" * 60)

        # 调用父类训练方法
        return super().train(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            use_feature_selection=use_feature_selection,
            min_return_threshold=min_return_threshold
        )

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

    if args.mode == 'train':
        model.train(
            codes=codes,
            start_date=args.start_date,
            end_date=args.end_date,
            use_feature_selection=args.use_feature_selection
        )
    elif args.mode == 'predict':
        model.predict(
            predict_date=args.predict_date,
            use_feature_cache=True,
            mode='production'
        )


if __name__ == '__main__':
    main()
