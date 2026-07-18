#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股市场级特征模块

计算双指数架构的市场级特征：
- 中证1000指数：中小盘Beta信号
- 创业板指：成长股风险偏好

特征包括：
- 强弱对比因子（Relative Strength）
- 市场情绪互动（Sentiment Ratio）
- 宏观利率对冲因子

创建时间：2026-07-17
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import (
    A_STOCK_MARKET_INDEX,
    A_STOCK_INDEX_TENCENT,
    A_STOCK_CACHE_DIR,
)

logger = logging.getLogger(__name__)

# 创建缓存目录
os.makedirs(A_STOCK_CACHE_DIR, exist_ok=True)


def get_index_data(stock_code, period_days=250):
    """
    获取指数数据（复用 a_stock_data.py）

    Args:
        stock_code (str): 指数代码
        period_days (int): 获取天数

    Returns:
        pandas.DataFrame: 指数数据
    """
    from data_services.a_stock_data import get_index_data as _get_index_data

    # 映射指数类型
    index_type_map = {
        '000852': 'csi1000',
        '399006': 'cyb',
        '000001': 'sh',
        '399001': 'sz',
    }

    index_type = index_type_map.get(stock_code, 'sh')
    return _get_index_data(index_type, period_days)


class AStockMarketFeatures:
    """A股市场级特征计算器"""

    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 3600  # 缓存1小时

    def load_index_data(self, period_days=250):
        """加载双指数数据"""
        csi1000_code = A_STOCK_MARKET_INDEX['csi1000']
        cyb_code = A_STOCK_MARKET_INDEX['cyb']

        # 获取中证1000数据
        csi1000_df = get_index_data(csi1000_code, period_days)
        # 获取创业板指数据
        cyb_df = get_index_data(cyb_code, period_days)

        return csi1000_df, cyb_df

    def calculate_market_features(self, df, stock_code, csi1000_df=None, cyb_df=None):
        """
        计算市场级特征

        Args:
            df: 股票数据 DataFrame
            stock_code: 股票代码
            csi1000_df: 中证1000数据（可选，不传则自动获取）
            cyb_df: 创业板指数据（可选，不传则自动获取）

        Returns:
            DataFrame: 添加市场特征后的数据
        """
        if csi1000_df is None or cyb_df is None:
            csi1000_df, cyb_df = self.load_index_data()

        # 确保索引对齐
        df = df.copy()

        # 计算CSI1000收益率
        if csi1000_df is not None and 'Close' in csi1000_df.columns:
            csi1000_df['Return'] = csi1000_df['Close'].pct_change()
            csi1000_df['Return_5d'] = csi1000_df['Return'].rolling(5).sum()
            csi1000_df['Return_20d'] = csi1000_df['Return'].rolling(20).sum()
            csi1000_df['Volume_5d_mean'] = csi1000_df['Volume'].rolling(5).mean()

            # 合并到股票数据
            for col in ['Return', 'Return_5d', 'Return_20d', 'Volume', 'Volume_5d_mean']:
                if col in csi1000_df.columns:
                    df[f'CSI1000_{col}'] = csi1000_df[col]

        # 计算创业板指收益率
        if cyb_df is not None and 'Close' in cyb_df.columns:
            cyb_df['Return'] = cyb_df['Close'].pct_change()
            cyb_df['Return_5d'] = cyb_df['Return'].rolling(5).sum()
            cyb_df['Return_20d'] = cyb_df['Return'].rolling(20).sum()
            cyb_df['Volume_5d_mean'] = cyb_df['Volume'].rolling(5).mean()

            # 合并到股票数据
            for col in ['Return', 'Return_5d', 'Return_20d', 'Volume', 'Volume_5d_mean']:
                if col in cyb_df.columns:
                    df[f'CYB_{col}'] = cyb_df[col]

        # ========== 特征1：强弱对比因子（Relative Strength）==========
        # 个股收益率 - 中证1000收益率
        if 'Return' in df.columns and 'CSI1000_Return' in df.columns:
            df['RS_CSI1000_1d'] = df['Return'] - df['CSI1000_Return']
        if 'Return_5d' in df.columns and 'CSI1000_Return_5d' in df.columns:
            df['RS_CSI1000_5d'] = df['Return_5d'] - df['CSI1000_Return_5d']

        # ========== 特征2：市场情绪互动（Sentiment Ratio）==========
        # 创业板5日均量 / 中证1000 5日均量
        if 'CYB_Volume_5d_mean' in df.columns and 'CSI1000_Volume_5d_mean' in df.columns:
            df['Sentiment_Ratio'] = df['CYB_Volume_5d_mean'] / (df['CSI1000_Volume_5d_mean'] + 1e-10)
            # 情绪比例变动
            df['Sentiment_Ratio_Change_5d'] = df['Sentiment_Ratio'].pct_change(5)

        return df

    def calculate_cross_market_features(self, df, use_shift=True):
        """
        计算跨市场联动特征（业界推荐）

        包括：
        - 商品期货联动（铜、原油）
        - 人民币汇率
        - AH溢价代理（使用港股相关ETF价格）

        Args:
            df: 股票数据 DataFrame
            use_shift: 是否使用滞后数据（避免数据泄漏）

        Returns:
            DataFrame: 添加跨市场特征后的数据
        """
        df = df.copy()
        shift_val = 1 if use_shift else 0

        try:
            # ========== 商品期货特征 ==========
            copper_df = self._get_commodity_futures_data('CU0')  # 铜主力
            if copper_df is not None and not copper_df.empty:
                copper_df['Copper_Return_5d'] = copper_df['收盘价'].pct_change(5).shift(shift_val)
                copper_df['Copper_Return_20d'] = copper_df['收盘价'].pct_change(20).shift(shift_val)

                # 合并到股票数据
                df = self._merge_cross_market_data(df, copper_df, ['Copper_Return_5d', 'Copper_Return_20d'])

            oil_df = self._get_commodity_futures_data('SC0')  # 原油主力
            if oil_df is not None and not oil_df.empty:
                oil_df['Oil_Return_5d'] = oil_df['收盘价'].pct_change(5).shift(shift_val)
                oil_df['Oil_Return_20d'] = oil_df['收盘价'].pct_change(20).shift(shift_val)

                df = self._merge_cross_market_data(df, oil_df, ['Oil_Return_5d', 'Oil_Return_20d'])

        except Exception as e:
            logger.warning(f"商品期货数据获取失败: {e}")

        try:
            # ========== 人民币汇率特征 ==========
            cny_df = self._get_cny_usd_rate()
            if cny_df is not None and not cny_df.empty:
                cny_df['CNY_USD_Change_5d'] = cny_df['rate'].pct_change(5).shift(shift_val)
                cny_df['CNY_USD_Z_Score'] = (
                    (cny_df['rate'] - cny_df['rate'].rolling(20).mean()) /
                    (cny_df['rate'].rolling(20).std() + 1e-10)
                ).shift(shift_val)

                df = self._merge_cross_market_data(df, cny_df, ['CNY_USD_Change_5d', 'CNY_USD_Z_Score'])

        except Exception as e:
            logger.warning(f"汇率数据获取失败: {e}")

        # 填充缺失值为0（表示无跨市场信号）
        cross_market_cols = [
            'Copper_Return_5d', 'Copper_Return_20d',
            'Oil_Return_5d', 'Oil_Return_20d',
            'CNY_USD_Change_5d', 'CNY_USD_Z_Score'
        ]
        for col in cross_market_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        logger.debug(f"跨市场联动特征计算完成")
        return df

    def _get_commodity_futures_data(self, symbol='CU0'):
        """获取商品期货数据（带缓存）"""
        import akshare as ak

        cache_key = f'futures_{symbol}'
        if cache_key in self._cache:
            cache_time, cache_data = self._cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=self._cache_ttl):
                return cache_data

        try:
            df = ak.futures_main_sina(symbol=symbol)
            if df is not None and not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
                df = df.rename(columns={'收盘价': '收盘价'})

                # 缓存数据
                self._cache[cache_key] = (datetime.now(), df)
                return df
        except Exception as e:
            logger.warning(f"获取商品期货 {symbol} 数据失败: {e}")

        return None

    def _get_cny_usd_rate(self):
        """获取人民币汇率数据（带缓存）"""
        import akshare as ak

        cache_key = 'cny_usd_rate'
        if cache_key in self._cache:
            cache_time, cache_data = self._cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=self._cache_ttl):
                return cache_data

        try:
            # 方法1: 尝试外汇即时行情
            df = ak.fx_spot_quote()
            if df is not None and not df.empty:
                # 检查可用列名（API格式已变更）
                price_cols = ['最新价', '买报价', '卖报价', '中间价']
                price_col = None
                for col in price_cols:
                    if col in df.columns:
                        # 检查是否有有效数据（非NaN）
                        if not df[col].isna().all():
                            price_col = col
                            break

                if price_col:
                    # 找到美元兑人民币
                    usd_cny = df[df['货币对'].str.contains('USD/CNY', case=False, na=False)]
                    if not usd_cny.empty and not pd.isna(usd_cny[price_col].iloc[0]):
                        result = pd.DataFrame({
                            'date': [datetime.now()],
                            'rate': [float(usd_cny[price_col].iloc[0])]
                        })
                        result = result.set_index('date')
                        self._cache[cache_key] = (datetime.now(), result)
                        return result

            # 方法2: 使用 yfinance 获取 USDCNY=X
            try:
                import yfinance as yf
                ticker = yf.Ticker("USDCNY=X")
                data = ticker.history(period="1d")
                if not data.empty:
                    result = pd.DataFrame({
                        'date': [datetime.now()],
                        'rate': [float(data['Close'].iloc[-1])]
                    })
                    result = result.set_index('date')
                    self._cache[cache_key] = (datetime.now(), result)
                    return result
            except Exception:
                pass

        except Exception as e:
            # 静默处理，汇率数据是辅助特征
            pass

        return None

    def _merge_cross_market_data(self, stock_df, market_df, columns):
        """合并跨市场数据到股票数据"""
        if market_df is None or market_df.empty:
            return stock_df

        # 统一时区
        if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
            stock_df.index = stock_df.index.tz_localize(None)

        market_df = market_df.copy()
        if hasattr(market_df.index, 'tz') and market_df.index.tz is not None:
            market_df.index = market_df.index.tz_localize(None)

        # 合并
        for col in columns:
            if col in market_df.columns:
                stock_df[col] = market_df[col].reindex(stock_df.index, method='ffill')

        return stock_df

    def calculate_interest_rate_features(self, df, us_yield_data=None):
        """
        计算利率对冲特征

        Args:
            df: 股票数据
            us_yield_data: 美债收益率数据（可选）

        Returns:
            DataFrame: 添加利率特征后的数据
        """
        df = df.copy()

        # 如果有美债收益率数据
        if us_yield_data is not None and 'US_10Y_Yield' in us_yield_data.columns:
            # 合并美债收益率
            df['US_10Y_Yield'] = us_yield_data['US_10Y_Yield']
            df['US_10Y_Yield_Change'] = us_yield_data['US_10Y_Yield'].pct_change()

            # ========== 特征3：宏观利率对冲 ==========
            # 结合个股growth评分（需要从sector_mapping获取）
            if 'CSI1000_Return_5d' in df.columns and 'US_10Y_Yield_Change' in df.columns:
                df['CSI1000_US10Y_Spread'] = df['CSI1000_Return_5d'] * df['US_10Y_Yield_Change']

        return df

    def get_feature_names(self):
        """返回市场特征名称列表"""
        return [
            # 中证1000特征
            'CSI1000_Return', 'CSI1000_Return_5d', 'CSI1000_Return_20d',
            'CSI1000_Volume', 'CSI1000_Volume_5d_mean',
            # 创业板指特征
            'CYB_Return', 'CYB_Return_5d', 'CYB_Return_20d',
            'CYB_Volume', 'CYB_Volume_5d_mean',
            # 强弱对比因子
            'RS_CSI1000_1d', 'RS_CSI1000_5d',
            # 市场情绪互动
            'Sentiment_Ratio', 'Sentiment_Ratio_Change_5d',
            # 利率对冲因子
            'CSI1000_US10Y_Spread',
        ]


def add_market_features_to_df(df, stock_code, csi1000_df=None, cyb_df=None):
    """
    为股票数据添加市场特征的便捷函数

    Args:
        df: 股票数据 DataFrame
        stock_code: 股票代码
        csi1000_df: 中证1000数据（可选）
        cyb_df: 创业板指数据（可选）

    Returns:
        DataFrame: 添加市场特征后的数据
    """
    calculator = AStockMarketFeatures()
    return calculator.calculate_market_features(df, stock_code, csi1000_df, cyb_df)


# 市场级特征单调性定义（用于交叉特征）
MARKET_FEATURE_MONOTONICITY = {
    'CSI1000_Return_1d': 'NEUTRAL',      # 市场涨跌对个股影响复杂
    'CSI1000_Return_5d': 'POSITIVE',     # 中期市场上涨有利
    'CSI1000_Return_20d': 'POSITIVE',    # 长期市场上涨有利
    'CYB_Return_5d': 'POSITIVE',         # 创业板上涨利好成长股
    'Sentiment_Ratio': 'NEUTRAL',        # 情绪比例方向不确定
    'Sentiment_Ratio_Change_5d': 'POSITIVE',  # 情绪升温有利
}


if __name__ == '__main__':
    # 测试市场特征计算
    from data_services.a_stock_data import get_a_stock_data

    print("测试A股市场特征模块...")

    # 获取测试股票数据
    test_code = '300440'
    df = get_a_stock_data(test_code, period_days=250)

    if df is not None:
        # 计算市场特征
        calculator = AStockMarketFeatures()
        df = calculator.calculate_market_features(df, test_code)

        # 显示结果
        feature_names = calculator.get_feature_names()
        existing_features = [f for f in feature_names if f in df.columns]

        print(f"\n成功计算 {len(existing_features)} 个市场特征:")
        for feat in existing_features:
            if feat in df.columns:
                latest = df[feat].dropna().iloc[-1] if len(df[feat].dropna()) > 0 else 'N/A'
                print(f"  - {feat}: {latest}")
    else:
        print(f"无法获取股票 {test_code} 的数据")