#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股市场数据获取模块
提供美股指数、VIX恐慌指数、美国国债收益率等数据
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class USMarketData:
    """美股市场数据获取类"""

    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=1)  # 缓存1小时

    def get_sp500_data(self, period_days=730):
        """获取标普500指数数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含标普500指数数据
        """
        cache_key = f'sp500_{period_days}'
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data

        try:
            # 使用 yfinance 获取标普500指数数据 (^GSPC)
            ticker = yf.Ticker('^GSPC')
            df = ticker.history(period=f'{period_days}d')

            if df.empty:
                print("⚠️ 无法获取标普500指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率
            df['SP500_Return'] = df['Close'].pct_change()
            df['SP500_Return_5d'] = df['Close'].pct_change(5)
            df['SP500_Return_20d'] = df['Close'].pct_change(20)

            # 缓存数据
            self.cache[cache_key] = (df, datetime.now())

            return df

        except Exception as e:
            print(f"⚠️ 获取标普500指数数据失败: {e}")
            return None

    def get_nasdaq_data(self, period_days=730):
        """获取纳斯达克指数数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含纳斯达克指数数据
        """
        cache_key = f'nasdaq_{period_days}'
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data

        try:
            # 使用 yfinance 获取纳斯达克指数数据 (^IXIC)
            ticker = yf.Ticker('^IXIC')
            df = ticker.history(period=f'{period_days}d')

            if df.empty:
                print("⚠️ 无法获取纳斯达克指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率
            df['NASDAQ_Return'] = df['Close'].pct_change()
            df['NASDAQ_Return_5d'] = df['Close'].pct_change(5)
            df['NASDAQ_Return_20d'] = df['Close'].pct_change(20)

            # 缓存数据
            self.cache[cache_key] = (df, datetime.now())

            return df

        except Exception as e:
            print(f"⚠️ 获取纳斯达克指数数据失败: {e}")
            return None

    def get_vix_data(self, period_days=730):
        """获取VIX恐慌指数数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含VIX恐慌指数数据
        """
        cache_key = f'vix_{period_days}'
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data

        try:
            # 使用 yfinance 获取VIX恐慌指数数据 (^VIX)
            ticker = yf.Ticker('^VIX')
            df = ticker.history(period=f'{period_days}d')

            if df.empty:
                print("⚠️ 无法获取VIX恐慌指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算VIX变化
            df['VIX_Change'] = df['Close'].pct_change()
            df['VIX_MA5'] = df['Close'].rolling(window=5).mean()
            df['VIX_MA20'] = df['Close'].rolling(window=20).mean()

            # VIX相对位置（相对于20日均值）
            df['VIX_Ratio_MA20'] = df['Close'] / df['VIX_MA20']

            # 缓存数据
            self.cache[cache_key] = (df, datetime.now())

            return df

        except Exception as e:
            print(f"⚠️ 获取VIX恐慌指数数据失败: {e}")
            return None

    def get_us_treasury_yield(self, period_days=730):
        """获取美国10年期国债收益率数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含美国10年期国债收益率数据
        """
        cache_key = f'treasury_{period_days}'
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data

        try:
            # 使用 yfinance 获取美国10年期国债收益率数据 (^TNX)
            # 注意：^TNX 的数据是百分比形式，需要除以100
            ticker = yf.Ticker('^TNX')
            df = ticker.history(period=f'{period_days}d')

            if df.empty:
                print("⚠️ 无法获取美国10年期国债收益率数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 转换为小数形式（百分比/100）
            df['US_10Y_Yield'] = df['Close'] / 100

            # 计算收益率变化
            df['US_10Y_Yield_Change'] = df['US_10Y_Yield'].pct_change()
            df['US_10Y_Yield_MA5'] = df['US_10Y_Yield'].rolling(window=5).mean()
            df['US_10Y_Yield_MA20'] = df['US_10Y_Yield'].rolling(window=20).mean()

            # 缓存数据
            self.cache[cache_key] = (df, datetime.now())

            return df

        except Exception as e:
            print(f"⚠️ 获取美国10年期国债收益率数据失败: {e}")
            return None

    def get_all_us_market_data(self, period_days=730):
        """获取所有美股市场数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 合并后的美股市场数据
        """
        # 获取各项数据
        sp500_df = self.get_sp500_data(period_days)
        nasdaq_df = self.get_nasdaq_data(period_days)
        vix_df = self.get_vix_data(period_days)
        treasury_df = self.get_us_treasury_yield(period_days)

        # 合并数据
        if sp500_df is not None:
            merged_df = sp500_df[['SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d']].copy()
        else:
            return None

        if nasdaq_df is not None:
            merged_df = merged_df.merge(
                nasdaq_df[['NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d']],
                left_index=True, right_index=True, how='left'
            )

        if vix_df is not None:
            merged_df = merged_df.merge(
                vix_df[['VIX_Change', 'VIX_Ratio_MA20', 'Close']],
                left_index=True, right_index=True, how='left'
            )
            # 重命名VIX绝对值
            merged_df.rename(columns={'Close': 'VIX_Level'}, inplace=True)

        if treasury_df is not None:
            merged_df = merged_df.merge(
                treasury_df[['US_10Y_Yield', 'US_10Y_Yield_Change']],
                left_index=True, right_index=True, how='left'
            )

        return merged_df

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        print("✅ 美股数据缓存已清除")


# 全局实例
us_market_data = USMarketData()


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("测试美股市场数据获取")
    print("=" * 70)

    # 获取所有美股市场数据
    print("\n📊 获取美股市场数据...")
    us_df = us_market_data.get_all_us_market_data(period_days=30)

    if us_df is not None:
        print(f"\n✅ 成功获取 {len(us_df)} 天的美股市场数据")
        print("\n📊 数据预览:")
        print(us_df.tail(10))

        print("\n📊 数据统计:")
        print(us_df.describe())

        print("\n📊 缺失值统计:")
        print(us_df.isnull().sum())
    else:
        print("\n❌ 获取美股市场数据失败")
