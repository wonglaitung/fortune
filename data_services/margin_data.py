#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融资融券数据服务

提供融资融券历史数据的获取、缓存和查询功能

数据来源：AKShare - 东方财富网
"""

import os
import sys
import pickle
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 缓存配置
CACHE_DIR = 'data/margin_cache'
CACHE_EXPIRE_HOURS = 6


class MarginDataService:
    """融资融券数据服务"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get_margin_data_sse(self, date):
        """
        获取沪市融资融券数据

        Args:
            date (str): 日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            DataFrame: 融资融券数据
        """
        # 标准化日期格式
        date_str = date.replace('-', '')

        cache_file = os.path.join(CACHE_DIR, f'sse_{date_str}.pkl')

        # 检查缓存
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRE_HOURS):
                try:
                    df = pd.read_pickle(cache_file)
                    return df
                except Exception:
                    pass

        try:
            import akshare as ak
            df = ak.stock_margin_detail_sse(date=date_str)

            if df is not None and not df.empty:
                df.to_pickle(cache_file)
            return df

        except Exception as e:
            print(f"  ⚠️ 获取沪市融资融券数据失败: {e}")
            return None

    def get_margin_data_szse(self, date):
        """
        获取深市融资融券数据

        Args:
            date (str): 日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            DataFrame: 融资融券数据
        """
        date_str = date.replace('-', '')

        cache_file = os.path.join(CACHE_DIR, f'szse_{date_str}.pkl')

        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRE_HOURS):
                try:
                    df = pd.read_pickle(cache_file)
                    return df
                except Exception:
                    pass

        try:
            import akshare as ak
            df = ak.stock_margin_detail_szse(date=date_str)

            if df is not None and not df.empty:
                df.to_pickle(cache_file)
            return df

        except Exception as e:
            print(f"  ⚠️ 获取深市融资融券数据失败: {e}")
            return None

    def get_stock_margin_data(self, stock_code, date):
        """
        获取个股融资融券数据

        Args:
            stock_code (str): 股票代码
            date (str): 日期

        Returns:
            dict: 融资融券特征
        """
        # 根据股票代码判断市场
        if stock_code.startswith('6'):
            df = self.get_margin_data_sse(date)
            # 沪市列名：标的证券代码（不是标的代码）
            code_col = '标的证券代码'
        else:
            df = self.get_margin_data_szse(date)
            # 深市列名：证券代码
            code_col = '证券代码'

        if df is None or df.empty:
            return {
                'Margin_Buy_Amount': 0,
                'Margin_Balance': 0,
                'Short_Sell_Volume': 0,
                'Short_Balance': 0,
            }

        # 查找该股票
        try:
            # 检查列名是否存在
            if code_col not in df.columns:
                # 尝试其他可能的列名
                possible_cols = ['标的代码', '标的证券代码', '证券代码', '股票代码']
                for col in possible_cols:
                    if col in df.columns:
                        code_col = col
                        break
                else:
                    return {
                        'Margin_Buy_Amount': 0,
                        'Margin_Balance': 0,
                        'Short_Sell_Volume': 0,
                        'Short_Balance': 0,
                    }

            row = df[df[code_col] == stock_code]
            if row.empty:
                return {
                    'Margin_Buy_Amount': 0,
                    'Margin_Balance': 0,
                    'Short_Sell_Volume': 0,
                    'Short_Balance': 0,
                }

            row = row.iloc[0]

            return {
                'Margin_Buy_Amount': float(row.get('融资买入额', 0) or 0),
                'Margin_Balance': float(row.get('融资余额', 0) or 0),
                'Short_Sell_Volume': float(row.get('融券卖出量', 0) or 0),
                'Short_Balance': float(row.get('融券余量', 0) or 0),
            }

        except Exception as e:
            # 静默处理，不打印警告（融资融券数据是增强特征，缺失不影响核心功能）
            return {
                'Margin_Buy_Amount': 0,
                'Margin_Balance': 0,
                'Short_Sell_Volume': 0,
                'Short_Balance': 0,
            }


def get_margin_features(stock_code, date):
    """获取指定股票的融资融券特征"""
    service = MarginDataService()
    return service.get_stock_margin_data(stock_code, date)


if __name__ == '__main__':
    print("=" * 60)
    print("融资融券数据测试")
    print("=" * 60)

    service = MarginDataService()

    # 测试获取最新数据
    today = datetime.now().strftime('%Y%m%d')

    print(f"\n测试沪市数据 ({today}):")
    df_sh = service.get_margin_data_sse(today)
    if df_sh is not None:
        print(f"  获取成功: {len(df_sh)} 条记录")
        print(f"  列名: {list(df_sh.columns)[:5]}...")
    else:
        print("  获取失败或无数据")

    print(f"\n测试深市数据 ({today}):")
    df_sz = service.get_margin_data_szse(today)
    if df_sz is not None:
        print(f"  获取成功: {len(df_sz)} 条记录")
        print(f"  列名: {list(df_sz.columns)[:5]}...")
    else:
        print("  获取失败或无数据")

    # 测试个股数据
    print(f"\n测试个股融资融券:")
    for code in ['600800', '300440']:
        features = get_margin_features(code, today)
        print(f"  {code}: {features}")
