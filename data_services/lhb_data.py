#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙虎榜数据服务

提供龙虎榜历史数据的获取、缓存和查询功能

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
CACHE_DIR = 'data/lhb_cache'
CACHE_EXPIRE_HOURS = 6


class LHBDataService:
    """龙虎榜数据服务"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get_lhb_data(self, start_date, end_date=None):
        """
        获取龙虎榜数据

        Args:
            start_date (str): 开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
            end_date (str): 结束日期，默认与开始日期相同

        Returns:
            DataFrame: 龙虎榜数据
        """
        if end_date is None:
            end_date = start_date

        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')

        cache_file = os.path.join(CACHE_DIR, f'lhb_{start_str}_{end_str}.pkl')

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
            df = ak.stock_lhb_detail_em(start_date=start_str, end_date=end_str)

            if df is not None and not df.empty:
                df.to_pickle(cache_file)
            return df

        except Exception as e:
            print(f"  ⚠️ 获取龙虎榜数据失败: {e}")
            return None

    def get_stock_lhb_data(self, stock_code, date):
        """
        获取个股龙虎榜数据

        Args:
            stock_code (str): 股票代码
            date (str): 日期

        Returns:
            dict: 龙虎榜特征
        """
        df = self.get_lhb_data(date)

        if df is None or df.empty:
            return {
                'LHB_Buy_Amount': 0,
                'LHB_Sell_Amount': 0,
                'LHB_Net_Buy': 0,
                'LHB_On_List': 0,
            }

        # 查找该股票
        try:
            # 尝试不同的代码列名
            code_col = None
            for col in ['代码', '股票代码', 'symbol', 'code']:
                if col in df.columns:
                    code_col = col
                    break

            if code_col is None:
                return {
                    'LHB_Buy_Amount': 0,
                    'LHB_Sell_Amount': 0,
                    'LHB_Net_Buy': 0,
                    'LHB_On_List': 0,
                }

            row = df[df[code_col].astype(str) == str(stock_code)]

            if row.empty:
                return {
                    'LHB_Buy_Amount': 0,
                    'LHB_Sell_Amount': 0,
                    'LHB_Net_Buy': 0,
                    'LHB_On_List': 0,
                }

            row = row.iloc[0]

            # 尝试不同的列名
            buy_col = '买入金额' if '买入金额' in df.columns else '买方合计'
            sell_col = '卖出金额' if '卖出金额' in df.columns else '卖方合计'

            buy_amount = float(row.get(buy_col, 0) or 0)
            sell_amount = float(row.get(sell_col, 0) or 0)

            return {
                'LHB_Buy_Amount': buy_amount,
                'LHB_Sell_Amount': sell_amount,
                'LHB_Net_Buy': buy_amount - sell_amount,
                'LHB_On_List': 1,  # 上榜
            }

        except Exception as e:
            print(f"  ⚠️ 获取个股龙虎榜数据失败: {e}")
            return {
                'LHB_Buy_Amount': 0,
                'LHB_Sell_Amount': 0,
                'LHB_Net_Buy': 0,
                'LHB_On_List': 0,
            }


def get_lhb_features(stock_code, date):
    """获取指定股票的龙虎榜特征"""
    service = LHBDataService()
    return service.get_stock_lhb_data(stock_code, date)


if __name__ == '__main__':
    print("=" * 60)
    print("龙虎榜数据测试")
    print("=" * 60)

    service = LHBDataService()

    # 测试获取最近7天的龙虎榜数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"\n获取龙虎榜数据 ({start_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')}):")
    df = service.get_lhb_data(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))

    if df is not None:
        print(f"  获取成功: {len(df)} 条记录")
        print(f"  列名: {list(df.columns)[:8]}...")

        # 统计上榜股票数量
        code_col = None
        for col in ['代码', '股票代码', 'symbol', 'code']:
            if col in df.columns:
                code_col = col
                break
        if code_col:
            print(f"  上榜股票数: {df[code_col].nunique()}")
    else:
        print("  获取失败或无数据")

    # 测试个股数据
    print(f"\n测试个股龙虎榜:")
    test_date = (end_date - timedelta(days=1)).strftime('%Y%m%d')
    for code in ['600800', '300440']:
        features = get_lhb_features(code, test_date)
        print(f"  {code}: {features}")
