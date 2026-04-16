#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股通南向资金数据服务

提供南向资金历史数据的获取、缓存和查询功能

数据来源：Akshare - 东方财富网
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 缓存配置
CACHE_DIR = 'data/southbound_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'southbound_history.pkl')
CACHE_EXPIRE_HOURS = 6  # 缓存过期时间（小时）


class SouthboundDataService:
    """港股通南向资金数据服务"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._cache = None

    def fetch_history(self, use_cache=True):
        """
        获取南向资金历史数据

        参数:
        - use_cache: 是否使用缓存

        返回:
        - DataFrame: 南向资金历史数据
        """
        # 检查缓存
        if use_cache:
            cached_data = self._load_cache()
            if cached_data is not None:
                return cached_data

        print("📊 正在获取南向资金历史数据...")

        try:
            import akshare as ak

            # 获取南向资金历史数据
            df = ak.stock_hsgt_hist_em(symbol='南向资金')

            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '当日成交净买额': 'net_buy',
                '买入成交额': 'buy_amount',
                '卖出成交额': 'sell_amount',
                '历史累计净买额': 'cumulative_net_buy',
                '当日资金流入': 'net_inflow',
                '当日余额': 'daily_balance',
                '持股市值': 'holding_value'
            })

            # 转换日期
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # 保存缓存
            self._save_cache(df)

            print(f"  ✅ 获取完成（{len(df)} 条记录）")
            print(f"  日期范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            print(f"  ❌ 获取失败: {e}")
            return None

    def _load_cache(self):
        """加载缓存数据"""
        if not os.path.exists(CACHE_FILE):
            return None

        try:
            # 检查缓存是否过期
            cache_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
            if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRE_HOURS):
                return None

            with open(CACHE_FILE, 'rb') as f:
                df = pickle.load(f)

            print(f"  ✅ 从缓存加载南向资金数据（{len(df)} 条）")
            return df

        except Exception as e:
            print(f"  ⚠️ 缓存加载失败: {e}")
            return None

    def _save_cache(self, df):
        """保存缓存数据"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(df, f)
            print(f"  💾 缓存已保存到: {CACHE_FILE}")
        except Exception as e:
            print(f"  ⚠️ 缓存保存失败: {e}")

    def get_features_for_date(self, date, df=None):
        """
        获取指定日期的南向资金特征

        参数:
        - date: 日期（str 或 datetime）
        - df: 南向资金数据（可选，默认从缓存获取）

        返回:
        - dict: 南向资金特征
        """
        if df is None:
            df = self.fetch_history()

        if df is None or df.empty:
            return {
                'Southbound_Net_Buy': 0,
                'Southbound_Net_Inflow': 0,
                'Southbound_Buy_Amount': 0,
                'Southbound_Sell_Amount': 0
            }

        # 转换日期格式
        if isinstance(date, str):
            date = pd.to_datetime(date)

        # 查找最接近的交易日
        try:
            if date in df.index:
                row = df.loc[date]
            else:
                # 找到最近的交易日
                mask = df.index <= date
                if mask.any():
                    row = df[mask].iloc[-1]
                else:
                    return {
                        'Southbound_Net_Buy': 0,
                        'Southbound_Net_Inflow': 0,
                        'Southbound_Buy_Amount': 0,
                        'Southbound_Sell_Amount': 0
                    }

            return {
                'Southbound_Net_Buy': float(row.get('net_buy', 0)) if pd.notna(row.get('net_buy', 0)) else 0,
                'Southbound_Net_Inflow': float(row.get('net_inflow', 0)) if pd.notna(row.get('net_inflow', 0)) else 0,
                'Southbound_Buy_Amount': float(row.get('buy_amount', 0)) if pd.notna(row.get('buy_amount', 0)) else 0,
                'Southbound_Sell_Amount': float(row.get('sell_amount', 0)) if pd.notna(row.get('sell_amount', 0)) else 0
            }

        except Exception as e:
            print(f"  ⚠️ 获取南向资金特征失败: {e}")
            return {
                'Southbound_Net_Buy': 0,
                'Southbound_Net_Inflow': 0,
                'Southbound_Buy_Amount': 0,
                'Southbound_Sell_Amount': 0
            }

    def get_latest(self):
        """
        获取最新的南向资金数据

        返回:
        - dict: 最新南向资金数据
        """
        df = self.fetch_history()
        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        return {
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'net_buy': float(latest.get('net_buy', 0)) if pd.notna(latest.get('net_buy', 0)) else 0,
            'net_inflow': float(latest.get('net_inflow', 0)) if pd.notna(latest.get('net_inflow', 0)) else 0,
            'buy_amount': float(latest.get('buy_amount', 0)) if pd.notna(latest.get('buy_amount', 0)) else 0,
            'sell_amount': float(latest.get('sell_amount', 0)) if pd.notna(latest.get('sell_amount', 0)) else 0
        }


# 便捷函数
def get_southbound_history():
    """获取南向资金历史数据"""
    service = SouthboundDataService()
    return service.fetch_history()


def get_southbound_features(date):
    """获取指定日期的南向资金特征"""
    service = SouthboundDataService()
    return service.get_features_for_date(date)


if __name__ == '__main__':
    # 测试
    service = SouthboundDataService()

    print("=" * 60)
    print("南向资金数据测试")
    print("=" * 60)

    # 获取历史数据
    df = service.fetch_history()
    print(f"\n数据概览:")
    print(df.describe())

    # 获取最新数据
    latest = service.get_latest()
    print(f"\n最新数据:")
    print(f"  日期: {latest['date']}")
    print(f"  净买入: {latest['net_buy']:.2f} 亿")
    print(f"  净流入: {latest['net_inflow']:.2f} 亿")
