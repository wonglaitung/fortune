#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主力资金流向数据服务

提供主力资金历史数据的获取、缓存和查询功能

数据来源：东方财富网 API
- 主力净流入 = 超大单 + 大单净流入
- 反映市场大资金流向，替代原北向资金指标

创建时间：2026-07-20
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 缓存配置
CACHE_DIR = 'data/main_fund_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'main_fund_history.pkl')
CACHE_EXPIRE_HOURS = 6  # 缓存过期时间（小时）

# 日志配置
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainFundFlowService:
    """主力资金流向数据服务"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._cache = None

    def fetch_history(self, use_cache=True, days=365):
        """
        获取主力资金历史数据

        参数:
        - use_cache: 是否使用缓存
        - days: 获取数据的天数

        返回:
        - DataFrame: 主力资金历史数据
        """
        # 检查缓存
        if use_cache:
            cached_data = self._load_cache()
            if cached_data is not None:
                return cached_data

        logger.info("正在获取主力资金流向数据...")

        try:
            # 东方财富主力资金流向接口
            url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            params = {
                "lmt": days,
                "klt": 101,  # 日线
                "secid": "1.000001",  # 上证指数
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56",
            }

            r = requests.get(url, params=params, timeout=15)
            data = r.json()

            if data.get("rc") != 0 or not data.get("data"):
                logger.error("获取主力资金数据失败")
                return None

            klines = data["data"].get("klines", [])
            if not klines:
                logger.error("主力资金数据为空")
                return None

            # 解析数据
            # 格式: 日期,主力净流入,超大单,大单,中单,小单,主力净占比,超大单占比,大单占比,中单占比,小单占比,收盘价,涨跌幅,flag
            parsed_data = []
            for kline in klines:
                parts = kline.split(',')
                if len(parts) >= 6:
                    parsed_data.append({
                        'date': pd.to_datetime(parts[0]),
                        'main_net_flow': float(parts[1]) / 1e8 if parts[1] else 0,  # 转换为亿元
                        'super_large': float(parts[2]) / 1e8 if parts[2] else 0,
                        'large': float(parts[3]) / 1e8 if parts[3] else 0,
                        'mid': float(parts[4]) / 1e8 if parts[4] else 0,
                        'small': float(parts[5]) / 1e8 if parts[5] else 0,
                    })

            if not parsed_data:
                logger.error("解析主力资金数据失败")
                return None

            df = pd.DataFrame(parsed_data)
            df = df.set_index('date')

            # 保存缓存
            self._save_cache(df)

            logger.info(f"获取完成（{len(df)} 条记录）")
            logger.info(f"日期范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            logger.error(f"获取主力资金数据失败: {e}")
            return None

    def fetch_latest(self):
        """
        获取当日主力资金数据

        返回:
        - dict: 当日主力资金数据
        """
        df = self.fetch_history(days=1)
        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        return {
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'main_net_flow': float(latest['main_net_flow']),
            'super_large': float(latest['super_large']),
            'large': float(latest['large']),
            'mid': float(latest['mid']),
            'small': float(latest['small']),
        }

    def get_latest(self):
        """
        获取最新的主力资金数据（兼容旧接口）

        返回:
        - dict: 最新主力资金数据
        """
        return self.fetch_latest()

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

            logger.info(f"从缓存加载主力资金数据（{len(df)} 条）")
            return df

        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None

    def _save_cache(self, df):
        """保存缓存数据"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"缓存已保存到: {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def get_features_for_date(self, date, df=None):
        """
        获取指定日期的主力资金特征

        参数:
        - date: 日期（str 或 datetime）
        - df: 主力资金数据（可选，默认从缓存获取）

        返回:
        - dict: 主力资金特征
        """
        if df is None:
            df = self.fetch_history()

        if df is None or df.empty:
            return {
                'MainFund_Net_Flow': 0,
                'MainFund_Super_Large': 0,
                'MainFund_Large': 0,
                'MainFund_Mid': 0,
                'MainFund_Small': 0,
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
                        'MainFund_Net_Flow': 0,
                        'MainFund_Super_Large': 0,
                        'MainFund_Large': 0,
                        'MainFund_Mid': 0,
                        'MainFund_Small': 0,
                    }

            return {
                'MainFund_Net_Flow': float(row.get('main_net_flow', 0)) if pd.notna(row.get('main_net_flow', 0)) else 0,
                'MainFund_Super_Large': float(row.get('super_large', 0)) if pd.notna(row.get('super_large', 0)) else 0,
                'MainFund_Large': float(row.get('large', 0)) if pd.notna(row.get('large', 0)) else 0,
                'MainFund_Mid': float(row.get('mid', 0)) if pd.notna(row.get('mid', 0)) else 0,
                'MainFund_Small': float(row.get('small', 0)) if pd.notna(row.get('small', 0)) else 0,
            }

        except Exception as e:
            logger.warning(f"获取主力资金特征失败: {e}")
            return {
                'MainFund_Net_Flow': 0,
                'MainFund_Super_Large': 0,
                'MainFund_Large': 0,
                'MainFund_Mid': 0,
                'MainFund_Small': 0,
            }


# 便捷函数
def get_main_fund_history():
    """获取主力资金历史数据"""
    service = MainFundFlowService()
    return service.fetch_history()


def get_main_fund_features(date):
    """获取指定日期的主力资金特征"""
    service = MainFundFlowService()
    return service.get_features_for_date(date)


if __name__ == '__main__':
    # 测试
    service = MainFundFlowService()

    print("=" * 60)
    print("主力资金流向数据测试")
    print("=" * 60)

    # 测试历史数据
    print("\n历史数据:")
    df = service.fetch_history()
    if df is not None:
        print(f"数据量: {len(df)} 条")
        print(f"最新数据:")
        print(df.tail(5).to_string())

    # 测试最新数据
    print("\n最新数据:")
    latest = service.get_latest()
    if latest:
        print(f"  日期: {latest.get('date', 'N/A')}")
        print(f"  主力净流入: {latest.get('main_net_flow', 0):.2f} 亿")
        print(f"  超大单: {latest.get('super_large', 0):.2f} 亿")
        print(f"  大单: {latest.get('large', 0):.2f} 亿")