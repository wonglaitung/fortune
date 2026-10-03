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
import threading
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 缓存配置
CACHE_DIR = 'data/margin_cache'
CACHE_EXPIRE_HOURS = 6

from data_services.a_data_freeze import is_frozen, load_frozen, save_frozen

# 网络硬超时（秒）：akshare/SSE/深交所接口偶发无超时挂起
# （2026-09-28 实测 query.sse.com.cn 阻塞 16 分钟，拖死整个 walk-forward）
NET_TIMEOUT_SEC = 20
# 电路闸：某市场接口超时一次后，本进程内不再请求该网络源（直接用缓存/默认值）
_net_disabled = {}


def _fetch_with_timeout(fn, key, *args, **kwargs):
    """守护线程调用网络接口，超时/禁用时返回 (ok=False, None)。

    超时线程保留在后台（daemon，不阻塞进程退出），结果若晚到会写入缓存供下次使用。
    """
    if _net_disabled.get(key):
        return False, None
    box = {}

    def _run():
        try:
            box['ret'] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 —— 原样抛给调用方处理
            box['err'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(NET_TIMEOUT_SEC)
    if t.is_alive():
        _net_disabled[key] = True
        print(f"  ⚠️ 融资融券接口[{key}]超时({NET_TIMEOUT_SEC}s)，"
              f"本次进程内跳过该网络源（用缓存/默认值）")
        return False, None
    if 'err' in box:
        raise box['err']
    return True, box.get('ret')


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

        # 冻结模式（2026-10-03）：跳过 6h TTL，保证 walk-forward 可复现。
        # 此前 6h TTL + _net_disabled 进程级熔断 → 一次超时后本进程全部归 0。
        if is_frozen():
            found, fd = load_frozen(f'margin_sse_{date_str}')
            if found:
                return fd
        elif os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRE_HOURS):
                try:
                    df = pd.read_pickle(cache_file)
                    return df
                except Exception:
                    pass

        try:
            import akshare as ak
            ok, df = _fetch_with_timeout(ak.stock_margin_detail_sse, 'sse', date=date_str)
            if not ok:
                return None

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

        # 冻结模式（2026-10-03）：同 SSE，跳过 6h TTL
        if is_frozen():
            found, fd = load_frozen(f'margin_szse_{date_str}')
            if found:
                return fd
        elif os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRE_HOURS):
                try:
                    df = pd.read_pickle(cache_file)
                    return df
                except Exception:
                    pass

        try:
            import akshare as ak
            ok, df = _fetch_with_timeout(ak.stock_margin_detail_szse, 'szse', date=date_str)
            if not ok:
                return None

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
