#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据获取模块

通过腾讯财经API和AKShare获取A股数据
- 个股历史数据
- 指数数据
- 实时报价
"""

import os
import sys
import requests
import pandas as pd

from data_services.a_data_freeze import is_frozen, load_frozen, save_frozen, save_failure
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import (
    get_market_code,
    A_STOCK_INDEX_TENCENT,
    A_STOCK_CACHE_DIR,
    A_STOCK_CACHE_DAYS,
)

# 创建缓存目录
os.makedirs(A_STOCK_CACHE_DIR, exist_ok=True)


def get_a_stock_data_tencent(stock_code, period_days=90):
    """
    通过腾讯财经接口获取A股股票数据

    Args:
        stock_code (str): 股票代码，例如 "300440"
        period_days (int): 获取数据的天数，默认90天

    Returns:
        pandas.DataFrame: 包含股票数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    # 获取市场代码
    market = get_market_code(stock_code)

    # 腾讯财经API URL（前复权数据）
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{stock_code},day,,,{period_days},qfq"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://stockapp.finance.qq.com/',
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # 解析返回的JSON数据
        data = response.json()

        # 检查数据是否有效
        if 'data' not in data:
            print(f"无法获取股票 {stock_code} 的数据")
            return None

        # 提取K线数据
        # 数据结构: data -> {market}{code} -> qfqday 或 day
        stock_key = f'{market}{stock_code}'
        if stock_key not in data['data']:
            print(f"无法获取股票 {stock_code} 的数据")
            return None

        stock_data = data['data'][stock_key]
        kline_data = None

        # 优先使用前复权数据
        if 'qfqday' in stock_data:
            kline_data = stock_data['qfqday']
        elif 'day' in stock_data:
            kline_data = stock_data['day']

        if kline_data is None or len(kline_data) == 0:
            print(f"无法获取股票 {stock_code} 的K线数据")
            return None

        # 解析数据
        # 数据格式: [日期, 开盘价, 收盘价, 最高价, 最低价, 成交量, ...]
        parsed_data = []
        for item in kline_data:
            if len(item) >= 6:
                parsed_data.append({
                    'Date': pd.to_datetime(item[0], utc=True),
                    'Open': float(item[1]),
                    'Close': float(item[2]),
                    'High': float(item[3]),
                    'Low': float(item[4]),
                    'Volume': int(float(item[5])),
                })

        # 创建DataFrame
        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            return df
        else:
            print(f"股票 {stock_code} 数据为空")
            return None

    except Exception as e:
        print(f"获取股票 {stock_code} 数据失败: {e}")
        return None


def get_a_stock_data_akshare(stock_code, period_days=90):
    """
    通过AKShare获取A股股票数据（备选数据源）

    Args:
        stock_code (str): 股票代码，例如 "300440"
        period_days (int): 获取数据的天数，默认90天

    Returns:
        pandas.DataFrame: 包含股票数据的DataFrame
    """
    try:
        import akshare as ak

        # 获取个股历史数据（前复权）；失败重试1次（AKShare 限流常见 RemoteDisconnected）
        import time as _time
        df = None
        for _attempt in range(2):
            try:
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
                if df is not None and not df.empty:
                    break
            except Exception as e:
                if _attempt == 0:
                    print(f"AKShare: 获取 {stock_code} 失败({type(e).__name__})，2秒后重试...")
                    _time.sleep(2)
                else:
                    print(f"AKShare: 获取股票 {stock_code} 数据失败: {e}")
                    df = None

        if df is None or df.empty:
            print(f"AKShare: 无法获取股票 {stock_code} 的数据")
            return None

        # 标准化列名
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
        })

        # 转换日期
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)

        # 选择需要的列
        df = df[['Open', 'Close', 'High', 'Low', 'Volume']]

        # 取最近 period_days 天
        df = df.tail(period_days)

        return df

    except Exception as e:
        print(f"AKShare: 获取股票 {stock_code} 数据失败: {e}")
        return None


def _get_a_stock_data_impl(stock_code, period_days=90, use_cache=True, min_rows=200):
    """
    获取A股股票数据（优先腾讯财经，失败后使用AKShare）

    Args:
        stock_code (str): 股票代码
        period_days (int): 获取数据的天数
        use_cache (bool): 是否使用缓存
        min_rows (int): 最小数据行数（低于此值尝试备用数据源）

    Returns:
        pandas.DataFrame: 股票数据
    """
    # 检查缓存。缓存文件名含请求窗口，避免"小窗口写入的短数据被大窗口复用"
    # （2026-10-02 实测：缓存 641 行、传 1460/2600 均直接返回，
    #  致个股只有 2.5 年历史，早期折 lookback 残缺——港股对应教训 lessons 三.26/三.31）
    cache_file = os.path.join(A_STOCK_CACHE_DIR, f'{stock_code}_{int(period_days)}d.pkl')
    legacy_cache = os.path.join(A_STOCK_CACHE_DIR, f'{stock_code}.pkl')
    if use_cache and not os.path.exists(cache_file) and os.path.exists(legacy_cache):
        # 旧版无窗口后缀的缓存：行数不足则视为不覆盖（丢弃），否则沿用避免全量重拉
        try:
            _old = pd.read_pickle(legacy_cache)
            if _old is not None and len(_old) < int(period_days) * 0.9:
                print(f"  ⚠️ 旧缓存仅 {len(_old)} 行 < 所需 {int(period_days)} 天的 90%，重新获取 {stock_code}")
            else:
                cache_file = legacy_cache
        except Exception:
            pass
    if use_cache and os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(days=A_STOCK_CACHE_DAYS):
            try:
                df = pd.read_pickle(cache_file)
                # 缓存行数不达标则视为坏缓存，跳过并重新拉取
                # （防：period_days 传小/数据源故障写入的短数据被缓存 7 天）
                if df is not None and len(df) >= min_rows:
                    print(f"  ✅ 从缓存加载 {stock_code} 数据")
                    return df
                print(f"  ⚠️ 缓存行数不足（{0 if df is None else len(df)} < {min_rows}），重新获取")
            except Exception as e:
                print(f"  ⚠️ 缓存加载失败: {e}")

    # 优先使用腾讯财经
    df = get_a_stock_data_tencent(stock_code, period_days)

    # 腾讯失败或数据不足，尝试AKShare
    if df is None or len(df) < min_rows:
        if df is not None:
            print(f"  ⚠️ 腾讯财经数据不足（{len(df)} < {min_rows}），尝试 AKShare...")
        else:
            print(f"  ⚠️ 腾讯财经获取失败，尝试 AKShare...")
        df_ak = get_a_stock_data_akshare(stock_code, period_days)
        # 使用数据量更多的来源
        if df_ak is not None and (df is None or len(df_ak) > len(df)):
            df = df_ak

    # 保存缓存
    if df is not None:
        try:
            # 注意：腾讯 A股日线接口对 period_days>~1460 一律只返回约 640 行
            # （2026-10-02 实测 pd=1460/1500/2000 均 640 行；港股同期可返 2600 行）。
            # 故 A股历史长度受**数据源**限制，非调用方参数——需更长历史只能走 AKShare 全量 hist。
            df.to_pickle(cache_file)
            print(f"  ✅ 数据已缓存: {cache_file}（{len(df)}行）")
        except Exception as e:
            print(f"  ⚠️ 缓存保存失败: {e}")

    return df


def get_a_stock_info_tencent(stock_code):
    """
    通过腾讯财经接口获取A股股票基本信息

    Args:
        stock_code (str): 股票代码，例如 "300440"

    Returns:
        dict: 包含股票基本信息的字典
    """
    market = get_market_code(stock_code)
    url = f"http://qt.gtimg.cn/q={market}{stock_code}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.text
        if data.startswith('v_'):
            parts = data.split('~')
            if len(parts) > 32:
                return {
                    'stock_code': stock_code,
                    'stock_name': parts[1],
                    'current_price': float(parts[3]) if parts[3] else None,
                    'prev_close': float(parts[4]) if parts[4] else None,
                    'change_amount': float(parts[31]) if parts[31] else None,
                    'change_percent': float(parts[32]) if parts[32] else None,
                    'high': float(parts[33]) if len(parts) > 33 and parts[33] else None,
                    'low': float(parts[34]) if len(parts) > 34 and parts[34] else None,
                    'volume': float(parts[36]) if len(parts) > 36 and parts[36] else None,
                    'amount': float(parts[37]) if len(parts) > 37 and parts[37] else None,
                }

        print(f"无法获取股票 {stock_code} 信息")
        return None

    except Exception as e:
        print(f"获取股票 {stock_code} 信息失败: {e}")
        return None


def get_index_data_tencent(index_type='sh', period_days=90):
    """
    通过腾讯财经接口获取指数数据

    Args:
        index_type (str): 指数类型，'sh'(上证指数), 'sz'(深证成指), 'cyb'(创业板指)
        period_days (int): 获取数据的天数

    Returns:
        pandas.DataFrame: 指数数据
    """
    index_code = A_STOCK_INDEX_TENCENT.get(index_type)
    if not index_code:
        print(f"不支持的指数类型: {index_type}")
        return None

    # 腾讯财经指数接口
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={index_code},day,,,{period_days},"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stockapp.finance.qq.com/',
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        if 'data' not in data or index_code not in data['data']:
            print(f"无法获取指数 {index_type} 的数据")
            return None

        index_data = data['data'][index_code]
        kline_data = index_data.get('day', [])

        if not kline_data:
            print(f"指数 {index_type} 数据为空")
            return None

        # 解析数据
        parsed_data = []
        for item in kline_data:
            if len(item) >= 6:
                parsed_data.append({
                    'Date': pd.to_datetime(item[0], utc=True),
                    'Open': float(item[1]),
                    'Close': float(item[2]),
                    'High': float(item[3]),
                    'Low': float(item[4]),
                    'Volume': int(float(item[5])) if len(item) > 5 else 0,
                })

        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            return df.tail(period_days)

        return None

    except Exception as e:
        print(f"获取指数 {index_type} 数据失败: {e}")
        return None


def get_index_data_akshare(index_type='sh', period_days=90):
    """
    通过AKShare获取指数数据（备选）

    Args:
        index_type (str): 指数类型
        period_days (int): 获取数据的天数

    Returns:
        pandas.DataFrame: 指数数据
    """
    try:
        import akshare as ak

        # 指数代码映射
        index_codes = {
            'sh': 'sh000001',    # 上证指数
            'sz': 'sz399001',    # 深证成指
            'cyb': 'sz399006',   # 创业板指
        }

        code = index_codes.get(index_type)
        if not code:
            return None

        df = ak.stock_zh_index_daily(symbol=code)

        if df is None or df.empty:
            return None

        # 标准化列名
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
        })

        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)

        return df.tail(period_days)

    except Exception as e:
        print(f"AKShare: 获取指数 {index_type} 数据失败: {e}")
        return None


def _get_index_data_impl(index_type='sh', period_days=90):
    """
    获取指数数据（优先腾讯，失败后使用AKShare）

    Args:
        index_type (str): 指数类型
        period_days (int): 获取数据的天数

    Returns:
        pandas.DataFrame: 指数数据
    """
    df = get_index_data_tencent(index_type, period_days)

    if df is None:
        df = get_index_data_akshare(index_type, period_days)

    return df


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("A股数据获取测试")
    print("=" * 60)

    # 测试个股数据
    test_stocks = ['300440', '002655', '600800', '300765']

    for code in test_stocks:
        print(f"\n测试 {code}:")
        df = get_a_stock_data(code, period_days=30, use_cache=False)
        if df is not None:
            print(f"  获取成功: {len(df)} 条记录")
            print(f"  日期范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
        else:
            print("  获取失败")

    # 测试指数数据
    print("\n" + "=" * 60)
    print("指数数据测试")
    print("=" * 60)

    for index_type in ['sh', 'sz', 'cyb']:
        print(f"\n测试 {index_type}:")
        df = get_index_data(index_type, period_days=30)
        if df is not None:
            print(f"  获取成功: {len(df)} 条记录")
        else:
            print("  获取失败")

    # 测试实时报价
    print("\n" + "=" * 60)
    print("实时报价测试")
    print("=" * 60)

    info = get_a_stock_info_tencent('300440')
    if info:
        print(f"  {info['stock_name']}: {info['current_price']} ({info['change_percent']}%)")


def get_a_stock_data(stock_code, period_days=90, use_cache=True, min_rows=200):
    """获取 A 股日线（冻结优先包装器）

    冻结模式（2026-10-03，A_STOCK_SNAPSHOT_DIR）下跳过 7 天 TTL 与坏缓存重拉，
    使个股行情输入在跨日/跨运行间保持一致 —— 否则 walk-forward 结果随日期漂移
    （lessons 三.29：同代码跨日跑结果不同）。
    未设置该变量时行为与从前完全一致。
    """
    if not is_frozen():
        return _get_a_stock_data_impl(stock_code, period_days, use_cache, min_rows)

    key = f'quote_{stock_code}_{int(period_days)}d'
    found, fd = load_frozen(key)
    if found:
        return fd                      # None 表示已冻结的失败哨兵

    df = _get_a_stock_data_impl(stock_code, period_days, use_cache, min_rows)
    if df is not None and not getattr(df, 'empty', False):
        save_frozen(key, df)
    else:
        save_failure(key)
    return df


def get_index_data(index_type='sh', period_days=90):
    """获取指数日线（冻结优先包装器）

    此前 get_index_data **完全无缓存**，每次调用都联网 —— 在 A 股 53 只股票 ×
    19 折的 walk-forward 中被反复调用，是剩余的不可复现来源之一（2026-10-03）。
    """
    if not is_frozen():
        return _get_index_data_impl(index_type, period_days)

    key = f'index_{index_type}_{int(period_days)}d'
    found, fd = load_frozen(key)
    if found:
        return fd
    df = _get_index_data_impl(index_type, period_days)
    if df is not None and not getattr(df, 'empty', False):
        save_frozen(key, df)
    else:
        save_failure(key)
    return df
