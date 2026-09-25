import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import glob
import pickle


def _request_json_retry(url, headers, timeout=15, attempts=3, base_delay=1.5):
    """带重试的 GET+JSON 解析（DNS/网络抖动兜底）"""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            print(f"  ⚠️ 请求失败({i+1}/{attempts}): {str(e)[:80]}")
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last


def _hk_yf_fallback(code, period_days):
    """备用源：yfinance 港股 .HK"""
    try:
        import yfinance as yf
        tk = f"{int(code):04d}.HK"
        hist = yf.Ticker(tk).history(period='max')
        if hist is None or len(hist) < 5:
            return None
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        now = pd.Timestamp.now(tz='UTC')
        df = df[df.index.tz_convert('UTC') >= (now - pd.Timedelta(days=period_days * 2))].tail(period_days)
        df.index = df.index.tz_convert('UTC')
        df.index.name = 'Date'
        print(f"  [备用源yfinance] 获取 {tk} 成功（{len(df)} 行）")
        return df
    except Exception as e:
        print(f"  [备用源yfinance失败] {code}: {str(e)[:60]}")
        return None


def _hk_cache_fallback(code):
    """兜底：读本地 stock_cache 最近缓存（可能过期，但保证有数据）"""
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        c4 = f"{int(code):04d}"
        pat = os.path.join(base, 'data', 'stock_cache', f"{c4}_*.pkl")
        files = sorted(glob.glob(pat))
        if not files:
            return None
        d = pickle.load(open(files[-1], 'rb'))
        df = d['data'] if isinstance(d, dict) and 'data' in d else d
        if isinstance(df, pd.DataFrame) and len(df) > 5 \
                and {'Open', 'Close', 'High', 'Low', 'Volume'} <= set(df.columns):
            print(f"  [缓存兜底] 使用 {os.path.basename(files[-1])}（{len(df)} 行，可能过期）")
            return df
    except Exception as e:
        print(f"  [缓存兜底失败] {code}: {str(e)[:60]}")
    return None


def get_hk_stock_data_tencent(stock_code, period_days=90):
    """
    通过腾讯财经接口获取港股股票数据

    Args:
        stock_code (str): 股票代码，例如 "00700" (腾讯)
        period_days (int): 获取数据的天数，默认90天

    Returns:
        pandas.DataFrame: 包含股票数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    # 确保股票代码是5位数字格式
    formatted_code = stock_code.zfill(5)

    # 腾讯财经API URL (历史交易数据)
    # 使用 hkfqkline 端点（绕过 WAF）
    # 注意：fqkline/get 端点会触发 WAF 501 错误
    url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param=hk{formatted_code},day,,,{period_days},qfq"

    try:
        # 添加请求头以模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://stockapp.finance.qq.com/',
            'Origin': 'https://stockapp.finance.qq.com',
            'Connection': 'keep-alive',
        }

        response = _request_json_retry(url, headers, timeout=15)

        # 解析返回的JSON数据
        data = response

        # 检查数据是否有效
        if 'data' not in data or f'hk{formatted_code}' not in data['data']:
            raise ValueError(f"无数据: {stock_code}")

        # 提取K线数据
        # 普通股票数据在 'qfqday' 键下（前复权数据）
        stock_data = data['data'][f'hk{formatted_code}']
        kline_data = None
        if 'qfqday' in stock_data:
            kline_data = stock_data['qfqday']
        elif 'day' in stock_data:
            kline_data = stock_data['day']

        if kline_data is None or len(kline_data) == 0:
            raise ValueError(f"无K线: {stock_code}")

        # 解析数据
        # 数据格式: [日期, 开盘价, 收盘价, 最高价, 最低价, 成交量, 其他信息, ?, 成交额(万元)]
        # 示例: ['2026-06-24', '148.800', '148.200', '149.900', '147.900', '10676159.000', {}, '0.060', '159009.520']
        parsed_data = []
        for item in kline_data:
            if len(item) >= 6:
                parsed_data.append({
                    'Date': pd.to_datetime(item[0], utc=True),
                    'Open': float(item[1]),
                    'Close': float(item[2]),
                    'High': float(item[3]),
                    'Low': float(item[4]),
                    'Volume': int(float(item[5]))  # 成交量可能是浮点数字符串
                })

        # 创建DataFrame
        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            return df
        else:
            raise ValueError(f"数据为空: {stock_code}")

    except Exception as e:
        print(f"获取股票 {stock_code} 数据失败（腾讯）: {str(e)[:80]}")
        # 备用源 + 缓存兜底
        fb = _hk_yf_fallback(formatted_code, period_days)
        if fb is not None:
            return fb
        fb = _hk_cache_fallback(formatted_code)
        return fb

def get_hk_stock_info_tencent(stock_code):
    """
    通过腾讯财经接口获取港股股票基本信息
    
    Args:
        stock_code (str): 股票代码，例如 "00700"
    
    Returns:
        dict: 包含股票基本信息的字典
    """
    # 腾讯财经API URL (实时数据)
    # 注意港股代码需要5位数字
    url = f"http://qt.gtimg.cn/q=hk{stock_code.zfill(5)}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 解析返回的数据
        data = response.text
        if data.startswith('v_'):
            # 提取数据部分
            data_parts = data.split('~')
            if len(data_parts) > 3:
                stock_name = data_parts[1]
                current_price = float(data_parts[3]) if data_parts[3] else None
                prev_close = float(data_parts[4]) if data_parts[4] else None
                change_amount = float(data_parts[31]) if data_parts[31] else None  # 涨跌额
                change_percent = float(data_parts[32]) if data_parts[32] else None  # 涨跌幅
                
                return {
                    "stock_name": stock_name,
                    "current_price": current_price,
                    "prev_close": prev_close,
                    "change_amount": change_amount,
                    "change_percent": change_percent
                }
            else:
                print(f"股票 {stock_code} 数据格式不正确")
                return None
        else:
            print(f"无法获取股票 {stock_code} 数据")
            return None
            
    except Exception as e:
        print(f"获取股票 {stock_code} 信息失败: {e}")
        return None

def get_hsi_data_tencent(period_days=90):
    """
    通过腾讯财经接口获取恒生指数数据

    Args:
        period_days (int): 获取数据的天数，默认90天

    Returns:
        pandas.DataFrame: 包含恒生指数数据的DataFrame，列包括Date, Open, High, Low, Close, Volume, Amount
    """
    # 腾讯财经API URL (历史交易数据)
    # 使用 hkfqkline 端点（绕过 WAF）
    # 注意：恒生指数也需要 qfq 参数，否则会返回 "bad params"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param=hkHSI,day,,,{period_days},qfq"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://stockapp.finance.qq.com/',
        }

        response = _request_json_retry(url, headers, timeout=15)

        # 解析返回的JSON数据
        data = response

        # 检查数据是否有效
        if 'data' not in data or 'hkHSI' not in data['data']:
            raise ValueError("无恒生指数数据")

        # 提取K线数据
        # 恒生指数数据在 'day' 键下（没有 qfqday）
        hsi_data = data['data']['hkHSI']
        kline_data = None
        if 'day' in hsi_data:
            kline_data = hsi_data['day']

        if kline_data is None or len(kline_data) == 0:
            print("无法获取恒生指数的K线数据")
            return None

        # 解析数据
        parsed_data = []
        for item in kline_data:
            # 腾讯财经API数据格式（恒生指数）:
            # [0] 日期, [1] 开盘, [2] 收盘, [3] 最高, [4] 最低, [5] 成交额(元), [6] {}, [7] ?, [8] 成交额(万元)
            # 示例: ['2026-04-16', '26122.801', '26394.260', '26403.070', '26122.801', '256227932806.00', {}, '0.00', '25622793.28', ...]
            if len(item) >= 6:
                row_data = {
                    'Date': pd.to_datetime(item[0], utc=True),
                    'Open': float(item[1]),
                    'Close': float(item[2]),
                    'High': float(item[3]),
                    'Low': float(item[4]),
                    'Volume': int(float(item[5]))  # 成交额（元）
                }
                # 添加成交额字段（如果有第8个字段，单位是万元）
                if len(item) >= 9:
                    # 字段[8]是成交额（万元），转换为亿港元
                    row_data['Amount'] = float(item[8]) / 10000  # 万元 -> 亿港元
                else:
                    # 从字段[5]计算（成交额元 -> 亿港元）
                    row_data['Amount'] = float(item[5]) / 100000000

                parsed_data.append(row_data)

        # 创建DataFrame
        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            # 取最近 period_days 天的数据
            df = df.tail(period_days)
            return df
        else:
            raise ValueError("恒生指数数据为空")

    except Exception as e:
        print(f"获取恒生指数数据失败（腾讯）: {str(e)[:80]}")
        # HSI 缓存兜底（data/hk_universe_cache/HSI.pkl）
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pkl = os.path.join(base, 'data', 'hk_universe_cache', 'HSI.pkl')
            if os.path.exists(pkl):
                hsi = pickle.load(open(pkl, 'rb'))
                hsi = hsi.tail(period_days)
                hsi.index = pd.to_datetime(hsi.index).tz_localize('UTC')
                print(f"  [HSI缓存兜底] 使用 HSI.pkl（{len(hsi)} 行，可能过期）")
                return hsi
        except Exception as ee:
            print(f"  [HSI缓存兜底失败]: {str(ee)[:60]}")
        return None