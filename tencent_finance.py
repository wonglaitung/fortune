import requests
import pandas as pd
from datetime import datetime, timedelta
import json

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
    # 使用正确的接口和参数格式
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param=hk{formatted_code},day,,,{period_days},qfq&r=0.123456"
    
    try:
        # 添加请求头以模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Referer': 'https://stockapp.finance.qq.com/'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 解析返回的JSON数据 (需要去除回调函数名)
        # 数据格式为: kline_dayqfq={...}
        text_data = response.text
        if text_data.startswith("kline_dayqfq="):
            # 直接提取等号后面的部分
            json_str = text_data[13:]  # 去除 "kline_dayqfq="
            data = json.loads(json_str)
        else:
            print(f"无法解析股票 {stock_code} 的返回数据: {text_data[:50]}")
            return None
        
        # 检查数据是否有效
        if 'data' not in data or f'hk{formatted_code}' not in data['data']:
            print(f"无法获取股票 {stock_code} 的数据")
            return None
            
        # 提取K线数据
        kline_data = None
        # 注意：数据在 'day' 键下
        if 'day' in data['data'][f'hk{formatted_code}']:
            kline_data = data['data'][f'hk{formatted_code}']['day']
        
        if kline_data is None or len(kline_data) == 0:
            print(f"无法获取股票 {stock_code} 的K线数据")
            return None
        
        # 解析数据
        # 数据格式: [日期, 开盘价, 收盘价, 最低价, 最高价, 成交量, 其他信息]
        parsed_data = []
        for item in kline_data:
            if len(item) >= 6:
                parsed_data.append({
                    'Date': pd.to_datetime(item[0]),
                    'Open': float(item[1]),
                    'Close': float(item[2]),
                    'Low': float(item[3]),
                    'High': float(item[4]),
                    'Volume': int(float(item[5]))  # 成交量可能是浮点数字符串
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
        pandas.DataFrame: 包含恒生指数数据的DataFrame，列包括Date, Open, High, Low, Close, Volume
    """
    # 腾讯财经API URL (历史交易数据)
    # 首先尝试获取前复权数据
    url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param=hkHSI,day,,,300,qfq"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 解析返回的JSON数据
        data = response.json()
        
        # 检查数据是否有效
        if 'data' not in data or 'hkHSI' not in data['data']:
            print("无法获取恒生指数的数据")
            return None
            
        # 提取K线数据
        # 尝试不同的数据键名
        kline_data = None
        if 'qfqday' in data['data']['hkHSI']:
            kline_data = data['data']['hkHSI']['qfqday']
        elif 'day' in data['data']['hkHSI']:
            kline_data = data['data']['hkHSI']['day']
        
        # 如果前复权数据为空，尝试获取原始数据
        if not kline_data or len(kline_data) == 0:
            url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param=hkHSI,day,,,300"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # 检查数据是否有效
            if 'data' not in data or 'hkHSI' not in data['data']:
                print("无法获取恒生指数的数据")
                return None
            
            # 提取K线数据
            if 'day' in data['data']['hkHSI']:
                kline_data = data['data']['hkHSI']['day']
        
        if kline_data is None or len(kline_data) == 0:
            print("无法获取恒生指数的K线数据")
            return None
        
        # 解析数据
        parsed_data = []
        for item in kline_data:
            # 数据格式: ["2023-10-26", "320.00", "325.00", "318.00", "322.00", "1000000", {}]
            # 日期,开盘价,收盘价,最高价,最低价,成交量,其他数据
            if len(item) >= 6:
                parsed_data.append({
                    'Date': pd.to_datetime(item[0]),
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
            # 取最近 period_days 天的数据
            df = df.tail(period_days)
            return df
        else:
            print("恒生指数数据为空")
            return None
            
    except Exception as e:
        print(f"获取恒生指数数据失败: {e}")
        return None