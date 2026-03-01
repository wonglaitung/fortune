#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取富时中国A50期货指数数据
使用新浪财经或东方财富API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

def get_a50_futures_index_sina():
    """
    通过新浪财经API获取富时中国A50期货指数

    Returns:
        dict: 包含价格、涨跌幅、成交量等信息的字典
    """
    try:
        # 新浪财经的新富时A50期货代码
        # 新浪代码格式：hf_IFX0 (IFX0是A50期货的主合约)
        # 新富时A50期货代码：hf_0500
        url = "http://hq.sinajs.cn/list=hf_0500"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'gbk'

        if response.status_code == 200:
            # 解析返回数据
            # 格式：var hq_str_hf_0500="现价,昨收,开盘,最高,最低,买一,卖一,成交量,成交额,日期,时间,..."
            data_str = response.text
            if 'hq_str_hf_0500=' in data_str:
                data = data_str.split('"')[1].split(',')

                if len(data) >= 10:
                    current_price = float(data[0])
                    prev_close = float(data[1])
                    open_price = float(data[2])
                    high_price = float(data[3])
                    low_price = float(data[4])
                    volume = int(data[7]) if data[7] else 0

                    # 计算涨跌幅
                    change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

                    return {
                        'name': '富时中国A50期货指数',
                        'price': current_price,
                        'change_pct': change_pct,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'volume': volume,
                        'prev_close': prev_close
                    }

        return None
    except Exception as e:
        print(f"新浪财经获取A50期货失败: {e}")
        return None


def get_a50_futures_index_eastmoney():
    """
    通过东方财富API获取富时中国A50期货指数

    Returns:
        dict: 包含价格、涨跌幅、成交量等信息的字典
    """
    try:
        # 东方财富的新富时A50期货API
        # API地址：http://push2.eastmoney.com/api/qt/stock/get?secid=113.0500&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f60,f107,f116,f117,f127,f152
        # secid格式: 市场代码.股票代码
        # 新富时A50期货在东方财富的市场代码可能是113
        url = "http://push2.eastmoney.com/api/qt/stock/get?secid=113.0500&fields=f43,f44,f45,f46,f60,f107,f116"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://quote.eastmoney.com/'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('data') and data['data'].get('f43'):
                current_price = data['data']['f43'] / 100  # 东方财富价格需要除以100
                prev_close = data['data']['f60'] / 100 if data['data'].get('f60') else current_price

                change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

                return {
                    'name': '富时中国A50期货指数',
                    'price': current_price,
                    'change_pct': change_pct,
                    'open': data['data']['f46'] / 100 if data['data'].get('f46') else current_price,
                    'high': data['data']['f44'] / 100 if data['data'].get('f44') else current_price,
                    'low': data['data']['f45'] / 100 if data['data'].get('f45') else current_price,
                    'volume': data['data'].get('f47', 0),
                    'prev_close': prev_close
                }

        return None
    except Exception as e:
        print(f"东方财富获取A50期货失败: {e}")
        return None


def get_a50_futures_index():
    """
    获取富时中国A50期货指数数据（自动选择可用的数据源）

    Returns:
        dict: 包含价格、涨跌幅、成交量等信息的字典
    """
    # 尝试新浪财经
    print("尝试从新浪财经获取A50期货指数...")
    data = get_a50_futures_index_sina()
    if data and data['price'] > 0:
        print(f"✅ 新浪财经 - A50期货指数: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
        return data

    # 尝试东方财富
    print("尝试从东方财富获取A50期货指数...")
    data = get_a50_futures_index_eastmoney()
    if data and data['price'] > 0:
        print(f"✅ 东方财富 - A50期货指数: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
        return data

    print("❌ 所有数据源均无法获取A50期货指数")
    return None


if __name__ == "__main__":
    print("测试获取富时中国A50期货指数...")
    print("="*60)

    data = get_a50_futures_index()

    if data:
        print("\n" + "="*60)
        print("A50期货指数数据:")
        print("="*60)
        print(f"名称: {data['name']}")
        print(f"当前价格: {data['price']:.2f}")
        print(f"昨收: {data['prev_close']:.2f}")
        print(f"涨跌幅: {data['change_pct']:+.2f}%")
        print(f"开盘: {data['open']:.2f}")
        print(f"最高: {data['high']:.2f}")
        print(f"最低: {data['low']:.2f}")
        print(f"成交量: {data['volume']}")
    else:
        print("❌ 无法获取A50期货指数数据")
