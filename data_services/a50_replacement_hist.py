#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取上证50和沪深300指数的历史数据来计算涨跌幅
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


def get_index_hist_sina(index_code):
    """
    通过新浪财经获取指数历史数据
    index_code: sh000050 (上证50), sh000300 (沪深300)
    """
    try:
        url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={index_code}&scale=240&ma=no&datalen=5"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            import json
            data = json.loads(response.text)

            if data and len(data) >= 2:
                latest = data[0]
                prev = data[1]

                return {
                    'name': '上证50指数' if '000050' in index_code else '沪深300指数',
                    'price': float(latest['close']),
                    'prev_close': float(prev['close']),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'volume': int(latest['volume']) if latest.get('volume') else 0
                }

        return None
    except Exception as e:
        print(f"新浪财经获取指数历史数据失败: {e}")
        return None


def get_a50_replacement_with_history():
    """
    获取A50期货替代指标（使用历史数据计算涨跌幅）
    """
    # 优先尝试上证50
    print("尝试获取上证50指数历史数据...")
    data = get_index_hist_sina('sh000050')

    if data and data['price'] > 0:
        change_pct = ((data['price'] - data['prev_close']) / data['prev_close']) * 100
        data['change_pct'] = change_pct
        data['name'] = '上证50指数（A50期货替代指标）'
        print(f"✅ 上证50指数: {data['price']:.2f}, 涨跌: {change_pct:+.2f}%")
        return data

    # 如果上证50失败，尝试沪深300
    print("尝试获取沪深300指数历史数据...")
    data = get_index_hist_sina('sh000300')

    if data and data['price'] > 0:
        change_pct = ((data['price'] - data['prev_close']) / data['prev_close']) * 100
        data['change_pct'] = change_pct
        data['name'] = '沪深300指数（A50期货替代指标）'
        print(f"✅ 沪深300指数: {data['price']:.2f}, 涨跌: {change_pct:+.2f}%")
        return data

    print("❌ 无法获取A50期货替代指标数据")
    return None


if __name__ == "__main__":
    print("测试获取A50期货替代指标（历史数据版）...")
    print("="*60)

    data = get_a50_replacement_with_history()

    if data:
        print("\n" + "="*60)
        print("A50期货替代指标数据:")
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
        print("❌ 无法获取A50期货替代指标数据")
