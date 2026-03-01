#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A50期货指数数据获取模块
使用上证50指数作为A50期货的替代指标
"""

import requests
from datetime import datetime, timedelta
import pandas as pd


def get_sse50_index_tencent():
    """
    通过腾讯财经API获取上证50指数实时数据
    上证50指数与A50期货高度相关，是可靠的替代指标

    Returns:
        dict: 包含价格、涨跌幅等信息的字典
    """
    try:
        url = "http://qt.gtimg.cn/q=s_sh000050"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200 and 'v_s_sh000050=' in response.text:
            # 腾讯API返回格式：~名称~~~当前价~昨收~开盘~最高~最低~...
            # 数据索引：1=名称, 3=当前价, 4=昨收, 5=开盘, 33=今高, 34=今低
            data = response.text.split('~')

            if len(data) >= 5:
                name = data[1]
                current_price = float(data[3]) if data[3] else 0
                prev_close = float(data[4]) if data[4] else current_price

                # 验证昨收价是否合理（应该在当前价的合理范围内）
                # 如果昨收价异常小，说明数据格式可能有问题
                if prev_close < current_price * 0.5 or prev_close > current_price * 2:
                    print(f"警告：昨收价异常 ({prev_close})，尝试使用其他字段")
                    # 尝试从其他位置获取昨收价
                    for idx in [6, 7, 32]:  # 可能的其他字段位置
                        if len(data) > idx and data[idx]:
                            try:
                                test_price = float(data[idx])
                                if current_price * 0.5 <= test_price <= current_price * 2:
                                    prev_close = test_price
                                    print(f"修正昨收价为: {prev_close}")
                                    break
                            except:
                                pass

                # 计算涨跌幅
                change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

                return {
                    'name': '上证50指数（A50期货替代指标）',
                    'price': current_price,
                    'change_pct': change_pct,
                    'open': float(data[5]) if len(data) > 5 and data[5] else current_price,
                    'high': float(data[33]) if len(data) > 33 and data[33] else current_price,
                    'low': float(data[34]) if len(data) > 34 and data[34] else current_price,
                    'volume': 0,  # 指数无成交量
                    'prev_close': prev_close
                }

        return None
    except Exception as e:
        print(f"腾讯财经获取上证50失败: {e}")
        return None


def get_hs300_index_tencent():
    """
    通过腾讯财经API获取沪深300指数实时数据
    沪深300指数也与A50期货高度相关

    Returns:
        dict: 包含价格、涨跌幅等信息的字典
    """
    try:
        url = "http://qt.gtimg.cn/q=s_sh000300"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200 and 'v_s_sh000300=' in response.text:
            data = response.text.split('~')

            if len(data) >= 5:
                name = data[1]
                current_price = float(data[3]) if data[3] else 0
                prev_close = float(data[4]) if data[4] else current_price

                # 验证昨收价是否合理
                if prev_close < current_price * 0.5 or prev_close > current_price * 2:
                    print(f"警告：昨收价异常 ({prev_close})，尝试修正")
                    for idx in [6, 7, 32]:
                        if len(data) > idx and data[idx]:
                            try:
                                test_price = float(data[idx])
                                if current_price * 0.5 <= test_price <= current_price * 2:
                                    prev_close = test_price
                                    break
                            except:
                                pass

                change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

                return {
                    'name': '沪深300指数（A50期货替代指标）',
                    'price': current_price,
                    'change_pct': change_pct,
                    'open': float(data[5]) if len(data) > 5 and data[5] else current_price,
                    'high': float(data[33]) if len(data) > 33 and data[33] else current_price,
                    'low': float(data[34]) if len(data) > 34 and data[34] else current_price,
                    'volume': 0,
                    'prev_close': prev_close
                }

        return None
    except Exception as e:
        print(f"腾讯财经获取沪深300失败: {e}")
        return None


def get_a50_replacement_index():
    """
    获取A50期货的替代指标（上证50或沪深300指数）
    优先使用上证50指数，如果失败则使用沪深300指数

    Returns:
        dict: 包含价格、涨跌幅等信息的字典
    """
    # 优先尝试上证50指数
    print("尝试获取上证50指数（A50期货替代指标）...")
    data = get_sse50_index_tencent()

    if data and data['price'] > 0:
        print(f"✅ 上证50指数: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
        return data

    # 如果上证50失败，尝试沪深300
    print("尝试获取沪深300指数（A50期货替代指标）...")
    data = get_hs300_index_tencent()

    if data and data['price'] > 0:
        print(f"✅ 沪深300指数: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
        return data

    print("❌ 无法获取A50期货替代指标数据")
    return None


if __name__ == "__main__":
    print("测试获取A50期货替代指标...")
    print("="*60)

    data = get_a50_replacement_index()

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
    else:
        print("❌ 无法获取A50期货替代指标数据")
