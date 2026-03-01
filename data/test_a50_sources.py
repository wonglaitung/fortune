#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同的A50期货指数数据源
"""

import requests
import re

def test_sina_qq_api():
    """测试新浪和腾讯的A50指数API"""
    test_urls = [
        ("新浪-富时A50", "http://hq.sinajs.cn/list=s_sh000001"),  # 上证指数（测试）
        ("新浪-IFX0", "http://hq.sinajs.cn/list=hf_IFX0"),    # A50期货
        ("新浪-CN00", "http://hq.sinajs.cn/list=hf_CN00"),     # A50期货
        ("新浪-0500", "http://hq.sinajs.cn/list=hk0500"),     # 港股0500（测试）
        ("新浪-000050", "http://hq.sinajs.cn/list=s_sh000050"), # 上证50指数
        ("新浪-000300", "http://hq.sinajs.cn/list=s_sh000300"), # 沪深300指数
    ]

    for name, url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            response.encoding = 'gbk'
            if response.status_code == 200 and len(response.text) > 50:
                print(f"✅ {name}")
                data = response.text[:200]
                print(f"   数据: {data[:100]}...")
                if '=' in data and '"' in data:
                    values = data.split('"')[1].split(',')
                    if len(values) > 0:
                        try:
                            price = float(values[0])
                            if price > 1:  # 排除明显的错误值
                                print(f"   价格: {price:.2f}")
                        except:
                            pass
            else:
                print(f"❌ {name} - 无数据")
        except Exception as e:
            print(f"❌ {name} - {str(e)[:30]}")
        print()

def test_tencent_api():
    """测试腾讯财经API"""
    print("测试腾讯财经API...")
    # 上证50指数
    url = "http://qt.gtimg.cn/q=s_sh000050"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and 'v_s_sh000050=' in response.text:
            data = response.text.split('~')
            if len(data) > 3:
                name = data[1]
                price = float(data[3]) if data[3] else 0
                prev_close = float(data[4]) if data[4] else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                print(f"✅ 上证50指数: {name}, 价格: {price:.2f}, 涨跌: {change_pct:+.2f}%")
                return True
    except Exception as e:
        print(f"❌ 腾讯财经失败: {e}")
    return False


if __name__ == "__main__":
    print("="*60)
    print("测试A50期货指数数据源")
    print("="*60)
    print()

    print("第一部分：测试新浪财经API")
    print("-"*60)
    test_sina_qq_api()

    print()
    print("第二部分：测试腾讯财经API（上证50指数）")
    print("-"*60)
    test_tencent_api()

    print()
    print("="*60)
    print("结论：如果无法直接获取A50期货指数，")
    print("建议使用上证50指数或沪深300指数作为替代，")
    print("这些指数与A50期货高度相关。")
    print("="*60)
