#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试雅虎财经API获取财报公告日
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# 测试股票列表
TEST_STOCKS = [
    ('0700.HK', '腾讯控股'),
    ('0939.HK', '建设银行'),
    ('1288.HK', '农业银行'),
]

def test_earnings_calendar():
    """测试获取财报公告日"""
    print("=" * 80)
    print("测试雅虎财经API获取财报公告日")
    print("=" * 80)

    for code, name in TEST_STOCKS:
        print(f"\n测试 {name} ({code})...")
        try:
            # 创建Ticker对象
            ticker = yf.Ticker(code)

            # 获取财报日历
            calendar = ticker.calendar

            if calendar is None or calendar.empty:
                print(f"  ⚠️ 未找到财报数据")
                continue

            print(f"  ✅ 找到 {len(calendar)} 条财报记录")
            print(f"\n  财报日历列: {calendar.columns.tolist()}")
            print(f"\n  财报日历预览:")
            print(calendar.head(10).to_string())

        except Exception as e:
            print(f"  ❌ 错误: {e}")

def test_yahoo_finance_available():
    """测试yfinance是否可用"""
    print("\n" + "=" * 80)
    print("测试yfinance库是否可用")
    print("=" * 80)

    try:
        import yfinance as yf
        print(f"  ✅ yfinance版本: {yf.__version__}")

        # 测试获取腾讯数据
        ticker = yf.Ticker('0700.HK')
        info = ticker.info
        print(f"  ✅ 能获取股票信息")

        # 检查可用属性
        available_attrs = [attr for attr in dir(ticker) if not attr.startswith('_')]
        print(f"\n  Ticker可用属性（前20个）:")
        for attr in available_attrs[:20]:
            print(f"    - {attr}")

        # 测试获取历史数据
        hist = ticker.history(period="1mo")
        print(f"\n  ✅ 能获取历史数据: {len(hist)} 条")

        return True

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False

if __name__ == "__main__":
    # 测试yfinance是否可用
    if test_yahoo_finance_available():
        # 测试获取财报日历
        test_earnings_calendar()
    else:
        print("\n❌ yfinance不可用，请安装: pip install yfinance")