#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试雅虎财经API获取财报公告日 - 调试版
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime

# 测试股票列表
TEST_STOCKS = [
    ('0700.HK', '腾讯控股'),
    ('0939.HK', '建设银行'),
]

def test_earnings_calendar_debug():
    """测试获取财报公告日 - 调试版"""
    print("=" * 80)
    print("测试雅虎财经API获取财报公告日（调试版）")
    print("=" * 80)

    for code, name in TEST_STOCKS:
        print(f"\n测试 {name} ({code})...")
        try:
            # 创建Ticker对象
            ticker = yf.Ticker(code)

            # 获取财报日历
            calendar = ticker.calendar

            print(f"  calendar类型: {type(calendar)}")
            print(f"  calendar内容: {calendar}")

            if calendar is None:
                print(f"  ⚠️ calendar is None")
                continue

            if isinstance(calendar, dict):
                print(f"\n  ✅ calendar是字典，键: {list(calendar.keys())}")
                for key, value in calendar.items():
                    print(f"    {key}: {value}")
                    if isinstance(value, pd.DataFrame):
                        print(f"      DataFrame形状: {value.shape}")
                        print(f"      DataFrame列: {value.columns.tolist()}")
                        print(f"      DataFrame前5行:")
                        print(value.head())
                    elif isinstance(value, pd.Series):
                        print(f"      Series形状: {value.shape}")
                        print(f"      Series值: {value.tolist()}")
            elif isinstance(calendar, pd.DataFrame):
                print(f"  ✅ calendar是DataFrame，形状: {calendar.shape}")
                print(f"  列: {calendar.columns.tolist()}")
                print(f"\n  前10行:")
                print(calendar.head(10))
            else:
                print(f"  ⚠️ 未知类型: {type(calendar)}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()

def test_earnings_dates():
    """测试使用earnings_dates属性"""
    print("\n" + "=" * 80)
    print("测试使用earnings_dates属性")
    print("=" * 80)

    for code, name in TEST_STOCKS:
        print(f"\n测试 {name} ({code})...")
        try:
            ticker = yf.Ticker(code)

            # 尝试earnings_dates
            earnings_dates = ticker.earnings_dates

            print(f"  earnings_dates类型: {type(earnings_dates)}")

            if earnings_dates is not None and isinstance(earnings_dates, pd.DataFrame):
                print(f"  ✅ 找到 {len(earnings_dates)} 条财报记录")
                print(f"  列: {earnings_dates.columns.tolist()}")
                print(f"\n  前10行:")
                print(earnings_dates.head(10))
            else:
                print(f"  ⚠️ earnings_dates不可用")

        except Exception as e:
            print(f"  ❌ 错误: {e}")

def test_earnings():
    """测试使用earnings属性"""
    print("\n" + "=" * 80)
    print("测试使用earnings属性")
    print("=" * 80)

    for code, name in TEST_STOCKS:
        print(f"\n测试 {name} ({code})...")
        try:
            ticker = yf.Ticker(code)

            # 尝试earnings
            earnings = ticker.earnings

            print(f"  earnings类型: {type(earnings)}")

            if earnings is not None:
                print(f"  ✅ earnings可用")
                print(f"  内容: {earnings}")
                if isinstance(earnings, pd.DataFrame):
                    print(f"  形状: {earnings.shape}")
                    print(f"  列: {earnings.columns.tolist()}")
                    print(earnings.head())
                elif isinstance(earnings, pd.Series):
                    print(f"  形状: {earnings.shape}")
                    print(earnings.head())
            else:
                print(f"  ⚠️ earnings不可用")

        except Exception as e:
            print(f"  ❌ 错误: {e}")

if __name__ == "__main__":
    # 测试calendar属性
    test_earnings_calendar_debug()

    # 测试earnings_dates属性
    test_earnings_dates()

    # 测试earnings属性
    test_earnings()
