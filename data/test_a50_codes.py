#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试各种A50期货指数代码
"""

import yfinance as yf

# 测试各种可能的A50代码
test_symbols = [
    "CNA50.SI",  # 新加坡交易所的CN A50
    "CN50.SI",   # CN A50
    "FTCH.SI",   # FTSE China
    "XIN.SI",    # Xinhua
    "0500.HK.HK",  # 港交所A50指数
    "A50.HK",    # A50 in HK
    "^FTCH",     # FTSE China A50 index
    "^CN50",     # CN50 index
]

print("测试各种A50期货指数代码...")
print("="*60)

for symbol in test_symbols:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty:
            latest = hist.iloc[-1]
            print(f"✅ {symbol}:")
            print(f"   最新价格: {latest['Close']:.2f}")
            print(f"   成交量: {latest['Volume']:.0f}")
            print(f"   日期: {hist.index[-1].strftime('%Y-%m-%d')}")
            print()
        else:
            print(f"❌ {symbol}: 无数据")
    except Exception as e:
        print(f"❌ {symbol}: {str(e)[:50]}")

print("="*60)
