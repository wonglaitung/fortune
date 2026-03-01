#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AkShare获取指数数据
"""

import akshare as ak
from datetime import datetime, timedelta

# 获取上证50
print("上证50指数:")
sse50 = ak.index_zh_a_hist(symbol='000050', period='daily', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
print(sse50.tail(3))
print(f"列名: {sse50.columns.tolist()}")
print()

# 获取沪深300
print("沪深300指数:")
hs300 = ak.index_zh_a_hist(symbol='000300', period='daily', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
print(hs300.tail(3))
print(f"列名: {hs300.columns.tolist()}")
