#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股配置文件

包含A股自选股列表、涨跌停规则、指数代码等配置
"""

# ========== A股自选股列表 ==========
# 核心持仓股（4只）

A_STOCK_WATCHLIST = {
    "300440": "运达科技",   # 创业板，20%涨跌停
    "002655": "共达电声",   # 深主板，10%涨跌停
    "600800": "渤海化学",   # 沪主板，10%涨跌停
    "300765": "石药创新",   # 创业板，20%涨跌停
}

# 训练用股票列表（可扩展）
A_STOCK_TRAINING_LIST = A_STOCK_WATCHLIST.copy()

# ========== 板块映射 ==========

A_STOCK_SECTOR_MAPPING = {
    "300440": {
        'sector': 'tech',
        'name': '运达科技',
        'type': 'software',
        'defensive': 30,
        'growth': 70,
        'cyclical': 40,
        'liquidity': 60,
        'risk': 60,
    },
    "002655": {
        'sector': 'tech',
        'name': '共达电声',
        'type': 'electronics',
        'defensive': 40,
        'growth': 60,
        'cyclical': 50,
        'liquidity': 50,
        'risk': 55,
    },
    "600800": {
        'sector': 'chemical',
        'name': '渤海化学',
        'type': 'chemical',
        'defensive': 50,
        'growth': 30,
        'cyclical': 70,
        'liquidity': 60,
        'risk': 50,
    },
    "300765": {
        'sector': 'pharmaceutical',
        'name': '石药创新',
        'type': 'biotech',
        'defensive': 40,
        'growth': 70,
        'cyclical': 30,
        'liquidity': 55,
        'risk': 65,
    },
}

# 板块名称映射
A_STOCK_SECTOR_NAME_MAPPING = {
    'tech': '科技股',
    'chemical': '化工股',
    'pharmaceutical': '医药股',
}

# ========== 涨跌停规则 ==========

LIMIT_RULES = {
    'main': 0.10,      # 主板 10%
    'gem': 0.20,       # 创业板 20%
    'star': 0.20,      # 科创板 20%
    'bse': 0.30,       # 北交所 30%
}

def get_limit_rate(stock_code):
    """
    根据股票代码判断涨跌停限制

    Args:
        stock_code (str): 股票代码，如 "300440"

    Returns:
        float: 涨跌停比例，如 0.20 表示 20%
    """
    # 创业板（300xxx, 301xxx）
    if stock_code.startswith('300') or stock_code.startswith('301'):
        return LIMIT_RULES['gem']

    # 科创板（688xxx）
    if stock_code.startswith('688'):
        return LIMIT_RULES['star']

    # 北交所（8xxxxx, 4xxxxx）
    if stock_code.startswith('8') or stock_code.startswith('4'):
        return LIMIT_RULES['bse']

    # 主板（60xxxx, 00xxxx）
    return LIMIT_RULES['main']

def get_market_code(stock_code):
    """
    根据股票代码判断市场代码

    Args:
        stock_code (str): 股票代码，如 "300440"

    Returns:
        str: 市场代码，'sh'（上海）或 'sz'（深圳）
    """
    # 上海交易所：6开头
    if stock_code.startswith('6'):
        return 'sh'
    # 深圳交易所：0、3开头
    return 'sz'

# ========== 指数代码 ==========

A_STOCK_INDEX = {
    'sh': '000001',      # 上证指数
    'sz': '399001',      # 深证成指
    'cyb': '399006',     # 创业板指
    'hs300': '000300',   # 沪深300
    'zz500': '000905',   # 中证500
}

# 指数腾讯代码（用于腾讯财经API）
A_STOCK_INDEX_TENCENT = {
    'sh': 'sh000001',    # 上证指数
    'sz': 'sz399001',    # 深证成指
    'cyb': 'sz399006',   # 创业板指
}

# ========== 交易时间 ==========

A_STOCK_TRADING_HOURS = {
    'morning': {
        'start': '09:30',
        'end': '11:30',
    },
    'afternoon': {
        'start': '13:00',
        'end': '15:00',
    },
}

# ========== 数据缓存配置 ==========

A_STOCK_CACHE_DIR = 'data/a_stock_cache'
A_STOCK_FEATURE_CACHE_DIR = 'data/a_stock_feature_cache'
A_STOCK_CACHE_DAYS = 7  # 缓存有效期（天）
