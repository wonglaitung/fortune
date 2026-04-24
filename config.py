#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置文件
"""

# 股票板块映射（58只股票，覆盖16个板块）
# 包含股票名称、板块代码、类型、评分（防御性、成长性、周期性、流动性、风险）
STOCK_SECTOR_MAPPING = {
    # 银行股 (bank)
    '0005.HK': {'sector': 'bank', 'name': '汇丰银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 70, 'risk': 20},
    '0939.HK': {'sector': 'bank', 'name': '建设银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 80, 'risk': 20},
    '1288.HK': {'sector': 'bank', 'name': '农业银行', 'type': 'bank', 'defensive': 95, 'growth': 25, 'cyclical': 20, 'liquidity': 85, 'risk': 15},
    '1398.HK': {'sector': 'bank', 'name': '工商银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20},
    '3968.HK': {'sector': 'bank', 'name': '招商银行', 'type': 'bank', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 25},
    '2388.HK': {'sector': 'bank', 'name': '中银香港', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 75, 'risk': 20},

    # 科技股 (tech)
    '0700.HK': {'sector': 'tech', 'name': '腾讯控股', 'type': 'tech', 'defensive': 40, 'growth': 85, 'cyclical': 30, 'liquidity': 90, 'risk': 60},
    '9988.HK': {'sector': 'tech', 'name': '阿里巴巴-SW', 'type': 'tech', 'defensive': 35, 'growth': 85, 'cyclical': 35, 'liquidity': 85, 'risk': 65},
    '3690.HK': {'sector': 'tech', 'name': '美团-W', 'type': 'tech', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 85, 'risk': 70},
    '1810.HK': {'sector': 'tech', 'name': '小米集团-W', 'type': 'tech', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 80, 'risk': 65},
    '9618.HK': {'sector': 'tech', 'name': '京东集团-SW', 'type': 'tech', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 80, 'risk': 70},
    '9999.HK': {'sector': 'tech', 'name': '网易-S', 'type': 'tech', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 75, 'risk': 65},
    '9888.HK': {'sector': 'tech', 'name': '百度集团-SW', 'type': 'tech', 'defensive': 30, 'growth': 75, 'cyclical': 45, 'liquidity': 75, 'risk': 70},
    '1024.HK': {'sector': 'tech', 'name': '快手-W', 'type': 'tech', 'defensive': 25, 'growth': 85, 'cyclical': 50, 'liquidity': 70, 'risk': 75},

    # 半导体股 (semiconductor)
    '0981.HK': {'sector': 'semiconductor', 'name': '中芯国际', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 75, 'risk': 75},
    '1347.HK': {'sector': 'semiconductor', 'name': '华虹半导体', 'type': 'semiconductor', 'defensive': 20, 'growth': 85, 'cyclical': 75, 'liquidity': 70, 'risk': 80},
    '02382.HK': {'sector': 'semiconductor', 'name': '舜宇光学科技', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 70, 'risk': 75},

    # 人工智能股 (ai)
    '6682.HK': {'sector': 'ai', 'name': '范式智能', 'type': 'ai', 'defensive': 20, 'growth': 90, 'cyclical': 50, 'liquidity': 60, 'risk': 85},
    '9660.HK': {'sector': 'ai', 'name': '地平线机器人', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 60, 'liquidity': 55, 'risk': 90},
    '2533.HK': {'sector': 'ai', 'name': '黑芝麻智能', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 65, 'liquidity': 50, 'risk': 90},
    '0020.HK': {'sector': 'ai', 'name': '商汤-W', 'type': 'ai', 'defensive': 15, 'growth': 90, 'cyclical': 55, 'liquidity': 55, 'risk': 85},

    # 新能源股 (new_energy)
    '1211.HK': {'sector': 'new_energy', 'name': '比亚迪股份', 'type': 'new_energy', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 80, 'risk': 70},
    '1798.HK': {'sector': 'new_energy', 'name': '赣锋锂业', 'type': 'new_energy', 'defensive': 20, 'growth': 90, 'cyclical': 70, 'liquidity': 65, 'risk': 80},
    '3800.HK': {'sector': 'new_energy', 'name': '保利协鑫能源', 'type': 'new_energy', 'defensive': 15, 'growth': 85, 'cyclical': 75, 'liquidity': 60, 'risk': 85},
    '2282.HK': {'sector': 'new_energy', 'name': '比亚迪电子', 'type': 'new_energy', 'defensive': 25, 'growth': 80, 'cyclical': 65, 'liquidity': 70, 'risk': 75},
    '0960.HK': {'sector': 'new_energy', 'name': '龙源电力', 'type': 'new_energy', 'defensive': 30, 'growth': 70, 'cyclical': 65, 'liquidity': 65, 'risk': 70},

    # 环保股 (environmental)
    '1330.HK': {'sector': 'environmental', 'name': '绿色动力环保', 'type': 'environmental', 'defensive': 25, 'growth': 75, 'cyclical': 80, 'liquidity': 60, 'risk': 80},
    '01257.HK': {'sector': 'environmental', 'name': '中国光大环境', 'type': 'environmental', 'defensive': 30, 'growth': 70, 'cyclical': 75, 'liquidity': 65, 'risk': 75},
    '01387.HK': {'sector': 'environmental', 'name': '中国水务', 'type': 'environmental', 'defensive': 40, 'growth': 65, 'cyclical': 70, 'liquidity': 70, 'risk': 70},

    # 能源股 (energy)
    '0883.HK': {'sector': 'energy', 'name': '中国海洋石油', 'type': 'energy', 'defensive': 30, 'growth': 50, 'cyclical': 90, 'liquidity': 75, 'risk': 75},
    '1088.HK': {'sector': 'energy', 'name': '中国神华', 'type': 'energy', 'defensive': 40, 'growth': 45, 'cyclical': 85, 'liquidity': 70, 'risk': 70},
    '1171.HK': {'sector': 'energy', 'name': '兖矿能源', 'type': 'energy', 'defensive': 25, 'growth': 55, 'cyclical': 90, 'liquidity': 65, 'risk': 80},
    '02883.HK': {'sector': 'energy', 'name': '中海油服', 'type': 'energy', 'defensive': 20, 'growth': 60, 'cyclical': 95, 'liquidity': 60, 'risk': 85},

    # 航运股 (shipping)
    '1138.HK': {'sector': 'shipping', 'name': '中远海能', 'type': 'shipping', 'defensive': 25, 'growth': 45, 'cyclical': 95, 'liquidity': 65, 'risk': 80},
    '01919.HK': {'sector': 'shipping', 'name': '中远海控', 'type': 'shipping', 'defensive': 20, 'growth': 50, 'cyclical': 95, 'liquidity': 60, 'risk': 85},
    '02866.HK': {'sector': 'shipping', 'name': '中远海运港口', 'type': 'shipping', 'defensive': 30, 'growth': 40, 'cyclical': 90, 'liquidity': 70, 'risk': 75},

    # 交易所 (exchange)
    '0388.HK': {'sector': 'exchange', 'name': '香港交易所', 'type': 'exchange', 'defensive': 25, 'growth': 50, 'cyclical': 90, 'liquidity': 70, 'risk': 75},

    # 公用事业股 (utility)
    '0728.HK': {'sector': 'utility', 'name': '中国电信', 'type': 'utility', 'defensive': 90, 'growth': 25, 'cyclical': 15, 'liquidity': 70, 'risk': 20},
    '0941.HK': {'sector': 'utility', 'name': '中国移动', 'type': 'utility', 'defensive': 95, 'growth': 30, 'cyclical': 15, 'liquidity': 80, 'risk': 15},
    '0002.HK': {'sector': 'utility', 'name': '中电控股', 'type': 'utility', 'defensive': 95, 'growth': 20, 'cyclical': 10, 'liquidity': 75, 'risk': 10},
    '0006.HK': {'sector': 'utility', 'name': '电能实业', 'type': 'utility', 'defensive': 95, 'growth': 20, 'cyclical': 10, 'liquidity': 70, 'risk': 10},

    # 保险股 (insurance)
    '1299.HK': {'sector': 'insurance', 'name': '友邦保险', 'type': 'insurance', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 30},
    '2318.HK': {'sector': 'insurance', 'name': '中国平安', 'type': 'insurance', 'defensive': 80, 'growth': 45, 'cyclical': 30, 'liquidity': 70, 'risk': 35},
    '2601.HK': {'sector': 'insurance', 'name': '中国太保', 'type': 'insurance', 'defensive': 80, 'growth': 40, 'cyclical': 30, 'liquidity': 65, 'risk': 35},
    '0966.HK': {'sector': 'insurance', 'name': '中国人寿', 'type': 'insurance', 'defensive': 85, 'growth': 35, 'cyclical': 25, 'liquidity': 70, 'risk': 30},
    '02318.HK': {'sector': 'insurance', 'name': '中国平安证券', 'type': 'insurance', 'defensive': 75, 'growth': 50, 'cyclical': 30, 'liquidity': 65, 'risk': 35},

    # 生物医药股 (biotech)
    '2269.HK': {'sector': 'biotech', 'name': '药明生物', 'type': 'biotech', 'defensive': 30, 'growth': 80, 'cyclical': 55, 'liquidity': 70, 'risk': 70},
    '02269.HK': {'sector': 'biotech', 'name': '药明康德', 'type': 'biotech', 'defensive': 30, 'growth': 75, 'cyclical': 60, 'liquidity': 65, 'risk': 75},
    '01177.HK': {'sector': 'biotech', 'name': '中国生物制药', 'type': 'biotech', 'defensive': 35, 'growth': 70, 'cyclical': 50, 'liquidity': 70, 'risk': 65},
    '02186.HK': {'sector': 'biotech', 'name': '绿叶制药', 'type': 'biotech', 'defensive': 30, 'growth': 65, 'cyclical': 55, 'liquidity': 60, 'risk': 70},

    # 指数基金 (index)
    '2800.HK': {'sector': 'index', 'name': '盈富基金', 'type': 'index', 'defensive': 80, 'growth': 40, 'cyclical': 30, 'liquidity': 90, 'risk': 25},
    '2828.HK': {'sector': 'index', 'name': '恒生中国企业', 'type': 'index', 'defensive': 75, 'growth': 45, 'cyclical': 35, 'liquidity': 85, 'risk': 30},

    # 房地产股 (real_estate)
    '1109.HK': {'sector': 'real_estate', 'name': '华润置地', 'type': 'real_estate', 'defensive': 30, 'growth': 40, 'cyclical': 85, 'liquidity': 60, 'risk': 75},
    '0012.HK': {'sector': 'real_estate', 'name': '恒基地产', 'type': 'real_estate', 'defensive': 20, 'growth': 30, 'cyclical': 95, 'liquidity': 50, 'risk': 85},
    '0016.HK': {'sector': 'real_estate', 'name': '新鸿基地产', 'type': 'real_estate', 'defensive': 25, 'growth': 35, 'cyclical': 90, 'liquidity': 55, 'risk': 80},

    # 消费股 (consumer)
    '00151.HK': {'sector': 'consumer', 'name': '中国旺旺', 'type': 'consumer', 'defensive': 50, 'growth': 45, 'cyclical': 45, 'liquidity': 70, 'risk': 45},
    '02228.HK': {'sector': 'consumer', 'name': '中国飞鹤', 'type': 'consumer', 'defensive': 45, 'growth': 50, 'cyclical': 50, 'liquidity': 65, 'risk': 50},

    # 汽车股 (auto)
    '02333.HK': {'sector': 'auto', 'name': '长城汽车', 'type': 'auto', 'defensive': 25, 'growth': 75, 'cyclical': 70, 'liquidity': 70, 'risk': 75},
    '1053.HK': {'sector': 'auto', 'name': '重庆长安汽车', 'type': 'auto', 'defensive': 20, 'growth': 80, 'cyclical': 75, 'liquidity': 65, 'risk': 80},
}

# 板块名称映射（统一中文名称）
SECTOR_NAME_MAPPING = {
    'bank': '银行股',
    'tech': '科技股',
    'semiconductor': '半导体股',
    'ai': '人工智能股',
    'new_energy': '新能源股',
    'environmental': '环保股',
    'energy': '能源股',
    'shipping': '航运股',
    'exchange': '交易所',
    'utility': '公用事业股',
    'insurance': '保险股',
    'biotech': '生物医药股',
    'real_estate': '房地产股',
    'index': '指数基金',
    'consumer': '消费股',
    'auto': '汽车股',
}

# 自选股列表（核心28只，用于预测和日常监控）
WATCHLIST = {
    "0005.HK": "汇丰银行",
    "0012.HK": "恒基地产",
    "0016.HK": "新鸿基地产",
    "0388.HK": "香港交易所",
    "0700.HK": "腾讯控股",
    "0728.HK": "中国电信",
    "0883.HK": "中国海洋石油",
    "0939.HK": "建设银行",
    "0941.HK": "中国移动",
    "0981.HK": "中芯国际",
    "1088.HK": "中国神华",
    "1109.HK": "华润置地",
    "1138.HK": "中远海能",
    "1288.HK": "农业银行",
    "1330.HK": "绿色动力环保",
    "1347.HK": "华虹半导体",
    "1398.HK": "工商银行",
    "1810.HK": "小米集团-W",
    "2269.HK": "药明生物",
    "2533.HK": "黑芝麻智能",
    "2800.HK": "盈富基金",
    "3690.HK": "美团-W",
    "3968.HK": "招商银行",
    "6682.HK": "范式智能",
    "9660.HK": "地平线机器人",
    "9988.HK": "阿里巴巴-SW",
    "1211.HK": "比亚迪股份",
    "1299.HK": "友邦保险",
}

# 训练用股票列表（扩展59只，用于模型训练以增加样本量）
# 包含WATCHLIST全部28只 + 31只额外股票
TRAINING_STOCKS = {
    **WATCHLIST,  # 继承核心28只
    # 额外31只股票（从STOCK_SECTOR_MAPPING补充）
    "0002.HK": "中电控股",
    "0006.HK": "电能实业",
    "00151.HK": "中国旺旺",
    "0020.HK": "商汤-W",
    "01177.HK": "中国生物制药",
    "01257.HK": "中国光大环境",
    "01387.HK": "中国水务",
    "01919.HK": "中远海控",
    "02186.HK": "绿叶制药",
    "02228.HK": "中国飞鹤",
    "02269.HK": "药明康德",
    "02318.HK": "中国平安证券",
    "02333.HK": "长城汽车",
    "02382.HK": "舜宇光学科技",
    "02866.HK": "中远海运港口",
    "02883.HK": "中海油服",
    "02828.HK": "恒生中国企业",
    "0960.HK": "龙源电力",
    "0966.HK": "中国人寿",
    "1024.HK": "快手-W",
    "1053.HK": "重庆长安汽车",
    "1171.HK": "兖矿能源",
    "1798.HK": "赣锋锂业",
    "2282.HK": "比亚迪电子",
    "2318.HK": "中国平安",
    "2388.HK": "中银香港",
    "2601.HK": "中国太保",
    "3800.HK": "保利协鑫能源",
    "9618.HK": "京东集团-SW",
    "9888.HK": "百度集团-SW",
    "9999.HK": "网易-S",
}
