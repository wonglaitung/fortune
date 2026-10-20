#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置文件
"""

# 股票板块映射（56只股票，覆盖17个板块）
# ⚠️ 2026-10-14 据腾讯行情逐只核对公司名（lessons 三.74）：修正多只错名 +
#    0960 龙湖集团（地产，原误标 new_energy）、2282 美高梅中国（消费，原误标 new_energy）、
#    2228 晶泰控股（生物医药，原误标 consumer）、1053 重庆钢铁股份（材料，原误标 auto）；
#    去掉重复键 2269（原并存 药明生物/药明康德 两条，后者静默覆盖前者）。
#    ⚠️ 来源存疑（未改代码，仅记录）：0960 原标「龙源电力」疑为 0916.HK 串位
#    （0916=龙源电力；数据按 0960 取=龙湖集团，故按 0960 归类）。
#    ⚠️ `sector` 进中性化选股；`type`/profile 进 Stock_Type/Stock_*_Score **候选特征**
#    ⇒ 下次重训须清特征缓存（本次未重训，不影响现行 pkl）。
#    另**删除幽灵条目 1387.HK**：腾讯/新浪/akshare 三源均取不到数据（非上市代码），
#    且从未进入任何一轮（无特征/无预测；见 lessons 三.74 附）⇒ 池 57 → **56**。
#    若将来要引入中国水务（0855.HK），属**改池**，须走 D31 式预登记 + 重跑。
# 包含股票名称、板块代码、类型、评分（防御性、成长性、周期性、流动性、风险）
STOCK_SECTOR_MAPPING = {
    # 银行股 (bank)
    '0005.HK': {'sector': 'bank', 'name': '汇丰控股', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 70, 'risk': 20},
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
    '2382.HK': {'sector': 'semiconductor', 'name': '舜宇光学科技', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 70, 'risk': 75},

    # 人工智能股 (ai)
    '6682.HK': {'sector': 'ai', 'name': '范式智能', 'type': 'ai', 'defensive': 20, 'growth': 90, 'cyclical': 50, 'liquidity': 60, 'risk': 85},
    '9660.HK': {'sector': 'ai', 'name': '地平线机器人', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 60, 'liquidity': 55, 'risk': 90},
    '2533.HK': {'sector': 'ai', 'name': '黑芝麻智能', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 65, 'liquidity': 50, 'risk': 90},
    '0020.HK': {'sector': 'ai', 'name': '商汤-W', 'type': 'ai', 'defensive': 15, 'growth': 90, 'cyclical': 55, 'liquidity': 55, 'risk': 85},
    '2513.HK': {'sector': 'ai', 'name': '智谱', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 50, 'liquidity': 65, 'risk': 90},

    # 新能源股 (new_energy)
    '1211.HK': {'sector': 'new_energy', 'name': '比亚迪股份', 'type': 'new_energy', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 80, 'risk': 70},
    '1798.HK': {'sector': 'new_energy', 'name': '大唐新能源', 'type': 'new_energy', 'defensive': 20, 'growth': 90, 'cyclical': 70, 'liquidity': 65, 'risk': 80},
    '3800.HK': {'sector': 'new_energy', 'name': '协鑫科技', 'type': 'new_energy', 'defensive': 15, 'growth': 85, 'cyclical': 75, 'liquidity': 60, 'risk': 85},
    '2282.HK': {'sector': 'consumer', 'name': '美高梅中国', 'type': 'consumer', 'defensive': 30, 'growth': 50, 'cyclical': 80, 'liquidity': 65, 'risk': 75},
    '0960.HK': {'sector': 'real_estate', 'name': '龙湖集团', 'type': 'real_estate', 'defensive': 20, 'growth': 30, 'cyclical': 95, 'liquidity': 50, 'risk': 90},

    # 环保股 (environmental)
    '1330.HK': {'sector': 'environmental', 'name': '绿色动力环保', 'type': 'environmental', 'defensive': 25, 'growth': 75, 'cyclical': 80, 'liquidity': 60, 'risk': 80},
    '1257.HK': {'sector': 'environmental', 'name': '中国光大绿色环保', 'type': 'environmental', 'defensive': 30, 'growth': 70, 'cyclical': 75, 'liquidity': 65, 'risk': 75},

    # 能源股 (energy)
    '0883.HK': {'sector': 'energy', 'name': '中国海洋石油', 'type': 'energy', 'defensive': 30, 'growth': 50, 'cyclical': 90, 'liquidity': 75, 'risk': 75},
    '1088.HK': {'sector': 'energy', 'name': '中国神华', 'type': 'energy', 'defensive': 40, 'growth': 45, 'cyclical': 85, 'liquidity': 70, 'risk': 70},
    '1171.HK': {'sector': 'energy', 'name': '兖矿能源', 'type': 'energy', 'defensive': 25, 'growth': 55, 'cyclical': 90, 'liquidity': 65, 'risk': 80},
    '2883.HK': {'sector': 'energy', 'name': '中海油田服务', 'type': 'energy', 'defensive': 20, 'growth': 60, 'cyclical': 95, 'liquidity': 60, 'risk': 85},

    # 航运股 (shipping)
    '1138.HK': {'sector': 'shipping', 'name': '中远海能', 'type': 'shipping', 'defensive': 25, 'growth': 45, 'cyclical': 95, 'liquidity': 65, 'risk': 80},
    '1919.HK': {'sector': 'shipping', 'name': '中远海控', 'type': 'shipping', 'defensive': 20, 'growth': 50, 'cyclical': 95, 'liquidity': 60, 'risk': 85},
    '2866.HK': {'sector': 'shipping', 'name': '中远海发', 'type': 'shipping', 'defensive': 30, 'growth': 40, 'cyclical': 90, 'liquidity': 70, 'risk': 75},

    # 燃气股 (gas)
    '1600.HK': {'sector': 'gas', 'name': '天伦燃气', 'type': 'gas', 'defensive': 70, 'growth': 45, 'cyclical': 30, 'liquidity': 55, 'risk': 40},

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
    '0966.HK': {'sector': 'insurance', 'name': '中国太平', 'type': 'insurance', 'defensive': 85, 'growth': 35, 'cyclical': 25, 'liquidity': 70, 'risk': 30},

    # 生物医药股 (biotech)
    '2269.HK': {'sector': 'biotech', 'name': '药明生物', 'type': 'biotech', 'defensive': 30, 'growth': 80, 'cyclical': 55, 'liquidity': 70, 'risk': 70},
    '1177.HK': {'sector': 'biotech', 'name': '中国生物制药', 'type': 'biotech', 'defensive': 35, 'growth': 70, 'cyclical': 50, 'liquidity': 70, 'risk': 65},
    '2186.HK': {'sector': 'biotech', 'name': '绿叶制药', 'type': 'biotech', 'defensive': 30, 'growth': 65, 'cyclical': 55, 'liquidity': 60, 'risk': 70},

    # 指数基金 (index)：2800.HK 盈富基金、2828.HK 恒生中国企业
    # 2026-10-06 自训练池剔除（ETF 参与个股横截面排序不合理，见 D14）；
    # 从 WATCHLIST / TRAINING_STOCKS / STOCK_SECTOR_MAPPING 三处同步移除以保持单一股票宇宙

    # 房地产股 (real_estate)
    '1109.HK': {'sector': 'real_estate', 'name': '华润置地', 'type': 'real_estate', 'defensive': 30, 'growth': 40, 'cyclical': 85, 'liquidity': 60, 'risk': 75},
    '0012.HK': {'sector': 'real_estate', 'name': '恒基地产', 'type': 'real_estate', 'defensive': 20, 'growth': 30, 'cyclical': 95, 'liquidity': 50, 'risk': 85},
    '0016.HK': {'sector': 'real_estate', 'name': '新鸿基地产', 'type': 'real_estate', 'defensive': 25, 'growth': 35, 'cyclical': 90, 'liquidity': 55, 'risk': 80},

    # 消费股 (consumer)
    '0151.HK': {'sector': 'consumer', 'name': '中国旺旺', 'type': 'consumer', 'defensive': 50, 'growth': 45, 'cyclical': 45, 'liquidity': 70, 'risk': 45},
    '2228.HK': {'sector': 'biotech', 'name': '晶泰控股', 'type': 'biotech', 'defensive': 25, 'growth': 85, 'cyclical': 50, 'liquidity': 60, 'risk': 85},

    # 汽车股 (auto)
    '2333.HK': {'sector': 'auto', 'name': '长城汽车', 'type': 'auto', 'defensive': 25, 'growth': 75, 'cyclical': 70, 'liquidity': 70, 'risk': 75},
    '1053.HK': {'sector': 'materials', 'name': '重庆钢铁股份', 'type': 'materials', 'defensive': 15, 'growth': 40, 'cyclical': 95, 'liquidity': 60, 'risk': 90},
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
    'gas': '燃气股',
    'materials': '材料股',
}

# 自选股列表（核心32只，用于预测和日常监控；2026-10-06 剔除 ETF 2800.HK 后）
# ⚠️ 名称已于 2026-10-14 据腾讯行情核对（lessons 三.74），与 STOCK_SECTOR_MAPPING 同步
WATCHLIST = {
    "0005.HK": "汇丰控股",
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
    "1600.HK": "天伦燃气",
    "1810.HK": "小米集团-W",
    "2269.HK": "药明生物",
    "2388.HK": "中银香港",
    "2533.HK": "黑芝麻智能",
    # "2800.HK": "盈富基金" —— 2026-10-06 剔除（ETF，见 D14）
    "3690.HK": "美团-W",
    "3968.HK": "招商银行",
    "6682.HK": "范式智能",
    "9660.HK": "地平线机器人",
    "2513.HK": "智谱",
    "9888.HK": "百度集团-SW",
    "9988.HK": "阿里巴巴-SW",
    "1211.HK": "比亚迪股份",
    "1299.HK": "友邦保险",
    "2318.HK": "中国平安",
}

# 训练用股票列表（扩展56只，用于模型训练以增加样本量）
# 包含WATCHLIST全部32只 + 24只额外股票（2026-10-14 删 1387.HK 幽灵条目）
TRAINING_STOCKS = {
    **WATCHLIST,  # 继承核心32只
    # 额外24只股票（从STOCK_SECTOR_MAPPING补充）
    "0002.HK": "中电控股",
    "0006.HK": "电能实业",
    "0151.HK": "中国旺旺",
    "0020.HK": "商汤-W",
    "1177.HK": "中国生物制药",
    "1257.HK": "中国光大绿色环保",
    "1919.HK": "中远海控",
    "2186.HK": "绿叶制药",
    "2228.HK": "晶泰控股",
    "2269.HK": "药明生物",
    "2333.HK": "长城汽车",
    "2382.HK": "舜宇光学科技",
    "2866.HK": "中远海发",
    "2883.HK": "中海油田服务",
    # "2828.HK": "恒生中国企业" —— 2026-10-06 剔除（ETF，见 D14）
    "0960.HK": "龙湖集团",
    "0966.HK": "中国太平",
    "1024.HK": "快手-W",
    "1053.HK": "重庆钢铁股份",
    "1171.HK": "兖矿能源",
    "1798.HK": "大唐新能源",
    "2282.HK": "美高梅中国",
    "2388.HK": "中银香港",
    "2601.HK": "中国太保",
    "3800.HK": "协鑫科技",
    "9618.HK": "京东集团-SW",
    "9888.HK": "百度集团-SW",
    "9999.HK": "网易-S",
}

# ========== 预测周期（唯一来源，D59：1/5/20 → 5/10/20） ==========
# ⚠️ 只用于「预测/评估/展示」周期。板块/网络等**特征工程**周期（如 sector_avg_change_1d/5d/20d）
#    是特征列名，改动等于改特征集，须重训 —— 不引用本常量。
PREDICTION_HORIZONS = (5, 10, 20)
