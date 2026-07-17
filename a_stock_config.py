#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股配置文件

包含A股自选股列表、涨跌停规则、指数代码等配置

版本：v2.0 (2026-07-17)
- 扩展股票池至53只（4核心+49扩展）
- 新增板块映射和样本权重配置
- 新增双指数市场基准（中证1000+创业板指）
"""

# ========== 核心持仓（4只）==========
# 用于预测、监控和交易的核心标的

A_STOCK_CORE_HOLDINGS = {
    "300440": "运达科技",   # 创业板，20%涨跌停，轨交IT核心
    "002655": "共达电声",   # 深主板，10%涨跌停，声学电子核心
    "600800": "渤海化学",   # 沪主板，10%涨跌停，化工周期核心
    "300765": "石药创新",   # 创业板，20%涨跌停，创新药核心
}

# ========== 扩展股票池（49只）==========
# 用于网络分析和样本量扩充

A_STOCK_EXTENSION_LIST = {
    # ========== 第一组：轨交IT与工业软件网络（围绕运达科技）==========
    # 核心竞争对手
    "300789": "唐源电气",    # 成都同城、轨交检测同业
    "603508": "思维列控",    # 轨交LKJ系统垄断者
    "300523": "凯发电气",    # 轨交供电自动化
    # 成都本地联动
    "300678": "中科信息",    # 成都本地政府/工业IT
    "688682": "成都华微",    # 成都本地特种芯片
    # 产业链上下游
    "300011": "鼎汉技术",    # 轨交地面及车载装备
    "600528": "中铁工业",    # 大型轨交工程装备
    "600458": "时代电气",    # 轨交牵引系统霸主
    # 高维节点
    "002920": "德赛西威",    # 智能座舱/工业智造
    "600150": "中国船舶",    # 泛工业装备大周期
    "688611": "杭州柯林",    # 电力/交通电气化监测
    "300098": "高新兴",      # 轨交物联网通信

    # ========== 第二组：声学、智能硬件与果链网络（围绕共达电声）==========
    # 声学龙头
    "002241": "歌尔股份",    # 声学全球龙头
    "300893": "朝阳科技",    # 微型电声、智能穿戴
    # 果链同盟
    "002475": "立讯精密",    # 消费电子绝对核心
    "002885": "杰美特",      # 智能手机配件
    "002456": "欧菲光",      # 光学/摄像头模组
    # 电子元器件网络
    "603936": "博敏电子",    # PCB电路板
    "002138": "顺络电子",    # 被动元器件龙头
    "300223": "北京君正",    # 车载/智能穿戴芯片
    # 智能穿戴联动
    "300416": "苏大维格",    # 光电子/微纳制造
    "300115": "长信科技",    # 消费电子显示模块
    "600203": "福日电子",    # 电子元器件OEM
    "300679": "电连技术",    # 微型电连接器
    "002635": "恒大高新",    # 泛消费电子互动节点

    # ========== 第三组：创新药、生物科技与研发网络（围绕石药创新）==========
    # CXO龙头
    "300759": "康龙化成",    # 大型CXO龙头
    "300725": "药石科技",    # 分子砌块/创新药上游
    "300347": "泰格医药",    # 临床CRO霸主
    # 创新药企
    "300558": "贝达药业",    # 抗肿瘤创新药
    "300009": "安科生物",    # 生物制品/生长激素
    "300122": "智飞生物",    # 疫苗巨头
    "688180": "君实生物",    # 科创板创新药
    # 高弹性品种
    "300142": "沃森生物",    # mRNA技术
    "300255": "常山药业",    # 多肽/创新药
    "300199": "翰宇药业",    # 多肽高成长
    # 医药总龙头
    "600276": "恒瑞医药",    # A股创新药总龙头
    # 体外诊断
    "300463": "迈克生物",    # 泛医药健康多元化

    # ========== 第四组：精细化工、周期与新能源材料网络（围绕渤海化学）==========
    # 化工龙头
    "600309": "万华化学",    # 全球MDI龙头
    "600346": "恒力石化",    # 大炼化巨头
    "601857": "中国石油",    # 能源终极源头
    "600028": "中国石化",    # 基础化工锚定
    # 氟化工/锂电材料
    "002407": "多氟多",      # 氟化工及锂电材料
    "300037": "新宙邦",      # 精细化工/电解液
    # 上下游联动
    "600688": "上海石化",    # 炼油/化工基础原料
    "002838": "道恩股份",    # 改性塑料/高分子
    "002340": "格林美",      # 新能源材料/循环
    "000301": "东方盛虹",    # 光伏材料/大炼化
    # 高端材料
    "002768": "国恩股份",    # 大健康及新材料
    "300054": "鼎龙股份",    # 半导体材料/精细化工
}

# ========== 训练用股票列表（53只）==========
A_STOCK_TRAINING_LIST = {
    **A_STOCK_CORE_HOLDINGS,
    **A_STOCK_EXTENSION_LIST,
}

# 自选股列表（兼容旧版本）
A_STOCK_WATCHLIST = A_STOCK_CORE_HOLDINGS

# ========== 样本权重配置 ==========
# 核心股权重3.0，扩展股权重1.0

A_STOCK_WEIGHTS = {
    # 核心持仓权重
    '300440': 3.0,  # 运达科技
    '002655': 3.0,  # 共达电声
    '300765': 3.0,  # 石药创新
    '600800': 3.0,  # 渤海化学
    # 扩展股默认权重
    'default': 1.0
}

def get_sample_weight(stock_code):
    """获取样本权重

    Args:
        stock_code (str): 股票代码

    Returns:
        float: 样本权重（核心股3.0，扩展股1.0）
    """
    return A_STOCK_WEIGHTS.get(stock_code, A_STOCK_WEIGHTS['default'])

def is_core_holding(stock_code):
    """判断是否为核心持仓

    Args:
        stock_code (str): 股票代码

    Returns:
        bool: 是否为核心持仓
    """
    return stock_code in A_STOCK_CORE_HOLDINGS

# ========== 板块映射 ==========

A_STOCK_SECTOR_MAPPING = {
    # ========== 核心持仓 ==========
    "300440": {
        'sector': 'railway_tech',
        'name': '运达科技',
        'type': 'software',
        'defensive': 30,
        'growth': 70,
        'cyclical': 40,
        'liquidity': 60,
        'risk': 60,
        'is_core': True,
    },
    "002655": {
        'sector': 'acoustics',
        'name': '共达电声',
        'type': 'electronics',
        'defensive': 40,
        'growth': 60,
        'cyclical': 50,
        'liquidity': 50,
        'risk': 55,
        'is_core': True,
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
        'is_core': True,
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
        'is_core': True,
    },

    # ========== 第一组：轨交IT与工业软件网络 ==========
    "300789": {
        'sector': 'railway_tech',
        'name': '唐源电气',
        'type': 'software',
        'defensive': 35, 'growth': 65, 'cyclical': 45, 'liquidity': 55, 'risk': 60,
        'is_core': False,
    },
    "603508": {
        'sector': 'railway_tech',
        'name': '思维列控',
        'type': 'software',
        'defensive': 45, 'growth': 55, 'cyclical': 35, 'liquidity': 65, 'risk': 50,
        'is_core': False,
    },
    "300523": {
        'sector': 'railway_tech',
        'name': '凯发电气',
        'type': 'equipment',
        'defensive': 30, 'growth': 60, 'cyclical': 50, 'liquidity': 50, 'risk': 65,
        'is_core': False,
    },
    "300678": {
        'sector': 'railway_tech',
        'name': '中科信息',
        'type': 'software',
        'defensive': 35, 'growth': 65, 'cyclical': 40, 'liquidity': 55, 'risk': 60,
        'is_core': False,
    },
    "688682": {
        'sector': 'semiconductor',
        'name': '成都华微',
        'type': 'chip',
        'defensive': 25, 'growth': 75, 'cyclical': 55, 'liquidity': 45, 'risk': 75,
        'is_core': False,
    },
    "300011": {
        'sector': 'railway_tech',
        'name': '鼎汉技术',
        'type': 'equipment',
        'defensive': 30, 'growth': 55, 'cyclical': 55, 'liquidity': 50, 'risk': 65,
        'is_core': False,
    },
    "600528": {
        'sector': 'railway_tech',
        'name': '中铁工业',
        'type': 'equipment',
        'defensive': 50, 'growth': 35, 'cyclical': 60, 'liquidity': 70, 'risk': 45,
        'is_core': False,
    },
    "600458": {
        'sector': 'railway_tech',
        'name': '时代电气',
        'type': 'equipment',
        'defensive': 40, 'growth': 50, 'cyclical': 45, 'liquidity': 75, 'risk': 50,
        'is_core': False,
    },
    "002920": {
        'sector': 'consumer_electronics',
        'name': '德赛西威',
        'type': 'auto_electronics',
        'defensive': 35, 'growth': 70, 'cyclical': 40, 'liquidity': 75, 'risk': 55,
        'is_core': False,
    },
    "600150": {
        'sector': 'industrial',
        'name': '中国船舶',
        'type': 'equipment',
        'defensive': 45, 'growth': 40, 'cyclical': 70, 'liquidity': 80, 'risk': 50,
        'is_core': False,
    },
    "688611": {
        'sector': 'new_energy',
        'name': '杭州柯林',
        'type': 'equipment',
        'defensive': 30, 'growth': 65, 'cyclical': 50, 'liquidity': 45, 'risk': 70,
        'is_core': False,
    },
    "300098": {
        'sector': 'railway_tech',
        'name': '高新兴',
        'type': 'iot',
        'defensive': 25, 'growth': 60, 'cyclical': 55, 'liquidity': 55, 'risk': 70,
        'is_core': False,
    },

    # ========== 第二组：声学、智能硬件与果链网络 ==========
    "002241": {
        'sector': 'acoustics',
        'name': '歌尔股份',
        'type': 'electronics',
        'defensive': 35, 'growth': 65, 'cyclical': 45, 'liquidity': 85, 'risk': 55,
        'is_core': False,
    },
    "300893": {
        'sector': 'acoustics',
        'name': '朝阳科技',
        'type': 'electronics',
        'defensive': 30, 'growth': 65, 'cyclical': 50, 'liquidity': 45, 'risk': 70,
        'is_core': False,
    },
    "002475": {
        'sector': 'consumer_electronics',
        'name': '立讯精密',
        'type': 'electronics',
        'defensive': 35, 'growth': 70, 'cyclical': 40, 'liquidity': 90, 'risk': 50,
        'is_core': False,
    },
    "002885": {
        'sector': 'consumer_electronics',
        'name': '杰美特',
        'type': 'accessories',
        'defensive': 25, 'growth': 55, 'cyclical': 55, 'liquidity': 40, 'risk': 70,
        'is_core': False,
    },
    "002456": {
        'sector': 'consumer_electronics',
        'name': '欧菲光',
        'type': 'optics',
        'defensive': 20, 'growth': 60, 'cyclical': 60, 'liquidity': 70, 'risk': 75,
        'is_core': False,
    },
    "603936": {
        'sector': 'electronics',
        'name': '博敏电子',
        'type': 'pcb',
        'defensive': 40, 'growth': 50, 'cyclical': 50, 'liquidity': 50, 'risk': 55,
        'is_core': False,
    },
    "002138": {
        'sector': 'electronics',
        'name': '顺络电子',
        'type': 'passive',
        'defensive': 40, 'growth': 55, 'cyclical': 45, 'liquidity': 65, 'risk': 50,
        'is_core': False,
    },
    "300223": {
        'sector': 'semiconductor',
        'name': '北京君正',
        'type': 'chip',
        'defensive': 30, 'growth': 70, 'cyclical': 50, 'liquidity': 70, 'risk': 60,
        'is_core': False,
    },
    "300416": {
        'sector': 'electronics',
        'name': '苏大维格',
        'type': 'optics',
        'defensive': 25, 'growth': 65, 'cyclical': 55, 'liquidity': 45, 'risk': 70,
        'is_core': False,
    },
    "300115": {
        'sector': 'consumer_electronics',
        'name': '长信科技',
        'type': 'display',
        'defensive': 35, 'growth': 55, 'cyclical': 45, 'liquidity': 65, 'risk': 55,
        'is_core': False,
    },
    "600203": {
        'sector': 'electronics',
        'name': '福日电子',
        'type': 'oem',
        'defensive': 35, 'growth': 45, 'cyclical': 55, 'liquidity': 50, 'risk': 60,
        'is_core': False,
    },
    "300679": {
        'sector': 'electronics',
        'name': '电连技术',
        'type': 'connector',
        'defensive': 40, 'growth': 55, 'cyclical': 40, 'liquidity': 55, 'risk': 50,
        'is_core': False,
    },
    "002635": {
        'sector': 'consumer_electronics',
        'name': '恒大高新',
        'type': 'services',
        'defensive': 30, 'growth': 50, 'cyclical': 50, 'liquidity': 45, 'risk': 65,
        'is_core': False,
    },

    # ========== 第三组：创新药、生物科技与研发网络 ==========
    "300759": {
        'sector': 'pharmaceutical',
        'name': '康龙化成',
        'type': 'cxo',
        'defensive': 35, 'growth': 70, 'cyclical': 35, 'liquidity': 75, 'risk': 55,
        'is_core': False,
    },
    "300725": {
        'sector': 'pharmaceutical',
        'name': '药石科技',
        'type': 'cxo',
        'defensive': 30, 'growth': 75, 'cyclical': 40, 'liquidity': 65, 'risk': 60,
        'is_core': False,
    },
    "300347": {
        'sector': 'pharmaceutical',
        'name': '泰格医药',
        'type': 'cxo',
        'defensive': 35, 'growth': 65, 'cyclical': 35, 'liquidity': 80, 'risk': 50,
        'is_core': False,
    },
    "300558": {
        'sector': 'pharmaceutical',
        'name': '贝达药业',
        'type': 'biotech',
        'defensive': 30, 'growth': 70, 'cyclical': 40, 'liquidity': 65, 'risk': 60,
        'is_core': False,
    },
    "300009": {
        'sector': 'pharmaceutical',
        'name': '安科生物',
        'type': 'biotech',
        'defensive': 40, 'growth': 55, 'cyclical': 35, 'liquidity': 60, 'risk': 55,
        'is_core': False,
    },
    "300122": {
        'sector': 'pharmaceutical',
        'name': '智飞生物',
        'type': 'vaccine',
        'defensive': 35, 'growth': 65, 'cyclical': 40, 'liquidity': 80, 'risk': 50,
        'is_core': False,
    },
    "688180": {
        'sector': 'pharmaceutical',
        'name': '君实生物',
        'type': 'biotech',
        'defensive': 25, 'growth': 80, 'cyclical': 45, 'liquidity': 60, 'risk': 75,
        'is_core': False,
    },
    "300142": {
        'sector': 'pharmaceutical',
        'name': '沃森生物',
        'type': 'vaccine',
        'defensive': 30, 'growth': 65, 'cyclical': 45, 'liquidity': 70, 'risk': 60,
        'is_core': False,
    },
    "300255": {
        'sector': 'pharmaceutical',
        'name': '常山药业',
        'type': 'biotech',
        'defensive': 25, 'growth': 60, 'cyclical': 55, 'liquidity': 55, 'risk': 70,
        'is_core': False,
    },
    "300199": {
        'sector': 'pharmaceutical',
        'name': '翰宇药业',
        'type': 'biotech',
        'defensive': 20, 'growth': 65, 'cyclical': 60, 'liquidity': 45, 'risk': 80,
        'is_core': False,
    },
    "600276": {
        'sector': 'pharmaceutical',
        'name': '恒瑞医药',
        'type': 'biotech',
        'defensive': 45, 'growth': 60, 'cyclical': 30, 'liquidity': 90, 'risk': 45,
        'is_core': False,
    },
    "300463": {
        'sector': 'pharmaceutical',
        'name': '迈克生物',
        'type': 'ivd',
        'defensive': 40, 'growth': 50, 'cyclical': 35, 'liquidity': 60, 'risk': 55,
        'is_core': False,
    },

    # ========== 第四组：精细化工、周期与新能源材料网络 ==========
    "600309": {
        'sector': 'chemical',
        'name': '万华化学',
        'type': 'chemical',
        'defensive': 45, 'growth': 55, 'cyclical': 60, 'liquidity': 90, 'risk': 40,
        'is_core': False,
    },
    "600346": {
        'sector': 'chemical',
        'name': '恒力石化',
        'type': 'refining',
        'defensive': 40, 'growth': 45, 'cyclical': 70, 'liquidity': 80, 'risk': 50,
        'is_core': False,
    },
    "601857": {
        'sector': 'energy',
        'name': '中国石油',
        'type': 'oil',
        'defensive': 55, 'growth': 35, 'cyclical': 65, 'liquidity': 95, 'risk': 35,
        'is_core': False,
    },
    "600028": {
        'sector': 'energy',
        'name': '中国石化',
        'type': 'refining',
        'defensive': 55, 'growth': 35, 'cyclical': 65, 'liquidity': 95, 'risk': 35,
        'is_core': False,
    },
    "002407": {
        'sector': 'new_energy',
        'name': '多氟多',
        'type': 'material',
        'defensive': 25, 'growth': 70, 'cyclical': 65, 'liquidity': 65, 'risk': 70,
        'is_core': False,
    },
    "300037": {
        'sector': 'new_energy',
        'name': '新宙邦',
        'type': 'material',
        'defensive': 35, 'growth': 65, 'cyclical': 50, 'liquidity': 75, 'risk': 55,
        'is_core': False,
    },
    "600688": {
        'sector': 'chemical',
        'name': '上海石化',
        'type': 'refining',
        'defensive': 45, 'growth': 30, 'cyclical': 75, 'liquidity': 70, 'risk': 55,
        'is_core': False,
    },
    "002838": {
        'sector': 'chemical',
        'name': '道恩股份',
        'type': 'polymer',
        'defensive': 35, 'growth': 50, 'cyclical': 55, 'liquidity': 55, 'risk': 60,
        'is_core': False,
    },
    "002340": {
        'sector': 'new_energy',
        'name': '格林美',
        'type': 'recycling',
        'defensive': 30, 'growth': 60, 'cyclical': 60, 'liquidity': 80, 'risk': 55,
        'is_core': False,
    },
    "000301": {
        'sector': 'new_energy',
        'name': '东方盛虹',
        'type': 'material',
        'defensive': 25, 'growth': 65, 'cyclical': 70, 'liquidity': 70, 'risk': 65,
        'is_core': False,
    },
    "002768": {
        'sector': 'chemical',
        'name': '国恩股份',
        'type': 'material',
        'defensive': 30, 'growth': 55, 'cyclical': 55, 'liquidity': 50, 'risk': 60,
        'is_core': False,
    },
    "300054": {
        'sector': 'semiconductor',
        'name': '鼎龙股份',
        'type': 'material',
        'defensive': 30, 'growth': 70, 'cyclical': 50, 'liquidity': 60, 'risk': 60,
        'is_core': False,
    },
}

# 板块名称映射
A_STOCK_SECTOR_NAME_MAPPING = {
    'railway_tech': '轨交IT',
    'acoustics': '声学电子',
    'pharmaceutical': '创新药',
    'chemical': '化工',
    'consumer_electronics': '消费电子',
    'semiconductor': '半导体',
    'new_energy': '新能源材料',
    'electronics': '电子元器件',
    'energy': '能源',
    'industrial': '工业装备',
}

# ========== 市场指数配置 ==========
# 双指数架构：中证1000（中小盘基准）+ 创业板指（成长股基准）

A_STOCK_MARKET_INDEX = {
    'csi1000': '000852',  # 中证1000 - 中小盘基准
    'cyb': '399006',      # 创业板指 - 成长股基准
    'sh': '000001',       # 上证指数（备选）
    'sz': '399001',       # 深证成指（备选）
    'hs300': '000300',    # 沪深300（备选）
}

# 指数腾讯代码（用于腾讯财经API）
A_STOCK_INDEX_TENCENT = {
    'csi1000': 'sh000852',
    'cyb': 'sz399006',
    'sh': 'sh000001',
    'sz': 'sz399001',
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

def get_market_type(stock_code):
    """
    根据股票代码判断市场类型

    Args:
        stock_code (str): 股票代码

    Returns:
        str: 市场类型 'gem'（创业板）或 'main'（主板）
    """
    if stock_code.startswith('300') or stock_code.startswith('301'):
        return 'gem'
    if stock_code.startswith('688'):
        return 'star'
    return 'main'

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
A_STOCK_NETWORK_FEATURES_DIR = 'data/a_stock_network_features'
A_STOCK_MODELS_DIR = 'data/a_stock_models'
A_STOCK_CACHE_DAYS = 7  # 缓存有效期（天）

# ========== 统计信息 ==========

def get_stock_pool_stats():
    """获取股票池统计信息"""
    core_count = len(A_STOCK_CORE_HOLDINGS)
    extension_count = len(A_STOCK_EXTENSION_LIST)
    total_count = len(A_STOCK_TRAINING_LIST)

    # 统计板块分布
    sector_counts = {}
    for stock_code in A_STOCK_TRAINING_LIST.keys():
        sector = A_STOCK_SECTOR_MAPPING.get(stock_code, {}).get('sector', 'unknown')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # 统计市场类型
    market_counts = {'gem': 0, 'star': 0, 'main': 0}
    for stock_code in A_STOCK_TRAINING_LIST.keys():
        market_type = get_market_type(stock_code)
        market_counts[market_type] = market_counts.get(market_type, 0) + 1

    return {
        'core_count': core_count,
        'extension_count': extension_count,
        'total_count': total_count,
        'sector_counts': sector_counts,
        'market_counts': market_counts,
    }


if __name__ == '__main__':
    # 打印股票池统计信息
    stats = get_stock_pool_stats()
    print("=" * 50)
    print("A股智能分析系统 - 股票池统计")
    print("=" * 50)
    print(f"核心持仓: {stats['core_count']} 只")
    print(f"扩展股票: {stats['extension_count']} 只")
    print(f"训练股票池: {stats['total_count']} 只")
    print()
    print("板块分布:")
    for sector, count in sorted(stats['sector_counts'].items()):
        sector_name = A_STOCK_SECTOR_NAME_MAPPING.get(sector, sector)
        print(f"  - {sector_name}: {count} 只")
    print()
    print("市场类型分布:")
    market_names = {'gem': '创业板', 'star': '科创板', 'main': '主板'}
    for market, count in stats['market_counts'].items():
        print(f"  - {market_names.get(market, market)}: {count} 只")