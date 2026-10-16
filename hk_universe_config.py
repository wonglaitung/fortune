#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股小宇宙配置 —— **按 A股方式**构建（种子 + 围绕每只种子的关系网络分组）

对标 `a_stock_config.py`（4 核心 + 扩展，按"同业竞争 / 本地联动 / 产业链 / 高维节点 / 龙头"分组）：
    HK_CORE_HOLDINGS   4 只种子（题材锚）
    HK_EXTENSION_LIST  扩展，**每个组 = 围绕某只种子的一种关系**
    HK_SECTOR_MAPPING  行业/类型/评分（防御·成长·周期·流动性·风险，仿 A股）
    HK_UNIVERSE_WEIGHTS 样本权重（种子 3.0 / 扩展 1.0，仿 A股）

四只种子（覆盖 AI 平台 / 半导体 / 银行 / 新能源车）：
    0700.HK 腾讯控股 · 0981.HK 中芯国际 · 0005.HK 汇丰控股 · 1211.HK 比亚迪股份

⚠️ **诚实前提**：
  - 这是**研究对象**，非生产池（港股生产池 = `config.py` 的**行业扁平池**，结构不同）；
  - **组间边 = 人工判断**（软边、无 PIT、不可复算）⇒ 建模须显式标注，**不得当硬数据**；
  - **不暗示 alpha**：题材分区（D51）+ 关联传播（D52）在可交易方向均 **null**；
  - 目的 = **组织清晰 / 关系可解释 / 便于做关系网络研究**，不是"能赚钱"。
"""

# ========== 核心（4 只种子）==========

HK_CORE_HOLDINGS = {
    "0700.HK": "腾讯控股",    # 互联网平台 / AI 核心
    "0981.HK": "中芯国际",    # 半导体代工核心
    "0005.HK": "汇丰控股",    # 国际银行核心
    "1211.HK": "比亚迪股份",  # 新能源车核心
}

# ========== 扩展（按"围绕种子的关系"分组）==========

HK_EXTENSION_LIST = {
    # ========== 第一组：互联网平台 / AI 网络（围绕腾讯控股）==========
    # 平台同业竞争
    "9988.HK": "阿里巴巴-W",     # 电商/云/大模型平台
    "3690.HK": "美团-W",         # 本地生活平台
    "9618.HK": "京东集团-SW",    # 电商平台
    "1024.HK": "快手-W",         # 短视频/直播
    "9999.HK": "网易-S",         # 游戏/内容
    "1810.HK": "小米集团-W",     # 智能硬件/互联网
    # AI 大模型 / 自动驾驶同盟
    "9888.HK": "百度集团-SW",    # 搜索/自动驾驶/文心
    "0020.HK": "商汤-W",         # 计算机视觉大模型
    "6682.HK": "范式智能",       # 第四范式，决策类 AI
    "9660.HK": "地平线机器人",   # 自动驾驶芯片/算法
    "2533.HK": "黑芝麻智能",     # 车规 AI 芯片
    "2513.HK": "智谱",           # 大模型
    # 云 / 企业软件上下游
    "3888.HK": "金山软件",       # 办公软件/云
    "0268.HK": "金蝶国际",       # 企业云 ERP

    # ========== 第二组：半导体产业链网络（围绕中芯国际）==========
    # 代工同业
    "1347.HK": "华虹半导体",     # 特色工艺代工
    # 设备 / 封测
    "0522.HK": "ASMPT",          # 半导体封装设备
    # 芯片设计
    "1385.HK": "上海复旦",       # FPGA/安全芯片
    "2878.HK": "晶门半导体",     # 显示驱动芯片
    # 功率 / 车规
    "3898.HK": "时代电气",       # 轨交/功率半导体（IGBT）
    # 应用（光学模组）
    "2382.HK": "舜宇光学科技",   # 手机/车载光学模组

    # ========== 第三组：银行 / 金融网络（围绕汇丰控股）==========
    # 国际银行同业
    "2888.HK": "渣打集团",       # 国际银行
    "2388.HK": "中银香港",       # 香港本地银行
    "0011.HK": "恒生银行",       # 香港本地银行
    # 内银 H 股（A-H 双重上市）
    "0939.HK": "建设银行",       # 内银 H
    "1398.HK": "工商银行",       # 内银 H
    "1288.HK": "农业银行",       # 内银 H
    "3988.HK": "中国银行",       # 内银 H
    "3968.HK": "招商银行",       # 股份行 H
    "1658.HK": "邮储银行",       # 内银 H
    "3328.HK": "交通银行",       # 内银 H
    "0998.HK": "中信银行",       # 股份行 H
    "1988.HK": "民生银行",       # 股份行 H
    # 泛金融高维节点
    "2318.HK": "中国平安",       # 保险/金融科技
    "1299.HK": "友邦保险",       # 寿险龙头

    # ========== 第四组：新能源车产业链网络（围绕比亚迪）==========
    # 整车同业竞争
    "2333.HK": "长城汽车",       # 传统整车
    "0175.HK": "吉利汽车",       # 传统整车
    "9866.HK": "蔚来-SW",        # 新势力整车
    "9868.HK": "小鹏汽车-W",     # 新势力整车
    "2015.HK": "理想汽车-W",     # 新势力整车
    # 上游锂电 / 材料
    "1772.HK": "赣锋锂业",       # 锂盐
    "9696.HK": "天齐锂业",       # 锂盐
    "3800.HK": "协鑫科技",       # 光伏材料（泛新能源材料）
    # 汽车零部件
    "3606.HK": "福耀玻璃",       # 汽车玻璃
}

# ========== 全池 ==========

HK_UNIVERSE = {**HK_CORE_HOLDINGS, **HK_EXTENSION_LIST}

# ========== 关系网络视图（核心 → 关系组）==========
# 仅用于**展示"围绕种子的关系网络"**；成员真值以 HK_EXTENSION_LIST 为准
# （validate() 会校验两者一致，防漂移）。

HK_RELATION_GROUPS = {
    "0700.HK": {  # 腾讯控股
        "平台同业竞争": ["9988.HK", "3690.HK", "9618.HK", "1024.HK", "9999.HK", "1810.HK"],
        "AI 同盟": ["9888.HK", "0020.HK", "6682.HK", "9660.HK", "2533.HK", "2513.HK"],
        "云 / 软件上下游": ["3888.HK", "0268.HK"],
    },
    "0981.HK": {  # 中芯国际
        "代工同业": ["1347.HK"],
        "设备 / 封测": ["0522.HK"],
        "芯片设计": ["1385.HK", "2878.HK"],
        "功率 / 车规": ["3898.HK"],
        "应用（光学模组）": ["2382.HK"],
    },
    "0005.HK": {  # 汇丰控股
        "国际银行同业": ["2888.HK", "2388.HK", "0011.HK"],
        "内银 H 股（A-H 双重）": ["0939.HK", "1398.HK", "1288.HK", "3988.HK",
                                   "3968.HK", "1658.HK", "3328.HK", "0998.HK", "1988.HK"],
        "泛金融高维": ["2318.HK", "1299.HK"],
    },
    "1211.HK": {  # 比亚迪股份
        "整车同业竞争": ["2333.HK", "0175.HK", "9866.HK", "9868.HK", "2015.HK"],
        "上游锂电 / 材料": ["1772.HK", "9696.HK", "3800.HK"],
        "汽车零部件": ["3606.HK"],
    },
}

# ========== 样本权重（种子 3.0 / 扩展 1.0，仿 A股）==========

HK_UNIVERSE_WEIGHTS = {code: 3.0 for code in HK_CORE_HOLDINGS}
HK_UNIVERSE_WEIGHTS["default"] = 1.0


def get_sample_weight(stock_code):
    return HK_UNIVERSE_WEIGHTS.get(stock_code, HK_UNIVERSE_WEIGHTS["default"])


def is_core_holding(stock_code):
    return stock_code in HK_CORE_HOLDINGS


# ========== 行业 / 类型 / 评分（仿 a_stock_config 的 STOCK_SECTOR_MAPPING）==========
# 评分口径与 A股一致：防御/成长/周期/流动性/风险 均为 0–100

HK_UNIVERSE_SECTOR_MAPPING = {
    # ---- 核心 ----
    "0700.HK": {'sector': 'ai_platform', 'name': '腾讯控股', 'type': 'internet', 'defensive': 40, 'growth': 85, 'cyclical': 30, 'liquidity': 95, 'risk': 60, 'is_core': True},
    "0981.HK": {'sector': 'semiconductor', 'name': '中芯国际', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 80, 'risk': 75, 'is_core': True},
    "0005.HK": {'sector': 'bank', 'name': '汇丰控股', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20, 'is_core': True},
    "1211.HK": {'sector': 'new_energy', 'name': '比亚迪股份', 'type': 'new_energy', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 85, 'risk': 70, 'is_core': True},

    # ---- 第一组：互联网平台 / AI ----
    "9988.HK": {'sector': 'ai_platform', 'name': '阿里巴巴-W', 'type': 'internet', 'defensive': 35, 'growth': 85, 'cyclical': 35, 'liquidity': 90, 'risk': 65, 'is_core': False},
    "3690.HK": {'sector': 'ai_platform', 'name': '美团-W', 'type': 'internet', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 85, 'risk': 70, 'is_core': False},
    "9618.HK": {'sector': 'ai_platform', 'name': '京东集团-SW', 'type': 'internet', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 85, 'risk': 70, 'is_core': False},
    "1024.HK": {'sector': 'ai_platform', 'name': '快手-W', 'type': 'internet', 'defensive': 25, 'growth': 85, 'cyclical': 50, 'liquidity': 75, 'risk': 75, 'is_core': False},
    "9999.HK": {'sector': 'ai_platform', 'name': '网易-S', 'type': 'internet', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 80, 'risk': 65, 'is_core': False},
    "1810.HK": {'sector': 'ai_platform', 'name': '小米集团-W', 'type': 'internet', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 85, 'risk': 65, 'is_core': False},
    "9888.HK": {'sector': 'ai', 'name': '百度集团-SW', 'type': 'ai', 'defensive': 30, 'growth': 75, 'cyclical': 45, 'liquidity': 80, 'risk': 70, 'is_core': False},
    "0020.HK": {'sector': 'ai', 'name': '商汤-W', 'type': 'ai', 'defensive': 15, 'growth': 90, 'cyclical': 55, 'liquidity': 55, 'risk': 85, 'is_core': False},
    "6682.HK": {'sector': 'ai', 'name': '范式智能', 'type': 'ai', 'defensive': 20, 'growth': 90, 'cyclical': 50, 'liquidity': 60, 'risk': 85, 'is_core': False},
    "9660.HK": {'sector': 'ai', 'name': '地平线机器人', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 60, 'liquidity': 55, 'risk': 90, 'is_core': False},
    "2533.HK": {'sector': 'ai', 'name': '黑芝麻智能', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 65, 'liquidity': 50, 'risk': 90, 'is_core': False},
    "2513.HK": {'sector': 'ai', 'name': '智谱', 'type': 'ai', 'defensive': 15, 'growth': 95, 'cyclical': 50, 'liquidity': 65, 'risk': 90, 'is_core': False},
    "3888.HK": {'sector': 'software', 'name': '金山软件', 'type': 'software', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 70, 'risk': 70, 'is_core': False},
    "0268.HK": {'sector': 'software', 'name': '金蝶国际', 'type': 'software', 'defensive': 30, 'growth': 80, 'cyclical': 45, 'liquidity': 65, 'risk': 75, 'is_core': False},

    # ---- 第二组：半导体 ----
    "1347.HK": {'sector': 'semiconductor', 'name': '华虹半导体', 'type': 'semiconductor', 'defensive': 20, 'growth': 85, 'cyclical': 75, 'liquidity': 75, 'risk': 80, 'is_core': False},
    "0522.HK": {'sector': 'semiconductor', 'name': 'ASMPT', 'type': 'semiconductor', 'defensive': 25, 'growth': 75, 'cyclical': 70, 'liquidity': 70, 'risk': 70, 'is_core': False},
    "1385.HK": {'sector': 'semiconductor', 'name': '上海复旦', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 65, 'liquidity': 60, 'risk': 75, 'is_core': False},
    "2878.HK": {'sector': 'semiconductor', 'name': '晶门半导体', 'type': 'semiconductor', 'defensive': 20, 'growth': 80, 'cyclical': 70, 'liquidity': 55, 'risk': 80, 'is_core': False},
    "3898.HK": {'sector': 'semiconductor', 'name': '时代电气', 'type': 'semiconductor', 'defensive': 35, 'growth': 70, 'cyclical': 75, 'liquidity': 70, 'risk': 70, 'is_core': False},
    "2382.HK": {'sector': 'semiconductor', 'name': '舜宇光学科技', 'type': 'semiconductor', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 75, 'risk': 75, 'is_core': False},

    # ---- 第三组：银行 / 金融 ----
    "2888.HK": {'sector': 'bank', 'name': '渣打集团', 'type': 'bank', 'defensive': 85, 'growth': 35, 'cyclical': 25, 'liquidity': 70, 'risk': 30, 'is_core': False},
    "2388.HK": {'sector': 'bank', 'name': '中银香港', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 75, 'risk': 20, 'is_core': False},
    "0011.HK": {'sector': 'bank', 'name': '恒生银行', 'type': 'bank', 'defensive': 90, 'growth': 25, 'cyclical': 20, 'liquidity': 65, 'risk': 20, 'is_core': False},
    "0939.HK": {'sector': 'bank', 'name': '建设银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20, 'is_core': False},
    "1398.HK": {'sector': 'bank', 'name': '工商银行', 'type': 'bank', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20, 'is_core': False},
    "1288.HK": {'sector': 'bank', 'name': '农业银行', 'type': 'bank', 'defensive': 95, 'growth': 25, 'cyclical': 20, 'liquidity': 85, 'risk': 15, 'is_core': False},
    "3988.HK": {'sector': 'bank', 'name': '中国银行', 'type': 'bank', 'defensive': 90, 'growth': 28, 'cyclical': 20, 'liquidity': 80, 'risk': 20, 'is_core': False},
    "3968.HK": {'sector': 'bank', 'name': '招商银行', 'type': 'bank', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 80, 'risk': 25, 'is_core': False},
    "1658.HK": {'sector': 'bank', 'name': '邮储银行', 'type': 'bank', 'defensive': 92, 'growth': 28, 'cyclical': 20, 'liquidity': 70, 'risk': 20, 'is_core': False},
    "3328.HK": {'sector': 'bank', 'name': '交通银行', 'type': 'bank', 'defensive': 90, 'growth': 28, 'cyclical': 20, 'liquidity': 75, 'risk': 20, 'is_core': False},
    "0998.HK": {'sector': 'bank', 'name': '中信银行', 'type': 'bank', 'defensive': 88, 'growth': 32, 'cyclical': 22, 'liquidity': 70, 'risk': 25, 'is_core': False},
    "1988.HK": {'sector': 'bank', 'name': '民生银行', 'type': 'bank', 'defensive': 85, 'growth': 30, 'cyclical': 25, 'liquidity': 65, 'risk': 30, 'is_core': False},
    "2318.HK": {'sector': 'insurance', 'name': '中国平安', 'type': 'insurance', 'defensive': 80, 'growth': 45, 'cyclical': 30, 'liquidity': 80, 'risk': 35, 'is_core': False},
    "1299.HK": {'sector': 'insurance', 'name': '友邦保险', 'type': 'insurance', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 80, 'risk': 30, 'is_core': False},

    # ---- 第四组：新能源车 ----
    "2333.HK": {'sector': 'auto', 'name': '长城汽车', 'type': 'auto', 'defensive': 25, 'growth': 75, 'cyclical': 70, 'liquidity': 75, 'risk': 75, 'is_core': False},
    "0175.HK": {'sector': 'auto', 'name': '吉利汽车', 'type': 'auto', 'defensive': 25, 'growth': 75, 'cyclical': 70, 'liquidity': 75, 'risk': 75, 'is_core': False},
    "9866.HK": {'sector': 'auto', 'name': '蔚来-SW', 'type': 'auto', 'defensive': 15, 'growth': 85, 'cyclical': 70, 'liquidity': 65, 'risk': 85, 'is_core': False},
    "9868.HK": {'sector': 'auto', 'name': '小鹏汽车-W', 'type': 'auto', 'defensive': 15, 'growth': 85, 'cyclical': 70, 'liquidity': 65, 'risk': 85, 'is_core': False},
    "2015.HK": {'sector': 'auto', 'name': '理想汽车-W', 'type': 'auto', 'defensive': 20, 'growth': 80, 'cyclical': 65, 'liquidity': 70, 'risk': 80, 'is_core': False},
    "1772.HK": {'sector': 'new_energy', 'name': '赣锋锂业', 'type': 'new_energy', 'defensive': 15, 'growth': 75, 'cyclical': 90, 'liquidity': 60, 'risk': 85, 'is_core': False},
    "9696.HK": {'sector': 'new_energy', 'name': '天齐锂业', 'type': 'new_energy', 'defensive': 15, 'growth': 70, 'cyclical': 90, 'liquidity': 55, 'risk': 90, 'is_core': False},
    "3800.HK": {'sector': 'new_energy', 'name': '协鑫科技', 'type': 'new_energy', 'defensive': 15, 'growth': 85, 'cyclical': 75, 'liquidity': 60, 'risk': 85, 'is_core': False},
    "3606.HK": {'sector': 'auto', 'name': '福耀玻璃', 'type': 'auto', 'defensive': 30, 'growth': 65, 'cyclical': 75, 'liquidity': 65, 'risk': 65, 'is_core': False},
}

HK_UNIVERSE_SECTOR_NAME_MAPPING = {
    'ai_platform': '互联网平台股',
    'semiconductor': '半导体股',
    'bank': '银行股',
    'ai': '人工智能股',
    'software': '软件股',
    'insurance': '保险股',
    'new_energy': '新能源股',
    'auto': '汽车股',
}


# ========== 自检 ==========

def get_stock_pool_stats():
    core = len(HK_CORE_HOLDINGS)
    total = len(HK_UNIVERSE)
    sectors = {}
    for m in HK_UNIVERSE_SECTOR_MAPPING.values():
        sectors[m['sector']] = sectors.get(m['sector'], 0) + 1
    return {"core_count": core, "ext_count": total - core, "total_count": total, "sectors": sectors}


def validate():
    """返回问题列表（空 = 通过）。"""
    import re
    problems = []
    seen = {}
    for grp, members in [("CORE", HK_CORE_HOLDINGS), ("EXT", HK_EXTENSION_LIST)]:
        for code in members:
            if not re.match(r"^\d{4}\.HK$", code):
                problems.append(f"非法代码格式: {code}")
            if code in seen:
                problems.append(f"重复代码: {code}（{seen[code]} 与 {grp}）")
            seen[code] = grp
    # 映射覆盖
    for code in HK_UNIVERSE:
        if code not in HK_UNIVERSE_SECTOR_MAPPING:
            problems.append(f"缺 sector 映射: {code}")
    for code in HK_UNIVERSE_SECTOR_MAPPING:
        if code not in HK_UNIVERSE:
            problems.append(f"映射里有非池内股票: {code}")
    # is_core 一致
    for code, m in HK_UNIVERSE_SECTOR_MAPPING.items():
        if m.get('is_core') != is_core_holding(code):
            problems.append(f"is_core 与核心列表不一致: {code}")
    # 关系组 ↔ 扩展列表 一致（防漂移）
    flat = [c for grp in HK_RELATION_GROUPS.values() for codes in grp.values() for c in codes]
    if set(flat) != set(HK_EXTENSION_LIST):
        only_g = set(flat) - set(HK_EXTENSION_LIST)
        only_e = set(HK_EXTENSION_LIST) - set(flat)
        problems.append(f"关系组与扩展列表不一致: 仅组内={sorted(only_g)} 仅扩展={sorted(only_e)}")
    if len(flat) != len(set(flat)):
        problems.append("关系组内成员有重复")
    return problems


def render_network_ascii():
    """打印"围绕每只种子的关系网络"（ASCII）。"""
    lines = []
    for core, groups in HK_RELATION_GROUPS.items():
        cname = HK_CORE_HOLDINGS.get(core, HK_UNIVERSE_SECTOR_MAPPING.get(core, {}).get('name', ''))
        lines.append(f"{core} {cname}")
        items = list(groups.items())
        for i, (gname, codes) in enumerate(items):
            branch = "└─" if i == len(items) - 1 else "├─"
            names = " / ".join(HK_UNIVERSE.get(c, c) for c in codes)
            lines.append(f"   {branch} {gname}: {names}")
    return "\n".join(lines)


if __name__ == "__main__":
    stats = get_stock_pool_stats()
    print("港股小宇宙（按 A股方式：种子 + 关系网络）")
    print(f"  种子 {stats['core_count']} + 扩展 {stats['ext_count']} = {stats['total_count']} 只")
    print("  行业分布:")
    for s, n in sorted(stats['sectors'].items(), key=lambda x: -x[1]):
        print(f"    {HK_UNIVERSE_SECTOR_NAME_MAPPING.get(s, s):10s} {n}")
    print("\n关系网络：")
    print(render_network_ascii())
    probs = validate()
    if probs:
        print("❌ 自检失败:")
        for p in probs:
            print("   - " + p)
    else:
        print("✅ 自检通过")
