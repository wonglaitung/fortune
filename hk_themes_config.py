#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股题材小宇宙配置（银行 / 人工智能 / 芯片）

用 **A股建池方式（种子 + 关系扩散）的"图"版** 组织：

    第0层  种子（题材锚）        HK_THEMES_CORE
    第1/2层 扩展（按边类型)      HK_THEMES_EXTENSION   ← 仿 a_stock_config.py 的"网络分组"
    边定义                         EDGE_TYPES / HK_THEMES_AH_PAIRS / HK_THEMES_GROUPS / HK_THEMES_SUPPLY
    过滤                           HK_THEMES_FILTER
    派生映射                       HK_THEMES_SECTOR_MAPPING（由上面推导）

与 A股建池方式的差异（港股特有的两条强边）：
    E1 A-H 双重上市 —— 同公司 H↔A，一一映射（把 A股题材接回港股）
    E5 南向共同持仓 —— 港股独有资金边

⚠️ 本文件**只负责建池（研究对象）**；"这套池有没有 alpha"另论，
   须走 D49/D50/D51 的功效与三道闸，**不得由本文件暗示 edge**。
⚠️ E3 产业链 / E4 集团 为**人工判断**（无 PIT、不可复算）⇒ 只能当"仅判断"候选边，
   建模时须显式标注，不得当硬数据。
"""

import re

# ========== 题材（三个）==========

THEMES = {
    "banks": {"name": "银行", "sector": "bank"},
    "ai": {"name": "人工智能", "sector": "ai"},
    "chips": {"name": "芯片", "sector": "semiconductor"},
}

# ========== 第0层：种子（题材锚）==========

HK_THEMES_CORE = {
    "banks": {
        "0005.HK": "汇丰控股",
        "2388.HK": "中银香港",
        "0011.HK": "恒生银行",
        "2888.HK": "渣打集团",
    },
    "ai": {
        "0700.HK": "腾讯控股",
        "0020.HK": "商汤-W",
        "9888.HK": "百度集团-SW",
        "9988.HK": "阿里巴巴-W",
    },
    "chips": {
        "0981.HK": "中芯国际",
        "1347.HK": "华虹半导体",
        "1385.HK": "上海复旦",
        "2878.HK": "晶门半导体",
    },
}

# ========== 边类型 ==========

EDGE_TYPES = {
    "E1_AH_DUAL": "A-H 双重上市（同公司 H↔A，港股特有硬边）",
    "E2_PEER": "同业竞争（同题材 / 同子行业）",
    "E3_SUPPLY": "产业链上下游（人工判断）",
    "E4_GROUP": "集团 / 股权（人工判断）",
    "E5_SOUTHBOUND": "南向共同持仓（数据推导）",
    "E6_CORR": "价格相关性（可计算，需行情）",
}

# 硬边（可复算 / 有数据源）；其余为软边（人工 / 推导），建模时须区分
HARD_EDGE_TYPES = ("E1_AH_DUAL", "E5_SOUTHBOUND", "E6_CORR")

# ========== 第1/2层：扩展（按边类型分组）==========
# 结构：[题材][边类型] = {code: name}

HK_THEMES_EXTENSION = {
    "banks": {
        "E2_PEER": {
            "0939.HK": "建设银行",
            "1288.HK": "农业银行",
            "1398.HK": "工商银行",
            "3968.HK": "招商银行",
            "1658.HK": "邮储银行",
            "3328.HK": "交通银行",
            "0998.HK": "中信银行",
            "1988.HK": "民生银行",
            "3988.HK": "中国银行",
        },
    },
    "ai": {
        "E2_PEER": {
            "1024.HK": "快手-W",
            "3690.HK": "美团-W",
            "9618.HK": "京东集团-SW",
            "9999.HK": "网易-S",
            "1810.HK": "小米集团-W",
            "6682.HK": "范式智能",
            "9660.HK": "地平线机器人",
            "2533.HK": "黑芝麻智能",
            "2513.HK": "智谱",
        },
    },
    "chips": {
        "E2_PEER": {
            "2382.HK": "舜宇光学科技",
            "0522.HK": "ASMPT",
            "3898.HK": "时代电气",
        },
    },
}

# ========== E1：A-H 双重上市对（hk_code -> a_code）==========
# 仅列**题材内**的双重上市；A 代码为 6 位沪/深代码

HK_THEMES_AH_PAIRS = {
    # 银行
    "0939.HK": "601939",  # 建设银行
    "1288.HK": "601288",  # 农业银行
    "1398.HK": "601398",  # 工商银行
    "3968.HK": "600036",  # 招商银行
    "1658.HK": "601658",  # 邮储银行
    "3328.HK": "601328",  # 交通银行
    "0998.HK": "601998",  # 中信银行
    "1988.HK": "600016",  # 民生银行
    "3988.HK": "601988",  # 中国银行
    # 芯片
    "0981.HK": "688981",  # 中芯国际
    "1347.HK": "688347",  # 华虹半导体
    "1385.HK": "688385",  # 上海复旦
    "3898.HK": "688187",  # 时代电气
}

# ========== E3：产业链上下游（人工判断，无 PIT）==========
# (上游/设计, 下游/应用, 说明)

HK_THEMES_SUPPLY = [
    ("1385.HK", "0981.HK", "芯片设计 → 代工"),
    ("0981.HK", "2382.HK", "代工 → 光学/模组应用"),
    ("2878.HK", "2382.HK", "显示驱动 → 模组"),
    ("0981.HK", "9660.HK", "芯片 → 自动驾驶 SoC"),
    ("2533.HK", "9660.HK", "车规芯片 ↔ 自动驾驶（同链上下游）"),
    ("3898.HK", "1211.HK", "功率半导体 → 新能源车（跨题材）"),
]

# ========== E4：集团 / 股权（人工判断，无 PIT）==========

HK_THEMES_GROUPS = {
    "tencent_affiliated": {
        "name": "腾讯系（腾讯持股/关联）",
        "members": ["0700.HK", "3690.HK", "9618.HK", "1024.HK"],
    },
    "alibaba_affiliated": {
        "name": "阿里系",
        "members": ["9988.HK"],
    },
    "state_financial": {
        "name": "国资金融（汇金/财政部控股）",
        "members": ["0939.HK", "1288.HK", "1398.HK", "3988.HK", "1658.HK", "3328.HK", "0998.HK", "1988.HK"],
    },
}

# ========== 过滤（功效诚实：小宇宙上限）==========

HK_THEMES_FILTER = {
    "min_price": 1.0,            # 港元；剔仙股
    "exclude_gem": True,         # 剔创业板 08xxx
    "min_avg_turnover": 5_000_000.0,  # 日均成交额（港元）下限；数据到手后过滤
    "max_stocks": 60,            # 小宇宙上限 —— 控多重检验 + 提醒相关性高致 N_eff 更小
}

# ========== 派生：题材/行业映射 ==========

_HK_CODE_RE = re.compile(r"^\d{4}\.HK$")


def _iter_theme_members():
    """产出 (theme, code, name, is_core, edge_type)。"""
    for theme, members in HK_THEMES_CORE.items():
        for code, name in members.items():
            yield theme, code, name, True, "CORE"
    for theme, by_edge in HK_THEMES_EXTENSION.items():
        for edge_type, members in by_edge.items():
            for code, name in members.items():
                yield theme, code, name, False, edge_type


def build_sector_mapping():
    """由种子 + 扩展推导题材/行业映射。"""
    mapping = {}
    for theme, code, name, is_core, edge_type in _iter_theme_members():
        mapping[code] = {
            "theme": theme,
            "theme_name": THEMES[theme]["name"],
            "sector": THEMES[theme]["sector"],
            "name": name,
            "is_core": is_core,
            "edge_type": edge_type,
        }
    return mapping


HK_THEMES_SECTOR_MAPPING = build_sector_mapping()

HK_THEMES_UNIVERSE = {code: meta["name"] for code, meta in HK_THEMES_SECTOR_MAPPING.items()}


# ========== 辅助 ==========

def normalize_hk(code):
    """把 4/5 位、带不带后缀的港股代码统一为 'NNNN.HK'（4 位补零）。"""
    if not code:
        return None
    c = str(code).strip().upper().replace(".HK", "")
    if not c.isdigit():
        return None
    return f"{int(c):04d}.HK"


def get_theme(code):
    """返回股票所属题材；不在宇宙内返回 None。"""
    meta = HK_THEMES_SECTOR_MAPPING.get(code)
    return meta["theme"] if meta else None


def get_all_stocks():
    """全部宇宙股票（种子 + 扩展）。"""
    return list(HK_THEMES_SECTOR_MAPPING.keys())


def get_seed_stocks(theme=None):
    """种子（题材锚）；theme 给定则只看该题材。"""
    if theme is None:
        out = []
        for members in HK_THEMES_CORE.values():
            out.extend(members.keys())
        return out
    return list(HK_THEMES_CORE.get(theme, {}).keys())


def get_theme_stocks(theme):
    """某题材的全部股票（种子 + 扩展）。"""
    return [c for c, m in HK_THEMES_SECTOR_MAPPING.items() if m["theme"] == theme]


def get_ah_links():
    """E1：返回 (hk_code, a_code, name) 列表，仅保留在宇宙内的 H 股。"""
    out = []
    for hk, a in HK_THEMES_AH_PAIRS.items():
        if hk in HK_THEMES_SECTOR_MAPPING:
            out.append((hk, a, HK_THEMES_SECTOR_MAPPING[hk]["name"]))
    return out


def validate():
    """自检：代码格式、唯一性、种子非空。返回问题列表（空 = 通过）。"""
    problems = []
    seen = {}
    for theme, code, name, is_core, edge_type in _iter_theme_members():
        if not _HK_CODE_RE.match(code):
            problems.append(f"非法代码格式: {code} ({name})")
        if code in seen:
            problems.append(f"重复代码: {code} 同时属于 {seen[code]} 与 {theme}")
        seen[code] = theme
    for theme in THEMES:
        if not HK_THEMES_CORE.get(theme):
            problems.append(f"题材 {theme} 无种子")
    for hk in HK_THEMES_AH_PAIRS:
        if hk not in HK_THEMES_SECTOR_MAPPING:
            problems.append(f"E1 A-H 对的 H 股不在宇宙内: {hk}")
    return problems


if __name__ == "__main__":
    probs = validate()
    print("港股题材小宇宙")
    print(f"  题材: {list(THEMES.keys())}")
    print(f"  股票数: {len(HK_THEMES_UNIVERSE)}（种子 {len(get_seed_stocks())}）")
    for theme in THEMES:
        seeds = len(HK_THEMES_CORE[theme])
        total = len(get_theme_stocks(theme))
        print(f"    - {THEMES[theme]['name']}({theme}): {total} 只（种子 {seeds}）")
    print(f"  E1 A-H 对: {len(get_ah_links())}")
    print(f"  E3 产业链边: {len(HK_THEMES_SUPPLY)}")
    print(f"  E4 集团: {list(HK_THEMES_GROUPS.keys())}")
    if probs:
        print("❌ 自检失败:")
        for p in probs:
            print("   - " + p)
    else:
        print("✅ 自检通过")
