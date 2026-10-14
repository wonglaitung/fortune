# -*- coding: utf-8 -*-
"""config 映射表完整性（lessons 三.74）

触发：2026-10-14 逐只核对名称时发现 `STOCK_SECTOR_MAPPING` 里 **2269.HK 出现两次**
（药明生物 / 药明康德），后者**静默覆盖**前者 —— Python dict 字面量重复键不报错。
同类错误此前也踩过：默认值多处定义（O5）、中性化两份实现（三.74）。
本文件把「配置表不得自相矛盾」变成可执行断言。
"""

import ast
import collections
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SRC = ROOT / 'config.py'
DICT_NAMES = ['STOCK_SECTOR_MAPPING', 'WATCHLIST', 'TRAINING_STOCKS']


def _dict_literals():
    tree = ast.parse(CONFIG_SRC.read_text(encoding='utf-8'))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in DICT_NAMES:
                    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                    out[t.id] = keys
    return out


def test_no_duplicate_keys_in_config_dicts():
    """源码级：配置表字面量不得有重复键（重复键会静默覆盖，运行时无感）。"""
    for name, keys in _dict_literals().items():
        dup = [k for k, c in collections.Counter(keys).items() if c > 1]
        assert not dup, f"{name} 存在重复键：{dup}"


def test_every_sector_has_display_name():
    """每只股票的 sector 必须能在 SECTOR_NAME_MAPPING 里找到中文名。"""
    known = set(config.SECTOR_NAME_MAPPING)
    bad = {c: m.get('sector') for c, m in config.STOCK_SECTOR_MAPPING.items()
           if m.get('sector') not in known}
    assert not bad, f"sector 无显示名：{bad}"


def test_pool_codes_all_have_sector_mapping():
    """WATCHLIST / TRAINING_STOCKS 的每只都必须有板块映射。"""
    have = set(config.STOCK_SECTOR_MAPPING)
    miss = sorted((set(config.WATCHLIST) | set(config.TRAINING_STOCKS)) - have)
    assert not miss, f"池内股票缺板块映射：{miss}"


def test_training_includes_watchlist():
    assert set(config.WATCHLIST) <= set(config.TRAINING_STOCKS)
