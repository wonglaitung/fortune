"""港股小宇宙配置（hk_universe_config.py，按 A股方式）—— D53。"""

import re

import hk_universe_config as cfg


def test_structure_and_selfcheck():
    assert cfg.validate() == []
    assert len(cfg.HK_CORE_HOLDINGS) == 4
    assert len(cfg.HK_UNIVERSE) == len(cfg.HK_CORE_HOLDINGS) + len(cfg.HK_EXTENSION_LIST)


def test_codes_valid_unique():
    assert len(set(cfg.HK_UNIVERSE)) == len(cfg.HK_UNIVERSE)
    for code in cfg.HK_UNIVERSE:
        assert re.match(r"^\d{4}\.HK$", code), code


def test_weights_reflect_core():
    for code in cfg.HK_CORE_HOLDINGS:
        assert cfg.get_sample_weight(code) == 3.0
        assert cfg.is_core_holding(code)
    for code in cfg.HK_EXTENSION_LIST:
        assert cfg.get_sample_weight(code) == 1.0
        assert not cfg.is_core_holding(code)


def test_relation_groups_match_extension():
    flat = [c for grp in cfg.HK_RELATION_GROUPS.values() for codes in grp.values() for c in codes]
    assert set(flat) == set(cfg.HK_EXTENSION_LIST)
    assert set(cfg.HK_RELATION_GROUPS) == set(cfg.HK_CORE_HOLDINGS)


def test_sector_mapping_covers_universe():
    for code in cfg.HK_UNIVERSE:
        m = cfg.HK_UNIVERSE_SECTOR_MAPPING[code]
        assert 0 <= m['growth'] <= 100 and 0 <= m['risk'] <= 100
        assert m['name']


def test_render_network_contains_cores():
    txt = cfg.render_network_ascii()
    for core, name in cfg.HK_CORE_HOLDINGS.items():
        assert core in txt and name in txt
