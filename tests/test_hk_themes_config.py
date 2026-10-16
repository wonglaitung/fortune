"""港股题材小宇宙配置（hk_themes_config.py）—— D51。"""

import re

import hk_themes_config as cfg


def test_themes_are_three():
    assert set(cfg.THEMES) == {"banks", "ai", "chips"}


def test_config_self_check_passes():
    assert cfg.validate() == []


def test_codes_are_valid_hk_format():
    for code in cfg.get_all_stocks():
        assert re.match(r"^\d{4}\.HK$", code), code


def test_helpers():
    assert cfg.get_theme("0700.HK") == "ai"
    assert cfg.get_theme("0981.HK") == "chips"
    assert cfg.get_theme("0000.HK") is None
    assert "0981.HK" in cfg.get_seed_stocks("chips")
    assert set(cfg.get_all_stocks()) >= set(cfg.get_seed_stocks())
    for theme in cfg.THEMES:
        assert cfg.get_theme_stocks(theme)


def test_ah_links_subset_universe_and_six_digit_a():
    links = cfg.get_ah_links()
    assert links
    for hk, a, name in links:
        assert hk in cfg.HK_THEMES_SECTOR_MAPPING
        assert re.match(r"^\d{6}$", a), a


def test_hard_edge_types_declared():
    assert set(cfg.HARD_EDGE_TYPES) <= set(cfg.EDGE_TYPES)


def test_normalize_hk():
    assert cfg.normalize_hk("1398") == "1398.HK"
    assert cfg.normalize_hk("01398.HK") == "1398.HK"
    assert cfg.normalize_hk("9988.HK") == "9988.HK"
    assert cfg.normalize_hk("0700") == "0700.HK"
    assert cfg.normalize_hk("abc") is None
    assert cfg.normalize_hk("") is None
