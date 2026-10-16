"""风险面板（ml_services/risk_panel.py）—— D55 邮件区块（波动轴已撤，只留同步）。"""

from ml_services.risk_panel import (
    SYNC_DISPLAY, _axis_line, _hint, risk_panel_html, tier_of)

FORBIDDEN = ("加仓", "减仓", "抄底", "逃顶", "看涨", "看跌")


def test_tier_thresholds():
    assert tier_of(0.2) == "green"
    assert tier_of(0.5) == "yellow"
    assert tier_of(0.74) == "yellow"
    assert tier_of(0.75) == "red"
    assert tier_of(float("nan")) is None
    assert tier_of(None) is None


def test_axis_line_labels_tiers():
    line = _axis_line({1: {"tier": "green"}, 5: {"tier": "yellow"}}, [1, 5], "tier")
    assert "未来 1 天：正常" in line and "未来 1 周：留意" in line


def test_hint_levels():
    assert _hint({1: {"tier": "green"}, 20: {"tier": "green"}}) == ""
    assert "留意" in _hint({5: {"tier": "yellow"}})
    assert "控制仓位" in _hint({20: {"tier": "red"}})


def test_html_only_sync_no_vol_axis():
    hs = {1: {"tier": "green", "vol_tier": "green"},
          5: {"tier": "green", "vol_tier": "yellow"},
          10: {"tier": "green"}, 20: {"tier": "red"}}
    html = risk_panel_html({"as_of": "2026-10-06", "horizons": hs})
    assert "同步" in html
    assert "波动" not in html                       # 波动轴已撤
    assert "未来 1 个月：当心" in html
    assert "不含方向判断" in html


def test_html_accepts_string_keys_and_no_directional_verbs():
    hs = {"1": {"tier": "green"}, "5": {"tier": "yellow"},
          "10": {"tier": "green"}, "20": {"tier": "red"}}
    html = risk_panel_html({"as_of": "x", "horizons": hs})
    assert "未来 1 周：留意" in html
    for w in FORBIDDEN:
        assert w not in html, w


def test_sync_display_covers_all_four_horizons():
    assert SYNC_DISPLAY == [1, 5, 10, 20]


def test_html_empty_payload():
    assert "暂未生成" in risk_panel_html({})
