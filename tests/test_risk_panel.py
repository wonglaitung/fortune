"""风险面板（ml_services/risk_panel.py）—— D55/D56 邮件区块。"""

from ml_services.risk_panel import (
    SYNC_DISPLAY, VOL_DISPLAY, _axis_line, _hint, risk_panel_html, tier_of)

FORBIDDEN = ("加仓", "减仓", "抄底", "逃顶", "看涨", "看跌")


def test_tier_thresholds():
    assert tier_of(0.2) == "green"
    assert tier_of(0.5) == "yellow"
    assert tier_of(0.74) == "yellow"
    assert tier_of(0.75) == "red"
    assert tier_of(float("nan")) is None
    assert tier_of(None) is None


def test_axis_line_labels_tiers():
    hs = {1: {"vol_tier": "green"}, 5: {"vol_tier": "yellow"}}
    line = _axis_line(hs, VOL_DISPLAY, "vol_tier")
    assert "未来 1 天：正常" in line and "未来 1 周：留意" in line


def test_hint_levels():
    assert _hint({1: {"vol_tier": "green"}, 5: {"tier": "green"}}) == ""
    assert "留意" in _hint({5: {"tier": "yellow"}})
    assert "控制仓位" in _hint({20: {"tier": "red"}})


def test_html_vol_before_sync_and_two_axes():
    hs = {1: {"tier": "green", "vol_tier": "green"},
          5: {"tier": "green", "vol_tier": "yellow"},
          10: {"tier": "green"},
          20: {"tier": "red"}}
    html = risk_panel_html({"as_of": "2026-10-06", "horizons": hs})
    assert html.index("波动") < html.index("同步")          # 波动在前
    assert "不含方向判断" in html
    assert "未来 1 个月：当心" in html


def test_html_accepts_string_keys_and_has_no_directional_verbs():
    # JSON 往返后键为字符串
    hs = {"1": {"tier": "green", "vol_tier": "green"},
          "5": {"tier": "yellow", "vol_tier": "green"},
          "10": {"tier": "green"}, "20": {"tier": "red"}}
    html = risk_panel_html({"as_of": "x", "horizons": hs})
    assert "未来 1 周：留意" in html
    for w in FORBIDDEN:
        assert w not in html, w


def test_html_empty_payload():
    assert "暂未生成" in risk_panel_html({})
