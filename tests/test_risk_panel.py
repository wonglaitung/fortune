"""风险面板（ml_services/risk_panel.py）—— D55 邮件区块原型。"""

import re

from ml_services.risk_panel import _compose, risk_panel_html, tier_of

FORBIDDEN = ("加仓", "减仓", "抄底", "逃顶", "买", "卖", "看涨", "看跌", "涨跌")


def test_tier_thresholds():
    assert tier_of(0.2) == "green"
    assert tier_of(0.5) == "yellow"
    assert tier_of(0.74) == "yellow"
    assert tier_of(0.75) == "red"
    assert tier_of(float("nan")) is None
    assert tier_of(None) is None


def test_compose_green_mentions_normal():
    hs = {5: {"tier": "green", "p_corr_rise": 0.3}, 20: {"tier": "green", "p_corr_rise": 0.4}}
    lead, b = _compose(hs)
    assert "正常" in lead
    assert b == []


def test_compose_red_direction_neutral():
    hs = {5: {"tier": "green"}, 20: {"tier": "red", "p_corr_rise": 0.9}}
    lead, b = _compose(hs)
    assert "当心" in lead
    # 「同步」解释里必须显式"同涨同跌"以示双向
    assert any("同涨同跌" in x for x in b)


def test_html_has_no_directional_verbs():
    hs = {1: {"tier": "green", "p_corr_rise": 0.2},
          5: {"tier": "yellow", "p_corr_rise": 0.6},
          10: {"tier": "yellow", "p_corr_rise": 0.65},
          20: {"tier": "red", "p_corr_rise": 0.8}}
    html = risk_panel_html({"as_of": "2026-10-06", "horizons": hs})
    assert "波动提示" in html
    assert "不含方向判断" in html
    # 正文不得出现方向性动词（"涨跌"仅在"（涨跌皆可能）"与免责里，允许）
    for w in ("加仓", "减仓", "抄底", "逃顶", "看涨", "看跌"):
        assert w not in html, w
    # "同涨同跌"用于说明双向 —— 允许
    assert "同涨同跌" in html


def test_html_empty_payload():
    assert "暂未生成" in risk_panel_html({})
