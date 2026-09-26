# -*- coding: utf-8 -*-
"""ML 概率直填/缺失展示 测试

背景：LLM 把概率抄成 null → 展示层 or 0 误显"0% 看跌"（3968.HK 案例）。
修复：数值不经 LLM，三周期值（DailyConfidence 已校准）直填；缺失显示"-"。
注意：Isotonic 不幂等（0.57 再校准→0.51），直填必须透传、禁止二次 transform。
"""
import os

import pytest

import comprehensive_analysis as ca


# ---------- fill_calibrated_ml_probs ----------

def test_fill_passes_through_calibrated_value():
    # three_horizon 值已由 DailyConfidence 校准（如 0.5659），直填=透传×100
    stock_data = {}
    th = {'3968.HK': {'predictions': {
        1: {'probability': 0.4102}, 5: {'probability': 0.4324}, 20: {'probability': 0.5659}}}}
    ca.fill_calibrated_ml_probs(stock_data, th, '3968.HK')
    assert stock_data['ml_prob_20d'] == pytest.approx(56.59, abs=0.01)
    assert stock_data['ml_prob_1d'] == pytest.approx(41.02, abs=0.01)
    assert stock_data['ml_prob_5d'] == pytest.approx(43.24, abs=0.01)


def test_fill_does_not_recalibrate():
    # 若二次 transform：0.5659 → 0.5124（Isotonic 不幂等）——断言透传排除
    stock_data = {}
    th = {'3968.HK': {'predictions': {20: {'probability': 0.5659}}}}
    ca.fill_calibrated_ml_probs(stock_data, th, '3968.HK')
    assert stock_data['ml_prob_20d'] == pytest.approx(56.59, abs=0.01)
    assert stock_data['ml_prob_20d'] != pytest.approx(51.24, abs=0.01)


def test_fill_calibrated_ml_probs_missing_stock():
    stock_data = {'ml_prob_20d': 66.0}  # LLM 旧值
    ca.fill_calibrated_ml_probs(stock_data, {'0700.HK': {}}, '3968.HK')
    # 无预测 → 不覆盖不写键（保留原值交由展示层）
    assert stock_data['ml_prob_20d'] == 66.0


def test_fill_calibrated_ml_probs_overwrites_llm_value():
    stock_data = {'ml_prob_20d': 0.0}  # LLM 抄成 null→0 的病值
    th = {'3968.HK': {'predictions': {20: {'probability': 0.5659}}}}
    ca.fill_calibrated_ml_probs(stock_data, th, '3968.HK')
    assert stock_data['ml_prob_20d'] == pytest.approx(56.59, abs=0.01)


# ---------- generate_stock_section_html 展示 ----------

def test_section_missing_prob_shows_dash_not_zero():
    html = ca.generate_stock_section_html({})
    assert '数据缺失（未取得当日预测）' in html
    assert '看跌，<50%硬约束禁止买入' not in html
    assert '<td class="metric-neutral">-</td>' in html
    # 三周期表方向也显示 "-"
    assert '>↑ 上涨<' not in html.split('二、三周期预测')[1].split('</table>')[0]


def test_section_with_prob_shows_calibrated_value():
    html = ca.generate_stock_section_html({
        'ml_prob_20d': 56.6, 'ml_prob_1d': 41.0, 'ml_prob_5d': 43.2})
    assert '<td>57%</td>' in html  # 20d 56.6 → 57%
    assert '中等置信度，50-60%' in html
    assert '看跌，<50%硬约束禁止买入' not in html.split('一、核心指标')[1].split('</table>')[0]
