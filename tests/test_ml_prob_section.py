# -*- coding: utf-8 -*-
"""ML 概率直填/校准/缺失展示 三件套测试

背景：LLM 把概率抄成 null → 展示层 or 0 误显"0% 看跌"（3968.HK 案例）。
修复：数值不经 LLM，raw→Isotonic 校准→直填；缺失显示"-"。
"""
import os

import pytest

import comprehensive_analysis as ca

CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'calibrators', 'prob_cal_20.pkl')


# ---------- calibrated_probability ----------

def test_calibrated_probability_none_raw():
    assert ca.calibrated_probability(None, 20) is None
    assert ca.calibrated_probability('bad', 20) is None


def test_calibrated_probability_fallback_raw(monkeypatch):
    monkeypatch.setattr(ca, '_load_prob_calibrator', lambda hz: None)
    ca._prob_calibrator_cache.clear()
    assert ca.calibrated_probability(0.8212, 20) == pytest.approx(0.8212)


@pytest.mark.skipif(not os.path.exists(CAL_PATH), reason='校准器文件不在仓库（data/calibrators 未入库）')
def test_calibrated_probability_real_calibrator():
    # 3968.HK 案例：raw 0.8212 → Isotonic 校准后 0.5659
    v = ca.calibrated_probability(0.8212, 20)
    assert v == pytest.approx(0.5659, abs=0.005)


# ---------- fill_calibrated_ml_probs ----------

def test_fill_calibrated_ml_probs(monkeypatch):
    monkeypatch.setattr(ca, '_load_prob_calibrator', lambda hz: None)
    ca._prob_calibrator_cache.clear()
    stock_data = {}
    th = {'3968.HK': {'predictions': {
        1: {'probability': 0.4342}, 5: {'probability': 0.7069}, 20: {'probability': 0.8212}}}}
    ca.fill_calibrated_ml_probs(stock_data, th, '3968.HK')
    assert stock_data['ml_prob_20d'] == pytest.approx(82.12, abs=0.01)
    assert stock_data['ml_prob_1d'] == pytest.approx(43.42, abs=0.01)
    assert stock_data['ml_prob_5d'] == pytest.approx(70.69, abs=0.01)


def test_fill_calibrated_ml_probs_missing_stock():
    stock_data = {'ml_prob_20d': 66.0}  # LLM 旧值
    ca.fill_calibrated_ml_probs(stock_data, {'0700.HK': {}}, '3968.HK')
    # 无预测 → 不覆盖不写键（保留原值交由展示层）
    assert stock_data['ml_prob_20d'] == 66.0


def test_fill_calibrated_ml_probs_overwrites_llm_value(monkeypatch):
    monkeypatch.setattr(ca, '_load_prob_calibrator', lambda hz: None)
    ca._prob_calibrator_cache.clear()
    stock_data = {'ml_prob_20d': 0.0}  # LLM 抄成 null→0 的病值
    th = {'3968.HK': {'predictions': {20: {'probability': 0.8212}}}}
    ca.fill_calibrated_ml_probs(stock_data, th, '3968.HK')
    assert stock_data['ml_prob_20d'] == pytest.approx(82.12, abs=0.01)


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
