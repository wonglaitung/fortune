"""HSI 预测邮件「风险覆盖层（刹车）」section 渲染测试（D35 §十一集成）。"""

import json

import hsi_prediction as hp


def _inst():
    return hp.HSI_Predictor.__new__(hp.HSI_Predictor)


def test_section_missing_json(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, 'data_dir', str(tmp_path))
    h = _inst()._risk_overlay_html()
    assert '大跌保护' in h and '暂未生成' in h


def test_section_cash(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, 'data_dir', str(tmp_path))
    (tmp_path / 'hsi_risk_overlay_latest.json').write_text(json.dumps({
        'as_of': '2026-09-16', 'enh': 'nasdaq', 'horizon': 20, 'prob': 0.38,
        'stance': 'CASH', 'avg_exposure': 0.57, 'maxdd_sig': -0.10, 'maxdd_hold': -0.24,
        'calmar_sig': 0.71, 'calmar_hold': 0.22, 'timing_increment': True,
    }, ensure_ascii=False), encoding='utf-8')
    h = _inst()._risk_overlay_html()
    assert '空仓' in h and '刹车' in h
    assert '必须' not in h  # 只读展示，不做指令
    assert '保证赚钱' in h  # 免责（大白话）必须带


def test_section_hold(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, 'data_dir', str(tmp_path))
    (tmp_path / 'hsi_risk_overlay_latest.json').write_text(json.dumps({
        'as_of': '2026-09-16', 'enh': 'southbound', 'horizon': 20, 'prob': 0.62,
        'stance': 'HOLD', 'avg_exposure': 0.52, 'maxdd_sig': -0.097, 'maxdd_hold': -0.24,
        'calmar_sig': 0.70, 'calmar_hold': 0.22, 'timing_increment': True,
    }, ensure_ascii=False), encoding='utf-8')
    h = _inst()._risk_overlay_html()
    assert '持有恒指' in h
