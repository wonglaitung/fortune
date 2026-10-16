"""风险面板滚动校准监控（scripts/risk_panel_monitor.py）—— D55/D56。"""

import csv

import scripts.risk_panel_monitor as m


def test_verdict_thresholds():
    assert m._verdict(0.60) == "OK"
    assert m._verdict(0.55) == "WARN"      # 边界：>0.55 才 OK
    assert m._verdict(0.52) == "WARN"
    assert m._verdict(0.50) == "FAIL"      # 边界：>0.50 才 WARN
    assert m._verdict(float("nan")) == "NODATA"


def test_fields_and_series_scope():
    # 波动轴只 1/5D；同步轴 1/5/10/20
    series = [f"{k}_{h}" for k, h in m.SERIES]
    assert "big_move_1" in series and "big_move_5" in series
    assert "big_move_10" not in series and "big_move_20" not in series
    assert {"corr_rise_1", "corr_rise_5", "corr_rise_10", "corr_rise_20"} <= set(series)
    assert "run_date" in m.FIELDS and "verdict" in m.FIELDS


def test_update_writes_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LEDGER", tmp_path / "led.csv")
    rows = [{"series": "corr_rise_1", "kind": "corr_rise", "H": 1, "n": 100,
             "auc": 0.88, "brier": 0.15, "skill": 0.3, "recent_auc": 0.9,
             "recent_brier": 0.14, "recent_skill": 0.35, "verdict": "OK"}]
    monkeypatch.setattr(m, "evaluate", lambda *a, **k: rows)
    m.update()
    with (tmp_path / "led.csv").open(encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 1
    assert got[0]["run_date"]           # 非空
    assert got[0]["series"] == "corr_rise_1"
    assert got[0]["verdict"] == "OK"
