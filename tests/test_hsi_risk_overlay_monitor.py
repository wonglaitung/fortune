"""HSI 风险覆盖层滚动验证/监控 冒烟测试。"""

import json

import numpy as np
import pandas as pd

import scripts.hsi_risk_overlay_monitor as mon


def test_maxdd_ir():
    r = np.array([0.01, -0.05, 0.02])
    assert mon._maxdd(r) < 0
    assert np.isfinite(mon._ir(r, 20))


def test_update_fills_realized(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, 'LEDGER', str(tmp_path / 'led.csv'))
    monkeypatch.setattr(mon, 'JSON_PATH', str(tmp_path / 'j.json'))
    idx = pd.bdate_range('2026-01-01', periods=40)
    s = pd.Series(np.linspace(100.0, 120.0, 40), index=idx)
    monkeypatch.setattr(mon, 'get_hsi_close', lambda: s)
    with open(mon.JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({'as_of': str(idx[0].date()), 'stance': 'CASH', 'prob': 0.3,
                   'exposure_flat': 0.0, 'horizon': 20}, f)
    mon.update()
    led = pd.read_csv(mon.LEDGER)
    assert len(led) == 1
    assert bool(led.loc[0, 'filled']) is True
    assert led.loc[0, 'hold_ret'] > 0          # 100 → 120
    assert led.loc[0, 'signal_ret'] < 0        # 空仓 + 成本 ⇒ 负


def test_update_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, 'LEDGER', str(tmp_path / 'led.csv'))
    monkeypatch.setattr(mon, 'JSON_PATH', str(tmp_path / 'j.json'))
    idx = pd.bdate_range('2026-01-01', periods=5)
    monkeypatch.setattr(mon, 'get_hsi_close', lambda: pd.Series(np.linspace(1, 2, 5), index=idx))
    with open(mon.JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({'as_of': str(idx[-1].date()), 'stance': 'HOLD', 'prob': 0.6,
                   'exposure_flat': 0.0, 'horizon': 20}, f)
    mon.update()
    mon.update()  # 同一 as_of 不重复追加
    assert len(pd.read_csv(mon.LEDGER)) == 1


def test_report_runs(tmp_path, monkeypatch, capsys):
    led = pd.DataFrame({
        'signal_date': ['2026-01-02', '2026-02-02', '2026-03-02'],
        'stance': ['HOLD', 'CASH', 'HOLD'], 'prob': [0.6, 0.3, 0.7],
        'exposure': [1.0, 0.0, 1.0], 'horizon': [20, 20, 20],
        'fill_date': ['2026-02-01', '2026-03-01', '2026-04-01'],
        'hsi_ret': [0.02, -0.03, 0.01], 'signal_ret': [0.02, -0.035, 0.01],
        'hold_ret': [0.02, -0.03, 0.01], 'filled': [True, True, True],
    })
    monkeypatch.setattr(mon, 'LEDGER', str(tmp_path / 'led.csv'))
    led.to_csv(mon.LEDGER, index=False)
    mon.report()
    assert '滚动' in capsys.readouterr().out
