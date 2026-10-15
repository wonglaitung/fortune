"""HSI 风险覆盖层（刹车原型）冒烟测试。"""

import numpy as np
import pandas as pd

from ml_services.hsi_risk_overlay import overlay, _maxdd, _ir


def _mk(tmp_path, n=120):
    dates = pd.bdate_range('2020-01-01', periods=n)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'fold': np.arange(n) // 20,
        'date': dates,
        'prob': rng.random(n),
        'pred': (rng.random(n) > 0.5).astype(int),
        'actual_return': rng.normal(0.001, 0.02, n),
    })
    p = tmp_path / 'pred.csv'
    df.to_csv(p, index=False)
    return str(p)


def test_overlay_shapes(tmp_path):
    last, wbar, rows = overlay(_mk(tmp_path), 20)
    assert 0.0 <= wbar <= 1.0
    assert len(rows) == 3
    for r in rows.values():
        assert _maxdd(r) <= 0.0
    assert 'pred' in last and 'date' in last


def test_flat_exposure_scaling(tmp_path):
    p = _mk(tmp_path)
    _, wbar0, rows0 = overlay(p, 20, exposure_flat=0.0)
    _, wbar5, rows5 = overlay(p, 20, exposure_flat=0.5)
    assert wbar5 >= wbar0  # Flat 半仓 ⇒ 平均敞口不降


def test_maxdd_and_ir():
    r = np.array([0.01, -0.05, 0.02, -0.03])
    assert _maxdd(r) < 0
    assert np.isfinite(_ir(r, 20))


def test_daily_overlay(tmp_path, monkeypatch):
    """逐日估值口径：按 20d 调仓、每日 mark；返回 3 条口径。"""
    import ml_services.hsi_risk_overlay as ov
    dates = pd.bdate_range('2020-01-01', periods=120)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'fold': np.arange(120) // 20, 'date': dates,
                       'prob': rng.random(120), 'pred': (rng.random(120) > 0.5).astype(int),
                       'actual_return': rng.normal(0, 0.02, 120)})
    p = tmp_path / 'pred.csv'
    df.to_csv(p, index=False)
    hsi = pd.Series(np.linspace(100, 130, 120), index=dates)
    monkeypatch.setattr(ov, '_hsi_daily_close', lambda: hsi)
    wbar, rows = ov.daily_overlay(str(p), 20, 0.0)
    assert 0.0 <= wbar <= 1.0
    assert len(rows) == 3
    assert '买入持有' in rows
