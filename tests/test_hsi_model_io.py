import os
import pickle

import lightgbm as lgb
import numpy as np
import pytest

import ml_services.hsi_ml_model as hsi


def _fit():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 3)
    y = (X[:, 0] > 0.5).astype(int)
    m = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    m.fit(X, y)
    return m


def test_save_load_roundtrip_by_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(hsi, 'MODEL_DIR', str(tmp_path))
    for h in (5, 10, 20):
        mdl = hsi.HSIModel(horizon=h)
        mdl.model = _fit()
        mdl.feature_names = ['a', 'b', 'c']
        mdl.save_model()

    names = os.listdir(tmp_path)
    assert any(f'hsi_lightgbm_5d_' in n for n in names)
    assert any(f'hsi_lightgbm_10d_' in n for n in names)
    assert any(f'hsi_lightgbm_20d_' in n for n in names)

    m = hsi.HSIModel(horizon=10)
    assert m.load_model() is True
    assert m.horizon == 10
    assert m.feature_names == ['a', 'b', 'c']
    assert '_10d_' in os.path.basename(m.model_path)


def test_horizon_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(hsi, 'MODEL_DIR', str(tmp_path))
    mdl = hsi.HSIModel(horizon=5)
    mdl.model = _fit()
    mdl.feature_names = ['a', 'b', 'c']
    mdl.save_model()

    assert hsi.HSIModel(horizon=20).load_model() is False


def test_rejects_legacy_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(hsi, 'MODEL_DIR', str(tmp_path))
    legacy = tmp_path / 'hsi_lightgbm_20d_20260101_000000.pkl'
    with open(legacy, 'wb') as f:
        pickle.dump(_fit(), f)

    with pytest.raises(ValueError):
        hsi.HSIModel(horizon=20).load_model()
