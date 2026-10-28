import numpy as np
import pandas as pd
import pytest

from ml_services.ml_trading_model import TradingModel


def _fit(tmp_path, monkeypatch, seeds):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LGBM_BAG_SEEDS', seeds)
    monkeypatch.setenv('LGBM_N_JOBS', '1')
    rng = np.random.RandomState(0)
    X = rng.rand(300, 2)
    y = (X[:, 0] + 0.1 * rng.rand(300)).clip(0, 1)
    m = TradingModel()
    m.feature_columns = ['a', 'b']
    m.categorical_encoders = {}
    m.label_mode = 'rank'
    m.community_ids = []
    m._train_lightgbm(None, False, None, 20, X, y)
    return m, X


def test_default_single_model(tmp_path, monkeypatch):
    m, _ = _fit(tmp_path, monkeypatch, '42')
    assert len(m.models) == 1
    assert m.model is m.models[0]


def test_bagging_averages_predictions(tmp_path, monkeypatch):
    m, X = _fit(tmp_path, monkeypatch, '1,2')
    assert len(m.models) == 2
    df = pd.DataFrame(X, columns=['a', 'b'])
    singles = [float(mod.predict(df)[0]) for mod in m.models]
    expected = float(np.clip(np.mean(singles), 0.0, 1.0))
    got = float(m.predict_proba(df)[0, 1])
    assert got == pytest.approx(expected, abs=1e-6)


def test_bagging_save_load_keeps_models(tmp_path, monkeypatch):
    m, X = _fit(tmp_path, monkeypatch, '1,2')
    p = str(tmp_path / 'm.pkl')
    m.save_model(p)
    m2 = TradingModel()
    m2.load_model(p)
    assert len(m2.models) == 2
    df = pd.DataFrame(X, columns=['a', 'b'])
    assert np.allclose(m.predict_proba(df), m2.predict_proba(df))
