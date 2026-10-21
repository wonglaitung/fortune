"""HMM 状态缓存 TTL：预测路径按需重训、回测路径永不重训（D67 配套）。"""

import numpy as np
import pandas as pd
import pytest

from data_services.regime_detector import RegimeDetector


class _FakeModel:
    """最小可用的 HMM 替身（避免测试里真训练）。"""

    def __init__(self, n_states=3):
        self.transmat_ = np.full((n_states, n_states), 0.9 / n_states)
        np.fill_diagonal(self.transmat_, 0.9)
        self.means_ = np.array([[0.01, 1.0, 0.0], [0.05, 1.2, 0.0], [-0.02, 2.0, 0.0]])
        self.n_states = n_states

    def predict(self, X):
        return np.arange(len(X)) % self.n_states

    def predict_proba(self, X):
        p = np.full((len(X), self.n_states), 1.0 / self.n_states)
        return p


@pytest.fixture()
def hsi_like():
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    rng = np.random.default_rng(7)
    close = 20000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "Close": close,
        "Volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=idx)


def _detector(ttl_days=30, trained_at="2020-01-01T00:00:00"):
    d = RegimeDetector(market="test_ttl", ttl_days=ttl_days)
    d.model = _FakeModel()
    d._state_mapping = {0: 0, 1: 1, 2: 2}
    d._trained_at = trained_at
    return d


def test_is_stale_without_timestamp():
    d = _detector(trained_at=None)
    assert d._is_stale() is True


def test_is_stale_respects_ttl():
    assert _detector(ttl_days=30, trained_at="2020-01-01T00:00:00")._is_stale() is True
    fresh = pd.Timestamp.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")
    assert _detector(ttl_days=30, trained_at=fresh)._is_stale() is False


def test_predict_refits_when_stale(monkeypatch, hsi_like):
    d = _detector()
    calls = []

    def fake_fit(df):
        calls.append(len(df))
        d._trained_at = pd.Timestamp.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")
        return d

    monkeypatch.setattr(d, "fit", fake_fit)
    d.predict(hsi_like)
    assert calls == [len(hsi_like)]


def test_predict_skips_refit_when_fresh(monkeypatch, hsi_like):
    fresh = pd.Timestamp.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")
    d = _detector(trained_at=fresh)
    calls = []
    monkeypatch.setattr(d, "fit", lambda df: calls.append(1))
    d.predict(hsi_like)
    assert calls == []


def test_predict_can_disable_refresh(monkeypatch, hsi_like):
    d = _detector()
    calls = []
    monkeypatch.setattr(d, "fit", lambda df: calls.append(1))
    d.predict(hsi_like, refresh_if_stale=False)
    assert calls == []


def test_backtest_path_never_refits(monkeypatch, hsi_like):
    """calculate_features(use_shift=True) 必须传 refresh_if_stale=False。"""
    d = _detector()
    seen = {}

    def fake_predict(df, refresh_if_stale=True):
        seen["refresh"] = refresh_if_stale
        return pd.DataFrame(index=df.index)

    monkeypatch.setattr(d, "predict", fake_predict)
    d.calculate_features(hsi_like.copy(), use_shift=True)
    assert seen["refresh"] is False


def test_production_path_allows_refit(monkeypatch, hsi_like):
    d = _detector()
    seen = {}

    def fake_predict(df, refresh_if_stale=True):
        seen["refresh"] = refresh_if_stale
        return pd.DataFrame(index=df.index)

    monkeypatch.setattr(d, "predict", fake_predict)
    d.calculate_features(hsi_like.copy(), use_shift=False)
    assert seen["refresh"] is True
