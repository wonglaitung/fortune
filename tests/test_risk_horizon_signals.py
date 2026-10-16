"""多周期风险信号（ml_services/risk_horizon_signals.py）—— D55。"""

import numpy as np
import pandas as pd

from ml_services.risk_horizon_signals import (
    FEATURES, HORIZONS, build_dataset, avg_pairwise_corr, walk_forward)


def _synthetic_rets(n=800, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    mkt = rng.normal(0, 0.012, n)
    cols = {}
    for i in range(k):
        cols[f"S{i}.HK"] = 0.6 * mkt + rng.normal(0, 0.01, n)
    return pd.DataFrame(cols, index=idx)


def test_avg_pairwise_corr_bounded():
    rets = _synthetic_rets()
    c = avg_pairwise_corr(rets, window=22)
    assert len(c) > 0
    assert ((c >= -1) & (c <= 1)).all()


def test_build_dataset_has_features_and_binary_labels():
    rets = _synthetic_rets()
    df, mkt = build_dataset(rets)
    for f in FEATURES:
        assert f in df.columns
    for H in HORIZONS:
        for kind in ("big_drop", "corr_rise"):
            col = df[f"{kind}_{H}"]
            vals = col.dropna().unique()
            assert set(vals) <= {0.0, 1.0}, (kind, H, vals)
    assert len(mkt) == len(rets)


def test_walk_forward_returns_metrics():
    rets = _synthetic_rets(n=900)
    df, _ = build_dataset(rets)
    r = walk_forward(df, "corr_rise_5")
    assert r is None or {"auc", "brier", "brier_clim", "n"} <= set(r)
