import numpy as np
import pandas as pd
import pytest

from scripts.ic_attribution import _norm_date, daily_ic, nw_t


def test_norm_date_handles_tz():
    s = pd.Series(['2023-06-01 00:00:00+00:00', '2023-06-02'])
    out = _norm_date(s)
    assert str(out.iloc[0]) == '2023-06-01 00:00:00'
    assert str(out.iloc[1]) == '2023-06-02 00:00:00'


def test_daily_ic_monotone():
    n = 30
    df = pd.DataFrame({
        'date': pd.Timestamp('2023-06-01'),
        'pred': np.arange(n, dtype=float),
        'ret': np.arange(n, dtype=float),
    })
    ics = daily_ic(df)[None]
    assert len(ics) == 1
    assert ics[0] == pytest.approx(1.0)


def test_daily_ic_skips_small_cross_section():
    df = pd.DataFrame({
        'date': pd.Timestamp('2023-06-01'),
        'pred': [0.1, 0.2, 0.3],
        'ret': [0.01, 0.02, 0.03],
    })
    assert daily_ic(df) == {}


def test_nw_t_positive_for_positive_mean():
    rng = np.random.RandomState(0)
    x = 0.02 + rng.normal(0, 0.01, size=200)
    assert nw_t(x, 20) > 2
