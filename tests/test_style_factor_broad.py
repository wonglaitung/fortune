"""D54 换宇宙风格因子工具（scripts/style_factor_broad.py）。"""

import pickle

import numpy as np
import pandas as pd

from scripts.style_factor_broad import CLIP, STYLES, _panel, _read_close


def _frame(vals):
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.DataFrame({"Close": vals}, index=idx)


def test_panel_clips_extremes(tmp_path):
    fp = tmp_path / "a.pkl"
    pickle.dump(_frame([10.0, 10.0, 1000.0, 10.0, 10.0]), open(fp, "wb"))  # 脏价 +9900%
    s = _read_close(fp)
    rets = _panel({"A.HK": s})
    mx = rets.abs().max().max()
    assert np.isfinite(mx)
    assert mx <= CLIP + 1e-9


def test_panel_no_inf(tmp_path):
    fp = tmp_path / "b.pkl"
    pickle.dump(_frame([10.0, 0.0, 10.0, 10.0]), open(fp, "wb"))  # 0 价
    s = _read_close(fp)
    rets = _panel({"B.HK": s})
    assert not np.isinf(rets.values).any()


def test_read_close_drops_nonpositive(tmp_path):
    fp = tmp_path / "c.pkl"
    pickle.dump(_frame([10.0, 0.0, np.nan, 11.0]), open(fp, "wb"))
    s = _read_close(fp)
    assert s.iloc[0] == 10.0
    assert s.iloc[1:3].isna().all()          # 0 与 NaN 被剔
    assert s.iloc[3] == 11.0


def test_style_list():
    assert "mom_60" in STYLES and "rev_5" in STYLES
