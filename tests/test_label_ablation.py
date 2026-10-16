"""标签口径消融（scripts/label_ablation.py）—— D58。"""

import numpy as np
import pandas as pd

from scripts.label_ablation import _rsi, make_labels


def test_rsi_bounds():
    s = pd.Series(np.linspace(100, 120, 60))
    r = _rsi(s).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    # 单调上升 ⇒ RSI 偏高
    assert r.iloc[-1] > 70


def test_make_labels_columns():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rows = []
    for c in ["A.HK", "B.HK"]:
        rows.append(pd.DataFrame({"date": idx, "code": c, "sector": "s",
                                  "fwd": np.linspace(0.01, 0.02, 30), "hsi_fwd": 0.015}))
    p = pd.concat(rows, ignore_index=True)
    out = make_labels(p)
    assert "V4 排序回归" in out and out["V4 排序回归"][1] is True
    assert "excess_ind" in p.columns and "rank" in p.columns
