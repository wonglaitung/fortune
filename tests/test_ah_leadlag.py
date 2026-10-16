"""D52 ② A-H 领先滞后工具（scripts/ah_leadlag_test.py）。"""

import numpy as np
import pandas as pd

from scripts.ah_leadlag_test import per_pair_corrs


def test_per_pair_corrs_detects_lead():
    n = 400
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, n)
    # H_t = base_{t-1} ⇒ A 的昨日驱动 H 今日 ⇒ A_leads_H 应接近 1
    rA = base
    rH = np.concatenate([[np.nan], base[:-1]])
    df = pd.DataFrame({"rA": rA, "rH": rH, "premium_pit": rng.normal(0, 0.01, n)})
    c = per_pair_corrs(df)
    assert np.isfinite(c["contemp"])
    assert c["A_leads_H"] > 0.5
    assert abs(c["H_leads_A"]) < 0.3      # 反向无领先


def test_per_pair_corrs_short_returns_nan():
    df = pd.DataFrame({"rA": [0.01, 0.02], "rH": [0.0, 0.01], "premium_pit": [0.0, 0.0]})
    c = per_pair_corrs(df)
    assert np.isnan(c["contemp"])
