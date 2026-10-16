"""港股题材 alpha 工具（scripts/hk_themes_alpha.py）—— D51。"""

import numpy as np
import pandas as pd

import hk_themes_config as cfg
from scripts.hk_themes_alpha import relaxed_backtest, theme_of


def test_theme_of_maps_and_unknown():
    assert theme_of("0700.HK") == "ai"
    assert theme_of("0981.HK") == "chips"
    assert theme_of("0001.HK") is None


def test_relaxed_backtest_net_calc():
    dates = pd.bdate_range("2020-01-01", periods=40)
    rows = []
    for i, d in enumerate(dates):
        for c in ["A", "B", "C", "D"]:
            rows.append({"date": d, "code": c,
                         "prob": 0.5 + 0.1 * (c == "A"),
                         "ret": 0.01 if c == "A" else 0.0})
    df = pd.DataFrame(rows)
    bt = relaxed_backtest(df, horizon=20, topk=1, cost=0.0)
    assert len(bt) >= 1
    # 零成本、A 始终最高 → top_net = A 的收益 = 0.01，bench = 全池均值 = 0.0025
    assert np.isclose(bt["top_net"].iloc[0], 0.01, atol=1e-9)
    assert np.isclose(bt["bench"].iloc[0], 0.0025, atol=1e-9)


def test_relaxed_backtest_skips_tiny():
    dates = pd.bdate_range("2020-01-01", periods=40)
    rows = [{"date": d, "code": c, "prob": 0.5, "ret": 0.0}
            for d in dates for c in ["A", "B", "C"]]   # 每期仅 3 只 < 4
    bt = relaxed_backtest(pd.DataFrame(rows), horizon=20, topk=1, cost=0.0)
    assert len(bt) == 0


def test_theme_pool_config_nonempty():
    assert cfg.get_theme_stocks("banks")
    assert cfg.get_theme_stocks("ai")
    assert cfg.get_theme_stocks("chips")
