"""用户规则回测（scripts/m200_sigma_rule_backtest.py）—— 冒烟测试。"""

import numpy as np
import pandas as pd

from scripts.m200_sigma_rule_backtest import simulate


def _trend_series(n=400):
    idx = pd.bdate_range("2015-01-01", periods=n)
    base = np.linspace(100.0, 200.0, n)
    noise = np.sin(np.arange(n) / 5.0) * 3.0
    return pd.Series(base + noise, index=idx)


def test_simulate_runs_and_reports(capsys):
    simulate(_trend_series(), 1.0, "test")
    out = capsys.readouterr().out
    assert "[test]" in out
    assert ("交易" in out) or ("无交易" in out)


def test_simulate_no_entry_on_flat(capsys):
    idx = pd.bdate_range("2015-01-01", periods=400)
    close = pd.Series(np.full(400, 100.0), index=idx)      # 平线 ⇒ 无上穿 ⇒ 无交易
    simulate(close, 1.0, "flat")
    assert "无交易" in capsys.readouterr().out
