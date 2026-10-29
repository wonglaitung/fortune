#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略评估的**唯一实现**（D45）：逐日 mark-to-market + 静态等仓位对照 + 功效/MDE。

教训来源：
- **D32 / D35 §十二**：降回撤必须用 **逐日 mark-to-market** + **静态等仓位对照**，
  否则会把"少投"误当"择时"（lessons 三.76 / 三.78）。
- **lessons 三.75**：功效 / MDE 必须随结论同报（"不显著" ≠ "没有"）。

所有**时序 / 择时类**策略评估一律 import 本模块，禁各自再写一份（口径漂移 = 三.73/三.74 同源）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_mark(asset_returns: pd.Series, w, round_trip_cost: float = 0.0, lag: int = 1) -> pd.Series:
    """逐日 mark-to-market：仓位 `w`（T 决定）T+lag 生效，扣换手成本。

    - 不重叠期收益的稀疏净值会**低估**日内回撤、虚报风控（lessons 三.78）——一律用本函数逐日记。
    - `w` 可为常数（静态）或序列（择时）。
    """
    w = pd.Series(w) if np.isscalar(w) else pd.Series(w)
    w = w.reindex(asset_returns.index).fillna(0.0)
    w_pos = w.shift(lag).fillna(0.0)
    turn = (w_pos - w_pos.shift(1)).abs()
    if len(turn):
        turn.iloc[0] = abs(w_pos.iloc[0])
    # D97：|Δw| 是单边变化；买+卖合计才是一次往返 ⇒ 乘 0.5
    return w_pos * asset_returns - 0.5 * turn * round_trip_cost


def static_matched(asset_returns: pd.Series, exposure: float) -> pd.Series:
    """静态等仓位对照：恒定 exposure、无择时（纯减仓）。

    '少投' 的基准。信号若**不比它更浅**回撤，则"降回撤"只是少投（D32/D35 §十二）。
    """
    return float(exposure) * asset_returns


def metrics_daily(net: pd.Series, horizon: int = 1) -> dict:
    """净值指标（CAGR/波动/夏普/maxDD/Calmar）。maxDD 为负。

    - `horizon=1`：`net` 为**逐日**收益（默认）；
    - `horizon=h>1`：`net` 为**非重叠 h 日**收益 ⇒ 年化因子 sqrt(252/h)（兼容旧 `_ir(r,h)` 口径）。
    """
    net = pd.Series(net).dropna()
    if len(net) == 0 or float(net.std()) == 0.0:
        nan = float('nan')
        return dict(n=len(net), cagr=nan, vol=nan, sharpe=nan, maxdd=nan, calmar=nan)
    factor = TRADING_DAYS / horizon
    cum = (1.0 + net).cumprod()
    cagr = float(cum.iloc[-1] ** (factor / len(net)) - 1.0)
    vol = float(net.std(ddof=1) * np.sqrt(factor))
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(factor)) if vol else float('nan')
    dd = cum / cum.cummax() - 1.0
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd else float('nan')
    return dict(n=len(net), cagr=cagr, vol=vol, sharpe=sharpe, maxdd=maxdd, calmar=calmar)


def mde_ir(n: int, horizon: int) -> float:
    """本窗口（n 个非重叠期）最小可探测「超额 IR」(α=0.05, 功率80% ⇒ z≈2.8)。"""
    return 2.8 * np.sqrt(TRADING_DAYS / (horizon * n)) if n else float('nan')


def has_timing_increment(net_sig: pd.Series, net_static: pd.Series, tol: float = 0.005) -> bool:
    """信号 maxDD 是否**浅于**静态等仓位（真会躲）？否则只是少投。"""
    return metrics_daily(net_sig)['maxdd'] > metrics_daily(net_static)['maxdd'] + tol
