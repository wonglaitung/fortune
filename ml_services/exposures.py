# -*- coding: utf-8 -*-
"""PIT 交易成本所需暴露：ADV20、σ20、Corwin-Schultz 价差（快照）。"""
import os

import numpy as np
import pandas as pd


def _norm(idx):
    idx = pd.to_datetime(idx)
    return idx.tz_convert(None) if idx.tz is not None else idx


def corwin_schultz(high, low):
    """日线买卖价差估计（比例，0..0.2）。用 t 与 t-1，避免前视。"""
    h, l = high.astype(float), low.astype(float)
    h1, l1 = h.shift(1), l.shift(1)
    beta = np.log(h / l) ** 2 + np.log(h1 / l1) ** 2
    hh = np.maximum(h, h1)
    ll = np.minimum(l, l1)
    gamma = np.log(hh / ll) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0.0, upper=0.2)


def load_exposures(codes, snapshot='data/hk_broad/snapshot'):
    """返回 (adv20, sigma20, spread)，均为 date×code 的 PIT 表（shift 1）。"""
    adv, sig, spr = {}, {}, {}
    for c in codes:
        p = os.path.join(snapshot, c.split('.')[0].zfill(5) + '.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        idx = _norm(df.index).normalize()
        df.index = idx
        amt = (df['Close'] * df['Volume']).rolling(20, min_periods=10).mean()
        ret = df['Close'].pct_change()
        adv[c] = amt
        sig[c] = ret.rolling(20, min_periods=10).std()
        from ml_services.costs import tick_spread
        spr[c] = pd.Series([tick_spread(x) for x in df['Close'].values], index=df.index)
    return (pd.DataFrame(adv).sort_index().shift(1),
            pd.DataFrame(sig).sort_index().shift(1),
            pd.DataFrame(spr).sort_index().shift(1))


def lookup(table, date, code):
    try:
        v = table.at[pd.Timestamp(date).normalize(), code]
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan
