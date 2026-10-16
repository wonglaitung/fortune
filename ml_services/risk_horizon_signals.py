#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多周期风险信号（D55）：未来 H 日 大跌 / 相关齐升 的**概率预测**。

方向不可测（D33/D43），但**波动 / 相关可测**（vol clustering；HAR，Corsi 2009）。
本模块把风险预测做成 **1/5/10/20d**，**用概率评分评估**（AUC/Brier/校准）——**出口 = 风控档位，非 alpha**。

口径见 DECISIONS D55 §二（预注册）：
- 市场代理 = 56 池等权日收益（`data/hk_market_snapshot`，2014-07→）。
- 标签（PIT）：`big_drop_H = fwd_H ≤ 过去 250d 的 20 分位`；`corr_rise_H = fwd_avg_corr_H > 过去 250d 中位数`。
- 预测子：`rv_1, rv_5, rv_22` + `corr_22`。
- 模型：Logistic + 膨胀窗 walk-forward（只用过去）。

用法：python3 ml_services/risk_horizon_signals.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SNAP = os.path.join(ROOT, "data", "hk_market_snapshot")
HORIZONS = [1, 5, 10, 20]
FEATURES = ["rv_1", "rv_5", "rv_22", "corr_22"]
LOOKBACK_Q = 250          # 标签分位窗口（只用过去）
CORR_WIN = 22
MIN_TRAIN = 500
STEP = 20


# ---------------------------------------------------------------- 数据
def load_returns():
    """56 池日收益面板 + 等权市场收益。"""
    closes = {}
    for fp in sorted(os.listdir(SNAP)):
        if not fp.endswith(".pkl"):
            continue
        try:
            df = pickle.load(open(os.path.join(SNAP, fp), "rb"))
        except Exception:  # noqa: BLE001
            continue
        s = pd.to_numeric(df["Close"], errors="coerce")
        s.index = pd.to_datetime(df.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s = s.where(np.isfinite(s) & (s > 0))
        closes[fp.replace(".pkl", "") + ".HK"] = s
    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return rets


def avg_pairwise_corr(rets, window=CORR_WIN):
    """过去 window 日平均两两相关（PIT）。"""
    out = {}
    arr = rets.values
    cols = rets.columns
    idx = rets.index
    for t in range(window, len(idx) + 1):
        w = arr[t - window:t]
        # 至少 5 只、且该窗无全 NaN 列
        valid = ~np.isnan(w).all(axis=0)
        ww = w[:, valid]
        if ww.shape[1] < 5:
            continue
        c = np.corrcoef(ww, rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        vals = c[iu]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[idx[t - 1]] = float(vals.mean())
    return pd.Series(out)


def build_dataset(rets):
    """特征 + 各 H 的标签（PIT）。"""
    mkt = rets.mean(axis=1)                      # 等权市场收益
    df = pd.DataFrame(index=rets.index)
    df["rv_1"] = mkt.abs()                       # 1d 已实现波动 ≈ |收益|
    df["rv_5"] = mkt.rolling(5).std() * np.sqrt(252)
    df["rv_22"] = mkt.rolling(22).std() * np.sqrt(252)
    df["corr_22"] = avg_pairwise_corr(rets, CORR_WIN)

    for H in HORIZONS:
        # 过去 H 日和（标签阈值用）
        pastH = mkt.rolling(H).sum()
        thr = pastH.rolling(LOOKBACK_Q).quantile(0.20)          # 只用过去
        fwdH = mkt.rolling(H).sum().shift(-H)                   # 未来 H 日和
        df[f"big_drop_{H}"] = (fwdH <= thr).astype(float)
        df.loc[fwdH.isna() | thr.isna(), f"big_drop_{H}"] = np.nan

        # 相关齐升：未来 H 日平均两两相关 > 过去中位数
        fwd_corr = avg_corr_series(df).rolling(H).mean().shift(-H)
        med = avg_corr_series(df).rolling(LOOKBACK_Q).median()
        df[f"corr_rise_{H}"] = (fwd_corr > med).astype(float)
        df.loc[fwd_corr.isna() | med.isna(), f"corr_rise_{H}"] = np.nan

        # 大波动（|r|，不分方向；D56）：未来 H 日 |累计和| ≥ 过去 250d |H 日和| 的 80 分位
        thr_abs = pastH.abs().rolling(LOOKBACK_Q).quantile(0.80)
        df[f"big_move_{H}"] = (fwdH.abs() >= thr_abs).astype(float)
        df.loc[fwdH.isna() | thr_abs.isna(), f"big_move_{H}"] = np.nan
    return df, mkt


def avg_corr_series(df):
    return df["corr_22"]


# ---------------------------------------------------------------- 评估
def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def walk_forward(df, label_col):
    from sklearn.linear_model import LogisticRegression
    d = df.dropna(subset=FEATURES + [label_col]).copy()
    if len(d) < MIN_TRAIN + 60:
        return None
    X, y = d[FEATURES].values, d[label_col].values
    idx = d.index
    ys, ps = [], []
    for i in range(MIN_TRAIN, len(d), STEP):
        Xtr, ytr = X[:i], y[:i]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=1000)
        clf.fit((Xtr - mu) / sd, ytr)
        ps.append(clf.predict_proba((X[i:i + STEP] - mu) / sd)[:, 1])
        ys.append(y[i:i + STEP])
    if not ys:
        return None
    yv = np.concatenate(ys); pv = np.concatenate(ps)
    from sklearn.metrics import brier_score_loss
    base = float(yv.mean())
    return {"n": int(len(yv)), "base_rate": base, "auc": _auc(yv, pv),
            "brier": float(brier_score_loss(yv, pv)),
            "brier_clim": base * (1 - base),
            "y": yv, "p": pv}


def calibration(y, p, bins=5):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 5:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def main():
    print("多周期风险信号（D55）—— 大跌 / 相关齐升（概率预测，非方向）")
    rets = load_returns()
    df, mkt = build_dataset(rets)
    print(f"面板 {rets.shape[1]} 只 | {rets.index.min().date()} ~ {rets.index.max().date()} | 样本 {len(df)}")
    print("=" * 84)
    for kind in ("big_move", "big_drop", "corr_rise"):
        print(f"\n### {kind}（AUC>0.55 且 Brier<climatology 为过）")
        print(f"{'H':>3}{'n':>6}{'基率':>8}{'AUC':>7}{'Brier':>8}{'气候Brier':>10}{'技能':>8}{'判':>5}")
        for H in HORIZONS:
            r = walk_forward(df, f"{kind}_{H}")
            if not r:
                print(f"{H:>3}  样本不足")
                continue
            skill = 1 - r["brier"] / r["brier_clim"] if r["brier_clim"] > 0 else float("nan")
            ok = (r["auc"] > 0.55) and (r["brier"] < r["brier_clim"])
            print(f"{H:>3}{r['n']:>6}{r['base_rate']*100:>7.1f}%{r['auc']:>7.3f}{r['brier']:>8.4f}"
                  f"{r['brier_clim']:>10.4f}{skill*100:>7.1f}%{'✅' if ok else '❌':>5}")
    print("\n判据(D55 §三): 各 H 分别 AUC>0.55 ∧ Brier<气候 ∧ 校准单调；Bonferroni over 8 检验")
    print("⚠️ 波动/相关可预测 ≠ 能赚方向；出口 = 风控档位（非 alpha）")


if __name__ == "__main__":
    main()
