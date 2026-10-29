#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签口径消融实验（D58）。

固定：56 池价格型特征 · LightGBM · 展开窗 + embargo。
只换**标签**：
  V0 绝对二分类(>0) | V1 成本阈值(>0.5%) | V2 HSI相对(>0) | V3 行业中性(>0) | V4 横截面百分位排序(回归)

评估：**逐（非重叠）测试日的横截面 Spearman IC**(score, 未来收益) → 均值 + 跨块 t；
      另报 TopK=10 毛超额（pp）。⚠️ 不用行级 z（lessons 三.29）。

用法：python3 scripts/label_ablation.py
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from config import STOCK_SECTOR_MAPPING  # noqa: E402
from scipy.stats import spearmanr, ttest_1samp  # noqa: E402

SNAP = os.path.join(ROOT, "data", "hk_market_snapshot")
HSI_PKL = os.path.join(ROOT, "data", "hk_universe_cache", "HSI.pkl")
H, LAG, STEP = 20, 1, 20
TOPK = 10
from ml_services.costs import COST_ROUND_TRIP
MIN_TRAIN_DATES = 600
FEATS = ["mom_20", "mom_60", "mom_120", "vol_20", "vol_60", "rsi_14",
         "vol_ratio", "ma_ratio_20", "ma_ratio_60", "ret_1", "ret_5"]

def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def _stock_features(df):
    c = pd.to_numeric(df["Close"], errors="coerce")
    v = pd.to_numeric(df["Volume"], errors="coerce")
    r = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["mom_20"] = c / c.shift(20) - 1
    out["mom_60"] = c / c.shift(60) - 1
    out["mom_120"] = c / c.shift(120) - 1
    out["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    out["vol_60"] = r.rolling(60).std() * np.sqrt(252)
    out["rsi_14"] = _rsi(c)
    out["vol_ratio"] = v / v.rolling(20).mean()
    out["ma_ratio_20"] = c / c.rolling(20).mean() - 1
    out["ma_ratio_60"] = c / c.rolling(60).mean() - 1
    out["ret_1"] = r
    out["ret_5"] = c / c.shift(5) - 1
    return out


def load_panel(h=H):
    rows = []
    for fp in sorted(os.listdir(SNAP)):
        if not fp.endswith(".pkl"):
            continue
        code = fp.replace(".pkl", "") + ".HK"
        df = pickle.load(open(os.path.join(SNAP, fp), "rb"))
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        f = _stock_features(df)
        c = pd.to_numeric(df["Close"], errors="coerce")
        f["fwd"] = c.shift(-(h + LAG)) / c.shift(-LAG) - 1
        f["code"] = code
        f.index.name = "date"
        rows.append(f.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["sector"] = panel["code"].map(lambda x: (STOCK_SECTOR_MAPPING.get(x) or {}).get("sector", "unknown"))
    # HSI 未来收益（V2）
    hsi = pickle.load(open(HSI_PKL, "rb"))["Close"].copy()
    hsi.index = pd.to_datetime(hsi.index)
    if getattr(hsi.index, "tz", None) is not None:
        hsi.index = hsi.index.tz_localize(None)
    hsi = hsi[~hsi.index.duplicated()].sort_index()
    hsi_fwd = (hsi.shift(-(h + LAG)) / hsi.shift(-LAG) - 1)
    hsi_fwd.index = hsi_fwd.index.normalize()
    panel["hsi_fwd"] = panel["date"].map(hsi_fwd)
    return panel


def make_labels(panel):
    """返回 {变体名: (标签列, is_regression)}。"""
    p = panel
    ind = p.groupby(["date", "sector"])["fwd"].transform("mean")
    p["excess_ind"] = p["fwd"] - ind
    p["excess_hsi"] = p["fwd"] - p["hsi_fwd"]
    p["rank"] = p.groupby("date")["fwd"].rank(pct=True)
    # LTR 用：当日排名的整档（0..4 五档 relevance）；保留 float（NaN 由 dropna 处理）
    p["grade"] = np.minimum(p["rank"] * 5, 4)
    return {
        "V0 绝对二分类": ("y_abs", "clf"),
        "V1 成本阈值": ("y_cost", "clf"),
        "V2 HSI相对": ("y_hsi", "clf"),
        "V3 行业中性": ("y_ind", "clf"),
        "V4 排序回归": ("y_rank", "reg"),
        "V5 真·LTR(lambdarank)": ("y_grade", "ltr"),
    }


def _fit_predict(Xtr, ytr, Xte, mode, groups=None):
    import lightgbm as lgb
    common = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                  subsample=0.75, colsample_bytree=0.8, random_state=42, verbose=-1)
    if mode == "ltr":
        m = lgb.LGBMRanker(objective="lambdarank", **common)
        m.fit(Xtr, ytr, group=groups)
        return m.predict(Xte)
    if mode == "reg":
        m = lgb.LGBMRegressor(**common)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    m = lgb.LGBMClassifier(**common)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def evaluate(panel, label_col, mode, h=H):
    d = panel.dropna(subset=FEATS + [label_col, "fwd"]).copy()
    dates = sorted(d["date"].unique())
    test_dates = dates[MIN_TRAIN_DATES::STEP]
    ics, excess = [], []
    for t in test_dates:
        cutoff = np.datetime64(t) - np.timedelta64(h + LAG, "D")
        tr = d[d["date"] <= cutoff]
        te = d[d["date"] == t]
        if len(tr) < 500 or len(te) < 20:
            continue
        if mode == "ltr":
            tr = tr.sort_values("date")
            groups = tr.groupby("date").size().values
            ytr = tr[label_col].astype(int).values
            sc = _fit_predict(tr[FEATS].values, ytr, te[FEATS].values, "ltr", groups=groups)
        else:
            ytr = tr[label_col].values
            if mode == "clf" and len(set(ytr)) < 2:
                continue
            sc = _fit_predict(tr[FEATS].values, ytr, te[FEATS].values, mode)
        ic = spearmanr(sc, te["fwd"].values).correlation
        if np.isfinite(ic):
            ics.append(ic)
        order = np.argsort(-sc)
        top = te["fwd"].values[order[:TOPK]]
        excess.append(float(top.mean() - te["fwd"].mean()))
    ics = np.asarray(ics)
    if len(ics) < 5:
        return None
    t_, p_ = ttest_1samp(ics, 0.0)
    h2 = len(ics) // 2
    return {"n": len(ics), "ic": float(ics.mean()), "t": float(t_), "p": float(p_),
            "excess_pp": float(np.mean(excess) * 100),
            "ic_h1": float(ics[:h2].mean()), "ic_h2": float(ics[h2:].mean())}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="标签口径消融（D58）")
    ap.add_argument("--horizon", type=int, default=20, help="预测周期（交易日）")
    args = ap.parse_args()
    h = args.horizon
    print(f"标签口径消融（D58）：固定特征+LightGBM，只换标签 · H={h}d")
    panel = load_panel(h)
    print(f"面板: {panel['code'].nunique()} 只 · {panel['date'].nunique()} 日 · "
          f"{panel['date'].min().date()}~{panel['date'].max().date()}")
    # 先生成中间量(excess_*/rank)，再定义各变体标签
    variants = make_labels(panel)
    panel["y_abs"] = (panel["fwd"] > 0).astype(int)
    panel["y_cost"] = (panel["fwd"] > COST_ROUND_TRIP).astype(int)
    panel["y_hsi"] = (panel["excess_hsi"] > 0).astype(int)
    panel["y_ind"] = (panel["excess_ind"] > 0).astype(int)
    panel["y_rank"] = panel["rank"]
    panel["y_grade"] = panel["grade"]
    print("=" * 92)
    print(f"{'变体':<22}{'块数':>6}{'IC均值':>9}{'t':>7}{'p':>8}{'前1/2':>9}{'后1/2':>9}{'TopK毛超额':>11}")
    res = {}
    for name, (col, mode) in variants.items():
        r = evaluate(panel, col, mode, h)
        res[name] = r
        if r:
            print(f"{name:<22}{r['n']:>6}{r['ic']:>+9.3f}{r['t']:>+7.2f}{r['p']:>8.3f}"
                  f"{r['ic_h1']:>+9.3f}{r['ic_h2']:>+9.3f}{r['excess_pp']:>+10.2f}%")
        else:
            print(f"{name:<22}  样本不足")
    print("-" * 78)
    print("判据(D58 §四): 变体 IC(fold聚类) > 基线 V0 且显著；Bonferroni over 5")
    print("⚠️ 先验偏 null；此为'只换标签/目标函数'的受控比较，非生产口径。")
    return res


if __name__ == "__main__":
    main()
