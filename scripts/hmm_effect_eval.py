#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMM（市场状态）效果评估（D57）。

回答三个问题：
  ① 状态**描述**：三态在观测空间是否分得开？（上涨=正收益+高量；下跌=高波动）
  ② 状态**预测**：各状态的前瞻 20d 收益/上涨率是否有别？（≈掷硬币则无信息）
  ③ 时长是否可信：模型 self-transition ⇒ 期望持续 vs **解码序列的经验持续**。

数据：yfinance `^HSI`（10y，含 Volume）；模型：缓存 `data/regime_cache/hmm_regime_model_hsi.pkl`。

⚠️ 仅**评估**（描述/叙事价值），**不是**预测信号；结论须带适用范围。

用法：python3 scripts/hmm_effect_eval.py [--period 10y]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LBL = {0: "低波震荡", 1: "上涨", 2: "高波偏弱"}


def _load_hsi(period):
    import yfinance as yf
    h = yf.Ticker("^HSI").history(period=period, auto_adjust=True, timeout=15)
    h.index = pd.to_datetime(h.index)
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    return h


def _runs(seq):
    out, cur, L = [], None, 0
    for s in seq:
        if s == cur:
            L += 1
        else:
            if cur is not None:
                out.append((cur, L))
            cur, L = s, 1
    if cur is not None:
        out.append((cur, L))
    return out


def evaluate(period="10y"):
    from data_services.regime_detector import RegimeDetector

    h = _load_hsi(period)
    det = RegimeDetector(market="hsi")
    loaded = det._load_model()
    res = det.predict(h).join(h["Close"])
    obs = det._prepare_observations(h).reindex(res.index)
    res["st"] = res["Market_Regime"].map(LBL)
    fwd = h["Close"].shift(-20) / h["Close"] - 1
    res["fwd20"] = fwd.reindex(res.index)

    out = {"period": period, "rows": int(len(h)), "model_loaded": bool(loaded),
           "separation": {}, "forward": {}, "persistence": {}}

    for s in (0, 1, 2):
        m = res["st"] == LBL[s]
        out["separation"][LBL[s]] = {
            "share_pct": round(float(m.mean()) * 100, 1),
            "return_20d": round(float(obs.loc[m, "return_20d"].mean()), 2),
            "vol_20d": round(float(obs.loc[m, "vol_20d"].mean()), 2),
            "volume_change": round(float(obs.loc[m, "volume_change"].mean()), 2),
        }
        x = res.loc[m, "fwd20"].dropna()
        out["forward"][LBL[s]] = {"mean_pct": round(float(x.mean()) * 100, 2),
                                  "up_pct": round(float((x > 0).mean()) * 100, 1), "n": int(len(x))}
    out["forward"]["全样本"] = {"mean_pct": round(float(res["fwd20"].mean()) * 100, 2)}

    T = det.model.transmat_
    seq = np.asarray(res["Market_Regime"].values)
    emp = np.zeros((3, 3))
    for a, b in zip(seq[:-1], seq[1:]):
        emp[a, b] += 1
    emp = emp / emp.sum(axis=1, keepdims=True)
    rr = _runs(seq)
    for s in (0, 1, 2):
        Ls = [L for (c, L) in rr[:-1] if c == s]
        model_dur = 1.0 / (1.0 - T[s, s])
        out["persistence"][LBL[s]] = {
            "model_self_trans": round(float(T[s, s]), 3),
            "emp_self_trans": round(float(emp[s, s]), 3),
            "model_exp_dur": round(float(model_dur), 1),
            "emp_median_dur": int(np.median(Ls)) if Ls else None,
            "emp_mean_dur": round(float(np.mean(Ls)), 1) if Ls else None,
        }

    # ④ 门控策略 vs 买入持有（成本 0.5%/边，T+1 生效）
    from ml_services.strategy_eval import daily_mark, metrics_daily
    r = h["Close"].pct_change().fillna(0.0)
    st = res["Market_Regime"].reindex(h.index)

    def _strat(w, name):
        net = daily_mark(r, w, round_trip_cost=0.005, lag=1)
        m = metrics_daily(net, horizon=1)
        return {"name": name, "sharpe": round(float(m["sharpe"]), 2),
                "cagr_pct": round(float(m["cagr"]) * 100, 1),
                "maxdd_pct": round(float(m["maxdd"]) * 100, 1),
                "turnover": int(w.diff().abs().fillna(0).sum())}

    out["gating"] = [
        _strat(pd.Series(1.0, index=h.index), "买入持有"),
        _strat((st != 2).astype(float).fillna(1.0), "避开高波偏弱(2)"),
        _strat((st == 1).astype(float).fillna(0.0), "只在上涨(1)"),
    ]
    return out


def _print(o):
    print("HMM（市场状态）效果评估（D57）")
    print(f"  HSI {o['period']} · {o['rows']} 行 · 缓存模型 loaded={o['model_loaded']}")
    print("\n[① 描述：状态分离度（观测 z 均值）]")
    print(f"  {'状态':<6}{'占比':>7}{'return_20d':>12}{'vol_20d':>9}{'volume':>9}")
    for k, v in o["separation"].items():
        print(f"  {k:<6}{v['share_pct']:>6.1f}%{v['return_20d']:>12}{v['vol_20d']:>9}{v['volume_change']:>9}")
    print("\n[② 预测：前瞻 20d 收益（≈50% 则无信息）]")
    for k, v in o["forward"].items():
        if "up_pct" in v:
            print(f"  {k:<6} 均值 {v['mean_pct']:+.2f}% | 上涨 {v['up_pct']:.1f}% | n={v['n']}")
        else:
            print(f"  {k}: 均值 {v['mean_pct']:+.2f}%")
    print("\n[③ 时长：模型 vs 经验]")
    print(f"  {'状态':<6}{'模型自转':>9}{'经验自转':>9}{'模型期望':>9}{'经验中位':>9}{'经验均值':>9}")
    for k, v in o["persistence"].items():
        print(f"  {k:<6}{v['model_self_trans']:>9}{v['emp_self_trans']:>9}"
              f"{v['model_exp_dur']:>9}{str(v['emp_median_dur']):>9}{str(v['emp_mean_dur']):>9}")
    print("\n[④ 门控策略 vs 买入持有（成本 0.5%/边，T+1）]")
    print(f"  {'策略':<16}{'夏普':>8}{'年化':>9}{'maxDD':>9}{'换手':>7}")
    for v in o["gating"]:
        print(f"  {v['name']:<16}{v['sharpe']:>+8.2f}{v['cagr_pct']:>+8.1f}%{v['maxdd_pct']:>+8.1f}%{v['turnover']:>7}")


def main():
    ap = argparse.ArgumentParser(description="HMM 市场状态效果评估")
    ap.add_argument("--period", default="10y")
    args = ap.parse_args()
    _print(evaluate(args.period))
    return 0


if __name__ == "__main__":
    sys.exit(main())
