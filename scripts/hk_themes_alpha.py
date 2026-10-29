#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股题材小宇宙 —— 题材内横截面 alpha 测试（D51 §六，先登记后跑）

口径与部署一致（20d、raw 打分、COST_ROUND_TRIP 与 WF 一致）。
分三层，**诚实标注功效**：

  A. 合并题材池（3 题材并集 ∩ 有预测）      —— 标准 `backtest`（≥20 只）
  B. 逐题材（bank/ai/chips）                 —— 放宽版（N<20，**方向参考**，极易不显著）
  C. N_eff 估计（E6 同题材平均相关推导）     —— 提醒「不显著 ≠ 没有」（D49）

⚠️ 模型在**全池 56** 训练，本测试把它**限制到题材子池**评估 = **分层检验**，
   **不是**重训题材专用模型（那会 N 更小、更不可信）。
⚠️ 判据（D51 §六）：净IR≥0.7 ∧ PBO<0.5 ∧ DSR≥0.95 ∧ 超额 95%CI 下界>0 ∧ Bonferroni over 3 题材。
   本脚本只算**净IR / 超额CI / HAC t**；PBO/DSR 用 `monthly_guardrail`。

用法：
    python3 scripts/hk_themes_alpha.py --pred output/20261015_141343_catboost_20d/prediction_analysis.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hk_themes_config as cfg  # noqa: E402
from ml_services.portfolio_backtest import load_panel, backtest, _stats  # noqa: E402
from ml_services.costs import COST_ROUND_TRIP  # noqa: E402
from ml_services.monthly_guardrail import min_detectable_ir, vs_baseline  # noqa: E402

HORIZON = 20
DEFAULT_PRED = "output/20261015_141343_catboost_20d/prediction_analysis.csv"


def theme_of(code):
    return cfg.get_theme(code)


def add_theme(panel):
    panel = panel.copy()
    panel["theme"] = panel["code"].map(theme_of)
    return panel


def relaxed_backtest(df_theme, horizon=HORIZON, topk=5, cost=COST_ROUND_TRIP):
    """逐题材放宽版回测（无『每期≥20只』硬门槛）。

    与 `portfolio_backtest.backtest` 同净口径：top_net = 毛 − 换手×成本；
    基准 = 当日该题材等权（bench）。
    """
    dates = sorted(df_theme["date"].unique())
    rb = dates[::horizon]
    rows, prev = [], set()
    for d in rb:
        g = df_theme[df_theme["date"] == d]
        n = len(g)
        if n < 4:
            continue
        g = g.sort_values("prob", ascending=False, kind="mergesort")
        k = max(1, min(topk, n // 2))
        sel = list(g["code"].head(k))
        long = g[g["code"].isin(sel)]
        cur = set(sel)
        turn = len(cur - prev) / k if prev else 1.0
        prev = cur
        rows.append(dict(date=d,
                         top_net=float(long["ret"].mean()) - turn * cost,
                         bench=float(g["ret"].mean()),
                         turnover=turn))
    return pd.DataFrame(rows)


def hac_t(x, lags=HORIZON - 1):
    try:
        from ml_services.overlap_stats import hac_t as _h
        return _h(np.asarray(x, dtype=float), lags=lags)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


def load_n_eff(pred_universe):
    """用 E6 边权（同题材平均相关）估 N_eff = N/(1+(N-1)r̄)；无 E6 则 None。"""
    edges = ROOT / "data" / "hk_themes" / "theme_graph_edges.csv"
    if not edges.exists():
        return {}
    e = pd.read_csv(edges)
    e = e[e["edge_type"] == "E6_CORR"]
    out = {}
    for t in cfg.THEMES:
        codes = [c for c in cfg.get_theme_stocks(t) if c in pred_universe]
        if len(codes) < 2:
            out[t] = None
            continue
        w = e[((e["u"].isin(codes)) & (e["v"].isin(codes)))]["weight"]
        if len(w) == 0:
            out[t] = None
            continue
        r = float(w.mean())
        n = len(codes)
        out[t] = round(n / (1 + (n - 1) * max(r, 0.0)), 2)
    return out


def run(pred_csv, topk_combined=5, topk_theme=3, out_dir=ROOT / "data" / "hk_themes"):
    panel = add_theme(load_panel(pred_csv, market="hk"))
    universe = set(panel["code"].unique())
    theme_universe = [c for c in cfg.get_all_stocks() if c in universe]

    report = {"pred_csv": pred_csv, "horizon": HORIZON, "cost": COST_ROUND_TRIP,
              "universe_theme": sorted(theme_universe), "themes": {}}

    # ---- A. 合并题材池 ----
    df_th = panel[panel["code"].isin(theme_universe)].copy()
    combined = None
    if len(df_th["code"].unique()) >= 20:
        bt = backtest(df_th, HORIZON, topk_combined, cost=COST_ROUND_TRIP)
        if len(bt) >= 5:
            st = _stats(bt["top_net"].values, HORIZON)
            sb = _stats(bt["bench"].values, HORIZON)
            ex = vs_baseline(bt["top_net"].values, bt["bench"].values, "题材池-topk − 题材池等权")
            mde = min_detectable_ir(st["n"], HORIZON)
            combined = {"n_periods": st["n"], "net_ir": st["ir"], "net_mean_pp": st["mean"] * 100,
                        "bench_ir": sb["ir"], "excess_pp": (ex or {}).get("diff", np.nan) * 100 if ex else np.nan,
                        "excess_ci_lo_pp": (ex or {}).get("ci_lo", np.nan) * 100 if ex else np.nan,
                        "excess_ci_hi_pp": (ex or {}).get("ci_hi", np.nan) * 100 if ex else np.nan,
                        "excess_p": (ex or {}).get("p", np.nan) if ex else np.nan,
                        "mde_ir": mde, "n_stocks": len(df_th["code"].unique())}
    report["combined"] = combined

    # ---- B. 逐题材 ----
    n_eff = load_n_eff(universe)
    for t in cfg.THEMES:
        codes = [c for c in cfg.get_theme_stocks(t) if c in universe]
        dt = panel[panel["code"].isin(codes)].copy()
        entry = {"n_stocks_pred": len(codes), "n_eff_est": n_eff.get(t),
                 "missing": [c for c in cfg.get_theme_stocks(t) if c not in universe]}
        bt = relaxed_backtest(dt, HORIZON, topk_theme, COST_ROUND_TRIP) if len(codes) >= 4 else pd.DataFrame()
        if len(bt) >= 5:
            st = _stats(bt["top_net"].values, HORIZON)
            ex = vs_baseline(bt["top_net"].values, bt["bench"].values, f"{t}-topk − {t}-等权")
            m, n_h, se_h, t_h = hac_t(bt["top_net"].values - bt["bench"].values)
            entry.update({"n_periods": st["n"], "net_ir": st["ir"], "net_mean_pp": st["mean"] * 100,
                          "excess_pp": (ex or {}).get("diff", np.nan) * 100 if ex else np.nan,
                          "excess_ci_lo_pp": (ex or {}).get("ci_lo", np.nan) * 100 if ex else np.nan,
                          "excess_ci_hi_pp": (ex or {}).get("ci_hi", np.nan) * 100 if ex else np.nan,
                          "excess_p": (ex or {}).get("p", np.nan) if ex else np.nan,
                          "excess_t_hac": float(t_h) if np.isfinite(t_h) else None,
                          "mde_ir": min_detectable_ir(st["n"], HORIZON)})
        else:
            entry["note"] = "N<4 或期数<5，不可测（N 太小）"
        report["themes"][t] = entry

    _print(report)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "theme_alpha_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📁 {Path(out_dir) / 'theme_alpha_report.json'}")
    return report


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _print(r):
    print("=" * 70)
    print(f"港股题材 alpha（部署口径 20d，净成本 {COST_ROUND_TRIP*100:.2f}%/边）")
    print(f"  预测: {r['pred_csv']}")
    print(f"  题材池 ∩ 有预测: {len(r['universe_theme'])} 只")
    print("=" * 70)
    c = r["combined"]
    print("\n[A. 合并题材池]")
    if c:
        print(f"  股票 {c['n_stocks']} · 期数 {c['n_periods']}")
        print(f"  净IR {_f(c['net_ir'])}（基准 {_f(c['bench_ir'])}）· 净均 {_f(c['net_mean_pp'])}pp")
        print(f"  超额 {_f(c['excess_pp'])}pp  95%CI [{_f(c['excess_ci_lo_pp'])}, {_f(c['excess_ci_hi_pp'])}]  p={_f(c['excess_p'],3)}")
        print(f"  MDE_IR {_f(c['mde_ir'])}（窗口功效下限；低于它测不出）")
    else:
        print("  不可测（题材池不足 20 只）")
    print("\n[B. 逐题材]")
    for t, meta in cfg.THEMES.items():
        e = r["themes"][t]
        print(f"  [{meta['name']}] 有预测 {e['n_stocks_pred']} 只 · N_eff≈{e['n_eff_est']}")
        if "note" in e:
            print(f"     {e['note']}")
        else:
            print(f"     净IR {_f(e['net_ir'])} · 净均 {_f(e['net_mean_pp'])}pp · "
                  f"超额 {_f(e['excess_pp'])}pp 95%CI[{_f(e['excess_ci_lo_pp'])},{_f(e['excess_ci_hi_pp'])}] "
                  f"HAC t={_f(e['excess_t_hac'])} · MDE_IR {_f(e['mde_ir'])}")
            if e.get("missing"):
                print(f"     ⚠️ 宇宙内 {len(e['missing'])} 只无预测（不在 56 池）: {', '.join(e['missing'])}")
    print("\n判据(D51 §六): 净IR≥0.7 ∧ PBO<0.5 ∧ DSR≥0.95 ∧ 超额CI下界>0 ∧ Bonferroni over 3 题材")
    print("⚠️ N 小 ⇒ 功效极低（MDE_IR 高）：不显著 ≠ 没有（D49）")


def main():
    ap = argparse.ArgumentParser(description="港股题材 alpha 测试（D51 §六）")
    ap.add_argument("--pred", default=DEFAULT_PRED or (os.environ.get("PRED_CSV", "")), help="prediction_analysis.csv")
    ap.add_argument("--topk-combined", type=int, default=5)
    ap.add_argument("--topk-theme", type=int, default=3)
    args = ap.parse_args()
    if not os.path.exists(args.pred):
        print(f"❌ 找不到 {args.pred}")
        return 1
    run(args.pred, topk_combined=args.topk_combined, topk_theme=args.topk_theme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
