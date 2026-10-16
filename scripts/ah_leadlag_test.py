#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D52 ② —— E1 A-H 领先滞后检验（关系网络"传播"类，非横截面）

问题：同一家公司的 A 股与 H 股，谁领先谁？**可交易方向只算"能否预测 H"**
（A 的收益我们不做）。

数据：`data/ah_premium/`（D21，13 对，2006–2026，日频）。
口径：逐对算相关系数 → **按 pair 聚类**（13 个 pair）做 t 检验；
      多重检验用 Bonferroni（5 个假设）。

假设：
  0. 同期  corr(rA_t, rH_t)            （基准：同公司应高）
  1. A→H   corr(rA_{t-1}, rH_t)        （A 领先 H —— 可交易方向）
  2. H→A   corr(rH_{t-1}, rA_t)        （H 领先 A —— 非我们方向）
  3. H 动量 corr(rH_{t-1}, rH_t)
  4. 溢价反转 corr(premium_{t-1}, rH_t)

⚠️ 逐对相关 + 跨对 t = pairwise 聚类；日频重叠已由"每对一条相关系数"吸收大半。
⚠️ 只诊断"有没有传播"，不直接构成策略；过闸仍须独立预登记。

用法：
    python3 scripts/ah_leadlag_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
AH_DIR = ROOT / "data" / "ah_premium"
N_HYPOTHESES = 5


def load_pairs():
    man = json.loads((AH_DIR / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for rec in man["pairs"].values():
        f = AH_DIR / rec["file"]
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "A_cny", "H_hkd"]).sort_values("date")
        df["rA"] = df["A_cny"].pct_change()
        df["rH"] = df["H_hkd"].pct_change()
        out.append({"name": rec["name"], "a": rec["a_code"], "h": rec["h_code"], "df": df})
    return out


def per_pair_corrs(d):
    """单对：5 个相关系数。"""
    x = d.dropna(subset=["rA", "rH"])
    rA, rH = x["rA"], x["rH"]
    prem = x["premium_pit"] if "premium_pit" in x.columns else pd.Series(np.nan, index=x.index)
    def corr(a, b):
        m = pd.concat([a, b], axis=1).dropna()
        if len(m) < 60:
            return np.nan
        return float(m.iloc[:, 0].corr(m.iloc[:, 1]))
    return {
        "contemp": corr(rA, rH),
        "A_leads_H": corr(rA.shift(1), rH),
        "H_leads_A": corr(rH.shift(1), rA),
        "H_mom": corr(rH.shift(1), rH),
        "prem_rev_H": corr(prem.shift(1), rH),
    }


def run():
    pairs = load_pairs()
    rows = []
    for p in pairs:
        c = per_pair_corrs(p["df"])
        c.update({"name": p["name"], "pair": f'{p["a"]}/{p["h"]}', "n": int(p["df"]["rA"].notna().sum())})
        rows.append(c)
    res = pd.DataFrame(rows)

    print("=" * 78)
    print("D52 ② E1 A-H 领先滞后（逐对相关；按 pair 聚类）")
    print("=" * 78)
    print(f"{'pair':14s} {'name':10s} {'同期':>7s} {'A→H':>7s} {'H→A':>7s} {'H动量':>7s} {'溢价→H':>7s}")
    for _, r in res.iterrows():
        print(f"{r['pair']:14s} {str(r['name'])[:9]:10s} "
              f"{r['contemp']:+7.3f} {r['A_leads_H']:+7.3f} {r['H_leads_A']:+7.3f} "
              f"{r['H_mom']:+7.3f} {r['prem_rev_H']:+7.3f}")

    print("\n" + "-" * 78)
    print(f"{'假设':16s} {'均值相关':>9s} {'跨对 t':>9s} {'p':>9s} {'Bonferroni α=0.05/5':>22s}")
    keymap = [("同期 corr(rA,rH)", "contemp"), ("① A→H (可交易)", "A_leads_H"),
              ("② H→A", "H_leads_A"), ("③ H 动量", "H_mom"), ("④ 溢价→H", "prem_rev_H")]
    summary = {}
    for label, k in keymap:
        v = res[k].dropna().values
        if len(v) < 3:
            continue
        t, p = ttest_1samp(v, 0.0)
        sig = "✅显著" if p < 0.05 / N_HYPOTHESES else "—"
        summary[k] = {"mean": float(np.mean(v)), "t": float(t), "p": float(p), "n_pairs": int(len(v))}
        print(f"{label:16s} {np.mean(v):+9.3f} {t:+9.2f} {p:9.4f} {sig:>22s}")

    print('\n判据(D52 §三): 聚类后 |t|>1.96 且 Bonferroni 后仍显著；方向须"能预测 H"。')
    print("⚠️ 逐对相关不是策略收益；先验偏 null（网络族已进模型仍 null）。")
    return summary


def main():
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
