#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取个股“最新报告期”财务（中期优先于年报），并拆出单季、算同比。

用途：防止用旧年报下结论（e.g. 已有 2026H1 却只看 FY2025）。
数据：data/fund_cache/{code}_{利润表|现金流量表}_{中期|年度}.pkl（累计口径）。

用法：python3 scripts/latest_financials.py --codes 0700.HK,9988.HK,3690.HK
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "fund_cache")
COLS = ["营业额", "毛利", "经营溢利", "销售及分销费用", "研发费用", "股东应占溢利"]
RATIOS = [("毛利率", "毛利", "营业额"), ("经营利润率", "经营溢利", "营业额"), ("净利率", "股东应占溢利", "营业额")]


def load(code, kind, per):
    p = os.path.join(CACHE, f"{code}_{kind}_{per}.pkl")
    return pd.read_pickle(p) if os.path.exists(p) else None


def combined(code):
    """合并中期/年度（同日优先中期），返回 (利润表, 现金流) 的累计序列。"""
    is_m, is_a = load(code, "利润表", "中期"), load(code, "利润表", "年度")
    cf_m, cf_a = load(code, "现金流量表", "中期"), load(code, "现金流量表", "年度")
    def merge(m, a):
        d = {}
        if a is not None:
            for x in a.index: d[pd.Timestamp(x)] = a.loc[x]
        if m is not None:
            for x in m.index: d[pd.Timestamp(x)] = m.loc[x]
        return pd.DataFrame(d).T.sort_index() if d else pd.DataFrame()
    return merge(is_m, is_a), merge(cf_m, cf_a)


def fy_end_month(code):
    a = load(code, "利润表", "年度")
    if a is None or len(a) == 0:
        return 12
    return pd.Timestamp(sorted(a.index)[-1]).month


def single_quarters(df: pd.DataFrame, fy_m: int) -> pd.DataFrame:
    out = {}
    dates = list(df.index)
    for d in dates:
        # 同一财年内的上一累计期（含财年末 Q4 = FY − 9M）
        if fy_m == 12:
            fy_start = pd.Timestamp(d.year, 1, 1)
        else:
            y = d.year if d.month > fy_m else d.year - 1
            fy_start = pd.Timestamp(y, fy_m + 1, 1)
        prev = [x for x in dates if x < d and x >= fy_start]
        if prev:
            out[d] = df.loc[d] - df.loc[prev[-1]]
        else:
            out[d] = df.loc[d]
    return pd.DataFrame(out).T.sort_index()


def yoy_table(df: pd.DataFrame, code: str, label: str) -> list[str]:
    lines = []
    dates = list(df.index)
    for d in dates[-6:]:
        prev = pd.Timestamp(d) - pd.DateOffset(years=1)
        if prev not in df.index:
            continue
        cur, old = df.loc[d], df.loc[prev]
        cells = []
        for c in COLS:
            if c in df.columns and pd.notna(cur.get(c)) and pd.notna(old.get(c)) and old.get(c):
                cells.append(f"{cur[c]/1e8:.0f}({(cur[c]/old[c]-1)*100:+.0f}%)")
            else:
                cells.append("—")
        ratios = []
        for name, a, b in RATIOS:
            if a in df.columns and b in df.columns and pd.notna(cur.get(b)) and cur.get(b):
                ratios.append(f"{name}{cur[a]/cur[b]*100:.1f}%")
        lines.append(f"| {str(d)[:10]} | " + " | ".join(cells) + " | " + " ".join(ratios) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    a = ap.parse_args()
    for code in [c.strip() for c in a.codes.split(",") if c.strip()]:
        is_, cf = combined(code)
        if is_.empty:
            print(f"\n## {code}\n数据缺失"); continue
        fy = fy_end_month(code)
        latest = is_.index.max()
        print(f"\n## {code}｜最新累计期 {str(latest)[:10]}｜财年末 {fy} 月")
        print("\n累计（亿元，括号=同比）")
        print("| 期间 | " + " | ".join(COLS) + " | 利润率 |")
        print("|---|" + "---|" * (len(COLS) + 1))
        rows = yoy_table(is_, code, "累计")
        print("\n".join(rows) if rows else "（无同比可比期）")
        sq = single_quarters(is_, fy)
        print("\n单季（亿元，括号=同比）")
        print("| 期间 | " + " | ".join(COLS) + " | 利润率 |")
        print("|---|" + "---|" * (len(COLS) + 1))
        rows = yoy_table(sq, code, "单季")
        print("\n".join(rows) if rows else "（无同比可比期）")
        if not cf.empty:
            for c in ["经营业务现金净额", "购建固定资产"]:
                if c in cf.columns:
                    tail = cf[c].dropna().tail(4)
                    print(f"\n{c}（累计/亿元）: " + "，".join(f"{str(i)[:10]}={v/1e8:.0f}" for i, v in tail.items()))


if __name__ == "__main__":
    main()
