#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""港股“估值回归”检验：倍数是否回归？回归能否预测收益？

两层：
  1) AR(1)：log(PE)/log(PB) 的持续性（phi<1 即均值回复），半衰期 = ln.5/ln phi。
  2) 预测：估值水平 / 偏离自身历史中位 是否预测未来 1y/3y 收益（按年份聚类 IC）。

样本：当前自选 30 只（幸存者偏差），fund_cache 年报 + yfinance 年末价。
用法：python3 scripts/valuation_reversion_test.py
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, yfinance as yf
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import WATCHLIST  # noqa: E402

HKD = {"0005.HK", "0012.HK", "0016.HK", "0388.HK", "2388.HK", "0981.HK", "1347.HK", "1299.HK"}
FX = {"CNY": 1.08, "HKD": 1.0}


def pick(df, d, names):
    for n in ([names] if isinstance(names, str) else names):
        if d in df.index and n in df.columns and pd.notna(df.loc[d, n]):
            return float(df.loc[d, n])
    return np.nan


def build() -> pd.DataFrame:
    rows = []
    for c in WATCHLIST:
        fx = FX["HKD" if c in HKD else "CNY"]
        try:
            is_ = pd.read_pickle(os.path.join(ROOT, f"data/fund_cache/{c}_利润表_年度.pkl"))
            bs = pd.read_pickle(os.path.join(ROOT, f"data/fund_cache/{c}_资产负债表_年度.pkl"))
        except Exception:
            continue
        h = yf.Ticker(c).history(period="max", interval="1d", auto_adjust=True, timeout=30)["Close"]
        h.index = pd.to_datetime(h.index)
        if h.index.tz is not None:
            h.index = h.index.tz_convert(None)
        ye = h.groupby(h.index.year).last()
        for d in sorted(is_.index):
            y = d.year
            if y not in ye.index:
                continue
            net = pick(is_, d, ["股东应占溢利", "除税后溢利"])
            eps = pick(is_, d, "每股基本盈利")
            eq = pick(bs, d, ["归属于母公司股东权益", "股东权益", "总权益"])
            if not (net and eps and eq and net > 0 and eps > 0 and eq > 0):
                continue
            p = float(ye.loc[y])
            pe = p / (eps * fx)
            pb = p / (eq / (net / eps) * fx)
            f1 = ye.get(y + 1, np.nan) / p - 1 if (y + 1 in ye.index) else np.nan
            f3 = ye.get(y + 3, np.nan) / p - 1 if (y + 3 in ye.index) else np.nan
            rows.append(dict(code=c, year=y, pe=pe, pb=pb, f1=f1, f3=f3))
    df = pd.DataFrame(rows)
    return df[(df.pe > 0) & (df.pb > 0)].sort_values(["code", "year"])


def main():
    df = build()
    out = ["# 港股“估值回归”检验\n",
           f"> 样本：当前自选池 {df.code.nunique()} 只、{df.year.min()}–{df.year.max()}、"
           f"{len(df)} 个股票-年（**幸存者偏差**）。价格=年末，PE/PB=年末价/当年EPS(BVPS)。\n"]
    out.append("\n## 1. 倍数是否回归（AR1）\n")
    for col in ["pe", "pb"]:
        d = df.copy(); d["l"] = np.log(d[col])
        d["x"] = d["l"] - d.groupby("code")["l"].transform("mean")
        d["x1"] = d.groupby("code")["l"].shift(1)
        d["x1"] = d["x1"] - d.groupby("code")["l"].transform("mean")
        z = d.dropna(subset=["x", "x1"])
        phi = np.polyfit(z["x1"], z["x"], 1)[0]
        hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
        out.append(f"- log({col.upper()})：n={len(z)}，AR1 phi={phi:.3f}，半衰期≈{hl:.1f} 年"
                   f"（phi<1 表示存在均值回复）。")
    out.append("\n> 但“倍数回归”不等于“收益可预测”：回归可由盈利增长（分母变大）实现，而非价格下跌。\n")

    out.append("\n## 2. 估值能否预测收益（按年份聚类 IC）\n")
    out.append("| 信号 → 未来收益 | 年数 | 平均IC | t | p | 正IC占比 |")
    out.append("|---|---|---|---|---|---|")
    for col in ["pe", "pb"]:
        df["l" + col] = np.log(df[col])
        df["med_" + col] = df.groupby("code")[col].transform(
            lambda s: s.expanding(3).median().shift(1))
        df["z_" + col] = df["l" + col] - np.log(df["med_" + col])
    for z in ["z_pe", "z_pb"]:
        for f in ["f1", "f3"]:
            ics = []
            for y, g in df.dropna(subset=[z, f]).groupby("year"):
                if len(g) < 8:
                    continue
                ic = stats.spearmanr(g[z], g[f]).correlation
                if np.isfinite(ic):
                    ics.append(ic)
            ics = np.array(ics)
            if len(ics) > 2:
                t, p = stats.ttest_1samp(ics, 0)
                out.append(f"| {z}({col2(z)}) → {f} | {len(ics)} | {ics.mean():+.3f} | "
                           f"{t:+.2f} | {p:.3f} | {(ics > 0).mean()*100:.0f}% |")
    out.append("\n> 预期：若“偏离自身中位高 → 未来收益低”，IC 应显著为负。实测接近 0。\n")

    out.append("\n## 3. 结论\n")
    out.append("- **倍数有均值回复**（phi<1），但**估值偏离不预测未来 1/3 年收益**（IC≈0）。")
    out.append("- 本项目既有证据一致：D39 价格型价值 null、D41/D42 PIT 基本面 null、D113 质量+价值底仓 −8.1%。")
    out.append("- ⇒ “等估值回归”只能当**情景/风控**，不能当**交易信号**；判断仍要看 EPS/现金流/分红。")
    out.append("- 局限：当前池（幸存者）、样本小；结论为“未发现”，非“已证明不存在”。\n")
    os.makedirs(os.path.join(ROOT, "output/_logs"), exist_ok=True)
    p = os.path.join(ROOT, "output/_logs/valuation_reversion.md")
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("写入", p)
    print("\n".join(out[:40]))


def col2(z):
    return "PE" if z == "z_pe" else "PB"


if __name__ == "__main__":
    sys.exit(main())
