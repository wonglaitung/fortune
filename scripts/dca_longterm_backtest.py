#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定投恒指 + 长期持有：能否赚钱 / 是否胜过存银行。预注册 docs/DCA_LONGTERM_PREREG.md

主口径 2800.HK（盈富，含息）；保守下界 ^HSI 价格（无分红）；银行=HKD 1M HIBOR。
只读 data/dca_snapshot/ 快照；每月首个交易日投固定额；成本 0.1%/次。
"""
from __future__ import annotations
import hashlib, json, os, pickle, sys
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "dca_snapshot")
HIBOR = os.path.join(ROOT, "data", "cnh_hibor", "hibor_hkd.csv")
OUT = os.path.join(ROOT, "output", "dca_longterm")
C, COST = 10000.0, 0.001
HORIZONS = [5, 10, 15, 20, 25]


def load(p):
    s = pickle.load(open(p, "rb")); s.index = pd.to_datetime(s.index); return s.sort_index().dropna()


def monthly_table(price: pd.Series):
    df = pd.DataFrame({"p": price})
    grp = df.groupby(price.index.to_period("M"))
    per = grp.first().index
    return (grp.first()["p"].to_numpy(), grp.last()["p"].to_numpy(), per, [str(p) for p in per])


def hibor_monthly():
    d = pd.read_csv(HIBOR, parse_dates=["date"]).set_index("date").sort_index()
    r = d["hibor_1m"].ffill() / 100.0
    return r.groupby(r.index.to_period("M")).last()


def irr_monthly(n: int, terminal: float, c: float) -> float:
    if not np.isfinite(terminal) or terminal <= 0 or n <= 0:
        return float("nan")
    lo, hi = -0.99, 1.0
    def f(r): return sum(c * (1 + r) ** (n - k) for k in range(n)) - terminal
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(120):
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0: hi, fhi = mid, fm
        else: lo, flo = mid, fm
    return (1 + (lo + hi) / 2) ** 12 - 1


def simulate(pf, pl, rates_dec, cost, a, b):
    n = b - a + 1
    sh = C * (1 - cost) / pf[a:b + 1]
    shares = sh.sum()
    terminal = shares * pl[b]
    contributed = C * n
    irr = irr_monthly(n, terminal, C)
    # bank: product of (1+r_j/12) from k..b
    lr = np.log1p(rates_dec[a:b + 1] / 12.0)
    lc = np.concatenate([[0.0], np.cumsum(lr)])  # lc[i]=sum lr[a..a+i-1]
    # product k..b = exp(lc[n] - lc[k-a])
    bank_term = C * float(np.sum(np.exp(lc[n] - lc[:n])))
    bank_irr = irr_monthly(n, bank_term, C) if np.isfinite(bank_term) else float("nan")
    # underwater
    cum_sh = np.cumsum(sh)
    val = cum_sh * pl[a:b + 1]
    basis = C * np.arange(1, n + 1)
    uw = float(np.min(val / basis - 1.0))
    return dict(n=n, multiple=terminal / contributed, irr=irr, bank_irr=bank_irr,
                excess_irr=irr - bank_irr, underwater=uw)


def run_universe(price, rates, label):
    pf, pl, periods, months = monthly_table(price)
    rm = hibor_monthly().reindex(periods).ffill().to_numpy()
    # 只保留 HIBOR 有数据之后（领先后无法 ffill）⇒ 从首月有效利率开始
    valid = np.where(np.isfinite(rm))[0]
    if len(valid) == 0:
        raise SystemExit("❌ HIBOR 与行情无重叠月份")
    a0 = int(valid[0])
    pf, pl, periods, months, rm = pf[a0:], pl[a0:], periods[a0:], months[a0:], rm[a0:]
    res, rows = {}, {}

    def collect(H, a, b):
        r = simulate(pf, pl, rm, COST, a, b)
        rows.setdefault(H, []).append((months[a], months[b], r))

    for H in HORIZONS:
        nm = H * 12
        for a in range(0, len(months) - nm + 1):
            collect(H, a, a + nm - 1)
        if not rows.get(H):
            continue
        arr = rows[H]
        mult = np.array([x[2]["multiple"] for x in arr])
        irr = np.array([x[2]["irr"] for x in arr])
        birr = np.array([x[2]["bank_irr"] for x in arr])
        uw = np.array([x[2]["underwater"] for x in arr])
        res[H] = dict(n_windows=len(arr),
                      pct_profit=float(np.mean(mult > 1.0)),
                      median_multiple=float(np.median(mult)),
                      median_irr=float(np.nanmedian(irr)),
                      worst_irr=float(np.nanmin(irr)),
                      pct_beat_bank=float(np.mean(irr > birr)),
                      median_bank_irr=float(np.nanmedian(birr)),
                      worst_underwater=float(np.nanmin(uw)))
    full = simulate(pf, pl, rm, COST, 0, len(months) - 1)
    return {"label": label, "months": [months[0], months[-1]], "by_horizon": res,
            "full": full, "rows": rows}


def main():
    os.makedirs(OUT, exist_ok=True)
    etf = run_universe(load(os.path.join(DIR, "etf2800.pkl")), None, "2800.HK(含息)")
    hsi = run_universe(load(os.path.join(DIR, "hsi.pkl")), None, "^HSI(价格,不含息)")

    def g(u, H, k):
        return u["by_horizon"].get(H, {}).get(k)

    K1 = (g(etf, 10, "pct_profit") == 1.0) and (g(etf, 15, "pct_profit") == 1.0)
    K2 = (g(etf, 10, "pct_beat_bank") is not None) and (g(etf, 10, "pct_beat_bank") >= 0.75)
    K3 = etf["full"]["irr"] > etf["full"]["bank_irr"]
    K4 = (g(hsi, 10, "pct_profit") is not None) and (g(hsi, 10, "pct_profit") >= 0.95)
    crit = {"K1": bool(K1), "K2": bool(K2), "K3": bool(K3), "K4": bool(K4)}

    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "etf2800_sha256": hashlib.sha256(open(os.path.join(DIR, "etf2800.pkl"), "rb").read()).hexdigest(),
        "hsi_sha256": hashlib.sha256(open(os.path.join(DIR, "hsi.pkl"), "rb").read()).hexdigest(),
        "hibor_sha256": hashlib.sha256(open(HIBOR, "rb").read()).hexdigest(),
        "contribution": C, "cost": COST, "horizons": HORIZONS,
        "primary": {k: v for k, v in etf.items() if k != "rows"},
        "conservative": {k: v for k, v in hsi.items() if k != "rows"},
        "criteria": crit,
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, sort_keys=True)
    # windows.csv（主口径 + 保守）
    wrows = []
    for u in (etf, hsi):
        for H, arr in u["rows"].items():
            for a, b, r in arr:
                wrows.append([u["label"], H, a, b, r["multiple"], r["irr"], r["bank_irr"],
                              r["excess_irr"], r["underwater"]])
    pd.DataFrame(wrows, columns=["universe", "horizon", "start", "end", "multiple",
                                 "irr", "bank_irr", "excess_irr", "underwater"]).to_csv(
        os.path.join(OUT, "windows.csv"), index=False, float_format="%.10f")

    def tbl(u):
        lines = ["| 持有期 | 窗口数 | 期末>投入占比 | 中位倍数 | 中位IRR | 最差IRR | 跑赢银行占比 | 最差浮亏 |",
                 "|---|---|---|---|---|---|---|---|"]
        for H in HORIZONS:
            if H not in u["by_horizon"]: continue
            d = u["by_horizon"][H]
            lines.append(f"| {H}y | {d['n_windows']} | {d['pct_profit']*100:.1f}% | "
                         f"{d['median_multiple']:.2f}x | {d['median_irr']*100:+.2f}% | "
                         f"{d['worst_irr']*100:+.2f}% | {d['pct_beat_bank']*100:.1f}% | "
                         f"{d['worst_underwater']*100:+.1f}% |")
        return "\n".join(lines)

    md = ["# 定投恒指 + 长期持有：验证结果\n",
          f"> 生成 {metrics['generated_at']}｜每月首日投 {C:.0f}、成本 {COST*100:.1f}%/次｜"
          f"HIBOR 1M｜窗口滚动\n",
          "> 预注册：`docs/DCA_LONGTERM_PREREG.md`\n",
          "\n## 1. 主口径：2800.HK（含息，2008→今）\n", tbl(etf),
          f"\n- 全期（{etf['months'][0]}→{etf['months'][1]}）：定投 IRR {etf['full']['irr']*100:+.2f}%，"
          f"银行 IRR {etf['full']['bank_irr']*100:+.2f}%，期末/投入 {etf['full']['multiple']:.2f}x，"
          f"最差浮亏 {etf['full']['underwater']*100:+.1f}%。\n",
          "\n## 2. 保守下界：^HSI 价格（不含分红）\n", tbl(hsi),
          f"\n- 全期（{hsi['months'][0]}→{hsi['months'][1]}）：定投 IRR {hsi['full']['irr']*100:+.2f}%，"
          f"银行 IRR {hsi['full']['bank_irr']*100:+.2f}%，期末/投入 {hsi['full']['multiple']:.2f}x。\n",
          "\n## 3. 预注册判据\n",
          "| 判据 | 要求 | 结果 | 通过 |", "|---|---|---|---|",
          f"| K1 ≥10/15y 窗口全赚 | 100% | 10y {g(etf,10,'pct_profit')*100:.1f}% / 15y {g(etf,15,'pct_profit')*100:.1f}% | {'✅' if K1 else '❌'} |",
          f"| K2 ≥10y 跑赢银行 | ≥75% | {g(etf,10,'pct_beat_bank')*100:.1f}% | {'✅' if K2 else '❌'} |",
          f"| K3 全期 IRR>银行 | >0 | {etf['full']['irr']*100:+.2f}% vs {etf['full']['bank_irr']*100:+.2f}% | {'✅' if K3 else '❌'} |",
          f"| K4 保守口径 ≥10y 全赚 | ≥95% | {g(hsi,10,'pct_profit')*100:.1f}% | {'✅' if K4 else '❌'} |",
          "| K5 双跑 md5 | 一致 | 外部比对 | — |",
          "\n## 4. 判决\n"]
    allpass = all(crit.values())
    md.append(f"- {'🟢 全部通过' if allpass else '🔴/🟡 未全过'}："
              f"2800 含息 10 年窗口全赚 {g(etf,10,'pct_profit')*100:.1f}%，"
              f"跑赢银行 {g(etf,10,'pct_beat_bank')*100:.1f}%，全期 IRR {etf['full']['irr']*100:+.2f}%。")
    md.append("- 注意：滚动窗口高度重叠，占比只作描述，不当独立显著性；未计通胀与税费。")
    md.append("- 适用范围：恒指/盈富、本快照、本窗口；**不构成投资建议**。\n")
    open(os.path.join(OUT, "result.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    print(f"[dca] 写入 {OUT}")
    print(f"  2800含息 10y: 全赚 {g(etf,10,'pct_profit')*100:.1f}% / 跑赢银行 {g(etf,10,'pct_beat_bank')*100:.1f}% / 最差IRR {g(etf,10,'worst_irr')*100:+.2f}%")
    print(f"  2800含息 全期 IRR {etf['full']['irr']*100:+.2f}% vs 银行 {etf['full']['bank_irr']*100:+.2f}%")
    print(f"  ^HSI价格 10y: 全赚 {g(hsi,10,'pct_profit')*100:.1f}% / 全期 IRR {hsi['full']['irr']*100:+.2f}% vs 银行 {hsi['full']['bank_irr']*100:+.2f}%")
    print(f"  判据: {crit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
