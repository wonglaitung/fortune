#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""季节性择时（逐月入市/离开）验证。预注册：docs/SEASONAL_TIMING_PREREG.md

主假设：把「历史最优月份配对」逐年滚动选股（扩张窗、天然 OOS）是否**样本外**优于买入持有。
次假设：固定规则 R1「11–4 月持有」仅作描述（含选择偏差，不作判决）。

口径：ml_services.strategy_eval（逐日 mark-to-market + 静态等仓位对照 + MDE），
成本往返 0.2%（D97），T+1 生效。只读 F1 快照（data/hsi_snapshot/hsi_daily.pkl），禁止实时取数。

用法：python3 scripts/seasonal_timing_backtest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ml_services.strategy_eval import (  # noqa: E402
    daily_mark, static_matched, metrics_daily, has_timing_increment)

COST_ROUND_TRIP = 0.002
COST_SENS = 0.005
SNAP = os.path.join(ROOT, "data", "hsi_snapshot", "hsi_daily.pkl")
OUT = os.path.join(ROOT, "output", "seasonal_timing")
FIRST_TRADE_YEAR = 1996
MIN_HIST_YEARS = 8
R1_MONTHS = {11, 12, 1, 2, 3, 4}
CN_MON = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]


def load_snapshot() -> pd.Series:
    if not os.path.exists(SNAP):
        raise SystemExit("❌ 缺少 F1 快照：先运行 python3 scripts/pin_hsi_snapshot.py")
    s = pickle.load(open(SNAP, "rb"))
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def monthly_pivot(close: pd.Series) -> tuple[np.ndarray, list[int]]:
    me = close.resample("M").last()
    r = me.pct_change().dropna()
    r.index = r.index.to_period("M")
    df = pd.DataFrame({"r": r})
    df["y"] = df.index.year
    df["m"] = df.index.month
    piv = df.pivot(index="y", columns="m", values="r").dropna(how="any")
    piv = piv.reindex(columns=range(1, 13))
    return piv.to_numpy(), list(piv.index)


def best_combo_from_history(R: np.ndarray) -> tuple[int, int]:
    """在 R（k×12 月度收益）上选均值最高的 (i,j)（0-based，i<=j）。"""
    n = len(R)
    C = np.cumprod(1.0 + R, axis=1)
    best, best_val = (0, 11), -np.inf
    for i in range(12):
        base = np.ones(n) if i == 0 else C[:, i - 1]
        for j in range(i, 12):
            v = np.mean(C[:, j] / base - 1.0)
            if v > best_val:
                best_val, best = v, (i, j)
    return best


def expanding_weights(ret_idx: pd.DatetimeIndex, R: np.ndarray, years: list[int]) -> pd.Series:
    """逐年扩张窗选配对，生成下一年权重（PIT）。"""
    w = pd.Series(0.0, index=ret_idx)
    for Y in range(FIRST_TRADE_YEAR, max(years) + 1):
        hist = [k for k, yr in enumerate(years) if yr < Y]
        if len(hist) < MIN_HIST_YEARS:
            continue
        i, j = best_combo_from_history(R[hist])
        mask = (ret_idx.year == Y) & (ret_idx.month >= i + 1) & (ret_idx.month <= j + 1)
        w[mask] = 1.0
    return w


def month_set_weights(ret_idx: pd.DatetimeIndex, months: set[int]) -> pd.Series:
    return pd.Series([1.0 if m in months else 0.0 for m in ret_idx.month], index=ret_idx)


def annual_net(net: pd.Series) -> pd.Series:
    return (1.0 + net).groupby(net.index.year).prod() - 1.0


def block_bootstrap_ci(x: np.ndarray, block: int = 3, n_boot: int = 5000,
                       seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=k)
        vals = np.concatenate([x[np.arange(s, s + block) % n] for s in starts])[:n]
        means[b] = vals.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def combo_annual_returns(R: np.ndarray, years: list[int], eval_years: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """66 个配对在每个评估年的持有收益（矩阵 T×66）。"""
    combos = [(i, j) for i in range(12) for j in range(i, 12)]
    idx = {yr: k for k, yr in enumerate(years)}
    rows = []
    for yr in eval_years:
        k = idx[yr]
        row = []
        for (i, j) in combos:
            row.append(float(np.prod(1.0 + R[k, i:j + 1]) - 1.0))
        rows.append(row)
    return np.array(rows), combos


def cscv_pbo(M: np.ndarray, S: int = 8) -> float:
    """CSCV 估计 PBO：IS 最优策略的 OOS 排名低于中位数的概率。"""
    T, N = M.shape
    if T < S or N < 2:
        return float("nan")
    blocks = np.array_split(np.arange(T), S)
    splits = list(combinations(range(S), S // 2))
    overfit = 0
    for is_idx in splits:
        oos_idx = [b for b in range(S) if b not in is_idx]
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in oos_idx])
        IS = M[is_rows]; OOS = M[oos_rows]
        sr_is = IS.mean(0) / (IS.std(0, ddof=1) + 1e-12)
        sr_oos = OOS.mean(0) / (OOS.std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (np.sum(sr_oos < sr_oos[best]) + 0.5 * np.sum(sr_oos == sr_oos[best])) / N
        if rank < 0.5:
            overfit += 1
    return overfit / len(splits)


def deflated_sharpe(sr: float, trials_sr: np.ndarray, T: int, skew: float, kurt: float) -> float:
    """Bailey–López de Prado DSR（对 N 次试验去偏）。"""
    N = len(trials_sr)
    if N < 2 or T < 2:
        return float("nan")
    gamma = 0.5772156649
    V = np.var(trials_sr, ddof=1)
    e_max = np.sqrt(V) * ((1 - gamma) * stats.norm.ppf(1 - 1.0 / N)
                          + gamma * stats.norm.ppf(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr * sr)
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - e_max) * np.sqrt(T - 1) / denom))


def md_table(headers, rows) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.4f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    close = load_snapshot()
    ret = close.pct_change().dropna()
    R, years = monthly_pivot(close)
    eval_years = [y for y in years if y >= FIRST_TRADE_YEAR]
    out_dir = OUT
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 策略 ----------
    net_bh = daily_mark(ret, 1.0, round_trip_cost=COST_ROUND_TRIP, lag=1)
    w_r1 = month_set_weights(ret.index, R1_MONTHS)
    net_r1 = daily_mark(ret, w_r1, round_trip_cost=COST_ROUND_TRIP, lag=1)
    w_p = expanding_weights(ret.index, R, years)
    net_p = daily_mark(ret, w_p, round_trip_cost=COST_ROUND_TRIP, lag=1)

    start = pd.Timestamp(f"{FIRST_TRADE_YEAR}-01-01")
    def sl(x): return x[x.index >= start]
    mbh, mr1, mp = metrics_daily(sl(net_bh)), metrics_daily(sl(net_r1)), metrics_daily(sl(net_p))

    exp_r1 = float(w_r1[w_r1.index >= start].mean())
    net_static = static_matched(sl(ret), exp_r1)
    mstatic = metrics_daily(net_static)
    timing_ok = has_timing_increment(sl(net_r1), net_static)

    # ---------- 年度超额 + bootstrap ----------
    a_bh = annual_net(sl(net_bh)); a_p = annual_net(sl(net_p)); a_r1 = annual_net(sl(net_r1))
    idx = a_p.index.intersection(a_bh.index)
    exc_p = (a_p.reindex(idx) - a_bh.reindex(idx)).to_numpy()
    exc_r1 = (a_r1.reindex(idx) - a_bh.reindex(idx)).to_numpy()
    lo_p, hi_p = block_bootstrap_ci(exc_p)
    lo_r1, hi_r1 = block_bootstrap_ci(exc_r1)
    n_yr = len(idx)
    mde = float(stats.t.ppf(0.975, n_yr - 1) * np.std(exc_p, ddof=1) / np.sqrt(n_yr))

    # ---------- PBO / DSR ----------
    M, combos = combo_annual_returns(R, years, eval_years)
    pbo = cscv_pbo(M, S=8)
    # 各配对年度 Sharpe（用于 DSR 的试验分布）
    trials_sr = M.mean(0) / (M.std(0, ddof=1) + 1e-12)
    ap = a_p.to_numpy()
    sr_p = float(np.mean(ap) / (np.std(ap, ddof=1) + 1e-12))
    dsr = deflated_sharpe(sr_p, trials_sr, len(ap),
                          float(stats.skew(ap)), float(stats.kurtosis(ap, fisher=False)))

    # ---------- 子窗 2010+ ----------
    exc_2010_cagr = float((1 + a_p[a_p.index >= 2010]).prod() ** (1 / len(a_p[a_p.index >= 2010])) - 1
                          - ((1 + a_bh[a_bh.index >= 2010]).prod() ** (1 / len(a_bh[a_bh.index >= 2010])) - 1))

    # ---------- 相邻参数（诊断） ----------
    adj = []
    for st in (10, 11, 12):
        for en in (3, 4, 5):
            months = set(range(st, 13)) | set(range(1, en + 1))
            w = month_set_weights(ret.index, months)
            net = daily_mark(ret, w, round_trip_cost=COST_ROUND_TRIP, lag=1)
            m = metrics_daily(sl(net))
            adj.append([f"{CN_MON[st-1]}→{CN_MON[en-1]}", m["cagr"] - mbh["cagr"]])

    # ---------- 判据 ----------
    P1 = mp["cagr"] - mbh["cagr"]
    P2 = lo_p > 0
    P3 = (mp["maxdd"] > mbh["maxdd"]) and timing_ok
    P4 = mp["sharpe"] >= mbh["sharpe"] - 0.05
    P5 = pbo < 0.5
    P6 = dsr >= 0.95
    P7 = exc_2010_cagr > 0
    # P8 由外部双跑 md5 判定

    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_sha256": hashlib.sha256(open(SNAP, "rb").read()).hexdigest(),
        "window": [str(sl(net_bh).index.min().date()), str(sl(net_bh).index.max().date())],
        "n_years": n_yr,
        "cost_round_trip": COST_ROUND_TRIP,
        "buy_hold": mbh,
        "fixed_R1": mr1,
        "procedure_P": mp,
        "static_matched": mstatic,
        "R1_exposure": exp_r1,
        "timing_increment_R1": bool(timing_ok),
        "excess_annual_P": {"mean": float(np.mean(exc_p)), "ci95": [lo_p, hi_p], "mde": mde},
        "excess_annual_R1": {"mean": float(np.mean(exc_r1)), "ci95": [lo_r1, hi_r1]},
        "excess_cagr_2010plus_P": exc_2010_cagr,
        "pbo": pbo,
        "dsr": dsr,
        "adjacent_R1": adj,
        "criteria": {"P1": bool(P1 > 0), "P2": bool(P2), "P3": bool(P3), "P4": bool(P4),
                     "P5": bool(P5), "P6": bool(P6), "P7": bool(P7)},
        "P1_excess_cagr": float(P1),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)

    eq = pd.DataFrame({"buy_hold": sl(net_bh), "fixed_R1": sl(net_r1),
                       "procedure_P": sl(net_p), "static_matched": net_static}).dropna(how="all")
    eq.index.name = "date"
    eq.round(10).to_csv(os.path.join(out_dir, "equity.csv"), float_format="%.10f")

    # ---------- markdown ----------
    md = ["# 季节性择时验证结果\n",
          f"> 生成：{metrics['generated_at']}｜窗口 {metrics['window'][0]} → {metrics['window'][1]}"
          f"（{n_yr} 年）｜成本往返 {COST_ROUND_TRIP*100:.1f}%｜快照 sha256={metrics['snapshot_sha256'][:16]}…\n",
          "> 预注册：`docs/SEASONAL_TIMING_PREREG.md`｜评估口径：`ml_services/strategy_eval.py`\n",
          "\n## 1. 策略表现（逐日 mark-to-market）\n",
          md_table(["策略", "CAGR", "波动", "Sharpe", "maxDD", "Calmar"],
                   [["买入持有", mbh["cagr"], mbh["vol"], mbh["sharpe"], mbh["maxdd"], mbh["calmar"]],
                    ["固定 R1（11–4月）", mr1["cagr"], mr1["vol"], mr1["sharpe"], mr1["maxdd"], mr1["calmar"]],
                    ["程序 P（OOS 滚动）", mp["cagr"], mp["vol"], mp["sharpe"], mp["maxdd"], mp["calmar"]],
                    ["静态同仓位（少投对照）", mstatic["cagr"], mstatic["vol"], mstatic["sharpe"],
                     mstatic["maxdd"], mstatic["calmar"]]]),
          "\n## 2. 年度超额（程序 P − 买入持有）\n",
          md_table(["项", "值"],
                   [["均值", float(np.mean(exc_p))],
                    ["95%CI", f"[{lo_p*100:+.2f}%, {hi_p*100:+.2f}%]"],
                    ["MDE（最小可探测）", mde],
                    ["2010+ 年化超额", exc_2010_cagr]]),
          "\n## 3. 过拟合 / 稳健\n",
          md_table(["项", "值"],
                   [["PBO（66 配对 CSCV）", pbo],
                    ["DSR（对 66 次试验）", dsr],
                    ["相邻参数 9 格 净超额 CAGR 范围",
                     f"[{min(a[1] for a in adj)*100:+.2f}%, {max(a[1] for a in adj)*100:+.2f}%]"],
                    ["相邻参数符号一致", all(a[1] > 0 for a in adj) or all(a[1] < 0 for a in adj)]]),
          "\n## 4. 预注册判据\n",
          md_table(["判据", "要求", "结果", "通过"],
                   [["P1 净超额 CAGR>0", ">0", f"{P1*100:+.2f}%", "✅" if P1 > 0 else "❌"],
                    ["P2 年度超额 CI 下界>0", ">0", f"下界 {lo_p*100:+.2f}%", "✅" if P2 else "❌"],
                    ["P3 maxDD 更浅 且 真择时", "是", f"maxDD {mp['maxdd']*100:.1f}% vs {mbh['maxdd']*100:.1f}% / 真择时={timing_ok}", "✅" if P3 else "❌"],
                    ["P4 Sharpe≥持有−0.05", "是", f"{mp['sharpe']:.2f} vs {mbh['sharpe']:.2f}", "✅" if P4 else "❌"],
                    ["P5 PBO<0.5", "<0.5", f"{pbo:.2f}", "✅" if P5 else "❌"],
                    ["P6 DSR≥0.95", "≥0.95", f"{dsr:.3f}", "✅" if P6 else "❌"],
                    ["P7 2010+ 超额>0", ">0", f"{exc_2010_cagr*100:+.2f}%", "✅" if P7 else "❌"],
                    ["P8 双跑 md5 一致", "一致", "见外部比对", "—"]]),
          "\n## 5. 判决\n"]
    all_pass = all([P1 > 0, P2, P3, P4, P5, P6, P7])
    md.append(f"- 程序 P（主假设）：{'🟢 全部通过' if all_pass else '🔴/🟡 未全过'}。"
              f"净超额 CAGR {P1*100:+.2f}%，年度超额 CI [{lo_p*100:+.2f}%, {hi_p*100:+.2f}%]，"
              f"PBO={pbo:.2f}，DSR={dsr:.3f}，MDE={mde:.3f}。")
    md.append("- 固定 R1 仅描述（含选择偏差），不参与判决。")
    md.append("- 适用范围：恒指本快照/本窗口；**外推生产 ✗**。若主假设失败 ⇒ 停止日历维度搜索。\n")
    with open(os.path.join(out_dir, "result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[seasonal] 结果写入 {out_dir}")
    print(f"  P: CAGR {mp['cagr']*100:.2f}% / 超额 {P1*100:+.2f}% / CI [{lo_p*100:+.2f}%,{hi_p*100:+.2f}%] "
          f"/ PBO {pbo:.2f} / DSR {dsr:.3f} / maxDD {mp['maxdd']*100:.1f}%")
    print(f"  判据 P1-P7: {metrics['criteria']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
