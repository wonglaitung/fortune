#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""年度“一笔买卖”策略验证。预注册：docs/ANNUAL_ONE_TRADE_PREREG.md

口径：每个自然年只做 1 笔（买入月第一个交易日买、卖出月最后一个交易日卖），年末空仓，次年重置。
选择目标 = 年度持有收益的 IR（风险调整）。主假设用扩张窗逐年选、天然 OOS。

评估口径复用 ml_services.strategy_eval（逐日 mark + 静态对照 + MDE），只读 F1 快照。

用法：python3 scripts/annual_one_trade_backtest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ml_services.strategy_eval import (  # noqa: E402
    daily_mark, static_matched, metrics_daily, has_timing_increment)
from seasonal_timing_backtest import (  # noqa: E402  复用同一实现的统计工具
    monthly_pivot, block_bootstrap_ci, cscv_pbo, deflated_sharpe, md_table)

COST = 0.002
SNAP = os.path.join(ROOT, "data", "hsi_snapshot", "hsi_daily.pkl")
OUT = os.path.join(ROOT, "output", "annual_one_trade")
FIRST_YEAR = 1996
MIN_HIST = 8
CN_MON = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]
PAIRS = [(i, j) for i in range(12) for j in range(i, 12)]


def load_snapshot() -> pd.Series:
    if not os.path.exists(SNAP):
        raise SystemExit("❌ 缺少 F1 快照：先运行 python3 scripts/pin_hsi_snapshot.py")
    s = pickle.load(open(SNAP, "rb"))
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def pair_annual_net(R: np.ndarray, i: int, j: int) -> np.ndarray:
    """R(年×12) → 每个年度该配对的净收益（含一次往返成本）。"""
    gross = np.prod(1.0 + R[:, i:j + 1], axis=1) - 1.0
    return gross - COST


def ir_of(R: np.ndarray, i: int, j: int) -> float:
    x = pair_annual_net(R, i, j)
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return float(np.mean(x) / sd) if sd > 1e-12 else -np.inf


def pick_pair(R_hist: np.ndarray) -> tuple[int, int]:
    vals = [(ir_of(R_hist, i, j), (i, j)) for (i, j) in PAIRS]
    return max(vals)[1]


def pair_weights(idx: pd.DatetimeIndex, i: int, j: int) -> pd.Series:
    m = idx.month
    return pd.Series(((m >= i + 1) & (m <= j + 1)).astype(float), index=idx)


def expanding_weights(idx, R, years) -> tuple[pd.Series, list]:
    w = pd.Series(0.0, index=idx)
    chosen = []
    for Y in range(FIRST_YEAR, max(years) + 1):
        hist = [k for k, yr in enumerate(years) if yr < Y]
        if len(hist) < MIN_HIST:
            continue
        i, j = pick_pair(R[hist])
        chosen.append((Y, i, j))
        mask = (idx.year == Y) & (idx.month >= i + 1) & (idx.month <= j + 1)
        w[mask] = 1.0
    return w, chosen


def annual_net(net: pd.Series) -> pd.Series:
    return (1.0 + net).groupby(net.index.year).prod() - 1.0


def main() -> int:
    close = load_snapshot()
    ret = close.pct_change().dropna()
    R, years = monthly_pivot(close)
    eval_years = [y for y in years if y >= FIRST_YEAR]
    os.makedirs(OUT, exist_ok=True)

    net_bh = daily_mark(ret, 1.0, round_trip_cost=COST, lag=1)
    # 固定最优格（描述用；按全评估窗 IR）
    ev = [k for k, yr in enumerate(years) if yr >= FIRST_YEAR]
    i_fix, j_fix = max(((ir_of(R[ev], i, j), (i, j)) for (i, j) in PAIRS))[1]
    w_fix = pair_weights(ret.index, i_fix, j_fix)
    net_fix = daily_mark(ret, w_fix, round_trip_cost=COST, lag=1)
    # 程序 P（OOS 扩张窗）
    w_p, chosen = expanding_weights(ret.index, R, years)
    net_p = daily_mark(ret, w_p, round_trip_cost=COST, lag=1)

    start = pd.Timestamp(f"{FIRST_YEAR}-01-01")
    def sl(x): return x[x.index >= start]
    mbh, mfix, mp = metrics_daily(sl(net_bh)), metrics_daily(sl(net_fix)), metrics_daily(sl(net_p))
    exp_p = float(w_p[w_p.index >= start].mean())
    net_static = static_matched(sl(ret), exp_p)
    mstatic = metrics_daily(net_static)
    timing_ok = has_timing_increment(sl(net_p), net_static)

    a_bh = annual_net(sl(net_bh)); a_p = annual_net(sl(net_p)); a_fix = annual_net(sl(net_fix))
    idx = a_p.index.intersection(a_bh.index)
    exc_p = (a_p.reindex(idx) - a_bh.reindex(idx)).to_numpy()
    exc_fix = (a_fix.reindex(idx) - a_bh.reindex(idx)).to_numpy()
    lo_p, hi_p = block_bootstrap_ci(exc_p)
    n_yr = len(idx)
    mde = float(stats.t.ppf(0.975, n_yr - 1) * np.std(exc_p, ddof=1) / np.sqrt(n_yr))

    # PBO / DSR（78 配对）
    idxmap = {yr: k for k, yr in enumerate(years)}
    M = np.array([[float(pair_annual_net(np.array([R[idxmap[yr]]]), i, j)[0]) for (i, j) in PAIRS]
                  for yr in eval_years])
    pbo = cscv_pbo(M, S=8)
    trials_sr = M.mean(0) / (M.std(0, ddof=1) + 1e-12)
    ap = a_p.to_numpy()
    sr_p = float(np.mean(ap) / (np.std(ap, ddof=1) + 1e-12))
    dsr = deflated_sharpe(sr_p, trials_sr, len(ap),
                          float(stats.skew(ap)), float(stats.kurtosis(ap, fisher=False)))

    # 2010+ ΔSharpe
    def sharpe_from(y0):
        return metrics_daily(sl(net_p)[sl(net_p).index.year >= y0])["sharpe"] - \
               metrics_daily(sl(net_bh)[sl(net_bh).index.year >= y0])["sharpe"]
    dsh_2010 = sharpe_from(2010)

    # 相邻参数（固定最优格 ±1 月）
    adj = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = min(max(i_fix + di, 0), 11); j = min(max(j_fix + dj, i), 11)
            w = pair_weights(ret.index, i, j)
            m = metrics_daily(sl(daily_mark(ret, w, round_trip_cost=COST, lag=1)))
            adj.append([f"{CN_MON[i]}→{CN_MON[j]}", m["sharpe"] - mbh["sharpe"]])
    adj_sign_ok = all(a[1] > 0 for a in adj) or all(a[1] < 0 for a in adj)

    P1 = mp["sharpe"] - mbh["sharpe"]
    crit = {
        "A1": bool(P1 > 0), "A2": bool(lo_p > 0),
        "A3": bool((mp["maxdd"] > mbh["maxdd"]) and timing_ok),
        "A4": bool(pbo < 0.5), "A5": bool(dsr >= 0.95),
        "A6": bool(dsh_2010 > 0), "A7": bool(adj_sign_ok),
    }
    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_sha256": hashlib.sha256(open(SNAP, "rb").read()).hexdigest(),
        "window": [str(sl(net_bh).index.min().date()), str(sl(net_bh).index.max().date())],
        "n_years": n_yr, "cost_round_trip": COST,
        "buy_hold": mbh, "fixed_best": mfix, "procedure_P": mp, "static_matched": mstatic,
        "P_exposure": exp_p, "timing_increment_P": bool(timing_ok),
        "fixed_best_pair": [CN_MON[i_fix], CN_MON[j_fix]],
        "procedure_chosen": [[y, CN_MON[i], CN_MON[j]] for y, i, j in chosen],
        "delta_sharpe_P": P1,
        "excess_annual_P": {"mean": float(np.mean(exc_p)), "ci95": [lo_p, hi_p], "mde": mde},
        "excess_annual_fixed": {"mean": float(np.mean(exc_fix))},
        "delta_sharpe_2010plus": dsh_2010,
        "pbo": pbo, "dsr": dsr,
        "adjacent_fixed": adj, "criteria": crit,
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    eq = pd.DataFrame({"buy_hold": sl(net_bh), "fixed_best": sl(net_fix),
                       "procedure_P": sl(net_p), "static_matched": net_static}).dropna(how="all")
    eq.index.name = "date"
    eq.round(10).to_csv(os.path.join(OUT, "equity.csv"), float_format="%.10f")

    md = ["# 年度“一笔买卖”验证结果\n",
          f"> 生成 {metrics['generated_at']}｜窗口 {metrics['window'][0]} → {metrics['window'][1]}"
          f"（{n_yr} 年）｜每年 1 买 1 卖、年末空仓｜成本往返 {COST*100:.1f}%｜"
          f"快照 {metrics['snapshot_sha256'][:16]}…\n",
          "> 预注册：`docs/ANNUAL_ONE_TRADE_PREREG.md`｜口径：`ml_services/strategy_eval.py`\n",
          "\n## 1. 表现（逐日 mark）\n",
          md_table(["策略", "CAGR", "波动", "Sharpe", "maxDD", "Calmar"],
                   [["买入持有", mbh["cagr"], mbh["vol"], mbh["sharpe"], mbh["maxdd"], mbh["calmar"]],
                    [f"固定最优 {CN_MON[i_fix]}→{CN_MON[j_fix]}（含选择偏差）", mfix["cagr"], mfix["vol"], mfix["sharpe"], mfix["maxdd"], mfix["calmar"]],
                    ["程序 P（OOS 逐年选）", mp["cagr"], mp["vol"], mp["sharpe"], mp["maxdd"], mp["calmar"]],
                    ["静态同仓位（少投对照）", mstatic["cagr"], mstatic["vol"], mstatic["sharpe"], mstatic["maxdd"], mstatic["calmar"]]]),
          f"\n> 程序 P 平均在场比例 {exp_p*100:.0f}%；固定最优格逐年选择见 metrics.json。\n",
          "\n## 2. 年度超额（P − 买入持有）\n",
          md_table(["项", "值"], [["均值", float(np.mean(exc_p))],
                                 ["95%CI", f"[{lo_p*100:+.2f}%, {hi_p*100:+.2f}%]"],
                                 ["MDE", mde], ["ΔSharpe(P)", P1], ["2010+ ΔSharpe", dsh_2010]]),
          "\n## 3. 过拟合/稳健\n",
          md_table(["项", "值"], [["PBO（78 配对）", pbo], ["DSR（78 试验）", dsr],
                                 ["相邻 9 格 ΔSharpe 范围", f"[{min(a[1] for a in adj):+.2f}, {max(a[1] for a in adj):+.2f}]"],
                                 ["相邻符号一致", adj_sign_ok]]),
          "\n## 4. 判据\n",
          md_table(["判据", "要求", "结果", "通过"],
                   [["A1 ΔSharpe>0", ">0", f"{P1:+.2f}", "✅" if crit["A1"] else "❌"],
                    ["A2 年度超额 CI 下界>0", ">0", f"{lo_p*100:+.2f}%", "✅" if crit["A2"] else "❌"],
                    ["A3 回撤更浅且真择时", "是", f"maxDD {mp['maxdd']*100:.1f}% vs {mbh['maxdd']*100:.1f}% / 真择时={timing_ok}", "✅" if crit["A3"] else "❌"],
                    ["A4 PBO<0.5", "<0.5", f"{pbo:.2f}", "✅" if crit["A4"] else "❌"],
                    ["A5 DSR≥0.95", "≥0.95", f"{dsr:.3f}", "✅" if crit["A5"] else "❌"],
                    ["A6 2010+ ΔSharpe>0", ">0", f"{dsh_2010:+.2f}", "✅" if crit["A6"] else "❌"],
                    ["A7 相邻符号一致", "一致", str(adj_sign_ok), "✅" if crit["A7"] else "❌"],
                    ["A8 双跑 md5", "一致", "外部比对", "—"]]),
          "\n## 5. 判决\n"]
    allpass = all(crit.values())
    md.append(f"- 程序 P（主假设）：{'🟢 全部通过' if allpass else '🔴 未全过'}；ΔSharpe {P1:+.2f}，"
              f"年度超额 CI [{lo_p*100:+.2f}%, {hi_p*100:+.2f}%]，PBO {pbo:.2f}，DSR {dsr:.3f}，MDE {mde:.3f}。")
    md.append("- 固定最优格仅描述（含选择偏差），不参与判决。")
    md.append("- 适用范围：恒指本快照/本窗口；**外推生产 ✗**。\n")
    with open(os.path.join(OUT, "result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[annual] 写入 {OUT}")
    print(f"  固定最优 {CN_MON[i_fix]}→{CN_MON[j_fix]}；P: Sharpe {mp['sharpe']:.2f} vs BH {mbh['sharpe']:.2f} "
          f"(Δ{P1:+.2f}) / CAGR {mp['cagr']*100:.2f}% vs {mbh['cagr']*100:.2f}%")
    print(f"  年度超额 CI [{lo_p*100:+.2f}%,{hi_p*100:+.2f}%] / PBO {pbo:.2f} / DSR {dsr:.3f} / MDE {mde:.3f}")
    print(f"  判据: {crit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
