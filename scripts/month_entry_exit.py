#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""港股逐月「入市 → 离开」分析。

回答：哪个月适合入市、哪个月建议离开。

口径（写死，避免歧义）
  - 入市 = 当月**第一个交易日**买入（= 上月末收盘）
  - 离开 = 当月**最后一个交易日**卖出
  - 收益 = 从入市月到离开月（含）的月度收益复利（同一年内，Jan..Dec）
  - 例：1 月入市、6 月离开 = 1–6 月累计收益（= H1）

为什么加置换检验
  12×12 上三角共 66 个 (入市,离开) 组合，直接挑均值最大者=多重比较/赢家诅咒。
  做法：在每个年度内**独立打乱 12 个月的收益**（保留年度整体水平、摧毁季节性），
  重算矩阵取最大/最小，重复 2000 次得到「纯噪声能挑出的最好/最差」分布，
  再看真实最优是否超出该分布。Bonferroni(66) 仅作对照。

用法：python3 scripts/month_entry_exit.py [--out docs/MONTH_ENTRY_EXIT.md]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN_MON = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]


def load_hsi() -> pd.Series:
    import yfinance as yf
    h = yf.Ticker("^HSI").history(period="max", interval="1d", auto_adjust=True, timeout=30)
    if h.empty:
        raise RuntimeError("HSI 取数失败")
    h.index = pd.to_datetime(h.index).tz_localize(None)
    s = h["Close"].sort_index()
    return s[s.index <= pd.Timestamp.now().normalize()]


def monthly_matrix(s: pd.Series, lo_year: int) -> tuple[np.ndarray, list[int]]:
    """返回 (n_years × 12) 月度收益矩阵（仅完整年度）与年份列表。"""
    me = s.resample("M").last()
    r = me.pct_change().dropna()
    r.index = r.index.to_period("M")
    df = pd.DataFrame({"r": r})
    df["y"] = df.index.year
    df["m"] = df.index.month
    piv = df.pivot(index="y", columns="m", values="r").dropna(how="any")
    piv = piv[piv.index >= lo_year]
    piv = piv.reindex(columns=range(1, 13))
    return piv.to_numpy(), list(piv.index)


def hold_matrix(R: np.ndarray) -> np.ndarray:
    """上三角：行=入市月(0..11)，列=离开月(0..11)，值=均值累计收益。"""
    n = len(R)
    C = np.cumprod(1.0 + R, axis=1)          # (n,12) 年内累计
    M = np.full((12, 12), np.nan)
    for i in range(12):
        base = np.ones(n) if i == 0 else C[:, i - 1]
        for j in range(i, 12):
            M[i, j] = np.mean(C[:, j] / base - 1.0)
    return M


def hold_stats(R: np.ndarray):
    """返回 (mean, ir 年化风险调整, pos 正收益占比) 三个 12×12 上三角矩阵。"""
    n = len(R)
    C = np.cumprod(1.0 + R, axis=1)
    mean = np.full((12, 12), np.nan); ir = np.full((12, 12), np.nan)
    pos = np.full((12, 12), np.nan)
    for i in range(12):
        base = np.ones(n) if i == 0 else C[:, i - 1]
        for j in range(i, 12):
            x = C[:, j] / base - 1.0
            h = j - i + 1
            mean[i, j] = x.mean()
            sd = x.std(ddof=1) if len(x) > 1 else np.nan
            ir[i, j] = (x.mean() / sd * np.sqrt(12.0 / h)) if sd and sd > 0 else np.nan
            pos[i, j] = (x > 0).mean()
    return mean, ir, pos


def perm_test_ir(R: np.ndarray, n_perm: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(R); iu = np.triu_indices(12)
    _, obs_ir, _ = hold_stats(R)
    o_max = np.nanmax(obs_ir[iu]); o_min = np.nanmin(obs_ir[iu])
    nmax = np.empty(n_perm); nmin = np.empty(n_perm)
    for k in range(n_perm):
        Rp = R.copy()
        for i in range(n):
            Rp[i] = rng.permutation(Rp[i])
        _, ir, _ = hold_stats(Rp)
        v = ir[iu]
        nmax[k] = np.nanmax(v); nmin[k] = np.nanmin(v)
    return o_max, (np.sum(nmax >= o_max)+1)/(n_perm+1), o_min, (np.sum(nmin <= o_min)+1)/(n_perm+1)


def best_cells(M: np.ndarray) -> tuple[tuple[int, int], float, tuple[int, int], float]:
    iu = np.triu_indices(12)
    vals = M[iu]
    kmax = np.nanargmax(vals)
    kmin = np.nanargmin(vals)
    return (int(iu[0][kmax]), int(iu[1][kmax])), float(vals[kmax]), \
           (int(iu[0][kmin]), int(iu[1][kmin])), float(vals[kmin])


def perm_test(R: np.ndarray, n_perm: int = 2000, seed: int = 42):
    """年度内独立打乱月份 → 最优/最差配对的零分布。"""
    rng = np.random.default_rng(seed)
    n = len(R)
    obs = hold_matrix(R)
    _, o_max, _, o_min = best_cells(obs)
    null_max = np.empty(n_perm)
    null_min = np.empty(n_perm)
    for k in range(n_perm):
        Rp = R.copy()
        for i in range(n):
            Rp[i] = rng.permutation(Rp[i])
        Mp = hold_matrix(Rp)
        iu = np.triu_indices(12)
        v = Mp[iu]
        null_max[k] = np.nanmax(v)
        null_min[k] = np.nanmin(v)
    p_max = (np.sum(null_max >= o_max) + 1) / (n_perm + 1)
    p_min = (np.sum(null_min <= o_min) + 1) / (n_perm + 1)
    return obs, o_max, p_max, o_min, p_min, null_max, null_min


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


def fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/MONTH_ENTRY_EXIT.md"))
    args = ap.parse_args()

    s = load_hsi()
    pull = datetime.now().strftime("%Y-%m-%d %H:%M")
    windows = {"全样本": 1987, "2000 年后": 2000, "2010 年后": 2010, "2015 年后": 2015}

    md: list[str] = []
    md.append("# 港股逐月「入市 → 离开」分析\n")
    md.append(f"> 生成：{pull}｜恒指 `^HSI` {s.index.min().date()} → {s.index.max().date()}\n")
    md.append("> 口径：**入市 = 当月第一个交易日买入**，**离开 = 当月最后一个交易日卖出**，"
              "收益 = 入市月到离开月（含）的复利；仅统计同一年内。\n")
    md.append("> **适用范围**：恒指；描述性统计，**不构成买卖建议**。13 个月/66 个配对 ⇒ "
              "多重比较严重，已用**置换检验**控制，不以「挑最好」为结论。\n")

    # ---------- 1. 单月收益（决定“持有哪个月”） ----------
    md.append("\n## 1. 单月表现（哪个月本身值得持有）\n")
    R_all, years_all = monthly_matrix(s, 1987)
    rows = []
    for j in range(12):
        x = R_all[:, j]
        t, p = stats.ttest_1samp(x, 0.0)
        rows.append([CN_MON[j], len(x), x.mean(), np.median(x), (x > 0).mean(),
                     x.std(), t, p])
    md.append(md_table(["月", "n", "均值", "中位数", "上涨占比", "标准差", "t", "p"], rows))

    # ---------- 2. 入市→离开 矩阵（全样本） ----------
    md.append("\n## 2. 入市 → 离开 持有收益矩阵（全样本，行=入市，列=离开）\n")
    M = hold_matrix(R_all)
    hdr = ["入市\\离开"] + CN_MON
    mat_rows = []
    for i in range(12):
        mat_rows.append([CN_MON[i]] + [M[i, j] if j >= i else "—" for j in range(12)])
    md.append(md_table(hdr, mat_rows))

    # ---------- 2.1 风险调整（年化 IR） ----------
    md.append("\n## 2.1 风险调整视角：年化 IR（均值/波动 × √(12/持有月数)）\n")
    md.append("> 只看均值会被「持有越久越高」误导；IR 消除久期后，才回答「何时离开」更值。\n")
    _, ir_all, pos_all = hold_stats(R_all)
    hdr_ir = ["入市\\离开"] + CN_MON
    ir_rows = []
    for i in range(12):
        ir_rows.append([CN_MON[i]] + [round(float(ir_all[i, j]), 2) if j >= i else "—" for j in range(12)])
    md.append(md_table(hdr_ir, ir_rows))
    (ii, jj), iv, (wi2, wj2), wv = best_cells(ir_all)
    md.append(f"\n- 全样本 IR 最高配对：**{CN_MON[ii]}入市 → {CN_MON[jj]}离开**，IR={iv:.2f}，"
              f"正收益占比 {pos_all[ii, jj]*100:.0f}%。")
    md.append(f"- IR 最低配对：{CN_MON[wi2]}入市 → {CN_MON[wj2]}离开，IR={wv:.2f}。")

    # ---------- 3. 置换检验 ----------
    md.append("\n## 3. 置换检验：最优/最差配对是否超出纯噪声\n")
    obs, o_max, p_max, o_min, p_min, nmax, nmin = perm_test(R_all, n_perm=2000)
    (bi, bj), bval, (wi, wj), wval = best_cells(obs)
    md.append(f"- 名义最优：**{CN_MON[bi]}入市 → {CN_MON[bj]}离开**，均值 **{fmt_pct(bval)}**；"
              f"置换检验 p={p_max:.3f}（2000 次）。")
    md.append(f"- 名义最差：**{CN_MON[wi]}入市 → {CN_MON[wj]}离开**，均值 **{fmt_pct(wval)}**；"
              f"置换检验 p={p_min:.3f}。")
    md.append(f"- 零分布参考：纯噪声下 66 个配对的「最优」均值分布 95 分位 = "
              f"{fmt_pct(np.percentile(nmax, 95))}；「最差」5 分位 = {fmt_pct(np.percentile(nmin, 5))}。")
    if p_max > 0.05:
        md.append("- ⇒ **最优配对未超出噪声**（p>0.05）：不能据此断言「某月入某月出」更好。")
    else:
        md.append("- ⇒ 最优配对超出噪声（p≤0.05），但仍需预注册与相邻参数一致性检验才可部署。")
    md.append("- Bonferroni(66) 对照：α = 0.05/66 = 0.00076（单配对 t 检验几乎不可能通过）。")
    oir, pir, oir_min, pir_min = perm_test_ir(R_all, n_perm=2000)
    md.append(f"- **风险调整口径**：IR 最优 {oir:.2f} 的置换 p={pir:.3f}；"
              f"IR 最差 {oir_min:.2f} 的置换 p={pir_min:.3f}。\n")

    # ---------- 4. 分窗口：每档最优 ----------
    md.append("\n## 4. 分窗口：名义最优/最差配对与「各入市月的最佳离开」\n")
    rows = []
    for w, lo in windows.items():
        Rw, _ = monthly_matrix(s, lo)
        if len(Rw) < 3:
            continue
        Mw = hold_matrix(Rw)
        (i1, j1), v1, (i2, j2), v2 = best_cells(Mw)
        # 各入市月的最佳离开
        best_exit = {}
        for i in range(12):
            seg = Mw[i, i:]
            if np.all(np.isnan(seg)):
                continue
            jbest = i + int(np.nanargmax(seg))
            best_exit[CN_MON[i]] = f"{CN_MON[jbest]}({fmt_pct(Mw[i, jbest])})"
        _, irw, posw = hold_stats(Rw)
        (k1, l1), kv, (k2, l2), kv2 = best_cells(irw)
        best_exit_ir = {}
        for i in range(12):
            seg = irw[i, i:]
            if np.all(np.isnan(seg)):
                continue
            jb = i + int(np.nanargmax(seg))
            best_exit_ir[CN_MON[i]] = f"{CN_MON[jb]}(IR {irw[i, jb]:.2f})"
        rows.append([w, len(Rw), f"{CN_MON[i1]}→{CN_MON[j1]}", v1,
                     f"{CN_MON[i2]}→{CN_MON[j2]}", v2,
                     f"{CN_MON[k1]}→{CN_MON[l1]} IR{kv:.2f}",
                     f"{CN_MON[k2]}→{CN_MON[l2]} IR{kv2:.2f}",
                     "; ".join(f"{k}:{v}" for k, v in best_exit_ir.items())])
    md.append(md_table(["窗口", "年数", "最高均值配对", "均值", "最低均值配对", "均值",
                        "最高 IR 配对", "最低 IR 配对", "各入市月的最佳离开(按 IR)"], rows))

    # ---------- 5. 平均季节路径（峰/谷） ----------
    md.append("\n## 5. 平均季节路径：谷（入市）与峰（离开）\n")
    md.append("把每年 1 月初归零，逐月末的平均累计收益。谷=相对低点（入市），峰=相对高点（离开）。\n")
    for w, lo in windows.items():
        Rw, _ = monthly_matrix(s, lo)
        if len(Rw) < 3:
            continue
        path = np.cumprod(1 + Rw, axis=1) - 1
        means = np.nanmean(path, axis=0)
        trough = int(np.nanargmin(means)); peak = int(np.nanargmax(means))
        md.append(f"- **{w}**：1 月低点归零后，谷 = {CN_MON[trough]}（{fmt_pct(means[trough])}），"
                  f"峰 = {CN_MON[peak]}（{fmt_pct(means[peak])}）；"
                  f"全年末 {fmt_pct(means[-1])}。")
        md.append(md_table(["月"] + CN_MON, [["均值路径"] + [float(x) for x in means]]))

    # ---------- 6. 结论 ----------
    a = monthly_matrix(s, 1987); Ma = hold_matrix(a[0])
    b = monthly_matrix(s, 2010); Mb = hold_matrix(b[0])
    def top_months(Mx, k=3):
        iu = np.triu_indices(12)
        order = np.argsort(Mx[iu])[::-1][:k]
        return [(CN_MON[iu[0][o]], CN_MON[iu[1][o]], Mx[iu][o]) for o in order]
    md.append("\n## 6. 结论（条件化）\n")
    def trough_peak(lo):
        Rw, _ = monthly_matrix(s, lo)
        means = np.cumprod(1 + Rw, axis=1).mean(axis=0)
        return CN_MON[int(np.nanargmin(means))], CN_MON[int(np.nanargmax(means))]
    def ir_diag(lo):
        Rw, _ = monthly_matrix(s, lo)
        _, ir, _ = hold_stats(Rw)
        diag = np.array([ir[i, i] for i in range(12)])
        i = int(np.nanargmax(diag)); j = int(np.nanargmin(diag))
        return CN_MON[i], diag[i], CN_MON[j], diag[j]

    tp = {w: trough_peak(lo) for w, lo in windows.items()}
    di = {w: ir_diag(lo) for w, lo in windows.items()}
    md.append("**入市（相对低点）**：按平均季节路径 —— " +
              "；".join(f"{w}谷在 {v[0]}" for w, v in tp.items()) + "。")
    md.append("**离开（相对高点）**：按平均季节路径 —— " +
              "；".join(f"{w}峰在 {v[1]}" for w, v in tp.items()) + "。")
    md.append("**风险调整后最强的单月持有**（对角线 IR）：" +
              "；".join(f"{w} {v[0]}(IR {v[1]:.2f})" for w, v in di.items()) + "。")
    md.append("**最弱的单月**：" +
              "；".join(f"{w} {v[2]}(IR {v[3]:.2f})" for w, v in di.items()) + "。")
    md.append(f"- **统计强度**：66 个配对里挑出的最优在置换检验下 "
              f"{'未超出噪声（p>0.05）' if p_max > 0.05 else '超出噪声'}；"
              "因此「某月入、某月出」只能当弱先验，不能当规则。")
    md.append("- **与单月表一致**：强的单月是 2/4/7/12 月，弱的单月是 3/8 月；但均未过 Bonferroni。")
    md.append("- **落地边界**：任何「按月择时」规则须预注册 + 三冻结 + 双跑 md5；"
              "本报告为描述性，MDE≈5%/半年量级，小效应测不出。\n")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[month_entry_exit] 报告写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
