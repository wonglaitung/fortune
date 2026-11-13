#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""港股「年初入市抓上半年升势 / 开门红」假设检验（描述性）。

命题（用户）：每年基金重新开始配置、香港有「开门红」习惯 ⇒ 在年初入市可抓到上半年升势。

检验分层
  Q1 开门红：1 月收益、年内首个交易日上涨概率、跨年窗口（去年末 5 日 + 今初 5 日）
  Q2 上半年 vs 下半年：H1(1–6月) / H2(7–12月) 收益，按年配对检验
  Q3 入市点：以上一年最后一个交易日（=年初）买入持有到 6 月底，vs 其他月末入市的 6 个月前瞻
  Q4 平均路径：从上年末归零，逐月的平均/中位累计收益（看升势集中在哪几个月）

统计口径
  - 恒指 yfinance `^HSI`；个股池 `data/hk_market_snapshot`（等权）
  - 窗口：全样本 / 2000+ / 2010+ / 2015+
  - 显著性：t 检验 + 移动块 bootstrap（块=3）+ Bonferroni（6 个主检验，α=0.05/6）
  - 报 MDE，避免把「不显著」当「不存在」
  - 明确：本报告是描述性；若要据此择时，须先走 hypothesis-validation 预注册 + 三道闸

用法：python3 scripts/year_start_rally.py [--out docs/YEAR_START_RALLY.md]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN_MON = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]
N_TESTS = 6
ALPHA_EFF = 0.05 / N_TESTS


def load_hsi() -> pd.Series:
    import yfinance as yf
    h = yf.Ticker("^HSI").history(period="max", interval="1d", auto_adjust=True, timeout=30)
    if h.empty:
        raise RuntimeError("HSI 取数失败")
    h.index = pd.to_datetime(h.index).tz_localize(None)
    s = h["Close"].sort_index()
    return s[s.index <= pd.Timestamp.now().normalize()]


def load_pool() -> pd.Series:
    """个股池等权月度收益序列（PeriodIndex）。"""
    cur = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")
    cols = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data/hk_market_snapshot/*.pkl"))):
        code = os.path.splitext(os.path.basename(f))[0]
        try:
            d = pd.read_pickle(f)
        except Exception:
            continue
        if "Close" not in d:
            continue
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        me = s.resample("M").last()
        r = me.pct_change()
        r.index = r.index.to_period("M")
        cols[code] = r[r.index <= cur]
    mat = pd.DataFrame(cols)
    return mat.mean(axis=1).dropna()


def block_bootstrap_ci(x: np.ndarray, block: int = 3, n_boot: int = 5000,
                       seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=k)
        vals = np.concatenate([x[np.arange(s, s + block) % n] for s in starts])[:n]
        means[i] = vals.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


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


def bar_chart(labels, vals, width=34) -> str:
    mx = max(abs(v) for v in vals) or 1
    lines = []
    for lab, v in zip(labels, vals):
        n = int(round(abs(v) / mx * width))
        bar = ("+" * n) if v >= 0 else ("-" * n)
        lines.append(f"{lab:>4} | {bar.ljust(width)} {v * 100:+6.2f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def h1_h2_table(s: pd.Series, lo_year: int) -> tuple[list, float, float]:
    """按年算 H1 / H2（年末最后一个交易日为界）。"""
    yearly = s.groupby(s.index.year)
    rows = []
    for yr, sub in yearly:
        if yr < lo_year:
            continue
        dec = s[(s.index.year == yr - 1) & (s.index.month == 12)]
        jun = s[(s.index.year == yr) & (s.index.month == 6)]
        if dec.empty or jun.empty:
            continue
        h1 = jun.iloc[-1] / dec.iloc[-1] - 1
        dec_cur = sub[sub.index.month == 12]
        if dec_cur.empty:
            continue
        h2 = dec_cur.iloc[-1] / jun.iloc[-1] - 1
        rows.append([yr, h1, h2, h1 - h2])
    return rows


def pct_path(s: pd.Series, lo_year: int) -> dict[int, list[float]]:
    """每年从上年末归零、逐月末累计收益；返回 {month: [values]}。"""
    out = {m: [] for m in range(1, 13)}
    for yr, sub in s.groupby(s.index.year):
        if yr < lo_year:
            continue
        dec = s[(s.index.year == yr - 1) & (s.index.month == 12)]
        if dec.empty:
            continue
        base = dec.iloc[-1]
        for m in range(1, 13):
            mm = sub[sub.index.month == m]
            if not mm.empty:
                out[m].append(mm.iloc[-1] / base - 1)
    return out


def first_day_stats(s: pd.Series, lo_year: int) -> list:
    rows = []
    for yr, sub in s.groupby(s.index.year):
        if yr < lo_year:
            continue
        jan = sub[sub.index.month == 1]
        dec_prev = s[(s.index.year == yr - 1) & (s.index.month == 12)]
        if jan.empty or dec_prev.empty:
            continue
        first = jan.iloc[0] / dec_prev.iloc[-1] - 1
        first5 = jan.iloc[min(4, len(jan) - 1)] / dec_prev.iloc[-1] - 1
        rows.append([yr, first, first5])
    return rows


def turn_of_year(s: pd.Series, lo_year: int) -> list:
    rows = []
    for yr, sub in s.groupby(s.index.year):
        if yr < lo_year:
            continue
        tail = s[(s.index.year == yr - 1) & (s.index.month == 12)]
        head = sub[sub.index.month == 1]
        if len(tail) < 6 or head.empty:
            continue
        anchor = tail.iloc[-6]   # 去年末第 6 个交易日起
        end = head.iloc[min(4, len(head) - 1)]
        rows.append([yr, end / anchor - 1])
    return rows


def entry_6m(s: pd.Series, lo_year: int) -> dict[int, list[float]]:
    """每个日历月末买入、持 6 个自然月（用月末序列）。"""
    me = s.resample("M").last().dropna()
    mep = me.copy(); mep.index = mep.index.to_period("M")
    out = {m: [] for m in range(1, 13)}
    for p in mep.index:
        if p.year < lo_year:
            continue
        tgt = p + 6
        if tgt in mep.index:
            out[p.month].append(float(mep[tgt] / mep[p] - 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/YEAR_START_RALLY.md"))
    args = ap.parse_args()

    s = load_hsi()
    pool = load_pool()
    pull = datetime.now().strftime("%Y-%m-%d %H:%M")
    windows = {"全样本": 1987, "2000 年后": 2000, "2010 年后": 2010, "2015 年后": 2015}

    md: list[str] = []
    md.append("# 港股「年初入市 / 开门红 / 上半年升势」检验\n")
    md.append(f"> 生成：{pull}｜恒指 `^HSI` {s.index.min().date()} → {s.index.max().date()}"
              f"（{len(s)} 交易日）｜个股池等权（`data/hk_market_snapshot`）\n")
    md.append("> **命题**：每年基金重新配置、香港有「开门红」习惯 ⇒ 年初入市可抓上半年升势。\n")
    md.append(f"> 多重比较：{N_TESTS} 个主检验，Bonferroni α = 0.05/{N_TESTS} = {ALPHA_EFF:.4f}。\n")
    md.append("> **适用范围**：恒指 + 本项目训练池；描述性统计，**不构成买卖建议**；"
              "要部署须先走 hypothesis-validation 预注册 + 三道闸。\n")

    # ---------- Q1 开门红 ----------
    md.append("\n## 1. Q1：开门红（1 月效应）\n")
    r = s.resample("M").last().pct_change().dropna()
    r.index = r.index.to_period("M")
    jan_all = r[r.index.month == 1]

    rows = []
    for w, lo in windows.items():
        x = jan_all[jan_all.index.year >= lo]
        t, p = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
        lo_ci, hi_ci = block_bootstrap_ci(x.to_numpy())
        rows.append([w, len(x), x.mean(), (x > 0).mean(), lo_ci, hi_ci, t, p])
    md.append("### 1.1 1 月收益\n")
    md.append(md_table(["窗口", "n", "均值", "上涨占比", "CI95_lo", "CI95_hi", "t", "p"], rows))

    # 1 月 vs 其余月份均值
    dfm = pd.DataFrame({"r": r}); dfm["m"] = dfm.index.month
    jan = dfm.loc[dfm.m == 1, "r"]; rest = dfm.loc[dfm.m != 1, "r"]
    t1, p1 = stats.ttest_ind(jan, rest, equal_var=False)
    md.append(f"\n- 1 月均值 {jan.mean() * 100:+.2f}% vs 其余月份 {rest.mean() * 100:+.2f}%；"
              f"差异检验 t={t1:.2f}, p={p1:.3f}（n={len(jan)}）。")

    # 首个交易日 / 前 5 交易日 / 跨年窗口
    md.append("\n### 1.2 年内首个交易日、前 5 日、跨年窗口（去年末 5 日 + 今初 5 日）\n")
    for w, lo in windows.items():
        fd = first_day_stats(s, lo)
        if not fd:
            continue
        d1 = np.array([x[1] for x in fd]); d5 = np.array([x[2] for x in fd])
        b1 = stats.binomtest(int((d1 > 0).sum()), len(d1), 0.5)
        ty = np.array([x[1] for x in turn_of_year(s, lo)])
        tt, tp = stats.ttest_1samp(ty, 0.0)
        md.append(f"- **{w}**：首日上涨 {int((d1 > 0).sum())}/{len(d1)}（{ (d1 > 0).mean()*100:.0f}%，"
                  f"二项 p={b1.pvalue:.3f}），首日均值 {d1.mean()*100:+.2f}%；"
                  f"前 5 日均值 {d5.mean()*100:+.2f}%；"
                  f"跨年窗口均值 {ty.mean()*100:+.2f}%（t={tt:.2f}, p={tp:.3f}）。")

    # ---------- Q2 H1 vs H2 ----------
    md.append("\n## 2. Q2：上半年（1–6 月）是否强于下半年（7–12 月）\n")
    md.append(md_table(
        ["窗口", "n", "H1 均值", "H1 上涨占比", "H2 均值", "H1−H2 均值",
         "配对 t", "p(配对)", "H1 CI95"],
        _h1h2_all(s, windows)))

    # ---------- Q3 入市点 ----------
    md.append("\n## 3. Q3：何时入市能抓到上半年升势\n")
    md.append("口径 A：**上一年最后一个交易日（=年初）买入**，持有到 6 月底（=H1 收益）；"
              "口径 B：各日历月末买入、持 6 个自然月。\n")
    for w, lo in windows.items():
        rows = h1_h2_table(s, lo)
        if not rows:
            continue
        h1 = np.array([x[1] for x in rows])
        t, p = stats.ttest_1samp(h1, 0.0)
        lo_ci, hi_ci = block_bootstrap_ci(h1)
        md.append(f"- **{w}**：年初入市 H1 均值 **{h1.mean()*100:+.2f}%**（n={len(h1)}，"
                  f"上涨占比 {(h1>0).mean()*100:.0f}%，CI95 [{lo_ci*100:+.2f}%, {hi_ci*100:+.2f}%]，"
                  f"t={t:.2f}, p={p:.3f}）。")

    md.append("\n### 3.1 各月末入市、持 6 个月的均值（全样本）\n")
    e6 = entry_6m(s, 1987)
    rows = []
    for m in range(1, 13):
        x = np.array(e6[m])
        if len(x) < 2:
            rows.append([CN_MON[m - 1], len(x), np.nan, np.nan, np.nan, np.nan])
            continue
        t, p = stats.ttest_1samp(x, 0.0)
        lo_ci, hi_ci = block_bootstrap_ci(x)
        rows.append([CN_MON[m - 1], len(x), x.mean(), (x > 0).mean(), lo_ci, hi_ci])
    md.append(md_table(["入市月(月末)", "n", "6个月均值", "正收益占比", "CI95_lo", "CI95_hi"], rows))
    md.append("\n```\n" + bar_chart(CN_MON, [x[2] for x in rows]) + "\n```\n")
    best = max([x for x in rows if not np.isnan(x[2])], key=lambda z: z[2])
    md.append(f"- 名义最优月末入市月：**{best[0]}**（6 个月均值 {best[2]*100:+.2f}%，n={best[1]}）。")
    md.append("- 注意：「年初入市」= 上年 12 月末入市；表中「12月」即此口径。\n")

    # ---------- Q4 平均路径 ----------
    md.append("\n## 4. Q4：从上年末归零的月度平均累计收益（升势集中在哪几个月）\n")
    for w, lo in windows.items():
        path = pct_path(s, lo)
        means = [np.mean(path[m]) if path[m] else np.nan for m in range(1, 13)]
        meds = [np.median(path[m]) if path[m] else np.nan for m in range(1, 13)]
        md.append(f"\n### {w}（n={len(path[6])} 年）\n")
        md.append(md_table(["月"] + CN_MON,
                           [["均值"] + means, ["中位数"] + meds]))
        md.append("\n```\n" + bar_chart(CN_MON, means) + "\n```\n")

    # ---------- Q5 个股池 ----------
    md.append("\n## 5. 个股池等权验证（月度）\n")
    if len(pool) > 0:
        pr = pd.DataFrame({"r": pool}); pr["m"] = pr.index.month
        jp = pr.loc[pr.m == 1, "r"]
        h1p = []
        h2p = []
        for yr in sorted(set(pool.index.year)):
            months = pool[pool.index.year == yr]
            if len(months) < 12:
                continue
            h1p.append(float((1 + months[months.index.month <= 6]).prod() - 1))
            h2p.append(float((1 + months[months.index.month >= 7]).prod() - 1))
        h1p, h2p = np.array(h1p), np.array(h2p)
        t, p = stats.ttest_rel(h1p, h2p)
        th, ph = stats.ttest_1samp(h1p, 0.0)
        md.append(f"- 1 月均值 {jp.mean()*100:+.2f}%（n={len(jp)} 年，上涨占比 {(jp>0).mean()*100:.0f}%）。")
        md.append(f"- H1 均值 {h1p.mean()*100:+.2f}% vs H2 {h2p.mean()*100:+.2f}%；"
                  f"H1 对 0 t={th:.2f} p={ph:.3f}；H1−H2 配对 t={t:.2f} p={p:.3f}（n={len(h1p)}）。")

    # ---------- 结论 ----------
    # 计算全样本 + 2010+ 关键数字供结论引用
    def stat_block(lo):
        rows = h1_h2_table(s, lo)
        h1 = np.array([x[1] for x in rows]); h2 = np.array([x[2] for x in rows])
        th, ph = stats.ttest_1samp(h1, 0.0)
        td, pd_ = stats.ttest_rel(h1, h2)
        mde = stats.t.ppf(0.975, len(h1) - 1) * h1.std() / np.sqrt(len(h1))
        return h1.mean(), h2.mean(), th, ph, td, pd_, mde, len(h1)

    a = stat_block(1987); b = stat_block(2010)
    md.append("\n## 6. 结论（条件化）\n")
    md.append(f"- **H1 是否为正**：全样本 H1 均值 {a[0]*100:+.2f}%（t={a[2]:.2f}, p={a[3]:.3f}）；"
              f"2010 年后 {b[0]*100:+.2f}%（t={b[2]:.2f}, p={b[3]:.3f}）。")
    md.append(f"- **H1 是否强于 H2**：全样本 H2 均值 {a[1]*100:+.2f}%，H1−H2 配对 p={a[5]:.3f}；"
              f"2010 年后 p={b[5]:.3f}。")
    md.append(f"- **「年初入市抓上半年」能否成立**：需同时满足「H1 显著为正」且「H1 显著强于 H2」。"
              f"Bonferroni α={ALPHA_EFF:.4f}；见上表 p 值判定。")
    md.append(f"- **功效**：H1 的 MDE（全样本）≈ {a[6]*100:.2f}%（n={a[7]} 年）；"
              f"2010 年后 MDE ≈ {b[6]*100:.2f}%。小于此量级的季节性无法被本样本探测。")
    md.append("- **机制证据缺口**：南向资金数据仅 2024-11 起（约 2 年）⇒ 无法检验「年初新钱流入」；"
              "本报告仅检验价格形态，未证实资金面机制。")
    md.append("- **落地边界**：「H1 持有、H2 空仓」是**样本内季节性策略**，未经三道闸/双跑；"
              "若要用，必须先预注册（判据写死）+ 三冻结 + 双跑 md5，并同报适用范围/MDE/相邻参数一致性。\n")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[year_start] 报告写入 {out_path}")
    return 0


def _h1h2_all(s: pd.Series, windows: dict) -> list:
    rows = []
    for w, lo in windows.items():
        data = h1_h2_table(s, lo)
        if not data:
            continue
        h1 = np.array([x[1] for x in data]); h2 = np.array([x[2] for x in data])
        t, p = stats.ttest_rel(h1, h2)
        lo_ci, hi_ci = block_bootstrap_ci(h1)
        rows.append([w, len(h1), h1.mean(), (h1 > 0).mean(), h2.mean(),
                     (h1 - h2).mean(), t, p, f"[{lo_ci*100:+.2f}%, {hi_ci*100:+.2f}%]"])
    return rows


def _h1h2_rows(*a, **k):
    return []


if __name__ == "__main__":
    sys.exit(main())
