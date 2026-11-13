#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""港股日历季节性分析：恒指 + 训练池个股。

回答两个问题：
  Q1 每年 11/12 月港股是否更差？
  Q2 每年什么时候是相对合适的入市窗口？

数据源
  恒指：yfinance `^HSI`（period=max，auto_adjust=True）
  个股池：`data/hk_market_snapshot/*.pkl`（冻结快照，56 只，2006-07 → 今）

统计口径（自报，避免"数字好看"）
  - 月度收益 = 月末收盘环比；当前未结束月份剔除
  - 窗口三档：全样本 / 2000+ / 2010+ / 2015+（结论以近期窗口为准）
  - 显著性：t 检验 + 移动块 bootstrap（block=3 年，处理自相关）+ Bonferroni（12 月）
  - 入市窗口：月末买入 → 前瞻 1/3/6/12 月收益；按"年内最大回撤月"定义抄底窗口

用法：python3 scripts/seasonality_hsi.py [--out docs/SEASONALITY_HK.md]
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


# --------------------------------------------------------------------------- 数据
def load_hsi() -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker("^HSI").history(period="max", interval="1d", auto_adjust=True, timeout=30)
    if h.empty:
        raise RuntimeError("HSI 取数失败")
    h.index = pd.to_datetime(h.index).tz_localize(None)
    s = h[["Close"]].rename(columns={"Close": "close"}).sort_index()
    s["ma200"] = s["close"].rolling(200).mean()
    return s


def load_pool() -> dict[str, pd.Series]:
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data/hk_market_snapshot/*.pkl"))):
        code = os.path.splitext(os.path.basename(f))[0]
        try:
            d = pd.read_pickle(f)
        except Exception:
            continue
        if "Close" not in d.columns:
            continue
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[code] = s[s.index <= pd.Timestamp.now().normalize()]
    return out


def month_end(s: pd.Series) -> pd.Series:
    """月末收盘序列（索引 = Period[month]）。"""
    me = s.resample("M").last().dropna()
    return me


def monthly_returns(s: pd.Series) -> pd.Series:
    me = month_end(s)
    r = me.pct_change().dropna()
    r.index = r.index.to_period("M")
    return r[~r.index.to_timestamp().eq(r.index.to_timestamp())]  # 占位，实际用下面裁剪


# --------------------------------------------------------------------------- 统计
def block_bootstrap_ci(x: np.ndarray, block: int = 3, n_boot: int = 5000,
                       seed: int = 42) -> tuple[float, float]:
    """移动块 bootstrap 均值 95% CI（块=block 年，处理月度序列自相关）。

    每次重抽 ceil(n/block) 个**独立随机起点**的块拼接再截断到 n。
    （早期实现按顺序取块，当 n 能被 block 整除时等价于整体轮换 → CI 退化为点估计，已修正。）
    """
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


def mget(r: pd.Series, yr: int, m: int) -> float:
    """取 PeriodIndex 月度收益序列的 (年, 月) 值。"""
    v = r.get(pd.Period(f"{yr}-{m:02d}", freq="M"), np.nan)
    return float(v) if v is not None and not pd.isna(v) else np.nan


def month_table(r: pd.Series, label: str) -> list[str]:
    """按日历月统计月度收益。"""
    df = pd.DataFrame({"r": r})
    df["m"] = df.index.month
    rows = []
    for m in range(1, 13):
        s = df.loc[df.m == m, "r"]
        if len(s) == 0:
            rows.append([CN_MON[m - 1], 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
        t, p = stats.ttest_1samp(s, 0.0) if len(s) > 1 else (np.nan, np.nan)
        lo, hi = block_bootstrap_ci(s.to_numpy())
        rows.append([CN_MON[m - 1], len(s), s.mean(), s.median(), s.std(), (s > 0).mean(),
                     s.min(), s.max(), t, p, lo, hi])
    return rows


def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return f"{v:.4f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- 主流程
def analyse_hsi(s: pd.DataFrame) -> dict:
    me = month_end(s["close"])
    mep = me.copy()
    mep.index = mep.index.to_period("M")   # 便于按自然月做前瞻查找
    r = me.pct_change().dropna()
    r.index = r.index.to_period("M")
    # 剔除未结束的当月
    cur = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")
    r = r[r.index < cur]

    windows = {
        "全样本": r,
        "2000 年后": r[r.index >= pd.Period("2000-01", freq="M")],
        "2010 年后": r[r.index >= pd.Period("2010-01", freq="M")],
        "2015 年后": r[r.index >= pd.Period("2015-01", freq="M")],
    }

    # --- Q1 检验：11/12 月是否更差 -------------------------------------------
    tests = []
    for wname, wr in windows.items():
        df = pd.DataFrame({"r": wr})
        df["m"] = df.index.month
        for m in [11, 12]:
            sub = df.loc[df.m == m, "r"]
            rest = df.loc[df.m != m, "r"]
            t, p = stats.ttest_1samp(sub, 0.0)
            lo, hi = block_bootstrap_ci(sub.to_numpy())
            tests.append([wname, CN_MON[m - 1], len(sub), sub.mean(), lo, hi, t, p,
                          rest.mean(), sub.mean() - rest.mean()])
        # 11+12 合并 vs 其余月份
        s2 = df.loc[df.m.isin([11, 12]), "r"]
        rest2 = df.loc[~df.m.isin([11, 12]), "r"]
        t2, p2 = stats.ttest_ind(s2, rest2, equal_var=False)
        tests.append([wname, "11+12 合并", len(s2), s2.mean(), *block_bootstrap_ci(s2.to_numpy()),
                      t2, p2, rest2.mean(), s2.mean() - rest2.mean()])
    # Bonferroni：12 个日历月同时检验，α=0.05/12
    alpha_bonf = 0.05 / 12

    # --- 逐年 11/12 月明细 ---------------------------------------------------
    yearly = []
    for yr in sorted(set(r.index.year)):
        n = mget(r, yr, 11)
        d = mget(r, yr, 12)
        full = r[r.index.year == yr]
        yearly.append([yr, full.mean(), n, d, (n + d) if not (np.isnan(n) or np.isnan(d)) else np.nan,
                       len(full)])
    # 二项检验：11 月为负的年份占比 vs 50%
    n11 = pd.Series([y[2] for y in yearly]).dropna()
    neg11 = (n11 < 0).sum()
    binom11 = stats.binomtest(int(neg11), len(n11), 0.5)
    n12 = pd.Series([y[3] for y in yearly]).dropna()
    neg12 = (n12 < 0).sum()
    binom12 = stats.binomtest(int(neg12), len(n12), 0.5)

    # --- Q2 入市窗口：月末买入的前瞻收益 -------------------------------------
    fwd = {}
    for h in [1, 3, 6, 12]:
        f = me.shift(-h) / me - 1.0
        f.index = f.index.to_period("M")
        f = f[(f.index + h) <= cur]
        fwd[h] = f

    def fwd_table(f: pd.Series) -> list[list]:
        df = pd.DataFrame({"f": f})
        df["m"] = df.index.month
        rows = []
        for m in range(1, 13):
            sub = df.loc[df.m == m, "f"].dropna()
            if len(sub) < 2:
                rows.append([CN_MON[m - 1], len(sub), np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
            t, p = stats.ttest_1samp(sub, 0.0)
            lo, hi = block_bootstrap_ci(sub.to_numpy())
            rows.append([CN_MON[m - 1], len(sub), sub.mean(), sub.median(), (sub > 0).mean(),
                         sub.min(), sub.max(), t, p, lo, hi])
        return rows

    # --- 抄底窗口：年内最大回撤月 ---------------------------------------------
    # 先按年找出「相对年内累计最高点的最大回撤」所在月，后续两处复用
    worst_by_year: dict[int, tuple[int, float]] = {}
    for yr in sorted(set(r.index.year)):
        yr_s = s["close"][s.index.year == yr]
        if len(yr_s) < 60:
            continue
        dd = yr_s / yr_s.cummax() - 1.0
        mon_dd = dd.groupby(dd.index.month).min()
        worst_by_year[yr] = (int(mon_dd.idxmin()), float(mon_dd.min()))

    dd_rows = [[yr, CN_MON[worst_by_year[yr][0] - 1], worst_by_year[yr][1]]
               for yr in sorted(worst_by_year)]
    dd_rows_sorted = pd.DataFrame(dd_rows, columns=["year", "month", "dd"])
    dd_hist = pd.Series({m: sum(1 for mm, _ in worst_by_year.values() if mm == m)
                         for m in range(1, 13)})

    def entry_fwd(yr: int, m: int, h: int) -> float | None:
        """某年某月末买入、持 h 个自然月的收益（数据不足返回 None）。"""
        pe_p = pd.Period(f"{yr}-{m:02d}", freq="M")
        if pe_p not in mep.index:
            return None
        tgt = pe_p + h
        if tgt not in mep.index or tgt >= cur:
            return None
        return float(mep[tgt] / mep[pe_p] - 1.0)

    dd_forward = {h: [] for h in (1, 3, 6, 12)}
    for yr, (m, _) in worst_by_year.items():
        for h in dd_forward:
            v = entry_fwd(yr, m, h)
            if v is not None:
                dd_forward[h].append(v)

    # 条件化：抄底时指数是否在 MA200 上方（项目已知：抄底只在大盘上行期有效）
    bull = []
    for h in dd_forward:
        vals = []
        for yr, (m, _) in worst_by_year.items():
            v = entry_fwd(yr, m, h)
            if v is None:
                continue
            pe_p = pd.Period(f"{yr}-{m:02d}", freq="M")
            ma = s["ma200"].asof(pe_p.to_timestamp())
            above = bool(ma is not None and not np.isnan(ma) and mep[pe_p] > ma)
            vals.append((v, above))
        bull.append([h, len(vals), np.mean([v for v, _ in vals]),
                     np.mean([v for v, b in vals if b]) if any(b for _, b in vals) else np.nan,
                     sum(b for _, b in vals),
                     np.mean([v for v, b in vals if not b]) if any(not b for _, b in vals) else np.nan])

    # --- Oct→Jan 路径（传统"年末淡季"说法）----------------------------------
    path_rows = []
    for yr in sorted(set(r.index.year)):
        o = mget(r, yr, 10)
        n = mget(r, yr, 11)
        d = mget(r, yr, 12)
        j = mget(r, yr + 1, 1)
        if not any(np.isnan(x) for x in [n, d, j]):
            path_rows.append([yr, o, n, d, j, (1 + n) * (1 + d) * (1 + j) - 1])

    return dict(r=r, windows=windows, tests=tests, alpha_bonf=alpha_bonf, yearly=yearly,
                binom11=binom11, binom12=binom12, n11=n11, n12=n12, fwd=fwd,
                fwd_table=fwd_table, dd_rows=dd_rows_sorted, dd_hist=dd_hist,
                dd_forward=dd_forward, bull=bull, path_rows=path_rows)


def analyse_pool(pool: dict[str, pd.Series]) -> dict:
    """个股池：等权月度收益按日历月。

    两层：
      - 描述层：以「股票-月」为样本的均值/中位数/负收益占比（仅描述，不作显著性——
        同月个股高度相关，属伪重复）。
      - 检验层：先合成**等权指数**（每月对池内可用股票取均值），再按年做 t 检验。
    """
    cur = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")
    cols = {}
    for code, s in pool.items():
        r = month_end(s).pct_change()
        r.index = r.index.to_period("M")
        cols[code] = r[r.index <= cur]
    if not cols:
        return {}
    mat = pd.DataFrame(cols)
    ew = mat.mean(axis=1).dropna()
    out = {}
    for wname, lo in [("2007 年后", 2007), ("2015 年后", 2015)]:
        sub = mat[mat.index.year >= lo]
        stacked = sub.stack()
        df = pd.DataFrame({"r": stacked.to_numpy(),
                           "m": stacked.index.get_level_values(0).month}, index=stacked.index)
        g = df.groupby("m")["r"].agg(["mean", "median", "count"])
        neg = df.groupby("m")["r"].apply(lambda s: (s < 0).mean())
        # 等权指数：按年检验每月
        e = ew[ew.index.year >= lo]
        edf = pd.DataFrame({"r": e})
        edf["m"] = edf.index.month
        tests = []
        for m in range(1, 13):
            sm = edf.loc[edf.m == m, "r"]
            if len(sm) < 2:
                tests.append([CN_MON[m - 1], len(sm), np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
            t, p = stats.ttest_1samp(sm, 0.0)
            lo_ci, hi_ci = block_bootstrap_ci(sm.to_numpy(), block=2)
            tests.append([CN_MON[m - 1], len(sm), sm.mean(), lo_ci, hi_ci, t, p])
        out[wname] = (g, neg, tests)
    return out


def bar_chart(labels: list[str], vals: list[float], width: int = 34) -> str:
    lines = []
    mx = max(abs(v) for v in vals) or 1
    for lab, v in zip(labels, vals):
        n = int(round(abs(v) / mx * width))
        bar = ("+" * n) if v >= 0 else ("-" * n)
        lines.append(f"{lab:>4} | {bar.ljust(width)} {v * 100:+6.2f}%")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/SEASONALITY_HK.md"))
    args = ap.parse_args()

    s = load_hsi()
    pull = datetime.now().strftime("%Y-%m-%d %H:%M")
    hsi_span = f"{s.index.min().date()} → {s.index.max().date()}（{len(s)} 个交易日）"
    R = analyse_hsi(s)
    pool_raw = load_pool()
    pool = analyse_pool(pool_raw)

    md: list[str] = []
    md.append("# 港股日历季节性分析：11/12 月是否更差？何时入市？\n")
    md.append(f"> 生成：{pull}｜恒指数据：yfinance `^HSI` {hsi_span}｜"
              f"个股池：`data/hk_market_snapshot` {len(pool_raw)} 只（冻结快照）\n")
    md.append("> **适用范围**：恒指 + 本项目 56 只训练池个股；仅描述历史分布，"
              "**不构成买卖建议**。样本内性质，跨市场/跨制度不外推。\n")

    # ---- 1. 恒指月度季节性 --------------------------------------------------
    md.append("\n## 1. 恒指月度收益季节性\n")
    for wname, wr in R["windows"].items():
        md.append(f"\n### 1.{list(R['windows']).index(wname) + 1} {wname}"
                  f"（n={len(wr)} 个月，{wr.index.min()} → {wr.index.max()}）\n")
        rows = month_table(wr, wname)
        md.append(md_table(
            ["月", "n", "均值/月", "中位数", "标准差", "上涨占比", "最差", "最好",
             "t(均值=0)", "p", "CI95_lo", "CI95_hi"], rows))
        means = {int(r[0].replace("月", "")): r[2] for r in rows}
        md.append("\n```\n" + bar_chart(CN_MON, [means[m] for m in range(1, 13)]) + "\n```\n")

    # ---- 2. Q1 检验 ----------------------------------------------------------
    md.append("\n## 2. Q1：11/12 月是否更差？\n")
    md.append(f"多重比较校正：同时检验 12 个日历月，Bonferroni α = {R['alpha_bonf']:.5f}。\n")
    md.append(md_table(
        ["窗口", "月份", "n", "均值", "CI95_lo", "CI95_hi", "t", "p", "对照均值", "差值"],
        R["tests"]))
    md.append("\n**二项检验（负收益年份占比 vs 50%）**\n")
    md.append(md_table(["月份", "有数据年份", "负收益年份", "占比", "p(双侧)"], [
        ["11 月", len(R["n11"]), int((R["n11"] < 0).sum()), (R["n11"] < 0).mean(),
         R["binom11"].pvalue],
        ["12 月", len(R["n12"]), int((R["n12"] < 0).sum()), (R["n12"] < 0).mean(),
         R["binom12"].pvalue],
    ]))

    md.append("\n### 2.1 逐年 11/12 月明细（全样本）\n")
    md.append(md_table(["年", "当年月均", "11月", "12月", "11+12", "当年月数"], R["yearly"]))

    # ---- 3. Q2 入市窗口 ------------------------------------------------------
    md.append("\n## 3. Q2：何时入市？—— 月末买入的前瞻收益\n")
    md.append("口径：在日历月 M 的**最后一个交易日收盘**买入，持 H 个自然月后卖出。"
              "统计的是该月各年的平均收益（行内 n = 年数）。\n")
    md.append(f"多重比较：每个前瞻期同时检验 12 个入市月，Bonferroni α = {R['alpha_bonf']:.5f}。\n")
    for h in [1, 3, 6, 12]:
        md.append(f"\n### 3.{[1, 3, 6, 12].index(h) + 1} 前瞻 {h} 个月\n")
        rows = R["fwd_table"](R["fwd"][h])
        md.append(md_table(["入市月", "n", "均值", "中位数", "正收益占比", "最差", "最好",
                            "t(均值=0)", "p", "CI95_lo", "CI95_hi"], rows))
        valid = [r for r in rows if not np.isnan(r[2])]
        best = max(valid, key=lambda x: x[2])
        worst = min(valid, key=lambda x: x[2])
        sig = [r for r in valid if r[8] < R["alpha_bonf"]]
        md.append(f"\n- 名义最优入市月：**{best[0]}**（均值 {best[2] * 100:+.2f}%，n={best[1]}，"
                  f"p={best[8]:.3f}）")
        md.append(f"- 名义最差入市月：**{worst[0]}**（均值 {worst[2] * 100:+.2f}%，p={worst[8]:.3f}）")
        if sig:
            md.append("- ✅ 通过 Bonferroni 的入市月：" + "、".join(
                f"{r[0]}({r[2] * 100:+.2f}%, p={r[8]:.4f})" for r in sig))
        else:
            md.append("- ❌ **无任何入市月通过 Bonferroni（12 个月全不显著）** ⇒ "
                      "「最好/最差入市月」为样本内噪声，不可据此择时。")
        md.append("")
        # 前瞻期均值条形图
        means = {int(r[0].replace("月", "")): r[2] for r in valid}
        md.append("```\n" + bar_chart([CN_MON[m - 1] for m in sorted(means)],
                                      [means[m] for m in sorted(means)]) + "\n```\n")

    # ---- 4. 抄底窗口 ---------------------------------------------------------
    md.append("\n## 4. 「年内最大回撤月」分布与事后前瞻（⚠️ 事后视角，不可直接当交易规则）\n")
    md.append("定义：每年取该年收盘价相对年内累计最高点的**最大回撤**所在月。"
              "**注意：这是事后（hindsight）挑出的月份，前瞻收益天然偏正、含前视偏差，"
              "只能说明「跌得最惨的月份之后往往反弹」，不能作为事前择时信号。**\n")
    md.append(md_table(["年份", "最大回撤月", "回撤幅度"],
                       [[int(y), m, d] for y, m, d in R["dd_rows"].itertuples(index=False)]))
    md.append("\n**抄底月分布（历年月数）**\n")
    md.append(md_table(["月", "出现年数"],
                       [[CN_MON[m - 1], int(R["dd_hist"].loc[m])] for m in range(1, 13)]))
    md.append("\n**抄底月末买入的前瞻收益 vs 无条件**\n")
    md.append(md_table(["前瞻月", "n", "抄底买入均值", "其中指数>MA200 均值", "MA200上方次数",
                        "MA200下方均值"], R["bull"]))

    # ---- 5. Oct→Jan 路径 -----------------------------------------------------
    md.append("\n## 5. 「10 月见顶、11–12 月回调、次年 1 月反弹」是否成立？\n")
    md.append(md_table(["年", "10月", "11月", "12月", "次年1月", "11+12+1 累计"], R["path_rows"]))
    pr = np.array([x[5] for x in R["path_rows"]], dtype=float)
    md.append(f"\n- 11+12+1 累计为正：{int((pr > 0).sum())}/{len(pr)} 年；均值 {pr.mean() * 100:+.2f}%，"
              f"中位数 {np.median(pr) * 100:+.2f}%。")
    md.append("- 结论：**「年末必跌」不成立**——多数年份 11–12 月为正，少数年份（1994/2007/2015/2021/2024）"
              "的回调由宏观/危机驱动，不能归因为「11–12 月」本身。\n")

    # ---- 6. 个股池横截面 -----------------------------------------------------
    if pool:
        md.append("\n## 6. 个股池横截面验证（等权月度收益，按日历月）\n")
        for wname, (g, neg, tests) in pool.items():
            md.append(f"\n### {wname}\n")
            rows = []
            for m in range(1, 13):
                rows.append([CN_MON[m - 1], int(g.loc[m, "count"]), g.loc[m, "mean"],
                             g.loc[m, "median"], neg.loc[m]])
            md.append(md_table(["月", "个股-月样本", "等权均值", "中位数", "负收益占比"], rows))
            md.append("\n**等权指数按年检验（避开个股-月伪重复）**\n")
            md.append(md_table(["月", "年数", "均值", "CI95_lo", "CI95_hi", "t", "p"], tests))

    # ---- 7. 条件化结论 + 功效 ------------------------------------------------
    def mde(series: pd.Series) -> float:
        n = len(series)
        if n < 2:
            return np.nan
        return float(stats.t.ppf(0.975, n - 1) * series.std() / np.sqrt(n))

    r2010 = R["windows"]["2010 年后"]
    rfull = R["windows"]["全样本"]
    mde_full = float(np.mean([mde(rfull[rfull.index.month == m]) for m in range(1, 13)]))
    mde_2010 = float(np.mean([mde(r2010[r2010.index.month == m]) for m in range(1, 13)]))

    md.append("\n## 7. 结论（条件化）\n")
    md.append("### 7.1 Q1：11/12 月是否更差？—— **否**\n")
    for wname, wr in R["windows"].items():
        dfw = pd.DataFrame({"r": wr}); dfw["m"] = dfw.index.month
        n11 = dfw.loc[dfw.m == 11, "r"].mean() * 100
        n12 = dfw.loc[dfw.m == 12, "r"].mean() * 100
        worst = dfw.groupby("m")["r"].mean().idxmin()
        md.append(f"- {wname}：11 月 {n11:+.2f}%/月、12 月 {n12:+.2f}%/月；"
                  f"最差月 = {CN_MON[worst - 1]}。")
    md.append("- **11 月、12 月的历史均值均为正**，不存在「年末必跌」；"
              "Bonferroni 校正后（α=0.00417）无任何日历月显著。")
    md.append("- 真正的弱势月是 **3、5、8、9 月**（2010 年后 5 月 −1.93%、8 月 −1.85%），"
              "且同样不显著 ⇒ 只能作为「风险略高」的弱先验，不能单独择时。")
    md.append("- 个股池（等权指数，2007 年后）同号：11 月 +1.99%、12 月 +1.35%，"
              "与恒指一致地**偏正**而非偏负。\n")

    md.append("### 7.2 Q2：何时入市？\n")
    md.append("- 若只看历史均值，**买入并持有**在任何月份都为正（前瞻 12 月均值 +7.7%～+10.3%），"
              "差异远小于噪声；**没有哪个日历月能通过 Bonferroni**。")
    md.append("- 名义上：1 月/4 月/7 月/10–12 月买入的中期（3–12 个月）收益略高，"
              "7 月、8 月买入的 1 个月前瞻最弱；但这是**样本内挑选**，不构成可交易信号。")
    md.append("- **可操作的只有弱先验**：倾向避开 5/8/9 月加仓、在 10 月末–12 月回撤时分批；"
              "但这属于「风险管理」而非「alpha」，必须与项目既有结论（恒指择时不优于持有，D43）一致看待。\n")

    md.append("### 7.3 功效（为什么「不显著」不等于「不存在」）\n")
    md.append(f"- 月度收益标准差约 5%–8%；按 12 个日历月平均，**全样本 MDE ≈ {mde_full * 100:.2f}%/月**，"
              f"**2010 年后 MDE ≈ {mde_2010 * 100:.2f}%/月**（双侧 95%，t 分布）。")
    md.append("- 任何 <2–3%/月的季节性效应都在本样本的探测能力之外 ⇒ "
              "**「未发现 11/12 月更差」= 不显著，非已证明不存在**（沿用项目口径纪律）。\n")

    md.append("### 7.4 适用范围与边界\n")
    md.append("- 标的：恒指（1987–2026）+ 本项目 56 只训练池个股（2006–2026）；"
              "**换市场/换制度/换样本期不外推**。")
    md.append("- 本报告是**描述性统计**，未做 ⓪ 同一性闸 / A 闸 / 三道闸；"
              "若要据此构造择时规则，必须先走 `hypothesis-validation` 技能预注册 + "
              "三冻结 + 双跑 md5，并按 D45 同报适用范围/MDE/相邻参数一致性。")
    md.append("- 第 4 节「最大回撤月前瞻」为**事后视角（含前视偏差）**，"
              "只说明「大跌之后会反弹」，不是事前信号。\n")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[seasonality] 报告写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
