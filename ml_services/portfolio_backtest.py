#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 最小版：把 OOS 预测分数转成组合，评估**扣成本后**的净表现

输入：Walk-forward 的 prediction_analysis.csv（逐日逐股 Predict_Prob + Actual_Return）
方法：
  - 非重叠调仓（每 horizon 天一次）
  - TopK 组合（等权）；可选行业中性（行业内 z-score 后再选）
  - 成本 = 单边换手率 × 双边成本(TOTAL_COST)
  - 多空 Q5-Q1
输出：净均收益 / 净IR / 胜率 / 换手 / 累计，并与等权基准对比

用法：
  python3 ml_services/portfolio_backtest.py --horizon 5 --topk 10
  python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST = 0.005  # 双边成本（与 walk_forward TOTAL_COST 一致）
DEFAULT_PRED = {
    5: 'output/20260922_212530_catboost_5d/prediction_analysis.csv',
    20: 'output/20260922_162806_catboost_20d/prediction_analysis.csv',
}


def load_panel(pred_csv):
    df = pd.read_csv(pred_csv)
    df = df.rename(columns={'Stock_Code': 'code', 'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['prob'] = pd.to_numeric(df['Predict_Prob'], errors='coerce')
    df['ret'] = pd.to_numeric(df['Actual_Return'], errors='coerce')
    df = df.dropna(subset=['date', 'code', 'prob', 'ret'])
    try:
        from config import STOCK_SECTOR_MAPPING
        df['sector'] = df['code'].map(lambda c: (STOCK_SECTOR_MAPPING.get(c) or {}).get('sector', 'unknown'))
    except Exception:
        df['sector'] = 'unknown'
    return df


def _sector_neutral_score(g):
    """行业内 z-score（对 prob），再全局排序"""
    s = g.groupby('sector')['prob']
    z = (g['prob'] - s.transform('mean')) / (s.transform('std').replace(0, np.nan))
    return z.fillna(0.0)


def backtest(df, horizon, topk, use_sector_neutral, cost=COST):
    dates = sorted(df['date'].unique())
    rb = dates[::horizon]  # 非重叠调仓
    rows = []
    prev_long, prev_short = set(), set()
    for d in rb:
        g = df[df['date'] == d].copy()
        if len(g) < 20:
            continue
        if use_sector_neutral:
            g['score'] = _sector_neutral_score(g)
        else:
            g['score'] = g['prob']
        g = g.sort_values('score', ascending=False)
        k = min(topk, len(g) // 4)
        long = g.head(k)
        short = g.tail(k)
        # 基准：等权全池
        bench = g['ret'].mean()
        # 多空
        ls_gross = long['ret'].mean() - short['ret'].mean()
        # 换手（单边）
        cur_long = set(long['code']); cur_short = set(short['code'])
        t_long = len(cur_long - prev_long) / k if prev_long else 1.0
        t_short = len(cur_short - prev_short) / k if prev_short else 1.0
        prev_long, prev_short = cur_long, cur_short
        top_gross = long['ret'].mean()
        top_net = top_gross - t_long * cost
        ls_net = ls_gross - (t_long + t_short) * cost
        rows.append(dict(date=d, bench=bench, top_gross=top_gross, top_net=top_net,
                         ls_gross=ls_gross, ls_net=ls_net, turnover=t_long))
    return pd.DataFrame(rows)


def bootstrap(x, horizon, n=2000, seed=42):
    """对期收益做 bootstrap，返回 净均收益/净IR 的 95%CI 与 P(IR>0.5)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return dict(mean_lo=np.nan, mean_hi=np.nan, ir_lo=np.nan, ir_hi=np.nan, p_ir=np.nan)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x))
    means, irs = [], []
    for _ in range(n):
        xs = x[rng.choice(idx, len(idx), replace=True)]
        m = xs.mean(); sd = xs.std(ddof=1)
        means.append(m)
        irs.append(m / sd * np.sqrt(252.0 / horizon) if sd > 0 else np.nan)
    means = np.array(means); irs = np.array(irs)
    return dict(mean_lo=float(np.percentile(means, 2.5)), mean_hi=float(np.percentile(means, 97.5)),
                ir_lo=float(np.nanpercentile(irs, 2.5)), ir_hi=float(np.nanpercentile(irs, 97.5)),
                p_ir=float(np.nanmean(irs > 0.5)))


def _stats(x, horizon):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return dict(n=len(x), mean=np.nan, ir=np.nan, win=np.nan, cum=np.nan)
    mean = float(np.mean(x)); std = float(np.std(x, ddof=1))
    ir = mean / std * np.sqrt(252.0 / horizon) if std > 0 else np.nan
    cum = float(np.prod(1 + x) - 1)
    return dict(n=len(x), mean=mean, ir=ir, win=float((x > 0).mean()), cum=cum)


def run(horizon, pred_csv, topk, out_md, cost=COST):
    df = load_panel(pred_csv)
    print(f"\n{'='*70}\nPhase3 最小组合回测  horizon={horizon}  TopK={topk}\n{'='*70}")
    print(f"样本: {len(df)}　交易日: {df['date'].nunique()}　股票: {df['code'].nunique()}")

    res = {}
    for name, neutral in [("TopK-raw", False), ("TopK-行业中性", True)]:
        res[name] = backtest(df, horizon, topk, neutral, cost=cost)

    # 汇总
    rows = []
    bench = _stats(res["TopK-raw"]['bench'].values, horizon)
    rows.append(("等权基准（全池）", bench, np.nan))
    for name in ("TopK-raw", "TopK-行业中性"):
        bt = res[name]
        st = _stats(bt['top_net'].values, horizon)
        rows.append((f"{name}（净）", st, float(bt['turnover'].mean())))
    st_ls = _stats(res["TopK-行业中性"]['ls_net'].values, horizon)
    rows.append(("多空 Q5-Q1（行业中性，净）", st_ls,
                 float(res["TopK-行业中性"]['turnover'].mean() * 2)))

    L = []
    L.append(f"# Phase 3 最小组合回测（{horizon}d，TopK={topk}）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 数据: `{pred_csv}`")
    L.append(f"- 调仓: 每 {horizon} 交易日（非重叠）　成本: {cost:.3f}（双边）\n")
    # bootstrap CI（对每个组合的净收益序列）
    series = {'等权基准（全池）': res["TopK-raw"]['bench'].values}
    series['TopK-raw（净）'] = res["TopK-raw"]['top_net'].values
    series['TopK-行业中性（净）'] = res["TopK-行业中性"]['top_net'].values
    series['多空 Q5-Q1（行业中性，净）'] = res["TopK-行业中性"]['ls_net'].values
    ci = {k: bootstrap(v, horizon) for k, v in series.items()}

    L.append("| 组合 | 期数 | 净均收益/期 | 净IR(年化) | 胜率 | 累计净收益 | 平均换手 | 净IR 95%CI | P(IR>0.5) |")
    L.append("|------|------|------------|-----------|------|-----------|---------|-----------|-----------|")
    for name, st, to in rows:
        to_s = '—' if (to is None or np.isnan(to)) else f"{to*100:.0f}%"
        c = ci.get(name, {})
        cistr = '—' if not c or np.isnan(c.get('ir_lo', np.nan)) else f"[{c['ir_lo']:.2f}, {c['ir_hi']:.2f}]"
        pstr = '—' if not c or np.isnan(c.get('p_ir', np.nan)) else f"{c['p_ir']*100:.0f}%"
        L.append(f"| {name} | {st['n']} | {_p(st['mean'])} | {_f(st['ir'])} | "
                 f"{_p(st['win'])} | {_p(st['cum'])} | {to_s} | {cistr} | {pstr} |")
    L.append("")
    L.append("> P(IR>0.5) = bootstrap 中净IR超过 0.5 的比例（越高越稳健）。\n")
    L.append("## 结论\n")
    best = "TopK-行业中性（净）" if res["TopK-行业中性"]['top_net'].mean() >= res["TopK-raw"]['top_net'].mean() else "TopK-raw（净）"
    bst = _stats(series[best], horizon)
    c = ci[best]
    if bst['ir'] > 0.5 and c['ir_lo'] > 0:
        verd = "✅ 净 IR>0.5 且 CI 下限>0：稳健，进入 Phase 3 完整版"
    elif bst['ir'] > 0 and c['ir_lo'] > 0:
        verd = "⚠️ 净 IR 正向且 CI 下限>0，但强度一般"
    elif bst['ir'] > 0:
        verd = "⚠️ 点估计正但 CI 跨 0 → 不稳健"
    else:
        verd = "❌ 扣成本后净 IR ≤ 0：当前信号无法覆盖成本"
    L.append(f"- 最佳组合（{best}）：净IR {bst['ir']:.2f} [{c['ir_lo']:.2f},{c['ir_hi']:.2f}]，"
             f"净均收益/期 {bst['mean']*100:+.2f}%，P(IR>0.5)={c['p_ir']*100:.0f}%")
    L.append(f"- 对照等权基准：净IR {bench['ir']:.2f}，均收益/期 {bench['mean']*100:+.2f}%")
    L.append(f"- 判定: {verd}")
    L.append("")

    # 逐年分解（最佳组合 vs 基准）
    bt = res["TopK-行业中性"].copy()
    bt['year'] = pd.to_datetime(bt['date']).dt.year
    L.append("## 逐年分解（行业中性 TopK vs 等权基准）\n")
    L.append("| 年份 | 期数 | 基准均收益/期 | TopK净均收益/期 | TopK净IR |")
    L.append("|------|------|--------------|----------------|---------|")
    for y, g in bt.groupby('year'):
        st = _stats(g['top_net'].values, horizon)
        bm = g['bench'].mean()
        L.append(f"| {int(y)} | {len(g)} | {_p(bm)} | {_p(st['mean'])} | {_f(st['ir'])} |")
    L.append("")
    L.append("> ⚠️ 逐年样本少（20d 每年仅 ~12 期）；若收益集中在单一年份，则稳健性存疑。\n")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def _p(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.1f}%"


def _f(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description='Phase3 最小 TopK 组合回测')
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 20])
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--cost', type=float, default=COST, help='双边成本（默认0.005）')
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()
    pred_csv = args.pred or DEFAULT_PRED[args.horizon]
    out = args.output or f"output/portfolio_{args.horizon}d_top{args.topk}.md"
    run(args.horizon, pred_csv, args.topk, out, cost=args.cost)


if __name__ == '__main__':
    main()
