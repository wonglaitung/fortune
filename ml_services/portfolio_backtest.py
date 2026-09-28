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
输出：净均收益 / 净IR / 胜率 / 换手 / 累计，并与等权基准对比；
含超额（TopK净 − 等权基准）bootstrap CI 与逐年超额分解

用法：
  python3 ml_services/portfolio_backtest.py --horizon 5 --topk 10
  python3 ml_services/portfolio_backtest.py --horizon 20 --pred <A股CSV> --market a
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


def load_panel(pred_csv, market='hk'):
    """读 prediction_analysis.csv → 面板（date/code/prob/ret/sector）

    market='a'（A股）：板块映射用 a_stock_config.A_STOCK_SECTOR_MAPPING，
    代码左补零到6位（read_csv 类型推断会丢前导零）——A_STOCK_REFORM_PLAN P2.1
    """
    df = pd.read_csv(pred_csv)
    df = df.rename(columns={'Stock_Code': 'code', 'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['prob'] = pd.to_numeric(df['Predict_Prob'], errors='coerce')
    df['ret'] = pd.to_numeric(df['Actual_Return'], errors='coerce')
    df = df.dropna(subset=['date', 'code', 'prob', 'ret'])
    df['code'] = df['code'].astype(str)
    is_a = str(market).lower().startswith('a')
    if is_a:
        df['code'] = df['code'].str.zfill(6)
    try:
        if is_a:
            from a_stock_config import A_STOCK_SECTOR_MAPPING as MAPPING
        else:
            from config import STOCK_SECTOR_MAPPING as MAPPING
        df['sector'] = df['code'].map(lambda c: (MAPPING.get(c) or {}).get('sector', 'unknown'))
    except Exception:
        df['sector'] = 'unknown'
    return df


def _sector_neutral_score(g):
    """行业内 z-score（对 prob），再全局排序"""
    s = g.groupby('sector')['prob']
    z = (g['prob'] - s.transform('mean')) / (s.transform('std').replace(0, np.nan))
    return z.fillna(0.0)


def backtest(df, horizon, topk, use_sector_neutral, cost=COST, dropout=0, vol_target=None):
    dates = sorted(df['date'].unique())
    rb = dates[::horizon]  # 非重叠调仓
    rows = []
    prev_long, prev_short = set(), set()
    held = []  # TopK-Dropout：当前持仓（按上次分数降序）
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
        bench = g['ret'].mean()

        # TopK-Dropout：保留仍在榜的旧持仓，剔除分数最低的 dropout 只，再补新
        if dropout > 0 and held:
            hold_score = g[g['code'].isin(held)].sort_values('score', ascending=False)
            keep = list(hold_score['code'].head(max(k - dropout, 1)))
            fill = [c for c in g['code'] if c not in keep and c not in held][:k - len(keep)]
            if len(keep) + len(fill) < k:
                fill += [c for c in g['code'] if c not in keep and c not in fill][:k - len(keep) - len(fill)]
            sel = keep + fill
        else:
            sel = list(g['code'].head(k))
        long = g[g['code'].isin(sel)]
        short = g.tail(k)

        cur_long = set(long['code']); cur_short = set(short['code'])
        t_long = len(cur_long - prev_long) / k if prev_long else 1.0
        t_short = len(cur_short - prev_short) / k if prev_short else 1.0
        prev_long, prev_short = cur_long, cur_short
        held = sel

        top_gross = long['ret'].mean()
        top_net = top_gross - t_long * cost
        ls_gross = top_gross - short['ret'].mean()
        ls_net = ls_gross - (t_long + t_short) * cost
        rows.append(dict(date=d, bench=bench, top_gross=top_gross, top_net=top_net,
                         ls_gross=ls_gross, ls_net=ls_net, turnover=t_long))
    bt = pd.DataFrame(rows)
    # 波动率目标：按上一期为止的基准已实现波动缩放敞口
    if vol_target and len(bt) > 3:
        rv = bt['bench'].rolling(6, min_periods=3).std().shift(1)
        ann_rv = rv * np.sqrt(252.0 / horizon)
        scale = (vol_target / ann_rv).clip(upper=2.0).fillna(0.0)
        bt['exposure'] = scale
        bt['top_net'] = bt['top_net'] * scale
        bt['ls_net'] = bt['ls_net'] * scale
    return bt


def bootstrap(x, horizon, n=2000, seed=42):
    """对期收益做 bootstrap，返回 净均收益/净IR 的 95%CI 与 P(IR>0.5)/P(IR>0)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return dict(mean_lo=np.nan, mean_hi=np.nan, ir_lo=np.nan, ir_hi=np.nan,
                    p_ir=np.nan, p_ir_pos=np.nan)
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
                p_ir=float(np.nanmean(irs > 0.5)), p_ir_pos=float(np.nanmean(irs > 0)))


def _stats(x, horizon):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return dict(n=len(x), mean=np.nan, ir=np.nan, win=np.nan, cum=np.nan)
    mean = float(np.mean(x)); std = float(np.std(x, ddof=1))
    ir = mean / std * np.sqrt(252.0 / horizon) if std > 0 else np.nan
    cum = float(np.prod(1 + x) - 1)
    return dict(n=len(x), mean=mean, ir=ir, win=float((x > 0).mean()), cum=cum)


def run(horizon, pred_csv, topk, out_md, cost=COST, dropout=0, vol_target=None, market='hk'):
    df = load_panel(pred_csv, market=market)
    print(f"\n{'='*70}\nPhase3 最小组合回测  horizon={horizon}  TopK={topk}\n{'='*70}")
    print(f"样本: {len(df)}　交易日: {df['date'].nunique()}　股票: {df['code'].nunique()}")

    res = {}
    for name, neutral in [("TopK-raw", False), ("TopK-行业中性", True)]:
        res[name] = backtest(df, horizon, topk, neutral, cost=cost, dropout=dropout, vol_target=vol_target)

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
    L.append(f"- 调仓: 每 {horizon} 交易日（非重叠）　成本: {cost:.3f}　Dropout={dropout}　波动率目标={vol_target}\n")
    # bootstrap CI（对每个组合的净收益序列）
    series = {'等权基准（全池）': res["TopK-raw"]['bench'].values}
    series['TopK-raw（净）'] = res["TopK-raw"]['top_net'].values
    series['TopK-行业中性（净）'] = res["TopK-行业中性"]['top_net'].values
    series['多空 Q5-Q1（行业中性，净）'] = res["TopK-行业中性"]['ls_net'].values
    ci = {k: bootstrap(v, horizon) for k, v in series.items()}

    # 超额序列（TopK净 − 等权基准，逐期）
    excess = {
        'TopK-raw（净）': res["TopK-raw"]['top_net'].values - res["TopK-raw"]['bench'].values,
        'TopK-行业中性（净）': res["TopK-行业中性"]['top_net'].values - res["TopK-行业中性"]['bench'].values,
        '多空 Q5-Q1（行业中性，净）': res["TopK-行业中性"]['ls_net'].values - res["TopK-行业中性"]['bench'].values,
    }
    ex_stats = {k: _stats(v, horizon) for k, v in excess.items()}
    ex_ci = {k: bootstrap(v, horizon) for k, v in excess.items()}

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

    L.append("## 超额检验（TopK净 − 等权基准，bootstrap 2000 次）\n")
    L.append("> 基准自身 CI 常跨 0，只看绝对净IR 会高估；超额口径 95%CI 下界 >0 才算组合层稳健。\n")
    L.append("| 组合 | 超额均收益/期 | 超额IR | 超额IR 95%CI | P(超额IR>0) | 超额>0占比 |")
    L.append("|------|-------------|--------|--------------|-------------|-----------|")
    for name, st in ex_stats.items():
        c = ex_ci[name]
        cistr = '—' if np.isnan(c['ir_lo']) else f"[{c['ir_lo']:.2f}, {c['ir_hi']:.2f}]"
        ppos = '—' if np.isnan(c['p_ir_pos']) else f"{c['p_ir_pos']*100:.0f}%"
        win = '—' if np.isnan(st['win']) else f"{st['win']*100:.0f}%"
        L.append(f"| {name} | {_p(st['mean'])} | {_f(st['ir'])} | {cistr} | {ppos} | {win} |")
    L.append("")

    L.append("## 结论\n")
    best = "TopK-行业中性（净）" if res["TopK-行业中性"]['top_net'].mean() >= res["TopK-raw"]['top_net'].mean() else "TopK-raw（净）"
    bst = _stats(series[best], horizon)
    c = ci[best]
    ec = ex_ci[best]
    est = ex_stats[best]
    ex_ok = not np.isnan(ec['ir_lo']) and ec['ir_lo'] > 0
    if bst['ir'] > 0.5 and c['ir_lo'] > 0 and ex_ok:
        verd = "✅ 净 IR>0.5、CI 下限>0 且超额CI 下限>0：组合层稳健"
    elif bst['ir'] > 0 and c['ir_lo'] > 0:
        verd = "⚠️ 净 IR 正向且 CI 下限>0，但超额口径未过（可能只是 beta）"
    elif bst['ir'] > 0:
        verd = "⚠️ 点估计正但 CI 跨 0 → 不稳健"
    else:
        verd = "❌ 扣成本后净 IR ≤ 0：当前信号无法覆盖成本"
    L.append(f"- 最佳组合（{best}）：净IR {bst['ir']:.2f} [{c['ir_lo']:.2f},{c['ir_hi']:.2f}]，"
             f"净均收益/期 {bst['mean']*100:+.2f}%，P(IR>0.5)={c['p_ir']*100:.0f}%")
    L.append(f"- 超额（vs 等权基准）：超额IR {est['ir']:.2f} [{ec['ir_lo']:.2f},{ec['ir_hi']:.2f}]，"
             f"超额均收益/期 {est['mean']*100:+.2f}%，P(超额IR>0)={ec['p_ir_pos']*100:.0f}%")
    L.append(f"- 对照等权基准：净IR {bench['ir']:.2f}，均收益/期 {bench['mean']*100:+.2f}%")
    L.append(f"- 判定: {verd}")
    L.append("")

    # 逐年分解（最佳组合 vs 基准）
    bt = res["TopK-行业中性"].copy()
    bt['year'] = pd.to_datetime(bt['date']).dt.year
    L.append("## 逐年分解（行业中性 TopK vs 等权基准）\n")
    L.append("| 年份 | 期数 | 基准均收益/期 | TopK净均收益/期 | 超额/期 | 超额IR |")
    L.append("|------|------|--------------|----------------|---------|--------|")
    for y, g in bt.groupby('year'):
        st = _stats(g['top_net'].values, horizon)
        bm = g['bench'].mean()
        ex = (g['top_net'] - g['bench']).mean()
        ex_ir = _stats((g['top_net'] - g['bench']).values, horizon)['ir']
        L.append(f"| {int(y)} | {len(g)} | {_p(bm)} | {_p(st['mean'])} | {_p(ex)} | {_f(ex_ir)} |")
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
    ap.add_argument('--dropout', type=int, default=0, help='TopK-Dropout 每期剔除只数')
    ap.add_argument('--vol-target', type=float, default=None, help='年化波动率目标（如 0.15）')
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--market', type=str, default='hk', choices=['hk', 'a'],
                    help='市场：hk=港股（默认），a=A股（A_STOCK_REFORM_PLAN P2.1）')
    args = ap.parse_args()
    pred_csv = args.pred or DEFAULT_PRED[args.horizon]
    suffix = '_a' if args.market == 'a' else ''
    out = args.output or f"output/portfolio_{args.horizon}d_top{args.topk}{suffix}.md"
    run(args.horizon, pred_csv, args.topk, out, cost=args.cost, dropout=args.dropout,
        vol_target=args.vol_target, market=args.market)


if __name__ == '__main__':
    main()
