#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 最小版：把 OOS 预测分数转成组合，评估**扣成本后**的净表现

输入：Walk-forward 的 prediction_analysis.csv（逐日逐股 Predict_Prob + Actual_Return）
方法：
  - 非重叠调仓（每 horizon 天一次）
  - TopK 组合（等权），按 raw prob 排序（D73；行业中性已删除）
  - 成本 = 单边换手率 × 往返成本(COST_ROUND_TRIP)
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

# 选股打分**唯一实现**在 ml_services.selection（lessons 三.73/三.74）。
# 本模块与 live_topk（生产侧）都从这里导入，禁止各自再写一份。
from ml_services.selection import (          # noqa: E402
    selection_score,
)

from ml_services.costs import COST_ROUND_TRIP, SideCostModel
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


def selection_ic(df, min_per_day=20, min_sector=2):
    """**部署口径**的选股能力度量（lessons 三.73）。

    口径 = 逐日 Spearman(selection_score, ret) 再取均值/中位。
    打分只有一个实现（raw prob），与 `backtest` 必然同源。

    同时返回三个对照（它们**不是同一个量**，报告必须分开写，禁互相替代）：
      - `ic` / `ic_med` 部署口径（逐日、用部署打分）
      - `ic_raw`        同一逐日口径，但用**原始 prob**（说明「打分变换」之差）
      - `ic_pooled`     全表跨日池化 Spearman（WF 报告口径的近亲）。**含日间共同层**
                        ⇒ 天然偏高（D30：0.0322~0.0442 vs 逐日 0.0136）
                        ⇒ **不可用于能力评估**（lessons 三.68）。
      - `ic_rank_vs_lin` 逐日 Pearson 对照：同一「逐日」口径下 rank 与线性可异号
                        ⇒ 该量在噪声水平时本身就是统计量依赖的（D30：rank +0.0136 / 线性 −0.0145）。

    本函数**不下任何显著性结论**；显著性由调用方用 HAC / 块 bootstrap 判（三.69）。
    """
    dc = 'date' if 'date' in df.columns else 'Date'
    d = df.dropna(subset=['prob', 'ret']).copy()
    s_ic, r_ic, l_ic = [], [], []
    for _, g in d.groupby(dc):
        if len(g) < min_per_day:
            continue
        if g['sector'].nunique() >= min_sector:
            sc = selection_score(g)
            v = sc.corr(g['ret'], method='spearman')
            if np.isfinite(v):
                s_ic.append(v)
            # 逐日 Pearson（线性）对照：同一「逐日」口径下 rank 与线性可异号
            w = (sc - sc.mean()).corr(g['ret'] - g['ret'].mean())
            if np.isfinite(w):
                l_ic.append(w)
        v = g['prob'].corr(g['ret'], method='spearman')
        if np.isfinite(v):
            r_ic.append(v)
    s_ic, r_ic, l_ic = np.asarray(s_ic), np.asarray(r_ic), np.asarray(l_ic)
    out = {
        'n_days': int(s_ic.size),
        'ic': float(s_ic.mean()) if s_ic.size else float('nan'),
        'ic_med': float(np.median(s_ic)) if s_ic.size else float('nan'),
        'ic_raw': float(r_ic.mean()) if r_ic.size else float('nan'),
        'ic_lin': float(l_ic.mean()) if l_ic.size else float('nan'),
        'ic_pooled': float(d['prob'].corr(d['ret'], method='spearman')),
        'ic_series': s_ic,
    }
    return out


def backtest(df, horizon, topk, cost=COST_ROUND_TRIP, dropout=0, vol_target=None,
             cost_mode='flat', cost_model=None, trade_value=50000.0):
    dates = sorted(df['date'].unique())
    if cost_mode == 'model' and cost_model is not None:
        from ml_services.exposures import load_exposures
        _adv, _sig, _spr = load_exposures(sorted(df['code'].unique()))
    rb = dates[::horizon]  # 非重叠调仓
    rows = []
    prev_long, prev_short = set(), set()
    held = []  # TopK-Dropout：当前持仓（按上次分数降序）
    for d in rb:
        g = df[df['date'] == d].copy()
        if len(g) < 20:
            continue
        g['score'] = selection_score(g)
        g = g.sort_values('score', ascending=False, kind='mergesort')  # mergesort：score 并列时顺序稳定（去噪声）
        k = min(topk, len(g) // 4)
        bench = g['ret'].mean()

        # TopK-Dropout：保留仍在榜的旧持仓，剔除分数最低的 dropout 只，再补新
        if dropout > 0 and held:
            hold_score = g[g['code'].isin(held)].sort_values('score', ascending=False, kind='mergesort')
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
        held = sel

        top_gross = long['ret'].mean()
        ls_gross = top_gross - short['ret'].mean()
        if cost_mode == 'model' and cost_model is not None:
            from ml_services.exposures import lookup
            def _cost_of(old, new_set):
                c = 0.0
                for code in (new_set - old) | (old - new_set):
                    av = lookup(_adv, d, code)
                    sg = lookup(_sig, d, code)
                    sp = lookup(_spr, d, code)
                    if np.isfinite(av) and np.isfinite(sg):
                        c += (1.0 / k) * cost_model.per_side(trade_value, av, sg, sp)['total']
                    else:
                        c += (1.0 / k) * (cost / 2.0)
                return c
            c_long = _cost_of(prev_long, cur_long)
            c_short = _cost_of(prev_short, cur_short)
        else:
            c_long = t_long * cost
            c_short = t_short * cost
        top_net = top_gross - c_long
        ls_net = ls_gross - c_long - c_short
        prev_long, prev_short = cur_long, cur_short
        rows.append(dict(date=d, bench=bench, top_gross=top_gross, top_net=top_net,
                         ls_gross=ls_gross, ls_net=ls_net, turnover=t_long,
                         cost=c_long, cost_short=c_short))
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


def run(horizon, pred_csv, topk, out_md, cost=COST_ROUND_TRIP, dropout=0, vol_target=None, market='hk',
        cost_mode='flat', cost_model=None, trade_value=50000.0):
    df = load_panel(pred_csv, market=market)
    print(f"\n{'='*70}\nPhase3 最小组合回测  horizon={horizon}  TopK={topk}\n{'='*70}")
    print(f"样本: {len(df)}　交易日: {df['date'].nunique()}　股票: {df['code'].nunique()}")

    res = {"TopK": backtest(df, horizon, topk, cost=cost, dropout=dropout, vol_target=vol_target,
                              cost_mode=cost_mode, cost_model=cost_model, trade_value=trade_value)}

    # 汇总
    rows = []
    bench = _stats(res["TopK"]['bench'].values, horizon)
    rows.append(("等权基准（全池）", bench, np.nan))
    bt = res["TopK"]
    rows.append((f"TopK（净）", _stats(bt['top_net'].values, horizon),
                 float(bt['turnover'].mean())))
    rows.append(("多空 Q5-Q1（净）", _stats(bt['ls_net'].values, horizon),
                 float(bt['turnover'].mean() * 2)))

    L = []
    L.append(f"# Phase 3 最小组合回测（{horizon}d，TopK={topk}）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 数据: `{pred_csv}`")
    L.append(f"- 调仓: 每 {horizon} 交易日（非重叠）　成本: {cost:.3f}　Dropout={dropout}　波动率目标={vol_target}")
    L.append(f"- 打分: raw prob（D73）\n")
    # bootstrap CI（对每个组合的净收益序列）
    series = {'等权基准（全池）': bt['bench'].values,
              'TopK（净）': bt['top_net'].values,
              '多空 Q5-Q1（净）': bt['ls_net'].values}
    ci = {k: bootstrap(v, horizon) for k, v in series.items()}

    # 超额序列（TopK净 − 等权基准，逐期）
    excess = {'TopK（净）': bt['top_net'].values - bt['bench'].values,
              '多空 Q5-Q1（净）': bt['ls_net'].values - bt['bench'].values}
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
    best = "TopK（净）"
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
    bt = res["TopK"].copy()
    bt['year'] = pd.to_datetime(bt['date']).dt.year
    L.append("## 逐年分解（TopK vs 等权基准）\n")
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
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 10, 20])
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--cost', type=float, default=COST_ROUND_TRIP, help='往返成本（flat 模式）')
    ap.add_argument('--cost-mode', choices=['flat', 'model'], default='flat')
    ap.add_argument('--broker-rate', type=float, default=0.001)
    ap.add_argument('--min-fee', type=float, default=50.0)
    ap.add_argument('--impact-k', type=float, default=1.0)
    ap.add_argument('--trade-value', type=float, default=50000.0)
    ap.add_argument('--dropout', type=int, default=0, help='TopK-Dropout 每期剔除只数')
    ap.add_argument('--vol-target', type=float, default=None, help='年化波动率目标（如 0.15）')
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--market', type=str, default='hk', choices=['hk', 'a'],
                    help='市场：hk=港股（默认），a=A股（A_STOCK_REFORM_PLAN P2.1）')
    args = ap.parse_args()
    pred_csv = args.pred or DEFAULT_PRED.get(args.horizon)
    if not pred_csv:
        raise SystemExit(f"未提供 --pred，且 horizon={args.horizon} 无默认 CSV（D60：10d 需显式 --pred）")
    suffix = '_a' if args.market == 'a' else ''
    out = args.output or f"output/portfolio_{args.horizon}d_top{args.topk}{suffix}.md"
    cm = SideCostModel(broker_rate=args.broker_rate, min_fee=args.min_fee, impact_k=args.impact_k) if args.cost_mode == 'model' else None
    run(args.horizon, pred_csv, args.topk, out, cost=args.cost, dropout=args.dropout,
        cost_mode=args.cost_mode, cost_model=cm, trade_value=args.trade_value,
        vol_target=args.vol_target, market=args.market)


if __name__ == '__main__':
    main()
