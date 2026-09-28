#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学习器 A/B 对照报告（A股 P4.1 / DECISIONS D10 口径）

对同周期的 CatBoost 与 LightGBM walk-forward 产物并排输出：
  - 合并准确率 + 95%CI + vs 随机 p
  - 超额 lift（胜率 − 无条件基准）+ p
  - 月度护栏：净IR [95%CI] / PBO / DSR → D2 判定
  - 组合层：TopK 行业中性 超额IR [95%CI]

判定原则（D3/D10）：只比 lift / 净IR / PBO / DSR 等决策指标，不比绝对准确率。

用法：
  python3 ml_services/learner_ab_report.py --horizons 20 5 1
  python3 ml_services/learner_ab_report.py --horizons 20 --no-guardrail
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services import backtest_eval as be
from ml_services.eval_overfit import build_matrix, cscv_pbo, deflated_sharpe
from ml_services.monthly_guardrail import signal_lift, COST
from ml_services.portfolio_backtest import load_panel, backtest, bootstrap, _stats

LEARNERS = ['catboost', 'lightgbm']
TOPK = 10


def _find_csv(learner, horizon, market='a'):
    if str(market).lower().startswith('a'):
        pat = f'output/*_a_stock_{learner}_{horizon}d/prediction_analysis.csv'
    else:
        pat = [f for f in glob.glob(f'output/*_{learner}_{horizon}d/prediction_analysis.csv')
               if '_a_stock_' not in f]
        files = pat
        return max(files) if files else None
    files = glob.glob(pat)
    return max(files, key=os.path.getmtime) if files else None


def _verdict(ir, pbo, dsr):
    if ir <= 0:
        return '🔴 停用（IR≤0）'
    if ir >= 0.7 and pbo < 0.5 and dsr is not None and dsr >= 0.95:
        return '🟢 通过 → 可升级'
    return '🟡 保留低配（未达升级门槛）'


def metrics_for(csv, horizon, market='a', with_guardrail=True):
    be.set_market(market)
    df = pd.read_csv(csv)
    ev = be.evaluate(df, horizon)
    pooled = ev['pooled']
    wr = ev['win_rate'] or {}

    out = {
        'csv': csv,
        'n': int(ev['n']),
        'folds': int(next(c for c in df.columns
                          if c.lower() == 'fold') and
                     df[next(c for c in df.columns if c.lower() == 'fold')].nunique()),
        'acc': pooled.get('accuracy'),
        'acc_lo': pooled.get('accuracy_ci_low'),
        'acc_hi': pooled.get('accuracy_ci_high'),
        'acc_p': pooled.get('p_value_vs_random'),
        'lift': (wr['lift'] * 100) if wr.get('lift') is not None else None,   # pp
        'lift_p': wr.get('p_value_vs_baseline'),
        'trades': wr.get('trades'),
        'win_rate': wr.get('win_rate'),
        'baseline': wr.get('baseline'),
    }

    if not with_guardrail:
        return out

    panel = load_panel(csv, market=market)
    bt = backtest(panel, horizon, TOPK, True, cost=COST)
    st = _stats(bt['top_net'].values, horizon)
    ci = bootstrap(bt['top_net'].values, horizon)
    _, base, lift = signal_lift(df)

    M = build_matrix(horizon, csv, market=market)
    pbo, _p = cscv_pbo(M.values)
    sr = M.mean() / M.std(ddof=1)
    best = sr.idxmax()
    from scipy.stats import skew, kurtosis
    dsr = deflated_sharpe(sr[best], len(M), M.shape[1], float(sr.var(ddof=1)),
                          float(skew(M[best].values)),
                          float(kurtosis(M[best].values, fisher=False)))

    ex = bt['top_net'].values - bt['bench'].values
    ex_st = _stats(ex, horizon)
    ex_ci = bootstrap(ex, horizon)

    out.update({
        'ir': st['ir'], 'ir_lo': ci['ir_lo'], 'ir_hi': ci['ir_hi'],
        'pbo': pbo, 'dsr': dsr, 'verdict': _verdict(st['ir'], pbo, dsr),
        'guard_lift': (lift * 100) if lift == lift else None, 'guard_base': base,
        'ex_ir': ex_st['ir'], 'ex_ir_lo': ex_ci['ir_lo'], 'ex_ir_hi': ex_ci['ir_hi'],
        'ex_p': ex_ci.get('p_ir_pos', np.nan),
    })
    return out


def _f(x, fmt='{:.3f}'):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def render(horizon, rows):
    """rows: {learner: metrics dict}"""
    L = [f"# A股学习器 A/B 对照（{horizon}d）\n",
         f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- TopK={TOPK} 行业中性，成本 {COST:.3f}，判定看 lift/净IR/PBO/DSR（D3/D10），不看绝对准确率\n"]

    has_g = all('ir' in r for r in rows.values())
    cols = [l for l in LEARNERS if l in rows]
    head = "| 指标 | " + " | ".join(cols) + " |"
    L += [head, "|" + "---|" * (len(cols) + 1)]

    def line(label, key, fmt='{:.3f}', extra=''):
        vals = []
        for c in cols:
            v = rows[c].get(key)
            vals.append('—' if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v) + extra)
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    line('样本数', 'n', '{:d}')
    line('折数', 'folds', '{:d}')
    line('合并准确率', 'acc', '{:.1%}')
    line('准确率95%CI', 'acc_lo', '{:.1%}')
    line('　CI 上界', 'acc_hi', '{:.1%}')
    line('vs 随机 p', 'acc_p', '{:.4f}')
    line('**超额 lift**', 'lift', '{:+.2f}', extra='pp')
    line('lift p', 'lift_p', '{:.4f}')
    if has_g:
        line('信号胜率 / 基准', 'win_rate', '{:.1%}')
        line('**净IR** [95%CI]', 'ir', '{:.2f}')
        line('　CI 下界', 'ir_lo', '{:.2f}')
        line('　CI 上界', 'ir_hi', '{:.2f}')
        line('**PBO**', 'pbo', '{:.2f}')
        line('**DSR**', 'dsr', '{:.3f}')
        line('超额IR（组合层）', 'ex_ir', '{:.2f}')
        line('　超额IR CI 下界', 'ex_ir_lo', '{:.2f}')
        line('P(超额IR>0)', 'ex_p', '{:.0%}')
    L.append('')

    for c in cols:
        lv = rows[c].get('lift')
        L.append(f"- **{c}**：" + (rows[c].get('verdict', '—') if has_g
                                  else 'lift —' if lv is None else f"lift {lv:+.2f}pp"))
        L.append(f"  - 源: `{rows[c]['csv']}`")
    L.append("")

    # A/B 结论：以 lift 与净IR 双指标比
    if len(cols) == 2 and has_g:
        a, b = rows['catboost'], rows['lightgbm']
        dl = (a.get('lift') or 0) - (b.get('lift') or 0)   # 已是 pp
        di = (a.get('ir') or 0) - (b.get('ir') or 0)
        win = 'CatBoost' if (dl > 0 and di > 0) else ('LightGBM' if (dl < 0 and di < 0) else '平手/口径分歧')
        L += [f"## A/B 结论（{horizon}d）",
              f"- 超额 lift 差（CB−LGBM）: {dl:+.2f}pp；净IR 差: {di:+.2f}",
              f"- 双指标同向 → **{win} 胜**；分歧则按 D10「分周期 A/B、不全局替换」保守处理",
              "- 判定只用决策指标；单轮结果受运行噪声影响（lessons 三.22），需与上轮/另一周期交叉验证"]
    return "\n".join(L) + "\n"


def main():
    global TOPK
    ap = argparse.ArgumentParser(description='学习器 A/B 对照报告')
    ap.add_argument('--horizons', type=int, nargs='+', default=[20, 5, 1])
    ap.add_argument('--market', type=str, default='a')
    ap.add_argument('--topk', type=int, default=TOPK)
    ap.add_argument('--no-guardrail', action='store_true', help='跳过净IR/PBO/DSR/组合层（快速）')
    ap.add_argument('--output', type=str, default=None, help='合并输出 md（默认按周期各存一份）')
    args = ap.parse_args()

    TOPK = args.topk

    written = []
    for h in args.horizons:
        rows = {}
        for learner in LEARNERS:
            csv = _find_csv(learner, h, args.market)
            if not csv:
                print(f"⚠️ {h}d {learner}: 未找到产物，跳过")
                continue
            print(f"▶ {h}d {learner}: {csv}")
            rows[learner] = metrics_for(csv, h, args.market,
                                        with_guardrail=not args.no_guardrail)
        if not rows:
            continue
        md = render(h, rows)
        print(md)
        if not args.output:
            out = f'output/learner_ab_a_stock_{h}d_{datetime.now():%Y%m%d}.md'
            os.makedirs('output', exist_ok=True)
            with open(out, 'w', encoding='utf-8') as f:
                f.write(md)
            written.append(out)
    if args.output and 'md' in dir():
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(md)
        written.append(args.output)
    for w in written:
        print(f"✅ 报告已保存: {w}")


if __name__ == '__main__':
    main()
