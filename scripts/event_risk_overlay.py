#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件风控 overlay（B 方案）：不改模型分数，只在组合层排除/减仓。

用法：
  python3 scripts/event_risk_overlay.py \
    --pred output/20261031_195014_lgbm_20d/prediction_analysis.csv \
    --events data/events/events_310.csv --topk 10 --horizon 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_services.costs import COST_ROUND_TRIP

SNAP = 'data/hk_broad/snapshot'
HSI = 'data/hk_universe_cache/HSI.pkl'

# 预登记策略：事件类型 → (窗口天数, 动作)
POLICY = {
    'PROFIT_WARNING': (20, 'block'),
    'PLACEMENT_COMPANY': (20, 'block'),
    'PLACEMENT_SHAREHOLDER': (10, 'block'),
    'CB_ISSUANCE': (20, 'block'),
    'LOCKUP': (20, 'block'),
    'RIGHTS_ISSUE': (20, 'block'),
}


def load_prices(codes):
    out = {}
    for c in codes:
        p = f'{SNAP}/{str(c).zfill(5)}.pkl'
        if not os.path.exists(p):
            continue
        s = pd.read_pickle(p)['Close'].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[str(c).zfill(5)] = s.sort_index()
    return out


def active_codes(d, events, code_map):
    act = set()
    for et, (win, _) in POLICY.items():
        sub = events[(events.event_type == et) & (events.confidence == 'high') &
                     (events.announce_ts <= d) &
                     (events.announce_ts >= d - pd.Timedelta(days=win))]
        act |= {code_map.get(str(c).zfill(5)) for c in sub.code}
    act.discard(None)
    return act


def daily_portfolio(d, end, codes, px, w):
    """等权组合在 [d, end) 的逐日收益；缺失股票按 0。"""
    dates = None
    for c in codes:
        s = px.get(str(c).split('.')[0].zfill(5))
        if s is None:
            continue
        idx = s.index[(s.index >= d) & (s.index < end)]
        dates = idx if dates is None else dates.union(idx)
    if dates is None or len(dates) == 0:
        return pd.Series(dtype=float)
    rets = []
    for dt in dates:
        vals = []
        for c in codes:
            s = px.get(str(c).split('.')[0].zfill(5))
            if s is None or dt not in s.index:
                continue
            pos = s.index.get_loc(dt)
            if pos == 0:
                continue
            vals.append(s.iloc[pos] / s.iloc[pos - 1] - 1)
        rets.append(np.mean(vals) if vals else 0.0)
    return pd.Series(rets, index=dates)


def run(pred, events, topk, horizon, cost):
    df = pd.read_csv(pred)
    df['date'] = pd.to_datetime(df.Date)
    df['code'] = df.Stock_Code.astype(str)
    df['code5'] = df.code.str.split('.').str[0].str.zfill(5)
    df['score'] = pd.to_numeric(df.Predict_Prob, errors='coerce')
    events = events.copy()
    events['announce_ts'] = pd.to_datetime(events.announce_ts).dt.tz_localize(None).dt.normalize()
    code_map = {c: f'{c}.HK' for c in df.code5.unique()}
    px = load_prices(df.code5.unique())
    dates = sorted(df.date.unique())
    rb = dates[::horizon]

    curves = {'base': [], 'overlay': [], 'scaled': []}
    exposures, turns = [], []
    prev = set()
    for i, d in enumerate(rb):
        g = df[df.date == d].dropna(subset=['score']).sort_values('score', ascending=False)
        if len(g) < 20:
            continue
        k = min(topk, len(g) // 4)
        end = rb[i + 1] if i + 1 < len(rb) else dates[-1] + pd.Timedelta(days=1)
        base_codes = list(g.code.head(k))
        act = active_codes(d, events, code_map)
        ov_codes = [c for c in g.code if c not in act][:k]

        # 换手成本（双边按往返 0.5%）
        t = len(set(ov_codes) - prev) / max(k, 1) if prev else 1.0
        prev = set(ov_codes)
        turns.append(t)
        exposures.append(len(ov_codes) / max(k, 1))

        b = daily_portfolio(d, end, base_codes, px, None)
        o = daily_portfolio(d, end, ov_codes, px, None)
        for name, s in [('base', b), ('overlay', o)]:
            if len(s) == 0:
                continue
            s = s.copy()
            if name == 'overlay':
                s.iloc[0] -= t * cost
            curves[name].append(s)

    def cat(xs):
        if not xs:
            return pd.Series(dtype=float)
        return pd.concat(xs).sort_index()

    base = cat(curves['base'])
    overlay = cat(curves['overlay'])
    # 等暴露对照：基线按 overlay 平均仓位缩放（成本同比例）
    avg_exp = float(np.mean(exposures)) if exposures else 1.0
    scaled = base * avg_exp

    def stats(x):
        if len(x) < 20:
            return {}
        eq = (1 + x).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        ann_ir = x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)
        cvar = x[x <= x.quantile(0.05)].mean()
        return {'n': len(x), 'ann_ret': x.mean() * 252, 'ann_ir': ann_ir,
                'maxdd': dd, 'cvar5': cvar, 'cum': eq.iloc[-1] - 1}

    out = {'base': stats(base), 'overlay': stats(overlay), 'scaled': stats(scaled),
           'avg_exposure': avg_exp, 'avg_turnover': float(np.mean(turns)) if turns else np.nan}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--events', default='data/events/events_310.csv')
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--cost', type=float, default=COST_ROUND_TRIP)
    ap.add_argument('--out', default='output/_logs/event_risk_overlay.md')
    args = ap.parse_args()
    ev = pd.read_csv(args.events)
    res = run(args.pred, ev, args.topk, args.horizon, args.cost)

    L = ['# 事件风控 overlay（B 方案）', '',
         f'- 预测：`{args.pred}`；TopK={args.topk}；horizon={args.horizon}d；成本={args.cost:.3f}',
         f'- 平均仓位（overlay）：{res["avg_exposure"]:.1%}；平均换手：{res["avg_turnover"]:.0%}',
         '', '| 臂 | 天数 | 年化收益 | 年化IR | 最大回撤(逐日) | CVaR5% | 累计 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for k in ['base', 'overlay', 'scaled']:
        s = res[k]
        if not s:
            continue
        L.append(f"| {k} | {s['n']} | {s['ann_ret']:.1%} | {s['ann_ir']:.2f} | {s['maxdd']:.1%} | {s['cvar5']:.2%} | {s['cum']:.1%} |")
    L += ['', '> `scaled` = 基线按 overlay 平均仓位缩放，用于区分“真风控”与“只是少投”。',
          '> 判据：overlay 回撤/CVaR 改善且 scaled 不能解释；IR 不劣于基线 0.2。']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
