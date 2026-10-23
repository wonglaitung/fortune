#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B′ 扩池：中性化口径对照（方案 B）。

在 300 池 WF 产物上，比较四种打分的「能力 IC + TopK 组合」：
  N1 行业中性（现行 D64 口径）
  N2 行业 + 规模 + 波动 三中性（本实验主命题）
  N3 仅规模 + 波动
  N4 不中性（对照）

规模代理 = log(20 日均成交额 HKD)，波动 = 20 日收益率标准差。
残差化在「逐日横截面」上做：rank(prob) ~ 常数 + 行业哑变量 + log_adv + vol。
打分残差越界不影响排序；组合评估复用 portfolio_backtest.backtest（非中性）。

用法：
    python3 scripts/neutral_ab_test.py --pred output/<dir>/prediction_analysis.csv --topk 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_services.overlap_stats import block_bootstrap_se, hac_t  # noqa: E402
from ml_services.portfolio_backtest import backtest, load_panel  # noqa: E402
from ml_services.portfolio_backtest import _stats  # noqa: E402


def attach_size_vol(df: pd.DataFrame) -> pd.DataFrame:
    src = os.path.join(ROOT, 'data', 'hk_broad')
    close = pd.read_parquet(os.path.join(src, 'panel_close.parquet'))
    volume = pd.read_parquet(os.path.join(src, 'panel_volume.parquet'))
    turn = (close * volume).stack().rename('amt').reset_index()
    turn.columns = ['date', 'code5', 'amt']
    turn['date'] = pd.to_datetime(turn['date'])
    turn = turn.sort_values(['code5', 'date'])
    turn['adv20'] = turn.groupby('code5')['amt'].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    c = close.stack().rename('px').reset_index()
    c.columns = ['date', 'code5', 'px']
    c['date'] = pd.to_datetime(c['date'])
    c = c.sort_values(['code5', 'date'])
    c['vol20'] = c.groupby('code5')['px'].transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std())

    d = df.copy()
    d['code5'] = d['code'].str.replace('.HK', '', regex=False).str.zfill(5)
    d = d.merge(turn[['date', 'code5', 'adv20']], on=['date', 'code5'], how='left')
    d = d.merge(c[['date', 'code5', 'vol20']], on=['date', 'code5'], how='left')
    return d


def residualize(g: pd.DataFrame, use_sector: bool, use_size_vol: bool) -> np.ndarray:
    y = g['prob'].rank(pct=True).to_numpy()
    cols = [np.ones(len(g))]
    if use_sector:
        dummies = pd.get_dummies(g['sector'], drop_first=True, dtype=float).to_numpy()
        cols.append(dummies)
    if use_size_vol:
        size = np.log1p(g['adv20'].clip(lower=1.0).to_numpy())
        vol = g['vol20'].fillna(g['vol20'].median()).to_numpy()
        # 横截面标准化，避免量纲主导
        size = (size - size.mean()) / (size.std() or 1.0)
        vol = (vol - vol.mean()) / (vol.std() or 1.0)
        cols.append(np.column_stack([size, vol]))
    X = np.column_stack([c if c.ndim == 2 else c.reshape(-1, 1) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def ic_stats(d: pd.DataFrame, score_col: str, min_per_day=20):
    s = []
    for _, g in d.groupby('date'):
        if len(g) < min_per_day:
            continue
        s.append(g[score_col].corr(g['ret'], method='spearman'))
    s = np.array([x for x in s if np.isfinite(x)])
    m, n, se, t = hac_t(s, 19)
    half = len(s) // 2
    return dict(ic=float(s.mean()), med=float(np.median(s)), t=t,
                first=float(s[:half].mean()), second=float(s[half:].mean()),
                pos=float((s > 0).mean()), ic_series=s)


def portfolio(d: pd.DataFrame, horizon: int, topk: int):
    bt = backtest(d, horizon, topk, use_sector_neutral=False)
    st = _stats(bt['top_net'].values, horizon)
    ex = bt['top_net'].values - bt['bench'].values
    se = block_bootstrap_se(ex, max(1, 252 // horizon))
    return dict(ir=st['ir'], excess_mean=float(ex.mean()) / horizon * 100,
                excess_ir=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252 / horizon)),
                excess_lo=float(ex.mean() / horizon * 100 - 1.96 * se / horizon * 100),
                excess_hi=float(ex.mean() / horizon * 100 + 1.96 * se / horizon * 100),
                wr=st.get('win_rate'), turnover=float(bt['turnover'].mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()

    df = attach_size_vol(load_panel(args.pred))
    print(f'样本 {len(df)} 行 / {df.date.nunique()} 个交易日 / '
          f'adv 缺失 {df.adv20.isna().mean():.1%} / vol 缺失 {df.vol20.isna().mean():.1%}')

    variants = {
        'N1 行业': (True, False),
        'N2 行业+规模+波动': (True, True),
        'N3 规模+波动': (False, True),
        'N4 不中性': (False, False),
    }
    rows = []
    for name, (us, uv) in variants.items():
        d = df.copy()
        d['score'] = np.nan
        for _, idx in d.groupby('date').groups.items():
            d.loc[idx, 'score'] = residualize(d.loc[idx], us, uv)
        d['prob'] = d['score']
        d['sector'] = 'ALL'
        r = ic_stats(d, 'score')
        p = portfolio(d, args.horizon, args.topk)
        rows.append(dict(变体=name, IC=r['ic'], IC中位=r['med'], HAC_t=r['t'],
                         前半=r['first'], 后半=r['second'], 正比例=r['pos'],
                         净IR=p['ir'], 超额IR=p['excess_ir'],
                         超额CI=f"[{p['excess_lo']:+.2f},{p['excess_hi']:+.2f}]",
                         换手=p['turnover']))
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(f"\nTopK={args.topk}  horizon={args.horizon}")
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
