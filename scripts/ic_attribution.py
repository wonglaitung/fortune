#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IC 归因：预测分数在流动性 / Beta / 行业分层内是否仍有信息。

诊断口径（不做 alpha 声明）：
- 逐日横截面 Spearman IC（预测分数 vs 实际收益），按层分组后取均值
- 显著性用 Newey-West（lag = horizon，覆盖重叠收益的自相关）
- 流动性代理 = 20 日 ADV（HKD，Close×Volume，shift 1）
- Beta = 60 日滚动 beta（对等权全池市场收益，shift 1）
- 规模维度**算不了**（无股数/流通股）——本脚本不输出市值分层

用法：
  python3 scripts/ic_attribution.py --input output/<run>/prediction_analysis.csv --horizon 20
  # 可选：--out-dir output/_logs/attr20 --write-subsets
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SECTOR_MAPPING


def _norm_date(x):
    # utc=True 统一时区/朴素混用；只取日期部分
    try:
        s = pd.to_datetime(x, utc=True)
    except (ValueError, TypeError):
        s = pd.to_datetime(x, utc=True, format='mixed')
    return s.dt.tz_localize(None).dt.normalize()


def load_panel(snapshot_dir, codes):
    """返回 (close, volume) 长表：index=date, columns=code。"""
    closes, vols = {}, {}
    for code in codes:
        num = code.split('.')[0].zfill(5)
        p = os.path.join(snapshot_dir, f'{num}.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        idx = pd.to_datetime(df.index)
        idx = idx.tz_convert(None) if idx.tz is not None else idx
        df.index = idx.normalize()
        closes[code] = df['Close'].astype(float)
        vols[code] = df['Volume'].astype(float)
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).sort_index()
    return close, vol


def build_exposures(close, vol):
    value = close * vol
    adv20 = value.rolling(20, min_periods=10).mean().shift(1)
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    beta60 = ret.rolling(60, min_periods=40).cov(mkt).div(
        mkt.rolling(60, min_periods=40).var(), axis=0).shift(1)
    return adv20, beta60


def nw_t(x, lags):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    mu = x.mean()
    e = x - mu
    s = (e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        s += 2 * w * (e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(mu / se)


def daily_ic(df, group_col=None):
    """逐日（可分层）IC 序列。"""
    out = {}
    for d, g in df.groupby('date'):
        groups = [(None, g)] if group_col is None else g.groupby(group_col, observed=True)
        for key, gg in groups:
            gg = gg.dropna(subset=['pred', 'ret'])
            if len(gg) < 10 or gg['pred'].nunique() < 2 or gg['ret'].nunique() < 2:
                continue
            ic = stats.spearmanr(gg['pred'], gg['ret']).correlation
            if np.isfinite(ic):
                out.setdefault(key, []).append(ic)
    return out


def bucket_report(df, col, labels, horizon):
    rows = []
    for key, ics in daily_ic(df, col).items():
        rows.append({
            'bucket': key, 'n_days': len(ics),
            'mean_ic': float(np.mean(ics)),
            'nw_t': nw_t(ics, horizon),
        })
    return pd.DataFrame(rows).sort_values('bucket')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--snapshot', default='data/hk_broad/snapshot')
    ap.add_argument('--out-dir', default='output/_logs/attr')
    ap.add_argument('--write-subsets', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df['date'] = _norm_date(df['Date'])
    df = df.rename(columns={'Stock_Code': 'code', 'Predict_Prob': 'pred',
                            'Actual_Return': 'ret'})
    codes = sorted(df['code'].unique())
    close, vol = load_panel(args.snapshot, codes)
    adv20, beta60 = build_exposures(close, vol)

    long = df.set_index(['date', 'code'])
    long['adv20'] = adv20.stack().reindex(long.index)
    long['beta60'] = beta60.stack().reindex(long.index)
    long = long.reset_index()
    long['sector'] = long['code'].map(
        lambda c: (STOCK_SECTOR_MAPPING.get(c) or {}).get('sector', 'unknown'))

    # 每日分位：q1=最低流动性/最低 beta
    for col in ('adv20', 'beta60'):
        long[col + '_q'] = long.groupby('date')[col].transform(
            lambda s: pd.qcut(s.rank(method='first'), 5,
                              labels=[1, 2, 3, 4, 5]) if s.notna().sum() >= 25 else np.nan)

    os.makedirs(args.out_dir, exist_ok=True)
    lines = [f'# IC 归因（{args.horizon}d）', '',
             f'- 输入: `{args.input}`', f'- 行数: {len(long)}',
             f'- 流动性缺失: {int(long["adv20"].isna().sum())} / {len(long)}', '']

    all_ics = daily_ic(long, None)[None]
    lines += ['## 总体', '',
              f'- 平均逐日 IC: {np.mean(all_ics):.4f}（NW t={nw_t(all_ics, args.horizon):.2f}, n={len(all_ics)} 日）', '']

    for col, name in (('adv20_q', '流动性五分位（1=最低）'), ('beta60_q', 'Beta 五分位（1=最低）')):
        rep = bucket_report(long, col, None, args.horizon)
        lines += [f'## {name}', '',
                  '| 分位 | 日数 | 平均 IC | NW t |', '|---|---|---|---|']
        for _, r in rep.iterrows():
            lines.append(f"| {int(r['bucket'])} | {int(r['n_days'])} | {r['mean_ic']:.4f} | {r['nw_t']:.2f} |")
        lines.append('')

    sec = bucket_report(long, 'sector', None, args.horizon)
    lines += ['## 行业', '', '| 行业 | 日数 | 平均 IC | NW t |', '|---|---|---|---|']
    for _, r in sec.iterrows():
        lines.append(f"| {r['bucket']} | {int(r['n_days'])} | {r['mean_ic']:.4f} | {r['nw_t']:.2f} |")
    lines.append('')

    report = os.path.join(args.out_dir, f'ic_attribution_{args.horizon}d.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'报告: {report}')

    if args.write_subsets:
        subsets = {
            'all': long,
            'liq_q1': long[long['adv20_q'] == 1],
            'liq_q5': long[long['adv20_q'] == 5],
            'liq_ge_q3': long[long['adv20_q'] >= 3],
            'liq_ge_q4': long[long['adv20_q'] >= 4],
        }
        for name, sub in subsets.items():
            out = os.path.join(args.out_dir, f'{name}_{args.horizon}d.csv')
            sub.drop(columns=['date', 'sector']).rename(
                columns={'code': 'Stock_Code', 'pred': 'Predict_Prob',
                         'ret': 'Actual_Return'}).to_csv(out, index=False)
            print(f'子集: {out} ({len(sub)} 行)')


if __name__ == '__main__':
    main()
