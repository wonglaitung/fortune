#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法#2 MVP：排序目标 vs 回归目标（同特征、同折，A/B 对照）

目的：判断把目标从"预测收益大小"改成"横截面排序(LambdaRank)"是否能提升 Rank IC/ICIR。

数据：data/feature_cache/*_shift.pkl（OHLC + 特征），特征取
      data/feature_selection/statistical_features_latest.txt（Top 500）。
折：按自然月滚动；train=测试月之前的全部数据（去掉测试前 embargo=horizon 天）。

评估：逐日横截面 Rank IC 的均值 / ICIR / 正比例；分位多空（Q5-Q1）。

用法：
  python3 ml_services/rank_model.py --horizon 20
  python3 ml_services/rank_model.py --horizon 5
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEAT_FILE = 'data/feature_selection/statistical_features_latest.txt'
CACHE_DIR = 'data/feature_cache'
TEST_START = '2023-06-01'
TEST_END = '2026-08-01'
MIN_TRAIN = 3000


def load_features():
    if not os.path.exists(FEAT_FILE):
        return None
    with open(FEAT_FILE, encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_panel(horizon, feats):
    """拼接 panel：index=(date,code)，columns=特征 + fwd_ret"""
    files = {}
    for p in glob.glob(os.path.join(CACHE_DIR, '*_shift.pkl')):
        code = os.path.basename(p).split('_')[0]
        if code not in files or p > files[code]:
            files[code] = p
    frames = []
    for code, p in files.items():
        try:
            df = pd.read_pickle(p)['data']['stock_df']
        except Exception:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        keep = ['Close'] + [c for c in feats if c in df.columns]
        d = df[keep].copy()
        d['fwd_ret'] = d['Close'].shift(-horizon) / d['Close'] - 1.0
        d['code'] = code
        d = d.dropna(subset=['fwd_ret'])
        frames.append(d)
    panel = pd.concat(frames)
    panel = panel[(panel.index >= '2022-01-01') & (panel.index < TEST_END)]
    return panel


def month_folds(panel):
    months = sorted(pd.Series(panel.index).dt.to_period('M').unique())
    return [m for m in months if pd.Period(TEST_START, 'M') <= m < pd.Period(TEST_END, 'M')]


def _fit_predict(tr, te, feature_cols, objective, horizon):
    import lightgbm as lgb
    params = dict(objective=objective, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.6,
                  n_estimators=200, verbose=-1, random_state=42)
    if objective == 'lambdarank':
        tr = tr.sort_values('date')
        y = tr['rank_label'].values.astype(int)
        grp = tr.groupby('date').size().values
        ds = lgb.Dataset(tr[feature_cols], label=y, group=grp, free_raw_data=False)
        params.pop('n_estimators')
        model = lgb.train(params, ds, num_boost_round=200)
        return model.predict(te[feature_cols])
    else:  # regression
        model = lgb.LGBMRegressor(**params)
        model.fit(tr[feature_cols], tr['y'].values)
        return model.predict(te[feature_cols])


def daily_rank_ic(df, score_col):
    ics = []
    for d, g in df.groupby('date'):
        if len(g) < 10:
            continue
        from scipy.stats import spearmanr
        ic = spearmanr(g[score_col], g['fwd_ret']).correlation
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    mean = ics.mean(); std = ics.std(ddof=1)
    return mean, (mean / std if std > 0 else np.nan), (ics > 0).mean(), len(ics)


def quantile_spread(df, score_col, q=5):
    rows = []
    for d, g in df.groupby('date'):
        if len(g) < q * 2:
            continue
        g = g.copy()
        g['qn'] = pd.qcut(g[score_col].rank(method='first'), q, labels=False)
        hi = g.loc[g['qn'] == q - 1, 'fwd_ret'].mean()
        lo = g.loc[g['qn'] == 0, 'fwd_ret'].mean()
        rows.append(hi - lo)
    return float(np.mean(rows)) if rows else np.nan


def run(horizon, out_md):
    feats = load_features()
    print(f"特征数: {len(feats) if feats else 0}")
    panel = load_panel(horizon, feats)
    feature_cols = [c for c in feats if c in panel.columns]
    print(f"panel: {panel.shape}  特征可用: {len(feature_cols)}  "
          f"股票: {panel['code'].nunique()}  日期: {panel.index.nunique()}")

    # rank 标签：按日分位（0..9）
    panel = panel.reset_index()
    panel = panel.rename(columns={panel.columns[0]: 'date'})
    panel['date'] = pd.to_datetime(panel['date'])
    panel['rank_label'] = panel.groupby('date')['fwd_ret'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=False))

    # 横截面标准化（按日 z-score）+ 相对收益标签（去市场 beta）
    eps = 1e-6
    gm = panel.groupby('date')[feature_cols].transform('mean')
    gs = panel.groupby('date')[feature_cols].transform('std')
    panel[feature_cols] = ((panel[feature_cols] - gm) / (gs + eps)).replace([np.inf, -np.inf], np.nan)
    panel['y'] = panel['fwd_ret'] - panel.groupby('date')['fwd_ret'].transform('mean')

    folds = month_folds(panel.set_index('date'))
    print(f"测试折数: {len(folds)}")

    results = {k: [] for k in ('mse', 'lambdarank')}
    for i, m in enumerate(folds):
        te_mask = panel['date'].dt.to_period('M') == m
        te = panel[te_mask]
        start = m.start_time - pd.Timedelta(days=horizon)
        tr = panel[panel['date'] < start]
        if len(tr) < MIN_TRAIN:
            continue
        for obj in ('mse', 'lambdarank'):
            try:
                sc = _fit_predict(tr, te, feature_cols, obj, horizon)
                t = te.copy(); t['score'] = sc
                t['objective'] = obj
                results[obj].append(t[['date', 'code', 'fwd_ret', 'score']])
            except Exception as e:
                print(f"  fold {m} {obj} 失败: {e}")
    L = []
    L.append(f"# 排序目标 A/B 对照（{horizon}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 特征: Top500（`{FEAT_FILE}`）　测试: {TEST_START} ~ {TEST_END}\n")
    L.append("| 目标 | Rank IC | ICIR | 正比例 | 期数 | 多空Q5-Q1/期 |")
    L.append("|------|---------|------|--------|------|-------------|")
    summary = {}
    for obj in ('mse', 'lambdarank'):
        if not results[obj]:
            continue
        d = pd.concat(results[obj])
        ic, icir, pos, n = daily_rank_ic(d, 'score')
        spread = quantile_spread(d, 'score')
        summary[obj] = (ic, icir, spread)
        L.append(f"| {obj} | {ic:.4f} | {icir:.3f} | {pos*100:.1f}% | {n} | {spread*100:+.2f}% |")
    L.append("")
    L.append("## 结论\n")
    if 'lambdarank' in summary and 'mse' in summary:
        dic = summary['lambdarank'][0] - summary['mse'][0]
        L.append(f"- LambdaRank 相对 回归：Rank IC {dic:+.4f}，"
                 f"多空 {summary['lambdarank'][2]*100-summary['mse'][2]*100:+.2f}pp")
        L.append(f"- 对照基线 CatBoost（20d 全期）Rank IC ≈ 0.023")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


def main():
    ap = argparse.ArgumentParser(description='排序目标 vs 回归目标 A/B')
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 20])
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()
    out = args.output or f"output/rank_model_{args.horizon}d.md"
    run(args.horizon, out)


if __name__ == '__main__':
    main()
