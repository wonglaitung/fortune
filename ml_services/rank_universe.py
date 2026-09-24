#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更大池模型重新学习（Part B 第 2 步）

目的：扩池后朴素 Alpha158 等权复合 IC 被稀释（§5.14），用 LightGBM 在更大池上
**重新学习**，看能否找回/提升信号。对照：
  - 124 只池：回归 / LambdaRank
  - 59 只池：回归（同特征同折）
参考：朴素等权复合 IC：59 只 0.0088 / 124 只 0.0049。

用法：python3 ml_services/rank_universe.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.alpha158_baseline import alpha_factors
from ml_services.alpha158_universe import load_ohlcv_from

FEAT_CACHE = 'data/feature_cache'
UNI_CACHE = 'data/hk_universe_cache'
TEST_START = '2021-10-01'
TEST_END = '2026-08-01'
MIN_TRAIN = 3000
EMBARGO_DAYS = 20


def build_panel(ohlcv, horizon=20):
    frames = []
    for code, d in ohlcv.items():
        f = alpha_factors(d)
        f['fwd_ret'] = d['Close'].shift(-horizon) / d['Close'] - 1
        f['code'] = code
        frames.append(f.dropna(subset=['fwd_ret']))
    panel = pd.concat(frames)
    panel = panel[(panel.index >= TEST_START) & (panel.index < TEST_END)]
    panel = panel.reset_index()
    panel = panel.rename(columns={panel.columns[0]: 'date'})
    panel['date'] = pd.to_datetime(panel['date'])
    feat = [c for c in alpha_factors(next(iter(ohlcv.values()))).columns if c in panel.columns]
    gm = panel.groupby('date')[feat].transform('mean')
    gs = panel.groupby('date')[feat].transform('std')
    panel[feat] = ((panel[feat] - gm) / (gs + 1e-9)).replace([np.inf, -np.inf], np.nan)
    panel['rank_label'] = panel.groupby('date')['fwd_ret'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=False))
    panel['y'] = panel['fwd_ret'] - panel.groupby('date')['fwd_ret'].transform('mean')
    return panel, feat


def run_wf(panel, feat, objective, horizon=20):
    import lightgbm as lgb
    months = sorted(panel['date'].dt.to_period('M').unique())
    test_months = [m for m in months if pd.Period(TEST_START, 'M') <= m < pd.Period(TEST_END, 'M')]
    out = []
    for m in test_months:
        te = panel[panel['date'].dt.to_period('M') == m]
        cut = m.start_time - pd.Timedelta(days=EMBARGO_DAYS)
        tr = panel[panel['date'] < cut]
        if len(tr) < MIN_TRAIN:
            continue
        try:
            if objective == 'lambdarank':
                tr = tr.sort_values('date')
                y = tr['rank_label'].values.astype(int)
                grp = tr.groupby('date').size().values
                ds = lgb.Dataset(tr[feat], label=y, group=grp, free_raw_data=False)
                model = lgb.train(dict(objective='lambdarank', learning_rate=0.05,
                                       num_leaves=31, min_child_samples=50, subsample=0.8,
                                       colsample_bytree=0.6, verbose=-1, random_state=42),
                                  ds, num_boost_round=200)
                sc = model.predict(te[feat])
            else:
                model = lgb.LGBMRegressor(objective='regression', learning_rate=0.05,
                                          num_leaves=31, min_child_samples=50, subsample=0.8,
                                          colsample_bytree=0.6, n_estimators=200, verbose=-1,
                                          random_state=42)
                model.fit(tr[feat], tr['y'].values)
                sc = model.predict(te[feat])
            out.append(te[['date', 'code', 'fwd_ret']].assign(score=sc))
        except Exception as e:
            print(f"  fold {m} {objective} 失败: {e}")
    return pd.concat(out) if out else pd.DataFrame(columns=['date', 'code', 'fwd_ret', 'score'])


def evaluate(d):
    ics = []
    for dte, g in d.groupby('date'):
        if len(g) < 15:
            continue
        ic = spearmanr(g['score'], g['fwd_ret']).correlation
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    spread = []
    for dte, g in d.groupby('date'):
        if len(g) < 20:
            continue
        g = g.copy()
        g['qn'] = pd.qcut(g['score'].rank(method='first'), 5, labels=False)
        spread.append(g.loc[g['qn'] == 4, 'fwd_ret'].mean() - g.loc[g['qn'] == 0, 'fwd_ret'].mean())
    return dict(ic=ics.mean(), icir=ics.mean() / ics.std(ddof=1), n=len(ics),
                pos=(ics > 0).mean(), spread=np.mean(spread) if spread else np.nan)


def main(out_md=None):
    uni = load_ohlcv_from(UNI_CACHE)
    feat59 = load_ohlcv_from(FEAT_CACHE)
    p_uni, f_uni = build_panel(uni)
    p_59, f_59 = build_panel(feat59)

    results = {}
    results['124只-回归'] = evaluate(run_wf(p_uni, f_uni, 'regression'))
    results['124只-LambdaRank'] = evaluate(run_wf(p_uni, f_uni, 'lambdarank'))
    results['59只-回归'] = evaluate(run_wf(p_59, f_59, 'regression'))

    L = []
    L.append("# 更大池模型重新学习（LightGBM）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 窗口: {TEST_START} ~ {TEST_END}　特征: Alpha158 核心 48 因子（横截面 z-score）\n")
    L.append("| 池/目标 | Rank IC | ICIR | 正比例 | 期数 | 多空Q5-Q1/期 |")
    L.append("|---------|--------|------|--------|------|-------------|")
    for k, v in results.items():
        L.append(f"| {k} | {v['ic']:+.4f} | {v['icir']:.3f} | {v['pos']*100:.0f}% | {v['n']} | {v['spread']*100:+.2f}% |")
    L.append("")
    L.append("## 对照\n")
    L.append("- 朴素等权复合：59只 IC +0.0088 / 124只 +0.0049")
    L.append("- 若 124只-模型 IC 明显 > 0.0049（尤其 >0.03），则'扩池+模型重学习'成立；")
    L.append("  若与朴素复合相当或更低，则扩池未带来可学信号。")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    main('output/rank_universe_20d.md')