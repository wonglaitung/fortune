#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关股扩池 vs 任意扩池对比（数据驱动关联度分类）

方法：
  1. 用"与现有 59 池收益的平均相关系数"衡量每只 universe 股的关联度（co-movement 代理）。
  2. 把 124 只 universe 按关联度中位拆分：related（高关联≈上下游/同行） vs unrelated（低关联）。
  3. 对 4 个池（59 / 59+related / 59+unrelated / 59+all）分别算：
     - Alpha158 全因子等权复合的日度横截面 Rank IC（均值/ICIR/std）
     - 有效独立样本 n_eff = m / (1 + (m-1)·ρ̄)，ρ̄=池内平均两两收益相关
结论：若 related 池 ICIR 显著优于 unrelated 池，则相关扩池在稳定性上更优；
      若两者都≈0，则相关扩池也救不了基础信号。

用法：python3 ml_services/related_pool_test.py
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.alpha158_baseline import alpha_factors, load_ohlcv as load_feat_ohlcv
from ml_services.alpha158_universe import load_ohlcv_from

UNI_CACHE = 'data/hk_universe_cache'
FEAT_CACHE = 'data/feature_cache'
TEST_START = '2021-10-01'
TEST_END = '2026-08-01'
HORIZON = 20


def daily_returns(ohlcv):
    r = {}
    for code, d in ohlcv.items():
        ret = d['Close'].pct_change()
        ret = ret[(ret.index >= TEST_START) & (ret.index < TEST_END)]
        r[code] = ret
    return pd.DataFrame(r)


def pool_ic(panel):
    """日度横截面 Rank IC 的 mean/ICIR/std"""
    ics = []
    for d, g in panel.groupby('date'):
        g = g.dropna(subset=['score', 'fwd_ret'])
        if len(g) < 15:
            continue
        ic = g['score'].corr(g['fwd_ret'], method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 30:
        return dict(n=len(ics), ic=np.nan, icir=np.nan, std=np.nan)
    return dict(n=len(ics), ic=ics.mean(), icir=ics.mean() / ics.std(ddof=1), std=ics.std(ddof=1))


def effective_n(ret_df, codes):
    m = len(codes)
    if m < 3:
        return np.nan
    sub = ret_df[list(codes)]
    c = sub.corr().values
    iu = np.triu_indices(m, k=1)
    rho = np.nanmean(c[iu]) if iu[0].size else np.nan
    if rho is None or not np.isfinite(rho) or (1 + (m - 1) * rho) <= 0:
        return m
    return m / (1 + (m - 1) * rho)


def main(out_md=None):
    feat = load_feat_ohlcv()               # 59 只（feature_cache）
    uni = load_ohlcv_from(UNI_CACHE)       # 124 只（hk_universe_cache）
    base_codes = sorted(feat.keys())
    uni_codes = [c for c in sorted(uni.keys())
                 if c not in set(base_codes) and c.lower() not in ('hsi', 'hsi.pkl')]

    rets = daily_returns({**feat, **{c: uni[c] for c in uni_codes}})

    # 关联度 = 与 59 池等权收益的相关（共同日期，逐股对齐）
    pool_mean = rets[base_codes].mean(axis=1)
    avg_corr = {}
    for c in uni_codes:
        s = rets[c].dropna()
        idx = s.index.intersection(pool_mean.dropna().index)
        if len(idx) < 60:
            avg_corr[c] = 0.0
            continue
        corr = s.loc[idx].corr(pool_mean.loc[idx])
        avg_corr[c] = corr if np.isfinite(corr) else 0.0
    ac = pd.Series(avg_corr).sort_values(ascending=False)
    med = ac.median()
    related = list(ac[ac >= med].index)
    unrelated = list(ac[ac < med].index)

    # 构建各池 panel（Alpha158 复合）
    def build_panel(codes):
        ohlcv = {c: feat[c] if c in feat else uni[c] for c in codes if c in feat or c in uni}
        frames = []
        for c, d in ohlcv.items():
            f = alpha_factors(d)
            f['fwd_ret'] = d['Close'].shift(-HORIZON) / d['Close'] - 1
            f['code'] = c
            frames.append(f.dropna(subset=['fwd_ret']))
        p = pd.concat(frames)
        p = p[(p.index >= TEST_START) & (p.index < TEST_END)].reset_index()
        p = p.rename(columns={p.columns[0]: 'date'})
        p['date'] = pd.to_datetime(p['date'])
        feat_cols = [x for x in alpha_factors(next(iter(ohlcv.values()))).columns if x in p.columns]
        gm = p.groupby('date')[feat_cols].transform('mean')
        gs = p.groupby('date')[feat_cols].transform('std')
        p[feat_cols] = ((p[feat_cols] - gm) / (gs + 1e-9)).replace([np.inf, -np.inf], np.nan)
        p['score'] = p[feat_cols].mean(axis=1)
        return p

    pools = {
        '59只（原）': base_codes,
        '59 + related(高关联)': base_codes + related,
        '59 + unrelated(低关联)': base_codes + unrelated,
        '59 + all(124)': base_codes + uni_codes,
    }

    L = []
    L.append("# 相关扩池 vs 任意扩池（数据驱动关联度）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 窗口: {TEST_START}~{TEST_END}　关联度=与59池收益平均相关（中位拆分）\n")
    L.append(f"- related {len(related)} 只（平均相关 {ac[related].mean():.3f}）| "
             f"unrelated {len(unrelated)} 只（平均相关 {ac[unrelated].mean():.3f}）\n")
    L.append("| 池 | 股票数 | 复合IC | ICIR | 日IC std | n_eff(有效独立) |")
    L.append("|----|--------|--------|------|---------|-----------------|")
    for name, codes in pools.items():
        codes = [c for c in codes if c in rets.columns]
        p = build_panel(codes)
        s = pool_ic(p)
        ne = effective_n(rets, codes)
        L.append(f"| {name} | {len(codes)} | {s['ic']:+.4f} | {s['icir']:.3f} | {s['std']:.4f} | {ne:.0f} |")
    L.append("")
    L.append("## 结论\n")
    L.append("- IC/ICIR 越高越好；n_eff 越高说明独立信息越多（相关性越低）。")
    L.append("- 若 related 池 ICIR > unrelated 池 → 相关扩池在稳定性上更优；")
    L.append("  若两者都≈0 → 相关扩池也救不了基础信号。")
    L.append("- 注意：同仓共动相关 ≠ 上下游领先滞后，后者是另一类信号。")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    main('output/related_pool_test.md')