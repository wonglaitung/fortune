#!/usr/bin/env python3
"""T2 枢纽实验：共同特征集跨市场 IC 对照（假设-演绎第一性原理）

假设 H2（停止规则判据）：若「同一批由 OHLCV 独立计算的共同特征」在 A 股有
横截面预测力而在港股没有，则差异源于市场对信号的横截面响应（结构性不可移植），
而非特征工程选型 → 触发停止规则，移植计划仅保留 E3（剔除绝对价格特征）。
反向（比值小）→ 特征响应无市场差异 → 继续 Phase 2（E1 关特征选择 / E3）。

【预注册判据 — 跑前写死，禁止事后修改】
窗口：2025-01-01 ~ 2026-07-31（与 T1/A 折测试期一致）
主口径 = 行业中性每日截面 IC 的窗口均值，每市场得 12 个特征的 IC 向量
  R = median_f(|IC_A|) / median_f(|IC_HK|)
  一致率 = share(sign(IC_A) == sign(IC_HK))
  R >= 3.0 且 一致率 <= 60%  → STOP_RULE（结构性不可移植，仅留 E3）
  R < 1.5                    → CONTINUE（进入 Phase 2：E1/E3）
  1.5 ~ 3.0                  → GRAY（仅低成本 E3，不开重实验）
原始（非中性）口径与非重叠 20d 子集 t 检验仅作辅助记录。

设计要点（隔离变量=市场）：
- 特征由本脚本同一函数从原始 OHLCV 计算，不经任一管线的特征矩阵
 （A 股无特征缓存；两管线特征集不同会污染对照）
- 特征整体 shift(1)（只用 T-1 及以前数据），标签 = Close(t+20)/Close(t)-1
  → 特征窗与标签窗零重叠
- 两市场完全相同的特征公式、中性化规则、聚合方式
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WINDOW_START = pd.Timestamp('2025-01-01')
WINDOW_END = pd.Timestamp('2026-07-31')
HORIZON = 20
MIN_COVER = 0.6
OUT_PATH = os.path.join(ROOT, 'scripts', 't2_common_feature_ic_result.json')

FEATS = ['ret_5', 'ret_20', 'ret_60', 'vol_20', 'vol_60', 'vol_ratio_20_60',
         'volume_ratio_20', 'volume_trend_5_20', 'hl_range_20',
         'ma_dist_20', 'ma_dist_60', 'up_days_20']


def compute_features(df):
    """从单股 OHLCV DataFrame 计算 12 个共同特征（最后整体 shift(1)）。"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    ret = c.pct_change(fill_method=None)
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    out = pd.DataFrame(index=df.index)
    out['ret_5'] = c / c.shift(5) - 1
    out['ret_20'] = c / c.shift(20) - 1
    out['ret_60'] = c / c.shift(60) - 1
    out['vol_20'] = vol20
    out['vol_60'] = vol60
    out['vol_ratio_20_60'] = vol20 / vol60
    out['volume_ratio_20'] = v / v.rolling(20).mean()
    out['volume_trend_5_20'] = v.rolling(5).mean() / v.rolling(20).mean()
    out['hl_range_20'] = (h.rolling(20).max() - l.rolling(20).min()) / c
    out['ma_dist_20'] = c / c.rolling(20).mean() - 1
    out['ma_dist_60'] = c / c.rolling(60).mean() - 1
    out['up_days_20'] = (ret > 0).astype(float).rolling(20).mean()
    return out.shift(1)  # 特征只用 T-1 及以前数据


def load_market(market):
    """返回 panel: dict[code -> DataFrame(OHLCV)]，sector dict。"""
    if market == 'HK':
        from config import TRAINING_STOCKS, STOCK_SECTOR_MAPPING
        codes = [c.replace('.HK', '') for c in TRAINING_STOCKS]
        import glob
        from ml_services.feature_cache_utils import latest_feature_cache_files
        _files = latest_feature_cache_files(os.path.join(ROOT, 'data', 'feature_cache'))
        latest = {c: (os.path.basename(p).split('_')[1], p) for c, p in _files.items()}
        panel, sec = {}, {}
        for code in codes:
            if code not in latest:
                continue
            try:
                with open(latest[code][1], 'rb') as fh:
                    df = pickle.load(fh)['data']['stock_df'][
                        ['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            except Exception:
                continue
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df = df[~df.index.duplicated(keep='last')].sort_index()
            panel[code] = df
            m = STOCK_SECTOR_MAPPING.get(f'{code}.HK', {})
            sec[code] = m.get('sector', 'unknown') if isinstance(m, dict) else m
        return panel, sec
    else:
        from a_stock_config import A_STOCK_TRAINING_LIST, A_STOCK_SECTOR_MAPPING
        panel, sec = {}, {}
        base = os.path.join(ROOT, 'data', 'a_stock_cache')
        for code in A_STOCK_TRAINING_LIST:
            path = os.path.join(base, f'{code}_1460d.pkl')
            if not os.path.exists(path):
                path = os.path.join(base, f'{code}.pkl')
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'rb') as fh:
                    df = pickle.load(fh)[
                        ['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            except Exception:
                continue
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df = df[~df.index.duplicated(keep='last')].sort_index()
            panel[code] = df
            m = A_STOCK_SECTOR_MAPPING.get(code, {})
            sec[code] = m.get('sector', 'unknown') if isinstance(m, dict) else m
        return panel, sec


def daily_ic(panel, sector, method):
    """逐日截面 IC：method='spearman' 原始 / 'neutral' 行业中性。

    返回 {feat: {date: ic}} 与有效日期列表。
    """
    feat_panels = {}
    fwd_panel = {}
    for code, df in panel.items():
        if len(df) < 90:
            continue
        feat_panels[code] = compute_features(df)
        c = df['Close']
        fwd_panel[code] = c.shift(-HORIZON) / c - 1.0
    feats = pd.DataFrame({f: pd.concat(
        {c: p[f] for c, p in feat_panels.items()}, axis=1).stack()
        for f in FEATS})
    fwd = pd.concat({c: s for c, s in fwd_panel.items()}, axis=1).stack()
    sec = pd.Series(sector)

    dates = sorted({d for d in feats.index.get_level_values(0)
                    if WINDOW_START <= d <= WINDOW_END})
    res = {f: {} for f in FEATS}
    used = []
    for d in dates:
        try:
            X = feats.xs(d, level=0)
            y = fwd.xs(d, level=0).reindex(X.index)
        except KeyError:
            continue
        mask = y.notna()
        X, y = X[mask], y[mask]
        if len(X) < max(8, MIN_COVER * len(panel)):
            continue
        if method == 'spearman':
            for f in FEATS:
                s = X[f].dropna()
                common = s.index.intersection(y.index)
                if len(common) >= 8:
                    res[f][d] = float(s.loc[common].corr(y.loc[common],
                                                         method='spearman'))
        else:  # neutral：特征与标签各自行业内去均值，单成员行业剔除
            g = pd.DataFrame({'y': y, 'g': sec.reindex(y.index)}).dropna()
            cnt = g.groupby('g')['g'].transform('size')
            keep = g[cnt >= 2].index
            g = g.loc[keep]
            if len(keep) < 8:
                continue
            y_res = g['y'] - g.groupby('g')['y'].transform('mean')
            for f in FEATS:
                s = X[f].reindex(keep).dropna()
                k2 = s.index.intersection(y_res.index)
                if len(k2) < 8:
                    continue
                gg = pd.DataFrame({'x': s.loc[k2],
                                   'g': sec.reindex(k2)}).dropna()
                cnt2 = gg.groupby('g')['g'].transform('size')
                gg = gg[cnt2 >= 2]
                if len(gg) < 8:
                    continue
                x_res = gg['x'] - gg.groupby('g')['x'].transform('mean')
                yr = y_res.reindex(x_res.index)
                if x_res.std() > 0 and yr.std() > 0:
                    res[f][d] = float(x_res.corr(yr))
        used.append(d)
    return res, used


def agg(res, dates):
    """窗口均值 + 非重叠子集 t 检验。"""
    out = {}
    sub = dates[::HORIZON]
    for f in FEATS:
        vals = pd.Series(res[f])
        v = vals.dropna()
        vs = vals.reindex(sub).dropna()
        t = None
        if len(vs) >= 5 and vs.std(ddof=1) > 0:
            t = float(vs.mean() / (vs.std(ddof=1) / np.sqrt(len(vs))))
        out[f] = {'ic_mean': float(v.mean()) if len(v) else None,
                  'n_dates': int(len(v)),
                  'ic_nonoverlap_mean': float(vs.mean()) if len(vs) else None,
                  't_nonoverlap': t}
    return out


def main():
    results = {}
    for market in ('A', 'HK'):
        panel, sec = load_market(market)
        raw, dates = daily_ic(panel, sec, 'spearman')
        neu, _ = daily_ic(panel, sec, 'neutral')
        results[market] = {'panel_size': len(panel), 'n_dates': len(dates),
                           'raw': agg(raw, dates), 'neutral': agg(neu, dates)}

    ic_a = {f: results['A']['neutral'][f]['ic_mean'] for f in FEATS}
    ic_h = {f: results['HK']['neutral'][f]['ic_mean'] for f in FEATS}
    med_a = float(np.median([abs(v) for v in ic_a.values() if v is not None]))
    med_h = float(np.median([abs(v) for v in ic_h.values() if v is not None]))
    ratio = med_a / med_h if med_h > 1e-9 else float('inf')
    agree = float(np.mean([np.sign(ic_a[f]) == np.sign(ic_h[f]) for f in FEATS]))

    if ratio >= 3.0 and agree <= 0.60:
        code = 'STOP_RULE'
        text = (f'R={ratio:.2f}>=3.0 且一致率{agree:.0%}<=60% → 停止规则触发：'
                '同特征集在 A 有预测力而 HK 无 → 结构性不可移植，仅保留 E3')
    elif ratio < 1.5:
        code = 'CONTINUE'
        text = f'R={ratio:.2f}<1.5 → 特征响应无显著市场差异 → 继续 Phase 2（E1/E3）'
    else:
        code = 'GRAY'
        text = (f'R={ratio:.2f}∈[1.5,3.0) → 灰区 → 仅跑低成本 E3，不开重实验')

    out = {
        'experiment': 'T2_common_feature_IC_hub_test',
        'preregistered_criteria': {
            'R>=3.0 and sign_agree<=0.60': 'STOP_RULE (structural, keep E3 only)',
            'R<1.5': 'CONTINUE (Phase 2: E1/E3)',
            'R in [1.5,3.0)': 'GRAY (E3 only, no heavy runs)',
        },
        'window': f'{WINDOW_START.date()} ~ {WINDOW_END.date()}',
        'horizon': HORIZON, 'features': FEATS,
        'per_feature_neutral_ic': {
            f: {'A': ic_a[f], 'HK': ic_h[f],
                'ratio_abs': (abs(ic_a[f]) / abs(ic_h[f])
                              if ic_h[f] and abs(ic_h[f]) > 1e-9 else None),
                'sign_agree': bool(np.sign(ic_a[f]) == np.sign(ic_h[f]))}
            for f in FEATS},
        'median_abs_ic': {'A': med_a, 'HK': med_h},
        'R_median_abs_ic': ratio, 'sign_agreement': agree,
        'verdict_code': code, 'verdict_text': text,
        'A': results['A'], 'HK': results['HK'],
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True,
                         default=str)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(payload)
    print(json.dumps({k: out[k] for k in
                      ['median_abs_ic', 'R_median_abs_ic', 'sign_agreement',
                       'verdict_code', 'verdict_text']},
                     ensure_ascii=False, indent=2))
    for f in FEATS:
        print(f"{f:24s} A={ic_a[f]:+.4f} HK={ic_h[f]:+.4f} "
              f"sign={'OK' if np.sign(ic_a[f]) == np.sign(ic_h[f]) else 'DIFF'}")
    print(f"MD5={hashlib.md5(payload.encode('utf-8')).hexdigest()}  OUT={OUT_PATH}")


if __name__ == '__main__':
    main()
