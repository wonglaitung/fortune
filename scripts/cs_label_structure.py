#!/usr/bin/env python3
"""T1 对照实验：标签横截面结构（20d 前向收益的截面分散度）

假设 H1：港股池内「原材料薄」——同一信号下可获得的收益分散度显著低于 A 股，
是中性 IC 0.029 vs 0.114 的池层成因（训练窗/涨跌停/穿越等已排除）。

【预注册判据 — 跑前写死，禁止事后修改】
窗口：2025-01-01 ~ 2026-07-31（前向收益起始日，与 A 股 19 折测试期对齐）
主指标 R = mean_date std_cross(20d前向收益, 行业中性) 的 HK / A 比值
  R < 0.60         → 支持 H1（港股标签分散度不足 A 股 60%）
  0.80 <= R <= 1.20 → H1 出局（两市场标签分散度相当）
  R > 1.20          → H1 出局（反向：港股更分散）
  0.60 ~ 0.80       → 灰区（弱支持；去留由 T2 枢纽判据决定）
辅助（不参与判定，仅记录）：同窗日收益平均两两相关、原始口径 σ、非重叠 20d 子集复核

数据源：港股 feature_cache 最新 _shift 文件（raw stock_cache 仅 12 只不够）；
        A 股 a_stock_cache/{code}_1460d.pkl（缺则 {code}.pkl）。
观察记录（已验证）：港股特征缓存索引标签 = 真实交易日 - 1 个日历日
（dow={Mon-Thu,Sun}，58/59 code 统一；帧内自洽无泄漏，窗口边界差 1 天可忽略）。
"""
import glob
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
MIN_COVER = 0.6  # 当日有效股票占比下限
OUT_PATH = os.path.join(ROOT, 'scripts', 't1_label_structure_result.json')


def load_hk_close():
    """港股：feature_cache 每 code 最新 _shift 文件 → Close（qfq）。"""
    from config import TRAINING_STOCKS, STOCK_SECTOR_MAPPING
    codes = [c.replace('.HK', '') for c in TRAINING_STOCKS]
    latest = {}
    from ml_services.feature_cache_utils import latest_feature_cache_files
    _files = latest_feature_cache_files(os.path.join(ROOT, 'data', 'feature_cache'))
    latest = {c: (os.path.basename(p).split('_')[1], p) for c, p in _files.items()}
    closes, sector, dropped = {}, {}, []
    for code in codes:
        if code not in latest:
            dropped.append({'code': code, 'reason': 'no_cache'})
            continue
        try:
            with open(latest[code][1], 'rb') as fh:
                obj = pickle.load(fh)
            s = obj['data']['stock_df']['Close'].copy()
        except Exception as e:
            dropped.append({'code': code, 'reason': f'read_err:{e}'})
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep='last')].sort_index()
        closes[code] = s
        m = STOCK_SECTOR_MAPPING.get(f'{code}.HK', {})
        sector[code] = m.get('sector', 'unknown') if isinstance(m, dict) else m
    return closes, sector, dropped, len(codes)


def load_a_close():
    """A 股：a_stock_cache _1460d 优先，缺则 plain pkl。"""
    from a_stock_config import A_STOCK_TRAINING_LIST, A_STOCK_SECTOR_MAPPING
    closes, sector, dropped = {}, {}, []
    for code in A_STOCK_TRAINING_LIST:
        base = os.path.join(ROOT, 'data', 'a_stock_cache')
        path = os.path.join(base, f'{code}_1460d.pkl')
        if not os.path.exists(path):
            path = os.path.join(base, f'{code}.pkl')
        if not os.path.exists(path):
            dropped.append({'code': code, 'reason': 'no_cache'})
            continue
        try:
            with open(path, 'rb') as fh:
                df = pickle.load(fh)
            s = df['Close'].copy()
        except Exception as e:
            dropped.append({'code': code, 'reason': f'read_err:{e}'})
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep='last')].sort_index()
        closes[code] = s
        m = A_STOCK_SECTOR_MAPPING.get(code, {})
        sector[code] = m.get('sector', 'unknown') if isinstance(m, dict) else m
    return closes, sector, dropped, len(A_STOCK_TRAINING_LIST)


def cross_section_stats(closes, sector):
    """逐日截面：20d 前向收益的原始/行业中性 std + 窗内日收益两两相关。"""
    close = pd.DataFrame(closes).sort_index()
    fwd = close.shift(-HORIZON) / close - 1.0
    ret = close.pct_change(fill_method=None)
    sec = pd.Series(sector)

    dates = [d for d in fwd.index
             if WINDOW_START <= d <= WINDOW_END
             and fwd.loc[d].notna().sum() >= max(8, MIN_COVER * fwd.shape[1])]
    raw_std, neu_std = [], []
    for d in dates:
        row = fwd.loc[d].dropna()
        raw_std.append(float(row.std(ddof=1)))
        g = pd.DataFrame({'y': row, 'g': sec.reindex(row.index)}).dropna()
        cnt = g.groupby('g')['g'].transform('size')
        g = g[cnt >= 2]  # 单成员行业无法去均值，当日剔除
        resid = g['y'] - g.groupby('g')['y'].transform('mean')
        neu_std.append(float(resid.std(ddof=1)) if len(resid) >= 8 else np.nan)

    sub = dates[::HORIZON]  # 非重叠 20d 子集复核
    sub_neu = [neu_std[dates.index(d)] for d in sub]
    sub_neu = [x for x in sub_neu if not np.isnan(x)]

    # 同窗日收益平均两两相关（描述性，不参与判定）
    r = ret.loc[(ret.index >= WINDOW_START) & (ret.index <= WINDOW_END)]
    r = r.dropna(axis=1, how='any')
    if r.shape[1] >= 3:
        cm = r.corr()
        iu = np.triu_indices_from(cm.values, k=1)
        avg_corr = float(cm.values[iu].mean())
    else:
        avg_corr = None

    return {
        'n_dates': len(dates),
        'n_stocks_used': int(close.shape[1]),
        'raw_std_mean': float(np.nanmean(raw_std)),
        'raw_std_std': float(np.nanstd(raw_std)),
        'neutral_std_mean': float(np.nanmean(neu_std)),
        'neutral_std_std': float(np.nanstd(neu_std)),
        'neutral_std_median': float(np.nanmedian(neu_std)),
        'neutral_std_nonoverlap_mean': float(np.mean(sub_neu)) if sub_neu else None,
        'n_nonoverlap': len(sub_neu),
        'avg_pairwise_corr_daily': avg_corr,
        'n_corr_stocks': int(r.shape[1]),
    }


def verdict(r_value):
    if r_value is None:
        return 'INVALID', '比值不可计算'
    if r_value < 0.60:
        return 'SUPPORT_H1', f'R={r_value:.3f} < 0.60 → 支持 H1「港股标签原材料薄」'
    if r_value <= 1.20:
        return 'H1_REJECTED', f'R={r_value:.3f} ∈ [0.80,1.20] → H1 出局' \
            if r_value >= 0.80 else \
            f'R={r_value:.3f} ∈ (0.60,0.80) → 灰区（弱支持，去留由 T2 枢纽判据决定）'
    return 'H1_REJECTED_INVERSE', f'R={r_value:.3f} > 1.20 → H1 出局（反向：港股更分散）'


def main():
    hk_close, hk_sec, hk_drop, hk_n = load_hk_close()
    a_close, a_sec, a_drop, a_n = load_a_close()

    hk = cross_section_stats(hk_close, hk_sec)
    a = cross_section_stats(a_close, a_sec)

    r_main = hk['neutral_std_mean'] / a['neutral_std_mean'] if a['neutral_std_mean'] else None
    r_nonoverlap = (hk['neutral_std_nonoverlap_mean'] / a['neutral_std_nonoverlap_mean']
                    if hk['neutral_std_nonoverlap_mean'] and a['neutral_std_nonoverlap_mean']
                    else None)
    r_raw = hk['raw_std_mean'] / a['raw_std_mean'] if a['raw_std_mean'] else None
    code, text = verdict(r_main)

    result = {
        'experiment': 'T1_label_cross_section_structure',
        'preregistered_criteria': {
            'R_neutral < 0.60': 'SUPPORT_H1',
            'R_neutral in [0.80,1.20]': 'H1_REJECTED',
            'R_neutral > 1.20': 'H1_REJECTED_INVERSE',
            'R_neutral in (0.60,0.80)': 'GRAY_ZONE -> decision left to T2',
        },
        'window': f'{WINDOW_START.date()} ~ {WINDOW_END.date()}', 'horizon': HORIZON,
        'HK': hk, 'A': a,
        'HK_pool_loaded': len(hk_close), 'HK_pool_expected': hk_n,
        'A_pool_loaded': len(a_close), 'A_pool_expected': a_n,
        'HK_dropped': hk_drop, 'A_dropped': a_drop,
        'ratios_HK_over_A': {
            'R_neutral_main': r_main,
            'R_neutral_nonoverlap': r_nonoverlap,
            'R_raw': r_raw,
        },
        'verdict_code': code, 'verdict_text': text,
        'avg_pairwise_corr': {'HK': hk['avg_pairwise_corr_daily'],
                              'A': a['avg_pairwise_corr_daily']},
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                         default=str)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(payload)
    md5 = hashlib.md5(payload.encode('utf-8')).hexdigest()
    print(payload)
    print(f'\nMD5={md5}  OUT={OUT_PATH}')


if __name__ == '__main__':
    main()
