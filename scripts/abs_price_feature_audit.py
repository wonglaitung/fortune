#!/usr/bin/env python3
"""T3 审计：港股模型特征中的绝对价格漏网（E3 实验输入）

问题：ABSOLUTE_PRICE_FEATURES 排除表（当前 42 项，AGENTS 文档写 40，
系实验2 连带改动 40→42）是否覆盖了模型实际使用的特征中所有「绝对价格量纲」项？
若有漏网 → 绝对价格跨股票不可比 → E3（剔除）实验的候选清单。

方法（两路交叉）：
1. 名称审计：50 folds top_features（importance top100）并集，筛名字像绝对价格
   （Price/Close/High/Low/MA\\d/ATR$ /OBV/VWAP/Turnover/Volume/Support/Resistance/
    Channel/TP 等模式）但不在 42 项排除表中的。
2. 值审计（防改名漏网）：top 并集 ∩ 特征缓存数值列，在窗口内 5 个截面日算
   跨股票 Spearman(特征, Close)——|均值| >= 0.8 且方向稳定 → 值量纲上是
   绝对价格代理。判 0.8 因子：比率类特征（MA_Ratio 等）与价格水平相关远低于此。
只读不改，输出 json。
"""
import glob
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FOLD_JSON = os.path.join(ROOT, 'output', '20261005_164759_lightgbm_20d',
                         'fold_metrics_detail.json')
OUT_PATH = os.path.join(ROOT, 'scripts', 't3_abs_price_audit_result.json')
VALUE_DATES = ['2026-02-02', '2026-03-02', '2026-04-01', '2026-06-01',
               '2026-07-01']
ABS_NAME_PATTERNS = [
    r'^MA\d', r'^Price_', r'_Price$', r'^Close', r'^Open$', r'^High', r'^Low',
    r'^ATR', r'_ATR$', r'^OBV', r'OBV', r'^VWAP', r'Turnover', r'^Volume',
    r'Support', r'Resistance', r'Channel', r'^TP$', r'^TP_', r'Stock_Price',
    r'Prev_Close', r'^TP\b', r'Year_High', r'Year_Low', r'52',
]

# 公式核实（2026-10-05 grep ml_trading_model.py 逐个确认）：名字模式初筛命中
# 但公式实为比率/百分比/分位归一 → 非绝对量纲，属假阳性。漏网判定必须扣除。
FORMULA_VERIFIED_NORMALIZED = {
    'ATR_Pct': 'ATR/prev_close',
    'ATR_Change_5d': 'ATR.pct_change(5)',
    'ATR_Change_10d': 'ATR.pct_change(10)',
    'ATR_Ratio_120d': 'ATR/ATR_MA120',
    'ATR_Risk_Score': 'ATR分位数',
    'ATR_Stop_Loss_Distance': '2*ATR/Close',
    'Channel_High_Ratio_20d': '比率',
    'Channel_Low_Ratio_20d': '比率',
    'Channel_Width_Ratio_20d': '比率',
    'Distance_Support_120d': '(Close-Support)/Close',
    'Distance_Resistance_120d': '(Resistance-Close)/Close',
    'High_Position_60d': '(Close-Low60)/(High60-Low60)',
    'MA120_Deviation': '(Close-MA120)/MA120*100',
    'MA250_Deviation': '(Close-MA250)/MA250*100',
    'MA120_Slope': '(MA120-MA120.shift(10))/MA120.shift(10)',
    'MA250_Slope': '(MA250-MA250.shift(20))/MA250.shift(20)',
    'OBV_Change_5d': 'OBV.pct_change(5)',
    'Price_Channel_Position_20d': '窗内位置',
    'Price_Pct_20d': '窗内min-max位置',
    'Price_Percentile': '60日分位',
    'Price_Position_Risk': 'Distance_Support比值',
    'Price_Ratio_MA50': 'Close/MA50',
    'Price_Ratio_MA120': 'Close/MA120',
    'Price_Return_Mean_30d': '30日收益均值',
    'Resistance_Ratio_120d': '比率',
    'Stock_Price_Stability_Score': '1-std/mean, clip[0,1]',
    'Support_Ratio_120d': '比率',
    'Turnover_Change_5d': 'Turnover.pct_change(5)',
    'Volume_Ratio_20d': 'Volume/Volume.rolling(20).mean()',
}


def main():
    from ml_services.ml_trading_model import ABSOLUTE_PRICE_FEATURES
    excl = list(ABSOLUTE_PRICE_FEATURES)

    with open(FOLD_JSON) as fh:
        folds = json.load(fh)['folds']
    freq, union = {}, set()
    for f in folds:
        for item in f.get('top_features', []):
            name = item['feature']
            union.add(name)
            freq[name] = freq.get(name, 0) + 1

    rx = re.compile('|'.join(ABS_NAME_PATTERNS), re.IGNORECASE)
    name_hits = sorted(n for n in union
                       if rx.search(n) and n not in excl)
    name_hits_listed = sorted(n for n in union if rx.search(n) and n in excl)

    # 值审计：最新 feature cache 各 code，5 个截面日 Spearman(特征, Close)
    latest = {}
    from ml_services.feature_cache_utils import latest_feature_cache_files
    _files = latest_feature_cache_files(os.path.join(ROOT, 'data', 'feature_cache'))
    latest = {c: (os.path.basename(p).split('_')[1], p) for c, p in _files.items()}
    value_hits = {}
    # 值审计：5 个日期各做一次跨股票截面 Spearman(特征, Close)
    cross = {}
    rows_by_date = {d: {} for d in VALUE_DATES}
    for code, (_, path) in latest.items():
        try:
            with open(path, 'rb') as fh:
                df = pickle.load(fh)['data']['stock_df']
        except Exception:
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        for d in VALUE_DATES:
            td = pd.Timestamp(d)
            idx = df.index[df.index >= td]
            if len(idx):
                rows_by_date[d][code] = df.loc[idx[0]]
    for d, rows in rows_by_date.items():
        if len(rows) < 20:
            continue
        mat = pd.DataFrame(rows).T
        if 'Close' not in mat.columns:
            continue
        for col in (union | {'MA5', 'MA250'}):  # MA5/MA250 为阳性对照列
            if col not in mat.columns or col == 'Close':
                continue
            s = pd.to_numeric(mat[col], errors='coerce')
            c = pd.to_numeric(mat['Close'], errors='coerce')
            k = s.notna() & c.notna()
            if k.sum() >= 20 and s[k].nunique() > 2:
                r = float(s[k].corr(c[k], method='spearman'))
                cross.setdefault(col, []).append(r)
    for col, rs in cross.items():
        mr = float(np.mean(rs))
        if abs(mr) >= 0.8 and len([x for x in rs if np.sign(x) == np.sign(mr)]) \
                == len(rs):
            value_hits[col] = {'mean_spearman_vs_close': round(mr, 3),
                               'per_date': [round(x, 3) for x in rs]}

    leak = sorted((set(name_hits) | set(value_hits)) - set(excl)
                  - set(FORMULA_VERIFIED_NORMALIZED))
    # 阳性对照：审计逻辑须能抓到已知绝对价格列（MA5），否则空结果不可信
    control = {k: [round(x, 3) for x in v] for k, v in cross.items()
               if k in ('MA5', 'MA250', 'Close')}
    control_ok = bool(control.get('MA5')) and all(
        abs(x) >= 0.9 for x in control['MA5'])
    out = {
        'experiment': 'T3_absolute_price_feature_audit',
        'abs_list_size': len(excl), 'abs_list': excl,
        'n_folds': len(folds), 'top_union_size': len(union),
        'name_pattern_hits_not_in_list': name_hits,
        'name_pattern_hits_in_list_count': len(name_hits_listed),
        'name_hits_formula_verified_normalized': sorted(
            set(name_hits) & set(FORMULA_VERIFIED_NORMALIZED)),
        'value_audit_spearman_ge_0.8': value_hits,
        'positive_control_MAspearman': control,
        'positive_control_ok': control_ok,
        'leak_candidates_not_in_exclusion_list': leak,
        'top15_by_fold_frequency': sorted(freq.items(),
                                          key=lambda kv: -kv[1])[:15],
        'verdict': ('CONTROL_FAILED' if not control_ok
                    else ('LEAK_FOUND' if leak else 'NO_LEAK_FOUND')),
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True,
                         default=str)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(payload)
    print(json.dumps({k: out[k] for k in
                      ['abs_list_size', 'top_union_size',
                       'name_pattern_hits_not_in_list',
                       'value_audit_spearman_ge_0.8',
                       'positive_control_ok',
                       'leak_candidates_not_in_exclusion_list', 'verdict']},
                     ensure_ascii=False, indent=2))
    print(f"MD5={hashlib.md5(payload.encode('utf-8')).hexdigest()}")


if __name__ == '__main__':
    main()
