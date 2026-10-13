#!/usr/bin/env python3
"""T4 对照：预测概率自稳定性（Predict_Prob 序列自相关）

观察：港股 20d PBO 0.90（组合期 39 vs A 股 71），概率排序稳定性差是否为
跨市场管线差异的可观测代理？

【预注册判据 — 跑前写死】
每市场 prediction_analysis.csv：按 Stock_Code 分组、Date 排序，
Predict_Prob 的行序 lag1/5/20 自相关（组内成对非 NaN）。
  |median_lag1_HK - median_lag1_A| <= 0.10 → 概率自稳定性无市场差异
  差值 > 0.10 → 有差异（方向与幅度记为 PBO 成因的描述性证据）
lag1>0.8 股票占比为辅助描述。只读不改。
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HK_CSV = os.path.join(ROOT, 'output', '20261005_164759_lightgbm_20d',
                      'prediction_analysis.csv')
A_CSV = os.path.join(ROOT, 'output', '20261004_000658_a_stock_catboost_20d',
                     'prediction_analysis.csv')
OUT_PATH = os.path.join(ROOT, 'scripts', 't4_prob_autocorr_result.json')
LAGS = [1, 5, 20]


def acf_by_stock(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    res = {k: [] for k in LAGS}
    per_stock_lag1 = {}
    for code, g in df.sort_values('Date').groupby('Stock_Code'):
        p = g['Predict_Prob'].reset_index(drop=True)
        for k in LAGS:
            a, b = p.shift(k), p
            m = a.notna() & b.notna()
            if m.sum() >= 30 and a[m].std() > 0 and b[m].std() > 0:
                r = float(a[m].corr(b[m]))
                res[k].append(r)
                if k == 1:
                    per_stock_lag1[str(code)] = round(r, 4)
    return res, per_stock_lag1


def summarize(res):
    out = {}
    for k in LAGS:
        v = np.array(res[k])
        out[f'lag{k}'] = {
            'median': float(np.median(v)) if len(v) else None,
            'mean': float(v.mean()) if len(v) else None,
            'n_stocks': int(len(v)),
        }
    v1 = np.array(res[1])
    out['lag1_share_gt_0.8'] = float((v1 > 0.8).mean()) if len(v1) else None
    return out


def main():
    hk_res, hk_per = acf_by_stock(HK_CSV)
    a_res, a_per = acf_by_stock(A_CSV)
    hk, a = summarize(hk_res), summarize(a_res)
    d = (hk['lag1']['median'] - a['lag1']['median']
         if hk['lag1']['median'] is not None and a['lag1']['median'] is not None
         else None)
    if d is None:
        code, text = 'INVALID', '无法计算'
    elif abs(d) <= 0.10:
        code, text = 'NO_DIFF', f'|Δmedian lag1|={abs(d):.3f} <= 0.10 → 概率自稳定性无市场差异'
    else:
        direction = 'HK更黏' if d > 0 else 'A更黏'
        code, text = 'DIFF', f'Δmedian lag1={d:+.3f} (>0.10) → 有差异（{direction}），记为PBO成因描述性证据'
    out = {'experiment': 'T4_prob_autocorrelation',
           'preregistered_criteria': {'|Δlag1_median|<=0.10': 'NO_DIFF',
                                      '>0.10': 'DIFF (descriptive for PBO)'},
           'HK': hk, 'A': a, 'delta_lag1_median_HK_minus_A': d,
           'verdict_code': code, 'verdict_text': text,
           'per_stock_lag1_HK': hk_per, 'per_stock_lag1_A': a_per}
    payload = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True,
                         default=str)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(payload)
    print(json.dumps({k: out[k] for k in ['HK', 'A', 'delta_lag1_median_HK_minus_A',
                                          'verdict_code', 'verdict_text']},
                     ensure_ascii=False, indent=2)[:2000])
    print(f"MD5={hashlib.md5(payload.encode('utf-8')).hexdigest()}")


if __name__ == '__main__':
    main()
