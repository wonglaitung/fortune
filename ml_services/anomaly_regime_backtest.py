#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常大跌抄底 + 行情感知增强版（DECISIONS 后第 2 步）

背景：基础"异常大跌抄底"（z≤−3、5d）净IR 1.55 显著，但强依赖行情
（2021/2025 正、2023-24/26 负）。本脚本加**恒指行情感知过滤**，看能否提升稳健性。

过滤规则（信号日恒指状态）：
  R1: HSI 20日收益 > 0（短期上行）
  R2: HSI 收盘 > MA200（长期上行）
  R3: HSI 5日收益 > 0（反弹中）
输出：基础 vs 各过滤后的 n/胜率/净IR/CI/逐年。

用法：python3 ml_services/anomaly_regime_backtest.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.anomaly_dip_backtest import load as load_dip_panel
from ml_services.anomaly_dip_backtest import eval_strategy, boot_mean

CACHE_DIR = 'data/feature_cache'
HSI_PKL = 'data/hk_universe_cache/HSI.pkl'
COST = 0.005
ZTHR = -3.0
HORIZON = 5


def load_hsi():
    h = pd.read_pickle(HSI_PKL)
    h['ret5'] = h['close'].pct_change(5)
    h['ret20'] = h['close'].pct_change(20)
    h['ma200'] = h['close'].rolling(200, min_periods=200).mean()
    return h


def main(out_md=None):
    panel = load_dip_panel()
    hsi = load_hsi()
    hsi = hsi[['ret5', 'ret20', 'ma200', 'close']]
    panel = panel.join(hsi, on='date')
    sig = panel[panel['z'] <= ZTHR].copy()
    base = panel

    rules = {
        '基础（无过滤）': None,
        'R1 恒指20d收益>0': panel['ret20'] > 0,
        'R2 恒指>MA200': panel['close'] > panel['ma200'],
        'R3 恒指5d收益>0（反弹中）': panel['ret5'] > 0,
        'R1+R2（短长均好）': (panel['ret20'] > 0) & (panel['close'] > panel['ma200']),
    }

    L = []
    L.append(f"# 异常大跌抄底 + 行情感知（z≤{ZTHR}，{HORIZON}d）\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 基准: 无条件买入（全样本 5d 胜率/收益）\n")
    L.append("| 过滤 | 信号数 | 胜率 | 基准胜率 | lift | 净均/期 | 净IR | 净IR 95%CI |")
    L.append("|------|--------|------|---------|------|---------|------|-----------|")
    results = {}
    for name, mask in rules.items():
        s = sig if mask is None else sig[mask.loc[sig.index]]
        r = eval_strategy(s, base)
        results[name] = r
        L.append(f"| {name} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | "
                 f"{r['lift']*100:+.1f}pp | {r['mean']*100:+.2f}% | {r['ir']:.2f} | "
                 f"[{r['lo']*100:+.1f}%,{r['hi']*100:+.1f}%] |")
    L.append("")
    L.append("## 逐年（R1+R2）\n")
    s12 = sig[(panel['ret20'] > 0) & (panel['close'] > panel['ma200'])].copy()
    L.append("| 年份 | 信号数 | 胜率 | 基准胜率 | 净均/期 | 净IR |")
    L.append("|------|--------|------|---------|---------|------|")
    for y, g in s12.groupby('year'):
        b = base[base['year'] == y]
        r = eval_strategy(g, b)
        L.append(f"| {int(y)} | {r['n']} | {r['win']*100:.1f}% | {r['base_win']*100:.1f}% | "
                 f"{r['mean']*100:+.2f}% | {r['ir']:.2f} |")
    L.append("")
    L.append("## 结论\n")
    base_r = results['基础（无过滤）']
    best = max([k for k in results if k != '基础（无过滤）'], key=lambda k: results[k]['ir'])
    r_best = results[best]
    L.append(f"- 基础：净IR {base_r['ir']:.2f}，CI [{base_r['lo']*100:+.1f}%,{base_r['hi']*100:+.1f}%]")
    L.append(f"- 最优过滤（{best}）：净IR {r_best['ir']:.2f}，CI [{r_best['lo']*100:+.1f}%,{r_best['hi']*100:+.1f}%]，"
             f"信号 {r_best['n']} 个（保留 {r_best['n']/base_r['n']*100:.0f}%）")
    L.append(f"- 判定: {'✅ 行情感知提升稳健性' if r_best['ir'] > base_r['ir'] and r_best['lo'] > 0 else '⚠️ 提升有限/样本过少'}（需注意样本变少）")
    L.append("")
    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")


if __name__ == '__main__':
    main('output/anomaly_regime_backtest.md')