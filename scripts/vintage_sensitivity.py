#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨数据快照稳健性区间（lessons 三.29）

背景：us_market_data 缓存「仅当天有效」，宏观特征输入逐日变化 → 同代码跨日重跑
必换快照（实测 abs20d 净IR 0.34/0.42、PBO 0.41/0.64）。本脚本对**多个不同数据快照
产出的 walk-forward CSV** 逐个跑月度护栏与组合回测，输出 [min, max] 区间而非单点，
用以回答两个问题：
  1) D2「20d 维持低配」的结论在输入漂移下是否稳健？
  2) 数值波动幅度有多大（是否大到可以随便挑一轮当结论）？

用法：
    python3 scripts/vintage_sensitivity.py --horizon 20 --topk 10
    python3 scripts/vintage_sensitivity.py --horizon 20 --csv output/A/prediction_analysis.csv \
                                            --csv output/B/prediction_analysis.csv
"""
import argparse
import glob
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_RE = {
    'net_ir': re.compile(r'净IR \| \*\*([-0-9.]+)\*\*'),
    'pbo': re.compile(r'\*\*PBO\*\* \| \*\*([0-9.]+)\*\*'),
    'dsr': re.compile(r'\*\*DSR\*\*[^|]*\| \*\*([0-9.]+)\*\*'),
    'lift': re.compile(r'超额 lift\*\* \| \*\*([+-][0-9.]+)pp\*\*'),
}


def run_guardrail(csv_path: str, horizon: int, topk: int):
    """跑 monthly_guardrail 并解析关键指标"""
    out = os.path.join('/tmp', f'vs_{os.path.basename(os.path.dirname(csv_path))}_{horizon}d.md')
    # A股 CSV（目录含 _a_stock_）必须传 --market a，否则走港股口径得出错误指标
    market = 'a' if '_a_stock_' in csv_path else 'hk'
    cmd = [sys.executable, os.path.join(BASE, 'ml_services', 'monthly_guardrail.py'),
           '--horizon', str(horizon), '--topk', str(topk), '--market', market,
           '--pred', os.path.relpath(csv_path, BASE), '--output', out]
    subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    if not os.path.exists(out):
        return None
    txt = open(out, encoding='utf-8').read()
    res = {}
    for k, rx in METRIC_RE.items():
        m = rx.search(txt)
        res[k] = float(m.group(1)) if m else float('nan')
    res['judge'] = '🟢' if '🟢 通过' in txt else ('🟡' if '🟡' in txt else '🔴')
    os.remove(out)
    return res


def main():
    ap = argparse.ArgumentParser(description='跨数据快照稳健性区间')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--csv', action='append', default=[],
                    help='指定 CSV（可多次）；不给则自动扫描全部港股同周期 CSV')
    ap.add_argument('--min-dates', type=int, default=200, help='最少交易日，剔除过短样本')
    args = ap.parse_args()

    if args.csv:
        csvs = args.csv
    else:
        pat = os.path.join(BASE, 'output', f'*_catboost_{args.horizon}d', 'prediction_analysis.csv')
        # 排除 A 股目录（命名含 _a_stock_），否则混入另一市场的快照会污染区间
        csvs = [c for c in sorted(glob.glob(pat)) if '_a_stock_' not in c]

    rows = []
    for c in csvs:
        try:
            import pandas as pd
            df = pd.read_csv(c, usecols=['Date'])
            if df['Date'].nunique() < args.min_dates:
                continue
        except Exception:
            continue
        r = run_guardrail(c, args.horizon, args.topk)
        if r:
            rows.append((os.path.basename(os.path.dirname(c)), r))

    if not rows:
        print('无可用 CSV（可用 --csv 显式指定）')
        return 1

    print('=' * 78)
    print(f'跨数据快照稳健性区间  horizon={args.horizon}d  TopK={args.topk}  快照数={len(rows)}')
    print('=' * 78)
    print(f"{'快照(输出目录)':<32}{'净IR':>9}{'PBO':>8}{'DSR':>8}{'lift(pp)':>10}{'判定':>6}")
    print('-' * 78)
    for name, r in rows:
        print(f"{name:<32}{r['net_ir']:>9.2f}{r['pbo']:>8.2f}{r['dsr']:>8.3f}"
              f"{r['lift']:>10.1f}{r['judge']:>6}")

    def spread(key):
        vals = [r[key] for _, r in rows if r[key] == r[key]]
        return (min(vals), max(vals), max(vals) - min(vals)) if vals else (float('nan'),) * 3

    print('-' * 78)
    print('区间（min ~ max, 极差）：')
    for k, label in [('net_ir', '净IR'), ('pbo', 'PBO'), ('dsr', 'DSR'), ('lift', 'lift(pp)')]:
        lo, hi, sp = spread(k)
        print(f'  {label:<10} {lo:>8.3f} ~ {hi:>8.3f}   极差 {sp:.3f}')

    verdicts = {r['judge'] for _, r in rows}
    print('-' * 78)
    if verdicts == {'🟡'}:
        print('结论：所有快照判定一致 = 🟡 保留低配 → D2 在输入漂移下稳健 ✅')
    elif len(verdicts) == 1:
        print(f'结论：所有快照判定一致 = {verdicts.pop()} → 结论稳健 ✅')
    else:
        print(f'⚠️ 结论：快照间判定不一致 {sorted(verdicts)}')
        print('   → 单轮判定不可作结论，必须多轮取区间；且禁止挑选最好的一轮')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())