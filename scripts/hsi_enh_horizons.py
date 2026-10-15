#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D35 HSI 方向增强 —— **六周期** A/B（1/3/5/10/15/20d × 5 变体）。

对每个 (变体, 周期)：跑 hsi_walk_forward.py（HSI_ENH 控制单变量）→ 预测 CSV →
用 hsi_enh_eval.evaluate 算 净IR/超额CI/折聚类t/maxDD → 输出网格 + 判定矩阵。

用法：python3 scripts/hsi_enh_horizons.py
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsi_enh_eval import evaluate  # noqa: E402
from hsi_signal_static_check import static_check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = {'baseline': '', 'southbound': 'southbound', 'ahpremium': 'ahpremium',
            'nasdaq': 'nasdaq', 'usdcnh': 'usdcnh'}
HORIZONS = [1, 3, 5, 10, 15, 20]
CSV_RE = re.compile(r'hsi_prediction_analysis_[0-9_]+\.csv')


def run_one(horizon, enh):
    env = dict(os.environ, HSI_ENH=enh)
    p = subprocess.run(
        [sys.executable, 'ml_services/hsi_walk_forward.py', '--horizon', str(horizon),
         '--train-window', '12', '--start-date', '2020-01-01', '--end-date', '2025-12-31'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    m = CSV_RE.findall(p.stdout)
    if not m:
        return None
    return os.path.join(ROOT, 'data', 'hsi_walk_forward', m[-1])


def verdict(r):
    main_ok = (r['ir_sig'] >= r['ir_base']) and (r['ex_lo'] > 0)
    skill_ok = (r['fold_t'] == r['fold_t']) and (r['fold_t'] > 1.96)
    dd_ok = r['maxdd_sig'] >= r['maxdd_base']
    return 'PASS' if (main_ok and skill_ok and dd_ok) else 'FAIL'


def main():
    res = {}
    for h in HORIZONS:
        for v, enh in VARIANTS.items():
            csv = run_one(h, enh)
            if csv is None or not os.path.exists(csv):
                print(f"⚠️ H={h} {v}: 无预测 CSV（跳过）", flush=True)
                continue
            r = evaluate(csv, h)
            sc = static_check(csv, h)
            r['adds'] = sc['adds']
            r['wbar'] = sc['wbar']
            res[(h, v)] = r
            print(f"  H={h:>2} {v:<12} 期数{r['n']:>3} 净IR {r['ir_sig']:>5.2f} "
                  f"超额[{r['ex_lo']*100:>+6.2f}%,{r['ex_hi']*100:>+6.2f}%] "
                  f"t {r['fold_t']:>5.2f} maxDD {r['maxdd_sig']*100:>5.1f}% "
                  f"w̄{sc['wbar']:.2f} 择时{'有' if sc['adds'] else '无'} -> {verdict(r)}",
                  flush=True)

    print("\n" + "=" * 100)
    print("净IR（信号 / 持有）  与  判定矩阵")
    print("=" * 100)
    hdr = f"{'变体':<12}" + "".join(f"{str(h)+'d':>14}" for h in HORIZONS)
    print(hdr)
    for v in VARIANTS:
        line = f"{v:<12}"
        for h in HORIZONS:
            r = res.get((h, v))
            line += (f"{r['ir_sig']:>7.2f}/{r['ir_base']:<5.2f}" if r else f"{'—':>14}")
        print(line)
    print("-" * 100)
    print(f"{'判定':<12}" + "".join(f"{'':>14}" for _ in HORIZONS))
    for v in VARIANTS:
        line = f"{v:<12}"
        for h in HORIZONS:
            r = res.get((h, v))
            line += f"{verdict(r) if r else '—':>14}"
        print(line)
    print("-" * 100)
    # 空仓率（Long/Flat 价值=降回撤）矩阵
    print(f"{'maxDD 信号(持有)':<12}" + "".join(f"{str(h)+'d':>14}" for h in HORIZONS))
    for v in VARIANTS:
        line = f"{v:<12}"
        for h in HORIZONS:
            r = res.get((h, v))
            line += (f"{r['maxdd_sig']*100:>6.1f}%/{r['maxdd_base']*100:<6.1f}%" if r else f"{'—':>14}")
        print(line)
    print("-" * 100)
    print(f"{'择时有增量(vs静态w̄)':<12}" + "".join(f"{str(h)+'d':>14}" for h in HORIZONS))
    for v in VARIANTS:
        line = f"{v:<12}"
        for h in HORIZONS:
            r = res.get((h, v))
            line += f"{('✅有' if r['adds'] else '❌无') if r else '—':>14}"
        print(line)


if __name__ == '__main__':
    main()
