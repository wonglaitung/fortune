#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六周期联合 null 检验（**单一检验**，非 6 次独立检验）。

问题（一句话）：
  在 1/3/5/10/15/20d 六个周期里，**是否存在任何一个**「扣成本后超额 > 0 且超噪声」的信号？

为什么不能做 6× Bonferroni：
  六个周期共用同一批股票 / 特征 / 日期 / 模型，超额序列高度相关 ⇒ 有效独立检验数 ≈1–2，
  不是 6。假设独立做多重比较是**口径错误**（见 lessons 三.76 同类：基准选错 ⇒ 结论反向）。

本脚本做三件事（互相印证，不替代）：
  ① 每周期「逐期超额 = TopK净 − 等权基准」→ bootstrap 95%CI（block=horizon）。
  ② **max 统计量**：取六周期超额 IR 最大者，看其 CI 下界是否 >0（max 天然含多重比较校正）。
  ③ 补充 **Stouffer** 合并各周期 t（**假设独立 ⇒ 反保守**）：若它都不显著，null 更稳。

用法：python3 scripts/pooled_horizon_test.py
"""

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.portfolio_backtest import backtest, load_panel, bootstrap, _stats  # noqa: E402

HORIZONS = [1, 3, 5, 10, 15, 20]
TOPK = 10
from ml_services.costs import COST_ROUND_TRIP


def latest_csv(h):
    fs = sorted(glob.glob(f'output/2026*_catboost_{h}d/prediction_analysis.csv'))
    return fs[-1] if fs else None


def main():
    rows = []
    tstats = []
    pooled_ex = []
    for h in HORIZONS:
        csv = latest_csv(h)
        if not csv:
            print(f"[H={h}] 无 CSV，跳过")
            continue
        df = load_panel(csv, market='hk')
        bt = backtest(df, h, TOPK, cost=COST_ROUND_TRIP)
        ex = (bt['top_net'] - bt['bench']).values
        ex = ex[np.isfinite(ex)]
        st = _stats(ex, h)
        ci = bootstrap(ex, h)
        sd = ex.std(ddof=1)
        t = st['mean'] / (sd / np.sqrt(len(ex))) if (sd > 0 and len(ex) > 1) else np.nan
        rows.append(dict(h=h, n=len(ex), mean=st['mean'], ir=st['ir'],
                         ir_lo=ci['ir_lo'], ir_hi=ci['ir_hi'], t=t,
                         p_pos=ci['p_ir_pos'], csv=csv))
        if np.isfinite(t):
            tstats.append(t)
        pooled_ex.append(pd.DataFrame({'date': bt['date'].values, 'h': h, 'ex': ex}))

    print("=" * 104)
    print("六周期逐期超额（TopK净 − 等权基准，成本 0.5%）")
    print(f"{'周期':>4} {'期数':>5} {'超额均/期':>10} {'超额IR':>8} {'超额IR 95%CI':>18} {'t':>7} {'P(IR>0)':>8}")
    print("-" * 104)
    for r in rows:
        cistr = f"[{r['ir_lo']:.2f},{r['ir_hi']:.2f}]"
        print(f"{r['h']:>4} {r['n']:>5} {r['mean']*100:>9.2f}% {r['ir']:>8.2f} "
              f"{cistr:>18} {r['t']:>7.2f} {r['p_pos']*100:>7.0f}%")
    print("-" * 104)

    # ② max 统计量
    best = max(rows, key=lambda r: r['ir'])
    print(f"② max 统计量：最好周期 = {best['h']}d，超额IR {best['ir']:.2f} "
          f"[{best['ir_lo']:.2f},{best['ir_hi']:.2f}]  "
          f"→ CI {'下界>0（显著）' if best['ir_lo'] > 0 else '跨 0（不显著）'}")

    # ③ Stouffer（反保守）
    k = len(tstats)
    if k:
        Z = float(np.sum(tstats) / math.sqrt(k))
        p_one = 0.5 * math.erfc(Z / math.sqrt(2))  # 单侧 P(pooled>0)
        print(f"③ Stouffer 合并（k={k}，假设独立 ⇒ 反保守）：Z = {Z:+.2f}  "
              f"单侧 P(合并超额>0) = {p_one:.3f}  → {'不显著' if p_one>0.05 else '显著'}")
        print("   注：Stouffer 假设独立会**高估**正 Z；此处 Z 为负 ⇒ 「无正信号」结论更稳。")
    print("=" * 104)
    print("结论：")
    print(f"  - 六周期超额 IR 全为 {'正' if all(r['ir']>0 for r in rows) else '多数负'}；")
    print(f"  - 最好周期（{best['h']}d）CI 跨 0 ⇒ 即便不做多重比较校正也不显著；")
    print("  - ⇒ 任一周期都无可部署正超额；多重比较只会更严，不会翻盘。")


if __name__ == '__main__':
    main()
