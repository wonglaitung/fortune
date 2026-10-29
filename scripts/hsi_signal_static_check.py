#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSI Long/Flat 信号的"是否只是少投"检查（D32 教训，D35 用）。

对给定预测 CSV：
  w̄ = 平均仓位（= Long 占比）
  信号   = pred*ret − 换手成本（择时：该空仓时空仓）
  静态   = w̄*ret（**恒定** w̄，无择时，纯减仓）
  买入持有 = ret
输出三者 maxDD / IR / 均收益。
判定：若"信号 maxDD 未明显低于 静态 w̄"，则"降回撤"**只是少投**，不是择时能力。

用法：python3 scripts/hsi_signal_static_check.py --horizon 20 --csv <pred.csv>
"""

import argparse

import numpy as np
import pandas as pd

from ml_services.costs import COST_ROUND_TRIP


def _maxdd(r):
    c = (1 + np.asarray(r, float)).cumprod()
    return float((c / np.maximum.accumulate(c) - 1).min())


def _ir(r, h):
    r = np.asarray(r, float)
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252.0 / h)) if s > 0 else float('nan')


def static_check(csv, horizon):
    """返回 dict：wbar / 信号·静态·持有 的 均收益·IR·maxDD / adds(择时有增量?)。"""
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    rb = df.iloc[::horizon].reset_index(drop=True)
    pos = rb['pred'].astype(int).values
    entry = np.abs(np.diff(np.r_[pos[0], pos])).clip(max=1)
    ret = rb['actual_return'].values
    sig = pos * ret - 0.5 * entry * COST_ROUND_TRIP
    wbar = float(pos.mean())
    static = wbar * ret
    dd_sig, dd_static = _maxdd(sig), _maxdd(static)
    return dict(n=len(rb), wbar=wbar,
                sig=(sig.mean(), _ir(sig, horizon), dd_sig),
                static=(static.mean(), _ir(static, horizon), dd_static),
                hold=(ret.mean(), _ir(ret, horizon), _maxdd(ret)),
                adds=(dd_sig > dd_static + 0.005))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--csv', required=True)
    a = ap.parse_args()
    r = static_check(a.csv, a.horizon)
    print(f"期数 {r['n']} | 平均仓位 w̄ = {r['wbar']:.2f}（Long 占比）")
    for name, k in [('信号 Long/Flat', 'sig'), (f"静态等仓位 {r['wbar']:.2f}", 'static'), ('买入持有', 'hold')]:
        m, ir, dd = r[k]
        print(f"  {name:<16} 均/期 {m*100:>+6.2f}%   IR {ir:>5.2f}   maxDD {dd*100:>6.1f}%")
    dd_sig, dd_static = r['sig'][2], r['static'][2]
    verdict = '✅ 择时有增量（信号回撤更浅）' if r['adds'] else '❌ 仅少投/无增量（静态 w̄ 回撤不差或更浅）'
    print(f"判定：maxDD 信号 {dd_sig*100:.1f}% vs 静态 w̄ {dd_static*100:.1f}%  ->  {verdict}")


if __name__ == '__main__':
    main()
