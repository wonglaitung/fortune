#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSI 方向风险覆盖层（"刹车"原型，D35 §十）。

⚠️ **不是 alpha**：Long 持恒指 / Flat 转现金，目标 = **降回撤**（期望收益 ≈ 买入持有）。
经 D35 §十 验证：**仅 20d + southbound / nasdaq 增强**有真"择时降回撤"增量；其余"降回撤"只是"少投"。

本原型跑一遍 Walk-forward（HSI 方向模型）→ 取逐日 Long/Flat → 覆盖层回测：
  信号(Long/Flat) vs 静态等仓位(w̄) vs 买入持有，报 均收益 / IR / maxDD / Calmar。
并报**当前立场**（今天该持有 or 空仓）。

用法：
  python3 ml_services/hsi_risk_overlay.py --enh nasdaq --horizon 20            # 回测 + 当前立场
  python3 ml_services/hsi_risk_overlay.py --enh southbound --horizon 20 --asof # 只看当前该不该持有
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST = 0.005  # 单边换手成本（进出各计一次）
CSV_RE = re.compile(r'hsi_prediction_analysis_[0-9_]+\.csv')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_wf(horizon, enh, start, end):
    """跑 Walk-forward，返回预测 CSV 路径。"""
    env = dict(os.environ, HSI_ENH=enh)
    p = subprocess.run(
        [sys.executable, 'ml_services/hsi_walk_forward.py', '--horizon', str(horizon),
         '--train-window', '12', '--start-date', start, '--end-date', end],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    m = CSV_RE.findall(p.stdout)
    if not m:
        raise RuntimeError("WF 未产出预测 CSV（见 stderr）:\n" + p.stderr[-800:])
    return os.path.join(ROOT, 'data', 'hsi_walk_forward', m[-1])


def _ir(r, h):
    r = np.asarray(r, float)
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252.0 / h)) if s > 0 else float('nan')


def _maxdd(r):
    c = (1 + np.asarray(r, float)).cumprod()
    return float((c / np.maximum.accumulate(c) - 1).min())


def _calmar(r, h):
    dd = _maxdd(r)
    if dd == 0:
        return float('nan')
    cagr = (1 + np.asarray(r, float)).prod() ** (252.0 / h / len(r)) - 1
    return float(cagr / abs(dd))


def save_latest(path, enh, horizon, last, wbar, rows, exposure_flat):
    """把当前立场 + 回测统计落 JSON（供 HSI 预测邮件读取）。"""
    sig, hold = rows['信号(Long/Flat)'], rows['买入持有']
    dd_sig = _maxdd(sig)
    dd_static = _maxdd(rows[f'静态等仓位({wbar:.2f})'])
    obj = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'as_of': str(last['date'])[:10], 'enh': enh or 'none', 'horizon': horizon,
        'prob': float(last['prob']), 'stance': 'HOLD' if int(last['pred']) == 1 else 'CASH',
        'exposure_flat': exposure_flat, 'avg_exposure': wbar,
        'ir_sig': _ir(sig, horizon), 'ir_hold': _ir(hold, horizon),
        'maxdd_sig': dd_sig, 'maxdd_static': dd_static, 'maxdd_hold': _maxdd(hold),
        'calmar_sig': _calmar(sig, horizon), 'calmar_hold': _calmar(hold, horizon),
        'timing_increment': bool(dd_sig > dd_static + 0.005),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return obj


def overlay(csv, horizon, exposure_flat=0.0):
    """覆盖层回测。exposure_flat: Flat 时敞口（0=全现金，0.5=半仓）。"""
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    last = df.iloc[-1]
    rb = df.iloc[::horizon].reset_index(drop=True)
    pos = np.where(rb['pred'].astype(int).values == 1, 1.0, exposure_flat)
    entry = np.abs(np.diff(np.r_[pos[0], pos]))
    ret = rb['actual_return'].values
    sig = pos * ret - entry * COST
    wbar = float(pos.mean())
    static = wbar * ret
    rows = {
        '信号(Long/Flat)': sig,
        f'静态等仓位({wbar:.2f})': static,
        '买入持有': ret,
    }
    return last, wbar, rows


def main():
    ap = argparse.ArgumentParser(description='HSI 方向风险覆盖层（刹车原型）')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--enh', type=str, default='nasdaq',
                    choices=['none', 'southbound', 'ahpremium', 'nasdaq', 'usdcnh'],
                    help='D35 增强项（仅 20d 下 southbound/nasdaq 有择时增量）')
    ap.add_argument('--start-date', type=str, default='2020-01-01')
    ap.add_argument('--end-date', type=str, default=date.today().isoformat())
    ap.add_argument('--exposure-flat', type=float, default=0.0, help='Flat 时敞口（0=现金）')
    ap.add_argument('--asof', action='store_true', help='只报当前立场')
    ap.add_argument('--save', type=str, default='data/hsi_risk_overlay_latest.json',
                    help='聚类立场+统计落 JSON（供 HSI 预测邮件读取）；相对项目根')
    ap.add_argument('--no-save', action='store_true', help='不落盘')
    a = ap.parse_args()

    enh = '' if a.enh == 'none' else a.enh
    csv = run_wf(a.horizon, enh, a.start_date, a.end_date)
    last, wbar, rows = overlay(csv, a.horizon, a.exposure_flat)

    if not a.no_save:
        p = a.save if os.path.isabs(a.save) else os.path.join(ROOT, a.save)
        j = save_latest(p, a.enh, a.horizon, last, wbar, rows, a.exposure_flat)
        print(f"💾 立场已落盘: {p}（stance={j['stance']} as_of={j['as_of']}）")

    print("=" * 84)
    print(f"HSI 方向风险覆盖层（刹车原型）  增强={a.enh}  周期={a.horizon}d  "
          f"Flat敞口={a.exposure_flat:.0%}")
    print("⚠️ 不是 alpha：目标=降回撤，期望收益 ≈ 买入持有（D35 §十）")
    print("=" * 84)

    if not a.asof:
        print(f"{'口径':<20}{'均/期':>9}{'年化IR':>9}{'maxDD':>9}{'Calmar':>9}")
        for name, r in rows.items():
            print(f"{name:<20}{r.mean()*100:>8.2f}%{_ir(r, a.horizon):>9.2f}"
                  f"{_maxdd(r)*100:>8.1f}%{_calmar(r, a.horizon):>9.2f}")
        dd_sig, dd_static = _maxdd(rows['信号(Long/Flat)']), _maxdd(rows[f'静态等仓位({wbar:.2f})'])
        adds = dd_sig > dd_static + 0.005
        print("-" * 84)
        print(f"择时增量(vs 静态 w̄)：{'✅ 有' if adds else '❌ 无（只是少投）'}  "
              f"(信号 maxDD {dd_sig*100:.1f}% vs 静态 {dd_static*100:.1f}%)")
        if not adds:
            print("⚠️ 该 (增强, 周期) 组合无真增量 → 直接用『少投』即可，别用信号。")
        print("-" * 84)

    stance = '持有恒指(EXPOSURE=100%)' if int(last['pred']) == 1 else f"空仓/现金(EXPOSURE={a.exposure_flat:.0%})"
    print(f"当前立场（{str(last['date'])[:10]}，概率 {last['prob']:.3f}）：{stance}")


if __name__ == '__main__':
    main()
