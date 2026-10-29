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

from ml_services.strategy_eval import (  # noqa: E402
    daily_mark, static_matched, metrics_daily, has_timing_increment)

from ml_services.costs import COST_ROUND_TRIP
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


# 评估口径统一委托 `ml_services.strategy_eval`（唯一实现，D45）
def _ir(r, h):
    return metrics_daily(pd.Series(np.asarray(r, float)), horizon=h)['sharpe']


def _maxdd(r):
    return metrics_daily(pd.Series(np.asarray(r, float)), horizon=1)['maxdd']


def _calmar(r, h):
    return metrics_daily(pd.Series(np.asarray(r, float)), horizon=h)['calmar']


def save_latest(path, enh, horizon, last, wbar, daily_rows, exposure_flat):
    """把当前立场 + **逐日估值**统计落 JSON（供 HSI 预测邮件读取）。

    ⚠️ 用**逐日 mark-to-market**（非 20d 稀疏采样）：20d 非重叠只每 20 天记一次净值，
    会**低估**中间的日内回撤，从而**虚报**"择时增量"（2026-10-15 实测）。
    """
    sig = daily_rows['信号(20d仓·逐日估值)']
    hold = daily_rows['买入持有']
    static = daily_rows[f'静态等仓位({wbar:.2f})']
    dd_sig, dd_static = _maxdd(sig), _maxdd(static)
    obj = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'as_of': str(last['date'])[:10], 'enh': enh or 'none', 'horizon': horizon,
        'prob': float(last['prob']), 'stance': 'HOLD' if int(last['pred']) == 1 else 'CASH',
        'exposure_flat': exposure_flat, 'avg_exposure': wbar, 'basis': 'daily_mark',
        'ir_sig': _ir(sig, 1), 'ir_hold': _ir(hold, 1),
        'maxdd_sig': dd_sig, 'maxdd_static': dd_static, 'maxdd_hold': _maxdd(hold),
        'calmar_sig': _calmar(sig, 1), 'calmar_hold': _calmar(hold, 1),
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
    sig = pos * ret - 0.5 * entry * COST_ROUND_TRIP
    wbar = float(pos.mean())
    static = wbar * ret
    rows = {
        '信号(Long/Flat)': sig,
        f'静态等仓位({wbar:.2f})': static,
        '买入持有': ret,
    }
    return last, wbar, rows


def _hsi_daily_close():
    import yfinance as yf
    h = yf.Ticker('^HSI').history(period='5y', interval='1d')
    if h.empty:
        return None
    s = h['Close'].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, 'tz', None) is not None:
        s.index = s.index.tz_convert(None)
    return s[~s.index.duplicated(keep='last')].sort_index()


def daily_overlay(csv, horizon, exposure_flat=0.0):
    """**同一刹车、逐日估值**：按 20d 节奏调仓（用 `iloc[::horizon]` 的立场并 ffill 到每日），
    **逐日 mark-to-market** 算净值/回撤（只在调仓日计换手成本）。

    与 `overlay`（20d 非重叠）的差别：**仅估值更细**——20d 非重叠只每 20 天记一次净值，
    会**低估夹在中间的日内回撤**。收益序列自相关 ⇒ 显著性须 HAC。
    """
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    h = _hsi_daily_close()
    if h is None:
        raise RuntimeError('无 HSI 日线')
    dr = h.pct_change()
    rb = df.iloc[::horizon]                       # 同 overlay：20d 调仓日
    start, end = rb['date'].min(), rb['date'].max()
    hi = h.index[(h.index >= start) & (h.index <= end)]
    stance = rb.set_index('date')['pred'].reindex(hi, method='ffill').fillna(0)
    pos = pd.Series(np.where(stance.values == 1, 1.0, exposure_flat), index=hi)
    d = dr.reindex(hi)
    sig = daily_mark(d, pos, round_trip_cost=COST_ROUND_TRIP, lag=1).dropna()   # 唯一实现
    wbar = float(pos.mean())
    rows = {'信号(20d仓·逐日估值)': sig,
            f'静态等仓位({wbar:.2f})': static_matched(d, wbar).reindex(sig.index),
            '买入持有': d.reindex(sig.index)}
    return wbar, rows


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
    ap.add_argument('--daily', action='store_true',
                    help='逐日 P&L 视图（更细的净值/回撤；IR 未做 HAC，仅作细粒度观察）')
    ap.add_argument('--save', type=str, default=None,
                    help='立场+统计落 JSON（默认 data/hsi_risk_overlay_latest_{周期}d.json）；相对项目根')
    ap.add_argument('--no-save', action='store_true', help='不落盘')
    a = ap.parse_args()

    enh = '' if a.enh == 'none' else a.enh
    csv = run_wf(a.horizon, enh, a.start_date, a.end_date)
    last, wbar_s, rows_s = overlay(csv, a.horizon, a.exposure_flat)     # 20d 非重叠（稀疏）
    wbar_d, rows_d = daily_overlay(csv, a.horizon, a.exposure_flat)     # 20d 仓·逐日估值（诚实）

    if not a.no_save:
        save = a.save or f'data/hsi_risk_overlay_latest_{a.horizon}d.json'
        p = save if os.path.isabs(save) else os.path.join(ROOT, save)
        j = save_latest(p, a.enh, a.horizon, last, wbar_d, rows_d, a.exposure_flat)
        print(f"💾 立场已落盘: {p}（stance={j['stance']} as_of={j['as_of']} basis=daily_mark）")

    wbar, rows, h = (wbar_d, rows_d, 1) if a.daily else (wbar_s, rows_s, a.horizon)
    sig_key = '信号(20d仓·逐日估值)' if a.daily else '信号(Long/Flat)'
    static_key = f'静态等仓位({wbar:.2f})'

    print("=" * 84)
    print(f"HSI 大跌保护（刹车）  增强={a.enh}  {a.horizon}d  Flat敞口={a.exposure_flat:.0%}")
    print("⚠️ 不是 alpha：目标=降回撤，期望收益 ≈ 买入持有（D35 §十）")
    print("=" * 84)

    if not a.asof:
        for label, wb_, rws_, hh in [
                ('20d 非重叠（稀疏 · 会低估日内回撤）', wbar_s, rows_s, a.horizon),
                ('20d 仓 · 逐日估值（诚实）', wbar_d, rows_d, 1)]:
            sk = '信号(Long/Flat)' if hh != 1 else '信号(20d仓·逐日估值)'
            stk = f'静态等仓位({wb_:.2f})'
            print(f"-- {label} --")
            print(f"{'口径':<22}{'均/期':>10}{'年化IR':>9}{'maxDD':>9}{'Calmar':>9}")
            for name, r in rws_.items():
                print(f"{name:<22}{r.mean()*100:>9.3f}%{_ir(r, hh):>9.2f}"
                      f"{_maxdd(r)*100:>8.1f}%{_calmar(r, hh):>9.2f}")
            dd_s, dd_st = _maxdd(rws_[sk]), _maxdd(rws_[stk])
            inc = has_timing_increment(pd.Series(rws_[sk]), pd.Series(rws_[stk]))
            print(f"  择时增量: {'✅ 有（回撤更浅）' if inc else '❌ 无（只是少投）'}  "
                  f"(信号 {dd_s*100:.1f}% vs 静态 {dd_st*100:.1f}%)")
        print("-" * 84)
        print("⭐ 以『逐日估值』为准：20d 非重叠只每 20 天记一次净值，会低估日内回撤、虚报增量。")

    stance = '持有恒指(EXPOSURE=100%)' if int(last['pred']) == 1 else f"空仓/现金(EXPOSURE={a.exposure_flat:.0%})"
    print(f"当前立场（{str(last['date'])[:10]}，概率 {last['prob']:.3f}）：{stance}")


if __name__ == '__main__':
    main()
