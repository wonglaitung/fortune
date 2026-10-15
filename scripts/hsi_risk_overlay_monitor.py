#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSI 风险覆盖层 —— 滚动（样本外）验证 / 监控（D35 §十一）。

目的：**在样本外持续追踪"刹车"信号是否仍有效**（D35 只有单 vintage、35 期、n=1 危机）。
- `--update`：读 `data/hsi_risk_overlay_latest.json`（由 hsi_risk_overlay.py / 邮件流程刷新）
  取最新立场 → 追加到 `data/hsi_risk_overlay_ledger.csv`；对已到期记录（signal_date + horizon 交易日）
  用**真实 HSI 行情**回填实现收益。
- `--report`：读 ledger，报 信号 vs 买入持有 vs 静态等仓位 的**实现**表现 + 是否仍灵。

⚠️ 这不是 alpha：目标=降回撤。

用法：
  python3 scripts/hsi_risk_overlay_monitor.py --update
  python3 scripts/hsi_risk_overlay_monitor.py --report
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEDGER = os.path.join(ROOT, 'data', 'hsi_risk_overlay_ledger.csv')
JSON_PATH = os.path.join(ROOT, 'data', 'hsi_risk_overlay_latest.json')
COST = 0.005
COLS = ['signal_date', 'stance', 'prob', 'exposure', 'horizon',
        'fill_date', 'hsi_ret', 'signal_ret', 'hold_ret', 'filled']


def get_hsi_close():
    """HSI 收盘序列（yf，失败回退缓存 HSI.pkl）。"""
    try:
        import yfinance as yf
        h = yf.Ticker('^HSI').history(period='5y', interval='1d')
        if not h.empty:
            s = h['Close'].copy()
            s.index = pd.to_datetime(s.index)
            if getattr(s.index, 'tz', None) is not None:
                s.index = s.index.tz_convert(None)
            return s[~s.index.duplicated(keep='last')].sort_index()
    except Exception as e:
        print(f"  ⚠️ yf ^HSI 获取失败: {e}")
    try:
        import pickle
        p = os.path.join(ROOT, 'data', 'hk_universe_cache', 'HSI.pkl')
        s = pickle.load(open(p, 'rb'))
        col = 'Close' if 'Close' in s.columns else 'close'
        s = s[col].copy()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, 'tz', None) is not None:
            s.index = s.index.tz_convert(None)
        return s[~s.index.duplicated(keep='last')].sort_index()
    except Exception:
        return None


def _load_ledger():
    if os.path.exists(LEDGER):
        return pd.read_csv(LEDGER, dtype={'signal_date': str, 'fill_date': str})
    return pd.DataFrame(columns=COLS)


def _first_on_or_after(idx, d):
    pos = idx.searchsorted(pd.Timestamp(d), side='left')
    return pos if pos < len(idx) else None


def update():
    if not os.path.exists(JSON_PATH):
        print(f"⚠️ 未找到 {JSON_PATH}（先运行 hsi_risk_overlay.py 或发一次 HSI 邮件）")
        return
    d = json.load(open(JSON_PATH, encoding='utf-8'))
    sig_date = d['as_of']
    exposure = 1.0 if d['stance'] == 'HOLD' else float(d.get('exposure_flat', 0.0))
    led = _load_ledger()
    if sig_date not in set(led['signal_date']):
        import warnings
        new = pd.DataFrame([{
            'signal_date': sig_date, 'stance': d['stance'], 'prob': d['prob'],
            'exposure': exposure, 'horizon': d['horizon'], 'fill_date': '',
            'hsi_ret': np.nan, 'signal_ret': np.nan, 'hold_ret': np.nan, 'filled': False,
        }], columns=COLS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            led = pd.concat([led, new], ignore_index=True)
        print(f"➕ 追加立场记录: {sig_date} {d['stance']} prob={d['prob']:.3f}")

    # 回填到期记录（真实 HSI 行情）
    hsi = get_hsi_close()
    if hsi is None:
        print("⚠️ 无 HSI 行情，暂不回填")
    else:
        idx = hsi.index
        for i, r in led.iterrows():
            if r['filled']:
                continue
            h = int(r['horizon'])
            p0 = _first_on_or_after(idx, r['signal_date'])
            if p0 is None or p0 + h >= len(idx):
                continue  # 未到期
            ret = float(hsi.iloc[p0 + h] / hsi.iloc[p0] - 1)
            led.at[i, 'fill_date'] = str(idx[p0 + h].date())
            led.at[i, 'hsi_ret'] = ret
            led.at[i, 'hold_ret'] = ret
            led.at[i, 'signal_ret'] = float(r['exposure']) * ret - COST * (1.0 if r['exposure'] != 1.0 else 0.0)
            led.at[i, 'filled'] = True
            print(f"✅ 回填 {r['signal_date']} → {idx[p0+h].date()}  HSI {ret*100:+.1f}%")

    led.to_csv(LEDGER, index=False)
    print(f"💾 ledger: {LEDGER}（{len(led)} 行，已回填 {int(led['filled'].sum())}）")


def _maxdd(r):
    c = (1 + np.asarray(r, float)).cumprod()
    return float((c / np.maximum.accumulate(c) - 1).min())


def _ir(r, h):
    r = np.asarray(r, float)
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252.0 / h)) if s > 0 else float('nan')


def report():
    led = _load_ledger()
    done = led[led['filled'] == True]  # noqa: E712
    print("=" * 88)
    print(f"HSI 风险覆盖层 —— 滚动（样本外）验证  ledger={LEDGER}")
    print(f"记录 {len(led)} 行 | 已到期回填 {len(done)} | 待回填 {len(led)-len(done)}")
    print("=" * 88)
    if len(done) < 3:
        print("⚠️ 已到期样本不足（<3）⇒ 尚不可判；继续按月 --update 累积。")
        return
    h = int(done['horizon'].median())
    sig, hold = done['signal_ret'].values, done['hold_ret'].values
    wbar = float(done['exposure'].mean())
    static = wbar * hold
    print(f"{'口径':<20}{'均/期':>9}{'IR':>8}{'maxDD':>9}")
    for name, r in [('信号(刹车)', sig), (f'静态等仓位({wbar:.2f})', static), ('买入持有', hold)]:
        print(f"{name:<20}{r.mean()*100:>8.2f}%{_ir(r, h):>8.2f}{_maxdd(r)*100:>8.1f}%")
    dd_sig, dd_static = _maxdd(sig), _maxdd(static)
    adds = dd_sig > dd_static + 0.005
    print("-" * 88)
    print(f"滚动结论：择时增量 {'✅ 仍灵' if adds else '❌ 退化/仅少投'}  "
          f"(信号 maxDD {dd_sig*100:.1f}% vs 静态 {dd_static*100:.1f}%)")
    if not adds:
        print("⚠️ 样本外未复现择时增量 ⇒ 应停用该刹车，或仅作『少投』。")


def main():
    ap = argparse.ArgumentParser(description='HSI 风险覆盖层滚动验证/监控')
    ap.add_argument('--update', action='store_true', help='追加最新立场 + 回填到期实现收益')
    ap.add_argument('--report', action='store_true', help='报滚动（样本外）表现')
    a = ap.parse_args()
    if not a.update and not a.report:
        a.update = a.report = True
    if a.update:
        update()
    if a.report:
        report()


if __name__ == '__main__':
    main()
