#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSI 风险覆盖层 —— 滚动（样本外）验证 / 监控（D35 §十一）。

目的：**在样本外持续追踪"刹车"信号是否仍有效**（D35 只有单 vintage、35 期、n=1 危机）。
- `--update`：读 `data/hsi_risk_overlay_latest.json`（由 hsi_risk_overlay.py / 邮件流程刷新）
  取最新立场 → 追加到 `data/hsi_risk_overlay_ledger.csv`；对已到期记录（signal_date + horizon 交易日）
  用**真实 HSI 行情**回填实现收益。
- `--report`：读 ledger，报 信号 vs 买入持有 vs 静态等仓位 的**实现**表现 + 是否仍灵。
  ⚠️ 结论以**逐日 mark-to-market**为准（稀疏按记录期会低估回撤、虚报风控，见 D35 §十二）。

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
JSON_PATH = os.path.join(ROOT, 'data', 'hsi_risk_overlay_latest_20d.json')
from ml_services.costs import COST_ROUND_TRIP
COLS = ['signal_date', 'stance', 'prob', 'exposure', 'horizon',
        'fill_date', 'hsi_ret', 'signal_ret', 'hold_ret', 'filled']


def get_hsi_close():
    """HSI 收盘序列（yf，失败回退缓存 HSI.pkl）。"""
    try:
        import yfinance as yf
        h = yf.Ticker('^HSI').history(period='5y', interval='1d', timeout=15)
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
            led.at[i, 'signal_ret'] = float(r['exposure']) * ret - 0.5 * abs(float(r['exposure']) - 1.0) * COST_ROUND_TRIP
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


def _daily_mark(led):
    """用 ledger 记录的立场 + 真实 HSI 日线做**逐日 mark-to-market**（诚实回撤，D35 §十二）。"""
    led = led.dropna(subset=['exposure']).copy()
    if len(led) < 2:
        return None
    h = get_hsi_close()
    if h is None:
        return None
    dr = h.pct_change()
    led['signal_date'] = pd.to_datetime(led['signal_date'])
    led = led.sort_values('signal_date')
    hi = h.index[(h.index >= led['signal_date'].min()) & (h.index <= led['signal_date'].max())]
    if len(hi) < 5:
        return None
    st = led.drop_duplicates('signal_date').set_index('signal_date')['exposure'].astype(float)
    pos = st.reindex(hi, method='ffill').fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    turn = (pos - pos_lag).abs()
    d = dr.reindex(hi)
    sig = (pos_lag * d - 0.5 * turn * COST_ROUND_TRIP).dropna()
    wbar = float(pos.mean())
    return wbar, {'信号(逐日刹车)': sig,
                  f'静态等仓位({wbar:.2f})': wbar * d.reindex(sig.index),
                  '买入持有': d.reindex(sig.index)}


def report():
    led = _load_ledger()
    done = led[led['filled'] == True]  # noqa: E712
    print("=" * 88)
    print(f"HSI 大跌保护（刹车）—— 滚动（样本外）验证  ledger={LEDGER}")
    print(f"记录 {len(led)} 行 | 已到期回填 {len(done)} | 待回填 {len(led)-len(done)}")
    print("=" * 88)
    if len(done) < 3:
        print(f"⚠️ 已到期样本不足（{len(done)}/3）⇒ 尚不可判；约每月 1 笔，"
              f"还要 ~{max(3 - len(done), 0)} 个月才出初步结论（约 1 年后才较可信）。")
        return
    h = int(done['horizon'].median())
    sig, hold = done['signal_ret'].values, done['hold_ret'].values
    wbar = float(done['exposure'].mean())
    print("-- 稀疏（按记录期 · 会低估日内回撤）--")
    print(f"{'口径':<20}{'均/期':>9}{'IR':>8}{'maxDD':>9}")
    for name, r in [('信号(刹车)', sig), (f'静态等仓位({wbar:.2f})', wbar * hold), ('买入持有', hold)]:
        print(f"{name:<20}{r.mean()*100:>8.2f}%{_ir(r, h):>8.2f}{_maxdd(r)*100:>8.1f}%")

    dm = _daily_mark(led)
    print("-" * 88)
    if dm is None:
        print("⚠️ 无法逐日估值（HSI 日线不足 / 记录 <2）")
        return
    dwbar, drows = dm
    print("-- 逐日 mark-to-market（诚实，以此为准）--")
    print(f"{'口径':<22}{'均/日':>9}{'IR':>8}{'maxDD':>9}")
    for name, r in drows.items():
        print(f"{name:<22}{r.mean()*100:>8.3f}%{_ir(r, 1):>8.2f}{_maxdd(r)*100:>8.1f}%")
    dsig, dstatic = drows['信号(逐日刹车)'], drows[f'静态等仓位({dwbar:.2f})']
    adds = _maxdd(dsig) > _maxdd(dstatic) + 0.005
    print("-" * 88)
    print(f"滚动结论（以逐日为准）：择时增量 {'✅ 仍灵' if adds else '❌ 退化 / 仅少投'}  "
          f"(信号 {_maxdd(dsig)*100:.1f}% vs 静态 {_maxdd(dstatic)*100:.1f}%)")
    if not adds:
        print("⚠️ 样本外未复现择时增量 ⇒ 应停用该刹车，或仅作『少投』。")


def summary():
    """紧凑对账摘要（供 HSI 邮件嵌入用）：返回 dict(text=白话一行, status=...)。"""
    led = _load_ledger()
    done = led[led['filled'] == True]  # noqa: E712
    n, m = len(led), len(done)
    if m < 3:
        return {'n': n, 'filled': m, 'status': 'insufficient',
                'text': (f'样本外对账：已记 {n} 笔、回填 {m} 笔 —— 样本还太少（不到 3 笔），'
                         f'暂时看不出灵不灵（约每月 1 笔，还要 ~{max(3 - m, 0)} 个月才出初步结论；'
                         f'约 1 年后才较可信）。')}
    dm = _daily_mark(led)
    if dm is None:
        return {'n': n, 'filled': m, 'status': 'nodata',
                'text': f'样本外对账：已记 {n} 笔、回填 {m} 笔；逐日估值暂不可用。'}
    dwbar, drows = dm
    dsig = drows['信号(逐日刹车)']
    dstatic = drows[f'静态等仓位({dwbar:.2f})']
    adds = _maxdd(dsig) > _maxdd(dstatic) + 0.005
    st = '仍灵（真会躲）' if adds else '退化（只是少投，不算会躲）'
    return {'n': n, 'filled': m, 'status': 'ok' if adds else 'degraded',
            'text': f'样本外对账（逐日口径）：已记 {n} 笔、回填 {m} 笔 —— {st}。'}


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
