#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""底仓（长期持有）vs 战术层（20d）——成本后对照实验（预注册版）。

预注册文档：docs/LONG_TERM_VS_TACTICAL_EXPERIMENT.md（判据写死在跑之前）。
本脚本只做**底仓臂**与**基准臂**；战术层数字取自 docs/BASELINES.md B2（同一 310 池 / 同窗口）。

口径要点
- PIT：财报可用日 = 财年结束日 + 披露滞后（默认 120 天）；调仓日只用已披露数据。
- 决策在调仓日收盘，执行在**下一交易日收盘**（与项目 T+1 成交口径一致）。
- 成本：0.5 × Σ|Δw| × 往返成本（ml_services/costs.py 口径）。
- 基准：① 310 池等权（每季调仓、无成本）② HSI ③ 静态等权买入持有（对照"只是少投"）。
- 回撤：逐日 mark-to-market；显著性：月度超额 + Newey-West；必报 MDE。

用法：
  python3 scripts/backtest_longterm_core.py
  python3 scripts/backtest_longterm_core.py --grid          # 相邻参数网格 + 安慰剂
  python3 scripts/backtest_longterm_core.py --out output/_logs/lt_experiment.json
"""
import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from screen_longterm_core import FX, ccy_of  # noqa: E402

FUND = os.path.join(ROOT, 'data', 'fund_cache')
SNAP = os.path.join(ROOT, 'data', 'hk_broad', 'snapshot')
UNIV = os.path.join(ROOT, 'data', 'hk_broad', 'universe_310.json')
HSI_CACHE = os.path.join(ROOT, 'data', 'hk_broad', 'hsi_daily.csv')

MAIN_START, MAIN_END = '2023-06-01', '2026-09-30'
OOS_START, OOS_END = '2016-01-01', '2023-05-31'
MIN_ADV = 5e7


# ---------------------------------------------------------------- 数据装载
def load_code_fys(code):
    """返回按财年排序的原始财务块（含财年结束日）。"""
    out = {}
    for kind in ('利润表_年度', '资产负债表_年度', '现金流量表_年度'):
        p = os.path.join(FUND, f'{code}_{kind}.pkl')
        if not os.path.exists(p):
            return []
        out[kind] = pd.read_pickle(p)
    is_, bs, cf = out['利润表_年度'], out['资产负债表_年度'], out['现金流量表_年度']
    dates = sorted(set(is_.index) & set(bs.index))
    rows = []
    for dt in dates:
        def pick(df, names):
            for n in names:
                if n in df.columns and df.loc[dt, n] == df.loc[dt, n]:
                    return float(df.loc[dt, n])
            return np.nan
        rev = pick(is_, ['营业额', '营业收入', '收入'])
        net = pick(is_, ['股东应占溢利', '除税后溢利'])
        eps = pick(is_, ['每股基本盈利'])
        gross = pick(is_, ['毛利'])
        eq = pick(bs, ['归属于母公司股东权益', '股东权益', '总权益'])
        assets = pick(bs, ['总资产', '总权益及总负债'])
        liab = pick(bs, ['总负债'])
        ocf = pick(cf, ['经营业务现金净额', '经营产生现金'])
        div = pick(cf, ['已付股息(融资)'])
        bb = pick(cf, ['回购股份'])
        iss = pick(cf, ['发行股份'])
        rows.append({'fy_end': pd.Timestamp(dt), 'rev': rev, 'net': net, 'eps': eps,
                     'gross': gross, 'eq': eq, 'assets': assets, 'liab': liab, 'ocf': ocf,
                     'div': abs(div) if div == div else np.nan,
                     'bb': bb, 'iss': iss})
    return rows


_MB_CACHE = {}


def metric_bundle_cached(code, fys, i):
    key = (code, i)
    if key not in _MB_CACHE:
        _MB_CACHE[key] = metric_bundle(fys, i)
    return _MB_CACHE[key]


def metric_bundle(fys, i):
    """给定财年序号 i，用 fys[i-4..i]（5 个财年）算指标（滞后无关，只随数据可得性选择）。"""
    w = fys[i - 4:i + 1]
    net = np.array([x['net'] for x in w])
    rev = np.array([x['rev'] for x in w])
    eq = np.array([x['eq'] for x in w])
    assets = np.array([x['assets'] for x in w])
    liab = np.array([x['liab'] for x in w])
    ocf = np.array([x['ocf'] for x in w])
    last = fys[i]
    shares = last['net'] / last['eps'] if last['net'] and last['eps'] and last['net'] == last['net'] and last['eps'] == last['eps'] else np.nan
    roes = [net[k] / eq[k] for k in (2, 3, 4) if eq[k] and eq[k] > 0 and net[k] == net[k]]
    with np.errstate(all='ignore'):
        yoy = np.where(np.abs(net[:-1]) > 0, net[1:] / np.abs(net[:-1]) - 1, np.nan)
        cc = ocf / net
        m = {
            'roe3': float(np.mean(roes)) if len(roes) == 3 else np.nan,
            'profit_cagr3': float((net[4] / net[1]) ** (1 / 3) - 1) if net[1] > 0 and net[4] > 0 else np.nan,
            'rev_cagr3': float((rev[4] / rev[1]) ** (1 / 3) - 1) if rev[1] > 0 and rev[4] > 0 else np.nan,
            'cash_conv3': float(np.nanmean(cc[2:])) if np.isfinite(cc[2:]).any() else np.nan,
            'dar': float(liab[4] / assets[4]) if assets[4] and liab[4] == liab[4] else np.nan,
            'worst_yoy': float(np.nanmin(yoy)) if np.isfinite(yoy).any() else np.nan,
            'neg_years': int(np.sum(net < 0)),
            'eps': last['eps'], 'shares': shares,
            'dps': (last['div'] / shares) if shares and last['div'] == last['div'] else np.nan,
            'bb': last['bb'], 'iss': last['iss'],
            'fy_end': last['fy_end'],
        }
    return m


def build_universe_data():
    """key 用 4 位代码（fund_cache 文件名口径，如 0700.HK），快照文件用 5 位（00700.pkl）。"""
    uni = json.load(open(UNIV))
    codes = sorted(uni.keys())
    meta = {c: uni[c] for c in codes}
    fys = {}
    for c in codes:
        f = load_code_fys(c)
        if len(f) >= 5:
            fys[c] = f
    return meta, fys


def load_prices(codes):
    close, turnover = {}, {}
    for c in codes:
        p = os.path.join(SNAP, c.split('.')[0].zfill(5) + '.pkl')
        if not os.path.exists(p):
            continue
        d = pd.read_pickle(p)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        close[c] = d['Close'].astype(float)
        turnover[c] = (d['Close'].astype(float) * d['Volume'].astype(float))
    C = pd.DataFrame(close).sort_index()
    T = pd.DataFrame(turnover).sort_index()
    return C, T


def load_hsi(start, end):
    if os.path.exists(HSI_CACHE):
        h = pd.read_csv(HSI_CACHE, index_col=0, parse_dates=True)['close']
        return h
    from data_services.tencent_finance import get_hsi_data_tencent
    d = get_hsi_data_tencent(period_days=3200)
    h = d['Close'].astype(float)
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    os.makedirs(os.path.dirname(HSI_CACHE), exist_ok=True)
    h.to_frame('close').to_csv(HSI_CACHE)
    return h


# ---------------------------------------------------------------- 选股
def rebalance_dates(idx, freq):
    if freq == 'Q':
        keys = idx.to_period('Q')
    elif freq == 'SA':
        keys = idx.to_period('Q')
        keys = keys.where(keys.quarter <= 2, keys - 1)  # 半年
    else:
        keys = idx.to_period('A')
    first = ~keys.duplicated()
    return list(idx[first])


def eligible_codes(T, fys, meta, px_row, adv_row, lag_days, min_adv=MIN_ADV):
    out = {}
    for c, f in fys.items():
        avail = [i for i, x in enumerate(f) if x['fy_end'] + timedelta(days=lag_days) <= T]
        if len(avail) < 5:
            continue
        i = avail[-1]
        m = metric_bundle_cached(c, f, i)
        px = px_row.get(c, np.nan)
        if not (px and px == px and px > 0):
            continue
        adv = adv_row.get(c, np.nan)
        if not (adv == adv and adv >= min_adv):
            continue
        ccy = ccy_of(c.split('.')[0].zfill(5))
        fx = FX[ccy]
        pe = px / (m['eps'] * fx) if m['eps'] and m['eps'] == m['eps'] and m['eps'] > 0 else np.nan
        dy = (m['dps'] * fx / px) if m['dps'] == m['dps'] else np.nan
        mcap = m['shares'] * px if m['shares'] == m['shares'] else np.nan
        nbb = (m['bb'] - m['iss']) if (m['bb'] == m['bb'] and m['iss'] == m['iss']) else 0.0
        bby = nbb / mcap if (mcap and mcap == mcap and mcap > 0) else 0.0
        m.update({'code': c, 'sector': meta[c].get('sector', ''), 'pe': pe, 'dy': dy, 'bby': bby,
                  'tsr': (dy if dy == dy else 0.0) + (bby if bby == bby else 0.0),
                  'financial': meta[c].get('sector', '') in ('bank', 'insurance', 'financial')})
        out[c] = m
    return out


def pass_filters(m):
    if m['neg_years'] > 0:
        return False
    if m['worst_yoy'] <= -0.30:
        return False
    if m['dar'] == m['dar'] and m['dar'] > 0.70:
        return False
    if m['financial']:
        return bool(m['roe3'] == m['roe3'] and m['roe3'] >= 0.08 and m['dy'] == m['dy'] and m['dy'] >= 0.03)
    return bool(m['roe3'] == m['roe3'] and m['roe3'] >= 0.10
                and m['cash_conv3'] == m['cash_conv3'] and m['cash_conv3'] >= 0.9
                and m['profit_cagr3'] == m['profit_cagr3'] and m['profit_cagr3'] >= 0
                and m['rev_cagr3'] == m['rev_cagr3'] and m['rev_cagr3'] >= 0)


def score_table(cands):
    keys = ['roe3', 'profit_cagr3', 'rev_cagr3', 'cash_conv3', 'pe', 'tsr']
    stats = {}
    for k in keys:
        v = np.array([m[k] for m in cands.values() if m[k] == m[k]], dtype=float)
        sv = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        if len(v) < 5 or not np.isfinite(sv) or sv == 0:
            return None
        stats[k] = (float(v.mean()), sv)
    out = {}
    w = {'roe3': 0.30, 'profit_cagr3': 0.15, 'rev_cagr3': 0.15, 'cash_conv3': 0.10, 'pe': 0.15, 'tsr': 0.15}
    for c, m in cands.items():
        s = 0.0
        ok = True
        for k, wk in w.items():
            if m[k] != m[k]:
                ok = False
                break
            mu, sd = stats[k]
            z = (m[k] - mu) / sd
            s += wk * (-z if k == 'pe' else z)
        if ok:
            out[c] = s
    return out


# ---------------------------------------------------------------- 组合模拟
def simulate(C, dates, sched, round_trip, apply_cost=True):
    """sched: {调仓日 -> {code: weight}}；决策在收盘，执行在下一交易日收盘。

    返回 {'value': 逐日净值, 'turnover': 单边换手, 'cost': 成本拖累}。
    """
    P = C.reindex(dates).ffill()
    px = P.values
    cols = {c: P.columns.get_loc(c) for c in P.columns}
    n = len(dates)
    value = 1.0
    shares = {}
    vals = np.zeros(n)
    turn = np.zeros(n)
    costs = np.zeros(n)
    pending = None
    for i, dt in enumerate(dates):
        if pending is not None and pending in sched:
            tgt = sched[pending]
            cur_val = sum(sh * px[i, cols[c]] for c, sh in shares.items()) if shares else value
            cur_w = {c: sh * px[i, cols[c]] / cur_val for c, sh in shares.items()} if cur_val else {}
            keys = set(cur_w) | set(tgt)
            delta = sum(abs(tgt.get(c, 0.0) - cur_w.get(c, 0.0)) for c in keys)
            turn[i] = 0.5 * delta
            cost = turn[i] * round_trip if apply_cost else 0.0
            costs[i] = cost
            val_after = cur_val - cost
            shares = {}
            for c, w in tgt.items():
                col = cols.get(c)
                if col is not None and px[i, col] == px[i, col] and px[i, col] > 0:
                    shares[c] = w * val_after / px[i, col]
            value = val_after
        elif shares:
            value = sum(sh * px[i, cols[c]] for c, sh in shares.items())
        vals[i] = value
        if dt in sched:
            pending = dt
    return {'value': vals, 'turnover': turn, 'cost': costs}


def ew_schedule(C, dates, rebal, elig_at, n=310):
    sched = {}
    for dt in rebal:
        elig = elig_at.get(dt, [])
        if not elig:
            continue
        w = 1.0 / len(elig)
        sched[dt] = {c: w for c in elig}
    return sched


# ---------------------------------------------------------------- 指标
def monthly_series(values, dates):
    s = pd.Series(values, index=dates)
    return s.resample('ME').last().pct_change().dropna()


def newey_west_t(x, lags=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    mu = x.mean()
    e = x - mu
    g0 = np.sum(e * e) / n
    var = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = np.sum(e[k:] * e[:-k]) / n
        var += 2 * (1 - k / (lags + 1)) * gk
    se = np.sqrt(max(var, 0) / n)
    if not se or not np.isfinite(se):
        return np.nan, np.nan
    t = mu / se
    return float(t), float(mu)


def max_drawdown(values):
    v = np.array(values, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def arm_stats(vals, dates, bench_vals, bench_dates, label=''):
    r_s = monthly_series(vals, dates)
    r_b = monthly_series(bench_vals, bench_dates)
    common = r_s.index.intersection(r_b.index)
    r_s, r_b = r_s.loc[common], r_b.loc[common]
    excess = (1 + r_s) / (1 + r_b) - 1
    n = len(excess)
    ann_ex = float(excess.mean() * 12)
    sd = float(excess.std(ddof=1) * np.sqrt(12)) if n > 2 else np.nan
    ir = ann_ex / sd if sd and sd == sd else np.nan
    t, _ = newey_west_t(excess.values)
    years = n / 12 if n else np.nan
    se_ann = float(excess.std(ddof=1) * np.sqrt(12) / np.sqrt(n)) if n > 2 else np.nan
    mde = float((1.96 + 0.84) * se_ann / sd) if (sd and sd == sd and se_ann == se_ann) else np.nan
    # 相对 HSI
    return {
        'label': label, 'months': n,
        'cagr': float(np.prod(1 + r_s.values) ** (12 / n) - 1) if n else np.nan,
        'excess_ann': ann_ex, 'excess_ir': ir, 'excess_t': t, 'excess_ci95': [ann_ex - 1.96 * se_ann, ann_ex + 1.96 * se_ann] if se_ann == se_ann else [np.nan, np.nan],
        'mde_ir': mde, 'maxdd': max_drawdown(vals),
        'terminal_multiple': float(vals[-1] / vals[0]),
        'yearly_excess': {str(k): float(v) for k, v in excess.groupby(excess.index.year).mean().items()},
        'monthly_excess': [float(x) for x in excess.values],
        'se_ann': se_ann,
    }


# ---------------------------------------------------------------- 多重检验
def deflated_sharpe(monthly, n_trials):
    """Deflated Sharpe（Bailey & Lopez de Prado 2014）：月度超额 + 试验次数惩罚。"""
    r = np.asarray(monthly, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 12:
        return np.nan
    sr = float(r.mean() / r.std(ddof=1))          # 逐期 SR
    sr_ann = sr * np.sqrt(12)
    if n_trials < 2:
        return 1.0 if sr_ann > 0 else 0.0
    var_sr = (1 + sr ** 2 / 2) / T
    z = lambda p: float(__import__('scipy.stats', fromlist=['norm']).norm.ppf(p))
    gamma = 0.5772156649
    e = np.e
    sr_max = np.sqrt(var_sr) * ((1 - gamma) * z(1 - 1 / n_trials) + gamma * z(1 - 1 / (n_trials * e)))
    dsr = float(__import__('scipy.stats', fromlist=['norm']).norm.cdf((sr - sr_max) * np.sqrt(T - 1)))
    return {'sr_ann': sr_ann, 'sr_max_ann': float(sr_max * np.sqrt(12)), 'dsr': dsr, 'trials': n_trials}


def pbo_cscv(monthly_matrix, n_slices=8):
    """CSCV-PBO：行=候选策略，列=期；返回 OOS 落入下半区的比例。"""
    M = np.asarray(monthly_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 2 * n_slices:
        return np.nan
    T = M.shape[1]
    per = T // n_slices
    slices = [M[:, i * per:(i + 1) * per] for i in range(n_slices)]
    worse = 0
    total = 0
    for combo in _combinations(list(range(n_slices)), n_slices // 2):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_slices) if i not in is_idx]
        is_mat = np.hstack([slices[i] for i in is_idx])
        oos_mat = np.hstack([slices[i] for i in oos_idx])
        is_score = np.nanmean(is_mat, axis=1)
        oos_score = np.nanmean(oos_mat, axis=1)
        best = int(np.nanargmax(is_score))
        rank = float((oos_score < oos_score[best]).sum()) / (len(oos_score) - 1)
        total += 1
        if rank < 0.5:
            worse += 1
    return worse / total if total else np.nan


def _combinations(items, k):
    from itertools import combinations
    return list(combinations(items, k))


# ---------------------------------------------------------------- 主流程
def run_arm(C, dates, rebal, elig_at, n_hold, lag, round_trip, apply_cost, rng=None, random_pick=False):
    sched = {}
    prev_hold = []
    for dt in rebal:
        elig = elig_at.get(dt, [])
        if not elig:
            continue
        if random_pick:
            pick_n = min(n_hold, len(elig))
            hold = list(rng.choice(elig, size=pick_n, replace=False))
        else:
            cands = {c: m for c, m in elig_at_metrics[dt].items() if pass_filters(m)}
            sc = score_table(cands)
            if not sc:
                continue
            ranked = sorted(sc, key=lambda c: -sc[c])
            rank = {c: i + 1 for i, c in enumerate(ranked)}
            keep = [c for c in prev_hold if c in rank and rank[c] <= 2 * n_hold]
            keep = sorted(keep, key=lambda c: -sc[c])[:n_hold]
            fill = [c for c in ranked if c not in keep][:n_hold - len(keep)]
            hold = keep + fill
        if not hold:
            continue
        w = 1.0 / len(hold)
        sched[dt] = {c: w for c in hold}
        prev_hold = hold
    return simulate(C, dates, sched, round_trip, apply_cost), sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', '_logs', 'lt_experiment.json'))
    ap.add_argument('--lag', type=int, default=120, help='财报披露滞后（天）')
    ap.add_argument('--round-trip', type=float, default=0.0102, help='往返成本（主口径 1.02%）')
    ap.add_argument('--n', type=int, default=20, help='持仓数')
    ap.add_argument('--freq', default='Q', choices=['Q', 'SA', 'A'])
    ap.add_argument('--grid', action='store_true', help='跑相邻参数网格 + 安慰剂')
    ap.add_argument('--placebo', type=int, default=20, help='安慰剂随机次数')
    args = ap.parse_args()

    global elig_at_metrics
    meta, fys = build_universe_data()
    codes = sorted(fys.keys())
    C, T = load_prices(codes)
    print(f'价格面板 {C.shape}（{C.index[0].date()} → {C.index[-1].date()}）')

    results = {'config': vars(args), 'arms': {}, 'grid': [], 'placebo': []}
    sector_mix = {}

    def build_window(start, end):
        idx = C.index[(C.index >= pd.Timestamp(start)) & (C.index <= pd.Timestamp(end))]
        return list(idx)

    windows = {'main': (MAIN_START, MAIN_END), 'oos': (OOS_START, OOS_END)}
    for wname, (ws, we) in windows.items():
        dates = build_window(ws, we)
        if len(dates) < 200:
            continue
        rebal = rebalance_dates(pd.DatetimeIndex(dates), args.freq)
        elig_at, elig_at_metrics_ = {}, {}
        for dt in rebal:
            i = C.index.get_loc(dt)
            px_row = C.iloc[i].to_dict()
            adv_row = T.iloc[max(0, i - 20):i + 1].median().to_dict() if i > 0 else {}
            elig_at_metrics_[dt] = eligible_codes(dt, fys, meta, px_row, adv_row, args.lag)
            elig_at[dt] = list(elig_at_metrics_[dt].keys())
        elig_at_metrics = elig_at_metrics_
        print(f'[{wname}] {dates[0].date()} → {dates[-1].date()}，调仓 {len(rebal)} 次')

        strat, sched = run_arm(C, dates, rebal, elig_at, args.n, args.lag, args.round_trip, True)
        bench_sched = ew_schedule(C, dates, rebal, elig_at)
        bench = simulate(C, dates, bench_sched, args.round_trip, apply_cost=False)
        # 静态等权买入持有（对照"只是少投"）
        static_sched = {}
        if rebal:
            first = rebal[0]
            elig = elig_at.get(first, [])
            if elig:
                w = 1.0 / len(elig)
                static_sched[first] = {c: w for c in elig}
        static = simulate(C, dates, static_sched, args.round_trip, apply_cost=False)

        st = arm_stats(strat['value'], dates, bench['value'], dates, '底仓(质量+价值)')
        st_bench = arm_stats(bench['value'], dates, bench['value'], dates, '等权池(基准)')
        st_static = arm_stats(static['value'], dates, bench['value'], dates, '静态等权买入持有(对照)')
        turn_yr = float(strat['turnover'].sum() / max(1, len(dates) / 252))
        st['turnover_per_year'] = turn_yr
        mix = {}
        for dt, hold in sched.items():
            for c, w in hold.items():
                sec = meta[c].get('sector', '?')
                mix[sec] = mix.get(sec, 0.0) + w / len(sched)
        st['sector_mix'] = dict(sorted(mix.items(), key=lambda kv: -kv[1]))
        st['cost_drag_ann'] = float(strat['cost'].sum() / max(1, len(dates) / 252))
        st_static['turnover_per_year'] = 0.0
        results['arms'][wname] = {'strategy': st, 'ew_bench': st_bench, 'static_ew': st_static,
                                  'n_holdings': len(sched[rebal[-1]]) if sched else 0,
                                  'rebalances': len(rebal)}
        # HSI
        try:
            hsi = load_hsi(ws, we)
            hsi = hsi[(hsi.index >= dates[0]) & (hsi.index <= dates[-1])]
            hvals = (hsi / hsi.iloc[0]).reindex(dates).ffill().values
            st_hsi = arm_stats(strat['value'], dates, hvals, dates, '底仓 vs HSI')
            st_hsi.pop('monthly_excess', None)
            results['arms'][wname]['vs_hsi'] = st_hsi
        except Exception as e:
            results['arms'][wname]['hsi_error'] = str(e)

        if args.grid and wname == 'main':
            for freq in ('Q', 'SA', 'A'):
                for n in (15, 20, 30):
                    for lag in (90, 120, 180):
                        rb = rebalance_dates(pd.DatetimeIndex(dates), freq)
                        ea, eam = {}, {}
                        for dt in rb:
                            i = C.index.get_loc(dt)
                            px_row = C.iloc[i].to_dict()
                            adv_row = T.iloc[max(0, i - 20):i + 1].median().to_dict() if i > 0 else {}
                            eam[dt] = eligible_codes(dt, fys, meta, px_row, adv_row, lag)
                            ea[dt] = list(eam[dt].keys())
                        elig_at_metrics = eam
                        s, _ = run_arm(C, dates, rb, ea, n, lag, args.round_trip, True)
                        b = simulate(C, dates, ew_schedule(C, dates, rb, ea), args.round_trip, apply_cost=False)
                        a = arm_stats(s['value'], dates, b['value'], dates, f'{freq}/{n}/{lag}')
                        results['grid'].append({'freq': freq, 'n': n, 'lag': lag,
                                                'excess_ann': a['excess_ann'], 'excess_ir': a['excess_ir'],
                                                'excess_t': a['excess_t'], 'ci': a['excess_ci95'],
                                                'mde_ir': a['mde_ir'], 'turnover_per_year': float(s['turnover'].sum() / max(1, len(dates) / 252)),
                                                'monthly_excess': a['monthly_excess']})
                        print(f'  网格 {freq}/{n}/{lag}: 年化超额 {a["excess_ann"]:+.2%}，IR {a["excess_ir"]:.2f}')

            # 安慰剂：随机选股，同流程
            rng = np.random.default_rng(42)
            for k in range(args.placebo):
                s, _ = run_arm(C, dates, rebal, elig_at, args.n, args.lag, args.round_trip, True,
                               rng=rng, random_pick=True)
                b = simulate(C, dates, ew_schedule(C, dates, rebal, elig_at), args.round_trip, apply_cost=False)
                a = arm_stats(s['value'], dates, b['value'], dates, 'placebo')
                results['placebo'].append({'excess_ann': a['excess_ann'], 'excess_ir': a['excess_ir']})
                print(f'  安慰剂 {k + 1}/{args.placebo}: 年化超额 {a["excess_ann"]:+.2%}，IR {a["excess_ir"]:.2f}')

    # 多重检验与过拟合闸（用网格全部候选）
    trials = max(1, len(results['grid']))
    for wname, r in results['arms'].items():
        st = r['strategy']
        try:
            st['dsr'] = deflated_sharpe(st['monthly_excess'], trials if trials > 1 else 2)
        except Exception as e:
            st['dsr'] = {'error': str(e)}
        if results['grid'] and wname == 'main':
            M = np.array([g['monthly_excess'] for g in results['grid']], dtype=float)
            L = min(M.shape[1], min(len(g['monthly_excess']) for g in results['grid']))
            try:
                st['pbo'] = float(pbo_cscv(M[:, -L:]))
            except Exception as e:
                st['pbo'] = str(e)
        if results['placebo']:
            pv = np.array([x['excess_ann'] for x in results['placebo']])
            st['placebo_mean'] = float(pv.mean())
            st['placebo_pct'] = float((pv < st['excess_ann']).mean())
        st.pop('monthly_excess', None)
        r['ew_bench'].pop('monthly_excess', None)
        r['static_ew'].pop('monthly_excess', None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    for wname, r in results['arms'].items():
        st = r['strategy']
        print(f'\n[{wname}] 底仓：年化超额 {st["excess_ann"]:+.2%}（IR {st["excess_ir"]:.2f}，t {st["excess_t"]:.2f}，'
              f'CI {st["excess_ci95"][0]:+.2%}~{st["excess_ci95"][1]:+.2%}，MDE {st["mde_ir"]:.2f}）')
        print(f'        CAGR {st["cagr"]:+.2%}，最大回撤 {st["maxdd"]:.1%}，年换手 {st["turnover_per_year"]:.1%}，'
              f'成本拖累 {st["cost_drag_ann"]:.2%}/年')
        print(f'        静态等权对照：年化超额 {r["static_ew"]["excess_ann"]:+.2%}')
        if isinstance(st.get('dsr'), dict):
            print(f'        DSR {st["dsr"]["dsr"]:.3f}（SR_ann {st["dsr"]["sr_ann"]:+.2f}，零假设 SR_max {st["dsr"]["sr_max_ann"]:+.2f}，试验 {st["dsr"]["trials"]} 次）')
        if isinstance(st.get('pbo'), float):
            print(f'        PBO {st["pbo"]:.3f}')
        if 'placebo_mean' in st:
            print(f'        安慰剂均值 {st["placebo_mean"]:+.2%}，真实值分位 {st["placebo_pct"]:.0%}')
    print(f'\n结果 → {args.out}')


if __name__ == '__main__':
    main()
