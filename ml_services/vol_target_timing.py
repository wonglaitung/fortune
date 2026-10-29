"""波动目标（Volatility Targeting / Inverse-Vol）风控层 —— 恒指 ETF 择时（standalone）

设计（修正版，经对抗审核）：

  - **不赌方向**：按实现波动率反向缩放仓位（波动高→减仓，波动低→满仓）。
    因此没有"买/卖"方向决策，不存在"知道买不知卖"的问题。
  - **只做 standalone 恒指 ETF**；**不做 TopK 覆盖层**——个股 alpha 已证 null，
    缩放一个 null 仍是 null，覆盖无意义。
  - **双基准**：vs 买入持有（bh）AND vs **静态等平均仓位对照**（st）。
    后者用于隔离"vol 择时增量" vs "只是平均少投"。
  - **含港股成本**（印花税+佣金+滑点，保守 0.13% 印花税口径）；决策严格 <=T，T+1 生效。
  - **定性：风险预算工具，不是信号/alpha；预期收益 <= 买入持有。**

预登记判据（跑前定稿，不得事后改）：
  HEADLINE = (target=12%, window=60)
  PASS 条件：
    ① maxDD(vt) <= 0.8 * maxDD(bh)                       # 回撤实质下降
    ② Calmar(vt) >= Calmar(st)  且 >= Calmar(bh)          # 优于"静态等仓位"才叫择时增量
    ③ 网格稳健：{10,12,15}% x {20,60} 六格中，①与②各 >=5/6 成立
  三条全过才部署；不过则落地"vol 目标无择时增量，不部署"。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# 港股单边成本 ≈ 23bps（印花税 0.13% + 佣金 + 滑点），保守取 0.13% 印花税口径
from ml_services.costs import COST_ROUND_TRIP
DD_REDUCTION = 0.8
GRID_TARGET = (0.10, 0.12, 0.15)
GRID_WINDOW = (20, 60)
GRID_MIN_PASS = 5  # 六格中至少 5 格成立
HEADLINE_TARGET = 0.12
HEADLINE_WINDOW = 60


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """滚动实现波动率（年化）。PIT：仅用 <=T 的历史，无未来函数。"""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def vol_exposure(returns: pd.Series, target_vol: float = HEADLINE_TARGET,
                 window: int = HEADLINE_WINDOW, cap: float = 1.0,
                 floor: float = 0.0) -> pd.Series:
    """波动目标仓位 w_t = clip(target_vol / realized_vol, floor, cap)。"""
    rv = realized_vol(returns, window)
    w = target_vol / rv
    return w.clip(lower=floor, upper=cap)


def to_monthly(w: pd.Series) -> pd.Series:
    """月频再平衡：取每月最后一个交易日的 w，其余日沿用（版本无关实现）。"""
    key = w.index.to_period('M')
    monthly = w.groupby(key).last()
    return pd.Series(monthly.reindex(key).to_numpy(), index=w.index)


def _turnover_cost(w_pos: pd.Series, round_trip_cost: float):
    """换手成本：|Δw| * 单边成本（首次建仓也算换手）。"""
    turn = w_pos.diff().abs()
    if len(turn) > 0:
        turn.iloc[0] = abs(w_pos.iloc[0])
    return 0.5 * turn * round_trip_cost, turn


def strat_returns(asset_returns: pd.Series, w: pd.Series, lag: int = 1,
                  round_trip_cost: float = COST_ROUND_TRIP):
    """给定目标仓位 w（T 决定），T+lag 生效；扣换手成本。

    Returns: (net, gross, turnover, w_pos)
    """
    w_pos = w.shift(lag).fillna(0.0)
    gross = w_pos * asset_returns
    cost, turn = _turnover_cost(w_pos, round_trip_cost)
    net = gross - cost
    return net, gross, turn, w_pos


def static_matched_returns(asset_returns: pd.Series, target_exposure: float,
                           round_trip_cost: float = COST_ROUND_TRIP) -> pd.Series:
    """静态等平均仓位对照：恒定 target_exposure，月频再平衡回该值（含成本）。"""
    w = pd.Series(float(target_exposure), index=asset_returns.index)
    w_m = to_monthly(w)
    net, _, _, _ = strat_returns(asset_returns, w_m, lag=1,
                                 round_trip_cost=round_trip_cost)
    return net


def metrics(returns: pd.Series) -> dict:
    """收益/风险指标。maxDD 为负；Calmar = CAGR / |maxDD|。"""
    r = returns.dropna()
    if len(r) == 0 or float(r.std()) == 0.0:
        nan = float('nan')
        return dict(n=len(r), total=nan, cagr=nan, ann_ret=nan, vol=nan,
                    sharpe=nan, sortino=nan, maxdd=nan, calmar=nan)
    n = len(r)
    cum = (1.0 + r).cumprod()
    total = float(cum.iloc[-1] - 1.0)
    cagr = float(cum.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float(r.mean() * TRADING_DAYS)
    sharpe = ann_ret / vol if vol else float('nan')
    downside = float(r[r < 0].std() * np.sqrt(TRADING_DAYS)) if (r < 0).any() else 0.0
    sortino = ann_ret / downside if downside else float('nan')
    dd = cum / cum.cummax() - 1.0
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd != 0 else float('nan')
    return dict(n=n, total=total, cagr=cagr, ann_ret=ann_ret, vol=vol,
                sharpe=sharpe, sortino=sortino, maxdd=maxdd, calmar=calmar)


def backtest(close: pd.Series, target_vol: float = HEADLINE_TARGET,
             window: int = HEADLINE_WINDOW,
             round_trip_cost: float = COST_ROUND_TRIP,
             monthly: bool = True) -> dict:
    """对恒指 ETF 做 vol-target 回测，返回 vt / bh / st 三条曲线与指标。

    vt = vol-target 择时；bh = 买入持有；st = 静态等平均仓位对照。
    """
    ret = close.pct_change().fillna(0.0)
    w_raw = vol_exposure(ret, target_vol, window)
    w = to_monthly(w_raw) if monthly else w_raw
    # 有效窗口：从首个有效 w（波动率 burn-in 之后）起，三条曲线同窗比较
    start = w.first_valid_index()
    if start is None:
        raise ValueError("有效仓位为空（样本不足 window 天？）")
    ret = ret.loc[start:]
    w = w.loc[start:]

    vt_net, vt_gross, turn, w_pos = strat_returns(ret, w, round_trip_cost=round_trip_cost)
    bh = ret
    mean_exp = float(w_pos.mean())
    st_net = static_matched_returns(ret, mean_exp, round_trip_cost)
    return dict(
        vt=vt_net, bh=bh, st=st_net, w_pos=w_pos, turnover=turn,
        mean_exp=mean_exp, ret=ret, start=start,
        m_vt=metrics(vt_net), m_bh=metrics(bh), m_st=metrics(st_net),
    )


def evaluate_config(close: pd.Series, target_vol: float, window: int,
                    round_trip_cost: float = COST_ROUND_TRIP) -> dict:
    """单格判据：DD 条件 + Calmar(vt>=st,bh)。"""
    r = backtest(close, target_vol, window, round_trip_cost)
    dd_ok = (r['m_vt']['maxdd'] >= DD_REDUCTION * r['m_bh']['maxdd'])  # 负数比较
    calmar_vt, calmar_st, calmar_bh = r['m_vt']['calmar'], r['m_st']['calmar'], r['m_bh']['calmar']
    calmar_ok = (calmar_vt >= calmar_st) and (calmar_vt >= calmar_bh)
    return dict(target=target_vol, window=window, r=r, dd_ok=dd_ok,
                calmar_ok=calmar_ok, calmar_vt=calmar_vt, calmar_st=calmar_st,
                calmar_bh=calmar_bh, maxdd_vt=r['m_vt']['maxdd'], maxdd_bh=r['m_bh']['maxdd'])


def run_grid(close: pd.Series, round_trip_cost: float = COST_ROUND_TRIP) -> dict:
    cells = [evaluate_config(close, t, w, round_trip_cost)
             for t in GRID_TARGET for w in GRID_WINDOW]
    n = len(cells)
    dd_pass = sum(c['dd_ok'] for c in cells)
    calmar_pass = sum(c['calmar_ok'] for c in cells)
    return dict(cells=cells, n=n, dd_pass=dd_pass, calmar_pass=calmar_pass,
                grid_ok=(dd_pass >= GRID_MIN_PASS and calmar_pass >= GRID_MIN_PASS))


def load_hsi(period_days: int = 2600, start: str | None = None) -> pd.Series:
    """加载恒指 qfq 收盘价序列（腾讯，项目同源），可选按起始日过滤。"""
    from data_services.tencent_finance import get_hsi_data_tencent
    df = get_hsi_data_tencent(period_days=period_days)
    if df is None or df.empty:
        raise RuntimeError("无法获取恒指数据")
    close = df['Close'].astype(float).sort_index()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, 'tz', None) is not None:
        close.index = close.index.tz_convert(None)
    close = close[~close.index.duplicated(keep='last')]
    if start:
        close = close.loc[close.index >= pd.Timestamp(start)]
    return close


def _fmt_row(name, m):
    return (f"{name:<14} 总收益 {m['total']*100:>7.1f}%  CAGR {m['cagr']*100:>6.1f}%  "
            f"波动 {m['vol']*100:>5.1f}%  夏普 {m['sharpe']:>5.2f}  索提诺 {m['sortino']:>5.2f}  "
            f"maxDD {m['maxdd']*100:>6.1f}%  Calmar {m['calmar']:>5.2f}")


def report(close: pd.Series, round_trip_cost: float = COST_ROUND_TRIP) -> bool:
    """跑 headline + 网格，打印并返回是否 PASS。"""
    head = evaluate_config(close, HEADLINE_TARGET, HEADLINE_WINDOW, round_trip_cost)
    r = head['r']
    print("=" * 96)
    print(f"波动目标风控层（standalone 恒指 ETF）  样本 {r['start'].date()} ~ {close.index[-1].date()}"
          f"  ({r['m_bh']['n']} 交易日)")
    print(f"headline: target={HEADLINE_TARGET:.0%}  window={HEADLINE_WINDOW}  "
          f"月频再平衡  单边成本={round_trip_cost*100:.2f}%  平均仓位={r['mean_exp']:.1%}")
    print("-" * 96)
    print(_fmt_row('vol-target', r['m_vt']))
    print(_fmt_row('买入持有', r['m_bh']))
    print(_fmt_row('静态等仓位', r['m_st']))
    print("-" * 96)
    dd1 = r['m_vt']['maxdd'] / r['m_bh']['maxdd'] if r['m_bh']['maxdd'] else float('nan')
    print(f"① maxDD(vt)/maxDD(bh) = {dd1:.2f}  (需 <= {DD_REDUCTION:.2f})  -> "
          f"{'PASS' if head['dd_ok'] else 'FAIL'}")
    print(f"② Calmar(vt)={head['calmar_vt']:.2f} vs (st)={head['calmar_st']:.2f} vs (bh)="
          f"{head['calmar_bh']:.2f}  -> {'PASS' if head['calmar_ok'] else 'FAIL'}")
    print("-" * 96)
    grid = run_grid(close, round_trip_cost)
    print(f"③ 网格 {{10,12,15}}%x{{20,60}}: DD 过 {grid['dd_pass']}/{grid['n']}, "
          f"Calmar 过 {grid['calmar_pass']}/{grid['n']}  (需各 >= {GRID_MIN_PASS})  -> "
          f"{'PASS' if grid['grid_ok'] else 'FAIL'}")
    print("  逐格：")
    for c in grid['cells']:
        print(f"    target={c['target']:.0%} win={c['window']:>2}  "
              f"maxDD vt/bh={c['maxdd_vt']*100:6.1f}%/{c['maxdd_bh']*100:6.1f}%  "
              f"Calmar vt/st/bh={c['calmar_vt']:.2f}/{c['calmar_st']:.2f}/{c['calmar_bh']:.2f}  "
              f"DD={'Y' if c['dd_ok'] else 'N'} Calmar={'Y' if c['calmar_ok'] else 'N'}")
    print("=" * 96)
    passed = head['dd_ok'] and head['calmar_ok'] and grid['grid_ok']
    print(f"预登记判定：{'PASS —— 可部署' if passed else 'FAIL —— vol 目标无择时增量，不部署'}")
    print("定性：风险预算工具，非信号/alpha；预期收益 <= 买入持有。")
    return passed


def main():
    ap = argparse.ArgumentParser(description='波动目标风控层（standalone 恒指 ETF）')
    ap.add_argument('--mode', choices=['backtest', 'asof'], default='backtest')
    ap.add_argument('--start', type=str, default='2020-01-01', help='回测起始日（PIT）')
    ap.add_argument('--target-vol', type=float, default=HEADLINE_TARGET)
    ap.add_argument('--window', type=int, default=HEADLINE_WINDOW)
    ap.add_argument('--cost', type=float, default=COST_ROUND_TRIP, help='往返成本（默认 0.5%，单边 0.25%）')
    ap.add_argument('--period-days', type=int, default=2600)
    args = ap.parse_args()

    close = load_hsi(period_days=args.period_days, start=args.start)
    if args.mode == 'asof':
        ret = close.pct_change().fillna(0.0)
        w = to_monthly(vol_exposure(ret, args.target_vol, args.window))
        rv = realized_vol(ret, args.window).iloc[-1]
        print(f"截至 {close.index[-1].date()}  恒指 {close.iloc[-1]:.1f}  "
              f"实现波动(年化) {rv*100:.1f}%  目标 {args.target_vol*100:.0f}%  "
              f"-> 建议仓位 {w.iloc[-1]*100:.0f}%")
        return
    report(close, round_trip_cost=args.cost)


if __name__ == '__main__':
    main()
