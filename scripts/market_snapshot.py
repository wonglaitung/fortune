#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""港股/美股运行时最新快照。

规则：
- 港股：盘中用最新分钟价；收盘后用收盘价。
- 美股：盘中/盘前/盘后都用最新分钟价；收盘后用收盘价。
- 1日涨跌 = 最新价 / 上一交易日收盘 - 1；5日同理。
- 每个数字都带“市场状态 + 最新时间”，不把盘中价写成收盘。

用法：
  python3 scripts/market_snapshot.py
  python3 scripts/market_snapshot.py --out output/market_snapshot.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "market_snapshot"
HKT = ZoneInfo("Asia/Hong_Kong")
ET = ZoneInfo("America/New_York")

HK = {"^HSI": "恒生指数", "^HSCE": "恒生国企"}
US = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX", "^TNX": "US 10Y", "DX-Y.NYB": "DXY"}


def market_state(now: datetime, market: str) -> str:
    if now.weekday() >= 5:
        return "休市"
    t = now.time()
    if market == "HK":
        if t < time(9, 30):
            return "盘前"
        if time(12, 0) <= t < time(13, 0):
            return "午休"
        if t < time(16, 0):
            return "盘中"
        return "收盘"
    if t < time(9, 30):
        return "盘前"
    if t < time(16, 0):
        return "盘中"
    return "盘后"


def fetch(sym: str) -> tuple[pd.Series, pd.DataFrame, pd.Timestamp]:
    daily = yf.download(sym, period="15d", interval="1d", auto_adjust=True, progress=False)
    intra = yf.download(sym, period="1d", interval="1m", auto_adjust=True, progress=False, prepost=True)
    dc = daily["Close"].dropna()
    ic = intra["Close"].dropna()
    if ic.empty:
        return dc, daily, dc.index[-1]
    return ic, daily, ic.index[-1]


def snapshot(sym: str, name: str, market: str, tz: ZoneInfo) -> dict | None:
    latest, daily, ts = fetch(sym)
    if latest.empty or daily.empty:
        return None
    now = datetime.now(tz)
    # 用最新价所在日期匹配日线；盘前/盘中都取“上一交易日收盘”做基准
    ts_local = pd.Timestamp(ts).tz_convert(tz)
    day = ts_local.date()
    dc = daily["Close"].dropna()
    dc.index = pd.to_datetime(dc.index)
    hist = dc[dc.index.date < day]
    if len(hist) < 5:
        return None
    h = hist.to_numpy().ravel()
    prev = float(h[-1])
    base5 = float(h[-5])
    px = float(latest.to_numpy().ravel()[-1])
    asof_hkt = pd.Timestamp(ts).tz_convert(HKT)
    return {
        "market": market,
        "name": name,
        "state": market_state(now, market),
        "asof": f"{asof_hkt:%m-%d %H:%M} HKT",
        "last": px,
        "r1": px / prev - 1,
        "r5": px / base5 - 1,
    }


def render(rows: list[dict]) -> str:
    now = datetime.now(HKT)
    lines = [
        "# 市场快照（运行时最新）",
        "",
        f"- 生成时间：{now:%Y-%m-%d %H:%M} HKT",
        "- 港股盘中=最新分钟价；美股盘中/盘前/盘后=最新分钟价；收盘后=收盘价。",
        "- 1日涨跌以“最新价 / 上一交易日收盘”计算；5日同理。",
        "- 数据源：yfinance；分钟数据可能有延迟。",
        "",
        "| 市场 | 指标 | 状态 | 最新时间 | 最新 | 1日至今 | 5日 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['market']} | {r['name']} | {r['state']} | {r['asof']} | {r['last']:.2f} | "
                     f"{r['r1']*100:+.1f}% | {r['r5']*100:+.1f}% |")
    lines += ["", "> 盘中/盘前/盘后价格不是最终收盘；不构成交易建议。"]
    return "\n".join(lines) + "\n"


def collect() -> list[dict]:
    rows = []
    for sym, name in HK.items():
        r = snapshot(sym, name, "港股", HKT)
        if r:
            rows.append(r)
    for sym, name in US.items():
        r = snapshot(sym, name, "美股", ET)
        if r:
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = collect()
    text = render(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    md = Path(args.out) if args.out else OUT / f"{datetime.now(HKT):%Y-%m-%d}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    print(text)
    print(f"💾 {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
