#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日事件雷达（原型）。

数据源：
- Nasdaq 经济日历 API：事件、GMT 时间、预期/实际/前值
- Federal Reserve RSS：最新官方新闻
- yfinance：S&P500 / Nasdaq / VIX / 美债10Y / 美元指数 / 恒指

用法：
  python3 scripts/event_radar.py
  python3 scripts/event_radar.py --days 3
  python3 scripts/event_radar.py --out output/event_radar/today.md

只做信息汇总与风险提示，不构成交易建议。
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "event_radar"
CAL_URL = "https://api.nasdaq.com/api/calendar/economicevents"
FED_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
HKT = timezone(timedelta(hours=8))
EST = timezone(timedelta(hours=-5))
TRACK = {"United States", "China", "Hong Kong", "Euro Zone", "Japan"}
KEY = [
    "FOMC", "Fed", "Interest Rate", "CPI", "PCE", "Nonfarm", "Unemployment",
    "GDP", "PMI", "Retail Sales", "Treasury", "Trade Balance", "Industrial Production",
]
ASSETS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^VIX": "VIX",
    "^TNX": "US 10Y",
    "DX-Y.NYB": "DXY",
    "^HSI": "恒生指数",
}


def clean(x) -> str:
    s = str(x or "").replace("&nbsp;", " ").replace("\xa0", " ").strip()
    return "" if s in {"", "-", "N/A", "n/a"} else s


def num(x):
    s = clean(x).replace(",", "").replace("%", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def surprise(actual, consensus):
    a, c = num(actual), num(consensus)
    if a is None or c is None:
        return "—"
    d = a - c
    if abs(c) > 1e-9:
        return f"{d:+.2f} ({d / abs(c) * 100:+.1f}%)"
    return f"{d:+.2f}"


def why(event: str) -> str:
    e = event.lower()
    if any(k in e for k in ("fomc", "interest rate", "fed")):
        return "利率路径/折现率，影响科技估值与银行 NIM"
    if any(k in e for k in ("cpi", "pce", "inflation")):
        return "通胀 → 联储路径 → 美债/美元/港股估值"
    if any(k in e for k in ("nonfarm", "unemployment", "employment")):
        return "就业 → 联储路径 → 美债/美元"
    if "gdp" in e:
        return "增长预期 → 盈利与风险偏好"
    if "pmi" in e:
        return "景气领先指标 → 周期/板块"
    if "retail" in e:
        return "消费强弱 → 盈利与利率"
    if "treasury" in e:
        return "供给/收益率 → 全球估值锚"
    return "宏观/政策信息，影响风险偏好"


def calendar(day: date) -> list[dict]:
    r = requests.get(CAL_URL, params={"date": day.isoformat()}, timeout=25,
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    r.raise_for_status()
    data = r.json().get("data") or {}
    out = []
    for x in data.get("rows") or []:
        country = clean(x.get("country"))
        event = clean(x.get("eventName"))
        if country not in TRACK:
            continue
        gmt = clean(x.get("gmt"))
        if not gmt or gmt.upper() == "24H":
            continue
        if not any(k.lower() in event.lower() for k in KEY):
            continue
        low = event.lower()
        if any(k in low for k in ("gdpnow", "balance sheet", "reserve balances", "private nonfarm", "u6 unemployment")):
            continue
        if "speaks" in low and not any(k in low for k in ("powell", "chair", "vice chair")):
            continue
        try:
            hh, mm = [int(v) for v in gmt.split(":")[:2]]
            dt = datetime.combine(day, time(hh, mm), tzinfo=EST).astimezone(HKT)
        except Exception:
            continue
        out.append({
            "hkt": dt.strftime("%m-%d %H:%M"),
            "country": country,
            "event": event,
            "consensus": clean(x.get("consensus")),
            "actual": clean(x.get("actual")),
            "previous": clean(x.get("previous")),
            "surprise": surprise(x.get("actual"), x.get("consensus")),
            "why": why(event),
        })
    return out


def market() -> tuple[list[dict], list[str]]:
    import yfinance as yf
    rows, flags = [], []
    d = yf.download(list(ASSETS), period="10d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    for tk, name in ASSETS.items():
        try:
            x = d[tk]["Close"].dropna()
            last, d1, d5 = float(x.iloc[-1]), float(x.iloc[-1] / x.iloc[-2] - 1), float(x.iloc[-1] / x.iloc[-6] - 1)
            rows.append({"asset": name, "date": str(x.index[-1].date()), "last": last, "d1": d1, "d5": d5})
        except Exception:
            rows.append({"asset": name, "date": "—", "last": None, "d1": None, "d5": None})
    m = {r["asset"]: r for r in rows}
    if m.get("S&P 500", {}).get("d1") is not None and m["S&P 500"]["d1"] <= -0.02:
        flags.append("US_EQUITY_DROP")
    if m.get("Nasdaq", {}).get("d1") is not None and m["Nasdaq"]["d1"] <= -0.03:
        flags.append("NASDAQ_DROP")
    if m.get("VIX", {}).get("d1") is not None and m["VIX"]["d1"] >= 0.20:
        flags.append("VIX_SPIKE")
    if m.get("US 10Y", {}).get("d5") is not None and m["US 10Y"]["d5"] >= 0.15:
        flags.append("US10Y_UP")
    if m.get("S&P 500", {}).get("d5") is not None and m["S&P 500"]["d5"] <= -0.03:
        flags.append("US_EQUITY_5D_DROP")
    if m.get("Nasdaq", {}).get("d5") is not None and m["Nasdaq"]["d5"] <= -0.03:
        flags.append("NASDAQ_5D_DROP")
    if m.get("VIX", {}).get("d5") is not None and m["VIX"]["d5"] >= 0.30:
        flags.append("VIX_5D_SPIKE")
    return rows, flags


def fed_news(n=3) -> list[dict]:
    try:
        r = requests.get(FED_RSS, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item")[:n]:
            out.append({"title": clean(item.findtext("title")), "date": clean(item.findtext("pubDate"))})
        return out
    except Exception:
        return []


def score_event(e: dict) -> float:
    name = e["event"].lower()
    s = 0.0
    if any(k in name for k in ("fomc", "interest rate decision", "fed interest rate")):
        s += 6
    if any(k in name for k in ("nonfarm payrolls", "cpi", "pce", "gdp")):
        s += 5
    if "unemployment rate" in name:
        s += 3
    if any(k in name for k in ("retail sales", "ism", "pmi")):
        s += 2
    if any(k in name for k in ("fed", "ecb", "boe", "boj")):
        s += 2
    if e["country"] in ("United States", "China", "Hong Kong"):
        s += 2
    elif e["country"] in ("Euro Zone", "Japan"):
        s += 1
    a, c = num(e["actual"]), num(e["consensus"])
    if a is not None and c is not None:
        s += min(abs(a - c) / (abs(c) + 1e-9) * 2, 3)
    return s


def brief_events(events: list[dict], max_n: int = 8) -> list[dict]:
    ranked = sorted(events, key=lambda e: (-score_event(e), e["hkt"]))
    seen, out = set(), []
    for e in ranked:
        name = e["event"].lower()
        group = e["event"]
        if "cpi" in name:
            group = "CPI"
        elif "nonfarm payrolls" in name:
            group = "Nonfarm Payrolls"
        elif "unemployment rate" in name:
            group = "Unemployment Rate"
        elif "retail sales" in name:
            group = "Retail Sales"
        key = (e["country"], group)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= max_n:
            break
    return sorted(out, key=lambda e: e["hkt"])


def render_brief(events: list[dict], days: int) -> str:
    lines = [
        "# 每日关键事件",
        "",
        f"- 生成时间：{datetime.now(HKT):%Y-%m-%d %H:%M} HKT",
        f"- 覆盖：近 1 日 + 未来 {days} 天；只列 5–10 条关键事件",
        "- 时间口径：源为 Nasdaq 日历，已按 EST→HKT 转换；发布日请与官方日历复核。",
        "",
        "| # | HKT | 地区 | 事件 | 预期 | 实际 | 为什么重要 |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for i, e in enumerate(events, 1):
        lines.append(f"| {i} | {e['hkt']} | {e['country']} | {e['event']} | "
                     f"{e['consensus'] or '—'} | {e['actual'] or '—'} | {e['why']} |")
    if not events:
        lines.append("| — | — | — | 无符合条件的已排期事件 | — | — | — |")
    lines += ["", "> 只做事件清单；不判断方向、不构成交易建议。"]
    return "\n".join(lines) + "\n"


def render(events: list[dict], rows: list[dict], flags: list[str], fed: list[dict], days: int) -> str:
    def pct(x):
        return "—" if x is None else f"{x * 100:+.1f}%"

    lines = [
        "# 每日事件雷达",
        "",
        f"- 生成时间：{datetime.now(HKT):%Y-%m-%d %H:%M} HKT",
        f"- 覆盖：未来 {days} 天；地区：US / China / HK / Euro Zone / Japan；只列关键事件",
        "",
        "## 1. 市场快照",
        "| 资产 | 日期 | 最新 | 1日 | 5日 | 含义 |",
        "|---|---|---:|---:|---:|---|",
    ]
    meaning = {
        "S&P 500": "美股大盘；单日 ≤−2% 触发风险标记",
        "Nasdaq": "科技成长；单日 ≤−3% 触发风险标记",
        "VIX": "恐慌指数；单日 +20% 触发风险标记",
        "US 10Y": "全球估值锚；5日 +15bp 触发利率上行标记",
        "DXY": "美元强弱；影响外资与人民币",
        "恒生指数": "港股大盘",
    }
    for r in rows:
        lines.append(f"| {r['asset']} | {r['date']} | {r['last']:.2f} | {pct(r['d1'])} | {pct(r['d5'])} | {meaning.get(r['asset'], '')} |")
    lines += ["", "## 2. 风险标记", "| 标记 | 当前 | 解释 |", "|---|---|---|"]
    defs = {
        "US_EQUITY_DROP": "S&P 500 单日 ≤ −2%",
        "NASDAQ_DROP": "Nasdaq 单日 ≤ −3%",
        "VIX_SPIKE": "VIX 单日 ≥ +20%",
        "US10Y_UP": "美债 10Y 5日 ≥ +15bp",
        "US_EQUITY_5D_DROP": "S&P 500 5日 ≤ −3%",
        "NASDAQ_5D_DROP": "Nasdaq 5日 ≤ −3%",
        "VIX_5D_SPIKE": "VIX 5日 ≥ +30%",
    }
    for k, v in defs.items():
        lines.append(f"| `{k}` | {'**是**' if k in flags else '否'} | {v} |")

    lines += ["", "## 3. 事件雷达", "| HKT | 地区 | 事件 | 预期 | 实际 | 超预期 | 为什么重要 |",
              "|---|---|---|---:|---:|---:|---|"]
    if events:
        for e in events:
            lines.append(f"| {e['hkt']} | {e['country']} | {e['event']} | {e['consensus'] or '—'} | "
                         f"{e['actual'] or '—'} | {e['surprise']} | {e['why']} |")
    else:
        lines.append("| — | — | 无符合条件的已排期事件 | — | — | — | — |")

    lines += ["", "## 4. Fed 最新", "| 日期 | 标题 |", "|---|---|"]
    for x in fed:
        lines.append(f"| {x['date']} | {x['title']} |")
    if not fed:
        lines.append("| — | 未取到 RSS |")

    lines += ["", "## 5. 数据源与限制",
              "- Nasdaq 经济日历：有 consensus/actual，但无重要性字段；本脚本用关键词筛选。",
              "- Fed RSS：官方新闻，非交易信号。",
              "- yfinance：行情延迟/复权口径有限；仅用于状态判断。",
              "- 事件时间已转 HKT；跨时区事件需注意港股是否开市。",
              "- 本报告只做信息汇总，不构成买卖建议。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="每日事件雷达")
    ap.add_argument("--days", type=int, default=3, help="覆盖未来天数")
    ap.add_argument("--brief", action="store_true", help="只输出 5–10 条关键事件")
    ap.add_argument("--max", type=int, default=8, help="brief 模式最多条数")
    ap.add_argument("--out", help="输出 md 路径")
    args = ap.parse_args()

    today = date.today()
    events = []
    for i in range(-1, args.days + 1):
        try:
            events += calendar(today + timedelta(days=i))
        except Exception as e:
            print(f"⚠️ {today + timedelta(days=i)} 日历失败: {e}")
    seen, uniq = set(), []
    for e in events:
        k = (e["hkt"], e["country"], e["event"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    events = sorted(uniq, key=lambda x: x["hkt"])
    if args.brief:
        events = brief_events(events, args.max)
        text = render_brief(events, args.days)
        OUT.mkdir(parents=True, exist_ok=True)
        md = Path(args.out) if args.out else OUT / f"brief_{today.isoformat()}.md"
        js = md.with_suffix(".json")
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(text, encoding="utf-8")
        js.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(text)
        print(f"💾 {md}\n💾 {js}")
        return 0

    rows, flags = market()
    fed = fed_news()
    text = render(events, rows, flags, fed, args.days)

    OUT.mkdir(parents=True, exist_ok=True)
    md = Path(args.out) if args.out else OUT / f"{today.isoformat()}.md"
    js = md.with_suffix(".json")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps({"events": events, "market": rows, "flags": flags, "fed": fed},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(f"💾 {md}\n💾 {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
