#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日关键宏观事件雷达。

数据源：
- Nasdaq 经济日历 API：事件、美东时间、预期/实际/前值
- Federal Reserve RSS：最新官方新闻
- yfinance：S&P500 / Nasdaq / VIX / 美债10Y / 美元指数 / 恒指

用法：
  python3 scripts/event_radar.py --brief --days 5
  python3 scripts/event_radar.py --days 5
  python3 scripts/event_radar.py --out output/event_radar/today.md

只做信息汇总与风险提示，不构成交易建议。
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "event_radar"
CAL_URL = "https://api.nasdaq.com/api/calendar/economicevents"
FED_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
HKT = ZoneInfo("Asia/Hong_Kong")
NY = ZoneInfo("America/New_York")
TRACK = {"United States", "China", "Hong Kong", "Euro Zone", "Japan"}
KEY = [
    "FOMC", "Fed", "Interest Rate", "CPI", "PCE", "Nonfarm", "Unemployment",
    "GDP", "PMI", "Retail Sales", "Treasury", "Trade Balance", "Industrial Production",
    "Industrial Profit", "Exports", "Imports",
]
DROP = [
    "gdpnow", "balance sheet", "reserve balances", "private nonfarm", "u6 unemployment",
    "crude oil", "eia", "natural gas storage", "gasoline", "forecast 1m", "forecast 2m",
]
FAM_SCORE = {
    "Rate Decision": 6.0, "CPI": 5.0, "Core PCE": 5.0, "Nonfarm Payrolls": 5.0,
    "GDP": 4.0, "Unemployment Rate": 3.0, "PMI": 3.0, "Retail Sales": 2.0,
    "Industrial Profits": 1.5, "Industrial Production": 1.5, "Trade": 1.0,
}
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


def fetch_rows(day: date) -> list[dict]:
    try:
        r = requests.get(CAL_URL, params={"date": day.isoformat()}, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        r.raise_for_status()
        data = r.json().get("data") or {}
        return data.get("rows") or []
    except Exception:
        return []


def event_time(day: date, gmt: str):
    try:
        hh, mm = [int(v) for v in gmt.split(":")[:2]]
        return datetime.combine(day, time(hh, mm), tzinfo=NY).astimezone(HKT)
    except Exception:
        return None


def calendar(day: date) -> list[dict]:
    out = []
    for x in fetch_rows(day):
        country, event, gmt = clean(x.get("country")), clean(x.get("eventName")), clean(x.get("gmt"))
        if country not in TRACK or not event or not gmt or gmt.upper() in ("24H", "ALL DAY"):
            continue
        if not any(k.lower() in event.lower() for k in KEY):
            continue
        low = event.lower()
        if any(k in low for k in DROP):
            continue
        if "speaks" in low and not any(k in low for k in ("powell", "chair", "vice chair")):
            continue
        dt = event_time(day, gmt)
        if dt is None:
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


def family(event: str):
    e = event.lower()
    if "core pce" in e:
        return "Core PCE"
    if "pce" in e:
        return "PCE"
    if "gdp price" in e or "gdp sales" in e:
        return None
    if "gdp" in e:
        return "GDP"
    if "cpi" in e:
        return "CPI"
    if "nonfarm" in e:
        return "Nonfarm Payrolls"
    if "unemployment rate" in e:
        return "Unemployment Rate"
    if "retail sales" in e:
        return "Retail Sales"
    if "pmi" in e:
        return "PMI"
    if "industrial profit" in e:
        return "Industrial Profits"
    if "industrial production" in e:
        return "Industrial Production"
    if e in ("exports", "imports", "trade balance"):
        return "Trade"
    if any(k in e for k in ("interest rate", "fomc", "fed funds")):
        return "Rate Decision"
    return "Other"


def importance(fam: str, country: str, actual, consensus) -> float:
    s = FAM_SCORE.get(fam, 0.5)
    s += 2 if country in ("United States", "China", "Hong Kong") else 1
    a, c = num(actual), num(consensus)
    if a is not None and c is not None:
        s += min(abs(a - c) / (abs(c) + 1e-9) * 2, 3)
    return s


def event_why(fam: str, country: str) -> str:
    if fam == "Rate Decision":
        return "利率路径 → 折现率/银行NIM"
    if fam in ("CPI", "Core PCE", "PCE"):
        return {
            "United States": "美国通胀 → 联储路径 → 美债/美元/港股",
            "China": "中国通胀 → 政策空间",
            "Japan": "日本通胀 → 日元/套息交易",
            "Euro Zone": "欧元区通胀 → ECB 路径",
            "Hong Kong": "香港通胀 → 本地利率",
        }.get(country, "通胀 → 利率路径")
    if fam in ("Nonfarm Payrolls", "Unemployment Rate"):
        return ("美国就业 → 联储路径 → 美债/美元" if country == "United States"
                else f"{country}就业 → 本地消费/政策")
    if fam == "GDP":
        return "增长 → 盈利/风险偏好"
    if fam == "PMI":
        return {
            "China": "中国景气 → 周期/中资板块",
            "United States": "美国景气 → 周期/估值",
            "Japan": "日本景气 → 日元/出口链",
            "Euro Zone": "欧元区景气 → ECB 路径",
            "Hong Kong": "香港景气 → 本地股",
        }.get(country, "景气 → 周期")
    if fam == "Retail Sales":
        return "消费强弱 → 盈利/利率"
    if fam == "Industrial Profits":
        return "中国工业盈利 → 中资周期"
    if fam == "Trade":
        return "贸易/外需 → 港股出口链"
    if fam == "Industrial Production":
        return "工业生产 → 周期/出口链"
    return "宏观信息 → 风险偏好"


def collect_macro_events(days: int = 5, include_released: bool = True, now=None) -> list[dict]:
    today = date.today()
    now = now or datetime.now(HKT)
    dates = [today + timedelta(days=i) for i in range(-1, days + 1)]
    end = today + timedelta(days=days)
    with ThreadPoolExecutor(max_workers=4) as ex:
        batches = list(ex.map(fetch_rows, dates))
    cand = []
    for day, rows in zip(dates, batches):
        for x in rows:
            c, ev, gmt = clean(x.get("country")), clean(x.get("eventName")), clean(x.get("gmt"))
            if c not in TRACK or not ev or not gmt or gmt.upper() in ("24H", "ALL DAY"):
                continue
            if not any(k.lower() in ev.lower() for k in KEY):
                continue
            low = ev.lower()
            if any(k in low for k in DROP):
                continue
            if "speaks" in low and not any(k in low for k in ("powell", "chair", "vice chair")):
                continue
            fam = family(ev)
            if fam in (None, "Other"):
                continue
            dt = event_time(day, gmt)
            if dt is None or dt.date() > end:
                continue
            released = dt <= now
            actual = clean(x.get("actual")) if (released and include_released) else ""
            cand.append({
                "dt": dt, "hkt": dt.strftime("%m-%d %H:%M"), "country": c, "event": ev,
                "fam": fam, "consensus": clean(x.get("consensus")), "previous": clean(x.get("previous")),
                "actual": actual, "score": importance(fam, c, actual if released else None, x.get("consensus")),
                "why": event_why(fam, c),
            })
    best = {}
    for e in cand:
        k = (e["country"], e["fam"], e["dt"])
        if k not in best or e["score"] > best[k]["score"]:
            best[k] = e
    items = list(best.values())
    core = {(e["country"], e["dt"]) for e in items if e["fam"] == "Core PCE"}
    items = [e for e in items if not (e["fam"] == "PCE" and (e["country"], e["dt"]) in core)]
    items.sort(key=lambda e: (-e["score"], e["dt"]))
    return items


def render_brief(events: list[dict], days: int, max_n: int = 10) -> str:
    lines = [
        "# 每日关键事件",
        "",
        f"- 生成时间：{datetime.now(HKT):%Y-%m-%d %H:%M} HKT",
        f"- 覆盖：近 1 日 + 未来 {days} 天；按重要性排序，最多 {max_n} 条",
        "- 时间口径：Nasdaq 日历美东时间已转 HKT；实际值仅在事件时间已过时显示；发布日请与官方日历复核。",
        "",
        "| # | HKT | 地区 | 事件 | 预期 | 前值 | 实际(超预期) | 分 | 为什么重要 |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for i, e in enumerate(events[:max_n], 1):
        act = f"{e['actual']} ({surprise(e['actual'], e['consensus'])})" if e.get("actual") else "—"
        lines.append(f"| {i} | {e['hkt']} | {e['country']} | {e['event']} | "
                     f"{e['consensus'] or '—'} | {e['previous'] or '—'} | {act} | "
                     f"{e['score']:.1f} | {e['why']} |")
    if not events:
        lines.append("| — | — | — | 无符合条件的已排期事件 | — | — | — | — | — |")
    lines += ["", "> 只做事件清单；不判断方向、不构成交易建议。"]
    return "\n".join(lines) + "\n"


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
              "- Nasdaq 经济日历：有 consensus/actual，但无重要性字段；本脚本用关键词与家族分级筛选。",
              "- Fed RSS：官方新闻，非交易信号。",
              "- yfinance：行情延迟/复权口径有限；仅用于状态判断。",
              "- 事件时间已按美东时区转 HKT；跨时区事件需注意港股是否开市。",
              "- 本报告只做信息汇总，不构成买卖建议。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="每日关键宏观事件雷达")
    ap.add_argument("--days", type=int, default=5, help="覆盖未来自然日数")
    ap.add_argument("--brief", action="store_true", help="只输出关键事件清单")
    ap.add_argument("--max", type=int, default=10, help="brief 模式最多条数")
    ap.add_argument("--out", help="输出 md 路径")
    args = ap.parse_args()

    today = date.today()
    if args.brief:
        events = collect_macro_events(days=args.days)[:args.max]
        text = render_brief(events, args.days, args.max)
        payload = {"events": events}
    else:
        raw = []
        for i in range(-1, args.days + 1):
            raw += calendar(today + timedelta(days=i))
        seen, events = set(), []
        for e in raw:
            k = (e["hkt"], e["country"], e["event"])
            if k not in seen:
                seen.add(k)
                events.append(e)
        events = sorted(events, key=lambda x: x["hkt"])
        rows, flags = market()
        fed = fed_news()
        text = render(events, rows, flags, fed, args.days)
        payload = {"events": events, "market": rows, "flags": flags, "fed": fed}

    OUT.mkdir(parents=True, exist_ok=True)
    md = Path(args.out) if args.out else OUT / (f"brief_{today.isoformat()}.md" if args.brief else f"{today.isoformat()}.md")
    js = md.with_suffix(".json")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(text)
    print(f"💾 {md}\n💾 {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
