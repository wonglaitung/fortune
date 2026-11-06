#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日重要新闻摘要（Google News RSS 聚合）。

用法：
  python3 scripts/news_digest.py
  python3 scripts/news_digest.py --top 8
  python3 scripts/news_digest.py --lang zh

输出：output/news_digest/YYYY-MM-DD.md + .json
只做新闻线索，不判断涨跌、不构成交易建议。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "news_digest"
HKT = timezone(timedelta(hours=8))

EN = [
    ("stock market when:1d", "美股大盘"),
    ("Federal Reserve when:1d", "美联储/利率"),
    ("Nvidia when:1d", "AI/芯片"),
    ("China stocks when:1d", "中国股市"),
    ("Hong Kong stocks when:1d", "港股"),
    ("oil prices when:1d", "原油/地缘"),
    ("Treasury yields when:1d", "美债/利率"),
    ("China economy when:1d", "中国经济"),
]
ZH = [
    ("港股 when:1d", "港股"),
    ("中国经济 when:1d", "中国经济"),
    ("美联储 when:1d", "美联储/利率"),
    ("英伟达 when:1d", "AI/芯片"),
]
GOOD = {"reuters", "bloomberg", "wall street journal", "wsj", "financial times", "ft",
        "cnbc", "associated press", "ap", "nikkei", "scmp", "south china morning post",
        "barron's", "forbes", "fortune", "the economist", "路透", "彭博", "财新", "华尔街见闻"}
BAD = {"yahoo finance", "marketscreener", "moomoo", "investing.com", "tradingview",
       "stock price", "quote", "history"}


def clean(s):
    s = re.sub(r"<[^>]+>", "", str(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch(q, zh=False):
    params = {"q": f"{q}", "hl": "zh-HK" if zh else "en-US", "gl": "HK" if zh else "US",
              "ceid": "HK:zh-Hant" if zh else "US:en"}
    r = requests.get("https://news.google.com/rss/search", params=params, timeout=25,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = []
    for it in root.findall(".//item"):
        title = clean(it.findtext("title"))
        src = clean(it.findtext("source"))
        if not title:
            continue
        dt = None
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate")).astimezone(HKT)
        except Exception:
            pass
        out.append({"title": title, "source": src, "time": dt, "link": it.findtext("link")})
    return out


def score(x, topic):
    t = x["title"].lower()
    s = 0.0
    src = x["source"].lower()
    if any(g in src for g in GOOD):
        s += 3
    if any(b in src or b in t for b in BAD):
        s -= 4
    if any(k in t for k in ("fed", "rate", "cpi", "inflation", "nvidia", "ai", "china",
                            "hong kong", "tariff", "oil", "treasury", "yield", "stimulus",
                            "美联储", "港股", "中国", "英伟达", "关税", "降息", "加息")):
        s += 2
    if x["time"]:
        age = (datetime.now(HKT) - x["time"]).total_seconds() / 3600
        if age <= 6:
            s += 2
        elif age <= 12:
            s += 1
        elif age > 30:
            s -= 2
    return s


def dedupe(items):
    out = []
    for x in items:
        words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", x["title"].lower()))
        if not words:
            continue
        dup = False
        for y in out:
            w2 = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", y["title"].lower()))
            if len(words & w2) / max(1, len(words | w2)) > 0.72:
                dup = True
                break
        if not dup:
            out.append(x)
    return out


def render(rows):
    lines = [
        "# 每日重要新闻",
        "",
        f"- 生成时间：{datetime.now(HKT):%Y-%m-%d %H:%M} HKT",
        "- 来源：Google News RSS 聚合；标题保留原意，链接指向原文。",
        "- 只列 5–10 条；按来源质量、时效、关键词排序，不判断涨跌。",
        "",
        "| # | 时间(HKT) | 主题 | 新闻标题 | 来源 |",
        "|---:|---|---|---|---|",
    ]
    for i, x in enumerate(rows, 1):
        tm = x["time"].strftime("%m-%d %H:%M") if x["time"] else "—"
        lines.append(f"| {i} | {tm} | {x['topic']} | [{x['title']}]({x['link']}) | {x['source'] or '—'} |")
    lines += ["", "> 新闻标题只是线索；同一事件可能有多篇报道，需点开原文核对。"]
    return "\n".join(lines) + "\n"


def collect(top: int = 8, lang: str = "en") -> list[dict]:
    queries = ZH if lang == "zh" else EN
    items = []
    for q, topic in queries:
        try:
            for x in fetch(q, zh=(lang == "zh"))[:4]:
                x["topic"] = topic
                x["score"] = score(x, topic)
                items.append(x)
        except Exception as e:
            print(f"⚠️ {q}: {e}")
    items = dedupe(items)
    picked, counts = [], {}
    for x in sorted(items, key=lambda z: z["score"], reverse=True):
        if counts.get(x["topic"], 0) >= 2:
            continue
        picked.append(x)
        counts[x["topic"]] = counts.get(x["topic"], 0) + 1
        if len(picked) >= top:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--out")
    args = ap.parse_args()

    picked = collect(args.top, args.lang)
    text = render(picked)
    OUT.mkdir(parents=True, exist_ok=True)
    md = Path(args.out) if args.out else OUT / f"{datetime.now(HKT):%Y-%m-%d}.md"
    js = md.with_suffix(".json")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps([{k: (v.isoformat() if k == "time" and v else v) for k, v in x.items()} for x in picked],
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(f"💾 {md}\n💾 {js}")


if __name__ == "__main__":
    main()
