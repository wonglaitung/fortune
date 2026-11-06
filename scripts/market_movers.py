#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大市值市场异动简报。

输出：output/event_radar/movers_YYYY-MM-DD.md
用法：
  python3 scripts/market_movers.py
  python3 scripts/market_movers.py --date 2026-11-05 --top 8
  python3 scripts/market_movers.py --date 2026-11-05 --min-move 0.04

只做“昨晚发生了什么”的行情+新闻摘要，不判断涨跌、不构成交易建议。
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "event_radar"
US = {
    "NVDA": "英伟达", "AMD": "AMD", "MU": "美光", "TSM": "台积电ADR", "ASML": "阿斯麦",
    "AVGO": "博通", "INTC": "英特尔", "SMCI": "超微电脑", "AAPL": "苹果", "MSFT": "微软",
    "GOOGL": "谷歌", "AMZN": "亚马逊", "META": "Meta", "TSLA": "特斯拉", "NFLX": "奈飞",
    "ORCL": "甲骨文", "CRM": "Salesforce", "PLTR": "Palantir", "QCOM": "高通", "ARM": "ARM",
    "BABA": "阿里ADR", "PDD": "拼多多", "JD": "京东", "BIDU": "百度ADR", "NIO": "蔚来",
    "XPEV": "小鹏ADR", "LI": "理想ADR",
}
HK = {
    "0700.HK": "腾讯", "9988.HK": "阿里巴巴", "3690.HK": "美团", "1810.HK": "小米",
    "1211.HK": "比亚迪", "0939.HK": "建设银行", "1398.HK": "工商银行",
    "0005.HK": "汇丰", "1299.HK": "友邦", "0388.HK": "港交所",
}
INDEX = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX", "^TNX": "US 10Y", "^HSI": "恒生指数"}


def clean(s):
    s = re.sub(r"<[^>]+>", "", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def download(tickers, start):
    import yfinance as yf
    d = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    return d


def row_for(ticker, d, day):
    try:
        x = d[ticker]["Close"].dropna()
        x = x[x.index <= pd.Timestamp(day)]
        if len(x) < 7:
            return None
        dt = x.index[-1]
        prev = x.index[-2]
        r1 = float(x.iloc[-1] / x.iloc[-2] - 1)
        r5 = float(x.iloc[-1] / x.iloc[-6] - 1)
        vol = d[ticker]["Volume"].reindex(x.index).iloc[-1]
        turnover = float(x.iloc[-1] * (0 if pd.isna(vol) else vol))
        return {"date": str(dt.date()), "ret1": r1, "ret5": r5, "turnover": turnover}
    except Exception:
        return None


def news(name, ticker):
    q = f"{name} stock"
    try:
        r = requests.get("https://news.google.com/rss/search", params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                         timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        titles = [clean(i.findtext("title")) for i in root.findall(".//item")[:3]]
        return "；".join(titles[:2]) if titles else "—"
    except Exception:
        return "—"


def render(day, idx, rows, max_n):
    lines = [
        "# 市场异动简报",
        "",
        f"- 交易日：{day}",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "- 只列大市值/高成交额异动；新闻标题仅作线索，不构成因果结论。",
        "",
        "## 指数",
        "| 指数 | 1日 | 5日 |",
        "|---|---:|---:|",
    ]
    for x in idx:
        lines.append(f"| {x['name']} | {x['ret1']*100:+.1f}% | {x['ret5']*100:+.1f}% |")
    lines += ["", "## 大市值异动", "| 股票 | 1日 | 5日 | 成交额 | 影响* | 新闻标题（线索） |", "|---|---:|---:|---:|---:|---|"]
    for x in rows[:max_n]:
        lines.append(f"| {x['name']}（{x['ticker']}） | {x['ret1']*100:+.1f}% | {x['ret5']*100:+.1f}% | "
                     f"{x['turnover']/1e9:.1f}bn | {abs(x['ret1'])*x['turnover']/1e9:.1f}bn | {x['news']} |")
    lines += ["", "* 影响 = |1日涨跌| × 成交额，仅用于排序，不是市值损失。", "",
              "> 行情来自 yfinance；新闻来自 Google News RSS。只做信息汇总，不构成交易建议。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="大市值市场异动简报")
    ap.add_argument("--date", help="交易日 YYYY-MM-DD；默认最新")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-move", type=float, default=0.04)
    ap.add_argument("--out")
    args = ap.parse_args()

    today = date.today()
    start = (today - timedelta(days=40)).isoformat()
    d = download(list(US) + list(HK) + list(INDEX), start)
    idx = []
    for t, n in INDEX.items():
        r = row_for(t, d, args.date or today.isoformat())
        if r:
            idx.append({"name": n, **r})

    rows = []
    for t, n in {**US, **HK}.items():
        r = row_for(t, d, args.date or today.isoformat())
        if r and abs(r["ret1"]) >= args.min_move:
            rows.append({"ticker": t, "name": n, "news": "—", **r})
    rows.sort(key=lambda x: abs(x["ret1"]) * x["turnover"], reverse=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(news, x["name"], x["ticker"]): x for x in rows[:args.top]}
        for f in as_completed(futs):
            futs[f]["news"] = f.result()

    day = rows[0]["date"] if rows else (idx[0]["date"] if idx else (args.date or today.isoformat()))
    text = render(day, idx, rows, args.top)

    OUT.mkdir(parents=True, exist_ok=True)
    md = Path(args.out) if args.out else OUT / f"movers_{day}.md"
    js = md.with_suffix(".json")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps({"date": day, "index": idx, "movers": rows[:args.top]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(f"💾 {md}\n💾 {js}")


if __name__ == "__main__":
    main()
