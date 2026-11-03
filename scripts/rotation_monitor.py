#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金融 vs 科技轮动监控。

输出：output/rotation_monitor/YYYY-MM-DD.md + .json
用法：
  python3 scripts/rotation_monitor.py
  python3 scripts/rotation_monitor.py --out output/rotation_monitor/latest.md
  python3 scripts/rotation_monitor.py --alert

口径：
- 金融篮子：HSBC / 工行 / 建行 / 招行 / 平安 / 友邦 / 中行 / 中银香港 / 港交所
- 科技篮子：腾讯 / 阿里 / 小米 / 美团 / 网易 / 中芯 / 京东 / 百度 / 快手 / 联想
- 等权、日频、前复权；相对强弱 = 金融累计净值 / 科技累计净值
- 本脚本只做状态监控，不是交易信号。
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "hk_broad" / "snapshot"
OUT = ROOT / "output" / "rotation_monitor"
START = pd.Timestamp("2020-01-02")

FIN = ["00005", "01398", "00939", "03968", "02318", "01299", "03988", "02388", "00388"]
TECH = ["00700", "09988", "01810", "03690", "09999", "00981", "09618", "09888", "01024", "00992"]

THRESHOLDS = {
    "fin_crowded_rel_1y": 0.50,
    "fin_crowded_pct": 90,
    "tech_weak_rel_1y": 0.30,
    "corr_low": 0.45,
    "rate_up_bp": 30,
    "rate_down_bp": -30,
    "cny_weak": 0.01,
    "cny_strong": -0.01,
    "southbound_out_bn": -20,
    "southbound_in_bn": 20,
}


def load_px(code: str) -> pd.Series:
    p = SNAP / f"{code}.pkl"
    if not p.exists():
        raise FileNotFoundError(p)
    x = pd.read_pickle(p)["Close"].dropna()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    return x.sort_index()


def basket_returns(codes: list[str]) -> pd.Series:
    px = pd.DataFrame({c: load_px(c) for c in codes}).sort_index()
    px = px[px.index >= START]
    ret = px.pct_change()
    return ret.mean(axis=1, skipna=True).dropna()


def relative_metrics() -> dict:
    fr = basket_returns(FIN)
    tr = basket_returns(TECH)
    df = pd.DataFrame({"fin": fr, "tech": tr}).dropna()
    cum = (1 + df).cumprod()
    rel = cum["fin"] / cum["tech"]
    pct = float((rel <= rel.iloc[-1]).mean() * 100)
    corr = float(df["fin"].rolling(60).corr(df["tech"]).iloc[-1])
    out = {
        "asof": str(df.index[-1].date()),
        "n_days": int(len(df)),
        "fin_total": float(cum["fin"].iloc[-1] - 1),
        "tech_total": float(cum["tech"].iloc[-1] - 1),
        "fin_ann": float(cum["fin"].iloc[-1] ** (252 / len(df)) - 1),
        "tech_ann": float(cum["tech"].iloc[-1] ** (252 / len(df)) - 1),
        "corr_60d": corr,
        "rel_pct": pct,
        "fin_5d": float((1 + df["fin"].tail(5)).prod() - 1),
        "tech_5d": float((1 + df["tech"].tail(5)).prod() - 1),
    }
    for d in (5, 20, 60, 120, 252):
        out[f"rel_{d}d"] = float(rel.iloc[-1] / rel.iloc[-1 - d] - 1)
    return out


def macro_metrics() -> dict:
    out = {"ust_asof": None, "ust_10y": None, "ust_20d_bp": None, "ust_5d_bp": None,
           "cny_asof": None, "usdcny": None, "usdcny_20d": None, "usdcny_5d": None,
           "hibor_asof": None, "hibor_3m": None, "hibor_20d_bp": None, "hibor_5d_bp": None,
           "sb_asof": None, "sb_20d_bn": None, "sb_5d_bn": None}

    p = ROOT / "data" / "us_market_snapshot" / "treasury_yield.pkl"
    if p.exists():
        d = pd.read_pickle(p)["data"]
        s = d["US_10Y_Yield"].dropna()
        if len(s) > 20:
            out.update(ust_asof=str(s.index[-1].date()), ust_10y=float(s.iloc[-1] * 100),
                       ust_20d_bp=float((s.iloc[-1] - s.iloc[-21]) * 10000),
                       ust_5d_bp=float((s.iloc[-1] - s.iloc[-6]) * 10000))

    p = ROOT / "data" / "cnh_hibor" / "usdcny.csv"
    if p.exists():
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["usdcny"].dropna()
        if len(s) > 20:
            out.update(cny_asof=str(s.index[-1].date()), usdcny=float(s.iloc[-1]),
                       usdcny_20d=float(s.iloc[-1] / s.iloc[-21] - 1),
                       usdcny_5d=float(s.iloc[-1] / s.iloc[-6] - 1))

    p = ROOT / "data" / "cnh_hibor" / "hibor_hkd.csv"
    if p.exists():
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["hibor_3m"].dropna()
        if len(s) > 20:
            out.update(hibor_asof=str(s.index[-1].date()), hibor_3m=float(s.iloc[-1]),
                       hibor_20d_bp=float((s.iloc[-1] - s.iloc[-21]) * 100),
                       hibor_5d_bp=float((s.iloc[-1] - s.iloc[-6]) * 100))

    p = ROOT / "data" / "southbound_holdings" / "southbound_long.csv"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["date"])
        sh = d.pivot_table(index="date", columns="pool_code", values="hold_shares").sort_index()
        px = d.pivot_table(index="date", columns="pool_code", values="close").sort_index()
        net = (sh.diff() * px).sum(axis=1, min_count=1) / 1e9
        if len(net.dropna()) >= 20:
            n = net.dropna()
            out.update(sb_asof=str(n.index[-1].date()), sb_20d_bn=float(n.tail(20).sum()),
                       sb_5d_bn=float(n.tail(5).sum()))

    return out


def flags(r: dict, m: dict) -> list[str]:
    f = []
    if r["rel_252d"] >= THRESHOLDS["fin_crowded_rel_1y"] and r["rel_pct"] >= THRESHOLDS["fin_crowded_pct"]:
        f.append("FIN_CROWDED")
    if r["rel_252d"] >= THRESHOLDS["tech_weak_rel_1y"]:
        f.append("TECH_WEAK")
    if r["corr_60d"] < THRESHOLDS["corr_low"]:
        f.append("CORR_LOW")
    if m["ust_20d_bp"] is not None and m["ust_20d_bp"] >= THRESHOLDS["rate_up_bp"]:
        f.append("RATE_UP")
    if m["ust_20d_bp"] is not None and m["ust_20d_bp"] <= THRESHOLDS["rate_down_bp"]:
        f.append("RATE_DOWN")
    if m["usdcny_20d"] is not None and m["usdcny_20d"] >= THRESHOLDS["cny_weak"]:
        f.append("CNY_WEAK")
    if m["usdcny_20d"] is not None and m["usdcny_20d"] <= THRESHOLDS["cny_strong"]:
        f.append("CNY_STRONG")
    if m["sb_20d_bn"] is not None and m["sb_20d_bn"] <= THRESHOLDS["southbound_out_bn"]:
        f.append("SOUTHBOUND_OUT")
    if m["sb_20d_bn"] is not None and m["sb_20d_bn"] >= THRESHOLDS["southbound_in_bn"]:
        f.append("SOUTHBOUND_IN")
    return f


def render(r: dict, m: dict, f: list[str]) -> str:
    def pct(x):
        return "—" if x is None or x != x else f"{x * 100:+.1f}%"

    def num(x, d=2):
        return "—" if x is None or x != x else f"{x:.{d}f}"

    def rel_read(x):
        if x is None or x != x:
            return "数据不足"
        if x >= 0.30:
            return "金融明显跑赢"
        if x >= 0.05:
            return "金融小幅跑赢"
        if x <= -0.30:
            return "科技明显跑赢"
        if x <= -0.05:
            return "科技小幅跑赢"
        return "大致持平"

    def cny_read(x):
        if x is None or x != x:
            return "数据不足"
        if x <= -0.01:
            return "人民币偏强，利好外资/科技"
        if x >= 0.01:
            return "人民币偏弱，压制外资/科技"
        return "人民币大致稳定"

    flag_defs = [
        ("FIN_CROWDED", "1y 相对 ≥ +50% 且分位 ≥90", "金融拥挤，赔率下降，不等于立即反转"),
        ("TECH_WEAK", "1y 相对 ≥ +30%", "科技相对弱，等盈利/利率/监管催化"),
        ("CORR_LOW", "60d 相关 <0.45", "分散度上升，轮动迹象增强"),
        ("RATE_UP", "美债 10Y 20d ≥ +30bp", "科技估值受压，银行 NIM 相对受益"),
        ("RATE_DOWN", "美债 10Y 20d ≤ −30bp", "科技估值减压，银行 NIM 相对受压"),
        ("CNY_WEAK", "USD/CNY 20d ≥ +1%", "人民币偏弱，外资/科技风险偏好受压"),
        ("CNY_STRONG", "USD/CNY 20d ≤ −1%", "人民币偏强，利好外资/科技"),
        ("SOUTHBOUND_OUT", "南向 20d ≤ −20bn", "南向资金流出，港股承接力下降"),
        ("SOUTHBOUND_IN", "南向 20d ≥ +20bn", "南向资金流入，港股承接力上升"),
    ]

    lines = [
        "# 金融 vs 科技轮动监控",
        "",
        f"- 截至：{r['asof']}",
        "- 口径：金融/科技等权自建篮子；相对强弱=金融累计净值÷科技累计净值；非官方行业指数。",
        "",
        "## 0. 本周变化（5 个交易日）",
        "| 指标 | 数值 | 含义 | 当前读法 |",
        "|---|---:|---|---|",
        f"| 金融本周 | {pct(r['fin_5d'])} | 金融篮子近 5 日涨跌 | {'下跌' if r['fin_5d'] < 0 else '上涨'} |",
        f"| 科技本周 | {pct(r['tech_5d'])} | 科技篮子近 5 日涨跌 | {'下跌' if r['tech_5d'] < 0 else '上涨'} |",
        f"| 本周相对 | {pct(r['rel_5d'])} | 金融相对科技的本周超额 | {rel_read(r['rel_5d'])} |",
        "",
        "## 1. 相对强弱",
        "| 指标 | 数值 | 含义 | 当前读法 |",
        "|---|---:|---|---|",
        f"| 金融篮子累计 | {pct(r['fin_total'])} | 2020 以来等权持有金融股的累计涨跌 | "
        f"{'金融长期占优' if r['fin_total'] > r['tech_total'] else '科技长期占优'} |",
        f"| 科技篮子累计 | {pct(r['tech_total'])} | 2020 以来等权持有科技股的累计涨跌 | "
        f"{'科技落后' if r['tech_total'] < r['fin_total'] else '科技领先'} |",
        f"| 金融年化 | {pct(r['fin_ann'])} | 累计收益折算成年化 | 长期复利速度 |",
        f"| 科技年化 | {pct(r['tech_ann'])} | 累计收益折算成年化 | 长期复利速度 |",
        f"| 20d 相对 | {pct(r['rel_20d'])} | 近 20 个交易日金融相对科技的超额 | {rel_read(r['rel_20d'])} |",
        f"| 60d 相对 | {pct(r['rel_60d'])} | 近 60 个交易日金融相对科技的超额 | {rel_read(r['rel_60d'])} |",
        f"| 120d 相对 | {pct(r['rel_120d'])} | 近 120 个交易日金融相对科技的超额 | {rel_read(r['rel_120d'])} |",
        f"| 1y 相对 | {pct(r['rel_252d'])} | 近 252 个交易日金融相对科技的超额 | {rel_read(r['rel_252d'])} |",
        f"| 相对强弱分位 | {r['rel_pct']:.0f}% | 当前相对强弱在 2020 以来历史中的位置 | "
        f"{'极值高位' if r['rel_pct'] >= 90 else ('极值低位' if r['rel_pct'] <= 10 else '中性区间')} |",
        f"| 60d 相关性 | {num(r['corr_60d'], 2)} | 金融与科技日收益的同步程度；越低越分散 | "
        f"{'低相关，轮动特征' if r['corr_60d'] < 0.45 else ('中等相关' if r['corr_60d'] < 0.65 else '高相关，同涨同跌')} |",
        "",
        "## 2. 宏观与资金",
        "| 指标 | 数值 | 含义 | 当前读法 |",
        "|---|---:|---|---|",
        f"| 美债 10Y | {num(m['ust_10y'])}% | 全球无风险利率锚；上行压科技估值 | "
        f"本周 {num(m['ust_5d_bp'], 1)}bp，20d {num(m['ust_20d_bp'], 1)}bp，{'利率上行' if (m['ust_20d_bp'] or 0) > 0 else '利率下行'} |",
        f"| USD/CNY | {num(m['usdcny'], 3)} | 人民币汇率；数值升=人民币贬 | "
        f"本周 {pct(m['usdcny_5d'])}，20d {pct(m['usdcny_20d'])}，{cny_read(m['usdcny_20d'])} |",
        f"| HIBOR 3M | {num(m['hibor_3m'])}% | 香港短期资金成本；升=流动性收紧 | "
        f"本周 {num(m['hibor_5d_bp'], 1)}bp，20d {num(m['hibor_20d_bp'], 1)}bp，{'成本上升' if (m['hibor_20d_bp'] or 0) > 0 else '成本下降'} |",
        f"| 南向 20d | {num(m['sb_20d_bn'], 1)} HK$bn | 南向资金估算净买入；正=流入 | "
        f"本周 {num(m['sb_5d_bn'], 1)}bn，20d {num(m['sb_20d_bn'], 1)}bn，"
        f"{'流入' if (m['sb_20d_bn'] or 0) > 0 else '流出'}，{'支撑港股' if (m['sb_20d_bn'] or 0) > 0 else '承接力偏弱'} |",
        "",
        "## 3. 状态标记",
        "| 标记 | 触发条件 | 当前是否触发 | 解释 |",
        "|---|---|---|---|",
    ]
    for key, cond, exp in flag_defs:
        lines.append(f"| `{key}` | {cond} | {'**是**' if key in f else '否'} | {exp} |")

    if "FIN_CROWDED" in f and "RATE_UP" in f:
        conclusion = ("金融相对拥挤且美债利率上行：这是“金融仍强、科技继续受压”的组合；"
                      "轮动到科技需要利率见顶或科技盈利超预期，当前未看到确认。")
    elif "FIN_CROWDED" in f:
        conclusion = "金融相对已到高位，赔率下降；可能收敛，但不等于立即反转。"
    elif "TECH_WEAK" in f:
        conclusion = "科技相对明显落后；等盈利/利率/监管催化，不接无催化下跌。"
    elif "CORR_LOW" in f:
        conclusion = "相关性下降，轮动特征增强；但方向未确认。"
    else:
        conclusion = "无强标记，维持观察；不要仅凭单一指标切换。"
    lines += ["", "## 4. 结论（条件化）", conclusion, "",
              "> 本监控不是交易信号；金融/科技为等权自建篮子，未计成本、股息、成分变化。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="金融 vs 科技轮动监控")
    ap.add_argument("--out", help="输出 md 路径")
    ap.add_argument("--alert", action="store_true", help="有强标记时非零退出")
    ap.add_argument("--weekly", action="store_true", help="输出 weekly_YYYY-MM-DD.md")
    args = ap.parse_args()

    r = relative_metrics()
    m = macro_metrics()
    f = flags(r, m)
    text = render(r, m, f)

    OUT.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    md = Path(args.out) if args.out else OUT / (f"weekly_{day}.md" if args.weekly else f"{day}.md")
    js = md.with_suffix(".json")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps({"relative": r, "macro": m, "flags": f}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(f"💾 {md}\n💾 {js}")
    return 1 if args.alert and f else 0


if __name__ == "__main__":
    raise SystemExit(main())
