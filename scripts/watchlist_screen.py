#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自选 30 只基本面筛选报告（可复现）。数据：scripts/stock_fundamentals.load_stock + 事件表。

口径注意：akshare 港股的报表币种，内地企业=CNY；香港/英美系（0005/0012/0016/0388/2388/0981/1347/1299）的数据为 HKD。
价格用 data/hk_broad/snapshot（yfinance 最新收盘）。缺 NPL/CET1/分部/一致预期。
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from stock_fundamentals import load_stock  # noqa: E402
from config import STOCK_SECTOR_MAPPING as SM  # noqa: E402

GROUPS = {
    "银行": ["0005.HK", "0939.HK", "1288.HK", "1398.HK", "2388.HK", "3968.HK"],
    "保险": ["1299.HK", "2318.HK"],
    "地产": ["0012.HK", "0016.HK", "1109.HK"],
    "能源": ["0883.HK", "1088.HK", "1138.HK"],
    "电信": ["0728.HK", "0941.HK"],
    "互联网平台": ["0700.HK", "3690.HK", "9888.HK", "9988.HK"],
    "AI/半导体/机器人": ["0981.HK", "1347.HK", "2533.HK", "6682.HK", "9660.HK", "2513.HK"],
    "消费/汽车": ["1810.HK", "1211.HK"],
    "交易所": ["0388.HK"],
    "生物科技": ["2269.HK"],
}
HKD_CCY = {"0005.HK", "0012.HK", "0016.HK", "0388.HK", "2388.HK", "0981.HK", "1347.HK", "1299.HK"}
BANKS = set(GROUPS["银行"])


def fmt(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if pct:
        return f"{v*100:+.1f}%"
    return f"{v:.{nd}f}"


STRUCT_LEV = BANKS | {"1299.HK", "2318.HK", "0388.HK"}  # 银行/保险/交易所杠杆结构性，非红旗


def flags(r):
    f = []
    roe = r.get("ROE", np.nan); ocfni = r.get("OCF÷净利", np.nan)
    if r["code"] not in STRUCT_LEV and isinstance(ocfni, float) and np.isfinite(ocfni):
        if ocfni < 0: f.append("OCF为负")
        elif ocfni < 0.8 and isinstance(roe, float) and roe > 0: f.append("现金流弱(OCF/净利<0.8)")
    py = r.get("派息率", np.nan)
    if isinstance(py, float) and np.isfinite(py) and py > 1.0: f.append("派息>净利")
    nbb = r.get("净回购(亿)", np.nan)
    if isinstance(nbb, float) and np.isfinite(nbb) and nbb < -1: f.append("净增发(稀释)")
    pb = r.get("PB", np.nan); pe = r.get("PE", np.nan); dar = r.get("资产负债率", np.nan)
    if isinstance(pb, float) and np.isfinite(pb) and pb < 0: f.append("负净资产")
    if isinstance(pe, float) and np.isfinite(pe) and pe < 0: f.append("亏损")
    if isinstance(dar, float) and np.isfinite(dar) and dar > 0.85 and r["code"] not in STRUCT_LEV:
        f.append("高杠杆(>85%)")
    return "；".join(f) if f else ""


def main():
    rows = []
    for g, codes in GROUPS.items():
        for c in codes:
            ccy = "HKD" if c in HKD_CCY else "CNY"
            r = load_stock(c, ccy=ccy, bank=(c in BANKS))
            if not r:
                rows.append({"group": g, "code": c, "name": SM.get(c, {}).get("name", c),
                             "FAIL": True}); continue
            r["group"] = g; r["name"] = SM.get(c, {}).get("name", c); r["FAIL"] = False
            rows.append(r)
    df = pd.DataFrame(rows)
    # 事件
    ev = pd.read_csv(os.path.join(ROOT, "data/events/events_310.csv"))
    ev["announce_ts"] = pd.to_datetime(ev["announce_ts"])
    ev["code5"] = ev["code"].astype(str).str.replace(".HK", "").str.zfill(5)
    cut = ev["announce_ts"].max() - pd.Timedelta(days=365)
    ev = ev[ev["announce_ts"] >= cut]
    wl = {c.replace(".HK", "").zfill(5): c for g in GROUPS for c in GROUPS[g]}
    ev = ev[ev.code5.isin(wl)]
    evsum = {c: ev[ev.code5 == c.replace(".HK", "").zfill(5)] for c in wl.values()}

    md = ["# 自选 30 只：基本面筛选（2026-11）\n",
          "> 数据：akshare 三表（最新年报，多为 FY2025）+ yfinance 收盘（**截至 2026-11-13**）；事件表 `data/events/events_310.csv`。\n",
          "> ⚠️ 口径：内地企业报表 CNY；香港/英美系（0005/0012/0016/0388/2388/0981/1347/1299）数据为 HKD。"
          "银行股缺 NPL/CET1/地产敞口；全体缺分部、一致预期。**不是买卖建议**。\n"]

    def table(cols, subset, bank=False):
        hdr = ["代码", "名称", "PB", "PE", "ROE", "净利率", "营收增", "净利增", "OCF/净利",
               "股息率", "派息率", "净回购(亿)", "总回报", "红旗"]
        if bank:
            hdr = ["代码", "名称", "PB", "PE", "ROE", "股息率", "派息率", "NIM", "成本收入", "贷存比", "合理PB", "空间", "红旗"]
        lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for _, r in subset.iterrows():
            if r.get("FAIL"):
                lines.append(f"| {r['code']} | {r['name']} | " + " | ".join(["数据缺失"] * (len(hdr) - 2)) + " |")
                continue
            fl = flags(r)
            if bank:
                vals = [r["code"], r["name"], fmt(r.get("PB")), fmt(r.get("PE")), fmt(r.get("ROE"), True),
                        fmt(r.get("股息率"), True), fmt(r.get("派息率"), True), fmt(r.get("NIM"), True),
                        fmt(r.get("成本收入"), True), fmt(r.get("贷存比"), True), fmt(r.get("合理PB")),
                        fmt(r.get("空间"), True), fl]
            else:
                vals = [r["code"], r["name"], fmt(r.get("PB")), fmt(r.get("PE")), fmt(r.get("ROE"), True),
                        fmt(r.get("净利率"), True), fmt(r.get("营收增速"), True), fmt(r.get("净利增速"), True),
                        fmt(r.get("OCF÷净利")), fmt(r.get("股息率"), True), fmt(r.get("派息率"), True),
                        fmt(r.get("净回购(亿)"), nd=0), fmt(r.get("总股东回报"), True), fl]
            lines.append("| " + " | ".join(str(x) for x in vals) + " |")
        return "\n".join(lines)

    md.append("\n## 0. 一页分组观察（规则分组，非建议）\n")
    md.append("- **稳健现金牛 / 高股息**：内地大行（0939/1398/1288/3968，PB 0.6–1.0、股息 4–5%）、"
              "电信（0728/0941，股息 ~6.7%）、能源（0883/1088，股息 6–11%）、保险（2318 PB 0.84、股息 6.7%）。"
              "**便宜+高息，但要先查 NPL/CET1 与分红可持续性。**")
    md.append("- **高质量成长（但估值已不低）**：0700（ROE 19.5%、净利率 30%）、9988、1810、1211、2269、0388。"
              "看增长能否兑现、OCF/净利是否跟得上。")
    md.append("- **周期/地产**：0012/0016/1109（PB 0.5–0.7，但 ROE 1.8–8.8%、部分派息超净利）、1138。"
              "便宜是周期底的特征，也可能是价值陷阱。")
    md.append("- **未盈利/高估值/稀释**：0981/1347（PE 77/321）、2533/6682/9660/2513（亏损，且 6682/9660/1810/1211 净增发）、"
              "3690（转亏）、9888（利润 -76%、OCF 为负）。**高风险，需看订单/现金流/融资。**")
    md.append("- **纯股息型**：1088（股息 10.7%，派息率 157%，含特别息可能）。\n")

    for i, (g, codes) in enumerate(GROUPS.items(), 1):
        md.append(f"\n## {i}. {g}\n")
        sub = df[df.group == g]
        md.append(table(None, sub, bank=(g == "银行")))
        if g == "AI/半导体/机器人":
            md.append("\n> ⚠️ 2513 智谱：负净资产 / 资产负债率 267%，可能为累计亏损所致，**必须回年报核对**；"
                      "亏损公司 PE/PB 不具可比性，仅看营收与现金流。")
        if g == "银行":
            md.append("\n> 合理PB=(ROE−g)/(COE−g)，COE 11%；「空间」>0 表示 PB 低于简化模型隐含值（**粗略**，非目标价）。")
        nums = sub[~sub.get("FAIL", False)] if "FAIL" in sub else sub
        if len(nums):
            def med(col, pct=False):
                v = nums[col].dropna() if col in nums else pd.Series(dtype=float)
                v = v[np.isfinite(v)]
                return f"{v.median()*100:+.1f}%" if pct and len(v) else (f"{v.median():.2f}" if len(v) else "—")
            md.append(f"\n> 板块中位：PB {med('PB')}、PE {med('PE')}、ROE {med('ROE',True)}、股息率 {med('股息率',True)}。")

    # 公司行动
    md.append("\n## 11. 近 12 个月公司行动（PIT 公告日）\n")
    md.append("| 代码 | 名称 | 事件 |")
    md.append("|---|---|---|")
    any_ev = False
    for c in [x for g in GROUPS for x in GROUPS[g]]:
        g = evsum.get(c)
        if g is None or len(g) == 0: continue
        any_ev = True
        types = "、".join(f"{k}×{v}" for k, v in g.event_type.value_counts().items())
        hi = [str(t)[:60] for t in g[g.confidence == "high"].title]
        md.append(f"| {c} | {SM.get(c,{}).get('name','')} | {types} |")
    if not any_ev:
        md.append("| — | — | 无 |")

    # 红旗汇总
    md.append("\n## 12. 红旗汇总（需人工核对）\n")
    fr = [(r["code"], r["name"], flags(r)) for _, r in df.iterrows() if not r.get("FAIL") and flags(r)]
    md.append("| 代码 | 名称 | 红旗 |"); md.append("|---|---|---|")
    for c, n, f in fr:
        md.append(f"| {c} | {n} | {f} |")

    md.append("\n## 13. 结论（条件化）\n")
    md.append("- 这是**筛选/尽调清单**，不是买卖建议；PE/PB 为 FY2025 年报口径 + 当前价。")
    md.append("- 便宜不等于好：低 PB 银行要先看 NPL/CET1；高 PE 成长股要看增长能否兑现（OCF/净利）。")
    md.append("- 红旗项（现金流弱/派息超净利/净增发/负净资产）必须回年报核对后再谈估值。")
    md.append("- 数据缺口：银行 NPL/关注类/拨备/CET1、地产土储与负债结构、科技分部与订单、一致预期——未评估。\n")
    out = os.path.join(ROOT, "docs", "WATCHLIST_30_SCREEN.md")
    open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("写入", out)
    print("失败:", df[df.get("FAIL") == True]["code"].tolist() if "FAIL" in df else [])


if __name__ == "__main__":
    sys.exit(main())
