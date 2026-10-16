#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险面板（risk panel）—— 把 D55 的多周期风险信号聚合成**邮件区块**。

**定位**：只报**"同步"（一起涨、一起跌 → 分散失效）**，**不含方向**（D55：大跌不可测）。
**出口**：仓位 / 杠杆提示（**非 alpha**；D35 §十二）。

⚠️ **波动轴（大波动 |r|，D56）已撤**：近 250 期 AUC ≤0.5（失效，见 risk_panel_monitor）⇒
   **不上面板**（保留在监控里作观察）。

档位阈值（**暂用**，正式上线须预注册 + 校准）：🟢 P<0.5 · 🟡 0.5≤P<0.75 · 🔴 P≥0.75

用法：python3 ml_services/risk_panel.py            # 打印区块 + 写 data/risk_panel_latest.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_services.risk_horizon_signals import (  # noqa: E402
    FEATURES, HORIZONS, build_dataset, load_returns)

H_LABEL = {1: "未来 1 天", 5: "未来 1 周", 10: "未来 2 周", 20: "未来 1 个月"}
# 同步轴（相关齐升）—— 全周期均通过判据（D55）
SYNC_DISPLAY = [1, 5, 10, 20]
TIER_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
TIER_WORD = {"green": "正常", "yellow": "留意", "red": "当心"}


def tier_of(p, yellow=0.5, red=0.75):
    if p is None or not np.isfinite(p):
        return None
    if p >= red:
        return "red"
    if p >= yellow:
        return "yellow"
    return "green"


def latest_panel(horizons=HORIZONS):
    """对每个 H 拟合（全史）并预测最后一行 → P(相关齐升)。返回 payload。"""
    from sklearn.linear_model import LogisticRegression

    rets = load_returns()
    df, _ = build_dataset(rets)
    as_of = str(df.index[-1].date())
    out = {}
    for H in horizons:
        col = f"corr_rise_{H}"
        d = df.dropna(subset=FEATURES + [col])
        p = None
        if len(d) > 300:
            X, y = d[FEATURES].values, d[col].values
            mu, sd = X.mean(axis=0), X.std(axis=0)
            sd[sd == 0] = 1.0
            clf = LogisticRegression(max_iter=1000)
            clf.fit((X - mu) / sd, y)
            xlast = ((df[FEATURES].iloc[-1].values - mu) / sd).reshape(1, -1)
            p = float(clf.predict_proba(xlast)[0, 1])
        out[H] = {"p_corr_rise": p, "tier": tier_of(p)}
    return {"as_of": as_of, "horizons": out,
            "tiers_note": "🟢<0.5 / 🟡0.5–0.75 / 🔴≥0.75（暂用，正式须预注册）",
            "axes": ["sync"]}


def risk_panel_html(payload):
    """邮件区块：只报"同步"一轴（方向中立、书面通俗）。"""
    if not payload or not payload.get("horizons"):
        return ('        <div class="section">\n'
                '            <div class="section-title">📌 风险提示</div>\n'
                '            <div style="font-size:13px;color:#6b7280;">暂未生成。</div>\n'
                '        </div>\n')

    hs = _norm_hs(payload)
    line = _axis_line(hs, SYNC_DISPLAY, "tier")
    hint = _hint(hs)
    body = (
        '            <div style="font-size:13px;color:#374151;line-height:1.9;">\n'
        f'                <div><b>同步</b>（一起涨、一起跌，分散效果下降）<br>{line}</div>\n'
        + (f'                <div style="margin-top:4px;">{hint}</div>\n' if hint else '')
        + '                <div style="font-size:11px;color:#6b7280;margin-top:6px;">'
        '⚠️ 本提示<b>不含方向判断</b>（涨跌不可预测）；仅反映"个股是否容易一起动"（分散效果）。</div>\n'
        '            </div>\n')
    return ('        <div class="section">\n'
            '            <div class="section-title">📌 风险提示</div>\n'
            + body + '        </div>\n')


def _norm_hs(payload):
    """JSON 往返后键变字符串（"1"）⇒ 归一化为 int。"""
    return {int(k): (v or {}) for k, v in (payload.get("horizons") or {}).items()}


def _axis_line(hs, horizons, key):
    parts = []
    for h in horizons:
        tier = (hs.get(h) or {}).get(key)
        if tier in TIER_WORD:
            parts.append(f"{H_LABEL[h]}：{TIER_WORD[tier]}")
    return " ｜ ".join(parts) if parts else "—"


def _hint(hs):
    """只在高档位给一句（方向中立）。"""
    order = {"green": 0, "yellow": 1, "red": 2}
    worst = "green"
    for h in SYNC_DISPLAY:
        t = (hs.get(h) or {}).get("tier")
        if t in order and order[t] > order[worst]:
            worst = t
    if worst == "red":
        return "同步偏高时，宜控制仓位与杠杆。"
    if worst == "yellow":
        return "同步略高，留意仓位与杠杆。"
    return ""


def _cli_html(payload):
    hs = _norm_hs(payload)
    lines = [f"📌 风险提示（截至 {payload.get('as_of')}）", "",
             "同步（一起涨、一起跌，分散效果下降）",
             "  " + _axis_line(hs, SYNC_DISPLAY, "tier")]
    h = _hint(hs)
    if h:
        lines += ["", h]
    lines += ["", "⚠️ 本提示不含方向判断（涨跌不可预测）；仅反映分散效果。"]
    return "\n".join(lines)


def main():
    from pathlib import Path
    payload = latest_panel()
    out = Path(ROOT) / "data" / "risk_panel_latest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_cli_html(payload))
    print(f"\n[面板数据] {out}")
    for h in HORIZONS:
        d = payload["horizons"][h]
        pv = d["p_corr_rise"]
        print(f"  {H_LABEL[h]:10s} 同步(相关齐升) P={pv if pv is None else round(pv, 3)}  档位={d['tier']}")


if __name__ == "__main__":
    main()
