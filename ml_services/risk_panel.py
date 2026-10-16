#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险面板（risk panel）—— 把 D55 的多周期风险信号聚合成**邮件区块**。

**定位**：只报**波动 / 同步**（风险），**不含方向**（D55：大跌不可测）。
**出口**：仓位 / 杠杆提示（**非 alpha**；D35 §十二）。

档位阈值（**暂用**，正式上线须预注册 + 校准）：
    🟢 P<0.5 · 🟡 0.5≤P<0.75 · 🔴 P≥0.75

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
    FEATURES, HORIZONS, VOL_HORIZONS, build_dataset, load_returns)

H_LABEL = {1: "未来 1 天", 5: "未来 1 周", 10: "未来 2 周", 20: "未来 1 个月"}
# 展示周期：波动轴只 1/5D；同步轴 1/5/10/20（均通过判据）
VOL_DISPLAY = [1, 5]
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
    """对每个 H 拟合（全史）并预测最后一行 → P(相关齐升) + P(大波动)。返回 payload。"""
    from sklearn.linear_model import LogisticRegression

    rets = load_returns()
    df, _ = build_dataset(rets)
    as_of = str(df.index[-1].date())

    def _fit_predict(col):
        d = df.dropna(subset=FEATURES + [col])
        if len(d) <= 300:
            return None
        X, y = d[FEATURES].values, d[col].values
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=1000)
        clf.fit((X - mu) / sd, y)
        xlast = ((df[FEATURES].iloc[-1].values - mu) / sd).reshape(1, -1)
        return float(clf.predict_proba(xlast)[0, 1])

    out = {}
    for H in horizons:
        p_sync = _fit_predict(f"corr_rise_{H}")
        out[H] = {"p_corr_rise": p_sync, "tier": tier_of(p_sync)}
        # 波动轴（大波动 |r|）**只采用 1/5D**（10/20D 未通过，D56 §七）
        if H in VOL_HORIZONS:
            p_vol = _fit_predict(f"big_move_{H}")
            out[H].update({"p_big_move": p_vol, "vol_tier": tier_of(p_vol)})
    return {"as_of": as_of, "horizons": out,
            "vol_axis_horizons": list(VOL_HORIZONS),
            "tiers_note": "🟢<0.5 / 🟡0.5–0.75 / 🔴≥0.75（暂用，正式须预注册）"}


def risk_panel_html(payload):
    """邮件区块：**两轴分开列，波动在前**（方向中立、书面通俗）。"""
    if not payload or not payload.get("horizons"):
        return ('        <div class="section">\n'
                '            <div class="section-title">📌 风险提示</div>\n'
                '            <div style="font-size:13px;color:#6b7280;">暂未生成。</div>\n'
                '        </div>\n')

    hs = _norm_hs(payload)
    vol_line = _axis_line(hs, VOL_DISPLAY, "vol_tier")
    sync_line = _axis_line(hs, SYNC_DISPLAY, "tier")
    hint = _hint(hs)

    body = (
        '            <div style="font-size:13px;color:#374151;line-height:1.9;">\n'
        f'                <div><b>波动</b>（大起大落，涨跌都算）<br>{vol_line}</div>\n'
        f'                <div style="margin-top:4px;"><b>同步</b>（一起涨、一起跌，分散效果下降）<br>{sync_line}</div>\n'
        + (f'                <div style="margin-top:4px;">{hint}</div>\n' if hint else '')
        + '                <div style="font-size:11px;color:#6b7280;margin-top:6px;">'
        '⚠️ 本提示<b>不含方向判断</b>（涨跌不可预测）；仅反映波动水平。'
        '（波动项只看短期 1 天 / 1 周；长周期只报"同步"。）</div>\n'
        '            </div>\n')
    return ('        <div class="section">\n'
            '            <div class="section-title">📌 风险提示</div>\n'
            + body + '        </div>\n')


def _norm_hs(payload):
    """JSON 往返后键变字符串（"1"）⇒ 归一化为 int。"""
    return {int(k): (v or {}) for k, v in (payload.get("horizons") or {}).items()}


def _axis_line(hs, horizons, key):
    """某轴一行：`未来 1 天：正常 ｜ 未来 1 周：留意`。"""
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
    for h in VOL_DISPLAY:
        t = (hs.get(h) or {}).get("vol_tier")
        if t in order and order[t] > order[worst]:
            worst = t
    if worst == "red":
        return "波动 / 同步偏高时，宜控制仓位与杠杆。"
    if worst == "yellow":
        return "波动 / 同步略高，留意仓位与杠杆。"
    return ""


def _cli_html(payload):
    """CLI 纯文本预览。"""
    hs = _norm_hs(payload)
    lines = [f"📌 风险提示（截至 {payload['as_of']}）", ""]
    lines.append("波动（大起大落，涨跌都算）")
    lines.append("  " + _axis_line(hs, VOL_DISPLAY, "vol_tier"))
    lines.append("同步（一起涨、一起跌，分散效果下降）")
    lines.append("  " + _axis_line(hs, SYNC_DISPLAY, "tier"))
    h = _hint(hs)
    if h:
        lines.append("")
        lines.append(h)
    lines.append("")
    lines.append("⚠️ 本提示不含方向判断（涨跌不可预测）；仅反映波动水平。")
    return "\n".join(lines)


def main():
    payload = latest_panel()
    Path = __import__("pathlib").Path
    out = ROOT and (Path(ROOT) / "data" / "risk_panel_latest.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_cli_html(payload))
    print(f"\n[面板数据] {out}")
    for h in HORIZONS:
        d = payload["horizons"][h]
        pv = d["p_corr_rise"]
        line = f"  {H_LABEL[h]:10s} 同步(相关齐升) P={pv if pv is None else round(pv,3)}  档位={d['tier']}"
        if "p_big_move" in d:
            pm = d["p_big_move"]
            line += f" | 波动(|r|) P={pm if pm is None else round(pm,3)}  档位={d.get('vol_tier')}"
        print(line)


if __name__ == "__main__":
    main()
