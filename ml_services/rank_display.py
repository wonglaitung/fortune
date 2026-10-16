#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 排序分数 · 展示层（D59 P3）。

回归 score 无概率语义 ⇒ 邮件不再显示"上涨概率"，改显示**当日横截面分位**（相对强弱）。
- `to_percentile`：把 score 转成当日分位（1.0=最强，0.0=最弱）；
- `direction_label`：分位 → 偏强 / 中性 / 偏弱（阈值可配）；
- `render_html`：渲染邮件区块（"上涨概率" → "相对强弱分位"）。

本模块**旁路**：不改 `comprehensive_analysis` 现有概率链路；供 V4 管线/预览用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRONG_PCT = 0.70   # 分位 ≥ 此值 = 偏强
WEAK_PCT = 0.30     # 分位 ≤ 此值 = 偏弱


def to_percentile(scores) -> pd.Series:
    """score → 相对分位（1.0=最高分）。等价 rank(pct=True)。"""
    s = pd.Series(scores, dtype="float64")
    return s.rank(pct=True)


def direction_label(pct: float, strong: float = STRONG_PCT, weak: float = WEAK_PCT) -> str:
    """分位 → 方向标签。⚠️ 这是**相对强弱**，不是"涨/跌概率"。"""
    if pct >= strong:
        return "偏强"
    if pct <= weak:
        return "偏弱"
    return "中性"


@dataclass
class RankRow:
    code: str
    name: str
    score: float
    pct: float
    label: str


def build_rows(scores, codes, names=None) -> list[RankRow]:
    pcts = to_percentile(scores).values
    names = names or [""] * len(codes)
    out = []
    for c, n, sc, p in zip(codes, names, np.asarray(scores, dtype=float), pcts):
        out.append(RankRow(code=c, name=n, score=float(sc), pct=float(p), label=direction_label(float(p))))
    return sorted(out, key=lambda r: -r.pct)


def render_html(rows: list[RankRow], horizon: int, top: int | None = None) -> str:
    """渲染邮件区块。显示"相对强弱分位"，**不**显示"上涨概率"。"""
    shown = rows if top is None else rows[:top]
    trs = "".join(
        f"<tr><td>{r.name}<br><span style='color:#888'>{r.code}</span></td>"
        f"<td style='text-align:right'>{r.pct*100:.0f}%</td>"
        f"<td style='text-align:right'>{r.label}</td></tr>"
        for r in shown
    )
    return (
        f"<div style='font-family:sans-serif;font-size:13px'>"
        f"<p><b>{horizon} 日相对强弱</b>（当日横截面分位，非涨跌概率）</p>"
        f"<table style='border-collapse:collapse'><tr><th align='left'>股票</th>"
        f"<th align='right'>分位</th><th align='right'>标签</th></tr>{trs}</table>"
        f"<p style='color:#888'>分位 ≥70% 偏强 / ≤30% 偏弱 / 其余中性；"
        f"<b>无概率语义</b>，仅表示在池内的相对位置。</p></div>"
    )


def render_text(rows: list[RankRow], horizon: int, top: int | None = None) -> str:
    shown = rows if top is None else rows[:top]
    lines = [f"{horizon}日相对强弱（当日横截面分位，非涨跌概率）", "-" * 40]
    for r in shown:
        lines.append(f"{r.code:<12}{r.name:<8}{r.pct*100:>5.0f}%  {r.label}")
    lines.append("分位 ≥70% 偏强 / ≤30% 偏弱 / 其余中性（无概率语义）")
    return "\n".join(lines)
