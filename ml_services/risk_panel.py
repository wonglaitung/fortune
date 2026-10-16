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

from ml_services.risk_horizon_signals import FEATURES, HORIZONS, build_dataset, load_returns  # noqa: E402

H_LABEL = {1: "未来 1 天", 5: "未来 1 周", 10: "未来 2 周", 20: "未来 1 个月"}
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
        p_vol = _fit_predict(f"big_move_{H}")
        out[H] = {"p_corr_rise": p_sync, "tier": tier_of(p_sync),
                  "p_big_move": p_vol, "vol_tier": tier_of(p_vol)}
    return {"as_of": as_of, "horizons": out,
            "tiers_note": "🟢<0.5 / 🟡0.5–0.75 / 🔴≥0.75（暂用，正式须预注册）"}


def risk_panel_html(payload):
    """邮件区块（**方向中立**、书面但通俗）。只用到"同步"轴（相关齐升）；
    "波动"轴（大波动 |r|）待 D56 接入。"""
    if not payload or not payload.get("horizons"):
        return ('        <div class="section">\n'
                '            <div class="section-title">📌 波动提示</div>\n'
                '            <div style="font-size:13px;color:#6b7280;">暂未生成。</div>\n'
                '        </div>\n')

    hs = payload["horizons"]
    lead, bullets = _compose(hs)
    body = (f'            <div style="font-size:13px;color:#374151;line-height:1.9;">\n'
            f'                {lead}\n')
    if bullets:
        body += '                <div style="margin-top:4px;">· ' + '<br>· '.join(bullets) + '</div>\n'
    body += (
        '                <div style="font-size:11px;color:#6b7280;margin-top:6px;">'
        '⚠️ 本提示<b>不含方向判断</b>（涨跌不可预测）；仅反映波动水平：波动上行时，宜控制仓位与杠杆。</div>\n'
        '            </div>\n')
    return ('        <div class="section">\n'
            '            <div class="section-title">📌 波动提示（未来 1 周 / 1 个月）</div>\n'
            + body + '        </div>\n')


def _compose(hs):
    """把各 horizon 档位写成人话（方向中立）。两轴：同步（相关齐升）+ 波动（大波动）。"""
    def st(h):
        return (hs.get(h) or {}).get("tier")

    def vt(h):
        return (hs.get(h) or {}).get("vol_tier")

    sync_flag = [(h, st(h)) for h in (5, 20) if st(h) in ("yellow", "red")]
    vol_flag = [(h, vt(h)) for h in (1, 5) if vt(h) in ("yellow", "red")]
    if not sync_flag and not vol_flag:
        return "未来 1 周 / 1 个月：正常。", []

    parts = [f"{H_LABEL[h]}·同步{TIER_WORD[t]}" for h, t in sync_flag]
    parts += [f"{H_LABEL[h]}·波动{TIER_WORD[t]}" for h, t in vol_flag]
    lead = "；".join(parts) + "。"

    bullets = []
    if sync_flag:
        bullets += ["同步 = 同涨同跌，组合分散效果下降",
                    "故账户净值波动大概率放大（涨跌皆可能）"]
    if vol_flag:
        bullets.append("波动可能放大（涨跌皆可能）")
    seen = set()
    bullets = [b for b in bullets if not (b in seen or seen.add(b))]
    if any(t == "red" for _, t in sync_flag + vol_flag):
        bullets.append("波动上行阶段，宜控制仓位与杠杆")
    return lead, bullets


def _cli_html(payload):
    """CLI 友好的纯文本预览。"""
    lead, bullets = _compose(payload["horizons"])
    lines = [f"📌 波动提示（截至 {payload['as_of']}）", "", lead]
    for b in bullets:
        lines.append("  · " + b)
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
        print(f"  {H_LABEL[h]:10s} 相关齐升 P={pv if pv is None else round(pv,3)}  档位={d['tier']}")


if __name__ == "__main__":
    main()
