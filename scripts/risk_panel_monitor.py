#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险面板 —— 滚动（样本外）校准监控（D55/D56）。

目的：**持续追踪"波动 / 同步"信号的样本外 AUC / Brier 有没有漂**（信号衰减即下架）。
- `--update`：重算各序列的 walk-forward **样本外**指标 → 追加 `data/risk_panel_ledger.csv`。
- `--report`：读 ledger，报"最新 vs 历史"漂移 + **健康判定**。
- `summary()`：供邮件/报告引用的一行结论。

判据（预注册，D55/D56）：**近窗 AUC > 0.55 = OK；0.50–0.55 = WARN；≤0.50 = FAIL**。
⚠️ 这不是 alpha；**出口 = 风控档位**。

用法：
  python3 scripts/risk_panel_monitor.py --update
  python3 scripts/risk_panel_monitor.py --report
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_services.risk_horizon_signals import (  # noqa: E402
    HORIZONS, SNAP, VOL_HORIZONS, build_dataset, load_returns, walk_forward)

LEDGER = ROOT / "data" / "risk_panel_ledger.csv"
RECENT = 250
DEPLOYED = [("corr_rise", H) for H in HORIZONS]   # 已上面板（同步轴）
WATCH = [("big_move", H) for H in VOL_HORIZONS]   # 已从面板撤下（观察，看会不会回来）
SERIES = DEPLOYED + WATCH
FIELDS = ["run_date", "series", "n", "auc", "brier", "skill",
          "recent_auc", "recent_brier", "recent_skill", "verdict"]


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def _brier(y, p):
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y, p))


def _verdict(recent_auc):
    if recent_auc is None or recent_auc != recent_auc:
        return "NODATA"
    if recent_auc > 0.55:
        return "OK"
    if recent_auc > 0.50:
        return "WARN"
    return "FAIL"


def _require_snapshot():
    """快照缺失时给出可执行指令，而不是抛 FileNotFoundError 原始栈。"""
    if not os.path.isdir(SNAP) or not any(f.endswith(".pkl") for f in os.listdir(SNAP)):
        raise SystemExit(
            f"❌ 缺港股行情快照：{SNAP}\n"
            "   先运行 python3 scripts/pin_hk_snapshot.py（该目录不入库，CI 必须先生成）"
        )


def evaluate(recent=RECENT):
    _require_snapshot()
    rets = load_returns()
    df, _ = build_dataset(rets)
    out = []
    for kind, H in SERIES:
        r = walk_forward(df, f"{kind}_{H}")
        row = {"series": f"{kind}_{H}", "kind": kind, "H": H, "verdict": "NODATA",
               "n": 0, "auc": float("nan"), "brier": float("nan"), "skill": float("nan"),
               "recent_auc": float("nan"), "recent_brier": float("nan"), "recent_skill": float("nan")}
        if r and r["n"] > 0:
            y, p = r["y"], r["p"]
            k = min(recent, len(y))
            yr, pr = y[-k:], p[-k:]
            base = float(yr.mean())
            clim = base * (1 - base)
            row.update({
                "n": r["n"], "auc": r["auc"], "brier": r["brier"],
                "skill": (1 - r["brier"] / r["brier_clim"]) if r["brier_clim"] > 0 else float("nan"),
                "recent_auc": _auc(yr, pr), "recent_brier": _brier(yr, pr),
                "recent_skill": (1 - _brier(yr, pr) / clim) if clim > 0 else float("nan"),
            })
            row["verdict"] = _verdict(row["recent_auc"])
        out.append(row)
    return out


def _load_ledger():
    if not LEDGER.exists():
        return []
    with LEDGER.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update():
    rows = evaluate()
    today = date.today().isoformat()
    existing = _load_ledger()
    seen = {(r["run_date"], r["series"]) for r in existing}
    new = []
    for r in rows:
        if (today, r["series"]) in seen:
            continue
        rr = {}
        for k in FIELDS:
            v = r.get(k)
            rr[k] = f"{v:.4f}" if isinstance(v, float) and v == v else v
        rr["run_date"] = today      # 必须最后赋（FIELDS 含 run_date，会被 r.get 覆盖成 None）
        new.append(rr)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        for r in new:
            w.writerow(r)
    print(f"💾 {LEDGER}（{len(existing) + len(new)} 行；本次新增 {len(new)}）")
    return rows


def report():
    rows = evaluate()
    print("风险面板 —— 滚动（样本外）校准")
    print("=" * 90)
    print(f"{'序列':<14}{'n':>6}{'全期AUC':>9}{'全期技能':>9}{'近%d期AUC' % RECENT:>12}{'近窗技能':>9}{'判定':>7}  角色")
    worst = "OK"
    order = {"OK": 0, "WARN": 1, "FAIL": 2, "NODATA": 1}
    for r in rows:
        deployed = r["kind"] == "corr_rise"
        # ⚠️ 整体判定**只看已部署（同步轴）**；big_move 已撤 ⇒ 仅观察
        if deployed and r["verdict"] != "NODATA" and order.get(r["verdict"], 0) > order.get(worst, 0):
            worst = r["verdict"]
        role = "已部署" if deployed else "观察·未部署"
        print(f"{r['series']:<14}{r['n']:>6}{r['auc']:>9.3f}{r['skill']*100:>8.1f}%"
              f"{r['recent_auc']:>12.3f}{r['recent_skill']*100:>8.1f}%{r['verdict']:>7}  {role}")
    print("-" * 90)
    print(f"整体判定（仅已部署）: {worst}（近窗 AUC>0.55=OK / 0.50–0.55=WARN / ≤0.50=FAIL）")
    print("⚠️ 衰减即下架；观察项若恢复（近窗 AUC>0.55）可考虑重新上线。本监控不是 alpha。")
    return rows, worst


def summary():
    """供邮件/报告引用。"""
    try:
        rows, worst = report()
        bad = [r["series"] for r in rows if r["verdict"] in ("FAIL", "WARN")]
        txt = ("风险信号校准正常（近窗 AUC 全 >0.55）" if worst == "OK"
               else f"风险信号校准需留意（{worst}: {', '.join(bad)}）")
        return {"text": txt, "verdict": worst}
    except Exception as e:  # noqa: BLE001
        return {"text": f"校准监控不可用（{e}）", "verdict": "NODATA"}


def main():
    ap = argparse.ArgumentParser(description="风险面板滚动校准监控")
    ap.add_argument("--update", action="store_true", help="追加最新样本外指标到 ledger")
    ap.add_argument("--report", action="store_true", help="报滚动漂移 + 健康判定")
    args = ap.parse_args()
    if args.update:
        update()
    if args.report or not args.update:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
