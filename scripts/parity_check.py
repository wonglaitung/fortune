#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⓪ 同一性闸 —— **可执行的部分**（REVIEW_GATES.md 标注的"已知待修风险"落地）。

⓪ 问的是「WF 的函数 = 生产的函数？」（4 问）。其中**大多需要运行时**才能判等；
本脚本只把**能离线读到**的部分自动化，**明确标注覆盖边界**，**不假装** ⓪ 已全自动：

| ⓪ 四问 | 本脚本 | 覆盖 |
|---|---|---|
| Q1 社区 ID 集合 | 读生产 pkl `community_ids` + 与 `net_comm_*` 特征**自洽性** | **部分**（无法与 WF 每折比） |
| Q2 特征列集合 | 读生产 pkl `feature_columns`：F1/HIBOR/net_* 计数 | **部分**（无 WF 列集可比） |
| Q3 时点（use_shift/mode） | **静态**扫描 `mode=` 调用点 | **启发式**（非运行时相等） |
| Q4 概率同源（校准器） | **静态**读 `oof_glob` / `cal_prefix` | **信息**（需人工确认语义） |
| Q5 快照 / Q7 门槛基准 | 读 env | **报告** |

**覆盖边界（必读）**：Q3/Q4 的**运行时相等**仍须人工 + `tests/test_pipeline_contract.py`。
本脚本**不能**替代 ⓪ 的完整核对（REVIEW_GATES.md 明令不得因有工具就当已自动化）。

非零退出 = 存在**可判定**的 FAIL（如生产 pkl 的社区特征与 `community_ids` 不一致）。

用法：
    python3 scripts/parity_check.py
    python3 scripts/parity_check.py --horizon 20 --model data/ml_trading_model_lgbm_20d.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAIL, PASS, INFO, WARN = "FAIL", "PASS", "INFO", "WARN"
_results: list[tuple[str, str, str]] = []


def _rec(status, name, detail=""):
    _results.append((status, name, detail))
    icon = {PASS: "✅", FAIL: "❌", INFO: "•", WARN: "⚠️"}[status]
    line = f"  {icon} {name}"
    if detail:
        line += f" —— {detail}"
    print(line)


def _count(cols, pat):
    r = re.compile(pat, re.I)
    return sum(1 for c in cols if r.search(c))


# ---------------------------------------------------------------- 产物层
def check_artifact(model_path: Path):
    if not model_path.exists():
        _rec(FAIL, "[1] 生产模型产物", f"缺失 {model_path}")
        return None
    try:
        with model_path.open("rb") as f:
            md = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        _rec(FAIL, "[1] 生产模型产物", f"无法加载: {e}")
        return None
    cols = list(md.get("feature_columns") or [])
    comm = md.get("community_ids")
    horizon = md.get("horizon")
    learner = md.get("learner", "catboost")
    if not cols:
        _rec(FAIL, "[1] 生产模型产物", "feature_columns 为空")
        return None
    n_f1 = _count(cols, r"^f1_|_f1_")
    n_hibor = _count(cols, r"hibor")
    n_net = _count(cols, r"^net_")
    _rec(PASS, "[1] 生产模型产物",
         f"{model_path.name} | learner={learner} horizon={horizon} | "
         f"特征 {len(cols)}（F1={n_f1} HIBOR={n_hibor} net_*={n_net}）")
    _rec(INFO, "[2] 列集合（Q2 部分）",
         "生产侧 F1/HIBOR 计数如上；**无 WF 每折列集可比** ⇒ 需人工/契约测试")
    # Q1 自洽性：有 net_comm_* 特征却无 community_ids = D30 那类不一致
    has_comm_feat = any(c.startswith("net_comm_") or c == "net_community_id" for c in cols)
    if has_comm_feat and not comm:
        _rec(FAIL, "[3] 社区自洽（Q1 部分）",
             "含 net_comm_* 特征但 community_ids=None ⇒ 训练/推理社区集合不一致风险")
    else:
        _rec(PASS, "[3] 社区自洽（Q1 部分）",
             f"community_ids={comm if comm else 'None'}（net_comm_* 特征={has_comm_feat}）")
    return md


# ---------------------------------------------------------------- 静态层
def check_static_timepoint():
    """Q3 静态启发式：扫描 mode= 调用点。"""
    files = [ROOT / "ml_services" / "ml_trading_model.py",
             ROOT / "ml_services" / "walk_forward_validation.py"]
    wf_prod = wf_bt = 0
    for fp in files:
        if not fp.exists():
            continue
        txt = fp.read_text(encoding="utf-8")
        if fp.name == "walk_forward_validation.py":
            wf_prod = len(re.findall(r"mode=['\"]production['\"]", txt))
            wf_bt = len(re.findall(r"mode=['\"]backtest['\"]", txt))
    if wf_bt:
        _rec(FAIL, "[4] 时点口径（Q3 静态）",
             f"WF 出现 mode='backtest' {wf_bt} 次 ⇒ 与生产 T 收盘口径不一致风险")
    else:
        _rec(INFO, "[4] 时点口径（Q3 静态）",
             f"WF 显式 mode='production' {wf_prod} 次，无 backtest ⇒ **静态**通过；"
             "运行时相等仍须人工（本检查非运行时）")


def check_static_calibrator():
    """Q4 静态：oof_glob / cal_prefix 配置（只报，不判）。"""
    daily = ROOT / "ml_services" / "daily_confidence.py"
    a_stock = ROOT / "a_stock_comprehensive_analysis.py"
    hk_oof = "未搜到"
    try:
        txt = daily.read_text(encoding="utf-8")
        hk_oof = "有 oof_glob 参数" if "oof_glob" in txt else "无"
    except OSError:
        pass
    a_oof = "A_STOCK_OOF_GLOB" in (a_stock.read_text(encoding="utf-8")
                                  if a_stock.exists() else "")
    _rec(INFO, "[5] 校准器（Q4 静态）",
         f"daily_confidence: {hk_oof}；A股传 oof_glob={a_oof}。"
         "HK 是否应传 oof_glob **需人工确认**（REVIEW_GATES 记录：港股曾未传）")


def check_env():
    """Q5 快照 / Q7 门槛基准。"""
    us = os.environ.get("US_MARKET_SNAPSHOT_DIR", "")
    hk = os.environ.get("HK_MARKET_SNAPSHOT_DIR", "")
    gate = os.environ.get("GATE_SOURCE_CSV", "")

    def _stat(p):
        return ("存在" if Path(p).exists() else "缺失") if p else "未设"

    _rec(INFO, "[6] 快照 / 门槛基准（Q5/Q7）",
         f"US={us or '未设'}({_stat(us)}) HK={hk or '未设'}({_stat(hk)}) GATE={gate or '未设'}({_stat(gate)})")


def main():
    ap = argparse.ArgumentParser(description="⓪ 同一性闸可执行检查（部分）")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    model_path = Path(args.model) if args.model else \
        ROOT / f"data/ml_trading_model_lgbm_{args.horizon}d.pkl"

    print("=" * 74)
    print("⓪ 同一性闸 —— 可执行检查（scripts/parity_check.py）")
    print("   覆盖边界：仅离线可读项；Q3/Q4 运行时相等仍须人工 + tests/test_pipeline_contract.py")
    print("=" * 74)
    check_artifact(model_path)
    check_static_timepoint()
    check_static_calibrator()
    check_env()

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    n_pass = sum(1 for s, _, _ in _results if s == PASS)
    n_info = sum(1 for s, _, _ in _results if s in (INFO, WARN))
    print("-" * 74)
    print(f"结论: {n_pass} PASS / {n_fail} FAIL / {n_info} INFO")
    if n_fail:
        print("🛑 存在可判定 FAIL —— 不得作为结论 / 部署依据")
        return 1
    print("✅ 无**可判定** FAIL（注意：这不等于 ⓪ 全通过，见覆盖边界）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
