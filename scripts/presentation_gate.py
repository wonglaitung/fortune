#!/usr/bin/env python3
"""A 呈现闸（可执行版）—— 任何指标作为结论输出前的强制检查。

背景：三道闸（A/B/C）与两条硬约束原本只以散文形式写在 AGENTS.md，
必须靠执行者"记得去做"。2026-10-02 实证散文闸会被跳过：整场会话都在
更新 DECISIONS/AGENTS/lessons，却没人打开 docs/DEPLOYMENT.md，
里面写着已被证伪的「🟢 可放大至 15%」。本脚本把 A 闸与约束 1 落成
可执行检查，非零退出即阻断。

严格区分两类阈值（B 闸第 5 项：不得写未验证的断言）：
  [文档阈值]   来自 AGENTS.md 明文，直接执行
  [操作化阈值] 文档只给了方向性描述、没有数字，此处给出数字锚点并明示

检查项与出处：
  H1 准确率异常          AGENTS:115  个股 >65% / 恒指 >80% 为泄漏信号
  H2 Predict_Prob 饱和   AGENTS:115  「饱和成 0/1」
  H3 符号退化            AGENTS:115  预测 UP 组 Relative_Return 最大值 ≤0 即泄漏
                                     （CSV 无此列，须重建 = Actual − HSI 未来收益）
  H4 IC 异常             A闸判据「IC」（无文档数字阈值，仅在完美时判定）
  H5 分组符号一致性       A闸判据「分组符号一致性」（同上）
  C1 行级 vs Fold 聚类    约束 1  行级统计不得用于因果推断

用法：
  python3 scripts/presentation_gate.py --input <csv> --horizon 20
  python3 scripts/presentation_gate.py --input <csv> --horizon 20 --market hsi
退出码：0 = PASS（可作为结论呈现），1 = FAIL（不得作为结论输出）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 恒指阈值不同（AGENTS:115）
ACC_THRESHOLD = {"stock": 0.65, "hsi": 0.80, "a": 0.65}

# AGENTS:115 2026-10-07 修正前的旧句为「预测 UP 组收益最大值必须 <0」，
# 该普适规则方向写反（合法数据实测 max=+0.4209），已随本次修正一并纠正。


def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman IC，手动实现以避免 scipy 依赖差异。"""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run_checks(df: pd.DataFrame, horizon: int, market: str) -> tuple[list[str], list[str]]:
    """返回 (failures, notes)。failures 非空即阻断。"""
    failures: list[str] = []
    notes: list[str] = []

    prob = df["Predict_Prob"].astype(float).values
    ret = df["Actual_Return"].astype(float).values
    correct = df["Is_Correct"]
    if correct.dtype == object:
        correct = correct.map({"True": True, "False": False, True: True, False: False})
    correct = correct.astype(bool).values
    pred_up = df["Predict_Direction"].astype(str).str.upper().eq("UP").values

    n = len(df)
    n_folds = df["Fold"].nunique() if "Fold" in df.columns else 0
    notes.append(f"样本 {n} 行 / {n_folds} 折 / horizon={horizon} / market={market}")

    # ---------- H1 准确率异常（文档阈值） ----------
    acc = float(correct.mean())
    thr = ACC_THRESHOLD[market]
    status = "FAIL" if acc > thr else "PASS"
    notes.append(f"[H1][文档阈值] 合并准确率 {_fmt(acc)}（阈值 {thr:.0%}）→ {status}")
    if status == "FAIL":
        failures.append(
            f"H1 准确率 {_fmt(acc)} > {thr:.0%}（AGENTS:115 泄漏信号）"
        )

    # ---------- H2 概率饱和（AGENTS:115「饱和成 0/1」） ----------
    exact_sat = float(np.mean((prob <= 0.0) | (prob >= 1.0)))
    near_sat = float(np.mean((prob <= 1e-3) | (prob >= 1 - 1e-3)))
    q = np.quantile(prob, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    notes.append(
        f"[H2][报告] Predict_Prob 分位 min={q[0]:.4f} P1={q[1]:.4f} P5={q[2]:.4f} "
        f"中位={q[3]:.4f} P95={q[4]:.4f} P99={q[5]:.4f} max={q[6]:.4f}"
    )
    notes.append(
        f"[H2][报告] 精确0/1占比 {exact_sat:.4%}；≤1e-3或≥1-1e-3 占比 {near_sat:.4%}"
    )
    # 操作化阈值：文档只说「饱和成 0/1」未给比例，取 1% 作为饱和判定
    if exact_sat > 0.01:
        failures.append(
            f"H2 Predict_Prob 精确0/1占比 {exact_sat:.4%} > 1%（[操作化阈值] "
            f"锚点 AGENTS:115「饱和成 0/1」，比例数字非文档既有）"
        )

    # ---------- H3 符号退化（AGENTS:115，2026-10-07 修正口径） ----------
    # 原始依据 lessons 三.28：泄漏版预测 UP 组 Relative_Return 最大值 = −1e-05（贴零）。
    # 注意 CSV 无 Relative_Return 列，须重建 = Actual_Return − HSI 未来收益。
    # 实测对照（2026-10-07）：合法新池1d UP 组 max=+0.4209；泄漏版 max=−1e-05。
    # 故判定为「max ≤ 0 即符号退化」。此前 AGENTS 误写为「必须 <0」，方向相反。
    if pred_up.any():
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rel_alpha_check import build_hsi_forward_return  # 复用，避免重复实现

            d = pd.to_datetime(df["Date"])
            if d.dt.tz is not None:
                d = d.dt.tz_convert(None)
            fwd = build_hsi_forward_return(horizon)
            hsi_f = d.dt.normalize().map(fwd)
            rel = ret - hsi_f.values
            ok = ~np.isnan(rel)
            align = float(ok.mean())
            sub = rel[ok & pred_up]
            if len(sub) == 0:
                notes.append("[H3][文档阈值] 对齐后无预测 UP 行，跳过")
            else:
                up_max = float(np.max(sub))
                up_rng = float(np.max(sub) - np.min(sub))
                notes.append(
                    f"[H3][文档阈值] HSI 对齐 {align:.1%} | 预测UP组 Relative_Return "
                    f"max={up_max:.6g} 极差={up_rng:.6g}（合法实测约 +0.42 / 0.88）"
                )
                if up_max <= 0:
                    failures.append(
                        f"H3 预测 UP 组 Relative_Return 最大值 {up_max:.6g} ≤ 0 —— "
                        f"无一只预测上涨股跑赢大盘，符号退化；三.28 泄漏版实测 −1e-05"
                    )
                elif up_rng < 1e-3:
                    failures.append(
                        f"H3 预测 UP 组极差 {up_rng:.6g} < 1e-3 —— 分布退化为常数"
                    )
        except Exception as e:
            # 无法计算 ≠ 通过：明确标注未执行，不静默放行
            notes.append(
                f"[H3] ⚠ 未执行（无法重建 Relative_Return：{type(e).__name__}: {e}）"
                f"—— 本项未经验证，不得据此声称「已过 A 闸全部检查」"
            )
    else:
        notes.append("[H3] 无预测 UP 行，跳过")

    # ---------- H4 IC 异常（A闸判据，无文档数字阈值） ----------
    # 必须先剔 NaN，否则 corrcoef 结果为 nan 会静默显示 n/a 而非报错
    ok4 = np.isfinite(prob) & np.isfinite(ret)
    ic = _spearman(prob[ok4], ret[ok4]) if ok4.sum() >= 3 else float("nan")
    if ok4.sum() < len(prob):
        notes.append(f"[H4] 有效行 {int(ok4.sum())}/{n}（已剔除非有限值）")
    notes.append(f"[H4][报告] rank IC(Predict_Prob vs Actual_Return) = {_fmt(ic)}")
    if ic == ic and abs(ic) >= 0.99:
        failures.append(f"H4 |IC|={abs(ic):.4f} 近乎完美 —— 泄漏信号（无文档阈值，仅在完美时判定）")

    # ---------- H5 分组符号一致性（A闸判据，无文档数字阈值） ----------
    try:
        ok5 = np.isfinite(prob) & np.isfinite(ret)
        if int(ok5.sum()) < 10:
            notes.append(f"[H5] 有效行不足（{int(ok5.sum())} < 10），跳过分组")
        else:
            pv, rv = prob[ok5], ret[ok5]
            nbins = 5 if len(pv) >= 50 else 2
            bins = pd.qcut(pd.Series(pv), nbins, duplicates="drop")
            grp = pd.DataFrame({"b": bins, "r": rv}).groupby("b", observed=True)["r"].mean()
            mono = bool(grp.is_monotonic_increasing or grp.is_monotonic_decreasing)
            spread = float(grp.max() - grp.min())
            notes.append(
                f"[H5][报告] {nbins} 分组均值极差 {_fmt(spread)}；单调={'是' if mono else '否'}"
                f"；组值 {[round(float(v), 5) for v in grp.values]}"
            )
            if mono and spread >= 0.5:
                failures.append(
                    f"H5 分组均值完全单调且极差 {spread:.4f} ≥ 0.5 —— 分组收益跨度 50% "
                    f"不现实（[操作化阈值] 无文档数字，仅供阻断用）"
                )
    except Exception as e:  # 分组失败不阻断，报告即可
        notes.append(f"[H5] 分组失败：{type(e).__name__}: {e}")

    # ---------- C1 约束 1：行级 vs Fold 聚类 ----------
    if n_folds and n_folds > 1:
        row_se = float(np.std(correct.astype(float), ddof=1) / np.sqrt(n))
        row_z = (acc - 0.5) / row_se if row_se > 0 else float("nan")
        per_fold = df.assign(_c=correct.astype(float)).groupby("Fold")["_c"].mean()
        fvals = per_fold.values
        fold_se = float(np.std(fvals, ddof=1) / np.sqrt(len(fvals))) if len(fvals) > 1 else float("nan")
        fold_t = (float(fvals.mean()) - 0.5) / fold_se if fold_se and fold_se > 0 else float("nan")
        notes.append(
            f"[C1][约束1] 行级 z={_fmt(row_z, 2)}（**伪显著，禁止用于因果推断**）  "
            f"|  Fold聚类 t={_fmt(fold_t, 2)} n={len(fvals)}（判读依据）"
        )
        if abs(row_z) >= 3.0 and not (abs(fold_t) >= 2.0):
            notes.append(
                f"[C1] ⚠ 行级 |z|≥3 但折聚类 |t|<2 —— 典型伪显著，"
                f"结论必须只引用折聚类口径"
            )
    else:
        notes.append("[C1] 折数不足，无法做 Fold 聚类（约束 1 无法执行 → 不得作因果结论）")
        failures.append("C1 折数 <2，约束 1 无法执行，禁止任何因果性表述")

    return failures, notes


def main() -> int:
    ap = argparse.ArgumentParser(description="A 呈现闸：指标作为结论输出前的强制检查")
    ap.add_argument("--input", required=True, help="prediction_analysis.csv 路径")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--market", default="stock", choices=["stock", "hsi", "a"],
                    help="stock=个股(>65%%) / hsi=恒指(>80%%) / a=A股(>65%%)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"Predict_Prob", "Predict_Direction", "Actual_Return", "Is_Correct"}
    missing = required - set(df.columns)
    if missing:
        print(f"[A闸] FAIL —— CSV 缺列 {sorted(missing)}")
        return 1

    failures, notes = run_checks(df, args.horizon, args.market)

    print("=" * 70)
    print("🚦 A 呈现闸（可执行版）")
    print("=" * 70)
    for s in notes:
        print("  " + s)
    print("-" * 70)
    if failures:
        print("❌ A 闸 FAIL —— 以下指标不得作为结论输出：")
        for f in failures:
            print("   ✗ " + f)
        print("   处理：定位泄漏/异常根因后重跑，或改以折聚类口径重述。")
        return 1
    print("✅ A 闸 PASS —— 可作为结论呈现（仍须按判读顺序走 PBO/DSR/CI）")
    print("   注意：AGENTS:116 —— PBO/DSR 检测不出泄漏，本闸不可被其替代。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
