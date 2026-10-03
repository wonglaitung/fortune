#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股 分位门槛（docs/DECISIONS.md D8 同款，计划 P3.2）

熊市/弱震荡的准入门槛不用绝对概率（0.70/0.65），改取 **A股 walk-forward 回测中
校准后概率分布的分位点**（PIT）：
- bear  = P92 → 前约 8% 通过
- weak  = P90 → 前约 10% 通过
- normal 保持绝对 0.50（硬约束语义：胜率>50%，与分位无关）

为何不用 prediction_history：自选股批量预测右尾过窄，前 8% 分位低于买入线会门槛失效
（与港股同因，见 ml_services/market_regime.py 注释）。

回退链：本地最新 A股 20d 回测 CSV + A股校准器 → 内嵌快照 A_GATE_SNAPSHOT →
绝对值 A_GATE_FALLBACK（原 0.70/0.65）。

⚠️ walk-forward 重跑后分位会漂移 → 重跑完执行
   `python3 -c "from ml_services.a_stock_gates import suggest_a_gate_snapshot; suggest_a_gate_snapshot()"`
   并更新 A_GATE_SNAPSHOT（progress.txt 记录）。
"""

import glob
import logging
import os
import re
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

A_GATE_QUANTILES = {'bear': 0.92, 'weak': 0.90}
A_GATE_FALLBACK = {'bear': 0.70, 'weak': 0.65}   # 快照/CSV 全无时的最终回退（原绝对值）
A_GATE_MIN_SAMPLES = 200
# 退化保护（对齐港股 lessons 三.30）：模型无 edge 时 Isotonic 把概率压回基准率 →
# 分位锚在 0.5 → 门控近乎不过滤（港股实测 bear 通过率 17.2%→98.2%）。故
# 校准后唯一值过少 / bear·weak 分位重合 → 放弃分位、回落 A_GATE_FALLBACK 绝对值。
A_GATE_MIN_UNIQUE = 30
A_GATE_MIN_SPREAD = 0.05
# 分层区分度下限（与港股 GATE_MIN_LAYER_SEP 同步，2026-10-03）。
# 与 A_GATE_MIN_SPREAD 语义重叠（A 股 quantiles 仅 bear/weak 两层，spread 即层间极差），
# 保留为**显式冗余**，防止将来 quantiles 扩层时失去「层间无区分度」判别能力。
# 实测 2026-10-03：A 股 P90=0.681529、P92=0.682713，间距 0.0012 → 判退化回退绝对值。
A_GATE_MIN_LAYER_SEP = 0.01

# 分位快照：由 output/20260928_144833_a_stock_catboost_20d
# （19,578 条，as_of=2026-07-31，与 a_stock_prob_cal_20.pkl 同源）算出。
A_GATE_SNAPSHOT = {'bear': 0.7, 'weak': 0.65}  # 退化保护后实际生效值（绝对阈值）；
# 原分位值 bear 0.7497/weak 0.7154 已不适用——校准后唯一值 37、极差 0.0343 < GATE_MIN_SPREAD
# → 分布退化，compute_a_gate_thresholds 判定放弃分位、回落绝对值（lessons 三.30）
A_GATE_SNAPSHOT_AS_OF = '2026-07-31'

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_A_GATE_GLOB = os.path.join(_BASE_DIR, 'output', '*_a_stock_catboost_20d',
                            'prediction_analysis.csv')
_A_GATE_CALIBRATOR_FILE = os.path.join(_BASE_DIR, 'data', 'calibrators',
                                       'a_stock_prob_cal_20.pkl')


def _latest_a_gate_source_csv() -> Optional[str]:
    """最新 A股 20d 回测 CSV（生产学习器 catboost；lightgbm A/B 目录不参与门槛）

    可用环境变量 A_GATE_SOURCE_CSV 显式固定基准（复现用，对齐港股 GATE_SOURCE_CSV，
    lessons 三.29）：默认取「磁盘上最新 CSV」，而回测运行期间 output/ 会新增目录，
    同一模型两次运行会因 PIT 分位样本不同而产生不同的 Dynamic_Threshold。
    """
    forced = os.environ.get('A_GATE_SOURCE_CSV')
    if forced:
        if os.path.exists(forced):
            return forced
        logger.warning("A_GATE_SOURCE_CSV=%s 不存在，回退到自动选择最新 CSV", forced)
    files = glob.glob(_A_GATE_GLOB)
    return max(files) if files else None


def _a_gates_fallback(reason: str) -> Dict[str, float]:
    if A_GATE_SNAPSHOT:
        logger.info("A股分位门槛：%s，使用内嵌快照 %s (as_of=%s)",
                    reason, A_GATE_SNAPSHOT, A_GATE_SNAPSHOT_AS_OF)
        return dict(A_GATE_SNAPSHOT)
    logger.warning("A股分位门槛：%s，回退绝对阈值 %s", reason, dict(A_GATE_FALLBACK))
    return dict(A_GATE_FALLBACK)


def compute_a_gate_thresholds(as_of: Optional[str] = None,
                              source_csv: Optional[str] = None,
                              calibrator_file: Optional[str] = None,
                              quantiles: Optional[Dict[str, float]] = None,
                              min_samples: int = A_GATE_MIN_SAMPLES,
                              use_snapshot_fallback: bool = True) -> Dict[str, float]:
    """按 as_of（PIT，含当日）计算 bear/weak 分位门槛。

    数据流：最新 A股 walk-forward 回测 CSV（20d）中 Date<=as_of 的 Predict_Prob
    → Isotonic 校准（a_stock_prob_cal_20）→ 分位数。

    Returns:
        {layer: threshold}；样本不足或失败时按内嵌快照 → 绝对值回退链处理。
    """
    quantiles = quantiles or A_GATE_QUANTILES

    def _fb(reason: str) -> Dict[str, float]:
        if use_snapshot_fallback:
            return _a_gates_fallback(reason)
        logger.warning("A股分位门槛：%s，回退绝对阈值 %s", reason, dict(A_GATE_FALLBACK))
        return {layer: A_GATE_FALLBACK.get(layer, 0.50) for layer in quantiles}

    source_csv = source_csv or _latest_a_gate_source_csv()
    calibrator_file = calibrator_file or _A_GATE_CALIBRATOR_FILE
    try:
        if not (source_csv and os.path.exists(source_csv) and os.path.exists(calibrator_file)):
            return _fb("回测CSV或校准器缺失")
        df = pd.read_csv(source_csv, usecols=['Date', 'Predict_Prob']).dropna()
        if as_of:
            df = df[df['Date'].astype(str) <= str(as_of)[:10]]
        probs = df['Predict_Prob'].to_numpy(dtype=float)
        if probs.size < min_samples:
            return _fb(f"PIT样本 {probs.size} < {min_samples}")
        iso = joblib.load(calibrator_file)
        cal = np.asarray(iso.predict(probs.reshape(-1, 1)), dtype=float)
        gates = {layer: float(np.percentile(cal, q * 100.0))
                 for layer, q in quantiles.items()}
        # 退化保护：分布被压平 → 分位无经济含义 → 用绝对值（港股教训三.30）
        n_uniq = int(len(np.unique(cal)))
        spread = max(gates.values()) - min(gates.values())
        layer_sep = abs(gates.get('bear', 0) - gates.get('weak', 0))
        if n_uniq < A_GATE_MIN_UNIQUE or spread < A_GATE_MIN_SPREAD \
                or layer_sep < A_GATE_MIN_LAYER_SEP:
            _why = []
            if n_uniq < A_GATE_MIN_UNIQUE:
                _why.append(f"唯一值 {n_uniq}<{A_GATE_MIN_UNIQUE}")
            if spread < A_GATE_MIN_SPREAD:
                _why.append(f"bear-weak 间距 {spread:.4f}<{A_GATE_MIN_SPREAD}")
            if layer_sep < A_GATE_MIN_LAYER_SEP:
                _why.append(f"分层间距 {layer_sep:.4f}<{A_GATE_MIN_LAYER_SEP}（bear≈weak）")
            return _fb("校准后分布退化（" + "、".join(_why) + "）→ 用绝对阈值")
        logger.info("A股分位门槛（as_of=%s, n=%d, src=%s）: %s",
                    as_of or 'latest', probs.size,
                    os.path.basename(os.path.dirname(source_csv)),
                    {k: round(v, 4) for k, v in gates.items()})
        return gates
    except Exception as e:
        return _fb(f"计算失败（{e}）")


_CACHED: Optional[Dict[str, float]] = None


def get_a_gates(refresh: bool = False) -> Dict[str, float]:
    """运行时门槛（邮件/报告/LLM prompt 用）：每次进程只算一次，失败走回退链。"""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = compute_a_gate_thresholds()
    return dict(_CACHED)


def suggest_a_gate_snapshot(as_of: Optional[str] = None) -> str:
    """walk-forward 重跑后打印建议的 A_GATE_SNAPSHOT 字符串（人工粘贴更新）。"""
    gates = compute_a_gate_thresholds(as_of=as_of, use_snapshot_fallback=False)
    src = _latest_a_gate_source_csv()
    print(f"A_GATE_SNAPSHOT = {gates}")
    print(f"# 源: {src}  as_of={as_of or A_GATE_SNAPSHOT_AS_OF}")


if __name__ == '__main__':
    print("当前门槛:", get_a_gates())
    suggest_a_gate_snapshot()
