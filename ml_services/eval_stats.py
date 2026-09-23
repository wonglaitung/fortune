#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估统计工具（单一真相源）

提供置信区间、显著性检验、有效独立样本数、可靠性判定等，
用于 Walk-forward / 回测的严谨评估，避免用重叠样本得出虚高结论。
"""

import math
from collections import Counter


def wilson_ci(k, n, z=1.96):
    """Wilson 置信区间（小样本/极端比例比 Wald 更稳健）

    Args:
        k: 正确数
        n: 样本数（有效独立样本）
        z: 置信水平对应 z 值（1.96 -> 95%）

    Returns:
        (low, high)
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_p_value(k, n, p0=0.5):
    """单样本比例检验（vs p0，双侧，正态近似 + 连续性校正）

    Returns:
        p 值（越小越显著）
    """
    if n <= 0:
        return 1.0
    phat = k / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 1.0
    z = (abs(phat - p0) - 0.5 / n) / se
    z = max(0.0, z)
    # 双侧 p 值
    return math.erfc(z / math.sqrt(2))


def effective_n(predictions, horizon=None):
    """有效独立样本数

    - 指定 horizon: n / horizon
    - 未指定: 按每条预测的 horizon 分组求和 Σ_h (n_h / h)

    说明：重叠窗口下，n 个 h 天预测约等价于 n/h 个独立观测。
    """
    if horizon is not None:
        n = len(predictions)
        return n / horizon if horizon > 0 else float(n)

    cnt = Counter()
    for p in predictions:
        h = p.get('horizon', 20)
        cnt[h] += 1
    return sum(n / h for h, n in cnt.items() if h > 0)


def classify_vs_random(ci_low, ci_high, p0=0.5):
    """根据置信区间判断与随机基准的关系"""
    if ci_low > p0:
        return 'better'
    if ci_high < p0:
        return 'worse'
    return 'indistinguishable'


def assess_reliability(n_eff, threshold=30):
    """样本可靠性判定"""
    return 'reliable' if n_eff >= threshold else 'insufficient'


def summarize(correct, total, horizon=None, p0=0.5, reliability_threshold=30):
    """汇总一组预测的评估指标

    Args:
        correct: 正确数
        total: 已验证样本数
        horizon: 预测周期（用于计算有效样本）
        p0: 随机基准（默认0.5）
        reliability_threshold: 可靠性阈值

    Returns:
        dict: accuracy / ci / n_effective / p_value / vs_random / reliability
    """
    if total <= 0:
        return {
            'total': 0, 'correct': 0, 'accuracy': None,
            'accuracy_ci_low': None, 'accuracy_ci_high': None,
            'n_effective': 0.0, 'p_value_vs_random': None,
            'vs_random': 'unknown', 'reliability': 'insufficient',
        }

    accuracy = correct / total
    n_eff = (total / horizon) if (horizon and horizon > 0) else float(total)
    # 以有效样本为分母计算区间：k_eff = accuracy * n_eff（保持 p 不变）
    k_eff = accuracy * n_eff
    ci_low, ci_high = wilson_ci(k_eff, n_eff)
    p_value = binom_p_value(k_eff, n_eff, p0=p0)

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_ci_low': ci_low,
        'accuracy_ci_high': ci_high,
        'n_effective': n_eff,
        'p_value_vs_random': p_value,
        'vs_random': classify_vs_random(ci_low, ci_high, p0=p0),
        'reliability': assess_reliability(n_eff, threshold=reliability_threshold),
    }
