"""
Tests for evaluation statistics utilities.
"""

import math
import pytest

from ml_services.eval_stats import (
    wilson_ci,
    binom_p_value,
    effective_n,
    classify_vs_random,
    assess_reliability,
    summarize,
)


def test_wilson_ci_bounds():
    """Wilson CI 边界：k=0 下界为0，k=n 上界为1"""
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0
    assert 0.0 < hi < 1.0

    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_ci_symmetric_around_half():
    """k=n/2 时区间应对称且覆盖0.5"""
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert abs((0.5 - lo) - (hi - 0.5)) < 1e-9


def test_wilson_ci_narrows_with_n():
    """样本越多，区间越窄"""
    lo1, hi1 = wilson_ci(60, 100)
    lo2, hi2 = wilson_ci(600, 1000)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_effective_n_single_horizon():
    preds = [{'horizon': 20} for _ in range(100)]
    assert effective_n(preds, horizon=20) == pytest.approx(5.0)


def test_effective_n_mixed_horizons():
    preds = [{'horizon': 1}] * 10 + [{'horizon': 5}] * 10 + [{'horizon': 20}] * 20
    # 10/1 + 10/5 + 20/20 = 10 + 2 + 1 = 13
    assert effective_n(preds) == pytest.approx(13.0)


def test_classify_vs_random():
    assert classify_vs_random(0.55, 0.75) == 'better'
    assert classify_vs_random(0.25, 0.45) == 'worse'
    assert classify_vs_random(0.45, 0.65) == 'indistinguishable'


def test_assess_reliability():
    assert assess_reliability(30) == 'reliable'
    assert assess_reliability(29.9) == 'insufficient'


def test_binom_p_value_significant():
    # 90/100 明显偏离0.5
    assert binom_p_value(90, 100) < 0.01
    # 50/100 完全不显著
    assert binom_p_value(50, 100) > 0.9


def test_summarize_denominator_uses_verified_only():
    """回归测试：准确率分母必须为已验证样本数（防'分母错误'复现）"""
    s = summarize(correct=62, total=115, horizon=20)
    assert s['total'] == 115
    assert s['accuracy'] == pytest.approx(62 / 115)
    # 有效样本 = 115/20
    assert s['n_effective'] == pytest.approx(5.75)
    assert s['reliability'] == 'insufficient'


def test_summarize_handles_empty():
    s = summarize(correct=0, total=0, horizon=20)
    assert s['accuracy'] is None
    assert s['reliability'] == 'insufficient'
