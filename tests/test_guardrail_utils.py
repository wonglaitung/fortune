"""月度护栏工具：功效/MDE（D4 件：功效声明默认化）。"""

import math

from ml_services.monthly_guardrail import min_detectable_ir


def test_mde_formula():
    # n=39, h=20 → 2.8·√(252/780) ≈ 1.59
    np_expected = 2.8 * math.sqrt(252.0 / (20 * 39))
    assert abs(min_detectable_ir(39, 20) - np_expected) < 1e-9


def test_mde_decreases_with_more_periods():
    assert min_detectable_ir(156, 20) < min_detectable_ir(39, 20)


def test_mde_nan_when_no_periods():
    assert math.isnan(min_detectable_ir(0, 20))
