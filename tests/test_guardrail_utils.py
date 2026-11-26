"""月度护栏工具：功效/MDE（D4 件：功效声明默认化）+ 同一性取源。"""

import math

from ml_services.monthly_guardrail import min_detectable_ir, latest_pred


def test_mde_formula():
    # n=39, h=20 → 2.8·√(252/780) ≈ 1.59
    np_expected = 2.8 * math.sqrt(252.0 / (20 * 39))
    assert abs(min_detectable_ir(39, 20) - np_expected) < 1e-9


def test_mde_decreases_with_more_periods():
    assert min_detectable_ir(156, 20) < min_detectable_ir(39, 20)


def test_mde_nan_when_no_periods():
    assert math.isnan(min_detectable_ir(0, 20))


def test_latest_pred_prefers_lgbm_then_catboost(monkeypatch):
    from ml_services import monthly_guardrail as mg
    lgbm = 'output/fake_lgbm_20d/prediction_analysis.csv'
    cat = 'output/fake_catboost_20d/prediction_analysis.csv'

    def glob_lgbm(p):
        return [lgbm] if 'lgbm' in p else ([] if 'catboost' in p else [])

    def glob_cat(p):
        return [cat] if 'catboost' in p else []

    monkeypatch.setattr(mg.glob, 'glob', glob_lgbm)
    monkeypatch.setattr(mg.os.path, 'getmtime', lambda p: 1)
    assert latest_pred(20) == lgbm
    monkeypatch.setattr(mg.glob, 'glob', glob_cat)
    assert latest_pred(20) == cat
