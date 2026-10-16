"""HMM 状态转换概率的邮件文案口径（comprehensive_analysis）—— 诚实化修正。"""

from pathlib import Path

from comprehensive_analysis import (
    hmm_empirical_duration, hmm_hazard_remaining, hmm_stability_desc)


def test_desc_none():
    assert "不可用" in hmm_stability_desc(None)


def test_desc_by_historical_percentile():
    assert "历史低位" in hmm_stability_desc(0.10)
    assert "历史中位" in hmm_stability_desc(0.50)
    assert "历史高位" in hmm_stability_desc(0.90)
    # 边界：25% → 中位；75% → 高位
    assert "历史中位" in hmm_stability_desc(0.25)
    assert "历史高位" in hmm_stability_desc(0.75)


def test_no_action_implication_in_hmm_text():
    """HMM 无 alpha（D57）⇒ 其叙事不得夹带买卖/行动暗示。"""
    src = (Path(__file__).resolve().parent.parent / "comprehensive_analysis.py").read_text(encoding="utf-8")
    for w in ("趋势明确", "可正常交易"):
        assert w not in src, w
    # 必须显式声明"只描述、不预测、不是买卖信号"
    assert "不预测涨跌" in src and "不是买卖信号" in src


def test_no_hardcoded_range_wording_in_source():
    """回归护栏：不得再用'实际范围'（暗示分位却实为硬编码阈值）。"""
    src = (Path(__file__).resolve().parent.parent / "comprehensive_analysis.py").read_text(encoding="utf-8")
    assert "实际范围" not in src
    # 必须以人话点明"看过去 vs 看未来"两个概念，且提示二者可能不一致
    assert "过去" in src and "未来" in src
    assert "别互相替代" in src


def test_empirical_duration_uses_elapsed_and_excludes_ongoing():
    # 状态 1 的已结束段：30、12；末段（25）进行中 ⇒ 排除
    seq = [0] * 5 + [1] * 30 + [0] * 8 + [1] * 12 + [2] * 4 + [1] * 25
    e10 = hmm_empirical_duration(seq, current_state=1, elapsed=10)
    assert e10["n"] == 2 and e10["median_total"] == 30
    assert e10["exp_remaining"] == 11.0           # mean([20, 2])
    e25 = hmm_empirical_duration(seq, current_state=1, elapsed=25)
    assert e25["exp_remaining"] == 5.0            # 只剩 30−25=5（12 段已不足）
    # 已持续 ≥ 所有历史段 ⇒ 无法再走（经验 0）
    e40 = hmm_empirical_duration(seq, current_state=1, elapsed=40)
    assert e40["exp_remaining"] == 0.0


def test_empirical_duration_none_when_no_completed_same_state():
    assert hmm_empirical_duration([1] * 20, current_state=1, elapsed=20) is None


def test_hazard_remaining_basic():
    # 状态 1 的已结束段都=10（6 次）+ 末段进行中 ⇒ 恒定风险率 ≈ 1/10
    seq = [2] * 3 + ([1] * 10 + [0] * 3) * 6 + [1] * 4
    r0 = hmm_hazard_remaining(seq, current_state=1, elapsed=0)
    assert r0["n"] == 6
    assert 0 < r0["hazard_now"] < 1
    assert r0["exp_remaining"] == r0["exp_total"]           # elapsed=0 时"剩余"="总"
    r5 = hmm_hazard_remaining(seq, current_state=1, elapsed=5)
    assert r5["exp_remaining"] < r0["exp_remaining"]        # 已走更久 ⇒ 剩余更少
    # 到第 9 天时同类段几乎都在这天退出 ⇒ 风险率接近 1
    r9 = hmm_hazard_remaining(seq, current_state=1, elapsed=9)
    assert r9["hazard_now"] > 0.8


def test_hazard_remaining_none_when_no_completed():
    assert hmm_hazard_remaining([1] * 20, current_state=1, elapsed=5) is None


