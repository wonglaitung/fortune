"""HMM 状态转换概率的邮件文案口径（comprehensive_analysis）—— 诚实化修正。"""

from pathlib import Path

from comprehensive_analysis import hmm_empirical_duration, hmm_stability_desc


def test_desc_none():
    assert "不可用" in hmm_stability_desc(None)


def test_desc_by_historical_percentile():
    assert "历史低位" in hmm_stability_desc(0.10)
    assert "历史中位" in hmm_stability_desc(0.50)
    assert "历史高位" in hmm_stability_desc(0.90)
    # 边界：25% → 中位；75% → 高位
    assert "历史中位" in hmm_stability_desc(0.25)
    assert "历史高位" in hmm_stability_desc(0.75)


def test_no_hardcoded_range_wording_in_source():
    """回归护栏：不得再用'实际范围'（暗示分位却实为硬编码阈值）。"""
    src = (Path(__file__).resolve().parent.parent / "comprehensive_analysis.py").read_text(encoding="utf-8")
    assert "实际范围" not in src
    # 必须点明"回看 vs 前瞻"两个概念的差异
    assert "回看" in src and "前瞻" in src


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

