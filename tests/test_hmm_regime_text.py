"""HMM 状态转换概率的邮件文案口径（comprehensive_analysis）—— 诚实化修正。"""

from pathlib import Path

from comprehensive_analysis import hmm_stability_desc


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
