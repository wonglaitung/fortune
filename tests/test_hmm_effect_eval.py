"""HMM 效果评估脚本（scripts/hmm_effect_eval.py）—— D57。"""

from scripts.hmm_effect_eval import LBL, _runs


def test_runs_basic():
    assert _runs([1, 1, 1, 0, 0, 2]) == [(1, 3), (0, 2), (2, 1)]
    assert _runs([0]) == [(0, 1)]
    assert _runs([]) == []


def test_labels():
    assert set(LBL) == {0, 1, 2}
    assert LBL[1] == "上涨" and LBL[2] == "高波偏弱"
