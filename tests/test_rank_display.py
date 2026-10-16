"""V4 展示层（ml_services/rank_display.py）—— D59 P3。"""

import pandas as pd

from ml_services.rank_display import (build_rows, direction_label, render_html,
                                      render_text, to_percentile)


def test_to_percentile_monotonic():
    p = to_percentile([1.0, 2.0, 3.0, 4.0])
    assert list(p) == [0.25, 0.5, 0.75, 1.0]


def test_direction_labels():
    assert direction_label(0.9) == "偏强"
    assert direction_label(0.1) == "偏弱"
    assert direction_label(0.5) == "中性"


def test_build_rows_sorted_and_labeled():
    rows = build_rows([0.1, 0.9, 0.5, 0.2], ["A.HK", "B.HK", "C.HK", "D.HK"])
    assert rows[0].code == "B.HK" and rows[0].label == "偏强"
    assert rows[-1].code == "A.HK" and rows[-1].label == "偏弱"
    assert [r.pct for r in rows] == sorted([r.pct for r in rows], reverse=True)


def test_render_has_no_probability_word():
    rows = build_rows([0.2, 0.8], ["A.HK", "B.HK"])
    for txt in (render_text(rows, 5), render_html(rows, 5)):
        assert "相对强弱" in txt
        assert "上涨概率" not in txt     # 展示层不得出现概率措辞
