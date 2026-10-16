"""V4 排序标签/分数 的**同一性契约**（D59 P4）。

锁定两个易漂移的约定：
1. **标签**：rank = 当日 `fwd` 的横截面百分位（`groupby(date).rank(pct=True)`）；
2. **输出**：回归路径产出连续 `score`（**不是** `Predict_Prob`/概率）；
3. **确定性**：同输入重复拟合结果一致（同机 bit 级）。
"""

import numpy as np
import pandas as pd

import scripts.v4_pipeline as vp
from scripts.label_ablation import _fit_predict


def test_rank_label_is_cross_sectional_percentile():
    df = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 4,
                       "code": list("abcd") * 2,
                       "fwd": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]})
    r = df.groupby("date")["fwd"].rank(pct=True)
    assert r.tolist()[:4] == [0.25, 0.5, 0.75, 1.0]
    assert r.tolist()[4:] == [1.0, 0.75, 0.5, 0.25]


def test_pipeline_emits_score_not_probability():
    rng = np.random.default_rng(0)
    p = pd.concat([
        pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=800),
                      "code": f"{c:03d}.HK", "fwd": rng.normal(0, 0.05, 800),
                      **{f: rng.normal(0, 1, 800) for f in vp.FEATS}})
        for c in range(22)
    ], ignore_index=True)
    p["rank"] = p.groupby("date")["fwd"].rank(pct=True)
    pred = vp.walk_forward_scores(p, 5)
    assert "score" in pred.columns
    assert "Predict_Prob" not in pred.columns


def test_fit_predict_deterministic():
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.normal(0, 1, (400, 11)), rng.normal(0, 1, (50, 11))
    y = rng.normal(0, 1, 400)
    a = _fit_predict(Xtr, y, Xte, "reg")
    b = _fit_predict(Xtr, y, Xte, "reg")
    assert np.array_equal(a, b)
