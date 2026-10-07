"""特征选择确定性回归测试。

背景（2026-10-07）：港股 1d 新池双跑 md5 不一致，根因是
`feature_selection_statistical` 在同分时按**输入列序**打破平分
（SelectKBest 稳定排序按列序取舍 + `set(range(N))` 建行序 +
`sort_values` 默认 quicksort 不稳定）。实测列序打乱后选中集合
对称差 22 个特征 → 双跑不可复现。

本测试锁定：选择结果必须是 `(分数, 特征名)` 的纯函数，与输入列序无关。
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from ml_services.feature_selection import feature_selection_statistical


def _make_tied_data(seed=7, n_features=1200, n_rows=4000):
    """构造在 top_k 边界处大量精确同分的数据，强制触发平分取舍。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features)
    y = (rng.rand(n_rows) > 0.5).astype(int)
    # 后 700 列完全相同 → F/MI 分数精确同分，top_k=500 必切在同分带里
    X[:, 500:] = X[:, 500:][:, :1]
    names = [f"f{i:04d}" for i in range(n_features)]
    return X, y, names


def test_same_input_twice_identical():
    """同输入两次调用必须选出完全相同的特征（顺序也相同）。"""
    X, y, names = _make_tied_data()
    s1, _ = feature_selection_statistical(X.copy(), y.copy(), names, top_k=500)
    s2, _ = feature_selection_statistical(X.copy(), y.copy(), names, top_k=500)
    assert np.array_equal(s1, s2)


def test_permuted_column_order_same_selection():
    """打乱特征列序后，选中集合与顺序必须完全不变（原缺陷复现点）。"""
    X, y, names = _make_tied_data()
    s1, _ = feature_selection_statistical(X.copy(), y.copy(), names, top_k=500)
    expected = [names[int(i)] for i in s1]

    rng = np.random.RandomState(123)
    perm = rng.permutation(len(names))
    X2 = X[:, perm]
    names2 = [names[i] for i in perm]
    s2, _ = feature_selection_statistical(X2, y.copy(), names2, top_k=500)
    got = [names2[int(i)] for i in s2]

    assert got == expected, f"列序影响了选择结果，对称差 {len(set(got) ^ set(expected))} 个特征"


def test_selection_is_descending_by_combined_score():
    """确定性修复不得破坏「按综合得分降序」这一基本语义。"""
    X, y, names = _make_tied_data(n_features=400, n_rows=2000)
    X[:, 200:] = X[:, 200:][:, :1]
    names = [f"g{i:04d}" for i in range(400)]
    s1, scores = feature_selection_statistical(X, y, names, top_k=100)
    sub = scores.head(100)
    vals = sub["Combined_Score"].values
    assert np.all(np.diff(vals) <= 1e-12), "top_k 内综合得分必须非增"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
