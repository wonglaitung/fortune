# -*- coding: utf-8 -*-
"""学习器与标签口径的契约：rank（连续）标签必须走回归器。

背景（2026-10-25）：D59 把个股标签改为横截面百分位 rank 后，
`_train_lightgbm` 仍无条件用 `LGBMClassifier(objective='binary')`，
在 310 池 A/B 上 **40/40 折全失败**：
`Unknown label type: continuous ... trying to fit a classifier`。
本测试锁死「rank ⇒ LGBMRegressor(objective='regression')」。
"""

import sys
import types

import numpy as np
import pandas as pd

import ml_services.ml_trading_model as m


def _bare(reg: bool):
    obj = m.TradingModel.__new__(m.TradingModel)
    obj.label_mode = "rank" if reg else "absolute"
    obj.horizon = 20
    obj.feature_columns = ["f1", "f2"]
    obj.categorical_encoders = {}
    obj.learner = "lightgbm"
    return obj


def _fake_lgb(used):
    class _Reg:
        def __init__(self, **kw):
            used.append(("reg", kw))
            self.n_estimators = kw.get("n_estimators")

        def fit(self, X, y):
            assert np.asarray(y).dtype.kind == "f", "regression 应吃连续标签"
            return self

    class _Clf:
        def __init__(self, **kw):
            used.append(("clf", kw))
            self.n_estimators = kw.get("n_estimators")

        def fit(self, X, y):
            return self

    return types.SimpleNamespace(LGBMRegressor=_Reg, LGBMClassifier=_Clf)


def test_rank_label_uses_regressor(monkeypatch):
    used = []
    monkeypatch.setitem(sys.modules, "lightgbm", _fake_lgb(used))
    X = np.random.RandomState(0).rand(20, 2)
    y = np.random.RandomState(1).rand(20)  # 连续（rank 分位）
    _bare(True)._train_lightgbm(pd.DataFrame(X), False, None, 20, X, y)
    assert used and used[0][0] == "reg", f"rank 标签误用分类器：{used}"
    assert used[0][1]["objective"] == "regression"


def test_absolute_label_uses_classifier(monkeypatch):
    used = []
    monkeypatch.setitem(sys.modules, "lightgbm", _fake_lgb(used))
    X = np.random.RandomState(0).rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)
    _bare(False)._train_lightgbm(pd.DataFrame(X), False, None, 20, X, y)
    assert used and used[0][0] == "clf", f"absolute 标签应保留分类器：{used}"
    assert used[0][1]["objective"] == "binary"
