# -*- coding: utf-8 -*-
"""P4.1 学习器 A/B 开关（A股，决策点3：三周期全做）

AStockTradingModel 唯一学习器 = LightGBM（同 TSCV 口径、样本权重、
准确率写 a_stock_lightgbm_{h}d 不覆盖 catboost 键），walk-forward 目录名带学习器。
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from a_stock_ml_model import AStockTradingModel
import a_stock_walk_forward as awf


def _tiny_xy(n=600, k=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(int)
    w = np.where(np.arange(n) % 3 == 0, 3.0, 1.0)
    cols = [f'f{i}' for i in range(k)]
    return X, y, w, cols


def test_learner_defaults_to_lightgbm():
    m = AStockTradingModel(horizon=20)
    assert getattr(m, 'learner', '') == 'lightgbm'


def test_lightgbm_train_and_predict():
    X, y, w, cols = _tiny_xy()
    m = AStockTradingModel(horizon=20)
    m.label_mode = 'absolute'   # 分类口径用于验证 classifier 分支
    m.feature_columns = cols
    m._save_accuracy = lambda *a, **k: None   # 不污染真实 model_accuracy.json

    res = m.train_with_weights(X, y, sample_weights=w, horizon=20)
    assert m.learner == 'lightgbm'
    assert type(m.model).__name__ == 'LGBMClassifier'
    assert set(res) >= {'accuracy', 'f1'}

    proba = m.predict_proba(pd.DataFrame(X, columns=cols))
    assert proba.shape == (len(y), 2)
    assert np.isfinite(proba[:, 1]).all()
    assert ((proba[:, 1] >= 0) & (proba[:, 1] <= 1)).all()


def test_rank_label_uses_regressor():
    """D59 rank 连续标签 ⇒ LGBMRegressor（与港股同源修复）。"""
    X, _, w, cols = _tiny_xy()
    y_cont = np.random.default_rng(1).random(len(X))
    m = AStockTradingModel(horizon=20)
    m.label_mode = 'rank'
    m.feature_columns = cols
    m._save_accuracy = lambda *a, **k: None
    m.train_with_weights(X, y_cont, sample_weights=w, horizon=20)
    assert type(m.model).__name__ == 'LGBMRegressor'
    proba = m.predict_proba(pd.DataFrame(X, columns=cols))
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba[:, 1]).all()


def test_validator_learner_is_fixed_lightgbm():
    v = awf.AStockWalkForwardValidator(horizon=20)
    assert v.learner == 'lightgbm'
