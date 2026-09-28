# -*- coding: utf-8 -*-
"""P4.1 学习器 A/B 开关（A股，决策点3：三周期全做）

AStockTradingModel(learner='lightgbm') 走 LightGBM 路径（同 TSCV 口径、样本权重、
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


def test_learner_defaults_to_catboost():
    m = AStockTradingModel(horizon=20)
    assert getattr(m, 'learner', 'catboost') == 'catboost'


def test_lightgbm_train_and_predict():
    X, y, w, cols = _tiny_xy()
    m = AStockTradingModel(horizon=20, learner='lightgbm')
    m.feature_columns = cols
    m._save_accuracy = lambda *a, **k: None   # 不污染真实 model_accuracy.json

    res = m.train_with_weights(X, y, sample_weights=w, horizon=20)
    assert m.learner == 'lightgbm'
    assert type(m.model).__name__ == 'LGBMClassifier'
    assert m.catboost_model is m.model          # 共用句柄，feature_importance 无需改
    assert set(res) >= {'accuracy', 'f1'}

    proba = m.predict_proba(pd.DataFrame(X, columns=cols))
    assert proba.shape == (len(y), 2)
    assert np.isfinite(proba[:, 1]).all()
    assert ((proba[:, 1] >= 0) & (proba[:, 1] <= 1)).all()


def test_accuracy_key_separated_by_learner(tmp_path, monkeypatch):
    """准确率写 a_stock_lightgbm_{h}d，不覆盖 a_stock_catboost_{h}d"""
    monkeypatch.chdir(tmp_path)
    os.makedirs('data', exist_ok=True)

    m = AStockTradingModel(horizon=20, learner='lightgbm')
    m._save_accuracy(0.51, 0.01, 0.49, 0.01, 20)

    m2 = AStockTradingModel(horizon=20, learner='catboost')
    m2._save_accuracy(0.55, 0.01, 0.53, 0.01, 20)

    data = json.load(open('data/model_accuracy.json'))
    assert 'a_stock_lightgbm_20d' in data
    assert 'a_stock_catboost_20d' in data
    assert data['a_stock_lightgbm_20d']['model_type'] == 'a_stock_lightgbm'
    assert data['a_stock_catboost_20d']['model_type'] == 'a_stock_catboost'


def test_validator_forwards_learner():
    v = awf.AStockWalkForwardValidator(horizon=20, learner='lightgbm')
    assert v.learner == 'lightgbm'
    assert awf.AStockWalkForwardValidator(horizon=20).learner == 'catboost'
