"""D59 模型层切换：rank 标签/回归/持久化 契约。"""

import os
import tempfile

import numpy as np
import pandas as pd

from ml_services.ml_trading_model import CatBoostModel


def test_create_label_rank_is_placeholder():
    m = CatBoostModel()
    df = pd.DataFrame({"Close": np.linspace(100, 120, 60)},
                      index=pd.bdate_range("2020-01-01", periods=60))
    out = m.feature_engineer.create_label(df.copy(), horizon=5, label_mode="rank")
    assert "Future_Return" in out.columns
    assert out["Label"].isna().all()               # 占位：全池合并后再算分位
    assert out["Label_Mode"].iloc[0] == "rank"


def test_get_feature_columns_excludes_label_mode():
    m = CatBoostModel()
    df = pd.DataFrame({"Close": [1.0], "Label": [0], "Label_Mode": ["rank"],
                       "Label_Threshold": [0.0], "RSI_14": [0.5], "Volume": [1.0]})
    cols = m.get_feature_columns(df)
    assert "Label_Mode" not in cols        # 潜在 bug：曾被当特征（全量训练时 KeyError）
    assert "Label" not in cols and "Label_Threshold" not in cols


def test_save_load_preserves_label_mode():
    m = CatBoostModel()
    m.label_mode = "rank"
    m.feature_columns = ["a"]
    m.actual_n_estimators = 10
    m.horizon = 5
    m.model_type = "catboost"
    m.categorical_encoders = {}
    m.community_ids = None
    m.catboost_model = None
    m.model = None
    m.learner = "catboost"
    p = os.path.join(tempfile.mkdtemp(), "m.pkl")
    m.save_model(p)
    m2 = CatBoostModel()
    m2.load_model(p)
    assert m2.label_mode == "rank"


def test_a_stock_rank_trainer_smoke():
    from a_stock_ml_model import AStockTradingModel
    m = AStockTradingModel(horizon=5)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5))
    y = rng.random(200)
    r = m._train_with_weights_rank(X, y, None, 5)
    assert m.catboost_model is not None
    assert r["accuracy"] == 0.0


def test_walk_forward_accepts_rank_label_mode():
    """D60：WF 引擎必须接受 rank 口径（此前 choices 只有 absolute/relative_hsi，
    导致生产 rank 管线无 WF 跑法、无法过 C 闸）。"""
    from ml_services.walk_forward_validation import WalkForwardValidator
    v = WalkForwardValidator(horizon=5, label_mode="rank")
    assert v.label_mode == "rank"
    # CLI 契约：choices 必须含 rank
    import inspect
    src = inspect.getsource(
        __import__("ml_services.walk_forward_validation", fromlist=["x"]))
    assert "choices=['absolute', 'relative_hsi', 'rank']" in src


def test_predict_proba_rank_returns_clipped_score():
    """rank 口径下 predict_proba 必须返回 [:,1]=clip(score,0,1)（回归器无 predict_proba）。"""
    from catboost import CatBoostRegressor
    m = CatBoostModel()
    m.label_mode = "rank"
    m.feature_columns = ["a", "b"]
    m.categorical_encoders = {}
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 2))
    y = rng.random(60)
    m.catboost_model = CatBoostRegressor(n_estimators=5, verbose=False, allow_writing_files=False)
    m.catboost_model.fit(X, y)
    df = pd.DataFrame(X, columns=["a", "b"])
    p = m.predict_proba(df)
    assert p.shape == (60, 2)
    assert np.allclose(p[:, 0] + p[:, 1], 1.0)
    assert p[:, 1].min() >= 0.0 and p[:, 1].max() <= 1.0


def test_cli_defaults_to_rank():
    """2026-10-22：训练/验证 CLI 默认口径必须是 rank；旧 absolute 须显式指定。"""
    import inspect
    import a_stock_ml_model
    import ml_services.ml_trading_model as hk
    import ml_services.walk_forward_validation as wf

    hk_src = inspect.getsource(hk)
    a_src = inspect.getsource(a_stock_ml_model)
    wf_src = inspect.getsource(wf)

    assert "default=os.environ.get('LABEL_MODE', 'rank')" in hk_src
    assert "default=os.environ.get('LABEL_MODE', 'rank')" in a_src
    assert "default='rank'" in wf_src

    from a_stock_ml_model import AStockTradingModel
    from ml_services.walk_forward_validation import WalkForwardValidator
    assert AStockTradingModel(horizon=5).label_mode == 'rank'
    assert WalkForwardValidator(horizon=5).label_mode == 'rank'
