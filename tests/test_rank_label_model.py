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
