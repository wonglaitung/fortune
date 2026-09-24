#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练并保存生产 LightGBM 20d 模型（复用 CatBoost 管线，learner=lightgbm）

产出：data/ml_trading_model_lightgbm_20d.pkl（comprehensive_analysis 优先加载）
用法：python3 scripts/train_lightgbm_20d.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import CatBoostModel
from config import WATCHLIST


def main():
    codes = list(WATCHLIST.keys())
    print(f"股票数量: {len(codes)}")
    m = CatBoostModel(class_weight='balanced')
    m.learner = 'lightgbm'
    m.train(codes, start_date='2020-10-01', end_date=None, horizon=20,
            use_feature_selection=True)
    out = 'data/ml_trading_model_lightgbm_20d.pkl'
    m.save_model(out)
    print(f"✅ 生产 LightGBM 20d 模型已保存: {out}")


if __name__ == '__main__':
    main()