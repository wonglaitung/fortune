# -*- coding: utf-8 -*-
"""A股 prediction_analysis.csv 列名对齐港股（A_STOCK_REFORM_PLAN P0.1）"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_walk_forward import PREDICTION_ANALYSIS_COLUMNS, build_prediction_analysis


def _raw_df(**overrides):
    base = {
        'fold': [1, 1],
        'Date': pd.to_datetime(['2026-01-05', '2026-01-06']),
        'Stock_Code': ['2318', 2655],
        'probability': [0.62, 0.41],
        'prediction': [1, 0],
        'actual_return': [0.03, -0.02],
        'Label': [1, 0],
        'market_layer': ['normal', 'bear'],
        'dynamic_threshold': [0.5, 0.7],
        'market_up_ratio_lag1': [0.55, 0.25],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_output_columns_match_hk_layout():
    out = build_prediction_analysis(_raw_df())
    assert out.columns.tolist() == [c for c in PREDICTION_ANALYSIS_COLUMNS if c in out.columns]
    for col in ('Fold', 'Date', 'Stock_Code', 'Predict_Prob', 'Predict_Direction',
                'Actual_Return', 'Actual_Direction', 'Is_Correct',
                'Market_Layer', 'Dynamic_Threshold'):
        assert col in out.columns


def test_direction_mapped_and_code_zfilled():
    out = build_prediction_analysis(_raw_df())
    assert out['Predict_Direction'].tolist() == ['UP', 'DOWN']
    assert out['Actual_Direction'].tolist() == ['UP', 'DOWN']
    assert out['Stock_Code'].tolist() == ['002318', '002655']
    assert out['Is_Correct'].tolist() == [True, True]


def test_code_column_renamed_when_stock_code_absent():
    # 没有 Stock_Code 列时用 Code 代替
    df = _raw_df().drop(columns=['Stock_Code'])
    df['Code'] = ['2318', '2655']
    out = build_prediction_analysis(df)
    assert 'Stock_Code' in out.columns
    assert 'Code' not in out.columns
    assert out['Stock_Code'].tolist() == ['002318', '002655']


def test_existing_up_down_strings_preserved():
    df = _raw_df()
    df['prediction'] = ['UP', 'DOWN']
    df['Label'] = ['UP', 'UP']
    out = build_prediction_analysis(df)
    assert out['Predict_Direction'].tolist() == ['UP', 'DOWN']
    assert out['Actual_Direction'].tolist() == ['UP', 'UP']
    assert out['Is_Correct'].tolist() == [True, False]


def test_missing_optional_columns_skipped():
    df = _raw_df().drop(columns=['market_layer', 'dynamic_threshold', 'market_up_ratio_lag1'])
    out = build_prediction_analysis(df)
    assert 'Market_Layer' not in out.columns
    assert 'Dynamic_Threshold' not in out.columns
    assert 'Market_Up_Ratio' not in out.columns
    assert 'Predict_Prob' in out.columns
