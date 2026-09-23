"""
Tests for PIT (point-in-time) network feature merging.

验证 merge_pit_features 的按日期对齐、前向填充与默认值填充逻辑。
"""

import pytest
import pandas as pd
import numpy as np

from ml_services.ml_trading_model import (
    merge_pit_features,
    DEFAULT_NETWORK_FEATURES,
)


def _make_stock_df(dates):
    idx = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({'Close': np.arange(len(idx), dtype=float)}, index=idx)


def test_merge_pit_forward_fill():
    """PIT 日期为步长点，应按日期前向填充到日频"""
    stock_df = _make_stock_df(['2023-06-01', '2023-06-10', '2023-06-20', '2023-06-30'])
    code_pit = {
        '2023-06-05': {'net_degree_centrality': 0.1, 'net_community_id': 3},
        '2023-06-25': {'net_degree_centrality': 0.2, 'net_community_id': 5},
    }
    out = merge_pit_features(stock_df, code_pit)

    # 06-10 应前向填充 06-05 的值；06-30 应用 06-25 的值
    assert out.loc[pd.Timestamp('2023-06-10', tz='UTC'), 'net_degree_centrality'] == 0.1
    assert out.loc[pd.Timestamp('2023-06-30', tz='UTC'), 'net_degree_centrality'] == 0.2
    assert out.loc[pd.Timestamp('2023-06-30', tz='UTC'), 'net_community_id'] == 5


def test_merge_pit_early_rows_use_defaults():
    """PIT 覆盖前的行应使用默认值"""
    stock_df = _make_stock_df(['2023-01-01', '2023-06-10'])
    code_pit = {
        '2023-06-05': {'net_degree_centrality': 0.1, 'net_community_id': 3},
    }
    out = merge_pit_features(stock_df, code_pit)

    # 01-01 在首个 PIT 日期之前 -> 默认值
    assert out.loc[pd.Timestamp('2023-01-01', tz='UTC'), 'net_community_id'] == -1
    assert out.loc[pd.Timestamp('2023-01-01', tz='UTC'), 'net_constraint'] == 1.0
    # 06-10 有 PIT 覆盖
    assert out.loc[pd.Timestamp('2023-06-10', tz='UTC'), 'net_community_id'] == 3


def test_merge_pit_empty_uses_all_defaults():
    """PIT 为空时全部使用默认值"""
    stock_df = _make_stock_df(['2023-06-01', '2023-06-02'])
    out = merge_pit_features(stock_df, {})

    for key, value in DEFAULT_NETWORK_FEATURES.items():
        assert key in out.columns
        assert (out[key] == value).all()


def test_merge_pit_preserves_index_and_existing_columns():
    """合并不应改变原有索引与列"""
    stock_df = _make_stock_df(['2023-06-01', '2023-06-02'])
    original_index = stock_df.index.copy()
    out = merge_pit_features(
        stock_df, {'2023-06-01': {'net_degree_centrality': 0.5}})

    assert out.index.equals(original_index)
    assert 'Close' in out.columns
    assert out.loc[original_index[0], 'net_degree_centrality'] == 0.5
