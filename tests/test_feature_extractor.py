"""
Tests for feature extractor.
"""

import pytest
import pandas as pd
import numpy as np
from anomaly_detector.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor():
    """Create a feature extractor instance."""
    return FeatureExtractor()


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start='2026-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = {
        'Open': np.cumsum(np.random.randn(100)) + 100,
        'High': np.cumsum(np.random.randn(100)) + 102,
        'Low': np.cumsum(np.random.randn(100)) + 98,
        'Close': np.cumsum(np.random.randn(100)) + 100,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_feature_extractor_initialization(extractor):
    """Test extractor initializes correctly."""
    assert len(extractor.feature_names) == 10
    assert 'return_rate' in extractor.feature_names
    assert 'rsi' in extractor.feature_names


def test_extract_features_basic(extractor, sample_df):
    """Test basic feature extraction."""
    features, timestamps = extractor.extract_features(sample_df)
    
    assert features is not None
    assert not features.empty
    assert len(features) == len(sample_df)
    assert len(timestamps) == len(sample_df)
    assert list(features.columns) == extractor.feature_names


def test_extract_features_with_indicators(extractor, sample_df):
    """Test feature extraction with pre-calculated indicators."""
    # Add pre-calculated indicators
    sample_df['RSI'] = np.random.uniform(0, 100, 100)
    sample_df['MACD'] = np.random.randn(100)
    sample_df['MACD_signal'] = np.random.randn(100)
    sample_df['BB_position'] = np.random.uniform(0, 1, 100)
    sample_df['MA20'] = sample_df['Close'].rolling(20).mean()
    sample_df['MA50'] = sample_df['Close'].rolling(50).mean()
    
    features, timestamps = extractor.extract_features(sample_df)
    
    assert 'rsi' in features.columns
    assert 'macd' in features.columns
    assert 'bb_position' in features.columns
    assert 'ma20_diff' in features.columns


def test_extract_features_empty_dataframe(extractor):
    """Test extractor handles empty DataFrame."""
    empty_df = pd.DataFrame()
    features, timestamps = extractor.extract_features(empty_df)
    
    assert features.empty
    assert len(timestamps) == 0


def test_feature_names_are_correct(extractor):
    """Test feature names are as expected."""
    expected_names = [
        'return_rate',
        'volatility_20d',
        'volume_ratio',
        'rsi',
        'macd',
        'macd_signal',
        'bb_position',
        'ma20_diff',
        'ma50_diff',
        'ma20_ma50_diff'
    ]
    
    assert extractor.feature_names == expected_names