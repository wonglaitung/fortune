"""
Tests for Z-Score anomaly detector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from anomaly_detector.zscore_detector import ZScoreDetector


@pytest.fixture
def detector():
    """Create a Z-Score detector instance."""
    return ZScoreDetector(window_size=30, threshold=3.0)


@pytest.fixture
def price_history():
    """Create sample price history."""
    dates = pd.date_range(start='2026-01-01', periods=60, freq='D')
    prices = [100 + i * 0.1 + np.random.randn() * 2 for i in range(60)]
    return pd.Series(prices, index=dates)


def test_zscore_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector.window_size == 30
    assert detector.threshold == 3.0


def test_detect_price_anomaly_no_anomaly(detector, price_history):
    """Test normal price is not detected as anomaly."""
    current_price = price_history.iloc[-1]
    timestamp = datetime(2026, 3, 1, 10, 0, 0)
    
    result = detector.detect_price_anomaly(current_price, price_history, timestamp)
    
    assert result is None


def test_detect_price_anomaly_with_anomaly(detector, price_history):
    """Test extreme price is detected as anomaly."""
    # Create extreme price (5 standard deviations from mean)
    mean = price_history.mean()
    std = price_history.std()
    extreme_price = mean + 5 * std
    
    timestamp = datetime(2026, 3, 1, 10, 0, 0)
    
    result = detector.detect_price_anomaly(extreme_price, price_history, timestamp)
    
    assert result is not None
    assert result['type'] == 'price'
    assert result['severity'] == 'high'
    assert result['z_score'] > 4.0


def test_detect_price_anomaly_insufficient_data(detector):
    """Test detector handles insufficient data gracefully."""
    short_history = pd.Series([100, 101, 102], index=pd.date_range(start='2026-01-01', periods=3))
    current_price = 105
    timestamp = datetime(2026, 1, 4, 10, 0, 0)
    
    result = detector.detect_price_anomaly(current_price, short_history, timestamp)
    
    assert result is None


def test_detect_volume_anomaly(detector):
    """Test volume anomaly detection."""
    dates = pd.date_range(start='2026-01-01', periods=60, freq='D')
    volumes = [1000000 + i * 10000 + np.random.randn() * 200000 for i in range(60)]
    volume_history = pd.Series(volumes, index=dates)
    
    # Create extreme volume (5 standard deviations to ensure high severity)
    mean = volume_history.mean()
    std = volume_history.std()
    extreme_volume = mean + 5 * std
    
    timestamp = datetime(2026, 3, 1, 10, 0, 0)
    
    result = detector.detect_volume_anomaly(extreme_volume, volume_history, timestamp)
    
    assert result is not None
    assert result['type'] == 'volume'
    assert result['severity'] == 'high'


def test_severity_classification(detector):
    """Test severity level classification."""
    # High severity (|Z-Score| >= 4)
    assert detector._get_severity(4.5) == 'high'
    assert detector._get_severity(-4.5) == 'high'
    
    # Medium severity (3 <= |Z-Score| < 4)
    assert detector._get_severity(3.5) == 'medium'
    assert detector._get_severity(-3.5) == 'medium'
    
    # Low severity (|Z-Score| < 3)
    assert detector._get_severity(2.5) == 'low'
    assert detector._get_severity(-2.5) == 'low'