"""
Tests for Isolation Forest anomaly detector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detector.isolation_forest_detector import IsolationForestDetector


@pytest.fixture
def detector():
    """Create an Isolation Forest detector instance."""
    return IsolationForestDetector(contamination=0.05, random_state=42)


@pytest.fixture
def sample_features():
    """Create sample feature matrix."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate normal data
    normal_data = np.random.randn(n_samples, n_features)
    
    # Add some anomalies
    anomalies = np.random.randn(5, n_features) * 5  # Extreme values
    
    all_data = np.vstack([normal_data, anomalies])
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    dates = pd.date_range(start='2026-01-01', periods=len(all_data))
    
    df = pd.DataFrame(all_data, columns=feature_names, index=dates)
    
    return df


def test_isolation_forest_initialization(detector):
    """Test detector initializes correctly."""
    assert detector.contamination == 0.05
    assert detector.random_state == 42
    assert detector.model is None


def test_train_model(detector, sample_features):
    """Test model training."""
    detector.train(sample_features)
    
    assert detector.model is not None
    assert hasattr(detector.model, 'predict')


def test_train_empty_features(detector):
    """Test training with empty features."""
    empty_features = pd.DataFrame()
    
    detector.train(empty_features)
    
    assert detector.model is None


def test_detect_anomalies_without_training(detector, sample_features):
    """Test detection without training raises error."""
    timestamps = sample_features.index.tolist()
    
    anomalies = detector.detect_anomalies(sample_features, timestamps)
    
    assert anomalies == []


def test_detect_anomalies_with_training(detector, sample_features):
    """Test anomaly detection with trained model."""
    # Train model
    detector.train(sample_features)
    
    # Detect anomalies
    timestamps = sample_features.index.tolist()
    anomalies = detector.detect_anomalies(sample_features, timestamps)
    
    assert isinstance(anomalies, list)
    assert len(anomalies) > 0  # Should detect the anomalies we added


def test_detect_anomalies_lookback_filter(detector, sample_features):
    """Test lookback days filter."""
    # Train model
    detector.train(sample_features)
    
    # Detect with small lookback (should return fewer anomalies)
    timestamps = sample_features.index.tolist()
    recent_anomalies = detector.detect_anomalies(sample_features, timestamps, lookback_days=1)
    
    # All anomalies with larger lookback
    all_anomalies = detector.detect_anomalies(sample_features, timestamps, lookback_days=30)
    
    assert len(recent_anomalies) <= len(all_anomalies)


def test_severity_classification(detector, sample_features):
    """Test severity classification based on anomaly score."""
    # Train model
    detector.train(sample_features)
    
    # Test different score ranges
    assert detector._get_severity(-0.9) == 'high'
    assert detector._get_severity(-0.6) == 'medium'
    assert detector._get_severity(-0.4) == 'low'
    assert detector._get_severity(-0.2) == 'low'