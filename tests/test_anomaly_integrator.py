"""
Tests for anomaly integrator.
"""

import pytest
from datetime import datetime
from anomaly_detector.anomaly_integrator import AnomalyIntegrator
from anomaly_detector.cache import AnomalyCache


@pytest.fixture
def cache():
    """Create a test cache instance."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    yield AnomalyCache(temp_file)
    import os
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def integrator(cache):
    """Create an anomaly integrator instance."""
    return AnomalyIntegrator(cache)


def test_integrator_initialization(integrator):
    """Test integrator initializes correctly."""
    assert integrator.cache is not None


def test_integrate_no_anomalies(integrator):
    """Test integration with no anomalies."""
    result = integrator.integrate([], [], datetime.now())
    
    assert result['has_anomaly'] == False
    assert result['anomalies'] == []
    assert result['severity'] is None


def test_integrate_zscore_anomalies(integrator):
    """Test integration with Z-Score anomalies."""
    timestamp = datetime(2026, 4, 3, 10, 0, 0)
    
    anomalies = [
        {
            'type': 'price',
            'timestamp': timestamp,
            'severity': 'high',
            'z_score': 4.5
        }
    ]
    
    result = integrator.integrate(anomalies, [], datetime.now())
    
    assert result['has_anomaly'] == True
    assert len(result['anomalies']) == 1
    assert result['severity'] == 'high'


def test_integrate_with_deduplication(integrator):
    """Test duplicate anomaly detection."""
    timestamp = datetime(2026, 4, 3, 10, 0, 0)
    
    # First integration
    anomalies = [
        {
            'type': 'price',
            'timestamp': timestamp,
            'severity': 'high',
            'z_score': 4.5
        }
    ]
    
    result1 = integrator.integrate(anomalies, [], datetime.now())
    assert len(result1['anomalies']) == 1
    
    # Second integration (should skip duplicate)
    result2 = integrator.integrate(anomalies, [], datetime.now())
    assert len(result2['anomalies']) == 0


def test_integrate_isolation_forest_anomalies(integrator):
    """Test integration with Isolation Forest anomalies."""
    timestamp = datetime(2026, 4, 3, 10, 0, 0)
    
    anomalies = [
        {
            'timestamp': timestamp,
            'anomaly_score': -0.8,
            'severity': 'high',
            'features': {},
            'detection_method': 'isolation_forest',
            'type': 'isolation_forest'
        }
    ]
    
    result = integrator.integrate([], anomalies, datetime.now())
    
    assert result['has_anomaly'] == True
    assert len(result['anomalies']) == 1
    assert result['severity'] == 'high'


def test_overall_severity_high(integrator):
    """Test overall severity calculation with high severity anomaly."""
    anomalies = [
        {'severity': 'high'},
        {'severity': 'low'}
    ]
    
    severity = integrator._get_overall_severity(anomalies)
    assert severity == 'high'


def test_overall_severity_medium(integrator):
    """Test overall severity calculation with medium severity anomaly."""
    anomalies = [
        {'severity': 'medium'},
        {'severity': 'low'}
    ]
    
    severity = integrator._get_overall_severity(anomalies)
    assert severity == 'medium'


def test_overall_severity_low(integrator):
    """Test overall severity calculation with only low severity anomalies."""
    anomalies = [
        {'severity': 'low'},
        {'severity': 'low'}
    ]
    
    severity = integrator._get_overall_severity(anomalies)
    assert severity == 'low'