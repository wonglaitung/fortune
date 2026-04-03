"""
Tests for anomaly cache module.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from anomaly_detector.cache import AnomalyCache


@pytest.fixture
def temp_cache_file():
    """Create a temporary cache file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    yield temp_file
    if os.path.exists(temp_file):
        os.unlink(temp_file)


def test_cache_initialization(temp_cache_file):
    """Test cache initializes correctly."""
    cache = AnomalyCache(temp_cache_file)
    assert cache.get_cache_size() == 0
    assert os.path.exists(temp_cache_file)


def test_add_anomaly(temp_cache_file):
    """Test adding anomaly to cache."""
    cache = AnomalyCache(temp_cache_file)
    timestamp = datetime(2026, 4, 3, 10, 0, 0)
    
    cache.add(
        anomaly_type='price',
        timestamp=timestamp,
        severity='high',
        z_score=4.5
    )
    
    assert cache.get_cache_size() == 1
    assert cache.exists('price', timestamp.date())


def test_cache_exists_check(temp_cache_file):
    """Test checking if anomaly exists in cache."""
    cache = AnomalyCache(temp_cache_file)
    timestamp = datetime(2026, 4, 3, 10, 0, 0)
    
    # Should not exist initially
    assert not cache.exists('price', timestamp.date())
    
    # Add anomaly
    cache.add(
        anomaly_type='price',
        timestamp=timestamp,
        severity='high',
        z_score=4.5
    )
    
    # Should exist now
    assert cache.exists('price', timestamp.date())
    
    # Different date should not exist
    next_day = timestamp.replace(day=4)
    assert not cache.exists('price', next_day.date())


def test_cleanup_expired_entries(temp_cache_file):
    """Test cleaning up expired cache entries."""
    cache = AnomalyCache(temp_cache_file)
    
    # Add old anomaly (older than 48 hours)
    old_timestamp = datetime.now() - timedelta(hours=50)
    cache.add(
        anomaly_type='price',
        timestamp=old_timestamp,
        severity='high',
        z_score=4.5
    )
    
    # Add recent anomaly
    recent_timestamp = datetime.now() - timedelta(hours=1)
    cache.add(
        anomaly_type='volume',
        timestamp=recent_timestamp,
        severity='medium',
        z_score=3.2
    )
    
    assert cache.get_cache_size() == 2
    
    # Cleanup
    cache.cleanup_expired(max_age_hours=48)
    
    # Only recent anomaly should remain
    assert cache.get_cache_size() == 1
    assert cache.exists('volume', recent_timestamp.date())