# Ethereum Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated anomaly detection to the existing crypto_email.py system using a hybrid approach with Z-Score (hourly) and Isolation Forest (daily) detection layers.

**Architecture:** Two-layer detection system - Layer 1 uses Moving Z-Score for real-time filtering, Layer 2 uses Isolation Forest for deep multi-dimensional analysis. Results integrated with existing technical analysis and sent via email.

**Tech Stack:** Python, scikit-learn, pandas, numpy, yfinance, pytest

---

## File Structure Overview

```
/data/fortune/
├── crypto_email.py                    # Modify: Add anomaly detection integration
├── anomaly_detector/                  # Create: New Python package
│   ├── __init__.py                    # Create: Package init
│   ├── zscore_detector.py            # Create: Z-Score detection
│   ├── isolation_forest_detector.py  # Create: Isolation Forest detection
│   ├── feature_extractor.py          # Create: Feature extraction
│   ├── anomaly_integrator.py         # Create: Result integration
│   └── cache.py                      # Create: Cache management
└── tests/
    ├── test_zscore_detector.py       # Create: Z-Score tests
    ├── test_isolation_forest.py      # Create: Isolation Forest tests
    ├── test_feature_extractor.py     # Create: Feature extraction tests
    ├── test_anomaly_integrator.py    # Create: Integration tests
    ├── test_cache.py                 # Create: Cache tests
    └── __init__.py                    # Create: Test package init
```

---

## Task 1: Create anomaly_detector package structure

**Files:**
- Create: `anomaly_detector/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create anomaly_detector package init file**

```python
# anomaly_detector/__init__.py
"""
Anomaly detection package for Ethereum price monitoring.

This package provides two-layer anomaly detection:
- Layer 1: Real-time Z-Score detection (hourly)
- Layer 2: Isolation Forest deep analysis (daily)
"""

__version__ = "1.0.0"
```

- [ ] **Step 2: Verify package structure**

Run: `python3 -c "import sys; sys.path.insert(0, '.'); from anomaly_detector import __version__; print(__version__)"`
Expected: `1.0.0`

- [ ] **Step 3: Create tests package init file**

```python
# tests/__init__.py
"""
Test package for anomaly detection system.
"""
```

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/__init__.py tests/__init__.py
git commit -m "feat: create anomaly_detector package structure"
```

---

## Task 2: Implement cache management module

**Files:**
- Create: `anomaly_detector/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write cache module with file-based storage**

```python
# anomaly_detector/cache.py
"""
Cache management for anomaly detection.
Stores reported anomalies to avoid duplicate alerts.
"""

import json
import fcntl
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyCache:
    """Manages anomaly cache with file-based storage and locking."""
    
    def __init__(self, cache_file: str = "data/anomaly_cache.json"):
        """
        Initialize cache.
        
        Args:
            cache_file: Path to cache file (default: data/anomaly_cache.json)
        """
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}
        self._ensure_cache_dir()
        self._load_cache()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _load_cache(self):
        """Load cache from file with file locking."""
        if not os.path.exists(self.cache_file):
            self.cache = {}
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock
                self.cache = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to file with file locking."""
        try:
            with open(self.cache_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                json.dump(self.cache, f, indent=2, default=str)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def add(self, anomaly_type: str, timestamp: datetime, severity: str, **metadata):
        """
        Add anomaly to cache.
        
        Args:
            anomaly_type: Type of anomaly (e.g., 'price', 'volume')
            timestamp: When the anomaly was detected
            severity: Severity level ('high', 'medium', 'low')
            **metadata: Additional metadata (z_score, anomaly_score, etc.)
        """
        date_key = timestamp.strftime('%Y-%m-%d')
        cache_key = f"{anomaly_type}_{date_key}"
        
        self.cache[cache_key] = {
            'timestamp': timestamp.isoformat(),
            'severity': severity,
            **metadata
        }
        
        self._save_cache()
        logger.info(f"Added to cache: {cache_key}")
    
    def exists(self, anomaly_type: str, date: datetime) -> bool:
        """
        Check if anomaly exists in cache.
        
        Args:
            anomaly_type: Type of anomaly
            date: Date to check
        
        Returns:
            True if anomaly exists in cache, False otherwise
        """
        date_key = date.strftime('%Y-%m-%d')
        cache_key = f"{anomaly_type}_{date_key}"
        return cache_key in self.cache
    
    def cleanup_expired(self, max_age_hours: int = 48):
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup (default: 48)
        """
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            timestamp_str = entry.get('timestamp', '')
            if not timestamp_str:
                expired_keys.append(key)
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age = now - timestamp
                
                if age > timedelta(hours=max_age_hours):
                    expired_keys.append(key)
            except Exception as e:
                logger.warning(f"Failed to parse timestamp for {key}: {e}")
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._save_cache()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
```

- [ ] **Step 2: Write cache tests**

```python
# tests/test_cache.py
"""
Tests for anomaly cache module.
"""

import pytest
import os
import tempfile
from datetime import datetime
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
```

- [ ] **Step 3: Run cache tests**

Run: `python3 -m pytest tests/test_cache.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/cache.py tests/test_cache.py
git commit -m "feat: implement cache management module"
```

---

## Task 3: Implement Z-Score detector

**Files:**
- Create: `anomaly_detector/zscore_detector.py`
- Create: `tests/test_zscore_detector.py`

- [ ] **Step 1: Write Z-Score detector with moving window**

```python
# anomaly_detector/zscore_detector.py
"""
Z-Score anomaly detector for real-time monitoring.
Uses moving window to detect price and volume anomalies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ZScoreDetector:
    """Detects anomalies using Moving Z-Score method."""
    
    def __init__(self, window_size: int = 30, threshold: float = 3.0):
        """
        Initialize Z-Score detector.
        
        Args:
            window_size: Rolling window size in days (default: 30)
            threshold: Z-Score threshold for anomaly detection (default: 3.0)
        """
        self.window_size = window_size
        self.threshold = threshold
    
    def detect_price_anomaly(
        self,
        current_price: float,
        price_history: pd.Series,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Detect price anomaly using Z-Score.
        
        Args:
            current_price: Current price value
            price_history: Historical price data (index: datetime)
            timestamp: Detection timestamp
        
        Returns:
            Anomaly dict if detected, None otherwise
        """
        if len(price_history) < self.window_size:
            logger.warning(f"Insufficient data for Z-Score: {len(price_history)} < {self.window_size}")
            return None
        
        # Calculate rolling statistics
        rolling_window = price_history.tail(self.window_size)
        mean = rolling_window.mean()
        std = rolling_window.std()
        
        if std == 0:
            logger.warning("Standard deviation is zero, cannot calculate Z-Score")
            return None
        
        # Calculate Z-Score
        z_score = (current_price - mean) / std
        
        # Check for anomaly
        if abs(z_score) >= self.threshold:
            severity = self._get_severity(z_score)
            
            return {
                'type': 'price',
                'timestamp': timestamp,
                'severity': severity,
                'z_score': z_score,
                'value': current_price,
                'mean': mean,
                'std': std,
                'detection_method': 'zscore'
            }
        
        return None
    
    def detect_volume_anomaly(
        self,
        current_volume: float,
        volume_history: pd.Series,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Detect volume anomaly using Z-Score.
        
        Args:
            current_volume: Current volume value
            volume_history: Historical volume data (index: datetime)
            timestamp: Detection timestamp
        
        Returns:
            Anomaly dict if detected, None otherwise
        """
        if len(volume_history) < self.window_size:
            logger.warning(f"Insufficient volume data: {len(volume_history)} < {self.window_size}")
            return None
        
        # Calculate rolling statistics
        rolling_window = volume_history.tail(self.window_size)
        mean = rolling_window.mean()
        std = rolling_window.std()
        
        if std == 0:
            logger.warning("Volume standard deviation is zero")
            return None
        
        # Calculate Z-Score
        z_score = (current_volume - mean) / std
        
        # Check for anomaly
        if abs(z_score) >= self.threshold:
            severity = self._get_severity(z_score)
            
            return {
                'type': 'volume',
                'timestamp': timestamp,
                'severity': severity,
                'z_score': z_score,
                'value': current_volume,
                'mean': mean,
                'std': std,
                'detection_method': 'zscore'
            }
        
        return None
    
    def _get_severity(self, z_score: float) -> str:
        """
        Get severity level based on Z-Score.
        
        Args:
            z_score: Z-Score value
        
        Returns:
            Severity level ('high', 'medium', 'low')
        """
        if abs(z_score) >= 4.0:
            return 'high'
        elif abs(z_score) >= 3.0:
            return 'medium'
        else:
            return 'low'
```

- [ ] **Step 2: Write Z-Score detector tests**

```python
# tests/test_zscore_detector.py
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
    
    # Create extreme volume
    mean = volume_history.mean()
    std = volume_history.std()
    extreme_volume = mean + 4 * std
    
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
```

- [ ] **Step 3: Run Z-Score detector tests**

Run: `python3 -m pytest tests/test_zscore_detector.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/zscore_detector.py tests/test_zscore_detector.py
git commit -m "feat: implement Z-Score anomaly detector"
```

---

## Task 4: Implement feature extractor

**Files:**
- Create: `anomaly_detector/feature_extractor.py`
- Create: `tests/test_feature_extractor.py`

- [ ] **Step 1: Write feature extractor with multi-dimensional features**

```python
# anomaly_detector/feature_extractor.py
"""
Feature extractor for Isolation Forest anomaly detection.
Extracts multi-dimensional features from historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts multi-dimensional features for anomaly detection."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
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
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Extract features from historical data.
        
        Args:
            df: DataFrame with OHLCV data and additional columns
        
        Returns:
            Tuple of (features DataFrame, timestamps list)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature extractor")
            return pd.DataFrame(columns=self.feature_names), []
        
        # Calculate features
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['return_rate'] = df['Close'].pct_change()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        
        # Volume features
        volume_ma20 = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / volume_ma20
        
        # Technical indicators (if available)
        if 'RSI' in df.columns:
            features['rsi'] = df['RSI']
        else:
            features['rsi'] = self._calculate_rsi(df['Close'])
        
        if 'MACD' in df.columns:
            features['macd'] = df['MACD']
            features['macd_signal'] = df['MACD_signal']
        else:
            macd, macd_signal = self._calculate_macd(df['Close'])
            features['macd'] = macd
            features['macd_signal'] = macd_signal
        
        if 'BB_position' in df.columns:
            features['bb_position'] = df['BB_position']
        else:
            features['bb_position'] = self._calculate_bb_position(df['Close'])
        
        # Trend features
        if 'MA20' in df.columns:
            features['ma20_diff'] = (df['Close'] - df['MA20']) / df['MA20']
        else:
            ma20 = df['Close'].rolling(20).mean()
            features['ma20_diff'] = (df['Close'] - ma20) / ma20
        
        if 'MA50' in df.columns:
            features['ma50_diff'] = (df['Close'] - df['MA50']) / df['MA50']
            features['ma20_ma50_diff'] = (df['MA20'] - df['MA50']) / df['MA50']
        else:
            ma50 = df['Close'].rolling(50).mean()
            features['ma50_diff'] = (df['Close'] - ma50) / ma50
            ma20 = df['Close'].rolling(20).mean()
            features['ma20_ma50_diff'] = (ma20 - ma50) / ma50
        
        # Clean and normalize
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Extract timestamps
        timestamps = df.index.tolist()
        
        logger.info(f"Extracted {len(features)} samples with {len(self.feature_names)} features")
        
        return features[self.feature_names], timestamps
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Calculate Bollinger Band position."""
        bb_middle = prices.rolling(window=period).mean()
        bb_std = prices.rolling(window=period).std()
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        return bb_position.fillna(0.5)
```

- [ ] **Step 2: Write feature extractor tests**

```python
# tests/test_feature_extractor.py
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
```

- [ ] **Step 3: Run feature extractor tests**

Run: `python3 -m pytest tests/test_feature_extractor.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/feature_extractor.py tests/test_feature_extractor.py
git commit -m "feat: implement feature extractor"
```

---

## Task 5: Implement Isolation Forest detector

**Files:**
- Create: `anomaly_detector/isolation_forest_detector.py`
- Create: `tests/test_isolation_forest.py`

- [ ] **Step 1: Write Isolation Forest detector with sklearn**

```python
# anomaly_detector/isolation_forest_detector.py
"""
Isolation Forest anomaly detector for deep analysis.
Uses sklearn IsolationForest for multi-dimensional anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Detects anomalies using Isolation Forest algorithm."""
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.05)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
    
    def train(self, features: pd.DataFrame):
        """
        Train Isolation Forest model.
        
        Args:
            features: Feature matrix (n_samples, n_features)
        """
        if features.empty:
            logger.warning("Empty feature matrix provided for training")
            return
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.model.fit(features)
        logger.info(f"Isolation Forest trained on {len(features)} samples")
    
    def detect_anomalies(
        self,
        features: pd.DataFrame,
        timestamps: List[datetime],
        lookback_days: int = 7
    ) -> List[Dict]:
        """
        Detect anomalies using trained model.
        
        Args:
            features: Feature matrix
            timestamps: Corresponding timestamps
            lookback_days: Only return anomalies from last N days (default: 7)
        
        Returns:
            List of anomaly dicts
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return []
        
        if features.empty:
            logger.warning("Empty feature matrix provided for detection")
            return []
        
        # Get anomaly scores and predictions
        anomaly_scores = self.model.decision_function(features)
        predictions = self.model.predict(features)
        
        # Identify anomalies (prediction = -1)
        anomalies = []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1:  # Anomaly
                timestamp = timestamps[i]
                
                # Only include recent anomalies
                if timestamp >= cutoff_date:
                    severity = self._get_severity(score)
                    
                    # Get feature values
                    feature_values = features.iloc[i].to_dict()
                    
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'severity': severity,
                        'features': feature_values,
                        'detection_method': 'isolation_forest'
                    })
        
        logger.info(f"Detected {len(anomalies)} anomalies in last {lookback_days} days")
        
        return anomalies
    
    def _get_severity(self, anomaly_score: float) -> str:
        """
        Get severity level based on anomaly score.
        
        Args:
            anomaly_score: Anomaly score from Isolation Forest (lower = more anomalous)
        
        Returns:
            Severity level ('high', 'medium', 'low')
        """
        if anomaly_score < -0.7:
            return 'high'
        elif anomaly_score < -0.5:
            return 'medium'
        else:
            return 'low'
```

- [ ] **Step 2: Write Isolation Forest tests**

```python
# tests/test_isolation_forest.py
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
```

- [ ] **Step 3: Run Isolation Forest tests**

Run: `python3 -m pytest tests/test_isolation_forest.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/isolation_forest_detector.py tests/test_isolation_forest.py
git commit -m "feat: implement Isolation Forest anomaly detector"
```

---

## Task 6: Implement anomaly integrator

**Files:**
- Create: `anomaly_detector/anomaly_integrator.py`
- Create: `tests/test_anomaly_integrator.py`

- [ ] **Step 1: Write anomaly integrator to combine results**

```python
# anomaly_detector/anomaly_integrator.py
"""
Anomaly integrator to combine results from multiple detectors.
Handles deduplication and severity classification.
"""

from datetime import datetime
from typing import Dict, List
import logging

from .cache import AnomalyCache

logger = logging.getLogger(__name__)


class AnomalyIntegrator:
    """Integrates anomaly detection results from multiple detectors."""
    
    def __init__(self, cache: AnomalyCache):
        """
        Initialize anomaly integrator.
        
        Args:
            cache: AnomalyCache instance for deduplication
        """
        self.cache = cache
    
    def integrate(
        self,
        zscore_anomalies: List[Dict],
        if_anomalies: List[Dict],
        timestamp: datetime
    ) -> Dict:
        """
        Integrate anomalies from both detectors.
        
        Args:
            zscore_anomalies: Anomalies from Z-Score detector
            if_anomalies: Anomalies from Isolation Forest detector
            timestamp: Current timestamp
        
        Returns:
            Integrated result dict
        """
        all_anomalies = []
        
        # Add Z-Score anomalies
        for anomaly in zscore_anomalies:
            # Check if already reported
            if self.cache.exists(anomaly['type'], anomaly['timestamp']):
                logger.info(f"Skipping duplicate {anomaly['type']} anomaly")
                continue
            
            all_anomalies.append(anomaly)
        
        # Add Isolation Forest anomalies
        for anomaly in if_anomalies:
            # Use 'isolation_forest' as type for deduplication
            cache_key = f"if_{anomaly['timestamp'].strftime('%Y-%m-%d')}"
            
            # Check if already reported (check cache directly)
            if cache_key in self.cache.cache:
                logger.info(f"Skipping duplicate Isolation Forest anomaly")
                continue
            
            all_anomalies.append(anomaly)
        
        if not all_anomalies:
            return {
                'has_anomaly': False,
                'anomalies': [],
                'severity': None
            }
        
        # Determine overall severity
        severity = self._get_overall_severity(all_anomalies)
        
        # Add anomalies to cache
        for anomaly in all_anomalies:
            anomaly_type = anomaly['type']
            anomaly_timestamp = anomaly['timestamp']
            
            if anomaly_type == 'isolation_forest':
                cache_key = f"if_{anomaly_timestamp.strftime('%Y-%m-%d')}"
                self.cache.cache[cache_key] = {
                    'timestamp': anomaly_timestamp.isoformat(),
                    'severity': anomaly['severity'],
                    'anomaly_score': anomaly.get('anomaly_score', 0)
                }
            else:
                self.cache.add(
                    anomaly_type=anomaly_type,
                    timestamp=anomaly_timestamp,
                    severity=anomaly['severity'],
                    z_score=anomaly.get('z_score', 0)
                )
        
        # Save cache
        self.cache._save_cache()
        
        return {
            'has_anomaly': True,
            'anomalies': all_anomalies,
            'severity': severity
        }
    
    def _get_overall_severity(self, anomalies: List[Dict]) -> str:
        """
        Get overall severity from multiple anomalies.
        
        Args:
            anomalies: List of anomaly dicts
        
        Returns:
            Overall severity level ('high', 'medium', 'low')
        """
        if not anomalies:
            return None
        
        # Check for high severity
        if any(a['severity'] == 'high' for a in anomalies):
            return 'high'
        
        # Check for medium severity
        if any(a['severity'] == 'medium' for a in anomalies):
            return 'medium'
        
        # Default to low
        return 'low'
```

- [ ] **Step 2: Write anomaly integrator tests**

```python
# tests/test_anomaly_integrator.py
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
```

- [ ] **Step 3: Run anomaly integrator tests**

Run: `python3 -m pytest tests/test_anomaly_integrator.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add anomaly_detector/anomaly_integrator.py tests/test_anomaly_integrator.py
git commit -m "feat: implement anomaly integrator"
```

---

## Task 7: Integrate anomaly detection into crypto_email.py

**Files:**
- Modify: `crypto_email.py`

- [ ] **Step 1: Add anomaly detection imports to crypto_email.py**

Add after existing imports (around line 30):

```python
# Anomaly detection imports
try:
    from anomaly_detector.zscore_detector import ZScoreDetector
    from anomaly_detector.isolation_forest_detector import IsolationForestDetector
    from anomaly_detector.feature_extractor import FeatureExtractor
    from anomaly_detector.anomaly_integrator import AnomalyIntegrator
    from anomaly_detector.cache import AnomalyCache
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Anomaly detection not available: {e}")
    ANOMALY_DETECTION_AVAILABLE = False
```

- [ ] **Step 2: Add anomaly detection function**

Add after `calculate_technical_indicators` function (around line 400):

```python
def run_anomaly_detection(prices):
    """
    Run anomaly detection on Ethereum data.
    
    Args:
        prices: Price data from CoinGecko API
    
    Returns:
        Anomaly detection results dict or None if no anomalies
    """
    if not ANOMALY_DETECTION_AVAILABLE:
        print("⚠️ Anomaly detection not available")
        return None
    
    if 'ethereum' not in prices:
        return None
    
    try:
        # Initialize components
        zscore_detector = ZScoreDetector(window_size=30, threshold=3.0)
        if_detector = IsolationForestDetector(contamination=0.05, random_state=42)
        feature_extractor = FeatureExtractor()
        cache = AnomalyCache()
        integrator = AnomalyIntegrator(cache)
        
        # Get historical data
        eth_ticker = yf.Ticker("ETH-USD")
        eth_hist = eth_ticker.history(period="6mo")
        
        if eth_hist.empty:
            print("⚠️ No historical data available for anomaly detection")
            return None
        
        # Run Z-Score detection (Layer 1)
        zscore_anomalies = []
        current_price = prices['ethereum']['usd']
        
        # Price anomaly
        price_anomaly = zscore_detector.detect_anomaly(
            metric_name='price',
            current_value=current_price,
            history=eth_hist['Close'],
            timestamp=datetime.now()
        )
        if price_anomaly:
            zscore_anomalies.append(price_anomaly)
        
        # Volume anomaly
        if 'usd_24hr_vol' in prices['ethereum']:
            current_volume = prices['ethereum']['usd_24hr_vol']
            volume_anomaly = zscore_detector.detect_anomaly(
                metric_name='volume',
                current_value=current_volume,
                history=eth_hist['Volume'],
                timestamp=datetime.now()
            )
            if volume_anomaly:
                zscore_anomalies.append(volume_anomaly)
        
        # Run Isolation Forest detection (Layer 2)
        if_anomalies = []
        
        # Extract features
        features, timestamps = feature_extractor.extract_features(eth_hist)
        
        if not features.empty:
            # Train model
            if_detector.train(features)
            
            # Detect anomalies (last 7 days)
            if_anomalies = if_detector.detect_anomalies(
                features=features,
                timestamps=timestamps,
                lookback_days=7
            )
        
        # Integrate results
        result = integrator.integrate(
            zscore_anomalies=zscore_anomalies,
            if_anomalies=if_anomalies,
            timestamp=datetime.now()
        )
        
        # Cleanup old cache entries
        cache.cleanup_expired(max_age_hours=48)
        
        return result
    
    except Exception as e:
        print(f"⚠️ Anomaly detection failed: {e}")
        return None


def format_anomaly_results(anomaly_result):
    """
    Format anomaly detection results for email.
    
    Args:
        anomaly_result: Anomaly detection result dict
    
    Returns:
        Formatted HTML string
    """
    if not anomaly_result or not anomaly_result['has_anomaly']:
        return None
    
    severity_emoji = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢'
    }
    
    html = """
        <div class="section">
            <h3>🚨 异常检测报告</h3>
    """
    
    # Add severity
    severity = anomaly_result['severity']
    html += f"<p><strong>严重程度:</strong> {severity_emoji.get(severity, '')} {severity.upper()}</p>"
    
    # Add anomalies list
    html += "<table><tr><th>类型</th><th>检测时间</th><th>严重程度</th><th>详情</th></tr>"
    
    for anomaly in anomaly_result['anomalies']:
        anomaly_type = anomaly['type']
        timestamp = anomaly['timestamp']
        anomaly_severity = anomaly['severity']
        
        # Format details
        if anomaly_type == 'isolation_forest':
            score = anomaly.get('anomaly_score', 0)
            details = f"异常评分: {score:.3f}"
        else:
            z_score = anomaly.get('z_score', 0)
            value = anomaly.get('value', 0)
            details = f"Z-Score: {z_score:.2f}, 值: {value:.2f}"
        
        html += f"""
            <tr>
                <td>{anomaly_type}</td>
                <td>{timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{anomaly_severity}</td>
                <td>{details}</td>
            </tr>
        """
    
    html += "</table></div>"
    
    return html
```

- [ ] **Step 3: Update main function to include anomaly detection**

Modify the main function section (around line 1040), add after `indicators = calculate_technical_indicators(prices)`:

```python
    # 运行异常检测
    anomaly_result = None
    if ANOMALY_DETECTION_AVAILABLE:
        anomaly_result = run_anomaly_detection(prices)
        
        # 如果检测到异常，强制发送邮件
        if anomaly_result and anomaly_result['has_anomaly']:
            has_signals = True
```

- [ ] **Step 4: Add anomaly results to email HTML**

Modify the email HTML generation section (around line 1070), add after technical analysis section:

```python
    # 添加异常检测结果（如果有）
    if anomaly_result and anomaly_result['has_anomaly']:
        anomaly_html = format_anomaly_results(anomaly_result)
        if anomaly_html:
            html += anomaly_html
```

- [ ] **Step 5: Add anomaly results to email text**

Modify the email text generation section (around line 800), add after technical analysis text:

```python
    # 添加异常检测结果到文本版本
    if anomaly_result and anomaly_result['has_anomaly']:
        text += "\n\n🚨 异常检测报告\n"
        text += f"严重程度: {anomaly_result['severity'].upper()}\n"
        text += f"检测到异常数量: {len(anomaly_result['anomalies'])}\n"
        
        for anomaly in anomaly_result['anomalies']:
            anomaly_type = anomaly['type']
            timestamp = anomaly['timestamp']
            anomaly_severity = anomaly['severity']
            
            if anomaly_type == 'isolation_forest':
                score = anomaly.get('anomaly_score', 0)
                text += f"\n- 类型: Isolation Forest"
                text += f"\n  时间: {timestamp.strftime('%Y-%m-%d %H:%M')}"
                text += f"\n  严重程度: {anomaly_severity}"
                text += f"\n  异常评分: {score:.3f}"
            else:
                z_score = anomaly.get('z_score', 0)
                value = anomaly.get('value', 0)
                text += f"\n- 类型: {anomaly_type}"
                text += f"\n  时间: {timestamp.strftime('%Y-%m-%d %H:%M')}"
                text += f"\n  严重程度: {anomaly_severity}"
                text += f"\n  Z-Score: {z_score:.2f}"
                text += f"\n  值: {value:.2f}"
```

- [ ] **Step 6: Test anomaly detection integration**

Run: `python3 crypto_email.py` (test without email)

Expected: No errors, anomaly detection runs successfully

- [ ] **Step 7: Commit**

```bash
git add crypto_email.py
git commit -m "feat: integrate anomaly detection into crypto_email.py"
```

---

## Task 8: Create GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ethereum-anomaly-detection.yml`

- [ ] **Step 1: Create GitHub Actions workflow file**

```yaml
name: Ethereum Anomaly Detection

on:
  schedule:
    # 每小时运行Z-Score实时检测
    - cron: '0 * * * *'  # 每小时
    # 每天凌晨2点运行Isolation Forest深度分析
    - cron: '0 2 * * *'  # 每天凌晨2点（香港时间）
  workflow_dispatch:  # 支持手动触发

jobs:
  anomaly-detection:
    runs-on: ubuntu-latest
    env:
      TZ: Asia/Hong_Kong
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run anomaly detection
        run: |
          python3 crypto_email.py
```

- [ ] **Step 2: Verify workflow syntax**

Run: `cat .github/workflows/ethereum-anomaly-detection.yml`

Expected: Valid YAML syntax

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ethereum-anomaly-detection.yml
git commit -m "feat: add GitHub Actions workflow for anomaly detection"
```

---

## Task 9: Add configuration support

**Files:**
- Modify: `set_key.sh.sample`

- [ ] **Step 1: Add anomaly detection configuration to sample**

Add to `set_key.sh.sample` after existing configurations:

```bash
# Anomaly Detection Configuration
export ANOMALY_DETECTION_ENABLED="true"
export ZSCORE_WINDOW_SIZE="30"
export ZSCORE_THRESHOLD="3.0"
export ISOLATION_FOREST_CONTAMINATION="0.05"
export ANOMALY_CACHE_HOURS="24"
```

- [ ] **Step 2: Commit**

```bash
git add set_key.sh.sample
git commit -m "feat: add anomaly detection configuration to set_key.sh.sample"
```

---

## Task 10: Run integration test

**Files:**
- None (testing only)

- [ ] **Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 2: Test anomaly detection end-to-end**

Run: `python3 crypto_email.py --no-email`

Expected: No errors, anomaly detection runs successfully

- [ ] **Step 3: Verify cache file created**

Run: `ls -la data/anomaly_cache.json`

Expected: File exists (may be empty initially)

- [ ] **Step 4: Commit final changes**

```bash
git add .
git commit -m "test: validate anomaly detection implementation"
```

---

## Task 11: Push to remote

**Files:**
- None (git operation)

- [ ] **Step 1: Push all commits to remote**

Run: `git push origin main`

Expected: All commits pushed successfully

- [ ] **Step 2: Verify GitHub Actions workflow**

Run: Check GitHub Actions tab in repository

Expected: New workflow file visible and scheduled

---

## Implementation Notes

### Testing Strategy
- All components have comprehensive unit tests
- Integration tests verify end-to-end functionality
- Use pytest for test execution
- Test coverage: Focus on core logic (Z-Score calculation, Isolation Forest prediction, cache management)

### Deployment Strategy
- Deploy to GitHub Actions first for monitoring
- Monitor for false positives and adjust thresholds if needed
- Consider adding a "dry run" mode for testing without email alerts

### Monitoring
- Watch cache file size (should remain reasonable)
- Monitor email frequency (should not spam)
- Track detection accuracy over time

### Future Enhancements
- Add support for other cryptocurrencies (Bitcoin)
- Implement adaptive threshold adjustment
- Add anomaly pattern classification
- Integrate with machine learning model predictions

---

## Completion Criteria

- [ ] All tasks completed and committed
- [ ] All tests passing
- [ ] GitHub Actions workflow deployed
- [ ] Manual testing successful (crypto_email.py runs without errors)
- [ ] Cache management verified
- [ ] Documentation updated (if needed)