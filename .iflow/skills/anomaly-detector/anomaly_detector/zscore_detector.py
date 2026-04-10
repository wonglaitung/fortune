"""
Z-Score anomaly detector for real-time monitoring.
Uses moving window to detect anomalies in time series data.
Supports multiple time intervals (minute, hour, day, week) and is scenario-agnostic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeInterval(Enum):
    """Time interval for rolling window."""
    MINUTE = 'minute'
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'
    
    @classmethod
    def from_string(cls, value: str) -> 'TimeInterval':
        """Convert string to TimeInterval."""
        for interval in cls:
            if interval.value.lower() == value.lower():
                return interval
        raise ValueError(f"Invalid time interval: {value}")


class ZScoreDetector:
    """Generic Z-Score anomaly detector for time series data."""
    
    def __init__(
        self,
        window_size: int = 30,
        threshold: float = 3.0,
        time_interval: Union[str, TimeInterval] = 'day'
    ):
        """
        Initialize Z-Score detector.
        
        Args:
            window_size: Rolling window size (number of intervals)
            threshold: Z-Score threshold for anomaly detection (default: 3.0)
            time_interval: Time interval type ('minute', 'hour', 'day', 'week')
        
        Example:
            # Detect anomalies using 30-day window
            detector = ZScoreDetector(window_size=30, time_interval='day')
            
            # Detect anomalies using 60-minute window
            detector = ZScoreDetector(window_size=60, time_interval='hour')
        """
        self.window_size = window_size
        self.threshold = threshold
        self.time_interval = (
            TimeInterval.from_string(time_interval)
            if isinstance(time_interval, str)
            else time_interval
        )
    
    def detect_anomaly(
        self,
        metric_name: str,
        current_value: float,
        history: pd.Series,
        timestamp: datetime,
        time_interval: Optional[Union[str, TimeInterval]] = None
    ) -> Optional[Dict]:
        """
        Detect anomaly in any time series metric using Z-Score.
        
        This is the generic method that can detect anomalies in any numeric time series,
        including prices, volumes, technical indicators, etc.
        
        Args:
            metric_name: Name of the metric being monitored (e.g., 'price', 'volume', 'RSI')
            current_value: Current value of the metric
            history: Historical data (index: datetime, values: numeric)
            timestamp: Detection timestamp
            time_interval: Time interval for window (overrides detector default)
        
        Returns:
            Anomaly dict if detected, None otherwise
            
        Example:
            >>> detector = ZScoreDetector(window_size=30, time_interval='day')
            >>> 
            >>> # Detect price anomaly
            >>> price_anomaly = detector.detect_anomaly(
            ...     metric_name='price',
            ...     current_value=150.0,
            ...     history=price_history,
            ...     timestamp=datetime.now()
            ... )
            >>>
            >>> # Detect RSI anomaly
            >>> rsi_anomaly = detector.detect_anomaly(
            ...     metric_name='RSI',
            ...     current_value=85.0,
            ...     history=rsi_history,
            ...     timestamp=datetime.now()
            ... )
            >>>
            >>> # Detect hourly crypto volume anomaly
            >>> crypto_detector = ZScoreDetector(window_size=60, time_interval='hour')
            >>> volume_anomaly = crypto_detector.detect_anomaly(
            ...     metric_name='volume',
            ...     current_value=5000000,
            ...     history=volume_history,
            ...     timestamp=datetime.now()
            ... )
        """
        if len(history) < self.window_size:
            logger.warning(
                f"Insufficient data for Z-Score ({metric_name}): "
                f"{len(history)} < {self.window_size}"
            )
            return None
        
        # Calculate rolling statistics
        rolling_window = history.tail(self.window_size)
        mean = rolling_window.mean()
        std = rolling_window.std()
        
        if std == 0:
            logger.warning(f"Standard deviation is zero for {metric_name}")
            return None
        
        # Calculate Z-Score
        z_score = (current_value - mean) / std
        
        # Check for anomaly
        if abs(z_score) >= self.threshold:
            severity = self._get_severity(z_score)
            interval = time_interval or self.time_interval
            
            return {
                'type': metric_name,
                'timestamp': timestamp,
                'severity': severity,
                'z_score': z_score,
                'value': current_value,
                'mean': mean,
                'std': std,
                'window_size': self.window_size,
                'time_interval': interval.value if isinstance(interval, TimeInterval) else interval,
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
