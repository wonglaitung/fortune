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