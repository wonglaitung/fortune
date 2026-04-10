"""
Isolation Forest anomaly detector for deep analysis.
Uses sklearn IsolationForest for multi-dimensional anomaly detection.
Supports multiple time intervals (minute, hour, day, week) and is scenario-agnostic.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeInterval(Enum):
    """Time interval for lookback window."""
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


class IsolationForestDetector:
    """Generic Isolation Forest anomaly detector for multi-dimensional data."""
    
    def __init__(
        self,
        contamination: float = 0.03,
        random_state: int = 42,
        anomaly_type: str = 'isolation_forest'
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.03)
            random_state: Random seed for reproducibility (default: 42)
            anomaly_type: Type label for anomalies (default: 'isolation_forest')
        
        Example:
            >>> detector = IsolationForestDetector(
            ...     contamination=0.03,
            ...     anomaly_type='multi_feature'
            ... )
        """
        self.contamination = contamination
        self.random_state = random_state
        self.anomaly_type = anomaly_type
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
        lookback_days: int = 7,
        time_interval: Optional[Union[str, TimeInterval]] = None
    ) -> List[Dict]:
        """
        Detect anomalies using trained model.
        
        Args:
            features: Feature matrix
            timestamps: Corresponding timestamps
            lookback_days: Only return anomalies from last N days (default: 7)
            time_interval: Time interval type for the data (for metadata)
        
        Returns:
            List of anomaly dicts with enhanced metadata
            
        Example:
            >>> detector = IsolationForestDetector(anomaly_type='multi_feature')
            >>> anomalies = detector.detect_anomalies(
            ...     features=feature_matrix,
            ...     timestamps=timestamps,
            ...     lookback_days=7,
            ...     time_interval='day'
            ... )
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
        # Use UTC datetime for comparison to avoid timezone issues
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        cutoff_date = utc_now - timedelta(days=lookback_days)
        
        for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1:  # Anomaly
                timestamp = timestamps[i]
                
                # Handle different timestamp types (datetime, int, str, etc.)
                if isinstance(timestamp, datetime):
                    # Normalize timestamp to UTC for comparison
                    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                        timestamp_utc = timestamp.astimezone(timezone.utc)
                    else:
                        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
                else:
                    # For non-datetime types (int index, string, etc.), skip time filtering
                    # Use current time as placeholder (all anomalies will be included)
                    timestamp_utc = utc_now
                
                # Only include recent anomalies
                if timestamp_utc >= cutoff_date:
                    severity = self._get_severity(score)
                    
                    # Get feature values
                    feature_values = features.iloc[i].to_dict()
                    
                    # Determine time interval string
                    interval_str = (
                        time_interval.value if isinstance(time_interval, TimeInterval)
                        else time_interval if time_interval
                        else 'unknown'
                    )
                    
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'severity': severity,
                        'features': feature_values,
                        'detection_method': 'isolation_forest',
                        'type': self.anomaly_type,
                        'lookback_days': lookback_days,
                        'time_interval': interval_str,
                        'feature_count': len(feature_values),
                        'feature_names': list(feature_values.keys())
                    })
        
        logger.info(f"Detected {len(anomalies)} anomalies in last {lookback_days} days")
        
        return anomalies
    
    def detect_anomalies_by_date(
        self,
        features: pd.DataFrame,
        timestamps: List[datetime],
        target_date: datetime,
        time_interval: Optional[Union[str, TimeInterval]] = None
    ) -> List[Dict]:
        """
        Detect anomalies on a specific date.
        
        Args:
            features: Feature matrix
            timestamps: Corresponding timestamps
            target_date: Target date to check for anomalies
            time_interval: Time interval type for the data (for metadata)
        
        Returns:
            List of anomaly dicts with enhanced metadata
            
        Example:
            >>> detector = IsolationForestDetector(anomaly_type='crypto_features')
            >>> anomalies = detector.detect_anomalies_by_date(
            ...     features=feature_matrix,
            ...     timestamps=timestamps,
            ...     target_date=datetime(2026, 4, 4),
            ...     time_interval='hour'
            ... )
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
        
        # Normalize target date to UTC for comparison
        from datetime import timezone
        if hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
            target_date_utc = target_date.astimezone(timezone.utc)
        else:
            target_date_utc = target_date.replace(tzinfo=timezone.utc)
        
        # Get date part only for comparison
        target_date_only = target_date_utc.date()
        
        # Identify anomalies (prediction = -1)
        anomalies = []
        
        for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1:  # Anomaly
                timestamp = timestamps[i]
                
                # Handle different timestamp types
                if isinstance(timestamp, datetime):
                    # 不转换到UTC，直接使用原始时区比较日期
                    # 如果timestamp没有时区，假设是香港时间（数据源默认）
                    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                        timestamp_date = timestamp.date()
                    else:
                        # 没有时区，假设是本地时间
                        timestamp_date = timestamp.date()
                    
                    # 获取目标日期的date部分（不转换时区）
                    target_date_only = target_date.date() if hasattr(target_date, 'date') else target_date
                    
                    # Only include anomalies on the target date
                    if timestamp_date != target_date_only:
                        continue
                else:
                    # For non-datetime types, skip date filtering and include all anomalies
                    pass
                
                # Only include anomalies on the target date
                if timestamp_date == target_date_only:
                    severity = self._get_severity(score)
                    
                    # Get feature values
                    feature_values = features.iloc[i].to_dict()
                    
                    # Determine time interval string
                    interval_str = (
                        time_interval.value if isinstance(time_interval, TimeInterval)
                        else time_interval if time_interval
                        else 'unknown'
                    )
                    
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'severity': severity,
                        'features': feature_values,
                        'detection_method': 'isolation_forest',
                        'type': self.anomaly_type,
                        'time_interval': interval_str,
                        'feature_count': len(feature_values),
                        'feature_names': list(feature_values.keys())
                    })
        
        # 调整日志消息：明确显示目标日期和实际检测结果
        if anomalies:
            anomaly_dates = sorted(set([a['timestamp'].strftime('%Y-%m-%d') for a in anomalies]))
            target_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
            if anomaly_dates == [target_date_str]:
                logger.info(f"Detected {len(anomalies)} anomalies on target date {target_date_str}")
            else:
                logger.warning(f"Requested date {target_date_str}, but found anomalies on {', '.join(anomaly_dates)}")
        else:
            target_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
            logger.info(f"No anomalies detected on target date {target_date_str}")
        
        return anomalies
    
    def _get_severity(self, anomaly_score: float) -> str:
        """
        Get severity level based on anomaly score.
        
        Args:
            anomaly_score: Anomaly score from Isolation Forest (lower = more anomalous)
        
        Returns:
            Severity level ('high', 'medium', 'low')
        
        Note:
            Based on 2-year HK stock data analysis (2024-04-01 to 2026-04-01):
            - All anomaly scores fall in range [-0.2, 0]
            - Original thresholds (< -0.6, < -0.4) were too strict
            - New thresholds calibrated to actual distribution
        """
        if anomaly_score < -0.10:
            return 'high'
        elif anomaly_score < -0.02:
            return 'medium'
        else:
            return 'low'
