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
        # Use UTC datetime for comparison to avoid timezone issues
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        cutoff_date = utc_now - timedelta(days=lookback_days)
        
        for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1:  # Anomaly
                timestamp = timestamps[i]
                
                # Normalize timestamp to UTC for comparison
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                    timestamp_utc = timestamp.astimezone(timezone.utc)
                else:
                    timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
                
                # Only include recent anomalies
                if timestamp_utc >= cutoff_date:
                    severity = self._get_severity(score)
                    
                    # Get feature values
                    feature_values = features.iloc[i].to_dict()
                    
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'severity': severity,
                        'features': feature_values,
                        'detection_method': 'isolation_forest',
                        'type': 'isolation_forest'
                    })
        
        logger.info(f"Detected {len(anomalies)} anomalies in last {lookback_days} days")
        
        return anomalies
    
    def detect_anomalies_by_date(
        self,
        features: pd.DataFrame,
        timestamps: List[datetime],
        target_date: datetime
    ) -> List[Dict]:
        """
        Detect anomalies on a specific date.
        
        Args:
            features: Feature matrix
            timestamps: Corresponding timestamps
            target_date: Target date to check for anomalies
        
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
                if timestamp_date == target_date_only:
                    severity = self._get_severity(score)
                    
                    # Get feature values
                    feature_values = features.iloc[i].to_dict()
                    
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'severity': severity,
                        'features': feature_values,
                        'detection_method': 'isolation_forest',
                        'type': 'isolation_forest'
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
        """
        if anomaly_score < -0.7:
            return 'high'
        elif anomaly_score < -0.5:
            return 'medium'
        else:
            return 'low'