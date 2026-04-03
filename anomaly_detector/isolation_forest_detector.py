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