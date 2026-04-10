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
        # 按日期去重：每个股票每个日期只保留一个异常
        seen_dates = set()
        for anomaly in if_anomalies:
            # 获取异常日期
            anomaly_date = anomaly['timestamp'].strftime('%Y-%m-%d')
            
            # 检查是否已经处理过这个日期
            if anomaly_date in seen_dates:
                logger.info(f"Skipping duplicate Isolation Forest anomaly for date {anomaly_date}")
                continue
            
            seen_dates.add(anomaly_date)
            
            # Use 'isolation_forest' as type for deduplication
            cache_key = f"if_{anomaly_date}"
            
            # Check if already reported (check cache directly)
            if cache_key in self.cache.cache:
                logger.info(f"Skipping cached Isolation Forest anomaly for {anomaly_date}")
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
