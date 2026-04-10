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
        from datetime import timezone
        now = datetime.now(timezone.utc)  # 使用 UTC 时区
        expired_keys = []
        
        for key, entry in self.cache.items():
            timestamp_str = entry.get('timestamp', '')
            if not timestamp_str:
                expired_keys.append(key)
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                # 如果时间戳没有时区信息，假设是 UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
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
