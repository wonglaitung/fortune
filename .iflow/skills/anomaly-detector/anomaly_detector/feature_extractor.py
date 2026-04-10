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
