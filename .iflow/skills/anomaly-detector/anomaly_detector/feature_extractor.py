"""
Generic feature extractor for Isolation Forest anomaly detection.
Extracts multi-dimensional features from any time series data.
Supports both general metrics and optional stock-specific indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Generic feature extractor for time series anomaly detection."""
    
    def __init__(self, ma_periods: List[int] = None, include_stock_indicators: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            ma_periods: List of moving average periods (default: [5, 10, 20, 50])
            include_stock_indicators: Whether to include stock-specific indicators 
                                      (RSI, MACD, Bollinger Band) when OHLCV data is available
        """
        self.ma_periods = ma_periods or [5, 10, 20, 50]
        self.include_stock_indicators = include_stock_indicators
    
    def extract_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None
    ) -> Tuple[pd.DataFrame, list]:
        """
        Extract features from time series data.
        
        Automatically detects:
        1. Numeric columns for general feature extraction
        2. OHLCV columns for stock-specific indicators (if enabled)
        
        Args:
            df: DataFrame with time series data
            columns: Specific columns to extract features from (optional, auto-detects numeric columns)
        
        Returns:
            Tuple of (features DataFrame, timestamps list)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature extractor")
            return pd.DataFrame(), []
        
        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns found in DataFrame")
            return pd.DataFrame(), []
        
        features = pd.DataFrame(index=df.index)
        
        # Check for OHLCV data (stock-specific)
        ohlcv_mapping = self._detect_ohlcv_columns(df)
        
        if ohlcv_mapping and self.include_stock_indicators:
            # Extract stock-specific indicators
            logger.info("Detected OHLCV columns, extracting stock indicators")
            features = self._extract_stock_features(df, ohlcv_mapping, features)
        else:
            # Extract general features for each numeric column
            logger.info(f"Extracting general features for columns: {columns}")
            for col in columns:
                col_features = self._extract_column_features(df[col], col)
                features = pd.concat([features, col_features], axis=1)
        
        # Clean and normalize
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Remove columns with zero variance
        features = features.loc[:, features.std() > 0]
        
        # Extract timestamps
        timestamps = df.index.tolist()
        
        logger.info(f"Extracted {len(features)} samples with {len(features.columns)} features")
        
        return features, timestamps
    
    def _detect_ohlcv_columns(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """
        Detect OHLCV columns in DataFrame (case-insensitive).
        
        Returns:
            Dict mapping standard names to actual column names, or None if not found
        """
        # Common column name variations (lowercase for comparison)
        variations = {
            'Open': ['open', '开市价', '开盘价', 'opening_price'],
            'High': ['high', '最高价', 'highest_price'],
            'Low': ['low', '最低价', 'lowest_price'],
            'Close': ['close', '收市价', '收盘价', 'closing_price'],
            'Volume': ['volume', '成交量', '成交额', 'vol']
        }
        
        mapping = {}
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for std_name, variants in variations.items():
            for variant in variants:
                if variant.lower() in df_cols_lower:
                    mapping[std_name] = df_cols_lower[variant.lower()]
                    break
        
        # Only return mapping if we have at least Close
        if 'Close' in mapping:
            return mapping
        return None
    
    def _extract_column_features(self, series: pd.Series, prefix: str) -> pd.DataFrame:
        """
        Extract general features from a single column.
        
        Features extracted:
        - Return rate (pct_change)
        - Volatility (rolling std)
        - Moving averages and deviations
        - Rate of change
        - Z-score position
        """
        features = pd.DataFrame(index=series.index)
        col_name = prefix.replace(' ', '_').replace('/', '_')
        
        # Return rate
        features[f'{col_name}_return'] = series.pct_change()
        
        # Volatility at different windows
        for period in [5, 10, 20]:
            if len(series) >= period:
                features[f'{col_name}_volatility_{period}'] = series.pct_change().rolling(period).std()
        
        # Moving averages and deviations
        for period in self.ma_periods:
            if len(series) >= period:
                ma = series.rolling(period).mean()
                features[f'{col_name}_ma{period}'] = ma
                features[f'{col_name}_ma{period}_diff'] = (series - ma) / ma
                features[f'{col_name}_ma{period}_ratio'] = series / ma
        
        # Rate of change
        for period in [1, 5, 10]:
            if len(series) > period:
                features[f'{col_name}_roc_{period}'] = series.pct_change(periods=period)
        
        # Z-score position (current value relative to rolling mean/std)
        for period in [20, 50]:
            if len(series) >= period:
                rolling_mean = series.rolling(period).mean()
                rolling_std = series.rolling(period).std()
                features[f'{col_name}_zscore_{period}'] = (series - rolling_mean) / rolling_std
        
        # Cumulative return
        features[f'{col_name}_cum_return'] = (series / series.iloc[0] - 1) if len(series) > 0 else 0
        
        return features
    
    def _extract_stock_features(
        self, 
        df: pd.DataFrame, 
        ohlcv_mapping: Dict[str, str],
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract stock-specific features when OHLCV data is available.
        """
        close = df[ohlcv_mapping['Close']]
        
        # Basic price features
        features['return_rate'] = close.pct_change(fill_method=None)
        features['volatility_20d'] = close.pct_change(fill_method=None).rolling(20).std()
        
        # Volume features (if available)
        if 'Volume' in ohlcv_mapping:
            volume = df[ohlcv_mapping['Volume']]
            volume_ma20 = volume.rolling(20).mean()
            features['volume_ratio'] = volume / volume_ma20
        
        # RSI
        features['rsi'] = self._calculate_rsi(close)
        
        # MACD
        macd, macd_signal = self._calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        
        # Bollinger Band position
        features['bb_position'] = self._calculate_bb_position(close)
        
        # Moving average differences
        ma20 = close.rolling(20).mean()
        features['ma20_diff'] = (close - ma20) / ma20
        
        ma50 = close.rolling(50).mean()
        features['ma50_diff'] = (close - ma50) / ma50
        features['ma20_ma50_diff'] = (ma20 - ma50) / ma50
        
        # Price range (if High and Low available)
        if 'High' in ohlcv_mapping and 'Low' in ohlcv_mapping:
            high = df[ohlcv_mapping['High']]
            low = df[ohlcv_mapping['Low']]
            features['price_range'] = (high - low) / close
            features['upper_shadow'] = (high - close) / (high - low + 1e-10)
            features['lower_shadow'] = (close - low) / (high - low + 1e-10)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def _calculate_bb_position(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.Series:
        """Calculate Bollinger Band position."""
        bb_middle = prices.rolling(window=period).mean()
        bb_std = prices.rolling(window=period).std()
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        return bb_position.fillna(0.5)