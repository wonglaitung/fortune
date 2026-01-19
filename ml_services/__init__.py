#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习服务模块
包含机器学习交易模型和预测邮件发送功能
"""

from .ml_trading_model import (
    MLTradingModel,
    GBDTLRModel,
    FeatureEngineer,
    WATCHLIST,
    STOCK_NAMES
)
from .ml_prediction_email import MLPredictionEmailSender
from .us_market_data import us_market_data
from .base_model_processor import BaseModelProcessor

__all__ = [
    'MLTradingModel',
    'GBDTLRModel',
    'FeatureEngineer',
    'WATCHLIST',
    'STOCK_NAMES',
    'MLPredictionEmailSender',
    'us_market_data',
    'BaseModelProcessor'
]