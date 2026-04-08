#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密货币异常阈值校准工具

目标：
1. 分析加密货币的实际波动率分布
2. 根据波动率确定合适的异常阈值
3. 验证新阈值的异常率和策略效果

校准方法：
- 股票波动率：2-3%
- 加密货币波动率：5-10%（约为股票的3倍）
- 阈值调整：股票阈值 * 1.5-2.0倍
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detector.zscore_detector import ZScoreDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_volatility_distribution(cryptos: List[str], period: str = '1y'):
    """分析加密货币的波动率分布"""
    logger.info(f"\n{'='*60}")
    logger.info("分析加密货币波动率分布")
    logger.info(f"{'='*60}")
    
    volatility_stats = {}
    
    for crypto in cryptos:
        logger.info(f"\n分析 {crypto}...")
        
        try:
            ticker = yf.Ticker(crypto)
            df = ticker.history(period=period, interval='1h')
            
            if df.empty:
                logger.warning(f"  {crypto}: 无数据")
                continue
            
            # 计算小时级收益率
            df['return'] = df['Close'].pct_change()
            
            # 计算滚动波动率（24小时窗口）
            df['volatility_24h'] = df['return'].rolling(window=24).std() * np.sqrt(24) * 100  # 年化波动率
            
            # 计算统计量
            returns = df['return'].dropna()
            volatility = df['volatility_24h'].dropna()
            
            stats = {
                'returns': {
                    'mean': returns.mean() * 100,
                    'std': returns.std() * 100,
                    'min': returns.min() * 100,
                    'max': returns.max() * 100,
                    'percentile_95': returns.quantile(0.95) * 100,
                    'percentile_99': returns.quantile(0.99) * 100,
                },
                'volatility': {
                    'mean': volatility.mean(),
                    'std': volatility.std(),
                    'median': volatility.median(),
                    'percentile_75': volatility.quantile(0.75),
                    'percentile_90': volatility.quantile(0.90),
                }
            }
            
            volatility_stats[crypto] = stats
            
            logger.info(f"  收益率统计:")
            logger.info(f"    均值: {stats['returns']['mean']:.4f}%")
            logger.info(f"    标准差: {stats['returns']['std']:.4f}%")
            logger.info(f"    95分位: {stats['returns']['percentile_95']:.4f}%")
            logger.info(f"    99分位: {stats['returns']['percentile_99']:.4f}%")
            logger.info(f"  波动率统计（年化）:")
            logger.info(f"    均值: {stats['volatility']['mean']:.2f}%")
            logger.info(f"    标准差: {stats['volatility']['std']:.2f}%")
            logger.info(f"    中位数: {stats['volatility']['median']:.2f}%")
            
        except Exception as e:
            logger.error(f"  {crypto}: 分析失败 - {e}")
    
    return volatility_stats


def calculate_zscore_distribution(cryptos: List[str], window_size: int = 72):
    """计算Z-Score分布，确定合适阈值"""
    logger.info(f"\n{'='*60}")
    logger.info("计算Z-Score分布")
    logger.info(f"{'='*60}")
    
    zscore_stats = {}
    
    for crypto in cryptos:
        logger.info(f"\n分析 {crypto}...")
        
        try:
            ticker = yf.Ticker(crypto)
            df = ticker.history(period='1y', interval='1h')
            
            if df.empty:
                continue
            
            # 计算Z-Score
            detector = ZScoreDetector(window_size=window_size, threshold=3.0, time_interval='hour')
            
            zscores = []
            for i in range(window_size, len(df)):
                current_price = df['Close'].iloc[i]
                history = df['Close'].iloc[i-window_size:i]
                
                # 计算Z-Score
                mean = history.mean()
                std = history.std()
                if std > 0:
                    z_score = abs((current_price - mean) / std)
                    zscores.append(z_score)
            
            zscores = np.array(zscores)
            
            # 计算Z-Score分布
            stats = {
                'mean': zscores.mean(),
                'std': zscores.std(),
                'median': np.median(zscores),
                'percentile_90': np.percentile(zscores, 90),
                'percentile_95': np.percentile(zscores, 95),
                'percentile_99': np.percentile(zscores, 99),
                'percentile_99_5': np.percentile(zscores, 99.5),
            }
            
            zscore_stats[crypto] = stats
            
            logger.info(f"  Z-Score分布:")
            logger.info(f"    均值: {stats['mean']:.2f}")
            logger.info(f"    标准差: {stats['std']:.2f}")
            logger.info(f"    中位数: {stats['median']:.2f}")
            logger.info(f"    90分位: {stats['percentile_90']:.2f}")
            logger.info(f"    95分位: {stats['percentile_95']:.2f}")
            logger.info(f"    99分位: {stats['percentile_99']:.2f}")
            logger.info(f"    99.5分位: {stats['percentile_99_5']:.2f}")
            
        except Exception as e:
            logger.error(f"  {crypto}: 分析失败 - {e}")
    
    return zscore_stats


def recommend_thresholds(zscore_stats: Dict, volatility_stats: Dict):
    """推荐异常阈值"""
    logger.info(f"\n{'='*60}")
    logger.info("推荐异常阈值")
    logger.info(f"{'='*60}")
    
    # 股票基准
    stock_thresholds = {
        'high': 4.0,
        'medium': 3.0
    }
    
    # 计算平均波动率比例
    avg_volatility = np.mean([stats['volatility']['mean'] for stats in volatility_stats.values()])
    stock_volatility = 2.5  # 股票平均波动率约2.5%
    
    volatility_ratio = avg_volatility / stock_volatility
    
    logger.info(f"\n波动率对比:")
    logger.info(f"  股票平均波动率: {stock_volatility:.2f}%")
    logger.info(f"  加密货币平均波动率: {avg_volatility:.2f}%")
    logger.info(f"  波动率比例: {volatility_ratio:.2f}x")
    
    # 方法1：基于波动率比例调整
    threshold_by_volatility = {
        'high': stock_thresholds['high'] * volatility_ratio,
        'medium': stock_thresholds['medium'] * volatility_ratio
    }
    
    # 方法2：基于Z-Score分布（使用99分位和95分位）
    avg_percentile_99 = np.mean([stats['percentile_99'] for stats in zscore_stats.values()])
    avg_percentile_95 = np.mean([stats['percentile_95'] for stats in zscore_stats.values()])
    
    threshold_by_percentile = {
        'high': avg_percentile_99,
        'medium': avg_percentile_95
    }
    
    logger.info(f"\n推荐阈值方案:")
    logger.info(f"  方案1（基于波动率）:")
    logger.info(f"    high: {threshold_by_volatility['high']:.2f}")
    logger.info(f"    medium: {threshold_by_volatility['medium']:.2f}")
    
    logger.info(f"  方案2（基于Z-Score分布）:")
    logger.info(f"    high: {threshold_by_percentile['high']:.2f}")
    logger.info(f"    medium: {threshold_by_percentile['medium']:.2f}")
    
    # 推荐方案（取平均）
    recommended = {
        'high': (threshold_by_volatility['high'] + threshold_by_percentile['high']) / 2,
        'medium': (threshold_by_volatility['medium'] + threshold_by_percentile['medium']) / 2
    }
    
    logger.info(f"\n✅ 最终推荐:")
    logger.info(f"    high: {recommended['high']:.2f}")
    logger.info(f"    medium: {recommended['medium']:.2f}")
    
    return recommended


def main():
    parser = argparse.ArgumentParser(description='加密货币异常阈值校准')
    parser.add_argument('--cryptos', nargs='+', default=['BTC-USD', 'ETH-USD'], help='加密货币列表')
    parser.add_argument('--period', default='1y', help='数据周期')
    
    args = parser.parse_args()
    
    # 分析波动率分布
    volatility_stats = analyze_volatility_distribution(args.cryptos, args.period)
    
    # 计算Z-Score分布
    zscore_stats = calculate_zscore_distribution(args.cryptos)
    
    # 推荐阈值
    recommended = recommend_thresholds(zscore_stats, volatility_stats)
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'cryptos': args.cryptos,
        'volatility_stats': volatility_stats,
        'zscore_stats': zscore_stats,
        'recommended_thresholds': recommended
    }
    
    import json
    with open('output/crypto_threshold_calibration.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"\n结果已保存: output/crypto_threshold_calibration.json")


if __name__ == "__main__":
    main()
