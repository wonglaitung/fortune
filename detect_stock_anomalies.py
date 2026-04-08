#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股异常检测脚本（改进版）

功能：
- 检测自选股价格和成交量异常（基于Z-Score）
- 支持高异常（Z-Score >= 4.0）和中异常（Z-Score >= 3.0）
- 支持快速模式（Z-Score only）和深度模式（Z-Score + Isolation Forest）
- 发送邮件警报
- 集成到综合分析系统

使用方法：
  python3 detect_stock_anomalies.py --mode standalone --mode-type deep   # 深度模式（Z-Score + Isolation Forest）
  python3 detect_stock_anomalies.py --mode integrated --mode-type deep   # 集成模式（返回数据）
  python3 detect_stock_anomalies.py --mode test --stocks 0700.HK 0939.HK  # 测试模式

依赖：
  - config.py：配置文件，包含WATCHLIST
  - anomaly_detector/zscore_detector.py：Z-Score异常检测器
  - anomaly_detector/isolation_forest_detector.py：Isolation Forest异常检测器
  - anomaly_detector/feature_extractor.py：特征提取器
  - anomaly_detector/anomaly_integrator.py：异常整合器
  - data_services/technical_analysis.py：技术指标计算
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WATCHLIST, STOCK_SECTOR_MAPPING
from data_services.technical_analysis import TechnicalAnalyzerV2

# Anomaly detection imports
try:
    from anomaly_detector.zscore_detector import ZScoreDetector
    from anomaly_detector.isolation_forest_detector import IsolationForestDetector
    from anomaly_detector.feature_extractor import FeatureExtractor
    from anomaly_detector.anomaly_integrator import AnomalyIntegrator
    from anomaly_detector.cache import AnomalyCache
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Anomaly detection not available: {e}")
    ANOMALY_DETECTION_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnomalyDetector:
    """股票异常检测器（使用 anomaly_detector 模块）"""

    def __init__(self, window_size: int = 30, threshold_high: float = 4.0, threshold_medium: float = 3.0, use_deep_analysis: bool = False, time_interval: str = 'day'):
        """
        初始化异常检测器
        
        Args:
            window_size: 滚动窗口大小（天或小时，取决于time_interval）
            threshold_high: 高异常Z-Score阈值
            threshold_medium: 中异常Z-Score阈值
            use_deep_analysis: 是否使用深度分析（Z-Score + Isolation Forest）
            time_interval: 时间间隔类型（'day' 或 'hour'）
        """
        self.window_size = window_size
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        self.use_deep_analysis = use_deep_analysis
        self.time_interval = time_interval
        self.technical_analyzer = TechnicalAnalyzerV2()
        
        # 使用 anomaly_detector 模块
        if ANOMALY_DETECTION_AVAILABLE:
            self.zscore_detector = ZScoreDetector(window_size=window_size, threshold=threshold_medium, time_interval=time_interval)
            
            # 只在深度分析模式下初始化 Isolation Forest
            if use_deep_analysis:
                # 根据时间间隔设置异常类型（股票检测应使用 stock 前缀）
                anomaly_type = f"stock_{time_interval}" if time_interval == 'hour' else 'stock'
                self.if_detector = IsolationForestDetector(
                    contamination=0.05,
                    random_state=42,
                    anomaly_type=anomaly_type
                )
                self.feature_extractor = FeatureExtractor()
                self.cache = AnomalyCache()
                self.integrator = AnomalyIntegrator(self.cache)
            else:
                self.if_detector = None
                self.feature_extractor = None
                self.cache = None
                self.integrator = None
        else:
            self.zscore_detector = None
            self.if_detector = None
            self.feature_extractor = None
            self.cache = None
            self.integrator = None

    def detect_anomalies(self, stocks: List[str], period: str = '3mo', target_date: str = None) -> List[Dict]:
        """
        检测股票异常（使用 anomaly_detector 模块）
        
        Args:
            stocks: 股票代码列表（如 ['0700.HK', '0939.HK']）
            period: 数据周期（默认3个月）
            target_date: 目标日期（YYYY-MM-DD格式），如果为None则检测当天异常
            
        Returns:
            异常列表，每个异常包含：
            {
                'stock': '0700.HK',
                'name': '腾讯控股',
                'type': 'price' or 'volume' or 'isolation_forest',
                'severity': 'high' or 'medium',
                'z_score': 4.2,  # Z-Score值（仅price和volume类型）
                'value': 320.50,  # 价格、成交量或异常评分
                'current_price': 320.50,
                'change_1d': 3.2,
                'change_5d': -5.8,
                'rsi': 68.5,
                'bollinger_position': 'upper',
                'macd_signal': 'bullish',
                'timestamp': datetime object,
                'anomaly_date': '2026-04-03',  # 异常发生日期
                'anomaly_reason': '价格异常波动（收盘价异常）'  # 异常原因
            }
        """
        # 确定目标日期
        if target_date:
            try:
                from datetime import timezone
                target_dt = datetime.strptime(target_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except ValueError:
                logger.error(f"无效的日期格式: {target_date}，使用今天")
                target_dt = datetime.now(timezone.utc)
        else:
            from datetime import timezone
            target_dt = datetime.now(timezone.utc)
        
        target_date_str = target_dt.strftime('%Y-%m-%d')
        logger.info(f"检测 {target_date_str} 的异常")
        
        anomalies = []
        
        for stock in stocks:
            try:
                logger.info(f"检测 {stock} 的异常...")
                
                # 获取股票数据
                df = self.get_stock_data(stock, period)
                
                if df is None or len(df) < self.window_size:
                    logger.warning(f"{stock} 数据不足，跳过")
                    continue
                
                # 使用 anomaly_detector 模块检测异常
                if not ANOMALY_DETECTION_AVAILABLE:
                    logger.warning(f"{stock} 异常检测模块不可用，跳过")
                    continue
                
                # Z-Score 检测（Layer 1） - 始终执行
                zscore_anomalies = []
                
                # Price anomaly
                current_price = df['Close'].iloc[-1]
                price_anomaly = self.zscore_detector.detect_anomaly(
                    metric_name='price',
                    current_value=current_price,
                    history=df['Close'],
                    timestamp=target_dt
                )
                if price_anomaly:
                    zscore_anomalies.append(price_anomaly)
                
                # Volume anomaly（如果有成交量数据）
                if 'Volume' in df.columns and len(df['Volume']) > 0:
                    current_volume = df['Volume'].iloc[-1]
                    volume_anomaly = self.zscore_detector.detect_anomaly(
                        metric_name='volume',
                        current_value=current_volume,
                        history=df['Volume'],
                        timestamp=target_dt
                    )
                    if volume_anomaly:
                        zscore_anomalies.append(volume_anomaly)
                
                # Isolation Forest 检测（Layer 2） - 仅在深度分析模式下执行
                if_anomalies = []
                if self.use_deep_analysis and self.if_detector and self.feature_extractor:
                    try:
                        # 提取特征
                        features, timestamps = self.feature_extractor.extract_features(df)
                        
                        if not features.empty:
                            # 训练模型
                            self.if_detector.train(features)
                            
                            # 检测异常（指定日期）
                            if_anomalies = self.if_detector.detect_anomalies_by_date(
                                features=features,
                                timestamps=timestamps,
                                target_date=target_dt
                            )
                    except Exception as e:
                        logger.warning(f"{stock} Isolation Forest 检测失败: {e}")
                
                # 整合结果
                if self.integrator:
                    result = self.integrator.integrate(
                        zscore_anomalies=zscore_anomalies,
                        if_anomalies=if_anomalies,
                        timestamp=target_dt
                    )
                    
                    # 转换为统一格式
                    if result['has_anomaly']:
                        for anomaly in result['anomalies']:
                            # 获取异常发生的日期
                            anomaly_timestamp = anomaly.get('timestamp', target_dt)
                            anomaly_date_str = anomaly_timestamp.strftime('%Y-%m-%d') if hasattr(anomaly_timestamp, 'strftime') else target_date_str
                            
                            # 找到异常日期在DataFrame中的索引位置
                            df_anomaly = df[df.index.normalize() == pd.Timestamp(anomaly_timestamp).normalize()]
                            
                            if df_anomaly.empty:
                                # 如果找不到精确匹配，尝试按日期匹配
                                target_date_only = pd.to_datetime(anomaly_date_str).date()
                                df_anomaly = df[df.index.date == target_date_only]
                            
                            if df_anomaly.empty:
                                # 如果还是找不到，使用最新数据
                                df_anomaly = df.tail(1)
                            
                            df_anomaly_row = df_anomaly.iloc[0]
                            
                            # 计算异常日期的技术指标（使用完整的历史数据）
                            # 技术指标如RSI需要足够的历史数据
                            indicators = self.calculate_indicators(df)
                            
                            # 映射异常严重程度
                            z_score = anomaly.get('z_score', 0)
                            severity = anomaly.get('severity', self.get_severity(z_score))
                            
                            # 分析异常原因
                            logger.info(f"分析异常: stock={stock}, anomaly_type={anomaly.get('type')}, features={anomaly.get('features', {})}")
                            anomaly_reason = self._analyze_anomaly_reason(anomaly, df_anomaly_row)
                            logger.info(f"异常原因: {anomaly_reason}")
                            
                            # 计算异常日期的涨跌幅
                            current_price = df_anomaly_row['Close']
                            change_1d = 0
                            change_5d = 0
                            
                            if len(df_anomaly) > 1:
                                idx = df_anomaly.index.get_loc(df_anomaly_row.name)
                                if idx > 0:
                                    prev_close = df_anomaly.iloc[idx-1]['Close']
                                    change_1d = (current_price / prev_close - 1) * 100
                                
                                if idx >= 5:
                                    prev_5d_close = df_anomaly.iloc[idx-5]['Close']
                                    change_5d = (current_price / prev_5d_close - 1) * 100
                            
                            anomalies.append({
                                'stock': stock,
                                'name': self.get_stock_name(stock),
                                'type': anomaly.get('type', 'price'),
                                'severity': severity,
                                'z_score': abs(z_score) if anomaly.get('type') != 'isolation_forest' else 0,
                                'value': anomaly.get('anomaly_score', 0) if anomaly.get('type') == 'isolation_forest' else anomaly.get('value', current_price),
                                'current_price': current_price,
                                'change_1d': change_1d,
                                'change_5d': change_5d,
                                'rsi': indicators.get('rsi', 0),
                                'bollinger_position': indicators.get('bollinger_position', 'middle'),
                                'macd_signal': indicators.get('macd_signal', 'neutral'),
                                'timestamp': anomaly_timestamp,
                                'anomaly_date': anomaly_date_str,
                                'anomaly_reason': anomaly_reason
                            })
                            
                            logger.info(f"{stock} {anomaly_date_str} 检测到 {anomaly.get('type')} {severity}级异常: {anomaly_reason}")
                else:
                    # 如果没有整合器，直接使用 Z-Score 结果
                    for anomaly in zscore_anomalies:
                        # 计算技术指标
                        indicators = self.calculate_indicators(df)
                        
                        # 映射异常严重程度
                        z_score = anomaly.get('z_score', 0)
                        severity = self.get_severity(z_score)
                        
                        anomalies.append({
                            'stock': stock,
                            'name': self.get_stock_name(stock),
                            'type': anomaly.get('type', 'price'),
                            'severity': anomaly.get('severity', severity),
                            'z_score': abs(z_score),
                            'value': anomaly.get('value', current_price),
                            'current_price': current_price,
                            'change_1d': (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0,
                            'change_5d': (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0,
                            'rsi': indicators.get('rsi', 0),
                            'bollinger_position': indicators.get('bollinger_position', 'middle'),
                            'macd_signal': indicators.get('macd_signal', 'neutral'),
                            'timestamp': anomaly.get('timestamp', datetime.now())
                        })
                        
                        logger.info(f"{stock} 检测到 {anomaly.get('type')} {severity}级异常（Z-Score: {abs(z_score):.2f}）")
                
                # 清理旧缓存条目
                if self.cache:
                    self.cache.cleanup_expired(max_age_hours=48)
                
            except Exception as e:
                logger.error(f"检测 {stock} 异常时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return anomalies

    def get_stock_data(self, stock: str, period: str = '3mo', interval: str = None) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock: 股票代码（如 '0700.HK'）
            period: 数据周期（默认3个月）
            interval: 数据间隔（如 '1d'、'1h'），默认为 None（使用 self.time_interval）
            
        Returns:
            DataFrame 包含价格数据，或 None
        """
        try:
            # 如果没有指定 interval，使用实例的 time_interval
            if interval is None:
                interval = '1h' if self.time_interval == 'hour' else '1d'
            
            logger.info(f"下载 {stock} 数据（周期：{period}，间隔：{interval}）...")
            ticker = yf.Ticker(stock)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"{stock} 数据下载失败：无数据")
                return None
            
            logger.info(f"{stock} 数据下载成功，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock} 数据失败: {e}")
            return None

    def get_stock_name(self, stock: str) -> str:
        """
        获取股票名称
        
        Args:
            stock: 股票代码（如 '0700.HK'）
            
        Returns:
            股票名称（如 '腾讯控股'）
        """
        if stock in STOCK_SECTOR_MAPPING:
            # 从 STOCK_SECTOR_MAPPING 获取（字典格式）
            return STOCK_SECTOR_MAPPING[stock].get('name', stock)
        elif stock in WATCHLIST:
            # 从 WATCHLIST 获取（字符串格式）
            return WATCHLIST[stock] if isinstance(WATCHLIST[stock], str) else WATCHLIST[stock].get('name', stock)
        return stock

    def get_severity(self, z_score: float) -> Optional[str]:
        """
        根据Z-Score判断异常级别
        
        Args:
            z_score: Z-Score值
            
        Returns:
            异常级别：'high', 'medium', 或 None（无异常）
        """
        abs_z_score = abs(z_score)
        
        if abs_z_score >= self.threshold_high:
            return 'high'
        elif abs_z_score >= self.threshold_medium:
            return 'medium'
        else:
            return None
    
    def _analyze_anomaly_reason(self, anomaly: Dict, row_data) -> str:
        """
        分析异常原因

        Args:
            anomaly: 异常字典
            row_data: 异常日期的数据行

        Returns:
            异常原因描述
        """
        anomaly_type = anomaly.get('type', 'price')
        features = anomaly.get('features', {})

        logger.debug(f"分析异常原因: type={anomaly_type}, features keys={list(features.keys()) if features else 'empty'}")
        logger.debug(f"Row data columns: {list(row_data.index)}")

        # 处理'stock'类型异常（来自Isolation Forest，包括 stock_hour 小时级别）
        # 同时支持 crypto_hour/crypto_hourly 以兼容旧缓存数据
        if anomaly_type in ('stock', 'stock_hour', 'isolation_forest', 'crypto_hour', 'crypto_hourly'):
            if not features:
                logger.warning("异常但没有特征数据")
                return '多维特征异常（综合指标异常）'

            # 特征名称映射和分析
            feature_analysis = []

            # 涨跌幅 (return_rate)
            if 'return_rate' in features:
                return_rate = features['return_rate']
                if abs(return_rate) > 0.05:  # 超过5%
                    feature_analysis.append(f'单日涨跌幅{return_rate*100:.2f}%')

            # 波动率 (volatility_20d)
            if 'volatility_20d' in features:
                volatility = features['volatility_20d']
                if volatility > 0.03:  # 超过3%
                    feature_analysis.append(f'高波动率{volatility*100:.1f}%')

            # 成交量比率 (volume_ratio)
            if 'volume_ratio' in features:
                vol_ratio = features['volume_ratio']
                if vol_ratio > 2.0:
                    feature_analysis.append(f'成交量{vol_ratio:.1f}倍')

            # RSI指标 (rsi)
            if 'rsi' in features:
                rsi = features['rsi']
                if rsi > 70:
                    feature_analysis.append(f'RSI超买({rsi:.1f})')
                elif rsi < 30:
                    feature_analysis.append(f'RSI超卖({rsi:.1f})')

            # 布林带位置 (bb_position)
            if 'bb_position' in features:
                bb_pos = features['bb_position']
                if bb_pos > 0.9:
                    feature_analysis.append('接近布林带上轨')
                elif bb_pos < 0.1:
                    feature_analysis.append('接近布林带下轨')

            # MACD
            if 'macd' in features:
                macd = features['macd']
                if abs(macd) > 5:
                    feature_analysis.append(f'MACD强势({macd:.2f})')

            # 均线差价 (ma20_diff, ma50_diff)
            if 'ma20_diff' in features:
                ma20_diff = features['ma20_diff']
                if abs(ma20_diff) > 0.05:
                    feature_analysis.append(f'偏离20日均线{ma20_diff*100:.1f}%')

            logger.debug(f"特征分析结果: {feature_analysis}")

            if feature_analysis:
                return f'多维特征异常（{", ".join(feature_analysis[:3])}）'
            else:
                # 如果没有明显异常特征，显示原始特征值
                feature_values = []
                for key, value in features.items():
                    if abs(value) > 0.05 or key in ['rsi', 'bb_position']:  # 显示有意义的值
                        feature_values.append(f"{key}={value:.3f}")
                if feature_values:
                    return f'多维特征异常（关键特征: {", ".join(feature_values[:3])}）'
                return '多维特征异常（综合指标异常，需关注）'

        if anomaly_type == 'price':
            # 价格异常：分析涨跌幅
            if 'Change %' in row_data.index:
                change_pct = row_data['Change %']
                if abs(change_pct) > 5:
                    return f'价格异常（单日涨跌幅{change_pct:.2f}%，大幅波动）'
                elif abs(change_pct) > 3:
                    return f'价格异常（单日涨跌幅{change_pct:.2f}%，显著波动）'
                else:
                    return '价格异常（收盘价显著偏离历史均值）'
            return '价格异常（收盘价显著偏离历史均值）'

        elif anomaly_type == 'volume':
            # 成交量异常
            if 'Volume' in row_data.index and len(row_data.index) > 1:
                volume = row_data['Volume']
                volume_mean = row_data.get('Volume_MA', volume)
                volume_ratio = volume / volume_mean if volume_mean > 0 else 1

                if volume_ratio > 3:
                    return f'成交量异常（成交量{volume:,.0f}，为均值的{volume_ratio:.1f}倍，巨量）'
                elif volume_ratio > 2:
                    return f'成交量异常（成交量{volume:,.0f}，为均值的{volume_ratio:.1f}倍，放量）'
                else:
                    return '成交量异常（成交量显著偏离历史均值）'
            return '成交量异常（成交量显著偏离历史均值）'

        logger.warning(f"未知的异常类型: {anomaly_type}")
        return '异常（原因未知）'

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算技术指标
        
        Args:
            df: 股票数据（包含价格和成交量）
            
        Returns:
            技术指标字典：
            {
                'rsi': 68.5,
                'bollinger_position': 'upper',
                'macd_signal': 'bullish'
            }
        """
        try:
            # 检查数据量，如果数据不足（少于20天），使用全部可用数据
            if len(df) < 20:
                logger.warning(f"数据量不足（{len(df)}天），计算指标可能不准确")
                # 继续尝试计算，但可能会得到一些nan值
            
            # 使用 TechnicalAnalyzerV2 计算指标
            indicators_df = self.technical_analyzer.calculate_all_indicators(df.copy())
            
            # 获取最新指标值
            latest = indicators_df.iloc[-1]
            
            rsi = latest.get('RSI', 50.0)
            # 如果RSI是nan或inf，使用默认值50
            if pd.isna(rsi) or not np.isfinite(rsi):
                logger.warning(f"RSI计算无效（{rsi}），使用默认值50")
                rsi = 50.0
            
            return {
                'rsi': rsi,
                'bollinger_position': self._get_bollinger_position(latest),
                'macd_signal': self._get_macd_signal(latest)
            }
            
        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'rsi': 50.0,
                'bollinger_position': 'middle',
                'macd_signal': 'neutral'
            }

    def _get_bollinger_position(self, latest_row: pd.Series) -> str:
        """
        获取布林带位置
        
        Args:
            latest_row: 最新数据行
            
        Returns:
            位置：'upper', 'middle', 'lower'
        """
        close = latest_row.get('Close', 0)
        bb_upper = latest_row.get('BB_Upper', 0)
        bb_lower = latest_row.get('BB_Lower', 0)
        bb_middle = latest_row.get('BB_Middle', 0)
        
        if close > bb_middle + (bb_upper - bb_middle) * 0.8:
            return 'upper'
        elif close < bb_middle - (bb_middle - bb_lower) * 0.8:
            return 'lower'
        else:
            return 'middle'

    def _get_macd_signal(self, latest_row: pd.Series) -> str:
        """
        获取MACD信号
        
        Args:
            latest_row: 最新数据行
            
        Returns:
            信号：'bullish', 'bearish', 'neutral'
        """
        macd = latest_row.get('MACD', 0)
        macd_signal = latest_row.get('MACD_signal', 0)
        
        if macd > macd_signal:
            return 'bullish'
        elif macd < macd_signal:
            return 'bearish'
        else:
            return 'neutral'


def format_anomaly_email(anomalies: List[Dict]) -> str:
    """
    格式化异常邮件内容
    
    Args:
        anomalies: 异常列表
        
    Returns:
        格式化的邮件内容（Markdown格式）
    """
    if not anomalies:
        lines = []
        lines.append("✅ 未检测到异常")
        lines.append("\n" + "=" * 80)
        lines.append("💡 提醒")
        lines.append("  - 异常是警告信号，不一定是交易信号")
        lines.append("  - 不要看到异常就立即卖出")
        lines.append("  - 综合考虑基本面和市场情绪")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    lines = []
    
    # 按严重程度和类型分组
    high_price_anomalies = [a for a in anomalies if a['severity'] == 'high' and a['type'] == 'price']
    high_volume_anomalies = [a for a in anomalies if a['severity'] == 'high' and a['type'] == 'volume']
    medium_price_anomalies = [a for a in anomalies if a['severity'] == 'medium' and a['type'] == 'price']
    medium_volume_anomalies = [a for a in anomalies if a['severity'] == 'medium' and a['type'] == 'volume']
    if_anomalies = [a for a in anomalies if a['type'] in ('isolation_forest', 'stock_hour', 'stock')]

    # Isolation Forest 异常去重：每个股票只显示一次（选择最严重的）
    if_anomalies_dedup = {}
    for a in if_anomalies:
        stock = a['stock']
        if stock not in if_anomalies_dedup:
            if_anomalies_dedup[stock] = a
        else:
            # 如果已有记录，选择更严重的
            existing = if_anomalies_dedup[stock]
            if (existing['severity'] == 'low' and a['severity'] != 'low') or \
               (existing['severity'] == a['severity'] and abs(a.get('value', 0)) > abs(existing.get('value', 0))):
                if_anomalies_dedup[stock] = a
    if_anomalies = list(if_anomalies_dedup.values())
    
    # 高异常（价格）
    if high_price_anomalies:
        lines.append("\n## 🔴 高异常（价格）\n")
        lines.append("| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in high_price_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | **{a['z_score']:.2f}** | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**风险提示**：")
        lines.append("⚠️ 价格异常波动，风险显著增加")
        lines.append("⚠️ 可能是突发利空消息或市场恐慌")
        lines.append("⚠️ 建议立即检查持仓，考虑减仓或止损")
    
    # 高异常（成交量）
    if high_volume_anomalies:
        lines.append("\n## 🔴 高异常（成交量）\n")
        lines.append("| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in high_volume_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | **{a['z_score']:.2f}** | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**风险提示**：")
        lines.append("⚠️ 成交量异常放大，可能有大户进出")
        lines.append("⚠️ 需要关注价格是否伴随成交量变化")
        lines.append("⚠️ 避免在异常期间加仓")
    
    # 中异常（价格）
    if medium_price_anomalies:
        lines.append("\n## ⚠️ 中异常（价格）\n")
        lines.append("| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in medium_price_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | {a['z_score']:.2f} | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**操作建议**：")
        lines.append("- 观察价格是否能突破阻力位")
        lines.append("- 注意止损位设置")
        lines.append("- 避免在异常期间加仓")
    
    # 中异常（成交量）
    if medium_volume_anomalies:
        lines.append("\n## ⚠️ 中异常（成交量）\n")
        lines.append("| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in medium_volume_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | {a['z_score']:.2f} | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**操作建议**：")
        lines.append("- 观察价格是否能突破阻力位")
        lines.append("- 注意止损位设置")
        lines.append("- 避免在异常期间加仓")
    
    # Isolation Forest 异常分组
    if_high_anomalies = [a for a in if_anomalies if a['severity'] == 'high']
    if_medium_anomalies = [a for a in if_anomalies if a['severity'] == 'medium']
    if_low_anomalies = [a for a in if_anomalies if a['severity'] == 'low']
    
    # 高严重度 Isolation Forest 异常
    if if_high_anomalies:
        lines.append("\n## 🔴 Isolation Forest 异常（高）\n")
        lines.append("| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in if_high_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | {a.get('anomaly_date', 'N/A')} | {a.get('anomaly_reason', '未知')} | **{a['value']:.3f}** | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**风险提示**：")
        lines.append("🔴 多维特征严重偏离正常模式")
        lines.append("🔴 可能存在价格操纵或重大消息")
        lines.append("🔴 建议立即检查持仓，考虑减仓")
    
    # 中等严重度 Isolation Forest 异常
    if if_medium_anomalies:
        lines.append("\n## ⚠️ Isolation Forest 异常（中）\n")
        lines.append("| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in if_medium_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | {a.get('anomaly_date', 'N/A')} | {a.get('anomaly_reason', '未知')} | {a['value']:.3f} | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**操作建议**：")
        lines.append("- 多维特征出现异常，值得关注")
        lines.append("- 观察后续价格走势确认")
        lines.append("- 避免在异常期间加仓")
    
    # 低严重度 Isolation Forest 异常
    if if_low_anomalies:
        lines.append("\n## 🔬 Isolation Forest 异常（低）\n")
        lines.append("| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | 5日变化 | RSI | 布林带 | MACD |")
        lines.append("|---------|---------|---------|---------|---------|---------|---------|---------|-----|--------|------|")
        
        for a in if_low_anomalies:
            change_1d_icon = "📈" if a['change_1d'] > 0 else "📉"
            change_5d_icon = "📈" if a['change_5d'] > 0 else "📉"
            lines.append(f"| {a['stock']} | {a['name']} | {a.get('anomaly_date', 'N/A')} | {a.get('anomaly_reason', '未知')} | {a['value']:.3f} | {a['current_price']:.2f} | {change_1d_icon} {a['change_1d']:+.2f}% | {change_5d_icon} {a['change_5d']:+.2f}% | {a['rsi']:.1f} | {get_bollinger_position_cn(a['bollinger_position'])} | {get_macd_signal_cn(a['macd_signal'])} |")
        
        lines.append("\n**说明**：")
        lines.append("- 基于多维特征检测的轻微异常")
        lines.append("- 可能是价格模式轻微变化")
        lines.append("- 建议结合基本面分析")
    
    lines.append("\n" + "=" * 80)
    lines.append("💡 提醒")
    lines.append("  - 异常是警告信号，不一定是交易信号")
    lines.append("  - 不要看到异常就立即卖出")
    lines.append("  - 综合考虑基本面和市场情绪")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def get_bollinger_position_cn(position: str) -> str:
    """获取布林带位置描述（中文）"""
    if position == 'upper':
        return "上轨（强势）"
    elif position == 'lower':
        return "下轨（弱势）"
    else:
        return "中轨（正常）"


def get_macd_signal_cn(signal: str) -> str:
    """获取MACD信号描述（中文）"""
    if signal == 'bullish':
        return "金叉（上涨）"
    elif signal == 'bearish':
        return "死叉（下跌）"
    else:
        return "中性"


def format_anomaly_email_html(anomalies: List[Dict]) -> str:
    """
    格式化异常邮件内容（HTML格式）
    
    Args:
        anomalies: 异常列表
        
    Returns:
        格式化的邮件内容（HTML格式）
    """
    # HTML 头部和样式
    html_header = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .high { color: #dc3545; font-weight: bold; }
            .medium { color: #ffc107; font-weight: bold; }
            .warning { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
            .positive { color: #28a745; }
            .negative { color: #dc3545; }
            .no-anomaly { text-align: center; padding: 20px; color: #28a745; font-size: 18px; }
        </style>
    </head>
    <body>
    """
    
    # HTML 尾部
    html_footer = """
    <hr>
    <p><strong>💡 提醒</strong></p>
    <ul>
        <li>异常是警告信号，不一定是交易信号</li>
        <li>不要看到异常就立即卖出</li>
        <li>综合考虑基本面和市场情绪</li>
    </ul>
    </body>
    </html>
    """
    
    if not anomalies:
        return html_header + '<p class="no-anomaly">✅ 未检测到异常</p>' + html_footer
    
    html_parts = []
    html_parts.append(html_header)
    
    # 按严重程度和类型分组
    high_price_anomalies = [a for a in anomalies if a['severity'] == 'high' and a['type'] == 'price']
    high_volume_anomalies = [a for a in anomalies if a['severity'] == 'high' and a['type'] == 'volume']
    medium_price_anomalies = [a for a in anomalies if a['severity'] == 'medium' and a['type'] == 'price']
    medium_volume_anomalies = [a for a in anomalies if a['severity'] == 'medium' and a['type'] == 'volume']
    if_anomalies = [a for a in anomalies if a['type'] in ('isolation_forest', 'stock_hour', 'stock')]

    # Isolation Forest 异常去重
    if_anomalies_dedup = {}
    for a in if_anomalies:
        stock = a['stock']
        if stock not in if_anomalies_dedup:
            if_anomalies_dedup[stock] = a
        else:
            existing = if_anomalies_dedup[stock]
            if (existing['severity'] == 'low' and a['severity'] != 'low') or \
               (existing['severity'] == a['severity'] and abs(a.get('value', 0)) > abs(existing.get('value', 0))):
                if_anomalies_dedup[stock] = a
    if_anomalies = list(if_anomalies_dedup.values())
    
    def format_change(value: float) -> str:
        """格式化涨跌幅"""
        css_class = 'positive' if value > 0 else 'negative'
        icon = '📈' if value > 0 else '📉'
        return f'<span class="{css_class}">{icon} {value:+.2f}%</span>'
    
    # 高异常（价格）
    if high_price_anomalies:
        html_parts.append("<h2>🔴 高异常（价格）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>Z-Score</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in high_price_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td class='high'>{a['z_score']:.2f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">⚠️ 价格异常波动，风险显著增加<br>⚠️ 可能是突发利空消息或市场恐慌<br>⚠️ 建议立即检查持仓，考虑减仓或止损</div>')
    
    # 高异常（成交量）
    if high_volume_anomalies:
        html_parts.append("<h2>🔴 高异常（成交量）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>Z-Score</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in high_volume_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td class='high'>{a['z_score']:.2f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">⚠️ 成交量异常放大，可能有大户进出<br>⚠️ 需要关注价格是否伴随成交量变化<br>⚠️ 避免在异常期间加仓</div>')
    
    # 中异常（价格）
    if medium_price_anomalies:
        html_parts.append("<h2>⚠️ 中异常（价格）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>Z-Score</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in medium_price_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td class='medium'>{a['z_score']:.2f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">- 观察价格是否能突破阻力位<br>- 注意止损位设置<br>- 避免在异常期间加仓</div>')
    
    # 中异常（成交量）
    if medium_volume_anomalies:
        html_parts.append("<h2>⚠️ 中异常（成交量）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>Z-Score</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in medium_volume_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td class='medium'>{a['z_score']:.2f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">- 观察价格是否能突破阻力位<br>- 注意止损位设置<br>- 避免在异常期间加仓</div>')
    
    # Isolation Forest 异常分组
    if_high_anomalies = [a for a in if_anomalies if a['severity'] == 'high']
    if_medium_anomalies = [a for a in if_anomalies if a['severity'] == 'medium']
    if_low_anomalies = [a for a in if_anomalies if a['severity'] == 'low']
    
    # 高严重度 Isolation Forest 异常
    if if_high_anomalies:
        html_parts.append("<h2>🔴 Isolation Forest 异常（高）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>异常日期</th><th>异常原因</th><th>异常评分</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in if_high_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td>{a.get('anomaly_date', 'N/A')}</td><td>{a.get('anomaly_reason', '未知')}</td><td class='high'>{a['value']:.3f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">🔴 多维特征严重偏离正常模式<br>🔴 可能存在价格操纵或重大消息<br>🔴 建议立即检查持仓，考虑减仓</div>')
    
    # 中等严重度 Isolation Forest 异常
    if if_medium_anomalies:
        html_parts.append("<h2>⚠️ Isolation Forest 异常（中）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>异常日期</th><th>异常原因</th><th>异常评分</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in if_medium_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td>{a.get('anomaly_date', 'N/A')}</td><td>{a.get('anomaly_reason', '未知')}</td><td class='medium'>{a['value']:.3f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">- 多维特征出现异常，值得关注<br>- 观察后续价格走势确认<br>- 避免在异常期间加仓</div>')
    
    # 低严重度 Isolation Forest 异常
    if if_low_anomalies:
        html_parts.append("<h2>🔬 Isolation Forest 异常（低）</h2>")
        html_parts.append("<table><tr><th>股票代码</th><th>股票名称</th><th>异常日期</th><th>异常原因</th><th>异常评分</th><th>当前价格</th><th>当日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>")
        for a in if_low_anomalies:
            html_parts.append(f"<tr><td><strong>{a['stock']}</strong></td><td>{a['name']}</td><td>{a.get('anomaly_date', 'N/A')}</td><td>{a.get('anomaly_reason', '未知')}</td><td>{a['value']:.3f}</td><td>{a['current_price']:.2f}</td><td>{format_change(a['change_1d'])}</td><td>{format_change(a['change_5d'])}</td><td>{a['rsi']:.1f}</td><td>{get_bollinger_position_cn(a['bollinger_position'])}</td><td>{get_macd_signal_cn(a['macd_signal'])}</td></tr>")
        html_parts.append("</table>")
        html_parts.append('<div class="warning">- 基于多维特征检测的轻微异常<br>- 可能是价格模式轻微变化<br>- 建议结合基本面分析</div>')
    
    # 添加尾部
    html_parts.append(html_footer)
    
    return "".join(html_parts)


def send_email_alert(anomalies: List[Dict], recipient_email: str = None) -> bool:
    """
    发送异常警报邮件
    
    Args:
        anomalies: 异常列表
        recipient_email: 收件人邮箱（可选，默认从环境变量获取）
        
    Returns:
        是否发送成功
    """
    try:
        # 导入邮件发送功能（从 comprehensive_analysis.py）
        from comprehensive_analysis import send_email
        
        # 获取收件人邮箱
        if recipient_email is None:
            recipient_email = os.environ.get('RECIPIENT_EMAIL', '')
            if ',' in recipient_email:
                recipient_email = recipient_email.split(',')[0]
        
        if not recipient_email:
            print("⚠️ 未配置收件人邮箱，跳过发送邮件")
            return False
        
        # 格式化邮件内容（文本和HTML）
        content = format_anomaly_email(anomalies)
        html_content = format_anomaly_email_html(anomalies)
        
        # 发送邮件（修正参数顺序：subject, content, html_content）
        subject = f"🚨 港股异常检测警报 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        success = send_email(subject, content, html_content)
        
        if success:
            logger.info("📧 邮件警报已发送")
        
        return success
        
    except Exception as e:
        logger.error(f"发送邮件失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='港股异常检测脚本')
    parser.add_argument('--mode', choices=['standalone', 'integrated', 'test'], default='standalone',
                       help='运行模式：standalone（独立运行，发送邮件）、integrated（集成模式，返回数据）、test（测试模式）')
    parser.add_argument('--mode-type', choices=['quick', 'deep'], default='deep',
                       help='异常检测模式：deep（Z-Score + Isolation Forest，深度，推荐）')
    parser.add_argument('--time-interval', choices=['day', 'hour'], default='day',
                       help='时间间隔：day（每日）、hour（每小时）')
    parser.add_argument('--stocks', nargs='+', help='股票代码列表（如 0700.HK 0939.HK）')
    parser.add_argument('--date', help='指定检测日期（YYYY-MM-DD格式），默认检测当天异常')
    parser.add_argument('--no-email', action='store_true', help='不发送邮件')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 确定股票列表
    if args.stocks:
        stocks = args.stocks
    else:
        # 使用配置文件中的WATCHLIST
        stocks = list(WATCHLIST.keys())
    
    # 根据时间间隔设置参数
    if args.time_interval == 'hour':
        window_size = 72  # 3天 = 72小时
        period = '1mo'    # 1个月数据
    else:  # day
        window_size = 30  # 30天
        period = '3mo'    # 3个月数据
    
    logger.info(f"开始检测异常，股票数量：{len(stocks)}")
    logger.info(f"检测模式：{args.mode}，分析模式：{args.mode_type}，时间间隔：{args.time_interval}")
    if args.date:
        logger.info(f"检测日期：{args.date}")
    
    # 创建检测器实例
    detector = StockAnomalyDetector(
        window_size=window_size,
        threshold_high=4.0,
        threshold_medium=3.0,
        use_deep_analysis=(args.mode_type == 'deep'),
        time_interval=args.time_interval
    )
    
    # 检测异常
    anomalies = detector.detect_anomalies(stocks, period=period, target_date=args.date)
    
    # 根据模式处理结果
    if args.mode == 'standalone':
        # 独立模式：发送邮件
        # 过滤低严重度异常（与 Z-Score 保持一致，只发送 high 和 medium）
        significant_anomalies = [a for a in anomalies if a['severity'] in ('high', 'medium')]
        
        if anomalies:
            print(f"\n🚨 检测到 {len(anomalies)} 个异常")
            if len(significant_anomalies) < len(anomalies):
                print(f"   其中 {len(significant_anomalies)} 个需要关注（已过滤 {len(anomalies) - len(significant_anomalies)} 个低严重度异常）\n")
            else:
                print()
            print(format_anomaly_email(anomalies))
            
            # 只发送高/中严重度异常的邮件（除非 --no-email）
            if not args.no_email:
                if significant_anomalies:
                    send_email_alert(significant_anomalies)
                else:
                    logger.info("没有高/中严重度异常，跳过邮件发送")
        else:
            print("\n✅ 未检测到异常")
    
    elif args.mode == 'integrated':
        # 集成模式：返回JSON格式
        if anomalies:
            print(f"检测到 {len(anomalies)} 个异常")
        
        # 输出JSON格式（便于其他程序解析）
        import json
        output = {
            'has_anomaly': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'timestamp': datetime.now().isoformat(),
            'anomalies': anomalies
        }
        
        print(json.dumps(output, ensure_ascii=False, indent=2))
    
    elif args.mode == 'test':
        # 测试模式：输出详细信息
        print(f"\n{'=' * 80}")
        print("异常检测结果")
        print(f"{'=' * 80}")
        
        if anomalies:
            print(f"\n检测到 {len(anomalies)} 个异常\n")
            
            # 按类型分组
            price_anomalies = [a for a in anomalies if a['type'] == 'price']
            volume_anomalies = [a for a in anomalies if a['type'] == 'volume']
            if_anomalies = [a for a in anomalies if a['type'] == 'isolation_forest']
            
            if price_anomalies:
                print("价格异常：")
                for a in price_anomalies:
                    severity_icon = '🔴' if a['severity'] == 'high' else '⚠️'
                    print(f"  {severity_icon} {a['name']}（{a['stock']}）：{a['severity']}（Z-Score: {a['z_score']:.2f}）")
                    print(f"     当前价格：{a['current_price']:.2f}，当日：{a['change_1d']:+.2f}%，5日：{a['change_5d']:+.2f}%")
            
            if volume_anomalies:
                print("\n成交量异常：")
                for a in volume_anomalies:
                    severity_icon = '🔴' if a['severity'] == 'high' else '⚠️'
                    print(f"  {severity_icon} {a['name']}（{a['stock']}）：{a['severity']}（Z-Score: {a['z_score']:.2f}）")
                    print(f"     当前价格：{a['current_price']:.2f}，当日：{a['change_1d']:+.2f}%，5日：{a['change_5d']:+.2f}%")
            
            if if_anomalies:
                print("\nIsolation Forest 异常：")
                for a in if_anomalies:
                    severity_icon = '🔬'
                    print(f"  {severity_icon} {a['name']}（{a['stock']}）：{a['severity']}（评分：{a['value']:.3f}）")
                    print(f"     当前价格：{a['current_price']:.2f}，当日：{a['change_1d']:+.2f}%，5日：{a['change_5d']:+.2f}%")
            
        else:
            print("✅ 未检测到异常")


if __name__ == "__main__":
    main()