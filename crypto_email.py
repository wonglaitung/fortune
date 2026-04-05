import os
import argparse
import requests
import smtplib
import json
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# 导入技术分析工具
try:
    from data_services.technical_analysis import TechnicalAnalyzer, TechnicalAnalyzerV2, TAVScorer, TAVConfig
    TECHNICAL_ANALYSIS_AVAILABLE = True
    TAV_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    TAV_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用简化指标计算")

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

def get_cryptocurrency_prices(include_market_cap=False, include_24hr_vol=False):
    # 注意：原 URL 末尾有空格，已修正
    url = "https://api.coingecko.com/api/v3/simple/price"
    
    params = {
        'ids': 'bitcoin,ethereum',
        'vs_currencies': 'usd,hkd',
        'include_24hr_change': 'true'
    }
    
    # 添加新参数
    if include_market_cap:
        params['include_market_cap'] = 'true'
    if include_24hr_vol:
        params['include_24hr_vol'] = 'true'
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching cryptocurrency prices: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception during API request: {e}")
        return None

def calculate_technical_indicators(prices):
    """
    计算加密货币技术指标
    """
    if not TECHNICAL_ANALYSIS_AVAILABLE:
        # 如果技术分析工具不可用，使用简化指标计算
        btc = prices.get('bitcoin', {})
        eth = prices.get('ethereum', {})
        
        # 简化的技术指标计算
        indicators = {
            'bitcoin': {
                'rsi': calculate_rsi(btc.get('usd_24h_change', 0)),
                'macd': calculate_macd(btc.get('usd', 0)),
                'price_position': calculate_price_position(btc.get('usd', 0)),
            },
            'ethereum': {
                'rsi': calculate_rsi(eth.get('usd_24h_change', 0)),
                'macd': calculate_macd(eth.get('usd', 0)),
                'price_position': calculate_price_position(eth.get('usd', 0)),
            }
        }
        
        return indicators
    
    # 使用技术分析工具计算更准确的指标
    # 根据TAV可用性选择分析器
    if TAV_AVAILABLE:
        analyzer = TechnicalAnalyzerV2(enable_tav=True)
        use_tav = True
    else:
        analyzer = TechnicalAnalyzer()
        use_tav = False
    
    # 获取历史数据进行技术分析
    indicators = {}
    
    # 获取比特币历史数据
    try:
        btc_ticker = yf.Ticker("BTC-USD")
        btc_hist = btc_ticker.history(period="1mo")  # 获取1个月的历史数据
        if not btc_hist.empty:
            # 计算技术指标（包含TAV分析）
            btc_indicators = analyzer.calculate_all_indicators(btc_hist.copy(), asset_type='crypto')
            
            # 生成买卖信号（如果启用TAV，使用TAV增强信号）
            if use_tav:
                btc_indicators_with_signals = analyzer.generate_buy_sell_signals(btc_indicators.copy(), use_tav=True, asset_type='crypto')
            else:
                btc_indicators_with_signals = analyzer.generate_buy_sell_signals(btc_indicators.copy())
            
            # 分析趋势
            btc_trend = analyzer.analyze_trend(btc_indicators_with_signals)
            
            # 获取TAV分析摘要（如果可用）
            btc_tav_summary = None
            if use_tav:
                btc_tav_summary = analyzer.get_tav_analysis_summary(btc_indicators_with_signals, 'crypto')
            
            # 获取最新的指标值
            latest_btc = btc_indicators_with_signals.iloc[-1]
            btc_rsi = latest_btc.get('RSI', 50.0)
            btc_macd = latest_btc.get('MACD', 0.0)
            btc_macd_signal = latest_btc.get('MACD_signal', 0.0)
            btc_bb_position = latest_btc.get('BB_position', 0.5) if 'BB_position' in latest_btc else 0.5
            
            # 获取TAV评分（如果可用）
            btc_tav_score = latest_btc.get('TAV_Score', 0) if use_tav else 0
            btc_tav_status = latest_btc.get('TAV_Status', '无TAV') if use_tav else '无TAV'
            
            # 检查最近的交易信号
            recent_signals = btc_indicators_with_signals.tail(5)
            buy_signals = []
            sell_signals = []
            
            if 'Buy_Signal' in recent_signals.columns:
                buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                for idx, row in buy_signals_df.iterrows():
                    # 从描述中提取买入信号部分
                    desc = row.get('Signal_Description', '')
                    if '买入信号:' in desc and '卖出信号:' in desc:
                        # 如果同时有买入和卖出信号，只提取买入部分
                        buy_part = desc.split('买入信号:')[1].split('卖出信号:')[0].strip()
                        # 移除可能的结尾分隔符
                        if buy_part.endswith('|'):
                            buy_part = buy_part[:-1].strip()
                        buy_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': f"买入信号: {buy_part}"
                        })
                    elif '买入信号:' in desc:
                        # 如果只有买入信号
                        buy_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': desc
                        })
            
            if 'Sell_Signal' in recent_signals.columns:
                sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                for idx, row in sell_signals_df.iterrows():
                    # 从描述中提取卖出信号部分
                    desc = row.get('Signal_Description', '')
                    if '买入信号:' in desc and '卖出信号:' in desc:
                        # 如果同时有买入和卖出信号，只提取卖出部分
                        sell_part = desc.split('卖出信号:')[1].strip()
                        sell_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': f"卖出信号: {sell_part}"
                        })
                    elif '卖出信号:' in desc:
                        # 如果只有卖出信号
                        sell_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': desc
                        })

            # 解析并解决同日冲突（如果有）
            final_buy_signals, final_sell_signals, signal_conflicts = resolve_conflicting_signals(
                buy_signals, sell_signals, tav_score=btc_tav_score if use_tav else None
            )
            
            indicators['bitcoin'] = {
                'rsi': btc_rsi,
                'macd': btc_macd,
                'macd_signal': btc_macd_signal,
                'price_position': calculate_price_position(latest_btc.get('Close', 0)),
                'bb_position': btc_bb_position,
                'trend': btc_trend,
                'recent_buy_signals': final_buy_signals,
                'recent_sell_signals': final_sell_signals,
                'signal_conflicts': signal_conflicts,
                'current_price': latest_btc.get('Close', 0),
                'ma20': latest_btc.get('MA20', 0),
                'ma50': latest_btc.get('MA50', 0),
                'tav_score': btc_tav_score,
                'tav_status': btc_tav_status,
                'tav_summary': btc_tav_summary,
            }
        else:
            # 如果无法获取历史数据，使用简化计算
            btc = prices.get('bitcoin', {})
            indicators['bitcoin'] = {
                'rsi': calculate_rsi(btc.get('usd_24h_change', 0)),
                'macd': calculate_macd(btc.get('usd', 0)),
                'price_position': calculate_price_position(btc.get('usd', 0)),
            }
    except Exception as e:
        print(f"⚠️ 获取比特币历史数据失败: {e}")
        # 如果获取历史数据失败，使用简化计算
        btc = prices.get('bitcoin', {})
        indicators['bitcoin'] = {
            'rsi': calculate_rsi(btc.get('usd_24h_change', 0)),
            'macd': calculate_macd(btc.get('usd', 0)),
            'price_position': calculate_price_position(btc.get('usd', 0)),
        }
    
    # 获取以太坊历史数据
    try:
        eth_ticker = yf.Ticker("ETH-USD")
        eth_hist = eth_ticker.history(period="1mo")  # 获取1个月的历史数据
        if not eth_hist.empty:
            # 计算技术指标（包含TAV分析）
            eth_indicators = analyzer.calculate_all_indicators(eth_hist.copy(), asset_type='crypto')
            
            # 生成买卖信号（如果启用TAV，使用TAV增强信号）
            if use_tav:
                eth_indicators_with_signals = analyzer.generate_buy_sell_signals(eth_indicators.copy(), use_tav=True, asset_type='crypto')
            else:
                eth_indicators_with_signals = analyzer.generate_buy_sell_signals(eth_indicators.copy())
            
            # 分析趋势
            eth_trend = analyzer.analyze_trend(eth_indicators_with_signals)
            
            # 获取TAV分析摘要（如果可用）
            eth_tav_summary = None
            if use_tav:
                eth_tav_summary = analyzer.get_tav_analysis_summary(eth_indicators_with_signals, 'crypto')
            
            # 获取最新的指标值
            latest_eth = eth_indicators_with_signals.iloc[-1]
            eth_rsi = latest_eth.get('RSI', 50.0)
            eth_macd = latest_eth.get('MACD', 0.0)
            eth_macd_signal = latest_eth.get('MACD_signal', 0.0)
            eth_bb_position = latest_eth.get('BB_position', 0.5) if 'BB_position' in latest_eth else 0.5
            
            # 获取TAV评分（如果可用）
            eth_tav_score = latest_eth.get('TAV_Score', 0) if use_tav else 0
            eth_tav_status = latest_eth.get('TAV_Status', '无TAV') if use_tav else '无TAV'
            
            # 检查最近的交易信号
            recent_signals = eth_indicators_with_signals.tail(5)
            buy_signals = []
            sell_signals = []
            
            if 'Buy_Signal' in recent_signals.columns:
                buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                for idx, row in buy_signals_df.iterrows():
                    # 从描述中提取买入信号部分
                    desc = row.get('Signal_Description', '')
                    if '买入信号:' in desc and '卖出信号:' in desc:
                        # 如果同时有买入和卖出信号，只提取买入部分
                        buy_part = desc.split('买入信号:')[1].split('卖出信号:')[0].strip()
                        # 移除可能的结尾分隔符
                        if buy_part.endswith('|'):
                            buy_part = buy_part[:-1].strip()
                        buy_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': f"买入信号: {buy_part}"
                        })
                    elif '买入信号:' in desc:
                        # 如果只有买入信号
                        buy_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': desc
                        })
            
            if 'Sell_Signal' in recent_signals.columns:
                sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                for idx, row in sell_signals_df.iterrows():
                    # 从描述中提取卖出信号部分
                    desc = row.get('Signal_Description', '')
                    if '买入信号:' in desc and '卖出信号:' in desc:
                        # 如果同时有买入和卖出信号，只提取卖出部分
                        sell_part = desc.split('卖出信号:')[1].strip()
                        sell_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': f"卖出信号: {sell_part}"
                        })
                    elif '卖出信号:' in desc:
                        # 如果只有卖出信号
                        sell_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': desc
                        })

            # 解析并解决同日冲突（如果有）
            final_buy_signals, final_sell_signals, signal_conflicts = resolve_conflicting_signals(
                buy_signals, sell_signals, tav_score=eth_tav_score if use_tav else None
            )
            
            indicators['ethereum'] = {
                'rsi': eth_rsi,
                'macd': eth_macd,
                'macd_signal': eth_macd_signal,
                'price_position': calculate_price_position(latest_eth.get('Close', 0)),
                'bb_position': eth_bb_position,
                'trend': eth_trend,
                'recent_buy_signals': final_buy_signals,
                'recent_sell_signals': final_sell_signals,
                'signal_conflicts': signal_conflicts,
                'current_price': latest_eth.get('Close', 0),
                'ma20': latest_eth.get('MA20', 0),
                'ma50': latest_eth.get('MA50', 0),
                'tav_score': eth_tav_score,
                'tav_status': eth_tav_status,
                'tav_summary': eth_tav_summary,
            }
        else:
            # 如果无法获取历史数据，使用简化计算
            eth = prices.get('ethereum', {})
            indicators['ethereum'] = {
                'rsi': calculate_rsi(eth.get('usd_24h_change', 0)),
                'macd': calculate_macd(eth.get('usd', 0)),
                'price_position': calculate_price_position(eth.get('usd', 0)),
            }
    except Exception as e:
        print(f"⚠️ 获取以太坊历史数据失败: {e}")
        # 如果获取历史数据失败，使用简化计算
        eth = prices.get('ethereum', {})
        indicators['ethereum'] = {
            'rsi': calculate_rsi(eth.get('usd_24h_change', 0)),
            'macd': calculate_macd(eth.get('usd', 0)),
            'price_position': calculate_price_position(eth.get('usd', 0)),
        }
    
    return indicators

def run_anomaly_detection(prices, use_deep_analysis=False, target_date=None):
    """
    Run anomaly detection on Ethereum data.
    
    Args:
        prices: Price data from CoinGecko API
        use_deep_analysis: If True, run full analysis (Z-Score + Isolation Forest).
                         If False, run only Z-Score detection (faster).
        target_date: Target date for anomaly detection (YYYY-MM-DD format), None for current date.
    
    Returns:
        Anomaly detection results dict or None if no anomalies
    """
    if not ANOMALY_DETECTION_AVAILABLE:
        print("⚠️ Anomaly detection not available")
        return None
    
    if 'ethereum' not in prices:
        return None
    
    # 确定目标日期
    if target_date:
        try:
            from datetime import timezone
            target_dt = datetime.strptime(target_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"⚠️ 无效的日期格式: {target_date}，使用今天")
            target_dt = datetime.now(timezone.utc)
    else:
        from datetime import timezone
        target_dt = datetime.now(timezone.utc)
    
    target_date_str = target_dt.strftime('%Y-%m-%d')
    print(f"🔍 检测 {target_date_str} 的异常")
    
    try:
        # Initialize components
        # Z-Score：小时级监控，3天窗口（72小时）
        zscore_detector = ZScoreDetector(window_size=72, threshold=3.0, time_interval='hour')
        
        # 只在深度分析模式下初始化 Isolation Forest
        if_detector = None
        feature_extractor = None
        
        if use_deep_analysis:
            # Isolation Forest：小时级监控，关注最近1天
            if_detector = IsolationForestDetector(
                contamination=0.05,
                random_state=42,
                anomaly_type='crypto_hourly'
            )
            feature_extractor = FeatureExtractor()
        
        cache = AnomalyCache()
        integrator = AnomalyIntegrator(cache)
        
        # Get historical data (小时级数据，1个月)
        eth_ticker = yf.Ticker("ETH-USD")
        eth_hist = eth_ticker.history(period="1mo", interval='1h')
        
        if eth_hist.empty:
            print("⚠️ No historical data available for anomaly detection")
            return None
        
        # Run Z-Score detection (Layer 1) - 始终执行
        zscore_anomalies = []
        current_price = prices['ethereum']['usd']
        
        # Price anomaly (小时级)
        price_anomaly = zscore_detector.detect_anomaly(
            metric_name='price',
            current_value=current_price,
            history=eth_hist['Close'],
            timestamp=target_dt,
            time_interval='hour'
        )
        if price_anomaly:
            zscore_anomalies.append(price_anomaly)
        
        # Volume anomaly (小时级)
        if 'usd_24hr_vol' in prices['ethereum']:
            current_volume = prices['ethereum']['usd_24hr_vol']
            volume_anomaly = zscore_detector.detect_anomaly(
                metric_name='volume',
                current_value=current_volume,
                history=eth_hist['Volume'],
                timestamp=target_dt,
                time_interval='hour'
            )
            if volume_anomaly:
                zscore_anomalies.append(volume_anomaly)
        
        # Run Isolation Forest detection (Layer 2) - 仅在深度分析模式下执行
        if_anomalies = []
        
        if use_deep_analysis and if_detector and feature_extractor:
            # Extract features
            features, timestamps = feature_extractor.extract_features(eth_hist)
            
            if not features.empty:
                # Train model
                if_detector.train(features)
                
                # Detect anomalies (指定日期，小时级)
                if_anomalies = if_detector.detect_anomalies_by_date(
                    features=features,
                    timestamps=timestamps,
                    target_date=target_dt,
                    time_interval='hour'
                )
        
        # Integrate results
        result = integrator.integrate(
            zscore_anomalies=zscore_anomalies,
            if_anomalies=if_anomalies,
            timestamp=target_dt
        )
        
        # 为每个异常添加异常日期和原因
        if result['has_anomaly']:
            for anomaly in result['anomalies']:
                # 获取异常发生的时间戳
                anomaly_timestamp = anomaly.get('timestamp', target_dt)
                anomaly_date_str = anomaly_timestamp.strftime('%Y-%m-%d') if hasattr(anomaly_timestamp, 'strftime') else target_date_str
                
                # 添加异常日期和原因
                anomaly['anomaly_date'] = anomaly_date_str
                anomaly['anomaly_reason'] = _analyze_anomaly_reason(anomaly)
                
                # 计算异常日期的技术指标
                # 修复：使用 .date() 方法替代 .normalize()，确保时区兼容性
                target_date_only = anomaly_timestamp.date() if hasattr(anomaly_timestamp, 'date') else pd.to_datetime(anomaly_date_str).date()
                df_anomaly = eth_hist[eth_hist.index.date == target_date_only]
                
                if not df_anomaly.empty:
                    df_anomaly_row = df_anomaly.iloc[0]
                    anomaly['indicators'] = _calculate_anomaly_indicators(eth_hist, df_anomaly_row)
        
        # Cleanup old cache entries
        cache.cleanup_expired(max_age_hours=48)
        
        return result
    
    except Exception as e:
        print(f"⚠️ Anomaly detection failed: {e}")
        return None


def format_anomaly_results(anomaly_result):
    """
    Format anomaly detection results for email.
    
    Args:
        anomaly_result: Anomaly detection result dict
    
    Returns:
        Formatted HTML string
    """
    if not anomaly_result or not anomaly_result['has_anomaly']:
        return None
    
    severity_emoji = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢'
    }
    
    html = """
        <div class="section">
            <h3>🚨 异常检测报告</h3>
    """
    
    # Add severity
    severity = anomaly_result['severity']
    html += f"<p><strong>严重程度:</strong> {severity_emoji.get(severity, '')} {severity.upper()}</p>"
    
    # Add anomalies table with detailed columns
    html += "<table><tr><th>类型</th><th>异常日期</th><th>异常原因</th><th>异常评分/值</th><th>当前价格</th><th>1日变化</th><th>5日变化</th><th>RSI</th><th>布林带</th><th>MACD</th></tr>"
    
    for anomaly in anomaly_result['anomalies']:
        anomaly_type = anomaly['type']
        anomaly_date = anomaly.get('anomaly_date', 'N/A')
        anomaly_reason = anomaly.get('anomaly_reason', '未知')
        indicators = anomaly.get('indicators', {})
        
        # Format score/value
        if anomaly_type == 'isolation_forest':
            score = anomaly.get('anomaly_score', 0)
            score_value = f"{score:.3f}"
        else:
            z_score = anomaly.get('z_score', 0)
            value = anomaly.get('value', 0)
            score_value = f"Z: {z_score:.2f}, 值: {value:.2f}"
        
        # Get indicator values
        current_price = indicators.get('current_price', 0)
        change_1d = indicators.get('change_1d', 0)
        change_5d = indicators.get('change_5d', 0)
        rsi = indicators.get('rsi', 0)
        bb_position = indicators.get('bollinger_position', 'middle')
        macd_signal = indicators.get('macd_signal', 'neutral')
        
        # Format position and signal
        bb_position_cn = get_bollinger_position_cn(bb_position)
        macd_signal_cn = get_macd_signal_cn(macd_signal)
        
        # Format change icons
        change_1d_icon = "📈" if change_1d > 0 else "📉"
        change_5d_icon = "📈" if change_5d > 0 else "📉"
        
        html += f"""
            <tr>
                <td>{anomaly_type}</td>
                <td>{anomaly_date}</td>
                <td>{anomaly_reason}</td>
                <td>{score_value}</td>
                <td>{current_price:.2f}</td>
                <td>{change_1d_icon} {change_1d:+.2f}%</td>
                <td>{change_5d_icon} {change_5d:+.2f}%</td>
                <td>{rsi:.1f}</td>
                <td>{bb_position_cn}</td>
                <td>{macd_signal_cn}</td>
            </tr>
        """
    
    html += "</table></div>"
    
    return html


def _analyze_anomaly_reason(anomaly):
    """
    分析异常原因

    Args:
        anomaly: 异常字典

    Returns:
        异常原因描述
    """
    anomaly_type = anomaly.get('type', 'price')
    features = anomaly.get('features', {})

    # 处理'crypto'类型异常（来自Isolation Forest）
    if anomaly_type == 'crypto' or anomaly_type == 'isolation_forest':
        if not features:
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
        return '价格异常波动（收盘价显著偏离历史均值）'
    elif anomaly_type == 'volume':
        return '成交量异常（成交量显著偏离历史均值）'
    elif anomaly_type == 'isolation_forest':
        # Isolation Forest 异常：分析哪个特征异常
        features = anomaly.get('features', {})
        if not features:
            return '多维特征异常（综合指标异常）'

        # 特征名称映射
        feature_names = {
            'price_change_1d': '1日涨跌幅',
            'price_change_5d': '5日涨跌幅',
            'volume_ratio': '成交量比率',
            'rsi': 'RSI指标',
            'macd': 'MACD指标',
            'bb_position': '布林带位置',
            'volatility_5d': '5日波动率',
            'volume_spike': '成交量激增',
            'price_gap': '价格缺口',
            'trend_strength': '趋势强度'
        }

        # 找出异常值最大的特征
        abnormal_features = []
        for feature_name, feature_value in features.items():
            # 如果特征值偏离中位数较多（假设标准化后的特征）
            if abs(feature_value) > 2:
                feature_cn = feature_names.get(feature_name, feature_name)
                abnormal_features.append(feature_cn)

        if abnormal_features:
            return f'多维特征异常（{", ".join(abnormal_features[:3])}异常）'
        else:
            return '多维特征异常（综合指标异常，需关注）'

    return '异常（原因未知）'


def _calculate_anomaly_indicators(eth_hist, df_anomaly_row):
    """
    计算异常日期的技术指标
    
    Args:
        eth_hist: 完整的历史数据
        df_anomaly_row: 异常日期的数据行
    
    Returns:
        技术指标字典
    """
    try:
        # 检查数据量
        if len(eth_hist) < 20:
            print("⚠️ 数据量不足（少于20天），计算指标可能不准确")
        
        # 使用技术分析工具计算指标
        if TECHNICAL_ANALYSIS_AVAILABLE:
            analyzer = TechnicalAnalyzer()
            indicators_df = analyzer.calculate_all_indicators(eth_hist.copy())
            
            # 获取异常日期的指标
            anomaly_date = df_anomaly_row.name
            if anomaly_date in indicators_df.index:
                latest = indicators_df.loc[anomaly_date]
            else:
                # 如果找不到精确匹配，使用最接近的日期
                idx = indicators_df.index.get_indexer([anomaly_date], method='nearest')[0]
                latest = indicators_df.iloc[idx]
            
            rsi = latest.get('RSI', 50.0)
            # 如果RSI是nan或inf，使用默认值50
            if pd.isna(rsi) or not np.isfinite(rsi):
                print(f"⚠️ RSI计算无效（{rsi}），使用默认值50")
                rsi = 50.0
            
            return {
                'rsi': rsi,
                'bollinger_position': _get_bollinger_position(latest),
                'macd_signal': _get_macd_signal(latest),
                'current_price': df_anomaly_row['Close'],
                'change_1d': 0,
                'change_5d': 0
            }
        else:
            # 简化计算
            return {
                'rsi': 50.0,
                'bollinger_position': 'middle',
                'macd_signal': 'neutral',
                'current_price': df_anomaly_row['Close'],
                'change_1d': 0,
                'change_5d': 0
            }
    except Exception as e:
        print(f"⚠️ 计算技术指标失败: {e}")
        return {
            'rsi': 50.0,
            'bollinger_position': 'middle',
            'macd_signal': 'neutral',
            'current_price': df_anomaly_row['Close'],
            'change_1d': 0,
            'change_5d': 0
        }


def _get_bollinger_position(latest_row):
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


def _get_macd_signal(latest_row):
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


def calculate_rsi(change_pct):
    """
    简化RSI计算（基于24小时变化率）
    """
    # 这是一个非常简化的计算，实际RSI需要14天的价格数据
    if change_pct > 0:
        return min(100, 50 + change_pct * 2)  # 简单映射
    else:
        return max(0, 50 + change_pct * 2)

def calculate_macd(price):
    """
    简化MACD计算（基于价格）
    """
    # 这是一个非常简化的计算，实际MACD需要历史价格数据
    return price * 0.01  # 简单映射

def calculate_price_position(price):
    """
    简化价格位置计算（假设价格在近期高低点之间）
    """
    # 这是一个非常简化的计算，实际需要历史价格数据
    return 50.0  # 假设在中位


# Helper functions for Bollinger Bands and MACD
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


def _get_bollinger_position(latest_row):
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


def _get_macd_signal(latest_row):
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


# --- 新增：信号冲突解析辅助函数 ---
def resolve_conflicting_signals(buy_signals, sell_signals, tav_score=None, buy_threshold=55, sell_threshold=45):
    """
    输入：
      buy_signals, sell_signals: 列表，每项形如 {'date': 'YYYY-MM-DD', 'description': '...'}
      tav_score: 可选的数值评分（0-100），用于解冲决策
      buy_threshold / sell_threshold: 用于基于 tav_score 的自动决策阈值

    返回：
      resolved_buy, resolved_sell, conflicts
      resolved_buy/resolved_sell: 列表，包含被最终判定为买/卖的信号，
        每项形如 {'date':..., 'description':..., 'reason':...}
      conflicts: 列表，包含当天同时有买卖但无法自动判定的条目（保留原始描述，便于人工查看）
    """
    # 按日期汇总
    by_date = {}
    for s in buy_signals:
        date = s.get('date')
        by_date.setdefault(date, {'buy': [], 'sell': []})
        by_date[date]['buy'].append(s.get('description'))
    for s in sell_signals:
        date = s.get('date')
        by_date.setdefault(date, {'buy': [], 'sell': []})
        by_date[date]['sell'].append(s.get('description'))

    resolved_buy = []
    resolved_sell = []
    conflicts = []

    for date, parts in sorted(by_date.items()):
        buys = parts.get('buy', [])
        sells = parts.get('sell', [])

        # 只有买或只有卖 —— 直接保留
        if buys and not sells:
            combined_desc = " | ".join(buys)
            resolved_buy.append({'date': date, 'description': combined_desc, 'reason': 'only_buy'})
            continue
        if sells and not buys:
            combined_desc = " | ".join(sells)
            resolved_sell.append({'date': date, 'description': combined_desc, 'reason': 'only_sell'})
            continue

        # 同一天同时存在买与卖 —— 尝试用 tav_score 自动解冲
        if buys and sells:
            if tav_score is not None:
                # 简单策略：高于 buy_threshold -> 选 buy；低于 sell_threshold -> 选 sell；否则冲突
                if tav_score >= buy_threshold and tav_score > sell_threshold:
                    combined_desc = "Buy: " + " | ".join(buys) + " ; Sell: " + " | ".join(sells)
                    resolved_buy.append({'date': date, 'description': combined_desc, 'reason': f'tav_decision({tav_score})'})
                elif tav_score <= sell_threshold and tav_score < buy_threshold:
                    combined_desc = "Sell: " + " | ".join(sells) + " ; Buy: " + " | ".join(buys)
                    resolved_sell.append({'date': date, 'description': combined_desc, 'reason': f'tav_decision({tav_score})'})
                else:
                    # tav_score 在不确定区间 -> 标记冲突
                    combined_desc = "同时包含买和卖信号。Buy: " + " | ".join(buys) + " ; Sell: " + " | ".join(sells)
                    conflicts.append({'date': date, 'description': combined_desc, 'tav_score': tav_score})
            else:
                # 没有 tav_score，无法自动判定 -> 标记冲突
                combined_desc = "同时包含买和卖信号。Buy: " + " | ".join(buys) + " ; Sell: " + " | ".join(sells)
                conflicts.append({'date': date, 'description': combined_desc, 'tav_score': None})

    return resolved_buy, resolved_sell, conflicts
# --- 新增结束 ---


def send_email(to, subject, text, html):
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.mail.yahoo.com")
    smtp_user = os.environ.get("EMAIL_SENDER")
    smtp_pass = os.environ.get("EMAIL_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("Error: Missing EMAIL_SENDER or EMAIL_PASSWORD in environment variables.")
        return False

    # 如果to是字符串，转换为列表
    if isinstance(to, str):
        to = [to]

    msg = MIMEMultipart("alternative")
    msg['From'] = f'<{sender_email}>'
    msg['To'] = ", ".join(to)  # 将收件人列表转换为逗号分隔的字符串
    msg['Subject'] = subject

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    # 根据SMTP服务器类型选择合适的端口和连接方式
    if "163.com" in smtp_server:
        # 163邮箱使用SSL连接，端口465
        smtp_port = 465
        use_ssl = True
    elif "gmail.com" in smtp_server:
        # Gmail使用TLS连接，端口587
        smtp_port = 587
        use_ssl = False
    else:
        # 默认使用TLS连接，端口587
        smtp_port = 587
        use_ssl = False

    # 发送邮件（增加重试机制）
    for attempt in range(3):
        try:
            if use_ssl:
                # 使用SSL连接
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                server.login(smtp_user, smtp_pass)
                server.sendmail(sender_email, to, msg.as_string())
                server.quit()
            else:
                # 使用TLS连接
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(sender_email, to, msg.as_string())
                server.quit()
            
            print("✅ Email sent successfully!")
            return True
        except Exception as e:
            print(f"❌ Error sending email (attempt {attempt+1}/3): {e}")
            if attempt < 2:  # 不是最后一次尝试，等待后重试
                import time
                time.sleep(5)
    
    print("❌ Failed to send email after 3 attempts")
    return False

# === 主逻辑 ===
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='加密货币价格监控和异常检测')
    parser.add_argument('--mode', type=str, choices=['quick', 'deep'], default='quick',
                       help='异常检测模式: quick (Z-Score only) 或 deep (Z-Score + Isolation Forest)')
    parser.add_argument('--date', help='指定检测日期（YYYY-MM-DD格式），默认检测当天异常')
    args = parser.parse_args()
    
    # 从参数获取异常检测模式
    # quick: 只执行 Z-Score 检测（快速）
    # deep: 执行完整分析（Z-Score + Isolation Forest）
    use_deep_analysis = (args.mode == 'deep')
    target_date = args.date
    
    if use_deep_analysis:
        print("🔍 深度异常检测模式（Z-Score + Isolation Forest）")
    else:
        print("⚡ 快速异常检测模式（Z-Score only）")
    
    if target_date:
        print(f"📅 检测日期: {target_date}")
    else:
        print("📅 检测日期: 今天")
    
    # 可以通过修改这里的参数来控制是否包含市值和24小时交易量
    prices = get_cryptocurrency_prices(include_market_cap=True, include_24hr_vol=True)

    if prices is None:
        print("Failed to fetch prices. Exiting.")
        exit(1)

    # 计算技术指标
    indicators = calculate_technical_indicators(prices)

    # 运行异常检测
    anomaly_result = None
    if ANOMALY_DETECTION_AVAILABLE:
        anomaly_result = run_anomaly_detection(prices, use_deep_analysis=use_deep_analysis, target_date=target_date)

    # 检查是否存在当天的交易信号
    has_signals = False
    today = datetime.now().date()
    
    if 'ethereum' in indicators:
        eth_recent_buy_signals = indicators['ethereum'].get('recent_buy_signals', [])
        eth_recent_sell_signals = indicators['ethereum'].get('recent_sell_signals', [])
        eth_conflicts = indicators['ethereum'].get('signal_conflicts', [])
        
        # 检查以太坊是否有今天的信号
        for signal in eth_recent_buy_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for signal in eth_recent_sell_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for c in eth_conflicts:
            if datetime.strptime(c['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
    
    if 'bitcoin' in indicators and not has_signals:
        btc_recent_buy_signals = indicators['bitcoin'].get('recent_buy_signals', [])
        btc_recent_sell_signals = indicators['bitcoin'].get('recent_sell_signals', [])
        btc_conflicts = indicators['bitcoin'].get('signal_conflicts', [])
        
        # 检查比特币是否有今天的信号
        for signal in btc_recent_buy_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for signal in btc_recent_sell_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for c in btc_conflicts:
            if datetime.strptime(c['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break

    # 检查是否有重要内容需要发送（交易信号或异常检测）
    has_important_content = has_signals  # 有交易信号
    
    # 检查异常检测结果
    if anomaly_result and anomaly_result['has_anomaly']:
        severity = anomaly_result.get('severity', 'low')
        # 只发送 high 或 medium 级别的异常
        if severity in ['high', 'medium']:
            has_important_content = True
    
    # 如果没有重要内容，则不发送邮件
    if not has_important_content:
        print("⚠️ 没有检测到交易信号或重要异常，跳过发送邮件。")
        exit(0)

    # 根据内容类型调整邮件标题
    if has_signals and anomaly_result and anomaly_result['has_anomaly']:
        subject = "加密货币更新 - 交易信号 + 异常检测"
    elif has_signals:
        subject = "加密货币更新 - 交易信号提醒"
    elif anomaly_result and anomaly_result['has_anomaly']:
        severity = anomaly_result.get('severity', 'low')
        severity_emoji = '🔴' if severity == 'high' else '🟡' if severity == 'medium' else '🟢'
        subject = f"加密货币更新 - 异常检测提醒 [{severity_emoji} {severity.upper()}]"
    else:
        subject = "加密货币更新 - 价格更新"

    text = ""
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h2 {{ color: #333; }}
            h3 {{ color: #555; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin: 20px 0; }}
            .highlight {{ background-color: #ffffcc; }}
            .buy-signal {{ background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .sell-signal {{ background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .conflict-signal {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h2>💰 加密货币价格更新 - 交易信号提醒</h2>
        <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    # 使用表格展示加密货币价格概览
    html += """
        <div class="section">
            <h3>💰 加密货币价格概览</h3>
            <table>
                <tr>
                    <th>资产名称</th>
                    <th>最新价格 (USD)</th>
                    <th>最新价格 (HKD)</th>
                    <th>24小时变化</th>
                    <th>市值 (USD)</th>
                    <th>24小时交易量 (USD)</th>
                </tr>
    """
    
    # Ethereum
    if 'ethereum' in prices:
        eth = prices['ethereum']
        eth_usd = eth['usd']
        eth_hkd = eth['hkd']
        eth_change = eth.get('usd_24h_change', 0.0)
        eth_market_cap = eth.get('usd_market_cap', 0.0) if 'usd_market_cap' in eth else 0.0
        eth_24hr_vol = eth.get('usd_24h_vol', 0.0) if 'usd_24h_vol' in eth else 0.0
        
        html += f"""
                <tr>
                    <td>Ethereum (ETH)</td>
                    <td>${eth_usd:,.2f}</td>
                    <td>${eth_hkd:,.2f}</td>
                    <td>{eth_change:+.2f}%</td>
                    <td>${eth_market_cap:,.2f}</td>
                    <td>${eth_24hr_vol:,.2f}</td>
                </tr>
        """
        
        text += f"Ethereum (ETH):\n"
        text += f"  价格: ${eth_usd:,.2f} USD ({eth_change:+.2f}% 24h)\n"
        text += f"  价格: ${eth_hkd:,.2f} HKD\n"
        if eth_market_cap > 0:
            text += f"  市值: ${eth_market_cap:,.2f} USD\n"
        if eth_24hr_vol > 0:
            text += f"  24小时交易量: ${eth_24hr_vol:,.2f} USD\n"
    
    # Bitcoin
    if 'bitcoin' in prices:
        btc = prices['bitcoin']
        btc_usd = btc['usd']
        btc_hkd = btc['hkd']
        btc_change = btc.get('usd_24h_change', 0.0)
        btc_market_cap = btc.get('usd_market_cap', 0.0) if 'usd_market_cap' in btc else 0.0
        btc_24hr_vol = btc.get('usd_24h_vol', 0.0) if 'usd_24h_vol' in btc else 0.0
        
        html += f"""
                <tr>
                    <td>Bitcoin (BTC)</td>
                    <td>${btc_usd:,.2f}</td>
                    <td>${btc_hkd:,.2f}</td>
                    <td>{btc_change:+.2f}%</td>
                    <td>${btc_market_cap:,.2f}</td>
                    <td>${btc_24hr_vol:,.2f}</td>
                </tr>
        """
        
        text += f"Bitcoin (BTC):\n"
        text += f"  价格: ${btc_usd:,.2f} USD ({btc_change:+.2f}% 24h)\n"
        text += f"  价格: ${btc_hkd:,.2f} HKD\n"
        if btc_market_cap > 0:
            text += f"  市值: ${btc_market_cap:,.2f} USD\n"
        if btc_24hr_vol > 0:
            text += f"  24小时交易量: ${btc_24hr_vol:,.2f} USD\n"
    
    html += """
            </table>
        </div>
    """

    # 使用表格展示技术分析
    html += """
        <div class=\"section\">
            <h3>🔬 技术分析</h3>
            <table>
                <tr>
                    <th>资产名称</th>
                    <th>趋势</th>
                    <th>RSI (14日)</th>
                    <th>MACD</th>
                    <th>MACD信号线</th>
                    <th>布林带位置</th>
                    <th>TAV评分</th>
                    <th>MA20</th>
                    <th>MA50</th>
                </tr>
    """
    
    # Ethereum 技术分析
    if 'ethereum' in prices and 'ethereum' in indicators:
        eth_rsi = indicators['ethereum'].get('rsi', 0.0)
        eth_macd = indicators['ethereum'].get('macd', 0.0)
        eth_macd_signal = indicators['ethereum'].get('macd_signal', 0.0)
        eth_bb_position = indicators['ethereum'].get('bb_position', 0.5)
        eth_trend = indicators['ethereum'].get('trend', '未知')
        eth_ma20 = indicators['ethereum'].get('ma20', 0)
        eth_ma50 = indicators['ethereum'].get('ma50', 0)
        eth_recent_buy_signals = indicators['ethereum'].get('recent_buy_signals', [])
        eth_recent_sell_signals = indicators['ethereum'].get('recent_sell_signals', [])
        eth_conflicts = indicators['ethereum'].get('signal_conflicts', [])
        eth_tav_score = indicators['ethereum'].get('tav_score', 0)
        eth_tav_status = indicators['ethereum'].get('tav_status', '无TAV')
        
        html += f"""
                <tr>
                    <td>Ethereum (ETH)</td>
                    <td>{eth_trend}</td>
                    <td>{eth_rsi:.2f}</td>
                    <td>{eth_macd:.4f}</td>
                    <td>{eth_macd_signal:.4f}</td>
                    <td>{eth_bb_position:.2f}</td>
                    <td>{eth_tav_score:.1f} ({eth_tav_status})</td>
                    <td>${eth_ma20:.2f}</td>
                    <td>${eth_ma50:.2f}</td>
                </tr>
            """
        
        # 添加交易信号到表格中
        if eth_recent_buy_signals:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"buy-signal\">
                            <strong>🔔 Ethereum (ETH) 最近买入信号:</strong><br>
            """
            for signal in eth_recent_buy_signals:
                reason = signal.get('reason', '')
                is_today = datetime.strptime(signal['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: green;'>• {bold_start}{signal['date']}: {signal['description']}{bold_end}"
                if reason:
                    html += f" （{reason}）"
                html += "</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        if eth_recent_sell_signals:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"sell-signal\">
                            <strong>🔻 Ethereum (ETH) 最近卖出信号:</strong><br>
            """
            for signal in eth_recent_sell_signals:
                reason = signal.get('reason', '')
                is_today = datetime.strptime(signal['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: red;'>• {bold_start}{signal['date']}: {signal['description']}{bold_end}"
                if reason:
                    html += f" （{reason}）"
                html += "</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        # 添加冲突区块
        if eth_conflicts:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"conflict-signal\">
                            <strong>⚠️ Ethereum (ETH) 信号冲突（需要人工确认）:</strong><br>
            """
            for c in eth_conflicts:
                tav_info = f" TAV={c.get('tav_score')}" if c.get('tav_score') is not None else ""
                is_today = datetime.strptime(c['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: #856404;'>• {bold_start}{c['date']}: {c['description']}{tav_info}{bold_end}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """

        text += f"  RSI: {eth_rsi:.2f}\n"
        text += f"  MACD: {eth_macd:.4f} (信号线: {eth_macd_signal:.4f})\n"
        text += f"  布林带位置: {eth_bb_position:.2f}\n"
        text += f"  趋势: {eth_trend}\n"
        text += f"  MA20: ${eth_ma20:.2f}\n"
        text += f"  MA50: ${eth_ma50:.2f}\n"
        
        # 添加TAV信息
        if TAV_AVAILABLE and indicators['ethereum'].get('tav_score') is not None:
            eth_tav_score = indicators['ethereum'].get('tav_score', 0)
            eth_tav_status = indicators['ethereum'].get('tav_status', '无TAV')
            text += f"  TAV评分: {eth_tav_score:.1f} ({eth_tav_status})\n"
            
            # 添加TAV详细分析
            eth_tav_summary = indicators['ethereum'].get('tav_summary')
            if eth_tav_summary:
                text += f"  TAV趋势分析: {eth_tav_summary.get('trend_analysis', 'N/A')}\n"
                text += f"  TAV动量分析: {eth_tav_summary.get('momentum_analysis', 'N/A')}\n"
                text += f"  TAV成交量分析: {eth_tav_summary.get('volume_analysis', 'N/A')}\n"
                text += f"  TAV建议: {eth_tav_summary.get('recommendation', 'N/A')}\n"
        
        # 添加交易信号信息到文本版本
        if eth_recent_buy_signals:
            text += f"  🔔 最近买入信号 ({len(eth_recent_buy_signals)} 个):\n"
            for signal in eth_recent_buy_signals:
                reason = signal.get('reason', '')
                text += f"    {signal['date']}: {signal['description']}"
                if reason:
                    text += f" （{reason}）"
                text += "\n"
        
        if eth_recent_sell_signals:
            text += f"  🔻 最近卖出信号 ({len(eth_recent_sell_signals)} 个):\n"
            for signal in eth_recent_sell_signals:
                reason = signal.get('reason', '')
                text += f"    {signal['date']}: {signal['description']}"
                if reason:
                    text += f" （{reason}）"
                text += "\n"

        if eth_conflicts:
            text += f"  ⚠️ 信号冲突 ({len(eth_conflicts)} 个)，需要人工确认：\n"
            for c in eth_conflicts:
                tav_info = f" TAV={c.get('tav_score')}" if c.get('tav_score') is not None else ""
                text += f"    {c['date']}: {c['description']}{tav_info}\n"
    
    # Bitcoin 技术分析
    if 'bitcoin' in prices and 'bitcoin' in indicators:
        btc_rsi = indicators['bitcoin'].get('rsi', 0.0)
        btc_macd = indicators['bitcoin'].get('macd', 0.0)
        btc_macd_signal = indicators['bitcoin'].get('macd_signal', 0.0)
        btc_bb_position = indicators['bitcoin'].get('bb_position', 0.5)
        btc_trend = indicators['bitcoin'].get('trend', '未知')
        btc_ma20 = indicators['bitcoin'].get('ma20', 0)
        btc_ma50 = indicators['bitcoin'].get('ma50', 0)
        btc_tav_score = indicators['bitcoin'].get('tav_score', 0)
        btc_tav_status = indicators['bitcoin'].get('tav_status', '无TAV')
        btc_recent_buy_signals = indicators['bitcoin'].get('recent_buy_signals', [])
        btc_recent_sell_signals = indicators['bitcoin'].get('recent_sell_signals', [])
        btc_conflicts = indicators['bitcoin'].get('signal_conflicts', [])
        
        html += f"""
                <tr>
                    <td>Bitcoin (BTC)</td>
                    <td>{btc_trend}</td>
                    <td>{btc_rsi:.2f}</td>
                    <td>{btc_macd:.4f}</td>
                    <td>{btc_macd_signal:.4f}</td>
                    <td>{btc_bb_position:.2f}</td>
                    <td>{btc_tav_score:.1f} ({btc_tav_status})</td>
                    <td>${btc_ma20:.2f}</td>
                    <td>${btc_ma50:.2f}</td>
                </tr>
        """
        
        # 添加交易信号到表格中
        if btc_recent_buy_signals:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"buy-signal\">
                            <strong>🔔 Bitcoin (BTC) 最近买入信号:</strong><br>
            """
            for signal in btc_recent_buy_signals:
                reason = signal.get('reason', '')
                is_today = datetime.strptime(signal['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: green;'>• {bold_start}{signal['date']}: {signal['description']}{bold_end}"
                if reason:
                    html += f" （{reason}）"
                html += "</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        if btc_recent_sell_signals:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"sell-signal\">
                            <strong>🔻 Bitcoin (BTC) 最近卖出信号:</strong><br>
            """
            for signal in btc_recent_sell_signals:
                reason = signal.get('reason', '')
                is_today = datetime.strptime(signal['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: red;'>• {bold_start}{signal['date']}: {signal['description']}{bold_end}"
                if reason:
                    html += f" （{reason}）"
                html += "</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        # 添加冲突区块
        if btc_conflicts:
            html += f"""
                <tr>
                    <td colspan=\"9\">
                        <div class=\"conflict-signal\">
                            <strong>⚠️ Bitcoin (BTC) 信号冲突（需要人工确认）:</strong><br>
            """
            for c in btc_conflicts:
                tav_info = f" TAV={c.get('tav_score')}" if c.get('tav_score') is not None else ""
                is_today = datetime.strptime(c['date'], '%Y-%m-%d').date() == today
                bold_start = "<strong>" if is_today else ""
                bold_end = "</strong>" if is_today else ""
                html += f"<span style='color: #856404;'>• {bold_start}{c['date']}: {c['description']}{tav_info}{bold_end}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """

        text += f"  RSI: {btc_rsi:.2f}\n"
        text += f"  MACD: {btc_macd:.4f} (信号线: {btc_macd_signal:.4f})\n"
        text += f"  布林带位置: {btc_bb_position:.2f}\n"
        text += f"  趋势: {btc_trend}\n"
        text += f"  MA20: ${btc_ma20:.2f}\n"
        text += f"  MA50: ${btc_ma50:.2f}\n"
        
        # 添加TAV信息
        if TAV_AVAILABLE and indicators['bitcoin'].get('tav_score') is not None:
            btc_tav_score = indicators['bitcoin'].get('tav_score', 0)
            btc_tav_status = indicators['bitcoin'].get('tav_status', '无TAV')
            text += f"  TAV评分: {btc_tav_score:.1f} ({btc_tav_status})\n"
            
            # 添加TAV详细分析
            btc_tav_summary = indicators['bitcoin'].get('tav_summary')
            if btc_tav_summary:
                text += f"  TAV趋势分析: {btc_tav_summary.get('trend_analysis', 'N/A')}\n"
                text += f"  TAV动量分析: {btc_tav_summary.get('momentum_analysis', 'N/A')}\n"
                text += f"  TAV成交量分析: {btc_tav_summary.get('volume_analysis', 'N/A')}\n"
                text += f"  TAV建议: {btc_tav_summary.get('recommendation', 'N/A')}\n"
        
        # 添加交易信号信息到文本版本
        if btc_recent_buy_signals:
            text += f"  🔔 最近买入信号 ({len(btc_recent_buy_signals)} 个):\n"
            for signal in btc_recent_buy_signals:
                reason = signal.get('reason', '')
                text += f"    {signal['date']}: {signal['description']}"
                if reason:
                    text += f" （{reason}）"
                text += "\n"
        
        if btc_recent_sell_signals:
            text += f"  🔻 最近卖出信号 ({len(btc_recent_sell_signals)} 个):\n"
            for signal in btc_recent_sell_signals:
                reason = signal.get('reason', '')
                text += f"    {signal['date']}: {signal['description']}"
                if reason:
                    text += f" （{reason}）"
                text += "\n"

        if btc_conflicts:
            text += f"  ⚠️ 信号冲突 ({len(btc_conflicts)} 个)，需要人工确认：\n"
            for c in btc_conflicts:
                tav_info = f" TAV={c.get('tav_score')}" if c.get('tav_score') is not None else ""
                text += f"    {c['date']}: {c['description']}{tav_info}\n"
    
    # 添加异常检测结果到文本版本
    if anomaly_result and anomaly_result['has_anomaly']:
        text += "\n\n🚨 异常检测报告\n"
        text += f"严重程度: {anomaly_result['severity'].upper()}\n"
        text += f"检测到异常数量: {len(anomaly_result['anomalies'])}\n"
        
        for anomaly in anomaly_result['anomalies']:
            anomaly_type = anomaly['type']
            timestamp = anomaly['timestamp']
            anomaly_severity = anomaly['severity']
            
            if anomaly_type == 'isolation_forest':
                score = anomaly.get('anomaly_score', 0)
                text += f"\n- 类型: Isolation Forest"
                text += f"\n  时间: {timestamp.strftime('%Y-%m-%d %H:%M')}"
                text += f"\n  严重程度: {anomaly_severity}"
                text += f"\n  异常评分: {score:.3f}"
            else:
                z_score = anomaly.get('z_score', 0)
                value = anomaly.get('value', 0)
                text += f"\n- 类型: {anomaly_type}"
                text += f"\n  时间: {timestamp.strftime('%Y-%m-%d %H:%M')}"
                text += f"\n  严重程度: {anomaly_severity}"
                text += f"\n  Z-Score: {z_score:.2f}"
                text += f"\n  值: {value:.2f}"
    
    html += """
            </table>
        </div>
    """

    # 添加异常检测结果（如果有）
    if anomaly_result and anomaly_result['has_anomaly']:
        anomaly_html = format_anomaly_results(anomaly_result)
        if anomaly_html:
            html += anomaly_html

    # 添加指标说明到文本版本
    text += "\n📋 指标说明:\n"
    text += "价格(USD/HKD)：加密货币的当前价格，分别以美元和港币计价。\n"
    text += "24小时变化(%)：过去24小时内价格的变化百分比。\n"
    text += "市值(Market Cap)：加密货币的总市值，以美元计价。\n"
    text += "24小时交易量：过去24小时内该加密货币的交易总额，以美元计价。\n"
    text += "RSI(相对强弱指数)：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。\n"
    text += "MACD(异同移动平均线)：判断价格趋势和动能的技术指标。\n"
    text += "MA20(20日移动平均线)：过去20个交易日的平均价格，反映短期趋势。\n"
    text += "MA50(50日移动平均线)：过去50个交易日的平均价格，反映中期趋势。\n"
    text += "布林带位置：当前价格在布林带中的相对位置，范围0-1。接近0表示价格接近下轨（可能超卖），接近1表示价格接近上轨（可能超买）。\n"
    text += "价格位置(%)：当前价格在近期价格区间的相对位置。\n"
    text += "趋势：市场当前的整体方向。\n"
    text += "TAV评分(趋势-动量-成交量综合评分)：基于趋势(Trend)、动量(Momentum)、成交量(Volume)三个维度的综合评分系统，范围0-100分：\n"
    text += "  - 计算方式：TAV评分 = 趋势评分 × 30% + 动量评分 × 45% + 成交量评分 × 25%（加密货币权重配置）\n"
    text += "  - 趋势评分(30%权重)：基于10日、30日、100日移动平均线的排列和价格位置计算，评估长期、中期、短期趋势的一致性\n"
    text += "  - 动量评分(45%权重)：结合RSI(14日)和MACD(12,26,9)指标，评估价格变化的动能强度和方向\n"
    text += "  - 成交量评分(25%权重)：基于20日成交量均线，分析成交量突增(>1.3倍为弱、>1.8倍为中、>2.5倍为强)或萎缩(<0.7倍)情况\n"
    text += "  - 评分等级：\n"
    text += "    * ≥80分：强共振 - 三个维度高度一致，强烈信号\n"
    text += "    * 55-79分：中等共振 - 多数维度一致，中等信号\n"
    text += "    * 30-54分：弱共振 - 部分维度一致，弱信号\n"
    text += "    * <30分：无共振 - 各维度分歧，无明确信号\n"

    text += "  强势多头：价格强劲上涨趋势，各周期均线呈多头排列（价格 > MA20 > MA50 > MA200）\n"
    text += "  多头趋势：价格上涨趋势，中期均线呈多头排列（价格 > MA20 > MA50）\n"
    text += "  弱势空头：价格持续下跌趋势，各周期均线呈空头排列（价格 < MA20 < MA50 < MA200）\n"
    text += "  空头趋势：价格下跌趋势，中期均线呈空头排列（价格 < MA20 < MA50）\n"
    text += "  震荡整理：价格在一定区间内波动，无明显趋势\n"
    text += "  短期上涨/下跌：基于最近价格变化的短期趋势判断\n"
    text += "\n"
    
    # 添加指标说明
    html += """
    <div class="section">
        <h3>📋 指标说明</h3>
        <div style="font-size:0.9em; line-height:1.4;">
        <ul>
          <li><b>价格(USD/HKD)</b>：加密货币的当前价格，分别以美元和港币计价。</li>
          <li><b>24小时变化(%)</b>：过去24小时内价格的变化百分比。</li>
          <li><b>市值(Market Cap)</b>：加密货币的总市值，以美元计价。</li>
          <li><b>24小时交易量</b>：过去24小时内该加密货币的交易总额，以美元计价。</li>
          <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
          <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
          <li><b>MA20(20日移动平均线)</b>：过去20个交易日的平均价格，反映短期趋势。</li>
          <li><b>MA50(50日移动平均线)</b>：过去50个交易日的平均价格，反映中期趋势。</li>
          <li><b>布林带位置</b>：当前价格在布林带中的相对位置，范围0-1。接近0表示价格接近下轨（可能超卖），接近1表示价格接近上轨（可能超买）。</li>
          <li><b>价格位置(%)</b>：当前价格在近期价格区间的相对位置。</li>
          <li><b>趋势</b>：市场当前的整体方向。
            <ul>
              <li><b>强势多头</b>：价格强劲上涨趋势，各周期均线呈多头排列（价格 > MA20 > MA50 > MA200）</li>
              <li><b>多头趋势</b>：价格上涨趋势，中期均线呈多头排列（价格 > MA20 > MA50）</li>
              <li><b>弱势空头</b>：价格持续下跌趋势，各周期均线呈空头排列（价格 < MA20 < MA50 < MA200）</li>
              <li><b>空头趋势</b>：价格下跌趋势，中期均线呈空头排列（价格 < MA20 < MA50）</li>
              <li><b>震荡整理</b>：价格在一定区间内波动，无明显趋势</li>
              <li><b>短期上涨/下跌</b>：基于最近价格变化的短期趋势判断</li>
            </ul>
          </li>
          <li><b>TAV评分(趋势-动量-成交量综合评分)</b>：基于趋势(Trend)、动量(Momentum)、成交量(Volume)三个维度的综合评分系统，范围0-100分：
            <ul>
              <li><b>计算方式</b>：TAV评分 = 趋势评分 × 30% + 动量评分 × 45% + 成交量评分 × 25%（加密货币权重配置）</li>
              <li><b>趋势评分(30%权重)</b>：基于10日、30日、100日移动平均线的排列和价格位置计算，评估长期、中期、短期趋势的一致性</li>
              <li><b>动量评分(45%权重)</b>：结合RSI(14日)和MACD(12,26,9)指标，评估价格变化的动能强度和方向</li>
              <li><b>成交量评分(25%权重)</b>：基于20日成交量均线，分析成交量突增(>1.3倍为弱、>1.8倍为中、>2.5倍为强)或萎缩(<0.7倍)情况</li>
              <li><b>评分等级</b>：
                <ul>
                  <li>≥80分：强共振 - 三个维度高度一致，强烈信号</li>
                  <li>55-79分：中等共振 - 多数维度一致，中等信号</li>
                  <li>30-54分：弱共振 - 部分维度一致，弱信号</li>
                  <li><30分：无共振 - 各维度分歧，无明确信号</li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
        </div>
    </div>
    """

    html += "</body></html>"

    # 获取收件人（默认 fallback）
    recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
    
    # 如果环境变量中有多个收件人（用逗号分隔），则拆分为列表
    if ',' in recipient_env:
        recipients = [recipient.strip() for recipient in recipient_env.split(',')]
    else:
        recipients = [recipient_env]

    print("🔔 检测到交易信号，发送邮件到:", ", ".join(recipients))
    print("📝 Subject:", subject)
    print("📄 Text preview:\n", text)

    success = send_email(recipients, subject, text, html)
    if not success:
        exit(1)
