#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数及港股主力资金追踪器股票价格监控和交易信号邮件通知系统
基于技术分析指标生成买卖信号，只在有交易信号时发送邮件
"""

import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# 导入技术分析工具
try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用简化指标计算")

# 从港股主力资金追踪器导入股票列表
try:
    from hk_smart_money_tracker import WATCHLIST
    STOCK_LIST = WATCHLIST
except ImportError:
    print("⚠️ 无法导入 hk_smart_money_tracker.WATCHLIST，使用默认股票列表")
    # 默认使用一些常见的港股股票
    STOCK_LIST = {
    "2800.HK": "盈富基金",
    "3968.HK": "招商银行",
    "0939.HK": "建设银行",
    "1398.HK": "工商银行",
    "1288.HK": "农业银行",
    "0005.HK": "汇丰银行",
    "0728.HK": "中国电信",
    "0941.HK": "中国移动",
    "6682.HK": "第四范式",
    "1347.HK": "华虹半导体",
    "1138.HK": "中远海能",
    "1088.HK": "中国神华",
    "0883.HK": "中国海洋石油",
    "0981.HK": "中芯国际",
    "0388.HK": "香港交易所",
    "0700.HK": "腾讯控股",
    "9988.HK": "阿里巴巴-SW",
    "3690.HK": "美团-W",
    "1810.HK": "小米集团-W",
    "9660.HK": "地平线机器人",
    "2533.HK": "黑芝麻智能",
    "1330.HK": "绿色动力环保",
    "1211.HK": "比亚迪股份",
    "2269.HK": "药明生物",
    "1299.HK": "友邦保险"
    }

def get_hsi_data():
    """获取恒生指数数据"""
    try:
        # 使用yfinance获取恒生指数数据
        hsi_ticker = yf.Ticker("^HSI")
        hist = hsi_ticker.history(period="6mo")  # 获取6个月的历史数据
        if hist.empty:
            print("❌ 无法获取恒生指数历史数据")
            return None
        
        # 获取最新数据
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        hsi_data = {
            'current_price': latest['Close'],
            'change_1d': (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0,
            'change_1d_points': latest['Close'] - prev['Close'],
            'open': latest['Open'],
            'high': latest['High'],
            'low': latest['Low'],
            'volume': latest['Volume'],
            'hist': hist
        }
        
        return hsi_data
    except Exception as e:
        print(f"❌ 获取恒生指数数据失败: {e}")
        return None

def get_stock_data(symbol):
    """获取指定股票的数据"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")  # 获取6个月的历史数据
        if hist.empty:
            print(f"❌ 无法获取 {symbol} 的历史数据")
            return None
        
        # 获取最新数据
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        stock_data = {
            'symbol': symbol,
            'name': STOCK_LIST.get(symbol, symbol),  # 使用导入的股票列表获取股票名称
            'current_price': latest['Close'],
            'change_1d': (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0,
            'change_1d_points': latest['Close'] - prev['Close'],
            'open': latest['Open'],
            'high': latest['High'],
            'low': latest['Low'],
            'volume': latest['Volume'],
            'hist': hist
        }
        
        return stock_data
    except Exception as e:
        print(f"❌ 获取 {symbol} 数据失败: {e}")
        return None

def calculate_technical_indicators(data):
    """
    计算技术指标（适用于恒生指数或个股）
    """
    if data is None:
        return None
    
    hist = data['hist']
    
    if not TECHNICAL_ANALYSIS_AVAILABLE:
        # 如果技术分析工具不可用，使用简化指标计算
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # 简化的技术指标计算
        indicators = {
            'rsi': calculate_rsi((latest['Close'] - prev['Close']) / prev['Close'] * 100),
            'macd': calculate_macd(latest['Close']),
            'price_position': calculate_price_position(latest['Close'], hist['Close'].min(), hist['Close'].max()),
        }
        
        return indicators
    
    # 使用技术分析工具计算更准确的指标
    analyzer = TechnicalAnalyzer()
    
    try:
        # 计算技术指标
        indicators = analyzer.calculate_all_indicators(hist.copy())
        
        # 生成买卖信号
        indicators_with_signals = analyzer.generate_buy_sell_signals(indicators.copy())
        
        # 分析趋势
        trend = analyzer.analyze_trend(indicators_with_signals)
        
        # 获取最新的指标值
        latest = indicators_with_signals.iloc[-1]
        rsi = latest.get('RSI', 50.0)
        macd = latest.get('MACD', 0.0)
        macd_signal = latest.get('MACD_signal', 0.0)
        bb_position = latest.get('BB_position', 0.5) if 'BB_position' in latest else 0.5
        
        # 检查最近的交易信号
        recent_signals = indicators_with_signals.tail(5)
        buy_signals = []
        sell_signals = []
        
        if 'Buy_Signal' in recent_signals.columns:
            buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
            for idx, row in buy_signals_df.iterrows():
                description = row['Signal_Description']
                # 如果描述中已经有"买入信号"字样，去除它，因为我们会在显示时添加
                if description.startswith('买入信号:'):
                    description = description[5:].strip()  # 去掉"买入信号:"和可能的空格
                elif description.startswith('买入信号'):
                    description = description[4:].strip()  # 去掉"买入信号"和可能的冒号和空格
                elif description.startswith('Buy Signal:'):
                    description = description[11:].strip()
                elif description.startswith('Buy Signal'):
                    description = description[10:].strip()
                buy_signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'description': description
                })
        
        if 'Sell_Signal' in recent_signals.columns:
            sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
            for idx, row in sell_signals_df.iterrows():
                description = row['Signal_Description']
                # 如果描述中已经有"卖出信号"字样，去除它，因为我们会在显示时添加
                if description.startswith('卖出信号:'):
                    description = description[5:].strip()  # 去掉"卖出信号:"和可能的空格
                elif description.startswith('卖出信号'):
                    description = description[4:].strip()  # 去掉"卖出信号"和可能的冒号和空格
                elif description.startswith('Sell Signal:'):
                    description = description[11:].strip()
                elif description.startswith('Sell Signal'):
                    description = description[10:].strip()
                sell_signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'description': description
                })
        
        return {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'price_position': calculate_price_position(latest.get('Close', 0), hist['Close'].min(), hist['Close'].max()),
            'bb_position': bb_position,
            'trend': trend,
            'recent_buy_signals': buy_signals,
            'recent_sell_signals': sell_signals,
            'current_price': latest.get('Close', 0),
            'ma20': latest.get('MA20', 0),
            'ma50': latest.get('MA50', 0),
            'ma200': latest.get('MA200', 0),
            'hist': hist
        }
    except Exception as e:
        print(f"⚠️ 计算技术指标失败: {e}")
        # 如果计算失败，使用简化计算
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        return {
            'rsi': calculate_rsi((latest['Close'] - prev['Close']) / prev['Close'] * 100),
            'macd': calculate_macd(latest['Close']),
            'price_position': calculate_price_position(latest['Close'], hist['Close'].min(), hist['Close'].max()),
        }

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

def calculate_price_position(current_price, min_price, max_price):
    """
    计算价格位置（在近期高低点之间的百分位）
    """
    if max_price == min_price:
        return 50.0
    
    return (current_price - min_price) / (max_price - min_price) * 100

def detect_continuous_signals_in_history_from_transactions(stock_code, hours=48, min_signals=3):
    """
    基于交易历史记录检测连续买卖信号
    - stock_code: 股票代码
    - hours: 检测的时间范围（小时）
    - min_signals: 判定为连续信号的最小信号数量
    返回: 连续信号状态（如"连续买入(3次)"、"买入2次,卖出1次"等）
    """
    try:
        import csv
        from collections import defaultdict
        
        # 读取交易记录文件
        if not os.path.exists('data/simulation_transactions.csv'):
            return "无交易记录"
        
        with open('data/simulation_transactions.csv', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 解析CSV内容
        lines = content.strip().split('\n')
        headers = lines[0].split(',')
        transactions = []
        
        for line in lines[1:]:
            fields = line.split(',')
            # 处理可能包含逗号的字段
            if len(fields) > len(headers):
                reconstructed = []
                i = 0
                while i < len(fields):
                    if fields[i].startswith('"') and not fields[i].endswith('"'):
                        j = i
                        while j < len(fields) and not fields[j].endswith('"'):
                            j += 1
                        reconstructed.append(','.join(fields[i:j+1]).strip('"'))
                        i = j + 1
                    else:
                        reconstructed.append(fields[i].strip('"'))
                        i += 1
                fields = reconstructed
            
            if len(fields) >= 10:  # 确保有足够的字段
                timestamp_str = fields[0]
                trans_type = fields[1]
                code = fields[2]
                name = fields[3] if len(fields) > 3 else ""
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    transactions.append({
                        'timestamp': timestamp,
                        'type': trans_type,
                        'code': code,
                        'name': name
                    })
                except ValueError as e:
                    print(f"解析时间戳失败: {timestamp_str}, 错误: {e}")
                    continue
    
        # 过滤指定时间范围内的交易
        now = datetime.now()
        time_threshold = now - timedelta(hours=hours)
        recent_transactions = [t for t in transactions if t['timestamp'] >= time_threshold and t['code'] == stock_code]
        
        # 按股票代码分组交易
        transactions_by_stock = defaultdict(lambda: {'BUY': [], 'SELL': []})
        for trans in recent_transactions:
            if trans['type'] in transactions_by_stock[trans['code']]:
                transactions_by_stock[trans['code']][trans['type']].append(trans)
        
        # 获取指定股票的交易
        trans_dict = transactions_by_stock[stock_code]
        buys = sorted(trans_dict['BUY'], key=lambda x: x['timestamp'])
        sells = sorted(trans_dict['SELL'], key=lambda x: x['timestamp'])
        
        buy_count = len(buys)
        sell_count = len(sells)
        
        # 根据买卖次数返回不同的状态
        if buy_count >= min_signals and sell_count == 0 and buy_count > 0:
            return f"连续买入({buy_count}次)"
        elif sell_count >= min_signals and buy_count == 0 and sell_count > 0:
            return f"连续卖出({sell_count}次)"
        elif buy_count > 0 and sell_count == 0:
            return f"买入({buy_count}次)"
        elif sell_count > 0 and buy_count == 0:
            return f"卖出({sell_count}次)"
        elif buy_count > 0 and sell_count > 0:
            return f"买入{buy_count}次,卖出{sell_count}次"
        else:
            return "无信号"
    
    except Exception as e:
        print(f"⚠️ 检测连续信号失败: {e}")
        return "检测失败"

def detect_continuous_signals_in_history(indicators_df, hours=48, min_signals=3):
    """
    检测历史数据中的连续买卖信号（基于交易记录）
    - indicators_df: 包含历史信号数据的DataFrame
    - hours: 检测的时间范围（小时）
    - min_signals: 判定为连续信号的最小信号数量
    返回: 连续信号状态（如"连续买入"、"连续卖出"、"无连续信号"）
    """
    # 这里应该检测基于交易记录的连续信号，而不是技术指标
    # 由于我们无法从indicators_df获取股票代码，需要另外处理
    return "无交易记录"  # 作为默认返回值，实际调用时会使用新的函数

def analyze_continuous_signals():
    """
    分析最近48小时内的连续买卖信号
    返回: 有连续买入信号的股票列表、有连续卖出信号的股票列表
    """
    import csv
    from collections import defaultdict
    
    # 读取交易记录文件
    if not os.path.exists('data/simulation_transactions.csv'):
        return [], []
    
    with open('data/simulation_transactions.csv', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 解析CSV内容
    lines = content.strip().split('\n')
    headers = lines[0].split(',')
    transactions = []
    
    for line in lines[1:]:
        fields = line.split(',')
        # 处理可能包含逗号的字段
        if len(fields) > len(headers):
            reconstructed = []
            i = 0
            while i < len(fields):
                if fields[i].startswith('"') and not fields[i].endswith('"'):
                    j = i
                    while j < len(fields) and not fields[j].endswith('"'):
                        j += 1
                    reconstructed.append(','.join(fields[i:j+1]).strip('"'))
                    i = j + 1
                else:
                    reconstructed.append(fields[i].strip('"'))
                    i += 1
            fields = reconstructed
        
        if len(fields) >= 10:  # 确保有足够的字段
            timestamp_str = fields[0]
            trans_type = fields[1]
            code = fields[2]
            name = fields[3] if len(fields) > 3 else ""
            shares_str = fields[4] if len(fields) > 4 else "0"
            price_str = fields[5] if len(fields) > 5 else "0"
            amount_str = fields[6] if len(fields) > 6 else "0"
            reason = fields[8] if len(fields) > 8 else ""  # reason is at index 8
            stop_loss_price = fields[10] if len(fields) > 10 else ""  # stop_loss_price is at index 10 (after success field at index 9)
            current_price = fields[11] if len(fields) > 11 else ""  # current_price is at index 11
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Format reason with stop_loss_price and current_price if they exist
                formatted_reason = reason
                has_additional_info = False
                
                # 检查止损价是否有效（不是空字符串、None、False、'None'、'nan'、'False'等）
                if stop_loss_price and stop_loss_price not in ["", "None", "nan", "False", "null"] and stop_loss_price is not None and stop_loss_price != "False":
                    try:
                        # 尝试将stop_loss_price转换为浮点数以检查是否有效
                        float_stop_loss = float(stop_loss_price)
                        if stop_loss_price and stop_loss_price != 'False' and not (float_stop_loss != float_stop_loss):  # 检查是否为NaN
                            formatted_reason += f", 止损价: {stop_loss_price}"
                            has_additional_info = True
                    except (ValueError, TypeError):
                        pass  # 如果无法转换为浮点数，则跳过
                
                # 检查现价是否有效（不是空字符串、None、False、'None'、'nan'、'False'等）
                if current_price and current_price not in ["", "None", "nan", "False", "null"] and current_price is not None and current_price != "False":
                    try:
                        # 尝试将current_price转换为浮点数以检查是否有效
                        float_current = float(current_price)
                        if current_price and current_price != 'False' and not (float_current != float_current):  # 检查是否为NaN
                            formatted_reason += f", 现价: {current_price}"
                            has_additional_info = True
                    except (ValueError, TypeError):
                        pass  # 如果无法转换为浮点数，则跳过
                
                # 如果添加了额外信息，确保正确的格式
                if has_additional_info and formatted_reason.startswith(", "):
                    formatted_reason = formatted_reason[2:]
                
                transactions.append({
                    'timestamp': timestamp,
                    'date': timestamp.date(),
                    'type': trans_type,
                    'code': code,
                    'name': name,
                    'shares': int(float(shares_str)),
                    'price': float(price_str),
                    'amount': float(amount_str),
                    'reason': formatted_reason.strip()
                })
            except ValueError as e:
                print(f"Error parsing line: {line[:100]}... Error: {e}")
    
    # 过滤最近48小时的交易
    now = datetime.now()
    time_48_hours_ago = now - timedelta(hours=48)
    recent_transactions = [t for t in transactions if t['timestamp'] >= time_48_hours_ago]
    
    # 按股票代码分组交易
    transactions_by_stock = defaultdict(lambda: {'BUY': [], 'SELL': []})
    for trans in recent_transactions:
        transactions_by_stock[trans['code']][trans['type']].append(trans)
    
    # 查找有3次或以上连续买入信号且无卖出信号的股票
    buy_without_sell_after = []
    sell_without_buy_after = []
    
    for stock_code, trans_dict in transactions_by_stock.items():
        buys = sorted(trans_dict['BUY'], key=lambda x: x['timestamp'])
        sells = sorted(trans_dict['SELL'], key=lambda x: x['timestamp'])
        
        # 检查是否有3次或以上买入且无卖出
        if len(buys) >= 3 and len(sells) == 0:
            stock_name = buys[0]['name'] if buys else 'Unknown'
            buy_times = [buy['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for buy in buys]
            buy_reasons = [buy['reason'] for buy in buys]
            buy_without_sell_after.append((stock_code, stock_name, buy_times, buy_reasons))
        elif len(sells) >= 3 and len(buys) == 0:
            stock_name = sells[0]['name'] if sells else 'Unknown'
            sell_times = [sell['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for sell in sells]
            sell_reasons = [sell['reason'] for sell in sells]
            sell_without_buy_after.append((stock_code, stock_name, sell_times, sell_reasons))
    
    return buy_without_sell_after, sell_without_buy_after

def has_any_signals(hsi_indicators, stock_results):
    """检查是否有任何股票有当天的交易信号"""
    today = datetime.now().date()
    
    # 检查恒生指数信号
    if hsi_indicators:
        recent_buy_signals = hsi_indicators.get('recent_buy_signals', [])
        recent_sell_signals = hsi_indicators.get('recent_sell_signals', [])
        
        for signal in recent_buy_signals:
            signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
            if signal_date == today:
                return True
        for signal in recent_sell_signals:
            signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
            if signal_date == today:
                return True
    
    # 检查持仓股票信号
    for stock_result in stock_results:
        indicators = stock_result.get('indicators')
        if indicators:
            recent_buy_signals = indicators.get('recent_buy_signals', [])
            recent_sell_signals = indicators.get('recent_sell_signals', [])
            
            for signal in recent_buy_signals:
                signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
                if signal_date == today:
                    return True
            for signal in recent_sell_signals:
                signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
                if signal_date == today:
                    return True
    
    return False

def generate_stock_analysis_html(stock_data, indicators):
    """为单只股票生成HTML分析部分"""
    if not indicators:
        return ""
    
    html = f"""
    <div class="section">
        <h3>📊 {stock_data['name']} ({stock_data['symbol']}) 分析</h3>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
            </tr>
    """
    
    html += f"""
            <tr>
                <td>当前价格</td>
                <td>{stock_data['current_price']:,.2f}</td>
            </tr>
            <tr>
                <td>24小时变化</td>
                <td>{stock_data['change_1d']:+.2f}% ({stock_data['change_1d_points']:+.2f})</td>
            </tr>
            <tr>
                <td>当日开盘</td>
                <td>{stock_data['open']:,.2f}</td>
            </tr>
            <tr>
                <td>当日最高</td>
                <td>{stock_data['high']:,.2f}</td>
            </tr>
            <tr>
                <td>当日最低</td>
                <td>{stock_data['low']:,.2f}</td>
            </tr>
            <tr>
                <td>成交量</td>
                <td>{stock_data['volume']:,.0f}</td>
            </tr>
    """
    
    # 添加技术指标
    rsi = indicators.get('rsi', 0.0)
    macd = indicators.get('macd', 0.0)
    macd_signal = indicators.get('macd_signal', 0.0)
    bb_position = indicators.get('bb_position', 0.5)
    trend = indicators.get('trend', '未知')
    ma20 = indicators.get('ma20', 0)
    ma50 = indicators.get('ma50', 0)
    ma200 = indicators.get('ma200', 0)
    
    html += f"""
            <tr>
                <td>趋势</td>
                <td>{trend}</td>
            </tr>
            <tr>
                <td>RSI (14日)</td>
                <td>{rsi:.2f}</td>
            </tr>
            <tr>
                <td>MACD</td>
                <td>{macd:.4f}</td>
            </tr>
            <tr>
                <td>MACD信号线</td>
                <td>{macd_signal:.4f}</td>
            </tr>
            <tr>
                <td>布林带位置</td>
                <td>{bb_position:.2f}</td>
            </tr>
            <tr>
                <td>MA20</td>
                <td>{ma20:,.2f}</td>
            </tr>
            <tr>
                <td>MA50</td>
                <td>{ma50:,.2f}</td>
            </tr>
            <tr>
                <td>MA200</td>
                <td>{ma200:,.2f}</td>
            </tr>
    """
    
    # 添加交易信号
    recent_buy_signals = indicators.get('recent_buy_signals', [])
    recent_sell_signals = indicators.get('recent_sell_signals', [])
    
    if recent_buy_signals:
        html += f"""
            <tr>
                <td colspan="2">
                    <div class="buy-signal">
                        <strong>🔔 最近买入信号:</strong><br>
        """
        for signal in recent_buy_signals:
            html += f"<span style='color: green;'>• {signal['date']}: {signal['description']}</span><br>"
        html += """
                    </div>
                </td>
            </tr>
        """
    
    if recent_sell_signals:
        html += f"""
            <tr>
                <td colspan="2">
                    <div class="sell-signal">
                        <strong>🔻 最近卖出信号:</strong><br>
        """
        for signal in recent_sell_signals:
            html += f"<span style='color: red;'>• {signal['date']}: {signal['description']}</span><br>"
        html += """
                    </div>
                </td>
            </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    return html

def send_email(to, subject, text, html):
    smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("❌ 缺少YAHOO_EMAIL或YAHOO_APP_PASSWORD环境变量")
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
            
            print("✅ 邮件发送成功!")
            return True
        except Exception as e:
            print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
            if attempt < 2:  # 不是最后一次尝试，等待后重试
                import time
                time.sleep(5)
    
    print("❌ 3次尝试后仍无法发送邮件")
    return False

# === 主逻辑 ===
if __name__ == "__main__":
    print("🔍 正在获取恒生指数数据...")
    
    # 获取恒生指数数据和指标
    hsi_data = get_hsi_data()
    if hsi_data is None:
        print("❌ 无法获取恒生指数数据")
        hsi_indicators = None
    else:
        print("📊 正在计算恒生指数技术指标...")
        hsi_indicators = calculate_technical_indicators(hsi_data)

    # 获取WATCHLIST中的股票并进行分析
    print(f"🔍 正在获取股票列表并分析 ({len(STOCK_LIST)} 只股票)...")
    stock_results = []
    
    for stock_code, stock_name in STOCK_LIST.items():
        print(f"🔍 正在分析 {stock_name} ({stock_code}) ...")
        stock_data = get_stock_data(stock_code)
        if stock_data:
            print(f"📊 正在计算 {stock_name} ({stock_code}) 技术指标...")
            indicators = calculate_technical_indicators(stock_data)
            stock_results.append({
                'code': stock_code,
                'name': stock_name,
                'data': stock_data,
                'indicators': indicators
            })

    # 检查是否有任何股票有交易信号
    if not has_any_signals(hsi_indicators, stock_results):
        print("⚠️ 没有检测到任何交易信号，跳过发送邮件。")
        exit(0)

    subject = "恒生指数及港股主力资金追踪器股票交易信号提醒"

    # 创建信号汇总
    all_signals = []  # 合并买入和卖出信号
    
    # 恒生指数信号
    if hsi_indicators:
        recent_buy_signals = hsi_indicators.get('recent_buy_signals', [])
        recent_sell_signals = hsi_indicators.get('recent_sell_signals', [])
        for signal in recent_buy_signals:
            all_signals.append(('恒生指数', 'HSI', signal, '买入'))
        for signal in recent_sell_signals:
            all_signals.append(('恒生指数', 'HSI', signal, '卖出'))
    
    # 创建股票趋势映射
    stock_trends = {}
    for stock_result in stock_results:
        indicators = stock_result['indicators']
        if indicators:
            trend = indicators.get('trend', '未知')
            stock_trends[stock_result['code']] = trend
    
    # 股票信号
    for stock_result in stock_results:
        indicators = stock_result['indicators']
        if indicators:
            recent_buy_signals = indicators.get('recent_buy_signals', [])
            recent_sell_signals = indicators.get('recent_sell_signals', [])
            for signal in recent_buy_signals:
                all_signals.append((stock_result['name'], stock_result['code'], signal, '买入'))
            for signal in recent_sell_signals:
                all_signals.append((stock_result['name'], stock_result['code'], signal, '卖出'))
    
    # 只保留当天的信号
    today = datetime.now().date()
    today_signals = []
    for stock_name, stock_code, signal, signal_type in all_signals:
        signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
        if signal_date == today:
            # 获取该股票的趋势
            trend = stock_trends.get(stock_code, '未知')
            today_signals.append((stock_name, stock_code, trend, signal, signal_type))
    
    # 按股票名称排序
    today_signals.sort(key=lambda x: x[0])  # 按股票名称排序

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
        </style>
    </head>
    <body>
        <h2>📈 恒生指数及港股主力资金追踪器股票交易信号提醒</h2>
        <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    # 交易信号总结
    html += """
        <div class="section">
            <h3>🔔 交易信号总结</h3>
            <table>
                <tr>
                    <th>股票名称</th>
                    <th>股票代码</th>
                    <th>趋势</th>
                    <th>信号类型</th>
                    <th>信号描述</th>
                    <th>48小时智能建议</th>
                </tr>
    """

    # 添加所有信号（买入和卖出已合并并排序，只显示当天的）
    for stock_name, stock_code, trend, signal, signal_type in today_signals:
        signal_display = f"{signal_type}信号"
        color_style = "color: green; font-weight: bold;" if signal_type == '买入' else "color: red; font-weight: bold;"
        
        # 获取连续信号状态
        continuous_signal_status = "无信号"
        if stock_code != 'HSI':  # 恒生指数不适用连续信号检测
            # 使用基于交易记录的连续信号检测
            continuous_signal_status = detect_continuous_signals_in_history_from_transactions(stock_code)
        
        html += f"""
                <tr>
                    <td>{stock_name}</td>
                    <td>{stock_code}</td>
                    <td>{trend}</td>
                    <td><span style=\"{color_style}\">{signal_display}</span></td>
                    <td>{signal['description']}</td>
                    <td>{continuous_signal_status}</td>
                </tr>
        """

    if not today_signals:
        html += """
                <tr>
                    <td colspan="5">当前没有检测到任何交易信号</td>
                </tr>
        """

    html += """
            </table>
        </div>
    """

    # 在文本版本中添加信号总结（只显示当天的信号）
    text += "🔔 交易信号总结:\n"
    if today_signals:
        text += f"  {'股票名称':<15} {'股票代码':<10} {'趋势':<10} {'信号类型':<6} {'信号描述':<30} {'48小时内人工智能买卖建议':<18}\n"
        for stock_name, stock_code, trend, signal, signal_type in today_signals:
            # 获取连续信号状态
            continuous_signal_status = "无信号"
            if stock_code != 'HSI':  # 恒生指数不适用连续信号检测
                # 使用基于交易记录的连续信号检测
                continuous_signal_status = detect_continuous_signals_in_history_from_transactions(stock_code)
            text += f"  {stock_name:<15} {stock_code:<10} {trend:<10} {signal_type:<6} {signal['description']:<30} {continuous_signal_status:<18}\n"
    else:
        text += "当前没有检测到任何交易信号\n"
    
    text += "\n"

    # 分析最近48小时内的连续信号
    print("🔍 正在分析最近48小时内的连续交易信号...")
    buy_without_sell_after, sell_without_buy_after = analyze_continuous_signals()

    # 检查是否存在符合条件的连续信号
    has_continuous_signals = len(buy_without_sell_after) > 0 or len(sell_without_buy_after) > 0

    # 连续信号分析 - HTML
    if has_continuous_signals:
        html += """
        <div class="section">
            <h3>🔔 48小时连续交易信号分析</h3>
        """
        
        # 连续买入信号
        if buy_without_sell_after:
            html += """
            <div class="section">
                <h3>📈 最近48小时内连续3次或以上建议买入同一只股票（期间没有卖出建议）</h3>
                <table>
                    <tr>
                        <th>股票代码</th>
                        <th>股票名称</th>
                        <th>建议次数</th>
                        <th>建议时间及理由</th>
                    </tr>
            """
            
            for code, name, times, reasons in buy_without_sell_after:
                # 合并时间和原因
                combined_str = ""
                for i in range(len(times)):
                    time_reason = f"{times[i]}: {reasons[i] if reasons[i] else '无具体理由'}"
                    if i < len(times) - 1:
                        combined_str += time_reason + "<br>"
                    else:
                        combined_str += time_reason
                html += f"""
                <tr>
                    <td>{code}</td>
                    <td>{name}</td>
                    <td>{len(times)}次</td>
                    <td>{combined_str}</td>
                </tr>
                """
            
            html += """
                </table>
            </div>
            """

        # 连续卖出信号
        if sell_without_buy_after:
            html += """
            <div class="section">
                <h3>📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）</h3>
                <table>
                    <tr>
                        <th>股票代码</th>
                        <th>股票名称</th>
                        <th>建议次数</th>
                        <th>建议时间及理由</th>
                    </tr>
            """
            
            for code, name, times, reasons in sell_without_buy_after:
                # 合并时间和原因
                combined_str = ""
                for i in range(len(times)):
                    time_reason = f"{times[i]}: {reasons[i] if reasons[i] else '无具体理由'}"
                    if i < len(times) - 1:
                        combined_str += time_reason + "<br>"
                    else:
                        combined_str += time_reason
                html += f"""
                <tr>
                    <td>{code}</td>
                    <td>{name}</td>
                    <td>{len(times)}次</td>
                    <td>{combined_str}</td>
                </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </div>
        """

    # 连续信号分析 - 文本
    if buy_without_sell_after:
        text += f"📈 最近48小时内连续3次或以上建议买入同一只股票（期间没有卖出建议）:\n"
        for code, name, times, reasons in buy_without_sell_after:
            # 合并时间和原因
            combined_list = []
            for i in range(len(times)):
                time_reason = f"{times[i]}: {reasons[i] if reasons[i] else '无具体理由'}"
                combined_list.append(time_reason)
            combined_str = "\n    ".join(combined_list)
            text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
        text += "\n"
    
    if sell_without_buy_after:
        text += f"📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）:\n"
        for code, name, times, reasons in sell_without_buy_after:
            # 合并时间和原因
            combined_list = []
            for i in range(len(times)):
                time_reason = f"{times[i]}: {reasons[i] if reasons[i] else '无具体理由'}"
                combined_list.append(time_reason)
            combined_str = "\n    ".join(combined_list)
            text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
        text += "\n"

    # 添加说明
    if has_continuous_signals:
        text += "📋 说明:\n"
        text += "连续买入：指在最近48小时内，某只股票收到3次或以上买入建议，且期间没有收到任何卖出建议。\n"
        text += "连续卖出：指在最近48小时内，某只股票收到3次或以上卖出建议，且期间没有收到任何买入建议。\n\n"
        
        html += """
        <div class="section">
            <h3>📋 说明</h3>
            <div style="font-size:0.9em; line-height:1.4;">
            <ul>
              <li><b>连续买入</b>：指在最近48小时内，某只股票收到3次或以上买入建议，且期间没有收到任何卖出建议。</li>
              <li><b>连续卖出</b>：指在最近48小时内，某只股票收到3次或以上卖出建议，且期间没有收到任何买入建议。</li>
            </ul>
            </div>
        </div>
        """

    # 恒生指数价格概览（如果数据可用）
    if hsi_data:
        html += """
            <div class="section">
                <h3>📈 恒生指数价格概览</h3>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                    </tr>
        """
        
        html += f"""
                <tr>
                    <td>当前指数</td>
                    <td>{hsi_data['current_price']:,.2f}</td>
                </tr>
                <tr>
                    <td>24小时变化</td>
                    <td>{hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)</td>
                </tr>
                <tr>
                    <td>当日开盘</td>
                    <td>{hsi_data['open']:,.2f}</td>
                </tr>
                <tr>
                    <td>当日最高</td>
                    <td>{hsi_data['high']:,.2f}</td>
                </tr>
                <tr>
                    <td>当日最低</td>
                    <td>{hsi_data['low']:,.2f}</td>
                </tr>
                <tr>
                    <td>成交量</td>
                    <td>{hsi_data['volume']:,.0f}</td>
                </tr>
        """
        
        if hsi_indicators:
            rsi = hsi_indicators.get('rsi', 0.0)
            macd = hsi_indicators.get('macd', 0.0)
            macd_signal = hsi_indicators.get('macd_signal', 0.0)
            bb_position = hsi_indicators.get('bb_position', 0.5)
            trend = hsi_indicators.get('trend', '未知')
            ma20 = hsi_indicators.get('ma20', 0)
            ma50 = hsi_indicators.get('ma50', 0)
            ma200 = hsi_indicators.get('ma200', 0)
            
            html += f"""
                <tr>
                    <td>趋势</td>
                    <td>{trend}</td>
                </tr>
                <tr>
                    <td>RSI (14日)</td>
                    <td>{rsi:.2f}</td>
                </tr>
                <tr>
                    <td>MACD</td>
                    <td>{macd:.4f}</td>
                </tr>
                <tr>
                    <td>MACD信号线</td>
                    <td>{macd_signal:.4f}</td>
                </tr>
                <tr>
                    <td>布林带位置</td>
                    <td>{bb_position:.2f}</td>
                </tr>
                <tr>
                    <td>MA20</td>
                    <td>{ma20:,.2f}</td>
                </tr>
                <tr>
                    <td>MA50</td>
                    <td>{ma50:,.2f}</td>
                </tr>
                <tr>
                    <td>MA200</td>
                    <td>{ma200:,.2f}</td>
                </tr>
            """
            
            # 添加交易信号
            recent_buy_signals = hsi_indicators.get('recent_buy_signals', [])
            recent_sell_signals = hsi_indicators.get('recent_sell_signals', [])
            
            if recent_buy_signals:
                html += f"""
                    <tr>
                        <td colspan="2">
                            <div class="buy-signal">
                                <strong>🔔 恒生指数最近买入信号:</strong><br>
                """
                for signal in recent_buy_signals:
                    html += f"<span style='color: green;'>• {signal['date']}: {signal['description']}</span><br>"
                html += """
                            </div>
                        </td>
                    </tr>
                """
            
            if recent_sell_signals:
                html += f"""
                    <tr>
                        <td colspan="2">
                            <div class="sell-signal">
                                <strong>🔻 恒生指数最近卖出信号:</strong><br>
                """
                for signal in recent_sell_signals:
                    html += f"<span style='color: red;'>• {signal['date']}: {signal['description']}</span><br>"
                html += """
                            </div>
                        </td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
        """

        # 在文本版本中添加恒生指数信息
        text += f"📈 恒生指数价格概览:\n"
        text += f"  当前指数: {hsi_data['current_price']:,.2f}\n"
        text += f"  24小时变化: {hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)\n"
        text += f"  当日开盘: {hsi_data['open']:,.2f}\n"
        text += f"  当日最高: {hsi_data['high']:,.2f}\n"
        text += f"  当日最低: {hsi_data['low']:,.2f}\n"
        text += f"  成交量: {hsi_data['volume']:,.0f}\n\n"
        
        if hsi_indicators:
            text += f"📊 恒生指数技术分析:\n"
            text += f"  趋势: {trend}\n"
            text += f"  RSI: {rsi:.2f}\n"
            text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
            text += f"  布林带位置: {bb_position:.2f}\n"
            text += f"  MA20: {ma20:,.2f}\n"
            text += f"  MA50: {ma50:,.2f}\n"
            text += f"  MA200: {ma200:,.2f}\n"
            
            # 添加交易信号信息到文本版本
            if recent_buy_signals:
                text += f"  🔔 最近买入信号 ({len(recent_buy_signals)} 个):\n"
                for signal in recent_buy_signals:
                    text += f"    {signal['date']}: {signal['description']}\n"
            
            if recent_sell_signals:
                text += f"  🔻 最近卖出信号 ({len(recent_sell_signals)} 个):\n"
                for signal in recent_sell_signals:
                    text += f"    {signal['date']}: {signal['description']}\n"
        
        text += "\n"
    
    # 添加股票分析结果
    for stock_result in stock_results:
        stock_data = stock_result['data']
        indicators = stock_result['indicators']
        
        if indicators:
            # 添加到HTML
            html += generate_stock_analysis_html(stock_data, indicators)
            
            # 添加到文本版本
            text += f"📊 {stock_result['name']} ({stock_result['code']}) 分析:\n"
            text += f"  当前价格: {stock_data['current_price']:,.2f}\n"
            text += f"  24小时变化: {stock_data['change_1d']:+.2f}% ({stock_data['change_1d_points']:+.2f})\n"
            text += f"  当日开盘: {stock_data['open']:,.2f}\n"
            text += f"  当日最高: {stock_data['high']:,.2f}\n"
            text += f"  当日最低: {stock_data['low']:,.2f}\n"
            text += f"  成交量: {stock_data['volume']:,.0f}\n"
            
            # 添加技术指标到文本版本
            rsi = indicators.get('rsi', 0.0)
            macd = indicators.get('macd', 0.0)
            macd_signal = indicators.get('macd_signal', 0.0)
            bb_position = indicators.get('bb_position', 0.5)
            trend = indicators.get('trend', '未知')
            ma20 = indicators.get('ma20', 0)
            ma50 = indicators.get('ma50', 0)
            ma200 = indicators.get('ma200', 0)
            
            text += f"  趋势: {trend}\n"
            text += f"  RSI: {rsi:.2f}\n"
            text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
            text += f"  布林带位置: {bb_position:.2f}\n"
            text += f"  MA20: {ma20:,.2f}\n"
            text += f"  MA50: {ma50:,.2f}\n"
            text += f"  MA200: {ma200:,.2f}\n"
            
            # 添加交易信号信息到文本版本
            recent_buy_signals = indicators.get('recent_buy_signals', [])
            recent_sell_signals = indicators.get('recent_sell_signals', [])
            
            if recent_buy_signals:
                text += f"  🔔 最近买入信号 ({len(recent_buy_signals)} 个):\n"
                for signal in recent_buy_signals:
                    text += f"    {signal['date']}: {signal['description']}\n"
            
            if recent_sell_signals:
                text += f"  🔻 最近卖出信号 ({len(recent_sell_signals)} 个):\n"
                for signal in recent_sell_signals:
                    text += f"    {signal['date']}: {signal['description']}\n"
            
            text += "\n"

    # 添加指标说明
    html += """
    <div class="section">
        <h3>📋 指标说明</h3>
        <div style="font-size:0.9em; line-height:1.4;">
        <ul>
          <li><b>当前指数/价格</b>：恒生指数或股票的实时点位/价格。</li>
          <li><b>24小时变化</b>：过去24小时内指数或股价的变化百分比和点数/金额。</li>
          <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
          <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
          <li><b>MA20(20日移动平均线)</b>：过去20个交易日的平均指数/股价，反映短期趋势。</li>
          <li><b>MA50(50日移动平均线)</b>：过去50个交易日的平均指数/股价，反映中期趋势。</li>
          <li><b>MA200(200日移动平均线)</b>：过去200个交易日的平均指数/股价，反映长期趋势。</li>
          <li><b>布林带位置</b>：当前指数/股价在布林带中的相对位置，范围0-1。接近0表示接近下轨（可能超卖），接近1表示接近上轨（可能超买）。</li>
          <li><b>趋势</b>：市场当前的整体方向。
            <ul>
              <li><b>强势多头</b>：强劲上涨趋势，各周期均线呈多头排列（指数/股价 > MA20 > MA50 > MA200）</li>
              <li><b>多头趋势</b>：上涨趋势，中期均线呈多头排列（指数/股价 > MA20 > MA50）</li>
              <li><b>弱势空头</b>：持续下跌趋势，各周期均线呈空头排列（指数/股价 < MA20 < MA50 < MA200）</li>
              <li><b>空头趋势</b>：下跌趋势，中期均线呈空头排列（指数/股价 < MA20 < MA50）</li>
              <li><b>震荡整理</b>：在一定区间内波动，无明显趋势</li>
              <li><b>短期上涨/下跌</b>：基于最近指数/股价变化的短期趋势判断</li>
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
    print("📝 主题:", subject)
    print("📄 文本预览:\n", text)

    success = send_email(recipients, subject, text, html)
    if not success:
        exit(1)
