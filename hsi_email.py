#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数价格监控和交易信号邮件通知系统
基于技术分析指标生成买卖信号，只在有交易信号时发送邮件
"""

import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
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

def calculate_technical_indicators(hsi_data):
    """
    计算恒生指数技术指标
    """
    if hsi_data is None:
        return None
    
    hist = hsi_data['hist']
    
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
                buy_signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'description': row['Signal_Description']
                })
        
        if 'Sell_Signal' in recent_signals.columns:
            sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
            for idx, row in sell_signals_df.iterrows():
                sell_signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'description': row['Signal_Description']
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
    
    # 获取恒生指数数据
    hsi_data = get_hsi_data()

    if hsi_data is None:
        print("❌ 无法获取恒生指数数据，退出。")
        exit(1)

    # 计算技术指标
    print("📊 正在计算技术指标...")
    indicators = calculate_technical_indicators(hsi_data)

    if indicators is None:
        print("❌ 无法计算技术指标，退出。")
        exit(1)

    # 检查是否存在当天的交易信号
    has_signals = False
    today = datetime.now().date()
    
    if indicators:
        recent_buy_signals = indicators.get('recent_buy_signals', [])
        recent_sell_signals = indicators.get('recent_sell_signals', [])
        
        # 检查是否有今天的买入信号
        for signal in recent_buy_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        # 检查是否有今天的卖出信号
        for signal in recent_sell_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break

    # 如果没有交易信号，则不发送邮件
    if not has_signals:
        print("⚠️ 没有检测到任何交易信号，跳过发送邮件。")
        exit(0)

    subject = "恒生指数交易信号提醒"

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
        <h2>📈 恒生指数交易信号提醒</h2>
        <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    # 恒生指数价格概览
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
    
    html += """
            </table>
        </div>
    """

    # 技术分析
    html += """
        <div class="section">
            <h3>🔬 技术分析</h3>
            <table>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                </tr>
    """
    
    rsi = indicators.get('rsi', 0.0)
    macd = indicators.get('macd', 0.0)
    macd_signal = indicators.get('macd_signal', 0.0)
    bb_position = indicators.get('bb_position', 0.5)
    trend = indicators.get('trend', '未知')
    ma20 = indicators.get('ma20', 0)
    ma50 = indicators.get('ma50', 0)
    ma200 = indicators.get('ma200', 0)
    recent_buy_signals = indicators.get('recent_buy_signals', [])
    recent_sell_signals = indicators.get('recent_sell_signals', [])
    
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
    
    # 添加交易信号到表格中
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

    # 在文本版本中也添加信息
    text += f"📈 恒生指数价格概览:\n"
    text += f"  当前指数: {hsi_data['current_price']:,.2f}\n"
    text += f"  24小时变化: {hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)\n"
    text += f"  当日开盘: {hsi_data['open']:,.2f}\n"
    text += f"  当日最高: {hsi_data['high']:,.2f}\n"
    text += f"  当日最低: {hsi_data['low']:,.2f}\n"
    text += f"  成交量: {hsi_data['volume']:,.0f}\n\n"
    
    text += f"📊 技术分析:\n"
    text += f"  趋势: {trend}\n"
    text += f"  RSI: {rsi:.2f}\n"
    text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
    text += f"  布林带位置: {bb_position:.2f}\n"
    text += f"  MA20: {ma20:,.2f}\n"
    text += f"  MA50: {ma50:,.2f}\n"
    text += f"  MA200: {ma200:,.2f}\n"
    
    # 添加交易信号信息到文本版本
    if recent_buy_signals:
        text += f"\n🔔 最近买入信号 ({len(recent_buy_signals)} 个):\n"
        for signal in recent_buy_signals:
            text += f"  {signal['date']}: {signal['description']}\n"
    
    if recent_sell_signals:
        text += f"\n🔻 最近卖出信号 ({len(recent_sell_signals)} 个):\n"
        for signal in recent_sell_signals:
            text += f"  {signal['date']}: {signal['description']}\n"
    
    # 添加指标说明到文本版本
    text += "\n📋 指标说明:\n"
    text += "当前指数：恒生指数的实时点位。\n"
    text += "24小时变化：过去24小时内指数的变化百分比和点数。\n"
    text += "RSI(相对强弱指数)：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。\n"
    text += "MACD(异同移动平均线)：判断价格趋势和动能的技术指标。\n"
    text += "MA20(20日移动平均线)：过去20个交易日的平均指数，反映短期趋势。\n"
    text += "MA50(50日移动平均线)：过去50个交易日的平均指数，反映中期趋势。\n"
    text += "MA200(200日移动平均线)：过去200个交易日的平均指数，反映长期趋势。\n"
    text += "布林带位置：当前指数在布林带中的相对位置，范围0-1。接近0表示指数接近下轨（可能超卖），接近1表示指数接近上轨（可能超买）。\n"
    text += "趋势：市场当前的整体方向。\n"
    text += "  强势多头：指数强劲上涨趋势，各周期均线呈多头排列（指数 > MA20 > MA50 > MA200）\n"
    text += "  多头趋势：指数上涨趋势，中期均线呈多头排列（指数 > MA20 > MA50）\n"
    text += "  弱势空头：指数持续下跌趋势，各周期均线呈空头排列（指数 < MA20 < MA50 < MA200）\n"
    text += "  空头趋势：指数下跌趋势，中期均线呈空头排列（指数 < MA20 < MA50）\n"
    text += "  震荡整理：指数在一定区间内波动，无明显趋势\n"
    text += "  短期上涨/下跌：基于最近指数变化的短期趋势判断\n"
    text += "\n"
    
    # 添加指标说明
    html += """
    <div class="section">
        <h3>📋 指标说明</h3>
        <div style="font-size:0.9em; line-height:1.4;">
        <ul>
          <li><b>当前指数</b>：恒生指数的实时点位。</li>
          <li><b>24小时变化</b>：过去24小时内指数的变化百分比和点数。</li>
          <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
          <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
          <li><b>MA20(20日移动平均线)</b>：过去20个交易日的平均指数，反映短期趋势。</li>
          <li><b>MA50(50日移动平均线)</b>：过去50个交易日的平均指数，反映中期趋势。</li>
          <li><b>MA200(200日移动平均线)</b>：过去200个交易日的平均指数，反映长期趋势。</li>
          <li><b>布林带位置</b>：当前指数在布林带中的相对位置，范围0-1。接近0表示指数接近下轨（可能超卖），接近1表示指数接近上轨（可能超买）。</li>
          <li><b>趋势</b>：市场当前的整体方向。
            <ul>
              <li><b>强势多头</b>：指数强劲上涨趋势，各周期均线呈多头排列（指数 > MA20 > MA50 > MA200）</li>
              <li><b>多头趋势</b>：指数上涨趋势，中期均线呈多头排列（指数 > MA20 > MA50）</li>
              <li><b>弱势空头</b>：指数持续下跌趋势，各周期均线呈空头排列（指数 < MA20 < MA50 < MA200）</li>
              <li><b>空头趋势</b>：指数下跌趋势，中期均线呈空头排列（指数 < MA20 < MA50）</li>
              <li><b>震荡整理</b>：指数在一定区间内波动，无明显趋势</li>
              <li><b>短期上涨/下跌</b>：基于最近指数变化的短期趋势判断</li>
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
