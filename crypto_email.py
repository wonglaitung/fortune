import os
import requests
import smtplib
import json
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import yfinance as yf
import pandas as pd



# 导入技术分析工具
try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用简化指标计算")

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
    analyzer = TechnicalAnalyzer()
    
    # 获取历史数据进行技术分析
    indicators = {}
    
    # 获取比特币历史数据
    try:
        btc_ticker = yf.Ticker("BTC-USD")
        btc_hist = btc_ticker.history(period="6mo")  # 获取6个月的历史数据
        if not btc_hist.empty:
            # 计算技术指标
            btc_indicators = analyzer.calculate_all_indicators(btc_hist.copy())
            
            # 生成买卖信号
            btc_indicators_with_signals = analyzer.generate_buy_sell_signals(btc_indicators.copy())
            
            # 分析趋势
            btc_trend = analyzer.analyze_trend(btc_indicators_with_signals)
            
            # 获取最新的指标值
            latest_btc = btc_indicators_with_signals.iloc[-1]
            btc_rsi = latest_btc.get('RSI', 50.0)
            btc_macd = latest_btc.get('MACD', 0.0)
            btc_macd_signal = latest_btc.get('MACD_signal', 0.0)
            btc_bb_position = latest_btc.get('BB_position', 0.5) if 'BB_position' in latest_btc else 0.5
            
            # 检查最近的交易信号
            recent_signals = btc_indicators_with_signals.tail(5)
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
            
            indicators['bitcoin'] = {
                'rsi': btc_rsi,
                'macd': btc_macd,
                'macd_signal': btc_macd_signal,
                'price_position': calculate_price_position(latest_btc.get('Close', 0)),
                'bb_position': btc_bb_position,
                'trend': btc_trend,
                'recent_buy_signals': buy_signals,
                'recent_sell_signals': sell_signals,
                'current_price': latest_btc.get('Close', 0),
                'ma20': latest_btc.get('MA20', 0),
                'ma50': latest_btc.get('MA50', 0),
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
        eth_hist = eth_ticker.history(period="6mo")  # 获取6个月的历史数据
        if not eth_hist.empty:
            # 计算技术指标
            eth_indicators = analyzer.calculate_all_indicators(eth_hist.copy())
            
            # 生成买卖信号
            eth_indicators_with_signals = analyzer.generate_buy_sell_signals(eth_indicators.copy())
            
            # 分析趋势
            eth_trend = analyzer.analyze_trend(eth_indicators_with_signals)
            
            # 获取最新的指标值
            latest_eth = eth_indicators_with_signals.iloc[-1]
            eth_rsi = latest_eth.get('RSI', 50.0)
            eth_macd = latest_eth.get('MACD', 0.0)
            eth_macd_signal = latest_eth.get('MACD_signal', 0.0)
            eth_bb_position = latest_eth.get('BB_position', 0.5) if 'BB_position' in latest_eth else 0.5
            
            # 检查最近的交易信号
            recent_signals = eth_indicators_with_signals.tail(5)
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
            
            indicators['ethereum'] = {
                'rsi': eth_rsi,
                'macd': eth_macd,
                'macd_signal': eth_macd_signal,
                'price_position': calculate_price_position(latest_eth.get('Close', 0)),
                'bb_position': eth_bb_position,
                'trend': eth_trend,
                'recent_buy_signals': buy_signals,
                'recent_sell_signals': sell_signals,
                'current_price': latest_eth.get('Close', 0),
                'ma20': latest_eth.get('MA20', 0),
                'ma50': latest_eth.get('MA50', 0),
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



def send_email(to, subject, text, html):
    smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
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
    # 可以通过修改这里的参数来控制是否包含市值和24小时交易量
    prices = get_cryptocurrency_prices(include_market_cap=True, include_24hr_vol=True)

    if prices is None:
        print("Failed to fetch prices. Exiting.")
        exit(1)

    # 计算技术指标
    indicators = calculate_technical_indicators(prices)

    # 检查是否存在当天的交易信号
    has_signals = False
    today = datetime.now().date()
    
    if 'ethereum' in indicators:
        eth_recent_buy_signals = indicators['ethereum'].get('recent_buy_signals', [])
        eth_recent_sell_signals = indicators['ethereum'].get('recent_sell_signals', [])
        
        # 检查以太坊是否有今天的信号
        for signal in eth_recent_buy_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for signal in eth_recent_sell_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
    
    if 'bitcoin' in indicators and not has_signals:
        btc_recent_buy_signals = indicators['bitcoin'].get('recent_buy_signals', [])
        btc_recent_sell_signals = indicators['bitcoin'].get('recent_sell_signals', [])
        
        # 检查比特币是否有今天的信号
        for signal in btc_recent_buy_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break
        for signal in btc_recent_sell_signals:
            if datetime.strptime(signal['date'], '%Y-%m-%d').date() == today:
                has_signals = True
                break

    # 如果没有交易信号，则不发送邮件
    if not has_signals:
        print("⚠️ 没有检测到任何交易信号，跳过发送邮件。")
        exit(0)

    subject = "Ethereum and Bitcoin Price Update - 交易信号提醒"

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
                    <td>${{eth_usd:,.2f}}</td>
                    <td>${{eth_hkd:,.2f}}</td>
                    <td>{{eth_change:+.2f}}%</td>
                    <td>${{eth_market_cap:,.2f}}</td>
                    <td>${{eth_24hr_vol:,.2f}}</td>
                </tr>
        """
        
        text += f"Ethereum (ETH):\n"
        text += f"  价格: ${{eth_usd:,.2f}} USD ({{eth_change:+.2f}}% 24h)\n"
        text += f"  价格: ${{eth_hkd:,.2f}} HKD\n"
        if eth_market_cap > 0:
            text += f"  市值: ${{eth_market_cap:,.2f}} USD\n"
        if eth_24hr_vol > 0:
            text += f"  24小时交易量: ${{eth_24hr_vol:,.2f}} USD\n"
    
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
                    <td>${{btc_usd:,.2f}}</td>
                    <td>${{btc_hkd:,.2f}}</td>
                    <td>{{btc_change:+.2f}}%</td>
                    <td>${{btc_market_cap:,.2f}}</td>
                    <td>${{btc_24hr_vol:,.2f}}</td>
                </tr>
        """
        
        text += f"Bitcoin (BTC):\n"
        text += f"  价格: ${{btc_usd:,.2f}} USD ({{btc_change:+.2f}}% 24h)\n"
        text += f"  价格: ${{btc_hkd:,.2f}} HKD\n"
        if btc_market_cap > 0:
            text += f"  市值: ${{btc_market_cap:,.2f}} USD\n"
        if btc_24hr_vol > 0:
            text += f"  24小时交易量: ${{btc_24hr_vol:,.2f}} USD\n"
    
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
                    <th>MA20</th>
                    <th>MA50</th>
                </tr>
    """
    
    # Ethereum 技术分析
    if 'ethereum' in prices:
        eth_rsi = indicators['ethereum'].get('rsi', 0.0)
        eth_macd = indicators['ethereum'].get('macd', 0.0)
        eth_macd_signal = indicators['ethereum'].get('macd_signal', 0.0)
        eth_bb_position = indicators['ethereum'].get('bb_position', 0.5)
        eth_trend = indicators['ethereum'].get('trend', '未知')
        eth_ma20 = indicators['ethereum'].get('ma20', 0)
        eth_ma50 = indicators['ethereum'].get('ma50', 0)
        eth_recent_buy_signals = indicators['ethereum'].get('recent_buy_signals', [])
        eth_recent_sell_signals = indicators['ethereum'].get('recent_sell_signals', [])
        
        html += f"""
                <tr>
                    <td>Ethereum (ETH)</td>
                    <td>{{eth_trend}}</td>
                    <td>{{eth_rsi:.2f}}</td>
                    <td>{{eth_macd:.4f}}</td>
                    <td>{{eth_macd_signal:.4f}}</td>
                    <td>{{eth_bb_position:.2f}}</td>
                    <td>${{eth_ma20:.2f}}</td>
                    <td>${{eth_ma50:.2f}}</td>
                </tr>
        """
        
        # 添加交易信号到表格中
        if eth_recent_buy_signals:
            html += f"""
                <tr>
                    <td colspan=\"8\">
                        <div class=\"buy-signal\">
                            <strong>🔔 Ethereum (ETH) 最近买入信号:</strong><br>
            """
            for signal in eth_recent_buy_signals:
                html += f"<span style='color: green;'>• {{signal['date']}}: {{signal['description']}}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        if eth_recent_sell_signals:
            html += f"""
                <tr>
                    <td colspan=\"8\">
                        <div class=\"sell-signal\">
                            <strong>🔻 Ethereum (ETH) 最近卖出信号:</strong><br>
            """
            for signal in eth_recent_sell_signals:
                html += f"<span style='color: red;'>• {{signal['date']}}: {{signal['description']}}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        text += f"  RSI: {{eth_rsi:.2f}}\n"
        text += f"  MACD: {{eth_macd:.4f}} (信号线: {{eth_macd_signal:.4f}})\n"
        text += f"  布林带位置: {{eth_bb_position:.2f}}\n"
        text += f"  趋势: {{eth_trend}}\n"
        text += f"  MA20: ${{eth_ma20:.2f}}\n"
        text += f"  MA50: ${{eth_ma50:.2f}}\n"
        
        # 添加交易信号信息到文本版本
        if eth_recent_buy_signals:
            text += f"  🔔 最近买入信号 ({{len(eth_recent_buy_signals)}} 个):\n"
            for signal in eth_recent_buy_signals:
                text += f"    {{signal['date']}}: {{signal['description']}}\n"
        
        if eth_recent_sell_signals:
            text += f"  🔻 最近卖出信号 ({{len(eth_recent_sell_signals)}} 个):\n"
            for signal in eth_recent_sell_signals:
                text += f"    {{signal['date']}}: {{signal['description']}}\n"
    
    # Bitcoin 技术分析
    if 'bitcoin' in prices:
        btc_rsi = indicators['bitcoin'].get('rsi', 0.0)
        btc_macd = indicators['bitcoin'].get('macd', 0.0)
        btc_macd_signal = indicators['bitcoin'].get('macd_signal', 0.0)
        btc_bb_position = indicators['bitcoin'].get('bb_position', 0.5)
        btc_trend = indicators['bitcoin'].get('trend', '未知')
        btc_ma20 = indicators['bitcoin'].get('ma20', 0)
        btc_ma50 = indicators['bitcoin'].get('ma50', 0)
        btc_recent_buy_signals = indicators['bitcoin'].get('recent_buy_signals', [])
        btc_recent_sell_signals = indicators['bitcoin'].get('recent_sell_signals', [])
        
        html += f"""
                <tr>
                    <td>Bitcoin (BTC)</td>
                    <td>{{btc_trend}}</td>
                    <td>{{btc_rsi:.2f}}</td>
                    <td>{{btc_macd:.4f}}</td>
                    <td>{{btc_macd_signal:.4f}}</td>
                    <td>{{btc_bb_position:.2f}}</td>
                    <td>${{btc_ma20:.2f}}</td>
                    <td>${{btc_ma50:.2f}}</td>
                </tr>
        """
        
        # 添加交易信号到表格中
        if btc_recent_buy_signals:
            html += f"""
                <tr>
                    <td colspan=\"8\">
                        <div class=\"buy-signal\">
                            <strong>🔔 Bitcoin (BTC) 最近买入信号:</strong><br>
            """
            for signal in btc_recent_buy_signals:
                html += f"<span style='color: green;'>• {{signal['date']}}: {{signal['description']}}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        if btc_recent_sell_signals:
            html += f"""
                <tr>
                    <td colspan=\"8\">
                        <div class=\"sell-signal\">
                            <strong>🔻 Bitcoin (BTC) 最近卖出信号:</strong><br>
            """
            for signal in btc_recent_sell_signals:
                html += f"<span style='color: red;'>• {{signal['date']}}: {{signal['description']}}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """
        
        text += f"  RSI: {{btc_rsi:.2f}}\n"
        text += f"  MACD: {{btc_macd:.4f}} (信号线: {{btc_macd_signal:.4f}})\n"
        text += f"  布林带位置: {{btc_bb_position:.2f}}\n"
        text += f"  趋势: {{btc_trend}}\n"
        text += f"  MA20: ${{btc_ma20:.2f}}\n"
        text += f"  MA50: ${{btc_ma50:.2f}}\n"
        
        # 添加交易信号信息到文本版本
        if btc_recent_buy_signals:
            text += f"  🔔 最近买入信号 ({{len(btc_recent_buy_signals)}} 个):\n"
            for signal in btc_recent_buy_signals:
                text += f"    {{signal['date']}}: {{signal['description']}}\n"
        
        if btc_recent_sell_signals:
            text += f"  🔻 最近卖出信号 ({{len(btc_recent_sell_signals)}} 个):\n"
            for signal in btc_recent_sell_signals:
                text += f"    {{signal['date']}}: {{signal['description']}}\n"
    
    html += """
            </table>
        </div>
    """

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

