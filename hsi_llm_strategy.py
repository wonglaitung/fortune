#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数(HSI)大模型策略分析器
此脚本用于获取当前恒生指数数据并调用大模型生成明确的交易策略建议
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import yfinance as yf

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Markdown到HTML的转换函数
# Markdown到HTML的转换函数
def markdown_to_html(md_text):
    if not md_text:
        return md_text

    # 保存原始文本并逐行处理
    lines = md_text.split('\n')
    html_lines = []
    in_list = False
    list_type = None  # 'ul' for unordered, 'ol' for ordered
    in_table = False  # 标记是否在表格中
    table_header_processed = False  # 标记表格头部是否已处理

    for line in lines:
        stripped_line = line.strip()
        
        # 检查是否是表格分隔行（包含 | 和 - 用于定义表格结构）
        table_separator_match = re.match(r'^\s*\|?\s*[:\-\s\|]*\|\s*$', line)
        if table_separator_match and '|' in line and any(c in line for c in ['-', ':']):
            # 这是表格的分隔行，跳过处理
            continue

        # 检查是否是表格行（包含 | 分隔符）
        is_table_row = '|' in line and not stripped_line.startswith('```')
        
        if is_table_row and not table_separator_match:
            # 处理表格行
            if not in_table:
                # 开始新表格
                in_table = True
                table_header_processed = False
                html_lines.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
            
            # 分割单元格并去除空白
            cells = [cell.strip() for cell in line.split('|')]
            # 过滤掉首尾的空字符串（因为 | 开头或结尾会产生空字符串）
            cells = [cell for i, cell in enumerate(cells) if i > 0 and i < len(cells) - 1]
            
            # 确定是表头还是数据行
            if not table_header_processed and any('---' in cell for cell in [c for c in cells if c.strip()]):
                # 如果这一行包含 ---，则认为是分隔行，跳过
                continue
            elif not table_header_processed:
                # 首次遇到非分隔行，作为表头处理
                html_lines.append('<thead><tr>')
                for cell in cells:
                    # 处理单元格内的粗体和斜体
                    cell_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    cell_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cell_content)
                    cell_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', cell_content)
                    cell_content = re.sub(r'_(.*?)_', r'<em>\1</em>', cell_content)
                    # 处理代码
                    cell_content = re.sub(r'`(.*?)`', r'<code>\1</code>', cell_content)
                    # 处理链接
                    cell_content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', cell_content)
                    html_lines.append(f'<th style="padding: 8px; text-align: left; border: 1px solid #ddd;">{cell_content}</th>')
                html_lines.append('</tr></thead><tbody>')
                table_header_processed = True
            else:
                # 数据行
                html_lines.append('<tr>')
                for cell in cells:
                    # 处理单元格内的粗体和斜体
                    cell_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    cell_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cell_content)
                    cell_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', cell_content)
                    cell_content = re.sub(r'_(.*?)_', r'<em>\1</em>', cell_content)
                    # 处理代码
                    cell_content = re.sub(r'`(.*?)`', r'<code>\1</code>', cell_content)
                    # 处理链接
                    cell_content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', cell_content)
                    html_lines.append(f'<td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{cell_content}</td>')
                html_lines.append('</tr>')
            continue

        # 如果当前行不是表格行，但之前在表格中，则关闭表格
        if in_table:
            html_lines.append('</tbody></table>')
            in_table = False
            table_header_processed = False

        # 检查是否是标题
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
            header_level = len(header_match.group(1))
            header_content = header_match.group(2)
            # 处理标题内的粗体和斜体
            header_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', header_content)
            header_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', header_content)
            header_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', header_content)
            header_content = re.sub(r'_(.*?)_', r'<em>\1</em>', header_content)
            html_lines.append(f'<h{header_level}>{header_content}</h{header_level}>')
            continue

        # 检查是否是列表项（无序）
        ul_match = re.match(r'^\s*[-*+]\s+(.*)', line)
        if ul_match:
            content = ul_match.group(1).strip()
            # 处理列表项内的粗体和斜体
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', content)
            content = re.sub(r'_(.*?)_', r'<em>\1</em>', content)
            
            if not in_list or list_type != 'ul':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            
            html_lines.append(f'<li>{content}</li>')
            continue

        # 检查是否是列表项（有序）
        ol_match = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if ol_match:
            content = ol_match.group(2).strip()
            # 处理列表项内的粗体和斜体
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', content)
            content = re.sub(r'_(.*?)_', r'<em>\1</em>', content)
            
            if not in_list or list_type != 'ol':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            
            html_lines.append(f'<li>{content}</li>')
            continue

        # 如果当前行不是列表项，但之前在列表中，则关闭列表
        if in_list:
            html_lines.append(f'</{list_type}>')
            in_list = False

        # 处理普通行
        if stripped_line:
            # 处理粗体和斜体
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            processed_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_line)
            processed_line = re.sub(r'__(.*?)__', r'<strong>\1</strong>', processed_line)
            processed_line = re.sub(r'_(.*?)_', r'<em>\1</em>', processed_line)
            # 处理代码
            processed_line = re.sub(r'`(.*?)`', r'<code>\1</code>', processed_line)
            # 处理链接
            processed_line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', processed_line)
            html_lines.append(processed_line)
        else:
            # 空行转为<br>
            html_lines.append('<br>')

    # 如果文档以列表结束，关闭列表
    if in_list:
        html_lines.append(f'</{list_type}>')

    # 如果文档以表格结束，关闭表格
    if in_table:
        html_lines.append('</tbody></table>')

    # 将所有行用<br>连接（但避免在已有HTML标签后添加额外的<br>）
    final_html = '<br>'.join(html_lines)
    # 修复多余的<br>标签
    final_html = re.sub(r'<br>(\s*<(ul|ol|h[1-6]|/ul|/ol|/h[1-6]|table|/table|/tbody|/thead|tr|/tr|td|/td|th|/th)>)', r'\1', final_html)
    final_html = re.sub(r'<br><br>', r'<br>', final_html)

    return final_html
# 邮件发送函数
def send_email(to, subject, text, html=None):
    """发送邮件功能"""
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
    
    # 如果提供了HTML内容，则也添加HTML版本
    if html:
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
    
    print("❌ Failed to send email after 3 attempts")
    return False

# 导入腾讯财经接口
from data_services.tencent_finance import get_hsi_data_tencent

# 导入技术分析工具
from data_services.technical_analysis import TechnicalAnalyzer

# 导入大模型服务
try:
    from llm_services.qwen_engine import chat_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("警告: 无法导入大模型服务，将跳过大模型分析功能")

def get_overseas_market_data():
    """
    获取隔夜美股及A50期货数据
    """
    print("🌍 开始获取隔夜美股及A50期货数据...")
    
    overseas_data = {}
    
    try:
        # 获取主要美股指数数据
        us_indices = {
            'SPY': '标普500 ETF', 
            'QQQ': '纳斯达克100 ETF', 
            'DIA': '道琼斯工业平均ETF',
            'TLT': '20+年国债ETF(反映利率情绪)'
        }
        
        for symbol, name in us_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")  # 获取最近5天的数据
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
                    change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
                    overseas_data[symbol] = {
                        'name': name,
                        'price': latest['Close'],
                        'change_pct': change_pct,
                        'volume': latest['Volume']
                    }
                    print(f"✅ {name}({symbol}): {latest['Close']:.2f}, 涨跌: {change_pct:+.2f}%")
                else:
                    print(f"⚠️ 无法获取 {symbol} 数据")
                    overseas_data[symbol] = {
                        'name': name,
                        'price': 0,
                        'change_pct': 0,
                        'volume': 0
                    }
            except Exception as e:
                print(f"⚠️ 获取 {symbol} 数据失败: {e}")
                overseas_data[symbol] = {
                    'name': name,
                    'price': 0,
                    'change_pct': 0,
                    'volume': 0
                }
        
        # 获取恐慌指数(VIX)
        try:
            vix_ticker = yf.Ticker("^VIX")  # VIX指数的正确符号
            hist = vix_ticker.history(period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
                change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
                overseas_data['VIX'] = {
                    'name': '恐慌指数(VIX)',
                    'price': latest['Close'],
                    'change_pct': change_pct,
                    'volume': latest['Volume']
                }
                print(f"✅ 恐慌指数(VIX): {latest['Close']:.2f}, 涨跌: {change_pct:+.2f}%")
            else:
                print(f"⚠️ 无法获取 VIX 数据")
                overseas_data['VIX'] = {
                    'name': '恐慌指数(VIX)',
                    'price': 0,
                    'change_pct': 0,
                    'volume': 0
                }
        except Exception as e:
            print(f"⚠️ 获取 VIX 数据失败: {e}")
            overseas_data['VIX'] = {
                'name': '恐慌指数(VIX)',
                'price': 0,
                'change_pct': 0,
                'volume': 0
            }
        
        # 获取A50期货数据 - 尝试不同的可能代码
        a50_symbols = ["CHI50.MI", "05101.HK", "FU50.CFE"]  # 尝试不同的A50期货代码
        a50_found = False
        
        for symbol in a50_symbols:
            try:
                a50_futures = yf.Ticker(symbol)
                hist = a50_futures.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
                    change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
                    overseas_data['A50_FUTURES'] = {
                        'name': f'富时中国A50指数期货({symbol})',
                        'price': latest['Close'],
                        'change_pct': change_pct,
                        'volume': latest['Volume']
                    }
                    print(f"✅ A50期货({symbol}): {latest['Close']:.2f}, 涨跌: {change_pct:+.2f}%")
                    a50_found = True
                    break
                else:
                    print(f"⚠️ 无法获取 {symbol} 数据")
            except Exception as e:
                print(f"⚠️ 获取 {symbol} 数据失败: {e}")
        
        if not a50_found:
            print(f"⚠️ 无法获取 A50期货 数据，所有尝试的代码均失败")
            overseas_data['A50_FUTURES'] = {
                'name': '富时中国A50指数期货',
                'price': 0,
                'change_pct': 0,
                'volume': 0
            }
        
        return overseas_data
    except Exception as e:
        print(f"❌ 获取海外数据时发生错误: {e}")
        return {}

def assess_risk_level(overseas_data):
    """
    评估隔夜市场对港股的潜在风险水平
    """
    risk_level = "中等"
    risk_factors = []
    
    if not overseas_data:
        return risk_level, risk_factors
    
    # 评估美股风险
    for symbol in ['SPY', 'QQQ', 'DIA']:
        if symbol in overseas_data:
            change_pct = overseas_data[symbol]['change_pct']
            if abs(change_pct) > 3:
                risk_factors.append(f"{overseas_data[symbol]['name']}({symbol})隔夜波动{change_pct:+.2f}%，波动较大")
            elif abs(change_pct) > 2:
                risk_factors.append(f"{overseas_data[symbol]['name']}({symbol})隔夜波动{change_pct:+.2f}%，波动明显")
    
    # 评估A50期货风险
    if 'A50_FUTURES' in overseas_data:
        a50_change = overseas_data['A50_FUTURES']['change_pct']
        # 如果A50期货数据是默认的0值，说明实际数据获取失败，忽略此数据
        if a50_change != 0 or overseas_data['A50_FUTURES']['price'] != 0:
            if abs(a50_change) > 2.5:
                risk_factors.append(f"A50期货隔夜波动{a50_change:+.2f}%，可能影响A股及港股走势")
            elif abs(a50_change) > 1.5:
                risk_factors.append(f"A50期货隔夜波动{a50_change:+.2f}%，对A股及港股有一定影响")
    
    # 评估恐慌指数(VIX)
    if 'VIX' in overseas_data:
        vix_value = overseas_data['VIX']['change_pct']
        if vix_value > 5:
            risk_factors.append(f"恐慌指数(VIX)大幅上升{vix_value:+.2f}%，市场避险情绪浓厚")
        elif vix_value > 2:
            risk_factors.append(f"恐慌指数(VIX)上升{vix_value:+.2f}%，市场情绪偏谨慎")
        elif vix_value < -5:
            risk_factors.append(f"恐慌指数(VIX)大幅下降{vix_value:+.2f}%，市场风险偏好过高需警惕")
    
    # 综合评估风险等级
    high_risk_factors = [f for f in risk_factors if "波动较大" in f or "大幅上升" in f or "大幅下降" in f or "数据缺失" in f]
    medium_risk_factors = [f for f in risk_factors if "波动明显" in f or "上升" in f or "下降" in f]
    
    if len(high_risk_factors) >= 2 or (len(high_risk_factors) >= 1 and len(medium_risk_factors) >= 2):
        risk_level = "高风险"
    elif len(high_risk_factors) >= 1 or len(medium_risk_factors) >= 2:
        risk_level = "中高风险"
    
    return risk_level, risk_factors

warnings.filterwarnings('ignore')

def generate_hsi_llm_strategy(overseas_data=None):
    """
    生成恒生指数大模型策略分析
    """
    print("🚀 开始获取恒生指数数据...")
    
    # 获取最新数据
    period_days = 90
    data = get_hsi_data_tencent(period_days=period_days)
    
    if data is None or data.empty:
        print("❌ 无法获取恒生指数数据")
        return None
    
    print(f"✅ 成功获取 {len(data)} 天的恒生指数数据")
    
    # 创建技术分析器并计算指标
    technical_analyzer = TechnicalAnalyzer()
    indicators = technical_analyzer.calculate_all_indicators(data.copy())
    
    # 计算额外的恒生指数专用指标
    # 计算价格位置（在最近N日内的百分位位置）
    price_window = 60
    if len(indicators) >= price_window:
        rolling_low = indicators['Close'].rolling(window=price_window).min()
        rolling_high = indicators['Close'].rolling(window=price_window).max()
        indicators['Price_Percentile'] = ((indicators['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
    else:
        # 如果数据不足，使用全部可用数据
        rolling_low = indicators['Close'].rolling(window=len(indicators)).min()
        rolling_high = indicators['Close'].rolling(window=len(indicators)).max()
        indicators['Price_Percentile'] = ((indicators['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
    
    # 计算成交量比率（相对于20日均量）
    indicators['Vol_MA20'] = indicators['Volume'].rolling(window=20).mean()
    indicators['Vol_Ratio'] = indicators['Volume'] / indicators['Vol_MA20']
    
    # 计算波动率（20日年化波动率）
    indicators['Returns'] = indicators['Close'].pct_change()
    indicators['Volatility'] = indicators['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    # 获取最新数据
    latest = indicators.iloc[-1]
    
    print(f"📊 当前恒生指数: {latest['Close']:.2f}")
    print(f"📈 RSI: {latest['RSI']:.2f}")
    print(f"📊 MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
    print(f"均线: MA20: {latest['MA20']:.2f}, MA50: {latest['MA50']:.2f}")
    print(f"价格位置: {latest['Price_Percentile']:.2f}%")
    print(f"波动率: {latest['Volatility']:.2f}%")
    print(f"量比: {latest['Vol_Ratio']:.2f}")
    
    # 如果没有提供海外数据，则获取
    if overseas_data is None:
        overseas_data = get_overseas_market_data()
    
    # 评估风险等级
    risk_level, risk_factors = assess_risk_level(overseas_data)
    
    # 构建分析报告内容作为大模型输入
    analysis_summary = []
    analysis_summary.append("恒生指数(HSI)技术分析数据:")
    analysis_summary.append(f"当前指数: {latest['Close']:.2f}")
    analysis_summary.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_summary.append("")
    
    # 添加关键技术指标
    analysis_summary.append("关键技术指标:")
    if 'RSI' in indicators.columns:
        analysis_summary.append(f"RSI: {latest['RSI']:.2f}")
    if 'MACD' in indicators.columns and 'MACD_signal' in indicators.columns:
        analysis_summary.append(f"MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
    if 'MA20' in indicators.columns:
        analysis_summary.append(f"MA20: {latest['MA20']:.2f}")
    if 'MA50' in indicators.columns:
        analysis_summary.append(f"MA50: {latest['MA50']:.2f}")
    if 'MA200' in indicators.columns:
        analysis_summary.append(f"MA200: {latest['MA200']:.2f}")
    if 'Price_Percentile' in indicators.columns:
        analysis_summary.append(f"价格位置: {latest['Price_Percentile']:.2f}%")
    if 'Volatility' in indicators.columns:
        analysis_summary.append(f"波动率: {latest['Volatility']:.2f}%")
    if 'Vol_Ratio' in indicators.columns:
        analysis_summary.append(f"量比: {latest['Vol_Ratio']:.2f}")
    analysis_summary.append("")
    
    # 添加趋势分析
    current_price = latest['Close']
    ma20 = latest['MA20'] if 'MA20' in indicators.columns and not pd.isna(latest['MA20']) else np.nan
    ma50 = latest['MA50'] if 'MA50' in indicators.columns and not pd.isna(latest['MA50']) else np.nan
    ma200 = latest['MA200'] if 'MA200' in indicators.columns and not pd.isna(latest['MA200']) else np.nan
    
    trend = "未知"
    if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
        if current_price > ma20 > ma50 > ma200:
            trend = "强势多头"
        elif current_price < ma20 < ma50 < ma200:
            trend = "弱势空头"
        else:
            trend = "震荡整理"
    elif not pd.isna(ma20) and not pd.isna(ma50):
        if current_price > ma20 > ma50:
            trend = "多头趋势"
        elif current_price < ma20 < ma50:
            trend = "空头趋势"
        else:
            trend = "震荡"
    
    analysis_summary.append(f"当前趋势: {trend}")
    analysis_summary.append("")
    
    # 获取历史数据用于趋势分析
    historical_data = indicators.tail(20)  # 最近20天的数据
    analysis_summary.append("最近20天指数变化:")
    for idx, row in historical_data.iterrows():
        analysis_summary.append(f"  {idx.strftime('%Y-%m-%d')}: {row['Close']:.2f}")
    analysis_summary.append("")
    
    # 添加隔夜海外市场数据
    if overseas_data:
        analysis_summary.append("隔夜海外市场数据:")
        for symbol, data in overseas_data.items():
            if 'A50_FUTURES' in symbol:
                # 如果A50期货数据是实际数据（非默认0值），才显示
                if data['change_pct'] != 0 or data['price'] != 0:
                    analysis_summary.append(f"{data['name']}: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
                else:
                    analysis_summary.append(f"{data['name']}: 数据缺失 (无法获取)")
            else:
                analysis_summary.append(f"{data['name']}: {data['price']:.2f}, 涨跌: {data['change_pct']:+.2f}%")
        analysis_summary.append("")
        
        # 分析隔夜市场对港股的潜在影响
        analysis_summary.append("隔夜市场影响分析:")
        
        # 分析美股三大指数
        us_impact = 0
        if 'SPY' in overseas_data:
            spy_change = overseas_data['SPY']['change_pct']
            if spy_change > 1:
                analysis_summary.append(f"• SPY(标普500)上涨 {spy_change:.2f}%，对港股形成正面带动")
                us_impact += 1
            elif spy_change < -1:
                analysis_summary.append(f"• SPY(标普500)下跌 {spy_change:.2f}%，对港股形成负面冲击")
                us_impact -= 1
            else:
                analysis_summary.append(f"• SPY(标普500)涨跌 {spy_change:.2f}%，对港股影响中性")
                
        if 'QQQ' in overseas_data:
            qqq_change = overseas_data['QQQ']['change_pct']
            if qqq_change > 1:
                analysis_summary.append(f"• QQQ(纳斯达克100)上涨 {qqq_change:.2f}%，利好科技股")
                us_impact += 1
            elif qqq_change < -1:
                analysis_summary.append(f"• QQQ(纳斯达克100)下跌 {qqq_change:.2f}%，对科技股形成压力")
                us_impact -= 1
            else:
                analysis_summary.append(f"• QQQ(纳斯达克100)涨跌 {qqq_change:.2f}%，对科技股影响中性")
                
        if 'DIA' in overseas_data:
            dia_change = overseas_data['DIA']['change_pct']
            if dia_change > 1:
                analysis_summary.append(f"• DIA(道琼斯)上涨 {dia_change:.2f}%，反映市场情绪向好")
            elif dia_change < -1:
                analysis_summary.append(f"• DIA(道琼斯)下跌 {dia_change:.2f}%，反映市场情绪偏弱")
            else:
                analysis_summary.append(f"• DIA(道琼斯)涨跌 {dia_change:.2f}%，对市场情绪影响中性")
        
        # 分析A50期货 - 只有当有实际数据时才分析
        a50_impact = 0
        if 'A50_FUTURES' in overseas_data:
            a50_change = overseas_data['A50_FUTURES']['change_pct']
            # 如果A50期货数据是实际数据（非默认0值），才进行分析
            if a50_change != 0 or overseas_data['A50_FUTURES']['price'] != 0:
                if a50_change > 1:
                    analysis_summary.append(f"• A50期货上涨 {a50_change:.2f}%，预示A股情绪向好，利好港股")
                    a50_impact += 1
                elif a50_change < -1:
                    analysis_summary.append(f"• A50期货下跌 {a50_change:.2f}%，预示A股情绪偏弱，利空港股")
                    a50_impact -= 1
                else:
                    analysis_summary.append(f"• A50期货涨跌 {a50_change:.2f}%，对A股及港股影响中性")
            # 如果A50期货数据缺失，则不进行分析，也不在影响评估中考虑
        
        # 总体影响评估 - 如果A50期货数据缺失，则只考虑美股影响
        if 'A50_FUTURES' in overseas_data and (overseas_data['A50_FUTURES']['change_pct'] != 0 or overseas_data['A50_FUTURES']['price'] != 0):
            # A50期货有实际数据，计入总体影响
            total_impact = us_impact + a50_impact
        else:
            # A50期货数据缺失，只考虑美股影响
            total_impact = us_impact
        
        if total_impact > 1:
            analysis_summary.append("综合影响: 隔夜市场整体向好，对港股开盘形成支撑")
        elif total_impact < -1:
            analysis_summary.append("综合影响: 隔夜市场整体偏弱，对港股开盘形成压力")
        else:
            analysis_summary.append("综合影响: 隔夜市场影响中性，港股更多将跟随自身逻辑")
        
        analysis_summary.append("")
    
    # 添加风险评估
    analysis_summary.append("风险评估:")
    analysis_summary.append(f"整体风险等级: {risk_level}")
    if risk_factors:
        analysis_summary.append("主要风险因素:")
        for factor in risk_factors:
            analysis_summary.append(f"• {factor}")
    else:
        analysis_summary.append("主要风险因素: 隔夜市场波动正常，暂无显著风险因素")
    analysis_summary.append("")
    
    # 构建大模型提示
    prompt = f"""
请分析以下恒生指数(HSI)技术分析数据及隔夜海外市场表现，并提供明确的交易策略建议：

{chr(10).join(analysis_summary)}

请特别关注隔夜美股对港股的潜在影响，在策略建议中充分考虑外部市场因素，避免黑天鹅事件。如果A50期货数据存在，则同时考虑A50期货对港股的潜在影响；如果A50期货数据缺失，请主要基于美股及其他市场因素进行分析。

请根据以下原则提供具体的交易策略：
1. 基于趋势分析：如果指数处于上升趋势，考虑多头策略；如果处于下降趋势，考虑空头或谨慎策略
2. 基于技术指标：利用RSI、MACD、移动平均线等指标判断买卖时机
3. 基于市场状态：考虑当前市场是处于高位、中位还是低位
4. 基于隔夜市场影响：充分考虑美股对港股的带动或冲击作用（如果A50期货数据存在，也考虑其影响）
5. 风险管理：在建议中包含止损和风险控制策略，特别在隔夜市场大幅波动时加强风险控制
6. 资金管理：考虑适当的仓位管理原则

策略定义参考：
- 保守型：偏好低风险、稳定收益的投资策略，如高股息股票，注重资本保值
- 平衡型：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进取型：偏好高风险、高收益的投资策略，如科技成长股，追求资本增值

请在回复的第一行提供一个明确的标题，反映当前市场情况和推荐的交易策略，例如：
- 如果市场趋势向好："📈 恒生指数强势多头策略 - 推荐进取型投资者积极布局"
- 如果市场趋势偏弱："📉 恒生指数谨慎观望策略 - 推荐保守型投资者控制仓位"
- 如果市场震荡："📊 恒生指数震荡整理策略 - 推荐平衡型投资者灵活操作"

请务必在回复的第二行用一句话总结当天的交易策略（例如：当前建议采取保守型投资者策略，重点关注隔夜市场风险，谨慎操作。），然后继续提供具体的交易策略，包括：
- 当前市场观点
- 交易方向建议（做多/做空/观望）
- 明确推荐一个最适合当前市场状况的投资者类型（保守型/平衡型/进取型）
- 具体操作建议
- 风险控制措施（特别是基于隔夜市场情况的风险警示）
- 目标价位和止损位

请确保策略符合港股市场特点和恒生指数的特性，并特别关注隔夜市场波动对港股开盘的潜在影响。
"""
    
    if LLM_AVAILABLE:
        try:
            print("\n🤖 正在调用大模型分析恒生指数策略...")
            response = chat_with_llm(prompt)
            
            # 提取策略标题（第一行作为标题）
            lines = response.split('\n')
            title = lines[0].strip() if lines else "🤖 大模型恒生指数交易策略分析"
            # 移除可能的标题符号
            title = title.lstrip('# ').strip()
            
            print("\n" + "="*60)
            print(f"🤖 {title}")
            print("="*60)
            print(response)
            print("="*60)
            
            # 保存大模型输出到固定文件名
            filename = "hsi_strategy_latest.txt"
            filepath = os.path.join("data", filename)
            
            # 确保 data 目录存在
            os.makedirs("data", exist_ok=True)
            
            # 写入文件（新内容覆盖旧内容）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"恒生指数策略分析报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(response)
            
            print(f"💾 策略报告已保存到: {filepath}")
            
            # 返回策略内容和标题
            return {
                'content': response,
                'title': title
            }
        except Exception as e:
            print(f"❌ 调用大模型失败: {str(e)}")
            print("💡 请确保已设置 QWEN_API_KEY 环境变量")
            return None
    else:
        print("❌ 大模型服务不可用")
        return None

def main():
    """主函数"""
    print("📈 恒生指数(HSI)大模型策略分析器")
    print("="*50)
    
    # 获取隔夜美股及A50期货数据
    overseas_data = get_overseas_market_data()
    
    # 生成策略分析
    strategy_result = generate_hsi_llm_strategy(overseas_data)
    
    if strategy_result:
        print("\n✅ 恒生指数大模型策略分析完成！")
        
        # 发送邮件
        recipients = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@gmail.com")
        # 如果是字符串，分割成列表
        if isinstance(recipients, str):
            recipients = [email.strip() for email in recipients.split(',')]
        
        subject = f"📈 恒生指数策略分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # 创建HTML版本的内容
        html_content = f"""
        <h2>📈 恒生指数(HSI)大模型策略分析报告</h2>
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;">
            {markdown_to_html(strategy_result['content'])}
        </div>
        
        <!-- 隔夜美股及A50期货数据 -->
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #b3d9ff;">
            <h3>🌍 隔夜美股及A50期货数据</h3>
            <ul>
                <li>✅ 标普500 ETF(SPY): {overseas_data.get('SPY', {}).get('price', 'N/A')} ({overseas_data.get('SPY', {}).get('change_pct', 0):+.2f}%)</li>
                <li>✅ 纳斯达克100 ETF(QQQ): {overseas_data.get('QQQ', {}).get('price', 'N/A')} ({overseas_data.get('QQQ', {}).get('change_pct', 0):+.2f}%)</li>
                <li>✅ 道琼斯工业平均ETF(DIA): {overseas_data.get('DIA', {}).get('price', 'N/A')} ({overseas_data.get('DIA', {}).get('change_pct', 0):+.2f}%)</li>
                <li>✅ 20+年国债ETF(反映利率情绪)(TLT): {overseas_data.get('TLT', {}).get('price', 'N/A')} ({overseas_data.get('TLT', {}).get('change_pct', 0):+.2f}%)</li>
                <li>✅ 恐慌指数(VIX): {overseas_data.get('VIX', {}).get('price', 'N/A')} ({overseas_data.get('VIX', {}).get('change_pct', 0):+.2f}%)</li>
                <li>✅ A50期货: {overseas_data.get('A50_FUTURES', {}).get('price', 'N/A')} ({overseas_data.get('A50_FUTURES', {}).get('change_pct', 0):+.2f}%)</li>
            </ul>
        </div>
        
        <p><em>--- 此邮件由恒生指数大模型策略分析器自动生成</em></p>
        """
        
        # 纯文本版本的内容
        text_content = f"""恒生指数(HSI)大模型策略分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【隔夜市场摘要 - 重要风险提示】
请在港股开盘前务必检查隔夜美股及A50期货走势，避免黑天鹅事件
本报告已整合隔夜市场数据，建议重点关注潜在风险因素

{strategy_result['content']}

【隔夜美股及A50期货数据】
✅ 标普500 ETF(SPY): {overseas_data.get('SPY', {}).get('price', 'N/A')} ({overseas_data.get('SPY', {}).get('change_pct', 0):+.2f}%)
✅ 纳斯达克100 ETF(QQQ): {overseas_data.get('QQQ', {}).get('price', 'N/A')} ({overseas_data.get('QQQ', {}).get('change_pct', 0):+.2f}%)
✅ 道琼斯工业平均ETF(DIA): {overseas_data.get('DIA', {}).get('price', 'N/A')} ({overseas_data.get('DIA', {}).get('change_pct', 0):+.2f}%)
✅ 20+年国债ETF(反映利率情绪)(TLT): {overseas_data.get('TLT', {}).get('price', 'N/A')} ({overseas_data.get('TLT', {}).get('change_pct', 0):+.2f}%)
✅ 恐慌指数(VIX): {overseas_data.get('VIX', {}).get('price', 'N/A')} ({overseas_data.get('VIX', {}).get('change_pct', 0):+.2f}%)
✅ A50期货: {overseas_data.get('A50_FUTURES', {}).get('price', 'N/A')} ({overseas_data.get('A50_FUTURES', {}).get('change_pct', 0):+.2f}%)

---
此邮件由恒生指数大模型策略分析器自动生成
"""
        
        print("📧 正在发送邮件...")
        success = send_email(recipients, subject, text_content, html_content)
        if success:
            print("✅ 邮件发送成功！")
        else:
            print("❌ 邮件发送失败！")
    else:
        print("\n❌ 恒生指数大模型策略分析失败")

if __name__ == "__main__":
    main()
