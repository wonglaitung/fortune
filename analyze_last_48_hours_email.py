#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最近24小时连续交易信号邮件通知系统
基于分析结果生成买卖信号，只在有交易信号时发送邮件
"""

import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
import csv

def analyze_last_24_hours():
    # Read the CSV file
    with open('data/simulation_transactions.csv', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the CSV content
    lines = content.strip().split('\n')
    headers = lines[0].split(',')
    transactions = []
    
    for line in lines[1:]:
        fields = line.split(',')
        # Handle cases where fields might contain commas within quotes
        if len(fields) > len(headers):
            # Reconstruct fields to match headers
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
        
        if len(fields) >= 6:  # Ensure we have enough fields
            timestamp_str = fields[0]
            trans_type = fields[1]
            code = fields[2]
            name = fields[3] if len(fields) > 3 else ""
            shares_str = fields[4] if len(fields) > 4 else "0"
            price_str = fields[5] if len(fields) > 5 else "0"
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                transactions.append({
                    'timestamp': timestamp,
                    'date': timestamp.date(),
                    'type': trans_type,
                    'code': code,
                    'name': name,
                    'shares': int(float(shares_str)),
                    'price': float(price_str)
                })
            except ValueError as e:
                print(f"Error parsing line: {line[:100]}... Error: {e}")
    
    # Filter for the last 48 hours
    now = datetime.now()
    time_48_hours_ago = now - timedelta(hours=48)
    recent_transactions = [t for t in transactions if t['timestamp'] >= time_48_hours_ago]
    
    if not recent_transactions:
        # If no transactions in the last 48 hours, check for the most recent date in the data
        if transactions:
            latest_date = max(transactions, key=lambda x: x['timestamp'])['timestamp'].date()
            time_48_hours_ago = datetime.combine(latest_date, datetime.min.time()) - timedelta(hours=48)
            recent_transactions = [t for t in transactions if t['timestamp'] >= time_48_hours_ago]
    
    # Group transactions by stock code
    transactions_by_stock = defaultdict(lambda: {'BUY': [], 'SELL': []})
    for trans in recent_transactions:
        transactions_by_stock[trans['code']][trans['type']].append(trans)
    
    # Find stocks with 3 or more consecutive buy signals without intervening sells
    buy_without_sell_after = []
    sell_without_buy_after = []
    
    for stock_code, trans_dict in transactions_by_stock.items():
        buys = sorted(trans_dict['BUY'], key=lambda x: x['timestamp'])
        sells = sorted(trans_dict['SELL'], key=lambda x: x['timestamp'])
        
        # Check if there are 3 or more buys and no sells for this stock in the period
        if len(buys) >= 3 and len(sells) == 0:
            stock_name = buys[0]['name'] if buys else 'Unknown'
            buy_times = [buy['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for buy in buys]
            buy_without_sell_after.append((stock_code, stock_name, buy_times))
        elif len(sells) >= 3 and len(buys) == 0:
            stock_name = sells[0]['name'] if sells else 'Unknown'
            sell_times = [sell['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for sell in sells]
            sell_without_buy_after.append((stock_code, stock_name, sell_times))
    
    return buy_without_sell_after, sell_without_buy_after

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
    msg['From'] = f'"24小时连续交易信号监控" <{sender_email}>'
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
    print("🔍 正在分析最近48小时内的交易信号...")
    
    # 分析最近48小时内的交易信号
    buy_without_sell_after, sell_without_buy_after = analyze_last_24_hours()

    # 检查是否存在符合条件的信号
    has_signals = len(buy_without_sell_after) > 0 or len(sell_without_buy_after) > 0

    # 如果没有交易信号，则不发送邮件
    if not has_signals:
        print("⚠️ 没有检测到连续3次或以上的买入或卖出信号，跳过发送邮件。")
        exit(0)

    subject = "最近48小时连续交易信号提醒"

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
        <h2>📊 最近48小时连续交易信号提醒</h2>
        <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
                    <th>建议时间</th>
                </tr>
        """
        
        for code, name, times in buy_without_sell_after:
            times_str = "<br>".join(times)
            html += f"""
            <tr>
                <td>{code}</td>
                <td>{name}</td>
                <td>{times_str}</td>
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
                    <th>建议时间</th>
                </tr>
        """
        
        for code, name, times in sell_without_buy_after:
            times_str = "<br>".join(times)
            html += f"""
            <tr>
                <td>{code}</td>
                <td>{name}</td>
                <td>{times_str}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """

    # 在文本版本中也添加信息
    if buy_without_sell_after:
        text += f"📈 最近48小时内连续3次或以上建议买入同一只股票（期间没有卖出建议）:\n"
        for code, name, times in buy_without_sell_after:
            times_str = ", ".join(times)
            text += f"  {code} ({name}) - 建议时间: {times_str}\n"
        text += "\n"
    
    if sell_without_buy_after:
        text += f"📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）:\n"
        for code, name, times in sell_without_buy_after:
            times_str = ", ".join(times)
            text += f"  {code} ({name}) - 建议时间: {times_str}\n"
        text += "\n"

    # 添加说明
    text += "📋 说明:\n"
    text += "连续买入：指在最近48小时内，某只股票收到3次或以上买入建议，且期间没有收到任何卖出建议。\n"
    text += "连续卖出：指在最近48小时内，某只股票收到3次或以上卖出建议，且期间没有收到任何买入建议。\n"
    
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
