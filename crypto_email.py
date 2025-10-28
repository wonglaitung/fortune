import os
import requests
import smtplib
import json
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# 导入大模型服务
try:
    from llm_services import qwen_engine
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ 大模型服务不可用，将跳过AI分析功能")

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
    # 这里简化处理，实际应用中可以获取历史价格数据进行更详细的分析
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

def build_llm_analysis_prompt(prices, indicators):
    """
    构建发送给大模型的加密货币数据分析提示词
    
    Args:
        prices (dict): 加密货币价格数据
        indicators (dict): 技术指标数据
        
    Returns:
        str: 构建好的提示词
    """
    # 构建加密货币数据表格
    table_header = "| 加密货币 | 价格(USD) | 24小时变化(%) | RSI | MACD | 价格位置(%) |\n"
    table_separator = "|----------|-----------|----------------|-----|------|--------------|\n"
    
    table_rows = []
    
    # Bitcoin数据
    if 'bitcoin' in prices:
        btc = prices['bitcoin']
        btc_ind = indicators['bitcoin']
        row = f"| Bitcoin | ${btc.get('usd', 0):,.2f} | {btc.get('usd_24h_change', 0):.2f} | {btc_ind.get('rsi', 0):.2f} | {btc_ind.get('macd', 0):.2f} | {btc_ind.get('price_position', 0):.2f} |"
        table_rows.append(row)
    
    # Ethereum数据
    if 'ethereum' in prices:
        eth = prices['ethereum']
        eth_ind = indicators['ethereum']
        row = f"| Ethereum | ${eth.get('usd', 0):,.2f} | {eth.get('usd_24h_change', 0):.2f} | {eth_ind.get('rsi', 0):.2f} | {eth_ind.get('macd', 0):.2f} | {eth_ind.get('price_position', 0):.2f} |"
        table_rows.append(row)
    
    crypto_table = table_header + table_separator + "\n".join(table_rows)
    
    # 构建提示词
    prompt = f"""
你是一个专业的加密货币分析师，请根据以下加密货币市场数据，分析并提供投资建议：

{crypto_table}

请从以下几个维度分析：
1. Bitcoin和Ethereum的当前市场表现对比
2. 技术指标分析（RSI、MACD等）
3. 市场趋势判断
4. 投资建议

请给出你的分析结论，包括：
1. 最值得关注的加密货币及其理由
2. 需要警惕的风险点
3. 短期和中长期市场趋势预测

请用中文回答，格式清晰易读。
"""
    
    return prompt

def analyze_with_llm(prices, indicators):
    """
    使用大模型分析加密货币数据
    
    Args:
        prices (dict): 加密货币价格数据
        indicators (dict): 技术指标数据
        
    Returns:
        str: 大模型分析结果
    """
    if not LLM_AVAILABLE:
        return "大模型服务不可用"
    
    try:
        prompt = build_llm_analysis_prompt(prices, indicators)
        analysis = qwen_engine.chat_with_llm(prompt)
        return analysis
    except Exception as e:
        print(f"⚠️ 大模型分析失败: {e}")
        return "大模型分析暂不可用"

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
    msg['From'] = f'"wonglaitung" <{sender_email}>'
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
    
    # 使用大模型分析数据
    llm_analysis = analyze_with_llm(prices, indicators)

    subject = "Ethereum and Bitcoin Price Update"

    text = ""
    html = "<html><body>"
    html += "<h2>加密货币价格更新</h2>"
    html += f"<p><strong>报告时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"

    # Ethereum (放在前面)
    if 'ethereum' in prices:
        eth = prices['ethereum']
        eth_usd = eth['usd']
        eth_hkd = eth['hkd']
        eth_change = eth.get('usd_24h_change', 0.0)
        eth_market_cap = eth.get('usd_market_cap', 0.0) if 'usd_market_cap' in eth else 0.0
        eth_24hr_vol = eth.get('usd_24h_vol', 0.0) if 'usd_24h_vol' in eth else 0.0
        eth_rsi = indicators['ethereum'].get('rsi', 0.0)
        eth_macd = indicators['ethereum'].get('macd', 0.0)
        
        text += f"Ethereum price: ${eth_usd:,.2f} USD ({eth_change:.2f}% 24h)\n"
        text += f"Ethereum price: ${eth_hkd:,.2f} HKD\n"
        if eth_market_cap > 0:
            text += f"Market Cap: ${eth_market_cap:,.2f} USD\n"
        if eth_24hr_vol > 0:
            text += f"24h Volume: ${eth_24hr_vol:,.2f} USD\n"
        text += f"RSI: {eth_rsi:.2f}\n"
        text += f"MACD: {eth_macd:.2f}\n"
        text += "\n"
        
        html += f"<p><strong>Ethereum</strong><br>"
        html += f"Price: ${eth_usd:,.2f} USD ({eth_change:.2f}% 24h)<br>"
        html += f"Price: ${eth_hkd:,.2f} HKD<br>"
        if eth_market_cap > 0:
            html += f"Market Cap: ${eth_market_cap:,.2f} USD<br>"
        if eth_24hr_vol > 0:
            html += f"24h Volume: ${eth_24hr_vol:,.2f} USD<br>"
        html += f"RSI: {eth_rsi:.2f}<br>"
        html += f"MACD: {eth_macd:.2f}<br>"
        html += "</p>"

    # Bitcoin
    if 'bitcoin' in prices:
        btc = prices['bitcoin']
        btc_usd = btc['usd']
        btc_hkd = btc['hkd']
        btc_change = btc.get('usd_24h_change', 0.0)
        btc_market_cap = btc.get('usd_market_cap', 0.0) if 'usd_market_cap' in btc else 0.0
        btc_24hr_vol = btc.get('usd_24h_vol', 0.0) if 'usd_24h_vol' in btc else 0.0
        btc_rsi = indicators['bitcoin'].get('rsi', 0.0)
        btc_macd = indicators['bitcoin'].get('macd', 0.0)
        
        text += f"Bitcoin price: ${btc_usd:,.2f} USD ({btc_change:.2f}% 24h)\n"
        text += f"Bitcoin price: ${btc_hkd:,.2f} HKD\n"
        if btc_market_cap > 0:
            text += f"Market Cap: ${btc_market_cap:,.2f} USD\n"
        if btc_24hr_vol > 0:
            text += f"24h Volume: ${btc_24hr_vol:,.2f} USD\n"
        text += f"RSI: {btc_rsi:.2f}\n"
        text += f"MACD: {btc_macd:.2f}\n"
        text += "\n"
        
        html += f"<p><strong>Bitcoin</strong><br>"
        html += f"Price: ${btc_usd:,.2f} USD ({btc_change:.2f}% 24h)<br>"
        html += f"Price: ${btc_hkd:,.2f} HKD<br>"
        if btc_market_cap > 0:
            html += f"Market Cap: ${btc_market_cap:,.2f} USD<br>"
        if btc_24hr_vol > 0:
            html += f"24h Volume: ${btc_24hr_vol:,.2f} USD<br>"
        html += f"RSI: {btc_rsi:.2f}<br>"
        html += f"MACD: {btc_macd:.2f}<br>"
        html += "</p>"

    # 添加大模型分析结果
    html += "<h3>🤖 大模型分析结果：</h3>"
    html += "<div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px;'>"
    if llm_analysis:
        # 将大模型分析结果中的换行符转换为HTML换行标签
        llm_analysis_html = llm_analysis.replace('\n', '<br>')
        html += f"<p>{llm_analysis_html}</p>"
    else:
        html += "<p>大模型分析暂不可用</p>"
    html += "</div>"

    # 添加指标说明
    html += """
    <h3>📋 指标说明</h3>
    <div style="font-size:0.9em; line-height:1.4;">
    <ul>
      <li><b>价格(USD/HKD)</b>：加密货币的当前价格，分别以美元和港币计价。</li>
      <li><b>24小时变化(%)</b>：过去24小时内价格的变化百分比。</li>
      <li><b>市值(Market Cap)</b>：加密货币的总市值，以美元计价。</li>
      <li><b>24小时交易量</b>：过去24小时内该加密货币的交易总额，以美元计价。</li>
      <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
      <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
      <li><b>价格位置(%)</b>：当前价格在近期价格区间的相对位置。</li>
    </ul>
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

    print("📧 Sending email to:", ", ".join(recipients))
    print("📝 Subject:", subject)
    print("📄 Text preview:\n", text)

    success = send_email(recipients, subject, text, html)
    if not success:
        exit(1)
