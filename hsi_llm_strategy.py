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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    msg['From'] = f'"wonglaitung" <{sender_email}>'
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
from tencent_finance import get_hsi_data_tencent

# 导入技术分析工具
from technical_analysis import TechnicalAnalyzer

# 导入大模型服务
try:
    from llm_services.qwen_engine import chat_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("警告: 无法导入大模型服务，将跳过大模型分析功能")

warnings.filterwarnings('ignore')

def generate_hsi_llm_strategy():
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
    
    # 构建大模型提示
    prompt = f"""
请分析以下恒生指数(HSI)技术分析数据，并提供明确的交易策略建议：

{chr(10).join(analysis_summary)}

请根据以下原则提供具体的交易策略：
1. 基于趋势分析：如果指数处于上升趋势，考虑多头策略；如果处于下降趋势，考虑空头或谨慎策略
2. 基于技术指标：利用RSI、MACD、移动平均线等指标判断买卖时机
3. 基于市场状态：考虑当前市场是处于高位、中位还是低位
4. 风险管理：在建议中包含止损和风险控制策略
5. 资金管理：考虑适当的仓位管理原则

策略定义参考：
- 保守型：偏好低风险、稳定收益的投资策略，如高股息股票，注重资本保值
- 平衡型：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进取型：偏好高风险、高收益的投资策略，如科技成长股，追求资本增值

请在报告的开头提供一个明确的标题，反映当前市场情况和推荐的交易策略，例如：
- 如果市场趋势向好："📈 恒生指数强势多头策略 - 推荐进取型投资者积极布局"
- 如果市场趋势偏弱："📉 恒生指数谨慎观望策略 - 推荐保守型投资者控制仓位"
- 如果市场震荡："📊 恒生指数震荡整理策略 - 推荐平衡型投资者灵活操作"

然后提供具体的交易策略，包括：
- 当前市场观点
- 交易方向建议（做多/做空/观望）
- 明确推荐一个最适合当前市场状况的投资者类型（保守型/平衡型/进取型）
- 具体操作建议
- 风险控制措施
- 目标价位和止损位

请确保策略符合港股市场特点和恒生指数的特性。
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
    
    # 生成策略分析
    strategy_result = generate_hsi_llm_strategy()
    
    if strategy_result:
        print("\n✅ 恒生指数大模型策略分析完成！")
        
        # 发送邮件
        recipients = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@gmail.com")
        # 如果是字符串，分割成列表
        if isinstance(recipients, str):
            recipients = [email.strip() for email in recipients.split(',')]
        
        subject = f"📈 恒生指数策略分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        content = f"""恒生指数(HSI)大模型策略分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{strategy_result['content']}

---
此邮件由恒生指数大模型策略分析器自动生成
"""
        
        print("📧 正在发送邮件...")
        success = send_email(recipients, subject, content)
        if success:
            print("✅ 邮件发送成功！")
        else:
            print("❌ 邮件发送失败！")
    else:
        print("\n❌ 恒生指数大模型策略分析失败")

if __name__ == "__main__":
    main()
