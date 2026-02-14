#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入大模型服务
from llm_services.qwen_engine import chat_with_llm


def extract_llm_recommendations(filepath):
    """
    从大模型建议文件中提取买卖建议
    
    参数:
    - filepath: 文件路径
    
    返回:
    - str: 提取的买卖建议文本
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取【短期建议】部分
        short_term_start = content.find("【短期建议】")
        if short_term_start != -1:
            short_term_end = content.find("\n\n", short_term_start + len("【短期建议】"))
            if short_term_end != -1:
                short_term_section = content[short_term_start:short_term_end]
            else:
                short_term_section = content[short_term_start:]
        else:
            short_term_section = ""
        
        # 提取【中期建议】部分
        medium_term_start = content.find("【中期建议】")
        if medium_term_start != -1:
            medium_term_end = content.find("\n\n", medium_term_start + len("【中期建议】"))
            if medium_term_end != -1:
                medium_term_section = content[medium_term_start:medium_term_end]
            else:
                medium_term_section = content[medium_term_start:]
        else:
            medium_term_section = ""
        
        # 提取买入和卖出建议
        buy_sell_recommendations = []
        
        # 从短期建议中提取
        if short_term_section:
            lines = short_term_section.split('\n')
            for line in lines:
                if '买入' in line or '加仓' in line or '卖出' in line or '减仓' in line or '清仓' in line:
                    buy_sell_recommendations.append(f"短期: {line.strip()}")
        
        # 从中期建议中提取
        if medium_term_section:
            lines = medium_term_section.split('\n')
            for line in lines:
                if '持有' in line or '加仓' in line or '减仓' in line or '清仓' in line or '买入' in line or '卖出' in line:
                    buy_sell_recommendations.append(f"中期: {line.strip()}")
        
        recommendations_text = "\n".join(buy_sell_recommendations)
        return recommendations_text if recommendations_text else "未找到买卖建议"
        
    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return ""


def extract_ml_predictions(filepath):
    """
    从ML预测文件中提取预测结果
    
    参数:
    - filepath: 文件路径
    
    返回:
    - str: 提取的预测结果文本
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取预测结果部分
        predictions_start = content.find("【预测结果】")
        statistics_start = content.find("【统计信息】")
        
        if predictions_start != -1:
            if statistics_start != -1:
                predictions_section = content[predictions_start:statistics_start]
            else:
                predictions_section = content[predictions_start:]
        else:
            predictions_section = ""
        
        # 提取预测上涨的股票
        up_stocks = []
        lines = predictions_section.split('\n')
        for line in lines:
            if '上涨' in line and not line.startswith('股票代码') and not line.startswith('-'):
                up_stocks.append(line.strip())
        
        predictions_text = "\n".join(up_stocks[:10])  # 只取前10个
        return predictions_text if predictions_text else "未找到预测上涨的股票"
        
    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return ""


def send_email(subject, content, html_content=None):
    """
    发送邮件通知
    
    参数:
    - subject: 邮件主题
    - content: 邮件文本内容
    - html_content: 邮件HTML内容（可选）
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # 从环境变量获取邮件配置
        sender_email = os.environ.get("YAHOO_EMAIL")
        email_password = os.environ.get("YAHOO_APP_PASSWORD")
        smtp_server = os.environ.get("YAHOO_SMTP", "smtp.163.com")
        recipient_email = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        
        if ',' in recipient_email:
            recipients = [recipient.strip() for recipient in recipient_email.split(',')]
        else:
            recipients = [recipient_email]
        
        if not sender_email or not email_password:
            print("❌ 邮件配置不完整，跳过邮件发送")
            return False
        
        # 根据SMTP服务器类型选择端口和SSL
        if "163.com" in smtp_server:
            smtp_port = 465
            use_ssl = True
        elif "gmail.com" in smtp_server:
            smtp_port = 587
            use_ssl = False
        else:
            smtp_port = 587
            use_ssl = False
        
        # 创建邮件对象
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)
        
        # 添加文本版本
        text_part = MIMEText(content, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # 如果有HTML版本，添加HTML版本
        if html_content:
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
        
        # 重试机制（3次）
        for attempt in range(3):
            try:
                if use_ssl:
                    server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                    server.login(sender_email, email_password)
                    server.sendmail(sender_email, recipients, msg.as_string())
                    server.quit()
                else:
                    server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                    server.starttls()
                    server.login(sender_email, email_password)
                    server.sendmail(sender_email, recipients, msg.as_string())
                    server.quit()
                
                print(f"✅ 邮件已发送到: {', '.join(recipients)}")
                return True
            except Exception as e:
                print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                if attempt < 2:
                    import time
                    time.sleep(5)
        
        print("❌ 3次尝试后仍无法发送邮件")
        return False
        
    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_analysis(llm_filepath, ml_filepath, output_filepath=None, send_email_flag=True):
    """
    运行综合分析
    
    参数:
    - llm_filepath: 大模型建议文件路径
    - ml_filepath: ML预测结果文件路径
    - output_filepath: 输出文件路径（可选）
    """
    try:
        print("=" * 80)
        print("🤖 综合分析开始")
        print("=" * 80)
        
        # 检查文件是否存在
        if not os.path.exists(llm_filepath):
            print(f"❌ 大模型建议文件不存在: {llm_filepath}")
            return None
        
        if not os.path.exists(ml_filepath):
            print(f"❌ ML预测结果文件不存在: {ml_filepath}")
            return None
        
        print(f"📊 大模型建议文件: {llm_filepath}")
        print(f"📊 ML预测结果文件: {ml_filepath}")
        print("")
        
        # 提取大模型建议
        print("📝 提取大模型买卖建议...")
        llm_recommendations = extract_llm_recommendations(llm_filepath)
        print(f"✅ 提取完成\n")
        
        # 提取ML预测
        print("📝 提取ML预测结果...")
        ml_predictions = extract_ml_predictions(ml_filepath)
        print(f"✅ 提取完成\n")
        
        # 生成日期
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 构建综合分析提示词
        prompt = f"""你是一位专业的投资分析师。请根据以下三部分信息，进行综合分析，给出实质的买卖建议。

=== 信息来源 ===

【1. 大模型短期和中期买卖建议】
{llm_recommendations}

【2. 机器学习20天预测结果】
{ml_predictions}

【3. 综合分析任务】
请基于上述信息，完成以下任务：

1. **一致性分析**：
   - 分析大模型短期建议与ML 20天预测的一致性
   - 如果两者都建议买入/上涨，标注为"强买入信号"
   - 如果两者建议相反，分析哪个更可信
   - 优先推荐两者一致的股票

2. **个股建议排序**：
   - 按照"强买入信号 > 中等信号 > 观望"的优先级排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出
   - 简要说明理由（不超过20字）

3. **综合推荐清单**：
   - 推荐买入的股票清单（按优先级排序）
   - 推荐卖出的股票清单（如有）
   - 需要关注的股票清单（观望）

4. **风险提示**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比）
   - 给出止损位建议（如果有的话）

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] - 原因（20字内）

## 买入信号（3-5只）
1. [股票代码] [股票名称] - 原因（20字内）

## 持有/观望
1. [股票代码] [股票名称] - 理由（20字内）

## 卖出信号（如有）
1. [股票代码] [股票名称] - 理由（20字内）

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 建议仓位百分比：[X]%
- 止损位设置：[策略]

---
分析日期：{date_str}
"""
        
        print("🤖 提交大模型进行综合分析...")
        print("")
        
        # 调用大模型
        response = chat_with_llm(prompt)
        
        if response:
            print("✅ 综合分析完成\n")
            print("=" * 80)
            print("📊 综合买卖建议")
            print("=" * 80)
            print("")
            print(response)
            print("")
            print("=" * 80)
            
            # 保存到文件
            if output_filepath is None:
                output_filepath = f'data/comprehensive_recommendations_{date_str}.txt'
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(f"{'=' * 80}\n")
                f.write(f"综合买卖建议\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析日期: {date_str}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(response)
            
            print(f"✅ 综合建议已保存到 {output_filepath}")
            
            # 发送邮件通知
            if send_email_flag:
                print("\n📧 准备发送邮件通知...")
                email_subject = f"【综合分析】港股买卖建议 - {date_str}"
                email_content = response
                send_email(email_subject, email_content)
            
            return response
        else:
            print("❌ 大模型分析失败")
            return None
        
    except Exception as e:
        print(f"❌ 综合分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='综合分析脚本 - 整合大模型建议和ML预测结果')
    parser.add_argument('--llm-file', type=str, default=None, 
                       help='大模型建议文件路径 (默认使用今天的文件)')
    parser.add_argument('--ml-file', type=str, default=None,
                       help='ML预测结果文件路径 (默认使用今天的文件)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认保存到data/comprehensive_recommendations_YYYY-MM-DD.txt)')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件通知')
    
    args = parser.parse_args()
    
    # 生成日期
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # 默认文件路径
    if args.llm_file is None:
        args.llm_file = f'data/llm_recommendations_{date_str}.txt'
    
    if args.ml_file is None:
        args.ml_file = f'data/ml_predictions_20d_{date_str}.txt'
    
    # 运行综合分析
    result = run_comprehensive_analysis(args.llm_file, args.ml_file, args.output, 
                                       send_email_flag=not args.no_email)
    
    if result:
        print("\n✅ 综合分析完成！")
        sys.exit(0)
    else:
        print("\n❌ 综合分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()