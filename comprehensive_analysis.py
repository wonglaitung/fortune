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
        
        # 提取买入、卖出、观察部分的股票
        recommendations = []
        
        # 查找所有买入、卖出、观察股票
        lines = content.split('\n')
        current_section = None
        capture_next = False
        
        for i, line in enumerate(lines):
            # 检测章节标记
            if '买入机会推荐' in line:
                current_section = '买入'
                continue
            elif '卖出机会推荐' in line:
                current_section = '卖出'
                continue
            elif '观察列表' in line or '观望建议' in line:
                current_section = '观察'
                continue
            
            # 捕获股票代码行（格式：X. 股票名称 (股票代码)）
            if current_section and line.strip() and line[0].isdigit() and '.' in line and '(' in line and ')' in line:
                stock_info = line.strip()
                # 提取股票代码
                code_start = stock_info.find('(') + 1
                code_end = stock_info.find(')')
                if code_start > 0 and code_end > code_start:
                    code = stock_info[code_start:code_end]
                    # 提取股票名称
                    name_end = stock_info.find('(')
                    name = stock_info[stock_info.find(' ') + 1:name_end].strip()
                    
                    # 获取推荐理由和操作建议（下一行或几行）
                    reason_parts = []
                    operation_advice = ""
                    price_guide = ""
                    risk_hint = ""
                    
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        
                        # 提取推荐理由
                        if next_line.startswith('- 推荐理由：'):
                            reason_text = next_line.replace('- 推荐理由：', '').strip()
                            reason_parts.append(f"推荐理由: {reason_text}")
                        elif next_line.startswith('- ') and '推荐理由' in next_line:
                            reason_text = next_line.replace('- 推荐理由：', '').strip()
                            reason_parts.append(f"推荐理由: {reason_text}")
                        
                        # 提取操作建议
                        elif next_line.startswith('- 操作建议：'):
                            operation_advice = next_line.replace('- 操作建议：', '').strip()
                        
                        # 提取价格指引
                        elif next_line.startswith('- 价格指引：'):
                            # 收集后续的价格指引行
                            price_guide_items = []
                            k = j + 1
                            while k < len(lines):
                                price_line = lines[k].strip()
                                if price_line.startswith('* '):
                                    # 去掉前导符号和多余空格
                                    price_item = price_line.replace('* ', '').strip()
                                    # 移除可能的缩进空格
                                    price_item = price_item.lstrip('· ')
                                    price_guide_items.append(price_item)
                                elif price_line.startswith('- ') or (price_line and not price_line.startswith('-') and not price_line.startswith('•')):
                                    # 遇到其他章节，停止收集
                                    break
                                else:
                                    break
                                k += 1
                            price_guide = " | ".join(price_guide_items)
                        
                        # 提取风险提示
                        elif next_line.startswith('- 风险提示：'):
                            risk_hint = next_line.replace('- 风险提示：', '').strip()
                        
                        # 遇到新股票或新章节，停止收集
                        elif next_line and next_line[0].isdigit() and '.' in next_line and '(' in next_line and ')' in next_line:
                            break
                        
                        j += 1
                    
                    # 去掉Markdown格式
                    reason_text = " ".join(reason_parts).replace('*', '').replace('**', '')
                    price_guide = price_guide.replace('*', '').replace('**', '')
                    operation_advice = operation_advice.replace('*', '').replace('**', '')
                    risk_hint = risk_hint.replace('*', '').replace('**', '')
                    
                    # 组合所有信息
                    full_info = f"{reason_text}"
                    if operation_advice:
                        full_info += f" | 操作建议: {operation_advice}"
                    if price_guide:
                        full_info += f" | 价格: {price_guide}"
                    if risk_hint:
                        full_info += f" | 风险: {risk_hint}"
                    
                    recommendations.append(f"{current_section}: {code} {name} - {full_info}")
        
        recommendations_text = "\n".join(recommendations) if recommendations else "未找到买卖建议"
        return recommendations_text
        
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


def generate_html_email(content, date_str):
    """
    生成HTML格式的邮件内容
    
    参数:
    - content: 综合分析文本内容
    - date_str: 分析日期
    
    返回:
    - str: HTML格式的邮件内容
    """
    # 将Markdown标题转换为HTML标题
    lines = content.split('\n')
    html_lines = []
    
    for line in lines:
        line = line.strip()
        
        # 转换主标题
        if line.startswith('# '):
            html_lines.append(f'<h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px;">{line[2:]}</h1>')
        
        # 转换二级标题
        elif line.startswith('## '):
            title = line[3:]
            color = '#e74c3c' if '卖出' in title else '#27ae60' if '买入' in title else '#f39c12' if '持有' in title else '#3498db'
            icon = '🔴' if '卖出' in title else '🟢' if '买入' in title else '🟡' if '持有' in title else '📊'
            html_lines.append(f'<h2 style="color: {color}; border-left: 4px solid {color}; padding-left: 15px; margin-top: 30px; margin-bottom: 15px;">{icon} {title}</h2>')
        
        # 转换三级标题（风险控制建议）
        elif line.startswith('### '):
            html_lines.append(f'<h3 style="color: #8e44ad; margin-top: 25px; margin-bottom: 10px;">📌 {line[4:]}</h3>')
        
        # 转换列表项（股票推荐）
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. ') or line.startswith('4. ') or line.startswith('5. '):
            # 提取股票代码和名称
            match = line.split(']')
            if len(match) > 1:
                stock_code_name = match[0].replace('[', '').replace(']', '')
                html_lines.append(f'<div style="background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 5px;">')
                html_lines.append(f'<h4 style="color: #2c3e50; margin: 0 0 10px 0;">📈 {stock_code_name}</h4>')
            else:
                html_lines.append(f'<div style="padding: 10px;">')
        
        # 转换子项（推荐理由、操作建议等）
        elif line.startswith('   - '):
            item_text = line.replace('   - ', '')
            
            # 识别字段类型并添加图标
            if '推荐理由' in item_text:
                html_lines.append(f'<p style="color: #34495e; margin: 8px 0;"><strong>💡 {item_text}</strong></p>')
            elif '操作建议' in item_text:
                html_lines.append(f'<p style="color: #27ae60; margin: 8px 0;"><strong>⚡ {item_text}</strong></p>')
            elif '建议仓位' in item_text:
                html_lines.append(f'<p style="color: #e67e22; margin: 8px 0;"><strong>📊 {item_text}</strong></p>')
            elif '价格指引' in item_text:
                html_lines.append(f'<p style="color: #2980b9; margin: 8px 0;"><strong>💰 {item_text}</strong></p>')
            elif '建议买入价' in item_text or '止损位' in item_text or '目标价' in item_text:
                html_lines.append(f'<p style="color: #16a085; margin: 5px 0 5px 20px;">&nbsp;&nbsp;&nbsp;&nbsp;• {item_text}</p>')
            elif '操作时机' in item_text:
                html_lines.append(f'<p style="color: #8e44ad; margin: 8px 0;"><strong>⏰ {item_text}</strong></p>')
            elif '风险提示' in item_text:
                html_lines.append(f'<p style="color: #c0392b; margin: 8px 0;"><strong>⚠️ {item_text}</strong></p>')
            elif '关注要点' in item_text:
                html_lines.append(f'<p style="color: #7f8c8d; margin: 8px 0;"><strong>👀 {item_text}</strong></p>')
            elif '建议卖出价' in item_text:
                html_lines.append(f'<p style="color: #c0392b; margin: 8px 0;"><strong>💵 {item_text}</strong></p>')
            else:
                html_lines.append(f'<p style="color: #34495e; margin: 8px 0;">{item_text}</p>')
        
        # 转换分隔线
        elif line.startswith('---'):
            html_lines.append('<hr style="border: none; border-top: 2px solid #ecf0f1; margin: 20px 0;">')
        
        # 转换日期
        elif line.startswith('分析日期：'):
            html_lines.append(f'<p style="color: #7f8c8d; font-size: 14px; margin-top: 30px; text-align: right;">📅 {line}</p>')
        
        # 转换段落
        elif line and not line.startswith('#') and not line.startswith('-'):
            if not line.startswith('   '):  # 不是子项
                html_lines.append(f'<p style="color: #34495e; line-height: 1.6; margin: 10px 0;">{line}</p>')
    
    # 组装完整的HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        h2 {{
            color: #3498db;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 35px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #8e44ad;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        h4 {{
            color: #2c3e50;
            margin: 0 0 12px 0;
            font-size: 18px;
        }}
        p {{
            color: #34495e;
            line-height: 1.8;
            margin: 10px 0;
        }}
        .stock-card {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 18px;
            margin: 18px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        .stock-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .highlight-buy {{
            border-left-color: #27ae60;
        }}
        .highlight-sell {{
            border-left-color: #e74c3c;
        }}
        .highlight-hold {{
            border-left-color: #f39c12;
        }}
    </style>
</head>
<body>
    <div class="container">
        {''.join(html_lines)}
        <div class="footer">
            <p>📧 本邮件由港股综合分析系统自动生成</p>
            <p>⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


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
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由，包含技术面、基本面、资金面等分析]
   - 操作建议：买入/卖出/持有/观望
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 买入信号（3-5只）
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由]
   - 操作建议：买入/持有
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 持有/观望
1. [股票代码] [股票名称] 
   - 推荐理由：[观望理由]
   - 操作建议：持有/观望
   - 关注要点：[需要关注的关键指标或事件]
   - 风险提示：[主要风险因素]

## 卖出信号（如有）
1. [股票代码] [股票名称] 
   - 推荐理由：[卖出理由]
   - 操作建议：卖出/减仓
   - 建议卖出价：HK$XX.XX
   - 止损位（如持有）：HK$XX.XX（-X.X%）
   - 风险提示：[主要风险因素]

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 建议仓位百分比：[X]%
- 止损位设置：[策略]
- 组合调整建议：[具体的组合调整建议]

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
                
                # 生成HTML格式邮件内容
                html_content = generate_html_email(response, date_str)
                send_email(email_subject, email_content, html_content)
            
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