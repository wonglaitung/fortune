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
    从大模型建议文件中提取买卖建议，分别提取短期和中期建议
    
    参数:
    - filepath: 文件路径
    
    返回:
    - dict: 包含短期和中期建议的字典
      {
        'short_term': str,  # 短期建议文本
        'medium_term': str  # 中期建议文本
      }
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        
        # 使用更精确的正则表达式提取短期建议
        # 匹配"### 稳健型短期分析"标题后到下一个"###"标题之前的内容
        short_term_match = re.search(
            r'^###.*稳健型短期分析.*?\n(.*?)(?=^###|\Z)',
            content,
            re.DOTALL | re.MULTILINE
        )
        
        # 使用更精确的正则表达式提取中期建议
        # 匹配"### 稳健型中期分析"标题后到文件末尾或下一个"###"标题之前的内容
        medium_term_match = re.search(
            r'^###.*稳健型中期分析.*?\n(.*?)(?=\Z|^###)',
            content,
            re.DOTALL | re.MULTILINE
        )
        
        result = {
            'short_term': short_term_match.group(1).strip() if short_term_match else '',
            'medium_term': medium_term_match.group(1).strip() if medium_term_match else ''
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return {'short_term': '', 'medium_term': ''}


def extract_ml_predictions(filepath):
    """
    从ML预测CSV文件中提取LightGBM和GBDT+LR的预测结果
    
    参数:
    - filepath: 文本预测文件路径（用于获取日期）
    
    返回:
    - dict: 包含LightGBM和GBDT+LR预测结果的字典
      {
        'lgbm': str,      # LightGBM预测结果
        'gbdt_lr': str   # GBDT+LR预测结果
      }
    """
    try:
        import pandas as pd
        from datetime import datetime
        import os
        
        # 从文件路径中提取日期
        date_str = filepath.split('_')[-1].replace('.txt', '')
        
        # 使用相对路径（从当前脚本位置推导data目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        lgbm_csv = os.path.join(data_dir, 'ml_trading_model_lgbm_predictions_20d.csv')
        gbdt_lr_csv = os.path.join(data_dir, 'ml_trading_model_gbdt_lr_predictions_20d.csv')
        
        result = {
            'lgbm': '',
            'gbdt_lr': ''
        }
        
        # 读取LightGBM预测结果
        if os.path.exists(lgbm_csv):
            df_lgbm = pd.read_csv(lgbm_csv)
            # 提取预测上涨的股票
            up_stocks_lgbm = df_lgbm[df_lgbm['prediction'] == 1].sort_values('probability', ascending=False)
            
            lgbm_text = "【LightGBM模型预测结果】\n"
            lgbm_text += f"预测日期: {date_str}\n\n"
            lgbm_text += "预测上涨的股票（按概率排序）:\n"
            lgbm_text += "-" * 80 + "\n"
            lgbm_text += f"{'股票代码':<12} {'股票名称':<12} {'上涨概率':<10} {'当前价格':<12}\n"
            lgbm_text += "-" * 80 + "\n"
            
            for _, row in up_stocks_lgbm.iterrows():
                lgbm_text += f"{row['code']:<12} {row['name']:<12} {row['probability']:<10.4f} {row['current_price']:<12}\n"
            
            lgbm_text += "-" * 80 + "\n"
            lgbm_text += f"预测上涨: {len(up_stocks_lgbm)} 只\n"
            lgbm_text += f"预测下跌: {len(df_lgbm) - len(up_stocks_lgbm)} 只\n"
            lgbm_text += f"平均上涨概率: {up_stocks_lgbm['probability'].mean():.4f}\n"
            
            result['lgbm'] = lgbm_text
        
        # 读取GBDT+LR预测结果
        if os.path.exists(gbdt_lr_csv):
            df_gbdt_lr = pd.read_csv(gbdt_lr_csv)
            # 提取预测上涨的股票
            up_stocks_gbdt_lr = df_gbdt_lr[df_gbdt_lr['prediction'] == 1].sort_values('probability', ascending=False)
            
            gbdt_lr_text = "【GBDT+LR模型预测结果】\n"
            gbdt_lr_text += f"预测日期: {date_str}\n\n"
            gbdt_lr_text += "预测上涨的股票（按概率排序）:\n"
            gbdt_lr_text += "-" * 80 + "\n"
            gbdt_lr_text += f"{'股票代码':<12} {'股票名称':<12} {'上涨概率':<10} {'当前价格':<12}\n"
            gbdt_lr_text += "-" * 80 + "\n"
            
            for _, row in up_stocks_gbdt_lr.iterrows():
                gbdt_lr_text += f"{row['code']:<12} {row['name']:<12} {row['probability']:<10.4f} {row['current_price']:<12}\n"
            
            gbdt_lr_text += "-" * 80 + "\n"
            gbdt_lr_text += f"预测上涨: {len(up_stocks_gbdt_lr)} 只\n"
            gbdt_lr_text += f"预测下跌: {len(df_gbdt_lr) - len(up_stocks_gbdt_lr)} 只\n"
            gbdt_lr_text += f"平均上涨概率: {up_stocks_gbdt_lr['probability'].mean():.4f}\n"
            
            result['gbdt_lr'] = gbdt_lr_text
        
        return result
        
    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'lgbm': '', 'gbdt_lr': ''}


def generate_html_email(content, date_str):
    """
    生成HTML格式的邮件内容
    
    参数:
    - content: 综合分析文本内容（Markdown格式）
    - date_str: 分析日期
    
    返回:
    - str: HTML格式的邮件内容
    """
    try:
        import markdown
    except ImportError:
        print("⚠️ 警告：未安装markdown库，使用简单转换")
        # 如果没有安装markdown库，使用简单转换
        simple_html = content.replace('\n', '<br>')
        return simple_html
    
    # 配置markdown扩展，使用更多功能以支持嵌套列表
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
    
    # 将Markdown转换为HTML
    html_content = md.convert(content)
    
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
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            background-color: #ffffff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 28px;
        }}
        h2 {{
            color: #3498db;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 35px;
            margin-bottom: 20px;
            font-size: 22px;
        }}
        h3 {{
            color: #8e44ad;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 20px;
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
        ul, ol {{
            color: #34495e;
            line-height: 1.8;
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .reference-section {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 4px solid #95a5a6;
        }}
        .reference-title {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .reference-content {{
            background: #ffffff;
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 港股综合买卖建议</h1>
        <p style="color: #7f8c8d; font-size: 14px;">📅 分析日期：{date_str}</p>
        
        <div class="content">
            {html_content}
        </div>
        
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
        print(f"   - 短期建议长度: {len(llm_recommendations['short_term'])} 字符")
        print(f"   - 中期建议长度: {len(llm_recommendations['medium_term'])} 字符\n")
        
        # 提取ML预测
        print("📝 提取ML预测结果...")
        ml_predictions = extract_ml_predictions(ml_filepath)
        print(f"✅ 提取完成\n")
        print(f"   - LightGBM预测长度: {len(ml_predictions['lgbm'])} 字符")
        print(f"   - GBDT+LR预测长度: {len(ml_predictions['gbdt_lr'])} 字符\n")
        
        # 生成日期
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 构建综合分析提示词
        prompt = f"""你是一位专业的投资分析师。请根据以下四部分信息，进行综合分析，给出实质的买卖建议。

=== 信息来源 ===

【主要信息源 - 决策依据】

【1. 大模型中期买卖建议（数周-数月）】
{llm_recommendations['medium_term']}

【2. LightGBM模型20天预测结果】
{ml_predictions['lgbm']}

【3. GBDT+LR模型20天预测结果】
{ml_predictions['gbdt_lr']}

【辅助信息源 - 操作时机参考】

【4. 大模型短期买卖建议（日内/数天）】
{llm_recommendations['short_term']}

=== 综合分析规则 ===

**规则1：时间维度匹配（业界最佳实践）**
- **短期信号（触发器）**：负责"何时做"（Timing）
- **中期信号（确认器）**：负责"是否做"（Direction）
- 只有短期和中期方向一致时，才采取行动
- 短期和中期冲突时，选择观望（避免不确定性）

**决策逻辑（短期触发 + 中期确认）**：
- 短期建议买入 + 中期建议买入 → 强买入信号
- 短期建议买入 + 中期建议观望 → 观望（等待中期确认）
- 短期建议买入 + 中期建议卖出 → 不买入（冲突，信号无效）
- 短期建议卖出 + 中期建议卖出 → 强卖出信号
- 短期建议卖出 + 中期建议观望 → 观望
- 短期建议卖出 + 中期建议买入 → 不卖出（冲突，信号无效）

**规则2：一致性判断标准（基于业界最佳实践）**

**核心原则：短期触发 + 中期确认 + ML验证**

- **强买入信号**：短期建议买入 AND 中期建议买入 AND (至少一个ML模型预测上涨且probability>0.62)
- **买入信号**：短期建议买入 AND 中期建议买入 AND (至少一个ML模型预测上涨且probability>0.60)
- **观望信号**：
  - 短期建议买入 AND 中期建议观望（等待中期确认）
  - 短期建议卖出 AND 中期建议观望（等待中期确认）
  - 短期建议买入 AND 中期建议卖出（冲突）
  - 短期建议卖出 AND 中期建议买入（冲突）
  - ML模型probability在0.45-0.55之间（低置信度）
  - 两个ML模型预测冲突（信号不一致）
- **卖出信号**：短期建议卖出 AND 中期建议卖出 AND (至少一个ML模型预测下跌且probability<0.40)

**阈值优化说明（2026-02-16最新）**：
- 当前20天模型准确率：LightGBM 57.63%（标准差±6.33%），GBDT+LR 57.80%（标准差±6.06%）
- 超增强正则化优化后，标准差从±7.17%降至±6.33%/±6.06%（-11.7%/-14.3%）
- 强买入阈值0.62略高于准确率，确保高置信度
- 买入阈值0.60接近准确率，平衡召回率和精确率
- 卖出阈值0.40确保下跌概率>60%
- 观望区间0.45-0.55避免低置信度决策

**重要说明 - LR算法probability含义**：
- probability字段始终代表**上涨概率**P(y=1|x)
- 当prediction=1时：probability > 0.5（上涨概率高）
- 当prediction=0时：probability <= 0.5（上涨概率低，即下跌概率高）
- 强烈下跌信号：prediction=0且probability < 0.40（即下跌概率 > 60%）
- 中性信号：probability在0.40-0.60之间（上涨或下跌概率都不超过60%）

**重要说明 - 信号优先级（业界标准）**：
- **短期信号（触发器）**：负责"何时做"（Timing），权重100%（必须满足）
- **中期信号（确认器）**：负责"是否做"（Direction），权重100%（必须满足）
- **ML预测（验证器）**：负责提升置信度，权重50%（辅助验证）
- **关键原则**：短期和中期必须一致（方向相同），ML预测用于验证和提升置信度

**重要说明 - 模型不确定性（2026-02-16最新）**：
- ML 20天模型准确率：LightGBM 57.63%（标准差±6.33%），GBDT+LR 57.80%（标准差±6.06%）
- 超增强正则化优化后，标准差显著降低（-11.7%/-14.3%），模型稳定性提升
- 即使probability>0.62，实际准确率也可能在51.30% ~ 63.96%（LightGBM）或51.74% ~ 63.86%（GBDT+LR）之间波动
- 建议：短期和中期一致是主要决策依据，ML预测用于验证和提升置信度
- 对于probability在0.55-0.65之间的股票，建议降低仓位控制风险

**重要说明 - 时间维度标准化**：
- 短期：1-5个交易日（日内到一周）
- 中期：10-20个交易日（2-4周）
- 长期：>20个交易日（超过1个月）
- 当前映射：大模型短期建议 ↔ ML次日模型（1天），大模型中期建议 ↔ ML 20天模型（20天）✅

**规则3：ML模型冲突处理**
- 如果LightGBM和GBDT+LR预测冲突（一个上涨，一个下跌）：
  - 优先相信预测概率更高的模型
  - 如果概率相近（相差<0.10），则参考大模型中期建议
- 如果两个ML模型预测一致（都上涨或都下跌）：
  - 信号可靠性高，优先级提升

**规则4：推荐理由格式**
- 必须说明：短期建议+中期建议+哪个ML模型预测+短期中期一致性程度
- 例如："短期建议买入（触发器），中期建议买入（确认器），短期中期方向一致，LightGBM预测上涨概率0.72，GBDT+LR预测上涨概率0.68，三重确认买入，综合置信度高"

请基于上述规则，完成以下任务：

1. **一致性分析**（方案A核心：短期触发 + 中期确认）：
   - **第一步（核心）**：分析短期建议与中期建议的一致性
     - 短期买入 + 中期买入 → 方向一致，考虑ML验证
     - 短期买入 + 中期观望 → 等待中期确认
     - 短期买入 + 中期卖出 → 冲突，观望
     - 短期卖出 + 中期卖出 → 方向一致，考虑ML验证
     - 短期卖出 + 中期观望 → 等待中期确认
     - 短期卖出 + 中期买入 → 冲突，观望
   - **第二步（验证）**：对短期中期一致的股票，分析ML预测验证
     - 如果ML模型预测支持（probability>0.60），提升为强信号
     - 如果ML模型预测冲突（probability<0.40），降低为弱信号或观望
     - 如果ML模型不确定（0.45-0.55），保持中等置信度
   - 标注符合"强买入信号"、"买入信号"、"观望信号"、"卖出信号"的股票

2. **个股建议排序**：
   - 优先级：强买入信号 > 买入信号 > 观望信号 > 卖出信号
   - 在相同优先级内，按ML预测概率排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 强烈买入信号（2-3只）：最高优先级
   - 买入信号（3-5只）：次优先级
   - 持有/观望（如有）：第三优先级
   - 卖出信号（如有）：最低优先级

4. **风险提示**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比）
   - 给出止损位建议（如果有的话）

请按照以下格式输出（不要添加任何额外说明文字）：

2. **个股建议排序**：
   - 按照"强买入信号 > 中等信号 > 观望"的优先级排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 推荐买入的股票清单（按优先级排序）
   - 推荐卖出的股票清单（如有）
   - 需要关注的股票清单（观望）

4. **风险控制建议**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比）
   - 给出止损位建议（如果有的话）
   
   **特别要求 - 考虑模型不确定性（2026-02-16最新）**：
   - 超增强正则化优化后，ML 20天模型标准差降至±6.33%（LightGBM）/±6.06%（GBDT+LR），稳定性显著提升
   - 对于probability在0.55-0.65之间的股票，建议仓位不超过2-3%（可适度提升）
   - 强买入信号（短期/中期一致买入且ML模型确认）建议仓位4-6%（可适度提升）
   - 总仓位控制在45%-55%（模型稳定性提升，可适度提升）
   - 必须设置止损位，单只股票最大亏损不超过-8%
   - **严格遵循"短期触发 + 中期确认"原则**：只有短期和中期方向一致时才行动，冲突时选择观望
   - 如果短期和中期建议冲突，优先选择观望，不进行交易
   - 采用"三重确认"策略：短期、中期、ML模型三者一致时才重仓操作

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由，必须说明：短期建议+中期建议+ML预测+一致性程度。例如："短期建议买入（触发器），中期建议买入（确认器），LightGBM预测上涨概率0.72，GBDT+LR预测上涨概率0.68，短中长期方向一致（短期/中期一致买入，ML模型验证上涨），综合置信度高。注意ML模型经过超增强正则化优化（2026-02-16），标准差已降至±6.33%/±6.06%，probability在0.72附近实际准确率可能在58% ~ 66%之间"]
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
        
        # 调用大模型（关闭思考模式，避免输出思考过程）
        response = chat_with_llm(prompt, enable_thinking=False)
        
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
                
                # 构建完整的邮件内容（综合买卖建议 + 信息参考）
                full_content = f"""{response}

---

# 信息参考

## 大模型短期买卖建议（日内/数天）
{llm_recommendations['short_term']}

## 大模型中期买卖建议（数周-数月）
{llm_recommendations['medium_term']}

## 机器学习预测结果（20天）

### LightGBM模型
{ml_predictions['lgbm']}

### GBDT+LR模型
{ml_predictions['gbdt_lr']}
"""
                
                # 生成HTML格式邮件内容（将完整内容转换为HTML）
                html_content = generate_html_email(full_content, date_str)
                send_email(email_subject, full_content, html_content)
            
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