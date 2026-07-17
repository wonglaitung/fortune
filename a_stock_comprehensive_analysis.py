#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议并发送邮件

⚠️ 运行时机：建议在A股收市后（15:00 CST）运行

版本：v2.0 (2026-07-18)
- 新增邮件发送功能
- 新增HTML格式邮件
- 复用港股的邮件服务模块
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入A股配置
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_SECTOR_MAPPING,
    get_limit_rate,
)

# 导入数据服务
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.northbound_data import NorthboundDataService

# 股票名称映射
STOCK_NAMES = A_STOCK_WATCHLIST
STOCK_LIST = list(A_STOCK_WATCHLIST.keys())


def read_llm_recommendations(llm_file):
    """
    读取大模型建议文件

    Args:
        llm_file: 大模型建议文件路径

    Returns:
        str: 大模型建议内容
    """
    if not os.path.exists(llm_file):
        print(f"⚠️ 大模型建议文件不存在: {llm_file}")
        return None

    with open(llm_file, 'r', encoding='utf-8') as f:
        return f.read()


def read_ml_predictions(horizon=20):
    """
    读取ML预测结果

    Args:
        horizon: 预测周期

    Returns:
        DataFrame: 预测结果
    """
    # 查找最新的预测文件（A股专用路径）
    import glob
    files = glob.glob(f'data/a_stock_models/ml_predictions_{horizon}d.csv')

    if not files:
        print(f"⚠️ 未找到 {horizon}d 预测文件")
        return None

    latest_file = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest_file)
    print(f"✅ 读取预测文件: {latest_file}")
    return df


def get_stock_analysis(stock_code):
    """
    获取单只股票的详细分析

    Args:
        stock_code: 股票代码

    Returns:
        dict: 股票分析结果
    """
    stock_name = A_STOCK_WATCHLIST.get(stock_code, stock_code)
    result = {
        'code': stock_code,
        'name': stock_name,
        'limit_rate': get_limit_rate(stock_code),
    }

    # 获取实时行情
    realtime = get_a_stock_info_tencent(stock_code)
    if realtime:
        result['current_price'] = realtime.get('current_price')
        result['change_percent'] = realtime.get('change_percent')
        result['prev_close'] = realtime.get('prev_close')

    # 获取历史数据
    df = get_a_stock_data(stock_code, period_days=100)
    if df is not None and not df.empty:
        # 计算技术指标
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        latest = df.iloc[-1]
        result['ma5'] = latest['MA5']
        result['ma10'] = latest['MA10']
        result['ma20'] = latest['MA20']
        result['ma60'] = latest.get('MA60')

        # 计算涨跌
        if len(df) >= 5:
            result['return_5d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        if len(df) >= 20:
            result['return_20d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi_14'] = 100 - (100 / (1 + rs.iloc[-1]))

        # 计算MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        result['macd'] = df['EMA12'].iloc[-1] - df['EMA26'].iloc[-1]

    return result


def analyze_market():
    """
    分析市场环境

    Returns:
        dict: 市场分析结果
    """
    result = {}

    # 上证指数
    sh_df = get_index_data('sh', period_days=30)
    if sh_df is not None and not sh_df.empty:
        latest = sh_df.iloc[-1]
        prev = sh_df.iloc[-2] if len(sh_df) > 1 else latest
        result['sh_close'] = latest['Close']
        result['sh_change'] = (latest['Close'] / prev['Close'] - 1) * 100

        # 计算MA
        sh_df['MA20'] = sh_df['Close'].rolling(20).mean()
        result['sh_ma20'] = sh_df['MA20'].iloc[-1]
        result['sh_vs_ma20'] = (latest['Close'] / sh_df['MA20'].iloc[-1] - 1) * 100

    # 北向资金
    northbound_service = NorthboundDataService()
    nb_data = northbound_service.get_latest()
    if nb_data:
        result['northbound_net_buy'] = nb_data.get('net_buy', 0)
        result['northbound_sh'] = nb_data.get('sh_net_buy', 0)
        result['northbound_sz'] = nb_data.get('sz_net_buy', 0)

    return result


def generate_comprehensive_report(llm_content, ml_predictions_20d, stock_analyses, market_data):
    """
    生成综合分析报告

    Args:
        llm_content: 大模型建议内容
        ml_predictions_20d: 20天ML预测
        stock_analyses: 股票分析结果
        market_data: 市场数据

    Returns:
        str: 综合报告
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    report = f"""{'=' * 80}
A股综合分析报告
日期: {date_str}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

## 一、市场概况

### 1.1 上证指数
- 收盘: {market_data.get('sh_close', 'N/A'):.2f}
- 涨跌: {market_data.get('sh_change', 0):+.2f}%
- MA20: {market_data.get('sh_ma20', 'N/A'):.2f}
- 相对MA20: {market_data.get('sh_vs_ma20', 0):+.2f}%

### 1.2 北向资金
- 净买入: {market_data.get('northbound_net_buy', 0):.2f} 亿
- 沪股通: {market_data.get('northbound_sh', 0):.2f} 亿
- 深股通: {market_data.get('northbound_sz', 0):.2f} 亿

---

## 二、自选股技术分析

"""

    # 添加每只股票的分析
    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        report += f"""### {name} ({code})

**基本数据**：
- 当前价格: {analysis.get('current_price', 'N/A')} 元
- 今日涨跌: {analysis.get('change_percent', 0):+.2f}%
- 涨跌停限制: {analysis.get('limit_rate', 0) * 100:.0f}%

**技术指标**：
- MA5: {analysis.get('ma5', 'N/A'):.2f}
- MA10: {analysis.get('ma10', 'N/A'):.2f}
- MA20: {analysis.get('ma20', 'N/A'):.2f}
- RSI(14): {analysis.get('rsi_14', 'N/A'):.1f}

**近期涨跌**：
- 5日: {analysis.get('return_5d', 0):+.2f}%
- 20日: {analysis.get('return_20d', 0):+.2f}%

"""

    # 添加ML预测结果
    report += """---

## 三、机器学习预测（20天周期）

"""
    if ml_predictions_20d is not None and not ml_predictions_20d.empty:
        for _, row in ml_predictions_20d.iterrows():
            code = row.get('Stock_Code', '')
            name = A_STOCK_WATCHLIST.get(code, code)
            pred_proba = row.get('Prediction_Proba', 0.5)
            pred_label = '上涨' if pred_proba >= 0.5 else '下跌'
            confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba

            report += f"""### {name} ({code})
- 预测方向: **{pred_label}**
- 置信度: {confidence:.1%}

"""
    else:
        report += "⚠️ 未找到ML预测结果\n\n"

    # 添加大模型建议
    report += f"""---

## 四、AI分析建议

{llm_content if llm_content else '⚠️ 未找到大模型建议'}

---

## 五、操作建议汇总

| 股票 | 代码 | 当前价 | 涨跌 | 建议 |
|------|------|--------|------|------|
"""

    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', 'N/A')
        change = analysis.get('change_percent', 0)

        # 简单建议逻辑
        if analysis.get('return_20d', 0) > 10 and analysis.get('rsi_14', 50) > 70:
            suggestion = '谨慎持有'
        elif analysis.get('return_20d', 0) < -20:
            suggestion = '观望'
        elif analysis.get('rsi_14', 50) < 30:
            suggestion = '关注'
        else:
            suggestion = '持有'

        report += f"| {name} | {code} | {price} | {change:+.2f}% | {suggestion} |\n"

    report += f"""
---

## 六、风险提示

1. **涨跌停风险**: 创业板/科创板涨跌停限制20%，主板10%
2. **北向资金**: 关注外资流向变化
3. **市场情绪**: 上证指数跌破MA20时需谨慎

---

*本报告仅供参考，不构成投资建议*
"""

    return report


def generate_html_email(llm_content, ml_predictions_20d, stock_analyses, market_data):
    """
    生成HTML格式的邮件内容

    Args:
        llm_content: 大模型建议内容
        ml_predictions_20d: ML预测结果
        stock_analyses: 股票分析结果
        market_data: 市场数据

    Returns:
        str: HTML格式邮件
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 市场涨跌颜色
    sh_change = market_data.get('sh_change', 0)
    sh_color = 'green' if sh_change >= 0 else 'red'

    # 北向资金颜色
    nb_buy = market_data.get('northbound_net_buy', 0)
    nb_color = 'green' if nb_buy >= 0 else 'red'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #007bff; margin-top: 30px; }}
        h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .market-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .stock-card {{ background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; }}
        .prediction-up {{ color: green; font-weight: bold; }}
        .prediction-down {{ color: red; font-weight: bold; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>📊 A股综合分析报告</h1>
    <p>日期: {date_str}</p>

    <div class="market-box">
        <h2>📈 市场概况</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>变化</th></tr>
            <tr>
                <td>上证指数</td>
                <td>{market_data.get('sh_close', 0):.2f}</td>
                <td class="{'positive' if sh_change >= 0 else 'negative'}">{sh_change:+.2f}%</td>
            </tr>
            <tr>
                <td>北向资金净买入</td>
                <td>{nb_buy:.2f} 亿</td>
                <td class="{'positive' if nb_buy >= 0 else 'negative'}">{'流入' if nb_buy >= 0 else '流出'}</td>
            </tr>
            <tr>
                <td>沪股通</td>
                <td>{market_data.get('northbound_sh', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>深股通</td>
                <td>{market_data.get('northbound_sz', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
        </table>
    </div>

    <h2>📋 自选股分析</h2>
    <table>
        <tr>
            <th>股票</th>
            <th>代码</th>
            <th>现价</th>
            <th>涨跌</th>
            <th>RSI</th>
            <th>涨跌停</th>
        </tr>
"""

    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', '-')
        change = analysis.get('change_percent', 0)
        rsi = analysis.get('rsi_14', 50)
        limit_rate = analysis.get('limit_rate', 0.1) * 100

        change_class = 'positive' if change >= 0 else 'negative'
        rsi_class = 'negative' if rsi > 70 else ('positive' if rsi < 30 else '')

        html += f"""        <tr>
            <td>{name}</td>
            <td>{code}</td>
            <td>{price:.2f}</td>
            <td class="{change_class}">{change:+.2f}%</td>
            <td class="{rsi_class}">{rsi:.1f}</td>
            <td>{limit_rate:.0f}%</td>
        </tr>
"""

    html += """    </table>

    <h2>🤖 机器学习预测（20天）</h2>
    <table>
        <tr><th>股票</th><th>代码</th><th>预测方向</th><th>置信度</th></tr>
"""

    if ml_predictions_20d is not None and not ml_predictions_20d.empty:
        for _, row in ml_predictions_20d.iterrows():
            code = row.get('Stock_Code', '')
            name = A_STOCK_WATCHLIST.get(code, code)
            pred_proba = row.get('Prediction_Proba', 0.5)
            pred_label = '上涨' if pred_proba >= 0.5 else '下跌'
            confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba
            pred_class = 'prediction-up' if pred_proba >= 0.5 else 'prediction-down'

            html += f"""        <tr>
            <td>{name}</td>
            <td>{code}</td>
            <td class="{pred_class}">{pred_label}</td>
            <td>{confidence:.1%}</td>
        </tr>
"""
    else:
        html += '        <tr><td colspan="4">暂无预测数据</td></tr>\n'

    html += """    </table>

    <h2>💡 AI 分析建议</h2>
    <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; white-space: pre-wrap;">
"""

    # 简化LLM内容显示
    if llm_content:
        # 提取关键建议（前500字符）
        llm_summary = llm_content[:500] + '...' if len(llm_content) > 500 else llm_content
        html += llm_summary.replace('\n', '<br>')
    else:
        html += '暂无AI建议'

    html += """
    </div>

    <h2>⚠️ 风险提示</h2>
    <ul>
        <li>创业板/科创板涨跌停限制20%，主板10%</li>
        <li>关注北向资金流向变化</li>
        <li>上证指数跌破MA20时需谨慎</li>
    </ul>

    <div class="footer">
        <p>📧 本邮件由A股综合分析系统自动生成</p>
        <p>⚠️ 本报告仅供参考，不构成投资建议</p>
    </div>
</body>
</html>"""

    return html


def send_email(subject, content, html_content=None):
    """
    发送邮件通知（使用统一消息服务模块）

    参数:
    - subject: 邮件主题
    - content: 邮件文本内容
    - html_content: 邮件HTML内容（可选）
    """
    try:
        from message_services import EmailSender
        sender = EmailSender()
        return sender.send_with_retry(subject, content, html_content)
    except ImportError:
        print("⚠️ 消息服务模块未安装，使用内置邮件发送")
        return _send_email_legacy(subject, content, html_content)


def _send_email_legacy(subject, content, html_content=None):
    """
    发送邮件通知（备用实现）

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
        sender_email = os.environ.get("EMAIL_SENDER")
        email_password = os.environ.get("EMAIL_PASSWORD")
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
        recipient_email = os.environ.get("RECIPIENT_EMAIL", "")

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
        elif "qq.com" in smtp_server:
            smtp_port = 465
            use_ssl = True
        elif "gmail.com" in smtp_server:
            smtp_port = 587
            use_ssl = False
        else:
            smtp_port = 465
            use_ssl = True

        # 创建邮件对象
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)

        # 添加文本内容
        msg.attach(MIMEText(content, 'plain', 'utf-8'))

        # 添加HTML内容（如果有）
        if html_content:
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
            server.login(sender_email, email_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        print(f"✅ 邮件已发送到: {', '.join(recipients)}")
        return True

    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='A股综合分析')
    parser.add_argument('--llm-file', type=str, default=None,
                       help='大模型建议文件路径')
    parser.add_argument('--use-cached-predictions', action='store_true',
                       help='使用缓存的预测结果')
    parser.add_argument('--horizon', type=int, default=20,
                       help='预测周期（默认20天）')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件')
    parser.add_argument('--email', action='store_true',
                       help='发送邮件（默认行为）')

    args = parser.parse_args()

    # 确定是否发送邮件（默认发送，除非指定 --no-email）
    send_email_flag = not args.no_email

    print("\n" + "=" * 60)
    print("📊 A股综合分析")
    print("=" * 60)
    print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 分析股票: {len(STOCK_LIST)} 只")
    print(f"📧 发送邮件: {'是' if send_email_flag else '否'}")
    print("=" * 60)

    # 1. 读取大模型建议
    print("\n📊 读取大模型建议...")
    llm_file = args.llm_file
    if llm_file is None:
        # 查找最新的建议文件
        import glob
        files = glob.glob('data/a_stock_llm_recommendations_*.txt')
        if files:
            llm_file = max(files, key=os.path.getmtime)
            print(f"  使用文件: {llm_file}")

    llm_content = None
    if llm_file:
        llm_content = read_llm_recommendations(llm_file)
        if llm_content:
            print("  ✅ 大模型建议读取成功")

    # 2. 读取ML预测结果
    print("\n📊 读取ML预测结果...")
    ml_predictions = read_ml_predictions(args.horizon)

    # 3. 分析市场
    print("\n📊 分析市场环境...")
    market_data = analyze_market()
    if market_data:
        print(f"  上证指数: {market_data.get('sh_close', 'N/A'):.2f} ({market_data.get('sh_change', 0):+.2f}%)")
        print(f"  北向资金: {market_data.get('northbound_net_buy', 0):.2f} 亿")

    # 4. 分析每只股票
    print("\n📊 分析自选股...")
    stock_analyses = {}
    for code in STOCK_LIST:
        print(f"  分析 {code}...")
        analysis = get_stock_analysis(code)
        stock_analyses[code] = analysis

    # 5. 生成综合报告
    print("\n📊 生成综合报告...")
    report = generate_comprehensive_report(llm_content, ml_predictions, stock_analyses, market_data)

    # 保存报告
    os.makedirs('data', exist_ok=True)
    report_file = f"data/a_stock_comprehensive_recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告已保存: {report_file}")

    # 6. 发送邮件
    if send_email_flag:
        print("\n📊 发送邮件...")
        html_content = generate_html_email(llm_content, ml_predictions, stock_analyses, market_data)

        date_str = datetime.now().strftime('%Y-%m-%d')
        email_subject = f"A股综合分析报告 - {date_str}"

        if send_email(email_subject, report, html_content):
            print("  ✅ 邮件发送成功")
        else:
            print("  ⚠️ 邮件发送失败")

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("📊 报告摘要")
    print("=" * 60)
    print(f"  市场状态: 上证 {market_data.get('sh_change', 0):+.2f}%")
    print(f"  北向资金: {market_data.get('northbound_net_buy', 0):.2f} 亿")
    print(f"\n  个股建议:")
    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', 'N/A')
        change = analysis.get('change_percent', 0)
        print(f"    {name}: {price} 元 ({change:+.2f}%)")

    print("\n" + "=" * 60)
    print(f"📄 完整报告: {report_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()