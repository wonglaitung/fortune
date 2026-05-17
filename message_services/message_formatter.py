#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息格式化工具

功能：
- 格式化 HTML 报告
- Markdown 转 HTML
- 生成邮件模板
"""

import re
from datetime import datetime
from typing import Optional


# 默认 HTML 样式
DEFAULT_HTML_STYLE = """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 20px; }
    h3 { color: #7f8c8d; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th { background-color: #3498db; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .positive { color: #27ae60; font-weight: bold; }
    .negative { color: #e74c3c; font-weight: bold; }
    .warning { color: #f39c12; }
    .info { background-color: #d5dbdb; padding: 10px; border-radius: 5px; }
    .timestamp { color: #95a5a6; font-size: 0.9em; }
</style>
"""


def format_html_report(
    title: str,
    content: str,
    style: str = "default",
    include_timestamp: bool = True
) -> str:
    """
    格式化 HTML 报告

    参数：
    - title: 报告标题
    - content: 报告内容（可以是 Markdown）
    - style: CSS 样式（default 或自定义）
    - include_timestamp: 是否包含时间戳

    返回：
    - str: 完整的 HTML 文档
    """
    # 获取样式
    css_style = DEFAULT_HTML_STYLE if style == "default" else style

    # 转换内容为 HTML
    html_content = markdown_to_html(content)

    # 构建完整 HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        css_style,
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        html_content,
    ]

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_parts.append(f"<p class='timestamp'>生成时间: {timestamp}</p>")

    html_parts.extend([
        "</body>",
        "</html>"
    ])

    return "\n".join(html_parts)


def markdown_to_html(md_content: str) -> str:
    """
    简单的 Markdown 转 HTML

    支持的格式：
    - 标题：# ## ###
    - 加粗：**text**
    - 列表：- item
    - 表格：| col1 | col2 |
    - 链接：[text](url)

    参数：
    - md_content: Markdown 内容

    返回：
    - str: HTML 内容
    """
    html = md_content

    # 转换标题
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # 转换加粗
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # 转换链接
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)

    # 转换列表
    lines = html.split('\n')
    in_list = False
    result_lines = []

    for line in lines:
        if line.startswith('- ') or line.startswith('* '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            result_lines.append(f'<li>{line[2:]}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)

    if in_list:
        result_lines.append('</ul>')

    html = '\n'.join(result_lines)

    # 转换表格（简单处理）
    if '|' in html:
        lines = html.split('\n')
        in_table = False
        result_lines = []

        for line in lines:
            # 检测表格行
            if line.strip().startswith('|') and line.strip().endswith('|'):
                cells = [c.strip() for c in line.strip().split('|')[1:-1]]

                if not in_table:
                    result_lines.append('<table>')
                    in_table = True
                    # 第一行作为表头
                    result_lines.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
                else:
                    # 跳过分隔行（如 |---|---|）
                    if not all(c.replace('-', '').replace(':', '') == '' for c in cells):
                        result_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            else:
                if in_table:
                    result_lines.append('</table>')
                    in_table = False
                result_lines.append(line)

        if in_table:
            result_lines.append('</table>')

        html = '\n'.join(result_lines)

    # 转换段落（连续的非空行）
    paragraphs = []
    current_para = []

    for line in html.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('<'):
            current_para.append(stripped)
        else:
            if current_para:
                paragraphs.append('<p>' + ' '.join(current_para) + '</p>')
                current_para = []
            if stripped:
                paragraphs.append(stripped)

    if current_para:
        paragraphs.append('<p>' + ' '.join(current_para) + '</p>')

    return '\n'.join(paragraphs)


def format_stock_alert_html(
    title: str,
    stocks: list,
    hsi_prediction: Optional[dict] = None
) -> str:
    """
    格式化股票预警 HTML

    参数：
    - title: 标题
    - stocks: 股票列表
    - hsi_prediction: 恒指预测信息

    返回：
    - str: HTML 内容
    """
    lines = [
        f"<h2>{title}</h2>",
        f"<p class='timestamp'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
    ]

    if hsi_prediction:
        direction = hsi_prediction.get('direction', '未知')
        confidence = hsi_prediction.get('confidence', 0)
        lines.append("<h3>恒生指数预测</h3>")
        lines.append(f"<p>方向: <strong>{direction}</strong></p>")
        lines.append(f"<p>置信度: {confidence:.1%}</p>")

    if stocks:
        lines.append("<h3>个股信号</h3>")
        lines.append("<table>")
        lines.append("<tr><th>股票</th><th>代码</th><th>信号</th><th>置信度</th></tr>")

        for stock in stocks[:10]:
            code = stock.get('code', '')
            name = stock.get('name', '')
            signal = stock.get('signal', '')
            confidence = stock.get('confidence', 0)

            signal_class = 'positive' if signal == '买入' else 'negative' if signal == '卖出' else ''
            lines.append(f"<tr><td>{name}</td><td>{code}</td><td class='{signal_class}'>{signal}</td><td>{confidence:.1%}</td></tr>")

        lines.append("</table>")

    return '\n'.join(lines)


def format_trading_report_html(
    report_data: dict
) -> str:
    """
    格式化交易报告 HTML

    参数：
    - report_data: 报告数据字典

    返回：
    - str: HTML 内容
    """
    lines = [
        "<h1>📊 港股智能分析日报</h1>",
        f"<p class='timestamp'>日期: {datetime.now().strftime('%Y-%m-%d')}</p>",
    ]

    # 市场概况
    if 'hsi' in report_data:
        hsi = report_data['hsi']
        lines.append("<h2>市场概况</h2>")
        lines.append(f"<p>恒生指数: <strong>{hsi.get('close', 'N/A')}</strong></p>")
        change = hsi.get('change_pct', 0)
        change_class = 'positive' if change >= 0 else 'negative'
        lines.append(f"<p>日涨跌: <span class='{change_class}'>{change:+.2f}%</span></p>")

    # 预测摘要
    if 'predictions' in report_data:
        preds = report_data['predictions']
        lines.append("<h2>模型预测</h2>")
        lines.append("<table>")
        lines.append("<tr><th>周期</th><th>方向</th><th>准确率</th></tr>")

        for period, data in preds.items():
            direction = data.get('direction', 'N/A')
            accuracy = data.get('accuracy', 0)
            dir_class = 'positive' if direction == '上涨' else 'negative'
            lines.append(f"<tr><td>{period}</td><td class='{dir_class}'>{direction}</td><td>{accuracy:.0%}</td></tr>")

        lines.append("</table>")

    # 风险提示
    lines.append("<h2>⚠️ 风险提示</h2>")
    lines.append("<div class='info'>")
    lines.append("<p>以上分析仅供参考，不构成投资建议</p>")
    lines.append("<p>个股预测准确率约 54%，需谨慎决策</p>")
    lines.append("</div>")

    return '\n'.join(lines)


if __name__ == "__main__":
    # 测试 Markdown 转 HTML
    test_md = """
## 测试标题

这是一段测试内容。

- 列表项 1
- 列表项 2

| 列1 | 列2 |
|-----|-----|
| A | B |
| C | D |
"""

    print(markdown_to_html(test_md))