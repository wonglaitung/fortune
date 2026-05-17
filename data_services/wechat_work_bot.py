#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业微信机器人通知模块

功能：
- 发送 Markdown 格式消息
- 发送文本消息
- 发送卡片消息
- 支持港股分析报告推送

使用方法：
1. 在企业微信群中添加机器人，获取 Webhook URL
2. 设置环境变量 WECHAT_WORK_WEBHOOK 或在代码中传入
3. 调用 send_markdown_msg() 或 send_trading_alert()

环境变量：
- WECHAT_WORK_WEBHOOK: 企业微信机器人 Webhook URL

参考文档：https://developer.work.weixin.qq.com/document/path/91770
"""

import os
import requests
import json
from datetime import datetime
from typing import Optional, List, Dict


class WeChatWorkBot:
    """企业微信机器人"""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        初始化企业微信机器人

        参数：
        - webhook_url: Webhook URL，如不传入则从环境变量读取
        """
        self.webhook_url = webhook_url or os.environ.get("WECHAT_WORK_WEBHOOK")
        if not self.webhook_url:
            print("⚠️ 未配置企业微信机器人 Webhook URL")
            print("   请设置环境变量 WECHAT_WORK_WEBHOOK")
            print("   或在初始化时传入 webhook_url 参数")

    def _send_request(self, data: dict) -> bool:
        """发送请求到企业微信 API"""
        if not self.webhook_url:
            print("❌ Webhook URL 未配置，无法发送消息")
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=data,
                timeout=10
            )
            result = response.json()

            if result.get('errcode') == 0:
                return True
            else:
                print(f"❌ 发送失败: {result.get('errmsg', '未知错误')}")
                return False

        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            return False
        except json.JSONDecodeError:
            print("❌ 响应解析失败")
            return False

    def send_text(self, content: str, mentioned_list: Optional[List[str]] = None) -> bool:
        """
        发送文本消息

        参数：
        - content: 文本内容
        - mentioned_list: 提醒人员的手机号或企业微信ID
        """
        data = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list or []
            }
        }
        return self._send_request(data)

    def send_markdown(self, content: str) -> bool:
        """
        发送 Markdown 消息

        参数：
        - content: Markdown 格式内容（支持标题、引用、加粗等）

        注意：
        - Markdown 内容最长不超过 4096 字节
        - 支持的格式：# 标题、> 引用、**加粗**、[链接](url)、代码块
        """
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }
        return self._send_request(data)

    def send_image(self, base64_image: str, md5: str) -> bool:
        """
        发送图片消息

        参数：
        - base64_image: 图片的 Base64 编码
        - md5: 图片的 MD5 值
        """
        data = {
            "msgtype": "image",
            "image": {
                "base64": base64_image,
                "md5": md5
            }
        }
        return self._send_request(data)

    def send_news(self, articles: List[Dict]) -> bool:
        """
        发送图文消息

        参数：
        - articles: 图文列表，每项包含 title, description, url, picurl
        """
        data = {
            "msgtype": "news",
            "news": {
                "articles": articles
            }
        }
        return self._send_request(data)

    def send_file(self, media_id: str) -> bool:
        """
        发送文件消息（需先上传文件获取 media_id）

        参数：
        - media_id: 文件 ID
        """
        data = {
            "msgtype": "file",
            "file": {
                "media_id": media_id
            }
        }
        return self._send_request(data)


def format_stock_alert(title: str, stocks: List[Dict],
                       hsi_prediction: Optional[Dict] = None) -> str:
    """
    格式化股票预警消息

    参数：
    - title: 标题
    - stocks: 股票列表，每项包含 code, name, signal, confidence 等
    - hsi_prediction: 恒指预测信息

    返回：
    - Markdown 格式的消息内容
    """
    lines = [f"## {title}", ""]

    # 添加时间
    lines.append(f"> 发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 添加恒指预测
    if hsi_prediction:
        direction = hsi_prediction.get('direction', '未知')
        confidence = hsi_prediction.get('confidence', 0)
        lines.append(f"### 恒生指数预测")
        lines.append(f"- **方向**: {direction}")
        lines.append(f"- **置信度**: {confidence:.1%}")
        lines.append("")

    # 添加股票列表
    if stocks:
        lines.append("### 个股信号")
        lines.append("")
        for stock in stocks[:10]:  # 最多显示10只
            code = stock.get('code', '')
            name = stock.get('name', '')
            signal = stock.get('signal', '')
            confidence = stock.get('confidence', 0)

            # 信号图标
            signal_icon = "📈" if signal == "买入" else "📉" if signal == "卖出" else "⏸️"

            lines.append(f"{signal_icon} **{name}** ({code})")
            lines.append(f"   - 信号: {signal}")
            lines.append(f"   - 置信度: {confidence:.1%}")
            lines.append("")

    return "\n".join(lines)


def format_daily_report(report_data: Dict) -> str:
    """
    格式化每日分析报告

    参数：
    - report_data: 报告数据

    返回：
    - Markdown 格式的消息内容
    """
    lines = [
        f"## 港股智能分析日报",
        "",
        f"> 日期: {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    # 市场概况
    if 'hsi' in report_data:
        hsi = report_data['hsi']
        lines.append("### 市场概况")
        lines.append(f"- 恒生指数: {hsi.get('close', 'N/A')}")
        lines.append(f"- 日涨跌: {hsi.get('change_pct', 'N/A'):+.2f}%")
        lines.append("")

    # 预测摘要
    if 'predictions' in report_data:
        preds = report_data['predictions']
        lines.append("### 模型预测")

        # 20天预测
        if '20d' in preds:
            p20 = preds['20d']
            lines.append(f"- **20天预测**: {p20.get('direction', 'N/A')} (准确率 {p20.get('accuracy', 'N/A'):.1%})")

        # 5天预测
        if '5d' in preds:
            p5 = preds['5d']
            lines.append(f"- 5天预测: {p5.get('direction', 'N/A')} (准确率 {p5.get('accuracy', 'N/A'):.1%})")

        lines.append("")

    # 重点关注
    if 'focus_stocks' in report_data:
        lines.append("### 重点关注")
        for stock in report_data['focus_stocks'][:5]:
            lines.append(f"- {stock.get('name', '')} ({stock.get('code', '')}): {stock.get('reason', '')}")
        lines.append("")

    # 风险提示
    lines.append("### ⚠️ 风险提示")
    lines.append("- 以上分析仅供参考，不构成投资建议")
    lines.append("- 个股预测准确率约 54%，需谨慎决策")

    return "\n".join(lines)


def send_trading_alert(title: str, content: str,
                       webhook_url: Optional[str] = None) -> bool:
    """
    发送交易预警消息（便捷函数）

    参数：
    - title: 标题
    - content: 内容
    - webhook_url: Webhook URL（可选）

    返回：
    - 是否发送成功
    """
    bot = WeChatWorkBot(webhook_url)

    # 格式化消息
    msg = f"## {title}\n\n{content}\n\n> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return bot.send_markdown(msg)


def send_daily_report(report_data: Dict,
                      webhook_url: Optional[str] = None) -> bool:
    """
    发送每日分析报告（便捷函数）

    参数：
    - report_data: 报告数据
    - webhook_url: Webhook URL（可选）

    返回：
    - 是否发送成功
    """
    bot = WeChatWorkBot(webhook_url)
    content = format_daily_report(report_data)
    return bot.send_markdown(content)


# 测试函数
def test_wechat_bot():
    """测试企业微信机器人"""
    webhook = os.environ.get("WECHAT_WORK_WEBHOOK")

    if not webhook:
        print("请先设置环境变量 WECHAT_WORK_WEBHOOK")
        print("示例: export WECHAT_WORK_WEBHOOK='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx'")
        return

    bot = WeChatWorkBot()

    # 测试 Markdown 消息
    test_msg = """## 港股分析测试消息

> 这是测试消息

### 测试内容
- ✅ Markdown 格式正常
- ✅ 标题显示正常
- ✅ 引用格式正常

**发送时间**: {}
""".format(datetime.now().strftime('%Y-%m-%d %H:%M'))

    success = bot.send_markdown(test_msg)

    if success:
        print("✅ 测试消息发送成功")
    else:
        print("❌ 测试消息发送失败")


if __name__ == "__main__":
    test_wechat_bot()