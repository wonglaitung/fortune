#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WxPusher 微信推送模块

功能：
- 发送文本、HTML、Markdown 消息
- 支持多用户推送
- 支持消息回调

使用方法：
1. 在 WxPusher 官网创建应用，获取 appToken
2. 关注应用获取 UID
3. 设置环境变量 WXPUSHER_TOKEN 和 WXPUSHER_UIDS
4. 调用 send_message() 发送推送

环境变量：
- WXPUSHER_TOKEN: WxPusher 应用的 appToken
- WXPUSHER_UIDS: 接收消息的用户 UID，多个用逗号分隔

官网：https://wxpusher.zjiecode.com/
文档：https://wxpusher.zjiecode.com/docs

免费额度：200 条/天
"""

import os
import requests
import json
from datetime import datetime
from typing import Optional, List, Union


class WxPusher:
    """WxPusher 推送客户端"""

    API_URL = "http://wxpusher.zjiecode.com/api/send/message"

    # 消息类型
    CONTENT_TYPE_TEXT = 1      # 文本
    CONTENT_TYPE_HTML = 2      # HTML
    CONTENT_TYPE_MD = 3        # Markdown

    def __init__(self, app_token: Optional[str] = None, uids: Optional[List[str]] = None):
        """
        初始化 WxPusher 客户端

        参数：
        - app_token: 应用 Token，如不传入则从环境变量读取
        - uids: 用户 UID 列表，如不传入则从环境变量读取
        """
        self.app_token = app_token or os.environ.get("WXPUSHER_TOKEN")

        # 解析 UIDs
        if uids:
            self.uids = uids
        else:
            uids_str = os.environ.get("WXPUSHER_UIDS", "")
            self.uids = [u.strip() for u in uids_str.split(",") if u.strip()]

        if not self.app_token:
            print("⚠️ 未配置 WxPusher Token")
            print("   请设置环境变量 WXPUSHER_TOKEN")

        if not self.uids:
            print("⚠️ 未配置接收用户 UID")
            print("   请设置环境变量 WXPUSHER_UIDS")

    def send(self, content: str, content_type: int = 1,
             summary: Optional[str] = None,
             url: Optional[str] = None) -> dict:
        """
        发送消息

        参数：
        - content: 消息内容
        - content_type: 消息类型 (1=文本, 2=HTML, 3=Markdown)
        - summary: 消息摘要（显示在通知栏，可选）
        - url: 点击消息跳转的 URL（可选）

        返回：
        - dict: API 响应结果
        """
        if not self.app_token or not self.uids:
            return {"success": False, "msg": "未配置 Token 或 UIDs"}

        payload = {
            "appToken": self.app_token,
            "content": content,
            "contentType": content_type,
            "uids": self.uids
        }

        if summary:
            payload["summary"] = summary

        if url:
            payload["url"] = url

        try:
            response = requests.post(
                self.API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            result = response.json()

            if result.get("code") == 1000:
                return {"success": True, "msg": "发送成功", "data": result.get("data")}
            else:
                return {"success": False, "msg": result.get("msg", "未知错误")}

        except requests.exceptions.Timeout:
            return {"success": False, "msg": "请求超时"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "msg": f"请求异常: {e}"}
        except json.JSONDecodeError:
            return {"success": False, "msg": "响应解析失败"}

    def send_text(self, content: str, summary: Optional[str] = None) -> dict:
        """发送文本消息"""
        return self.send(content, self.CONTENT_TYPE_TEXT, summary)

    def send_html(self, content: str, summary: Optional[str] = None) -> dict:
        """发送 HTML 消息"""
        return self.send(content, self.CONTENT_TYPE_HTML, summary)

    def send_markdown(self, content: str, summary: Optional[str] = None) -> dict:
        """发送 Markdown 消息"""
        return self.send(content, self.CONTENT_TYPE_MD, summary)

    def query_users(self) -> dict:
        """查询关注应用的用户列表"""
        if not self.app_token:
            return {"success": False, "msg": "未配置 Token"}

        url = f"http://wxpusher.zjiecode.com/api/fun/wxuser?appToken={self.app_token}"

        try:
            response = requests.get(url, timeout=10)
            result = response.json()

            if result.get("code") == 1000:
                return {"success": True, "users": result.get("data", [])}
            else:
                return {"success": False, "msg": result.get("msg")}

        except Exception as e:
            return {"success": False, "msg": str(e)}


def format_trading_alert_md(title: str, content: str) -> str:
    """
    格式化交易预警 Markdown 消息

    参数：
    - title: 标题
    - content: 内容

    返回：
    - Markdown 格式消息
    """
    return f"""# {title}

{content}

---
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""


def format_daily_report_md(report_data: dict) -> str:
    """
    格式化每日分析报告 Markdown 消息

    参数：
    - report_data: 报告数据

    返回：
    - Markdown 格式消息
    """
    lines = [
        "# 📊 港股智能分析日报",
        "",
        f"**日期**: {datetime.now().strftime('%Y-%m-%d')}",
        ""
    ]

    # 市场概况
    if 'hsi' in report_data:
        hsi = report_data['hsi']
        lines.append("## 市场概况")
        lines.append(f"- 恒生指数: **{hsi.get('close', 'N/A')}**")
        change = hsi.get('change_pct', 0)
        change_icon = "📈" if change >= 0 else "📉"
        lines.append(f"- 日涨跌: {change_icon} **{change:+.2f}%**")
        lines.append("")

    # 预测摘要
    if 'predictions' in report_data:
        preds = report_data['predictions']
        lines.append("## 模型预测")

        if '20d' in preds:
            p20 = preds['20d']
            direction = p20.get('direction', 'N/A')
            accuracy = p20.get('accuracy', 0)
            icon = "🟢" if direction == "上涨" else "🔴"
            lines.append(f"- 20天: {icon} **{direction}** (准确率 {accuracy:.0%})")

        if '5d' in preds:
            p5 = preds['5d']
            direction = p5.get('direction', 'N/A')
            accuracy = p5.get('accuracy', 0)
            icon = "🟢" if direction == "上涨" else "🔴"
            lines.append(f"- 5天: {icon} {direction} (准确率 {accuracy:.0%})")

        lines.append("")

    # 重点关注
    if 'focus_stocks' in report_data and report_data['focus_stocks']:
        lines.append("## 重点关注")
        for stock in report_data['focus_stocks'][:5]:
            name = stock.get('name', '')
            code = stock.get('code', '')
            reason = stock.get('reason', '')
            lines.append(f"- **{name}** ({code}): {reason}")
        lines.append("")

    # 风险提示
    lines.append("## ⚠️ 风险提示")
    lines.append("- 以上分析仅供参考，不构成投资建议")
    lines.append("- 个股预测准确率约 54%，需谨慎决策")

    return "\n".join(lines)


# 便捷函数
def send_alert(title: str, content: str,
               app_token: Optional[str] = None,
               uids: Optional[List[str]] = None) -> dict:
    """
    发送预警消息（便捷函数）

    参数：
    - title: 标题
    - content: 内容
    - app_token: 应用 Token（可选）
    - uids: 用户 UID 列表（可选）

    返回：
    - dict: 发送结果
    """
    pusher = WxPusher(app_token, uids)
    msg = format_trading_alert_md(title, content)
    return pusher.send_markdown(msg, summary=title)


def send_report(report_data: dict,
                app_token: Optional[str] = None,
                uids: Optional[List[str]] = None) -> dict:
    """
    发送每日报告（便捷函数）

    参数：
    - report_data: 报告数据
    - app_token: 应用 Token（可选）
    - uids: 用户 UID 列表（可选）

    返回：
    - dict: 发送结果
    """
    pusher = WxPusher(app_token, uids)
    msg = format_daily_report_md(report_data)
    return pusher.send_markdown(msg, summary="港股智能分析日报")


def test_wxpusher():
    """测试 WxPusher 推送"""
    token = os.environ.get("WXPUSHER_TOKEN")
    uids = os.environ.get("WXPUSHER_UIDS")

    if not token or not uids:
        print("请先设置环境变量：")
        print("  export WXPUSHER_TOKEN='your_app_token'")
        print("  export WXPUSHER_UIDS='uid1,uid2'")
        print("\n获取方法：")
        print("  1. 访问 https://wxpusher.zjiecode.com/ 创建应用")
        print("  2. 获取 appToken")
        print("  3. 扫码关注应用获取 UID")
        return

    pusher = WxPusher()

    # 测试消息
    result = pusher.send_markdown(
        "# 测试消息\n\n这是来自港股分析系统的测试消息。\n\n✅ 推送正常工作！",
        summary="港股系统测试"
    )

    if result["success"]:
        print("✅ 测试消息发送成功")
    else:
        print(f"❌ 发送失败: {result['msg']}")


if __name__ == "__main__":
    test_wxpusher()