#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息服务模块

提供统一的通知接口，支持：
- 邮件发送
- 企业微信机器人
- WxPusher 推送

使用方法：
    from message_services import send_email, notify_all

    # 发送邮件
    send_email("主题", "内容", html_content="<html>...</html>")

    # 发送到所有渠道
    notify_all("标题", "内容")
"""

from .email_sender import EmailSender, send_email, send_email_simple
from .message_formatter import (
    format_html_report,
    markdown_to_html,
    format_stock_alert_html,
    format_trading_report_html
)
from .notifier import (
    Notifier,
    notify_all,
    notify_email,
    notify_wechat
)

# 延迟导入微信模块（避免循环导入）
__all__ = [
    # 邮件
    'EmailSender',
    'send_email',
    'send_email_simple',

    # 格式化
    'format_html_report',
    'markdown_to_html',
    'format_stock_alert_html',
    'format_trading_report_html',

    # 通知管理
    'Notifier',
    'notify_all',
    'notify_email',
    'notify_wechat',
]


def get_wechat_work_bot():
    """获取企微机器人类（延迟导入）"""
    from .wechat_work_bot import WeChatWorkBot
    return WeChatWorkBot


def get_wxpusher():
    """获取 WxPusher 类（延迟导入）"""
    from .wxpusher_bot import WxPusher
    return WxPusher