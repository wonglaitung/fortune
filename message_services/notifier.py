#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一通知管理器

功能：
- 统一管理邮件、企微机器人、WxPusher 等通知渠道
- 一键发送到多个渠道
- 渠道状态检测

使用方法：
    from message_services import Notifier, notify_all

    # 使用便捷函数
    notify_all("标题", "内容")

    # 使用类
    notifier = Notifier(channels=['email', 'wechat_work'])
    notifier.notify("标题", "内容")
"""

import os
from typing import List, Optional, Dict
from datetime import datetime

from .email_sender import EmailSender
from .message_formatter import format_html_report


class Notifier:
    """统一通知管理器"""

    # 支持的通知渠道
    AVAILABLE_CHANNELS = ['email', 'wechat_work', 'wxpusher']

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        email_sender: Optional[EmailSender] = None
    ):
        """
        初始化通知管理器

        参数：
        - channels: 启用的通知渠道列表（默认全部）
        - email_sender: 自定义邮件发送器（可选）
        """
        # 默认启用所有渠道
        self.channels = channels or self.AVAILABLE_CHANNELS.copy()

        # 初始化邮件发送器
        self._email_sender = email_sender or EmailSender()

        # 初始化微信模块（延迟加载）
        self._wechat_work_bot = None
        self._wxpusher = None

    def _get_wechat_work_bot(self):
        """获取企微机器人实例（延迟加载）"""
        if self._wechat_work_bot is None:
            try:
                from .wechat_work_bot import WeChatWorkBot
                self._wechat_work_bot = WeChatWorkBot()
            except ImportError:
                print("⚠️ 企微机器人模块未安装")
        return self._wechat_work_bot

    def _get_wxpusher(self):
        """获取 WxPusher 实例（延迟加载）"""
        if self._wxpusher is None:
            try:
                from .wxpusher_bot import WxPusher
                self._wxpusher = WxPusher()
            except ImportError:
                print("⚠️ WxPusher 模块未安装")
        return self._wxpusher

    def notify(
        self,
        title: str,
        content: str,
        channels: Optional[List[str]] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        发送通知到多个渠道

        参数：
        - title: 通知标题
        - content: 通知内容
        - channels: 指定渠道（可选，默认使用初始化时的配置）
        - html_content: HTML 格式内容（仅邮件使用）

        返回：
        - dict: 各渠道发送结果 {'email': True, 'wechat_work': False, ...}
        """
        target_channels = channels or self.channels
        results = {}

        for channel in target_channels:
            if channel == 'email':
                results['email'] = self.notify_email(title, content, html_content)
            elif channel == 'wechat_work':
                results['wechat_work'] = self.notify_wechat_work(title, content)
            elif channel == 'wxpusher':
                results['wxpusher'] = self.notify_wxpusher(title, content)
            else:
                print(f"⚠️ 未知的通知渠道: {channel}")
                results[channel] = False

        return results

    def notify_email(
        self,
        title: str,
        content: str,
        html_content: Optional[str] = None
    ) -> bool:
        """
        发送邮件通知

        参数：
        - title: 邮件主题
        - content: 纯文本内容
        - html_content: HTML 内容（可选）

        返回：
        - bool: 是否发送成功
        """
        return self._email_sender.send_with_retry(title, content, html_content)

    def notify_wechat_work(self, title: str, content: str) -> bool:
        """
        发送企微机器人通知

        参数：
        - title: 消息标题
        - content: 消息内容

        返回：
        - bool: 是否发送成功
        """
        bot = self._get_wechat_work_bot()
        if bot is None:
            return False

        # 格式化为 Markdown
        msg = f"## {title}\n\n{content}\n\n> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return bot.send_markdown(msg)

    def notify_wxpusher(self, title: str, content: str) -> bool:
        """
        发送 WxPusher 推送

        参数：
        - title: 消息标题
        - content: 消息内容

        返回：
        - bool: 是否发送成功
        """
        pusher = self._get_wxpusher()
        if pusher is None:
            return False

        # 格式化为 Markdown
        msg = f"# {title}\n\n{content}\n\n---\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        result = pusher.send_markdown(msg, summary=title)
        return result.get('success', False)

    def check_channels(self) -> Dict[str, bool]:
        """
        检查各通知渠道的配置状态

        返回：
        - dict: 各渠道配置状态
        """
        status = {}

        # 检查邮件
        status['email'] = bool(
            self._email_sender.sender and
            self._email_sender.password and
            self._email_sender.recipients
        )

        # 检查企微机器人
        status['wechat_work'] = bool(os.environ.get("WECHAT_WORK_WEBHOOK"))

        # 检查 WxPusher
        status['wxpusher'] = bool(
            os.environ.get("WXPUSHER_TOKEN") and
            os.environ.get("WXPUSHER_UIDS")
        )

        return status

    def get_available_channels(self) -> List[str]:
        """
        获取已配置的通知渠道

        返回：
        - list: 已配置的渠道列表
        """
        status = self.check_channels()
        return [ch for ch, configured in status.items() if configured]


def notify_all(title: str, content: str, html_content: Optional[str] = None) -> Dict[str, bool]:
    """
    发送通知到所有已配置渠道（便捷函数）

    参数：
    - title: 通知标题
    - content: 通知内容
    - html_content: HTML 内容（仅邮件使用）

    返回：
    - dict: 各渠道发送结果
    """
    notifier = Notifier()
    available = notifier.get_available_channels()

    if not available:
        print("⚠️ 没有配置任何通知渠道")
        return {}

    print(f"📢 发送通知到: {', '.join(available)}")
    return notifier.notify(title, content, available, html_content)


def notify_email(title: str, content: str, html_content: Optional[str] = None) -> bool:
    """
    仅发送邮件通知（便捷函数）

    参数：
    - title: 邮件主题
    - content: 纯文本内容
    - html_content: HTML 内容（可选）

    返回：
    - bool: 是否发送成功
    """
    notifier = Notifier(channels=['email'])
    return notifier.notify_email(title, content, html_content)


def notify_wechat(title: str, content: str) -> Dict[str, bool]:
    """
    发送微信通知（企微 + WxPusher）（便捷函数）

    参数：
    - title: 通知标题
    - content: 通知内容

    返回：
    - dict: 各渠道发送结果
    """
    notifier = Notifier(channels=['wechat_work', 'wxpusher'])
    return notifier.notify(title, content)


# 测试函数
def test_notifier():
    """测试通知管理器"""
    notifier = Notifier()

    print("=== 通知渠道状态 ===")
    status = notifier.check_channels()
    for channel, configured in status.items():
        icon = "✅" if configured else "❌"
        print(f"  {icon} {channel}")

    print("\n=== 已配置渠道 ===")
    available = notifier.get_available_channels()
    print(f"  {available if available else '无'}")


if __name__ == "__main__":
    test_notifier()