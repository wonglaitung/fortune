#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一邮件发送模块

功能：
- 支持 HTML 和纯文本邮件
- 支持多收件人
- 自动 SSL/TLS 检测
- 重试机制
- 超时处理

环境变量：
- SMTP_SERVER: SMTP 服务器地址（默认 smtp.163.com）
- EMAIL_SENDER: 发件人邮箱
- EMAIL_PASSWORD: 邮箱密码/应用密码
- RECIPIENT_EMAIL: 收件人邮箱（多个用逗号分隔）

使用方法：
    from message_services import send_email

    # 快速发送
    send_email("主题", "内容", html_content="<html>...</html>")

    # 使用类
    sender = EmailSender()
    sender.send("主题", "内容")
"""

import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Union


class EmailSender:
    """统一邮件发送器"""

    # 默认端口配置
    SSL_PORT = 465
    TLS_PORT = 587

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender: Optional[str] = None,
        password: Optional[str] = None,
        recipients: Optional[List[str]] = None
    ):
        """
        初始化邮件发送器

        参数：
        - smtp_server: SMTP 服务器地址（默认从环境变量读取）
        - smtp_port: SMTP 端口（自动检测）
        - sender: 发件人邮箱（默认从环境变量读取）
        - password: 邮箱密码（默认从环境变量读取）
        - recipients: 收件人列表（默认从环境变量读取）
        """
        self.smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.163.com")
        self.sender = sender or os.environ.get("EMAIL_SENDER")
        self.password = password or os.environ.get("EMAIL_PASSWORD")

        # 解析收件人
        if recipients:
            self.recipients = recipients
        else:
            recipients_str = os.environ.get("RECIPIENT_EMAIL", "")
            self.recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

        # 自动检测端口
        self.smtp_port = smtp_port or self._detect_port()

        # 检测是否使用 SSL
        self.use_ssl = self._detect_ssl()

        # 验证配置
        if not self.sender or not self.password:
            print("⚠️ 邮件发送器配置不完整")
            print("   请设置环境变量: EMAIL_SENDER, EMAIL_PASSWORD")

    def _detect_port(self) -> int:
        """自动检测 SMTP 端口"""
        # 根据服务器域名判断
        if "163.com" in self.smtp_server or "qq.com" in self.smtp_server:
            return self.SSL_PORT
        else:
            return self.TLS_PORT

    def _detect_ssl(self) -> bool:
        """检测是否使用 SSL"""
        return self.smtp_port == self.SSL_PORT

    def _build_message(
        self,
        subject: str,
        content: str,
        html_content: Optional[str] = None,
        recipients: Optional[List[str]] = None
    ) -> MIMEMultipart:
        """构建邮件消息"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender

        # 设置收件人
        to_list = recipients or self.recipients
        msg["To"] = ", ".join(to_list)

        # 添加纯文本内容
        msg.attach(MIMEText(content, "plain", "utf-8"))

        # 添加 HTML 内容（如果有）
        if html_content:
            msg.attach(MIMEText(html_content, "html", "utf-8"))

        return msg

    def send(
        self,
        subject: str,
        content: str,
        html_content: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        timeout: int = 30
    ) -> bool:
        """
        发送邮件

        参数：
        - subject: 邮件主题
        - content: 纯文本内容
        - html_content: HTML 内容（可选）
        - recipients: 收件人列表（可选，默认使用配置的收件人）
        - timeout: 超时时间（秒）

        返回：
        - bool: 是否发送成功
        """
        if not self.sender or not self.password:
            print("❌ 邮件配置不完整，无法发送")
            return False

        to_list = recipients or self.recipients
        if not to_list:
            print("❌ 未指定收件人")
            return False

        # 构建消息
        msg = self._build_message(subject, content, html_content, to_list)

        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=timeout)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=timeout)
                server.starttls()

            server.login(self.sender, self.password)
            server.sendmail(self.sender, to_list, msg.as_string())
            server.quit()

            print(f"✅ 邮件发送成功: {subject}")
            return True

        except smtplib.SMTPException as e:
            print(f"❌ SMTP 错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    def send_with_retry(
        self,
        subject: str,
        content: str,
        html_content: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> bool:
        """
        带重试机制的邮件发送

        参数：
        - subject: 邮件主题
        - content: 纯文本内容
        - html_content: HTML 内容（可选）
        - recipients: 收件人列表（可选）
        - max_retries: 最大重试次数
        - retry_delay: 重试间隔（秒）

        返回：
        - bool: 是否发送成功
        """
        for attempt in range(max_retries):
            if self.send(subject, content, html_content, recipients):
                return True

            if attempt < max_retries - 1:
                print(f"⏳ 第 {attempt + 1} 次重试，等待 {retry_delay} 秒...")
                time.sleep(retry_delay)

        print(f"❌ 重试 {max_retries} 次后仍失败")
        return False


def send_email(
    subject: str,
    content: str,
    html_content: Optional[str] = None,
    recipients: Optional[List[str]] = None
) -> bool:
    """
    快速发送邮件（便捷函数）

    参数：
    - subject: 邮件主题
    - content: 纯文本内容
    - html_content: HTML 内容（可选）
    - recipients: 收件人列表（可选）

    返回：
    - bool: 是否发送成功
    """
    sender = EmailSender()
    return sender.send_with_retry(subject, content, html_content, recipients)


def send_email_simple(subject: str, content: str) -> bool:
    """
    发送简单文本邮件

    参数：
    - subject: 邮件主题
    - content: 邮件内容

    返回：
    - bool: 是否发送成功
    """
    return send_email(subject, content)


# 测试函数
def test_email_sender():
    """测试邮件发送器"""
    sender = EmailSender()

    print(f"SMTP 服务器: {sender.smtp_server}")
    print(f"SMTP 端口: {sender.smtp_port}")
    print(f"使用 SSL: {sender.use_ssl}")
    print(f"发件人: {sender.sender}")
    print(f"收件人: {sender.recipients}")

    if sender.sender and sender.password and sender.recipients:
        print("\n✅ 邮件配置完整")
    else:
        print("\n⚠️ 邮件配置不完整，请检查环境变量")


if __name__ == "__main__":
    test_email_sender()