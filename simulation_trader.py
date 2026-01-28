#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股模拟交易系统
基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易

新增功能：
1. 根据市场情况自动建议买入股票
"""

# 全局参数配置
DEFAULT_ANALYSIS_FREQUENCY = 60  # 默认分析频率（分钟）

import os
import sys
import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import threading
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入hk_smart_money_tracker模块和腾讯财经接口
import hk_smart_money_tracker
from tencent_finance import get_hk_stock_data_tencent

class SimulationTrader:
    def __init__(self, initial_capital=1000000, analysis_frequency=DEFAULT_ANALYSIS_FREQUENCY, investor_type="进取型"):
        """
        初始化模拟交易系统
        
        Args:
            initial_capital (float): 初始资金，默认100万港元
            analysis_frequency (int): 分析频率（分钟），默认15分钟
            investor_type (str): 投资者风险偏好，默认为"进取型"
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 持仓 {code: {'shares': 数量, 'avg_price': 平均买入价, 'stop_loss_price': 止损价格}}
        self.transaction_history = []  # 交易历史
        self.portfolio_history = []  # 投资组合价值历史
        self.start_date = datetime.now()
        self.is_trading_hours = True  # 模拟港股交易时间 (9:30-16:00)
        self.analysis_frequency = analysis_frequency  # 分析频率（分钟）
        self.investor_type = investor_type  # 投资者风险偏好
        self.decision_history = []  # 存储历史决策以供大模型分析
        
        # 确保data目录存在
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 持久化文件
        self.state_file = os.path.join(self.data_dir, "simulation_state.json")
        # 日志文件路径会在需要时动态生成
        
        # 尝试从文件恢复状态
        self.load_state()
        
        # 如果是新开始，在当天的日志文件中记录初始信息
        if not self.transaction_history:
            today_log_file = self.get_daily_log_file()
            with open(today_log_file, "w", encoding="utf-8") as f:
                f.write(f"模拟交易日志 - 开始时间: {self.start_date}\n")
                f.write(f"初始资金: HK${self.initial_capital:,.2f}\n")
                f.write("="*50 + "\n")
    
    @staticmethod
    def convert_investor_type_to_english(investor_type):
        """
        将中文投资者类型转换为英文
        
        Args:
            investor_type (str): 中文投资者类型（进取型、稳健型、保守型）
            
        Returns:
            str: 英文投资者类型（aggressive、moderate、conservative）
        """
        type_mapping = {
            '进取型': 'aggressive',
            '稳健型': 'moderate',
            '保守型': 'conservative'
        }
        return type_mapping.get(investor_type, 'moderate')
    
    def send_email_notification(self, subject, content):
        """
        发送邮件通知
        
        Args:
            subject (str): 邮件主题
            content (str): 邮件内容
        """
        try:
            smtp_server = os.environ.get("YAHOO_SMTP", "smtp.gmail.com")
            smtp_user = os.environ.get("YAHOO_EMAIL")
            smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
            sender_email = smtp_user

            if not smtp_user or not smtp_pass:
                self.log_message("警告: 缺少 YAHOO_EMAIL 或 YAHOO_APP_PASSWORD 环境变量，无法发送邮件")
                return False

            recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
            recipients = [r.strip() for r in recipient_env.split(',')] if ',' in recipient_env else [recipient_env]

            # 创建邮件
            msg = MIMEMultipart("alternative")
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            # 添加文本内容
            text_part = MIMEText(content, "plain", "utf-8")
            msg.attach(text_part)

            # 根据SMTP服务器类型选择合适的端口和连接方式
            if "163.com" in smtp_server:
                # 163邮箱使用SSL连接，端口465
                smtp_port = 465
                use_ssl = True
            elif "gmail.com" in smtp_server:
                # Gmail使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False
            else:
                # 默认使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False

            # 发送邮件（增加重试机制）
            for attempt in range(3):
                try:
                    if use_ssl:
                        # 使用SSL连接
                        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()
                    else:
                        # 使用TLS连接
                        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                        server.starttls()
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()
                    
                    self.log_message(f"邮件发送成功: {subject}")
                    return True
                except Exception as e:
                    self.log_message(f"发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试，等待后重试
                        time.sleep(5)
            
            self.log_message(f"发送邮件失败，已重试3次")
            return False
        except Exception as e:
            self.log_message(f"发送邮件失败: {e}")
            return False
    
    def send_trading_notification(self, notification_type, **kwargs):
        """
        发送交易通知邮件的统一方法
        
        Args:
            notification_type (str): 通知类型
            **kwargs: 其他参数，根据通知类型而定
        """
        # 构建持仓详情文本
        positions_detail = self.build_positions_detail()
        
        if notification_type == "buy":
            # 买入通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            price = kwargs.get('price')
            shares = kwargs.get('shares')
            amount = kwargs.get('amount')
            reason = kwargs.get('reason')
            is_new_stock = kwargs.get('is_new_stock', False)
            
            subject_prefix = "【新买入通知】" if is_new_stock else "【加仓通知】"
            subject = f"{subject_prefix}{name} ({code})"
            content = f"""
模拟交易系统买入通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
买入价格：HK${price:.2f}
买入数量：{shares} 股
买入金额：HK${amount:.2f}
买入原因：{reason if reason else '未提供理由'}
交易时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        elif notification_type == "sell":
            # 卖出通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            price = kwargs.get('price')
            shares = kwargs.get('shares')
            amount = kwargs.get('amount')
            avg_price = kwargs.get('avg_price')
            profit_loss = kwargs.get('profit_loss')
            reason = kwargs.get('reason')
            
            profit_loss_status = "盈利" if profit_loss >= 0 else "亏损"
            subject = f"【卖出通知】{name} ({code})"
            content = f"""
模拟交易系统卖出通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
卖出价格：HK${price:.2f}
卖出数量：{shares} 股
卖出金额：HK${amount:.2f}
平均成本：HK${avg_price:.2f}
盈亏金额：HK${profit_loss:+.2f} ({profit_loss_status})
卖出原因：{reason if reason else '未提供理由'}
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        elif notification_type == "cannot_sell":
            # 无法卖出通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            reason = kwargs.get('reason')
            stop_loss_triggered = kwargs.get('stop_loss_triggered', False)
            
            subject = f"【无法卖出通知】{name} ({code})"
            content = f"""
模拟交易系统无法按大模型建议卖出通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
大模型建议卖出理由：{reason}
是否止损触发：{stop_loss_triggered}
无法卖出原因：未持有该股票
交易时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{positions_detail}
            """
        elif notification_type == "stop_loss":
            # 止损触发通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            current_price = kwargs.get('current_price')
            stop_loss_price = kwargs.get('stop_loss_price')
            
            subject = f"【止损触发通知】{name} ({code})"
            content = f"""
模拟交易系统止损触发通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
当前价格：HK${current_price:.2f}
止损价格：HK${stop_loss_price:.2f}
触发原因：当前价格跌破止损价格
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{positions_detail}
            """
        elif notification_type == "insufficient_funds":
            # 资金不足无法买入通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            reason = kwargs.get('reason')
            allocation_pct = kwargs.get('allocation_pct')
            stop_loss_price = kwargs.get('stop_loss_price')
            required_amount = kwargs.get('required_amount')
            
            subject = f"【无法买入通知】{name} ({code})"
            content = f"""
模拟交易系统无法按大模型建议买入通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
大模型建议买入理由：{reason}
资金分配比例：{allocation_pct}
建议止损价格：{stop_loss_price}
建议买入金额：HK${required_amount:.2f}
无法买入原因：资金不足
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        elif notification_type == "max_position_reached":
            # 持仓比例已达到建议值通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            reason = kwargs.get('reason')
            allocation_pct = kwargs.get('allocation_pct')
            stop_loss_price = kwargs.get('stop_loss_price')
            
            subject = f"【持仓比例已达上限通知】{name} ({code})"
            content = f"""
模拟交易系统持仓比例已达建议上限通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
大模型建议买入理由：{reason}
资金分配比例：{allocation_pct}
建议止损价格：{stop_loss_price}
无法买入原因：当前持仓价值已达到大模型建议的比例上限
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        elif notification_type == "buy_failed":
            # 买入执行失败通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            reason = kwargs.get('reason')
            allocation_pct = kwargs.get('allocation_pct')
            stop_loss_price = kwargs.get('stop_loss_price')
            
            subject = f"【无法买入通知】{name} ({code})"
            content = f"""
模拟交易系统无法按大模型建议买入通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
大模型建议买入理由：{reason}
资金分配比例：{allocation_pct}
建议止损价格：{stop_loss_price}
无法买入原因：交易执行失败
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        elif notification_type == "manual_cannot_sell":
            # 手工卖出无法执行通知
            code = kwargs.get('code')
            name = kwargs.get('name')
            
            subject = f"【无法卖出通知】{name} ({code})"
            content = f"""
模拟交易系统无法按手动指令卖出通知：

投资者风险偏好：{self.investor_type}
股票名称：{name}
股票代码：{code}
无法卖出原因：未持有该股票
交易时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{positions_detail}
            """
        elif notification_type == "test":
            # 邮件功能测试
            subject = "港股模拟交易系统 - 邮件功能测试"
            content = f"""
这是港股模拟交易系统的邮件功能测试邮件。

投资者风险偏好：{self.investor_type}
时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
系统状态：
- 初始资金: HK${self.initial_capital:,.2f}
- 当前资金: HK${self.capital:,.2f}
- 持仓数量: {len(self.positions)}

{positions_detail}
            """
        else:
            # 未知通知类型
            self.log_message(f"未知的邮件通知类型: {notification_type}")
            return False
        
        return self.send_email_notification(subject, content)
    
    def get_daily_log_file(self):
        """获取当天的日志文件路径"""
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.data_dir, f"simulation_trade_log_{today}.txt")
    
    def log_message(self, message):
        """记录交易日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 获取当天的日志文件路径
        today_log_file = self.get_daily_log_file()
        
        # 写入日志文件
        with open(today_log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    
    def save_state(self):
        """保存交易状态到文件"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'capital': self.capital,
                'positions': self.positions,
                'transaction_history': self.transaction_history,
                'portfolio_history': self.portfolio_history,
                'decision_history': self.decision_history,
                'start_date': self.start_date.isoformat() if isinstance(self.start_date, datetime) else str(self.start_date)
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.log_message(f"保存状态失败: {e}")
    
    def load_state(self):
        """从文件加载交易状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.initial_capital = state.get('initial_capital', self.initial_capital)
                self.capital = state.get('capital', self.initial_capital)
                self.positions = {k: v for k, v in state.get('positions', {}).items()}
                self.transaction_history = state.get('transaction_history', [])
                self.portfolio_history = state.get('portfolio_history', [])
                self.decision_history = state.get('decision_history', [])
                
                # 恢复开始日期
                start_date_str = state.get('start_date')
                if start_date_str:
                    try:
                        self.start_date = datetime.fromisoformat(start_date_str)
                    except:
                        self.start_date = datetime.now()
                
                self.log_message(f"从文件恢复状态成功: {len(self.transaction_history)} 笔交易, {len(self.positions)} 个持仓, {len(self.decision_history)} 条决策历史")
                return True
        except Exception as e:
            self.log_message(f"加载状态失败: {e}")
        return False
    
    def get_current_stock_price(self, code):
        """
        获取股票当前价格（使用腾讯财经接口）
        
        Args:
            code (str): 股票代码
            
        Returns:
            float: 当前价格，如果获取失败返回None
        """
        try:
            # 移除代码中的.HK后缀，腾讯财经接口不需要
            stock_code = code.replace('.HK', '')
            
            # 获取最近3天的数据
            hist = get_hk_stock_data_tencent(stock_code, period_days=3)
            if hist is not None and not hist.empty:
                # 返回最新的收盘价
                return hist['Close'].iloc[-1]
        except Exception as e:
            self.log_message(f"获取股票 {code} 价格失败: {e}")
        return None
    
    def get_portfolio_value(self):
        """
        计算当前投资组合总价值
        
        Returns:
            float: 投资组合总价值
        """
        total_value = self.capital
        
        # 计算持仓价值
        for code, position in self.positions.items():
            current_price = self.get_current_stock_price(code)
            if current_price is not None:
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    
    
    
    
    def calculate_shares_to_buy(self, code, name, allocation_pct, current_price):
        """
        根据资金分配比例计算应买入的股数
        
        Args:
            code (str): 股票代码
            name (str): 股票名称
            allocation_pct: 资金分配比例
            current_price (float): 当前价格
            
        Returns:
            tuple: (shares, required_amount, reason) 股数、所需金额和原因
        """
        try:
            # 解析资金分配比例
            allocation_pct_value = 0
            if isinstance(allocation_pct, str):
                # 处理百分比格式，如"10%"
                if '%' in allocation_pct:
                    allocation_pct_value = float(allocation_pct.replace('%', '')) / 100
                else:
                    # 对于不带%的字符串，判断是小数还是百分比
                    # 如果值大于1，假设是百分比形式（如"15.0"表示15.0%）
                    # 如果值小于等于1，假设是小数形式（如"0.15"表示15%）
                    allocation_pct_value = float(allocation_pct)
                    if allocation_pct_value > 1:
                        allocation_pct_value = allocation_pct_value / 100
            else:
                # 数值形式的分配比例，需要判断是小数还是百分比
                # 如果值大于1，假设是百分比形式（如19.36表示19.36%）
                # 如果值小于等于1，假设是小数形式（如0.15表示15%）
                allocation_pct_value = float(allocation_pct)
                if allocation_pct_value > 1:
                    allocation_pct_value = allocation_pct_value / 100
            
            # 获取当前投资组合总价值
            current_portfolio_value = self.get_portfolio_value()
            
            # 根据资金分配比例计算应买入的金额（基于投资组合总价值）
            target_investment = current_portfolio_value * allocation_pct_value
            
            # 检查是否有足够现金
            if target_investment > self.capital:
                # 如果按投资组合比例计算的金额超过可用现金，则使用可用现金进行投资
                # 或者根据可用现金重新计算实际能投资的比例
                self.log_message(f"按大模型建议的{allocation_pct_value*100:.2f}%资金分配比例计算出的投资金额超出可用现金，限制买入金额至现金余额")
                target_investment = self.capital
            
            # 计算对应的股数（确保是100的倍数）
            shares = int(target_investment / current_price // 100) * 100
            
            # 限制股数不超过可用资金允许的最大股数（确保是100的倍数）
            max_affordable_shares = int(self.capital / current_price // 100) * 100
            shares = min(shares, max_affordable_shares)
            
            required_amount = shares * current_price
            
            # 检查是否有足够资金
            if required_amount > self.capital:
                # 如果资金不足，计算在当前资金下最多能买入多少股（确保是100的倍数）
                max_shares = int(self.capital / current_price // 100) * 100
                
                if max_shares > 0:
                    shares = max_shares
                    required_amount = shares * current_price
                    self.log_message(f"资金不足按大模型建议比率买入 {name} ({code})，改为买入 {shares} 股")
                    return shares, required_amount, "资金不足"
                else:
                    self.log_message(f"资金不足买入 {name} ({code})，即使买入100股也不够资金（当前资金: HK${self.capital:.2f}, 股价: HK${current_price:.2f}）")
                    return 0, required_amount, "资金不足"
            
            # 计算买入后股票在投资组合中的预期比例
            expected_stock_value = shares * current_price
            expected_portfolio_value = current_portfolio_value - required_amount  # 买入后投资组合价值 = 原价值 - 买入金额
            expected_allocation = (expected_stock_value / expected_portfolio_value) * 100 if expected_portfolio_value > 0 else 0
            
            # 如果预期持仓比例超过大模型建议的比例，则调整买入数量
            if expected_allocation > (allocation_pct_value * 100) and expected_portfolio_value > 0:
                # 计算在不超过建议比例的前提下，最大可买入的金额
                current_position_value = self.get_current_stock_price(code) * self.positions.get(code, {}).get('shares', 0) if code in self.positions and self.get_current_stock_price(code) is not None else 0
                max_allowed_total_value = current_portfolio_value * allocation_pct_value  # 按建议比例允许的该股票总价值
                max_investment_for_allocation = max_allowed_total_value - current_position_value  # 还可以投资的金额
                
                if max_investment_for_allocation > 0:
                    max_shares_for_allocation = int(max_investment_for_allocation / current_price // 100) * 100
                    if max_shares_for_allocation > 0:
                        # 取资金限制和比例限制下的较小股数
                        shares = min(shares, max_shares_for_allocation)
                        required_amount = shares * current_price
                        self.log_message(f"调整买入股数以符合大模型建议的持仓比例: {shares} 股 {name} ({code})")
                        return shares, required_amount, "持仓比例调整"
                elif max_investment_for_allocation <= 0:
                    # 如果当前持仓价值已经等于或超过建议比例，说明不需要再买入
                    # 或者如果当前持仓价值超过建议比例，说明已经超配了
                    self.log_message(f"当前持仓价值已达到或超过大模型建议的持仓比例，无需买入 {name} ({code})")
                    return 0, 0, "持仓比例已达到建议值"  # 返回0股数和0金额，原因是指定的文本
                    
            # 检查是否因为资金不足导致无法买入任何股票
            if shares <= 0 and target_investment > 0:
                # 如果目标投资金额大于0，但计算出的股数为0，说明资金不足以购买最小单位（100股）
                self.log_message(f"资金不足买入 {name} ({code})，即使买入100股也不够资金（目标投资: HK${target_investment:.2f}, 当前资金: HK${self.capital:.2f}, 股价: HK${current_price:.2f}）")
                return 0, required_amount, "资金不足"
            else:
                return shares, required_amount, "正常买入"
        except (ValueError, TypeError):
            # 如果无法解析资金分配比例
            self.log_message(f"无法解析资金分配比例 {allocation_pct}，跳过买入 {name} ({code})")
            return 0, 0, "无法解析资金分配比例"

    def buy_stock_by_shares(self, code, name, shares, reason=None, stop_loss_price=None, price_at_calculation=None, skip_decision_record=False, target_price=None, validity_period=None):
        """
        按指定股数买入股票
        
        Args:
            code (str): 股票代码
            name (str): 股票名称
            shares (int): 买入股数
            reason (str): 买入原因
            stop_loss_price (float): 止损价格
            price_at_calculation (float): 计算股数时的股价（可选），如果提供则使用此价格而非重新获取
            skip_decision_record (bool): 是否跳过在执行阶段的交易记录（当在决策阶段已记录时）
        """
        # 检查是否在交易时间
        if not self.is_trading_hours:
            self.log_message(f"非交易时间，跳过买入 {name} ({code})")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                amount = shares * (price_at_calculation if price_at_calculation is not None else 0) if shares > 0 and (price_at_calculation if price_at_calculation is not None else 0) > 0 else 0
                actual_current_price = price_at_calculation if price_at_calculation is not None else self.get_current_stock_price(code)
                self.record_transaction('BUY', code, name, shares, price_at_calculation if price_at_calculation is not None else 0, amount, reason, False, stop_loss_price=stop_loss_price, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 检查股数是否为0
        if shares <= 0:
            self.log_message(f"买入股数为0或负数，跳过买入 {name} ({code})")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                amount = shares * (price_at_calculation if price_at_calculation is not None else 0) if shares > 0 and (price_at_calculation if price_at_calculation is not None else 0) > 0 else 0
                actual_current_price = price_at_calculation if price_at_calculation is not None else self.get_current_stock_price(code)
                self.record_transaction('BUY', code, name, shares, price_at_calculation if price_at_calculation is not None else 0, amount, reason, False, stop_loss_price=stop_loss_price, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 获取当前价格（如果提供了计算时的价格，则使用计算时的价格）
        current_price = price_at_calculation if price_at_calculation is not None else self.get_current_stock_price(code)
        if current_price is None:
            self.log_message(f"无法获取 {name} ({code}) 的当前价格，跳过买入")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                amount = shares * 0 if shares > 0 else 0
                actual_current_price = current_price if current_price is not None else self.get_current_stock_price(code)
                self.record_transaction('BUY', code, name, shares, 0, amount, reason, False, stop_loss_price=stop_loss_price, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 计算实际投资金额
        actual_invest = shares * current_price
        
        # 检查是否有足够资金
        if actual_invest > self.capital:
            self.log_message(f"资金不足买入 {shares} 股 {name} ({code})，需要 HK${actual_invest:.2f}，当前资金 HK${self.capital:.2f}")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                self.record_transaction('BUY', code, name, shares, current_price, actual_invest, reason, False, stop_loss_price=stop_loss_price, current_price=current_price, target_price=target_price, validity_period=validity_period)
            
            # 返回一个特殊值来表示资金不足
            return "insufficient_funds"
            
        # 检查是否是新买入的股票
        is_new_stock = code not in self.positions
        
        # 执行买入
        self.capital -= actual_invest
        
        # 更新持仓
        if code in self.positions:
            # 如果已有持仓，更新平均买入价
            existing_shares = self.positions[code]['shares']
            existing_avg_price = self.positions[code]['avg_price']
            new_avg_price = (existing_shares * existing_avg_price + shares * current_price) / (existing_shares + shares)
            self.positions[code]['shares'] += shares
            self.positions[code]['avg_price'] = new_avg_price
            # 如果有新的止损价格建议，更新止损价格
            if stop_loss_price is not None and stop_loss_price != '未提供':
                try:
                    stop_loss_price_float = float(stop_loss_price)
                    self.positions[code]['stop_loss_price'] = stop_loss_price_float
                except (ValueError, TypeError):
                    # 如果无法转换为数字，忽略
                    pass
        else:
            # 新建持仓
            position_info = {
                'shares': shares,
                'avg_price': current_price
            }
            # 添加止损价格（如果提供）
            if stop_loss_price is not None and stop_loss_price != '未提供':
                try:
                    stop_loss_price_float = float(stop_loss_price)
                    position_info['stop_loss_price'] = stop_loss_price_float
                except (ValueError, TypeError):
                    # 如果无法转换为数字，忽略
                    pass
            self.positions[code] = position_info
            
        # 记录交易
        self.record_transaction('BUY', code, name, shares, current_price, actual_invest, reason, True, stop_loss_price=stop_loss_price, current_price=current_price, target_price=target_price, validity_period=validity_period)
        
        # 记录投资组合价值
        portfolio_value = self.get_portfolio_value()
        self.portfolio_history.append({
            'timestamp': datetime.now().isoformat(),
            'capital': self.capital,
            'portfolio_value': portfolio_value,
            'positions': dict(self.positions)
        })
        
        # 保存状态
        self.save_state()
        self.save_portfolio_to_csv()  # 立即保存投资组合历史到CSV
        
        self.log_message(f"买入 {shares} 股 {name} ({code}) @ HK${current_price:.2f}, 总金额: HK${actual_invest:.2f}")
        
        # 发送买入通知邮件（无论是否为新股票）
        self.send_trading_notification(
            notification_type="buy",
            code=code,
            name=name,
            price=current_price,
            shares=shares,
            amount=actual_invest,
            reason=reason,
            is_new_stock=is_new_stock
        )
        
        return True
    
    def sell_stock(self, code, name, percentage=1.0, reason=None, skip_decision_record=False, target_price=None, validity_period=None):
        """
        卖出股票
        
        Args:
            code (str): 股票代码
            name (str): 股票名称
            percentage (float): 卖出比例，默认100%
            reason (str): 卖出原因
            skip_decision_record (bool): 是否跳过在执行阶段的交易记录（当在决策阶段已记录时）
        """
        # 检查是否在交易时间
        if not self.is_trading_hours:
            self.log_message(f"非交易时间，跳过卖出 {name} ({code})")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                actual_shares = int(self.positions.get(code, {}).get('shares', 0) * percentage) if code in self.positions else 0
                actual_current_price = self.get_current_stock_price(code)  # 获取当前价格作为参考
                self.record_transaction('SELL', code, name, actual_shares, 0, 0, reason, False, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 检查是否有持仓
        if code not in self.positions:
            self.log_message(f"未持有 {name} ({code})，无法卖出")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                amount = 0 * 0 if 0 > 0 else 0
                current_price = self.get_current_stock_price(code)  # 获取当前价格作为参考
                self.record_transaction('SELL', code, name, 0, 0, amount, reason, False, current_price=current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        position = self.positions[code]
        avg_price = position['avg_price']
        current_price = self.get_current_stock_price(code)
        if current_price is None:
            self.log_message(f"无法获取 {name} ({code}) 的当前价格，跳过卖出")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                actual_shares = int(self.positions.get(code, {}).get('shares', 0) * percentage)
                actual_current_price = self.get_current_stock_price(code)  # 即使获取不到也尝试一次
                self.record_transaction('SELL', code, name, actual_shares, 0, 0, reason, False, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 计算卖出股数
        shares_to_sell = int(position['shares'] * percentage)
        if shares_to_sell <= 0:
            self.log_message(f"卖出股数为0，跳过卖出 {name} ({code})")
            
            # 如果没有在决策阶段记录过，则在执行阶段记录失败的交易
            if not skip_decision_record:
                actual_current_price = self.get_current_stock_price(code)
                self.record_transaction('SELL', code, name, 0, current_price, 0, reason, False, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
            return False
            
        # 计算卖出金额
        sell_amount = shares_to_sell * current_price
        
        # 计算盈亏金额
        profit_loss = (current_price - avg_price) * shares_to_sell
        
        # 执行卖出
        self.capital += sell_amount
        
        # 更新持仓
        self.positions[code]['shares'] -= shares_to_sell
        if self.positions[code]['shares'] <= 0:
            del self.positions[code]
            
        # 记录交易
        self.record_transaction('SELL', code, name, shares_to_sell, current_price, sell_amount, reason, True, profit_loss, current_price=current_price)
        
        # 记录投资组合价值
        portfolio_value = self.get_portfolio_value()
        self.portfolio_history.append({
            'timestamp': datetime.now().isoformat(),
            'capital': self.capital,
            'portfolio_value': portfolio_value,
            'positions': dict(self.positions)
        })
        
        # 保存状态
        self.save_state()
        self.save_portfolio_to_csv()  # 立即保存投资组合历史到CSV
        
        # 确定盈亏状态文本
        profit_loss_status = "盈利" if profit_loss >= 0 else "亏损"
        
        self.log_message(f"卖出 {shares_to_sell} 股 {name} ({code}) @ HK${current_price:.2f}, 总金额: HK${sell_amount:.2f}, {profit_loss_status}: HK${abs(profit_loss):.2f}")
        
        # 发送卖出通知邮件
        self.send_trading_notification(
            notification_type="sell",
            code=code,
            name=name,
            price=current_price,
            shares=shares_to_sell,
            amount=sell_amount,
            avg_price=avg_price,
            profit_loss=profit_loss,
            reason=reason
        )
        
        return True
    
    def is_trading_time(self):
        """
        检查是否为港股交易时间
        港股交易时间: 9:30-12:00, 13:00-16:00
        
        Returns:
            bool: 是否为交易时间
        """
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # 周末不交易
        if weekday >= 5:
            return False
            
        # 检查交易时间
        hour = now.hour
        minute = now.minute
        
        # 上午交易时间 9:30-12:00
        if (hour == 9 and minute >= 30) or (hour > 9 and hour < 12):
            return True
            
        # 下午交易时间 13:00-16:00
        if hour >= 13 and hour < 16:
            return True
            
        return False
    
    def parse_llm_recommendations(self, llm_analysis):
        """
        解析大模型的推荐结果
        
        Args:
            llm_analysis (str): 大模型分析结果
            
        Returns:
            dict: 解析后的推荐结果 {'buy': [股票信息列表], 'sell': [股票信息列表]}
        """
        recommendations = {
            'buy': [],
            'sell': []
        }
        
        # 解析JSON格式的输出
        try:
            import json
            # 尝试从大模型输出中提取JSON部分
            json_start = llm_analysis.find('{')
            json_end = llm_analysis.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = llm_analysis[json_start:json_end]
                parsed_json = json.loads(json_str)
                
                # 验证JSON格式是否符合预期
                if 'buy' in parsed_json and 'sell' in parsed_json:
                    # 验证买入股票是否在自选股列表中
                    buy_stocks = []
                    for stock in parsed_json['buy']:
                        if isinstance(stock, dict) and 'code' in stock:
                            if stock['code'] in hk_smart_money_tracker.WATCHLIST:
                                buy_stocks.append(stock)
                    
                    # 验证卖出股票是否在自选股列表中（允许建议卖出任何在自选股中的股票，即使没有持仓）
                    sell_stocks = []
                    for stock in parsed_json['sell']:
                        if isinstance(stock, dict) and 'code' in stock:
                            if stock['code'] in hk_smart_money_tracker.WATCHLIST:
                                sell_stocks.append(stock)
                    
                    recommendations['buy'] = buy_stocks
                    recommendations['sell'] = sell_stocks
                    self.log_message("成功解析JSON格式的买卖信号")
                    return recommendations
                else:
                    self.log_message("JSON格式不包含预期的buy/sell字段")
            else:
                self.log_message("未找到JSON格式的输出")
        except Exception as e:
            self.log_message(f"解析JSON格式失败: {e}")
        
        # 如果JSON解析失败，跳过本次交易
        self.log_message("JSON格式解析失败，跳过本次交易")
        return recommendations
    
    def get_recent_decisions_context(self):
        """
        获取最近的决策历史作为上下文提供给大模型，特别关注时间窗口分析
        
        Returns:
            str: 决策历史上下文文本
        """
        if not self.decision_history:
            return "无历史决策记录。"
        
        # 获取当前时间
        now = datetime.now()
        today = now.date()
        
        # 筛选出交易记录（只包含BUY和SELL，不包含持仓信息）
        trade_records = []
        for record in self.decision_history:
            timestamp_str = record.get('timestamp', '未知时间')
            decision_type = record.get('type', '未知类型')
            
            # 只处理交易记录（BUY和SELL），忽略其他类型
            if decision_type in ['BUY', 'SELL']:
                try:
                    # 解析时间戳
                    if 'Z' in timestamp_str:
                        decision_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        decision_time = datetime.fromisoformat(timestamp_str)
                    
                    # 计算时间差
                    time_diff = now - decision_time
                    hours_diff = time_diff.total_seconds() / 3600
                    
                    # 添加时间差信息
                    record_copy = record.copy()
                    record_copy['hours_diff'] = hours_diff
                    record_copy['datetime'] = decision_time
                    
                    trade_records.append(record_copy)
                except:
                    # 如果时间戳格式不正确，尝试直接检查日期部分
                    if timestamp_str.startswith(today.strftime('%Y-%m-%d')):
                        record_copy = record.copy()
                        record_copy['hours_diff'] = 24  # 假设是今天但无法解析具体时间
                        trade_records.append(record_copy)
        
        if not trade_records:
            return "今天无交易记录。"
        
        # 按时间排序（最新的在前）
        trade_records.sort(key=lambda x: x.get('hours_diff', float('inf')))
        
        # 分类不同时间窗口的交易
        recent_3h = [r for r in trade_records if r.get('hours_diff', float('inf')) <= 3]
        recent_24h = [r for r in trade_records if r.get('hours_diff', float('inf')) <= 24]
        
        # 构建上下文
        context = f"交易历史分析（当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}）:\n\n"
        
        # 最近3小时的交易（关键决策约束）
        if recent_3h:
            context += "【最近3小时内的交易 - 关键约束窗口】:\n"
            for i, record in enumerate(recent_3h):
                decision_type = record.get('type', '未知类型')
                name = record.get('name', record.get('code', '未知'))
                code = record.get('code', '未知代码')
                shares = record.get('shares', '未知数量')
                price = record.get('price', '未知价格')
                amount = record.get('amount', '未知金额')
                reason = record.get('reason', '未提供理由')
                hours_ago = record.get('hours_diff', 0)
                
                context += f"  {i+1}. {decision_type} {name}({code}) - {hours_ago:.1f}小时前\n"
                context += f"     数量: {shares}股, 价格: HK${price:.2f}, 金额: HK${amount:.2f}\n"
                context += f"     原因: {reason}\n"
        else:
            context += "【最近3小时内无交易记录】\n"
        
        # 最近24小时的交易（决策参考窗口）
        if recent_24h:
            context += "\n【最近24小时内的交易 - 决策参考窗口】:\n"
            # 按股票分组，显示每只股票的最近操作
            stock_operations = {}
            for record in recent_24h:
                code = record.get('code', '未知')
                if code not in stock_operations:
                    stock_operations[code] = []
                stock_operations[code].append(record)
            
            for code, operations in stock_operations.items():
                name = operations[0].get('name', code)
                context += f"  {name}({code}):\n"
                for op in operations:
                    decision_type = op.get('type', '未知')
                    hours_ago = op.get('hours_diff', 0)
                    price = op.get('price', 0)
                    context += f"    - {decision_type} {hours_ago:.1f}小时前 @ HK${price:.2f}\n"
        else:
            context += "\n【最近24小时内无交易记录】\n"
        
        # 特别提醒
        context += "\n【决策一致性提醒】:\n"
        context += "1. 3小时窗口内严禁对同一股票进行相反操作\n"
        context += "2. 24小时窗口内只允许在有明确技术信号变化时调整方向\n"
        context += "3. 如果没有重大变化，应该选择'观望'而非频繁操作\n"
        
        return context

    

    def record_transaction(self, transaction_type, code, name, shares, price, amount, reason, success, profit_loss=None, stop_loss_price=None, current_price=None, target_price=None, validity_period=None):
        """
        记录交易（成功或失败）
        
        Args:
            transaction_type (str): 交易类型 ('BUY' 或 'SELL')
            code (str): 股票代码
            name (str): 股票名称
            shares (int): 交易股数
            price (float): 交易价格
            amount (float): 交易金额
            reason (str): 交易原因
            success (bool): 交易是否成功
            profit_loss (float, optional): 盈亏金额，仅适用于卖出交易
            stop_loss_price (float, optional): 止损价格
            current_price (float, optional): 当前价格
            target_price (float, optional): 目标价格
            validity_period (int, optional): 有效期（天数）
        """
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': transaction_type,
            'code': code,
            'name': name,
            'shares': shares,
            'price': price,
            'amount': amount,
            'capital_after': self.capital,
            'reason': reason if reason else '未提供理由',
            'success': success,
            'stop_loss_price': stop_loss_price,
            'current_price': current_price,
            'target_price': target_price,
            'validity_period': validity_period
        }
        self.transaction_history.append(transaction)
        self.save_transactions_to_csv()  # 立即保存到CSV
        
        # 将交易记录添加到决策历史中
        trade_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': transaction_type,
            'code': code,
            'name': name,
            'shares': shares,
            'price': price,
            'amount': amount,
            'reason': reason,
            'capital_after': self.capital,
            'success': success,
            'stop_loss_price': stop_loss_price,
            'current_price': current_price,
            'target_price': target_price,
            'validity_period': validity_period
        }
        
        # 如果是卖出交易且提供了盈亏信息，则添加到记录中
        if transaction_type == 'SELL' and profit_loss is not None:
            trade_record['profit_loss'] = profit_loss
            
        self.decision_history.append(trade_record)
        
        # 限制决策历史记录数量，只保留最近的50条记录
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]

    def execute_trades(self):
        """执行交易决策"""
        # 更新交易时间状态
        self.is_trading_hours = self.is_trading_time()
        
        # 检查是否为交易时间
        if not self.is_trading_hours:
            self.log_message("非交易时间，暂停交易")
            return
            
        self.log_message("开始执行交易决策...")
        
        # 设定投资者风险偏好
        # 保守型：偏好低风险、稳定收益的股票
        # 平衡型：平衡风险与收益
        # 进取型：偏好高风险、高收益的股票
        # 使用实例变量存储投资者风险偏好
        
        # 运行股票分析
        try:
            self.log_message("正在分析股票...")
            results = []
            for code, name in hk_smart_money_tracker.WATCHLIST.items():
                res = hk_smart_money_tracker.analyze_stock(code, name)
                if res:
                    results.append(res)
                    
            if not results:
                self.log_message("股票分析无结果")
                return
                
            # 构建大模型分析提示
            # 转换投资者类型为英文
            investor_type_english = self.convert_investor_type_to_english(self.investor_type)
            llm_prompt = hk_smart_money_tracker.build_llm_analysis_prompt(results, None, None, investor_type_english)
            
            # 调用大模型分析（真实调用）
            self.log_message("正在调用大模型分析...")
        
            # 导入大模型服务
            from llm_services import qwen_engine
            llm_analysis = qwen_engine.chat_with_llm(llm_prompt)
            self.log_message("大模型分析调用成功")
            self.log_message(f"大模型分析结果:\n{llm_analysis}")
                
            # 获取历史决策上下文
            decision_history_context = self.get_recent_decisions_context()
                
            # 再次调用大模型，要求以固定格式输出买卖信号和资金分配建议
            format_prompt = f"""
请分析以下港股分析报告，考虑投资者风险偏好为{self.investor_type}，并严格按照以下JSON格式输出买卖信号和资金分配建议：

报告内容：
{llm_analysis}

【关键决策约束】请严格遵守以下规则，避免今日出现的频繁矛盾交易：

1. 决策一致性检查（必须执行）：
   - 检查最近3小时内是否有同一股票的相反操作记录
   - 检查最近24小时内是否有同一股票的买卖记录
   - 如果存在上述记录，除非出现明确的技术指标反转信号（如MACD死叉/金叉、放量突破等）或重大新闻事件，否则必须维持原方向

2. 时间窗口管理：
   - 3小时窗口：严禁对同一股票进行相反操作
   - 24小时窗口：只允许在有明确技术信号变化时调整
   - 对于已触发Trailing Stop的股票，需要确认是否是正常波动而非趋势反转

3. 风险控制优先级：
   - 资金保护 > 收益追求：减少交易频率，提高决策质量
   - 单只股票日内不应出现超过1次买卖方向转换
   - 如果建议与历史决策相反，必须在理由中明确说明"市场发生重大变化：[具体变化内容]"

4. 决策解释要求：
   - 如果维持原决策，请明确说明"维持此前决策，不建议操作"
   - 如果改变决策方向，必须解释具体的技术指标或基本面变化
   - 对于没有重大变化的股票，应该选择"观望"而非频繁操作

【历史决策记录】
{decision_history_context}

投资者风险偏好：{self.investor_type}
- 保守型：偏好低风险、稳定收益的股票，如高股息银行股，注重资本保值
- 平衡型：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进进型：偏好高风险、高收益的股票，如科技成长股，追求资本增值

请严格按照以下格式输出：
{{
    "buy": [
        {{"code": "股票代码1", "name": "股票名称1", "reason": "买入理由", "allocation_pct": 数字, "stop_loss_price": 数字, "target_price": 数字, "validity_period": 数字}},
        {{"code": "股票代码2", "name": "股票名称2", "reason": "买入理由", "allocation_pct": 数字, "stop_loss_price": 数字, "target_price": 数字, "validity_period": 数字}}
    ],
    "sell": [
        {{"code": "股票代码3", "name": "股票名称3", "reason": "卖出理由", "stop_loss_triggered": 布尔值, "target_price": 数字, "validity_period": 数字}},
        {{"code": "股票代码4", "name": "股票名称4", "reason": "卖出理由", "stop_loss_triggered": 布尔值, "target_price": 数字, "validity_period": 数字}}
    ]
}}

要求：
1. 只输出JSON格式，不要包含其他文字
2. "buy"字段包含建议买入的股票信息列表，每项包含代码、名称、理由、资金分配比例、止损价格、目标价格和有效期
3. "sell"字段包含建议卖出的股票信息列表，每项包含代码、名称、理由、是否由止损机制触发、目标价格和有效期
4. **资金分配比例必须是数字格式**，例如：15 表示 15%，而不是 "15%" 或 "0.15"。数值范围通常在 0-100 之间
5. **止损价格必须是数字格式**，例如：120.5，而不是字符串
6. **目标价格必须是数字格式**，表示预期的目标价位，例如：150.5
7. **有效期必须是数字格式**，表示建议的有效天数，例如：7 表示7天有效期
8. **是否由止损机制触发必须是布尔值**，true 或 false
9. 如果没有明确的买卖建议，对应的字段为空数组
10. 根据投资者风险偏好筛选适合的股票
11. 买入列表中的股票应按投资价值从高到低排序，最有价值的股票排在最前面
12. 资金分配策略：单只股票投资金额不应超过总投资金额的一定比例（保守型不超过10%，平衡型不超过15%，进取型不超过20%）
13. 止损策略：建议设置合理的止损价格（如低于买入价格5-10%），以控制潜在亏损
14. 目标价格策略：基于技术分析和市场预期，设定合理的盈利目标价格
15. 有效期策略：根据市场波动性和分析可靠性，设定建议的有效期限（通常3-30天）
16. 【重要】决策一致性：必须严格遵守上述决策约束，特别是3小时和24小时窗口的限制，避免频繁交易
"""
                
            self.log_message("正在请求大模型以固定格式输出买卖信号...")
            formatted_output = qwen_engine.chat_with_llm(format_prompt)
            self.log_message(f"格式化输出结果:\n{formatted_output}")
                
            # 将大模型的格式化输出传递给解析函数
            llm_analysis = formatted_output
        except Exception as e:
            self.log_message(f"股票分析或大模型调用失败: {e}")
            return
            
        # 解析大模型推荐
        try:
            recommendations = self.parse_llm_recommendations(llm_analysis)
            self.log_message(f"解析后推荐: 买入 {recommendations['buy']}, 卖出 {recommendations['sell']}")
            
            # 保存决策历史记录
            decision_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'buy': recommendations.get('buy', []),
                'sell': recommendations.get('sell', [])
            }
            self.decision_history.append(decision_record)
            
            # 限制决策历史记录数量，只保留最近的10次决策
            if len(self.decision_history) > 10:
                self.decision_history = self.decision_history[-10:]
            
            # 如果没有推荐，跳过本次交易
            if not recommendations['buy'] and not recommendations['sell']:
                self.log_message("大模型未提供明确推荐，跳过本次交易")
                return
        except Exception as e:
            self.log_message(f"解析大模型推荐失败: {e}")
            return
            
        # 执行卖出操作（严格按照大模型建议）
        for stock in recommendations['sell']:
            code = stock['code']
            name = stock.get('name', hk_smart_money_tracker.WATCHLIST.get(code, code))
            reason = stock.get('reason', '未提供理由')
            stop_loss_triggered = stock.get('stop_loss_triggered', False)
            target_price = stock.get('target_price', None)
            validity_period = stock.get('validity_period', None)
            
            if code in hk_smart_money_tracker.WATCHLIST:
                # 记录卖出决策（无论是否成功执行）
                if code not in self.positions:
                    # 没有持仓，无法卖出，发送邮件通知
                    self.log_message(f"未持有 {name} ({code})，无法按大模型建议卖出")
                    
                    # 记录失败的交易 - 确保记录到交易历史中
                    amount = 0.0
                    actual_current_price = self.get_current_stock_price(code)
                    self.record_transaction('SELL', code, name, 0, 0.0, amount, reason, False, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
                    
                    # 发送无法卖出通知邮件
                    success = self.send_trading_notification(
                        notification_type="cannot_sell",
                        code=code,
                        name=name,
                        reason=reason,
                        stop_loss_triggered=stop_loss_triggered
                    )
                    if success:
                        self.log_message(f"已发送无法卖出通知邮件: {name} ({code})")
                    else:
                        self.log_message(f"发送无法卖出通知邮件失败: {name} ({code})")
                else:
                    # 卖出全部持仓
                    sell_pct = 1.0
                    self.log_message(f"按大模型建议卖出 {name} ({code})，理由: {reason}，止损触发: {stop_loss_triggered}")
                    self.sell_stock(code, name, sell_pct, reason, skip_decision_record=True, target_price=target_price, validity_period=validity_period)  # 已在决策阶段记录，跳过执行阶段的记录
                
        # 检查是否需要止损
        positions_to_check = list(self.positions.items())  # 创建副本以避免修改时出错
        for code, position in positions_to_check:
            if code in hk_smart_money_tracker.WATCHLIST:
                # 检查是否设置了止损价格
                if 'stop_loss_price' in position and position['stop_loss_price'] is not None:
                    current_price = self.get_current_stock_price(code)
                    if current_price is not None:
                        # 如果当前价格低于止损价格，则卖出
                        if current_price < position['stop_loss_price']:
                            name = hk_smart_money_tracker.WATCHLIST.get(code, code)
                            self.log_message(f"触发止损: {name} ({code}) 当前价格 HK${current_price:.2f} < 止损价格 HK${position['stop_loss_price']:.2f}")
                            # 发送止损通知邮件
                            self.send_trading_notification(
                                notification_type="stop_loss",
                                code=code,
                                name=name,
                                current_price=current_price,
                                stop_loss_price=position['stop_loss_price']
                            )
                            
                            # 执行止损卖出
                            self.sell_stock(code, name, 1.0, f"止损触发: 当前价格HK${current_price:.2f} < 止损价格HK${position['stop_loss_price']:.2f}")
        
        # 执行买入操作（严格按照大模型建议）
        for stock in recommendations['buy']:
            code = stock['code']
            name = stock.get('name', hk_smart_money_tracker.WATCHLIST.get(code, code))
            reason = stock.get('reason', '未提供理由')
            allocation_pct = stock.get('allocation_pct', '未提供')
            stop_loss_price = stock.get('stop_loss_price', '未提供')
            target_price = stock.get('target_price', None)
            validity_period = stock.get('validity_period', None)
            
            if code in hk_smart_money_tracker.WATCHLIST:
                # 检查是否提供了资金分配比例
                if allocation_pct == '未提供' or allocation_pct is None:
                    self.log_message(f"大模型未提供 {name} ({code}) 的资金分配比例，跳过买入")
                    # 发送无法买入通知邮件
                    self.send_trading_notification(
                        notification_type="buy_failed",
                        code=code,
                        name=name,
                        reason=reason,
                        allocation_pct=allocation_pct,
                        stop_loss_price=stop_loss_price
                    )
                    continue
                
                # 获取当前价格
                current_price = self.get_current_stock_price(code)
                if current_price is None:
                    self.log_message(f"无法获取 {name} ({code}) 的当前价格，跳过买入")
                    # 发送无法买入通知邮件
                    self.send_trading_notification(
                        notification_type="buy_failed",
                        code=code,
                        name=name,
                        reason=reason,
                        allocation_pct=allocation_pct,
                        stop_loss_price=stop_loss_price
                    )
                    continue
                
                # 记录买入决策尝试
                # 根据大模型建议的资金分配比例计算应买入的股数
                shares, required_amount, calculation_reason = self.calculate_shares_to_buy(code, name, allocation_pct, current_price)
                
                # 检查计算出的股数是否为0
                if shares <= 0:
                    self.log_message(f"按大模型建议计算出的买入股数为0，跳过买入 {name} ({code})，原因: {calculation_reason}")
                    
                    # 记录失败的交易（只保存大模型的原始建议，不附加失败原因）
                    amount = 0 * current_price if 0 > 0 else 0
                    actual_current_price = current_price if current_price is not None else self.get_current_stock_price(code)
                    self.record_transaction('BUY', code, name, 0, current_price, amount, reason, False, stop_loss_price=stop_loss_price, current_price=actual_current_price, target_price=target_price, validity_period=validity_period)
                    
                    # 根据计算原因决定是否发送资金不足通知
                    if calculation_reason == "资金不足":
                        # 只在真正资金不足时发送资金不足通知
                        self.send_trading_notification(
                            notification_type="insufficient_funds",
                            code=code,
                            name=name,
                            reason=reason,
                            allocation_pct=allocation_pct,
                            stop_loss_price=stop_loss_price,
                            required_amount=required_amount
                        )
                    elif calculation_reason == "持仓比例已达到建议值":
                        # 当持仓比例已达到建议值时，发送持仓比例已达上限通知
                        self.log_message(f"当前持仓价值已达到大模型建议的比例，无需买入 {name} ({code})")
                        self.send_trading_notification(
                            notification_type="max_position_reached",
                            code=code,
                            name=name,
                            reason=reason,
                            allocation_pct=allocation_pct,
                            stop_loss_price=stop_loss_price
                        )
                    else:
                        # 其他原因导致无法买入时，发送一般性的无法买入通知
                        self.send_trading_notification(
                            notification_type="buy_failed",
                            code=code,
                            name=name,
                            reason=reason,
                            allocation_pct=allocation_pct,
                            stop_loss_price=stop_loss_price
                        )
                    continue
                
                # 如果计算出的股数大于0，则执行买入操作
                self.buy_stock_by_shares(code, name, shares, reason, stop_loss_price, current_price, skip_decision_record=False, target_price=target_price, validity_period=validity_period)
                
        # 保存状态
        self.save_state()
        
        # 计算当前投资组合价值
        portfolio_value = self.get_portfolio_value()
        
        # 记录当前状态
        self.log_message(f"当前资金: HK${self.capital:,.2f}")
        self.log_message(f"投资组合价值: HK${portfolio_value:,.2f}")
        self.log_message(f"持仓情况: {self.positions}")
        
        # 计算收益率
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        self.log_message(f"总收益率: {total_return:.2f}%")
    
    def run_hourly_analysis(self):
        """按计划频率运行分析和交易"""
        self.log_message(f"开始每{self.analysis_frequency}分钟分析...")
        self.execute_trades()
        
    def run_daily_summary(self):
        """每日总结"""
        if not self.portfolio_history:
            return
            
        self.log_message("="*90)
        self.log_message("每日交易总结")
        self.log_message("="*90)
        
        # 计算当日收益
        if len(self.portfolio_history) >= 2:
            today_value = self.portfolio_history[-1]['portfolio_value']
            yesterday_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (today_value - yesterday_value) / yesterday_value * 100
            self.log_message(f"当日收益率: {daily_return:.2f}%")
            
        # 总体收益
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital * 100
        self.log_message(f"总体收益率: {total_return:.2f}%")
        
        # 持仓详情
        positions_info, total_stock_value = self.get_detailed_positions_info()
        
        if positions_info:
            self.log_message("")
            self.log_message("当前持仓详情:")
            self.log_message("-" * 120)
            self.log_message(f"{'股票代码':<12} {'股票名称':<12} {'持有数量':<12} {'平均成本':<10} {'当前价格':<10} {'止损价格':<10} {'持有金额':<15} {'盈亏金额':<15}")
            self.log_message("-" * 120)
            
            for pos in positions_info:
                profit_loss_str = f"{pos['profit_loss']:>+.2f}" if pos['profit_loss'] >= 0 else f"{pos['profit_loss']:>+.2f}"
                stop_loss_str = f"{pos['stop_loss_price']:>10.2f}" if pos['stop_loss_price'] is not None else "N/A"
                self.log_message(f"{pos['code']:<12} {pos['name']:<12} {pos['shares']:>12,} {pos['avg_price']:>10.2f} {pos['current_price']:>10.2f} {stop_loss_str:>10} {pos['position_value']:>15,.0f} {profit_loss_str:>15}")
            
            self.log_message("-" * 120)
            
            # 显示投资组合资金分配比例
            self.log_message("")
            self.log_message("投资组合资金分配比例:")
            self.log_message("-" * 60)
            portfolio_allocation = self.calculate_portfolio_allocation()
            if portfolio_allocation:
                self.log_message(f"{'股票代码':<12} {'股票名称':<12} {'持有金额':<15} {'分配比例':<10}")
                self.log_message("-" * 60)
                for code, info in portfolio_allocation.items():
                    self.log_message(f"{code:<12} {info['name']:<12} {info['value']:>15,.0f} {info['percentage']:>9.2f}%")
                self.log_message("-" * 60)
        
        self.log_message(f"{'现金余额:':<75} {self.capital:>15,.2f}")
        self.log_message(f"{'股票总价值:':<75} {total_stock_value:>15,.2f}")
        self.log_message(f"{'投资组合总价值:':<75} {self.capital + total_stock_value:>15,.2f}")
        self.log_message(f"{'可用资金:':<75} {self.capital:>15,.2f}")
        
        
    def get_detailed_positions_info(self):
        """获取详细的持仓信息，包括当前价格和持有金额"""
        try:
            # 获取所有持仓股票的当前价格
            current_prices = {}
            total_stock_value = 0
            
            # 股票代码列表
            stock_codes = []
            for code in self.positions.keys():
                stock_code = code.replace('.HK', '')
                stock_codes.append(stock_code)
                
            # 获取所有股票的当前价格
            for stock_code in stock_codes:
                try:
                    hist = get_hk_stock_data_tencent(stock_code, period_days=3)
                    if hist is not None and not hist.empty:
                        current_prices[stock_code] = hist['Close'].iloc[-1]
                    else:
                        current_prices[stock_code] = 0
                except:
                    current_prices[stock_code] = 0
            
            # 计算每只股票的持有金额和盈亏
            positions_info = []
            for code, pos in self.positions.items():
                shares = pos['shares']
                avg_price = pos['avg_price']
                
                # 获取当前价格
                stock_code = code.replace('.HK', '')
                current_price = current_prices.get(stock_code, 0)
                
                # 计算持有金额
                position_value = shares * current_price
                total_stock_value += position_value
                
                # 计算盈亏金额
                profit_loss = (current_price - avg_price) * shares
                
                # 获取股票名称
                name = hk_smart_money_tracker.WATCHLIST.get(code, code)
                
                # 获取止损价格（如果存在）
                stop_loss_price = self.positions[code].get('stop_loss_price', None)
                
                positions_info.append({
                    'code': code,
                    'name': name,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'profit_loss': profit_loss,
                    'stop_loss_price': stop_loss_price
                })
            
            return positions_info, total_stock_value
        except Exception as e:
            self.log_message(f"获取持仓详情失败: {e}")
            return [], 0

    def calculate_portfolio_allocation(self):
        """
        计算投资组合中各股票的资金分配比例
        
        Returns:
            dict: 各股票的资金分配比例
        """
        try:
            # 获取当前投资组合总价值
            portfolio_value = self.get_portfolio_value()
            if portfolio_value <= 0:
                return {}
            
            # 获取持仓详情
            positions_info, _ = self.get_detailed_positions_info()
            
            # 计算各股票的资金分配比例
            allocation = {}
            for pos in positions_info:
                code = pos['code']
                position_value = pos['position_value']
                allocation_pct = (position_value / portfolio_value) * 100
                allocation[code] = {
                    'name': pos['name'],
                    'value': position_value,
                    'percentage': allocation_pct
                }
            
            return allocation
        except Exception as e:
            self.log_message(f"计算投资组合资金分配比例失败: {e}")
            return {}

    def build_positions_detail(self):
        """
        构建持仓详情文本
        
        Returns:
            str: 格式化的持仓详情文本
        """
        # 获取详细的持仓信息
        positions_info, total_stock_value = self.get_detailed_positions_info()
        
        # 构建持仓详情文本
        if positions_info:
            positions_detail = "当前持仓详情:\n"
            positions_detail += "-" * 95 + "\n"
            positions_detail += f"{'股票代码':<12} {'股票名称':<12} {'持有数量':<12} {'平均成本':<10} {'当前价格':<10} {'止损价格':<10} {'持有金额':<15} {'盈亏金额':<15}\n"
            positions_detail += "-" * 95 + "\n"
            for pos in positions_info:
                profit_loss_str = f"{pos['profit_loss']:>+.2f}" if pos['profit_loss'] >= 0 else f"{pos['profit_loss']:>+.2f}"
                stop_loss_str = f"{pos['stop_loss_price']:>10.2f}" if pos['stop_loss_price'] is not None else "N/A"
                positions_detail += f"{pos['code']:<12} {pos['name']:<12} {pos['shares']:>12,} {pos['avg_price']:>10.2f} {pos['current_price']:>10.2f} {stop_loss_str:>10} {pos['position_value']:>15,.0f} {profit_loss_str:>15}\n"
            positions_detail += "-" * 95 + "\n"
            positions_detail += f"{'现金余额:':<80} {self.capital:>15,.2f}\n"
            positions_detail += f"{'股票总价值:':<80} {total_stock_value:>15,.2f}\n"
            positions_detail += f"{'投资组合总价值:':<80} {self.capital + total_stock_value:>15,.2f}\n"
        else:
            positions_detail = "暂无持仓\n"
        
        return positions_detail
    
    def save_transactions_to_csv(self):
        """将交易历史保存到CSV文件"""
        try:
            df_transactions = pd.DataFrame(self.transaction_history)
            df_transactions.to_csv(os.path.join(self.data_dir, "simulation_transactions.csv"), index=False, encoding="utf-8")
            # 不打印日志，避免重复输出
        except Exception as e:
            self.log_message(f"保存交易历史失败: {e}")
    
    def save_portfolio_to_csv(self):
        """将投资组合历史保存到CSV文件"""
        try:
            df_portfolio = pd.DataFrame(self.portfolio_history)
            df_portfolio.to_csv(os.path.join(self.data_dir, "simulation_portfolio.csv"), index=False, encoding="utf-8")
            # 不打印日志，避免重复输出
        except Exception as e:
            self.log_message(f"保存投资组合历史失败: {e}")
    
    def generate_final_report(self):
        """生成最终报告"""
        # 保存最终状态
        self.save_state()
        
        self.log_message("="*60)
        self.log_message("模拟交易最终报告")
        self.log_message("="*60)
        
        # 总体收益
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        self.log_message(f"初始资金: HK${self.initial_capital:,.2f}")
        self.log_message(f"最终价值: HK${final_value:,.2f}")
        self.log_message(f"总收益率: {total_return:.2f}%")
        
        # 交易统计
        buy_count = sum(1 for t in self.transaction_history if t['type'] == 'BUY')
        sell_count = sum(1 for t in self.transaction_history if t['type'] == 'SELL')
        self.log_message(f"总交易次数: {len(self.transaction_history)} (买入: {buy_count}, 卖出: {sell_count})")
        
        # 持仓情况
        self.log_message(f"最终持仓: {self.positions}")
        
        # 交易历史已在每次交易后保存，这里仅作最终确认
        self.log_message("模拟交易系统最终报告生成完成")
            
        # 投资组合历史已在每次交易后保存，这里仅作最终确认
        self.log_message("投资组合历史已保存到 data/simulation_portfolio.csv")

    def manual_sell_stock(self, code, percentage=1.0):
        """
        手工卖出股票
        
        Args:
            code (str): 股票代码
            percentage (float): 卖出比例，默认100%
        """
        # 从WATCHLIST获取股票名称，如果找不到则使用代码
        name = hk_smart_money_tracker.WATCHLIST.get(code, code)
        
        # 检查是否有持仓
        if code not in self.positions:
            self.log_message(f"未持有 {name} ({code})，无法卖出")
            
            # 记录失败的交易
            amount = 0 * 0 if 0 > 0 else 0
            actual_current_price = self.get_current_stock_price(code)
            self.record_transaction('SELL', code, name, 0, 0, amount, '手工卖出指令', False, current_price=actual_current_price)
            
            # 发送无法卖出通知邮件
            success = self.send_trading_notification(
                notification_type="manual_cannot_sell",
                code=code,
                name=name
            )
            if success:
                self.log_message(f"已发送手工无法卖出通知邮件: {name} ({code})")
            else:
                self.log_message(f"发送手工无法卖出通知邮件失败: {name} ({code})")
            return False
        
        return self.sell_stock(code, name, percentage, None)
    
    def test_email_notification(self):
        """测试邮件发送功能"""
        self.log_message("测试邮件发送功能...")
        
        success = self.send_trading_notification(notification_type="test")
        if success:
            self.log_message("邮件功能测试成功")
        else:
            self.log_message("邮件功能测试失败")
        return success

def run_simulation(duration_days=30, analysis_frequency=DEFAULT_ANALYSIS_FREQUENCY, investor_type="进取型"):
    """
    运行模拟交易
    
    Args:
        duration_days (int): 模拟天数，默认30天
        analysis_frequency (int): 分析频率（分钟），默认15分钟
        investor_type (str): 投资者风险偏好，默认为"进取型"
    """
    # 创建模拟交易器
    trader = SimulationTrader(initial_capital=1000000, analysis_frequency=analysis_frequency, investor_type=investor_type)
    
    # 记录开始信息
    trader.log_message(f"开始港股模拟交易，模拟周期: {duration_days} 天")
    trader.log_message("初始资金: HK$1,000,000")
    
    # 测试邮件功能
    trader.test_email_notification()
    
    # 启动时先执行一次交易分析
    trader.log_message("启动时执行首次交易分析...")
    trader.run_hourly_analysis()
    
    # 计划随机间隔执行交易分析（每15-60分钟之间随机）
    next_analysis_time = datetime.now() + timedelta(minutes=random.randint(15, 60))
    
    # 计划每天收盘后生成总结
    schedule.every().day.at("16:05").do(trader.run_daily_summary)
    
    # 模拟运行指定天数
    end_time = datetime.now() + timedelta(days=duration_days)
    
    try:
        while datetime.now() < end_time:
            # 检查是否到了随机交易时间
            if datetime.now() >= next_analysis_time:
                trader.run_hourly_analysis()
                # 设置下一次随机交易时间（15-60分钟之间）
                next_analysis_time = datetime.now() + timedelta(minutes=random.randint(15, 60))
                trader.log_message(f"下一次交易分析将在 {next_analysis_time.strftime('%Y-%m-%d %H:%M:%S')} 执行")
            
            # 运行计划任务（用于每日总结等）
            schedule.run_pending()
            
            # 每分钟检查一次
            time.sleep(60)
            
    except KeyboardInterrupt:
        trader.log_message("\n模拟交易被手动中断")
    finally:
        # 生成最终报告
        trader.generate_final_report()
        trader.log_message(f"模拟交易完成，详细日志请查看: {trader.get_daily_log_file()}")

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='港股模拟交易系统')
    parser.add_argument('--duration-days', type=int, default=90, help='模拟天数，默认90天')
    parser.add_argument('--analysis-frequency', type=int, default=DEFAULT_ANALYSIS_FREQUENCY, help=f'分析频率（分钟），默认{DEFAULT_ANALYSIS_FREQUENCY}分钟')
    parser.add_argument('--manual-sell', type=str, help='手工卖出股票代码（例如：0700.HK）')
    parser.add_argument('--sell-percentage', type=float, default=1.0, help='卖出比例（0.0-1.0），默认1.0（100%）')
    parser.add_argument('--investor-type', type=str, default='moderate', choices=['aggressive', 'moderate', 'conservative'], help='投资者风险偏好：aggressive(进取型)、moderate(稳健型)、conservative(保守型)，默认为稳健型')
    args = parser.parse_args()
    
    # 如果指定了手工卖出股票，则执行手工卖出
    if args.manual_sell:
        # 创建模拟交易器
        trader = SimulationTrader(initial_capital=1000000, analysis_frequency=DEFAULT_ANALYSIS_FREQUENCY, investor_type=args.investor_type)
        
        # 执行手工卖出
        success = trader.manual_sell_stock(args.manual_sell, args.sell_percentage)
        if success:
            trader.log_message(f"成功卖出 {args.manual_sell} ({args.sell_percentage*100:.1f}%)")
        else:
            trader.log_message(f"卖出 {args.manual_sell} 失败")
        exit(0)
    
    # 运行模拟交易
    run_simulation(duration_days=args.duration_days, analysis_frequency=args.analysis_frequency, investor_type=args.investor_type)
