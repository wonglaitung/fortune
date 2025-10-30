#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股模拟交易系统
基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易

新增功能：
1. 根据不同投资者类型（保守型、平衡型、进取型）自动进行盈亏比例交易
2. 根据市场情况自动建议买入股票
"""

import os
import sys
import time
import json
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
    def __init__(self, initial_capital=1000000, analysis_frequency=15):
        """
        初始化模拟交易系统
        
        Args:
            initial_capital (float): 初始资金，默认100万港元
            analysis_frequency (int): 分析频率（分钟），默认15分钟
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 持仓 {code: {'shares': 数量, 'avg_price': 平均买入价}}
        self.transaction_history = []  # 交易历史
        self.portfolio_history = []  # 投资组合价值历史
        self.start_date = datetime.now()
        self.is_trading_hours = True  # 模拟港股交易时间 (9:30-16:00)
        self.analysis_frequency = analysis_frequency  # 分析频率（分钟）
        
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
    
    def get_daily_log_file(self):
        """获取当天的日志文件路径"""
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.data_dir, f"simulation_trade_log_{today}.txt")
    
    def log_message(self, message):
        """记录交易日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
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
                
                # 恢复开始日期
                start_date_str = state.get('start_date')
                if start_date_str:
                    try:
                        self.start_date = datetime.fromisoformat(start_date_str)
                    except:
                        self.start_date = datetime.now()
                
                self.log_message(f"从文件恢复状态成功: {len(self.transaction_history)} 笔交易, {len(self.positions)} 个持仓")
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
    
    def get_current_positions_list(self):
        """
        获取当前持仓的股票代码和名称清单
        
        Returns:
            list: 包含股票代码和名称的字符串列表
        """
        positions_list = []
        for code, position in self.positions.items():
            # 从WATCHLIST获取股票名称，如果找不到则使用代码
            name = hk_smart_money_tracker.WATCHLIST.get(code, code)
            shares = position['shares']
            avg_price = position['avg_price']
            positions_list.append(f"{code} {name} ({shares}股 @ HK${avg_price:.2f})")
        return positions_list
    
    def buy_stock(self, code, name, amount_percentage=0.1):
        """
        买入股票
        
        Args:
            code (str): 股票代码
            name (str): 股票名称
            amount_percentage (float): 投资金额占可用资金的比例，默认10%
        """
        # 检查是否在交易时间
        if not self.is_trading_hours:
            self.log_message(f"非交易时间，跳过买入 {name} ({code})")
            return False
            
        current_price = self.get_current_stock_price(code)
        if current_price is None:
            self.log_message(f"无法获取 {name} ({code}) 的当前价格，跳过买入")
            return False
            
        # 计算可投资金额
        invest_amount = self.capital * amount_percentage
        if invest_amount <= 0:
            self.log_message(f"资金不足，无法买入 {name} ({code})")
            return False
            
        # 计算可买入股数（考虑手续费）
        shares = int(invest_amount / current_price / 100) * 100  # 以100股为单位
        if shares <= 0:
            self.log_message(f"资金不足以买入100股 {name} ({code})")
            return False
            
        # 计算实际投资金额
        actual_invest = shares * current_price
        
        # 检查是否有足够资金
        if actual_invest > self.capital:
            self.log_message(f"资金不足买入 {shares} 股 {name} ({code})")
            return False
            
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
        else:
            # 新建持仓
            self.positions[code] = {
                'shares': shares,
                'avg_price': current_price
            }
            
        # 记录交易
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': 'BUY',
            'code': code,
            'name': name,
            'shares': shares,
            'price': current_price,
            'amount': actual_invest,
            'capital_after': self.capital
        }
        self.transaction_history.append(transaction)
        
        # 保存状态
        self.save_state()
        
        self.log_message(f"买入 {shares} 股 {name} ({code}) @ HK${current_price:.2f}, 总金额: HK${actual_invest:.2f}")
        
        # 如果是新买入的股票，发送邮件通知
        if is_new_stock:
            # 发送买入通知邮件
            # 构建持仓详情文本
            positions_detail = self.build_positions_detail()
            
            subject = f"【买入通知】{name} ({code})"
            content = f"""
模拟交易系统买入通知：

股票名称：{name}
股票代码：{code}
买入价格：HK${current_price:.2f}
买入数量：{shares} 股
买入金额：HK${actual_invest:.2f}
交易时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
            self.send_email_notification(subject, content)
        
        return True
    
    def sell_stock(self, code, name, percentage=1.0):
        """
        卖出股票
        
        Args:
            code (str): 股票代码
            name (str): 股票名称
            percentage (float): 卖出比例，默认100%
        """
        # 检查是否在交易时间
        if not self.is_trading_hours:
            self.log_message(f"非交易时间，跳过卖出 {name} ({code})")
            return False
            
        # 检查是否有持仓
        if code not in self.positions:
            self.log_message(f"未持有 {name} ({code})，无法卖出")
            return False
            
        position = self.positions[code]
        current_price = self.get_current_stock_price(code)
        if current_price is None:
            self.log_message(f"无法获取 {name} ({code}) 的当前价格，跳过卖出")
            return False
            
        # 计算卖出股数
        shares_to_sell = int(position['shares'] * percentage)
        if shares_to_sell <= 0:
            self.log_message(f"卖出股数为0，跳过卖出 {name} ({code})")
            return False
            
        # 计算卖出金额
        sell_amount = shares_to_sell * current_price
        
        # 执行卖出
        self.capital += sell_amount
        
        # 更新持仓
        self.positions[code]['shares'] -= shares_to_sell
        if self.positions[code]['shares'] <= 0:
            del self.positions[code]
            
        # 记录交易
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': 'SELL',
            'code': code,
            'name': name,
            'shares': shares_to_sell,
            'price': current_price,
            'amount': sell_amount,
            'capital_after': self.capital
        }
        self.transaction_history.append(transaction)
        
        # 保存状态
        self.save_state()
        
        self.log_message(f"卖出 {shares_to_sell} 股 {name} ({code}) @ HK${current_price:.2f}, 总金额: HK${sell_amount:.2f}")
        
        # 发送卖出通知邮件
        # 构建持仓详情文本
        positions_detail = self.build_positions_detail()
        
        subject = f"【卖出通知】{name} ({code})"
        content = f"""
模拟交易系统卖出通知：

股票名称：{name}
股票代码：{code}
卖出价格：HK${current_price:.2f}
卖出数量：{shares_to_sell} 股
卖出金额：HK${sell_amount:.2f}
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
        """
        self.send_email_notification(subject, content)
        
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
            dict: 解析后的推荐结果 {'buy': [股票代码列表], 'sell': [股票代码列表]}
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
                    # 验证股票代码是否在自选股列表中
                    buy_codes = [code for code in parsed_json['buy'] if code in hk_smart_money_tracker.WATCHLIST]
                    sell_codes = [code for code in parsed_json['sell'] if code in hk_smart_money_tracker.WATCHLIST and code in self.positions]
                    
                    recommendations['buy'] = buy_codes
                    recommendations['sell'] = sell_codes
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
        investor_type = "进取型"  # 可以根据需要调整为"保守型"或"平衡型"
        
        # 检查持仓股票的盈亏情况，根据投资者类型决定是否自动卖出
        self.check_positions_for_auto_trade(investor_type)
        
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
            llm_prompt = hk_smart_money_tracker.build_llm_analysis_prompt(results)
            
            # 调用大模型分析（真实调用）
            self.log_message("正在调用大模型分析...")
            try:
                # 导入大模型服务
                from llm_services import qwen_engine
                llm_analysis = qwen_engine.chat_with_llm(llm_prompt)
                self.log_message("大模型分析调用成功")
                self.log_message(f"大模型分析结果:\n{llm_analysis}")
                
                # 再次调用大模型，要求以固定格式输出买卖信号
                format_prompt = f"""
请分析以下港股分析报告，考虑投资者风险偏好为{investor_type}，并严格按照以下JSON格式输出买卖信号：

报告内容：
{llm_analysis}

投资者风险偏好：{investor_type}
- 保守型：偏好低风险、稳定收益的股票，如高股息银行股
- 平衡型：平衡风险与收益，兼顾价值与成长
- 进取型：偏好高风险、高收益的股票，如科技成长股

请严格按照以下格式输出：
{{
    \"buy\": [\"股票代码1\", \"股票代码2\", ...],
    \"sell\": [\"股票代码3\", \"股票代码4\", ...]
}}

要求：
1. 只输出JSON格式，不要包含其他文字
2. \"buy\"字段包含建议买入的股票代码列表
3. \"sell\"字段包含建议卖出的股票代码列表
4. 如果没有明确的买卖建议，对应的字段为空数组
5. 只包含在自选股列表中的股票代码：{list(hk_smart_money_tracker.WATCHLIST.keys())}
6. 根据投资者风险偏好筛选适合的股票
7. 注意：系统还会根据盈亏比例自动进行交易，无需在建议中包含已经达到盈亏阈值的股票
"""
                
                self.log_message("正在请求大模型以固定格式输出买卖信号...")
                formatted_output = qwen_engine.chat_with_llm(format_prompt)
                self.log_message(f"格式化输出结果:\n{formatted_output}")
                
                # 将大模型的格式化输出传递给解析函数
                llm_analysis = formatted_output
            except Exception as e:
                self.log_message(f"大模型分析调用失败: {e}")
                self.log_message("由于大模型分析失败，跳过本次交易决策")
                return
            
        except Exception as e:
            self.log_message(f"股票分析或大模型调用失败: {e}")
            return
            
        # 解析大模型推荐
        try:
            recommendations = self.parse_llm_recommendations(llm_analysis)
            self.log_message(f"解析后推荐: 买入 {recommendations['buy']}, 卖出 {recommendations['sell']}")
            
            # 如果没有推荐，跳过本次交易
            if not recommendations['buy'] and not recommendations['sell']:
                self.log_message("大模型未提供明确推荐，跳过本次交易")
                return
        except Exception as e:
            self.log_message(f"解析大模型推荐失败: {e}")
            return
            
        # 执行卖出操作（严格按照大模型建议）
        # 注意：这里不包括自动卖出的股票，自动卖出在check_positions_for_auto_trade中处理
        for code in recommendations['sell']:
            if code in hk_smart_money_tracker.WATCHLIST:
                name = hk_smart_money_tracker.WATCHLIST[code]
                # 检查是否持有该股票
                if code not in self.positions:
                    # 没有持仓，无法卖出，发送邮件通知
                    self.log_message(f"未持有 {name} ({code})，无法按大模型建议卖出")
                    # 发送无法卖出通知邮件
                    # 构建持仓详情文本
                    positions_detail = self.build_positions_detail()
                    
                    subject = f"【无法卖出通知】{name} ({code})"
                    content = f"""
模拟交易系统无法按大模型建议卖出通知：

股票名称：{name}
股票代码：{code}
无法卖出原因：未持有该股票
交易时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{positions_detail}
                    """
                    self.send_email_notification(subject, content)
                else:
                    # 卖出全部持仓
                    sell_pct = 1.0
                    self.sell_stock(code, name, sell_pct)
                
        # 执行买入操作（严格按照大模型建议）
        # 根据推荐的股票数量动态调整投资比例，确保不会超过总资金
        buy_count = len(recommendations['buy'])
        if buy_count > 0:
            # 计算每只股票的投资比例
            investment_pct = self.calculate_investment_percentage(investor_type, buy_count)
            
            # 将买入股票分为两类：没有持仓的股票和已有持仓的股票
            new_stocks = [code for code in recommendations['buy'] if code not in self.positions]
            existing_stocks = [code for code in recommendations['buy'] if code in self.positions]
            
            # 优先买入没有持仓的股票
            ordered_buy_list = new_stocks + existing_stocks
            
            for code in ordered_buy_list:
                if code in hk_smart_money_tracker.WATCHLIST:
                    name = hk_smart_money_tracker.WATCHLIST[code]
                    # 尝试买入股票
                    buy_result = self.buy_stock(code, name, investment_pct)
                    # 如果买入失败，发送邮件通知
                    if not buy_result:
                        self.log_message(f"无法按大模型建议买入 {name} ({code})，发送邮件通知")
                        # 发送无法买入通知邮件
                        # 构建持仓详情文本
                        positions_detail = self.build_positions_detail()
                        
                        subject = f"【无法买入通知】{name} ({code})"
                        content = f"""
模拟交易系统无法按大模型建议买入通知：

股票名称：{name}
股票代码：{code}
无法买入原因：资金不足或其他原因
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前资金：HK${self.capital:,.2f}

{positions_detail}
                        """
                        self.send_email_notification(subject, content)
        
        
                
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
        
        # 记录当前状态
        self.log_message(f"当前资金: HK${self.capital:,.2f}")
        self.log_message(f"投资组合价值: HK${portfolio_value:,.2f}")
        self.log_message(f"持仓情况: {self.positions}")
        
        # 计算收益率
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        self.log_message(f"总收益率: {total_return:.2f}%")
    
    def calculate_investment_percentage(self, investor_type, buy_count):
        """
        根据投资者类型和推荐买入股票数量计算每只股票的投资比例
        
        Args:
            investor_type (str): 投资者类型 ("保守型", "平衡型", "进取型")
            buy_count (int): 推荐买入的股票数量
            
        Returns:
            float: 每只股票的投资比例
        """
        # 根据投资者类型调整风险偏好
        # 保守型：总投资不超过30%，单只股票不超过5%
        # 平衡型：总投资不超过50%，单只股票不超过10%
        # 进取型：总投资不超过80%，单只股票不超过20%
        if investor_type == "保守型":
            max_total_pct = 0.3
            max_single_pct = 0.05
        elif investor_type == "平衡型":
            max_total_pct = 0.5
            max_single_pct = 0.1
        else:  # 进取型
            max_total_pct = 0.8
            max_single_pct = 0.2
        
        # 平均分配资金
        investment_pct = min(max_total_pct / buy_count, max_single_pct) if buy_count > 0 else 0
        return investment_pct
    
    def check_positions_for_auto_trade(self, investor_type):
        """
        根据投资者类型和盈亏比例检查持仓，决定是否自动买入或卖出
        
        Args:
            investor_type (str): 投资者类型 ("保守型", "平衡型", "进取型")
        """
        # 定义不同投资者类型的盈亏比例阈值
        if investor_type == "保守型":
            # 保守型：亏损超过5%卖出，盈利超过10%卖出
            loss_threshold = -0.05  # 亏损5%
            profit_threshold = 0.10  # 盈利10%
        elif investor_type == "平衡型":
            # 平衡型：亏损超过10%卖出，盈利超过15%卖出
            loss_threshold = -0.10  # 亏损10%
            profit_threshold = 0.15  # 盈利15%
        else:  # 进取型
            # 进取型：亏损超过15%卖出，盈利超过20%卖出
            loss_threshold = -0.15  # 亏损15%
            profit_threshold = 0.20  # 盈利20%
        
        # 检查持仓中的股票
        for code, position in list(self.positions.items()):
            # 获取当前价格
            current_price = self.get_current_stock_price(code)
            if current_price is None:
                continue
                
            # 计算盈亏比例
            avg_price = position['avg_price']
            profit_loss_ratio = (current_price - avg_price) / avg_price
            
            # 检查是否达到卖出条件
            if profit_loss_ratio <= loss_threshold or profit_loss_ratio >= profit_threshold:
                name = hk_smart_money_tracker.WATCHLIST.get(code, code)
                self.log_message(f"{name} ({code}) 盈亏比例: {profit_loss_ratio:.2%}，达到{investor_type}投资者的交易阈值")
                
                if profit_loss_ratio >= profit_threshold:
                    # 盈利达到阈值，卖出获利
                    self.log_message(f"自动卖出 {name} ({code}) - 盈利达到阈值 {profit_threshold:.2%}")
                    # 发送自动卖出通知邮件，明确说明是基于盈亏比例的自动交易
                    self.send_auto_trade_notification(name, code, 'SELL', current_price, position['shares'], 
                                                     f"基于{investor_type}投资者盈亏比例策略自动卖出 - 盈利达到阈值 {profit_threshold:.2%}")
                    self.sell_stock(code, name, 1.0)
                elif profit_loss_ratio <= loss_threshold:
                    # 亏损达到阈值，止损卖出
                    self.log_message(f"自动卖出 {name} ({code}) - 亏损达到止损阈值 {loss_threshold:.2%}")
                    # 发送自动卖出通知邮件，明确说明是基于盈亏比例的自动交易
                    self.send_auto_trade_notification(name, code, 'SELL', current_price, position['shares'], 
                                                     f"基于{investor_type}投资者盈亏比例策略自动卖出 - 亏损达到止损阈值 {loss_threshold:.2%}")
                    self.sell_stock(code, name, 1.0)
    
    
    
    def send_auto_trade_notification(self, name, code, trade_type, price, shares, reason):
        """
        发送自动交易通知邮件
        
        Args:
            name (str): 股票名称
            code (str): 股票代码
            trade_type (str): 交易类型 ('BUY' 或 'SELL')
            price (float): 交易价格
            shares (int): 交易股数
            reason (str): 交易原因
        """
        # 构建持仓详情文本
        positions_detail = self.build_positions_detail()
        
        if trade_type == 'BUY':
            subject = f"【自动买入通知】{name} ({code}) - {reason}"
            content = f"""
模拟交易系统自动买入通知：

股票名称：{name}
股票代码：{code}
买入价格：HK${price:.2f}
买入数量：{shares} 股
买入金额：HK${price * shares:.2f}
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
交易原因：{reason}

当前资金：HK${self.capital:,.2f}

{positions_detail}
            """
        else:  # SELL
            subject = f"【自动卖出通知】{name} ({code}) - {reason}"
            content = f"""
模拟交易系统自动卖出通知：

股票名称：{name}
股票代码：{code}
卖出价格：HK${price:.2f}
卖出数量：{shares} 股
卖出金额：HK${price * shares:.2f}
交易时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
交易原因：{reason}

当前资金：HK${self.capital:,.2f}

{positions_detail}
        """
        
        self.send_email_notification(subject, content)
    
    def run_hourly_analysis(self):
        """按计划频率运行分析和交易"""
        self.log_message(f"开始每{self.analysis_frequency}分钟分析...")
        self.execute_trades()
        
    def run_daily_summary(self):
        """每日总结"""
        if not self.portfolio_history:
            return
            
        self.log_message("="*80)
        self.log_message("每日交易总结")
        self.log_message("="*80)
        
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
            self.log_message("-" * 100)
            self.log_message(f"{'股票代码':<12} {'股票名称':<12} {'持有数量':<12} {'平均成本':<10} {'当前价格':<10} {'持有金额':<15} {'盈亏金额':<15}")
            self.log_message("-" * 100)
            
            for pos in positions_info:
                profit_loss_str = f"{pos['profit_loss']:>+.2f}" if pos['profit_loss'] >= 0 else f"{pos['profit_loss']:>+.2f}"
                self.log_message(f"{pos['code']:<12} {pos['name']:<12} {pos['shares']:>12,} {pos['avg_price']:>10.2f} {pos['current_price']:>10.2f} {pos['position_value']:>15,.0f} {profit_loss_str:>15}")
            
            self.log_message("-" * 100)
        
        self.log_message(f"{'现金余额:':<65} {self.capital:>15,.2f}")
        self.log_message(f"{'股票总价值:':<65} {total_stock_value:>15,.2f}")
        self.log_message(f"{'投资组合总价值:':<65} {self.capital + total_stock_value:>15,.2f}")
        self.log_message(f"{'可用资金:':<65} {self.capital:>15,.2f}")
        
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
                
                positions_info.append({
                    'code': code,
                    'name': name,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'profit_loss': profit_loss
                })
            
            return positions_info, total_stock_value
        except Exception as e:
            self.log_message(f"获取持仓详情失败: {e}")
            return [], 0

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
            positions_detail += "-" * 85 + "\n"
            positions_detail += f"{'股票代码':<12} {'股票名称':<12} {'持有数量':<12} {'平均成本':<10} {'当前价格':<10} {'持有金额':<15} {'盈亏金额':<15}\n"
            positions_detail += "-" * 85 + "\n"
            for pos in positions_info:
                profit_loss_str = f"{pos['profit_loss']:>+.2f}" if pos['profit_loss'] >= 0 else f"{pos['profit_loss']:>+.2f}"
                positions_detail += f"{pos['code']:<12} {pos['name']:<12} {pos['shares']:>12,} {pos['avg_price']:>10.2f} {pos['current_price']:>10.2f} {pos['position_value']:>15,.0f} {profit_loss_str:>15}\n"
            positions_detail += "-" * 85 + "\n"
            positions_detail += f"{'现金余额:':<70} {self.capital:>15,.2f}\n"
            positions_detail += f"{'股票总价值:':<70} {total_stock_value:>15,.2f}\n"
            positions_detail += f"{'投资组合总价值:':<70} {self.capital + total_stock_value:>15,.2f}\n"
        else:
            positions_detail = "暂无持仓\n"
        
        return positions_detail
    
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
        
        # 保存交易历史到文件
        try:
            df_transactions = pd.DataFrame(self.transaction_history)
            df_transactions.to_csv(os.path.join(self.data_dir, "simulation_transactions.csv"), index=False, encoding="utf-8")
            self.log_message("交易历史已保存到 data/simulation_transactions.csv")
        except Exception as e:
            self.log_message(f"保存交易历史失败: {e}")
            
        # 保存投资组合历史到文件
        try:
            df_portfolio = pd.DataFrame(self.portfolio_history)
            df_portfolio.to_csv(os.path.join(self.data_dir, "simulation_portfolio.csv"), index=False, encoding="utf-8")
            self.log_message("投资组合历史已保存到 data/simulation_portfolio.csv")
        except Exception as e:
            self.log_message(f"保存投资组合历史失败: {e}")

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
            return False
        
        return self.sell_stock(code, name, percentage)
    
    def test_email_notification(self):
        """测试邮件发送功能"""
        self.log_message("测试邮件发送功能...")
        # 构建持仓详情文本
        positions_detail = self.build_positions_detail()
        
        subject = "港股模拟交易系统 - 邮件功能测试"
        content = f"""
这是港股模拟交易系统的邮件功能测试邮件。

时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
系统状态：
- 初始资金: HK${self.initial_capital:,.2f}
- 当前资金: HK${self.capital:,.2f}
- 持仓数量: {len(self.positions)}

当前持仓详情：
{positions_detail}
        """
        
        success = self.send_email_notification(subject, content)
        if success:
            self.log_message("邮件功能测试成功")
        else:
            self.log_message("邮件功能测试失败")
        return success

def run_simulation(duration_days=30, analysis_frequency=15):
    """
    运行模拟交易
    
    Args:
        duration_days (int): 模拟天数，默认30天
        analysis_frequency (int): 分析频率（分钟），默认15分钟
    """
    print(f"开始港股模拟交易，模拟周期: {duration_days} 天")
    print("初始资金: HK$1,000,000")
    
    # 创建模拟交易器
    trader = SimulationTrader(initial_capital=1000000, analysis_frequency=analysis_frequency)
    
    # 测试邮件功能
    trader.test_email_notification()
    
    # 计划按指定频率执行交易分析
    schedule.every(analysis_frequency).minutes.do(trader.run_hourly_analysis)
    
    # 计划每天收盘后生成总结
    schedule.every().day.at("16:05").do(trader.run_daily_summary)
    
    # 模拟运行指定天数
    end_time = datetime.now() + timedelta(days=duration_days)
    
    try:
        while datetime.now() < end_time:
            # 运行计划任务
            schedule.run_pending()
            
            # 每分钟检查一次
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n模拟交易被手动中断")
    finally:
        # 生成最终报告
        trader.generate_final_report()
        print(f"模拟交易完成，详细日志请查看: {trader.log_file}")

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='港股模拟交易系统')
    parser.add_argument('--duration-days', type=int, default=90, help='模拟天数，默认90天')
    parser.add_argument('--analysis-frequency', type=int, default=15, help='分析频率（分钟），默认15分钟')
    parser.add_argument('--manual-sell', type=str, help='手工卖出股票代码（例如：0700.HK）')
    parser.add_argument('--sell-percentage', type=float, default=1.0, help='卖出比例（0.0-1.0），默认1.0（100%）')
    args = parser.parse_args()
    
    # 如果指定了手工卖出股票，则执行手工卖出
    if args.manual_sell:
        # 创建模拟交易器
        trader = SimulationTrader(initial_capital=1000000, analysis_frequency=15)
        
        # 执行手工卖出
        success = trader.manual_sell_stock(args.manual_sell, args.sell_percentage)
        if success:
            print(f"成功卖出 {args.manual_sell} ({args.sell_percentage*100:.1f}%)")
        else:
            print(f"卖出 {args.manual_sell} 失败")
        exit(0)
    
    # 运行模拟交易
    run_simulation(duration_days=args.duration_days, analysis_frequency=args.analysis_frequency)
