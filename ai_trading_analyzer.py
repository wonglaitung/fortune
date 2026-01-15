#!/usr/bin/env python3
"""
人工智能股票交易盈利能力分析器

基于交叉验证后的算法，分析AI推荐的股票交易策略的盈利能力。

本版本改进：
- 增加了更合理的回报率计算：
  - ROI（总投入回报率）
  - XIRR（考虑现金流时间的年化收益率）
  - TWR / 等效的基于净值序列的年化收益（用于评估策略表现）
  - 风险指标：最大回撤、年化波动率、夏普比率（假设无风险利率为0）
- 生成现金流列表、净值时间序列，用于这些计算
- 在报告中输出上述指标

指标含义与假设：
1. 初始资本设定：200万港元（可配置）
2. 资金分配策略：每只股票分配初始资本的固定比例（默认10%）
3. 买入规则：
   - 按资金分配比例计算应买入金额
   - 如果现金不足，使用所有可用现金
   - 确保总投入不超过初始资本
4. 卖出规则：清仓全部持仓，释放现金用于后续买入
5. 持仓市值可以因盈利而超过初始资本
6. 交易成本：包含佣金、印花税、平台费等港股标准费率
7. ROI（总投入回报率）= 总盈亏 / 初始资本
8. XIRR（基于现金流的内部收益率）
   - 计算方法：使用二分法求解使现金流净现值为0的折现率
   - 假设：所有现金流按时间连续复利计算
   - 注意：对于短时间周期（<30天），年化值可能不稳定
   - 基数：现金流序列（买入为负，卖出为正）
9. TWR（时间加权回报）
   - 计算方法：基于每日净值序列的复合收益率
   - 假设：每日收益独立，不考虑资金流入流出影响
   - 基数：初始资本
10. 最大回撤
    - 计算方法：净值从峰值到谷底的最大跌幅
    - 假设：衡量策略的最大潜在损失
    - 基数：净值序列
11. 年化波动率
    - 计算方法：日收益率标准差 × √252（交易日）
    - 假设：收益率服从正态分布
    - 基数：日收益率序列
12. 夏普比率
    - 计算方法：年化收益率 / 年化波动率
    - 假设：无风险利率为0
    - 基数：年化收益率和年化波动率

重要假设与限制：
- 初始资本默认为200万港元（可配置）
- 单只股票资金分配比例默认为10%（可配置）
- 买入时确保总投入不超过初始资本
- 卖出后释放现金，可用于后续买入
- 持仓市值可以因盈利而超过初始资本
- 考虑交易手续费和印花税
- 不考虑股息收入
- 排除价格为0的异常交易
- XIRR对于短时间周期（<30天）可能不稳定
- 净值序列基于初始资本+累计盈亏计算
"""

import pandas as pd
import argparse
import sys
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import math
import re
import time
from tencent_finance import get_hk_stock_info_tencent


class AITradingAnalyzer:
    """AI股票交易分析器"""
    
    def __init__(self, csv_file: str = 'data/simulation_transactions.csv', 
                 initial_capital: float = 2000000.0,
                 allocation_pct: float = 10.0):
        """
        初始化分析器
        
        Args:
            csv_file: 交易记录CSV文件路径
            initial_capital: 初始资本（港元），默认200万港元
            allocation_pct: 单只股票的资金分配比例（百分比），默认10%
        """
        self.csv_file = csv_file
        self.df = None
        self.excluded_stocks = set()
        self.initial_capital = initial_capital
        self.allocation_pct = allocation_pct
    
    def calculate_shares(self, price: float, allocation_pct: float = 10.0) -> int:
        """
        计算可买入的股数（以100股为倍数）
        
        Args:
            price: 股价（港元）
            allocation_pct: 资金分配比例（百分比），例如15表示15%
            
        Returns:
            可买入的股数（100股的倍数），如果资金不足则返回0
        """
        shares_per_lot = 100  # 港股每手100股
        
        # 基于初始资本的资金分配
        target_investment = self.initial_capital * (allocation_pct / 100.0)
        
        # 计算可买手数
        lots = int(target_investment / (price * shares_per_lot))
        
        # 如果资金不足，返回0
        if lots <= 0:
            return 0
        
        return lots * shares_per_lot
    
    def calculate_transaction_cost(self, amount: float, is_sell: bool = False) -> float:
        """
        计算交易成本（港股标准费率）
        
        Args:
            amount: 交易金额（港元）
            is_sell: 是否为卖出交易（卖出需要印花税）
            
        Returns:
            总交易成本（港元）
        """
        # 港股交易成本标准费率
        commission_rate = 0.001  # 佣金 0.1%
        stamp_duty_rate = 0.0013  # 印花税 0.13%（仅卖出时收取）
        platform_fee = 50.0  # 平台费（固定费用，每笔交易）
        trading_fee_rate = 0.00005  # 交易费 0.005%
        settlement_fee_rate = 0.00002  # 交收费 0.002%
        
        # 计算各项费用
        commission = amount * commission_rate
        stamp_duty = amount * stamp_duty_rate if is_sell else 0.0
        trading_fee = amount * trading_fee_rate
        settlement_fee = amount * settlement_fee_rate
        
        # 佣金和交易费有最低收费（每笔至少HK$100）
        min_commission = 100.0
        commission = max(commission, min_commission)
        
        # 交易费最低HK$5
        min_trading_fee = 5.0
        trading_fee = max(trading_fee, min_trading_fee)
        
        # 总成本
        total_cost = commission + stamp_duty + platform_fee + trading_fee + settlement_fee
        
        return total_cost
    
    def send_email_notification(self, subject: str, content: str) -> bool:
        """
        发送邮件通知
        
        Args:
            subject (str): 邮件主题
            content (str): 邮件内容
            
        Returns:
            bool: 发送成功返回True，失败返回False
        """
        try:
            smtp_server = os.environ.get("YAHOO_SMTP", "smtp.163.com")
            smtp_user = os.environ.get("YAHOO_EMAIL")
            smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
            sender_email = smtp_user

            if not smtp_user or not smtp_pass:
                print("警告: 缺少 YAHOO_EMAIL 或 YAHOO_APP_PASSWORD 环境变量，无法发送邮件")
                return False

            recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
            recipients = [r.strip() for r in recipient_env.split(',')] if ',' in recipient_env else [recipient_env]

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            # 只添加HTML内容（支持颜色显示）
            html_content = self._format_text_to_html(content)
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(html_part)

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
                    
                    print("✅ 邮件发送成功！")
                    return True
                except Exception as e:
                    print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试，等待后重试
                        time.sleep(5)
            
            print("❌ 邮件发送失败，已尝试3次")
            return False
        except Exception as e:
            print(f"❌ 邮件发送过程中发生错误: {e}")
            return False
    
    def _format_text_to_html(self, text: str) -> str:
        """
        将文本内容转换为HTML格式，并为盈亏添加颜色
        
        Args:
            text: 纯文本内容
            
        Returns:
            HTML格式的内容
        """
        lines = text.split('\n')
        html_lines = []
        
        for line in lines:
            # 转义HTML特殊字符
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # 识别盈亏并添加颜色
            # 匹配格式：盈亏HK$X,XXX.XX (X.XX%) 或 盈亏HK$-X,XXX.XX (-X.XX%)
            pattern = r'(盈亏HK\$-?[\d,]+\.?\d*)\s*\(([-\d.]+)%\)'
            
            def add_profit_color(match):
                value = match.group(2)
                try:
                    v = float(value)
                except:
                    v = 0.0
                if v >= 0:
                    # 盈利用绿色
                    return f'<span style="color: green; font-weight: bold;">{match.group(1)} ({value}%)</span>'
                else:
                    # 亏损用红色
                    return f'<span style="color: red; font-weight: bold;">{match.group(1)} ({value}%)</span>'
            
            line = re.sub(pattern, add_profit_color, line)
            
            # 识别总体盈亏并添加颜色
            pattern2 = r'(总体盈亏:\s*HK\$-?[\d,]+\.?\d*)'
            
            def add_total_profit_color(match):
                value_str = match.group(1).replace('总体盈亏:', '').replace('HK$', '').replace(',', '').strip()
                try:
                    value = float(value_str)
                    if value >= 0:
                        return f'<span style="color: green; font-weight: bold;">{match.group(1)}</span>'
                    else:
                        return f'<span style="color: red; font-weight: bold;">{match.group(1)}</span>'
                except:
                    return match.group(0)
            
            line = re.sub(pattern2, add_total_profit_color, line)
            
            # 识别已实现盈亏和未实现盈亏并添加颜色
            pattern3 = r'(已实现盈亏:\s*HK\$-?[\d,]+\.?\d*)|(未实现盈亏:\s*HK\$-?[\d,]+\.?\d*)'
            
            def add_component_profit_color(match):
                text0 = match.group(0)
                try:
                    value_str = text0.split('HK$')[1].replace(',', '').strip()
                    value = float(value_str)
                    if value >= 0:
                        return f'<span style="color: green;">{text0}</span>'
                    else:
                        return f'<span style="color: red;">{text0}</span>'
                except:
                    return text0
            
            line = re.sub(pattern3, add_component_profit_color, line)
            
            html_lines.append(line)
        
        # 包装在HTML标签中
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Courier New', Courier, monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                }}
            </style>
        </head>
        <body>
        {'<br/>'.join(html_lines)}
        </body>
        </html>
        """
        
        return html_content
        
    def load_transactions(self) -> bool:
        """
        加载交易记录
        
        Returns:
            加载成功返回True，失败返回False
        """
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            return True
        except Exception as e:
            print(f"错误：无法加载交易记录文件 - {e}")
            return False
    
    def filter_transactions(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        按日期范围过滤交易记录
        
        Args:
            start_date: 起始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        Returns:
            过滤后的交易记录DataFrame
        """
        df = self.df.copy()
        
        # 如果没有指定起始日期，使用最早的交易日期
        if start_date is None:
            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        
        # 如果没有指定结束日期，使用最新的交易日期
        if end_date is None:
            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        
        # 转换为日期时间并过滤
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date + ' 23:59:59')
        
        return df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
    
    def identify_excluded_stocks(self, df: pd.DataFrame) -> set:
        """
        识别需要排除的异常股票（现价为0）
        
        Args:
            df: 交易记录DataFrame
            
        Returns:
            需要排除的股票代码集合
        """
        excluded = set()
        all_stocks = df['code'].unique()
        
        for stock_code in all_stocks:
            stock_trades = df[df['code'] == stock_code]
            if not stock_trades.empty:
                latest_record = stock_trades.iloc[-1]
                # 优先使用current_price，如果为空则使用price
                latest_price = latest_record['current_price']
                if pd.isna(latest_price):
                    latest_price = latest_record['price']
                if latest_price <= 0:
                    excluded.add(stock_code)
        
        return excluded
    
    def detect_abnormal_cashflows(self, cashflows: List[Tuple[datetime, float]], 
                                 threshold_ratio: float = 0.3) -> List[Tuple[datetime, float, str]]:
        """
        检测异常现金流（如重复记账或数据导出故障）
        
        Args:
            cashflows: 现金流列表
            threshold_ratio: 异常阈值比例（相对于最大峰值的比例）
            
        Returns:
            异常现金流列表 [(datetime, amount, reason)]
        """
        if not cashflows:
            return []
        
        abnormal_cashflows = []
        
        # 计算所有现金流的绝对值
        abs_amounts = [abs(amt) for _, amt in cashflows]
        
        if not abs_amounts:
            return []
        
        # 计算买入(负值)绝对峰值
        inflow_amounts = [abs(amt) for _, amt in cashflows if amt < 0]
        max_inflow = max(inflow_amounts, default=0)
        
        # 计算总流入和总流出（正数）
        total_outflow = sum([amt for _, amt in cashflows if amt > 0])  # 卖出收回
        total_inflow = sum(inflow_amounts)  # 买入总额，正数
        
        # 检测异常大额流入（可能是重复记账）
        for dt, amt in cashflows:
            if amt > 0:
                # 如果单笔流入超过总流出的阈值比例，标记为异常
                if total_outflow > 0 and amt > total_outflow * threshold_ratio:
                    reason = f"单笔流入占总流出比例过高: {amt/total_outflow*100:.1f}% (阈值: {threshold_ratio*100:.1f}%)"
                    abnormal_cashflows.append((dt, amt, reason))
                
                # 如果单笔流入超过最大峰值的阈值比例，且没有对应的卖出记录，标记为异常
                if max_inflow > 0 and amt > max_inflow * threshold_ratio:
                    reason = f"单笔流入超过最大峰值比例: {amt/max_inflow*100:.1f}% (阈值: {threshold_ratio*100:.1f}%)"
                    abnormal_cashflows.append((dt, amt, reason))
        
        # 检测现金流不平衡
        if total_inflow > 0:  # 避免除零
            net_flow = total_outflow - total_inflow
            if abs(net_flow) > total_inflow * 0.1:  # 净现金流超过总投入的10%
                reason = f"现金流不平衡: 净流入 HK${net_flow:,.2f}，占总投入 {abs(net_flow/total_inflow*100):.1f}%"
                abnormal_cashflows.append((cashflows[-1][0], net_flow, reason))
        
        return abnormal_cashflows
    
    def analyze_trades(self, df: pd.DataFrame, excluded_stocks: set) -> Tuple[float, Dict]:
        """
        分析交易，计算现金流和持仓
        
        复盘规则：
        1. 买入信号：每次买入目标金额以内的最大股数（100股的倍数），如果已持仓则跳过
        2. 卖出信号：卖出全部持仓
        
        Args:
            df: 交易记录DataFrame
            excluded_stocks: 需要排除的股票代码集合
            
        Returns:
            (现金流, 持仓字典)
        """
        cash_flow = 0.0
        portfolio = {}  # {股票代码: [数量, 成本, 名称]}
        
        # 按时间顺序处理交易
        df_sorted = df.sort_values('timestamp')
        
        for _, row in df_sorted.iterrows():
            stock_code = row['code']
            stock_name = row['name']
            # 优先使用current_price，如果为空则使用price
            price = row['current_price']
            if pd.isna(price):
                price = row['price']
            transaction_type = row['type']
            
            # 跳过排除的股票
            if stock_code in excluded_stocks:
                continue
            
            # 复盘只关注交易信号，忽略shares=0的失败交易
            if transaction_type == 'BUY':
                # 买入信号：如果没有持仓，则买入目标金额以内的最大股数（100股的倍数）
                if stock_code not in portfolio or portfolio[stock_code][0] == 0:
                    shares = self.calculate_shares(price)
                    amount = shares * price
                    cash_flow -= amount  # 买入是现金流出
                    portfolio[stock_code] = [shares, price, stock_name]
            
            elif transaction_type == 'SELL':
                # 卖出信号：如果有持仓，则卖出全部
                if stock_code in portfolio and portfolio[stock_code][0] > 0:
                    shares = portfolio[stock_code][0]
                    amount = shares * price
                    cash_flow += amount  # 卖出是现金流入
                    portfolio[stock_code][0] = 0
        
        return cash_flow, portfolio
    
    # --- XIRR helpers ---
    def _xnpv(self, rate: float, cashflows: List[Tuple[datetime, float]]) -> float:
        """
        计算 NPV 给定年化贴现率（rate）和现金流
        cashflows: list of (datetime, amount)
        """
        if rate <= -1.0:
            return float('inf')
        t0 = cashflows[0][0]
        total = 0.0
        for d, amt in cashflows:
            days = (d - t0).days + (d - t0).seconds / 86400.0
            total += amt / ((1.0 + rate) ** (days / 365.0))
        return total

    def xirr(self, cashflows: List[Tuple[datetime, float]], guess: float = 0.1,
           filter_abnormal: bool = True) -> Optional[float]:
        """
        通过二分法求解 XIRR（年化内部收益率）
        返回年化率，例如 0.12 表示 12%
        如果无法收敛或现金流不支持（例如全为同号），返回 None
        注意：对于短时间周期（<30天），仍返回年化值但可能不稳定
        
        Args:
            cashflows: 现金流列表
            guess: 初始猜测值
            filter_abnormal: 是否过滤异常现金流
        """
        if not cashflows:
            return None
        
        # 检测并过滤异常现金流
        if filter_abnormal:
            abnormal_flows = self.detect_abnormal_cashflows(cashflows)
            if abnormal_flows:
                # 移除异常现金流
                abnormal_timestamps = [dt for dt, _, _ in abnormal_flows]
                filtered_cashflows = [(dt, amt) for dt, amt in cashflows 
                                    if dt not in abnormal_timestamps]
                
                # 如果过滤后现金流为空或只有单向现金流，返回None
                if not filtered_cashflows:
                    return None
                
                signs = set([1 if amt > 0 else -1 if amt < 0 else 0 for _, amt in filtered_cashflows])
                if not (1 in signs and -1 in signs):
                    return None
                
                cashflows = filtered_cashflows
        
        # 必须至少包含一次正流入和一次负流出
        signs = set([1 if amt > 0 else -1 if amt < 0 else 0 for _, amt in cashflows])
        if not (1 in signs and -1 in signs):
            return None

        # 排序现金流
        cashflows_sorted = sorted(cashflows, key=lambda x: x[0])
        
        # 计算时间周期（天数）
        start_date = cashflows_sorted[0][0]
        end_date = cashflows_sorted[-1][0]
        days = (end_date - start_date).days
        
        # 计算年化 XIRR（无论时间周期长短都返回年化值）
        low = -0.999999999  # 扩展 low 区间以提高稳定性
        high = 10.0
        f_low = self._xnpv(low, cashflows_sorted)
        f_high = self._xnpv(high, cashflows_sorted)
        # 扩展区间确保包含根
        for _ in range(100):
            if f_low * f_high < 0:
                break
            high *= 2
            f_high = self._xnpv(high, cashflows_sorted)
        if f_low * f_high > 0:
            # 无法找到符号变化
            return None

        # 二分求解
        for _ in range(200):
            mid = (low + high) / 2.0
            f_mid = self._xnpv(mid, cashflows_sorted)
            if abs(f_mid) < 1e-8:
                return mid
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2.0

    # --- 净值序列与风险指标计算 ---
    def build_nav_series(self, df: pd.DataFrame, excluded_stocks: set) -> pd.Series:
        """
        构建按日期（天）索引的净值（NAV）序列。
        方法：
        - 按时间顺序处理交易，维护持仓与现金
        - 在每个交易时间点计算 NAV（现金 + 各持仓按当时已知价格估值）
        - 将 NAV 序列按天重采样（每天取最后一次已知 NAV，前向填充），返回每日 NAV（pd.Series）
        注意：为了避免负值导致的收益率计算异常，我们使用累计盈亏作为净值基准
              净值 = 初始资本(initial_capital) + 累计盈亏
        """
        df_sorted = df.sort_values('timestamp')
        # 初始化
        cash = 0.0
        holdings = {}  # code -> [shares, cost]
        last_price_map = {}  # code -> last known price
        nav_times = []
        nav_values = []
        base_value = self.initial_capital  # 使用初始资本作为基准值

        # 处理每个交易时间点，记录 NAV
        for _, row in df_sorted.iterrows():
            code = row['code']
            # 优先使用 current_price，否则 price
            price = row['current_price'] if pd.notna(row['current_price']) else row['price']
            if price is None or pd.isna(price) or price <= 0:
                # 更新 last price 但不计入 NAV 如果 price <=0 就不更新
                pass
            else:
                last_price_map[code] = price

            if code in excluded_stocks:
                # 跳过这些股票的交易（就好像未发生）
                # 仍然记录 NAV（price 可能被更新但我们忽略）
                total_holdings_value = 0.0
                for k, v in holdings.items():
                    shares, cost = v
                    current_price = last_price_map.get(k, 0.0)
                    if current_price > 0:
                        total_holdings_value += shares * current_price
                nav = base_value + cash + total_holdings_value
                nav_times.append(row['timestamp'])
                nav_values.append(nav)
                continue

            ttype = row['type']
            # 买入：如果当前没有持仓，买入目标金额以内的最大股数（100股的倍数）
            if ttype == 'BUY':
                if code not in holdings or holdings[code][0] == 0:
                    if price > 0:
                        shares = self.calculate_shares(price)
                        amount = shares * price
                        cash -= amount
                        holdings[code] = [shares, price]
                        last_price_map[code] = price
            elif ttype == 'SELL':
                if code in holdings and holdings[code][0] > 0:
                    if price > 0:
                        shares, cost = holdings[code]
                        amount = shares * price
                        cash += amount
                        holdings[code] = [0, 0.0]  # 清空持仓
                        last_price_map[code] = price

            # 计算当前 NAV（基准值 + 累计盈亏）
            total_holdings_value = 0.0
            for k, v in holdings.items():
                shares, cost = v
                current_price = last_price_map.get(k, 0.0)
                if current_price > 0:
                    total_holdings_value += shares * current_price
            nav = base_value + cash + total_holdings_value
            nav_times.append(row['timestamp'])
            nav_values.append(nav)

        if not nav_times:
            return pd.Series(dtype=float)

        nav_df = pd.DataFrame({'timestamp': nav_times, 'nav': nav_values})
        nav_df['date'] = nav_df['timestamp'].dt.floor('D')
        # 取每天最后一个 NAV（即交易当日最后一次 NAV）
        daily_nav = nav_df.groupby('date')['nav'].last().sort_index()

        # 如果只有一天，则直接返回那一天的值
        if daily_nav.empty:
            return pd.Series(dtype=float)

        # 填充从第一天到最后一天的每天 NAV（前向填充）
        idx = pd.date_range(start=daily_nav.index.min(), end=daily_nav.index.max(), freq='D')
        daily_nav = daily_nav.reindex(idx, method='ffill')
        daily_nav.index.name = 'date'
        return daily_nav

    def calculate_max_drawdown(self, nav_series: pd.Series) -> float:
        """
        计算最大回撤（以比例表示，例如0.25表示25%）
        包含已实现和未实现盈亏
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        
        # 计算累积最大值
        cumulative_max = nav_series.cummax()
        
        # 计算回撤
        drawdown = (nav_series - cumulative_max) / cumulative_max
        
        # 获取最大回撤
        max_dd = drawdown.min()
        
        return abs(max_dd) if not math.isnan(max_dd) else 0.0
    
    
    
    def _get_trade_rule(self, grade: str) -> str:
        """
        获取交易规则说明
        
        Args:
            grade: 股票等级
        
        Returns:
            交易规则说明
        """
        rule_map = {
            'S': '全信号执行 + 15%仓位',
            'A': '仅执行首次买入/卖出',
            'B': '仅执行首次信号 + 5%仓位',
            'C': '完全禁用'
        }
        return rule_map.get(grade, '未知')

    def calculate_annualized_volatility(self, nav_series: pd.Series) -> float:
        """
        计算年化波动率（基于日收益率），采用交易日252
        
        修正说明：
        - 使用日收益率的标准差计算
        - 日收益率 = (当日净值 - 前一日净值) / 前一日净值
        - 年化波动率 = 日收益率标准差 × √252
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = nav_series.pct_change().dropna()
        
        if daily_returns.empty:
            return 0.0
        
        # 计算日收益率标准差（使用总体标准差，ddof=0）
        daily_std = daily_returns.std(ddof=0)
        
        # 年化：日标准差 × √252（交易日数）
        annual_volatility = daily_std * math.sqrt(252)
        
        return annual_volatility

    def calculate_time_weighted_return(self, nav_series: pd.Series) -> float:
        """
        计算时间加权回报（TWR）：基于每日净值序列，按日收益连乘
        返回年化TWR（例如0.12表示12%年化）
        注意：对于短时间周期（<30天），返回累计收益率而非年化收益率
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        # 计算每天的简单回报
        daily_ret = nav_series.pct_change().dropna()
        # 复合总回报
        cumulative_return = (1 + daily_ret).prod() - 1
        
        # 计算天数
        days = (nav_series.index[-1] - nav_series.index[0]).days
        if days <= 0:
            return 0.0
        
        # 如果时间周期少于30天，返回累计收益率而非年化
        if days < 30:
            return cumulative_return
        
        # 年化：根据天数
        years = days / 365.0
        try:
            annualized = (1 + cumulative_return) ** (1.0 / years) - 1 if cumulative_return > -1 else -1.0
        except Exception:
            annualized = 0.0
        return annualized
    
    def calculate_sharpe_ratio(self, nav_series: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            nav_series: 净值序列
            risk_free_rate: 无风险利率（年化），默认为0
            
        Returns:
            夏普比率
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        
        # 计算年化收益率
        annualized_return = self.calculate_time_weighted_return(nav_series)
        
        # 计算年化波动率
        annual_volatility = self.calculate_annualized_volatility(nav_series)
        
        # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
        if annual_volatility > 0:
            return (annualized_return - risk_free_rate) / annual_volatility
        return 0.0
    
    def calculate_calmar_ratio(self, nav_series: pd.Series) -> float:
        """
        计算卡尔玛比率（年化收益率 / 最大回撤）
        
        Args:
            nav_series: 净值序列
            
        Returns:
            卡尔玛比率
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        
        # 计算年化收益率
        annualized_return = self.calculate_time_weighted_return(nav_series)
        
        # 计算最大回撤
        max_drawdown = self.calculate_max_drawdown(nav_series)
        
        # 卡尔玛比率 = 年化收益率 / 最大回撤
        if max_drawdown > 0:
            return annualized_return / max_drawdown
        return 0.0
    
    def calculate_annualized_return(self, total_profit: float, total_investment: float, 
                                   start_date: datetime, end_date: datetime) -> float:
        """
        计算年化收益率
        
        Args:
            total_profit: 总盈亏
            total_investment: 总投资
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            年化收益率（小数形式，例如0.12表示12%）
        """
        if total_investment == 0:
            return 0.0
        
        # 计算累计收益率
        cumulative_return = total_profit / total_investment
        
        # 计算天数
        days = (end_date - start_date).days
        
        if days <= 0:
            return cumulative_return
        
        # 如果时间周期少于30天，返回累计收益率而非年化
        if days < 30:
            return cumulative_return
        
        # 年化：根据天数
        years = days / 365.0
        try:
            annualized = (1 + cumulative_return) ** (1.0 / years) - 1 if cumulative_return > -1 else -1.0
        except Exception:
            annualized = 0.0
        
        return annualized
    
    def get_benchmark_return(self, start_date: datetime, end_date: datetime) -> float:
        """
        获取基准收益率（恒生指数）
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            基准收益率（百分比形式，例如5.0表示5%）
        """
        try:
            from tencent_finance import get_hsi_data
            
            # 获取恒生指数数据
            hsi_df = get_hsi_data(start_date.strftime('%Y-%m-%d'), 
                                   end_date.strftime('%Y-%m-%d'))
            
            if hsi_df.empty:
                return 0.0
            
            # 计算基准收益率
            start_price = hsi_df.iloc[0]['close']
            end_price = hsi_df.iloc[-1]['close']
            
            if start_price > 0:
                return ((end_price - start_price) / start_price) * 100
            return 0.0
        except Exception:
            # 如果获取失败，返回0
            return 0.0
    
    def get_market_trend(self, start_date: datetime, end_date: datetime) -> str:
        """
        获取市场趋势（牛市/熊市/震荡）
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            市场趋势：'bull'（牛市）、'bear'（熊市）、'neutral'（震荡）
        """
        try:
            benchmark_return = self.get_benchmark_return(start_date, end_date)
            
            # 简单判断：涨幅 > 10% 为牛市，跌幅 < -10% 为熊市，否则为震荡
            if benchmark_return > 10.0:
                return 'bull'
            elif benchmark_return < -10.0:
                return 'bear'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'

    def calculate_profit_loss(self, df: pd.DataFrame, excluded_stocks: set) -> Dict:
        """
        计算盈亏情况，并扩展返回更多用于回报/风险计算的数据:
         - cashflows: list of (datetime, amount) 用于 XIRR
         - total_invested: 所有买入的总投入
         - nav_series: 每日净值序列（pd.Series）
        """
        results = {
            'realized_profit': 0.0,  # 已实现盈亏
            'unrealized_profit': 0.0,  # 未实现盈亏
            'total_profit': 0.0,  # 总盈亏
            'stock_details': [],  # 股票明细
            'sold_stocks': [],  # 已卖出股票
            'holding_stocks': [],  # 持仓中股票
            'peak_investment': 0.0,  # 最高峰资金需求
            # 以下为新增
            'cashflows': [],  # list of (datetime, amount)
            'total_invested': 0.0,
            'nav_series': pd.Series(dtype=float),
            'total_transaction_cost': 0.0,  # 总交易成本
            'abnormal_cashflows': [],  # 异常现金流记录
            # 逐笔交易统计（用于 classify_stocks）
            'stock_trade_outcomes': {},  # code -> list of per-trade profits (realized)
            'stock_completed_trades': {},    # code -> completed trades count (已完成的交易对/平仓次数)
        }
        
        # 获取所有股票
        all_stocks = set(df['code'].unique()) - excluded_stocks
        
        # 资金管理变量
        available_cash = self.initial_capital  # 可用现金
        current_investment = 0.0  # 当前总投入
        current_holdings = {}  # {股票代码: [数量, 成本]}
        peak_investment = 0.0  # 最高峰资金需求
        
        # 首先，按时间顺序处理所有交易，计算最高峰资金需求和现金流
        all_trades = df[df['code'].isin(all_stocks)].sort_values('timestamp')
        for _, row in all_trades.iterrows():
            stock_code = row['code']
            transaction_type = row['type']
            # 优先使用current_price，如果为空则使用price
            price = row['current_price']
            if pd.isna(price):
                price = row['price']
            
            # 跳过价格为0或无效的交易
            if price <= 0:
                continue
            
            if transaction_type == 'BUY':
                # 买入信号：基于初始资本的资金分配
                if stock_code not in current_holdings:
                    # 计算应买入的金额（基于初始资本的比例）
                    target_investment = self.initial_capital * (self.allocation_pct / 100.0)
                    
                    # 检查是否有足够现金
                    if target_investment > available_cash:
                        # 如果现金不足，使用所有可用现金
                        target_investment = available_cash
                    
                    if target_investment > 0:
                        # 计算可买入的股数
                        shares = self.calculate_shares(price, self.allocation_pct)
                        # 限制股数不超过可用现金
                        max_shares = int(available_cash / price // 100) * 100
                        shares = min(shares, max_shares)
                        
                        if shares > 0:
                            actual_investment = shares * price
                            current_holdings[stock_code] = [shares, actual_investment]
                            available_cash -= actual_investment
                            current_investment += actual_investment
                            peak_investment = max(peak_investment, current_investment)
                
            elif transaction_type == 'SELL':
                # 卖出信号：卖出全部持仓
                if stock_code in current_holdings:
                    shares, investment = current_holdings[stock_code]
                    # 假设按当前价格卖出
                    returns = shares * price
                    available_cash += returns
                    current_investment -= investment
                    del current_holdings[stock_code]
        
        # 将最高峰资金需求添加到结果中
        results['peak_investment'] = peak_investment
        results['available_cash'] = available_cash  # 最终可用现金
        
        # 另外我们也需要生成现金流（用于 XIRR）和按交易时间点的 NAV（用于 TWR/max drawdown等）
        cashflows: List[Tuple[datetime, float]] = []
        
        # 重新初始化资金管理变量，用于现金流计算
        available_cash = self.initial_capital
        current_investment = 0.0
        current_holdings = {}  # {股票代码: [数量, 成本]}
        
        # 用于分析每只股票
        for stock_code in all_stocks:
            stock_trades = df[df['code'] == stock_code].sort_values('timestamp')
            stock_name = stock_trades.iloc[0]['name'] if not stock_trades.empty else stock_code
            
            # 统计建议的买卖次数（所有交易信号）
            suggested_buy_count = 0
            suggested_sell_count = 0
            for _, row in stock_trades.iterrows():
                transaction_type = row['type']
                price = row['current_price'] if pd.notna(row['current_price']) else row['price']
                if price > 0:
                    if transaction_type == 'BUY':
                        suggested_buy_count += 1
                    elif transaction_type == 'SELL':
                        suggested_sell_count += 1
            
            # 按时间顺序处理交易
            portfolio = {
                'shares': 0,  # 持仓数量
                'cost': 0.0,  # 平均成本
                'investment': 0.0  # 总投资
            }
            
            stock_realized_profit = 0.0  # 该股票的已实现盈亏
            buy_count = 0  # 实际买入次数
            sell_count = 0  # 实际卖出次数
            
            for _, row in stock_trades.iterrows():
                transaction_type = row['type']
                # 优先使用current_price，如果为空则使用price
                price = row['current_price']
                if pd.isna(price):
                    price = row['price']
                
                # 跳过价格为0或无效的交易
                if price <= 0:
                    continue
                
                if transaction_type == 'BUY':
                    # 买入信号：基于初始资本的资金分配
                    if portfolio['shares'] == 0:
                        # 计算应买入的金额（基于初始资本的比例）
                        target_investment = self.initial_capital * (self.allocation_pct / 100.0)
                        
                        # 检查是否有足够现金
                        if target_investment > available_cash:
                            # 如果现金不足，使用所有可用现金
                            target_investment = available_cash
                        
                        if target_investment > 0:
                            # 计算可买入的股数
                            shares = self.calculate_shares(price, self.allocation_pct)
                            # 限制股数不超过可用现金
                            max_shares = int(available_cash / price // 100) * 100
                            shares = min(shares, max_shares)
                            
                            if shares > 0:
                                actual_investment = shares * price
                                portfolio['shares'] = shares
                                portfolio['cost'] = price
                                portfolio['investment'] = actual_investment
                                buy_count += 1
                                
                                # 计算买入交易成本
                                buy_cost = self.calculate_transaction_cost(actual_investment, is_sell=False)
                                results['total_transaction_cost'] += buy_cost
                                
                                # 现金流（买入为负，包含交易成本）
                                cashflows.append((row['timestamp'].to_pydatetime(), -(actual_investment + buy_cost)))
                                results['total_invested'] += actual_investment
                                
                                # 更新资金管理变量
                                available_cash -= (actual_investment + buy_cost)
                                current_investment += actual_investment
                                current_holdings[stock_code] = [shares, actual_investment]
                
                elif transaction_type == 'SELL':
                    # 卖出信号：卖出全部持仓
                    if portfolio['shares'] > 0:
                        shares = portfolio['shares']
                        returns = shares * price
                        profit = returns - portfolio['investment']
                        
                        # 计算卖出交易成本
                        sell_cost = self.calculate_transaction_cost(returns, is_sell=True)
                        results['total_transaction_cost'] += sell_cost
                        
                        # 扣除交易成本后的实际收益
                        net_returns = returns - sell_cost
                        stock_realized_profit += (net_returns - portfolio['investment'])
                        
                        sell_count += 1
                        # 记录现金流（卖出为正，扣除交易成本）
                        cashflows.append((row['timestamp'].to_pydatetime(), net_returns))
                        
                        # 更新资金管理变量
                        available_cash += net_returns
                        current_investment -= portfolio['investment']
                        if stock_code in current_holdings:
                            del current_holdings[stock_code]
                        
                        # 记录逐笔交易盈亏（用于 classify_stocks），使用净收益（含手续费）
                        trade_profit_net = net_returns - portfolio['investment']
                        results['stock_trade_outcomes'].setdefault(stock_code, []).append(trade_profit_net)
                        results['stock_completed_trades'][stock_code] = results['stock_completed_trades'].get(stock_code, 0) + 1
                        
                        # 清空持仓
                        portfolio['shares'] = 0
                        portfolio['cost'] = 0.0
                        portfolio['investment'] = 0.0
            
            # 处理该股票的最终状态
            if buy_count > 0 or sell_count > 0:
                if portfolio['shares'] > 0:
                    # 持仓中 - 获取最新价格
                    # 优先调用腾讯财经接口获取实时价格
                    latest_price = None
                    try:
                        stock_info = get_hk_stock_info_tencent(stock_code)
                        if stock_info and stock_info.get('current_price'):
                            latest_price = stock_info['current_price']
                            print(f"✓ 获取股票 {stock_code} 实时价格: {latest_price}")
                    except Exception as e:
                        print(f"✗ 获取股票 {stock_code} 实时价格失败: {e}")
                    
                    # 如果实时价格获取失败，使用交易记录中的价格
                    if latest_price is None or latest_price <= 0:
                        latest_record = stock_trades.iloc[-1]
                        latest_price = latest_record['current_price'] if pd.notna(latest_record['current_price']) else latest_record['price']
                        print(f"⚠ 使用交易记录中的价格: {latest_price}")
                    
                    if latest_price > 0:
                        current_value = portfolio['shares'] * latest_price
                        profit = current_value - portfolio['investment']
                        results['unrealized_profit'] += profit
                        
                        stock_detail = {
                            'code': stock_code,
                            'name': stock_name,
                            'status': '持仓中',
                            'investment': portfolio['investment'],
                            'current_value': current_value,
                            'profit': profit,
                            'buy_count': buy_count,
                            'sell_count': sell_count,
                            'suggested_buy_count': suggested_buy_count,
                            'suggested_sell_count': suggested_sell_count
                        }
                        results['holding_stocks'].append(stock_detail)
                        results['stock_details'].append(stock_detail)
                else:
                    # 已完全卖出
                    results['realized_profit'] += stock_realized_profit
                    
                    # 计算总投资和总回报（冗余计算以确保准确）
                    total_investment = 0.0
                    total_returns = 0.0
                    
                    # 重新遍历计算总投资和总回报
                    temp_portfolio = {'shares': 0, 'investment': 0.0}
                    for _, row in stock_trades.iterrows():
                        transaction_type = row['type']
                        price = row['current_price'] if pd.notna(row['current_price']) else row['price']
                        
                        if price <= 0:
                            continue
                        
                        if transaction_type == 'BUY' and temp_portfolio['shares'] == 0:
                            shares = self.calculate_shares(price)
                            temp_portfolio['shares'] = shares
                            temp_portfolio['investment'] = shares * price
                            total_investment += temp_portfolio['investment']
                        
                        elif transaction_type == 'SELL' and temp_portfolio['shares'] > 0:
                            shares = temp_portfolio['shares']
                            returns = shares * price
                            total_returns += returns
                            temp_portfolio['shares'] = 0
                            temp_portfolio['investment'] = 0.0
                    
                    stock_detail = {
                        'code': stock_code,
                        'name': stock_name,
                        'status': '已卖出',
                        'investment': total_investment,
                        'returns': total_returns,
                        'profit': stock_realized_profit,
                        'buy_count': buy_count,
                        'sell_count': sell_count,
                        'suggested_buy_count': suggested_buy_count,
                        'suggested_sell_count': suggested_sell_count
                    }
                    results['sold_stocks'].append(stock_detail)
                    results['stock_details'].append(stock_detail)
        
        # 期末，把未平仓的市值按最后可得价格加入作为终值现金流（用于 XIRR），并作为 NAV 的最后一条
        # 计算当前持仓市值（使用 current_holdings 中的实际持仓）
        holdings_value = 0.0
        last_ts = df['timestamp'].max().to_pydatetime()
        
        # 使用 current_holdings 中的实际持仓数据
        for code, (shares, investment) in current_holdings.items():
            # 优先调用腾讯财经接口获取实时价格
            latest_price = None
            try:
                stock_info = get_hk_stock_info_tencent(code)
                if stock_info and stock_info.get('current_price'):
                    latest_price = stock_info['current_price']
                    print(f"✓ 获取股票 {code} 实时价格: {latest_price}")
            except Exception as e:
                print(f"✗ 获取股票 {code} 实时价格失败: {e}")
            
            # 如果实时价格获取失败，使用交易记录中的价格
            if latest_price is None or latest_price <= 0:
                stock_trades = df[df['code'] == code].sort_values('timestamp')
                latest_record = stock_trades.iloc[-1]
                latest_price = latest_record['current_price'] if pd.notna(latest_record['current_price']) else latest_record['price']
                print(f"⚠ 使用交易记录中的价格: {latest_price}")
            
            if latest_price > 0 and shares > 0:
                current_value = shares * latest_price
                holdings_value += current_value
        
        # 期末现金流：将持仓市值作为终值流入（相当于假设在分析终点清仓）
        if holdings_value != 0:
            cashflows.append((last_ts, holdings_value))
        else:
            # 如果没有持仓，现金流最后一笔可能已经是卖出流入
            pass
        
        # 计算最终总资产（现金 + 持仓市值）
        final_total_assets = available_cash + holdings_value
        results['final_total_assets'] = final_total_assets

        # 填充结果的 cashflows
        # 显式排序：先按 timestamp 排序，同一 timestamp 内按金额排序（买入为负，卖出为正，确保买入在前）
        results['cashflows'] = sorted(cashflows, key=lambda x: (x[0], x[1]))
        results['total_invested'] = results.get('total_invested', 0.0)

        # 计算总体已实现+未实现利润
        results['total_profit'] = results['realized_profit'] + results['unrealized_profit']

        # 生成 NAV 序列（按天）
        results['nav_series'] = self.build_nav_series(df, excluded_stocks)

        return results
    
    def classify_stocks(self, stock_details: List[Dict], trade_outcomes_map: Dict[str, List[float]], 
                       completed_trades: Dict[str, int], min_trades: int = 4,
                       start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                       nav_series: Optional[pd.Series] = None) -> Dict[str, List[Dict]]:
        """
        基于真实交易统计对股票进行分级（改进版）
        
        分级规则（考虑风险指标和时间维度）：
        - S：profit_rate > 5% AND profit_factor > 2.0 AND win_rate > 60% AND trades >= 6 
             AND max_drawdown < 5% AND sharpe_ratio > 1.5
        - A：profit_rate > 2% AND profit_factor > 1.5 AND win_rate > 55% AND trades >= 4
             AND max_drawdown < 8% AND sharpe_ratio > 1.0
        - B：profit_rate > 0% AND trades >= 2
        - C：其余或样本不足
        
        Args:
            stock_details: 股票明细列表
            trade_outcomes_map: 股票代码到逐笔盈亏列表的映射
            completed_trades: 股票代码到已完成交易次数的映射（已完成的交易对/平仓次数）
            min_trades: 最小交易次数阈值（默认为4）
            start_date: 分析起始日期（用于计算年化收益率和基准比较）
            end_date: 分析结束日期（用于计算年化收益率和基准比较）
            nav_series: 净值序列（用于计算风险指标）
            
        Returns:
            分级结果字典，包含 S、A、B、C、INSUFFICIENT 五个等级的股票列表
        """
        classification = {'S': [], 'A': [], 'B': [], 'C': [], 'INSUFFICIENT': []}
        
        # 计算时间维度相关的阈值调整
        threshold_multiplier = 1.0
        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 0:
                # 根据分析周期调整阈值
                if days < 30:  # 短期
                    threshold_multiplier = 0.5
                elif days < 90:  # 中期
                    threshold_multiplier = 0.8
                else:  # 长期
                    threshold_multiplier = 1.0
        
        # 获取市场趋势并动态调整阈值
        market_trend = 'neutral'
        if start_date and end_date:
            market_trend = self.get_market_trend(start_date, end_date)
        
        # 根据市场环境调整阈值
        profit_threshold_adjustment = 1.0
        if market_trend == 'bull':
            # 牛市中提高阈值
            profit_threshold_adjustment = 1.2
        elif market_trend == 'bear':
            # 熊市中降低阈值
            profit_threshold_adjustment = 0.8
        
        # 计算整体阈值调整系数
        overall_multiplier = threshold_multiplier * profit_threshold_adjustment
        
        # 计算全局风险指标（如果提供了 nav_series）
        global_max_drawdown = 0.0
        global_sharpe_ratio = 0.0
        if nav_series is not None and not nav_series.empty:
            global_max_drawdown = self.calculate_max_drawdown(nav_series)
            global_sharpe_ratio = self.calculate_sharpe_ratio(nav_series)
        
        # 获取基准收益率
        benchmark_return = 0.0
        if start_date and end_date:
            benchmark_return = self.get_benchmark_return(start_date, end_date)
        
        for stock in stock_details:
            code = stock.get('code')
            total_investment = stock.get('investment', 0.0)
            total_profit = stock.get('profit', 0.0)
            
            # 从逐笔盈亏获得更可靠统计
            trades = trade_outcomes_map.get(code, [])
            trade_count = completed_trades.get(code, 0)
            
            # 基本统计
            win_trades = [p for p in trades if p > 0]
            loss_trades = [p for p in trades if p < 0]
            win_count = len(win_trades)
            loss_count = len(loss_trades)
            win_rate = (win_count / len(trades)) if trades else 0.0
            gross_profit = sum(win_trades) if win_trades else 0.0
            gross_loss = -sum(loss_trades) if loss_trades else 0.0  # positive number
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
            avg_win = (sum(win_trades) / win_count) if win_count > 0 else 0.0
            avg_loss = (sum(loss_trades) / loss_count) if loss_count > 0 else 0.0
            
            # 全部交易不足或投资太小，标记为样本不足
            if trade_count < min_trades or (total_investment and total_investment < 1e-6):
                stock_res = stock.copy()
                stock_res.update({
                    'trade_count': trade_count,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'reason': 'insufficient_samples' if trade_count < min_trades else 'low_investment'
                })
                classification['INSUFFICIENT'].append(stock_res)
                continue
            
            # 计算收益率（如果 investment 信息可用）
            profit_rate = (total_profit / total_investment * 100) if total_investment > 0 else 0.0
            
            # 计算年化收益率（如果提供了日期）
            annualized_return = 0.0
            if start_date and end_date and total_investment > 0:
                annualized_return = self.calculate_annualized_return(
                    total_profit, total_investment, start_date, end_date
                )
            
            # 计算超额收益（相对于基准）
            excess_return = profit_rate - benchmark_return
            excess_return_pct = (excess_return / benchmark_return * 100) if benchmark_return != 0 else 0.0
            
            # 应用阈值调整
            adjusted_profit_threshold_s = 5.0 * overall_multiplier
            adjusted_profit_threshold_a = 2.0 * overall_multiplier
            
            # 分级规则（改进版）
            grade = 'C'
            
            # S级：更严格的要求
            if (profit_rate > adjusted_profit_threshold_s and 
                profit_factor > 2.0 and 
                win_rate > 0.60 and 
                trade_count >= 6 and
                global_max_drawdown < 0.05 and  # 最大回撤 < 5%
                global_sharpe_ratio > 1.5):   # 夏普比率 > 1.5
                grade = 'S'
            
            # A级：中等要求
            elif (profit_rate > adjusted_profit_threshold_a and 
                  profit_factor > 1.5 and 
                  win_rate > 0.55 and 
                  trade_count >= 4 and
                  global_max_drawdown < 0.08 and  # 最大回撤 < 8%
                  global_sharpe_ratio > 1.0):   # 夏普比率 > 1.0
                grade = 'A'
            
            # B级：基本要求
            elif profit_rate > 0.0 and trade_count >= 2:
                grade = 'B'
            
            # 基准加成：如果超额收益 > 基准的50%，提升一级
            if benchmark_return != 0 and excess_return_pct > 50.0:
                if grade == 'B':
                    grade = 'A'
                elif grade == 'A':
                    grade = 'S'
            
            stock_res = stock.copy()
            stock_res.update({
                'trade_count': trade_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_rate': profit_rate,
                'annualized_return': annualized_return,
                'excess_return': excess_return,
                'benchmark_return': benchmark_return,
                'max_drawdown': global_max_drawdown,
                'sharpe_ratio': global_sharpe_ratio,
                'market_trend': market_trend,
                'threshold_multiplier': overall_multiplier,
                'grade': grade,
                'recommended_position': self._get_recommended_position(grade),
                'trade_rule': self._get_trade_rule(grade)
            })
            classification[grade].append(stock_res)
        
        return classification
    
    def _get_recommended_position(self, grade: str) -> str:
        """
        根据股票等级获取推荐仓位
        
        Args:
            grade: 股票等级（S、A、B、C）
            
        Returns:
            推荐仓位字符串
        """
        position_map = {
            'S': '重仓（20%-30%）',
            'A': '中仓（10%-20%）',
            'B': '轻仓（5%-10%）',
            'C': '不建议持有',
            'INSUFFICIENT': '数据不足，建议观望'
        }
        return position_map.get(grade, '未知等级')
    
    def _get_trade_rule(self, grade: str) -> str:
        """
        根据股票等级获取交易规则
        
        Args:
            grade: 股票等级（S、A、B、C）
            
        Returns:
            交易规则字符串
        """
        rule_map = {
            'S': '积极买入，逢低加仓，严格止损（-5%）',
            'A': '逢低买入，正常止损（-8%）',
            'B': '谨慎买入，严格止损（-10%）',
            'C': '不建议买入，如有持仓建议减仓或清仓',
            'INSUFFICIENT': '建议积累更多数据后再做决策'
        }
        return rule_map.get(grade, '未知等级')
    
    def generate_report(self, start_date: str, end_date: str, cash_flow: float, 
                       holdings_value: float, profit_results: Dict, 
                       excluded_stocks: set) -> str:
        """
        生成分析报告，根据时间周期动态调整内容
        
        时间周期分类：
        - 短期（<5天）：只显示基础指标
        - 中期（5-20天）：显示大部分指标
        - 长期（≥20天）：显示所有指标
        """
        # 使用最高峰资金需求
        peak_investment = profit_results.get('peak_investment', 0.0)
        
        # 计算已收回资金（卖出所得）
        sold_returns = 0
        for stock in profit_results['sold_stocks']:
            sold_returns += stock.get('returns', 0.0)
        
        # 总体盈亏 = 已实现盈亏 + 未实现盈亏
        total_profit = profit_results['realized_profit'] + profit_results['unrealized_profit']
        
        # 总交易成本
        total_transaction_cost = profit_results.get('total_transaction_cost', 0.0)
        
        # 扣除交易成本后的净盈亏
        net_profit = total_profit - total_transaction_cost
        
        # 最终总资产
        final_total_assets = profit_results.get('final_total_assets', 0.0)
        
        # ROI（基于总投入）
        total_invested = profit_results.get('total_invested', 0.0)
        roi = (total_profit / total_invested * 100) if total_invested != 0 else 0.0
        
        # 扣除交易成本后的ROI
        net_roi = (net_profit / total_invested * 100) if total_invested != 0 else 0.0

        # XIRR（年化内部收益率）
        cashflows = profit_results.get('cashflows', [])
        xirr_value = None
        xirr_without_final = None  # 不含期末清算的XIRR
        try:
            xirr_value = self.xirr(cashflows)
            # 计算不含期末清算的XIRR（移除最后一笔大额流入）
            if len(cashflows) >= 2:
                cashflows_without_final = cashflows[:-1]
                xirr_without_final = self.xirr(cashflows_without_final)
        except Exception:
            xirr_value = None
            xirr_without_final = None
        
        # 检测异常现金流
        abnormal_cashflows = []
        if cashflows:
            abnormal_cashflows = self.detect_abnormal_cashflows(cashflows)
        
        # NAV 序列与 TWR、回撤、波动率、夏普
        nav_series: pd.Series = profit_results.get('nav_series', pd.Series(dtype=float))
        max_drawdown = self.calculate_max_drawdown(nav_series)
        twr_annual = self.calculate_time_weighted_return(nav_series)
        # 计算年化波动率（基于净值序列）
        annual_vol = self.calculate_annualized_volatility(nav_series)
        # TWR 和 CAGR 计算基于净值序列
        twr_annual = self.calculate_time_weighted_return(nav_series)
        # CAGR 基于净值序列起止值正确计算
        if not nav_series.empty and len(nav_series) >= 2:
            start_val = nav_series.iloc[0]
            end_val = nav_series.iloc[-1]
            days = (nav_series.index[-1] - nav_series.index[0]).days
            if start_val > 0 and days > 0:
                # 如果时间周期少于30天，返回累计收益率而非年化
                if days < 30:
                    cagr = (end_val / start_val) - 1
                else:
                    years = days / 365.0
                    try:
                        cagr = (end_val / start_val) ** (1.0 / years) - 1
                    except Exception:
                        cagr = 0.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0

        # 计算时间周期（天数）
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days
        except:
            days = 0

        # 夏普比率（假设无风险利率为0）
        # 修正：使用年化收益率而非TWR
        # 年化收益率 = (最终总资产 / 初始资本 - 1) * (365 / 天数)
        if days > 0:
            annual_return = (final_total_assets / self.initial_capital - 1) * (365 / days)
        else:
            annual_return = 0.0
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0.0

        report = []
        report.append("=" * 60)
        report.append("人工智能股票交易盈利能力分析报告")
        report.append("=" * 60)
        report.append(f"分析期间: {start_date} 至 {end_date}")
        report.append("")
        
        # 根据时间周期添加说明
        if days < 5:
            report.append("【时间周期说明】")
            report.append("⚠️ 短期数据（<5天），指标仅供参考")
            report.append("   建议关注：ROI、总体盈亏、交易次数")
            report.append("   谨慎解读：XIRR（年化值不稳定）")
            report.append("")
        elif days < 20:
            report.append("【时间周期说明】")
            report.append("✓ 中期数据（5-20天），指标相对稳定")
            report.append("   可作为策略评估参考")
            report.append("")
        else:
            report.append("【时间周期说明】")
            report.append("✓ 长期数据（≥20天），指标具有较高参考价值")
            report.append("   适合全面评估策略表现")
            report.append("")
        
        # 总体概览
        report.append("【总体概览】")
        report.append(f"初始资本: HK${self.initial_capital:,.2f}")
        report.append(f"最终总资产: HK${profit_results.get('final_total_assets', 0.0):,.2f}")
        report.append(f"总盈亏: HK${total_profit:,.2f}")
        report.append(f"总收益率: {(total_profit/self.initial_capital*100):.2f}%")
        report.append(f"总交易成本: HK${total_transaction_cost:,.2f} (佣金+印花税+平台费)")
        report.append(f"扣除成本后净盈亏: HK${net_profit:,.2f}")
        report.append(f"扣除成本后净收益率: {(net_profit/self.initial_capital*100):.2f}%")
        report.append(f"资金利用率: {(profit_results.get('peak_investment', 0.0)/self.initial_capital*100):.2f}%")
        report.append("")
        
        # 交易成本明细
        if total_transaction_cost > 0:
            report.append("【交易成本明细】")
            report.append(f"总交易成本: HK${total_transaction_cost:,.2f}")
            report.append(f"占总投入比例: {(total_transaction_cost/total_invested*100):.2f}%" if total_invested > 0 else "占总投入比例: N/A")
            
            # 计算交易次数和单笔分析
            total_trades = len(profit_results.get('sold_stocks', [])) + len(profit_results.get('holding_stocks', []))
            if total_trades > 0:
                avg_cost_per_trade = total_transaction_cost / total_trades
                report.append(f"平均单笔交易成本: HK${avg_cost_per_trade:.2f}")
            
            # 计算成本吞噬比例
            gross_profit = total_profit + total_transaction_cost  # 毛利润（不含成本）
            if gross_profit > 0:
                cost_ratio = (total_transaction_cost / gross_profit) * 100
                report.append(f"成本吞噬比例: {cost_ratio:.1f}% (成本占毛利润比例)")
            
            report.append("")
        
        # 异常现金流警告
        if abnormal_cashflows:
            report.append("【⚠️ 异常现金流警告】")
            report.append("检测到以下异常现金流，可能影响XIRR计算的准确性：")
            for dt, amt, reason in abnormal_cashflows:
                report.append(f"  - {dt.strftime('%Y-%m-%d %H:%M:%S')}: HK${amt:,.2f}")
                report.append(f"    原因: {reason}")
            report.append("  这些异常值已从XIRR计算中排除")
            report.append("")
        
        # XIRR / 回报指标
        report.append("【回报指标】")
        if xirr_value is not None:
            if days < 5:
                # 短期数据：添加警告
                report.append(f"XIRR（含期末清算）: {xirr_value * 100:.2f}% ⚠️ 短期数据，仅供参考")
            else:
                report.append(f"XIRR（含期末清算）: {xirr_value * 100:.2f}%")
        else:
            report.append("XIRR: 无法计算（现金流可能不包含正负两类流）")
        
        # 添加不含期末清算的XIRR
        if xirr_without_final is not None:
            report.append(f"XIRR（不含期末清算）: {xirr_without_final * 100:.2f}%")
            if xirr_value is not None:
                diff = xirr_value - xirr_without_final
                report.append(f"  期末清算影响: +{diff * 100:.2f}%")
        else:
            report.append("XIRR（不含期末清算）: 无法计算")
        
        # 添加年化收益率（修正后的）
        if days > 0:
            report.append(f"年化收益率: {annual_return * 100:.2f}%")
        report.append("")
        
        # 根据时间周期决定是否显示风险指标
        if days >= 5:
            report.append("")
            report.append("【风险指标】")
            report.append(f"最大回撤: {max_drawdown * 100:.2f}%")
            
            if days >= 20:
                # 长期数据：显示完整风险指标
                report.append(f"年化波动率: {annual_vol * 100:.2f}%")
                report.append(f"夏普比率（假设无风险利率=0）: {sharpe:.2f}")
            else:
                # 中期数据：不显示波动率和夏普比率
                report.append("年化波动率: 数据不足，暂不显示（需≥20天）")
                report.append("夏普比率: 数据不足，暂不显示（需≥20天）")
        
        report.append("")
        
        # 盈亏构成
        report.append("【盈亏构成】")
        report.append(f"已实现盈亏: HK${profit_results['realized_profit']:,.2f}")
        report.append(f"未实现盈亏: HK${profit_results['unrealized_profit']:,.2f}")
        report.append("")
        
        # 已卖出股票
        if profit_results['sold_stocks']:
            report.append("【已卖出股票】")
            # 按股票代码排序
            sorted_sold = sorted(profit_results['sold_stocks'], key=lambda x: x['code'])
            for stock in sorted_sold:
                profit_rate_stock = (stock['profit'] / stock['investment'] * 100) if stock['investment'] != 0 else 0
                report.append(f"{stock['name']}({stock['code']}): "
                           f"盈亏HK${stock['profit']:,.2f} ({profit_rate_stock:.2f}%) "
                           f"(买入{stock['buy_count']}次, 卖出{stock['sell_count']}次, "
                           f"建议买入{stock['suggested_buy_count']}次, 建议卖出{stock['suggested_sell_count']}次)")
            report.append("")
        
        # 持仓中股票
        if profit_results['holding_stocks']:
            report.append("【持仓中股票】")
            # 按股票代码排序
            sorted_holding = sorted(profit_results['holding_stocks'], key=lambda x: x['code'])
            for stock in sorted_holding:
                profit_rate_stock = (stock['profit'] / stock['investment'] * 100) if stock['investment'] != 0 else 0
                report.append(f"{stock['name']}({stock['code']}): "
                           f"盈亏HK${stock['profit']:,.2f} ({profit_rate_stock:.2f}%) "
                           f"(买入{stock['buy_count']}次, 卖出{stock['sell_count']}次, "
                           f"建议买入{stock['suggested_buy_count']}次, 建议卖出{stock['suggested_sell_count']}次)")
            report.append("")
        
        # 股票分级（基于真实交易统计）
        report.append("【股票分级（基于真实交易统计）】")
        trade_outcomes_map = profit_results.get('stock_trade_outcomes', {})
        completed_trades = profit_results.get('stock_completed_trades', {})
        
        # 转换日期字符串为 datetime 对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        
        # 获取净值序列
        nav_series = profit_results.get('nav_series', pd.Series(dtype=float))
        
        classification = self.classify_stocks(
            profit_results['stock_details'], 
            trade_outcomes_map, 
            completed_trades,
            min_trades=4,
            start_date=start_dt,
            end_date=end_dt,
            nav_series=nav_series
        )
        
        # 展示各级别股票
        grade_order = ['S', 'A', 'B', 'C', 'INSUFFICIENT']
        for grade in grade_order:
            stocks = classification[grade]
            if stocks:
                report.append(f"{grade}级 ({len(stocks)}只):")
                for stock in stocks:
                    if grade == 'INSUFFICIENT':
                        # 样本不足的股票显示原因
                        report.append(f"  {stock['name']}({stock['code']}): {stock.get('reason', '数据不足')}")
                    else:
                        # 显示详细统计信息
                        profit_factor = stock.get('profit_factor', 0)
                        profit_factor_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "inf"
                        
                        # 构建显示信息
                        info_parts = []
                        info_parts.append(f"收益率{stock.get('profit_rate', 0):.2f}%")
                        info_parts.append(f"胜率{stock.get('win_rate', 0)*100:.1f}%")
                        info_parts.append(f"盈亏比{profit_factor_str}")
                        info_parts.append(f"交易{stock.get('trade_count', 0)}次")
                        
                        # 添加年化收益率（如果有）
                        if stock.get('annualized_return', 0) != 0:
                            info_parts.append(f"年化{stock.get('annualized_return', 0)*100:.2f}%")
                        
                        # 添加超额收益（如果有基准）
                        if stock.get('benchmark_return', 0) != 0:
                            excess = stock.get('excess_return', 0)
                            excess_str = f"+{excess:.2f}%" if excess > 0 else f"{excess:.2f}%"
                            info_parts.append(f"超额{excess_str}")
                        
                        # 添加风险指标（如果有）
                        if stock.get('max_drawdown', 0) > 0:
                            info_parts.append(f"回撤{stock.get('max_drawdown', 0)*100:.2f}%")
                        if stock.get('sharpe_ratio', 0) != 0:
                            info_parts.append(f"夏普{stock.get('sharpe_ratio', 0):.2f}")
                        
                        report.append(f"  {stock['name']}({stock['code']}): "
                                   f"{', '.join(info_parts)}, "
                                   f"推荐仓位: {stock.get('recommended_position', '')}, "
                                   f"交易规则: {stock.get('trade_rule', '')}")
                report.append("")
        
        # 排除的股票
        if excluded_stocks:
            report.append("【排除的异常股票】")
            for stock_code in excluded_stocks:
                stock_name = self.df[self.df['code'] == stock_code].iloc[0]['name']
                report.append(f"{stock_name}({stock_code}): 价格异常，已排除")
            report.append("")
        
        # 交易规则说明
        report.append("【交易规则说明】")
        report.append("1. 初始资本：200万港元（可配置）")
        report.append("2. 买入规则：每只股票分配初始资本的10%（可配置）")
        report.append("3. 资金管理：确保总投入不超过初始资本，卖出后释放现金")
        report.append("4. 持仓市值可以因盈利而超过初始资本")
        report.append("5. 交易成本：包含佣金、印花税、平台费等港股标准费率")
        report.append("6. 异常处理：排除价格为0的异常交易")
        report.append("")
        
        # 附加：现金流摘要（供 XIRR 校验）
        report.append("【现金流摘要（用于 XIRR 计算）】")
        if cashflows:
            for d, amt in cashflows:
                report.append(f"{d.strftime('%Y-%m-%d %H:%M:%S')}: {'+' if amt>=0 else ''}{amt:,.2f}")
        else:
            report.append("无现金流数据")
        report.append("")
        
        return "\n".join(report)
    
    def analyze(self, start_date: Optional[str] = None, 
                end_date: Optional[str] = None, 
                send_email: bool = True) -> str:
        """
        执行分析
        
        Args:
            start_date: 起始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            send_email: 是否发送邮件通知，默认为True
            
        Returns:
            分析报告字符串
        """
        # 加载交易记录
        if not self.load_transactions():
            return "错误：无法加载交易记录文件"
        
        # 过滤交易记录
        df_filtered = self.filter_transactions(start_date, end_date)
        if df_filtered.empty:
            return f"警告：指定日期范围内没有交易记录 ({start_date} 至 {end_date})"
        
        # 识别排除的股票
        self.excluded_stocks = self.identify_excluded_stocks(df_filtered)
        
        # 分析交易
        cash_flow, portfolio = self.analyze_trades(df_filtered, self.excluded_stocks)
        
        # 计算盈亏 & 生成现金流 与 NAV
        profit_results = self.calculate_profit_loss(df_filtered, self.excluded_stocks)
        
        # 计算持仓市值（从 profit_results 中获取）
        holdings_value = sum(
            stock.get('current_value', 0.0) 
            for stock in profit_results.get('holding_stocks', [])
        )
        
        # 确定日期范围
        actual_start = df_filtered['timestamp'].min().strftime('%Y-%m-%d')
        actual_end = df_filtered['timestamp'].max().strftime('%Y-%m-%d')
        
        # 生成报告
        report = self.generate_report(actual_start, actual_end, cash_flow, 
                                    holdings_value, profit_results, 
                                    self.excluded_stocks)
        
        # 发送邮件通知
        if send_email:
            # 使用 XIRR
            xirr_value = profit_results.get('cashflows') and self.xirr(profit_results.get('cashflows')) or None
            
            subject = f"AI交易分析报告 - {actual_start} 至 {actual_end}"
            # 在邮件主题中添加总体盈亏信息和回报指标
            # 计算总体盈亏和扣除成本后的净盈亏
            total_profit = profit_results['realized_profit'] + profit_results['unrealized_profit']
            total_transaction_cost = profit_results.get('total_transaction_cost', 0.0)
            net_profit = total_profit - total_transaction_cost
            net_roi = (net_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

            if net_profit > 0:
                profit_part = f"扣除成本后净盈亏 HK${net_profit:,.2f}"
            elif net_profit < 0:
                profit_part = f"扣除成本后净亏损 HK${abs(net_profit):,.2f}"
            else:
                profit_part = "扣除成本后净盈亏 HK$0.00"

            if xirr_value is not None:
                subject += f" ({profit_part}, 净收益率 {net_roi:.2f}%, XIRR {xirr_value*100:.2f}%)"
            else:
                subject += f" ({profit_part}, 净收益率 {net_roi:.2f}%)"
            
            # 发送邮件
            email_sent = self.send_email_notification(subject, report)
            if email_sent:
                print("\n📧 分析报告已通过邮件发送")
            else:
                print("\n❌ 邮件发送失败，请检查环境变量配置")
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人工智能股票交易盈利能力分析器')
    parser.add_argument('--start-date', '-s', type=str, 
                       help='起始日期 (YYYY-MM-DD)，默认为最早交易日期')
    parser.add_argument('--end-date', '-e', type=str, 
                       help='结束日期 (YYYY-MM-DD)，默认为最新交易日期')
    parser.add_argument('--file', '-f', type=str, 
                       default='data/simulation_transactions.csv',
                       help='交易记录CSV文件路径')
    parser.add_argument('--no-email', action='store_true', 
                       help='不发送邮件通知')
    
    args = parser.parse_args()
    
    # 验证日期格式
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print("错误：起始日期格式不正确，请使用YYYY-MM-DD格式")
            sys.exit(1)
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print("错误：结束日期格式不正确，请使用YYYY-MM-DD格式")
            sys.exit(1)
    
    # 创建分析器并执行分析
    analyzer = AITradingAnalyzer(args.file)
    report = analyzer.analyze(args.start_date, args.end_date, send_email=not args.no_email)
    
    # 输出报告
    print(report)


if __name__ == "__main__":
    main()
