#!/usr/bin/env python3
"""
人工智能股票交易盈利能力分析器

基于交叉验证后的算法，分析AI推荐的股票交易策略的盈利能力。
"""

import pandas as pd
import argparse
import sys
import smtplib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


class AITradingAnalyzer:
    """AI股票交易分析器"""
    
    def __init__(self, csv_file: str = 'data/simulation_transactions.csv'):
        """
        初始化分析器
        
        Args:
            csv_file: 交易记录CSV文件路径
        """
        self.csv_file = csv_file
        self.df = None
        self.excluded_stocks = set()
    
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
                    
                    print("✅ 邮件发送成功！")
                    return True
                except Exception as e:
                    print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试，等待后重试
                        import time
                        time.sleep(5)
            
            print("❌ 邮件发送失败，已尝试3次")
            return False
        except Exception as e:
            print(f"❌ 邮件发送过程中发生错误: {e}")
            return False
        
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
    
    def analyze_trades(self, df: pd.DataFrame, excluded_stocks: set) -> Tuple[float, Dict]:
        """
        分析交易，计算现金流和持仓
        
        复盘规则：
        1. 买入信号：每次买入1000股，如果已持仓则跳过
        2. 卖出信号：卖出全部持仓
        
        Args:
            df: 交易记录DataFrame
            excluded_stocks: 需要排除的股票代码集合
            
        Returns:
            (现金流, 持仓字典)
        """
        cash_flow = 0.0
        portfolio = {}  # {股票代码: [数量, 成本]}
        
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
                # 买入信号：如果没有持仓，则买入1000股
                if stock_code not in portfolio or portfolio[stock_code][0] == 0:
                    shares = 1000
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
    
    def calculate_holdings_value(self, portfolio: Dict, df: pd.DataFrame) -> float:
        """
        计算持仓市值
        
        Args:
            portfolio: 持仓字典
            df: 交易记录DataFrame
            
        Returns:
            持仓总市值
        """
        holdings_value = 0.0
        
        for stock_code, (shares, cost, name) in portfolio.items():
            if shares > 0:
                # 获取该股票的最新价格
                stock_trades = df[df['code'] == stock_code]
                if not stock_trades.empty:
                    latest_record = stock_trades.iloc[-1]
                    # 优先使用current_price，如果为空则使用price
                    latest_price = latest_record['current_price']
                    if pd.isna(latest_price):
                        latest_price = latest_record['price']
                    market_value = shares * latest_price
                    holdings_value += market_value
        
        return holdings_value
    
    def calculate_profit_loss(self, df: pd.DataFrame, excluded_stocks: set) -> Dict:
        """
        计算盈亏情况
        
        复盘规则：
        1. 每次买入信号固定买入1000股
        2. 卖出信号清仓全部持仓
        3. 支持同一股票的多次买卖交易
        
        Args:
            df: 交易记录DataFrame
            excluded_stocks: 需要排除的股票代码集合
            
        Returns:
            盈亏结果字典
        """
        results = {
            'realized_profit': 0.0,  # 已实现盈亏
            'unrealized_profit': 0.0,  # 未实现盈亏
            'total_profit': 0.0,  # 总盈亏
            'stock_details': [],  # 股票明细
            'sold_stocks': [],  # 已卖出股票
            'holding_stocks': []  # 持仓中股票
        }
        
        # 获取所有股票
        all_stocks = set(df['code'].unique()) - excluded_stocks
        
        for stock_code in all_stocks:
            stock_trades = df[df['code'] == stock_code].sort_values('timestamp')
            stock_name = stock_trades.iloc[0]['name']
            
            # 按时间顺序处理交易
            portfolio = {
                'shares': 0,  # 持仓数量
                'cost': 0.0,  # 平均成本
                'investment': 0.0  # 总投资
            }
            
            stock_realized_profit = 0.0  # 该股票的已实现盈亏
            buy_count = 0  # 买入次数
            sell_count = 0  # 卖出次数
            
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
                    # 买入信号：如果没有持仓，则买入1000股；如果有持仓，则跳过
                    if portfolio['shares'] == 0:
                        shares = 1000
                        portfolio['shares'] = shares
                        portfolio['cost'] = price
                        portfolio['investment'] = shares * price
                        buy_count += 1
                
                elif transaction_type == 'SELL':
                    # 卖出信号：卖出全部持仓
                    if portfolio['shares'] > 0:
                        shares = portfolio['shares']
                        returns = shares * price
                        profit = returns - portfolio['investment']
                        stock_realized_profit += profit
                        sell_count += 1
                        
                        # 清空持仓
                        portfolio['shares'] = 0
                        portfolio['cost'] = 0.0
                        portfolio['investment'] = 0.0
            
            # 处理该股票的最终状态
            if buy_count > 0 or sell_count > 0:
                if portfolio['shares'] > 0:
                    # 持仓中 - 获取最新价格
                    latest_record = stock_trades.iloc[-1]
                    latest_price = latest_record['current_price'] if pd.notna(latest_record['current_price']) else latest_record['price']
                    
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
                            'sell_count': sell_count
                        }
                        results['holding_stocks'].append(stock_detail)
                        results['stock_details'].append(stock_detail)
                else:
                    # 已完全卖出
                    results['realized_profit'] += stock_realized_profit
                    
                    # 计算总投资和总回报
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
                            shares = 1000
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
                        'sell_count': sell_count
                    }
                    results['sold_stocks'].append(stock_detail)
                    results['stock_details'].append(stock_detail)
        
        results['total_profit'] = results['realized_profit'] + results['unrealized_profit']
        
        return results
    
    def generate_report(self, start_date: str, end_date: str, cash_flow: float, 
                       holdings_value: float, profit_results: Dict, 
                       excluded_stocks: set) -> str:
        """
        生成分析报告
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            cash_flow: 现金流（负数表示支出）
            holdings_value: 持仓市值
            profit_results: 盈亏结果
            excluded_stocks: 排除的股票
            
        Returns:
            格式化的报告字符串
        """
        # 计算总投资
        total_investment = 0
        for stock in profit_results['stock_details']:
            total_investment += stock['investment']
        
        # 计算已收回资金（卖出所得）
        sold_returns = 0
        for stock in profit_results['sold_stocks']:
            sold_returns += stock['returns']
        
        # 总体盈亏 = 已实现盈亏 + 未实现盈亏
        total_profit = profit_results['realized_profit'] + profit_results['unrealized_profit']
        
        # 计算盈亏率
        profit_rate = (total_profit / total_investment * 100) if total_investment != 0 else 0
        
        report = []
        report.append("=" * 60)
        report.append("人工智能股票交易盈利能力分析报告")
        report.append("=" * 60)
        report.append(f"分析期间: {start_date} 至 {end_date}")
        report.append("")
        
        # 总体概览
        report.append("【总体概览】")
        report.append(f"总投入资金: ¥{total_investment:,.2f}")
        report.append(f"已收回资金: ¥{sold_returns:,.2f}")
        report.append(f"当前持仓市值: ¥{holdings_value:,.2f}")
        report.append(f"总体盈亏: ¥{total_profit:,.2f}")
        report.append(f"盈亏率: {profit_rate:.2f}%")
        report.append("")
        
        # 盈亏构成
        report.append("【盈亏构成】")
        report.append(f"已实现盈亏: ¥{profit_results['realized_profit']:,.2f}")
        report.append(f"未实现盈亏: ¥{profit_results['unrealized_profit']:,.2f}")
        report.append("")
        
        # 已卖出股票
        if profit_results['sold_stocks']:
            report.append("【已卖出股票】")
            for stock in profit_results['sold_stocks']:
                report.append(f"{stock['name']}({stock['code']}): "
                           f"投资¥{stock['investment']:,.2f}, "
                           f"回收¥{stock['returns']:,.2f}, "
                           f"盈亏¥{stock['profit']:,.2f} "
                           f"(买入{stock['buy_count']}次, 卖出{stock['sell_count']}次)")
            report.append("")
        
        # 持仓中股票
        if profit_results['holding_stocks']:
            report.append("【持仓中股票】")
            for stock in profit_results['holding_stocks']:
                report.append(f"{stock['name']}({stock['code']}): "
                           f"投资¥{stock['investment']:,.2f}, "
                           f"现值¥{stock['current_value']:,.2f}, "
                           f"盈亏¥{stock['profit']:,.2f} "
                           f"(买入{stock['buy_count']}次, 卖出{stock['sell_count']}次)")
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
        report.append("1. 买入信号：每次买入信号固定买入1000股，如果已持仓则跳过")
        report.append("2. 卖出信号：卖出全部持仓")
        report.append("3. 异常处理：排除价格为0的异常交易")
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
        
        # 计算持仓市值
        holdings_value = self.calculate_holdings_value(portfolio, df_filtered)
        
        # 计算盈亏
        profit_results = self.calculate_profit_loss(df_filtered, self.excluded_stocks)
        
        # 确定日期范围
        actual_start = df_filtered['timestamp'].min().strftime('%Y-%m-%d')
        actual_end = df_filtered['timestamp'].max().strftime('%Y-%m-%d')
        
        # 生成报告
        report = self.generate_report(actual_start, actual_end, cash_flow, 
                                    holdings_value, profit_results, 
                                    self.excluded_stocks)
        
        # 发送邮件通知
        if send_email:
            subject = f"AI交易分析报告 - {actual_start} 至 {actual_end}"
            # 在邮件主题中添加总体盈亏信息
            total_profit = profit_results['realized_profit'] + profit_results['unrealized_profit']
            if total_profit >= 0:
                subject += f" (盈利 ¥{total_profit:,.2f})"
            else:
                subject += f" (亏损 ¥{abs(total_profit):,.2f})"
            
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
