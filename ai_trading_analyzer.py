#!/usr/bin/env python3
"""
人工智能股票交易盈利能力分析器

基于交叉验证后的算法，分析AI推荐的股票交易策略的盈利能力。
"""

import pandas as pd
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional


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
                latest_price = stock_trades.iloc[-1]['price']
                if latest_price <= 0:
                    excluded.add(stock_code)
        
        return excluded
    
    def analyze_trades(self, df: pd.DataFrame, excluded_stocks: set) -> Tuple[float, Dict]:
        """
        分析交易，计算现金流和持仓
        
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
            price = row['price']
            transaction_type = row['type']
            
            # 跳过排除的股票
            if stock_code in excluded_stocks:
                continue
            
            if transaction_type == 'BUY':
                # 只有在没有持仓时才买入
                if stock_code not in portfolio or portfolio[stock_code][0] == 0:
                    shares = 1000
                    amount = shares * price
                    cash_flow -= amount
                    portfolio[stock_code] = [shares, price, stock_name]
            
            elif transaction_type == 'SELL':
                # 只有有持仓且价格大于0时才卖出
                if (stock_code in portfolio and portfolio[stock_code][0] > 0 
                    and price > 0):
                    shares = portfolio[stock_code][0]
                    amount = shares * price
                    cash_flow += amount
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
                    latest_price = stock_trades.iloc[-1]['price']
                    market_value = shares * latest_price
                    holdings_value += market_value
        
        return holdings_value
    
    def calculate_profit_loss(self, df: pd.DataFrame, excluded_stocks: set) -> Dict:
        """
        计算盈亏情况
        
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
            
            # 获取买入和卖出记录
            buys = stock_trades[stock_trades['type'] == 'BUY']
            sells = stock_trades[stock_trades['type'] == 'SELL']
            
            if not buys.empty:
                buy_price = buys.iloc[0]['price']
                investment = 1000 * buy_price
                
                # 检查是否有有效卖出
                valid_sells = sells[sells['price'] > 0]
                if not valid_sells.empty:
                    # 已卖出
                    sell_price = valid_sells.iloc[0]['price']
                    returns = 1000 * sell_price
                    profit = returns - investment
                    results['realized_profit'] += profit
                    
                    stock_detail = {
                        'code': stock_code,
                        'name': stock_name,
                        'status': '已卖出',
                        'investment': investment,
                        'returns': returns,
                        'profit': profit
                    }
                    results['sold_stocks'].append(stock_detail)
                else:
                    # 持仓中
                    latest_price = stock_trades.iloc[-1]['price']
                    current_value = 1000 * latest_price
                    profit = current_value - investment
                    results['unrealized_profit'] += profit
                    
                    stock_detail = {
                        'code': stock_code,
                        'name': stock_name,
                        'status': '持仓中',
                        'investment': investment,
                        'current_value': current_value,
                        'profit': profit
                    }
                    results['holding_stocks'].append(stock_detail)
                
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
            cash_flow: 现金流
            holdings_value: 持仓市值
            profit_results: 盈亏结果
            excluded_stocks: 排除的股票
            
        Returns:
            格式化的报告字符串
        """
        total_assets = cash_flow + holdings_value
        
        # 计算总投资
        total_investment = 0
        for stock in profit_results['stock_details']:
            total_investment += stock['investment']
        
        # 计算盈亏率
        profit_rate = (total_assets / total_investment * 100) if total_investment != 0 else 0
        
        report = []
        report.append("=" * 60)
        report.append("人工智能股票交易盈利能力分析报告")
        report.append("=" * 60)
        report.append(f"分析期间: {start_date} 至 {end_date}")
        report.append("")
        
        # 总体概览
        report.append("【总体概览】")
        report.append(f"总投入资金: ¥{total_investment:,.2f}")
        
        # 计算已收回资金（卖出所得）
        sold_returns = 0
        for stock in profit_results['sold_stocks']:
            sold_returns += stock['returns']
        
        report.append(f"已收回资金: ¥{sold_returns:,.2f}")
        report.append(f"当前持仓市值: ¥{holdings_value:,.2f}")
        report.append(f"总资产价值: ¥{total_assets:,.2f}")
        report.append(f"总体盈亏: ¥{total_assets:,.2f}")
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
                           f"盈亏¥{stock['profit']:,.2f}")
            report.append("")
        
        # 持仓中股票
        if profit_results['holding_stocks']:
            report.append("【持仓中股票】")
            for stock in profit_results['holding_stocks']:
                report.append(f"{stock['name']}({stock['code']}): "
                           f"投资¥{stock['investment']:,.2f}, "
                           f"现值¥{stock['current_value']:,.2f}, "
                           f"盈亏¥{stock['profit']:,.2f}")
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
        report.append("1. 买入信号：每次买入1000股，如果已持仓则跳过")
        report.append("2. 卖出信号：卖出全部持仓")
        report.append("3. 异常处理：排除价格为0的异常交易")
        report.append("")
        
        return "\n".join(report)
    
    def analyze(self, start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> str:
        """
        执行分析
        
        Args:
            start_date: 起始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
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
        return self.generate_report(actual_start, actual_end, cash_flow, 
                                  holdings_value, profit_results, 
                                  self.excluded_stocks)


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
    report = analyzer.analyze(args.start_date, args.end_date)
    
    # 输出报告
    print(report)


if __name__ == "__main__":
    main()
