#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股模拟交易系统
基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易
"""

import os
import sys
import time
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import threading
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入hk_smart_money_tracker模块
import hk_smart_money_tracker

class SimulationTrader:
    def __init__(self, initial_capital=1000000):
        """
        初始化模拟交易系统
        
        Args:
            initial_capital (float): 初始资金，默认100万港元
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 持仓 {code: {'shares': 数量, 'avg_price': 平均买入价}}
        self.transaction_history = []  # 交易历史
        self.portfolio_history = []  # 投资组合价值历史
        self.start_date = datetime.now()
        self.is_trading_hours = True  # 模拟港股交易时间 (9:30-16:00)
        
        # 持久化文件
        self.state_file = "simulation_state.json"
        self.log_file = "simulation_trade_log.txt"
        
        # 尝试从文件恢复状态
        self.load_state()
        
        # 如果是新开始，创建交易日志文件
        if not self.transaction_history:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"模拟交易日志 - 开始时间: {self.start_date}\n")
                f.write(f"初始资金: HK${self.initial_capital:,.2f}\n")
                f.write("="*50 + "\n")
    
    def log_message(self, message):
        """记录交易日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        # 写入日志文件
        with open(self.log_file, "a", encoding="utf-8") as f:
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
        获取股票当前价格
        
        Args:
            code (str): 股票代码
            
        Returns:
            float: 当前价格，如果获取失败返回None
        """
        try:
            ticker = yf.Ticker(code)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            else:
                # 如果1分钟数据不可用，尝试获取日线数据
                hist = ticker.history(period="2d")
                if len(hist) >= 1:
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
            
        # 执行卖出
        sell_amount = shares_to_sell * current_price
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
            
        # 执行买入操作（严格按照大模型建议）
        # 根据推荐的股票数量动态调整投资比例，确保不会超过总资金
        buy_count = len(recommendations['buy'])
        if buy_count > 0:
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
            investment_pct = min(max_total_pct / buy_count, max_single_pct)
            for code in recommendations['buy']:
                if code in hk_smart_money_tracker.WATCHLIST:
                    name = hk_smart_money_tracker.WATCHLIST[code]
                    self.buy_stock(code, name, investment_pct)
                
        # 执行卖出操作（严格按照大模型建议）
        for code in recommendations['sell']:
            if code in hk_smart_money_tracker.WATCHLIST:
                name = hk_smart_money_tracker.WATCHLIST[code]
                # 卖出全部持仓
                sell_pct = 1.0
                self.sell_stock(code, name, sell_pct)
                
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
    
    def run_hourly_analysis(self):
        """每小时运行一次分析和交易"""
        self.log_message("开始每小时分析...")
        self.execute_trades()
        
    def run_daily_summary(self):
        """每日总结"""
        if not self.portfolio_history:
            return
            
        self.log_message("="*50)
        self.log_message("每日交易总结")
        self.log_message("="*50)
        
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
        
        # 持仓情况
        self.log_message(f"当前持仓: {self.positions}")
        self.log_message(f"可用资金: HK${self.capital:,.2f}")
        
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
            df_transactions.to_csv("simulation_transactions.csv", index=False, encoding="utf-8")
            self.log_message("交易历史已保存到 simulation_transactions.csv")
        except Exception as e:
            self.log_message(f"保存交易历史失败: {e}")
            
        # 保存投资组合历史到文件
        try:
            df_portfolio = pd.DataFrame(self.portfolio_history)
            df_portfolio.to_csv("simulation_portfolio.csv", index=False, encoding="utf-8")
            self.log_message("投资组合历史已保存到 simulation_portfolio.csv")
        except Exception as e:
            self.log_message(f"保存投资组合历史失败: {e}")

def run_simulation(duration_days=30):
    """
    运行模拟交易
    
    Args:
        duration_days (int): 模拟天数，默认30天
    """
    print(f"开始港股模拟交易，模拟周期: {duration_days} 天")
    print("初始资金: HK$1,000,000")
    
    # 创建模拟交易器
    trader = SimulationTrader(initial_capital=1000000)
    
    # 计划每15分钟执行一次交易分析
    schedule.every(15).minutes.do(trader.run_hourly_analysis)
    
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
    # 运行90天的模拟交易
    run_simulation(duration_days=90)