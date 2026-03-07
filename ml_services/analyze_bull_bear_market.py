#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析2024-2025年股票的牛、熊市分布

功能：
- 分析市场环境（牛/熊市）分布
- 评估每个股票在不同市场环境下的表现
- 分析股票与市场环境的关联关系
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_market_environment(start_date='2024-01-01', end_date='2025-12-31'):
    """
    分析市场环境（牛/熊市）
    
    参数:
    - start_date: 开始日期
    - end_date: 结束日期
    
    返回:
    DataFrame: 每个月的市场环境数据
    """
    print(f"获取恒生指数数据: {start_date} 至 {end_date}")
    hsi_ticker = yf.Ticker("^HSI")
    hsi_df = hsi_ticker.history(start=start_date, end=end_date)
    
    if len(hsi_df) == 0:
        print("❌ 错误：无法获取恒生指数数据")
        return None
    
    print(f"✅ 成功获取 {len(hsi_df)} 行恒生指数数据")
    
    # 重置索引
    hsi_df = hsi_df.reset_index()
    hsi_df['Date'] = pd.to_datetime(hsi_df['Date']).dt.normalize()
    
    # 按月分组
    hsi_df['YearMonth'] = hsi_df['Date'].dt.to_period('M')
    
    # 计算每月的市场环境
    monthly_env = []
    for period, group in hsi_df.groupby('YearMonth'):
        if len(group) < 10:  # 至少需要10个交易日
            continue
            
        # 计算月收益率
        monthly_return = (group['Close'].iloc[-1] - group['Close'].iloc[0]) / group['Close'].iloc[0]
        
        # 计算市场状态
        if monthly_return > 0.03:
            market_state = 'bull'  # 牛市
        elif monthly_return < -0.03:
            market_state = 'bear'  # 熊市
        else:
            market_state = 'neutral'  # 震荡市
        
        # 计算波动率
        volatility = group['Close'].pct_change().std() * np.sqrt(21)  # 年化波动率
        
        monthly_env.append({
            'YearMonth': period,
            'Start_Date': group['Date'].iloc[0],
            'End_Date': group['Date'].iloc[-1],
            'Monthly_Return': monthly_return,
            'Market_State': market_state,
            'Volatility': volatility,
            'HSI_Close_Start': group['Close'].iloc[0],
            'HSI_Close_End': group['Close'].iloc[-1]
        })
    
    market_env_df = pd.DataFrame(monthly_env)
    
    print(f"\n市场环境分析结果:")
    print(f"  总月份数: {len(market_env_df)}")
    print(f"  牛市月份: {len(market_env_df[market_env_df['Market_State'] == 'bull'])}")
    print(f"  熊市月份: {len(market_env_df[market_env_df['Market_State'] == 'bear'])}")
    print(f"  震荡市月份: {len(market_env_df[market_env_df['Market_State'] == 'neutral'])}")
    
    return market_env_df

def analyze_stock_performance_by_market(trades_df, market_env_df):
    """
    分析每个股票在不同市场环境下的表现
    
    参数:
    - trades_df: 交易记录DataFrame
    - market_env_df: 市场环境DataFrame
    
    返回:
    DataFrame: 每个股票在不同市场环境下的表现
    """
    # 将交易记录转换为日期格式
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
    
    # 为每笔交易添加市场环境标签
    def get_market_state(date):
        # 确保日期是Timestamp类型并移除时区信息
        date = pd.Timestamp(date)
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_localize(None)
        
        # 找到包含该日期的月份
        for _, row in market_env_df.iterrows():
            start_date = pd.Timestamp(row['Start_Date'])
            end_date = pd.Timestamp(row['End_Date'])
            
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            if start_date <= date <= end_date:
                return row['Market_State']
            # 如果日期不在任何月份中，找最近的月份
            elif date < start_date and date >= start_date - timedelta(days=10):
                return row['Market_State']
        return 'unknown'
    
    trades_df['Market_State'] = trades_df['buy_date'].apply(get_market_state)
    
    # 筛除未知市场环境的交易
    trades_df = trades_df[trades_df['Market_State'] != 'unknown']
    
    # 分析每个股票在不同市场环境下的表现
    results = []
    
    for stock_code in trades_df['stock_code'].unique():
        stock_trades = trades_df[trades_df['stock_code'] == stock_code]
        
        # 牛市表现
        bull_trades = stock_trades[stock_trades['Market_State'] == 'bull']
        bull_count = len(bull_trades)
        if bull_count > 0:
            bull_avg_return = bull_trades['actual_change'].mean()
            bull_win_rate = (bull_trades['actual_change'] > 0).sum() / bull_count
            bull_accuracy = bull_trades['prediction_correct'].sum() / bull_count
            bull_correct_decision = bull_win_rate * bull_accuracy
            bull_avg_prob = bull_trades['probability'].mean()
            bull_prob_std = bull_trades['probability'].std()
        else:
            bull_avg_return = np.nan
            bull_win_rate = np.nan
            bull_accuracy = np.nan
            bull_correct_decision = np.nan
            bull_avg_prob = np.nan
            bull_prob_std = np.nan
        
        # 熊市表现
        bear_trades = stock_trades[stock_trades['Market_State'] == 'bear']
        bear_count = len(bear_trades)
        if bear_count > 0:
            bear_avg_return = bear_trades['actual_change'].mean()
            bear_win_rate = (bear_trades['actual_change'] > 0).sum() / bear_count
            bear_accuracy = bear_trades['prediction_correct'].sum() / bear_count
            bear_correct_decision = bear_win_rate * bear_accuracy
            bear_avg_prob = bear_trades['probability'].mean()
            bear_prob_std = bear_trades['probability'].std()
        else:
            bear_avg_return = np.nan
            bear_win_rate = np.nan
            bear_accuracy = np.nan
            bear_correct_decision = np.nan
            bear_avg_prob = np.nan
            bear_prob_std = np.nan
        
        # 震荡市表现
        neutral_trades = stock_trades[stock_trades['Market_State'] == 'neutral']
        neutral_count = len(neutral_trades)
        if neutral_count > 0:
            neutral_avg_return = neutral_trades['actual_change'].mean()
            neutral_win_rate = (neutral_trades['actual_change'] > 0).sum() / neutral_count
            neutral_accuracy = neutral_trades['prediction_correct'].sum() / neutral_count
            neutral_correct_decision = neutral_win_rate * neutral_accuracy
            neutral_avg_prob = neutral_trades['probability'].mean()
            neutral_prob_std = neutral_trades['probability'].std()
        else:
            neutral_avg_return = np.nan
            neutral_win_rate = np.nan
            neutral_accuracy = np.nan
            neutral_correct_decision = np.nan
            neutral_avg_prob = np.nan
            neutral_prob_std = np.nan
        
        # 计算与市场的关联性
        # 关联性 = (牛市收益率 - 熊市收益率) / (牛市收益率 + 熊市收益率)
        if not np.isnan(bull_avg_return) and not np.isnan(bear_avg_return):
            market_correlation = (bull_avg_return - bear_avg_return) / abs(bull_avg_return + bear_avg_return)
        else:
            market_correlation = np.nan
        
        results.append({
            'stock_code': stock_code,
            'bull_count': bull_count,
            'bear_count': bear_count,
            'neutral_count': neutral_count,
            'bull_avg_return': bull_avg_return,
            'bull_win_rate': bull_win_rate,
            'bull_accuracy': bull_accuracy,
            'bull_correct_decision': bull_correct_decision,
            'bull_avg_prob': bull_avg_prob,
            'bull_prob_std': bull_prob_std,
            'bear_avg_return': bear_avg_return,
            'bear_win_rate': bear_win_rate,
            'bear_accuracy': bear_accuracy,
            'bear_correct_decision': bear_correct_decision,
            'bear_avg_prob': bear_avg_prob,
            'bear_prob_std': bear_prob_std,
            'neutral_avg_return': neutral_avg_return,
            'neutral_win_rate': neutral_win_rate,
            'neutral_accuracy': neutral_accuracy,
            'neutral_correct_decision': neutral_correct_decision,
            'neutral_avg_prob': neutral_avg_prob,
            'neutral_prob_std': neutral_prob_std,
            'market_correlation': market_correlation
        })
    
    results_df = pd.DataFrame(results)
    
    return results_df

def generate_analysis_report(market_env_df, stock_performance_df):
    """
    生成分析报告
    """
    print("\n" + "=" * 100)
    print("2024-2025年市场环境与股票表现分析报告")
    print("=" * 100)
    
    # 1. 市场环境分布
    print("\n一、市场环境分布")
    print("-" * 100)
    print(f"{'市场状态':<12} {'月份数量':<10} {'占比':<10} {'平均月收益率':<15}")
    print("-" * 100)
    
    for state in ['bull', 'bear', 'neutral']:
        state_df = market_env_df[market_env_df['Market_State'] == state]
        count = len(state_df)
        ratio = count / len(market_env_df) * 100
        avg_return = state_df['Monthly_Return'].mean()
        
        state_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}[state]
        print(f"{state_name:<12} {count:<10} {ratio:<10.1f}% {avg_return:>14.2%}")
    
    print(f"{'总计':<12} {len(market_env_df):<10} 100.0% {market_env_df['Monthly_Return'].mean():>14.2%}")
    
    # 2. 详细月度数据
    print("\n二、详细月度数据")
    print("-" * 100)
    print(f"{'月份':<15} {'市场状态':<12} {'恒指收益率':<12} {'恒指起点':<12} {'恒指终点':<12}")
    print("-" * 100)
    
    for _, row in market_env_df.iterrows():
        period_str = str(row['YearMonth'])
        state_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}[row['Market_State']]
        print(f"{period_str:<15} {state_name:<12} {row['Monthly_Return']:>10.2%} {row['HSI_Close_Start']:>12.0f} {row['HSI_Close_End']:>12.0f}")
    
    # 3. 股票表现汇总（按市场关联性排序）
    print("\n三、股票表现汇总（按市场关联性排序）")
    print("-" * 100)
    
    # 过滤掉没有足够数据的股票
    valid_stocks = stock_performance_df[
        (stock_performance_df['bull_count'] >= 10) & 
        (stock_performance_df['bear_count'] >= 10)
    ].copy()
    
    valid_stocks = valid_stocks.sort_values('market_correlation', ascending=False)
    
    print(f"{'股票代码':<12} {'牛市收益率':<12} {'熊市收益率':<12} {'市场关联性':<12} {'牛市胜率':<10} {'熊市胜率':<10}")
    print("-" * 100)
    
    for _, row in valid_stocks.iterrows():
        print(f"{row['stock_code']:<12} {row['bull_avg_return']:>10.2%} {row['bear_avg_return']:>10.2%} "
              f"{row['market_correlation']:>10.2f} {row['bull_win_rate']:>8.1%} {row['bear_win_rate']:>8.1%}")
    
    # 4. 分析结论
    print("\n四、分析结论")
    print("-" * 100)
    
    # 高市场关联性股票
    high_corr_stocks = valid_stocks[valid_stocks['market_correlation'] > 0.3]
    print(f"\n高市场关联性股票（关联性 > 0.3）: {len(high_corr_stocks)} 只")
    if len(high_corr_stocks) > 0:
        print(f"  平均牛市收益率: {high_corr_stocks['bull_avg_return'].mean():.2%}")
        print(f"  平均熊市收益率: {high_corr_stocks['bear_avg_return'].mean():.2%}")
    
    # 低市场关联性股票
    low_corr_stocks = valid_stocks[abs(valid_stocks['market_correlation']) < 0.2]
    print(f"\n低市场关联性股票（|关联性| < 0.2）: {len(low_corr_stocks)} 只")
    if len(low_corr_stocks) > 0:
        print(f"  平均牛市收益率: {low_corr_stocks['bull_avg_return'].mean():.2%}")
        print(f"  平均熊市收益率: {low_corr_stocks['bear_avg_return'].mean():.2%}")
    
    # 预测概率范围分析
    print("\n五、预测概率范围分析")
    print("-" * 100)
    
    print(f"\n牛市环境预测概率:")
    print(f"  平均值: {valid_stocks['bull_avg_prob'].mean():.4f}")
    print(f"  标准差: {valid_stocks['bull_prob_std'].mean():.4f}")
    print(f"  范围: [{valid_stocks['bull_avg_prob'].min():.4f}, {valid_stocks['bull_avg_prob'].max():.4f}]")
    
    print(f"\n熊市环境预测概率:")
    print(f"  平均值: {valid_stocks['bear_avg_prob'].mean():.4f}")
    print(f"  标准差: {valid_stocks['bear_prob_std'].mean():.4f}")
    print(f"  范围: [{valid_stocks['bear_avg_prob'].min():.4f}, {valid_stocks['bear_avg_prob'].max():.4f}]")
    
    return valid_stocks

if __name__ == "__main__":
    # 1. 分析市场环境
    market_env_df = analyze_market_environment('2024-01-01', '2025-12-31')
    
    if market_env_df is None:
        sys.exit(1)
    
    # 2. 读取交易记录
    trades_df = pd.read_csv('output/backtest_20d_trades_20260307_002039.csv')
    print(f"\n✅ 成功读取 {len(trades_df)} 条交易记录")
    
    # 3. 分析股票表现
    stock_performance_df = analyze_stock_performance_by_market(trades_df, market_env_df)
    
    # 4. 生成分析报告
    valid_stocks = generate_analysis_report(market_env_df, stock_performance_df)
    
    # 5. 保存结果
    output_file = 'output/bull_bear_market_analysis_20260307.csv'
    valid_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 分析结果已保存到: {output_file}")
    
    print("\n" + "=" * 100)
    print("✅ 分析完成")
    print("=" * 100)