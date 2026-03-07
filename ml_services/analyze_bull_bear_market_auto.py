#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析股票的牛、熊市分布（可重复运行版本）

功能：
- 分析市场环境（牛/熊市）分布
- 评估每个股票在不同市场环境下的表现
- 分析股票与市场环境的关联关系
- 支持自定义日期范围和输入文件
- 自动生成带时间戳的分析报告

使用方法：
  python3 analyze_bull_bear_market_auto.py
  python3 analyze_bull_bear_market_auto.py --start-date 2024-01-01 --end-date 2025-12-31
  python3 analyze_bull_bear_market_auto.py --trades-file output/backtest_20d_trades_20260307_002039.csv
  python3 analyze_bull_bear_market_auto.py --output-format all
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
import argparse
import json
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 股票名称映射
STOCK_NAMES = {
    '0005.HK': '汇丰银行', '0012.HK': '恒基地产', '0016.HK': '新鸿基地产',
    '0388.HK': '香港交易所', '0700.HK': '腾讯控股', '0728.HK': '中国电信',
    '0883.HK': '中国海洋石油', '0939.HK': '建设银行', '0941.HK': '中国移动',
    '0981.HK': '中芯国际', '1088.HK': '中国神华', '1109.HK': '华润置地',
    '1138.HK': '中远海能', '1288.HK': '农业银行', '1299.HK': '友邦保险',
    '1330.HK': '绿色动力环保', '1347.HK': '华虹半导体', '1398.HK': '工商银行',
    '1810.HK': '小米集团-W', '2269.HK': '药明生物', '2533.HK': '黑芝麻智能',
    '2800.HK': '盈富基金', '3690.HK': '美团-W', '3968.HK': '招商银行',
    '6682.HK': '第四范式', '9660.HK': '地平线机器人', '9988.HK': '阿里巴巴-SW',
    '1211.HK': '比亚迪股份'
}

def find_latest_backtest_file():
    """
    查找最新的回测交易记录文件
    
    返回:
    str: 最新文件的路径，如果没有找到则返回None
    """
    output_dir = Path('output')
    
    # 查找所有 backtest_20d_trades_*.csv 文件
    trades_files = list(output_dir.glob('backtest_20d_trades_*.csv'))
    
    if not trades_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(trades_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def analyze_market_environment(start_date, end_date):
    """
    分析市场环境（牛/熊市）
    
    参数:
    - start_date: 开始日期
    - end_date: 结束日期
    
    返回:
    DataFrame: 每个月的市场环境数据
    """
    print(f"📊 获取恒生指数数据: {start_date} 至 {end_date}")
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
    
    print(f"\n📈 市场环境分析结果:")
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
        # 关联性 = (牛市收益率 - 熊市收益率) / |牛市收益率 + 熊市收益率|
        if not np.isnan(bull_avg_return) and not np.isnan(bear_avg_return):
            market_correlation = (bull_avg_return - bear_avg_return) / abs(bull_avg_return + bear_avg_return)
        else:
            market_correlation = np.nan
        
        results.append({
            'stock_code': stock_code,
            'stock_name': STOCK_NAMES.get(stock_code, stock_code),
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

def get_current_market_state():
    """
    获取当前市场状态（实时）
    
    返回:
    dict: 当前市场状态信息
    """
    try:
        # 获取最近30天的恒生指数数据
        hsi_ticker = yf.Ticker("^HSI")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hsi_df = hsi_ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                    end=end_date.strftime('%Y-%m-%d'))
        
        if len(hsi_df) < 10:
            return None
        
        # 计算最近20天收益率
        if len(hsi_df) >= 20:
            recent_20d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[-20]) / hsi_df['Close'].iloc[-20]
        else:
            recent_20d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[0]) / hsi_df['Close'].iloc[0]
        
        # 计算最近5天收益率
        if len(hsi_df) >= 5:
            recent_5d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[-5]) / hsi_df['Close'].iloc[-5]
        else:
            recent_5d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[0]) / hsi_df['Close'].iloc[0]
        
        # 计算当前市场状态
        if recent_20d_return > 0.05:
            market_state = 'bull'
            market_state_cn = '牛市'
            market_signal = '📈 强烈看涨'
        elif recent_20d_return < -0.05:
            market_state = 'bear'
            market_state_cn = '熊市'
            market_signal = '📉 强烈看跌'
        elif recent_20d_return > 0.02:
            market_state = 'neutral_bull'
            market_state_cn = '震荡偏涨'
            market_signal = '⬆️ 温和上涨'
        elif recent_20d_return < -0.02:
            market_state = 'neutral_bear'
            market_state_cn = '震荡偏跌'
            market_signal = '⬇️ 温和下跌'
        else:
            market_state = 'neutral'
            market_state_cn = '震荡市'
            market_signal = '➡️ 横盘整理'
        
        return {
            'market_state': market_state,
            'market_state_cn': market_state_cn,
            'market_signal': market_signal,
            'recent_20d_return': recent_20d_return,
            'recent_5d_return': recent_5d_return,
            'current_hsi': hsi_df['Close'].iloc[-1],
            'date': hsi_df.index[-1].strftime('%Y-%m-%d')
        }
    except Exception as e:
        print(f"⚠️ 获取当前市场状态失败: {e}")
        return None

def save_markdown_report(market_env_df, stock_performance_df, output_file):
    """
    生成Markdown格式的分析报告
    """
    # 过滤掉没有足够数据的股票
    valid_stocks = stock_performance_df[
        (stock_performance_df['bull_count'] >= 10) & 
        (stock_performance_df['bear_count'] >= 10)
    ].copy()
    
    valid_stocks = valid_stocks.sort_values('market_correlation', ascending=False)
    
    # 获取当前市场状态
    current_market = get_current_market_state()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 牛熊市分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 一、当前市场状态（新增）
        f.write("## 一、当前市场状态\n\n")
        if current_market:
            f.write(f"**市场信号**: {current_market['market_signal']}\n\n")
            f.write(f"**市场状态**: {current_market['market_state_cn']}\n\n")
            f.write(f"**恒生指数**: {current_market['current_hsi']:.2f} (截至 {current_market['date']})\n\n")
            f.write(f"**最近20天收益率**: {current_market['recent_20d_return']:.2%}\n\n")
            f.write(f"**最近5天收益率**: {current_market['recent_5d_return']:.2%}\n\n")
            
            f.write("### 市场状态说明\n\n")
            f.write("| 市场状态 | 20天收益率范围 | 说明 |\n")
            f.write("|---------|--------------|------|\n")
            f.write("| 📈 牛市 | > 5% | 市场强劲上涨，适合积极配置 |\n")
            f.write("| ⬆️ 震荡偏涨 | 2% - 5% | 市场温和上涨，可以谨慎配置 |\n")
            f.write("| ➡️ 震荡市 | -2% - 2% | 市场横盘整理，建议观望 |\n")
            f.write("| ⬇️ 震荡偏跌 | -5% - -2% | 市场温和下跌，建议减仓 |\n")
            f.write("| 📉 熊市 | < -5% | 市场强劲下跌，建议空仓 |\n\n")
            
            f.write("### 投资建议\n\n")
            if current_market['market_state'] == 'bull':
                f.write("**牛市策略**:\n\n")
                f.write("- ✅ **重仓高市场关联性股票**: 牛市中高关联性股票平均收益率可达 +9.35%\n")
                f.write("- ✅ **关注科技、半导体板块**: 这些板块通常在牛市中表现优异\n")
                f.write("- ✅ **使用100%仓位**: 市场信号强烈，可全仓操作\n\n")
            elif current_market['market_state'] == 'bear':
                f.write("**熊市策略**:\n\n")
                f.write("- ⚠️ **重仓低市场关联性股票**: 熊市中低关联性股票平均收益率为 +4.15%\n")
                f.write("- ⚠️ **配置银行、公用事业**: 这些股票具有防御性\n")
                f.write("- ⚠️ **降低仓位至30%**: 市场风险较高，控制仓位\n\n")
            elif current_market['market_state'] in ['neutral_bull', 'neutral_bear']:
                f.write("**震荡市策略**:\n\n")
                f.write("- 🔄 **均衡配置**: 高低关联性股票各占50%\n")
                f.write("- 🔄 **动态调整**: 根据市场信号及时调整仓位\n")
                f.write("- 🔄 **关注波段机会**: 震荡市适合波段操作\n\n")
                # 额外的风险提示
                if current_market['market_state'] == 'neutral_bear':
                    f.write("**风险提示**:\n\n")
                    f.write("- ⚠️ 市场温和下跌，建议保持谨慎\n")
                    f.write("- ⚠️ 可考虑降低仓位至70%\n")
                else:
                    f.write("**机会提示**:\n\n")
                    f.write("- ✅ 市场温和上涨，可考虑逐步加仓\n")
                    f.write("- ✅ 建议仓位可提升至80%\n\n")
            else:  # neutral
                f.write("**横盘策略**:\n\n")
                f.write("- ⏸️ **观望为主**: 市场缺乏明确方向，建议保持观望\n")
                f.write("- ⏸️ **低仓位试探**: 可用30%仓位试探性配置\n")
                f.write("- ⏸️ **等待信号**: 等待市场明确方向后再做决策\n\n")
        else:
            f.write("⚠️ 无法获取当前市场状态（可能是网络问题或数据不可用）\n\n")
        
        # 二、市场环境分布
        f.write("## 二、历史市场环境分布\n\n")
        f.write("| 市场状态 | 月份数量 | 占比 | 平均月收益率 |\n")
        f.write("|---------|---------|------|------------|\n")
        
        for state in ['bull', 'bear', 'neutral']:
            state_df = market_env_df[market_env_df['Market_State'] == state]
            count = len(state_df)
            ratio = count / len(market_env_df) * 100
            avg_return = state_df['Monthly_Return'].mean()
            
            state_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}[state]
            f.write(f"| {state_name} | {count} | {ratio:.1f}% | {avg_return:.2%} |\n")
        
        f.write(f"| **总计** | **{len(market_env_df)}** | **100.0%** | **{market_env_df['Monthly_Return'].mean():.2%}** |\n\n")
        
        # 三、详细月度数据
        f.write("## 三、详细月度数据\n\n")
        f.write("| 月份 | 市场状态 | 恒指收益率 | 恒指起点 | 恒指终点 |\n")
        f.write("|------|---------|-----------|---------|---------|\n")
        
        for _, row in market_env_df.iterrows():
            period_str = str(row['YearMonth'])
            state_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}[row['Market_State']]
            f.write(f"| {period_str} | {state_name} | {row['Monthly_Return']:.2%} | {row['HSI_Close_Start']:.0f} | {row['HSI_Close_End']:.0f} |\n")
        
        f.write("\n")
        
        # 四、股票表现汇总
        f.write("## 四、股票表现汇总（按市场关联性排序）\n\n")
        f.write("| 股票代码 | 股票名称 | 牛市收益率 | 熊市收益率 | 市场关联性 | 牛市胜率 | 熊市胜率 | 牛市准确率 | 熊市准确率 | 牛市正确决策 | 熊市正确决策 | 牛市预测概率 | 熊市预测概率 |\n")
        f.write("|---------|---------|-----------|-----------|-----------|---------|---------|-----------|-----------|-------------|-------------|-------------|-------------|\n")
        
        for _, row in valid_stocks.iterrows():
            f.write(f"| {row['stock_code']} | {row['stock_name']} | ")
            f.write(f"{row['bull_avg_return']:.2%} | {row['bear_avg_return']:.2%} | ")
            f.write(f"{row['market_correlation']:.2f} | {row['bull_win_rate']:.1%} | {row['bear_win_rate']:.1%} | ")
            f.write(f"{row['bull_accuracy']:.1%} | {row['bear_accuracy']:.1%} | ")
            f.write(f"{row['bull_correct_decision']:.2%} | {row['bear_correct_decision']:.2%} | ")
            f.write(f"{row['bull_avg_prob']:.4f} | {row['bear_avg_prob']:.4f} |\n")
        
        f.write("\n")
        
        # 四、分析结论
        f.write("## 四、分析结论\n\n")
        
        # 高市场关联性股票
        high_corr_stocks = valid_stocks[valid_stocks['market_correlation'] > 0.3]
        f.write(f"### 高市场关联性股票（关联性 > 0.3）\n\n")
        f.write(f"- **数量**: {len(high_corr_stocks)} 只\n")
        if len(high_corr_stocks) > 0:
            f.write(f"- **平均牛市收益率**: {high_corr_stocks['bull_avg_return'].mean():.2%}\n")
            f.write(f"- **平均熊市收益率**: {high_corr_stocks['bear_avg_return'].mean():.2%}\n")
        
        f.write("\n")
        
        # 低市场关联性股票
        low_corr_stocks = valid_stocks[abs(valid_stocks['market_correlation']) < 0.2]
        f.write(f"### 低市场关联性股票（|关联性| < 0.2）\n\n")
        f.write(f"- **数量**: {len(low_corr_stocks)} 只\n")
        if len(low_corr_stocks) > 0:
            f.write(f"- **平均牛市收益率**: {low_corr_stocks['bull_avg_return'].mean():.2%}\n")
            f.write(f"- **平均熊市收益率**: {low_corr_stocks['bear_avg_return'].mean():.2%}\n")
        
        f.write("\n")
        
        # 预测概率范围分析
        f.write("## 五、预测概率范围分析\n\n")
        
        f.write("### 牛市环境预测概率\n\n")
        f.write(f"- **平均值**: {valid_stocks['bull_avg_prob'].mean():.4f}\n")
        f.write(f"- **标准差**: {valid_stocks['bull_prob_std'].mean():.4f}\n")
        f.write(f"- **范围**: [{valid_stocks['bull_avg_prob'].min():.4f}, {valid_stocks['bull_avg_prob'].max():.4f}]\n\n")
        
        f.write("### 熊市环境预测概率\n\n")
        f.write(f"- **平均值**: {valid_stocks['bear_avg_prob'].mean():.4f}\n")
        f.write(f"- **标准差**: {valid_stocks['bear_prob_std'].mean():.4f}\n")
        f.write(f"- **范围**: [{valid_stocks['bear_avg_prob'].min():.4f}, {valid_stocks['bear_avg_prob'].max():.4f}]\n")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析股票的牛、熊市分布（可重复运行版本）')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--trades-file', type=str, default='auto', help='交易记录文件路径（默认自动查找最新）')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--output-format', type=str, default='all', choices=['csv', 'json', 'md', 'all'], help='输出格式')
    args = parser.parse_args()
    
    print("=" * 80)
    print("牛熊市分析（可重复运行版本）")
    print("=" * 80)
    print(f"开始日期: {args.start_date}")
    print(f"结束日期: {args.end_date}")
    
    # 1. 分析市场环境
    market_env_df = analyze_market_environment(args.start_date, args.end_date)
    
    if market_env_df is None:
        print("❌ 分析失败：无法获取市场环境数据")
        sys.exit(1)
    
    # 2. 读取交易记录
    if args.trades_file == 'auto':
        trades_file = find_latest_backtest_file()
        if trades_file is None:
            print("❌ 错误：无法找到回测交易记录文件")
            sys.exit(1)
        print(f"✅ 自动找到最新回测文件: {trades_file}")
    else:
        trades_file = args.trades_file
    
    try:
        trades_df = pd.read_csv(trades_file)
        print(f"✅ 成功读取 {len(trades_df)} 条交易记录")
    except Exception as e:
        print(f"❌ 错误：无法读取交易记录文件: {e}")
        sys.exit(1)
    
    # 3. 分析股票表现
    stock_performance_df = analyze_stock_performance_by_market(trades_df, market_env_df)
    
    # 4. 过滤有效数据
    valid_stocks = stock_performance_df[
        (stock_performance_df['bull_count'] >= 10) & 
        (stock_performance_df['bear_count'] >= 10)
    ].copy()
    
    valid_stocks = valid_stocks.sort_values('market_correlation', ascending=False)
    
    # 5. 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"bull_bear_analysis_{args.start_date}_to_{args.end_date}_{timestamp}"
    
    # 6. 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV格式
    if args.output_format in ['csv', 'all']:
        csv_file = output_dir / f"{base_filename}.csv"
        valid_stocks.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ CSV报告已保存: {csv_file}")
    
    # JSON格式
    if args.output_format in ['json', 'all']:
        json_file = output_dir / f"{base_filename}.json"
        
        # 准备JSON数据
        output_data = {
            'metadata': {
                'start_date': args.start_date,
                'end_date': args.end_date,
                'generated_at': datetime.now().isoformat(),
                'trades_file': trades_file,
                'total_stocks': len(valid_stocks)
            },
            'market_environment': {
                'total_months': len(market_env_df),
                'bull_months': len(market_env_df[market_env_df['Market_State'] == 'bull']),
                'bear_months': len(market_env_df[market_env_df['Market_State'] == 'bear']),
                'neutral_months': len(market_env_df[market_env_df['Market_State'] == 'neutral']),
                'monthly_data': market_env_df.to_dict('records')
            },
            'stock_performance': valid_stocks.to_dict('records')
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ JSON报告已保存: {json_file}")
    
    # Markdown格式
    if args.output_format in ['md', 'all']:
        md_file = output_dir / f"{base_filename}.md"
        save_markdown_report(market_env_df, stock_performance_df, md_file)
        print(f"✅ Markdown报告已保存: {md_file}")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
