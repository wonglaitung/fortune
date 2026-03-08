#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理新回测数据 - 生成综合分析报告
基于 batch_backtest 生成的数据
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

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

def process_new_backtest_data(summary_file, detailed_file):
    """处理新的回测数据"""
    print("📊 加载回测数据...")
    
    # 加载汇总数据
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # 加载详细数据
    with open(detailed_file, 'r', encoding='utf-8') as f:
        detailed_data = json.load(f)
    
    print(f"✅ 加载了 {len(summary_data)} 只股票的汇总数据")
    
    # 构建股票汇总表
    stock_summary = []
    trades_all = []
    
    for i, stock_data in enumerate(summary_data):
        stock_code = stock_data['stock_code']
        stock_name = stock_data['stock_name']
        
        # 获取对应股票的详细数据
        detailed_stock = detailed_data[i]
        
        # 计算平均收益率（使用所有交易的收益率）
        total_return = stock_data['total_return'] * 100  # 转换为百分比
        total_trades = stock_data['total_trades']
        win_rate = stock_data['win_rate'] * 100  # 转换为百分比
        
        # 计算平均收益率（总收益率 / 交易次数）
        avg_return_per_trade = total_return / total_trades if total_trades > 0 else 0
        
        # 准确率和正确决策比例暂时设为NaN（需要从预测数据计算）
        accuracy = np.nan
        correct_decision_ratio = np.nan
        
        stock_summary.append({
            '股票代码': stock_code,
            '股票名称': stock_name,
            '交易次数': total_trades,
            '平均收益率': avg_return_per_trade,
            '总收益率': total_return,
            '胜率': win_rate,
            '准确率': accuracy,
            '正确决策比例': correct_decision_ratio
        })
        
        # 提取交易记录（用于月度分析）
        if 'winning_trades' in detailed_stock:
            for trade in detailed_stock['winning_trades']:
                # 计算收益率
                if 'buy_price' in trade and 'sell_price' in trade:
                    return_rate = (trade['sell_price'] - trade['buy_price']) / trade['buy_price'] * 100
                    trades_all.append({
                        'stock_code': stock_code,
                        'return_rate': return_rate,
                        'is_win': True
                    })
        
        if 'losing_trades' in detailed_stock:
            for trade in detailed_stock['losing_trades']:
                if 'buy_price' in trade and 'sell_price' in trade:
                    return_rate = (trade['sell_price'] - trade['buy_price']) / trade['buy_price'] * 100
                    trades_all.append({
                        'stock_code': stock_code,
                        'return_rate': return_rate,
                        'is_win': False
                    })
    
    # 创建DataFrame
    stock_summary_df = pd.DataFrame(stock_summary)
    
    # 计算总体平均
    overall_avg_return = stock_summary_df['平均收益率'].mean()
    overall_winrate = stock_summary_df['胜率'].mean()
    
    print(f"\n📈 总体统计:")
    print(f"  平均收益率: {overall_avg_return:.2f}%")
    print(f"  平均胜率: {overall_winrate:.2f}%")
    print(f"  总交易次数: {stock_summary_df['交易次数'].sum()}")
    
    return stock_summary_df, pd.DataFrame(trades_all)

def main():
    print("=" * 80)
    print("处理新回测数据 - 生成综合分析报告")
    print("=" * 80)
    
    # 新的回测数据文件
    summary_file = 'output/batch_backtest_catboost_20d_20260307_000644.json'
    detailed_file = 'output/batch_backtest_detailed_catboost_20d_20260307_000644.json'
    
    # 处理数据
    stock_summary_df, trades_df = process_new_backtest_data(summary_file, detailed_file)
    
    # 生成文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 股票汇总CSV
    stock_summary_csv = f'output/backtest_stock_summary_new_{timestamp}.csv'
    stock_summary_df.to_csv(stock_summary_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 股票汇总CSV已保存: {stock_summary_csv}")
    
    # 2. 交易记录CSV
    trades_csv = f'output/backtest_trades_new_{timestamp}.csv'
    trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 交易记录CSV已保存: {trades_csv}")
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    
    print("\n📊 股票汇总表（前10只）:")
    print(stock_summary_df.head(10)[['股票代码', '股票名称', '交易次数', '平均收益率', '总收益率', '胜率']].to_string(index=False))

if __name__ == '__main__':
    main()