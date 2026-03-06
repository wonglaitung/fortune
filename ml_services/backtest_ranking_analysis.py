#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年全年CatBoost 20天模型回测 - 按不同指标TOP 10排名分析
"""

import pandas as pd
import numpy as np

# 读取回测数据
trades_df = pd.read_csv('output/backtest_20d_trades_20260305_115839.csv')

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

# 计算每只股票的统计数据
stock_stats = trades_df.groupby('stock_code').agg({
    'actual_change': ['count', 'mean', 'median', 'std'],
    'prediction_correct': 'mean',
    'probability': 'mean'
}).round(4)
stock_stats.columns = ['交易次数', '平均收益率', '收益率中位数', '收益率标准差', '准确率', '平均预测概率']
stock_stats['胜率'] = trades_df.groupby('stock_code').apply(lambda x: (x['actual_change'] > 0).mean()).round(4)

print("=" * 100)
print("2025年全年CatBoost 20天模型回测 - 股票表现TOP 10排名（按不同指标）")
print("=" * 100)
print()

# 1. 按平均收益率排名
print("=" * 100)
print("一、按平均收益率排名 TOP 10")
print("=" * 100)
print(f"{'排名':<4} {'股票代码':<10} {'股票名称':<15} {'交易次数':>8} {'平均收益率':>12} {'胜率':>8} {'准确率':>8}")
print("-" * 100)

by_return = stock_stats.sort_values('平均收益率', ascending=False).head(10)
for i, (stock_code, row) in enumerate(by_return.iterrows(), 1):
    stock_name = STOCK_NAMES.get(stock_code, stock_code)
    print(f"{i:<4} {stock_code:<10} {stock_name:<15} {int(row['交易次数']):>8d} {row['平均收益率']*100:>11.2f}% {row['胜率']*100:>7.2f}% {row['准确率']*100:>7.2f}%")
print()

# 2. 按胜率排名
print("=" * 100)
print("二、按胜率排名 TOP 10")
print("=" * 100)
print(f"{'排名':<4} {'股票代码':<10} {'股票名称':<15} {'交易次数':>8} {'平均收益率':>12} {'胜率':>8} {'准确率':>8}")
print("-" * 100)

by_win_rate = stock_stats.sort_values('胜率', ascending=False).head(10)
for i, (stock_code, row) in enumerate(by_win_rate.iterrows(), 1):
    stock_name = STOCK_NAMES.get(stock_code, stock_code)
    print(f"{i:<4} {stock_code:<10} {stock_name:<15} {int(row['交易次数']):>8d} {row['平均收益率']*100:>11.2f}% {row['胜率']*100:>7.2f}% {row['准确率']*100:>7.2f}%")
print()

# 3. 按准确率排名
print("=" * 100)
print("三、按准确率排名 TOP 10")
print("=" * 100)
print(f"{'排名':<4} {'股票代码':<10} {'股票名称':<15} {'交易次数':>8} {'平均收益率':>12} {'胜率':>8} {'准确率':>8}")
print("-" * 100)

by_accuracy = stock_stats.sort_values('准确率', ascending=False).head(10)
for i, (stock_code, row) in enumerate(by_accuracy.iterrows(), 1):
    stock_name = STOCK_NAMES.get(stock_code, stock_code)
    print(f"{i:<4} {stock_code:<10} {stock_name:<15} {int(row['交易次数']):>8d} {row['平均收益率']*100:>11.2f}% {row['胜率']*100:>7.2f}% {row['准确率']*100:>7.2f}%")
print()

# 4. 综合分析 - 三项指标都进入前15的股票
print("=" * 100)
print("四、综合优秀股票（三项指标均在前15名）")
print("=" * 100)

top15_return = set(by_return.index[:15])
top15_win_rate = set(by_win_rate.index[:15])
top15_accuracy = set(by_accuracy.index[:15])

excellent_stocks = top15_return & top15_win_rate & top15_accuracy

if excellent_stocks:
    print(f"{'股票代码':<10} {'股票名称':<15} {'平均收益率排名':>15} {'胜率排名':>10} {'准确率排名':>10}")
    print("-" * 80)
    
    for stock_code in excellent_stocks:
        stock_name = STOCK_NAMES.get(stock_code, stock_code)
        return_rank = stock_stats.sort_values('平均收益率', ascending=False).index.get_loc(stock_code) + 1
        win_rate_rank = stock_stats.sort_values('胜率', ascending=False).index.get_loc(stock_code) + 1
        accuracy_rank = stock_stats.sort_values('准确率', ascending=False).index.get_loc(stock_code) + 1
        
        print(f"{stock_code:<10} {stock_name:<15} {return_rank:>13d}  {win_rate_rank:>8d}  {accuracy_rank:>8d}")
else:
    print("没有股票在三项指标中均进入前15名")
print()

# 5. 对比分析 - 不同排名的股票差异
print("=" * 100)
print("五、排名对比分析")
print("=" * 100)

# 平均收益率TOP 3 vs 准确率TOP 3
print(f"【平均收益率TOP 3 vs 准确率TOP 3】")
print(f"平均收益率TOP 3: {', '.join([f'{STOCK_NAMES.get(code, code)}' for code in by_return.index[:3]])}")
print(f"准确率TOP 3: {', '.join([f'{STOCK_NAMES.get(code, code)}' for code in by_accuracy.index[:3]])}")
print()

# 胜率TOP 3 vs 准确率TOP 3
print(f"【胜率TOP 3 vs 准确率TOP 3】")
print(f"胜率TOP 3: {', '.join([f'{STOCK_NAMES.get(code, code)}' for code in by_win_rate.index[:3]])}")
print(f"准确率TOP 3: {', '.join([f'{STOCK_NAMES.get(code, code)}' for code in by_accuracy.index[:3]])}")
print()

# 6. 统计总结
print("=" * 100)
print("六、统计总结")
print("=" * 100)

print(f"【平均收益率统计】")
print(f"  最高: {stock_stats['平均收益率'].max()*100:.2f}% ({STOCK_NAMES.get(stock_stats['平均收益率'].idxmax(), stock_stats['平均收益率'].idxmax())})")
print(f"  最低: {stock_stats['平均收益率'].min()*100:.2f}% ({STOCK_NAMES.get(stock_stats['平均收益率'].idxmin(), stock_stats['平均收益率'].idxmin())})")
print(f"  平均: {stock_stats['平均收益率'].mean()*100:.2f}%")
print(f"  中位数: {stock_stats['平均收益率'].median()*100:.2f}%")
print()

print(f"【胜率统计】")
print(f"  最高: {stock_stats['胜率'].max()*100:.2f}% ({STOCK_NAMES.get(stock_stats['胜率'].idxmax(), stock_stats['胜率'].idxmax())})")
print(f"  最低: {stock_stats['胜率'].min()*100:.2f}% ({STOCK_NAMES.get(stock_stats['胜率'].idxmin(), stock_stats['胜率'].idxmin())})")
print(f"  平均: {stock_stats['胜率'].mean()*100:.2f}%")
print(f"  中位数: {stock_stats['胜率'].median()*100:.2f}%")
print()

print(f"【准确率统计】")
print(f"  最高: {stock_stats['准确率'].max()*100:.2f}% ({STOCK_NAMES.get(stock_stats['准确率'].idxmax(), stock_stats['准确率'].idxmax())})")
print(f"  最低: {stock_stats['准确率'].min()*100:.2f}% ({STOCK_NAMES.get(stock_stats['准确率'].idxmin(), stock_stats['准确率'].idxmin())})")
print(f"  平均: {stock_stats['准确率'].mean()*100:.2f}%")
print(f"  中位数: {stock_stats['准确率'].median()*100:.2f}%")
print()

print("=" * 100)
print("分析完成")
print("=" * 100)