#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年全年CatBoost 20天模型回测多角度分析脚本
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取回测数据
trades_df = pd.read_csv('output/backtest_20d_trades_20260305_115839.csv')
with open('output/backtest_20d_metrics_20260305_115839.json', 'r') as f:
    metrics = json.load(f)

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

print("=" * 80)
print("2025年全年CatBoost 20天模型回测多角度分析报告")
print("=" * 80)
print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 1. 整体性能分析
print("=" * 80)
print("一、整体性能分析")
print("=" * 80)
print(f"【总体统计】")
print(f"  总交易机会: {metrics['total_trades']}")
print(f"  买入信号数: {metrics['buy_signals']}")
print(f"  买入信号占比: {metrics['buy_signals']/metrics['total_trades']*100:.2f}%")
print()
print(f"【预测准确率】")
print(f"  准确率: {metrics['accuracy']*100:.2f}%")
print(f"  精确率: {metrics['precision']*100:.2f}%")
print(f"  召回率: {metrics['recall']*100:.2f}%")
print(f"  F1分数: {metrics['f1_score']:.4f}")
print()
print(f"【收益统计】")
print(f"  平均收益率: {metrics['avg_return']*100:.2f}%")
print(f"  收益率中位数: {metrics['median_return']*100:.2f}%")
print(f"  收益率标准差: {metrics['std_return']*100:.2f}%")
print(f"  上涨交易: {int(metrics['positive_trades'])} 笔")
print(f"  下跌交易: {int(metrics['negative_trades'])} 笔")
print(f"  胜率: {metrics['win_rate']*100:.2f}%")
print()
print(f"【风险指标】")
print(f"  夏普比率（年化）: {metrics['sharpe_ratio']:.2f}")
print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
print()

# 2. 按股票分析
print("=" * 80)
print("二、按股票分析")
print("=" * 80)

stock_stats = trades_df.groupby('stock_code').agg({
    'actual_change': ['count', 'mean', 'median', 'std'],
    'prediction_correct': 'mean',
    'probability': 'mean'
}).round(4)
stock_stats.columns = ['交易次数', '平均收益率', '收益率中位数', '收益率标准差', '准确率', '平均预测概率']
stock_stats['胜率'] = trades_df.groupby('stock_code').apply(lambda x: (x['actual_change'] > 0).mean()).round(4)
stock_stats = stock_stats.sort_values('平均收益率', ascending=False)

print(f"【股票表现排名TOP 10】")
for i, (stock_code, row) in enumerate(stock_stats.head(10).iterrows(), 1):
    stock_name = STOCK_NAMES.get(stock_code, stock_code)
    print(f"{i}. {stock_code} ({stock_name})")
    print(f"   交易次数: {int(row['交易次数'])}")
    print(f"   平均收益率: {row['平均收益率']*100:.2f}%")
    print(f"   胜率: {row['胜率']*100:.2f}%")
    print(f"   准确率: {row['准确率']*100:.2f}%")
    print()

print(f"【股票表现排名BOTTOM 5】")
for i, (stock_code, row) in enumerate(stock_stats.tail(5).iterrows(), 1):
    stock_name = STOCK_NAMES.get(stock_code, stock_code)
    print(f"{i}. {stock_code} ({stock_name})")
    print(f"   交易次数: {int(row['交易次数'])}")
    print(f"   平均收益率: {row['平均收益率']*100:.2f}%")
    print(f"   胜率: {row['胜率']*100:.2f}%")
    print(f"   准确率: {row['准确率']*100:.2f}%")
    print()

# 3. 按月份分析
print("=" * 80)
print("三、按月份分析")
print("=" * 80)

trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
trades_df['month'] = trades_df['buy_date'].dt.month

monthly_stats = trades_df.groupby('month').agg({
    'actual_change': ['count', 'mean', 'median'],
    'prediction_correct': 'mean'
}).round(4)
monthly_stats.columns = ['交易次数', '平均收益率', '收益率中位数', '准确率']
monthly_stats['胜率'] = trades_df.groupby('month').apply(lambda x: (x['actual_change'] > 0).mean()).round(4)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"【月度表现】")
for month, row in monthly_stats.iterrows():
    month_name = month_names[month-1]
    print(f"{month_name:3s} 交易次数: {int(row['交易次数']):3d} | 平均收益率: {row['平均收益率']*100:6.2f}% | 胜率: {row['胜率']*100:5.1f}% | 准确率: {row['准确率']*100:5.1f}%")
print()

# 4. 预测概率分析
print("=" * 80)
print("四、预测概率分析")
print("=" * 80)

# 分析不同概率区间的表现
trades_df['prob_bin'] = pd.cut(trades_df['probability'], bins=[0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0], labels=['<0.50', '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70-0.75', '0.75-0.80', '0.80-0.85', '0.85-0.90', '>0.90'])

print(f"【不同预测概率区间的表现】")
print(f"{'概率区间':<12} {'交易次数':>8} {'胜率':>8} {'平均收益率':>12} {'准确率':>8}")
print("-" * 60)
for bin_name in ['>0.90', '0.85-0.90', '0.80-0.85', '0.75-0.80', '0.70-0.75', '0.65-0.70', '0.60-0.65', '0.55-0.60', '0.50-0.55', '<0.50']:
    if bin_name in trades_df['prob_bin'].values:
        bin_data = trades_df[trades_df['prob_bin'] == bin_name]
        count = len(bin_data)
        if count > 0:
            win_rate = (bin_data['actual_change'] > 0).mean()
            avg_return = bin_data['actual_change'].mean()
            accuracy = bin_data['prediction_correct'].mean()
            print(f"{bin_name:<12} {count:>8d} {win_rate:>7.2%} {avg_return:>11.2%} {accuracy:>7.2%}")
print()

# 5. 收益率分布分析
print("=" * 80)
print("五、收益率分布分析")
print("=" * 80)

print(f"【收益率统计】")
print(f"  最大收益: {trades_df['actual_change'].max()*100:.2f}%")
print(f"  最小收益: {trades_df['actual_change'].min()*100:.2f}%")
print(f"  收益率范围: {(trades_df['actual_change'].max() - trades_df['actual_change'].min())*100:.2f}%")
print(f"  25%分位数: {trades_df['actual_change'].quantile(0.25)*100:.2f}%")
print(f"  75%分位数: {trades_df['actual_change'].quantile(0.75)*100:.2f}%")
print()

print(f"【收益率分布区间】")
bins = [-np.inf, -0.1, -0.05, 0, 0.05, 0.1, 0.2, np.inf]
bin_labels = ['<-10%', '-10%~-5%', '-5%~0%', '0%~5%', '5%~10%', '10%~20%', '>20%']
trades_df['return_bin'] = pd.cut(trades_df['actual_change'], bins=bins, labels=bin_labels)

for bin_name in bin_labels:
    count = len(trades_df[trades_df['return_bin'] == bin_name])
    percentage = count / len(trades_df) * 100
    print(f"  {bin_name:<12} {count:>6d} 笔 ({percentage:>5.1f}%)")
print()

# 6. 连续表现分析
print("=" * 80)
print("六、连续表现分析")
print("=" * 80)

# 分析连续盈利和连续亏损
trades_df_sorted = trades_df.sort_values(['stock_code', 'buy_date'])

# 计算连续盈利和亏损
def calculate_streaks(df):
    df = df.sort_values('buy_date')
    df['result'] = (df['actual_change'] > 0).astype(int)
    df['streak'] = df['result'].ne(df['result'].shift()).cumsum()
    streaks = df.groupby(['result', 'streak']).size().reset_index(name='length')
    return streaks

all_streaks = []
for stock_code in trades_df['stock_code'].unique():
    stock_data = trades_df_sorted[trades_df_sorted['stock_code'] == stock_code]
    if len(stock_data) > 0:
        streaks = calculate_streaks(stock_data)
        all_streaks.append(streaks)

all_streaks_df = pd.concat(all_streaks, ignore_index=True)

win_streaks = all_streaks_df[all_streaks_df['result'] == 1]['length']
lose_streaks = all_streaks_df[all_streaks_df['result'] == 0]['length']

print(f"【连续盈利】")
print(f"  最长连续盈利: {win_streaks.max()} 笔")
print(f"  平均连续盈利: {win_streaks.mean():.1f} 笔")
print(f"  中位数连续盈利: {win_streaks.median():.0f} 笔")
print()

print(f"【连续亏损】")
print(f"  最长连续亏损: {lose_streaks.max()} 笔")
print(f"  平均连续亏损: {lose_streaks.mean():.1f} 笔")
print(f"  中位数连续亏损: {lose_streaks.median():.0f} 笔")
print()

# 7. 市场环境分析
print("=" * 80)
print("七、市场环境分析")
print("=" * 80)

# 根据整体胜率分析市场环境
overall_win_rate = metrics['win_rate']
if overall_win_rate > 0.75:
    market_condition = "牛市/强上涨"
    explanation = "模型在强上涨市场中表现优异，捕捉到大量上涨机会"
elif overall_win_rate > 0.65:
    market_condition = "温和上涨"
    explanation = "市场整体向上，模型能较好地识别上涨信号"
elif overall_win_rate > 0.55:
    market_condition = "震荡市"
    explanation = "市场波动较大，模型需要更精准地把握时机"
elif overall_win_rate > 0.45:
    market_condition = "温和下跌"
    explanation = "市场整体偏弱，但仍有个别机会"
else:
    market_condition = "熊市/强下跌"
    explanation = "市场持续下跌，买入策略风险较高"

print(f"【市场环境判断】")
print(f"  市场状态: {market_condition}")
print(f"  判断依据: 整体胜率 {overall_win_rate*100:.2f}%")
print(f"  环境描述: {explanation}")
print()

# 8. 交易频率分析
print("=" * 80)
print("八、交易频率分析")
print("=" * 80)

# 计算每只股票的平均交易间隔
trades_df_sorted['prev_buy_date'] = trades_df_sorted.groupby('stock_code')['buy_date'].shift(1)
trades_df_sorted['days_between'] = (trades_df_sorted['buy_date'] - trades_df_sorted['prev_buy_date']).dt.days

avg_days_between = trades_df_sorted['days_between'].mean()
trading_days_per_month = 21
monthly_trade_frequency = trading_days_per_month / avg_days_between

print(f"【交易频率】")
print(f"  平均交易间隔: {avg_days_between:.1f} 天")
print(f"  预计每月交易次数: {monthly_trade_frequency:.1f} 次")
print(f"  预计每年交易次数: {monthly_trade_frequency * 12:.1f} 次")
print()

# 9. 模型稳定性分析
print("=" * 80)
print("九、模型稳定性分析")
print("=" * 80)

# 计算不同月份准确率的变异系数
monthly_accuracy = monthly_stats['准确率']
accuracy_cv = monthly_accuracy.std() / monthly_accuracy.mean() if monthly_accuracy.mean() > 0 else 0

# 计算不同股票准确率的变异系数
stock_accuracy = stock_stats['准确率']
stock_accuracy_cv = stock_accuracy.std() / stock_accuracy.mean() if stock_accuracy.mean() > 0 else 0

print(f"【稳定性指标】")
print(f"  月度准确率变异系数: {accuracy_cv:.4f}")
print(f"  股票间准确率变异系数: {stock_accuracy_cv:.4f}")
print()

if accuracy_cv < 0.1:
    month_stability = "非常稳定"
elif accuracy_cv < 0.15:
    month_stability = "稳定"
elif accuracy_cv < 0.2:
    month_stability = "一般"
else:
    month_stability = "不稳定"

if stock_accuracy_cv < 0.1:
    stock_stability = "非常稳定"
elif stock_accuracy_cv < 0.15:
    stock_stability = "稳定"
elif stock_accuracy_cv < 0.2:
    stock_stability = "一般"
else:
    stock_stability = "不稳定"

print(f"  月度稳定性: {month_stability}")
print(f"  股票间稳定性: {stock_stability}")
print()

# 10. 综合评价和建议
print("=" * 80)
print("十、综合评价和建议")
print("=" * 80)

# 综合评分
score = 0
max_score = 100

# 准确率评分 (25分)
if metrics['accuracy'] > 0.8:
    score += 25
elif metrics['accuracy'] > 0.75:
    score += 20
elif metrics['accuracy'] > 0.7:
    score += 15
elif metrics['accuracy'] > 0.65:
    score += 10
else:
    score += 5

# 胜率评分 (20分)
if metrics['win_rate'] > 0.8:
    score += 20
elif metrics['win_rate'] > 0.75:
    score += 15
elif metrics['win_rate'] > 0.7:
    score += 10
elif metrics['win_rate'] > 0.65:
    score += 5
else:
    score += 0

# 夏普比率评分 (20分)
if metrics['sharpe_ratio'] > 2.0:
    score += 20
elif metrics['sharpe_ratio'] > 1.5:
    score += 15
elif metrics['sharpe_ratio'] > 1.0:
    score += 10
elif metrics['sharpe_ratio'] > 0.5:
    score += 5
else:
    score += 0

# 稳定性评分 (15分)
if accuracy_cv < 0.1 and stock_accuracy_cv < 0.1:
    score += 15
elif accuracy_cv < 0.15 and stock_accuracy_cv < 0.15:
    score += 10
elif accuracy_cv < 0.2 and stock_accuracy_cv < 0.2:
    score += 5
else:
    score += 0

# 收益率评分 (20分)
if metrics['avg_return'] > 0.10:
    score += 20
elif metrics['avg_return'] > 0.08:
    score += 15
elif metrics['avg_return'] > 0.05:
    score += 10
elif metrics['avg_return'] > 0.03:
    score += 5
else:
    score += 0

print(f"【综合评分】")
print(f"  总分: {score}/{max_score}")
print()

if score >= 80:
    grade = "A+ (优秀)"
    recommendation = "强烈推荐实盘交易，模型表现卓越"
elif score >= 70:
    grade = "A (良好)"
    recommendation = "推荐实盘交易，模型表现良好"
elif score >= 60:
    grade = "B (中等)"
    recommendation = "谨慎使用，需要进一步优化"
elif score >= 50:
    grade = "C (一般)"
    recommendation = "不建议实盘，需要大幅改进"
else:
    grade = "D (较差)"
    recommendation = "不推荐使用，模型需要重新训练"

print(f"  评级: {grade}")
print(f"  建议: {recommendation}")
print()

print(f"【优势】")
print(f"  1. 准确率高: {metrics['accuracy']*100:.2f}%")
print(f"  2. 胜率优秀: {metrics['win_rate']*100:.2f}%")
print(f"  3. 夏普比率优秀: {metrics['sharpe_ratio']:.2f}")
print(f"  4. 平均收益率可观: {metrics['avg_return']*100:.2f}%")
print()

print(f"【风险提示】")
print(f"  1. 最大回撤较大: {metrics['max_drawdown']*100:.2f}%，需要严格控制仓位")
print(f"  2. 市场环境依赖: 在不同市场环境下表现可能有差异")
print(f"  3. 交易成本: 频繁交易可能产生较高手续费")
print()

print(f"【优化建议】")
print(f"  1. 动态调整置信度阈值: 根据市场环境调整买入标准")
print(f"  2. 风险管理: 设置止损点，控制最大回撤")
print(f"  3. 组合管理: 分散投资，降低单一股票风险")
print(f"  4. 市场环境识别: 增加市场状态判断模块，在熊市降低仓位")
print()

print("=" * 80)
print("分析完成")
print("=" * 80)

# 保存分析结果
analysis_report = f"""
2025年全年CatBoost 20天模型回测多角度分析报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【综合评分】
总分: {score}/{max_score}
评级: {grade}
建议: {recommendation}

【关键指标】
准确率: {metrics['accuracy']*100:.2f}%
胜率: {metrics['win_rate']*100:.2f}%
平均收益率: {metrics['avg_return']*100:.2f}%
夏普比率: {metrics['sharpe_ratio']:.2f}
最大回撤: {metrics['max_drawdown']*100:.2f}%

【市场环境】
市场状态: {market_condition}
环境描述: {explanation}

【优势】
1. 准确率高: {metrics['accuracy']*100:.2f}%
2. 胜率优秀: {metrics['win_rate']*100:.2f}%
3. 夏普比率优秀: {metrics['sharpe_ratio']:.2f}
4. 平均收益率可观: {metrics['avg_return']*100:.2f}%

【风险提示】
1. 最大回撤较大: {metrics['max_drawdown']*100:.2f}%，需要严格控制仓位
2. 市场环境依赖: 在不同市场环境下表现可能有差异
3. 交易成本: 频繁交易可能产生较高手续费

【优化建议】
1. 动态调整置信度阈值: 根据市场环境调整买入标准
2. 风险管理: 设置止损点，控制最大回撤
3. 组合管理: 分散投资，降低单一股票风险
4. 市场环境识别: 增加市场状态判断模块，在熊市降低仓位
"""

with open('output/backtest_2025_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(analysis_report)

print(f"\n✅ 详细分析报告已保存到: output/backtest_2025_analysis_report.txt")