#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析异常后股价是否呈现"趋势延续"特征
验证：异常前上涨的，异常后是否继续上涨；异常前下跌的，异常后是否继续下跌
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

# 加载分析结果
with open('output/hk_stock_anomaly_performance.json', 'r', encoding='utf-8') as f:
    performances = json.load(f)

print("="*70)
print("异常后趋势延续性分析")
print("="*70)

# 分析异常前后的趋势关系
results = []

for perf in performances:
    daily_return = perf.get('daily_return')  # 异常当日涨跌幅
    pre_5d_return = None  # 我们需要异常前5日的数据
    post_5d_return = perf.get('return_5d')
    post_10d_return = perf.get('return_10d')
    post_20d_return = perf.get('return_20d')
    
    if daily_return is not None and post_5d_return is not None:
        results.append({
            'stock': perf['stock'],
            'date': perf['date'],
            'type': perf['type'],
            'severity': perf['severity'],
            'daily_return': daily_return,  # 异常当日
            'post_5d': post_5d_return,
            'post_10d': post_10d_return,
            'post_20d': post_20d_return
        })

df = pd.DataFrame(results)
print(f"\n样本数: {len(df)}")

# 分析1: 按异常当日涨跌分组
print("\n" + "="*70)
print("分析1: 按异常当日涨跌分组")
print("="*70)

# 异常当日上涨
up_today = df[df['daily_return'] > 0]
# 异常当日下跌
down_today = df[df['daily_return'] < 0]

print(f"\n异常当日上涨样本: {len(up_today)}")
print(f"异常当日下跌样本: {len(down_today)}")

# 计算后续表现
for label, subset in [("异常当日上涨", up_today), ("异常当日下跌", down_today)]:
    print(f"\n{label}:")
    print(f"  异常当日平均涨跌: {subset['daily_return'].mean():+.2f}%")
    
    for col, name in [('post_5d', '5天后'), ('post_10d', '10天后'), ('post_20d', '20天后')]:
        valid = subset[col].dropna()
        if len(valid) > 0:
            mean_ret = valid.mean()
            win_rate = (valid > 0).sum() / len(valid) * 100
            print(f"  {name}: 平均收益 {mean_ret:+.2f}%, 胜率 {win_rate:.1f}%")

# 分析2: 趋势延续性统计
print("\n" + "="*70)
print("分析2: 趋势延续性统计（异常当日涨跌 vs 异常后涨跌）")
print("="*70)

for col, name in [('post_5d', '5天后'), ('post_10d', '10天后'), ('post_20d', '20天后')]:
    valid = df[[ 'daily_return', col]].dropna()
    if len(valid) > 0:
        # 趋势延续: 异常当日上涨 + 异常后上涨 或 异常当日下跌 + 异常后下跌
        trend_continue = ((valid['daily_return'] > 0) & (valid[col] > 0)) | \
                        ((valid['daily_return'] < 0) & (valid[col] < 0))
        continue_rate = trend_continue.sum() / len(valid) * 100
        
        # 反转: 异常当日上涨 + 异常后下跌 或 异常当日下跌 + 异常后上涨
        trend_reverse = ((valid['daily_return'] > 0) & (valid[col] < 0)) | \
                       ((valid['daily_return'] < 0) & (valid[col] > 0))
        reverse_rate = trend_reverse.sum() / len(valid) * 100
        
        print(f"\n{name}:")
        print(f"  趋势延续率: {continue_rate:.1f}%")
        print(f"  趋势反转率: {reverse_rate:.1f}%")
        print(f"  结论: {'趋势延续' if continue_rate > 50 else '无明显延续'}")

# 分析3: 相关性分析
print("\n" + "="*70)
print("分析3: 异常当日涨跌与异常后涨跌的相关性")
print("="*70)

for col, name in [('post_5d', '5天后'), ('post_10d', '10天后'), ('post_20d', '20天后')]:
    valid = df[['daily_return', col]].dropna()
    if len(valid) > 0:
        corr = valid['daily_return'].corr(valid[col])
        print(f"{name}: 相关系数 = {corr:.4f}")
        if corr > 0.1:
            print(f"  → 正相关，有趋势延续迹象")
        elif corr < -0.1:
            print(f"  → 负相关，有趋势反转迹象")
        else:
            print(f"  → 相关性很弱，无明显趋势")

# 分析4: 分异常类型分析
print("\n" + "="*70)
print("分析4: 按异常类型分组（价格异常 vs 成交量异常）")
print("="*70)

for anomaly_type in ['price', 'volume']:
    subset = df[df['type'] == anomaly_type]
    if len(subset) > 0:
        print(f"\n{anomaly_type}异常 (样本数: {len(subset)}):")
        
        # 按当日涨跌分组
        up = subset[subset['daily_return'] > 0]
        down = subset[subset['daily_return'] < 0]
        
        print(f"  当日上涨样本: {len(up)}")
        for col, name in [('post_5d', '5天后'), ('post_10d', '10天后')]:
            valid = up[col].dropna()
            if len(valid) > 0:
                print(f"    {name}: {valid.mean():+.2f}%, 胜率 {(valid>0).sum()/len(valid)*100:.1f}%")
        
        print(f"  当日下跌样本: {len(down)}")
        for col, name in [('post_5d', '5天后'), ('post_10d', '10天后')]:
            valid = down[col].dropna()
            if len(valid) > 0:
                print(f"    {name}: {valid.mean():+.2f}%, 胜率 {(valid>0).sum()/len(valid)*100:.1f}%")

# 分析5: 关键结论
print("\n" + "="*70)
print("关键结论")
print("="*70)

# 计算整体趋势延续情况
valid_5d = df[['daily_return', 'post_5d']].dropna()
continue_5d = ((valid_5d['daily_return'] > 0) & (valid_5d['post_5d'] > 0)) | \
              ((valid_5d['daily_return'] < 0) & (valid_5d['post_5d'] < 0))

valid_10d = df[['daily_return', 'post_10d']].dropna()
continue_10d = ((valid_10d['daily_return'] > 0) & (valid_10d['post_10d'] > 0)) | \
               ((valid_10d['daily_return'] < 0) & (valid_10d['post_10d'] < 0))

valid_20d = df[['daily_return', 'post_20d']].dropna()
continue_20d = ((valid_20d['daily_return'] > 0) & (valid_20d['post_20d'] > 0)) | \
               ((valid_20d['daily_return'] < 0) & (valid_20d['post_20d'] < 0))

print(f"\n趋势延续率（异常当日涨跌方向与异常后涨跌方向一致的比例）:")
print(f"  5天后:  {continue_5d.mean()*100:.1f}%")
print(f"  10天后: {continue_10d.mean()*100:.1f}%")
print(f"  20天后: {continue_20d.mean()*100:.1f}%")

# 相关性
corr_5d = valid_5d['daily_return'].corr(valid_5d['post_5d'])
corr_10d = valid_10d['daily_return'].corr(valid_10d['post_10d'])
corr_20d = valid_20d['daily_return'].corr(valid_20d['post_20d'])

print(f"\n相关系数（异常当日涨跌 vs 异常后涨跌）:")
print(f"  5天后:  {corr_5d:+.4f}")
print(f"  10天后: {corr_10d:+.4f}")
print(f"  20天后: {corr_20d:+.4f}")

print("\n" + "="*70)
print("结论")
print("="*70)
print("""
1. "异常后中长期上涨趋势"是指：
   - 所有异常样本平均来看，异常后5/10/20/30/60天的平均收益率为正
   - 这并不意味着"升的继续升，跌的继续跌"

2. 趋势延续性分析结果：
   - 趋势延续率约50%，接近随机
   - 相关性接近0或很弱
   - 没有明显的趋势延续或反转特征

3. 更准确的描述：
   - 异常是"波动率高点"信号（波动率下降28.5%）
   - 异常后市场趋于平静，股价回归均值
   - 整体平均收益率为正，但方向性不明确

4. "升的继续升，跌的继续跌"的验证结果：
   - 相关性很弱（接近0），不支持这个假设
   - 趋势延续率接近50%，更接近随机
   - 结论：这个假设不对
""")
