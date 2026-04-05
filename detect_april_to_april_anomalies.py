#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测去年4月1日到今年4月1日的异常并分析后表现
"""

import subprocess
import json
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_anomalies_for_date_range(start_date, end_date):
    """检测指定日期范围内的异常"""
    print(f"\n检测日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print("="*60)

    results = []
    total_anomalies = 0
    days_with_anomalies = 0

    # 生成日期列表
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')

        # 跳过周末
        if current_date.weekday() < 5:  # 0=周一, 4=周五
            # 检测异常
            cmd = [
                'python3', 'detect_stock_anomalies.py',
                '--mode', 'standalone',
                '--mode-type', 'deep',
                '--time-interval', 'day',
                '--date', date_str,
                '--no-email'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr

                # 提取异常信息
                anomalies = []
                lines = output.split('\n')
                for line in lines:
                    if r'\d{4}\.HK.*检测到.*异常:' in line:
                        anomalies.append(line.strip())

                # 查找总异常数量
                anomaly_count = 0
                for line in lines:
                    if r'检测到 \d+ 个异常' in line:
                        import re
                        match = re.search(r'检测到 (\d+) 个异常', line)
                        if match:
                            anomaly_count = int(match.group(1))
                        break

                if anomaly_count > 0:
                    days_with_anomalies += 1
                    total_anomalies += anomaly_count
                    print(f"✓ {date_str}: {anomaly_count} 个异常")
                    results.append({
                        'date': date_str,
                        'anomalies': anomalies,
                        'total_anomalies': anomaly_count
                    })
                else:
                    if current_date.day in [1, 15]:  # 只显示每月1号和15号的无异常信息
                        print(f"  {date_str}: 无异常")
            except subprocess.TimeoutExpired:
                print(f"⚠️ {date_str}: 检测超时")
            except Exception as e:
                print(f"❌ {date_str}: 错误 - {e}")

        current_date += timedelta(days=1)

    return results, total_anomalies, days_with_anomalies


def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    try:
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        # 统一时区
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"⚠️ 获取 {stock_code} 数据失败: {e}")
        return None


def analyze_post_anomaly_performance(anomalies_data):
    """分析异常后的股价表现"""
    print(f"\n{'='*60}")
    print("分析异常后的股价表现")
    print(f"{'='*60}")

    # 提取所有异常
    all_anomalies = []
    for date_data in anomalies_data:
        for anomaly_line in date_data['anomalies']:
            import re
            match = re.search(r'(\d{4}\.HK)', anomaly_line)
            if match:
                all_anomalies.append({
                    'date': date_data['date'],
                    'stock': match.group(1),
                    'original_line': anomaly_line
                })

    print(f"共找到 {len(all_anomalies)} 个异常")

    if len(all_anomalies) == 0:
        return None

    # 扩展时间范围以获取异常后的数据
    all_dates = [pd.Timestamp(a['date']) for a in all_anomalies]
    min_date = min(all_dates) - timedelta(days=30)
    max_date = max(all_dates) + timedelta(days=60)

    print(f"分析时间范围: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")

    # 分析每个异常后的表现
    results = []
    for i, anomaly in enumerate(all_anomalies):
        print(f"\n分析异常 {i+1}/{len(all_anomalies)}: {anomaly['stock']} {anomaly['date']}")

        # 获取股票数据
        df = get_stock_data(anomaly['stock'], min_date, max_date)
        if df is None or len(df) == 0:
            continue

        try:
            # 转换日期格式，统一时区
            anomaly_date = pd.Timestamp(anomaly['date'])
            if hasattr(anomaly_date, 'tz') and anomaly_date.tz is not None:
                anomaly_date = anomaly_date.tz_localize(None)

            # 确保异常日期在数据中
            if anomaly_date not in df.index:
                idx = df.index.get_indexer([anomaly_date], method='nearest')[0]
                anomaly_date = df.index[idx]

            # 获取异常日期的位置
            idx = df.index.get_loc(anomaly_date)

            # 计算异常后的收益率
            performance = {}
            for window in [1, 3, 5, 10, 20, 30, 60]:
                if idx + window < len(df):
                    future_close = df['Close'].iloc[idx + window]
                    current_close = df['Close'].iloc[idx]
                    return_pct = (future_close / current_close - 1) * 100
                    performance[f'return_{window}d'] = return_pct
                else:
                    performance[f'return_{window}d'] = None

            # 计算异常后的波动率
            if idx + 5 < len(df):
                post_anomaly_data = df['Close'].iloc[idx:idx+6]
                volatility = post_anomaly_data.pct_change().std() * 100
                performance['volatility_5d'] = volatility

            # 计算异常前的波动率
            if idx >= 5:
                pre_anomaly_data = df['Close'].iloc[idx-5:idx+1]
                volatility_pre = pre_anomaly_data.pct_change().std() * 100
                performance['volatility_pre_5d'] = volatility_pre

            # 计算异常当日涨跌幅
            if idx > 0:
                prev_close = df['Close'].iloc[idx-1]
                current_close = df['Close'].iloc[idx]
                daily_return = (current_close / prev_close - 1) * 100
                performance['daily_return'] = daily_return

            results.append({
                **anomaly,
                'performance': performance
            })

        except Exception as e:
            print(f"⚠️ 分析 {anomaly['stock']} 异常表现失败: {e}")
            continue

    print(f"\n成功分析 {len(results)} 个异常")
    return results


def generate_report(results, total_anomalies, days_with_anomalies):
    """生成分析报告"""
    print(f"\n{'='*60}")
    print("异常后股价表现统计")
    print(f"{'='*60}")

    if not results or len(results) == 0:
        print("❌ 没有异常数据可分析")
        return

    # 统计各时间窗口的收益率
    for window in [1, 3, 5, 10, 20, 30, 60]:
        returns = [r['performance'].get(f'return_{window}d') for r in results
                   if r['performance'].get(f'return_{window}d') is not None]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            positive_rate = sum(1 for r in returns if r > 0) / len(returns) * 100

            print(f"\n{window}天后:")
            print(f"  平均收益率: {avg_return:+.2f}%")
            print(f"  标准差: {std_return:.2f}%")
            print(f"  上涨概率: {positive_rate:.1f}%")
            print(f"  样本数: {len(returns)}")

    # 统计波动率变化
    volatilities = [r['performance'].get('volatility_5d') for r in results
                    if r['performance'].get('volatility_5d') is not None]
    volatilities_pre = [r['performance'].get('volatility_pre_5d') for r in results
                        if r['performance'].get('volatility_pre_5d') is not None]

    if volatilities:
        avg_vol = np.mean(volatilities)
        print(f"\n异常后5日平均波动率: {avg_vol:.2f}%")

    if volatilities_pre:
        avg_vol_pre = np.mean(volatilities_pre)
        print(f"异常前5日平均波动率: {avg_vol_pre:.2f}%")

    if volatilities and volatilities_pre:
        vol_change = (avg_vol - avg_vol_pre) / avg_vol_pre * 100
        print(f"波动率变化: {vol_change:+.1f}%")

    # 统计异常当日涨跌幅
    daily_returns = [r['performance'].get('daily_return') for r in results
                     if r['performance'].get('daily_return') is not None]
    if daily_returns:
        avg_daily = np.mean(daily_returns)
        print(f"\n异常当日平均涨跌幅: {avg_daily:+.2f}%")

    # 保存结果
    output_file = 'output/april_to_april_anomalies_with_reasons.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")


def main():
    """主函数"""
    print("开始检测去年4月1日到今年4月1日的异常...")
    print("="*60)

    # 定义日期范围
    start_date = datetime(2025, 4, 1)
    end_date = datetime(2026, 4, 1)

    # 检测异常
    anomalies_data, total_anomalies, days_with_anomalies = detect_anomalies_for_date_range(start_date, end_date)

    # 汇总统计
    print(f"\n{'='*60}")
    print("异常检测汇总")
    print(f"{'='*60}")
    print(f"总检测天数: {len([d for d in pd.date_range(start_date, end_date) if d.weekday() < 5])}")
    print(f"有异常的天数: {days_with_anomalies}")
    print(f"无异常的天数: {days_with_anomalies}")
    print(f"总异常数量: {total_anomalies}")

    # 分析异常后的表现
    if anomalies_data:
        results = analyze_post_anomaly_performance(anomalies_data)
        if results:
            generate_report(results, total_anomalies, days_with_anomalies)


if __name__ == '__main__':
    main()
