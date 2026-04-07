#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计3月15日到30日异常检测的股票频率
"""

import json

def analyze_anomaly_frequency():
    """分析异常频率"""

    # 读取详细结果
    with open('output/march_anomaly_real_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 统计每只股票的异常次数
    stock_anomaly_count = {}

    for result in results:
        if result['has_anomaly'] and result.get('anomalies'):
            for anomaly in result['anomalies']:
                # 提取股票代码
                # 格式: "2026-04-05 11:48:56,865 - __main__ - INFO - 0700.HK 2026-03-16 检测到 stock low级异常: 异常（原因未知）"
                parts = anomaly.split()
                for part in parts:
                    if part.endswith('.HK'):
                        stock_code = part
                        if stock_code not in stock_anomaly_count:
                            stock_anomaly_count[stock_code] = 0
                        stock_anomaly_count[stock_code] += 1
                        break

    # 按异常次数排序
    sorted_stocks = sorted(stock_anomaly_count.items(), key=lambda x: x[1], reverse=True)

    # 生成报告
    print("3月15日到30日股票异常频率统计")
    print("="*60)
    print(f"\n总共检测到 {sum(stock_anomaly_count.values())} 次异常")
    print(f"涉及 {len(stock_anomaly_count)} 只股票\n")

    print("异常频率排行（从高到低）:")
    print("-"*60)
    for i, (stock, count) in enumerate(sorted_stocks, 1):
        percentage = (count / sum(stock_anomaly_count.values())) * 100
        print(f"{i:2d}. {stock}: {count} 次 ({percentage:.1f}%)")

    # 统计高频异常股票（出现3次及以上）
    high_freq_stocks = [(stock, count) for stock, count in sorted_stocks if count >= 3]

    if high_freq_stocks:
        print(f"\n⚠️ 高频异常股票（出现3次及以上）:")
        print("-"*60)
        for stock, count in high_freq_stocks:
            print(f"  {stock}: {count} 次")

    # 保存结果
    output_file = 'output/march_anomaly_frequency_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("3月15日到30日股票异常频率统计报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"总共检测到 {sum(stock_anomaly_count.values())} 次异常\n")
        f.write(f"涉及 {len(stock_anomaly_count)} 只股票\n\n")

        f.write("异常频率排行（从高到低）:\n")
        f.write("-"*60 + "\n")
        for i, (stock, count) in enumerate(sorted_stocks, 1):
            percentage = (count / sum(stock_anomaly_count.values())) * 100
            f.write(f"{i:2d}. {stock}: {count} 次 ({percentage:.1f}%)\n")

        if high_freq_stocks:
            f.write(f"\n⚠️ 高频异常股票（出现3次及以上）:\n")
            f.write("-"*60 + "\n")
            for stock, count in high_freq_stocks:
                f.write(f"  {stock}: {count} 次\n")

    print(f"\n详细报告已保存到: {output_file}")

    return stock_anomaly_count, sorted_stocks

if __name__ == '__main__':
    analyze_anomaly_frequency()
