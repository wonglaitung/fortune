#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试3月15日到30日的异常检测（带详细原因）
"""

import subprocess
import json
from datetime import datetime, timedelta
import sys
import os
import re

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def extract_anomalies_with_reasons(date_str):
    """提取异常及其原因"""
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
            # 查找具体的异常检测记录
            # 格式: "0700.HK 2026-03-16 检测到 stock low级异常: 多维特征异常（接近布林带上轨, 偏离20日均线5.8%）"
            if re.search(r'\d{4}\.HK.*检测到.*异常:', line):
                anomalies.append(line.strip())

        # 查找总异常数量
        total_anomalies = 0
        for line in lines:
            if re.search(r'检测到 \d+ 个异常', line):
                match = re.search(r'检测到 (\d+) 个异常', line)
                if match:
                    total_anomalies = int(match.group(1))
                break

        return {
            'date': date_str,
            'anomalies': anomalies,
            'total_anomalies': total_anomalies,
            'has_anomaly': total_anomalies > 0
        }

    except subprocess.TimeoutExpired:
        return {
            'date': date_str,
            'error': '检测超时',
            'has_anomaly': False,
            'total_anomalies': 0
        }
    except Exception as e:
        return {
            'date': date_str,
            'error': str(e),
            'has_anomaly': False,
            'total_anomalies': 0
        }

def main():
    """主函数"""
    print("开始测试3月15日到30日的异常检测（带详细原因）...")
    print("="*60)

    # 生成日期列表（2026-03-15 到 2026-03-30）
    start_date = datetime(2026, 3, 15)
    end_date = datetime(2026, 3, 30)

    results = []
    days_with_anomalies = 0
    total_anomalies = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        result = extract_anomalies_with_reasons(date_str)
        results.append(result)

        # 打印简要结果
        if result['has_anomaly']:
            print(f"✓ {date_str}: 发现 {result['total_anomalies']} 个异常")
            days_with_anomalies += 1
            total_anomalies += result['total_anomalies']
        else:
            print(f"  {date_str}: 无异常")

        current_date += timedelta(days=1)

    # 汇总结果
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")

    anomaly_dates = [r for r in results if r['has_anomaly']]

    if anomaly_dates:
        print(f"\n发现异常的日期 ({len(anomaly_dates)} 天):")
        for result in anomaly_dates:
            print(f"\n{result['date']} ({result['total_anomalies']} 个异常):")
            if result.get('anomalies'):
                for anomaly in result['anomalies']:
                    print(f"  - {anomaly}")
    else:
        print("\n✓ 所有测试日期均无异常")

    print(f"\n统计:")
    print(f"  - 测试天数: {len(results)}")
    print(f"  - 有异常的天数: {days_with_anomalies}")
    print(f"  - 无异常的天数: {len(results) - days_with_anomalies}")
    print(f"  - 总异常数量: {total_anomalies}")

    # 保存详细结果
    output_file = 'output/march_anomaly_with_reasons.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 生成详细报告
    report_file = 'output/march_anomaly_detailed_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("3月15日到30日股票异常检测报告（含详细原因）\n")
        f.write("="*60 + "\n\n")

        if anomaly_dates:
            f.write(f"发现异常的日期 ({len(anomaly_dates)} 天):\n\n")
            for result in anomaly_dates:
                f.write(f"{result['date']} ({result['total_anomalies']} 个异常):\n")
                if result.get('anomalies'):
                    for anomaly in result['anomalies']:
                        f.write(f"  - {anomaly}\n")
                f.write("\n")
        else:
            f.write("所有测试日期均无异常\n\n")

        f.write(f"统计:\n")
        f.write(f"  - 测试天数: {len(results)}\n")
        f.write(f"  - 有异常的天数: {days_with_anomalies}\n")
        f.write(f"  - 无异常的天数: {len(results) - days_with_anomalies}\n")
        f.write(f"  - 总异常数量: {total_anomalies}\n")

    print(f"详细报告已保存到: {report_file}")

if __name__ == '__main__':
    main()
