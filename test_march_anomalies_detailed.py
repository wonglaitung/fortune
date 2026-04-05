#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试3月15日到30日的股票异常检测
包括Z-Score和Isolation Forest检测
"""

import subprocess
import json
from datetime import datetime, timedelta
import sys
import os
import re

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_date_detailed(date_str):
    """测试指定日期的异常 - 详细版本"""
    print(f"\n{'='*60}")
    print(f"测试日期: {date_str}")
    print(f"{'='*60}")

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

        # 查找所有可能的异常信息
        all_info = []

        # 查找检测到的异常
        lines = output.split('\n')
        current_stock = None
        anomalies_for_stock = []

        for line in lines:
            # 识别股票代码
            if re.match(r'\d{4}\.HK.*异常', line):
                current_stock = line.split()[0]
                continue

            # 查找异常信息
            if '异常' in line and '检测' not in line:
                # 排除"No anomalies"和正常的日志信息
                if 'No anomalies' not in line and 'INFO' not in line:
                    if current_stock:
                        anomalies_for_stock.append(f"{current_stock}: {line.strip()}")
                    else:
                        all_info.append(line.strip())

        # 查找Z-Score信息
        zscore_anomalies = []
        for line in lines:
            if 'Z-Score' in line or 'zscore' in line:
                zscore_anomalies.append(line.strip())

        # 查找Isolation Forest信息
        if_anomalies = []
        for line in lines:
            if 'Isolation Forest' in line or 'isolation forest' in line.lower():
                if_anomalies.append(line.strip())

        # 汇总该日期的检测结果
        result_summary = {
            'date': date_str,
            'zscore_anomalies': zscore_anomalies,
            'isolation_forest_anomalies': if_anomalies,
            'other_anomalies': anomalies_for_stock + all_info,
            'has_anomaly': len(anomalies_for_stock + all_info) > 0
        }

        # 打印简要结果
        if result_summary['has_anomaly']:
            print(f"⚠️ 发现异常:")
            for anomaly in anomalies_for_stock + all_info:
                print(f"  - {anomaly}")
        else:
            print("✓ 无异常")

        return result_summary

    except subprocess.TimeoutExpired:
        print(f"⚠️ 超时")
        return {
            'date': date_str,
            'error': '检测超时',
            'has_anomaly': True
        }
    except Exception as e:
        print(f"❌ 错误: {e}")
        return {
            'date': date_str,
            'error': str(e),
            'has_anomaly': True
        }

def main():
    """主函数"""
    print("开始详细测试3月15日到30日的股票异常检测...")
    print("="*60)

    # 生成日期列表（2026-03-15 到 2026-03-30）
    start_date = datetime(2026, 3, 15)
    end_date = datetime(2026, 3, 30)

    results = []
    days_with_anomalies = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        result = test_date_detailed(date_str)
        results.append(result)
        if result['has_anomaly']:
            days_with_anomalies += 1
        current_date += timedelta(days=1)

    # 汇总结果
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")

    # 统计异常情况
    anomaly_dates = [r for r in results if r['has_anomaly']]

    if anomaly_dates:
        print(f"\n发现异常的日期 ({len(anomaly_dates)} 天):")
        for result in anomaly_dates:
            print(f"\n{result['date']}:")
            if 'error' in result:
                print(f"  错误: {result['error']}")
            if result.get('zscore_anomalies'):
                print(f"  Z-Score异常:")
                for anomaly in result['zscore_anomalies'][:3]:  # 只显示前3个
                    print(f"    - {anomaly}")
            if result.get('isolation_forest_anomalies'):
                print(f"  Isolation Forest异常:")
                for anomaly in result['isolation_forest_anomalies'][:3]:
                    print(f"    - {anomaly}")
            if result.get('other_anomalies'):
                print(f"  其他异常:")
                for anomaly in result['other_anomalies'][:3]:
                    print(f"    - {anomaly}")
    else:
        print("\n✓ 所有测试日期均无异常")

    print(f"\n统计:")
    print(f"  - 测试天数: {len(results)}")
    print(f"  - 有异常的天数: {days_with_anomalies}")
    print(f"  - 无异常的天数: {len(results) - days_with_anomalies}")

    # 保存详细结果
    output_file = 'output/march_anomaly_detailed_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 生成简要报告
    report_file = 'output/march_anomaly_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("3月15日到30日股票异常检测报告\n")
        f.write("="*60 + "\n\n")

        if anomaly_dates:
            f.write(f"发现异常的日期 ({len(anomaly_dates)} 天):\n\n")
            for result in anomaly_dates:
                f.write(f"{result['date']}:\n")
                if 'error' in result:
                    f.write(f"  错误: {result['error']}\n")
                if result.get('zscore_anomalies'):
                    f.write(f"  Z-Score异常: {len(result['zscore_anomalies'])} 条\n")
                if result.get('isolation_forest_anomalies'):
                    f.write(f"  Isolation Forest异常: {len(result['isolation_forest_anomalies'])} 条\n")
                if result.get('other_anomalies'):
                    f.write(f"  其他异常: {len(result['other_anomalies'])} 条\n")
                f.write("\n")
        else:
            f.write("所有测试日期均无异常\n\n")

        f.write(f"统计:\n")
        f.write(f"  - 测试天数: {len(results)}\n")
        f.write(f"  - 有异常的天数: {days_with_anomalies}\n")
        f.write(f"  - 无异常的天数: {len(results) - days_with_anomalies}\n")

    print(f"简要报告已保存到: {report_file}")

if __name__ == '__main__':
    main()
