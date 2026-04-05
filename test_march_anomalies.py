#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试3月15日到30日的股票异常检测
"""

import subprocess
import json
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_date(date_str):
    """测试指定日期的异常"""
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

        # 查找异常信息
        anomalies = []
        lines = output.split('\n')
        for line in lines:
            if '异常检测完成' in line:
                print(line)
            elif '异常' in line and '检测' not in line and 'No anomalies' not in line:
                if '股票' in line or 'Stock' in line or '价格' in line or '成交量' in line:
                    anomalies.append(line)

        if anomalies:
            print(f"\n发现异常 ({len(anomalies)}):")
            for anomaly in anomalies:
                print(f"  - {anomaly}")
            return date_str, anomalies
        else:
            print("✓ 无异常")
            return date_str, []

    except subprocess.TimeoutExpired:
        print(f"⚠️ 超时")
        return date_str, ["检测超时"]
    except Exception as e:
        print(f"❌ 错误: {e}")
        return date_str, [f"错误: {str(e)}"]

def main():
    """主函数"""
    print("开始批量测试3月15日到30日的股票异常检测...")
    print("="*60)

    # 生成日期列表（2026-03-15 到 2026-03-30）
    start_date = datetime(2026, 3, 15)
    end_date = datetime(2026, 3, 30)

    results = {}
    total_anomalies = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        date_str, anomalies = test_date(date_str)
        results[date_str] = anomalies
        total_anomalies += len(anomalies)
        current_date += timedelta(days=1)

    # 汇总结果
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")

    days_with_anomalies = {k: v for k, v in results.items() if v}

    if days_with_anomalies:
        print(f"\n发现异常的日期 ({len(days_with_anomalies)} 天):")
        for date_str, anomalies in days_with_anomalies.items():
            print(f"\n{date_str}:")
            for anomaly in anomalies:
                print(f"  - {anomaly}")
    else:
        print("\n✓ 所有测试日期均无异常")

    print(f"\n统计:")
    print(f"  - 测试天数: {len(results)}")
    print(f"  - 有异常的天数: {len(days_with_anomalies)}")
    print(f"  - 总异常数量: {total_anomalies}")

    # 保存结果
    output_file = 'output/march_anomaly_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
