#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票表现排名分析脚本 - 按不同指标TOP 10排名
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WATCHLIST as STOCK_LIST

# 股票名称映射
STOCK_NAMES = STOCK_LIST


def load_and_analyze(trades_file):
    """加载回测数据并计算统计数据"""
    print(f"加载回测交易记录: {trades_file}")
    trades_df = pd.read_csv(trades_file)
    
    # 计算每只股票的统计数据
    stock_stats = trades_df.groupby('stock_code').agg({
        'actual_change': ['count', 'mean', 'median', 'std'],
        'prediction_correct': 'mean',
        'probability': 'mean'
    }).round(4)
    stock_stats.columns = ['交易次数', '平均收益率', '收益率中位数', '收益率标准差', '准确率', '平均预测概率']
    stock_stats['胜率'] = trades_df.groupby('stock_code').apply(lambda x: (x['actual_change'] > 0).mean()).round(4)
    
    return stock_stats


def analyze_rankings(stock_stats):
    """分析不同指标的排名"""
    results = {
        'by_return': {},
        'by_win_rate': {},
        'by_accuracy': {},
        'excellent_stocks': [],
        'statistics': {}
    }
    
    # 1. 按平均收益率排名
    by_return = stock_stats.sort_values('平均收益率', ascending=False)
    results['by_return'] = {
        'top10': by_return.head(10),
        'full': by_return
    }
    
    # 2. 按胜率排名
    by_win_rate = stock_stats.sort_values('胜率', ascending=False)
    results['by_win_rate'] = {
        'top10': by_win_rate.head(10),
        'full': by_win_rate
    }
    
    # 3. 按准确率排名
    by_accuracy = stock_stats.sort_values('准确率', ascending=False)
    results['by_accuracy'] = {
        'top10': by_accuracy.head(10),
        'full': by_accuracy
    }
    
    # 4. 综合优秀股票（三项指标均在前15名）
    top15_return = set(by_return.index[:15])
    top15_win_rate = set(by_win_rate.index[:15])
    top15_accuracy = set(by_accuracy.index[:15])
    excellent_stocks = top15_return & top15_win_rate & top15_accuracy
    
    for stock_code in excellent_stocks:
        return_rank = by_return.index.get_loc(stock_code) + 1
        win_rate_rank = by_win_rate.index.get_loc(stock_code) + 1
        accuracy_rank = by_accuracy.index.get_loc(stock_code) + 1
        results['excellent_stocks'].append({
            'code': stock_code,
            'name': STOCK_NAMES.get(stock_code, stock_code),
            'return_rank': return_rank,
            'win_rate_rank': win_rate_rank,
            'accuracy_rank': accuracy_rank,
            'avg_return': stock_stats.loc[stock_code, '平均收益率'],
            'win_rate': stock_stats.loc[stock_code, '胜率'],
            'accuracy': stock_stats.loc[stock_code, '准确率']
        })
    
    # 5. 统计总结
    results['statistics'] = {
        'return': {
            'max': stock_stats['平均收益率'].max(),
            'max_stock': STOCK_NAMES.get(stock_stats['平均收益率'].idxmax(), stock_stats['平均收益率'].idxmax()),
            'min': stock_stats['平均收益率'].min(),
            'min_stock': STOCK_NAMES.get(stock_stats['平均收益率'].idxmin(), stock_stats['平均收益率'].idxmin()),
            'mean': stock_stats['平均收益率'].mean(),
            'median': stock_stats['平均收益率'].median()
        },
        'win_rate': {
            'max': stock_stats['胜率'].max(),
            'max_stock': STOCK_NAMES.get(stock_stats['胜率'].idxmax(), stock_stats['胜率'].idxmax()),
            'min': stock_stats['胜率'].min(),
            'min_stock': STOCK_NAMES.get(stock_stats['胜率'].idxmin(), stock_stats['胜率'].idxmin()),
            'mean': stock_stats['胜率'].mean(),
            'median': stock_stats['胜率'].median()
        },
        'accuracy': {
            'max': stock_stats['准确率'].max(),
            'max_stock': STOCK_NAMES.get(stock_stats['准确率'].idxmax(), stock_stats['准确率'].idxmax()),
            'min': stock_stats['准确率'].min(),
            'min_stock': STOCK_NAMES.get(stock_stats['准确率'].idxmin(), stock_stats['准确率'].idxmin()),
            'mean': stock_stats['准确率'].mean(),
            'median': stock_stats['准确率'].median()
        }
    }
    
    return results


def generate_csv_report(results, output_file):
    """生成CSV报告"""
    rows = []
    
    # 1. 按平均收益率排名
    for i, (stock_code, row) in enumerate(results['by_return']['top10'].iterrows(), 1):
        rows.append({
            '排名类别': '平均收益率',
            '排名': i,
            '股票代码': stock_code,
            '股票名称': STOCK_NAMES.get(stock_code, stock_code),
            '交易次数': int(row['交易次数']),
            '平均收益率': f"{row['平均收益率']*100:.2f}%",
            '胜率': f"{row['胜率']*100:.2f}%",
            '准确率': f"{row['准确率']*100:.2f}%"
        })
    
    # 2. 按胜率排名
    for i, (stock_code, row) in enumerate(results['by_win_rate']['top10'].iterrows(), 1):
        rows.append({
            '排名类别': '胜率',
            '排名': i,
            '股票代码': stock_code,
            '股票名称': STOCK_NAMES.get(stock_code, stock_code),
            '交易次数': int(row['交易次数']),
            '平均收益率': f"{row['平均收益率']*100:.2f}%",
            '胜率': f"{row['胜率']*100:.2f}%",
            '准确率': f"{row['准确率']*100:.2f}%"
        })
    
    # 3. 按准确率排名
    for i, (stock_code, row) in enumerate(results['by_accuracy']['top10'].iterrows(), 1):
        rows.append({
            '排名类别': '准确率',
            '排名': i,
            '股票代码': stock_code,
            '股票名称': STOCK_NAMES.get(stock_code, stock_code),
            '交易次数': int(row['交易次数']),
            '平均收益率': f"{row['平均收益率']*100:.2f}%",
            '胜率': f"{row['胜率']*100:.2f}%",
            '准确率': f"{row['准确率']*100:.2f}%"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV报告已生成: {output_file}")


def generate_json_report(results, output_file):
    """生成JSON报告"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'top10_by_return': [],
        'top10_by_win_rate': [],
        'top10_by_accuracy': [],
        'excellent_stocks': results['excellent_stocks'],
        'statistics': results['statistics']
    }
    
    # 转换TOP10数据
    for stock_code, row in results['by_return']['top10'].iterrows():
        report['top10_by_return'].append({
            'code': stock_code,
            'name': STOCK_NAMES.get(stock_code, stock_code),
            'trades': int(row['交易次数']),
            'avg_return': round(row['平均收益率'], 4),
            'win_rate': round(row['胜率'], 4),
            'accuracy': round(row['准确率'], 4)
        })
    
    for stock_code, row in results['by_win_rate']['top10'].iterrows():
        report['top10_by_win_rate'].append({
            'code': stock_code,
            'name': STOCK_NAMES.get(stock_code, stock_code),
            'trades': int(row['交易次数']),
            'avg_return': round(row['平均收益率'], 4),
            'win_rate': round(row['胜率'], 4),
            'accuracy': round(row['准确率'], 4)
        })
    
    for stock_code, row in results['by_accuracy']['top10'].iterrows():
        report['top10_by_accuracy'].append({
            'code': stock_code,
            'name': STOCK_NAMES.get(stock_code, stock_code),
            'trades': int(row['交易次数']),
            'avg_return': round(row['平均收益率'], 4),
            'win_rate': round(row['胜率'], 4),
            'accuracy': round(row['准确率'], 4)
        })
    
    # 转换统计数据
    report['statistics'] = {
        'return': {
            'max': round(results['statistics']['return']['max'], 4),
            'max_stock': results['statistics']['return']['max_stock'],
            'min': round(results['statistics']['return']['min'], 4),
            'min_stock': results['statistics']['return']['min_stock'],
            'mean': round(results['statistics']['return']['mean'], 4),
            'median': round(results['statistics']['return']['median'], 4)
        },
        'win_rate': {
            'max': round(results['statistics']['win_rate']['max'], 4),
            'max_stock': results['statistics']['win_rate']['max_stock'],
            'min': round(results['statistics']['win_rate']['min'], 4),
            'min_stock': results['statistics']['win_rate']['min_stock'],
            'mean': round(results['statistics']['win_rate']['mean'], 4),
            'median': round(results['statistics']['win_rate']['median'], 4)
        },
        'accuracy': {
            'max': round(results['statistics']['accuracy']['max'], 4),
            'max_stock': results['statistics']['accuracy']['max_stock'],
            'min': round(results['statistics']['accuracy']['min'], 4),
            'min_stock': results['statistics']['accuracy']['min_stock'],
            'mean': round(results['statistics']['accuracy']['mean'], 4),
            'median': round(results['statistics']['accuracy']['median'], 4)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON报告已生成: {output_file}")


def generate_markdown_report(results, output_file):
    """生成Markdown报告"""
    lines = []
    lines.append("# 股票表现TOP 10排名分析报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 按平均收益率排名
    lines.append("\n## 一、按平均收益率排名 TOP 10")
    lines.append("\n| 排名 | 股票代码 | 股票名称 | 交易次数 | 平均收益率 | 胜率 | 准确率 |")
    lines.append("|------|----------|----------|----------|------------|------|--------|")
    
    for i, (stock_code, row) in enumerate(results['by_return']['top10'].iterrows(), 1):
        stock_name = STOCK_NAMES.get(stock_code, stock_code)
        lines.append(f"| {i} | {stock_code} | {stock_name} | {int(row['交易次数'])} | {row['平均收益率']*100:.2f}% | {row['胜率']*100:.2f}% | {row['准确率']*100:.2f}% |")
    
    # 2. 按胜率排名
    lines.append("\n## 二、按胜率排名 TOP 10")
    lines.append("\n| 排名 | 股票代码 | 股票名称 | 交易次数 | 平均收益率 | 胜率 | 准确率 |")
    lines.append("|------|----------|----------|----------|------------|------|--------|")
    
    for i, (stock_code, row) in enumerate(results['by_win_rate']['top10'].iterrows(), 1):
        stock_name = STOCK_NAMES.get(stock_code, stock_code)
        lines.append(f"| {i} | {stock_code} | {stock_name} | {int(row['交易次数'])} | {row['平均收益率']*100:.2f}% | {row['胜率']*100:.2f}% | {row['准确率']*100:.2f}% |")
    
    # 3. 按准确率排名
    lines.append("\n## 三、按准确率排名 TOP 10")
    lines.append("\n| 排名 | 股票代码 | 股票名称 | 交易次数 | 平均收益率 | 胜率 | 准确率 |")
    lines.append("|------|----------|----------|----------|------------|------|--------|")
    
    for i, (stock_code, row) in enumerate(results['by_accuracy']['top10'].iterrows(), 1):
        stock_name = STOCK_NAMES.get(stock_code, stock_code)
        lines.append(f"| {i} | {stock_code} | {stock_name} | {int(row['交易次数'])} | {row['平均收益率']*100:.2f}% | {row['胜率']*100:.2f}% | {row['准确率']*100:.2f}% |")
    
    # 4. 综合优秀股票
    lines.append("\n## 四、综合优秀股票（三项指标均在前15名）")
    
    if results['excellent_stocks']:
        lines.append("\n| 股票代码 | 股票名称 | 平均收益率排名 | 胜率排名 | 准确率排名 | 平均收益率 | 胜率 | 准确率 |")
        lines.append("|----------|----------|--------------|----------|------------|------------|------|--------|")
        
        for stock in results['excellent_stocks']:
            lines.append(f"| {stock['code']} | {stock['name']} | {stock['return_rank']} | {stock['win_rate_rank']} | {stock['accuracy_rank']} | {stock['avg_return']*100:.2f}% | {stock['win_rate']*100:.2f}% | {stock['accuracy']*100:.2f}% |")
    else:
        lines.append("\n没有股票在三项指标中均进入前15名")
    
    # 5. 排名对比分析
    lines.append("\n## 五、排名对比分析")
    
    top3_return = ', '.join([f"{STOCK_NAMES.get(code, code)}" for code in results['by_return']['top10'].index[:3]])
    top3_accuracy = ', '.join([f"{STOCK_NAMES.get(code, code)}" for code in results['by_accuracy']['top10'].index[:3]])
    lines.append(f"\n### 平均收益率TOP 3 vs 准确率TOP 3")
    lines.append(f"- 平均收益率TOP 3: {top3_return}")
    lines.append(f"- 准确率TOP 3: {top3_accuracy}")
    
    top3_win_rate = ', '.join([f"{STOCK_NAMES.get(code, code)}" for code in results['by_win_rate']['top10'].index[:3]])
    lines.append(f"\n### 胜率TOP 3 vs 准确率TOP 3")
    lines.append(f"- 胜率TOP 3: {top3_win_rate}")
    lines.append(f"- 准确率TOP 3: {top3_accuracy}")
    
    # 6. 统计总结
    lines.append("\n## 六、统计总结")
    
    stats = results['statistics']
    lines.append("\n### 平均收益率统计")
    lines.append(f"- 最高: {stats['return']['max']*100:.2f}% ({stats['return']['max_stock']})")
    lines.append(f"- 最低: {stats['return']['min']*100:.2f}% ({stats['return']['min_stock']})")
    lines.append(f"- 平均: {stats['return']['mean']*100:.2f}%")
    lines.append(f"- 中位数: {stats['return']['median']*100:.2f}%")
    
    lines.append("\n### 胜率统计")
    lines.append(f"- 最高: {stats['win_rate']['max']*100:.2f}% ({stats['win_rate']['max_stock']})")
    lines.append(f"- 最低: {stats['win_rate']['min']*100:.2f}% ({stats['win_rate']['min_stock']})")
    lines.append(f"- 平均: {stats['win_rate']['mean']*100:.2f}%")
    lines.append(f"- 中位数: {stats['win_rate']['median']*100:.2f}%")
    
    lines.append("\n### 准确率统计")
    lines.append(f"- 最高: {stats['accuracy']['max']*100:.2f}% ({stats['accuracy']['max_stock']})")
    lines.append(f"- 最低: {stats['accuracy']['min']*100:.2f}% ({stats['accuracy']['min_stock']})")
    lines.append(f"- 平均: {stats['accuracy']['mean']*100:.2f}%")
    lines.append(f"- 中位数: {stats['accuracy']['median']*100:.2f}%")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ Markdown报告已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='股票表现排名分析脚本')
    parser.add_argument('--trades-file', type=str, required=True, help='回测交易记录文件路径')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--output-format', type=str, default='all', choices=['csv', 'json', 'markdown', 'all'], help='输出格式')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("股票表现TOP 10排名分析脚本")
    print("=" * 100)
    print(f"交易记录文件: {args.trades_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出格式: {args.output_format}")
    print("=" * 100)
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载和分析数据
    stock_stats = load_and_analyze(args.trades_file)
    results = analyze_rankings(stock_stats)
    
    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output_format in ['csv', 'all']:
        csv_file = os.path.join(args.output_dir, f'ranking_analysis_{timestamp}.csv')
        generate_csv_report(results, csv_file)
    
    if args.output_format in ['json', 'all']:
        json_file = os.path.join(args.output_dir, f'ranking_analysis_{timestamp}.json')
        generate_json_report(results, json_file)
    
    if args.output_format in ['markdown', 'all']:
        md_file = os.path.join(args.output_dir, f'ranking_analysis_{timestamp}.md')
        generate_markdown_report(results, md_file)
    
    print()
    print("=" * 100)
    print("✅ 排名分析完成！")
    print("=" * 100)
    print()
    print("关键发现:")
    
    # 最佳收益股票
    best_return_stock = results['by_return']['top10'].index[0]
    best_return_name = STOCK_NAMES.get(best_return_stock, best_return_stock)
    best_return = results['by_return']['top10'].loc[best_return_stock, '平均收益率']
    print(f"  💰 平均收益率最高: {best_return_name} ({best_return*100:.2f}%)")
    
    # 最高胜率股票
    best_win_rate_stock = results['by_win_rate']['top10'].index[0]
    best_win_rate_name = STOCK_NAMES.get(best_win_rate_stock, best_win_rate_stock)
    best_win_rate = results['by_win_rate']['top10'].loc[best_win_rate_stock, '胜率']
    print(f"  🎯 胜率最高: {best_win_rate_name} ({best_win_rate*100:.2f}%)")
    
    # 最高准确率股票
    best_accuracy_stock = results['by_accuracy']['top10'].index[0]
    best_accuracy_name = STOCK_NAMES.get(best_accuracy_stock, best_accuracy_stock)
    best_accuracy = results['by_accuracy']['top10'].loc[best_accuracy_stock, '准确率']
    print(f"  🎯 准确率最高: {best_accuracy_name} ({best_accuracy*100:.2f}%)")
    
    # 综合优秀股票
    if results['excellent_stocks']:
        print(f"  🏆 综合优秀股票数量: {len(results['excellent_stocks'])}")
        for stock in results['excellent_stocks']:
            print(f"     - {stock['name']} (收益率排名:{stock['return_rank']}, 胜率排名:{stock['win_rate_rank']}, 准确率排名:{stock['accuracy_rank']})")


if __name__ == '__main__':
    main()