#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块表现分析脚本 - 按股票类型分析模型准确度
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WATCHLIST as STOCK_LIST

# 股票类型映射（从 ml_trading_model.py 复制）
STOCK_TYPE_MAPPING = {
    # 银行股
    '0005.HK': {'type': 'bank', 'name': '汇丰银行'},
    '0939.HK': {'type': 'bank', 'name': '建设银行'},
    '1288.HK': {'type': 'bank', 'name': '农业银行'},
    '1398.HK': {'type': 'bank', 'name': '工商银行'},
    '3968.HK': {'type': 'bank', 'name': '招商银行'},
    
    # 公用事业股
    '0728.HK': {'type': 'utility', 'name': '中国电信'},
    '0941.HK': {'type': 'utility', 'name': '中国移动'},
    
    # 科技股
    '0700.HK': {'type': 'tech', 'name': '腾讯控股'},
    '9988.HK': {'type': 'tech', 'name': '阿里巴巴-SW'},
    '3690.HK': {'type': 'tech', 'name': '美团-W'},
    '1810.HK': {'type': 'tech', 'name': '小米集团-W'},
    
    # 半导体股
    '0981.HK': {'type': 'semiconductor', 'name': '中芯国际'},
    '1347.HK': {'type': 'semiconductor', 'name': '华虹半导体'},
    
    # 人工智能股
    '6682.HK': {'type': 'ai', 'name': '第四范式'},
    '9660.HK': {'type': 'ai', 'name': '地平线机器人'},
    '2533.HK': {'type': 'ai', 'name': '黑芝麻智能'},
    
    # 新能源股
    '1211.HK': {'type': 'new_energy', 'name': '比亚迪股份'},
    '1330.HK': {'type': 'environmental', 'name': '绿色动力环保'},
    
    # 能源/周期股
    '0883.HK': {'type': 'energy', 'name': '中国海洋石油'},
    '1088.HK': {'type': 'energy', 'name': '中国神华'},
    '1138.HK': {'type': 'shipping', 'name': '中远海能'},
    '0388.HK': {'type': 'exchange', 'name': '香港交易所'},
    
    # 保险股
    '1299.HK': {'type': 'insurance', 'name': '友邦保险'},
    
    # 生物医药股
    '2269.HK': {'type': 'biotech', 'name': '药明生物'},
    
    # 房地产股
    '0012.HK': {'type': 'real_estate', 'name': '恒基地产'},
    '0016.HK': {'type': 'real_estate', 'name': '新鸿基地产'},
    '1109.HK': {'type': 'real_estate', 'name': '华润置地'},
    
    # 指数基金
    '2800.HK': {'type': 'index', 'name': '盈富基金'},
}

# 板块名称映射
SECTOR_NAMES = {
    'bank': '银行股',
    'utility': '公用事业股',
    'tech': '科技股',
    'semiconductor': '半导体股',
    'ai': '人工智能股',
    'new_energy': '新能源股',
    'environmental': '环保股',
    'energy': '能源股',
    'shipping': '航运股',
    'exchange': '交易所',
    'insurance': '保险股',
    'biotech': '生物医药股',
    'real_estate': '房地产股',
    'index': '指数基金',
}


def load_backtest_results(trades_file):
    """加载回测交易记录"""
    print(f"加载回测交易记录: {trades_file}")
    df = pd.read_csv(trades_file)
    
    # 计算每只股票的统计指标
    stock_stats = defaultdict(lambda: {
        'trades': 0,
        'buy_signals': 0,
        'correct_predictions': 0,
        'wrong_predictions': 0,
        'total_return': 0.0,
        'returns': [],
    })
    
    for _, row in df.iterrows():
        code = row['stock_code']
        stock_stats[code]['trades'] += 1
        stock_stats[code]['buy_signals'] += 1  # 每条记录都是买入
        
        # 判断预测是否正确
        if row['prediction_correct']:
            stock_stats[code]['correct_predictions'] += 1
        else:
            stock_stats[code]['wrong_predictions'] += 1
        
        # 累计收益率
        stock_stats[code]['returns'].append(row['actual_change'])
    
    # 计算每只股票的最终统计
    for code, stats in stock_stats.items():
        stats['total_return'] = sum(stats['returns'])
        if stats['returns']:
            stats['avg_return'] = np.mean(stats['returns'])
            stats['return_std'] = np.std(stats['returns'])
            stats['sharpe_ratio'] = stats['avg_return'] / stats['return_std'] if stats['return_std'] > 0 else 0
        else:
            stats['avg_return'] = 0.0
            stats['return_std'] = 0.0
            stats['sharpe_ratio'] = 0.0
        
        stats['accuracy'] = stats['correct_predictions'] / stats['trades'] if stats['trades'] > 0 else 0
        stats['win_rate'] = stats['correct_predictions'] / stats['buy_signals'] if stats['buy_signals'] > 0 else 0
    
    return stock_stats


def analyze_sector_performance(stock_stats):
    """分析板块表现"""
    sector_stats = defaultdict(lambda: {
        'stocks': [],
        'count': 0,
        'total_trades': 0,
        'total_buy_signals': 0,
        'total_correct': 0,
        'total_wrong': 0,
        'total_return': 0.0,
        'avg_return': 0.0,
        'return_std': 0.0,
        'sharpe_ratio': 0.0,
        'accuracy': 0.0,
        'win_rate': 0.0,
    })
    
    # 将股票归类到板块
    for code, stats in stock_stats.items():
        if code in STOCK_TYPE_MAPPING:
            sector_type = STOCK_TYPE_MAPPING[code]['type']
            sector_stats[sector_type]['stocks'].append({
                'code': code,
                'name': STOCK_TYPE_MAPPING[code]['name'],
                'accuracy': stats['accuracy'],
                'avg_return': stats['avg_return'],
                'win_rate': stats['win_rate'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'trades': stats['trades'],
                'buy_signals': stats['buy_signals'],
            })
            
            # 累加统计
            sector_stats[sector_type]['count'] += 1
            sector_stats[sector_type]['total_trades'] += stats['trades']
            sector_stats[sector_type]['total_buy_signals'] += stats['buy_signals']
            sector_stats[sector_type]['total_correct'] += stats['correct_predictions']
            sector_stats[sector_type]['total_wrong'] += stats['wrong_predictions']
            sector_stats[sector_type]['total_return'] += stats['total_return']
    
    # 计算每个板块的统计指标
    for sector_type, stats in sector_stats.items():
        if stats['count'] > 0:
            stats['accuracy'] = stats['total_correct'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            stats['win_rate'] = stats['total_correct'] / stats['total_buy_signals'] if stats['total_buy_signals'] > 0 else 0
            stats['avg_return'] = stats['total_return'] / stats['count']  # 平均每只股票的总收益
            
            # 计算板块整体的平均收益率（按交易次数加权）
            all_returns = []
            for stock in stats['stocks']:
                all_returns.extend([stock['avg_return']] * stock['trades'])
            if all_returns:
                stats['sector_avg_return'] = np.mean(all_returns)
                stats['sector_return_std'] = np.std(all_returns)
                stats['sector_sharpe'] = stats['sector_avg_return'] / stats['sector_return_std'] if stats['sector_return_std'] > 0 else 0
            else:
                stats['sector_avg_return'] = 0.0
                stats['sector_return_std'] = 0.0
                stats['sector_sharpe'] = 0.0
    
    return sector_stats


def generate_csv_report(sector_stats, output_file):
    """生成CSV报告"""
    rows = []
    for sector_type, stats in sector_stats.items():
        if stats['count'] > 0:
            sector_name = SECTOR_NAMES.get(sector_type, sector_type)
            rows.append({
                '板块类型': sector_name,
                '股票数量': stats['count'],
                '总交易次数': stats['total_trades'],
                '买入信号数': stats['total_buy_signals'],
                '准确率': f"{stats['accuracy']:.2%}",
                '胜率': f"{stats['win_rate']:.2%}",
                '板块平均收益率': f"{stats['sector_avg_return']:.2%}",
                '板块收益标准差': f"{stats['sector_return_std']:.2%}",
                '板块夏普比率': f"{stats['sector_sharpe']:.2f}",
                '股票代码列表': ', '.join([s['code'] for s in stats['stocks']]),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV报告已生成: {output_file}")


def generate_json_report(sector_stats, stock_stats, output_file):
    """生成JSON报告"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sectors': {},
        'stock_details': {},
    }
    
    for sector_type, stats in sector_stats.items():
        if stats['count'] > 0:
            sector_name = SECTOR_NAMES.get(sector_type, sector_type)
            report['sectors'][sector_type] = {
                'name': sector_name,
                'stock_count': stats['count'],
                'total_trades': stats['total_trades'],
                'buy_signals': stats['total_buy_signals'],
                'accuracy': round(stats['accuracy'], 4),
                'win_rate': round(stats['win_rate'], 4),
                'sector_avg_return': round(stats['sector_avg_return'], 4),
                'sector_return_std': round(stats['sector_return_std'], 4),
                'sector_sharpe': round(stats['sector_sharpe'], 4),
                'stocks': stats['stocks'],
            }
    
    # 添加股票详细信息
    for code, stats in stock_stats.items():
        if code in STOCK_TYPE_MAPPING:
            report['stock_details'][code] = {
                'name': STOCK_TYPE_MAPPING[code]['name'],
                'type': STOCK_TYPE_MAPPING[code]['type'],
                'trades': stats['trades'],
                'buy_signals': stats['buy_signals'],
                'accuracy': round(stats['accuracy'], 4),
                'win_rate': round(stats['win_rate'], 4),
                'avg_return': round(stats['avg_return'], 4),
                'sharpe_ratio': round(stats['sharpe_ratio'], 4),
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON报告已生成: {output_file}")


def generate_markdown_report(sector_stats, stock_stats, output_file):
    """生成Markdown报告"""
    lines = []
    lines.append("# 板块表现分析报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 按准确率排序
    sorted_sectors = sorted(
        [(k, v) for k, v in sector_stats.items() if v['count'] > 0],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # 整体统计
    lines.append("\n## 整体统计")
    total_stocks = sum(s['count'] for _, s in sorted_sectors)
    total_trades = sum(s['total_trades'] for _, s in sorted_sectors)
    total_correct = sum(s['total_correct'] for _, s in sorted_sectors)
    overall_accuracy = total_correct / total_trades if total_trades > 0 else 0
    
    lines.append(f"- 总板块数: {len(sorted_sectors)}")
    lines.append(f"- 总股票数: {total_stocks}")
    lines.append(f"- 总交易次数: {total_trades}")
    lines.append(f"- 总体准确率: {overall_accuracy:.2%}")
    
    # 板块排名
    lines.append("\n## 板块排名（按准确率）")
    lines.append("\n| 排名 | 板块 | 股票数量 | 交易次数 | 准确率 | 胜率 | 平均收益率 | 夏普比率 |")
    lines.append("|------|------|----------|----------|--------|------|------------|----------|")
    
    for i, (sector_type, stats) in enumerate(sorted_sectors, 1):
        sector_name = SECTOR_NAMES.get(sector_type, sector_type)
        lines.append(f"| {i} | {sector_name} | {stats['count']} | {stats['total_trades']} | {stats['accuracy']:.2%} | {stats['win_rate']:.2%} | {stats['sector_avg_return']:.2%} | {stats['sector_sharpe']:.2f} |")
    
    # 详细板块分析
    lines.append("\n## 详细板块分析")
    
    for sector_type, stats in sorted_sectors:
        sector_name = SECTOR_NAMES.get(sector_type, sector_type)
        lines.append(f"\n### {sector_name} ({sector_type})")
        lines.append(f"- 股票数量: {stats['count']}")
        lines.append(f"- 总交易次数: {stats['total_trades']}")
        lines.append(f"- 买入信号数: {stats['total_buy_signals']}")
        lines.append(f"- 准确率: {stats['accuracy']:.2%}")
        lines.append(f"- 胜率: {stats['win_rate']:.2%}")
        lines.append(f"- 板块平均收益率: {stats['sector_avg_return']:.2%}")
        lines.append(f"- 板块收益标准差: {stats['sector_return_std']:.2%}")
        lines.append(f"- 板块夏普比率: {stats['sector_sharpe']:.2f}")
        
        lines.append(f"\n#### 股票详情")
        lines.append("| 股票代码 | 股票名称 | 交易次数 | 准确率 | 胜率 | 平均收益率 | 夏普比率 |")
        lines.append("|----------|----------|----------|--------|------|------------|----------|")
        
        # 按准确率排序股票
        sorted_stocks = sorted(stats['stocks'], key=lambda x: x['accuracy'], reverse=True)
        for stock in sorted_stocks:
            lines.append(f"| {stock['code']} | {stock['name']} | {stock['trades']} | {stock['accuracy']:.2%} | {stock['win_rate']:.2%} | {stock['avg_return']:.2%} | {stock['sharpe_ratio']:.2f} |")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ Markdown报告已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='板块表现分析脚本')
    parser.add_argument('--trades-file', type=str, required=True, help='回测交易记录文件路径')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--output-format', type=str, default='all', choices=['csv', 'json', 'markdown', 'all'], help='输出格式')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("板块表现分析脚本")
    print("=" * 80)
    print(f"交易记录文件: {args.trades_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出格式: {args.output_format}")
    print("=" * 80)
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载回测结果
    stock_stats = load_backtest_results(args.trades_file)
    
    # 分析板块表现
    sector_stats = analyze_sector_performance(stock_stats)
    
    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output_format in ['csv', 'all']:
        csv_file = os.path.join(args.output_dir, f'sector_performance_analysis_{timestamp}.csv')
        generate_csv_report(sector_stats, csv_file)
    
    if args.output_format in ['json', 'all']:
        json_file = os.path.join(args.output_dir, f'sector_performance_analysis_{timestamp}.json')
        generate_json_report(sector_stats, stock_stats, json_file)
    
    if args.output_format in ['markdown', 'all']:
        md_file = os.path.join(args.output_dir, f'sector_performance_analysis_{timestamp}.md')
        generate_markdown_report(sector_stats, stock_stats, md_file)
    
    print()
    print("=" * 80)
    print("✅ 板块表现分析完成！")
    print("=" * 80)
    print()
    print("关键发现:")
    
    # 找出表现最好的板块
    sorted_sectors = sorted(
        [(k, v) for k, v in sector_stats.items() if v['count'] > 0],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    if sorted_sectors:
        best_sector_type, best_stats = sorted_sectors[0]
        best_sector_name = SECTOR_NAMES.get(best_sector_type, best_sector_type)
        print(f"  🏆 表现最佳板块: {best_sector_name} (准确率: {best_stats['accuracy']:.2%})")
    
    # 找出表现最差的板块
    if len(sorted_sectors) > 1:
        worst_sector_type, worst_stats = sorted_sectors[-1]
        worst_sector_name = SECTOR_NAMES.get(worst_sector_type, worst_sector_type)
        print(f"  ⚠️  表现最差板块: {worst_sector_name} (准确率: {worst_stats['accuracy']:.2%})")


if __name__ == '__main__':
    main()