#!/usr/bin/env python3
"""
CatBoost 20天模型回测 - 月度分析脚本
分析平均收益率、胜率、准确率与月份的关系
分析单个股票与总体趋势的差异
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# 文件路径
trades_file = os.path.join(project_dir, 'output', 'backtest_20d_trades_20260306_091148.csv')
stock_summary_file = os.path.join(project_dir, 'output', 'backtest_20d_stock_summary_20260306_091148.csv')
metrics_file = os.path.join(project_dir, 'output', 'backtest_20d_metrics_20260306_091148.json')

# 股票名称映射
stock_names = {
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

def load_data():
    """加载回测数据"""
    print("📊 加载回测数据...")
    
    # 加载交易记录
    trades_df = pd.read_csv(trades_file)
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
    trades_df['month'] = trades_df['buy_date'].dt.to_period('M')
    
    # 加载股票汇总
    stock_summary_df = pd.read_csv(stock_summary_file)
    
    # 加载性能指标
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"✅ 已加载 {len(trades_df)} 条交易记录")
    print(f"✅ 已加载 {len(stock_summary_df)} 只股票的汇总数据")
    
    return trades_df, stock_summary_df, metrics

def monthly_analysis(trades_df):
    """按月份分析"""
    print("\n📈 按月份分析...")
    
    # 按月份分组统计
    monthly_stats = trades_df.groupby('month').agg({
        'stock_code': 'count',  # 交易次数
        'actual_change': 'mean',  # 平均收益率
        'prediction_correct': lambda x: (x == True).sum() / len(x) * 100  # 准确率
    }).rename(columns={
        'stock_code': '交易次数',
        'actual_change': '平均收益率',
        'prediction_correct': '准确率'
    })
    
    # 计算胜率（盈利交易占比）
    monthly_winrate = trades_df[trades_df['actual_direction'] == 1].groupby('month').size() / \
                      trades_df.groupby('month').size() * 100
    monthly_stats['胜率'] = monthly_winrate
    
    # 添加月份名称
    monthly_stats['月份'] = monthly_stats.index.strftime('%Y-%m')
    
    # 重置索引
    monthly_stats = monthly_stats.reset_index(drop=True)
    
    # 重新排列列顺序
    monthly_stats = monthly_stats[['月份', '交易次数', '平均收益率', '胜率', '准确率']]
    
    # 转换为百分比格式
    monthly_stats['平均收益率'] = monthly_stats['平均收益率'] * 100
    monthly_stats['平均收益率'] = monthly_stats['平均收益率'].round(2)
    monthly_stats['胜率'] = monthly_stats['胜率'].round(2)
    monthly_stats['准确率'] = monthly_stats['准确率'].round(2)
    
    return monthly_stats

def stock_monthly_analysis(trades_df):
    """按股票和月份分析"""
    print("\n📊 按股票和月份分析...")
    
    # 按股票和月份分组统计
    stock_monthly_stats = trades_df.groupby(['stock_code', 'month']).agg({
        'actual_change': 'mean',  # 平均收益率
        'prediction_correct': lambda x: (x == True).sum() / len(x) * 100  # 准确率
    }).rename(columns={
        'actual_change': '平均收益率',
        'prediction_correct': '准确率'
    })
    
    # 计算胜率（盈利交易占比）
    stock_monthly_winrate = trades_df[trades_df['actual_direction'] == 1].groupby(['stock_code', 'month']).size() / \
                            trades_df.groupby(['stock_code', 'month']).size() * 100
    stock_monthly_stats['胜率'] = stock_monthly_winrate
    
    # 添加月份名称
    stock_monthly_stats['月份'] = stock_monthly_stats.index.get_level_values(1).strftime('%Y-%m')
    
    # 重置索引
    stock_monthly_stats = stock_monthly_stats.reset_index()
    
    # 添加股票名称
    stock_monthly_stats['股票名称'] = stock_monthly_stats['stock_code'].map(stock_names)
    
    # 重新排列列顺序（先重命名列）
    stock_monthly_stats = stock_monthly_stats.rename(columns={'stock_code': '股票代码'})
    stock_monthly_stats = stock_monthly_stats[['股票代码', '股票名称', '月份', '平均收益率', '胜率', '准确率']]
    
    # 转换为百分比格式
    stock_monthly_stats['平均收益率'] = stock_monthly_stats['平均收益率'] * 100
    stock_monthly_stats['平均收益率'] = stock_monthly_stats['平均收益率'].round(2)
    stock_monthly_stats['胜率'] = stock_monthly_stats['胜率'].round(2)
    stock_monthly_stats['准确率'] = stock_monthly_stats['准确率'].round(2)
    
    return stock_monthly_stats

def analyze_trend_differences(monthly_stats, stock_monthly_stats, stock_summary_df):
    """分析单个股票与总体趋势的差异"""
    print("\n🔍 分析单个股票与总体趋势的差异...")
    
    # 计算总体平均
    overall_avg_return = stock_summary_df['平均收益率'].mean() * 100
    overall_winrate = stock_summary_df['胜率'].mean() * 100
    overall_accuracy = stock_summary_df['准确率'].mean() * 100
    
    print(f"\n总体平均:")
    print(f"  平均收益率: {overall_avg_return:.2f}%")
    print(f"  胜率: {overall_winrate:.2f}%")
    print(f"  准确率: {overall_accuracy:.2f}%")
    
    # 计算每月平均
    monthly_avg_return = monthly_stats['平均收益率'].mean()
    monthly_avg_winrate = monthly_stats['胜率'].mean()
    monthly_avg_accuracy = monthly_stats['准确率'].mean()
    
    print(f"\n每月平均:")
    print(f"  平均收益率: {monthly_avg_return:.2f}%")
    print(f"  胜率: {monthly_avg_winrate:.2f}%")
    print(f"  准确率: {monthly_avg_accuracy:.2f}%")
    
    # 分析各股票与总体趋势的差异
    stock_summary_df['收益率差异'] = (stock_summary_df['平均收益率'] * 100 - overall_avg_return).round(2)
    stock_summary_df['胜率差异'] = (stock_summary_df['胜率'] * 100 - overall_winrate).round(2)
    stock_summary_df['准确率差异'] = (stock_summary_df['准确率'] * 100 - overall_accuracy).round(2)
    
    # 找出表现最好的股票
    top_stocks_return = stock_summary_df.nlargest(5, '平均收益率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    top_stocks_winrate = stock_summary_df.nlargest(5, '胜率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    top_stocks_accuracy = stock_summary_df.nlargest(5, '准确率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    
    # 找出表现最差的股票
    bottom_stocks_return = stock_summary_df.nsmallest(5, '平均收益率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    bottom_stocks_winrate = stock_summary_df.nsmallest(5, '胜率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    bottom_stocks_accuracy = stock_summary_df.nsmallest(5, '准确率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
    
    return {
        'overall_avg': {
            'return': overall_avg_return,
            'winrate': overall_winrate,
            'accuracy': overall_accuracy
        },
        'monthly_avg': {
            'return': monthly_avg_return,
            'winrate': monthly_avg_winrate,
            'accuracy': monthly_avg_accuracy
        },
        'top_stocks': {
            'return': top_stocks_return,
            'winrate': top_stocks_winrate,
            'accuracy': top_stocks_accuracy
        },
        'bottom_stocks': {
            'return': bottom_stocks_return,
            'winrate': bottom_stocks_winrate,
            'accuracy': bottom_stocks_accuracy
        }
    }

def generate_report(monthly_stats, stock_monthly_stats, trend_analysis, metrics):
    """生成分析报告"""
    print("\n📝 生成分析报告...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 生成CSV文件
    monthly_csv = os.path.join(project_dir, 'output', f'backtest_20d_monthly_analysis_{timestamp}.csv')
    stock_monthly_csv = os.path.join(project_dir, 'output', f'backtest_20d_stock_monthly_analysis_{timestamp}.csv')
    report_txt = os.path.join(project_dir, 'output', f'backtest_20d_monthly_analysis_report_{timestamp}.txt')
    
    # 保存月度分析CSV
    monthly_stats.to_csv(monthly_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 月度分析已保存到: {monthly_csv}")
    
    # 保存股票月度分析CSV
    stock_monthly_csv_data = stock_monthly_stats
    stock_monthly_csv_data.to_csv(stock_monthly_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 股票月度分析已保存到: {stock_monthly_csv}")
    
    # 生成文本报告
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CatBoost 20天模型回测 - 月度分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"回测日期范围: 2025-01-01 至 2025-12-31\n\n")
        
        # 总体性能
        f.write("=" * 80 + "\n")
        f.write("一、总体性能\n")
        f.write("=" * 80 + "\n")
        f.write(f"总交易机会: {metrics.get('total_opportunities', 'N/A')}\n")
        f.write(f"买入信号数: {metrics.get('buy_signals', 'N/A')}\n")
        f.write(f"准确率: {metrics.get('accuracy', 'N/A')}\n")
        f.write(f"平均收益率: {metrics.get('avg_return', 'N/A')}\n")
        f.write(f"胜率: {metrics.get('win_rate', 'N/A')}\n")
        f.write(f"夏普比率（年化）: {metrics.get('sharpe_ratio', 'N/A')}\n")
        f.write(f"最大回撤: {metrics.get('max_drawdown', 'N/A')}\n\n")
        
        # 总体平均
        f.write("=" * 80 + "\n")
        f.write("二、总体平均\n")
        f.write("=" * 80 + "\n")
        f.write(f"平均收益率: {trend_analysis['overall_avg']['return']:.2f}%\n")
        f.write(f"胜率: {trend_analysis['overall_avg']['winrate']:.2f}%\n")
        f.write(f"准确率: {trend_analysis['overall_avg']['accuracy']:.2f}%\n\n")
        
        # 每月平均
        f.write("=" * 80 + "\n")
        f.write("三、每月平均\n")
        f.write("=" * 80 + "\n")
        f.write(f"平均收益率: {trend_analysis['monthly_avg']['return']:.2f}%\n")
        f.write(f"胜率: {trend_analysis['monthly_avg']['winrate']:.2f}%\n")
        f.write(f"准确率: {trend_analysis['monthly_avg']['accuracy']:.2f}%\n\n")
        
        # 月度分析
        f.write("=" * 80 + "\n")
        f.write("四、月度分析\n")
        f.write("=" * 80 + "\n")
        f.write(monthly_stats.to_string(index=False))
        f.write("\n\n")
        
        # 按收益率排名TOP 5
        f.write("=" * 80 + "\n")
        f.write("五、按收益率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['top_stocks']['return'].to_string(index=False))
        f.write("\n\n")
        
        # 按胜率排名TOP 5
        f.write("=" * 80 + "\n")
        f.write("六、按胜率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['top_stocks']['winrate'].to_string(index=False))
        f.write("\n\n")
        
        # 按准确率排名TOP 5
        f.write("=" * 80 + "\n")
        f.write("七、按准确率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['top_stocks']['accuracy'].to_string(index=False))
        f.write("\n\n")
        
        # 按收益率排名BOTTOM 5
        f.write("=" * 80 + "\n")
        f.write("八、按收益率排名BOTTOM 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['bottom_stocks']['return'].to_string(index=False))
        f.write("\n\n")
        
        # 按胜率排名BOTTOM 5
        f.write("=" * 80 + "\n")
        f.write("九、按胜率排名BOTTOM 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['bottom_stocks']['winrate'].to_string(index=False))
        f.write("\n\n")
        
        # 按准确率排名BOTTOM 5
        f.write("=" * 80 + "\n")
        f.write("十、按准确率排名BOTTOM 5\n")
        f.write("=" * 80 + "\n")
        f.write(trend_analysis['bottom_stocks']['accuracy'].to_string(index=False))
        f.write("\n\n")
        
        # 趋势分析
        f.write("=" * 80 + "\n")
        f.write("十一、趋势分析\n")
        f.write("=" * 80 + "\n")
        
        # 分析收益率趋势
        return_trend = monthly_stats['平均收益率'].values
        if len(return_trend) > 1:
            first_half = np.mean(return_trend[:len(return_trend)//2])
            second_half = np.mean(return_trend[len(return_trend)//2:])
            f.write(f"收益率趋势:\n")
            f.write(f"  上半年平均收益率: {first_half:.2f}%\n")
            f.write(f"  下半年平均收益率: {second_half:.2f}%\n")
            f.write(f"  趋势: {'上升' if second_half > first_half else '下降'}\n\n")
        
        # 分析胜率趋势
        winrate_trend = monthly_stats['胜率'].values
        if len(winrate_trend) > 1:
            first_half = np.mean(winrate_trend[:len(winrate_trend)//2])
            second_half = np.mean(winrate_trend[len(winrate_trend)//2:])
            f.write(f"胜率趋势:\n")
            f.write(f"  上半年平均胜率: {first_half:.2f}%\n")
            f.write(f"  下半年平均胜率: {second_half:.2f}%\n")
            f.write(f"  趋势: {'上升' if second_half > first_half else '下降'}\n\n")
        
        # 分析准确率趋势
        accuracy_trend = monthly_stats['准确率'].values
        if len(accuracy_trend) > 1:
            first_half = np.mean(accuracy_trend[:len(accuracy_trend)//2])
            second_half = np.mean(accuracy_trend[len(accuracy_trend)//2:])
            f.write(f"准确率趋势:\n")
            f.write(f"  上半年平均准确率: {first_half:.2f}%\n")
            f.write(f"  下半年平均准确率: {second_half:.2f}%\n")
            f.write(f"  趋势: {'上升' if second_half > first_half else '下降'}\n\n")
        
        # 找出表现最好的月份
        best_return_month = monthly_stats.loc[monthly_stats['平均收益率'].idxmax()]
        f.write(f"表现最好的月份（收益率）: {best_return_month['月份']}, 收益率: {best_return_month['平均收益率']:.2f}%\n")
        
        best_winrate_month = monthly_stats.loc[monthly_stats['胜率'].idxmax()]
        f.write(f"表现最好的月份（胜率）: {best_winrate_month['月份']}, 胜率: {best_winrate_month['胜率']:.2f}%\n")
        
        best_accuracy_month = monthly_stats.loc[monthly_stats['准确率'].idxmax()]
        f.write(f"表现最好的月份（准确率）: {best_accuracy_month['月份']}, 准确率: {best_accuracy_month['准确率']:.2f}%\n\n")
        
        # 找出表现最差的月份
        worst_return_month = monthly_stats.loc[monthly_stats['平均收益率'].idxmin()]
        f.write(f"表现最差的月份（收益率）: {worst_return_month['月份']}, 收益率: {worst_return_month['平均收益率']:.2f}%\n")
        
        worst_winrate_month = monthly_stats.loc[monthly_stats['胜率'].idxmin()]
        f.write(f"表现最差的月份（胜率）: {worst_winrate_month['月份']}, 胜率: {worst_winrate_month['胜率']:.2f}%\n")
        
        worst_accuracy_month = monthly_stats.loc[monthly_stats['准确率'].idxmin()]
        f.write(f"表现最差的月份（准确率）: {worst_accuracy_month['月份']}, 准确率: {worst_accuracy_month['准确率']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ 分析报告已保存到: {report_txt}")
    
    return monthly_csv, stock_monthly_csv, report_txt

def main():
    print("=" * 80)
    print("CatBoost 20天模型回测 - 月度分析")
    print("=" * 80)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CatBoost 20天模型回测 - 月度分析')
    parser.add_argument('--trades-file', type=str, default='output/backtest_20d_trades_20260306_091148.csv',
                       help='交易记录文件路径')
    parser.add_argument('--stock-summary-file', type=str, default='output/backtest_20d_stock_summary_20260306_091148.csv',
                       help='股票汇总文件路径')
    args = parser.parse_args()
    
    # 加载回测数据
    print("\n📊 加载回测数据...")
    trades_df = pd.read_csv(args.trades_file)
    print(f"✅ 已加载 {len(trades_df)} 条交易记录")
    
    stock_summary_df = pd.read_csv(args.stock_summary_file)
    print(f"✅ 已加载 {len(stock_summary_df)} 只股票的汇总数据")
    
    # 转换日期格式并添加月份列
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['month'] = trades_df['buy_date'].dt.to_period('M')
    
    # 按月份分析
    monthly_stats = monthly_analysis(trades_df)
    
    # 按股票和月份分析
    stock_monthly_stats = stock_monthly_analysis(trades_df)
    
    # 分析单个股票与总体趋势的差异
    analyze_trend_differences(stock_monthly_stats, monthly_stats, stock_summary_df)
    
    # 生成分析报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    monthly_csv = f'output/backtest_20d_monthly_analysis_{timestamp}.csv'
    stock_monthly_csv = f'output/backtest_20d_stock_monthly_analysis_{timestamp}.csv'
    report_txt = f'output/backtest_20d_monthly_analysis_report_{timestamp}.txt'
    
    # 保存月度分析结果
    monthly_stats.to_csv(monthly_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 月度分析已保存到: {monthly_csv}")
    
    # 保存股票月度分析结果
    stock_monthly_stats.to_csv(stock_monthly_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 股票月度分析已保存到: {stock_monthly_csv}")
    
    # 生成文本报告
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CatBoost 20天模型回测 - 月度分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"回测日期范围: {trades_df['buy_date'].min()} 至 {trades_df['buy_date'].max()}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("一、总体性能\n")
        f.write("=" * 80 + "\n")
        f.write(f"总交易机会: {len(trades_df)}\n")
        f.write(f"买入信号数: {len(trades_df[trades_df['prediction'] == 1])}\n")
        f.write(f"准确率: {(trades_df['prediction_correct'].sum() / len(trades_df)):.10f}\n")
        f.write(f"平均收益率: {trades_df['actual_change'].mean():.20f}\n")
        f.write(f"胜率: {(len(trades_df[trades_df['actual_direction'] == 1]) / len(trades_df[trades_df['prediction'] == 1])):.20f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("二、月度分析\n")
        f.write("=" * 80 + "\n")
        f.write(monthly_stats.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("三、按收益率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        top5_return = stock_summary_df.nlargest(5, '平均收益率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
        f.write(top5_return.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("四、按胜率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        top5_winrate = stock_summary_df.nlargest(5, '胜率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
        f.write(top5_winrate.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("五、按准确率排名TOP 5\n")
        f.write("=" * 80 + "\n")
        top5_accuracy = stock_summary_df.nlargest(5, '准确率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
        f.write(top5_accuracy.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("六、按收益率排名BOTTOM 5\n")
        f.write("=" * 80 + "\n")
        bottom5_return = stock_summary_df.nsmallest(5, '平均收益率')[['股票代码', '股票名称', '平均收益率', '胜率', '准确率']]
        f.write(bottom5_return.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("七、趋势分析\n")
        f.write("=" * 80 + "\n")
        # 上半年和下半年对比
        monthly_stats['年月'] = pd.to_datetime(monthly_stats['月份'])
        monthly_stats['年'] = monthly_stats['年月'].dt.year
        monthly_stats['月'] = monthly_stats['年月'].dt.month
        
        h1_stats = monthly_stats[monthly_stats['月'] <= 6]
        h2_stats = monthly_stats[monthly_stats['月'] > 6]
        
        h1_avg_return = h1_stats['平均收益率'].mean()
        h2_avg_return = h2_stats['平均收益率'].mean()
        
        f.write(f"收益率趋势:\n")
        f.write(f"  上半年平均收益率: {h1_avg_return:.2f}%\n")
        f.write(f"  下半年平均收益率: {h2_avg_return:.2f}%\n")
        if h1_avg_return > h2_avg_return:
            f.write(f"  趋势: 下降\n")
        else:
            f.write(f"  趋势: 上升\n\n")
        
        best_return_month = monthly_stats.loc[monthly_stats['平均收益率'].idxmax()]
        f.write(f"表现最好的月份（收益率）: {best_return_month['月份']}, 收益率: {best_return_month['平均收益率']:.2f}%\n")
        
        worst_return_month = monthly_stats.loc[monthly_stats['平均收益率'].idxmin()]
        f.write(f"表现最差的月份（收益率）: {worst_return_month['月份']}, 收益率: {worst_return_month['平均收益率']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ 分析报告已保存到: {report_txt}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  1. 月度分析CSV: {monthly_csv}")
    print(f"  2. 股票月度分析CSV: {stock_monthly_csv}")
    print(f"  3. 分析报告: {report_txt}")
    
    print("\n📊 月度分析结果:")
    print(monthly_stats.to_string(index=False))

if __name__ == '__main__':
    main()