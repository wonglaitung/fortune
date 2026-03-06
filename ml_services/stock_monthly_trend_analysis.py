#!/usr/bin/env python3
"""
分析单个股票与月度趋势的差异
计算相关性系数和波动性
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# 文件路径
stock_monthly_file = os.path.join(project_dir, 'output', 'backtest_20d_stock_monthly_analysis_20260306_091400.csv')
monthly_file = os.path.join(project_dir, 'output', 'backtest_20d_monthly_analysis_20260306_091400.csv')

def load_data():
    """加载数据"""
    print("📊 加载数据...")
    
    # 加载股票月度数据
    stock_monthly_df = pd.read_csv(stock_monthly_file)
    stock_monthly_df['月份'] = pd.to_datetime(stock_monthly_df['月份']).dt.to_period('M')
    
    # 加载月度数据
    monthly_df = pd.read_csv(monthly_file)
    monthly_df['月份'] = pd.to_datetime(monthly_df['月份']).dt.to_period('M')
    
    print(f"✅ 已加载 {len(stock_monthly_df)} 条股票月度记录")
    print(f"✅ 已加载 {len(monthly_df)} 个月度记录")
    
    return stock_monthly_df, monthly_df

def calculate_correlation(stock_monthly_df, monthly_df):
    """计算单个股票与总体趋势的相关性"""
    print("\n🔍 计算相关性...")
    
    # 创建股票与总体趋势的对比表
    correlation_results = []
    
    # 获取所有股票
    stocks = stock_monthly_df['股票代码'].unique()
    
    for stock_code in stocks:
        stock_data = stock_monthly_df[stock_monthly_df['股票代码'] == stock_code].copy()
        stock_data = stock_data.sort_values('月份')
        
        # 合并总体月度数据
        merged = pd.merge(stock_data, monthly_df[['月份', '平均收益率']], on='月份', suffixes=('', '_总体'))
        
        # 计算相关性
        if len(merged) >= 3:  # 至少需要3个月的数据
            correlation = merged['平均收益率'].corr(merged['平均收益率_总体'])
            
            # 计算股票收益率标准差
            stock_std = merged['平均收益率'].std()
            
            # 计算总体收益率标准差
            overall_std = merged['平均收益率_总体'].std()
            
            # 计算股票收益率变异系数
            stock_cv = stock_std / merged['平均收益率'].mean() if merged['平均收益率'].mean() != 0 else 0
            
            correlation_results.append({
                '股票代码': stock_code,
                '股票名称': stock_data['股票名称'].iloc[0],
                '相关系数': correlation,
                '股票收益率标准差': stock_std,
                '总体收益率标准差': overall_std,
                '股票变异系数': stock_cv,
                '数据点数': len(merged)
            })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('相关系数', ascending=False)
    
    return correlation_df

def analyze_trend_patterns(stock_monthly_df, monthly_df, correlation_df):
    """分析趋势模式"""
    print("\n📈 分析趋势模式...")
    
    # 分析高相关性股票（相关系数 > 0.7）
    high_correlation = correlation_df[correlation_df['相关系数'] > 0.7]
    print(f"\n🔹 高相关性股票（相关系数 > 0.7）：{len(high_correlation)} 只")
    if len(high_correlation) > 0:
        print(high_correlation[['股票代码', '股票名称', '相关系数']].to_string(index=False))
    
    # 分析低相关性股票（相关系数 < 0.3）
    low_correlation = correlation_df[correlation_df['相关系数'] < 0.3]
    print(f"\n🔹 低相关性股票（相关系数 < 0.3）：{len(low_correlation)} 只")
    if len(low_correlation) > 0:
        print(low_correlation[['股票代码', '股票名称', '相关系数']].to_string(index=False))
    
    # 分析负相关性股票（相关系数 < 0）
    negative_correlation = correlation_df[correlation_df['相关系数'] < 0]
    print(f"\n🔹 负相关性股票（相关系数 < 0）：{len(negative_correlation)} 只")
    if len(negative_correlation) > 0:
        print(negative_correlation[['股票代码', '股票名称', '相关系数']].to_string(index=False))
    
    # 计算平均相关系数
    avg_correlation = correlation_df['相关系数'].mean()
    print(f"\n🔸 平均相关系数：{avg_correlation:.4f}")
    
    # 计算相关系数标准差
    correlation_std = correlation_df['相关系数'].std()
    print(f"🔸 相关系数标准差：{correlation_std:.4f}")
    
    return {
        'high_correlation': high_correlation,
        'low_correlation': low_correlation,
        'negative_correlation': negative_correlation,
        'avg_correlation': avg_correlation,
        'correlation_std': correlation_std
    }

def analyze_volatility(stock_monthly_df):
    """分析波动性"""
    print("\n📊 分析波动性...")
    
    volatility_results = []
    
    stocks = stock_monthly_df['股票代码'].unique()
    
    for stock_code in stocks:
        stock_data = stock_monthly_df[stock_monthly_df['股票代码'] == stock_code].copy()
        stock_data = stock_data.sort_values('月份')
        
        # 计算收益率标准差
        if len(stock_data) >= 3:
            return_std = stock_data['平均收益率'].std()
            return_mean = stock_data['平均收益率'].mean()
            
            # 计算变异系数
            cv = return_std / return_mean if return_mean != 0 else 0
            
            # 计算最大回撤
            cumulative_returns = (1 + stock_data['平均收益率'] / 100).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
            
            volatility_results.append({
                '股票代码': stock_code,
                '股票名称': stock_data['股票名称'].iloc[0],
                '收益率标准差': return_std,
                '收益率均值': return_mean,
                '变异系数': cv,
                '最大回撤': max_drawdown * 100
            })
    
    volatility_df = pd.DataFrame(volatility_results)
    volatility_df = volatility_df.sort_values('收益率标准差', ascending=False)
    
    return volatility_df

def analyze_monthly_outliers(stock_monthly_df, monthly_df):
    """分析月度异常值"""
    print("\n🔍 分析月度异常值...")
    
    outliers = []
    
    # 按月份分组，找出偏离总体趋势的股票
    for month in monthly_df['月份'].unique():
        month_data = monthly_df[monthly_df['月份'] == month].iloc[0]
        overall_return = month_data['平均收益率']
        
        # 获取该月所有股票的数据
        month_stocks = stock_monthly_df[stock_monthly_df['月份'] == month]
        
        if len(month_stocks) > 0:
            # 计算该月股票收益率的标准差
            month_std = month_stocks['平均收益率'].std()
            month_mean = month_stocks['平均收益率'].mean()
            
            # 找出偏离度超过2个标准差的股票
            threshold = 2 * month_std
            outlier_stocks = month_stocks[
                (month_stocks['平均收益率'] - month_mean).abs() > threshold
            ]
            
            for _, row in outlier_stocks.iterrows():
                outliers.append({
                    '月份': str(month),
                    '股票代码': row['股票代码'],
                    '股票名称': row['股票名称'],
                    '收益率': row['平均收益率'],
                    '总体收益率': overall_return,
                    '偏离度': row['平均收益率'] - overall_return
                })
    
    outliers_df = pd.DataFrame(outliers)
    outliers_df = outliers_df.sort_values('偏离度', key=abs, ascending=False)
    
    return outliers_df

def generate_report(correlation_df, volatility_df, outliers_df, trend_patterns):
    """生成分析报告"""
    print("\n📝 生成分析报告...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(project_dir, 'output', f'backtest_20d_stock_monthly_trend_analysis_{timestamp}.txt')
    csv_file = os.path.join(project_dir, 'output', f'backtest_20d_stock_monthly_trend_analysis_{timestamp}.csv')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("单个股票与月度趋势差异分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析股票数量: {len(correlation_df)}\n\n")
        
        # 总体相关性分析
        f.write("=" * 80 + "\n")
        f.write("一、总体相关性分析\n")
        f.write("=" * 80 + "\n")
        f.write(f"平均相关系数: {trend_patterns['avg_correlation']:.4f}\n")
        f.write(f"相关系数标准差: {trend_patterns['correlation_std']:.4f}\n")
        f.write(f"高相关性股票数量（>0.7）: {len(trend_patterns['high_correlation'])}\n")
        f.write(f"低相关性股票数量（<0.3）: {len(trend_patterns['low_correlation'])}\n")
        f.write(f"负相关性股票数量（<0）: {len(trend_patterns['negative_correlation'])}\n\n")
        
        # 高相关性股票
        f.write("=" * 80 + "\n")
        f.write("二、高相关性股票（相关系数 > 0.7）\n")
        f.write("=" * 80 + "\n")
        if len(trend_patterns['high_correlation']) > 0:
            f.write(trend_patterns['high_correlation'][['股票代码', '股票名称', '相关系数']].to_string(index=False))
        else:
            f.write("无高相关性股票\n")
        f.write("\n\n")
        
        # 低相关性股票
        f.write("=" * 80 + "\n")
        f.write("三、低相关性股票（相关系数 < 0.3）\n")
        f.write("=" * 80 + "\n")
        if len(trend_patterns['low_correlation']) > 0:
            f.write(trend_patterns['low_correlation'][['股票代码', '股票名称', '相关系数']].to_string(index=False))
        else:
            f.write("无低相关性股票\n")
        f.write("\n\n")
        
        # 负相关性股票
        f.write("=" * 80 + "\n")
        f.write("四、负相关性股票（相关系数 < 0）\n")
        f.write("=" * 80 + "\n")
        if len(trend_patterns['negative_correlation']) > 0:
            f.write(trend_patterns['negative_correlation'][['股票代码', '股票名称', '相关系数']].to_string(index=False))
        else:
            f.write("无负相关性股票\n")
        f.write("\n\n")
        
        # 波动性分析
        f.write("=" * 80 + "\n")
        f.write("五、波动性分析TOP 10\n")
        f.write("=" * 80 + "\n")
        f.write(volatility_df.head(10)[['股票代码', '股票名称', '收益率标准差', '变异系数', '最大回撤']].to_string(index=False))
        f.write("\n\n")
        
        # 稳定性分析
        f.write("=" * 80 + "\n")
        f.write("六、稳定性分析TOP 10（收益率标准差最低）\n")
        f.write("=" * 80 + "\n")
        f.write(volatility_df.tail(10)[['股票代码', '股票名称', '收益率标准差', '变异系数', '最大回撤']].to_string(index=False))
        f.write("\n\n")
        
        # 月度异常值分析
        f.write("=" * 80 + "\n")
        f.write("七、月度异常值分析TOP 20\n")
        f.write("=" * 80 + "\n")
        f.write(outliers_df.head(20).to_string(index=False))
        f.write("\n\n")
        
        # 关键发现
        f.write("=" * 80 + "\n")
        f.write("八、关键发现\n")
        f.write("=" * 80 + "\n")
        
        if trend_patterns['avg_correlation'] < 0.3:
            f.write("❌ 单个股票与月度趋势相关性较低，股票表现存在较大差异\n")
        elif trend_patterns['avg_correlation'] < 0.6:
            f.write("⚠️ 单个股票与月度趋势相关性中等，部分股票表现存在差异\n")
        else:
            f.write("✅ 单个股票与月度趋势相关性较高，股票表现较为一致\n")
        
        f.write(f"\n相关系数标准差：{trend_patterns['correlation_std']:.4f}\n")
        if trend_patterns['correlation_std'] > 0.5:
            f.write("⚠️ 相关系数波动较大，股票与总体趋势的差异较大\n")
        else:
            f.write("✅ 相关系数波动较小，股票与总体趋势的差异较小\n")
        
        f.write(f"\n月度异常值数量：{len(outliers_df)}\n")
        if len(outliers_df) > 20:
            f.write("⚠️ 存在大量月度异常值，部分股票在特定月份表现极端\n")
        elif len(outliers_df) > 10:
            f.write("⚠️ 存在一定数量的月度异常值，部分股票在特定月份表现偏离\n")
        else:
            f.write("✅ 月度异常值较少，股票表现相对稳定\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    # 保存CSV文件
    correlation_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ 分析报告已保存到: {report_file}")
    print(f"✅ 相关性数据已保存到: {csv_file}")
    
    return report_file, csv_file

def main():
    """主函数"""
    print("=" * 80)
    print("单个股票与月度趋势差异分析")
    print("=" * 80)
    
    # 加载数据
    stock_monthly_df, monthly_df = load_data()
    
    # 计算相关性
    correlation_df = calculate_correlation(stock_monthly_df, monthly_df)
    
    # 分析趋势模式
    trend_patterns = analyze_trend_patterns(stock_monthly_df, monthly_df, correlation_df)
    
    # 分析波动性
    volatility_df = analyze_volatility(stock_monthly_df)
    
    # 分析月度异常值
    outliers_df = analyze_monthly_outliers(stock_monthly_df, monthly_df)
    
    # 生成报告
    report_file, csv_file = generate_report(correlation_df, volatility_df, outliers_df, trend_patterns)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  1. 分析报告: {report_file}")
    print(f"  2. 相关性数据: {csv_file}")
    
    # 打印关键结果
    print("\n📊 关键结果:")
    print(f"  平均相关系数: {trend_patterns['avg_correlation']:.4f}")
    print(f"  相关系数标准差: {trend_patterns['correlation_std']:.4f}")
    print(f"  高相关性股票数量: {len(trend_patterns['high_correlation'])}")
    print(f"  低相关性股票数量: {len(trend_patterns['low_correlation'])}")
    print(f"  负相关性股票数量: {len(trend_patterns['negative_correlation'])}")
    print(f"  月度异常值数量: {len(outliers_df)}")

if __name__ == '__main__':
    main()