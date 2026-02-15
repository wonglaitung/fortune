#!/usr/bin/env python3
"""
板块轮动河流图生成器（简化版）
生成展示过去半年内各个板块排名变化的河流图

使用方法：
    python generate_sector_rotation_river_plot.py
"""

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_services.tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent


def get_historical_sector_performance(days=365):
    """
    获取历史板块表现数据
    
    Args:
        days: 获取天数，默认180天（半年）
    
    Returns:
        DataFrame: 包含日期、板块、涨跌幅、排名的数据
    """
    print(f"Getting historical sector performance data for past {days} days...")
    
    # 板块定义
    sectors = {
        'bank': ['0005.HK', '0939.HK', '1288.HK', '1398.HK', '3968.HK'],
        'tech': ['0700.HK', '1810.HK', '3690.HK', '9988.HK'],
        'semiconductor': ['0981.HK', '1347.HK'],
        'ai': ['2533.HK', '6682.HK', '9660.HK'],
        'new_energy': ['1211.HK'],
        'environmental': ['1330.HK'],
        'energy': ['0883.HK', '1088.HK'],
        'shipping': ['1138.HK'],
        'exchange': ['0388.HK'],
        'utility': ['0941.HK', '0728.HK'],
        'insurance': ['1299.HK'],
        'biotech': ['2269.HK'],
        'index': ['2800.HK']
    }
    
    sector_names = {
        'bank': 'Banking',
        'tech': 'Technology',
        'semiconductor': 'Semiconductor',
        'ai': 'AI',
        'new_energy': 'New Energy',
        'environmental': 'Environmental',
        'energy': 'Energy',
        'shipping': 'Shipping',
        'exchange': 'Exchange',
        'utility': 'Utility',
        'insurance': 'Insurance',
        'biotech': 'Biotech',
        'index': 'Index Fund'
    }
    
    # 先获取所有股票的历史数据（避免重复请求）
    print("Step 1/3: Getting historical data for all stocks...")
    stock_data_cache = {}
    all_stocks = [stock for stocks in sectors.values() for stock in stocks]
    
    for i, stock in enumerate(all_stocks):
        print(f"  Getting stock {i+1}/{len(all_stocks)}: {stock}")
        try:
            stock_code_simple = stock.replace('.HK', '')
            df = get_hk_stock_data_tencent(stock_code_simple, period_days=days)
            if df is not None and not df.empty:
                # DataFrame的索引已经是'Date'了
                stock_data_cache[stock] = df
        except Exception as e:
            print(f"    Warning: Cannot get data for {stock} - {str(e)}")
            continue
    
    print(f"Successfully got {len(stock_data_cache)}/{len(all_stocks)} stocks data")
    
    # 获取实际可用的日期范围
    if stock_data_cache:
        # 找到所有股票数据的日期范围
        all_dates = []
        for stock, df in stock_data_cache.items():
            all_dates.extend(df.index.tolist())
        
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            print(f"Data date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            
            # 使用实际可用的日期范围
            start_date = min_date
            end_date = max_date
        else:
            print("Error: No available date data")
            return None
    else:
        print("Error: No stock data retrieved")
        return None
    
    # 按双周采样数据（避免数据点过多）
    dates = pd.date_range(start=start_date, end=end_date, freq='2W-Fri')
    # 数据已经有UTC时区，直接使用
    dates = dates.tz_convert('UTC')
    
    results = []
    
    print(f"\nStep 2/3: Calculating sector performance for {len(dates)} time points...")
    for i, date in enumerate(dates):
        print(f"  Processing date {i+1}/{len(dates)}: {date.strftime('%Y-%m-%d')}")
        
        sector_changes = []
        
        for sector_code, stocks in sectors.items():
            try:
                total_change = 0
                valid_stocks = 0
                
                for stock in stocks:
                    if stock not in stock_data_cache:
                        continue
                    
                    df = stock_data_cache[stock]
                    
                    # 找到最接近目标日期的数据
                    date_diff = abs(df.index - date)
                    if date_diff.empty:
                        continue
                        
                    closest_idx = np.argmin(date_diff)
                    closest_date_idx = df.index[closest_idx]
                    
                    # 调试信息
                    if i == 0 and sector_code == 'tech' and stock == '0700.HK':
                        print(f"    调试: {stock} - 目标日期: {date}, 最近日期: {closest_date_idx}, 差异: {date_diff[closest_idx].days}天")
                    
                    if date_diff[closest_idx].days <= 30:  # 放宽到30天（一个月）
                        close_price = df.loc[closest_date_idx, 'Close']
                        
                        # 找到一周前的价格
                        week_ago_idx = max(0, df.index.get_loc(closest_date_idx) - 5)
                        week_ago_price = df.iloc[week_ago_idx]['Close']
                        
                        if week_ago_price > 0:
                            change_pct = ((close_price - week_ago_price) / week_ago_price) * 100
                            total_change += change_pct
                            valid_stocks += 1
                
                if valid_stocks > 0:
                    avg_change = total_change / valid_stocks
                    sector_changes.append({
                        'date': date,
                        'sector_code': sector_code,
                        'sector_name': sector_names[sector_code],
                        'change_pct': avg_change
                    })
            except Exception as e:
                continue
        
        # 计算排名
        if sector_changes:
            df_temp = pd.DataFrame(sector_changes)
            df_temp['rank'] = df_temp['change_pct'].rank(ascending=False).astype(int)
            results.extend(sector_changes)
    
    if results:
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        # 重新计算排名（确保所有时间点的排名是相对的）
        df['rank'] = df.groupby('date')['change_pct'].rank(ascending=False).astype(int)
        print(f"\nStep 3/3: Complete! Generated {len(results)} data records")
        return df
    else:
        print("Warning: No sector data retrieved")
        return None


def create_river_plot(df, output_path='output/sector_rotation_river_plot.png'):
    """
    创建板块轮动河流图（含恒生指数对比）
    
    Args:
        df: 包含历史板块数据的DataFrame
        output_path: 输出文件路径
    """
    if df is None or df.empty:
        print("Error: No available data")
        return
    
    print("Generating river plot...")
    
    # 准备数据：按日期和板块汇总
    df_pivot = df.pivot(index='date', columns='sector_name', values='rank')
    
    # 按日期排序
    df_pivot = df_pivot.sort_index()
    
    # 获取恒生指数数据
    print("Getting HSI data for comparison...")
    try:
        hsi_df = get_hsi_data_tencent(period_days=365)
        if hsi_df is not None and not hsi_df.empty:
            # 重采样到与板块数据相同的频率
            hsi_df = hsi_df.resample('2W-Fri').last()
            # 只保留与板块数据相同的日期
            hsi_df = hsi_df[hsi_df.index.isin(df_pivot.index)]
            print(f"Successfully retrieved HSI data for {len(hsi_df)} periods")
        else:
            print("Warning: Cannot retrieve HSI data")
            hsi_df = None
    except Exception as e:
        print(f"Warning: Cannot retrieve HSI data - {str(e)}")
        hsi_df = None
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 颜色映射（每个板块一个颜色）
    sectors = df_pivot.columns.tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))
    color_map = {sector: colors[i] for i, sector in enumerate(sectors)}
    
    # 为每个板块绘制线条
    for sector in sectors:
        rank_data = df_pivot[sector].dropna()
        if len(rank_data) > 1:
            # 绘制排名线
            ax.plot(rank_data.index, rank_data.values, 
                   color=color_map[sector], 
                   linewidth=2.5, 
                   alpha=0.8,
                   label=sector)
            
            # 标记最新排名
            last_date = rank_data.index[-1]
            last_rank = rank_data.values[-1]
            ax.scatter(last_date, last_rank, 
                      color=color_map[sector], 
                      s=150, 
                      zorder=5,
                      edgecolors='white',
                      linewidth=2)
            
            # 添加排名标注
            ax.annotate(f'{int(last_rank)}', 
                      xy=(last_date, last_rank),
                      xytext=(5, 0),
                      textcoords='offset points',
                      fontsize=9,
                      fontweight='bold',
                      va='center')
    
    # 如果有HSI数据，绘制HSI曲线
    if hsi_df is not None and not hsi_df.empty:
        # 标准化HSI数据到0-13范围（与排名对应）
        hsi_values = hsi_df['Close'].values
        if len(hsi_values) > 0:
            hsi_min, hsi_max = min(hsi_values), max(hsi_values)
            if hsi_max != hsi_min:
                normalized_hsi = 1 + (hsi_values - hsi_min) / (hsi_max - hsi_min) * 12
            else:
                normalized_hsi = np.full_like(hsi_values, 7)  # 中间值
        else:
            normalized_hsi = np.array([])
        
        ax2 = ax.twinx()
        ax2.plot(hsi_df.index, normalized_hsi, 
                color='black', 
                linewidth=2, 
                alpha=0.8, 
                label='HSI',
                linestyle='--')
        ax2.scatter(hsi_df.index[-1], normalized_hsi[-1], 
                   color='black', 
                   s=150, 
                   zorder=5,
                   edgecolors='white',
                   linewidth=2)
        ax2.annotate(f'{hsi_df["Close"].iloc[-1]:.0f}', 
                    xy=(hsi_df.index[-1], normalized_hsi[-1]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    va='center',
                    color='black')
        ax2.set_ylabel('HSI Index Level', fontsize=12, fontweight='bold', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 14)
    
    # 设置Y轴反转（排名1在顶部）
    ax.invert_yaxis()
    ax.set_ylim(df_pivot.max().max() + 1, 0.5)
    
    # 设置标题和标签
    ax.set_title('HK Sector Rotation Trend (Past 12 Months)\nRiver Plot with HSI Comparison', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sector Ranking (1=Best Performer)', fontsize=12, fontweight='bold')
    
    # 设置Y轴刻度
    ax.set_yticks(range(1, len(sectors) + 1))
    ax.set_yticklabels(range(1, len(sectors) + 1))
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.grid(True, axis='x', alpha=0.1, linestyle='--')
    
    # 添加图例
    ax.legend(loc='upper right', 
              bbox_to_anchor=(1.15, 1),
              ncol=1,
              fontsize=10,
              framealpha=0.9,
              fancybox=True,
              shadow=True)
    
    # 设置X轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"River plot saved to: {output_path}")
    
    # 显示图表
    plt.show()


def create_heatmap(df, output_path='output/sector_rotation_heatmap.png'):
    """
    创建板块轮动热力图（含恒生指数对比）
    
    Args:
        df: 包含历史板块数据的DataFrame
        output_path: 输出文件路径
    """
    if df is None or df.empty:
        print("Error: No available data")
        return
    
    print("Generating heatmap...")
    
    # 准备数据
    df_pivot = df.pivot(index='date', columns='sector_name', values='rank')
    df_pivot = df_pivot.sort_index()
    
    # 按月采样（避免数据点过多）
    df_pivot_monthly = df_pivot.resample('M').last()
    
    # 获取恒生指数数据
    print("Getting HSI data for comparison...")
    try:
        hsi_df = get_hsi_data_tencent(period_days=365)
        if hsi_df is not None and not hsi_df.empty:
            # 重采样到与板块数据相同的频率
            hsi_df = hsi_df.resample('M').last()
            # 只保留与板块数据相同的日期
            hsi_df = hsi_df[hsi_df.index.isin(df_pivot_monthly.index)]
            print(f"Successfully retrieved HSI data for {len(hsi_df)} periods")
        else:
            print("Warning: Cannot retrieve HSI data")
            hsi_df = None
    except Exception as e:
        print(f"Warning: Cannot retrieve HSI data - {str(e)}")
        hsi_df = None
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 转置数据以便板块在Y轴
    data = df_pivot_monthly.T
    
    # 使用发散 colormap（排名1用绿色，排名13用红色）
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=13)
    
    # 设置刻度
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels([d.strftime('%Y-%m') for d in data.columns], rotation=45, ha='right')
    ax.set_yticklabels(data.index)
    
    # 添加数值标注
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            value = data.iloc[i, j]
            if not pd.isna(value):
                text_color = 'white' if value < 7 else 'black'
                ax.text(j, i, f'{int(value)}', 
                       ha='center', va='center',
                       color=text_color,
                       fontsize=8,
                       fontweight='bold')
    
    # 如果有HSI数据，添加HSI数据到热力图
    if hsi_df is not None:
        hsi_df_monthly = hsi_df.resample('M').last()
        hsi_data = hsi_df_monthly['Close'].values
        
        # 创建一个单独的子图显示HSI数据
        ax2 = ax.twinx()
        if len(hsi_data) > 0:
            ax2.plot(range(len(hsi_data)), hsi_data, color='white', linewidth=2, label='HSI')
            ax2.scatter(range(len(hsi_data)), hsi_data, color='white', s=50, zorder=5)
            ax2.set_ylabel('HSI Level', fontsize=12, fontweight='bold', color='white')
            ax2.tick_params(axis='y', labelcolor='white')
            ax2.set_ylim(min(hsi_data) * 0.99, max(hsi_data) * 1.01)
        else:
            ax2.plot([], color='white', linewidth=2, label='HSI')
            ax2.set_ylabel('HSI Level', fontsize=12, fontweight='bold', color='white')
            ax2.tick_params(axis='y', labelcolor='white')
        
        # 添加颜色条
        if ax2.collections:
            cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar2.set_label('HSI Level', rotation=270, labelpad=20, fontsize=10)
    
    # 设置标题和标签
    ax.set_title('HK Sector Rotation Heatmap (Past 12 Months)\nMonthly Sector Rankings with HSI Comparison', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sector', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ranking (1=Best Performer, 13=Lowest Performer)', 
                    rotation=270, labelpad=20, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    
    # 显示图表
    plt.show()


def analyze_sector_rotation(df):
    """
    分析板块轮动特征
    
    Args:
        df: 包含历史板块数据的DataFrame
    """
    if df is None or df.empty:
        print("Error: No available data")
        return
    
    print("\n=== Sector Rotation Analysis ===\n")
    
    # 计算每个板块的平均排名
    avg_rank = df.groupby('sector_name')['rank'].mean().sort_values()
    print("Average Rankings (Past 12 Months):")
    for sector, rank in avg_rank.items():
        print(f"  {sector:12s}: {rank:.2f}")
    
    # 计算排名波动性
    rank_std = df.groupby('sector_name')['rank'].std().sort_values(ascending=False)
    print("\nRanking Volatility (Standard Deviation):")
    for sector, std in rank_std.items():
        print(f"  {sector:12s}: {std:.2f}")
    
    # 识别近期强势板块（最近4周）
    recent_df = df[df['date'] >= df['date'].max() - timedelta(weeks=4)]
    recent_avg_rank = recent_df.groupby('sector_name')['rank'].mean().sort_values()
    print("\nRecent Strong Sectors (Last 4 Weeks):")
    for sector, rank in recent_avg_rank.items():
        print(f"  {sector:12s}: {rank:.2f}")
    
    # 识别近期弱势板块
    print("\nRecent Weak Sectors (Last 4 Weeks):")
    for sector, rank in list(recent_avg_rank.items())[::-1]:
        print(f"  {sector:12s}: {rank:.2f}")
    
    # 分析与恒生指数的贴合度
    print("\n=== HSI Correlation Analysis ===")
    print("Analyzing correlation between sector rankings and HSI...")
    
    # 获取HSI数据
    try:
        hsi_df = get_hsi_data_tencent(period_days=365)
        print(f"HSI data retrieved: {hsi_df is not None}, shape: {hsi_df.shape if hsi_df is not None else 'None'}")
        if hsi_df is not None and not hsi_df.empty:
            # 重采样到与板块数据相同的频率
            hsi_df = hsi_df.resample('2W-Fri').last()
            print(f"After resample: HSI data shape: {hsi_df.shape}")
            print(f"HSI index sample: {hsi_df.index.tolist()[:3] if len(hsi_df.index) > 0 else 'Empty'}")
            print(f"DF index sample: {df['date'].tolist()[:3] if len(df['date']) > 0 else 'Empty'}")
            
            # 只保留与板块数据相同的日期
            common_dates = set(hsi_df.index).intersection(set(df['date']))
            hsi_df = hsi_df[hsi_df.index.isin(common_dates)]
            print(f"After filtering: HSI data shape: {hsi_df.shape}")
            
            # 计算每个板块与HSI的相关性
            correlations = {}
            print(f"HSI data shape: {hsi_df.shape}")
            print(f"HSI data columns: {hsi_df.columns.tolist()}")
            print(f"HSI data index sample: {hsi_df.index.tolist()[:5] if len(hsi_df.index) > 0 else 'Empty'}")
            print(f"DF date range: {df['date'].min()} to {df['date'].max()}")
            
            # 简化相关性分析
            print("HSI data shape:", hsi_df.shape)
            print("HSI data columns:", hsi_df.columns.tolist())
            
            # 直接计算每个板块与HSI的相关性
            for sector in df['sector_name'].unique():
                sector_data = df[df['sector_name'] == sector]
                
                # 找到与HSI数据日期匹配的板块数据
                matched_data = []
                for _, row in sector_data.iterrows():
                    date = row['date']
                    # 找到最接近的HSI日期
                    hsi_date = min(hsi_df.index, key=lambda x: abs((x - date).days))
                    if abs((hsi_date - date).days) <= 7:  # 7天内算匹配
                        matched_data.append({
                            'rank': row['rank'],
                            'hsi_close': hsi_df.loc[hsi_date, 'Close']
                        })
                
                if len(matched_data) > 1:
                    matched_df = pd.DataFrame(matched_data)
                    correlation = matched_df['hsi_close'].corr(matched_df['rank'])
                    correlations[sector] = correlation
                    print(f"  {sector}: {len(matched_data)} matching points, correlation: {correlation:.3f}")
                else:
                    print(f"  {sector}: No matching data")
            
            # 按相关性排序
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\nCorrelation with HSI (Absolute Value - Higher = Better Correlation):")
            for sector, corr in sorted_correlations:
                trend = "Positive" if corr > 0 else "Negative"
                print(f"  {sector:12s}: {corr:7.3f} ({trend})")
            
            # 找出与HSI最贴合的板块
            if correlations:
                best_correlation = max(correlations.items(), key=lambda x: abs(x[1]))
                print(f"\nBest Correlation with HSI: {best_correlation[0]} (r={best_correlation[1]:.3f})")
                
                # 显示与HSI同向和反向的板块
                positive_correlations = [(s, c) for s, c in correlations.items() if c > 0]
                negative_correlations = [(s, c) for s, c in correlations.items() if c < 0]
                
                print(f"\nPositive Correlation ({len(positive_correlations)} sectors):")
                for sector, corr in sorted(positive_correlations, key=lambda x: x[1], reverse=True):
                    print(f"  {sector:12s}: {corr:7.3f}")
                
                print(f"\nNegative Correlation ({len(negative_correlations)} sectors):")
                for sector, corr in sorted(negative_correlations, key=lambda x: x[1]):
                    print(f"  {sector:12s}: {corr:7.3f}")
            else:
                print("\nWarning: No correlation data available")
        else:
            print("Warning: Cannot retrieve HSI data for correlation analysis")
    except Exception as e:
        print(f"Warning: Cannot perform correlation analysis - {str(e)}")


def main():
    """Main function"""
    print("=" * 60)
    print("HK Sector Rotation River Plot Generator")
    print("=" * 60)
    
    # 获取历史数据（过去一年）
    df = get_historical_sector_performance(days=365)
    
    if df is not None and not df.empty:
        # 分析板块轮动特征
        analyze_sector_rotation(df)
        
        # 生成河流图
        print("\nGenerating river plot...")
        create_river_plot(df, 'output/sector_rotation_river_plot.png')
        
        # 生成热力图
        print("\nGenerating heatmap...")
        create_heatmap(df, 'output/sector_rotation_heatmap.png')
        
        print("\nComplete! Charts saved to output/ directory")
    else:
        print("Error: Cannot retrieve historical data")
        sys.exit(1)


if __name__ == '__main__':
    main()
