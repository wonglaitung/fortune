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

from data_services.tencent_finance import get_hk_stock_data_tencent


def get_historical_sector_performance(days=180):
    """
    获取历史板块表现数据
    
    Args:
        days: 获取天数，默认180天（半年）
    
    Returns:
        DataFrame: 包含日期、板块、涨跌幅、排名的数据
    """
    print(f"正在获取过去 {days} 天的板块历史数据...")
    
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
    print("步骤 1/3: 获取所有股票的历史数据...")
    stock_data_cache = {}
    all_stocks = [stock for stocks in sectors.values() for stock in stocks]
    
    for i, stock in enumerate(all_stocks):
        print(f"  获取股票 {i+1}/{len(all_stocks)}: {stock}")
        try:
            stock_code_simple = stock.replace('.HK', '')
            df = get_hk_stock_data_tencent(stock_code_simple, period_days=days)
            if df is not None and not df.empty:
                # DataFrame的索引已经是'Date'了
                stock_data_cache[stock] = df
        except Exception as e:
            print(f"    警告: 无法获取 {stock} 的数据 - {str(e)}")
            continue
    
    print(f"成功获取 {len(stock_data_cache)}/{len(all_stocks)} 只股票的数据")
    
    # 获取实际可用的日期范围
    if stock_data_cache:
        # 找到所有股票数据的日期范围
        all_dates = []
        for stock, df in stock_data_cache.items():
            all_dates.extend(df.index.tolist())
        
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            print(f"数据日期范围: {min_date.strftime('%Y-%m-%d')} 到 {max_date.strftime('%Y-%m-%d')}")
            
            # 使用实际可用的日期范围
            start_date = min_date
            end_date = max_date
        else:
            print("错误：没有可用的日期数据")
            return None
    else:
        print("错误：没有获取到任何股票数据")
        return None
    
    # 按双周采样数据（避免数据点过多）
    dates = pd.date_range(start=start_date, end=end_date, freq='2W-Fri')
    # 数据已经有UTC时区，直接使用
    dates = dates.tz_convert('UTC')
    
    results = []
    
    print(f"\n步骤 2/3: 计算 {len(dates)} 个时间点的板块表现...")
    for i, date in enumerate(dates):
        print(f"  处理日期 {i+1}/{len(dates)}: {date.strftime('%Y-%m-%d')}")
        
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
        print(f"\n步骤 3/3: 完成！共生成 {len(results)} 条数据记录")
        return df
    else:
        print("警告：未能获取到任何板块数据")
        return None


def create_river_plot(df, output_path='output/sector_rotation_river_plot.png'):
    """
    创建板块轮动河流图
    
    Args:
        df: 包含历史板块数据的DataFrame
        output_path: 输出文件路径
    """
    if df is None or df.empty:
        print("错误：没有可用的数据")
        return
    
    print("正在生成河流图...")
    
    # 准备数据：按日期和板块汇总
    df_pivot = df.pivot(index='date', columns='sector_name', values='rank')
    
    # 按日期排序
    df_pivot = df_pivot.sort_index()
    
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
    
    # 设置Y轴反转（排名1在顶部）
    ax.invert_yaxis()
    ax.set_ylim(df_pivot.max().max() + 1, 0.5)
    
    # 设置标题和标签
    ax.set_title('港股板块轮动趋势（过去半年）\n排名变化河流图', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax.set_ylabel('板块排名（1=表现最好）', fontsize=12, fontweight='bold')
    
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
    print(f"河流图已保存到: {output_path}")
    
    # 显示图表
    plt.show()


def create_heatmap(df, output_path='output/sector_rotation_heatmap.png'):
    """
    创建板块轮动热力图
    
    Args:
        df: 包含历史板块数据的DataFrame
        output_path: 输出文件路径
    """
    if df is None or df.empty:
        print("错误：没有可用的数据")
        return
    
    print("正在生成热力图...")
    
    # 准备数据
    df_pivot = df.pivot(index='date', columns='sector_name', values='rank')
    df_pivot = df_pivot.sort_index()
    
    # 按月采样（避免数据点过多）
    df_pivot_monthly = df_pivot.resample('M').last()
    
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
    
    # 设置标题和标签
    ax.set_title('港股板块轮动热力图（过去半年）\n每月板块排名', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax.set_ylabel('板块', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('排名（1=表现最好，13=表现最差）', 
                    rotation=270, labelpad=20, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {output_path}")
    
    # 显示图表
    plt.show()


def analyze_sector_rotation(df):
    """
    分析板块轮动特征
    
    Args:
        df: 包含历史板块数据的DataFrame
    """
    if df is None or df.empty:
        print("错误：没有可用的数据")
        return
    
    print("\n=== 板块轮动分析 ===\n")
    
    # 计算每个板块的平均排名
    avg_rank = df.groupby('sector_name')['rank'].mean().sort_values()
    print("各板块平均排名（过去半年）：")
    for sector, rank in avg_rank.items():
        print(f"  {sector:12s}: {rank:.2f}")
    
    # 计算排名波动性
    rank_std = df.groupby('sector_name')['rank'].std().sort_values(ascending=False)
    print("\n各板块排名波动性（标准差）：")
    for sector, std in rank_std.items():
        print(f"  {sector:12s}: {std:.2f}")
    
    # 识别近期强势板块（最近4周）
    recent_df = df[df['date'] >= df['date'].max() - timedelta(weeks=4)]
    recent_avg_rank = recent_df.groupby('sector_name')['rank'].mean().sort_values()
    print("\n近期强势板块（最近4周）：")
    for sector, rank in recent_avg_rank.items():
        print(f"  {sector:12s}: {rank:.2f}")
    
    # 识别近期弱势板块
    print("\n近期弱势板块（最近4周）：")
    for sector, rank in list(recent_avg_rank.items())[::-1]:
        print(f"  {sector:12s}: {rank:.2f}")


def main():
    """主函数"""
    print("=" * 60)
    print("板块轮动河流图生成器")
    print("=" * 60)
    
    # 获取历史数据（过去半年）
    df = get_historical_sector_performance(days=180)
    
    if df is not None and not df.empty:
        # 分析板块轮动特征
        analyze_sector_rotation(df)
        
        # 生成河流图
        print("\n生成河流图...")
        create_river_plot(df, 'output/sector_rotation_river_plot.png')
        
        # 生成热力图
        print("\n生成热力图...")
        create_heatmap(df, 'output/sector_rotation_heatmap.png')
        
        print("\n完成！图表已保存到 output/ 目录")
    else:
        print("错误：无法获取历史数据")
        sys.exit(1)


if __name__ == '__main__':
    main()
