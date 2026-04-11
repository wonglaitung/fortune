#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载工商银行（1398.HK）最近两年的股票数据
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_icbc_stock_data():
    """下载工商银行股票数据并保存到Excel"""
    
    # 工商银行港股代码
    stock_code = "1398.HK"
    
    # 计算日期范围（最近两年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 两年 = 730天
    
    print(f"正在下载 {stock_code} 的股票数据...")
    print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # 使用yfinance下载数据
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"❌ 无法获取 {stock_code} 的数据")
            return
        
        # 提取需要的列：开市价、最高价、最低价、收市价
        # yfinance的列名：Open, High, Low, Close
        df_selected = df[['Open', 'High', 'Low', 'Close']].copy()
        
        # 重命名列为中文
        df_selected.columns = ['开市价', '最高价', '最低价', '收市价']
        
        # 将索引（日期）转换为列
        df_selected.reset_index(inplace=True)
        df_selected.rename(columns={'Date': '日期'}, inplace=True)
        
        # 格式化日期
        df_selected['日期'] = pd.to_datetime(df_selected['日期']).dt.strftime('%Y-%m-%d')
        
        # 生成文件名
        filename = f"工商银行股票数据_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        # 保存到Excel
        df_selected.to_excel(filename, index=False, engine='openpyxl')
        
        print(f"\n✅ 数据下载成功！")
        print(f"📊 共获取 {len(df_selected)} 条记录")
        print(f"📁 文件已保存至: {filename}")
        print(f"\n数据预览（前5行）:")
        print(df_selected.head().to_string(index=False))
        print(f"\n数据统计:")
        print(f"  日期范围: {df_selected['日期'].iloc[0]} 至 {df_selected['日期'].iloc[-1]}")
        print(f"  收市价范围: {df_selected['收市价'].min():.3f} - {df_selected['收市价'].max():.3f} 港元")
        
        return filename
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

if __name__ == "__main__":
    download_icbc_stock_data()
