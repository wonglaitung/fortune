# -*- coding: utf-8 -*-
"""
港股主力资金追踪器 - 历史数据分析
作者：AI助手
说明：
- 分析过去三个月每天的数据
- 识别建仓和出货信号日期
"""

import warnings
import os
import math
import time
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial

warnings.filterwarnings("ignore")
os.environ['MPLBACKEND'] = 'Agg'

import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np

# 导入hk_smart_money_tracker.py中的配置和函数
from hk_smart_money_tracker import (
    WATCHLIST, DAYS_ANALYSIS, VOL_WINDOW, PRICE_WINDOW, 
    BUILDUP_MIN_DAYS, DISTRIBUTION_MIN_DAYS,
    PRICE_LOW_PCT, PRICE_HIGH_PCT, VOL_RATIO_BUILDUP, 
    VOL_RATIO_DISTRIBUTION, SOUTHBOUND_UNIT_CONVERSION, 
    SOUTHBOUND_THRESHOLD, OUTPERFORMS_REQUIRE_POSITIVE, 
    OUTPERFORMS_USE_RS, AK_CALL_SLEEP, southbound_cache, 
    fetch_ggt_components, mark_runs, safe_round, get_hsi_return
)

def analyze_stock_historical(code, name, start_date, end_date):
    """
    分析指定日期范围内的股票数据，识别建仓和出货信号
    """
    try:
        print(f"\n🔍 分析 {name} ({code}) 从 {start_date} 到 {end_date}...")
        ticker = yf.Ticker(code)
        
        # 获取指定日期范围的数据，额外获取 PRICE_WINDOW 天以确保有足够的历史数据
        extended_start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=PRICE_WINDOW)
        full_hist = ticker.history(start=extended_start_date.strftime('%Y-%m-%d'), end=end_date, repair=True)
            
        if len(full_hist) < PRICE_WINDOW:
            print(f"⚠️  {name} 数据不足（需要至少 {PRICE_WINDOW} 日）")
            return None

        # 保留交易日
        full_hist = full_hist[full_hist.index.weekday < 5]
        
        # 数据质量检查
        if full_hist.empty:
            print(f"⚠️  {name} 数据为空")
            return None
            
        # 检查是否有足够的数据点
        if len(full_hist) < 5:
            print(f"⚠️  {name} 数据不足")
            return None
            
        # 检查数据是否包含必要的列
        required_columns = ['Open', 'Close', 'Volume']
        for col in required_columns:
            if col not in full_hist.columns:
                print(f"⚠️  {name} 缺少必要的列 {col}")
                return None
                
        # 检查数据是否包含有效的数值
        if full_hist['Close'].isna().all() or full_hist['Volume'].isna().all():
            print(f"⚠️  {name} 数据包含大量缺失值")
            return None
            
        # 移除包含异常值的行
        full_hist = full_hist.dropna(subset=['Close', 'Volume'])
        full_hist = full_hist[(full_hist['Close'] > 0) & (full_hist['Volume'] >= 0)]
        
        if len(full_hist) < 5:
            print(f"⚠️  {name} 清理异常值后数据不足")
            return None

        # 基础指标（在 full_hist 上计算）
        full_hist['Vol_MA20'] = full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).mean()
        full_hist['MA5'] = full_hist['Close'].rolling(5, min_periods=1).mean()
        full_hist['MA10'] = full_hist['Close'].rolling(10, min_periods=1).mean()

        # MACD
        full_hist['EMA12'] = full_hist['Close'].ewm(span=12, adjust=False).mean()
        full_hist['EMA26'] = full_hist['Close'].ewm(span=26, adjust=False).mean()
        full_hist['MACD'] = full_hist['EMA12'] - full_hist['EMA26']
        full_hist['MACD_Signal'] = full_hist['MACD'].ewm(span=9, adjust=False).mean()

        # RSI (Wilder)
        delta_full = full_hist['Close'].diff()
        gain = delta_full.clip(lower=0)
        loss = -delta_full.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        full_hist['RSI'] = 100 - (100 / (1 + rs))

        # Returns & Volatility (年化)
        full_hist['Returns'] = full_hist['Close'].pct_change()
        # 使用 min_periods=10 保证样本充足再年化
        full_hist['Volatility'] = full_hist['Returns'].rolling(20, min_periods=10).std() * math.sqrt(252)

        # OBV 从 full_hist 累计
        full_hist['OBV'] = 0.0
        for i in range(1, len(full_hist)):
            if full_hist['Close'].iat[i] > full_hist['Close'].iat[i-1]:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1] + full_hist['Volume'].iat[i]
            elif full_hist['Close'].iat[i] < full_hist['Close'].iat[i-1]:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1] - full_hist['Volume'].iat[i]
            else:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1]

        # 获取恒生指数数据用于对比
        hsi_ticker = yf.Ticker("^HSI")
        hsi_hist = hsi_ticker.history(start=extended_start_date.strftime('%Y-%m-%d'), end=end_date)
        if hsi_hist.empty:
            print(f"⚠️ 无法获取恒生指数数据用于 {name}")
            return None

        # 存储结果
        results = []

        # 遍历每个交易日进行分析
        for i in range(PRICE_WINDOW, len(full_hist)):
            # 确定分析窗口
            current_date = full_hist.index[i]
            window_start_idx = max(0, i - DAYS_ANALYSIS + 1)
            window_end_idx = i + 1
            
            if window_end_idx - window_start_idx < 5:
                continue
                
            main_hist = full_hist.iloc[window_start_idx:window_end_idx].copy()
            
            # 计算价格百分位
            price_window_start = max(0, i - PRICE_WINDOW + 1)
            price_window_data = full_hist.iloc[price_window_start:window_end_idx]
            low60 = price_window_data['Close'].min()
            high60 = price_window_data['Close'].max()
            
            if high60 == low60:
                main_hist['Price_Percentile'] = 50.0
            else:
                main_hist['Price_Percentile'] = ((main_hist['Close'] - low60) / (high60 - low60) * 100).clip(0, 100)

            # 计算其他指标
            main_hist['Vol_Ratio'] = main_hist['Volume'] / main_hist['Vol_MA20']
            main_hist['MA5'] = full_hist['MA5'].reindex(main_hist.index, method='ffill')
            main_hist['MA10'] = full_hist['MA10'].reindex(main_hist.index, method='ffill')
            main_hist['MACD'] = full_hist['MACD'].reindex(main_hist.index, method='ffill')
            main_hist['MACD_Signal'] = full_hist['MACD_Signal'].reindex(main_hist.index, method='ffill')
            main_hist['RSI'] = full_hist['RSI'].reindex(main_hist.index, method='ffill')
            main_hist['Volatility'] = full_hist['Volatility'].reindex(main_hist.index, method='ffill')
            main_hist['OBV'] = full_hist['OBV'].reindex(main_hist.index, method='ffill').fillna(0.0)

            # 南向资金：按日期获取并缓存，转换为"万"
            main_hist['Southbound_Net'] = 0.0
            for ts in main_hist.index:
                # 排除周六日
                if ts.weekday() >= 5:
                    continue
                date_str = ts.strftime('%Y%m%d')
                try:
                    df_ggt = fetch_ggt_components(code, date_str)
                    if df_ggt is None:
                        continue
                    # 获取南向资金净买入数据
                    if '持股市值变化-1日' in df_ggt.columns and not df_ggt.empty:
                        net_val = df_ggt['持股市值变化-1日'].iloc[0]
                        if pd.notna(net_val):
                            # 转换为万元
                            main_hist.at[ts, 'Southbound_Net'] = float(net_val) / SOUTHBOUND_UNIT_CONVERSION
                except Exception as e:
                    print(f"⚠️ 处理南向资金数据时出错 {code} {date_str}: {e}")
                    pass

            # 计算区间收益
            start_date_analysis, end_date_analysis = main_hist.index[0], main_hist.index[-1]
            stock_ret = (main_hist['Close'].iloc[-1] - main_hist['Close'].iloc[0]) / main_hist['Close'].iloc[0]
            
            # 获取恒指收益
            hsi_ret = get_hsi_return(start_date_analysis, end_date_analysis)
            if pd.isna(hsi_ret):
                hsi_ret = 0.0
            
            rs_diff = stock_ret - hsi_ret
            if (1.0 + hsi_ret) == 0:
                rs_ratio = float('inf') if (1.0 + stock_ret) > 0 else float('-inf')
            else:
                rs_ratio = (1.0 + stock_ret) / (1.0 + hsi_ret) - 1.0

            # === 建仓信号 ===
            def is_buildup(row):
                # 基本条件
                price_cond = row['Price_Percentile'] < PRICE_LOW_PCT
                vol_cond = pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_BUILDUP
                sb_cond = pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] > SOUTHBOUND_THRESHOLD
                
                # 辅助条件
                # MACD线上穿信号线
                macd_cond = pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] > row['MACD_Signal']
                # RSI超卖（调整阈值从30到35）
                rsi_cond = pd.notna(row.get('RSI')) and row['RSI'] < 35
                # OBV上升
                obv_cond = pd.notna(row.get('OBV')) and row['OBV'] > 0
                # 价格相对于5日均线位置（价格低于5日均线）
                ma5_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA5')) and row['Close'] < row['MA5']
                # 价格相对于10日均线位置（价格低于10日均线）
                ma10_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA10')) and row['Close'] < row['MA10']
                
                # 计算满足的辅助条件数量
                aux_conditions = [macd_cond, rsi_cond, obv_cond, ma5_cond, ma10_cond]
                satisfied_aux_count = sum(aux_conditions)
                
                # 如果满足至少1个辅助条件，或者满足多个条件中的部分条件（更宽松的策略）
                aux_cond = satisfied_aux_count >= 1
                
                # 调试信息
                if price_cond and vol_cond and sb_cond and not aux_cond:
                    print(f"  ⚠️  {code} {row.name.strftime('%Y-%m-%d')} 满足基本条件但不满足辅助条件")
                    print(f"    价格百分位: {row['Price_Percentile']:.2f} (< {PRICE_LOW_PCT})")
                    print(f"    量比: {row['Vol_Ratio']:.2f} (> {VOL_RATIO_BUILDUP})")
                    print(f"    南向资金: {row.get('Southbound_Net', 'N/A')}")
                    print(f"    MACD: {row.get('MACD', 'N/A')}, MACD信号线: {row.get('MACD_Signal', 'N/A')}, 条件: {macd_cond}")
                    print(f"    RSI: {row.get('RSI', 'N/A')}, 条件: {rsi_cond}")
                    print(f"    OBV: {row.get('OBV', 'N/A')}, 条件: {obv_cond}")
                    print(f"    MA5: {row.get('Close', 'N/A')} < {row.get('MA5', 'N/A')}, 条件: {ma5_cond}")
                    print(f"    MA10: {row.get('Close', 'N/A')} < {row.get('MA10', 'N/A')}, 条件: {ma10_cond}")
                    print(f"    满足的辅助条件数: {satisfied_aux_count}")
                
                return price_cond and vol_cond and sb_cond and aux_cond

            main_hist['Buildup_Signal'] = main_hist.apply(is_buildup, axis=1)
            main_hist['Buildup_Confirmed'] = mark_runs(main_hist['Buildup_Signal'], BUILDUP_MIN_DAYS)

            # === 出货信号 ===
            main_hist['Prev_Close'] = main_hist['Close'].shift(1)
            def is_distribution(row):
                # 基本条件
                price_cond = row['Price_Percentile'] > PRICE_HIGH_PCT
                vol_cond = (pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_DISTRIBUTION)
                sb_cond = (pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] < -SOUTHBOUND_THRESHOLD)
                price_down_cond = (pd.notna(row.get('Prev_Close')) and (row['Close'] < row['Prev_Close'])) or (row['Close'] < row['Open'])
                
                # 辅助条件
                # MACD线下穿信号线
                macd_cond = pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] < row['MACD_Signal']
                # RSI超买（调整阈值从70到65）
                rsi_cond = pd.notna(row.get('RSI')) and row['RSI'] > 65
                # OBV下降
                obv_cond = pd.notna(row.get('OBV')) and row['OBV'] < 0
                # 价格相对于5日均线位置（价格高于5日均线）
                ma5_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA5')) and row['Close'] > row['MA5']
                # 价格相对于10日均线位置（价格高于10日均线）
                ma10_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA10')) and row['Close'] > row['MA10']
                
                # 计算满足的辅助条件数量
                aux_conditions = [macd_cond, rsi_cond, obv_cond, ma5_cond, ma10_cond]
                satisfied_aux_count = sum(aux_conditions)
                
                # 如果满足至少1个辅助条件，或者满足多个条件中的部分条件（更宽松的策略）
                aux_cond = satisfied_aux_count >= 1
                
                # 调试信息
                if price_cond and vol_cond and sb_cond and price_down_cond and not aux_cond:
                    print(f"  ⚠️  {code} {row.name.strftime('%Y-%m-%d')} 满足出货基本条件但不满足辅助条件")
                    print(f"    价格百分位: {row['Price_Percentile']:.2f} (> {PRICE_HIGH_PCT})")
                    print(f"    量比: {row['Vol_Ratio']:.2f} (> {VOL_RATIO_DISTRIBUTION})")
                    print(f"    南向资金: {row.get('Southbound_Net', 'N/A')}")
                    print(f"    价格下行: {price_down_cond}")
                    print(f"    MACD: {row.get('MACD', 'N/A')}, MACD信号线: {row.get('MACD_Signal', 'N/A')}, 条件: {macd_cond}")
                    print(f"    RSI: {row.get('RSI', 'N/A')}, 条件: {rsi_cond}")
                    print(f"    OBV: {row.get('OBV', 'N/A')}, 条件: {obv_cond}")
                    print(f"    MA5: {row.get('Close', 'N/A')} > {row.get('MA5', 'N/A')}, 条件: {ma5_cond}")
                    print(f"    MA10: {row.get('Close', 'N/A')} > {row.get('MA10', 'N/A')}, 条件: {ma10_cond}")
                    print(f"    满足的辅助条件数: {satisfied_aux_count}")
                
                return price_cond and vol_cond and sb_cond and price_down_cond and aux_cond

            main_hist['Distribution_Signal'] = main_hist.apply(is_distribution, axis=1)
            main_hist['Distribution_Confirmed'] = mark_runs(main_hist['Distribution_Signal'], DISTRIBUTION_MIN_DAYS)
            
            # === 放量上涨和缩量回调信号 ===
            # 放量上涨：收盘价 > 开盘价 且 Vol_Ratio > 1.5
            main_hist['Strong_Volume_Up'] = (main_hist['Close'] > main_hist['Open']) & (main_hist['Vol_Ratio'] > 1.5)
            # 缩量回调：收盘价 < 前一日收盘价 且 Vol_Ratio < 1.0 且跌幅 < 2%
            main_hist['Weak_Volume_Down'] = (main_hist['Close'] < main_hist['Prev_Close']) & (main_hist['Vol_Ratio'] < 1.0) & ((main_hist['Prev_Close'] - main_hist['Close']) / main_hist['Prev_Close'] < 0.02)

            # 只记录当前日期的数据，避免重复记录历史日期
            # 使用窗口内的最后一天作为当前日期
            if len(main_hist) > 0:
                row = main_hist.iloc[-1]  # 取窗口内的最后一天数据
                has_buildup = row['Buildup_Confirmed']
                has_distribution = row['Distribution_Confirmed']
                strong_volume_up = row['Strong_Volume_Up']  # 放量上涨信号
                weak_volume_down = row['Weak_Volume_Down']  # 缩量回调信号
                
                # 计算换手率
                float_shares = None
                try:
                    float_shares = ticker.info.get('floatShares', 0)
                    if float_shares is None or float_shares == 0:
                        float_shares = ticker.info.get('sharesOutstanding', 0)
                except Exception:
                    pass
                
                turnover_rate = (row['Volume'] / float_shares) * 100 if float_shares is not None and float_shares > 0 else None

                results.append({
                    'date': main_hist.index[-1].strftime('%Y-%m-%d'),  # 使用窗口内的最后一天日期
                    'code': code,
                    'name': name,
                    'last_close': safe_round(row['Close'], 2),
                    'price_percentile': safe_round(row['Price_Percentile'], 2),
                    'vol_ratio': safe_round(row['Vol_Ratio'], 2),
                    'southbound': safe_round(row['Southbound_Net'], 2),
                    'relative_strength': safe_round(rs_ratio, 4),  # 保持小数形式
                    'relative_strength_diff': safe_round(rs_diff, 4),  # 保持小数形式
                    'turnover_rate': safe_round(turnover_rate, 2),
                    'has_buildup': has_buildup,
                    'has_distribution': has_distribution,
                    'strong_volume_up': strong_volume_up,  # 放量上涨信号
                    'weak_volume_down': weak_volume_down   # 缩量回调信号
                })

        return results

    except Exception as e:
        print(f"❌ {name} 分析出错: {e}")
        return None

def main():
    import argparse
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='港股主力资金追踪器 - 历史数据分析')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD 格式)', default=None)
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD 格式)', default=None)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 港股主力资金追踪器 - 历史数据分析")
    print("="*80)

    # 设置日期范围
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        # 默认为过去三个月
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
    
    print(f"分析时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"分析 {len(WATCHLIST)} 只股票")

    all_results = []
    
    # 使用线程池并行分析多只股票
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 创建偏函数以固定日期参数
        analyze_func = partial(analyze_stock_historical, 
                              start_date=start_date.strftime('%Y-%m-%d'), 
                              end_date=end_date.strftime('%Y-%m-%d'))
        
        # 提交所有任务
        future_to_stock = {executor.submit(analyze_func, code, name): (code, name) 
                          for code, name in WATCHLIST.items()}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_stock):
            code, name = future_to_stock[future]
            try:
                results = future.result(timeout=300)  # 设置超时时间为5分钟
                if results:
                    all_results.extend(results)
            except concurrent.futures.TimeoutError:
                print(f"⚠️  {name} ({code}) 分析超时")
            except Exception as e:
                print(f"❌ {name} ({code}) 分析出错: {e}")

    # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 检查是否有数据
        print(f"📊 收集到 {len(all_results)} 条原始数据")
        if df.empty:
            print("❌ 无结果")
            # 即使没有信号数据，也生成一个空的报告文件
            df_empty = pd.DataFrame(columns=[
                '股票名称', '代码', '最新价', '换手率(%)',
                '位置(%)', '量比',
                '相对强度', '相对强度差值',
                '南向资金(万)', '建仓信号', '出货信号',
                '放量上涨', '缩量回调', '日期'
            ])
            with pd.ExcelWriter('hk_smart_money_historical_report.xlsx', engine='openpyxl') as writer:
                df_empty.to_excel(writer, sheet_name='所有信号', index=False)
                df_empty.to_excel(writer, sheet_name='建仓信号', index=False)
                df_empty.to_excel(writer, sheet_name='出货信号', index=False)
            print("⚠️ 已生成空的报告文件")
            return
        
        print(f"📊 共收集到 {len(df)} 条信号数据")
        
        # 为展示方便，添加展示列（百分比形式）但保留原始数值列用于机器化处理
        df['RS_ratio_%'] = df['relative_strength'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else None)
        df['RS_diff_%'] = df['relative_strength_diff'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else None)
        
        # 分离各种信号
        buildup_signals = df[df['has_buildup'] == True]
        distribution_signals = df[df['has_distribution'] == True]
        strong_volume_up_signals = df[df['strong_volume_up'] == True]
        weak_volume_down_signals = df[df['weak_volume_down'] == True]
        
        # 保存结果到Excel
        with pd.ExcelWriter('hk_smart_money_historical_report.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='所有信号', index=False)
            buildup_signals.to_excel(writer, sheet_name='建仓信号', index=False)
            distribution_signals.to_excel(writer, sheet_name='出货信号', index=False)
            strong_volume_up_signals.to_excel(writer, sheet_name='放量上涨', index=False)
            weak_volume_down_signals.to_excel(writer, sheet_name='缩量回调', index=False)
    
    print("\n" + "="*120)
    print("📊 历史信号分析结果")
    print("="*120)
    
    # 选择并重命名列用于最终报告
    df_report = df[[
        'name', 'code', 'last_close', 'turnover_rate',
        'price_percentile', 'vol_ratio',
        'RS_ratio_%', 'RS_diff_%',
        'southbound', 'has_buildup', 'has_distribution',
        'strong_volume_up', 'weak_volume_down', 'date'
    ]]
    df_report.columns = [
        '股票名称', '代码', '最新价', '换手率(%)',
        '位置(%)', '量比',
        '相对强度(%)', '相对强度差值(%)',
        '南向资金(万)', '建仓信号', '出货信号',
        '放量上涨', '缩量回调', '日期'
    ]
    
    if not buildup_signals.empty:
        print("\n🟢 建仓信号:")
        buildup_summary = df_report[df_report['建仓信号'] == True][[
            '股票名称', '代码', '最新价', '位置(%)', '量比', '南向资金(万)', '相对强度(%)', '日期'
        ]].copy()
        print(buildup_summary.to_string(index=False))
    
    if not distribution_signals.empty:
        print("\n🔴 出货信号:")
        distribution_summary = df_report[df_report['出货信号'] == True][[
            '股票名称', '代码', '最新价', '位置(%)', '量比', '南向资金(万)', '相对强度(%)', '日期'
        ]].copy()
        print(distribution_summary.to_string(index=False))
        
    if not strong_volume_up_signals.empty:
        print("\n📈 放量上涨信号:")
        strong_volume_up_summary = df_report[df_report['放量上涨'] == True][[
            '股票名称', '代码', '最新价', '量比', '南向资金(万)', '日期'
        ]].copy()
        print(strong_volume_up_summary.to_string(index=False))
        
    if not weak_volume_down_signals.empty:
        print("\n📉 缩量回调信号:")
        weak_volume_down_summary = df_report[df_report['缩量回调'] == True][[
            '股票名称', '代码', '最新价', '量比', '南向资金(万)', '日期'
        ]].copy()
        print(weak_volume_down_summary.to_string(index=False))
    
    print(f"\n📈 总结:")
    print(f"  - 检测到建仓信号 {len(buildup_signals)} 次")
    print(f"  - 检测到出货信号 {len(distribution_signals)} 次")
    print(f"  - 检测到放量上涨信号 {len(strong_volume_up_signals)} 次")
    print(f"  - 检测到缩量回调信号 {len(weak_volume_down_signals)} 次")
    print(f"  - 详细报告已保存到: hk_smart_money_historical_report.xlsx")
    
    # 按日期统计信号
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        daily_signals = df.groupby('date').agg({
            'has_buildup': 'sum',
            'has_distribution': 'sum',
            'strong_volume_up': 'sum',
            'weak_volume_down': 'sum'
        }).reset_index()
        daily_signals.columns = ['日期', '建仓信号次数', '出货信号次数', '放量上涨次数', '缩量回调次数']
        daily_signals = daily_signals[(daily_signals['建仓信号次数'] > 0) | 
                                     (daily_signals['出货信号次数'] > 0) |
                                     (daily_signals['放量上涨次数'] > 0) |
                                     (daily_signals['缩量回调次数'] > 0)]
        
        if not daily_signals.empty:
            print("\n📅 按日期统计的信号:")
            print(daily_signals.to_string(index=False))

if __name__ == "__main__":
    main()