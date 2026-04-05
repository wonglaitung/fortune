#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析异常后的股价表现

研究内容：
1. 异常后不同时间窗口的涨跌幅
2. 异常后的波动率变化
3. 异常后的胜率
4. 异常与股价的因果关系（Granger因果检验）
5. 异常与股价的时间关系（交叉相关分析）
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WATCHLIST


def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    try:
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        # 统一时区
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"⚠️ 获取 {stock_code} 数据失败: {e}")
        return None


def calculate_returns(df, windows=[1, 3, 5, 10, 20]):
    """计算不同时间窗口的收益率"""
    returns = {}
    for window in windows:
        returns[f'return_{window}d'] = df['Close'].pct_change(window)
    return returns


def analyze_post_anomaly_performance(stock_code, anomaly_date, df):
    """分析异常后的股价表现"""
    try:
        # 转换日期格式，统一时区
        anomaly_date = pd.Timestamp(anomaly_date)
        if hasattr(anomaly_date, 'tz') and anomaly_date.tz is not None:
            anomaly_date = anomaly_date.tz_localize(None)

        # 统一DataFrame索引的时区
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        # 确保异常日期在数据中
        if anomaly_date not in df.index:
            # 找到最接近的日期
            idx = df.index.get_indexer([anomaly_date], method='nearest')[0]
            anomaly_date = df.index[idx]

        # 获取异常日期的位置
        idx = df.index.get_loc(anomaly_date)

        # 计算异常后的收益率
        results = {}
        for window in [1, 3, 5, 10, 20]:
            if idx + window < len(df):
                future_close = df['Close'].iloc[idx + window]
                current_close = df['Close'].iloc[idx]
                return_pct = (future_close / current_close - 1) * 100
                results[f'return_{window}d'] = return_pct
            else:
                results[f'return_{window}d'] = None

        # 计算异常后的波动率
        if idx + 5 < len(df):
            post_anomaly_data = df['Close'].iloc[idx:idx+6]
            volatility = post_anomaly_data.pct_change().std() * 100
            results['volatility_5d'] = volatility

        # 计算异常前的波动率
        if idx >= 5:
            pre_anomaly_data = df['Close'].iloc[idx-5:idx+1]
            volatility_pre = pre_anomaly_data.pct_change().std() * 100
            results['volatility_pre_5d'] = volatility_pre

        # 计算异常当日涨跌幅
        if idx > 0:
            prev_close = df['Close'].iloc[idx-1]
            current_close = df['Close'].iloc[idx]
            daily_return = (current_close / prev_close - 1) * 100
            results['daily_return'] = daily_return

        return results

    except Exception as e:
        print(f"⚠️ 分析 {stock_code} 异常表现失败: {e}")
        return None


def load_march_anomalies():
    """加载3月异常数据"""
    try:
        with open('output/march_anomaly_with_reasons.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print("⚠️ 找不到3月异常数据文件")
        return []


def extract_anomalies_from_data(anomaly_data):
    """从异常数据中提取异常信息"""
    anomalies = []

    for date_data in anomaly_data:
        date = date_data['date']
        if date_data.get('has_anomaly') and date_data.get('anomalies'):
            for anomaly_line in date_data['anomalies']:
                # 解析股票代码
                # 格式: "2026-04-05 11:56:33,304 - __main__ - INFO - 0700.HK 2026-03-16 检测到 stock low级异常: ..."
                import re
                match = re.search(r'(\d{4}\.HK)', anomaly_line)
                if match:
                    stock_code = match.group(1)
                    anomalies.append({
                        'date': date,
                        'stock': stock_code,
                        'original_line': anomaly_line
                    })

    return anomalies


def analyze_granger_causality(stock_code, df, max_lag=5):
    """分析异常与股价的Granger因果关系"""
    try:
        # 创建异常序列（如果有异常日期，则为1，否则为0）
        anomaly_dates = set()
        for data in load_march_anomalies():
            if data.get('has_anomaly') and data.get('anomalies'):
                for line in data['anomalies']:
                    if stock_code in line:
                        anomaly_dates.add(data['date'])

        # 创建异常指示序列
        df = df.copy()
        df['anomaly'] = 0
        for date in anomaly_dates:
            date_ts = pd.Timestamp(date)
            if date_ts in df.index:
                df.loc[date_ts, 'anomaly'] = 1

        # 准备数据
        data = df[['Close', 'anomaly']].dropna()
        if len(data) < 20:
            return None

        # Granger因果检验
        results = {}
        try:
            # 测试异常是否预测股价变化
            test_result = grangercausalitytests(data[['Close', 'anomaly']], maxlag=maxlag, verbose=False)
            results['anomaly_to_price'] = {
                'p_values': [test_result[i][0]['ssr_ftest'][1] for i in range(max_lag)],
                'significant': any(test_result[i][0]['ssr_ftest'][1] < 0.05 for i in range(max_lag))
            }
        except Exception as e:
            results['anomaly_to_price'] = {'error': str(e)}

        return results

    except Exception as e:
        print(f"⚠️ Granger因果检验失败: {e}")
        return None


def main():
    """主函数"""
    print("开始分析异常后的股价表现...")
    print("="*60)

    # 加载异常数据
    anomaly_data = load_march_anomalies()
    if not anomaly_data:
        print("❌ 没有异常数据可分析")
        return

    # 提取异常信息
    anomalies = extract_anomalies_from_data(anomaly_data)
    print(f"共找到 {len(anomalies)} 个异常")

    # 扩展时间范围以获取异常后的数据
    all_dates = [pd.Timestamp(a['date']) for a in anomalies]
    min_date = min(all_dates) - timedelta(days=30)
    max_date = max(all_dates) + timedelta(days=30)

    print(f"分析时间范围: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")

    # 分析每个异常后的表现
    results = []
    for i, anomaly in enumerate(anomalies):
        print(f"\n分析异常 {i+1}/{len(anomalies)}: {anomaly['stock']} {anomaly['date']}")

        # 获取股票数据
        df = get_stock_data(anomaly['stock'], min_date, max_date)
        if df is None or len(df) == 0:
            continue

        # 分析异常后的表现
        performance = analyze_post_anomaly_performance(anomaly['stock'], anomaly['date'], df)
        if performance:
            results.append({
                **anomaly,
                'performance': performance
            })

    print(f"\n成功分析 {len(results)} 个异常")

    # 汇总统计
    print(f"\n{'='*60}")
    print("异常后股价表现统计")
    print(f"{'='*60}")

    # 统计各时间窗口的收益率
    for window in [1, 3, 5, 10, 20]:
        returns = [r['performance'].get(f'return_{window}d') for r in results
                   if r['performance'].get(f'return_{window}d') is not None]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            positive_rate = sum(1 for r in returns if r > 0) / len(returns) * 100

            print(f"\n{window}天后:")
            print(f"  平均收益率: {avg_return:+.2f}%")
            print(f"  标准差: {std_return:.2f}%")
            print(f"  上涨概率: {positive_rate:.1f}%")
            print(f"  样本数: {len(returns)}")

    # 统计波动率变化
    volatilities = [r['performance'].get('volatility_5d') for r in results
                    if r['performance'].get('volatility_5d') is not None]
    volatilities_pre = [r['performance'].get('volatility_pre_5d') for r in results
                        if r['performance'].get('volatility_pre_5d') is not None]

    if volatilities:
        avg_vol = np.mean(volatilities)
        print(f"\n异常后5日平均波动率: {avg_vol:.2f}%")

    if volatilities_pre:
        avg_vol_pre = np.mean(volatilities_pre)
        print(f"异常前5日平均波动率: {avg_vol_pre:.2f}%")

    if volatilities and volatilities_pre:
        vol_change = (avg_vol - avg_vol_pre) / avg_vol_pre * 100
        print(f"波动率变化: {vol_change:+.1f}%")

    # 统计异常当日涨跌幅
    daily_returns = [r['performance'].get('daily_return') for r in results
                     if r['performance'].get('daily_return') is not None]
    if daily_returns:
        avg_daily = np.mean(daily_returns)
        print(f"\n异常当日平均涨跌幅: {avg_daily:+.2f}%")

    # 保存结果
    output_file = 'output/anomaly_post_performance.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 生成报告
    report_file = 'output/anomaly_post_performance_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("异常后股价表现分析报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"分析样本数: {len(results)}\n\n")

        # 各时间窗口收益率
        f.write("各时间窗口收益率统计:\n\n")
        for window in [1, 3, 5, 10, 20]:
            returns = [r['performance'].get(f'return_{window}d') for r in results
                       if r['performance'].get(f'return_{window}d') is not None]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                positive_rate = sum(1 for r in returns if r > 0) / len(returns) * 100

                f.write(f"{window}天后:\n")
                f.write(f"  平均收益率: {avg_return:+.2f}%\n")
                f.write(f"  标准差: {std_return:.2f}%\n")
                f.write(f"  上涨概率: {positive_rate:.1f}%\n")
                f.write(f"  样本数: {len(returns)}\n\n")

        # 波动率分析
        if volatilities and volatilities_pre:
            f.write("波动率分析:\n\n")
            f.write(f"异常前5日平均波动率: {avg_vol_pre:.2f}%\n")
            f.write(f"异常后5日平均波动率: {avg_vol:.2f}%\n")
            f.write(f"波动率变化: {vol_change:+.1f}%\n\n")

        # 异常当日表现
        if daily_returns:
            f.write("异常当日表现:\n\n")
            f.write(f"平均涨跌幅: {avg_daily:+.2f}%\n")

        # 详细数据
        f.write("\n详细数据:\n\n")
        for result in results:
            f.write(f"{result['stock']} {result['date']}:\n")
            perf = result['performance']
            for key, value in perf.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:+.2f}%\n")
            f.write("\n")

    print(f"报告已保存到: {report_file}")


if __name__ == '__main__':
    main()
