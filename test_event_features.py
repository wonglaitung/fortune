#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试事件驱动特征 - 单只股票快速测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import FeatureEngineer
from data_services.tencent_finance import get_hk_stock_data_tencent

# 测试股票
TEST_CODE = '0700.HK'
TEST_NAME = '腾讯控股'

def test_event_driven_features():
    """测试事件驱动特征"""
    print("=" * 80)
    print(f"测试事件驱动特征 - {TEST_NAME} ({TEST_CODE})")
    print("=" * 80)

    # 创建特征工程器
    feature_engineer = FeatureEngineer()

    # 获取股票数据
    print(f"\n📊 获取 {TEST_NAME} 数据...")
    stock_df = get_hk_stock_data_tencent(TEST_CODE.replace('.HK', ''), period_days=365)

    if stock_df is None or stock_df.empty:
        print(f"❌ 无法获取股票数据")
        return False

    print(f"✅ 成功获取 {len(stock_df)} 条数据")
    print(f"日期范围: {stock_df.index[0]} 至 {stock_df.index[-1]}")

    # 计算技术指标
    print(f"\n🔧 计算技术指标...")
    stock_df = feature_engineer.calculate_technical_features(stock_df)
    print(f"✅ 技术指标完成，列数: {len(stock_df.columns)}")

    # 添加事件驱动特征
    print(f"\n🎯 添加事件驱动特征（9个）...")
    stock_df = feature_engineer.create_event_driven_features(TEST_CODE, stock_df)
    print(f"✅ 事件驱动特征完成，列数: {len(stock_df.columns)}")

    # 检查事件驱动特征列
    event_features = [
        'Ex_Dividend_In_7d',
        'Ex_Dividend_In_30d',
        'Dividend_Frequency_12m',
        'Earnings_Announcement_In_7d',
        'Earnings_Announcement_In_30d',
        'Days_Since_Last_Earnings',
        'Earnings_Surprise_Score',
        'Earnings_Surprise_Avg_3',
        'Earnings_Surprise_Trend'
    ]

    print(f"\n📋 检查事件驱动特征:")
    found_features = []
    for feat in event_features:
        if feat in stock_df.columns:
            non_null_count = stock_df[feat].notna().sum()
            sample_values = stock_df[feat].dropna().tail(5).tolist()
            print(f"  ✅ {feat}: {non_null_count} 个非空值")
            if sample_values:
                print(f"      样本值: {sample_values}")
            found_features.append(feat)
        else:
            print(f"  ❌ {feat}: 未找到")

    # 统计特征摘要
    if found_features:
        print(f"\n📊 事件驱动特征摘要:")
        summary_df = stock_df[found_features].describe()
        print(summary_df)
    else:
        print(f"\n⚠️ 未找到任何事件驱动特征")

    # 列出所有列名（调试用）
    print(f"\n🔍 所有列名（包含'Event', 'Earning', 'Dividend', 'Ex_'的列）:")
    matching_cols = [col for col in stock_df.columns if any(keyword in col.lower() for keyword in ['event', 'earning', 'dividend', 'ex_'])]
    for col in matching_cols:
        print(f"  - {col}")

    return True

if __name__ == "__main__":
    success = test_event_driven_features()
    if success:
        print(f"\n✅ 测试通过")
    else:
        print(f"\n❌ 测试失败")