#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态风险控制系统 - 使用真实市场数据

功能：
- 使用真实的恒生指数和VIX数据
- 验证极端市场环境识别
- 测试动态仓位管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.dynamic_risk_control import DynamicRiskControl

def test_march_2025_icbc_scenario():
    """测试2025年3月工商银行悲剧场景"""
    print("=" * 80)
    print("测试：2025年3月工商银行悲剧场景（使用真实市场数据）")
    print("=" * 80)

    # 创建动态风险控制系统
    risk_control = DynamicRiskControl()

    # 获取2025年3月的恒生指数数据
    start_date = "2025-03-01"
    end_date = "2025-03-31"
    
    print(f"\n获取恒生指数数据: {start_date} 至 {end_date}")
    hsi_ticker = yf.Ticker("^HSI")
    hsi_df = hsi_ticker.history(start=start_date, end=end_date)
    
    if len(hsi_df) == 0:
        print("❌ 错误：无法获取恒生指数数据")
        return
    
    print(f"✅ 成功获取 {len(hsi_df)} 行恒生指数数据")
    print(f"日期范围: {hsi_df.index.min()} 至 {hsi_df.index.max()}")
    
    # 获取VIX数据
    print(f"\n获取VIX数据: {start_date} 至 {end_date}")
    vix_ticker = yf.Ticker("^VIX")
    vix_df = vix_ticker.history(start=start_date, end=end_date)
    
    if len(vix_df) == 0:
        print("❌ 错误：无法获取VIX数据")
        return
    
    print(f"✅ 成功获取 {len(vix_df)} 行VIX数据")
    
    # 分析3月份的市场环境
    print("\n" + "=" * 80)
    print("2025年3月市场环境分析")
    print("=" * 80)
    
    # 计算关键指标
    hsi_return_5d = hsi_df['Close'].pct_change(5).iloc[-1] if len(hsi_df) >= 6 else 0
    hsi_return_20d = hsi_df['Close'].pct_change(20).iloc[-1] if len(hsi_df) >= 21 else 0
    
    # 检测极端市场环境
    hsi_data = hsi_df.reset_index()
    hsi_data = hsi_data.rename(columns={'index': 'Date'})
    hsi_data['Date'] = pd.to_datetime(hsi_data['Date']).dt.normalize()
    
    vix_level = float(vix_df['Close'].iloc[-1])
    
    # 模拟股票数据
    stock_data = pd.DataFrame({
        'Return': np.random.normal(0, 0.02, 5)
    })
    
    # 检测极端市场环境
    is_extreme, extreme_conditions, extreme_count = risk_control.detect_extreme_market_conditions(
        hsi_data, vix_level, stock_data
    )
    
    print(f"\n恒生指数5日收益率: {hsi_return_5d:.2%}")
    print(f"恒生指数20日收益率: {hsi_return_20d:.2%}")
    print(f"VIX恐慌指数: {vix_level:.2f}")
    print(f"\n极端市场检测结果:")
    print(f"  是否极端市场: {is_extreme}")
    print(f"  触发条件数量: {extreme_count}/6")
    print(f"  详细条件:")
    for condition, triggered in extreme_conditions.items():
        status = "✅ 触发" if triggered else "❌ 未触发"
        print(f"    - {condition}: {status}")
    
    # 计算市场环境评分
    market_env_score = risk_control.assess_market_environment(hsi_data, vix_level)
    print(f"\n市场环境评分: {market_env_score}/100")
    
    # 模拟交易决策
    print("\n" + "=" * 80)
    print("交易决策测试")
    print("=" * 80)
    
    test_scenarios = [
        ("高置信度买入", 0.75),
        ("中等置信度买入", 0.60),
        ("低置信度买入", 0.52),
    ]
    
    for scenario_name, prediction_prob in test_scenarios:
        # 计算动态仓位
        adjusted_prob, position_size, risk_level = risk_control.get_dynamic_position_size(
            prediction_prob, 'bear', vix_level, market_env_score
        )
        
        print(f"\n{scenario_name}:")
        print(f"  模型预测概率: {prediction_prob:.2%}")
        print(f"  调整后概率: {adjusted_prob:.2%}")
        print(f"  最终仓位: {position_size*100:.0f}%")
        print(f"  风险等级: {risk_level}")
        
        if is_extreme:
            print(f"  ✅ 风险控制生效：极端市场环境下停止交易")
        elif position_size == 0:
            print(f"  ⚠️  风险控制生效：仓位为0，停止交易")
        elif position_size < 1.0:
            print(f"  ⚠️  风险控制生效：降低仓位至 {position_size*100:.0f}%")
        else:
            print(f"  ✅ 正常交易：满仓操作")

def test_extreme_market_detection():
    """测试极端市场环境识别"""
    print("\n" + "=" * 80)
    print("测试：极端市场环境识别（真实数据）")
    print("=" * 80)
    
    # 创建动态风险控制系统
    risk_control = DynamicRiskControl()
    
    # 测试不同历史事件
    test_cases = [
        ("2025年3月", "2025-03-01", "2025-03-31"),
        ("2024年8月", "2024-08-01", "2024-08-31"),
        ("2025年1月", "2025-01-01", "2025-01-31"),
    ]
    
    for case_name, start_date, end_date in test_cases:
        print(f"\n{'─' * 60}")
        print(f"测试案例: {case_name} ({start_date} 至 {end_date})")
        print(f"{'─' * 60}")
        
        try:
            # 获取恒生指数数据
            hsi_ticker = yf.Ticker("^HSI")
            hsi_df = hsi_ticker.history(start=start_date, end=end_date)
            
            # 获取VIX数据
            vix_ticker = yf.Ticker("^VIX")
            vix_df = vix_ticker.history(start=start_date, end=end_date)
            
            if len(hsi_df) == 0 or len(vix_df) == 0:
                print(f"❌ 数据获取失败")
                continue
            
            # 准备数据
            hsi_data = hsi_df.reset_index()
            hsi_data = hsi_data.rename(columns={'index': 'Date'})
            hsi_data['Date'] = pd.to_datetime(hsi_data['Date']).dt.normalize()
            
            vix_level = float(vix_df['Close'].iloc[-1])
            
            # 模拟股票数据
            stock_data = pd.DataFrame({
                'Return': np.random.normal(0, 0.02, 5)
            })
            
            # 检测极端市场环境
            is_extreme, extreme_conditions, extreme_count = risk_control.detect_extreme_market_conditions(
                hsi_data, vix_level, stock_data
            )
            
            # 计算市场环境评分
            market_env_score = risk_control.assess_market_environment(hsi_data, vix_level)
            
            # 计算收益率
            hsi_return_5d = hsi_df['Close'].pct_change(5).iloc[-1] if len(hsi_df) >= 6 else 0
            hsi_return_20d = hsi_df['Close'].pct_change(20).iloc[-1] if len(hsi_df) >= 21 else 0
            
            print(f"恒生指数5日收益率: {hsi_return_5d:.2%}")
            print(f"恒生指数20日收益率: {hsi_return_20d:.2%}")
            print(f"VIX恐慌指数: {vix_level:.2f}")
            print(f"市场环境评分: {market_env_score:.0f}/100")
            print(f"极端市场: {'是' if is_extreme else '否'}")
            print(f"触发条件: {extreme_count}/6")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    # 运行测试
    test_march_2025_icbc_scenario()
    test_extreme_market_detection()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)
