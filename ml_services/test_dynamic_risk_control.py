#!/usr/bin/env python3
"""
测试动态风险控制系统
验证极端市场环境识别和动态仓位管理功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.dynamic_risk_control import DynamicRiskControl, calculate_market_beta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_extreme_market_detection():
    """测试极端市场环境检测"""
    print("=" * 60)
    print("测试1: 极端市场环境检测")
    print("=" * 60)

    risk_control = DynamicRiskControl()

    # 模拟正常市场数据（DataFrame格式）
    np.random.seed(42)
    n = 30
    dates = pd.date_range(end=datetime.now(), periods=n)
    normal_hsi_data = pd.DataFrame({
        'Close': 18500 * (1 + np.random.normal(0, 0.005, n).cumsum() / 10),
        'Volume': np.random.normal(1000000, 100000, n),
        'Date': dates
    })

    # 模拟极端市场数据（DataFrame格式，股灾）
    # 让跌幅更大，连续下跌，成交量萎缩
    extreme_dates = pd.date_range(end=datetime.now(), periods=n)
    extreme_prices = []
    price = 18500
    for i in range(n):
        if i < n-10:
            price = price * (1 + np.random.normal(0, 0.005))
        else:
            # 最后10天连续暴跌
            price = price * (1 + np.random.normal(-0.06, 0.01))
        extreme_prices.append(price)
    
    extreme_hsi_data = pd.DataFrame({
        'Close': extreme_prices,
        'Volume': np.concatenate([np.random.normal(1000000, 100000, n-10), np.random.normal(400000, 50000, 10)]),
        'Date': extreme_dates
    })

    # 测试正常市场
    print("\n【正常市场场景】")
    vix_level = 20
    stock_data = pd.DataFrame({'Return': [0.01, 0.02, -0.01, 0.03, -0.02]})
    is_extreme, conditions, count = risk_control.detect_extreme_market_conditions(
        normal_hsi_data, vix_level, stock_data
    )
    print(f"是否极端市场: {is_extreme}")
    print(f"触发条件数量: {count}")
    print(f"详细条件: {conditions}")
    assert not is_extreme, "正常市场不应被识别为极端市场"

    # 测试极端市场
    print("\n【极端市场场景（股灾）】")
    vix_level = 45  # VIX > 30 触发
    stock_data = pd.DataFrame({'Return': [-0.05, -0.08, -0.03, -0.06, -0.07]})  # 下跌占比 > 80% 触发
    is_extreme, conditions, count = risk_control.detect_extreme_market_conditions(
        extreme_hsi_data, vix_level, stock_data
    )
    print(f"是否极端市场: {is_extreme}")
    print(f"触发条件数量: {count}")
    print(f"详细条件: {conditions}")
    assert is_extreme, "股灾市场应被识别为极端市场"

    print("\n✅ 极端市场环境检测测试通过")

def test_dynamic_position_management():
    """测试动态仓位管理"""
    print("\n" + "=" * 60)
    print("测试2: 动态仓位管理")
    print("=" * 60)

    risk_control = DynamicRiskControl()

    # 测试不同场景下的仓位调整
    test_cases = [
        {
            'name': '高置信度 + 低风险市场',
            'prob': 0.70,
            'market_regime': 'bull',
            'vix_level': 15,
            'expected_risk': 'LOW',
            'expected_min_position': 0.8
        },
        {
            'name': '中等置信度 + 低风险市场',
            'prob': 0.60,
            'market_regime': 'bull',
            'vix_level': 15,
            'expected_risk': 'LOW',
            'expected_min_position': 0.6
        },
        {
            'name': '高置信度 + 中等风险市场（熊市）',
            'prob': 0.70,
            'market_regime': 'bear',
            'vix_level': 25,
            'expected_risk': 'HIGH',
            'expected_min_position': 0.3
        },
        {
            'name': '极端市场环境（VIX > 30）',
            'prob': 0.80,
            'market_regime': 'bear',
            'vix_level': 35,  # VIX > 30，触发极端市场
            'expected_risk': 'CRITICAL',
            'expected_min_position': 0.0
        }
    ]

    for case in test_cases:
        print(f"\n【{case['name']}】")
        print(f"  预测概率: {case['prob']}")
        print(f"  市场状态: {case['market_regime']}")
        print(f"  VIX水平: {case['vix_level']}")

        adjusted_prob, position_size, risk_level = risk_control.get_dynamic_position_size(
            case['prob'],
            case['market_regime'],
            case['vix_level']
        )

        print(f"  调整后概率: {adjusted_prob:.4f}")
        print(f"  仓位大小: {position_size:.2f}")
        print(f"  风险等级: {risk_level}")

        assert risk_level == case['expected_risk'], f"风险等级应为 {case['expected_risk']}, 实际: {risk_level}"
        assert position_size >= case['expected_min_position'], f"仓位应至少为 {case['expected_min_position']}, 实际: {position_size}"
        assert position_size <= 1.0, "仓位不应超过1.0"

    print("\n✅ 动态仓位管理测试通过")

def test_market_beta_calculation():
    """测试Beta计算"""
    print("\n" + "=" * 60)
    print("测试3: Beta计算")
    print("=" * 60)

    # 模拟数据
    np.random.seed(42)
    n = 100
    market_returns = np.random.normal(0.01, 0.02, n)
    stock_returns = 0.8 * market_returns + np.random.normal(0, 0.015, n)

    beta = calculate_market_beta(stock_returns, market_returns)
    print(f"计算得到的Beta值: {beta:.4f}")
    assert 0.5 < beta < 1.5, f"Beta值应在合理范围内 (0.5-1.5), 实际: {beta}"

    print("✅ Beta计算测试通过")

def test_complete_workflow():
    """测试完整工作流程"""
    print("\n" + "=" * 60)
    print("测试4: 完整工作流程")
    print("=" * 60)

    risk_control = DynamicRiskControl()

    # 模拟市场数据（DataFrame格式）
    np.random.seed(42)
    n = 30
    dates = pd.date_range(end=datetime.now(), periods=n)
    hsi_data = pd.DataFrame({
        'Close': 18000 * (1 + np.concatenate([np.zeros(n-10), np.random.normal(-0.04, 0.01, 10)])),
        'Volume': np.random.normal(1500000, 200000, n),
        'Date': dates
    })
    
    vix_level = 38
    stock_data = pd.DataFrame({'Return': [-0.03, -0.05, -0.02, -0.04, -0.06]})

    # 模拟模型预测
    prediction_prob = 0.65
    market_regime = 'bear'

    print(f"\n【初始状态】")
    print(f"  模型预测概率: {prediction_prob}")
    print(f"  市场状态: {market_regime}")
    print(f"  VIX水平: {vix_level}")

    # 步骤1: 检测极端市场
    is_extreme, conditions, count = risk_control.detect_extreme_market_conditions(
        hsi_data, vix_level, stock_data
    )
    print(f"\n【步骤1: 极端市场检测】")
    print(f"  是否极端市场: {is_extreme}")
    print(f"  触发条件数量: {count}")

    if is_extreme:
        print("  ⚠️  停止交易（极端市场环境）")
        return

    # 步骤2: 计算市场环境评分
    market_env_score = risk_control.assess_market_environment(hsi_data, vix_level)
    print(f"\n【步骤2: 市场环境评分】")
    print(f"  市场环境评分: {market_env_score:.2f}")

    # 步骤3: 动态仓位管理
    adjusted_prob, position_size, risk_level = risk_control.get_dynamic_position_size(
        prediction_prob, market_regime, vix_level, market_env_score
    )
    print(f"\n【步骤3: 动态仓位管理】")
    print(f"  调整后概率: {adjusted_prob:.4f}")
    print(f"  最终仓位: {position_size:.2f}")
    print(f"  风险等级: {risk_level}")

    # 步骤4: 交易决策
    final_threshold = 0.5
    signal = 1 if adjusted_prob > final_threshold else 0
    print(f"\n【步骤4: 交易决策】")
    print(f"  决策阈值: {final_threshold}")
    print(f"  交易信号: {'买入' if signal == 1 else '不买入'}")

    if signal == 1:
        print(f"  💡 执行买入交易，仓位: {position_size * 100:.0f}%")
    else:
        print(f"  ⚠️  不执行买入交易")

    print("\n✅ 完整工作流程测试通过")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("动态风险控制系统测试")
    print("=" * 60)

    try:
        test_extreme_market_detection()
        test_dynamic_position_management()
        test_market_beta_calculation()
        test_complete_workflow()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)