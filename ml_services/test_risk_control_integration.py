#!/usr/bin/env python3
"""
集成测试：验证动态风险控制系统在回测中的效果
模拟2025年3月工商银行的悲剧场景，验证风险控制是否能够避免损失
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.dynamic_risk_control import DynamicRiskControl

def simulate_march_2025_tragedy():
    """模拟2025年3月工商银行的悲剧场景"""
    print("=" * 60)
    print("集成测试：模拟2025年3月工商银行悲剧场景")
    print("=" * 60)

    # 工商银行2025年3月的实际情况：
    # - 准确率：90.48%（模型正确预测了下跌趋势）
    # - 胜率：9.52%（几乎所有交易都亏损）
    # - 平均收益率：-4.53%（严重亏损）
    # - 问题：模型预测了下跌，但仍然执行了买入信号

    print("\n【2025年3月工商银行实际情况】")
    print(f"  准确率: 90.48%")
    print(f"  胜率: 9.52%")
    print(f"  平均收益率: -4.53%")
    print(f"  问题: 模型预测了下跌，但仍然执行了买入信号")

    # 模拟2025年3月的市场环境（熊市、高VIX）
    np.random.seed(42)
    n = 22  # 3月大约22个交易日
    dates = pd.date_range(start='2025-03-01', end='2025-03-31', freq='B')

    # 模拟恒生指数暴跌
    hsi_data = pd.DataFrame({
        'Close': np.concatenate([
            np.linspace(18000, 17500, 5),  # 3月初
            np.linspace(17500, 16000, 10),  # 3月中旬暴跌
            np.linspace(16000, 15800, 7)   # 3月底
        ]),
        'Volume': np.concatenate([
            np.random.normal(1000000, 100000, 5),
            np.random.normal(1500000, 200000, 10),
            np.random.normal(1200000, 150000, 7)
        ])
    })

    # VIX水平（恐慌）
    vix_level = 45  # 高于阈值30，触发极端市场

    # 模拟股票数据
    stock_data = pd.DataFrame({
        'Return': np.concatenate([
            np.random.normal(-0.01, 0.02, 5),
            np.random.normal(-0.03, 0.03, 10),
            np.random.normal(-0.02, 0.025, 7)
        ])
    })

    # 模拟模型预测（模型预测了下跌，但置信度不高）
    prediction_prob = 0.48  # 预测概率 < 0.5，模型预测下跌
    market_regime = 'bear'

    print("\n【模拟市场环境】")
    print(f"  市场状态: {market_regime}")
    print(f"  VIX水平: {vix_level}（>30，恐慌）")
    print(f"  恒生指数变化: {hsi_data['Close'].iloc[-1] / hsi_data['Close'].iloc[0] - 1:.2%}")
    print(f"  模型预测概率: {prediction_prob:.2f}（预测下跌）")

    # 测试动态风险控制系统
    risk_control = DynamicRiskControl()

    print("\n【测试1: 传统模式（无动态风险控制）】")
    print(f"  置信度阈值: 0.55")
    print(f"  预测概率: {prediction_prob:.2f}")
    print(f"  决策: {'买入' if prediction_prob > 0.55 else '不买入'}")
    if prediction_prob <= 0.55:
        print(f"  ✅ 幸运：模型没有发出买入信号")
        print(f"  但是：这只是偶然，因为预测概率刚好低于阈值")

    print("\n【测试2: 动态风险控制模式】")

    # 步骤1: 检测极端市场
    is_extreme, conditions, count = risk_control.detect_extreme_market_conditions(
        hsi_data, vix_level, stock_data
    )
    print(f"  步骤1 - 极端市场检测:")
    print(f"    是否极端市场: {is_extreme}")
    print(f"    触发条件数量: {count}")
    print(f"    触发条件: {list(conditions.keys())[list(conditions.values()).index(True)] if any(conditions.values()) else '无'}")

    if is_extreme:
        print(f"  ⚠️  极端市场环境触发！停止交易")
        print(f"  ✅ 风险控制系统生效：避免了可能的损失")
        return True
    else:
        print(f"  继续执行风险评估...")

        # 步骤2: 计算市场环境评分
        market_env_score = risk_control.assess_market_environment(hsi_data, vix_level)
        print(f"  步骤2 - 市场环境评分: {market_env_score:.2f}")

        # 步骤3: 动态仓位管理
        adjusted_prob, position_size, risk_level = risk_control.get_dynamic_position_size(
            prediction_prob, market_regime, vix_level, market_env_score
        )
        print(f"  步骤3 - 动态仓位管理:")
        print(f"    调整后概率: {adjusted_prob:.4f}")
        print(f"    风险等级: {risk_level}")
        print(f"    仓位大小: {position_size:.2f}")

        # 步骤4: 交易决策
        final_threshold = 0.5
        signal = 1 if adjusted_prob > final_threshold else 0
        print(f"  步骤4 - 交易决策:")
        print(f"    决策阈值: {final_threshold}")
        print(f"    交易信号: {'买入' if signal == 1 else '不买入'}")

        if signal == 1:
            print(f"  💡 执行买入交易，仓位: {position_size * 100:.0f}%")
            if position_size < 1.0:
                print(f"  ✅ 风险控制系统生效：仓位已降低")
            else:
                print(f"  ⚠️  风险控制系统未生效：仍然满仓")
                return False
        else:
            print(f"  ✅ 风险控制系统生效：不执行买入交易")
            return True

    return False

def test_comparison_scenario():
    """对比测试：有风险控制 vs 无风险控制"""
    print("\n" + "=" * 60)
    print("对比测试：有风险控制 vs 无风险控制")
    print("=" * 60)

    risk_control = DynamicRiskControl()

    # 模拟10个交易场景
    np.random.seed(42)
    scenarios = []

    for i in range(10):
        # 随机生成市场环境
        vix = np.random.uniform(10, 50)
        market_regime = np.random.choice(['bull', 'bear', 'neutral'])
        prediction_prob = np.random.uniform(0.45, 0.70)

        # 模拟恒生指数数据
        n = 20
        hsi_data = pd.DataFrame({
            'Close': 18000 * (1 + np.random.normal(0, 0.01, n).cumsum() / 10),
            'Volume': np.random.normal(1000000, 100000, n)
        })

        # 模拟股票数据
        stock_data = pd.DataFrame({'Return': np.random.normal(0, 0.02, 5)})

        scenarios.append({
            'vix': vix,
            'market_regime': market_regime,
            'prediction_prob': prediction_prob,
            'hsi_data': hsi_data,
            'stock_data': stock_data
        })

    print(f"\n【模拟 {len(scenarios)} 个交易场景】\n")

    # 统计结果
    traditional_trades = 0
    risk_controlled_trades = 0

    for i, scenario in enumerate(scenarios, 1):
        # 传统模式
        traditional_signal = 1 if scenario['prediction_prob'] > 0.55 else 0

        # 动态风险控制模式
        is_extreme, _, _ = risk_control.detect_extreme_market_conditions(
            scenario['hsi_data'], scenario['vix'], scenario['stock_data']
        )

        if is_extreme:
            risk_controlled_signal = 0
        else:
            market_env_score = risk_control.assess_market_environment(
                scenario['hsi_data'], scenario['vix']
            )
            adjusted_prob, position_size, _ = risk_control.get_dynamic_position_size(
                scenario['prediction_prob'],
                scenario['market_regime'],
                scenario['vix'],
                market_env_score
            )
            risk_controlled_signal = 1 if adjusted_prob > 0.5 and position_size > 0 else 0

        if traditional_signal == 1:
            traditional_trades += 1
        if risk_controlled_signal == 1:
            risk_controlled_trades += 1

        print(f"场景 {i}: VIX={scenario['vix']:.1f}, 预测={scenario['prediction_prob']:.2f}, "
              f"传统={'买入' if traditional_signal else '不买'}, "
              f"风控={'买入' if risk_controlled_signal else '不买'}")

    print(f"\n【统计结果】")
    print(f"  传统模式交易次数: {traditional_trades}/{len(scenarios)}")
    print(f"  风险控制模式交易次数: {risk_controlled_trades}/{len(scenarios)}")
    print(f"  风险控制减少交易: {traditional_trades - risk_controlled_trades} 次")
    print(f"  交易减少比例: {(traditional_trades - risk_controlled_trades) / traditional_trades * 100:.1f}%")

    if risk_controlled_trades < traditional_trades:
        print(f"\n✅ 风险控制系统有效：减少了不必要的交易，降低了风险")
        return True
    else:
        print(f"\n⚠️  风险控制系统可能需要调整")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("动态风险控制系统集成测试")
    print("=" * 60)

    try:
        result1 = simulate_march_2025_tragedy()
        result2 = test_comparison_scenario()

        print("\n" + "=" * 60)
        if result1 and result2:
            print("🎉 所有集成测试通过！")
            print("✅ 动态风险控制系统能够有效避免2025年3月类型的悲剧")
        else:
            print("⚠️  部分测试未通过，可能需要调整风险控制参数")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
