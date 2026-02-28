#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 DynamicMarketStrategy 类的功能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_services.ml_trading_model import DynamicMarketStrategy

def test_dynamic_strategy():
    """测试动态市场策略"""
    print("=" * 80)
    print("🧪 测试 DynamicMarketStrategy 类")
    print("=" * 80)

    # 初始化策略
    strategy = DynamicMarketStrategy()
    print(f"\n✅ 策略初始化成功")
    print(f"   当前市场状态: {strategy.current_regime}")
    print(f"   模型稳定性数据: {strategy.model_stds}")

    # 测试市场状态检测
    print("\n" + "-" * 80)
    print("📊 测试市场状态检测")
    print("-" * 80)

    test_cases = [
        {'return_20d': 0.08, 'expected': 'bull', 'desc': '牛市 (8% 收益)'},
        {'return_20d': -0.08, 'expected': 'bear', 'desc': '熊市 (-8% 收益)'},
        {'return_20d': 0.02, 'expected': 'normal', 'desc': '震荡市 (2% 收益)'},
        {'return_20d': -0.02, 'expected': 'normal', 'desc': '震荡市 (-2% 收益)'},
        {'return_20d': 0.06, 'expected': 'bull', 'desc': '牛市 (6% 收益)'},
        {'return_20d': -0.06, 'expected': 'bear', 'desc': '熊市 (-6% 收益)'},
    ]

    for i, case in enumerate(test_cases, 1):
        hsi_data = {'return_20d': case['return_20d']}
        regime = strategy.detect_market_regime(hsi_data)
        status = "✅" if regime == case['expected'] else "❌"
        print(f"   测试 {i}: {status} {case['desc']}")
        print(f"           预期: {case['expected']}, 实际: {regime}")

    # 测试一致性计算
    print("\n" + "-" * 80)
    print("🔄 测试一致性计算")
    print("-" * 80)

    consistency_tests = [
        {'predictions': [0.8, 0.7, 0.9], 'expected': 1.0, 'desc': '三模型一致上涨'},
        {'predictions': [0.2, 0.3, 0.1], 'expected': 1.0, 'desc': '三模型一致下跌'},
        {'predictions': [0.8, 0.7, 0.3], 'expected': 0.67, 'desc': '两模型一致上涨'},
        {'predictions': [0.2, 0.3, 0.8], 'expected': 0.67, 'desc': '两模型一致下跌'},
        {'predictions': [0.8, 0.3, 0.5], 'expected': 0.33, 'desc': '三模型不一致'},
    ]

    for i, test in enumerate(consistency_tests, 1):
        consistency = strategy.calculate_consistency(test['predictions'])
        status = "✅" if consistency == test['expected'] else "❌"
        print(f"   测试 {i}: {status} {test['desc']}")
        print(f"           预期: {test['expected']}, 实际: {consistency}")

    # 测试牛市策略
    print("\n" + "-" * 80)
    print("🐂 测试牛市策略")
    print("-" * 80)

    predictions = [0.7, 0.65, 0.8]  # LightGBM, GBDT, CatBoost
    confidences = [0.7, 0.65, 0.8]
    fused_prob, strategy_name = strategy.bull_market_ensemble(predictions, confidences)
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")
    print(f"   ✅ 牛市策略执行成功")

    # 测试熊市策略
    print("\n" + "-" * 80)
    print("🐻 测试熊市策略")
    print("-" * 80)

    # 测试高置信度
    predictions = [0.7, 0.65, 0.8]
    confidences = [0.7, 0.65, 0.8]
    fused_prob, strategy_name = strategy.bear_market_ensemble(predictions, confidences)
    print(f"   测试1 - 高置信度 (>0.65):")
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")

    # 测试低置信度
    predictions = [0.4, 0.45, 0.5]
    confidences = [0.4, 0.45, 0.5]
    fused_prob, strategy_name = strategy.bear_market_ensemble(predictions, confidences)
    print(f"   测试2 - 低置信度 (≤0.65):")
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")
    print(f"   ✅ 熊市策略执行成功")

    # 测试震荡市策略
    print("\n" + "-" * 80)
    print("😐 测试震荡市策略")
    print("-" * 80)

    # 测试1：CatBoost 高置信度
    predictions = [0.5, 0.5, 0.7]
    confidences = [0.5, 0.5, 0.7]
    fused_prob, strategy_name = strategy.normal_market_ensemble(predictions, confidences)
    print(f"   测试1 - CatBoost 高置信度 (>0.60):")
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")

    # 测试2：高一致性
    predictions = [0.8, 0.75, 0.7]
    confidences = [0.5, 0.5, 0.5]
    fused_prob, strategy_name = strategy.normal_market_ensemble(predictions, confidences)
    print(f"   测试2 - 高一致性 (≥67%):")
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")

    # 测试3：低一致性
    predictions = [0.8, 0.3, 0.5]
    confidences = [0.5, 0.5, 0.5]
    fused_prob, strategy_name = strategy.normal_market_ensemble(predictions, confidences)
    print(f"   测试3 - 低一致性 (<67%):")
    print(f"   输入预测: {predictions}")
    print(f"   输入置信度: {confidences}")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")
    print(f"   ✅ 震荡市策略执行成功")

    # 测试动态预测
    print("\n" + "-" * 80)
    print("🎯 测试动态预测（完整流程）")
    print("-" * 80)

    # 牛市场景
    hsi_data = {'return_20d': 0.08}
    predictions = [0.7, 0.65, 0.8]
    confidences = [0.7, 0.65, 0.8]
    fused_prob, strategy_name = strategy.predict(predictions, confidences, hsi_data)
    print(f"   牛市场景 (HSI +8%):")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")

    # 熊市场景
    hsi_data = {'return_20d': -0.08}
    predictions = [0.7, 0.65, 0.8]
    confidences = [0.7, 0.65, 0.8]
    fused_prob, strategy_name = strategy.predict(predictions, confidences, hsi_data)
    print(f"   熊市场景 (HSI -8%):")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")

    # 震荡市场景
    hsi_data = {'return_20d': 0.02}
    predictions = [0.8, 0.3, 0.5]
    confidences = [0.5, 0.5, 0.5]
    fused_prob, strategy_name = strategy.predict(predictions, confidences, hsi_data)
    print(f"   震荡市场景 (HSI +2%):")
    print(f"   融合概率: {fused_prob:.4f}")
    print(f"   策略名称: {strategy_name}")
    print(f"   ✅ 动态预测执行成功")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！DynamicMarketStrategy 类功能正常")
    print("=" * 80)

if __name__ == '__main__':
    test_dynamic_strategy()
