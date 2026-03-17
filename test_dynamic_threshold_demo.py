#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态阈值功能演示和验证

快速测试三种方案的阈值调整逻辑，无需完整训练模型
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_services.ml_trading_model import CatBoostModel

print("="*70)
print("动态阈值功能演示")
print("="*70)

# 测试三种配置
configs = [
    {
        'name': '方案A: Balanced + Fixed',
        'class_weight': 'balanced',
        'use_dynamic_threshold': False,
        'desc': '固定阈值0.55，不使用动态调整'
    },
    {
        'name': '方案B: Balanced + Dynamic', 
        'class_weight': 'balanced',
        'use_dynamic_threshold': True,
        'desc': '自动平衡权重 + 动态阈值'
    },
    {
        'name': '方案C: Strong(1.3x) + Dynamic',
        'class_weight': {0: 1.0, 1: 1.3},
        'use_dynamic_threshold': True,
        'desc': '强权重(上涨1.3倍) + 动态阈值'
    }
]

print("\n1. 测试动态阈值计算逻辑")
print("-"*70)

for config in configs:
    print(f"\n{config['name']}")
    print(f"  说明: {config['desc']}")
    
    model = CatBoostModel(
        class_weight=config['class_weight'],
        use_dynamic_threshold=config['use_dynamic_threshold']
    )
    
    # 测试不同市场环境下的阈值
    test_cases = [
        ('牛市 (bull)', 'bull', None),
        ('熊市 (bear)', 'bear', None),
        ('震荡市 (normal)', 'normal', None),
        ('牛市+高波动', 'bull', 30),
        ('熊市+高波动', 'bear', 30),
        ('震荡市+低波动', 'normal', 12),
    ]
    
    for desc, regime, vix in test_cases:
        threshold = model.get_dynamic_threshold(
            market_regime=regime,
            vix_level=vix,
            base_threshold=0.55
        )
        vix_str = f", VIX={vix}" if vix else ""
        print(f"    {desc}: 阈值 = {threshold:.2f}{vix_str}")

print("\n\n2. 类别权重配置验证")
print("-"*70)

for config in configs:
    model = CatBoostModel(
        class_weight=config['class_weight'],
        use_dynamic_threshold=config['use_dynamic_threshold']
    )
    print(f"\n{config['name']}:")
    print(f"  class_weight: {model.class_weight}")
    print(f"  use_dynamic_threshold: {model.use_dynamic_threshold}")

print("\n\n3. 使用示例代码")
print("-"*70)
print("""
# 方案A：温和类别权重 + 固定阈值（推荐入门）
model_a = CatBoostModel(class_weight='balanced', use_dynamic_threshold=False)
model_a.train(stock_list, horizon=20)
# 预测时始终使用 0.55 阈值

# 方案B：温和类别权重 + 动态阈值（推荐进阶）
model_b = CatBoostModel(class_weight='balanced', use_dynamic_threshold=True)
model_b.train(stock_list, horizon=20)
# 预测时根据市场环境自动调整阈值
threshold = model_b.get_dynamic_threshold(market_regime='bull')  # 0.52

# 方案C：强类别权重 + 动态阈值（激进）
model_c = CatBoostModel(class_weight={0: 1.0, 1: 1.3}, use_dynamic_threshold=True)
model_c.train(stock_list, horizon=20)
# 模型更关注上涨样本，配合动态阈值使用
""")

print("\n" + "="*70)
print("✅ 动态阈值功能验证完成")
print("="*70)
print("\n建议下一步：")
print("  1. 使用 test_class_weight_dynamic_threshold.py 进行完整回测对比")
print("  2. 选择 2-3 只代表性股票进行快速验证")
print("  3. 根据结果选择最佳方案应用于生产环境")
