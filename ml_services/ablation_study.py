#!/usr/bin/env python3
"""
消融实验：找出导致 IC 变差的 Regime Shift 修复方案

实验设计：
1. 基线：关闭所有修复
2. 仅开启单调约束
3. 仅开启时间衰减
4. 仅开启滚动百分位
5. 全部开启（当前状态）

对比 IC 指标找出问题源
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.walk_forward_validation import WalkForwardValidator
from config import WATCHLIST
import json
from datetime import datetime

# WATCHLIST 是字典，需要提取 keys
STOCK_LIST = list(WATCHLIST.keys())

def run_experiment(name, use_monotone, time_decay, use_rolling_percentile):
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"单调约束={use_monotone}, 时间衰减={time_decay}, 滚动百分位={use_rolling_percentile}")
    print(f"{'='*60}")

    validator = WalkForwardValidator(
        model_type='catboost',
        train_window_months=12,
        test_window_months=1,
        step_window_months=1,
        horizon=20,
        confidence_threshold=0.55,
        use_feature_selection=False,  # 关闭特征选择，减少变量
        use_monotone_constraints=use_monotone,
        time_decay_lambda=time_decay,
        use_rolling_percentile=use_rolling_percentile
    )

    results = validator.validate(
        stock_list=STOCK_LIST[:10],  # 使用10只股票加速实验
        start_date='2024-01-01',
        end_date='2025-12-31'  # 需要足够长的时间范围（至少13个月）
    )

    # 提取关键指标（从 overall_metrics 子字典中获取）
    overall = results.get('overall_metrics', {})
    metrics = {
        'name': name,
        'config': {
            'use_monotone_constraints': use_monotone,
            'time_decay_lambda': time_decay,
            'use_rolling_percentile': use_rolling_percentile
        },
        'avg_accuracy': overall.get('avg_accuracy', 0),
        'avg_sharpe_ratio': overall.get('avg_sharpe_ratio', 0),
        'avg_max_drawdown': overall.get('avg_max_drawdown', 0),
        'avg_ic': overall.get('avg_ic', 0),
        'avg_rank_ic': overall.get('avg_rank_ic', 0),
        'avg_prediction_std': overall.get('avg_prediction_std', 0),
        'overall_score': overall.get('overall_score', 0)
    }

    print(f"\n结果:")
    print(f"  准确率: {metrics['avg_accuracy']:.2%}")
    print(f"  夏普比率: {metrics['avg_sharpe_ratio']:.4f}")
    print(f"  IC: {metrics['avg_ic']:.4f}")
    print(f"  Rank IC: {metrics['avg_rank_ic']:.4f}")
    print(f"  预测分散度: {metrics['avg_prediction_std']:.4f}")

    return metrics

def main():
    """运行消融实验"""
    print("="*60)
    print("消融实验：找出导致 IC 变差的 Regime Shift 修复方案")
    print("="*60)

    experiments = [
        # (名称, 单调约束, 时间衰减, 滚动百分位)
        ("基线（无修复）", False, 0.0, False),
        ("仅单调约束", True, 0.0, False),
        ("仅时间衰减(λ=0.5)", False, 0.5, False),
        ("仅滚动百分位", False, 0.0, True),
    ]

    # 注：全部修复的结果已有（IC=-0.0201），无需重复运行

    results = []
    for name, use_monotone, time_decay, use_rolling_percentile in experiments:
        try:
            metrics = run_experiment(name, use_monotone, time_decay, use_rolling_percentile)
            results.append(metrics)
        except Exception as e:
            print(f"实验 '{name}' 失败: {e}")
            results.append({
                'name': name,
                'error': str(e)
            })

    # 输出对比表
    print("\n" + "="*60)
    print("消融实验结果对比")
    print("="*60)
    print(f"| 实验 | 准确率 | 夏普比率 | IC | Rank IC | 预测分散度 |")
    print(f"|------|--------|----------|-----|---------|------------|")
    for r in results:
        if 'error' not in r:
            print(f"| {r['name']} | {r['avg_accuracy']:.2%} | {r['avg_sharpe_ratio']:.4f} | {r['avg_ic']:.4f} | {r['avg_rank_ic']:.4f} | {r['avg_prediction_std']:.4f} |")
        else:
            print(f"| {r['name']} | 错误: {r['error'][:30]} |")

    # 找出 IC 最差的配置
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        baseline_ic = valid_results[0]['avg_ic']
        print(f"\n分析:")
        print(f"基线 IC: {baseline_ic:.4f}")

        for r in valid_results[1:]:
            ic_change = r['avg_ic'] - baseline_ic
            if ic_change < -0.001:
                print(f"⚠️ {r['name']}: IC 下降 {abs(ic_change):.4f}")
            elif ic_change > 0.001:
                print(f"✅ {r['name']}: IC 提升 {ic_change:.4f}")
            else:
                print(f"➖ {r['name']}: IC 基本不变")

    # 保存结果
    output_file = f"output/ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
