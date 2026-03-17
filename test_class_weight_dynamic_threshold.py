#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别权重 + 动态阈值方案对比测试

对比三种方案：
1. 方案A：class_weight='balanced' + 固定阈值 0.55
2. 方案B：class_weight='balanced' + 动态阈值
3. 方案C：class_weight={0: 1.0, 1: 1.3} + 动态阈值

测试目标：提高买入胜率（解决准确率与胜率背离问题）
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_services.ml_trading_model import CatBoostModel
from ml_services.backtest_evaluator import BacktestEvaluator
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

logger = get_logger('test_class_weight')

# 选择测试股票（代表性样本：消费股、银行股、科技股各一只）
TEST_STOCKS = ['00151.HK', '0939.HK', '0700.HK']

# 回测参数
HORIZON = 20
TEST_START = '2024-01-01'
TEST_END = '2025-06-30'
TRAIN_START = '2022-01-01'
TRAIN_END = '2023-12-31'


class ModelConfig:
    """模型配置类"""
    def __init__(self, name: str, class_weight, use_dynamic_threshold: bool, 
                 base_threshold: float = 0.55):
        self.name = name
        self.class_weight = class_weight
        self.use_dynamic_threshold = use_dynamic_threshold
        self.base_threshold = base_threshold


def prepare_data(codes: List[str], start_date: str, end_date: str, horizon: int):
    """准备数据（复用 CatBoostModel 的数据准备逻辑）"""
    # 使用一个临时模型来准备数据
    temp_model = CatBoostModel()
    df = temp_model.prepare_data(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        for_backtest=True
    )
    return df


def train_model(config: ModelConfig, train_df: pd.DataFrame) -> CatBoostModel:
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"训练模型: {config.name}")
    print(f"  class_weight: {config.class_weight}")
    print(f"  use_dynamic_threshold: {config.use_dynamic_threshold}")
    print(f"{'='*70}")
    
    model = CatBoostModel(
        class_weight=config.class_weight,
        use_dynamic_threshold=config.use_dynamic_threshold
    )
    
    # 获取特征列
    feature_columns = model.get_feature_columns(train_df)
    model.feature_columns = feature_columns
    
    # 准备训练数据
    X_train = train_df[feature_columns]
    y_train = train_df['Label']
    
    print(f"训练数据: {len(X_train)} 条")
    print(f"类别分布: 上涨={sum(y_train==1)}, 下跌={sum(y_train==0)}")
    
    # 这里简化处理，直接调用内部训练逻辑
    # 实际训练中应该调用 model.train()，但为了快速测试，我们使用简化版
    model.train(
        codes=TEST_STOCKS,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        horizon=HORIZON
    )
    
    return model


def backtest_single_model(model: CatBoostModel, config: ModelConfig, 
                         test_df: pd.DataFrame) -> Dict:
    """对单只股票进行回测"""
    evaluator = BacktestEvaluator(initial_capital=100000)
    
    results = []
    
    for stock_code in test_df['Code'].unique():
        single_stock_df = test_df[test_df['Code'] == stock_code].sort_index()
        
        if len(single_stock_df) < 50:
            continue
            
        X_test = single_stock_df[model.feature_columns]
        y_test = single_stock_df['Label']
        prices = single_stock_df['Close']
        
        # 确定使用的阈值
        if config.use_dynamic_threshold and hasattr(model, 'get_dynamic_threshold'):
            # 简化的市场环境判断（基于近期收益率）
            recent_return = prices.pct_change(20).iloc[-1] if len(prices) > 20 else 0
            if recent_return > 0.05:
                market_regime = 'bull'
            elif recent_return < -0.05:
                market_regime = 'bear'
            else:
                market_regime = 'normal'
            
            threshold = model.get_dynamic_threshold(
                market_regime=market_regime,
                base_threshold=config.base_threshold
            )
        else:
            threshold = config.base_threshold
        
        print(f"  {stock_code}: 使用阈值 {threshold}")
        
        try:
            result = evaluator.backtest_model(
                model=model,
                test_data=X_test,
                test_labels=pd.Series(y_test.values, index=X_test.index),
                test_prices=prices,
                confidence_threshold=threshold
            )
            result['stock_code'] = stock_code
            result['threshold_used'] = threshold
            results.append(result)
        except Exception as e:
            logger.error(f"回测 {stock_code} 失败: {e}")
            continue
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """计算汇总指标"""
    if not results:
        return {}
    
    # 计算平均买入胜率
    win_rates = [r['win_rate'] for r in results if 'win_rate' in r]
    avg_win_rate = np.mean(win_rates) if win_rates else 0
    
    # 计算准确率（需要重新计算，backtest_evaluator 输出中没有直接提供）
    # 准确率 = 预测正确的交易数 / 总交易机会数
    # 这里我们用买入胜率作为代理
    
    # 计算总收益率
    returns = [r['total_return'] for r in results if 'total_return' in r]
    avg_return = np.mean(returns) if returns else 0
    
    # 计算夏普比率
    sharpe_ratios = [r['sharpe_ratio'] for r in results if 'sharpe_ratio' in r]
    avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
    
    return {
        'avg_win_rate': avg_win_rate,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'num_stocks': len(results)
    }


def print_comparison_table(all_results: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("📊 方案对比结果")
    print("="*80)
    print(f"{'方案':<30} {'买入胜率':<15} {'总收益率':<15} {'夏普比率':<15}")
    print("-"*80)
    
    for name, metrics in all_results.items():
        if metrics:
            print(f"{name:<30} {metrics['avg_win_rate']*100:>6.2f}%        "
                  f"{metrics['avg_return']*100:>6.2f}%        "
                  f"{metrics['avg_sharpe']:>6.2f}")
        else:
            print(f"{name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*80)
    
    # 找出最佳方案
    best_win_rate = max(all_results.items(), key=lambda x: x[1].get('avg_win_rate', 0) if x[1] else 0)
    best_return = max(all_results.items(), key=lambda x: x[1].get('avg_return', 0) if x[1] else 0)
    
    print(f"\n🏆 买入胜率最高: {best_win_rate[0]} ({best_win_rate[1]['avg_win_rate']*100:.2f}%)")
    print(f"🏆 总收益率最高: {best_return[0]} ({best_return[1]['avg_return']*100:.2f}%)")


def main():
    """主函数"""
    print("\n" + "#"*80)
    print("# 类别权重 + 动态阈值方案对比测试")
    print("#"*80)
    print(f"测试股票: {', '.join(TEST_STOCKS)}")
    print(f"预测周期: {HORIZON} 天")
    print(f"训练期: {TRAIN_START} 至 {TRAIN_END}")
    print(f"测试期: {TEST_START} 至 {TEST_END}")
    
    # 准备数据
    print("\n" + "="*70)
    print("准备数据...")
    print("="*70)
    
    # 准备训练数据
    print("\n准备训练数据...")
    train_df = prepare_data(TEST_STOCKS, TRAIN_START, TRAIN_END, HORIZON)
    print(f"训练数据: {len(train_df)} 条记录")
    
    # 准备测试数据
    print("\n准备测试数据...")
    test_df = prepare_data(TEST_STOCKS, TEST_START, TEST_END, HORIZON)
    print(f"测试数据: {len(test_df)} 条记录")
    
    # 定义三种方案
    configs = [
        ModelConfig(
            name="A: Balanced + Fixed(0.55)",
            class_weight='balanced',
            use_dynamic_threshold=False,
            base_threshold=0.55
        ),
        ModelConfig(
            name="B: Balanced + Dynamic",
            class_weight='balanced',
            use_dynamic_threshold=True,
            base_threshold=0.55
        ),
        ModelConfig(
            name="C: Strong(1.3x) + Dynamic",
            class_weight={0: 1.0, 1: 1.3},
            use_dynamic_threshold=True,
            base_threshold=0.55
        ),
    ]
    
    # 测试每种方案
    all_results = {}
    
    for config in configs:
        print(f"\n{'#'*70}")
        print(f"# 测试方案: {config.name}")
        print(f"{'#'*70}")
        
        # 训练模型
        model = train_model(config, train_df)
        
        # 回测
        results = backtest_single_model(model, config, test_df)
        
        # 计算指标
        metrics = calculate_metrics(results)
        all_results[config.name] = metrics
        
        print(f"\n方案 {config.name} 结果:")
        print(f"  平均买入胜率: {metrics['avg_win_rate']*100:.2f}%")
        print(f"  平均总收益率: {metrics['avg_return']*100:.2f}%")
        print(f"  平均夏普比率: {metrics['avg_sharpe']:.2f}")
        print(f"  测试股票数: {metrics['num_stocks']}")
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'output/class_weight_test_results_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
