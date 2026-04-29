#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost 超参数调优脚本（改进版）

改进点：
1. 使用更多股票和fold，避免过拟合
2. 使用 Walk-forward 验证而非简单 TimeSeriesSplit
3. 对最优参数运行完整验证确认效果
4. 添加早停机制，避免无效搜索

使用方法：
  # 标准调优（推荐）
  python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30

  # 快速测试
  python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 10 --quick

  # 完整调优（耗时较长）
  python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 50 --full
"""

import warnings
import os
import sys
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 导入项目模块
from ml_services.ml_trading_model import CatBoostModel, FeatureEngineer
from ml_services.logger_config import get_logger
from config import TRAINING_STOCKS as STOCK_LIST

logger = get_logger('hyperparameter_tuner')


# ==================== 搜索空间定义 ====================

SEARCH_SPACE = {
    'n_estimators': [400, 500, 600, 800],
    'depth': [5, 6, 7, 8],
    'learning_rate': [0.03, 0.04, 0.05, 0.06],
    'l2_leaf_reg': [1, 2, 3, 5],
    'subsample': [0.6, 0.7, 0.75, 0.8],
    'colsample_bylevel': [0.6, 0.7, 0.75, 0.8]
}

# 当前最优参数（基准）
BASELINE_PARAMS = {
    'n_estimators': 600,
    'depth': 7,
    'learning_rate': 0.03,
    'l2_leaf_reg': 2,
    'subsample': 0.75,
    'colsample_bylevel': 0.75
}


class HyperparameterTuner:
    """超参数调优器（改进版）"""

    def __init__(
        self,
        horizon: int = 20,
        n_folds: int = 6,
        quick_mode: bool = False,
        full_mode: bool = False
    ):
        """
        初始化调优器

        Args:
            horizon: 预测周期（天）
            n_folds: 验证fold数量
            quick_mode: 快速模式（更少股票和fold）
            full_mode: 完整模式（使用全部股票）
        """
        self.horizon = horizon
        self.n_folds = n_folds
        self.quick_mode = quick_mode
        self.full_mode = full_mode

        # 根据模式配置
        if quick_mode:
            self.n_folds = 4
            self.n_stocks = 20
            self.val_ratio = 0.2  # 验证集比例
        elif full_mode:
            self.n_folds = 8
            self.n_stocks = 59  # 全部股票
            self.val_ratio = 0.15
        else:
            self.n_folds = 6
            self.n_stocks = 40
            self.val_ratio = 0.2

        logger.info(f"初始化超参数调优器")
        logger.info(f"预测周期: {horizon}天")
        logger.info(f"验证fold数: {self.n_folds}")
        logger.info(f"股票数量: {self.n_stocks}")
        logger.info(f"模式: {'快速' if quick_mode else '完整' if full_mode else '标准'}")

    def evaluate_params(self, params: dict, stock_list: list) -> dict:
        """
        评估单组参数（使用 Walk-forward 风格验证）

        Args:
            params: 参数字典
            stock_list: 股票列表

        Returns:
            dict: 评估结果
        """
        # 创建模型实例
        model = CatBoostModel()

        # 选择股票子集（固定随机种子确保可重现）
        rng = random.Random(42)
        selected_stocks = rng.sample(stock_list, min(self.n_stocks, len(stock_list)))

        try:
            # 准备数据
            train_data = model.prepare_data(
                selected_stocks,
                horizon=self.horizon,
                for_backtest=False
            )

            if len(train_data) < 200:
                return {'score': -999, 'accuracy': 0, 'sharpe': 0, 'error': '样本不足'}

            # 获取特征列
            exclude_cols = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                           'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                           'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                           'BB_upper', 'BB_lower', 'BB_middle', 'Low_Min', 'High_Max',
                           '+DM', '-DM', '+DI', '-DI', 'TP', 'MF_Multiplier', 'MF_Volume']
            feature_cols = [c for c in train_data.columns if c not in exclude_cols
                           and train_data[c].dtype in ['float64', 'float32', 'int64', 'int32']]

            if len(feature_cols) == 0:
                return {'score': -999, 'accuracy': 0, 'sharpe': 0, 'error': '无特征'}

            # 使用 Walk-forward 风格验证
            total_samples = len(train_data)
            fold_size = total_samples // (self.n_folds + 1)

            fold_results = []

            for fold in range(self.n_folds):
                # 训练集：从开始到当前fold
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = min(val_start + fold_size, total_samples)

                if val_end <= val_start:
                    continue

                train_df = train_data.iloc[:train_end]
                val_df = train_data.iloc[val_start:val_end]

                # 检查标签多样性
                if train_df['Label'].nunique() < 2 or val_df['Label'].nunique() < 2:
                    continue

                X_train = train_df[feature_cols].values
                y_train = train_df['Label'].values
                X_val = val_df[feature_cols].values
                y_val = val_df['Label'].values

                # 处理NaN
                X_train = np.nan_to_num(X_train, nan=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0)

                # 训练CatBoost
                from catboost import CatBoostClassifier
                catboost_params = {
                    'loss_function': 'Logloss',
                    'eval_metric': 'Accuracy',
                    'depth': params['depth'],
                    'learning_rate': params['learning_rate'],
                    'iterations': params['n_estimators'],
                    'l2_leaf_reg': params['l2_leaf_reg'],
                    'subsample': params['subsample'],
                    'colsample_bylevel': params['colsample_bylevel'],
                    'random_seed': 42,
                    'verbose': 0,
                    'early_stopping_rounds': 50,
                    'allow_writing_files': False
                }

                clf = CatBoostClassifier(**catboost_params)
                clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

                # 预测
                y_pred = clf.predict(X_val)

                # 计算指标
                accuracy = (y_pred == y_val).mean()

                # 计算夏普比率
                if 'Future_Return' in val_df.columns:
                    returns = val_df['Future_Return'].values
                    strategy_returns = returns * (2 * y_pred - 1)
                    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                    else:
                        sharpe = 0
                else:
                    sharpe = 0

                fold_results.append({
                    'accuracy': accuracy,
                    'sharpe': sharpe
                })

            if not fold_results:
                return {'score': -999, 'accuracy': 0, 'sharpe': 0, 'error': '无有效fold'}

            # 计算平均指标
            avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
            avg_sharpe = np.mean([r['sharpe'] for r in fold_results])
            std_accuracy = np.std([r['accuracy'] for r in fold_results])
            std_sharpe = np.std([r['sharpe'] for r in fold_results])

            # 综合得分：准确率 + 夏普比率 - 稳定性惩罚
            # 更重视准确率，同时考虑夏普比率的稳定性
            score = (
                avg_accuracy * 100 +  # 准确率权重
                avg_sharpe * 5 -       # 夏普比率权重（降低）
                std_accuracy * 30 -    # 准确率稳定性惩罚
                std_sharpe * 2         # 夏普稳定性惩罚
            )

            return {
                'score': score,
                'accuracy': avg_accuracy,
                'sharpe': avg_sharpe,
                'std_accuracy': std_accuracy,
                'std_sharpe': std_sharpe,
                'n_folds': len(fold_results)
            }

        except Exception as e:
            logger.error(f"参数评估失败: {e}")
            return {'score': -999, 'accuracy': 0, 'sharpe': 0, 'error': str(e)}

    def random_search(self, n_iter: int = 30, stock_list: list = None) -> dict:
        """
        随机搜索

        Args:
            n_iter: 迭代次数
            stock_list: 股票列表

        Returns:
            dict: 最优参数和结果
        """
        print("\n" + "="*80)
        print("🎲 随机搜索超参数优化（改进版）")
        print("="*80)
        print(f"迭代次数: {n_iter}")
        print(f"股票数量: {self.n_stocks}")
        print(f"验证fold数: {self.n_folds}")
        total_combinations = (
            len(SEARCH_SPACE['n_estimators']) *
            len(SEARCH_SPACE['depth']) *
            len(SEARCH_SPACE['learning_rate']) *
            len(SEARCH_SPACE['l2_leaf_reg']) *
            len(SEARCH_SPACE['subsample']) *
            len(SEARCH_SPACE['colsample_bylevel'])
        )
        print(f"搜索空间: {total_combinations} 种组合")
        print("="*80)

        if stock_list is None:
            stock_list = list(STOCK_LIST.keys())

        best_score = -float('inf')
        best_params = None
        best_result = None
        all_results = []

        start_time = time.time()
        no_improve_count = 0
        early_stop_threshold = max(5, n_iter // 4)  # 早停阈值

        for i in range(n_iter):
            # 随机采样参数
            params = {
                'n_estimators': random.choice(SEARCH_SPACE['n_estimators']),
                'depth': random.choice(SEARCH_SPACE['depth']),
                'learning_rate': random.choice(SEARCH_SPACE['learning_rate']),
                'l2_leaf_reg': random.choice(SEARCH_SPACE['l2_leaf_reg']),
                'subsample': random.choice(SEARCH_SPACE['subsample']),
                'colsample_bylevel': random.choice(SEARCH_SPACE['colsample_bylevel'])
            }

            print(f"\n[{i+1}/{n_iter}] 评估参数: {params}")

            # 评估参数
            result = self.evaluate_params(params, stock_list)

            if 'error' in result and result.get('error'):
                print(f"  ⚠️ 评估失败: {result['error']}")
                continue

            print(f"  结果: 得分={result['score']:.2f}, 准确率={result['accuracy']:.2%}, 夏普={result['sharpe']:.4f}")
            print(f"        稳定性: 准确率std={result['std_accuracy']:.4f}, 夏普std={result['std_sharpe']:.4f}")

            # 记录结果
            all_results.append({
                'params': params,
                'result': result
            })

            # 更新最优
            if result['score'] > best_score:
                best_score = result['score']
                best_params = params
                best_result = result
                print(f"  ✅ 新最优!")
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 早停检查
            if no_improve_count >= early_stop_threshold and i > n_iter // 2:
                print(f"\n⏹️ 早停：连续 {no_improve_count} 次无改进")
                break

        elapsed_time = time.time() - start_time

        print("\n" + "="*80)
        print("🏆 随机搜索完成")
        print("="*80)
        print(f"耗时: {elapsed_time:.1f}秒")
        print(f"评估次数: {len(all_results)}")
        print(f"最优参数: {best_params}")
        print(f"最优结果: 得分={best_result['score']:.2f}, 准确率={best_result['accuracy']:.2%}, 夏普={best_result['sharpe']:.4f}")
        print("="*80)

        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': all_results,
            'elapsed_time': elapsed_time
        }

    def final_validation(self, params: dict, stock_list: list = None) -> dict:
        """
        对最优参数运行完整验证（使用全部股票和更多fold）

        Args:
            params: 参数字典
            stock_list: 股票列表

        Returns:
            dict: 验证结果
        """
        print("\n" + "="*80)
        print("🔬 最终验证（完整模式）")
        print("="*80)
        print(f"使用全部 {len(stock_list) if stock_list else 59} 只股票")
        print(f"验证fold数: 8")
        print("="*80)

        # 临时设置为完整模式
        original_n_stocks = self.n_stocks
        original_n_folds = self.n_folds
        self.n_stocks = 59
        self.n_folds = 8

        result = self.evaluate_params(params, stock_list or list(STOCK_LIST.keys()))

        # 恢复原始设置
        self.n_stocks = original_n_stocks
        self.n_folds = original_n_folds

        print(f"\n最终验证结果:")
        print(f"  准确率: {result['accuracy']:.2%}")
        print(f"  夏普比率: {result['sharpe']:.4f}")
        print(f"  稳定性: 准确率std={result['std_accuracy']:.4f}")
        print("="*80)

        return result

    def save_results(self, results: dict, output_dir: str = 'output'):
        """保存调优结果"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存最优参数
        best_params_path = os.path.join(output_dir, f'best_params_{timestamp}.json')
        with open(best_params_path, 'w') as f:
            json.dump(results['best_params'], f, indent=2)
        print(f"✅ 最优参数已保存: {best_params_path}")

        # 保存详细结果
        results_path = os.path.join(output_dir, f'tuning_results_{timestamp}.json')
        serializable_results = {
            'best_params': results['best_params'],
            'best_result': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in results['best_result'].items()},
            'elapsed_time': results['elapsed_time'],
            'config': {
                'n_stocks': self.n_stocks,
                'n_folds': self.n_folds,
                'horizon': self.horizon
            }
        }
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✅ 详细结果已保存: {results_path}")

        return best_params_path


def main():
    parser = argparse.ArgumentParser(description='CatBoost 超参数调优（改进版）')
    parser.add_argument('--n-iter', type=int, default=30,
                        help='迭代次数 (默认: 30)')
    parser.add_argument('--horizon', type=int, default=20,
                        choices=[1, 5, 20],
                        help='预测周期 (默认: 20)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（更少股票和fold）')
    parser.add_argument('--full', action='store_true',
                        help='完整模式（使用全部股票）')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录 (默认: output)')
    parser.add_argument('--skip-final', action='store_true',
                        help='跳过最终验证')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 CatBoost 超参数调优（改进版）")
    print("="*80)
    print(f"迭代次数: {args.n_iter}")
    print(f"预测周期: {args.horizon}天")
    print(f"模式: {'快速' if args.quick else '完整' if args.full else '标准'}")
    print(f"输出目录: {args.output_dir}")
    print("="*80)

    # 创建调优器
    tuner = HyperparameterTuner(
        horizon=args.horizon,
        quick_mode=args.quick,
        full_mode=args.full
    )

    # 执行调优
    results = tuner.random_search(n_iter=args.n_iter)

    # 最终验证
    if not args.skip_final:
        final_result = tuner.final_validation(results['best_params'])
        results['final_result'] = final_result

    # 保存结果
    tuner.save_results(results, args.output_dir)

    # 打印对比
    print("\n" + "="*80)
    print("📊 参数对比")
    print("="*80)
    print(f"{'参数':<20} {'基准值':<15} {'优化值':<15} {'变化':<10}")
    print("-"*60)
    for param in BASELINE_PARAMS.keys():
        baseline = BASELINE_PARAMS[param]
        optimized = results['best_params'][param]
        if isinstance(baseline, (int, float)):
            change = f"{((optimized - baseline) / baseline * 100):+.1f}%"
        else:
            change = "-"
        print(f"{param:<20} {baseline:<15} {optimized:<15} {change:<10}")
    print("="*80)

    # 最终验证对比
    if 'final_result' in results:
        print("\n📊 最终验证 vs 调优评估")
        print("-"*60)
        print(f"{'指标':<15} {'调优评估':<15} {'最终验证':<15}")
        print(f"{'准确率':<15} {results['best_result']['accuracy']:.2%}          {results['final_result']['accuracy']:.2%}")
        print(f"{'夏普比率':<15} {results['best_result']['sharpe']:.4f}          {results['final_result']['sharpe']:.4f}")
        print("="*80)

    print("\n💡 下一步:")
    print(f"1. 如果最终验证结果满意，更新 ml_trading_model.py 中的参数配置")
    print(f"2. 运行完整 Walk-forward 验证确认:")
    print(f"   python3 ml_services/walk_forward_validation.py --model-type catboost --horizon {args.horizon}")


if __name__ == "__main__":
    main()
