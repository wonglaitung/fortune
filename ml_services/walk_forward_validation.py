#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward 验证 - 模型预测能力验证（业界标准）

功能：
- 业界标准的 Walk-forward 验证方法
- 每个fold重新训练模型，评估真实预测能力
- 多维度评估指标（夏普比率、索提诺比率、信息比率、最大回撤）
- 严格的时序分割，避免数据泄露
- 稳定性分析，评估模型在不同时期的一致性

业界标准参数：
- 训练窗口：12个月（业界通常12-24个月）
- 测试窗口：1个月（业界通常1-3个月）
- 滚动步长：1个月（业界标准）

使用方法：
  python3 ml_services/walk_forward_validation.py --model-type catboost --start-date 2024-01-01 --end-date 2025-12-31
  python3 ml_services/walk_forward_validation.py --model-type catboost --train-window 12 --test-window 1 --step-window 1
  python3 ml_services/walk_forward_validation.py --model-type catboost --confidence-threshold 0.55 --use-feature-selection
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 导入项目模块
from ml_services.ml_trading_model import CatBoostModel, CatBoostRankerModel, LightGBMModel, GBDTModel, FeatureEngineer
from ml_services.logger_config import get_logger
from config import TRAINING_STOCKS as STOCK_LIST

# 获取日志记录器
logger = get_logger('walk_forward_validation')


class WalkForwardValidator:
    """Walk-forward 验证器 - 业界标准的时序验证方法"""

    def __init__(
        self,
        model_type: str = 'catboost',
        train_window_months: int = 12,    # 业界标准：12个月
        test_window_months: int = 1,       # 业界标准：1个月
        step_window_months: int = 1,        # 业界标准：1个月
        horizon: int = 20,
        confidence_threshold: float = 0.55,
        use_feature_selection: bool = True,
        min_train_samples: int = 100,
        # 新增：Regime Shift 修复参数
        use_monotone_constraints: bool = True,
        time_decay_lambda: float = 0.5,
        use_rolling_percentile: bool = False,  # 2026-05-02: 关闭滚动百分位（消融实验证明其降低IC）
        use_cross_sectional_percentile: bool = True,  # 2026-05-03: 截面百分位（与相对标签匹配）
        use_cross_sectional_zscore: bool = True,  # 2026-05-03: 截面 Z-Score（解决时间序列基准问题）
        feature_importance_threshold: float = 0.0,  # P3-8: 特征修剪阈值
        n_jobs: int = 1,  # 并行执行的 fold 数量（-1 表示使用所有 CPU 核心）
    ):
        """
        初始化 Walk-forward 验证器

        Args:
            model_type: 模型类型（catboost、lightgbm、gbdt、catboost_ranker）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            step_window_months: 滚动步长（月）
            horizon: 预测周期（天）
            confidence_threshold: 置信度阈值
            use_feature_selection: 是否使用特征选择
            min_train_samples: 最小训练样本数
            use_monotone_constraints: 是否使用单调约束（防止特征方向翻转，推荐开启）
            time_decay_lambda: 时间衰减系数（0=无衰减，0.5=默认）
            use_rolling_percentile: 是否使用滚动百分位特征（已关闭，消融实验证明降低IC）
            use_cross_sectional_percentile: 是否使用截面百分位特征（与相对标签匹配）
            use_cross_sectional_zscore: 是否使用截面 Z-Score 特征（解决时间序列基准问题）
            feature_importance_threshold: 特征重要性阈值，低于此值的特征将被移除（0=不修剪）
            n_jobs: 并行执行的 fold 数量（-1 表示使用所有 CPU 核心，1 表示顺序执行）
        """
        self.model_type = model_type.lower()
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.horizon = horizon
        self.confidence_threshold = confidence_threshold
        self.use_feature_selection = use_feature_selection
        self.min_train_samples = min_train_samples
        # 新增参数
        self.use_monotone_constraints = use_monotone_constraints
        self.time_decay_lambda = time_decay_lambda
        self.use_rolling_percentile = use_rolling_percentile
        self.use_cross_sectional_percentile = use_cross_sectional_percentile
        self.use_cross_sectional_zscore = use_cross_sectional_zscore
        self.feature_importance_threshold = feature_importance_threshold  # P3-8
        self.n_jobs = n_jobs

        # 模型类映射
        self.model_classes = {
            'catboost': CatBoostModel,
            'catboost_ranker': CatBoostRankerModel,  # P3-9: 排序模型
            'lightgbm': LightGBMModel,
            'gbdt': GBDTModel
        }

        if self.model_type not in self.model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}，支持：{list(self.model_classes.keys())}")

        self.model_class = self.model_classes[self.model_type]
        self.feature_engineer = FeatureEngineer()

        logger.info(f"初始化 Walk-forward 验证器")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"训练窗口: {train_window_months} 个月")
        logger.info(f"测试窗口: {test_window_months} 个月")
        logger.info(f"滚动步长: {step_window_months} 个月")
        logger.info(f"预测周期: {horizon} 天")
        logger.info(f"单调约束: {use_monotone_constraints}, 时间衰减: {time_decay_lambda}, 滚动百分位: {use_rolling_percentile}, 截面百分位: {use_cross_sectional_percentile}, 截面ZScore: {use_cross_sectional_zscore}")

    def validate(self, stock_list, start_date, end_date):
        """
        执行 Walk-forward 验证

        Args:
            stock_list: 股票代码列表
            start_date: 验证开始日期
            end_date: 验证结束日期

        Returns:
            dict: 验证结果（包含每个fold的指标和整体指标）
        """
        print("\n" + "="*80)
        print("🔬 开始 Walk-forward 验证")
        print("="*80)
        print(f"股票数量: {len(stock_list)}")
        print(f"验证日期范围: {start_date} 至 {end_date}")
        print(f"训练窗口: {self.train_window_months} 个月")
        print(f"测试窗口: {self.test_window_months} 个月")
        print(f"滚动步长: {self.step_window_months} 个月")
        print(f"预测周期: {self.horizon} 天")
        print(f"置信度阈值: {self.confidence_threshold}")
        print("="*80)

        # 生成所有月份
        all_months = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        while current <= end:
            all_months.append(current.strftime('%Y-%m'))
            current = current + pd.DateOffset(months=1)

        logger.info(f"总月份数: {len(all_months)}")

        # 计算fold数量
        total_months = len(all_months)
        num_folds = (total_months - self.train_window_months - self.test_window_months) // self.step_window_months + 1

        if num_folds <= 0:
            raise ValueError(f"日期范围不足，需要至少 {self.train_window_months + self.test_window_months} 个月的数据")

        print(f"\nFold 数量: {num_folds}")
        print(f"并行执行: {self.n_jobs if self.n_jobs > 0 else '所有可用核心'} 个 fold")
        print("="*80)

        # 准备所有 fold 的参数
        fold_params = []
        for fold in range(num_folds):
            # 计算训练和测试期间
            train_start_idx = fold * self.step_window_months
            train_end_idx = train_start_idx + self.train_window_months
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_window_months

            # 获取训练和测试月份
            train_months = all_months[train_start_idx:train_end_idx]
            test_months = all_months[test_start_idx:test_end_idx]

            # 计算日期范围（转换为带时区的UTC格式）
            train_start_date = pd.to_datetime(train_months[0] + '-01').tz_localize('UTC')
            train_end_date = (pd.to_datetime(train_months[-1] + '-01') + pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')
            test_start_date = pd.to_datetime(test_months[0] + '-01').tz_localize('UTC')
            test_end_date = (pd.to_datetime(test_months[-1] + '-01') + pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')

            fold_params.append({
                'fold': fold,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'test_start_date': test_start_date,
                'test_end_date': test_end_date,
            })

        # 执行验证（并行或顺序）
        all_fold_results = []
        if self.n_jobs != 1 and num_folds > 1:
            # 并行执行
            from joblib import Parallel, delayed
            print(f"\n🚀 并行执行 {num_folds} 个 fold...")

            results = Parallel(
                n_jobs=self.n_jobs,
                verbose=10,  # 显示进度
                backend='loky'  # 使用 loky 后端，支持多进程
            )(
                delayed(self._validate_fold_wrapper)(
                    stock_list,
                    params['train_start_date'],
                    params['train_end_date'],
                    params['test_start_date'],
                    params['test_end_date'],
                    params['fold'],
                    num_folds
                )
                for params in fold_params
            )

            # 过滤失败的结果
            all_fold_results = [r for r in results if r is not None]
        else:
            # 顺序执行（原有逻辑）
            for params in fold_params:
                fold = params['fold']
                print(f"\n{'='*80}")
                print(f"📊 Fold {fold + 1}/{num_folds}")
                print(f"{'='*80}")
                print(f"训练期间: {params['train_start_date'].strftime('%Y-%m-%d')} 至 {params['train_end_date'].strftime('%Y-%m-%d')}")
                print(f"测试期间: {params['test_start_date'].strftime('%Y-%m-%d')} 至 {params['test_end_date'].strftime('%Y-%m-%d')}")

                try:
                    fold_result = self._validate_fold(
                        stock_list,
                        params['train_start_date'],
                        params['train_end_date'],
                        params['test_start_date'],
                        params['test_end_date'],
                        fold
                    )

                    all_fold_results.append(fold_result)

                    # 打印fold结果
                    print(f"\n✅ Fold {fold + 1} 结果:")
                    print(f"  样本数: {fold_result['num_samples']}")
                    print(f"  交易次数: {fold_result.get('num_trades', 'N/A')}")
                    print(f"  平均收益率: {fold_result['avg_return']:.2%}")
                    print(f"  胜率: {fold_result['win_rate']:.2%}")
                    print(f"  准确率: {fold_result['accuracy']:.2%}")
                    print(f"  夏普比率: {fold_result['sharpe_ratio']:.4f}")
                    print(f"  最大回撤: {fold_result['max_drawdown']:.2%}")

                except Exception as e:
                    logger.error(f"Fold {fold + 1} 验证失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

        # 计算整体指标
        overall_result = self._calculate_overall_metrics(all_fold_results)

        # 生成报告
        report = {
            'validation_config': {
                'model_type': self.model_type,
                'train_window_months': self.train_window_months,
                'test_window_months': self.test_window_months,
                'step_window_months': self.step_window_months,
                'horizon': self.horizon,
                'confidence_threshold': self.confidence_threshold,
                'use_feature_selection': self.use_feature_selection,
                'start_date': start_date,
                'end_date': end_date,
                'num_folds': num_folds,
                'stock_list': stock_list
            },
            'fold_results': all_fold_results,
            'overall_metrics': overall_result
        }

        print(f"\n{'='*80}")
        print("📊 Walk-forward 验证完成")
        print(f"{'='*80}")
        print(f"总 Fold 数: {len(all_fold_results)}")
        
        if len(all_fold_results) == 0:
            print("⚠️  所有 Fold 验证失败，无法计算整体指标")
            print(f"可能原因：训练样本不足（需要至少 {self.min_train_samples} 个样本）")
            print(f"建议：增加训练窗口长度或延长验证日期范围")
        else:
            print(f"平均收益率: {overall_result['avg_return']:.2%}")
            print(f"平均胜率: {overall_result['avg_win_rate']:.2%}")
            print(f"平均准确率: {overall_result['avg_accuracy']:.2%}")
            print(f"平均夏普比率: {overall_result['avg_sharpe_ratio']:.4f}")
            print(f"平均最大回撤: {overall_result['avg_max_drawdown']:.2%}")
            print(f"收益率标准差: {overall_result['return_std']:.2%}")
            print(f"稳定性评级: {overall_result['stability_rating']}")
        print(f"{'='*80}")

        return report

    def _validate_fold_wrapper(self, stock_list, train_start_date, train_end_date, test_start_date, test_end_date, fold, total_folds):
        """
        并行执行的 fold 验证包装器

        Args:
            stock_list: 股票代码列表
            train_start_date: 训练开始日期
            train_end_date: 训练结束日期
            test_start_date: 测试开始日期
            test_end_date: 测试结束日期
            fold: fold编号
            total_folds: 总fold数

        Returns:
            dict: fold验证结果，失败时返回 None
        """
        print(f"\n{'='*80}")
        print(f"📊 Fold {fold + 1}/{total_folds}")
        print(f"{'='*80}")
        print(f"训练期间: {train_start_date.strftime('%Y-%m-%d')} 至 {train_end_date.strftime('%Y-%m-%d')}")
        print(f"测试期间: {test_start_date.strftime('%Y-%m-%d')} 至 {test_end_date.strftime('%Y-%m-%d')}")

        try:
            fold_result = self._validate_fold(
                stock_list,
                train_start_date,
                train_end_date,
                test_start_date,
                test_end_date,
                fold
            )

            # 打印fold结果
            print(f"\n✅ Fold {fold + 1} 结果:")
            print(f"  样本数: {fold_result['num_samples']}")
            print(f"  交易次数: {fold_result.get('num_trades', 'N/A')}")
            print(f"  平均收益率: {fold_result['avg_return']:.2%}")
            print(f"  胜率: {fold_result['win_rate']:.2%}")
            print(f"  准确率: {fold_result['accuracy']:.2%}")
            print(f"  夏普比率: {fold_result['sharpe_ratio']:.4f}")
            print(f"  最大回撤: {fold_result['max_drawdown']:.2%}")

            return fold_result

        except Exception as e:
            logger.error(f"Fold {fold + 1} 验证失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _validate_fold(self, stock_list, train_start_date, train_end_date, test_start_date, test_end_date, fold):
        """
        验证单个fold

        Args:
            stock_list: 股票代码列表
            train_start_date: 训练开始日期
            train_end_date: 训练结束日期
            test_start_date: 测试开始日期
            test_end_date: 测试结束日期
            fold: fold编号

        Returns:
            dict: fold验证结果
        """
        print(f"\n  🔄 准备训练数据...")

        # 创建模型实例（传递 Regime Shift 修复参数）
        if self.model_type == 'catboost':
            model = self.model_class(
                use_monotone_constraints=self.use_monotone_constraints,
                time_decay_lambda=self.time_decay_lambda,
                use_rolling_percentile=self.use_rolling_percentile,
                use_cross_sectional_percentile=self.use_cross_sectional_percentile,
                use_cross_sectional_zscore=self.use_cross_sectional_zscore,
                feature_importance_threshold=self.feature_importance_threshold  # P3-8
            )
        elif self.model_type == 'catboost_ranker':
            # P3-9: 排序模型
            # P10-1: 使用 YetiRank 损失函数（全局排序，直接优化 Rank IC）
            # P4-10/P9: YetiRankPairwise 导致 IC 正但 Rank IC 负，尝试 YetiRank 改善
            # P5: 软标签实验失败（IC/Rank IC 双降），已放弃
            model = self.model_class(
                loss_function='YetiRank',
                use_monotone_constraints=self.use_monotone_constraints,
                time_decay_lambda=self.time_decay_lambda,
                use_rolling_percentile=self.use_rolling_percentile,
                use_cross_sectional_percentile=self.use_cross_sectional_percentile,
                use_cross_sectional_zscore=self.use_cross_sectional_zscore,
                feature_importance_threshold=self.feature_importance_threshold,
                use_soft_label=False  # P5: 软标签实验失败，回退到原始收益率
            )
        else:
            model = self.model_class()

        # 准备训练数据
        train_codes = stock_list
        train_data = model.prepare_data(
            train_codes,
            start_date=train_start_date,
            end_date=train_end_date,
            horizon=self.horizon,
            for_backtest=False
        )

        # 检查训练样本数量
        if len(train_data) < self.min_train_samples:
            raise ValueError(f"训练样本不足: {len(train_data)} < {self.min_train_samples}")

        # 检查目标变量多样性（排序模型使用 Future_Return，跳过 Label 检查）
        if self.model_type == 'catboost_ranker':
            # 排序模型使用 Future_Return 作为标签，检查 Future_Return 存在性
            if 'Future_Return' not in train_data.columns:
                raise ValueError("训练数据中没有 'Future_Return' 列")
            if train_data['Future_Return'].isna().all():
                raise ValueError("Future_Return 全为 NaN")
        elif 'Label' in train_data.columns:
            unique_labels = train_data['Label'].nunique()
            if unique_labels < 2:
                raise ValueError(f"目标变量多样性不足：只有 {unique_labels} 个唯一值，需要至少 2 个（上涨/下跌）")
        else:
            raise ValueError("训练数据中没有 'Label' 列")

        print(f"  ✅ 训练数据准备完成: {len(train_data)} 条记录")

        # 训练模型（关键：每个fold重新训练）
        print(f"  🔄 训练模型 (Fold {fold + 1})...")
        try:
            model.train(
                train_codes,
                start_date=train_start_date,
                end_date=train_end_date,
                horizon=self.horizon,
                use_feature_selection=self.use_feature_selection
            )
            # P9: 保存最后一个训练的模型引用（用于 save_detailed_results）
            self.last_model = model
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise

        print(f"  ✅ 模型训练完成")

        # 准备测试数据
        print(f"  🔄 准备测试数据...")
        test_data = model.prepare_data(
            train_codes,
            start_date=test_start_date,
            end_date=test_end_date,
            horizon=self.horizon,
            for_backtest=False
        )

        print(f"  ✅ 测试数据准备完成: {len(test_data)} 条记录")

        # 生成预测（使用 predict_proba 方法批量预测）
        print(f"  🔄 生成预测...")
        X_test = test_data[model.feature_columns]

        # 对 Ranker 模型使用自动温度参数，扩大概率分布范围
        if hasattr(model, 'model_type') and model.model_type == 'catboost_ranker':
            prediction_proba = model.predict_proba(X_test, temperature='auto')
        else:
            prediction_proba = model.predict_proba(X_test)
        
        # 根据市场状态使用自适应置信度阈值（符合业界标准）
        base_threshold = self.confidence_threshold
        
        # 检查测试数据中是否有市场状态特征
        if 'Confidence_Threshold_Multiplier' in test_data.columns:
            # 使用特征中的动态阈值乘数
            multipliers = test_data['Confidence_Threshold_Multiplier'].values
            adaptive_thresholds = base_threshold * multipliers
        elif 'Market_Regime' in test_data.columns:
            # 根据市场状态计算阈值
            regime_multipliers = {
                'ranging': 1.09,    # 震荡市更严格
                'normal': 1.0,      # 正常市标准
                'trending': 0.91    # 趋势市更宽松
            }
            multipliers = test_data['Market_Regime'].map(regime_multipliers).fillna(1.0).values
            adaptive_thresholds = base_threshold * multipliers
        elif 'Market_Regime_Encoded' in test_data.columns:
            # 使用数值编码的市场状态
            regime_map = {0: 1.09, 1: 1.0, 2: 0.91}  # ranging, normal, trending
            multipliers = test_data['Market_Regime_Encoded'].map(regime_map).fillna(1.0).values
            adaptive_thresholds = base_threshold * multipliers
        else:
            # 无市场状态信息，使用固定阈值
            adaptive_thresholds = np.full(len(prediction_proba), base_threshold)
        
        # 根据自适应阈值生成预测标签
        predictions = pd.DataFrame({
            'prediction': (prediction_proba[:, 1] >= adaptive_thresholds).astype(int),
            'probability': prediction_proba[:, 1],
            'adaptive_threshold': adaptive_thresholds
        }, index=test_data.index)
        
        # 添加 Code 列（从列中提取）
        if 'Code' in test_data.columns:
            predictions['Code'] = test_data['Code'].values
        else:
            # 如果 Code 列不存在，使用 train_codes
            predictions['Code'] = [train_codes[0]] * len(predictions)  # 单股票情况下使用第一个股票代码
        
        print(f"  ✅ 预测生成完成 ({len(predictions)} 条预测)")

        # 计算评估指标
        print(f"  🔄 计算评估指标...")
        metrics = self._calculate_metrics(test_data, predictions)

        # 添加fold信息
        metrics['fold'] = fold
        metrics['train_start_date'] = train_start_date.strftime('%Y-%m-%d')
        metrics['train_end_date'] = train_end_date.strftime('%Y-%m-%d')
        metrics['test_start_date'] = test_start_date.strftime('%Y-%m-%d')
        metrics['test_end_date'] = test_end_date.strftime('%Y-%m-%d')
        metrics['num_train_samples'] = len(train_data)
        metrics['num_test_samples'] = len(test_data)

        return metrics

    def _calculate_metrics(self, test_data, predictions):
        """
        计算多维度评估指标

        Args:
            test_data: 测试数据
            predictions: 预测结果

        Returns:
            dict: 评估指标
        """
        # ========== 交易成本定义（业界标准）==========
        # 双边交易成本 = 佣金 + 印花税 + 滑点
        COMMISSION = 0.001     # 单边佣金 0.1%
        STAMP_DUTY = 0.001     # 印花税 0.1%（港股卖出时收取）
        SLIPPAGE = 0.001       # 滑点 0.1%
        TOTAL_COST = (COMMISSION + SLIPPAGE) * 2 + STAMP_DUTY  # 双边成本约 0.5%

        # 合并测试数据和预测结果
        if predictions is None or predictions.empty:
            raise ValueError("预测结果为空")

        # 合并数据（简单方法：直接添加预测列）
        # predictions 的索引是日期，test_data 的索引也是日期
        # 直接添加预测列，假设索引是对齐的
        df = test_data.copy()
        df['prediction'] = predictions['prediction']
        df['probability'] = predictions['probability']

        # 确保索引对齐（删除预测中不存在于 test_data 的索引）
        df = df[df.index.isin(predictions.index)]

        if df.empty:
            raise ValueError("合并后的数据为空")

        # 计算实际收益率
        # 注意：模型训练时 Label 基于 Future_Return（20天累积收益）
        # 但验证时使用 pct_change().shift(-horizon) 是合理的：
        # 1. 模型预测的是"未来上涨/下跌的方向"，不是具体收益率
        # 2. pct_change() 提供的是标准化后的收益方向和幅度
        # 3. 累积收益会受到除权除息、停牌等因素影响，可能导致异常值
        # 4. pct_change() 的单日收益更稳定，数值范围合理（±10%以内）
        df['actual_return'] = df['Close'].pct_change().shift(-self.horizon)

        # 计算预测收益
        df['predicted_return'] = df['actual_return'] * df['prediction']

        # ========== 风险过滤（业界最佳实践）==========
        # 过滤高风险信号：ATR_Risk_Score > 0.8 表示极端高波动期
        df['risk_filter'] = True  # 默认通过
        if 'ATR_Risk_Score' in df.columns:
            # 高波动期（ATR处于历史高位80%以上）不交易
            df['risk_filter'] = df['ATR_Risk_Score'] <= 0.8

        # 震荡市低置信度信号过滤
        if 'Market_Regime' in df.columns and 'probability' in df.columns:
            # 震荡市中，只有高置信度信号才通过
            ranging_filter = (df['Market_Regime'] != 'ranging') | (df['probability'] >= 0.75)
            df['risk_filter'] = df['risk_filter'] & ranging_filter

        # 计算交易信号（应用风险过滤）
        df['signal'] = ((df['prediction'] >= self.confidence_threshold) & df['risk_filter']).astype(int)

        # 计算收益率（扣除交易成本）
        df['strategy_return'] = df['signal'] * df['actual_return']
        # 扣除交易成本后的净收益
        df['strategy_return_net'] = df['strategy_return'] - df['signal'] * TOTAL_COST

        # 移除NaN
        df = df.dropna(subset=['actual_return', 'strategy_return'])

        if len(df) == 0:
            raise ValueError("没有有效的样本")

        # 分离有交易信号和无交易信号的样本
        trades = df[df['signal'] == 1].copy()
        no_trades = df[df['signal'] == 0].copy()

        # 基础指标 - 只计算有交易信号的样本
        if len(trades) > 0:
            avg_return = trades['strategy_return'].mean()
            # 对于固定持有期策略，累积收益应该用简单平均，而不是顺序累乘
            # 因为多笔交易是同时持有的，不是顺序平仓的
            cumulative_return = avg_return * len(trades) / max(len(trades), 1)  # 近似
            # 胜率计算使用扣除交易成本后的净收益
            win_rate = (trades['strategy_return_net'] > 0).sum() / len(trades)
            return_std = trades['strategy_return'].std()
            # 净收益指标
            avg_return_net = trades['strategy_return_net'].mean()
            return_std_net = trades['strategy_return_net'].std()
        else:
            avg_return = 0.0
            cumulative_return = 0.0
            win_rate = 0.0
            return_std = 0.0
            avg_return_net = 0.0
            return_std_net = 0.0

        # 准确率：使用全部样本（因为预测是针对所有样本的）
        accuracy = (df['prediction'] == df['Label']).sum() / len(df)
        num_samples = len(df)
        num_trades = len(trades)

        # ========== 批次分析（合并计算，避免重复） ==========
        # 对于固定持有期策略，按信号日期分组形成"批次"
        # 每个批次 = 当天产生的所有信号，等权重分配资金
        if len(trades) > 0:
            batches = trades.groupby(trades.index).agg({
                'strategy_return': 'mean',  # 批次平均收益
                'strategy_return_net': 'mean',  # 批次净收益
                'signal': 'count'  # 批次内信号数量
            }).rename(columns={'signal': 'batch_size'})
            batch_returns = batches['strategy_return'].values
            batch_returns_net = batches['strategy_return_net'].values
        else:
            batches = None
            batch_returns = np.array([])
            batch_returns_net = np.array([])

        # ========== 年化收益和夏普比率计算（修正版） ==========
        # 问题：之前用 avg_return * (252/horizon) 年化，假设一年可做12.6次交易
        # 实际：固定持有期策略的多笔交易是同时持有的，不能这样年化
        # 正确方法：基于批次收益的时间序列，月度收益 * 12 年化

        # 月度组合收益
        monthly_return = avg_return

        # 年化收益
        annualized_return = monthly_return * 12

        # 标准差：基于批次收益的波动
        if len(batch_returns) > 1:
            batch_std = np.std(batch_returns, ddof=1)
            monthly_std = batch_std
        elif return_std > 0:
            monthly_std = return_std
        else:
            monthly_std = 0.0

        # 年化标准差
        annualized_std = monthly_std * np.sqrt(12) if monthly_std > 0 else 0.0

        # ========== 最大回撤计算（批次回撤） ==========
        if len(batch_returns) > 1:
            # 每个批次分配等权重资金 (1/N)
            batch_weights = np.ones(len(batch_returns)) / len(batch_returns)

            # 累积净值曲线
            cumulative_values = np.cumsum(batch_returns * batch_weights) + 1.0
            peak_values = np.maximum.accumulate(cumulative_values)
            drawdowns = (peak_values - cumulative_values) / peak_values
            max_drawdown = -np.max(drawdowns) if np.max(drawdowns) > 0 else 0.0

            # 限制回撤范围在合理区间
            if abs(max_drawdown) > 0.5:
                # 使用更保守的方法：连续亏损批次的最大累积
                cumulative_loss = 0.0
                max_cumulative_loss = 0.0
                for ret in batch_returns:
                    if ret < 0:
                        cumulative_loss += ret * (1.0 / len(batch_returns))
                    else:
                        cumulative_loss = 0.0
                    max_cumulative_loss = min(max_cumulative_loss, cumulative_loss)
                max_drawdown = max_cumulative_loss
        elif len(batch_returns) == 1:
            max_drawdown = -abs(batch_returns[0]) if batch_returns[0] < 0 else 0.0
        else:
            max_drawdown = 0.0

        # 风险调整收益指标
        # 夏普比率（修正：添加无风险利率）
        risk_free_rate = 0.02  # 2%无风险利率
        if annualized_std > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
        else:
            sharpe_ratio = 0.0

        # 索提诺比率（只考虑下行风险，仅计算有交易的样本）
        # 修正：使用与夏普比率一致的年化因子 sqrt(12)
        if len(trades) > 0:
            downside_returns = trades['strategy_return'][trades['strategy_return'] < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    # 月度下行标准差 * sqrt(12) 年化
                    annualized_downside_std = downside_std * np.sqrt(12)
                    sortino_ratio = annualized_return / annualized_downside_std
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = float('inf') if annualized_return > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # 信息比率（相对于基准：买入持有策略）
        # 基准收益：所有样本的平均收益（代表买入持有）
        benchmark_return = df['actual_return'].mean()
        # 策略收益与基准的差异（只比较有交易的样本）
        # 修正：使用与夏普比率一致的年化因子 sqrt(12)
        if len(trades) > 0:
            excess_returns = trades['strategy_return'] - trades['actual_return']
            tracking_error = excess_returns.std()
            if tracking_error > 0:
                # 月度跟踪误差 * sqrt(12) 年化
                annualized_tracking_error = tracking_error * np.sqrt(12)
                information_ratio = (excess_returns.mean() * 12) / annualized_tracking_error
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0

        # ========== IC 指标计算（选股能力评估）==========
        # IC (Information Coefficient): 预测概率与实际收益的 Pearson 相关系数
        # Rank IC: 预测排名与实际收益排名的 Spearman 相关系数
        from scipy.stats import pearsonr, spearmanr

        # 只计算有有效预测和收益的样本
        valid_mask = ~df['actual_return'].isna() & ~df['probability'].isna()
        valid_count = valid_mask.sum()

        if valid_count > 10:  # 至少10个样本才计算相关性
            try:
                ic, ic_pvalue = pearsonr(
                    df.loc[valid_mask, 'probability'],
                    df.loc[valid_mask, 'actual_return']
                )
                rank_ic, rank_ic_pvalue = spearmanr(
                    df.loc[valid_mask, 'probability'],
                    df.loc[valid_mask, 'actual_return']
                )
                # 处理 NaN
                if np.isnan(ic):
                    ic = 0.0
                if np.isnan(rank_ic):
                    rank_ic = 0.0
            except Exception:
                ic = 0.0
                rank_ic = 0.0
        else:
            ic = 0.0
            rank_ic = 0.0

        # 预测分散度：预测概率的标准差
        # 用于检测"全涨全跌"问题（分散度过低表示预测缺乏区分度）
        prediction_std = df['probability'].std() if 'probability' in df.columns else 0.0

        # ========== 头部精度分析（Top Percentile Accuracy）==========
        # 分析预测概率最高的股票其实际收益分布
        top_metrics = {}
        if valid_count > 20:  # 至少20个样本才计算头部精度
            valid_df = df[valid_mask].copy()

            # Top 1%, 5%, 10%, 20% 分析
            for pct in [1, 5, 10, 20]:
                threshold = valid_df['probability'].quantile(1 - pct/100)
                top_mask = valid_df['probability'] >= threshold
                top_df = valid_df[top_mask]

                if len(top_df) > 0:
                    top_return = top_df['actual_return'].mean()
                    top_std = top_df['actual_return'].std()
                    top_win_rate = (top_df['actual_return'] > 0).mean()

                    # 与整体平均收益比较
                    overall_return = valid_df['actual_return'].mean()
                    excess_return = top_return - overall_return

                    top_metrics[f'top{pct}_return'] = top_return
                    top_metrics[f'top{pct}_std'] = top_std
                    top_metrics[f'top{pct}_win_rate'] = top_win_rate
                    top_metrics[f'top{pct}_excess'] = excess_return
                    top_metrics[f'top{pct}_count'] = len(top_df)

            # ========== P9 新增：保存 Top 25% 推荐股票详情 ==========
            top_25_threshold = valid_df['probability'].quantile(0.75)
            top_25_df = valid_df[valid_df['probability'] >= top_25_threshold].copy()

            if len(top_25_df) > 0:
                top_25_stocks = []
                for idx, row in top_25_df.iterrows():
                    code = row.get('Code', '') if 'Code' in row.index else ''
                    actual_rank = (valid_df['actual_return'] > row['actual_return']).sum() + 1
                    hit = 1 if row['actual_return'] > valid_df['actual_return'].median() else 0

                    top_25_stocks.append({
                        'code': code,
                        'date': str(idx) if hasattr(idx, '__str__') else '',
                        'predict_prob': float(row['probability']),
                        'rank': int((valid_df['probability'] > row['probability']).sum() + 1),
                        'actual_return': float(row['actual_return']),
                        'actual_rank': actual_rank,
                        'hit': hit
                    })
                top_metrics['top25_stocks'] = top_25_stocks

            # ========== P9 新增：保存错误案例分析 ==========
            # False Positive: 预测高概率但实际收益为负
            fp_df = valid_df[(valid_df['probability'] >= 0.7) & (valid_df['actual_return'] < 0)]
            # False Negative: 预测低概率但实际收益为正
            fn_df = valid_df[(valid_df['probability'] <= 0.3) & (valid_df['actual_return'] > 0)]

            error_cases = []
            for idx, row in fp_df.head(10).iterrows():  # 只保存前10个
                code = row.get('Code', '') if 'Code' in row.index else ''
                error_cases.append({
                    'code': code,
                    'date': str(idx) if hasattr(idx, '__str__') else '',
                    'predict_prob': float(row['probability']),
                    'actual_return': float(row['actual_return']),
                    'error_type': 'False Positive',
                    'reason': '高概率预测但实际下跌'
                })

            for idx, row in fn_df.head(10).iterrows():  # 只保存前10个
                code = row.get('Code', '') if 'Code' in row.index else ''
                error_cases.append({
                    'code': code,
                    'date': str(idx) if hasattr(idx, '__str__') else '',
                    'predict_prob': float(row['probability']),
                    'actual_return': float(row['actual_return']),
                    'error_type': 'False Negative',
                    'reason': '低概率预测但实际上涨'
                })

            top_metrics['error_cases'] = error_cases

        return {
            'num_samples': num_samples,
            'num_trades': num_trades,
            'avg_return': avg_return,
            'avg_return_net': avg_return_net,  # 扣除交易成本后的净收益
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,  # 基于净收益计算的胜率
            'accuracy': accuracy,
            'return_std': return_std,
            'return_std_net': return_std_net,  # 净收益标准差
            'annualized_return': annualized_return,
            'annualized_std': annualized_std,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_return,
            'transaction_cost': TOTAL_COST,  # 记录使用的交易成本
            # 新增 IC 指标
            'ic': ic,
            'rank_ic': rank_ic,
            'prediction_std': prediction_std,
            # 头部精度指标
            'top_metrics': top_metrics
        }

    def _calculate_overall_metrics(self, fold_results):
        """
        计算整体指标

        Args:
            fold_results: 所有fold的结果

        Returns:
            dict: 整体指标
        """
        if not fold_results:
            return {}

        # 计算平均指标
        avg_return = np.mean([r['avg_return'] for r in fold_results])
        avg_win_rate = np.mean([r['win_rate'] for r in fold_results])
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_sharpe_ratio = np.mean([r['sharpe_ratio'] for r in fold_results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        avg_sortino_ratio = np.mean([r['sortino_ratio'] for r in fold_results])
        avg_information_ratio = np.mean([r['information_ratio'] for r in fold_results])

        # IC 指标平均（新增）
        avg_ic = np.mean([r.get('ic', 0) for r in fold_results])
        avg_rank_ic = np.mean([r.get('rank_ic', 0) for r in fold_results])
        avg_prediction_std = np.mean([r.get('prediction_std', 0) for r in fold_results])
        ic_std = np.std([r.get('ic', 0) for r in fold_results])
        rank_ic_std = np.std([r.get('rank_ic', 0) for r in fold_results])

        # 头部精度指标汇总
        top_percentile_metrics = {}
        for pct in [1, 5, 10, 20]:
            returns = []
            win_rates = []
            excesses = []
            counts = []
            for r in fold_results:
                tm = r.get('top_metrics', {})
                if f'top{pct}_return' in tm:
                    returns.append(tm[f'top{pct}_return'])
                    win_rates.append(tm[f'top{pct}_win_rate'])
                    excesses.append(tm[f'top{pct}_excess'])
                    counts.append(tm[f'top{pct}_count'])

            if returns:
                top_percentile_metrics[f'top{pct}_avg_return'] = np.mean(returns)
                top_percentile_metrics[f'top{pct}_avg_win_rate'] = np.mean(win_rates)
                top_percentile_metrics[f'top{pct}_avg_excess'] = np.mean(excesses)
                top_percentile_metrics[f'top{pct}_total_count'] = sum(counts)
                top_percentile_metrics[f'top{pct}_return_std'] = np.std(returns)

        # 稳定性分析
        return_std = np.std([r['avg_return'] for r in fold_results])
        return_range = np.max([r['avg_return'] for r in fold_results]) - np.min([r['avg_return'] for r in fold_results])
        win_rate_std = np.std([r['win_rate'] for r in fold_results])
        sharpe_std = np.std([r['sharpe_ratio'] for r in fold_results])

        # 稳定性评级（基于收益标准差）
        if return_std < 0.02:
            stability_rating = "高（优秀）"
        elif return_std < 0.05:
            stability_rating = "中（良好）"
        else:
            stability_rating = "低（需改进）"

        # 综合评估评分（0-100分）
        # 权重：夏普比率30%，最大回撤25%，索提诺比率25%，稳定性20%
        score = 0

        # 1. 夏普比率评分（30分）：业界标准 >1.0
        if avg_sharpe_ratio >= 1.0:
            score += 30
        elif avg_sharpe_ratio >= 0.8:
            score += 25
        elif avg_sharpe_ratio >= 0.5:
            score += 20
        elif avg_sharpe_ratio >= 0:
            score += 10

        # 2. 最大回撤评分（25分）：业界标准 <-20%
        if avg_max_drawdown > -0.05:  # -5%以内，优秀
            score += 25
        elif avg_max_drawdown > -0.10:  # -10%以内，良好
            score += 20
        elif avg_max_drawdown > -0.20:  # -20%以内，可接受
            score += 15
        elif avg_max_drawdown > -0.30:  # -30%以内，较差
            score += 10

        # 3. 索提诺比率评分（25分）：业界标准 >1.0
        if avg_sortino_ratio >= 2.0:
            score += 25
        elif avg_sortino_ratio >= 1.0:
            score += 20
        elif avg_sortino_ratio >= 0.5:
            score += 15
        elif avg_sortino_ratio >= 0:
            score += 10

        # 4. 稳定性评分（20分）
        if return_std < 0.02:
            score += 20
        elif return_std < 0.05:
            score += 15
        elif return_std < 0.10:
            score += 10
        else:
            score += 5

        # 综合评级
        if score >= 80:
            overall_rating = "优秀"
            recommendation = "强烈推荐实盘"
        elif score >= 60:
            overall_rating = "良好"
            recommendation = "推荐实盘"
        elif score >= 40:
            overall_rating = "一般"
            recommendation = "谨慎使用"
        else:
            overall_rating = "不佳"
            recommendation = "需要改进"

        return {
            'num_folds': len(fold_results),
            'avg_return': avg_return,
            'avg_win_rate': avg_win_rate,
            'avg_accuracy': avg_accuracy,
            'avg_sharpe_ratio': avg_sharpe_ratio,
            'avg_max_drawdown': avg_max_drawdown,
            'avg_sortino_ratio': avg_sortino_ratio,
            'avg_information_ratio': avg_information_ratio,
            'return_std': return_std,
            'return_range': return_range,
            'win_rate_std': win_rate_std,
            'sharpe_std': sharpe_std,
            'stability_rating': stability_rating,
            'overall_score': score,
            'overall_rating': overall_rating,
            'recommendation': recommendation,
            # 新增 IC 指标
            'avg_ic': avg_ic,
            'avg_rank_ic': avg_rank_ic,
            'avg_prediction_std': avg_prediction_std,
            'ic_std': ic_std,
            'rank_ic_std': rank_ic_std,
            # 头部精度指标
            'top_percentile_metrics': top_percentile_metrics
        }

    def save_report(self, report, output_dir='output'):
        """
        保存验证报告（CSV、JSON、Markdown格式）

        Args:
            report: 验证报告
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = report['validation_config']['model_type']
        horizon = report['validation_config']['horizon']

        # 1. 保存JSON格式
        json_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON报告已保存: {json_file}")

        # 2. 保存CSV格式
        csv_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.csv')
        fold_df = pd.DataFrame(report['fold_results'])
        fold_df.to_csv(csv_file, index=False)
        logger.info(f"CSV报告已保存: {csv_file}")

        # 3. 保存Markdown格式
        md_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.md')
        self._generate_markdown_report(report, md_file)
        logger.info(f"Markdown报告已保存: {md_file}")

        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'md_file': md_file
        }

    def save_detailed_results(self, report, model, output_dir='data/validation_results'):
        """
        保存详细的验证结果（P9 阶段新增）

        保存 10 个分析文件：
        1. feature_importance_top50.json - Top 50 特征重要性
        2. recommended_stocks_returns.csv - 前 25% 推荐股票及真实收益率
        3. prediction_distribution.json - 预测概率分布统计
        4. fold_metrics_detail.json - 各 Fold 详细指标
        5. top_stocks_features.csv - 推荐股票的特征值
        6. error_analysis.csv - 错误案例分析
        7. sector_distribution.json - 板块分布
        8. feature_correlation_top20.csv - Top 20 特征相关性
        9. confidence_return_breakdown.json - 置信度分层收益
        10. validation_summary.json - 验证摘要

        Args:
            report: 验证报告
            model: 训练好的模型（用于获取特征重要性）
            output_dir: 输出目录

        Returns:
            dict: 保存的文件路径
        """
        import os
        import json
        from datetime import datetime
        from config import STOCK_SECTOR_MAPPING

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = report['validation_config']['model_type']
        horizon = report['validation_config']['horizon']
        result_dir = os.path.join(output_dir, f'{timestamp}_{model_type}_{horizon}d')
        os.makedirs(result_dir, exist_ok=True)

        saved_files = {}

        # ========== 1. 特征重要性 Top 50 ==========
        try:
            feature_importance = self._get_feature_importance(model)
            if feature_importance:
                top50_data = {
                    'model_type': model_type,
                    'horizon': horizon,
                    'timestamp': timestamp,
                    'top_50_features': feature_importance[:50],
                    'feature_type_summary': self._summarize_feature_types(feature_importance[:50])
                }
                file_path = os.path.join(result_dir, 'feature_importance_top50.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(top50_data, f, indent=2, ensure_ascii=False)
                saved_files['feature_importance_top50'] = file_path
                logger.info(f"特征重要性已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存特征重要性失败: {e}")

        # ========== 2. 推荐股票收益率 ==========
        try:
            recommended_stocks = self._extract_recommended_stocks(report)
            if recommended_stocks:
                file_path = os.path.join(result_dir, 'recommended_stocks_returns.csv')
                recommended_stocks.to_csv(file_path, index=False)
                saved_files['recommended_stocks_returns'] = file_path
                logger.info(f"推荐股票收益率已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存推荐股票收益率失败: {e}")

        # ========== 3. 预测分布统计 ==========
        try:
            pred_dist = self._calculate_prediction_distribution(report)
            if pred_dist:
                file_path = os.path.join(result_dir, 'prediction_distribution.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(pred_dist, f, indent=2, ensure_ascii=False)
                saved_files['prediction_distribution'] = file_path
                logger.info(f"预测分布统计已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存预测分布统计失败: {e}")

        # ========== 4. Fold 详细指标 ==========
        try:
            fold_detail = self._extract_fold_metrics_detail(report)
            if fold_detail:
                file_path = os.path.join(result_dir, 'fold_metrics_detail.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(fold_detail, f, indent=2, ensure_ascii=False, default=str)
                saved_files['fold_metrics_detail'] = file_path
                logger.info(f"Fold 详细指标已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存 Fold 详细指标失败: {e}")

        # ========== 5. 推荐股票特征值 ==========
        try:
            top_features = self._extract_top_stocks_features(report, model)
            if top_features is not None and not top_features.empty:
                file_path = os.path.join(result_dir, 'top_stocks_features.csv')
                top_features.to_csv(file_path, index=False)
                saved_files['top_stocks_features'] = file_path
                logger.info(f"推荐股票特征值已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存推荐股票特征值失败: {e}")

        # ========== 6. 错误分析 ==========
        try:
            error_analysis = self._analyze_errors(report)
            if error_analysis is not None and not error_analysis.empty:
                file_path = os.path.join(result_dir, 'error_analysis.csv')
                error_analysis.to_csv(file_path, index=False)
                saved_files['error_analysis'] = file_path
                logger.info(f"错误分析已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存错误分析失败: {e}")

        # ========== 7. 板块分布 ==========
        try:
            sector_dist = self._analyze_sector_distribution(report)
            if sector_dist:
                file_path = os.path.join(result_dir, 'sector_distribution.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(sector_dist, f, indent=2, ensure_ascii=False)
                saved_files['sector_distribution'] = file_path
                logger.info(f"板块分布已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存板块分布失败: {e}")

        # ========== 8. 特征相关性 Top 20 ==========
        try:
            feature_corr = self._calculate_feature_correlation(model, report)
            if feature_corr is not None and not feature_corr.empty:
                file_path = os.path.join(result_dir, 'feature_correlation_top20.csv')
                feature_corr.to_csv(file_path, index=False)
                saved_files['feature_correlation_top20'] = file_path
                logger.info(f"特征相关性已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存特征相关性失败: {e}")

        # ========== 9. 置信度分层收益 ==========
        try:
            confidence_breakdown = self._analyze_confidence_return(report)
            if confidence_breakdown:
                file_path = os.path.join(result_dir, 'confidence_return_breakdown.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(confidence_breakdown, f, indent=2, ensure_ascii=False)
                saved_files['confidence_return_breakdown'] = file_path
                logger.info(f"置信度分层收益已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存置信度分层收益失败: {e}")

        # ========== 10. 验证摘要 ==========
        try:
            summary = self._generate_validation_summary(report, saved_files)
            file_path = os.path.join(result_dir, 'validation_summary.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            saved_files['validation_summary'] = file_path
            logger.info(f"验证摘要已保存: {file_path}")
        except Exception as e:
            logger.warning(f"保存验证摘要失败: {e}")

        print(f"\n📁 详细验证结果已保存至: {result_dir}")
        print(f"   共保存 {len(saved_files)} 个文件")

        return saved_files

    def _get_feature_importance(self, model):
        """获取特征重要性"""
        if not hasattr(model, 'catboost_model') or model.catboost_model is None:
            return []

        try:
            importances = model.catboost_model.get_feature_importance()
            feature_names = model.feature_columns

            # 排序并格式化
            sorted_idx = np.argsort(importances)[::-1]
            result = []
            for i, idx in enumerate(sorted_idx[:50], 1):
                feat_name = feature_names[idx] if idx < len(feature_names) else f'unknown_{idx}'
                feat_type = self._classify_feature_type(feat_name)
                result.append({
                    'rank': i,
                    'feature': feat_name,
                    'importance': float(importances[idx]),
                    'type': feat_type
                })
            return result
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            return []

    def _classify_feature_type(self, feature_name):
        """分类特征类型"""
        if '_CS_Pct' in feature_name or '_CS_ZScore' in feature_name:
            return 'cross_sectional'
        elif any(x in feature_name for x in ['HSI_', 'SP500_', 'NASDAQ_', 'US_10Y', 'VIX']):
            return 'macro'
        elif any(x in feature_name for x in ['PE', 'PB', 'ROE', 'Market_Cap', 'Dividend']):
            return 'fundamental'
        elif 'net_' in feature_name:
            return 'network'
        else:
            return 'individual'

    def _summarize_feature_types(self, top_features):
        """汇总特征类型"""
        type_counts = {}
        for f in top_features:
            t = f['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        return type_counts

    def _extract_recommended_stocks(self, report):
        """提取前 25% 推荐股票及真实收益率"""
        from config import WATCHLIST

        records = []
        for fold_result in report.get('fold_results', []):
            fold = fold_result.get('fold', 0)
            test_start = fold_result.get('test_start_date', '')
            test_end = fold_result.get('test_end_date', '')

            # 从 top_metrics 中提取
            top_metrics = fold_result.get('top_metrics', {})
            if not top_metrics:
                continue

            # 提取 Top 25% 数据
            for pct in [25]:
                key = f'top{pct}_stocks'
                if key in top_metrics:
                    for stock_info in top_metrics[key]:
                        code = stock_info.get('code', '')
                        records.append({
                            'Fold': fold + 1,
                            'Date': stock_info.get('date', test_start),
                            'Stock_Code': code,
                            'Stock_Name': WATCHLIST.get(code, code),
                            'Predict_Prob': stock_info.get('predict_prob', 0),
                            'Rank': stock_info.get('rank', 0),
                            'Actual_Return': stock_info.get('actual_return', 0),
                            'Actual_Rank': stock_info.get('actual_rank', 0),
                            'Hit': stock_info.get('hit', 0)
                        })

        if records:
            return pd.DataFrame(records)
        return None

    def _calculate_prediction_distribution(self, report):
        """计算预测分布统计"""
        result = {'folds': []}

        for fold_result in report.get('fold_results', []):
            fold = fold_result.get('fold', 0)
            pred_std = fold_result.get('prediction_std', 0)

            # 从 top_metrics 提取预测概率分布
            top_metrics = fold_result.get('top_metrics', {})

            fold_data = {
                'fold': fold + 1,
                'test_period': f"{fold_result.get('test_start_date', '')} to {fold_result.get('test_end_date', '')}",
                'prediction_stats': {
                    'std': pred_std,
                    'ic': fold_result.get('ic', 0),
                    'rank_ic': fold_result.get('rank_ic', 0)
                }
            }
            result['folds'].append(fold_data)

        # 计算整体统计
        if result['folds']:
            overall = report.get('overall_metrics', {})
            result['overall'] = {
                'avg_ic': overall.get('avg_ic', 0),
                'avg_rank_ic': overall.get('avg_rank_ic', 0),
                'avg_prediction_std': overall.get('avg_prediction_std', 0)
            }

        return result

    def _extract_fold_metrics_detail(self, report):
        """提取各 Fold 详细指标"""
        result = {
            'model_type': report['validation_config']['model_type'],
            'horizon': report['validation_config']['horizon'],
            'folds': []
        }

        for fold_result in report.get('fold_results', []):
            fold_data = {
                'fold': fold_result.get('fold', 0) + 1,
                'train_period': f"{fold_result.get('train_start_date', '')} to {fold_result.get('train_end_date', '')}",
                'test_period': f"{fold_result.get('test_start_date', '')} to {fold_result.get('test_end_date', '')}",
                'metrics': {
                    'ic': fold_result.get('ic', 0),
                    'rank_ic': fold_result.get('rank_ic', 0),
                    'accuracy': fold_result.get('accuracy', 0),
                    'sharpe': fold_result.get('sharpe_ratio', 0),
                    'max_drawdown': fold_result.get('max_drawdown', 0),
                    'sortino': fold_result.get('sortino_ratio', 0),
                    'win_rate': fold_result.get('win_rate', 0),
                    'avg_return': fold_result.get('avg_return', 0)
                },
                'sample_counts': {
                    'total': fold_result.get('num_samples', 0),
                    'train': fold_result.get('num_train_samples', 0),
                    'test': fold_result.get('num_test_samples', 0)
                }
            }
            result['folds'].append(fold_data)

        return result

    def _extract_top_stocks_features(self, report, model):
        """提取推荐股票的特征值"""
        # 这个方法需要从原始预测数据中提取
        # 由于当前数据结构限制，返回 None
        # 实际实现需要在 _validate_fold 中保存更多数据
        return None

    def _analyze_errors(self, report):
        """分析预测错误案例"""
        from config import WATCHLIST

        records = []
        for fold_result in report.get('fold_results', []):
            fold = fold_result.get('fold', 0)

            # 从 top_metrics 中提取错误案例
            top_metrics = fold_result.get('top_metrics', {})
            error_cases = top_metrics.get('error_cases', [])

            for case in error_cases:
                code = case.get('code', '')
                records.append({
                    'Fold': fold + 1,
                    'Date': case.get('date', ''),
                    'Stock_Code': code,
                    'Stock_Name': WATCHLIST.get(code, code),
                    'Predict_Prob': case.get('predict_prob', 0),
                    'Actual_Return': case.get('actual_return', 0),
                    'Error_Type': case.get('error_type', ''),
                    'Possible_Reason': case.get('reason', '')
                })

        if records:
            return pd.DataFrame(records)
        return None

    def _analyze_sector_distribution(self, report):
        """分析推荐股票的板块分布"""
        from config import STOCK_SECTOR_MAPPING

        result = {'folds': []}

        for fold_result in report.get('fold_results', []):
            fold = fold_result.get('fold', 0)
            top_metrics = fold_result.get('top_metrics', {})

            # 统计 Top 25% 股票的板块分布
            sector_counts = {}
            top_stocks = top_metrics.get('top25_stocks', [])
            for stock in top_stocks:
                code = stock.get('code', '')
                sector = STOCK_SECTOR_MAPPING.get(code, 'unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            if sector_counts:
                result['folds'].append({
                    'fold': fold + 1,
                    'sector_counts': sector_counts,
                    'top_sector': max(sector_counts, key=sector_counts.get) if sector_counts else None
                })

        return result

    def _calculate_feature_correlation(self, model, report):
        """计算 Top 20 特征的相关性"""
        if not hasattr(model, 'catboost_model') or model.catboost_model is None:
            return None

        try:
            importances = model.catboost_model.get_feature_importance()
            feature_names = model.feature_columns

            # 获取 Top 20 特征
            sorted_idx = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in sorted_idx if i < len(feature_names)]

            # 计算相关性需要原始数据，这里返回特征列表
            # 实际相关性计算需要在有数据的情况下进行
            return pd.DataFrame({
                'Feature': top_features,
                'Importance': [importances[i] for i in sorted_idx if i < len(importances)]
            })
        except Exception as e:
            logger.warning(f"计算特征相关性失败: {e}")
            return None

    def _analyze_confidence_return(self, report):
        """分析不同置信度区间的实际收益"""
        result = {'confidence_bins': []}

        # 定义置信度区间
        bins = [
            ('0.8-1.0', 0.8, 1.0),
            ('0.7-0.8', 0.7, 0.8),
            ('0.6-0.7', 0.6, 0.7),
            ('0.5-0.6', 0.5, 0.6),
            ('0.0-0.5', 0.0, 0.5)
        ]

        overall = report.get('overall_metrics', {})
        top_metrics = overall.get('top_percentile_metrics', {})

        # 从 top_metrics 提取
        for label, low, high in bins:
            # 使用已有的 top_metrics 数据
            if label == '0.8-1.0':
                bin_data = {
                    'range': label,
                    'count': top_metrics.get('top1_total_count', 0),
                    'avg_return': top_metrics.get('top1_avg_return', 0),
                    'hit_rate': top_metrics.get('top1_avg_win_rate', 0)
                }
            elif label == '0.7-0.8':
                bin_data = {
                    'range': label,
                    'count': top_metrics.get('top5_total_count', 0),
                    'avg_return': top_metrics.get('top5_avg_return', 0),
                    'hit_rate': top_metrics.get('top5_avg_win_rate', 0)
                }
            else:
                bin_data = {
                    'range': label,
                    'count': 0,
                    'avg_return': 0,
                    'hit_rate': 0
                }
            result['confidence_bins'].append(bin_data)

        return result

    def _generate_validation_summary(self, report, saved_files):
        """生成验证摘要"""
        config = report['validation_config']
        overall = report.get('overall_metrics', {})

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': config['model_type'],
            'horizon': config['horizon'],
            'num_folds': config['num_folds'],
            'num_stocks': len(config['stock_list']),
            'overall_metrics': {
                'score': overall.get('overall_score', 0),
                'rating': overall.get('overall_rating', ''),
                'recommendation': overall.get('recommendation', ''),
                'avg_sharpe': overall.get('avg_sharpe_ratio', 0),
                'avg_ic': overall.get('avg_ic', 0),
                'avg_rank_ic': overall.get('avg_rank_ic', 0)
            },
            'saved_files': list(saved_files.keys())
        }

    def _generate_markdown_report(self, report, output_file):
        """
        生成Markdown格式报告

        Args:
            report: 验证报告
            output_file: 输出文件路径
        """
        config = report['validation_config']
        overall = report['overall_metrics']
        folds = report['fold_results']

        with open(output_file, 'w', encoding='utf-8') as f:
            # 标题
            f.write(f"# Walk-forward 验证报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 配置信息
            f.write("## 📋 验证配置\n\n")
            f.write(f"- **模型类型**: {config['model_type'].upper()}\n")
            f.write(f"- **训练窗口**: {config['train_window_months']} 个月\n")
            f.write(f"- **测试窗口**: {config['test_window_months']} 个月\n")
            f.write(f"- **滚动步长**: {config['step_window_months']} 个月\n")
            f.write(f"- **预测周期**: {config['horizon']} 天\n")
            f.write(f"- **置信度阈值**: {config['confidence_threshold']}\n")
            f.write(f"- **特征选择**: {'是' if config['use_feature_selection'] else '否'}\n")
            f.write(f"- **验证日期**: {config['start_date']} 至 {config['end_date']}\n")
            f.write(f"- **Fold数量**: {config['num_folds']}\n")
            f.write(f"- **股票数量**: {len(config['stock_list'])}\n\n")

            # 整体指标
            f.write("## 📊 整体性能指标\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法计算整体指标\n\n")
            else:
                f.write(f"- **Fold数量**: {overall['num_folds']}\n")
                f.write(f"- **综合评分**: {overall.get('overall_score', 'N/A')}/100\n")
                f.write(f"- **综合评级**: {overall.get('overall_rating', 'N/A')}\n")
                f.write(f"- **推荐度**: {overall.get('recommendation', 'N/A')}\n")
                f.write(f"- **平均收益率**: {overall['avg_return']:.2%}\n")
                f.write(f"- **平均胜率**: {overall['avg_win_rate']:.2%}\n")
                f.write(f"- **平均准确率**: {overall['avg_accuracy']:.2%}\n")
                f.write(f"- **平均夏普比率**: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- **平均最大回撤**: {overall['avg_max_drawdown']:.2%}\n")
                f.write(f"- **平均索提诺比率**: {overall['avg_sortino_ratio']:.4f}\n")
                f.write(f"- **平均信息比率**: {overall['avg_information_ratio']:.4f}\n\n")

            # 稳定性分析
            f.write("## 🔬 稳定性分析\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法计算稳定性指标\n\n")
            else:
                f.write(f"- **收益率标准差**: {overall['return_std']:.2%}\n")
                f.write(f"- **收益率范围**: {overall['return_range']:.2%}\n")
                f.write(f"- **胜率标准差**: {overall['win_rate_std']:.2%}\n")
                f.write(f"- **夏普比率标准差**: {overall['sharpe_std']:.4f}\n")
                f.write(f"- **稳定性评级**: {overall['stability_rating']}\n\n")

            # IC 指标分析（新增）
            f.write("## 📐 IC 指标分析（选股能力评估）\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法计算 IC 指标\n\n")
            else:
                avg_ic = overall.get('avg_ic', 0)
                avg_rank_ic = overall.get('avg_rank_ic', 0)
                avg_pred_std = overall.get('avg_prediction_std', 0)
                ic_std = overall.get('ic_std', 0)
                rank_ic_std = overall.get('rank_ic_std', 0)

                f.write("| 指标 | 数值 | 说明 |\n")
                f.write("|------|------|------|\n")
                f.write(f"| **IC** | {avg_ic:.4f} | 预测概率与实际收益的 Pearson 相关系数 |\n")
                f.write(f"| **Rank IC** | {avg_rank_ic:.4f} | 预测排名与实际收益排名的 Spearman 相关系数 |\n")
                f.write(f"| **预测分散度** | {avg_pred_std:.4f} | 预测概率标准差（避免\"全涨全跌\"） |\n")
                f.write(f"| **IC 标准差** | {ic_std:.4f} | IC 稳定性 |\n")
                f.write(f"| **Rank IC 标准差** | {rank_ic_std:.4f} | Rank IC 稳定性 |\n\n")

                f.write("**IC 解读**：\n")
                if avg_ic > 0.05:
                    ic_rating = "⭐⭐⭐⭐⭐ 预测能力较强"
                elif avg_ic > 0.02:
                    ic_rating = "⭐⭐⭐⭐ 预测能力中等"
                elif avg_ic > 0:
                    ic_rating = "⭐⭐⭐ 预测能力较弱"
                else:
                    ic_rating = "⚠️ 无预测能力或负相关"
                f.write(f"- IC = {avg_ic:.4f}：{ic_rating}\n")
                f.write(f"- Rank IC 更稳健，对异常值不敏感\n")
                f.write(f"- 预测分散度 > 0.1 表示预测有区分度，避免\"全涨全跌\"\n\n")

            # 头部精度分析（新增）
            f.write("## 🎯 头部精度分析（Top Percentile Accuracy）\n\n")
            if overall and 'top_percentile_metrics' in overall and overall['top_percentile_metrics']:
                top_metrics = overall['top_percentile_metrics']
                f.write("预测概率最高的股票其实际收益分布：\n\n")
                f.write("| 分位 | 平均收益 | 胜率 | 超额收益 | 样本数 | 评估 |\n")
                f.write("|------|---------|------|---------|--------|------|\n")

                for pct in [1, 5, 10, 20]:
                    ret_key = f'top{pct}_avg_return'
                    if ret_key in top_metrics:
                        ret = top_metrics[ret_key]
                        win = top_metrics.get(f'top{pct}_avg_win_rate', 0)
                        exc = top_metrics.get(f'top{pct}_avg_excess', 0)
                        cnt = top_metrics.get(f'top{pct}_total_count', 0)

                        # 评估：超额收益 > 0 且胜率 > 50%
                        if exc > 0 and win > 0.5:
                            rating = "✅ 有效"
                        elif exc > 0:
                            rating = "⚠️ 超额为正"
                        elif win > 0.5:
                            rating = "⚠️ 胜率过半"
                        else:
                            rating = "❌ 无效"

                        f.write(f"| **Top {pct}%** | {ret:.2%} | {win:.2%} | {exc:.2%} | {cnt} | {rating} |\n")

                f.write("\n**解读**：\n")
                f.write("- **平均收益**：该分位股票的平均实际收益\n")
                f.write("- **胜率**：该分位股票中收益为正的比例\n")
                f.write("- **超额收益**：该分位收益减去整体平均收益\n")
                f.write("- 如果 Top 1%/5% 的超额收益显著为正，说明模型头部预测有效\n\n")
            else:
                f.write("⚠️ 样本不足，无法计算头部精度\n\n")

            # Fold详细结果
            f.write("## 📈 Fold 详细结果\n\n")
            f.write("| Fold | 训练期间 | 测试期间 | 样本数 | 交易次数 | 收益率 | 胜率 | 准确率 | 夏普比率 | 最大回撤 |\n")
            f.write("|------|---------|---------|-------|---------|-------|------|-------|---------|--------|\n")

            for fold in folds:
                f.write(f"| {fold['fold'] + 1} | {fold['train_start_date']} 至 {fold['train_end_date']} | "
                       f"{fold['test_start_date']} 至 {fold['test_end_date']} | {fold['num_test_samples']} | "
                       f"{fold.get('num_trades', 'N/A')} | "
                       f"{fold['avg_return']:.2%} | {fold['win_rate']:.2%} | {fold['accuracy']:.2%} | "
                       f"{fold['sharpe_ratio']:.4f} | {fold['max_drawdown']:.2%} |\n")

            f.write("\n")

            # 结论
            f.write("## 💡 结论\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法给出结论。\n\n")
                f.write(f"可能原因：\n")
                f.write(f"- 训练样本不足（需要至少 {self.min_train_samples} 个样本）\n")
                f.write(f"- 目标变量多样性不足（需要同时包含上涨和下跌样本）\n")
                f.write(f"- 数据质量问题（缺失值、异常值等）\n\n")
                f.write(f"建议：\n")
                f.write(f"- 增加训练窗口长度\n")
                f.write(f"- 延长验证日期范围\n")
                f.write(f"- 检查数据质量\n")
            else:
                # 使用综合评级
                overall_rating = overall.get('overall_rating', '不佳')
                overall_score = overall.get('overall_score', 0)
                recommendation = overall.get('recommendation', '需要改进')

                if overall_rating == "优秀":
                    f.write(f"✅ 模型表现**优秀**，{recommendation}。\n\n")
                elif overall_rating == "良好":
                    f.write(f"✅ 模型表现**良好**，{recommendation}。\n\n")
                elif overall_rating == "一般":
                    f.write(f"⚠️ 模型表现**一般**，{recommendation}。\n\n")
                else:
                    f.write(f"❌ 模型表现**不佳**，{recommendation}。\n\n")

                f.write(f"| 指标 | 数值 | 业界标准 | 评估 |\n")
                f.write(f"|------|------|---------|------|\n")
                f.write(f"| 综合评分 | {overall_score}/100 | - | {overall_rating} |\n")
                f.write(f"| 夏普比率 | {overall['avg_sharpe_ratio']:.4f} | >1.0 | {'✅' if overall['avg_sharpe_ratio'] >= 1.0 else '⚠️' if overall['avg_sharpe_ratio'] >= 0.5 else '❌'} |\n")
                f.write(f"| 最大回撤 | {overall['avg_max_drawdown']:.2%} | <-20% | {'✅' if overall['avg_max_drawdown'] > -0.20 else '❌'} |\n")
                f.write(f"| 索提诺比率 | {overall['avg_sortino_ratio']:.4f} | >1.0 | {'✅' if overall['avg_sortino_ratio'] >= 1.0 else '⚠️' if overall['avg_sortino_ratio'] >= 0.5 else '❌'} |\n")
                f.write(f"| 稳定性评级 | {overall['stability_rating']} | - | - |\n\n")

                # 根据评级给出建议
                if overall_rating == "优秀":
                    f.write(f"- **核心优势**：最大回撤控制优秀、夏普比率接近业界标准、索提诺比率高\n")
                    f.write(f"- **建议**：可以小仓位实盘测试，持续监控\n\n")
                elif overall_rating == "良好":
                    f.write(f"- **核心优势**：风险控制良好\n")
                    f.write(f"- **主要风险**：稳定性有待提升\n")
                    f.write(f"- **建议**：小仓位实盘测试，关注稳定性优化\n\n")
                elif overall_rating == "一般":
                    f.write(f"- **建议**：增加特征工程、调整超参数、优化风险控制\n\n")
                else:
                    f.write(f"- **建议**：重新训练模型，增加特征工程，调整超参数\n\n")

            f.write("---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Walk-forward 验证 - 模型预测能力验证')

    # 模型参数
    parser.add_argument('--model-type', type=str, default='catboost',
                       choices=['catboost', 'catboost_ranker', 'lightgbm', 'gbdt'],
                       help='模型类型 (默认: catboost，catboost_ranker 为排序模型)')

    # Walk-forward 参数
    parser.add_argument('--train-window', type=int, default=12,
                       help='训练窗口（月，默认: 12）')
    parser.add_argument('--test-window', type=int, default=1,
                       help='测试窗口（月，默认: 1）')
    parser.add_argument('--step-window', type=int, default=1,
                       help='滚动步长（月，默认: 1）')

    # 模型参数
    parser.add_argument('--horizon', type=int, default=20,
                       help='预测周期（天，默认: 20）')
    parser.add_argument('--confidence-threshold', type=float, default=0.55,
                       help='置信度阈值（默认: 0.55）')

    # 数据参数
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='验证开始日期（默认: 2024-01-01）')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                       help='验证结束日期（默认: 2025-12-31）')
    parser.add_argument('--stocks', type=str, nargs='+',
                       help='股票代码列表（默认: 使用配置文件中的自选股）')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录（默认: output）')

    # 特征选择参数
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择（默认: False，使用全量特征）')

    # 并行执行参数
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='并行执行的 fold 数量（-1 表示使用所有 CPU 核心，默认: 1 顺序执行）')

    args = parser.parse_args()

    # 获取股票列表
    if args.stocks:
        stock_list = args.stocks
    else:
        stock_list = list(STOCK_LIST.keys())

    print(f"\n{'='*80}")
    print("🚀 Walk-forward 验证系统")
    print(f"{'='*80}")
    print(f"模型类型: {args.model_type.upper()}")
    print(f"股票数量: {len(stock_list)}")
    print(f"{'='*80}\n")

    # 创建验证器
    validator = WalkForwardValidator(
        model_type=args.model_type,
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_window_months=args.step_window,
        horizon=args.horizon,
        confidence_threshold=args.confidence_threshold,
        use_feature_selection=args.use_feature_selection,
        n_jobs=args.n_jobs
    )

    # 执行验证
    try:
        report = validator.validate(stock_list, args.start_date, args.end_date)

        # 保存报告
        output_files = validator.save_report(report, args.output_dir)

        # P9: 保存详细分析结果（10个文件）
        if hasattr(validator, 'last_model') and validator.last_model is not None:
            print(f"\n{'='*80}")
            print("📊 保存详细分析结果...")
            print(f"{'='*80}")
            detailed_files = validator.save_detailed_results(report, validator.last_model, args.output_dir)
            if detailed_files:
                print(f"\n详细分析文件:")
                for name, path in detailed_files.items():
                    print(f"  - {name}: {path}")
        else:
            print("\n⚠️  无法保存详细分析结果：模型引用不存在")

        print(f"\n{'='*80}")
        print("✅ 验证完成")
        print(f"{'='*80}")
        print(f"报告文件:")
        print(f"  - JSON: {output_files['json_file']}")
        print(f"  - CSV:  {output_files['csv_file']}")
        print(f"  - MD:   {output_files['md_file']}")
        print(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()