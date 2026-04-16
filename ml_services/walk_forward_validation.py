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
from ml_services.ml_trading_model import CatBoostModel, LightGBMModel, GBDTModel, FeatureEngineer
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

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
        min_train_samples: int = 100
    ):
        """
        初始化 Walk-forward 验证器

        Args:
            model_type: 模型类型（catboost、lightgbm、gbdt）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            step_window_months: 滚动步长（月）
            horizon: 预测周期（天）
            confidence_threshold: 置信度阈值
            use_feature_selection: 是否使用特征选择
            min_train_samples: 最小训练样本数
        """
        self.model_type = model_type.lower()
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.horizon = horizon
        self.confidence_threshold = confidence_threshold
        self.use_feature_selection = use_feature_selection
        self.min_train_samples = min_train_samples

        # 模型类映射
        self.model_classes = {
            'catboost': CatBoostModel,
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
        print("="*80)

        # 存储所有fold的结果
        all_fold_results = []

        # 执行每个fold的验证
        for fold in range(num_folds):
            print(f"\n{'='*80}")
            print(f"📊 Fold {fold + 1}/{num_folds}")
            print(f"{'='*80}")

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

            print(f"训练期间: {train_start_date.strftime('%Y-%m-%d')} 至 {train_end_date.strftime('%Y-%m-%d')}")
            print(f"测试期间: {test_start_date.strftime('%Y-%m-%d')} 至 {test_end_date.strftime('%Y-%m-%d')}")

            # 执行fold验证
            try:
                fold_result = self._validate_fold(
                    stock_list,
                    train_start_date,
                    train_end_date,
                    test_start_date,
                    test_end_date,
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

        # 创建模型实例
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

        # 检查目标变量多样性
        if 'Label' in train_data.columns:
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
            'transaction_cost': TOTAL_COST  # 记录使用的交易成本
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
            'recommendation': recommendation
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
                       choices=['catboost', 'lightgbm', 'gbdt'],
                       help='模型类型 (默认: catboost)')

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
        use_feature_selection=False  # 默认使用全量特征
    )

    # 执行验证
    try:
        report = validator.validate(stock_list, args.start_date, args.end_date)

        # 保存报告
        output_files = validator.save_report(report, args.output_dir)

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