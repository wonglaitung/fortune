#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股 Walk-forward 验证 - 完整版本

复用港股 Walk-forward 架构，调用 AStockTradingModel.train() 完成训练：
- 调用 model.train() 确保训练/预测一致性
- 1000+ 特征（技术指标 + 市场特征 + 网络特征 + 波动率等）
- 市场情绪过滤器
- 样本权重训练（核心股3.0，扩展股1.0）
- 完整评估指标（IC、夏普比率、最大回撤等）

使用方法：
  python3 a_stock_walk_forward.py --horizon 20
  python3 a_stock_walk_forward.py --horizon 20 --train-window 12 --folds 6
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

# 导入A股配置
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_TRAINING_LIST,
    get_limit_rate,
)

# 导入A股模型
from a_stock_ml_model import AStockTradingModel

# 导入日志
from ml_services.logger_config import get_logger
logger = get_logger('a_stock_walk_forward')

# 导入市场情绪过滤器
from ml_services.market_regime import MarketSentimentFilter

# 常量
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_YEAR = 240


class AStockWalkForwardValidator:
    """A股 Walk-forward 验证器 - 与港股架构一致"""

    def __init__(
        self,
        horizon: int = 20,
        train_window_months: int = 12,    # 业界标准：12个月
        test_window_months: int = 1,       # 业界标准：1个月
        step_window_months: int = 1,       # 滚动步长：1个月
        confidence_threshold: float = 0.50,
        use_market_filter: bool = True,    # 启用市场情绪过滤器
        min_train_samples: int = 100,
    ):
        """
        初始化验证器

        Args:
            horizon: 预测周期（天）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            step_window_months: 滚动步长（月）
            confidence_threshold: 置信度阈值
            use_market_filter: 是否使用市场情绪过滤器
            min_train_samples: 最小训练样本数
        """
        self.horizon = horizon
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.confidence_threshold = confidence_threshold
        self.use_market_filter = use_market_filter
        self.min_train_samples = min_train_samples
        self.stock_list = list(A_STOCK_TRAINING_LIST.keys())

        # 市场情绪过滤器（延迟初始化）
        self.market_filter = None

        # 存储所有预测详情（用于生成 prediction_analysis.csv）
        self.all_predictions = []

        logger.info(f"初始化 A股 Walk-forward 验证器")
        logger.info(f"训练窗口: {train_window_months} 个月")
        logger.info(f"测试窗口: {test_window_months} 个月")
        logger.info(f"滚动步长: {step_window_months} 个月")
        logger.info(f"预测周期: {horizon} 天")
        logger.info(f"市场情绪过滤: {'启用' if use_market_filter else '禁用'}")
        logger.info(f"股票数量: {len(self.stock_list)}")

    def validate(self, start_date='2024-01-01', end_date='2026-07-01'):
        """
        执行 Walk-forward 验证

        Args:
            start_date: 验证开始日期
            end_date: 验证结束日期

        Returns:
            dict: 验证结果
        """
        print("\n" + "="*80)
        print("🔬 开始 A股 Walk-forward 验证")
        print("="*80)
        print(f"股票数量: {len(self.stock_list)}")
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

        # 存储所有fold结果
        all_fold_results = []

        # 执行每个fold验证
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
                    self.stock_list,
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
                print(f"  IC: {fold_result.get('ic', 0):.4f}")

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
                'train_window_months': self.train_window_months,
                'test_window_months': self.test_window_months,
                'step_window_months': self.step_window_months,
                'horizon': self.horizon,
                'confidence_threshold': self.confidence_threshold,
                'start_date': start_date,
                'end_date': end_date,
                'num_folds': num_folds,
                'stock_list': self.stock_list,
                'use_market_filter': self.use_market_filter,
            },
            'fold_results': all_fold_results,
            'overall_metrics': overall_result,
        }

        # 打印整体结果
        self._print_overall_results(overall_result)

        # 保存报告
        self._save_report(report)

        return report

    def _validate_fold(self, stock_list, train_start_date, train_end_date, test_start_date, test_end_date, fold):
        """
        验证单个fold - 调用 AStockTradingModel.train() 确保一致性

        Args:
            stock_list: 股票代码列表
            train_start_date: 训练开始日期
            train_end_date: 训练结束日期
            test_start_date: 测试开始日期
            test_end_date: 测试结束日期
            fold: fold编号

        Returns:
            dict: fold结果
        """
        print(f"\n  🔄 准备训练数据...")

        # 创建模型实例
        model = AStockTradingModel(horizon=self.horizon)

        # 准备训练数据（调用模型的 prepare_data 方法）
        train_data = model.prepare_data(
            stock_list,
            start_date=train_start_date,
            end_date=train_end_date,
            horizon=self.horizon,
            mode='backtest'  # 使用滞后数据
        )

        # 检查训练样本数量
        if train_data is None or len(train_data) < self.min_train_samples:
            raise ValueError(f"训练样本不足: {len(train_data) if train_data is not None else 0} < {self.min_train_samples}")

        # 检查目标变量多样性
        if 'Label' in train_data.columns:
            unique_labels = train_data['Label'].nunique()
            if unique_labels < 2:
                raise ValueError(f"目标变量多样性不足：只有 {unique_labels} 个唯一值")
        else:
            raise ValueError("训练数据中没有 'Label' 列")

        print(f"  ✅ 训练数据准备完成: {len(train_data)} 条记录")

        # 训练模型（关键：调用 model.train() 确保与生产一致）
        print(f"  🔄 训练模型 (Fold {fold + 1})...")
        try:
            model.train(
                stock_list,
                start_date=train_start_date,
                end_date=train_end_date,
                horizon=self.horizon,
                use_feature_selection=False,
                use_sample_weights=True
            )
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise

        print(f"  ✅ 模型训练完成")

        # 准备测试数据
        print(f"  🔄 准备测试数据...")
        test_data = model.prepare_data(
            stock_list,
            start_date=test_start_date,
            end_date=test_end_date,
            horizon=self.horizon,
            mode='backtest'
        )

        if test_data is None or len(test_data) < 10:
            raise ValueError(f"测试数据不足: {len(test_data) if test_data is not None else 0}")

        print(f"  ✅ 测试数据准备完成: {len(test_data)} 条记录")

        # 生成预测（使用模型的特征列）
        print(f"  🔄 生成预测...")
        X_test = test_data[model.feature_columns]
        prediction_proba = model.predict_proba(X_test)

        # 使用标准阈值 0.5 生成初始预测
        predictions = pd.DataFrame({
            'prediction': (prediction_proba[:, 1] >= 0.5).astype(int),
            'probability': prediction_proba[:, 1],
        }, index=test_data.index)

        # 添加 Code 列
        if 'Stock_Code' in test_data.columns:
            predictions['Code'] = test_data['Stock_Code'].values

        print(f"  ✅ 预测生成完成 ({len(predictions)} 条预测)")

        # 应用市场情绪过滤器
        if self.use_market_filter:
            predictions = self._apply_market_filter(test_data, predictions, fold)

        # 计算评估指标
        print(f"  🔄 计算评估指标...")
        metrics = self._calculate_metrics(test_data, predictions)

        # 收集预测详情
        if 'prediction_df' in metrics:
            pred_df = metrics['prediction_df'].copy()
            pred_df['fold'] = fold + 1
            pred_df['train_start'] = train_start_date.strftime('%Y-%m-%d')
            pred_df['train_end'] = train_end_date.strftime('%Y-%m-%d')
            pred_df['test_start'] = test_start_date.strftime('%Y-%m-%d')
            pred_df['test_end'] = test_end_date.strftime('%Y-%m-%d')
            self.all_predictions.append(pred_df)

        # 添加fold信息
        metrics['fold'] = fold
        metrics['train_start_date'] = train_start_date.strftime('%Y-%m-%d')
        metrics['train_end_date'] = train_end_date.strftime('%Y-%m-%d')
        metrics['test_start_date'] = test_start_date.strftime('%Y-%m-%d')
        metrics['test_end_date'] = test_end_date.strftime('%Y-%m-%d')
        metrics['num_train_samples'] = len(train_data)
        metrics['num_test_samples'] = len(test_data)

        # 获取特征重要性（Top 20）
        try:
            feat_imp = model.get_feature_importance(top_n=20)
            if feat_imp is not None:
                metrics['top_features'] = feat_imp.to_dict('records')
                print(f"  ✅ 特征重要性已记录")
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            metrics['top_features'] = []

        return metrics

    def _apply_market_filter(self, test_data, predictions, fold):
        """
        应用市场情绪过滤器

        使用滞后1天的市场上涨比例动态调整预测阈值。

        Args:
            test_data: 测试数据
            predictions: 预测结果
            fold: fold编号

        Returns:
            pd.DataFrame: 过滤后的预测结果
        """
        print(f"  🔄 应用市场情绪过滤器...")

        # 初始化市场过滤器（首次调用时）
        if self.market_filter is None:
            self.market_filter = MarketSentimentFilter(lookback_days=1)

        # 从测试数据中提取收益率
        if 'Return_1d' in test_data.columns:
            # 使用 reset_index 避免索引和列名冲突
            returns_df = test_data[['Return_1d']].reset_index()
            returns_df.columns = ['Date', 'Return_1d']

            # 调试：检查数据
            print(f"  📊 市场情绪数据: {len(returns_df)} 条记录, {returns_df['Date'].nunique()} 个交易日")
            up_ratio_daily = returns_df.groupby('Date')['Return_1d'].apply(lambda x: (x > 0).mean())
            print(f"     平均上涨比例: {up_ratio_daily.mean()*100:.1f}%")

            self.market_filter.prepare_market_schedule(
                returns_df,
                date_col='Date',
                ret_col='Return_1d'
            )
        else:
            logger.warning("无法准备市场情绪数据，跳过过滤")
            predictions['filtered_signal'] = predictions['prediction']
            return predictions

        # 构建预测 DataFrame
        pred_df = predictions.copy()
        pred_df['Date'] = predictions.index
        pred_df['Predict_Prob'] = predictions['probability']
        pred_df['Predict_Direction'] = predictions['prediction'].map({1: 'UP', 0: 'DOWN'})

        # 应用过滤
        filtered_df = self.market_filter.apply_filter(
            pred_df,
            date_col='Date',
            prob_col='Predict_Prob',
            direction_col='Predict_Direction'
        )

        # 更新预测结果
        predictions['market_up_ratio_lag1'] = filtered_df['market_up_ratio_lag1'].values
        predictions['dynamic_threshold'] = filtered_df['dynamic_threshold'].values
        predictions['market_layer'] = filtered_df['market_layer'].values
        predictions['filtered_signal'] = filtered_df['filtered_signal'].values

        # 统计过滤效果
        original_signals = (predictions['prediction'] == 1).sum()
        filtered_signals = predictions['filtered_signal'].sum()
        reduction_pct = (original_signals - filtered_signals) / original_signals if original_signals > 0 else 0

        print(f"  ✅ 市场情绪过滤完成")
        print(f"     原始信号: {original_signals}, 过滤后: {filtered_signals}, 减少: {reduction_pct:.1%}")

        return predictions

    def _calculate_metrics(self, test_data, predictions):
        """
        计算多维度评估指标（与港股一致）

        Args:
            test_data: 测试数据
            predictions: 预测结果

        Returns:
            dict: 评估指标
        """
        # ========== 交易成本定义（A股标准）==========
        # A股双边交易成本 = 佣金 + 印花税 + 滑点
        COMMISSION = 0.0005    # 单边佣金 0.05%（A股较低）
        STAMP_DUTY = 0.001     # 印花税 0.1%（A股卖出时收取）
        SLIPPAGE = 0.001       # 滑点 0.1%
        TOTAL_COST = (COMMISSION + SLIPPAGE) * 2 + STAMP_DUTY  # 双边成本约 0.4%

        # 合并数据
        df = test_data.copy()
        df['prediction'] = predictions['prediction'].values
        df['probability'] = predictions['probability'].values

        # 保存 Code 列
        if 'Code' not in df.columns and 'Code' in predictions.columns:
            df['Code'] = predictions['Code'].values

        # 保存市场情绪数据
        if 'market_layer' in predictions.columns:
            df['market_layer'] = predictions['market_layer'].values
        if 'dynamic_threshold' in predictions.columns:
            df['dynamic_threshold'] = predictions['dynamic_threshold'].values
        if 'filtered_signal' in predictions.columns:
            df['filtered_signal'] = predictions['filtered_signal'].values

        # 确保索引对齐
        df = df[df.index.isin(predictions.index)]

        if df.empty:
            raise ValueError("合并后的数据为空")

        # ========== 计算实际收益率 ==========
        if 'Future_Return' in df.columns and df['Future_Return'].notna().sum() > 0:
            df['actual_return'] = df['Future_Return']
        else:
            df['actual_return'] = df['Close'].shift(-self.horizon) / df['Close'] - 1

        # ========== 计算 IC 和 Rank IC ==========
        valid_mask = df['actual_return'].notna() & df['probability'].notna()
        if valid_mask.sum() > 1:
            ic = df.loc[valid_mask, 'probability'].corr(df.loc[valid_mask, 'actual_return'])
            rank_ic = df.loc[valid_mask, 'probability'].rank().corr(df.loc[valid_mask, 'actual_return'].rank())
        else:
            ic = 0.0
            rank_ic = 0.0

        # 计算交易信号
        if 'filtered_signal' in predictions.columns:
            df['signal'] = predictions['filtered_signal'].values
        else:
            df['signal'] = (df['prediction'] >= self.confidence_threshold).astype(int)

        # 计算收益率（扣除交易成本）
        df['strategy_return'] = df['signal'] * df['actual_return']
        df['strategy_return_net'] = df['strategy_return'] - df['signal'] * TOTAL_COST

        # 移除NaN
        df = df.dropna(subset=['actual_return', 'strategy_return'])

        if len(df) == 0:
            raise ValueError("没有有效的样本")

        # 分离有交易信号的样本
        trades = df[df['signal'] == 1].copy()

        # 基础指标
        if len(trades) > 0:
            avg_return = trades['strategy_return'].mean()
            win_rate = (trades['strategy_return_net'] > 0).sum() / len(trades)
            return_std = trades['strategy_return'].std()
            avg_return_net = trades['strategy_return_net'].mean()
        else:
            avg_return = 0.0
            win_rate = 0.0
            return_std = 0.0
            avg_return_net = 0.0

        # 准确率
        accuracy = (df['prediction'] == df['Label']).sum() / len(df) if 'Label' in df.columns else 0.0
        num_samples = len(df)
        num_trades = len(trades)

        # ========== 年化收益和夏普比率计算 ==========
        holding_period_factor = 252 / self.horizon
        annualized_return = avg_return * holding_period_factor

        if len(trades) > 1:
            batch_std = trades['strategy_return'].std()
        else:
            batch_std = return_std if return_std > 0 else 0.0

        annualized_std = batch_std * np.sqrt(holding_period_factor) if batch_std > 0 else 0.0

        # ========== 最大回撤计算 ==========
        if len(trades) > 1:
            cumulative = (1 + trades['strategy_return']).cumprod()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0

        # 夏普比率
        risk_free_rate = 0.02
        if annualized_std > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
        else:
            sharpe_ratio = 0.0

        return {
            'num_samples': num_samples,
            'num_trades': num_trades,
            'avg_return': avg_return,
            'avg_return_net': avg_return_net,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'return_std': return_std,
            'annualized_return': annualized_return,
            'annualized_std': annualized_std,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'ic': ic if not np.isnan(ic) else 0.0,
            'rank_ic': rank_ic if not np.isnan(rank_ic) else 0.0,
            'prediction_df': df,  # 返回预测详情用于保存
        }

    def _calculate_overall_metrics(self, fold_results):
        """计算整体指标（与港股一致）"""
        if not fold_results:
            return {}

        # 计算平均指标
        avg_return = np.mean([r['avg_return'] for r in fold_results])
        avg_win_rate = np.mean([r['win_rate'] for r in fold_results])
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_sharpe_ratio = np.mean([r['sharpe_ratio'] for r in fold_results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])

        # 稳定性分析
        return_std = np.std([r['avg_return'] for r in fold_results])
        return_range = np.max([r['avg_return'] for r in fold_results]) - np.min([r['avg_return'] for r in fold_results])
        sharpe_std = np.std([r['sharpe_ratio'] for r in fold_results])

        # 稳定性评级
        if return_std < 0.02:
            stability_rating = "高（优秀）"
        elif return_std < 0.05:
            stability_rating = "中（良好）"
        else:
            stability_rating = "低（需改进）"

        # 综合评分
        score = 0

        # 1. 夏普比率评分（25分）
        if avg_sharpe_ratio >= 1.5:
            score += 25
        elif avg_sharpe_ratio >= 1.0:
            score += 22
        elif avg_sharpe_ratio >= 0.5:
            score += 18
        elif avg_sharpe_ratio >= 0:
            score += 10

        # 2. IC指标评分（25分）
        avg_ic = np.mean([r.get('ic', 0) for r in fold_results])
        if avg_ic >= 0.20:
            score += 25
        elif avg_ic >= 0.15:
            score += 22
        elif avg_ic >= 0.10:
            score += 18
        elif avg_ic >= 0.05:
            score += 15
        elif avg_ic >= 0:
            score += 10

        # 3. 最大回撤评分（25分）
        if avg_max_drawdown > -0.05:
            score += 25
        elif avg_max_drawdown > -0.10:
            score += 20
        elif avg_max_drawdown > -0.20:
            score += 15
        else:
            score += 10

        # 4. 稳定性评分（25分）
        if return_std < 0.02 and sharpe_std < 2.0:
            score += 25
        elif return_std < 0.05 and sharpe_std < 3.0:
            score += 20
        else:
            score += 10

        # 综合评级
        if score >= 85:
            overall_rating = "优秀"
            recommendation = "强烈推荐实盘"
        elif score >= 70:
            overall_rating = "良好"
            recommendation = "推荐实盘"
        elif score >= 55:
            overall_rating = "一般"
            recommendation = "谨慎使用"
        else:
            overall_rating = "不佳"
            recommendation = "需要改进"

        avg_rank_ic = np.mean([r.get('rank_ic', 0) for r in fold_results])

        return {
            'num_folds': len(fold_results),
            'avg_return': avg_return,
            'avg_win_rate': avg_win_rate,
            'avg_accuracy': avg_accuracy,
            'avg_sharpe_ratio': avg_sharpe_ratio,
            'avg_max_drawdown': avg_max_drawdown,
            'return_std': return_std,
            'return_range': return_range,
            'sharpe_std': sharpe_std,
            'stability_rating': stability_rating,
            'overall_score': score,
            'overall_rating': overall_rating,
            'recommendation': recommendation,
            'avg_ic': avg_ic,
            'avg_rank_ic': avg_rank_ic,
            'total_samples': sum(r['num_samples'] for r in fold_results),
        }

    def _print_overall_results(self, overall):
        """打印整体结果"""
        print(f"\n{'='*80}")
        print("📊 Walk-forward 验证完成")
        print(f"{'='*80}")

        if not overall:
            print("⚠️  所有 Fold 验证失败，无法计算整体指标")
            return

        print(f"\n综合指标:")
        print(f"  Fold 数量: {overall['num_folds']}")
        print(f"  总样本数: {overall['total_samples']}")
        print(f"  综合评分: {overall['overall_score']}/100")
        print(f"  综合评级: {overall['overall_rating']}")
        print(f"  推荐度: {overall['recommendation']}")
        print(f"  平均收益率: {overall['avg_return']:.2%}")
        print(f"  平均胜率: {overall['avg_win_rate']:.2%}")
        print(f"  平均准确率: {overall['avg_accuracy']:.2%}")
        print(f"  平均夏普比率: {overall['avg_sharpe_ratio']:.4f}")
        print(f"  平均最大回撤: {overall['avg_max_drawdown']:.2%}")
        print(f"  平均 IC: {overall['avg_ic']:.4f}")
        print(f"  平均 Rank IC: {overall['avg_rank_ic']:.4f}")
        print(f"  稳定性评级: {overall['stability_rating']}")

        print(f"\n模型评估:")
        if overall['avg_accuracy'] > 0.65:
            print("  ⚠️ 准确率 > 65%，可能存在数据泄漏风险")
        elif overall['avg_accuracy'] > 0.55:
            print("  ✅ 准确率在合理范围内 (55%-65%)")
        else:
            print("  ⚠️ 准确率 < 55%，模型预测能力较弱")

        if overall['avg_return'] > 0:
            print(f"  ✅ 平均收益为正 ({overall['avg_return']:.2%})")
        else:
            print(f"  ⚠️ 平均收益为负 ({overall['avg_return']:.2%})")

        if overall['avg_ic'] > 0.1:
            print(f"  ✅ IC > 0.1，特征有效")
        else:
            print(f"  ⚠️ IC < 0.1，特征预测能力较弱")

        print(f"{'='*80}")

    def _save_report(self, report):
        """保存验证报告（与港股一致）"""
        os.makedirs('output', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        horizon = report['validation_config']['horizon']

        # 创建详细结果目录
        detail_dir = os.path.join('output', f'{timestamp}_a_stock_catboost_{horizon}d')
        os.makedirs(detail_dir, exist_ok=True)

        # 1. 保存 validation_summary.json
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'a_stock_catboost',
            'horizon': horizon,
            'num_folds': report['validation_config']['num_folds'],
            'num_stocks': len(report['validation_config']['stock_list']),
            'overall_metrics': {
                'score': report['overall_metrics'].get('overall_score', 0),
                'rating': report['overall_metrics'].get('overall_rating', 'N/A'),
                'recommendation': report['overall_metrics'].get('recommendation', 'N/A'),
                'avg_sharpe': report['overall_metrics'].get('avg_sharpe_ratio', 0),
                'avg_ic': report['overall_metrics'].get('avg_ic', 0),
                'avg_accuracy': report['overall_metrics'].get('avg_accuracy', 0),
            }
        }
        summary_file = os.path.join(detail_dir, 'validation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"汇总报告已保存: {summary_file}")

        # 2. 保存 fold_metrics_detail.json
        fold_metrics = {
            'model_type': 'a_stock_catboost',
            'horizon': horizon,
            'folds': []
        }
        for fold in report['fold_results']:
            fold_data = {
                'fold': fold.get('fold', 0) + 1,
                'train_period': f"{fold.get('train_start_date', '')} to {fold.get('train_end_date', '')}",
                'test_period': f"{fold.get('test_start_date', '')} to {fold.get('test_end_date', '')}",
                'metrics': {
                    'ic': fold.get('ic', 0),
                    'rank_ic': fold.get('rank_ic', 0),
                    'accuracy': fold.get('accuracy', 0),
                    'sharpe': fold.get('sharpe_ratio', 0),
                    'max_drawdown': fold.get('max_drawdown', 0),
                    'win_rate': fold.get('win_rate', 0),
                    'avg_return': fold.get('avg_return', 0)
                },
                'sample_counts': {
                    'total': fold.get('num_samples', 0),
                    'train': fold.get('num_train_samples', 0),
                    'test': fold.get('num_test_samples', 0)
                },
                'top_features': fold.get('top_features', [])
            }
            fold_metrics['folds'].append(fold_data)

        metrics_file = os.path.join(detail_dir, 'fold_metrics_detail.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(fold_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Fold详细指标已保存: {metrics_file}")

        # 3. 保存完整JSON
        json_file = os.path.join('output', f'walk_forward_a_stock_catboost_{horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON报告已保存: {json_file}")

        # 4. 保存CSV
        csv_file = os.path.join('output', f'walk_forward_a_stock_catboost_{horizon}d_{timestamp}.csv')
        fold_df = pd.DataFrame(report['fold_results'])
        fold_df.to_csv(csv_file, index=False)
        logger.info(f"CSV报告已保存: {csv_file}")

        # 5. 保存Markdown报告
        md_file = os.path.join('output', f'walk_forward_a_stock_catboost_{horizon}d_{timestamp}.md')
        self._generate_markdown_report(report, md_file)
        logger.info(f"Markdown报告已保存: {md_file}")

        # 6. 保存 prediction_analysis.csv（用于计算个股盈亏比）
        if self.all_predictions:
            all_pred_df = pd.concat(self.all_predictions, ignore_index=True)

            # 标准化列名（与港股格式一致）
            pred_analysis = all_pred_df.rename(columns={
                'Code': 'code',
                'prediction': 'Predicted_Direction',
                'probability': 'Predict_Prob',
                'actual_return': 'Actual_Return',
                'Label': 'Actual_Direction',
            })

            # 转换预测方向
            pred_analysis['Predicted_Direction'] = pred_analysis['Predicted_Direction'].map({1: 'UP', 0: 'DOWN'})
            pred_analysis['Actual_Direction'] = pred_analysis['Actual_Direction'].map({1: 'UP', 0: 'DOWN'})

            # 计算是否正确
            pred_analysis['Is_Correct'] = pred_analysis['Predicted_Direction'] == pred_analysis['Actual_Direction']

            # 确保股票代码为6位字符串格式（保留前导零）
            if 'code' in pred_analysis.columns:
                pred_analysis['code'] = pred_analysis['code'].astype(str).str.zfill(6)

            # 选择需要的列
            columns_to_save = ['code', 'fold', 'Predicted_Direction', 'Predict_Prob',
                               'Actual_Direction', 'Actual_Return', 'Is_Correct']
            # 添加日期列（如果存在）
            if 'Date' in pred_analysis.columns or pred_analysis.index.name == 'Date':
                if pred_analysis.index.name == 'Date':
                    pred_analysis['Date'] = pred_analysis.index
                columns_to_save.insert(0, 'Date')

            pred_analysis_file = os.path.join(detail_dir, 'prediction_analysis.csv')
            pred_analysis[columns_to_save].to_csv(pred_analysis_file, index=False)
            logger.info(f"预测详情已保存: {pred_analysis_file}")
            print(f"  - 预测详情: {pred_analysis_file}")

        print(f"\n📄 报告已保存:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV: {csv_file}")
        print(f"  - MD: {md_file}")
        print(f"  - 详情目录: {detail_dir}")

    def _generate_markdown_report(self, report, output_file):
        """生成Markdown格式报告"""
        config = report['validation_config']
        overall = report['overall_metrics']
        folds = report['fold_results']

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# A股 Walk-forward 验证报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📋 验证配置\n\n")
            f.write(f"- **模型类型**: A股 CatBoost\n")
            f.write(f"- **训练窗口**: {config['train_window_months']} 个月\n")
            f.write(f"- **测试窗口**: {config['test_window_months']} 个月\n")
            f.write(f"- **滚动步长**: {config['step_window_months']} 个月\n")
            f.write(f"- **预测周期**: {config['horizon']} 天\n")
            f.write(f"- **验证日期**: {config['start_date']} 至 {config['end_date']}\n")
            f.write(f"- **Fold数量**: {config['num_folds']}\n")
            f.write(f"- **股票数量**: {len(config['stock_list'])}\n\n")

            f.write("## 📊 整体性能指标\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败\n\n")
            else:
                f.write(f"- **综合评分**: {overall.get('overall_score', 0)}/100\n")
                f.write(f"- **综合评级**: {overall.get('overall_rating', 'N/A')}\n")
                f.write(f"- **推荐度**: {overall.get('recommendation', 'N/A')}\n")
                f.write(f"- **平均收益率**: {overall['avg_return']:.2%}\n")
                f.write(f"- **平均胜率**: {overall['avg_win_rate']:.2%}\n")
                f.write(f"- **平均准确率**: {overall['avg_accuracy']:.2%}\n")
                f.write(f"- **平均夏普比率**: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- **平均最大回撤**: {overall['avg_max_drawdown']:.2%}\n")
                f.write(f"- **平均 IC**: {overall.get('avg_ic', 0):.4f}\n")
                f.write(f"- **稳定性评级**: {overall.get('stability_rating', 'N/A')}\n\n")

            f.write("## 📈 Fold 详细结果\n\n")
            f.write("| Fold | 训练期间 | 测试期间 | 样本数 | 收益率 | 胜率 | 准确率 | 夏普比率 | IC |\n")
            f.write("|------|---------|---------|-------|-------|------|-------|---------|------|\n")

            for fold in folds:
                f.write(f"| {fold['fold'] + 1} | {fold['train_start_date']} 至 {fold['train_end_date']} | "
                       f"{fold['test_start_date']} 至 {fold['test_end_date']} | {fold['num_samples']} | "
                       f"{fold['avg_return']:.2%} | {fold['win_rate']:.2%} | {fold['accuracy']:.2%} | "
                       f"{fold['sharpe_ratio']:.4f} | {fold.get('ic', 0):.4f} |\n")

            f.write("\n---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股 Walk-forward 验证')

    # Walk-forward 参数
    parser.add_argument('--train-window', type=int, default=12,
                       help='训练窗口（月，默认: 12）')
    parser.add_argument('--test-window', type=int, default=1,
                       help='测试窗口（月，默认: 1）')
    parser.add_argument('--step-window', type=int, default=1,
                       help='滚动步长（月，默认: 1）')

    # 模型参数
    parser.add_argument('--horizon', type=int, default=20,
                       choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月（默认）')
    parser.add_argument('--confidence-threshold', type=float, default=0.50,
                       help='置信度阈值（默认: 0.50）')

    # 数据参数
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='验证开始日期')
    parser.add_argument('--end-date', type=str, default='2026-07-01',
                       help='验证结束日期')

    # 功能开关
    parser.add_argument('--no-market-filter', action='store_true',
                       help='禁用市场情绪过滤器')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("🚀 A股 Walk-forward 验证系统")
    print(f"{'='*80}\n")

    # 创建验证器
    validator = AStockWalkForwardValidator(
        horizon=args.horizon,
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_window_months=args.step_window,
        confidence_threshold=args.confidence_threshold,
        use_market_filter=not args.no_market_filter,
    )

    # 执行验证
    try:
        report = validator.validate(
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print(f"\n{'='*80}")
        print("✅ 验证完成")
        print(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
