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
from ml_services.market_regime import MarketSentimentFilter
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
        fp_penalty: float = None           # False Positive 惩罚系数（非对称损失函数）
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
            fp_penalty: False Positive 惩罚系数（非对称损失函数）
                - None: 不使用额外惩罚
                - float (如 2.5): 对类别0（下跌）施加 fp_penalty 倍惩罚
        """
        self.model_type = model_type.lower()
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.horizon = horizon
        self.confidence_threshold = confidence_threshold
        self.use_feature_selection = use_feature_selection
        self.min_train_samples = min_train_samples
        self.fp_penalty = fp_penalty

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

        # 市场情绪过滤器（使用滞后数据避免前瞻性偏差）
        self.market_filter = None
        self.use_market_filter = True  # 默认启用

        logger.info(f"初始化 Walk-forward 验证器")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"训练窗口: {train_window_months} 个月")
        logger.info(f"测试窗口: {test_window_months} 个月")
        logger.info(f"滚动步长: {step_window_months} 个月")
        logger.info(f"预测周期: {horizon} 天")
        if self.fp_penalty is not None:
            logger.info(f"非对称损失函数: FP惩罚={self.fp_penalty}x")

        # 预加载社区 ID（确保训练/预测一致性）
        self.preloaded_community_ids = None
        network_features_file = 'output/network_features_for_ml.json'
        if os.path.exists(network_features_file):
            try:
                with open(network_features_file, 'r') as f:
                    network_features_data = json.load(f)
                all_community_ids = set()
                for stock_code, net_features in network_features_data.items():
                    if 'net_community_id' in net_features:
                        comm_id = net_features['net_community_id']
                        if comm_id >= 0:
                            all_community_ids.add(int(comm_id))
                if all_community_ids:
                    self.preloaded_community_ids = sorted(list(all_community_ids))
                    logger.info(f"预加载社区 ID: {self.preloaded_community_ids}")
            except Exception as e:
                logger.warning(f"预加载网络特征失败: {e}")

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

        # 创建模型实例（支持非对称损失函数）
        if self.fp_penalty is not None:
            model = self.model_class(fp_penalty=self.fp_penalty)
            print(f"  ⚠️ 使用非对称损失函数: FP惩罚={self.fp_penalty}x")
        else:
            model = self.model_class()

        # 准备训练数据
        train_codes = stock_list
        train_data = model.prepare_data(
            train_codes,
            start_date=train_start_date,
            end_date=train_end_date,
            horizon=self.horizon,
            for_backtest=False,
            community_ids=self.preloaded_community_ids  # 使用预加载的社区 ID
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

        # 准备测试数据（使用训练时保存的社区 ID，确保特征一致性）
        print(f"  🔄 准备测试数据...")
        test_data = model.prepare_data(
            train_codes,
            start_date=test_start_date,
            end_date=test_end_date,
            horizon=self.horizon,
            for_backtest=False,
            community_ids=model.community_ids  # 使用训练时的社区 ID
        )

        print(f"  ✅ 测试数据准备完成: {len(test_data)} 条记录")

        # 生成预测（使用 predict_proba 方法批量预测）
        print(f"  🔄 生成预测...")
        X_test = test_data[model.feature_columns]
        prediction_proba = model.predict_proba(X_test)

        # 使用标准阈值 0.5 生成初始预测（与 comprehensive_analysis.py 一致）
        # 市场情绪过滤器会在后续步骤中应用动态阈值
        predictions = pd.DataFrame({
            'prediction': (prediction_proba[:, 1] >= 0.5).astype(int),
            'probability': prediction_proba[:, 1],
            'adaptive_threshold': np.full(len(prediction_proba), 0.5)  # 固定阈值，后续由市场情绪过滤器调整
        }, index=test_data.index)
        
        # 添加 Code 列（从列中提取）
        if 'Code' in test_data.columns:
            predictions['Code'] = test_data['Code'].values
        else:
            # 如果 Code 列不存在，使用 train_codes
            predictions['Code'] = [train_codes[0]] * len(predictions)  # 单股票情况下使用第一个股票代码
        
        print(f"  ✅ 预测生成完成 ({len(predictions)} 条预测)")

        # ========== 应用市场情绪过滤器 ==========
        if self.use_market_filter:
            predictions = self._apply_market_filter(test_data, predictions, fold)

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

        # 获取特征重要性（Top 20）
        try:
            if hasattr(model, 'catboost_model') and model.catboost_model is not None:
                feat_imp = pd.DataFrame({
                    'Feature': model.feature_columns,
                    'Importance': model.catboost_model.feature_importances_
                })
                feat_imp = feat_imp.sort_values('Importance', ascending=False)
                # 保存 Top 100 特征重要性
                top_features = feat_imp.head(100).to_dict('records')
                metrics['top_features'] = [
                    {'feature': r['Feature'], 'importance': round(r['Importance'], 4)}
                    for r in top_features
                ]
                print(f"  ✅ 特征重要性已记录 (Top 100)")
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            metrics['top_features'] = []

        return metrics

    def _apply_market_filter(self, test_data, predictions, fold):
        """
        应用市场情绪过滤器

        使用滞后1天的市场上涨比例动态调整预测阈值，
        在极端市场环境时提高门槛，减少 False Positive。

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

        # 每个 fold 都需要准备市场收益率数据（因为日期范围不同）
        # 关键：使用所有股票的收益率数据，按日期计算上涨比例
        # 不能只使用单只股票的 Return_1d，需要聚合所有股票

        # 从 test_data 中提取所有股票的收益率数据
        if 'Return_1d' in test_data.columns:
            # 使用 reset_index 避免索引和列名冲突
            # 关键：直接传入原始收益率数据，让 prepare_market_schedule 内部进行 groupby
            # 不要在这里预先计算上涨比例，否则 prepare_market_schedule 会再次 groupby 导致错误
            returns_df = test_data[['Return_1d']].reset_index()
            returns_df.columns = ['Date', 'Return_1d']

            self.market_filter.prepare_market_schedule(
                returns_df,
                date_col='Date',
                ret_col='Return_1d'
            )
        elif 'Close' in test_data.columns:
            # 计算收益率
            returns_df = test_data[['Close']].reset_index()
            returns_df.columns = ['Date', 'Close']
            returns_df['Return_1d'] = returns_df['Close'].pct_change()

            # 直接传入收益率数据
            self.market_filter.prepare_market_schedule(
                returns_df[['Date', 'Return_1d']],
                date_col='Date',
                ret_col='Return_1d'
            )
        else:
            logger.warning("无法准备市场情绪数据，跳过过滤")
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

        # 统计各层级分布
        layer_counts = predictions['market_layer'].value_counts()
        print(f"     层级分布: {layer_counts.to_dict()}")

        return predictions

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

        # 保存 Code 列（用于错误分析）
        if 'Code' not in df.columns and 'Code' in predictions.columns:
            df['Code'] = predictions['Code'].values

        # 确保索引对齐（删除预测中不存在于 test_data 的索引）
        df = df[df.index.isin(predictions.index)]

        if df.empty:
            raise ValueError("合并后的数据为空")

        # ========== 计算实际收益率（与训练时一致）==========
        # 优先使用 Future_Return（在过滤前计算，包含正确的未来收益率）
        # 如果 Future_Return 不存在，则使用 Close.shift(-horizon) 计算
        if 'Future_Return' in df.columns and df['Future_Return'].notna().sum() > 0:
            df['actual_return'] = df['Future_Return']
        else:
            # 训练时使用累积收益率：Close[t+horizon] / Close[t] - 1
            df['actual_return'] = df['Close'].shift(-self.horizon) / df['Close'] - 1

        # ========== 计算 IC 和 Rank IC ==========
        # IC: 预测概率与实际收益率的相关系数（不是与二元标签）
        # 这是业界标准定义：IC = Corr(Prediction, Actual_Return)
        if 'actual_return' in df.columns:
            valid_mask = df['actual_return'].notna() & df['probability'].notna()
            if valid_mask.sum() > 1:
                ic = df.loc[valid_mask, 'probability'].corr(df.loc[valid_mask, 'actual_return'])
                rank_ic = df.loc[valid_mask, 'probability'].rank().corr(df.loc[valid_mask, 'actual_return'].rank())
            else:
                ic = 0.0
                rank_ic = 0.0
        else:
            ic = 0.0
            rank_ic = 0.0

        # 预测分布统计
        prediction_std = df['probability'].std()

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

        # 计算交易信号
        # 优先使用市场情绪过滤后的信号，否则使用风险过滤
        if 'filtered_signal' in predictions.columns:
            # 使用市场情绪过滤后的信号
            df['signal'] = predictions['filtered_signal'].values
            print(f"  📊 使用市场情绪过滤信号")
        else:
            # 使用风险过滤（原有逻辑）
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

        # ========== 年化收益和夏普比率计算（修正版 v3） ==========
        # 修正说明：
        # - 批次收益是20天持有期的收益，不是日收益
        # - 年化因子应考虑持有期：一年约252个交易日，20天持有期可做 252/20 ≈ 12.6 次交易
        # - 年化收益 = 批次收益均值 * (252/horizon)
        # - 年化波动率 = 批次收益标准差 * sqrt(252/horizon)

        # 持有期调整因子
        holding_period_factor = 252 / self.horizon  # 20天持有期 = 12.6

        # 年化收益
        annualized_return = avg_return * holding_period_factor

        # 标准差：基于批次收益的波动
        if len(batch_returns) > 1:
            batch_std = np.std(batch_returns, ddof=1)
        elif return_std > 0:
            batch_std = return_std
        else:
            batch_std = 0.0

        # 年化标准差（考虑持有期）
        annualized_std = batch_std * np.sqrt(holding_period_factor) if batch_std > 0 else 0.0

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
        # 修正：使用持有期调整的年化因子 sqrt(252/horizon)
        if len(trades) > 0:
            downside_returns = trades['strategy_return'][trades['strategy_return'] < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    # 持有期调整的年化下行标准差
                    annualized_downside_std = downside_std * np.sqrt(holding_period_factor)
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
        # 修正：使用持有期调整的年化因子
        if len(trades) > 0:
            excess_returns = trades['strategy_return'] - trades['actual_return']
            tracking_error = excess_returns.std()
            if tracking_error > 0:
                # 持有期调整的年化跟踪误差
                annualized_tracking_error = tracking_error * np.sqrt(holding_period_factor)
                information_ratio = (excess_returns.mean() * holding_period_factor) / annualized_tracking_error
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0

        # ========== 置信度与收益关系分析 ==========
        confidence_bins = self._calculate_confidence_return_breakdown(df)

        # ========== 错误分析 ==========
        error_analysis = self._analyze_errors(df)

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
            # 新增指标
            'ic': ic if not np.isnan(ic) else 0.0,
            'rank_ic': rank_ic if not np.isnan(rank_ic) else 0.0,
            'prediction_std': prediction_std,
            'confidence_breakdown': confidence_bins,
            'error_analysis': error_analysis,
            # 百分位预测效果分析
            'percentile_performance': self._calculate_percentile_performance(df)
        }

    def _calculate_percentile_performance(self, df):
        """
        计算不同百分位预测股票的实际效果

        方法：每个交易日按预测概率排名，分成 Top 1%/5%/10%/20% 四组，
        计算每组的平均实际收益。

        Args:
            df: 包含 probability, actual_return, Code 的 DataFrame

        Returns:
            list: 各百分位的收益统计
        """
        if 'Code' not in df.columns or 'actual_return' not in df.columns:
            return []

        # 导入股票名称映射
        try:
            from config import WATCHLIST as STOCK_NAMES
        except ImportError:
            STOCK_NAMES = {}

        # 按日期分组，计算截面百分位
        df = df.copy()
        # 提取日期（处理时区问题）
        if hasattr(df.index, 'date'):
            df['date'] = df.index.date
        elif hasattr(df.index[0], 'strftime'):
            df['date'] = df.index.strftime('%Y-%m-%d')
        else:
            df['date'] = df.index

        percentiles = [1, 5, 10, 20]
        results = {f'Top_{p}pct': {'count': 0, 'total_return': 0.0, 'returns': [], 'stocks': []}
                   for p in percentiles}

        for date, group in df.groupby('date'):
            if len(group) < 5:  # 样本太少跳过
                continue

            # 过滤掉 actual_return 为 NaN 的样本
            group = group.dropna(subset=['actual_return', 'probability'])
            if len(group) < 5:
                continue

            # 按预测概率降序排列
            group = group.sort_values('probability', ascending=False)

            for p in percentiles:
                # 计算该百分位的股票数量
                n_stocks = max(1, int(len(group) * p / 100))
                top_stocks = group.head(n_stocks)

                # 计算平均收益
                avg_return = top_stocks['actual_return'].mean()
                if not np.isnan(avg_return):
                    results[f'Top_{p}pct']['count'] += n_stocks
                    results[f'Top_{p}pct']['total_return'] += avg_return * n_stocks
                    results[f'Top_{p}pct']['returns'].append(avg_return)

                    # 记录股票详情（只记录 Top 1% 和 Top 5%）
                    if p <= 5:
                        for _, row in top_stocks.iterrows():
                            code = row.get('Code', 'Unknown')
                            stock_name = STOCK_NAMES.get(code, code) if isinstance(STOCK_NAMES, dict) else code
                            results[f'Top_{p}pct']['stocks'].append({
                                'date': str(date),
                                'code': code,
                                'name': stock_name,
                                'probability': round(float(row.get('probability', 0)), 4),
                                'actual_return': round(float(row.get('actual_return', 0)), 4)
                            })

        # 计算最终统计
        final_results = []
        for p in percentiles:
            key = f'Top_{p}pct'
            if results[key]['count'] > 0:
                result_item = {
                    'percentile': f'Top {p}%',
                    'total_stocks': results[key]['count'],
                    'avg_return': float(results[key]['total_return'] / results[key]['count']),
                    'return_std': float(np.std(results[key]['returns'])) if results[key]['returns'] else 0.0,
                    'num_periods': len(results[key]['returns'])
                }
                # 只在 Top 1% 和 Top 5% 中包含股票详情
                if p <= 5 and results[key]['stocks']:
                    result_item['top_stocks'] = results[key]['stocks'][:50]  # 最多记录50条
                final_results.append(result_item)

        return final_results

    def _calculate_confidence_return_breakdown(self, df):
        """计算置信度与收益关系"""
        bins = [
            (0.8, 1.0, '0.8-1.0'),
            (0.7, 0.8, '0.7-0.8'),
            (0.6, 0.7, '0.6-0.7'),
            (0.5, 0.6, '0.5-0.6'),
            (0.0, 0.5, '0.0-0.5')
        ]

        confidence_breakdown = []
        for low, high, label in bins:
            mask = (df['probability'] >= low) & (df['probability'] < high)
            subset = df[mask]
            if len(subset) > 0:
                avg_return = subset['actual_return'].mean()
                hit_rate = (subset['Label'] == 1).sum() / len(subset) if 'Label' in subset.columns else 0
                confidence_breakdown.append({
                    'range': label,
                    'count': len(subset),
                    'avg_return': float(avg_return) if not np.isnan(avg_return) else 0,
                    'hit_rate': float(hit_rate)
                })
            else:
                confidence_breakdown.append({
                    'range': label,
                    'count': 0,
                    'avg_return': 0,
                    'hit_rate': 0
                })

        return confidence_breakdown

    def _analyze_errors(self, df):
        """分析所有预测（包括正确和错误预测）"""
        predictions = []

        # 需要有 Code 列才能进行分析
        if 'Code' not in df.columns:
            return predictions

        for idx, row in df.iterrows():
            prob = row['probability']
            actual_return = row.get('actual_return', 0)
            code = row.get('Code', 'Unknown')

            # 处理 NaN 值
            if pd.isna(actual_return):
                continue

            # 判断预测方向和实际方向
            predict_up = prob >= 0.5
            actual_up = actual_return > 0

            # 判断预测是否正确
            is_correct = (predict_up and actual_up) or (not predict_up and not actual_up)

            # 确定预测类型
            if is_correct:
                if predict_up:
                    pred_type = 'True Positive'  # 预测涨，实际涨
                    reason = '正确预测上涨'
                else:
                    pred_type = 'True Negative'  # 预测跌，实际跌
                    reason = '正确预测下跌'
            else:
                if predict_up:
                    pred_type = 'False Positive'  # 预测涨，实际跌
                    reason = '预测涨但实际下跌'
                else:
                    pred_type = 'False Negative'  # 预测跌，实际涨
                    reason = '预测跌但实际上涨'

            predictions.append({
                'Date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'Stock_Code': code,
                'Predict_Prob': round(prob, 4),
                'Predict_Direction': 'UP' if predict_up else 'DOWN',
                'Actual_Return': round(actual_return, 4),
                'Actual_Direction': 'UP' if actual_up else 'DOWN',
                'Is_Correct': is_correct,
                'Prediction_Type': pred_type,
                'Note': reason
            })

        return predictions

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
        # 权重：夏普比率25%，IC指标25%，最大回撤25%，稳定性25%
        # 修正说明：年化因子修正后夏普比率约0.97，评分标准相应调整
        score = 0

        # 1. 夏普比率评分（25分）：修正后业界标准 >0.5 为良好，>1.0 为优秀
        if avg_sharpe_ratio >= 1.5:
            score += 25
        elif avg_sharpe_ratio >= 1.0:
            score += 22
        elif avg_sharpe_ratio >= 0.5:
            score += 18
        elif avg_sharpe_ratio >= 0.3:
            score += 15
        elif avg_sharpe_ratio >= 0:
            score += 10

        # 2. IC指标评分（25分）：IC >0.05 表示有效预测能力
        avg_ic = np.mean([r.get('ic', 0) for r in fold_results])
        avg_rank_ic = np.mean([r.get('rank_ic', 0) for r in fold_results])
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

        # 3. 最大回撤评分（25分）：业界标准 <-20%
        # 注意：当前为信号级回测，实际回撤可能更大
        if avg_max_drawdown > -0.05:  # -5%以内，优秀
            score += 25
        elif avg_max_drawdown > -0.10:  # -10%以内，良好
            score += 20
        elif avg_max_drawdown > -0.20:  # -20%以内，可接受
            score += 15
        elif avg_max_drawdown > -0.30:  # -30%以内，较差
            score += 10
        else:
            score += 5

        # 4. 稳定性评分（25分）：基于收益标准差和Fold波动
        # Fold波动大（准确率标准差>0.08）扣分
        if return_std < 0.02 and sharpe_std < 2.0:
            score += 25
        elif return_std < 0.05 and sharpe_std < 3.0:
            score += 20
        elif return_std < 0.10:
            score += 15
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

        # Rank IC 在评分中已计算 avg_ic
        avg_rank_ic = np.mean([r.get('rank_ic', 0) for r in fold_results])

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
            # 新增指标
            'avg_ic': avg_ic,
            'avg_rank_ic': avg_rank_ic
        }

    def save_report(self, report, output_dir='output'):
        """
        保存验证报告（CSV、JSON、Markdown格式 + 详细分析文件）

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

        # 创建详细结果目录
        detail_dir = os.path.join(output_dir, f'{timestamp}_{model_type}_{horizon}d')
        os.makedirs(detail_dir, exist_ok=True)

        # 1. 保存 validation_summary.json
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'horizon': horizon,
            'num_folds': report['validation_config']['num_folds'],
            'num_stocks': len(report['validation_config']['stock_list']),
            'overall_metrics': {
                'score': report['overall_metrics'].get('overall_score', 0),
                'rating': report['overall_metrics'].get('overall_rating', 'N/A'),
                'recommendation': report['overall_metrics'].get('recommendation', 'N/A'),
                'avg_sharpe': report['overall_metrics'].get('avg_sharpe_ratio', 0),
                'avg_ic': report['overall_metrics'].get('avg_ic', 0),
                'avg_rank_ic': report['overall_metrics'].get('avg_rank_ic', 0)
            },
            'saved_files': ['prediction_distribution', 'fold_metrics_detail', 'prediction_analysis', 'confidence_return_breakdown', 'percentile_performance']
        }
        summary_file = os.path.join(detail_dir, 'validation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"汇总报告已保存: {summary_file}")

        # 2. 保存 fold_metrics_detail.json
        fold_metrics = {
            'model_type': model_type,
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
                    'sortino': fold.get('sortino_ratio', 0),
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

        # 3. 保存 prediction_distribution.json
        pred_dist = {
            'folds': [],
            'overall': {
                'avg_ic': report['overall_metrics'].get('avg_ic', 0),
                'avg_rank_ic': report['overall_metrics'].get('avg_rank_ic', 0),
                'avg_prediction_std': np.mean([f.get('prediction_std', 0) for f in report['fold_results']])
            }
        }
        for fold in report['fold_results']:
            pred_dist['folds'].append({
                'fold': fold.get('fold', 0) + 1,
                'test_period': f"{fold.get('test_start_date', '')} to {fold.get('test_end_date', '')}",
                'prediction_stats': {
                    'std': fold.get('prediction_std', 0),
                    'ic': fold.get('ic', 0),
                    'rank_ic': fold.get('rank_ic', 0)
                }
            })

        pred_file = os.path.join(detail_dir, 'prediction_distribution.json')
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(pred_dist, f, indent=2, ensure_ascii=False)
        logger.info(f"预测分布已保存: {pred_file}")

        # 4. 保存 prediction_analysis.csv（所有预测，包括正确和错误）
        all_predictions = []
        for fold in report['fold_results']:
            fold_num = fold.get('fold', 0) + 1
            for pred in fold.get('error_analysis', []):
                pred['Fold'] = fold_num
                all_predictions.append(pred)

        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            # 调整列顺序
            cols = ['Fold', 'Date', 'Stock_Code', 'Predict_Prob', 'Predict_Direction',
                    'Actual_Return', 'Actual_Direction', 'Is_Correct', 'Prediction_Type', 'Note']
            pred_df = pred_df[[c for c in cols if c in pred_df.columns]]
            pred_file = os.path.join(detail_dir, 'prediction_analysis.csv')
            pred_df.to_csv(pred_file, index=False)
            logger.info(f"预测分析已保存: {pred_file}")

        # 5. 保存 confidence_return_breakdown.json
        # 汇总所有 fold 的置信度分析
        confidence_summary = {'confidence_bins': []}
        bins_labels = ['0.8-1.0', '0.7-0.8', '0.6-0.7', '0.5-0.6', '0.0-0.5']

        for label in bins_labels:
            total_count = 0
            total_return = 0
            total_hit_rate = 0
            fold_count = 0

            for fold in report['fold_results']:
                for breakdown in fold.get('confidence_breakdown', []):
                    if breakdown['range'] == label:
                        total_count += breakdown['count']
                        total_return += breakdown['avg_return'] * breakdown['count']
                        total_hit_rate += breakdown['hit_rate'] * breakdown['count']
                        fold_count += 1
                        break

            if total_count > 0:
                confidence_summary['confidence_bins'].append({
                    'range': label,
                    'count': total_count,
                    'avg_return': total_return / total_count,
                    'hit_rate': total_hit_rate / total_count
                })
            else:
                confidence_summary['confidence_bins'].append({
                    'range': label,
                    'count': 0,
                    'avg_return': 0,
                    'hit_rate': 0
                })

        conf_file = os.path.join(detail_dir, 'confidence_return_breakdown.json')
        with open(conf_file, 'w', encoding='utf-8') as f:
            json.dump(confidence_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"置信度分析已保存: {conf_file}")

        # 6. 保存 percentile_performance.json
        all_percentile_results = []
        for fold in report['fold_results']:
            fold_num = fold.get('fold', 0) + 1
            for pct_result in fold.get('percentile_performance', []):
                pct_result_copy = pct_result.copy()
                pct_result_copy['fold'] = fold_num
                all_percentile_results.append(pct_result_copy)

        if all_percentile_results:
            # 汇总各百分位的平均收益
            percentile_summary = {}
            for result in all_percentile_results:
                pct = result['percentile']
                if pct not in percentile_summary:
                    percentile_summary[pct] = {'total_return': 0.0, 'count': 0, 'returns': []}
                percentile_summary[pct]['total_return'] += result['avg_return'] * result['total_stocks']
                percentile_summary[pct]['count'] += result['total_stocks']
                percentile_summary[pct]['returns'].append(result['avg_return'])

            final_summary = []
            for pct in ['Top 1%', 'Top 5%', 'Top 10%', 'Top 20%']:
                if pct in percentile_summary:
                    data = percentile_summary[pct]
                    final_summary.append({
                        'percentile': pct,
                        'avg_return': float(data['total_return'] / data['count']) if data['count'] > 0 else 0.0,
                        'return_std': float(np.std(data['returns'])) if data['returns'] else 0.0,
                        'total_stocks': data['count']
                    })

            percentile_file = os.path.join(detail_dir, 'percentile_performance.json')
            with open(percentile_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': final_summary,
                    'by_fold': all_percentile_results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"百分位效果分析已保存: {percentile_file}")

        # 7. 保存完整JSON格式（原有逻辑）
        json_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON报告已保存: {json_file}")

        # 7. 保存CSV格式（原有逻辑）
        csv_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.csv')
        fold_df = pd.DataFrame(report['fold_results'])
        fold_df.to_csv(csv_file, index=False)
        logger.info(f"CSV报告已保存: {csv_file}")

        # 8. 保存Markdown格式（原有逻辑）
        md_file = os.path.join(output_dir, f'walk_forward_{model_type}_{horizon}d_{timestamp}.md')
        self._generate_markdown_report(report, md_file)
        logger.info(f"Markdown报告已保存: {md_file}")

        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'md_file': md_file,
            'detail_dir': detail_dir
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
                f.write(f"- **平均信息比率**: {overall['avg_information_ratio']:.4f}\n")
                # 新增 IC 和 Rank IC
                f.write(f"- **平均 IC**: {overall.get('avg_ic', 0):.4f}\n")
                f.write(f"- **平均 Rank IC**: {overall.get('avg_rank_ic', 0):.4f}\n\n")

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
            f.write("| Fold | 训练期间 | 测试期间 | 样本数 | 交易次数 | 收益率 | 胜率 | 准确率 | 夏普比率 | 最大回撤 | IC | Rank IC |\n")
            f.write("|------|---------|---------|-------|---------|-------|------|-------|---------|--------|------|--------|\n")

            for fold in folds:
                f.write(f"| {fold['fold'] + 1} | {fold['train_start_date']} 至 {fold['train_end_date']} | "
                       f"{fold['test_start_date']} 至 {fold['test_end_date']} | {fold['num_test_samples']} | "
                       f"{fold.get('num_trades', 'N/A')} | "
                       f"{fold['avg_return']:.2%} | {fold['win_rate']:.2%} | {fold['accuracy']:.2%} | "
                       f"{fold['sharpe_ratio']:.4f} | {fold['max_drawdown']:.2%} | "
                       f"{fold.get('ic', 0):.4f} | {fold.get('rank_ic', 0):.4f} |\n")

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

    # 特征选择参数
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择（默认: False，使用全量特征）')

    # 非对称损失函数参数
    parser.add_argument('--fp-penalty', type=float, default=None,
                       help='False Positive 惩罚系数（非对称损失函数），如 2.5 表示对FP错误施加2.5倍惩罚')

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
        fp_penalty=args.fp_penalty
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