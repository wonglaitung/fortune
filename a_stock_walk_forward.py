#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股 Walk-forward 验证 - 完整版本

复用港股 Walk-forward 架构，调用 AStockTradingModel 完整特征流程：
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

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 导入A股配置和数据
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_TRAINING_LIST,
    get_limit_rate,
)

# 导入日志
from ml_services.logger_config import get_logger
logger = get_logger('a_stock_walk_forward')

# 常量
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_YEAR = 240


class AStockWalkForwardValidator:
    """A股 Walk-forward 验证器 - 完整版本"""

    def __init__(
        self,
        horizon: int = 20,
        train_window_months: int = 12,    # 业界标准：12个月
        test_window_months: int = 1,       # 业界标准：1个月
        confidence_threshold: float = 0.50,
        use_market_filter: bool = True,    # 启用市场情绪过滤器
    ):
        """
        初始化验证器

        Args:
            horizon: 预测周期（天）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            confidence_threshold: 置信度阈值
            use_market_filter: 是否使用市场情绪过滤器
        """
        self.horizon = horizon
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.confidence_threshold = confidence_threshold
        self.use_market_filter = use_market_filter
        self.stock_list = list(A_STOCK_TRAINING_LIST.keys())

        # 市场情绪过滤器（延迟初始化）
        self.market_filter = None

        print(f"\n{'='*60}")
        print("🔬 A股 Walk-forward 验证器（完整版）")
        print(f"{'='*60}")
        print(f"预测周期: {horizon} 天")
        print(f"训练窗口: {train_window_months} 个月")
        print(f"测试窗口: {test_window_months} 个月")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"市场情绪过滤: {'启用' if use_market_filter else '禁用'}")
        print(f"股票数量: {len(self.stock_list)}")

    def validate(self, start_date='2024-01-01', end_date='2026-07-01', num_folds=6):
        """
        执行 Walk-forward 验证

        Args:
            start_date: 验证开始日期
            end_date: 验证结束日期
            num_folds: fold数量

        Returns:
            dict: 验证结果
        """
        print(f"\n{'='*60}")
        print(f"验证日期范围: {start_date} 至 {end_date}")
        print(f"Fold 数量: {num_folds}")
        print(f"{'='*60}")

        # 存储所有fold结果
        all_fold_results = []

        # 计算每个fold的日期范围
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in all_dates if d.weekday() < 5]  # 简单过滤周末

        total_days = len(trading_days)
        train_days = self.train_window_months * TRADING_DAYS_PER_MONTH
        test_days = self.test_window_months * TRADING_DAYS_PER_MONTH
        step_days = (total_days - train_days - test_days) // max(num_folds - 1, 1)

        for fold in range(num_folds):
            print(f"\n{'='*60}")
            print(f"📊 Fold {fold + 1}/{num_folds}")
            print(f"{'='*60}")

            # 计算训练和测试期间
            fold_start_idx = fold * step_days
            train_end_idx = fold_start_idx + train_days
            test_end_idx = train_end_idx + test_days

            if test_end_idx > total_days:
                print(f"  ⚠️ 数据不足，跳过此fold")
                continue

            train_start = trading_days[fold_start_idx].strftime('%Y-%m-%d')
            train_end = trading_days[train_end_idx - 1].strftime('%Y-%m-%d')
            test_start = trading_days[train_end_idx].strftime('%Y-%m-%d')
            test_end = trading_days[min(test_end_idx - 1, total_days - 1)].strftime('%Y-%m-%d')

            print(f"训练期间: {train_start} 至 {train_end}")
            print(f"测试期间: {test_start} 至 {test_end}")

            # 执行fold验证
            try:
                fold_result = self._validate_fold(
                    train_start, train_end,
                    test_start, test_end,
                    fold
                )
                all_fold_results.append(fold_result)

                # 打印结果
                print(f"\n✅ Fold {fold + 1} 结果:")
                print(f"  样本数: {fold_result['num_samples']}")
                print(f"  交易次数: {fold_result.get('num_trades', 'N/A')}")
                print(f"  准确率: {fold_result['accuracy']:.2%}")
                print(f"  胜率: {fold_result['win_rate']:.2%}")
                print(f"  平均收益率: {fold_result['avg_return']:.2%}")
                print(f"  夏普比率: {fold_result.get('sharpe_ratio', 'N/A'):.4f}")
                print(f"  IC: {fold_result.get('ic', 'N/A'):.4f}")
                print(f"  特征数量: {fold_result.get('num_features', 'N/A')}")

            except Exception as e:
                print(f"  ❌ Fold {fold + 1} 验证失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 计算整体指标
        overall_result = self._calculate_overall_metrics(all_fold_results)

        # 生成报告
        report = {
            'validation_config': {
                'horizon': self.horizon,
                'train_window_months': self.train_window_months,
                'test_window_months': self.test_window_months,
                'num_folds': num_folds,
                'start_date': start_date,
                'end_date': end_date,
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

    def _validate_fold(self, train_start, train_end, test_start, test_end, fold):
        """
        验证单个fold - 调用 AStockTradingModel 完整流程

        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            test_end: 测试结束日期
            fold: fold编号

        Returns:
            dict: fold结果
        """
        from a_stock_ml_model import AStockTradingModel

        # 创建模型实例
        print(f"\n  🔄 准备训练数据...")
        model = AStockTradingModel(horizon=self.horizon)

        # 准备训练数据（完整特征流程）
        train_data = model.prepare_data(
            self.stock_list,
            start_date=pd.to_datetime(train_start).tz_localize('UTC'),
            end_date=pd.to_datetime(train_end).tz_localize('UTC'),
            horizon=self.horizon,
            mode='backtest'  # 使用滞后数据
        )

        if train_data is None or len(train_data) < 100:
            raise ValueError(f"训练数据不足: {len(train_data) if train_data is not None else 0}")

        print(f"  ✅ 训练数据准备完成: {len(train_data)} 条记录")

        # 获取特征列
        feature_cols = model.get_feature_columns(train_data)
        print(f"  ✅ 特征数量: {len(feature_cols)}")

        # 准备训练数据
        # 只删除标签缺失的行，特征缺失用 fillna 处理
        train_data_clean = train_data.dropna(subset=['Label'])

        # 对特征缺失值填充（使用中位数）
        for col in feature_cols:
            if col in train_data_clean.columns:
                if train_data_clean[col].isna().any():
                    # 检查是否为数值型
                    if pd.api.types.is_numeric_dtype(train_data_clean[col]):
                        median_val = train_data_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0  # 如果中位数也是NaN，用0填充
                        train_data_clean[col] = train_data_clean[col].fillna(median_val)
                    else:
                        # 非数值型，用 'unknown' 填充
                        train_data_clean[col] = train_data_clean[col].fillna('unknown')

        # 处理分类特征：转换为字符串并编码
        cat_features = []
        for col in feature_cols:
            if col in train_data_clean.columns:
                if not pd.api.types.is_numeric_dtype(train_data_clean[col]):
                    # 标记为分类特征
                    cat_features.append(feature_cols.index(col))
                    # 转换为字符串
                    train_data_clean[col] = train_data_clean[col].astype(str)

        X_train = train_data_clean[feature_cols]
        y_train = train_data_clean['Label']
        sample_weights = train_data_clean['sample_weight'].values if 'sample_weight' in train_data_clean.columns else None

        print(f"  训练样本: {len(X_train)}")

        # 训练模型
        print(f"  🔄 训练模型...")
        from catboost import CatBoostClassifier

        catboost_model = CatBoostClassifier(
            iterations=400,
            depth=8,
            learning_rate=0.06,
            l2_leaf_reg=2,
            random_seed=42,
            verbose=0,
            cat_features=cat_features if cat_features else None,
        )

        if sample_weights is not None:
            catboost_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            catboost_model.fit(X_train, y_train)

        print(f"  ✅ 模型训练完成")

        # 准备测试数据
        print(f"  🔄 准备测试数据...")
        test_data = model.prepare_data(
            self.stock_list,
            start_date=pd.to_datetime(test_start).tz_localize('UTC'),
            end_date=pd.to_datetime(test_end).tz_localize('UTC'),
            horizon=self.horizon,
            mode='backtest'
        )

        if test_data is None or len(test_data) < 10:
            raise ValueError(f"测试数据不足: {len(test_data) if test_data is not None else 0}")

        print(f"  ✅ 测试数据准备完成: {len(test_data)} 条记录")

        # 准备测试数据
        # 只删除标签缺失的行，特征缺失用 fillna 处理
        test_data_clean = test_data.dropna(subset=['Label'])

        # 对特征缺失值填充（使用训练集的中位数，如果没有则用0）
        for col in feature_cols:
            if col in test_data_clean.columns:
                if test_data_clean[col].isna().any():
                    # 检查是否为数值型
                    if pd.api.types.is_numeric_dtype(test_data_clean[col]):
                        # 尝试使用训练集的中位数
                        if col in train_data_clean.columns and pd.api.types.is_numeric_dtype(train_data_clean[col]):
                            median_val = train_data_clean[col].median()
                            if pd.isna(median_val):
                                median_val = 0
                        else:
                            median_val = 0
                        test_data_clean[col] = test_data_clean[col].fillna(median_val)
                    else:
                        # 非数值型，用 'unknown' 填充
                        test_data_clean[col] = test_data_clean[col].fillna('unknown')

        # 处理分类特征：转换为字符串
        for idx in cat_features:
            col = feature_cols[idx]
            if col in test_data_clean.columns:
                test_data_clean[col] = test_data_clean[col].astype(str)

        X_test = test_data_clean[feature_cols]
        y_test = test_data_clean['Label']

        print(f"  测试样本: {len(X_test)}")

        # 预测
        print(f"  🔄 生成预测...")
        y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 构建预测 DataFrame
        predictions = pd.DataFrame({
            'prediction': y_pred,
            'probability': y_pred_proba,
        }, index=test_data_clean.index)

        # 添加 Code 列
        if 'Stock_Code' in test_data_clean.columns:
            predictions['Code'] = test_data_clean['Stock_Code'].values

        # 应用市场情绪过滤器
        if self.use_market_filter:
            predictions = self._apply_market_filter(test_data_clean, predictions)

        # 计算指标
        print(f"  🔄 计算评估指标...")
        metrics = self._calculate_metrics(test_data_clean, predictions, y_test, y_pred)

        # 添加fold信息
        metrics['fold'] = fold + 1
        metrics['train_start'] = train_start
        metrics['train_end'] = train_end
        metrics['test_start'] = test_start
        metrics['test_end'] = test_end
        metrics['num_train_samples'] = len(X_train)
        metrics['num_features'] = len(feature_cols)

        # 特征重要性
        feature_importance = dict(zip(feature_cols, catboost_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics['top_features'] = top_features

        return metrics

    def _apply_market_filter(self, test_data, predictions):
        """
        应用市场情绪过滤器

        使用所有股票的收益率计算上涨比例，动态调整预测阈值。

        Args:
            test_data: 测试数据
            predictions: 预测结果

        Returns:
            pd.DataFrame: 过滤后的预测结果
        """
        print(f"  🔄 应用市场情绪过滤器...")

        # 初始化市场过滤器（首次调用时）
        if self.market_filter is None:
            from ml_services.market_regime import MarketSentimentFilter
            self.market_filter = MarketSentimentFilter(lookback_days=1)

        # 从测试数据中提取收益率
        if 'Return_1d' not in test_data.columns:
            # 计算收益率
            test_data = test_data.copy()
            if 'Close' in test_data.columns:
                test_data['Return_1d'] = test_data['Close'].pct_change()
            else:
                logger.warning("无法计算收益率，跳过市场情绪过滤")
                predictions['filtered_signal'] = predictions['prediction']
                return predictions

        # 准备市场收益率数据
        returns_df = test_data[['Return_1d']].reset_index()
        returns_df.columns = ['Date', 'Return_1d']

        # 计算每日上涨比例
        up_ratio_daily = returns_df.groupby('Date')['Return_1d'].apply(lambda x: (x > 0).mean())

        if len(up_ratio_daily) > 0:
            print(f"     平均上涨比例: {up_ratio_daily.mean()*100:.1f}%")

            # 准备市场过滤器数据
            self.market_filter.prepare_market_schedule(
                returns_df,
                date_col='Date',
                ret_col='Return_1d'
            )

            # 应用过滤
            pred_df = predictions.copy()
            pred_df['Date'] = predictions.index
            pred_df['Predict_Prob'] = predictions['probability']
            pred_df['Predict_Direction'] = predictions['prediction'].map({1: 'UP', 0: 'DOWN'})

            filtered_df = self.market_filter.apply_filter(
                pred_df,
                date_col='Date',
                prob_col='Predict_Prob',
                direction_col='Predict_Direction'
            )

            # 更新预测结果
            predictions['filtered_signal'] = filtered_df['filtered_signal'].values
            predictions['market_layer'] = filtered_df['market_layer'].values
            predictions['dynamic_threshold'] = filtered_df['dynamic_threshold'].values

            # 统计过滤效果
            original_signals = (predictions['prediction'] == 1).sum()
            filtered_signals = predictions['filtered_signal'].sum()
            reduction_pct = (original_signals - filtered_signals) / original_signals if original_signals > 0 else 0

            print(f"  ✅ 市场情绪过滤完成")
            print(f"     原始信号: {original_signals}, 过滤后: {filtered_signals}, 减少: {reduction_pct:.1%}")
        else:
            predictions['filtered_signal'] = predictions['prediction']

        return predictions

    def _calculate_metrics(self, test_data, predictions, y_test, y_pred):
        """
        计算多维度评估指标

        Args:
            test_data: 测试数据
            predictions: 预测结果
            y_test: 真实标签
            y_pred: 预测标签

        Returns:
            dict: 评估指标
        """
        from sklearn.metrics import accuracy_score

        # 合并数据
        df = test_data.copy()
        df['prediction'] = predictions['prediction'].values
        df['probability'] = predictions['probability'].values

        # 使用过滤后的信号（如果有）
        if 'filtered_signal' in predictions.columns:
            df['signal'] = predictions['filtered_signal'].values
        else:
            df['signal'] = df['prediction']

        # 计算实际收益率
        if 'Future_Return' in df.columns:
            df['actual_return'] = df['Future_Return']
        else:
            # 使用 Close 计算未来收益
            df['actual_return'] = df['Close'].shift(-self.horizon) / df['Close'] - 1

        # 准确率
        accuracy = accuracy_score(y_test, y_pred)

        # 计算 IC（预测概率与实际收益率的相关性）
        valid_mask = df['actual_return'].notna() & df['probability'].notna()
        if valid_mask.sum() > 1:
            ic = df.loc[valid_mask, 'probability'].corr(df.loc[valid_mask, 'actual_return'])
            rank_ic = df.loc[valid_mask, 'probability'].rank().corr(df.loc[valid_mask, 'actual_return'].rank())
        else:
            ic = 0.0
            rank_ic = 0.0

        # 计算交易收益
        trades = df[df['signal'] == 1].copy()

        if len(trades) > 0:
            avg_return = trades['actual_return'].mean()
            win_rate = (trades['actual_return'] > 0).mean()
            return_std = trades['actual_return'].std()

            # 夏普比率（年化）
            if return_std > 0:
                sharpe_ratio = avg_return / return_std * np.sqrt(252 / self.horizon)
            else:
                sharpe_ratio = 0.0

            # 最大回撤
            cumulative = (1 + trades['actual_return']).cumprod()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            avg_return = 0.0
            win_rate = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            return_std = 0.0

        return {
            'num_samples': len(y_test),
            'num_trades': len(trades) if 'trades' in dir() else 0,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'ic': ic,
            'rank_ic': rank_ic,
            'return_std': return_std,
        }

    def _calculate_overall_metrics(self, fold_results):
        """计算整体指标"""
        if not fold_results:
            return {}

        accuracies = [r['accuracy'] for r in fold_results]
        avg_returns = [r['avg_return'] for r in fold_results]
        win_rates = [r['win_rate'] for r in fold_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in fold_results]
        ics = [r.get('ic', 0) for r in fold_results]

        return {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'avg_return': np.mean(avg_returns),
            'std_return': np.std(avg_returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_ic': np.mean(ics),
            'num_folds': len(fold_results),
            'total_samples': sum(r['num_samples'] for r in fold_results),
        }

    def _print_overall_results(self, overall):
        """打印整体结果"""
        print(f"\n{'='*60}")
        print("📊 Walk-forward 验证结果")
        print(f"{'='*60}")

        if not overall:
            print("无有效结果")
            return

        print(f"\n综合指标:")
        print(f"  Fold 数量: {overall['num_folds']}")
        print(f"  总样本数: {overall['total_samples']}")
        print(f"  平均准确率: {overall['avg_accuracy']:.2%} (±{overall['std_accuracy']:.2%})")
        print(f"  平均胜率: {overall['avg_win_rate']:.2%}")
        print(f"  平均收益率: {overall['avg_return']:.2%}")
        print(f"  平均夏普比率: {overall['avg_sharpe_ratio']:.4f}")
        print(f"  平均 IC: {overall['avg_ic']:.4f}")

        # 评估模型可信度
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

    def _save_report(self, report):
        """保存验证报告"""
        os.makedirs('data/a_stock_walk_forward', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'data/a_stock_walk_forward/validation_report_{timestamp}.json'

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📄 报告已保存: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='A股 Walk-forward 验证')
    parser.add_argument('--horizon', type=int, default=20, choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月（默认）')
    parser.add_argument('--train-window', type=int, default=12,
                       help='训练窗口（月），默认12个月')
    parser.add_argument('--folds', type=int, default=6,
                       help='Fold数量，默认6')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='验证开始日期')
    parser.add_argument('--end-date', type=str, default='2026-07-01',
                       help='验证结束日期')
    parser.add_argument('--no-market-filter', action='store_true',
                       help='禁用市场情绪过滤器')

    args = parser.parse_args()

    # 创建验证器
    validator = AStockWalkForwardValidator(
        horizon=args.horizon,
        train_window_months=args.train_window,
        use_market_filter=not args.no_market_filter,
    )

    # 执行验证
    validator.validate(
        start_date=args.start_date,
        end_date=args.end_date,
        num_folds=args.folds,
    )


if __name__ == '__main__':
    main()
