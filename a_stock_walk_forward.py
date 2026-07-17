#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股 Walk-forward 验证 - 模型预测能力验证

功能：
- 业界标准的 Walk-forward 验证方法
- A股特有特征：涨跌停限制、北向资金
- 多维度评估指标

使用方法：
  python3 a_stock_walk_forward.py --horizon 20
  python3 a_stock_walk_forward.py --horizon 20 --train-window 6 --folds 6
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
from data_services.a_stock_data import get_a_stock_data, get_index_data

# 常量
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_YEAR = 240


class AStockWalkForwardValidator:
    """A股 Walk-forward 验证器"""

    def __init__(
        self,
        horizon: int = 20,
        train_window_months: int = 6,
        test_window_months: int = 1,
        confidence_threshold: float = 0.50,
    ):
        """
        初始化验证器

        Args:
            horizon: 预测周期（天）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            confidence_threshold: 置信度阈值
        """
        self.horizon = horizon
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.confidence_threshold = confidence_threshold
        self.stock_list = list(A_STOCK_TRAINING_LIST.keys())

        print(f"\n{'='*60}")
        print("🔬 A股 Walk-forward 验证器")
        print(f"{'='*60}")
        print(f"预测周期: {horizon} 天")
        print(f"训练窗口: {train_window_months} 个月")
        print(f"测试窗口: {test_window_months} 个月")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"股票数量: {len(self.stock_list)}")

    def prepare_data(self, stock_code, start_date=None, end_date=None):
        """
        准备股票数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 股票数据
        """
        # 获取足够长的历史数据（不使用缓存，直接获取）
        period_days = 500  # 约2年数据
        df = get_a_stock_data(stock_code, period_days=period_days, use_cache=False)

        if df is None or df.empty:
            return None

        # 确保索引是datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除时区信息
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 过滤日期范围
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def calculate_features(self, df, stock_code):
        """
        计算技术特征

        Args:
            df: 股票数据
            stock_code: 股票代码

        Returns:
            DataFrame: 特征数据
        """
        df = df.copy()

        # 收益率
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)

        # 移动平均
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        # 移动平均比率
        df['MA5_Ratio'] = df['Close'] / df['MA5'] - 1
        df['MA10_Ratio'] = df['Close'] / df['MA10'] - 1
        df['MA20_Ratio'] = df['Close'] / df['MA20'] - 1

        # 波动率
        df['Volatility_5d'] = df['Return_1d'].rolling(5).std()
        df['Volatility_10d'] = df['Return_1d'].rolling(10).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(20).std()

        # 成交量比率
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5'] - 1

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 布林带
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        df['BB_Ratio'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # A股特有：涨跌停
        limit_rate = get_limit_rate(stock_code)
        df['High_Limit'] = df['Close'].shift(1) * (1 + limit_rate)
        df['Low_Limit'] = df['Close'].shift(1) * (1 - limit_rate)
        df['Limit_Up'] = (df['Close'] >= df['High_Limit'] * 0.995).astype(int)
        df['Limit_Down'] = (df['Close'] <= df['Low_Limit'] * 1.005).astype(int)

        # 未来收益（标签）
        df['Future_Return'] = df['Close'].shift(-self.horizon) / df['Close'] - 1

        # 标签：未来收益 > 0 为上涨
        df['Label'] = (df['Future_Return'] > 0).astype(int)

        return df

    def get_feature_columns(self):
        """获取特征列名"""
        return [
            'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
            'MA5_Ratio', 'MA10_Ratio', 'MA20_Ratio',
            'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
            'Volume_Ratio',
            'RSI_14',
            'MACD', 'MACD_Signal',
            'BB_Ratio',
            'Limit_Up', 'Limit_Down',
        ]

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
                print(f"  准确率: {fold_result['accuracy']:.2%}")
                print(f"  胜率: {fold_result['win_rate']:.2%}")
                print(f"  平均收益率: {fold_result['avg_return']:.2%}")

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
        验证单个fold

        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            test_end: 测试结束日期
            fold: fold编号

        Returns:
            dict: fold结果
        """
        from catboost import CatBoostClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # 收集训练和测试数据
        train_data_list = []
        test_data_list = []

        for stock_code in self.stock_list:
            # 获取数据
            df = self.prepare_data(stock_code)
            if df is None or len(df) < 200:
                continue

            # 计算特征
            df = self.calculate_features(df, stock_code)

            # 分割训练和测试
            df_train = df[(df.index >= train_start) & (df.index <= train_end)]
            df_test = df[(df.index >= test_start) & (df.index <= test_end)]

            if len(df_train) > 50 and len(df_test) > 5:
                train_data_list.append(df_train)
                test_data_list.append(df_test)

        if not train_data_list:
            raise ValueError("训练数据不足")

        # 合并数据
        train_df = pd.concat(train_data_list, ignore_index=False)
        test_df = pd.concat(test_data_list, ignore_index=False)

        # 准备特征和标签
        feature_cols = self.get_feature_columns()
        train_df = train_df.dropna(subset=feature_cols + ['Label'])
        test_df = test_df.dropna(subset=feature_cols + ['Label'])

        X_train = train_df[feature_cols]
        y_train = train_df['Label']
        X_test = test_df[feature_cols]
        y_test = test_df['Label']

        print(f"  训练样本: {len(X_train)}")
        print(f"  测试样本: {len(X_test)}")

        # 训练模型
        model = CatBoostClassifier(
            iterations=400,
            depth=8,
            learning_rate=0.06,
            l2_leaf_reg=2,
            random_seed=42,
            verbose=0,
        )

        model.fit(X_train, y_train)

        # 预测
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.confidence_threshold).astype(int)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)

        # 计算交易收益
        test_df_copy = test_df.copy()
        test_df_copy['Pred_Proba'] = y_pred_proba
        test_df_copy['Pred'] = y_pred

        # 只在预测上涨且概率超过阈值时买入
        trades = test_df_copy[test_df_copy['Pred'] == 1]
        if len(trades) > 0:
            avg_return = trades['Future_Return'].mean()
            win_rate = (trades['Future_Return'] > 0).mean()
        else:
            avg_return = 0
            win_rate = 0

        # 特征重要性
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'fold': fold + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'num_samples': len(X_test),
            'num_trades': len(trades) if 'trades' in dir() else 0,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'top_features': top_features,
        }

    def _calculate_overall_metrics(self, fold_results):
        """计算整体指标"""
        if not fold_results:
            return {}

        accuracies = [r['accuracy'] for r in fold_results]
        avg_returns = [r['avg_return'] for r in fold_results]
        win_rates = [r['win_rate'] for r in fold_results]

        return {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'avg_return': np.mean(avg_returns),
            'std_return': np.std(avg_returns),
            'avg_win_rate': np.mean(win_rates),
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
    parser.add_argument('--train-window', type=int, default=6,
                       help='训练窗口（月），默认6个月')
    parser.add_argument('--folds', type=int, default=6,
                       help='Fold数量，默认6')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='验证开始日期')
    parser.add_argument('--end-date', type=str, default='2026-07-01',
                       help='验证结束日期')

    args = parser.parse_args()

    # 创建验证器
    validator = AStockWalkForwardValidator(
        horizon=args.horizon,
        train_window_months=args.train_window,
    )

    # 执行验证
    validator.validate(
        start_date=args.start_date,
        end_date=args.end_date,
        num_folds=args.folds,
    )


if __name__ == '__main__':
    main()
