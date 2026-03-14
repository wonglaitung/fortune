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

        print(f"  ✅ 训练数据准备完成: {len(train_data)} 条记录")

        # 训练模型（关键：每个fold重新训练）
        print(f"  🔄 训练模型 (Fold {fold + 1})...")
        try:
            model.train(
                train_codes,
                start_date=train_start_date,
                end_date=train_end_date,
                horizon=self.horizon,
                use_feature_selection=self.use_feature_selection,
                skip_feature_selection=True  # 跳过特征选择，使用已有的特征
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

        # 生成预测
        print(f"  🔄 生成预测...")
        predictions = model.predict(
            train_codes,
            start_date=test_start_date,
            end_date=test_end_date,
            horizon=self.horizon
        )

        print(f"  ✅ 预测生成完成")

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
        # 合并测试数据和预测结果
        if predictions is None or predictions.empty:
            raise ValueError("预测结果为空")

        # 合并数据
        df = test_data.merge(predictions, on=['Code', 'Date'], how='inner')

        if df.empty:
            raise ValueError("合并后的数据为空")

        # 计算实际收益率
        df['actual_return'] = df['Close'].pct_change().shift(-self.horizon)

        # 计算预测收益
        df['predicted_return'] = df['actual_return'] * df['prediction']

        # 计算交易信号
        df['signal'] = (df['prediction'] >= self.confidence_threshold).astype(int)

        # 计算收益率（仅在有信号时）
        df['strategy_return'] = df['signal'] * df['actual_return']

        # 移除NaN
        df = df.dropna(subset=['actual_return', 'strategy_return'])

        if len(df) == 0:
            raise ValueError("没有有效的样本")

        # 基础指标
        avg_return = df['strategy_return'].mean()
        cumulative_return = (1 + df['strategy_return']).prod() - 1
        win_rate = (df['strategy_return'] > 0).sum() / len(df)
        accuracy = (df['prediction'] == df['Label']).sum() / len(df)
        num_samples = len(df)

        # 风险指标
        return_std = df['strategy_return'].std()
        annualized_return = avg_return * (252 / self.horizon)
        annualized_std = return_std * np.sqrt(252 / self.horizon)

        # 计算最大回撤
        cumulative_returns = (1 + df['strategy_return']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 风险调整收益指标
        # 夏普比率
        if return_std > 0:
            sharpe_ratio = annualized_return / annualized_std
        else:
            sharpe_ratio = 0.0

        # 索提诺比率（只考虑下行风险）
        downside_returns = df['strategy_return'][df['strategy_return'] < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = annualized_return / (downside_std * np.sqrt(252 / self.horizon))
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0.0

        # 信息比率（相对于基准：买入持有策略）
        benchmark_return = df['actual_return'].mean()
        excess_returns = df['strategy_return'] - df['actual_return']
        tracking_error = excess_returns.std()
        if tracking_error > 0:
            information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252 / self.horizon)
        else:
            information_ratio = 0.0

        return {
            'num_samples': num_samples,
            'avg_return': avg_return,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'return_std': return_std,
            'annualized_return': annualized_return,
            'annualized_std': annualized_std,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_return
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

        # 稳定性评级
        if return_std < 0.02:
            stability_rating = "高（优秀）"
        elif return_std < 0.05:
            stability_rating = "中（良好）"
        else:
            stability_rating = "低（需改进）"

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
            'stability_rating': stability_rating
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
            f.write(f"- **Fold数量**: {overall['num_folds']}\n")
            f.write(f"- **平均收益率**: {overall['avg_return']:.2%}\n")
            f.write(f"- **平均胜率**: {overall['avg_win_rate']:.2%}\n")
            f.write(f"- **平均准确率**: {overall['avg_accuracy']:.2%}\n")
            f.write(f"- **平均夏普比率**: {overall['avg_sharpe_ratio']:.4f}\n")
            f.write(f"- **平均最大回撤**: {overall['avg_max_drawdown']:.2%}\n")
            f.write(f"- **平均索提诺比率**: {overall['avg_sortino_ratio']:.4f}\n")
            f.write(f"- **平均信息比率**: {overall['avg_information_ratio']:.4f}\n\n")

            # 稳定性分析
            f.write("## 🔬 稳定性分析\n\n")
            f.write(f"- **收益率标准差**: {overall['return_std']:.2%}\n")
            f.write(f"- **收益率范围**: {overall['return_range']:.2%}\n")
            f.write(f"- **胜率标准差**: {overall['win_rate_std']:.2%}\n")
            f.write(f"- **夏普比率标准差**: {overall['sharpe_std']:.4f}\n")
            f.write(f"- **稳定性评级**: {overall['stability_rating']}\n\n")

            # Fold详细结果
            f.write("## 📈 Fold 详细结果\n\n")
            f.write("| Fold | 训练期间 | 测试期间 | 样本数 | 收益率 | 胜率 | 准确率 | 夏普比率 | 最大回撤 |\n")
            f.write("|------|---------|---------|-------|-------|------|-------|---------|--------|\n")

            for fold in folds:
                f.write(f"| {fold['fold'] + 1} | {fold['train_start_date']} 至 {fold['train_end_date']} | "
                       f"{fold['test_start_date']} 至 {fold['test_end_date']} | {fold['num_test_samples']} | "
                       f"{fold['avg_return']:.2%} | {fold['win_rate']:.2%} | {fold['accuracy']:.2%} | "
                       f"{fold['sharpe_ratio']:.4f} | {fold['max_drawdown']:.2%} |\n")

            f.write("\n")

            # 结论
            f.write("## 💡 结论\n\n")
            if overall['stability_rating'] == "高（优秀）":
                f.write(f"✅ 模型表现**优秀**，稳定性高，适合实盘交易。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n\n")
            elif overall['stability_rating'] == "中（良好）":
                f.write(f"⚠️ 模型表现**良好**，但需要优化稳定性。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n")
                f.write(f"- 建议：增加正则化，降低模型复杂度\n\n")
            else:
                f.write(f"❌ 模型表现**不佳**，需要重新训练或调整参数。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n")
                f.write(f"- 建议：重新训练模型，增加特征工程，调整超参数\n\n")

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

    # 特征参数
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择')

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
        use_feature_selection=args.use_feature_selection
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