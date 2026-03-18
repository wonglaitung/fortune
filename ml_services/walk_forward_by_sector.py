#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按板块进行 Walk-forward 验证（方案1：独立板块验证）

功能：
- 为每个板块独立进行Walk-forward验证
- 每个板块独立训练模型，评估模型在不同板块的表现
- 生成每个板块的独立验证报告
- 汇总所有板块的对比分析

优势：
- 独立验证：每个板块独立训练，避免板块间相互干扰
- 精准分析：识别模型在不同板块的强项和弱项
- 投资指导：为板块配置提供数据支持

使用方法：
  python3 ml_services/walk_forward_by_sector.py --sector bank
  python3 ml_services/walk_forward_by_sector.py --sectors bank tech semiconductor
  python3 ml_services/walk_forward_by_sector.py --all-sectors
"""

import warnings
import os
import sys
import argparse
from datetime import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 导入项目模块
from ml_services.ml_trading_model import CatBoostModel, FeatureEngineer
from ml_services.logger_config import get_logger
from config import STOCK_SECTOR_MAPPING, SECTOR_NAME_MAPPING

# 获取日志记录器
logger = get_logger('walk_forward_by_sector')


class SectorWalkForwardValidator:
    """按板块进行Walk-forward验证"""

    def __init__(
        self,
        model_type: str = 'catboost',
        train_window_months: int = 12,
        test_window_months: int = 1,
        step_window_months: int = 1,
        horizon: int = 20,
        confidence_threshold: float = 0.55,
        use_feature_selection: bool = True,
        min_train_samples: int = 100,
        class_weight='balanced'
    ):
        """
        初始化板块Walk-forward验证器

        Args:
            model_type: 模型类型（默认catboost）
            train_window_months: 训练窗口（月）
            test_window_months: 测试窗口（月）
            step_window_months: 滚动步长（月）
            horizon: 预测周期（天）
            confidence_threshold: 置信度阈值
            use_feature_selection: 是否使用特征选择
            min_train_samples: 最小训练样本数
            class_weight: 类别权重策略（'balanced', None, 或字典如{0:1.0, 1:1.5}）
        """
        self.model_type = model_type.lower()
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.horizon = horizon
        self.confidence_threshold = confidence_threshold
        self.use_feature_selection = use_feature_selection
        self.min_train_samples = min_train_samples
        self.class_weight = class_weight

        self.feature_engineer = FeatureEngineer()

        logger.info(f"初始化板块Walk-forward验证器")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"训练窗口: {train_window_months} 个月")
        logger.info(f"测试窗口: {test_window_months} 个月")
        logger.info(f"滚动步长: {step_window_months} 个月")

    def validate_sector(self, sector_code, stock_list, start_date, end_date):
        """
        验证单个板块

        Args:
            sector_code: 板块代码
            stock_list: 该板块的股票代码列表
            start_date: 验证开始日期
            end_date: 验证结束日期

        Returns:
            dict: 板块验证结果
        """
        print("\n" + "="*80)
        print(f"🔬 开始验证板块: {SECTOR_NAME_MAPPING.get(sector_code, sector_code)} ({sector_code})")
        print("="*80)
        print(f"股票数量: {len(stock_list)}")
        print(f"股票列表: {', '.join(stock_list)}")
        print("="*80)

        # 生成所有月份
        all_months = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        while current <= end:
            all_months.append(current.strftime('%Y-%m'))
            current = current + pd.DateOffset(months=1)

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

            # 计算日期范围
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
                print(f"  买入信号数: {fold_result['num_buy_signals']}")
                print(f"  平均收益率: {fold_result['avg_return']:.2%}")
                print(f"  买入信号胜率: {fold_result['win_rate']:.2%}")
                print(f"  正确决策比例: {fold_result['correct_decision_ratio']:.2%}")
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

        # 生成板块报告
        sector_report = {
            'sector_code': sector_code,
            'sector_name': SECTOR_NAME_MAPPING.get(sector_code, sector_code),
            'stock_list': stock_list,
            'num_stocks': len(stock_list),
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
                'num_folds': num_folds
            },
            'fold_results': all_fold_results,
            'overall_metrics': overall_result
        }

        print(f"\n{'='*80}")
        print(f"📊 板块 {SECTOR_NAME_MAPPING.get(sector_code, sector_code)} 验证完成")
        print(f"{'='*80}")
        print(f"总 Fold 数: {len(all_fold_results)}")
        
        if len(all_fold_results) == 0:
            print("⚠️  所有 Fold 验证失败，无法计算整体指标")
        else:
            print(f"平均收益率: {overall_result['avg_return']:.2%}")
            print(f"买入信号胜率: {overall_result['avg_win_rate']:.2%}")
            print(f"正确决策比例: {overall_result['avg_correct_decision_ratio']:.2%}")
            print(f"平均准确率: {overall_result['avg_accuracy']:.2%}")
            print(f"平均夏普比率: {overall_result['avg_sharpe_ratio']:.4f}")
            print(f"平均最大回撤: {overall_result['avg_max_drawdown']:.2%}")
            print(f"收益率标准差: {overall_result['return_std']:.2%}")
            print(f"稳定性评级: {overall_result['stability_rating']}")
        print(f"{'='*80}")

        return sector_report

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

        # 创建模型实例（传入类别权重参数）
        model = CatBoostModel(class_weight=self.class_weight)

        # 准备训练数据
        train_data = model.prepare_data(
            stock_list,
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
                stock_list,
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
            stock_list,
            start_date=test_start_date,
            end_date=test_end_date,
            horizon=self.horizon,
            for_backtest=False
        )

        print(f"  ✅ 测试数据准备完成: {len(test_data)} 条记录")

        # 生成预测
        print(f"  🔄 生成预测...")
        X_test = test_data[model.feature_columns]
        prediction_proba = model.predict_proba(X_test)
        
        # 根据置信度阈值转换为预测标签
        predictions = pd.DataFrame({
            'prediction': (prediction_proba[:, 1] >= self.confidence_threshold).astype(int),
            'probability': prediction_proba[:, 1]
        }, index=test_data.index)
        
        # 添加 Code 列
        if 'Code' in test_data.columns:
            predictions['Code'] = test_data['Code'].values
        else:
            predictions['Code'] = [stock_list[0]] * len(predictions)
        
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
        """计算多维度评估指标"""
        if predictions is None or predictions.empty:
            raise ValueError("预测结果为空")

        df = test_data.copy()
        df['prediction'] = predictions['prediction']
        df['probability'] = predictions['probability']
        
        df = df[df.index.isin(predictions.index)]

        if df.empty:
            raise ValueError("合并后的数据为空")

        # 计算实际收益率
        df['actual_return'] = df['Close'].pct_change().shift(-self.horizon)

        # 计算预测收益
        df['predicted_return'] = df['actual_return'] * df['prediction']

        # 计算交易信号
        df['signal'] = (df['prediction'] >= self.confidence_threshold).astype(int)

        # 计算收益率
        df['strategy_return'] = df['signal'] * df['actual_return']

        # 移除NaN
        df = df.dropna(subset=['actual_return', 'strategy_return'])

        if len(df) == 0:
            raise ValueError("没有有效的样本")

        # 基础指标
        avg_return = df['strategy_return'].mean()
        cumulative_return = (1 + df['strategy_return']).prod() - 1
        accuracy = (df['prediction'] == df['Label']).sum() / len(df)
        num_samples = len(df)

        # 买入信号相关指标
        buy_signals = df[df['signal'] == 1]
        num_buy_signals = len(buy_signals)
        if num_buy_signals > 0:
            # 买入信号胜率：盈利买入数 / 买入信号总数
            win_rate = (buy_signals['strategy_return'] > 0).sum() / num_buy_signals
            buy_signal_return = buy_signals['strategy_return'].mean()
        else:
            win_rate = 0.0
            buy_signal_return = 0.0

        # 正确决策比例 = (盈利买入数 + 正确不买入数) / 总决策数
        correct_buys = ((df['signal'] == 1) & (df['strategy_return'] > 0)).sum()
        correct_no_buys = ((df['signal'] == 0) & (df['actual_return'] < 0)).sum()
        correct_decision_ratio = (correct_buys + correct_no_buys) / len(df) if len(df) > 0 else 0.0

        # 风险指标
        return_std = df['strategy_return'].std()
        annualized_return = avg_return * (252 / self.horizon)
        annualized_std = return_std * np.sqrt(252 / self.horizon)

        # 计算最大回撤
        cumulative_returns = (1 + df['strategy_return']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 夏普比率
        if return_std > 0:
            sharpe_ratio = annualized_return / annualized_std
        else:
            sharpe_ratio = 0.0

        # 索提诺比率
        downside_returns = df['strategy_return'][df['strategy_return'] < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = annualized_return / (downside_std * np.sqrt(252 / self.horizon))
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0.0

        # 信息比率
        benchmark_return = df['actual_return'].mean()
        excess_returns = df['strategy_return'] - df['actual_return']
        tracking_error = excess_returns.std()
        if tracking_error > 0:
            information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252 / self.horizon)
        else:
            information_ratio = 0.0

        return {
            'num_samples': num_samples,
            'num_buy_signals': num_buy_signals,
            'avg_return': avg_return,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'buy_signal_return': buy_signal_return,
            'accuracy': accuracy,
            'correct_decision_ratio': correct_decision_ratio,
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
        """计算整体指标"""
        if not fold_results:
            return {}

        # 计算平均指标
        avg_return = np.mean([r['avg_return'] for r in fold_results])
        avg_win_rate = np.mean([r['win_rate'] for r in fold_results])
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_correct_decision_ratio = np.mean([r['correct_decision_ratio'] for r in fold_results])
        avg_sharpe_ratio = np.mean([r['sharpe_ratio'] for r in fold_results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in fold_results])
        avg_sortino_ratio = np.mean([r['sortino_ratio'] for r in fold_results])
        avg_information_ratio = np.mean([r['information_ratio'] for r in fold_results])

        # 计算年化收益率（基于平均收益率）
        annualized_return = avg_return * (252 / self.horizon)

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
            'annualized_return': annualized_return,
            'avg_win_rate': avg_win_rate,
            'avg_accuracy': avg_accuracy,
            'avg_correct_decision_ratio': avg_correct_decision_ratio,
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

    def save_sector_report(self, sector_report, output_dir='output'):
        """保存板块验证报告"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sector_code = sector_report['sector_code']
        model_type = sector_report['validation_config']['model_type']
        horizon = sector_report['validation_config']['horizon']

        # 1. 保存JSON格式
        json_file = os.path.join(output_dir, f'walk_forward_sector_{sector_code}_{model_type}_{horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sector_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"板块JSON报告已保存: {json_file}")

        # 2. 保存CSV格式
        csv_file = os.path.join(output_dir, f'walk_forward_sector_{sector_code}_{model_type}_{horizon}d_{timestamp}.csv')
        fold_df = pd.DataFrame(sector_report['fold_results'])
        fold_df.to_csv(csv_file, index=False)
        logger.info(f"板块CSV报告已保存: {csv_file}")

        # 3. 保存Markdown格式
        md_file = os.path.join(output_dir, f'walk_forward_sector_{sector_code}_{model_type}_{horizon}d_{timestamp}.md')
        self._generate_sector_markdown_report(sector_report, md_file)
        logger.info(f"板块Markdown报告已保存: {md_file}")

        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'md_file': md_file
        }

    def _generate_sector_markdown_report(self, sector_report, output_file):
        """生成板块Markdown格式报告"""
        config = sector_report['validation_config']
        overall = sector_report['overall_metrics']
        folds = sector_report['fold_results']
        sector_name = sector_report['sector_name']
        sector_code = sector_report['sector_code']

        with open(output_file, 'w', encoding='utf-8') as f:
            # 标题
            f.write(f"# 板块Walk-forward验证报告：{sector_name}\n\n")
            f.write(f"**板块代码**: {sector_code}\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 股票列表
            f.write("## 📋 板块股票列表\n\n")
            f.write(f"**股票数量**: {sector_report['num_stocks']}\n\n")
            f.write("| 股票代码 | 股票名称 |\n")
            f.write("|---------|---------|\n")
            for code in sector_report['stock_list']:
                name = STOCK_SECTOR_MAPPING.get(code, {}).get('name', code)
                f.write(f"| {code} | {name} |\n")
            f.write("\n")

            # 配置信息
            f.write("## 🔬 验证配置\n\n")
            f.write(f"- **模型类型**: {config['model_type'].upper()}\n")
            f.write(f"- **训练窗口**: {config['train_window_months']} 个月\n")
            f.write(f"- **测试窗口**: {config['test_window_months']} 个月\n")
            f.write(f"- **滚动步长**: {config['step_window_months']} 个月\n")
            f.write(f"- **预测周期**: {config['horizon']} 天\n")
            f.write(f"- **置信度阈值**: {config['confidence_threshold']}\n")
            f.write(f"- **特征选择**: {'是' if config['use_feature_selection'] else '否'}\n")
            f.write(f"- **验证日期**: {config['start_date']} 至 {config['end_date']}\n")
            f.write(f"- **Fold数量**: {config['num_folds']}\n\n")

            # 整体指标
            f.write("## 📊 整体性能指标\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法计算整体指标\n\n")
            else:
                f.write(f"- **Fold数量**: {overall['num_folds']}\n")
                f.write(f"- **年化收益率**: {overall['avg_return'] * (252 / config['horizon']):.2%}\n")
                f.write(f"- **买入信号胜率**: {overall['avg_win_rate']:.2%} （仅统计实际买入的交易）\n")
                f.write(f"- **平均准确率**: {overall['avg_accuracy']:.2%}\n")
                f.write(f"- **正确决策比例**: {overall['avg_correct_decision_ratio']:.2%} （盈利买入 + 正确不买入）\n")
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
            f.write("| Fold | 训练期间 | 测试期间 | 样本数 | 买入信号 | 收益率 | 买入胜率 | 准确率 | 正确决策率 | 夏普比率 |\n")
            f.write("|------|---------|---------|-------|---------|-------|---------|-------|-----------|---------|\n")

            for fold in folds:
                f.write(f"| {fold['fold'] + 1} | {fold['train_start_date']} 至 {fold['train_end_date']} | "
                       f"{fold['test_start_date']} 至 {fold['test_end_date']} | {fold['num_test_samples']} | "
                       f"{fold.get('num_buy_signals', 0)} | {fold['avg_return']:.2%} | {fold['win_rate']:.2%} | "
                       f"{fold['accuracy']:.2%} | {fold.get('correct_decision_ratio', 0):.2%} | {fold['sharpe_ratio']:.4f} |\n")

            f.write("\n")

            # 结论
            f.write("## 💡 结论\n\n")
            if not overall:
                f.write("⚠️  所有 Fold 验证失败，无法给出结论。\n\n")
            elif overall['stability_rating'] == "高（优秀）":
                f.write(f"✅ 模型在该板块表现**优秀**，稳定性高，适合实盘交易。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n\n")
            elif overall['stability_rating'] == "中（良好）":
                f.write(f"⚠️ 模型在该板块表现**良好**，但需要优化稳定性。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n\n")
            else:
                f.write(f"❌ 模型在该板块表现**不佳**，需要重新训练或调整参数。\n\n")
                f.write(f"- 平均收益率: {overall['avg_return']:.2%}\n")
                f.write(f"- 夏普比率: {overall['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 稳定性评级: {overall['stability_rating']}\n\n")

            f.write("---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    def generate_summary_report(self, all_sector_reports, output_dir='output'):
        """生成所有板块的汇总报告"""
        if not all_sector_reports:
            logger.warning("没有板块报告，无法生成汇总报告")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_file = os.path.join(output_dir, f'walk_forward_sectors_summary_{timestamp}.md')

        # 提取所有板块的整体指标
        sector_summary = []
        for report in all_sector_reports:
            if report.get('overall_metrics'):
                sector_summary.append({
                    'sector_code': report['sector_code'],
                    'sector_name': report['sector_name'],
                    'num_stocks': report['num_stocks'],
                    'num_folds': report['overall_metrics']['num_folds'],
                    'avg_return': report['overall_metrics']['avg_return'],
                    'annualized_return': report['overall_metrics']['annualized_return'],
                    'avg_win_rate': report['overall_metrics']['avg_win_rate'],
                    'avg_accuracy': report['overall_metrics']['avg_accuracy'],
                    'avg_correct_decision_ratio': report['overall_metrics']['avg_correct_decision_ratio'],
                    'avg_sharpe_ratio': report['overall_metrics']['avg_sharpe_ratio'],
                    'avg_max_drawdown': report['overall_metrics']['avg_max_drawdown'],
                    'return_std': report['overall_metrics']['return_std'],
                    'stability_rating': report['overall_metrics']['stability_rating']
                })

        # 排序
        sector_summary_df = pd.DataFrame(sector_summary)
        sector_summary_df = sector_summary_df.sort_values('avg_sharpe_ratio', ascending=False)

        with open(md_file, 'w', encoding='utf-8') as f:
            # 标题
            f.write(f"# 所有板块Walk-forward验证汇总报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**板块数量**: {len(sector_summary)}\n\n")

            # 配置信息
            if sector_summary:
                first_config = all_sector_reports[0]['validation_config']
                f.write("## 🔬 验证配置\n\n")
                f.write(f"- **模型类型**: {first_config['model_type'].upper()}\n")
                f.write(f"- **训练窗口**: {first_config['train_window_months']} 个月\n")
                f.write(f"- **测试窗口**: {first_config['test_window_months']} 个月\n")
                f.write(f"- **滚动步长**: {first_config['step_window_months']} 个月\n")
                f.write(f"- **预测周期**: {first_config['horizon']} 天\n")
                f.write(f"- **置信度阈值**: {first_config['confidence_threshold']}\n")
                f.write(f"- **特征选择**: {'是' if first_config['use_feature_selection'] else '否'}\n")
                f.write(f"- **验证日期**: {first_config['start_date']} 至 {first_config['end_date']}\n")
                f.write(f"- **指标说明**: 收益率为年化，胜率仅统计买入信号，正确决策比例 = (盈利买入 + 正确不买入) / 总决策\n\n")

            # 板块对比表格
            f.write("## 📊 板块性能对比\n\n")
            f.write("| 排名 | 板块 | 股票数 | Fold数 | 年化收益率 | 买入胜率 | 准确率 | 正确决策率 | 夏普比率 | 最大回撤 | 稳定性 |\n")
            f.write("|------|------|-------|-------|-----------|---------|-------|-----------|---------|--------|--------|\n")

            for idx, row in sector_summary_df.iterrows():
                f.write(f"| {int(idx) + 1} | {row['sector_name']} ({row['sector_code']}) | {row['num_stocks']} | "
                       f"{row['num_folds']} | {row['annualized_return']:.2%} | {row['avg_win_rate']:.2%} | "
                       f"{row['avg_accuracy']:.2%} | {row['avg_correct_decision_ratio']:.2%} | "
                       f"{row['avg_sharpe_ratio']:.4f} | {row['avg_max_drawdown']:.2%} | {row['stability_rating']} |\n")

            f.write("\n")

            # 排名分析
            f.write("## 🏆 板块排名分析\n\n")

            # TOP 5 夏普比率
            top_sharpe = sector_summary_df.nlargest(5, 'avg_sharpe_ratio')
            f.write("### 夏普比率 TOP 5\n\n")
            for idx, row in top_sharpe.iterrows():
                f.write(f"{int(idx) + 1}. **{row['sector_name']}**: 夏普比率 {row['avg_sharpe_ratio']:.4f}, "
                       f"收益率 {row['avg_return']:.2%}, 年化收益率 {row['annualized_return']:.2%}, 胜率 {row['avg_win_rate']:.2%}\n")
            f.write("\n")

            # TOP 5 收益率
            top_return = sector_summary_df.nlargest(5, 'avg_return')
            f.write("### 平均收益率 TOP 5\n\n")
            for idx, row in top_return.iterrows():
                f.write(f"{int(idx) + 1}. **{row['sector_name']}**: 收益率 {row['avg_return']:.2%}, 年化收益率 {row['annualized_return']:.2%}, "
                       f"夏普比率 {row['avg_sharpe_ratio']:.4f}, 胜率 {row['avg_win_rate']:.2%}\n")
            f.write("\n")

            # 稳定性分析
            high_stability = sector_summary_df[sector_summary_df['stability_rating'] == "高（优秀）"]
            f.write("### 稳定性分析\n\n")
            f.write(f"- **高稳定性板块**: {len(high_stability)} 个\n")
            if len(high_stability) > 0:
                f.write(f"  - {', '.join(high_stability['sector_name'].tolist())}\n")
            f.write("\n")

            # 投资建议
            f.write("## 💡 投资建议\n\n")
            if len(top_sharpe) > 0:
                best_sector = top_sharpe.iloc[0]
                f.write(f"### 推荐板块\n\n")
                f.write(f"**{best_sector['sector_name']}** ({best_sector['sector_code']})\n\n")
                f.write(f"- 夏普比率: {best_sector['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 平均收益率: {best_sector['avg_return']:.2%}\n")
                f.write(f"- 年化收益率: {best_sector['annualized_return']:.2%}\n")
                f.write(f"- 胜率: {best_sector['avg_win_rate']:.2%}\n")
                f.write(f"- 准确率: {best_sector['avg_accuracy']:.2%}\n")
                f.write(f"- 稳定性: {best_sector['stability_rating']}\n\n")
                f.write(f"建议：该板块模型表现优秀，稳定性高，适合重点配置。\n\n")

            f.write("---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        logger.info(f"汇总报告已保存: {md_file}")
        return md_file


def get_stocks_by_sector(sector_code):
    """根据板块代码获取股票列表"""
    stock_list = []
    for code, info in STOCK_SECTOR_MAPPING.items():
        if info.get('sector') == sector_code:
            stock_list.append(code)
    return stock_list


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='按板块进行Walk-forward验证')

    # 板块选择
    parser.add_argument('--sector', type=str,
                       help='单个板块代码（如 bank、tech、semiconductor）')
    parser.add_argument('--sectors', type=str, nargs='+',
                       help='多个板块代码（空格分隔）')
    parser.add_argument('--all-sectors', action='store_true',
                       help='验证所有板块')

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
    parser.add_argument('--class-weight', type=str, default='balanced',
                       help='类别权重策略（balanced/none/custom，默认: balanced）')

    # 特征参数
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择')

    # 数据参数
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='验证开始日期（默认: 2024-01-01）')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                       help='验证结束日期（默认: 2025-12-31）')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录（默认: output）')

    args = parser.parse_args()

    # 确定要验证的板块
    if args.all_sectors:
        # 获取所有板块
        all_sectors = set()
        for info in STOCK_SECTOR_MAPPING.values():
            if 'sector' in info:
                all_sectors.add(info['sector'])
        sectors_to_validate = sorted(list(all_sectors))
        print(f"\n{'='*80}")
        print(f"🚀 验证所有板块（共 {len(sectors_to_validate)} 个）")
        print(f"{'='*80}")
    elif args.sectors:
        sectors_to_validate = args.sectors
        print(f"\n{'='*80}")
        print(f"🚀 验证指定板块（共 {len(sectors_to_validate)} 个）")
        print(f"{'='*80}")
    elif args.sector:
        sectors_to_validate = [args.sector]
        print(f"\n{'='*80}")
        print(f"🚀 验证单个板块: {SECTOR_NAME_MAPPING.get(args.sector, args.sector)}")
        print(f"{'='*80}")
    else:
        print("错误：请指定要验证的板块（--sector、--sectors 或 --all-sectors）")
        print(f"\n可用板块：{', '.join(sorted(set([info['sector'] for info in STOCK_SECTOR_MAPPING.values() if 'sector' in info])))}")
        sys.exit(1)

    # 处理类别权重参数
    class_weight = args.class_weight
    if class_weight.lower() == 'none':
        class_weight = None
    elif class_weight.lower() == 'balanced':
        class_weight = 'balanced'
    elif ':' in class_weight:  # 格式如 "0:1.0,1:1.5"
        weight_dict = {}
        for pair in class_weight.split(','):
            key, val = pair.split(':')
            weight_dict[int(key)] = float(val)
        class_weight = weight_dict
        print(f"使用自定义类别权重: {class_weight}")
    
    # 创建验证器
    validator = SectorWalkForwardValidator(
        model_type='catboost',
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_window_months=args.step_window,
        horizon=args.horizon,
        confidence_threshold=args.confidence_threshold,
        use_feature_selection=args.use_feature_selection,
        class_weight=class_weight
    )

    # 存储所有板块的报告
    all_sector_reports = []

    # 验证每个板块
    for sector_code in sectors_to_validate:
        print(f"\n\n{'#'*80}")
        print(f"# 开始验证板块: {SECTOR_NAME_MAPPING.get(sector_code, sector_code)}")
        print(f"{'#'*80}")

        # 获取该板块的股票列表
        stock_list = get_stocks_by_sector(sector_code)

        if not stock_list:
            print(f"⚠️  板块 {sector_code} 没有股票，跳过")
            continue

        try:
            # 验证板块
            sector_report = validator.validate_sector(
                sector_code,
                stock_list,
                args.start_date,
                args.end_date
            )

            # 保存板块报告
            output_files = validator.save_sector_report(sector_report, args.output_dir)

            print(f"\n✅ 板块 {SECTOR_NAME_MAPPING.get(sector_code, sector_code)} 验证完成")
            print(f"报告文件:")
            print(f"  - JSON: {output_files['json_file']}")
            print(f"  - CSV:  {output_files['csv_file']}")
            print(f"  - MD:   {output_files['md_file']}")

            all_sector_reports.append(sector_report)

        except Exception as e:
            logger.error(f"板块 {sector_code} 验证失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # 生成汇总报告
    if all_sector_reports:
        print(f"\n\n{'#'*80}")
        print(f"# 生成汇总报告")
        print(f"{'#'*80}")

        summary_file = validator.generate_summary_report(all_sector_reports, args.output_dir)

        print(f"\n{'='*80}")
        print("✅ 所有板块验证完成")
        print(f"{'='*80}")
        print(f"成功验证板块数: {len(all_sector_reports)}")
        if summary_file:
            print(f"汇总报告: {summary_file}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠️  没有板块验证成功")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
