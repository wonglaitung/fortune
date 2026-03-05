#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20天持有期回测脚本 - 正确评估CatBoost 20天模型的实际性能
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import CatBoostModel
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

logger = get_logger('backtest_20d_horizon')

# 股票名称映射
STOCK_NAMES = STOCK_LIST


class Backtest20DHoldPeriod:
    """20天持有期回测器 - 符合CatBoost 20天模型的实际预测逻辑"""

    def __init__(self, model, confidence_threshold=0.55, commission=0.001, slippage=0.001):
        """
        初始化回测器

        参数:
        - model: 训练好的模型
        - confidence_threshold: 置信度阈值
        - commission: 交易佣金
        - slippage: 滑点
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.commission = commission
        self.slippage = slippage

    def backtest_single_stock(self, stock_code, test_df, feature_columns, start_date, end_date):
        """
        对单只股票进行20天持有期回测

        关键逻辑：
        - 第i天的预测是"第i+20天是否会上涨"
        - 如果预测上涨且置信度>阈值，则在第i天买入，第i+20天卖出
        - 持有20天，不考虑中间的信号变化

        参数:
        - stock_code: 股票代码
        - test_df: 测试数据（包含多只股票）
        - feature_columns: 特征列名列表
        - start_date: 开始日期
        - end_date: 结束日期

        返回:
        dict: 回测结果
        """
        # 获取单只股票的数据
        single_stock_df = test_df[test_df['Code'] == stock_code].sort_index()
        prices = single_stock_df['Close']

        # 确保索引是 tz-naive（移除时区）
        if hasattr(single_stock_df.index, 'tz') and single_stock_df.index.tz is not None:
            single_stock_df.index = single_stock_df.index.tz_localize(None)
            prices.index = prices.index.tz_localize(None)

        # 转换日期为 tz-naive Timestamp
        start_ts = pd.Timestamp(start_date).tz_localize(None)
        end_ts = pd.Timestamp(end_date).tz_localize(None)

        # 过滤日期范围
        mask = (single_stock_df.index >= start_ts) & (single_stock_df.index <= end_ts)
        single_stock_df_filtered = single_stock_df[mask]
        prices_filtered = prices[mask]

        if len(prices_filtered) < 22:  # 至少需要22天数据（1天买入 + 20天持有 + 1天卖出）
            logger.warning(f"{stock_code}: 日期范围内数据不足（{len(prices_filtered)} 天）")
            return None

        # 准备测试数据
        X_test = single_stock_df_filtered[feature_columns].copy()
        y_test = single_stock_df_filtered['Label'].values

        # 生成预测
        if hasattr(self.model, 'catboost_model'):
            from catboost import Pool
            categorical_encoders = getattr(self.model, 'categorical_encoders', {})
            model_features = getattr(self.model, 'feature_columns', [])
            catboost_model = self.model.catboost_model

            # 确保特征列正确
            available_features = [col for col in model_features if col in X_test.columns]
            if len(available_features) < len(model_features):
                missing_cols = [col for col in model_features if col not in X_test.columns]
                X_test = X_test.copy()
                for col in missing_cols:
                    X_test[col] = 0.0

            X_test = X_test[model_features]

            # 处理分类特征
            categorical_features = [model_features.index(col) for col in categorical_encoders.keys() if col in model_features]
            for cat_idx in categorical_features:
                col_name = model_features[cat_idx]
                if col_name in X_test.columns:
                    X_test[col_name] = X_test[col_name].astype(np.int32)

            # 使用 Pool 对象进行预测
            test_pool = Pool(data=X_test)
            predictions = catboost_model.predict_proba(test_pool)[:, 1]
        else:
            predictions = self.model.predict_proba(X_test)[:, 1]

        horizon = 20
        capital = 100000
        trades = []

        # 逐个交易机会进行回测
        for i in range(len(prices_filtered) - horizon):
            buy_date = prices_filtered.index[i]
            sell_date = prices_filtered.index[i + horizon]
            buy_price = prices_filtered.iloc[i]
            sell_price = prices_filtered.iloc[i + horizon]

            # 检查日期是否在范围内
            if buy_date > end_ts or sell_date > end_ts:
                continue

            # 模型预测
            prob = predictions[i]
            signal = 1 if prob > self.confidence_threshold else 0

            # 实际涨跌
            actual_change = (sell_price - buy_price) / buy_price
            actual_direction = 1 if actual_change > 0 else 0

            # 记录交易
            trades.append({
                'stock_code': stock_code,
                'buy_date': buy_date.strftime('%Y-%m-%d'),
                'sell_date': sell_date.strftime('%Y-%m-%d'),
                'buy_price': buy_price,
                'sell_price': sell_price,
                'prediction': signal,
                'probability': prob,
                'actual_change': actual_change,
                'actual_direction': actual_direction,
                'prediction_correct': signal == actual_direction
            })

        return trades

    def backtest_all_stocks(self, test_df, feature_columns, start_date, end_date):
        """
        对所有股票进行回测

        参数:
        - test_df: 测试数据
        - feature_columns: 特征列名列表
        - start_date: 开始日期
        - end_date: 结束日期

        返回:
        dict: 所有股票的回测结果
        """
        unique_stocks = test_df['Code'].unique()
        logger.info(f"开始20天持有期回测，共 {len(unique_stocks)} 只股票")

        all_trades = []

        for i, stock_code in enumerate(unique_stocks, 1):
            print(f"[{i}/{len(unique_stocks)}] 回测股票: {stock_code}")

            trades = self.backtest_single_stock(
                stock_code, test_df, feature_columns, start_date, end_date
            )

            if trades:
                all_trades.extend(trades)

        return all_trades


def calculate_performance_metrics(all_trades):
    """
    计算性能指标

    参数:
    - all_trades: 所有交易记录

    返回:
    dict: 性能指标
    """
    if not all_trades:
        return {}

    df = pd.DataFrame(all_trades)
    df['buy_date'] = pd.to_datetime(df['buy_date'])

    # 基本统计
    total_trades = len(df)
    correct_predictions = df['prediction_correct'].sum()
    accuracy = correct_predictions / total_trades if total_trades > 0 else 0

    # 收益统计
    buy_signals = df[df['prediction'] == 1]

    if len(buy_signals) > 0:
        avg_return = buy_signals['actual_change'].mean()
        median_return = buy_signals['actual_change'].median()
        std_return = buy_signals['actual_change'].std()
        positive_trades = (buy_signals['actual_change'] > 0).sum()
        negative_trades = (buy_signals['actual_change'] <= 0).sum()

        # 夏普比率（年化）
        returns = buy_signals['actual_change'].values
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252/20) if returns.std() > 0 else 0

        # 胜率（基于买入信号）
        win_rate = positive_trades / len(buy_signals) if len(buy_signals) > 0 else 0

        # F1分数
        precision = correct_predictions / len(buy_signals) if len(buy_signals) > 0 else 0
        recall = correct_predictions / (df['actual_direction'] == 1).sum() if (df['actual_direction'] == 1).sum() > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + buy_signals.sort_values('buy_date')['actual_change']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 每日性能统计
        daily_metrics = []
        for buy_date in sorted(buy_signals['buy_date'].unique()):
            daily_df = buy_signals[buy_signals['buy_date'] == buy_date]
            daily_total = len(daily_df)
            daily_correct = daily_df['prediction_correct'].sum()
            daily_accuracy = daily_correct / daily_total if daily_total > 0 else 0
            daily_avg_return = daily_df['actual_change'].mean()
            daily_median_return = daily_df['actual_change'].median()
            daily_std_return = daily_df['actual_change'].std()
            daily_positive = (daily_df['actual_change'] > 0).sum()
            daily_win_rate = daily_positive / daily_total if daily_total > 0 else 0

            daily_metrics.append({
                'buy_date': buy_date.strftime('%Y-%m-%d'),
                'total_trades': daily_total,
                'correct_predictions': int(daily_correct),
                'accuracy': float(daily_accuracy),
                'avg_return': float(daily_avg_return) if not np.isnan(daily_avg_return) else 0.0,
                'median_return': float(daily_median_return) if not np.isnan(daily_median_return) else 0.0,
                'std_return': float(daily_std_return) if not np.isnan(daily_std_return) else 0.0,
                'positive_trades': int(daily_positive),
                'win_rate': float(daily_win_rate)
            })

        return {
            'total_trades': total_trades,
            'buy_signals': len(buy_signals),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_return': avg_return,
            'median_return': median_return,
            'std_return': std_return,
            'positive_trades': positive_trades,
            'negative_trades': negative_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_metrics': daily_metrics
        }
    else:
        return {
            'total_trades': total_trades,
            'buy_signals': 0,
            'accuracy': accuracy,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'win_rate': 0,
            'daily_metrics': []
        }


def save_results(all_trades, metrics, output_dir='output'):
    """
    保存回测结果

    参数:
    - all_trades: 所有交易记录
    - metrics: 性能指标
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细交易记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = os.path.join(output_dir, f"backtest_20d_trades_{timestamp}.csv")

    df = pd.DataFrame(all_trades)
    df.to_csv(trades_file, index=False, encoding='utf-8')
    print(f"\n✅ 交易记录已保存到: {trades_file}")

    # 保存性能指标
    metrics_file = os.path.join(output_dir, f"backtest_20d_metrics_{timestamp}.json")

    # 转换为可序列化的格式
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ 性能指标已保存到: {metrics_file}")

    # 生成股票级别汇总CSV
    stock_summary_file = os.path.join(output_dir, f"backtest_20d_stock_summary_{timestamp}.csv")
    
    stock_summary = []
    for stock_code in df['stock_code'].unique():
        stock_df = df[df['stock_code'] == stock_code]
        
        total_trades = len(stock_df)
        buy_signals = stock_df[stock_df['prediction'] == 1]
        
        if len(buy_signals) > 0:
            avg_return = buy_signals['actual_change'].mean()
            win_rate = (buy_signals['actual_change'] > 0).mean()
        else:
            avg_return = 0
            win_rate = 0
        
        accuracy = stock_df['prediction_correct'].mean()
        
        stock_summary.append({
            '股票代码': stock_code,
            '股票名称': STOCK_NAMES.get(stock_code, stock_code),
            '交易次数': total_trades,
            '平均收益率': avg_return,
            '胜率': win_rate,
            '准确率': accuracy
        })
    
    stock_summary_df = pd.DataFrame(stock_summary)
    stock_summary_df = stock_summary_df.sort_values('平均收益率', ascending=False)
    stock_summary_df.to_csv(stock_summary_file, index=False, encoding='utf-8')
    print(f"✅ 股票汇总已保存到: {stock_summary_file}")

    # 生成文本报告
    report_file = os.path.join(output_dir, f"backtest_20d_report_{timestamp}.txt")

    report = generate_text_report(metrics, df)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 文本报告已保存到: {report_file}")

    return trades_file, metrics_file, report_file, stock_summary_file


def generate_text_report(metrics, df):
    """
    生成文本报告

    参数:
    - metrics: 性能指标
    - df: 交易数据

    返回:
    str: 文本报告
    """
    report = f"""
{'='*80}
20天持有期回测报告 - CatBoost 20天模型
{'='*80}

【回测参数】
  模型类型: CatBoost 20天
  持有期: 20个交易日
  评估方式: 买入后持有20天（符合模型预测周期）

【性能指标】
  总交易机会: {metrics.get('total_trades', 0)}
  买入信号数: {metrics.get('buy_signals', 0)}

【预测准确率】
  准确率: {metrics.get('accuracy', 0):.2%} ({metrics.get('correct_predictions', 0)}/{metrics.get('total_trades', 0)})
  精确率: {metrics.get('precision', 0):.2%}
  召回率: {metrics.get('recall', 0):.2%}
  F1分数: {metrics.get('f1_score', 0):.4f}

【收益统计】（仅买入信号）
  平均收益率: {metrics.get('avg_return', 0):.2%}
  收益率中位数: {metrics.get('median_return', 0):.2%}
  收益率标准差: {metrics.get('std_return', 0):.2%}
  上涨交易: {metrics.get('positive_trades', 0)} 笔
  下跌交易: {metrics.get('negative_trades', 0)} 笔
  胜率: {metrics.get('win_rate', 0):.2%}

【风险指标】
  夏普比率（年化）: {metrics.get('sharpe_ratio', 0):.2f}
  最大回撤: {metrics.get('max_drawdown', 0):.2%}

{'='*80}
"""

    if len(df) > 0:
        buy_signals_df = df[df['prediction'] == 1].sort_values('actual_change', ascending=False)

        if len(buy_signals_df) > 0:
            report += "\n【最佳交易（TOP 10）】\n"
            for i, (_, row) in enumerate(buy_signals_df.head(10).iterrows(), 1):
                report += f"{i}. {row['stock_code']} | {row['buy_date']} → {row['sell_date']} | "
                report += f"收益率: {row['actual_change']:.2%} | 预测概率: {row['probability']:.4f}\n"

            report += "\n【最差交易（BOTTOM 10）】\n"
            for i, (_, row) in enumerate(buy_signals_df.tail(10).iterrows(), 1):
                report += f"{i}. {row['stock_code']} | {row['buy_date']} → {row['sell_date']} | "
                report += f"收益率: {row['actual_change']:.2%} | 预测概率: {row['probability']:.4f}\n"

    return report


def main():
    parser = argparse.ArgumentParser(description='20天持有期回测 CatBoost 20天模型')
    parser.add_argument('--horizon', type=int, default=20, help='预测周期（天）')
    parser.add_argument('--confidence-threshold', type=float, default=0.55, help='置信度阈值')
    parser.add_argument('--start-date', type=str, default='2026-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2026-01-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--use-feature-selection', action='store_true', help='使用特征选择')
    parser.add_argument('--skip-feature-selection', action='store_true', help='跳过特征选择')

    args = parser.parse_args()

    logger.info(f"开始20天持有期回测")
    print(f"   模型类型: CatBoost {args.horizon}天")
    print(f"   持有期: {args.horizon}天（符合模型预测周期）")
    print(f"   置信度阈值: {args.confidence_threshold}")
    print(f"   回测日期范围: {args.start_date} 至 {args.end_date}")

    # 加载模型
    print("\n🔧 加载 CatBoost 模型...")
    model = CatBoostModel()
    model.load_model(f'data/ml_trading_model_catboost_{args.horizon}d.pkl')
    print(f"✅ 模型已加载")

    # 加载特征选择结果
    selected_features = None
    if args.use_feature_selection:
        try:
            selected_features = model.load_selected_features()
            if selected_features is None:
                logger.error("错误：未找到特征选择结果，请先运行特征选择")
                return
            print(f"✅ 已加载 {len(selected_features)} 个精选特征")
        except Exception as e:
            logger.warning(f" 无法加载特征选择结果: {e}")
            selected_features = None

    # 准备测试数据
    print("\n📊 准备测试数据...")
    from config import WATCHLIST

    test_df = model.prepare_data(
        codes=list(WATCHLIST.keys()),
        horizon=args.horizon,
        for_backtest=True
    )

    if test_df is None or len(test_df) == 0:
        logger.error("错误：没有可用数据")
        return

    # 获取特征列
    if args.use_feature_selection and selected_features is not None:
        feature_columns = selected_features
    else:
        feature_columns = model.feature_columns

    print(f"✅ 测试数据准备完成: {len(test_df)} 条，特征列数: {len(feature_columns)}")

    # 创建回测器
    backtester = Backtest20DHoldPeriod(
        model=model,
        confidence_threshold=args.confidence_threshold
    )

    # 运行回测
    print(f"\n🚀 开始20天持有期回测...")
    all_trades = backtester.backtest_all_stocks(
        test_df=test_df,
        feature_columns=feature_columns,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if not all_trades:
        logger.error("没有回测结果")
        return

    # 计算性能指标
    print(f"\n📊 计算性能指标...")
    metrics = calculate_performance_metrics(all_trades)

    # 保存结果
    print(f"\n💾 保存结果...")
    save_results(all_trades, metrics)

    # 打印汇总报告
    print(f"\n{'='*80}")
    print(f"20天持有期回测完成！")
    print(f"{'='*80}")
    print(f"回测日期范围: {args.start_date} 至 {args.end_date}")
    print(f"总交易机会: {metrics['total_trades']}")
    print(f"买入信号数: {metrics['buy_signals']}")
    print(f"准确率: {metrics['accuracy']:.2%} (与训练时可比)")
    print(f"F1分数: {metrics['f1_score']:.4f} (与训练时可比)")
    print(f"平均收益率: {metrics['avg_return']:.2%} (买入信号)")
    print(f"胜率: {metrics['win_rate']:.2%} (买入信号)")
    print(f"夏普比率（年化）: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")


if __name__ == '__main__':
    main()