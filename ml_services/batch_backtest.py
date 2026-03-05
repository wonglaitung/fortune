#!/usr/bin/env python3
"""
批量回测脚本 - 对所有股票逐一进行回测
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_evaluator import BacktestEvaluator

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import (
    MLTradingModel, GBDTModel, CatBoostModel, EnsembleModel
)
from ml_services.logger_config import get_logger

logger = get_logger('batch_backtest')
from config import WATCHLIST as STOCK_LIST

# 股票名称映射
STOCK_NAMES = STOCK_LIST


def batch_backtest_all_stocks(model, test_df, feature_columns, confidence_threshold=0.55):
    """
    对所有股票逐一进行回测

    Args:
        model: 训练好的模型
        test_df: 测试数据（包含多只股票）
        feature_columns: 特征列名列表
        confidence_threshold: 置信度阈值

    Returns:
        list: 所有股票的回测结果列表
    """
    unique_stocks = test_df['Code'].unique()
    logger.info(f"开始批量回测，共 {len(unique_stocks)} 只股票")

    results = []
    evaluator = BacktestEvaluator(initial_capital=100000)

    for i, stock_code in enumerate(unique_stocks, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(unique_stocks)}] 回测股票: {stock_code}")
        print(f"{'='*80}")

        # 获取单只股票的数据
        single_stock_df = test_df[test_df['Code'] == stock_code].sort_index()
        prices = single_stock_df['Close']

        if len(prices) < 50:  # 数据太少，跳过
            logger.warning(f" 跳过 {stock_code}：数据量不足（{len(prices)} 条）")
            continue

        print(f"价格数据: {len(prices)} 条")

        # 准备测试数据
        X_test = single_stock_df[feature_columns].copy()
        y_test = single_stock_df['Label'].values

        # 运行回测
        try:
            stock_result = evaluator.backtest_model(
                model=model,
                test_data=X_test,
                test_labels=pd.Series(y_test, index=single_stock_df.index),
                test_prices=prices,
                confidence_threshold=confidence_threshold
            )

            # 添加股票信息（包含股票名称）
            stock_result['stock_code'] = stock_code
            stock_result['stock_name'] = STOCK_NAMES.get(stock_code, stock_code)
            stock_result['data_points'] = len(prices)

            # 添加详细交易记录
            stock_result['trades'] = convert_to_serializable(evaluator.trades)

            results.append(stock_result)

            # 打印简要结果
            logger.info(f"{stock_code} 回测完成:")
            print(f"   总收益率: {stock_result['total_return']*100:.2f}%")
            print(f"   夏普比率: {stock_result['sharpe_ratio']:.2f}")
            print(f"   最大回撤: {stock_result['max_drawdown']*100:.2f}%")
            print(f"   胜率: {stock_result['win_rate']*100:.2f}%")

        except Exception as e:
            logger.error(f"{stock_code} 回测失败: {e}")
            continue

    return results


def convert_to_serializable(obj):
    """递归转换对象为可序列化的格式"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_batch_results(results, model_type, horizon, fusion_method=None):
    """
    保存批量回测结果

    Args:
        results: 回测结果列表
        model_type: 模型类型
        horizon: 预测周期
        fusion_method: 融合方法（仅用于融合模型）
    """
    # 创建保存目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fusion_method:
        filename = f"batch_backtest_{model_type}_{fusion_method}_{horizon}d_{timestamp}.json"
    else:
        filename = f"batch_backtest_{model_type}_{horizon}d_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    # 转换为可序列化的格式 - 保存完整结果（包含详细交易记录）
    results_full = []
    results_summary = []
    for result in results:
        # 完整结果（包含交易记录）
        result_full = convert_to_serializable(result)
        results_full.append(result_full)

        # 汇总结果（仅关键指标）
        result_summary = {
            'stock_code': result['stock_code'],
            'stock_name': result.get('stock_name', result['stock_code']),
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'final_capital': result['final_capital'],
            'sharpe_ratio': result['sharpe_ratio'],
            'sortino_ratio': result['sortino_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'winning_trades_count': len(result.get('winning_trades', [])),
            'losing_trades_count': len(result.get('losing_trades', [])),
            'benchmark_return': result.get('benchmark_return', 0),
            'benchmark_annual_return': result.get('benchmark_annual_return', 0),
            'benchmark_sharpe': result.get('benchmark_sharpe', 0),
            'benchmark_max_drawdown': result.get('benchmark_max_drawdown', 0),
            'excess_return': result.get('excess_return', 0),
            'information_ratio': result.get('information_ratio', 0),
            'data_points': result.get('data_points', 0),
            'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        results_summary.append(result_summary)

    # 保存完整结果（包含详细交易记录）
    detailed_filename = f"batch_backtest_detailed_{model_type}_{horizon}d_{timestamp}.json"
    detailed_filepath = os.path.join(output_dir, detailed_filename)
    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        json.dump(results_full, f, indent=2, ensure_ascii=False)

    # 保存汇总结果（仅关键指标）
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 批量回测结果已保存到: {filepath}")
    print(f"✅ 详细交易记录已保存到: {detailed_filepath}")
    print(f"   回测股票数量: {len(results_summary)}")

    # 生成汇总报告
    summary = generate_summary(results_summary)
    summary_filename = f"batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt"
    summary_filepath = os.path.join(output_dir, summary_filename)

    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(summary)

    logger.info(f"汇总报告已保存到: {summary_filepath}")

    return results_summary


def generate_summary(results):
    """
    生成批量回测汇总报告

    Args:
        results: 回测结果列表

    Returns:
        str: 汇总报告文本
    """
    if not results:
        return "没有回测结果"

    # 计算汇总统计
    total_returns = [r['total_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    max_drawdowns = [r['max_drawdown'] for r in results]
    win_rates = [r['win_rate'] for r in results]

    summary = f"""
================================================================================
批量回测汇总报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【总体统计】
  回测股票数量: {len(results)}
  平均总收益率: {np.mean(total_returns)*100:.2f}%
  平均夏普比率: {np.mean(sharpe_ratios):.2f}
  平均最大回撤: {np.mean(max_drawdowns)*100:.2f}%
  平均胜率: {np.mean(win_rates)*100:.2f}%

【收益分布】
  最高收益率: {np.max(total_returns)*100:.2f}% ({results[np.argmax(total_returns)]['stock_code']} {results[np.argmax(total_returns)].get('stock_name', '')})
  最低收益率: {np.min(total_returns)*100:.2f}% ({results[np.argmin(total_returns)]['stock_code']} {results[np.argmin(total_returns)].get('stock_name', '')})
  收益率中位数: {np.median(total_returns)*100:.2f}%
  收益率标准差: {np.std(total_returns)*100:.2f}%

【夏普比率分布】
  最高夏普比率: {np.max(sharpe_ratios):.2f} ({results[np.argmax(sharpe_ratios)]['stock_code']} {results[np.argmax(sharpe_ratios)].get('stock_name', '')})
  最低夏普比率: {np.min(sharpe_ratios):.2f} ({results[np.argmin(sharpe_ratios)]['stock_code']} {results[np.argmin(sharpe_ratios)].get('stock_name', '')})
  夏普比率中位数: {np.median(sharpe_ratios):.2f}

【回撤分布】
  最大回撤（最好）: {np.min(max_drawdowns)*100:.2f}% ({results[np.argmin(max_drawdowns)]['stock_code']} {results[np.argmin(max_drawdowns)].get('stock_name', '')})
  最大回撤（最差）: {np.max(max_drawdowns)*100:.2f}% ({results[np.argmax(max_drawdowns)]['stock_code']} {results[np.argmax(max_drawdowns)].get('stock_name', '')})
  平均最大回撤: {np.mean(max_drawdowns)*100:.2f}%

【胜率分布】
  最高胜率: {np.max(win_rates)*100:.2f}% ({results[np.argmax(win_rates)]['stock_code']} {results[np.argmax(win_rates)].get('stock_name', '')})
  最低胜率: {np.min(win_rates)*100:.2f}% ({results[np.argmin(win_rates)]['stock_code']} {results[np.argmin(win_rates)].get('stock_name', '')})
  平均胜率: {np.mean(win_rates)*100:.2f}%

================================================================================
【详细结果清单】
================================================================================

"""

    # 按收益率排序
    sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        stock_name = result.get('stock_name', result['stock_code'])
        summary += f"""
{i}. {result['stock_code']} ({stock_name})
   总收益率: {result['total_return']*100:.2f}%
   年化收益率: {result['annual_return']*100:.2f}%
   夏普比率: {result['sharpe_ratio']:.2f}
   索提诺比率: {result['sortino_ratio']:.2f}
   最大回撤: {result['max_drawdown']*100:.2f}%
   胜率: {result['win_rate']*100:.2f}%
   总交易次数: {result['total_trades']}
   盈利交易: {result.get('winning_trades_count', 0)}
   亏损交易: {result.get('losing_trades_count', 0)}
   数据点数: {result.get('data_points', 'N/A')}
"""

    return summary


def main():
    parser = argparse.ArgumentParser(description='批量回测所有股票')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['lgbm', 'gbdt', 'catboost', 'ensemble'],
                        help='模型类型')
    parser.add_argument('--horizon', type=int, default=20,
                        help='预测周期（天）')
    parser.add_argument('--confidence-threshold', type=float, default=0.55,
                        help='置信度阈值')
    parser.add_argument('--fusion-method', type=str, default='weighted',
                        choices=['average', 'weighted', 'voting', 'dynamic-market', 'advanced-dynamic'],
                        help='融合方法（仅用于融合模型）')
    parser.add_argument('--use-feature-selection', action='store_true',
                        help='使用特征选择')
    parser.add_argument('--skip-feature-selection', action='store_true',
                        help='跳过特征选择（配合 --use-feature-selection 使用）')

    args = parser.parse_args()

    logger.info(f"开始批量回测")
    print(f"   模型类型: {args.model_type}")
    print(f"   预测周期: {args.horizon} 天")
    print(f"   置信度阈值: {args.confidence_threshold}")
    if args.model_type == 'ensemble':
        print(f"   融合方法: {args.fusion_method}")

    # 加载模型
    model = None
    data_prep_model = None  # 用于准备数据的模型

    if args.model_type == 'lgbm':
        model = MLTradingModel()
        # 优先使用新训练的模型，否则使用旧路径
        new_model_path = f'output/models_with_feature_selection/lightgbm_stable_high_100_h5.pkl'
        if os.path.exists(new_model_path):
            logger.info(f"使用推荐模型: {new_model_path}")
            model.load_model(new_model_path)
        else:
            logger.info(f"使用标准模型: data/ml_trading_model_lgbm_{args.horizon}d.pkl")
            model.load_model(f'data/ml_trading_model_lgbm_{args.horizon}d.pkl')
        data_prep_model = model
    elif args.model_type == 'gbdt':
        model = GBDTModel()
        model.load_model(f'data/ml_trading_model_gbdt_{args.horizon}d.pkl')
        data_prep_model = model
    elif args.model_type == 'catboost':
        model = CatBoostModel()
        model.load_model(f'data/ml_trading_model_catboost_{args.horizon}d.pkl')
        data_prep_model = model
    elif args.model_type == 'ensemble':
        # 使用 EnsembleModel.load_models() 方法自动加载三个子模型和准确率
        print("🔧 加载融合模型...")
        model = EnsembleModel(fusion_method=args.fusion_method)
        model.load_models(horizon=args.horizon)
        # 使用 LightGBM 子模型来准备数据（因为 LightGBM 模型有完整的数据准备逻辑）
        data_prep_model = model.lgbm_model
        logger.info("融合模型已加载（包含3个子模型和准确率）")

    logger.info(f"模型已加载")

    # 加载特征选择结果
    selected_features = None
    if args.use_feature_selection:
        try:
            selected_features = data_prep_model.load_selected_features()
            if selected_features is None:
                logger.error("错误：未找到特征选择结果，请先运行特征选择")
                return
            logger.info(f"已加载 {len(selected_features)} 个精选特征")
        except Exception as e:
            logger.warning(f" 无法加载特征选择结果: {e}")
            selected_features = None

    # 准备测试数据 - 使用主脚本的数据准备逻辑
    logger.info(f"准备测试数据...")
    from config import WATCHLIST

    # 使用主脚本的数据准备逻辑
    # 简化版本：直接使用模型的数据准备方法
    test_df = data_prep_model.prepare_data(
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
        feature_columns = data_prep_model.feature_columns

    logger.info(f"测试数据准备完成: {len(test_df)} 条，特征列数: {len(feature_columns)}")

    # 运行批量回测
    results = batch_backtest_all_stocks(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
        confidence_threshold=args.confidence_threshold
    )

    # 保存结果
    if results:
        fusion_method = args.fusion_method if args.model_type == 'ensemble' else None
        save_batch_results(results, args.model_type, args.horizon, fusion_method)

        # 打印汇总报告
        summary = generate_summary(results)
        print(summary)
    else:
        logger.error("没有回测结果")


if __name__ == '__main__':
    main()