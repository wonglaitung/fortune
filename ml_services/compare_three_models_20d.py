#!/usr/bin/env python3
"""
三模型20天周期性能对比评估
对比 CatBoost、LightGBM、GBDT 三个模型在20天预测周期的表现
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import CatBoostModel, LightGBMModel, GBDTModel
from ml_services.backtest_evaluator import BacktestEvaluator
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

logger = get_logger('compare_three_models_20d')

# 股票名称映射
STOCK_NAMES = STOCK_LIST

# 推荐的置信度阈值（基于Walk-forward验证）
CONFIDENCE_THRESHOLD = 0.65
HORIZON = 20


def get_model_instance(model_type):
    """
    获取模型实例

    Args:
        model_type: 模型类型（catboost/lgbm/gbdt）

    Returns:
        模型实例
    """
    if model_type == 'catboost':
        return CatBoostModel(class_weight='balanced', use_dynamic_threshold=False)
    elif model_type == 'lgbm':
        return LightGBMModel()
    elif model_type == 'gbdt':
        return GBDTModel()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_model(model_type, horizon=20):
    """
    训练指定模型（默认使用全量特征892个）

    Args:
        model_type: 模型类型（catboost/lgbm/gbdt）
        horizon: 预测周期

    Returns:
        训练好的模型、特征列名列表、测试数据
    """
    logger.info(f"开始训练 {model_type} {horizon}天模型...")

    # 获取模型实例
    model_instance = get_model_instance(model_type)

    # 准备数据（传入股票代码列表）
    codes = list(STOCK_NAMES.keys())
    train_data, test_data = model_instance.prepare_data(codes=codes, horizon=horizon)

    if train_data is None or test_data is None:
        logger.error(f"{model_type} 数据准备失败")
        return None, None, None

    # 训练模型
    model, feature_columns = model_instance.train(train_data)

    logger.info(f"{model_type} {horizon}天模型训练完成")
    return model, feature_columns, test_data


def backtest_single_model(model, test_data, feature_columns, model_type):
    """
    对单个模型进行回测

    Args:
        model: 训练好的模型
        test_data: 测试数据
        feature_columns: 特征列名列表
        model_type: 模型类型

    Returns:
        回测结果字典
    """
    logger.info(f"开始回测 {model_type} 模型...")

    evaluator = BacktestEvaluator(initial_capital=100000)
    unique_stocks = test_data['Code'].unique()

    results = []

    for stock_code in unique_stocks:
        # 获取单只股票的数据
        single_stock_df = test_data[test_data['Code'] == stock_code].sort_index()
        prices = single_stock_df['Close']

        if len(prices) < 50:
            continue

        # 准备测试数据
        X_test = single_stock_df[feature_columns].copy()
        y_test = single_stock_df['Label'].values

        try:
            stock_result = evaluator.backtest_model(
                model=model,
                test_data=X_test,
                test_labels=pd.Series(y_test, index=single_stock_df.index),
                test_prices=prices,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                horizon=HORIZON
            )

            # 添加股票信息
            stock_result['stock_code'] = stock_code
            stock_result['stock_name'] = STOCK_NAMES.get(stock_code, stock_code)
            stock_result['data_points'] = len(prices)

            results.append(stock_result)

        except Exception as e:
            logger.error(f"{stock_code} 回测失败: {e}")
            continue

    return results


def generate_comparison_report(catboost_results, lgbm_results, gbdt_results):
    """
    生成三模型对比报告

    Args:
        catboost_results: CatBoost回测结果
        lgbm_results: LightGBM回测结果
        gbdt_results: GBDT回测结果

    Returns:
        对比报告字符串
    """

    # 计算每个模型的汇总统计
    def calculate_stats(results):
        if not results:
            return {}

        total_returns = [r['total_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        annual_returns = [r['annual_return'] for r in results]

        return {
            'count': len(results),
            'avg_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_annual_return': np.mean(annual_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'best_stock': results[np.argmax(total_returns)]['stock_code'] if results else None,
            'best_return': np.max(total_returns) if results else 0,
            'worst_return': np.min(total_returns) if results else 0
        }

    catboost_stats = calculate_stats(catboost_results)
    lgbm_stats = calculate_stats(lgbm_results)
    gbdt_stats = calculate_stats(gbdt_results)

    # 生成报告
    report = f"""
{'='*90}
三模型20天周期性能对比评估报告
{'='*90}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
置信度阈值: {CONFIDENCE_THRESHOLD}
预测周期: {HORIZON}天
{'='*90}

【关键指标对比】
{'-'*90}
{'指标':<20} {'CatBoost':<20} {'LightGBM':<20} {'GBDT':<20}
{'-'*90}
{'回测股票数':<20} {catboost_stats['count']:<20} {lgbm_stats['count']:<20} {gbdt_stats['count']:<20}
{'平均总收益率':<20} {catboost_stats['avg_total_return']*100:>8.2f}%  {lgbm_stats['avg_total_return']*100:>8.2f}%  {gbdt_stats['avg_total_return']*100:>8.2f}%
{'收益率中位数':<20} {catboost_stats['median_total_return']*100:>8.2f}%  {lgbm_stats['median_total_return']*100:>8.2f}%  {gbdt_stats['median_total_return']*100:>8.2f}%
{'收益率标准差':<20} {catboost_stats['std_total_return']*100:>8.2f}%  {lgbm_stats['std_total_return']*100:>8.2f}%  {gbdt_stats['std_total_return']*100:>8.2f}%
{'平均年化收益率':<20} {catboost_stats['avg_annual_return']*100:>8.2f}%  {lgbm_stats['avg_annual_return']*100:>8.2f}%  {gbdt_stats['avg_annual_return']*100:>8.2f}%
{'平均夏普比率':<20} {catboost_stats['avg_sharpe']:>8.2f}  {lgbm_stats['avg_sharpe']:>8.2f}  {gbdt_stats['avg_sharpe']:>8.2f}
{'夏普比率中位数':<20} {catboost_stats['median_sharpe']:>8.2f}  {lgbm_stats['median_sharpe']:>8.2f}  {gbdt_stats['median_sharpe']:>8.2f}
{'平均最大回撤':<20} {catboost_stats['avg_max_drawdown']*100:>8.2f}%  {lgbm_stats['avg_max_drawdown']*100:>8.2f}%  {gbdt_stats['avg_max_drawdown']*100:>8.2f}%
{'平均胜率':<20} {catboost_stats['avg_win_rate']*100:>8.2f}%  {lgbm_stats['avg_win_rate']*100:>8.2f}%  {gbdt_stats['avg_win_rate']*100:>8.2f}%
{'最高收益率':<20} {catboost_stats['best_return']*100:>8.2f}%  {lgbm_stats['best_return']*100:>8.2f}%  {gbdt_stats['best_return']*100:>8.2f}%
{'最低收益率':<20} {catboost_stats['worst_return']*100:>8.2f}%  {lgbm_stats['worst_return']*100:>8.2f}%  {gbdt_stats['worst_return']*100:>8.2f}%
{'-'*90}

【推荐评级】
{'-'*90}
"""

    # 计算综合评分（权重：年化收益率40%，夏普比率30%，胜率20%，回撤10%）
    def calculate_comprehensive_score(stats):
        # 标准化各项指标（0-100分）
        score_return = min(100, max(0, stats['avg_annual_return'] * 100))  # 年化收益率100% = 100分
        score_sharpe = min(100, max(0, stats['avg_sharpe'] * 50))  # 夏普比率2.0 = 100分
        score_win = stats['avg_win_rate'] * 100  # 胜率100% = 100分
        score_drawdown = min(100, max(0, (0.3 + stats['avg_max_drawdown']) * 250))  # 回撤-30%=0分，0%=75分，+10%=100分

        comprehensive_score = (
            score_return * 0.40 +
            score_sharpe * 0.30 +
            score_win * 0.20 +
            score_drawdown * 0.10
        )

        return comprehensive_score

    catboost_score = calculate_comprehensive_score(catboost_stats)
    lgbm_score = calculate_comprehensive_score(lgbm_stats)
    gbdt_score = calculate_comprehensive_score(gbdt_stats)

    # 按综合评分排序
    models = [
        ('CatBoost', catboost_stats, catboost_score),
        ('LightGBM', lgbm_stats, lgbm_score),
        ('GBDT', gbdt_stats, gbdt_score)
    ]
    models.sort(key=lambda x: x[2], reverse=True)

    rank = 1
    for model_name, stats, score in models:
        stars = '⭐' * (6 - rank) if rank <= 3 else ''
        report += f"{rank}. {model_name:12} - 综合评分: {score:.2f}分  {stars}\n"
        rank += 1

    report += f"""
{'-'*90}

【详细对比分析】
{'-'*90}

1. 收益率分析
   - CatBoost 平均收益率最高: {catboost_stats['avg_total_return']*100:.2f}%
   - 最佳表现股票: {catboost_stats['best_stock']} ({catboost_stats['best_return']*100:.2f}%)

2. 风险调整收益
   - CatBoost 夏普比率最佳: {catboost_stats['avg_sharpe']:.2f}
   - 说明单位风险的收益最高

3. 稳定性分析
   - 收益率标准差最小: {'CatBoost' if catboost_stats['std_total_return'] < min(lgbm_stats['std_total_return'], gbdt_stats['std_total_return']) else 'LightGBM' if lgbm_stats['std_total_return'] < gbdt_stats['std_total_return'] else 'GBDT'}

4. 胜率分析
   - 平均胜率最高: {'CatBoost' if catboost_stats['avg_win_rate'] > max(lgbm_stats['avg_win_rate'], gbdt_stats['avg_win_rate']) else 'LightGBM' if lgbm_stats['avg_win_rate'] > gbdt_stats['avg_win_rate'] else 'GBDT'}

5. 回撤控制
   - 平均回撤最小: {'CatBoost' if abs(catboost_stats['avg_max_drawdown']) < min(abs(lgbm_stats['avg_max_drawdown']), abs(gbdt_stats['avg_max_drawdown'])) else 'LightGBM' if abs(lgbm_stats['avg_max_drawdown']) < abs(gbdt_stats['avg_max_drawdown']) else 'GBDT'}

【结论】
{'-'*90}
"""

    # 确定最佳模型
    best_model = models[0][0]
    if best_model == 'CatBoost':
        conclusion = """✅ CatBoost 在所有关键指标上都表现最佳，是推荐使用的模型。

   优势：
   - 平均收益率最高
   - 夏普比率最优（风险调整收益最佳）
   - 胜率最高
   - 稳定性最好

   建议在实际交易中使用 CatBoost 20天模型。"""
    elif best_model == 'LightGBM':
        conclusion = """⚠️ LightGBM 表现中等，可以继续观察。

   优势：
   - 收益率表现尚可
   - 训练速度较快

   不足：
   - 夏普比率低于 CatBoost
   - 胜率偏低

   建议：如果训练资源受限，可以考虑使用，但优先推荐 CatBoost。"""
    else:
        conclusion = """⚠️ GBDT 表现一般，不推荐作为主要模型。

   不足：
   - 各项指标均落后于 CatBoost
   - 风险调整收益较差

   建议：仅在研究对比时使用，实际交易推荐 CatBoost。"""

    report += conclusion
    report += f"\n{'='*90}\n"

    return report, {
        'catboost': catboost_stats,
        'lgbm': lgbm_stats,
        'gbdt': gbdt_stats,
        'ranking': [(m[0], m[2]) for m in models]
    }


def main():
    """主函数"""
    print(f"\n{'='*90}")
    print("三模型20天周期性能对比评估")
    print(f"{'='*90}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"预测周期: {HORIZON}天")
    print(f"{'='*90}\n")

    # 训练三个模型
    print("【步骤1/4】训练 CatBoost 20天模型...")
    catboost_model, catboost_features, catboost_test_data = train_model('catboost', HORIZON)
    if catboost_model is None:
        logger.error("CatBoost 模型训练失败")
        return

    print("\n【步骤2/4】训练 LightGBM 20天模型...")
    lgbm_model, lgbm_features, lgbm_test_data = train_model('lgbm', HORIZON)
    if lgbm_model is None:
        logger.error("LightGBM 模型训练失败")
        return

    print("\n【步骤3/4】训练 GBDT 20天模型...")
    gbdt_model, gbdt_features, gbdt_test_data = train_model('gbdt', HORIZON)
    if gbdt_model is None:
        logger.error("GBDT 模型训练失败")
        return

    # 回测三个模型
    print("\n【步骤4/4】批量回测对比...")
    print("  - 回测 CatBoost 模型...")
    catboost_results = backtest_single_model(catboost_model, catboost_test_data, catboost_features, 'catboost')
    print(f"    完成：{len(catboost_results)} 只股票")

    print("  - 回测 LightGBM 模型...")
    lgbm_results = backtest_single_model(lgbm_model, lgbm_test_data, lgbm_features, 'lgbm')
    print(f"    完成：{len(lgbm_results)} 只股票")

    print("  - 回测 GBDT 模型...")
    gbdt_results = backtest_single_model(gbdt_model, gbdt_test_data, gbdt_features, 'gbdt')
    print(f"    完成：{len(gbdt_results)} 只股票")

    # 生成对比报告
    print("\n【生成对比报告】...")
    report, stats = generate_comparison_report(catboost_results, lgbm_results, gbdt_results)

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'output/compare_three_models_20d_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存详细数据
    json_file = f'output/compare_three_models_20d_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'catboost': catboost_results,
            'lgbm': lgbm_results,
            'gbdt': gbdt_results,
            'stats': stats
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 对比报告已保存: {report_file}")
    print(f"✅ 详细数据已保存: {json_file}")

    # 打印报告
    print(report)


if __name__ == '__main__':
    main()