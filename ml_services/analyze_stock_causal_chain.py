#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股因果链分析（完整模型）：1d→5d 和 5d→20d 的传导关系

使用完整模型（918个特征），对比恒指结果，检验个股是否具有相同的因果链模式。

创建时间：2026-04-26
"""

import warnings
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SECTOR_MAPPING

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
OUTPUT_DIR = "output"


# CatBoost 参数（完整版，按周期调整）
def get_full_catboost_params(horizon):
    if horizon == 1:
        return {'iterations': 100, 'learning_rate': 0.15, 'depth': 5,
                'l2_leaf_reg': 3, 'random_seed': RANDOM_SEED,
                'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 'verbose': 0}
    elif horizon == 5:
        return {'iterations': 100, 'learning_rate': 0.15, 'depth': 4,
                'l2_leaf_reg': 3, 'subsample': 0.7, 'random_seed': RANDOM_SEED,
                'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 'verbose': 0}
    else:
        return {'iterations': 100, 'learning_rate': 0.15, 'depth': 4,
                'l2_leaf_reg': 3, 'subsample': 0.8, 'random_seed': RANDOM_SEED,
                'loss_function': 'Logloss', 'auto_class_weights': 'Balanced', 'verbose': 0}


class FullFeatureEngineer:
    """完整特征工程（复用 ml_trading_model.py 的 FeatureEngineer）"""

    def __init__(self):
        from ml_services.ml_trading_model import FeatureEngineer
        self.feature_engineer = FeatureEngineer()

    def calculate_features(self, df, stock_code):
        """计算完整特征（918个）"""
        df = df.copy()

        # 技术指标
        df = self.feature_engineer.calculate_technical_features(df)
        df = self.feature_engineer.calculate_multi_period_metrics(df)

        # 基本面
        fundamental_features = self.feature_engineer.create_fundamental_features(stock_code)
        for key, value in fundamental_features.items():
            df[key] = value

        # 股票类型
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        for key, value in stock_type_features.items():
            df[key] = value

        # 情感
        sentiment_features = self.feature_engineer.create_sentiment_features(stock_code, df)
        for key, value in sentiment_features.items():
            df[key] = value

        # 板块
        sector_features = self.feature_engineer.create_sector_features(stock_code, df)
        for key, value in sector_features.items():
            df[key] = value

        # 事件驱动
        df = self.feature_engineer.create_event_driven_features(stock_code, df)

        # 交互特征
        df = self.feature_engineer.create_technical_fundamental_interactions(df)
        df = self.feature_engineer.create_interaction_features(df)

        return df


def get_available_features(df, stock_code):
    """获取可用特征列表（排除非特征列）"""
    exclude_columns = {'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                       'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                       'BB_upper', 'BB_lower', 'BB_middle', 'Dividends', 'Stock Splits',
                       'Future_Return_1d', 'Future_Return_5d', 'Future_Return_20d',
                       'Target_1d', 'Target_5d', 'Target_20d'}

    available_features = [col for col in df.columns if col not in exclude_columns]

    # 过滤非数值特征
    numeric_features = [col for col in available_features
                        if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
    available_features = numeric_features

    # 过滤高NaN率特征（>50%）
    nan_ratio = df[available_features].isna().sum() / len(df)
    valid_features = nan_ratio[nan_ratio <= 0.5].index.tolist()
    available_features = valid_features

    return available_features


def fetch_stock_data(stock_code, period_days=1460):
    """获取个股历史数据"""
    try:
        from ml_services.ml_trading_model import get_stock_data_with_cache
        df = get_stock_data_with_cache(stock_code.replace('.HK', ''), period_days=period_days)
        if df is not None and not df.empty and len(df) >= 300:
            return df
    except Exception:
        pass

    try:
        import yfinance as yf
        ticker = yf.Ticker(stock_code)
        df = ticker.history(period=f"{period_days}d", interval="1d")
        if df.empty or len(df) < 300:
            return None
        return df
    except Exception as e:
        print(f"  ⚠️ 获取 {stock_code} 数据失败: {e}")
        return None


def run_single_stock_walkforward(stock_code, stock_name, feature_engineer):
    """对单只股票运行完整模型 Walk-forward 三周期预测"""
    try:
        # 获取数据
        df = fetch_stock_data(stock_code)
        if df is None:
            print(f"  ⚠️ {stock_code} 数据不足")
            return None

        print(f"  原始数据: {len(df)} 条")

        # 计算完整特征
        df = feature_engineer.calculate_features(df, stock_code)
        available_features = get_available_features(df, stock_code)
        print(f"  可用特征: {len(available_features)} 个")

        if len(available_features) < 10:
            return None

        # 填充NaN
        for col in available_features:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 创建目标
        for h in [1, 5, 20]:
            df[f'Future_Return_{h}d'] = df['Close'].pct_change(h).shift(-h)
            df[f'Target_{h}d'] = (df[f'Future_Return_{h}d'] > 0).astype(int)

        # Walk-forward
        train_days = 252
        step_days = 40  # 完整模型用更大步长加速
        total_days = len(df)
        num_folds = (total_days - train_days) // step_days

        if num_folds < 5:
            print(f"  ⚠️ {stock_code} folds不足({num_folds})")
            return None

        print(f"  Walk-forward: {num_folds} folds, step={step_days}天")

        all_predictions = {1: [], 5: [], 20: []}

        for fold in range(num_folds):
            train_start = fold * step_days
            train_end = train_start + train_days
            test_start = train_end
            test_end = min(test_start + step_days, total_days)

            if test_start >= total_days:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            for horizon in [1, 5, 20]:
                target_col = f'Target_{horizon}d'
                return_col = f'Future_Return_{horizon}d'

                train_clean = train_df.dropna(subset=available_features + [target_col])
                test_clean = test_df.dropna(subset=available_features + [target_col, return_col])

                if len(train_clean) < 50 or len(test_clean) < 1:
                    continue

                X_train = train_clean[available_features]
                y_train = train_clean[target_col]

                try:
                    model = CatBoostClassifier(**get_full_catboost_params(horizon))
                    model.fit(X_train, y_train)

                    X_test = test_clean[available_features]
                    probs = model.predict_proba(X_test)[:, 1]
                    preds = (probs > 0.5).astype(int)

                    for i, (idx, row) in enumerate(test_clean.iterrows()):
                        all_predictions[horizon].append({
                            'date': idx,
                            'prediction': preds[i],
                            'probability': probs[i],
                            'actual_direction': int(row[target_col]),
                            'actual_return': float(row[return_col]),
                            'close': row['Close']
                        })
                except Exception:
                    continue

        # 检查样本量
        for h in [1, 5, 20]:
            if len(all_predictions[h]) < 100:
                print(f"  ⚠️ {stock_code} horizon={h}样本不足({len(all_predictions[h])})")
                return None

        return all_predictions

    except Exception as e:
        print(f"  ⚠️ {stock_code} 失败: {e}")
        return None


def analyze_three_horizon_patterns(all_predictions):
    """分析三周期预测模式和正确性传导（参考恒指脚本）"""
    # 构建DataFrame
    df_1d = pd.DataFrame(all_predictions[1])
    df_5d = pd.DataFrame(all_predictions[5])
    df_20d = pd.DataFrame(all_predictions[20])

    if len(df_1d) == 0 or len(df_5d) == 0 or len(df_20d) == 0:
        return None

    df_1d.set_index('date', inplace=True)
    df_5d.set_index('date', inplace=True)
    df_20d.set_index('date', inplace=True)

    # 去重并排序
    for df in [df_1d, df_5d, df_20d]:
        df.drop_duplicates(keep='last', inplace=True)
        df.sort_index(inplace=True)

    # 对齐数据
    common_dates = df_1d.index.intersection(df_5d.index).intersection(df_20d.index)
    if len(common_dates) < 50:
        return None

    pred_1d = df_1d.loc[common_dates, 'prediction'].astype(int)
    pred_5d = df_5d.loc[common_dates, 'prediction'].astype(int)
    pred_20d = df_20d.loc[common_dates, 'prediction'].astype(int)

    actual_1d = df_1d.loc[common_dates, 'actual_direction'].astype(int)
    actual_5d = df_5d.loc[common_dates, 'actual_direction'].astype(int)
    actual_20d = df_20d.loc[common_dates, 'actual_direction'].astype(int)

    prob_1d = df_1d.loc[common_dates, 'probability']
    prob_5d = df_5d.loc[common_dates, 'probability']

    results = {'samples': len(common_dates)}

    # ========== 1. 独立准确率 ==========
    results['independent_accuracy'] = {
        '1d': float((pred_1d == actual_1d).mean() * 100),
        '5d': float((pred_5d == actual_5d).mean() * 100),
        '20d': float((pred_20d == actual_20d).mean() * 100),
    }

    # ========== 2. 预测正确性传导 ==========
    correct_1d = pred_1d == actual_1d
    correct_5d = pred_5d == actual_5d

    if correct_1d.sum() > 0:
        results['1d_correct_then_20d_accuracy'] = float(
            (pred_20d[correct_1d] == actual_20d[correct_1d]).mean() * 100
        )
    if (~correct_1d).sum() > 0:
        results['1d_wrong_then_20d_accuracy'] = float(
            (pred_20d[~correct_1d] == actual_20d[~correct_1d]).mean() * 100
        )
    if correct_5d.sum() > 0:
        results['5d_correct_then_20d_accuracy'] = float(
            (pred_20d[correct_5d] == actual_20d[correct_5d]).mean() * 100
        )
    if (correct_1d & correct_5d).sum() > 0:
        results['1d_5d_correct_then_20d_accuracy'] = float(
            (pred_20d[correct_1d & correct_5d] == actual_20d[correct_1d & correct_5d]).mean() * 100
        )

    # ========== 3. 八大模式分析 ==========
    patterns = {
        '111': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 1)),
        '110': ((pred_1d == 1) & (pred_5d == 1) & (pred_20d == 0)),
        '101': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 1)),
        '100': ((pred_1d == 1) & (pred_5d == 0) & (pred_20d == 0)),
        '011': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 1)),
        '010': ((pred_1d == 0) & (pred_5d == 1) & (pred_20d == 0)),
        '001': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 1)),
        '000': ((pred_1d == 0) & (pred_5d == 0) & (pred_20d == 0)),
    }

    results['patterns'] = {}
    for pattern, mask in patterns.items():
        count = mask.sum()
        if count > 0:
            results['patterns'][pattern] = {
                'count': int(count),
                'pct': float(count / len(common_dates) * 100),
                '20d_accuracy': float((pred_20d[mask] == actual_20d[mask]).mean() * 100),
                '1d_accuracy': float((pred_1d[mask] == actual_1d[mask]).mean() * 100),
                '5d_accuracy': float((pred_5d[mask] == actual_5d[mask]).mean() * 100),
            }

    # ========== 4. 一致信号分析 ==========
    consensus_bull = patterns['111']
    consensus_bear = patterns['000']

    results['consensus_analysis'] = {
        'bull': {
            'count': int(consensus_bull.sum()),
            '1d_actual_up_rate': float(actual_1d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
            '5d_actual_up_rate': float(actual_5d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
            '20d_actual_up_rate': float(actual_20d[consensus_bull].mean() * 100) if consensus_bull.sum() > 0 else 0,
        },
        'bear': {
            'count': int(consensus_bear.sum()),
            '1d_actual_down_rate': float((1 - actual_1d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
            '5d_actual_down_rate': float((1 - actual_5d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
            '20d_actual_down_rate': float((1 - actual_20d[consensus_bear]).mean() * 100) if consensus_bear.sum() > 0 else 0,
        }
    }

    # ========== 5. 概率相关性 ==========
    results['probability_correlation'] = {
        '1d_vs_5d': float(prob_1d.corr(prob_5d)),
        '1d_vs_20d_actual': float(prob_1d.corr(actual_20d)),
        '5d_vs_20d_actual': float(prob_5d.corr(actual_20d)),
    }

    return results


def analyze_causal_chain(all_predictions, chain_type='1d_to_5d'):
    """分析因果链"""
    if chain_type == '1d_to_5d':
        df_short = pd.DataFrame(all_predictions[1])
        df_long = pd.DataFrame(all_predictions[5])
        chain_len = 5
        step = 1
    else:
        df_short = pd.DataFrame(all_predictions[5])
        df_long = pd.DataFrame(all_predictions[20])
        chain_len = 4
        step = 5

    if len(df_short) == 0 or len(df_long) == 0:
        return None

    df_short.set_index('date', inplace=True)
    df_long.set_index('date', inplace=True)

    df_short = df_short[~df_short.index.duplicated(keep='last')]
    df_long = df_long[~df_long.index.duplicated(keep='last')]
    df_short.sort_index(inplace=True)
    df_long.sort_index(inplace=True)

    results = []

    for date in df_long.index:
        if date not in df_short.index:
            continue

        date_idx = df_short.index.get_loc(date)
        required_indices = [date_idx + i * step for i in range(chain_len)]

        if any(idx >= len(df_short.index) for idx in required_indices):
            continue

        dates_needed = [df_short.index[idx] for idx in required_indices]

        pred_long = int(df_long.loc[date, 'prediction'])
        actual_long = int(df_long.loc[date, 'actual_direction'])
        prob_long = float(df_long.loc[date, 'probability'])

        preds_short = [int(df_short.loc[d, 'prediction']) for d in dates_needed]
        actuals_short = [int(df_short.loc[d, 'actual_direction']) for d in dates_needed]
        probs_short = [float(df_short.loc[d, 'probability']) for d in dates_needed]

        consensus_count = sum(preds_short)
        majority_vote = 1 if consensus_count >= (chain_len / 2 + 0.5) else 0
        correct_short_count = sum(1 for p, a in zip(preds_short, actuals_short) if p == a)
        avg_prob_short = np.mean(probs_short)

        results.append({
            'pred_long': pred_long,
            'actual_long': actual_long,
            'consensus_count': consensus_count,
            'majority_vote': majority_vote,
            'correct_short_count': correct_short_count,
            'avg_prob_short': avg_prob_short,
            'long_correct': pred_long == actual_long,
            'majority_correct': majority_vote == actual_long,
        })

    if not results:
        return None

    return pd.DataFrame(results)


def print_three_horizon_summary(all_stock_results):
    """打印三周期模式汇总（参考恒指格式）"""
    print("\n" + "=" * 90)
    print("📊 个股三周期预测分析汇总")
    print("=" * 90)

    # 恒指基准数据
    hsi_baseline = {
        'independent_accuracy': {'1d': 49.67, '5d': 62.36, '20d': 81.24},
        'causal_analysis': {
            '1d_correct_then_20d_accuracy': 83.56,
            '1d_wrong_then_20d_accuracy': 78.95,
            '5d_correct_then_20d_accuracy': 81.95,
            '1d_5d_correct_then_20d_accuracy': 81.49,
        },
        'patterns': {
            '101': {'20d_accuracy': 95.00, 'count': 60},
            '010': {'20d_accuracy': 85.98, 'count': 107},
            '001': {'20d_accuracy': 84.00, 'count': 100},
            '111': {'20d_accuracy': 80.62, 'count': 129},
            '000': {'20d_accuracy': 79.57, 'count': 186},
        },
        'probability_correlation': {'5d_vs_20d_actual': 0.35},
    }

    # ========== 1. 独立准确率对比 ==========
    print("\n【独立准确率对比】")
    print(f"  {'指标':<20} {'恒指':<12} {'个股平均':<12} {'差异':<10}")
    print("  " + "-" * 50)

    acc_1d_list = []
    acc_5d_list = []
    acc_20d_list = []

    for code, result in all_stock_results.items():
        if 'three_horizon' in result:
            acc = result['three_horizon'].get('independent_accuracy', {})
            if '1d' in acc:
                acc_1d_list.append(acc['1d'])
            if '5d' in acc:
                acc_5d_list.append(acc['5d'])
            if '20d' in acc:
                acc_20d_list.append(acc['20d'])

    if acc_1d_list:
        avg_1d = np.mean(acc_1d_list)
        print(f"  {'1天准确率':<20} {hsi_baseline['independent_accuracy']['1d']:.2f}%{'':<6} "
              f"{avg_1d:.2f}%{'':<6} {hsi_baseline['independent_accuracy']['1d'] - avg_1d:+.2f}%")
    if acc_5d_list:
        avg_5d = np.mean(acc_5d_list)
        print(f"  {'5天准确率':<20} {hsi_baseline['independent_accuracy']['5d']:.2f}%{'':<6} "
              f"{avg_5d:.2f}%{'':<6} {hsi_baseline['independent_accuracy']['5d'] - avg_5d:+.2f}%")
    if acc_20d_list:
        avg_20d = np.mean(acc_20d_list)
        print(f"  {'20天准确率':<20} {hsi_baseline['independent_accuracy']['20d']:.2f}%{'':<6} "
              f"{avg_20d:.2f}%{'':<6} {hsi_baseline['independent_accuracy']['20d'] - avg_20d:+.2f}%")

    # ========== 2. 预测正确性传导对比 ==========
    print("\n【预测正确性传导对比】")
    print(f"  {'指标':<30} {'恒指':<12} {'个股平均':<12} {'差异':<10}")
    print("  " + "-" * 60)

    causal_1d_correct_list = []
    causal_1d_wrong_list = []
    causal_5d_correct_list = []
    causal_1d_5d_correct_list = []

    for code, result in all_stock_results.items():
        if 'three_horizon' in result:
            th = result['three_horizon']
            if '1d_correct_then_20d_accuracy' in th:
                causal_1d_correct_list.append(th['1d_correct_then_20d_accuracy'])
            if '1d_wrong_then_20d_accuracy' in th:
                causal_1d_wrong_list.append(th['1d_wrong_then_20d_accuracy'])
            if '5d_correct_then_20d_accuracy' in th:
                causal_5d_correct_list.append(th['5d_correct_then_20d_accuracy'])
            if '1d_5d_correct_then_20d_accuracy' in th:
                causal_1d_5d_correct_list.append(th['1d_5d_correct_then_20d_accuracy'])

    if causal_1d_correct_list:
        avg = np.mean(causal_1d_correct_list)
        print(f"  {'1天正确→20天也正确':<30} {hsi_baseline['causal_analysis']['1d_correct_then_20d_accuracy']:.2f}%{'':<6} "
              f"{avg:.2f}%{'':<6} {hsi_baseline['causal_analysis']['1d_correct_then_20d_accuracy'] - avg:+.2f}%")
    if causal_1d_wrong_list:
        avg = np.mean(causal_1d_wrong_list)
        print(f"  {'1天错误→20天正确':<30} {hsi_baseline['causal_analysis']['1d_wrong_then_20d_accuracy']:.2f}%{'':<6} "
              f"{avg:.2f}%{'':<6} {hsi_baseline['causal_analysis']['1d_wrong_then_20d_accuracy'] - avg:+.2f}%")
    if causal_5d_correct_list:
        avg = np.mean(causal_5d_correct_list)
        print(f"  {'5天正确→20天也正确':<30} {hsi_baseline['causal_analysis']['5d_correct_then_20d_accuracy']:.2f}%{'':<6} "
              f"{avg:.2f}%{'':<6} {hsi_baseline['causal_analysis']['5d_correct_then_20d_accuracy'] - avg:+.2f}%")
    if causal_1d_5d_correct_list:
        avg = np.mean(causal_1d_5d_correct_list)
        print(f"  {'1+5天正确→20天也正确':<30} {hsi_baseline['causal_analysis']['1d_5d_correct_then_20d_accuracy']:.2f}%{'':<6} "
              f"{avg:.2f}%{'':<6} {hsi_baseline['causal_analysis']['1d_5d_correct_then_20d_accuracy'] - avg:+.2f}%")

    # ========== 3. 八大模式对比 ==========
    print("\n【八大交易模式对比（20天准确率）】")
    print(f"  {'模式':<8} {'名称':<12} {'恒指':<12} {'个股平均':<12} {'差异':<10}")
    print("  " + "-" * 50)

    pattern_names = {
        '101': '假突破⭐',
        '010': '反弹失败⭐',
        '001': '下跌中继',
        '111': '一致看涨',
        '000': '一致看跌',
        '110': '震荡回调',
        '011': '探底回升',
        '100': '冲高回落',
    }

    for pattern in ['101', '010', '001', '111', '000', '110', '011', '100']:
        pattern_acc_list = []
        for code, result in all_stock_results.items():
            if 'three_horizon' in result and 'patterns' in result['three_horizon']:
                if pattern in result['three_horizon']['patterns']:
                    pattern_acc_list.append(result['three_horizon']['patterns'][pattern]['20d_accuracy'])

        hsi_acc = hsi_baseline['patterns'].get(pattern, {}).get('20d_accuracy', 0)
        if pattern_acc_list:
            avg_acc = np.mean(pattern_acc_list)
            diff = hsi_acc - avg_acc
            print(f"  {pattern:<8} {pattern_names.get(pattern, ''):<12} "
                  f"{hsi_acc:.2f}%{'':<6} {avg_acc:.2f}%{'':<6} {diff:+.2f}%")
        elif hsi_acc > 0:
            print(f"  {pattern:<8} {pattern_names.get(pattern, ''):<12} "
                  f"{hsi_acc:.2f}%{'':<6} {'-':<12}")

    # ========== 4. 概率相关性对比 ==========
    print("\n【概率相关性对比】")
    print(f"  {'指标':<30} {'恒指':<12} {'个股平均':<12} {'说明':<20}")
    print("  " + "-" * 70)

    prob_corr_list = []
    for code, result in all_stock_results.items():
        if 'three_horizon' in result:
            pc = result['three_horizon'].get('probability_correlation', {})
            if '5d_vs_20d_actual' in pc:
                prob_corr_list.append(pc['5d_vs_20d_actual'])

    if prob_corr_list:
        avg_corr = np.mean(prob_corr_list)
        hsi_corr = hsi_baseline['probability_correlation']['5d_vs_20d_actual']
        direction = "正向（顺势）" if avg_corr > 0 else "反向（反转）"
        print(f"  {'5d预测概率与20d实际方向相关性 r':<30} "
              f"+{hsi_corr:.2f}{'':<8} {avg_corr:.4f}{'':<6} {direction}")
        print(f"\n  💡 关键发现：")
        if avg_corr < 0:
            print(f"     ⚠️ 个股预测概率与实际方向呈负相关（r={avg_corr:.4f})")
            print(f"     ⚠️ 高预测概率反而预示反转，与恒指完全相反")
            print(f"     ⚠️ 不能将恒指策略直接套用于个股")
        else:
            print(f"     ✅ 个股预测概率与实际方向呈正相关（r={avg_corr:.4f})")


def print_summary(all_stock_results):
    """打印汇总对比表"""
    hsi_baseline = {
        '1d_to_5d': {'model_accuracy': 64.14, 'majority_accuracy': 52.15,
                     'strong_accuracy': 64.69, 'mixed_accuracy': 63.12,
                     'transmission_corr': 0.0358, 'prob_corr': 0.0246},
        '5d_to_20d': {'model_accuracy': 82.76, 'majority_accuracy': 66.38,
                      'strong_accuracy': 84.36, 'mixed_accuracy': 78.70,
                      'transmission_corr': 0.0811, 'prob_corr': 0.4294},
    }

    print("\n" + "=" * 90)
    print("📊 个股（完整模型）vs 恒指 因果链对比总结")
    print("=" * 90)

    # 1d→5d
    print(f"\n【1d→5d 因果链（完整模型）】")
    print(f"  {'股票':<16} {'模型准确率':<12} {'投票准确率':<12} {'强信号':<10} {'混合信号':<10} {'传导r':<10} {'概率r':<10}")
    print("  " + "-" * 80)

    model_accs, vote_accs, strong_accs, mixed_accs, trans_corrs, prob_corrs = [], [], [], [], [], []

    for code, result in sorted(all_stock_results.items()):
        if '1d_to_5d' not in result:
            continue
        d = result['1d_to_5d']
        model_accs.append(d['model_accuracy'])
        vote_accs.append(d['majority_accuracy'])
        strong_accs.append(d.get('strong_accuracy', d['model_accuracy']))
        mixed_accs.append(d.get('mixed_accuracy', d['model_accuracy']))
        trans_corrs.append(d['transmission_corr'])
        prob_corrs.append(d['prob_corr'])

        print(f"  {result['stock_name']:<16} {d['model_accuracy']:.2f}%{'':<6} "
              f"{d['majority_accuracy']:.2f}%{'':<6} "
              f"{d.get('strong_accuracy', 0):.2f}%{'':<4} "
              f"{d.get('mixed_accuracy', 0):.2f}%{'':<4} "
              f"{d['transmission_corr']:.4f}{'':<4} "
              f"{d['prob_corr']:.4f}")

    print(f"  {'恒生指数':<16} {hsi_baseline['1d_to_5d']['model_accuracy']:.2f}%{'':<6} "
          f"{hsi_baseline['1d_to_5d']['majority_accuracy']:.2f}%{'':<6} "
          f"{hsi_baseline['1d_to_5d']['strong_accuracy']:.2f}%{'':<4} "
          f"{hsi_baseline['1d_to_5d']['mixed_accuracy']:.2f}%{'':<4} "
          f"{hsi_baseline['1d_to_5d']['transmission_corr']:.4f}{'':<4} "
          f"{hsi_baseline['1d_to_5d']['prob_corr']:.4f}")

    if model_accs:
        print(f"  {'--- 个股平均 ---':<16} {np.mean(model_accs):.2f}%{'':<6} "
              f"{np.mean(vote_accs):.2f}%{'':<6} "
              f"{np.mean(strong_accs):.2f}%{'':<4} "
              f"{np.mean(mixed_accs):.2f}%{'':<4} "
              f"{np.mean(trans_corrs):.4f}{'':<4} "
              f"{np.mean(prob_corrs):.4f}")

    # 5d→20d
    print(f"\n【5d→20d 因果链（完整模型）】")
    print(f"  {'股票':<16} {'模型准确率':<12} {'投票准确率':<12} {'强信号':<10} {'混合信号':<10} {'传导r':<10} {'概率r':<10}")
    print("  " + "-" * 80)

    model_accs2, vote_accs2, strong_accs2, mixed_accs2, trans_corrs2, prob_corrs2 = [], [], [], [], [], []

    for code, result in sorted(all_stock_results.items()):
        if '5d_to_20d' not in result:
            continue
        d = result['5d_to_20d']
        model_accs2.append(d['model_accuracy'])
        vote_accs2.append(d['majority_accuracy'])
        strong_accs2.append(d.get('strong_accuracy', d['model_accuracy']))
        mixed_accs2.append(d.get('mixed_accuracy', d['model_accuracy']))
        trans_corrs2.append(d['transmission_corr'])
        prob_corrs2.append(d['prob_corr'])

        print(f"  {result['stock_name']:<16} {d['model_accuracy']:.2f}%{'':<6} "
              f"{d['majority_accuracy']:.2f}%{'':<6} "
              f"{d.get('strong_accuracy', 0):.2f}%{'':<4} "
              f"{d.get('mixed_accuracy', 0):.2f}%{'':<4} "
              f"{d['transmission_corr']:.4f}{'':<4} "
              f"{d['prob_corr']:.4f}")

    print(f"  {'恒生指数':<16} {hsi_baseline['5d_to_20d']['model_accuracy']:.2f}%{'':<6} "
          f"{hsi_baseline['5d_to_20d']['majority_accuracy']:.2f}%{'':<6} "
          f"{hsi_baseline['5d_to_20d']['strong_accuracy']:.2f}%{'':<4} "
          f"{hsi_baseline['5d_to_20d']['mixed_accuracy']:.2f}%{'':<4} "
          f"{hsi_baseline['5d_to_20d']['transmission_corr']:.4f}{'':<4} "
          f"{hsi_baseline['5d_to_20d']['prob_corr']:.4f}")

    if model_accs2:
        print(f"  {'--- 个股平均 ---':<16} {np.mean(model_accs2):.2f}%{'':<6} "
              f"{np.mean(vote_accs2):.2f}%{'':<6} "
              f"{np.mean(strong_accs2):.2f}%{'':<4} "
              f"{np.mean(mixed_accs2):.2f}%{'':<4} "
              f"{np.mean(trans_corrs2):.4f}{'':<4} "
              f"{np.mean(prob_corrs2):.4f}")

    # ========== 核心差异 ==========
    print(f"\n{'='*90}")
    print("💡 恒指 vs 个股 核心差异")
    print(f"{'='*90}")

    if model_accs and model_accs2:
        print(f"\n  【1d→5d 链】")
        print(f"    恒指模型准确率: {hsi_baseline['1d_to_5d']['model_accuracy']:.2f}%")
        print(f"    个股平均准确率: {np.mean(model_accs):.2f}%  (差: {hsi_baseline['1d_to_5d']['model_accuracy'] - np.mean(model_accs):+.2f}%)")
        print(f"    恒指传导r: {hsi_baseline['1d_to_5d']['transmission_corr']:.4f}, 个股平均: {np.mean(trans_corrs):.4f}")
        print(f"    恒指概率r: {hsi_baseline['1d_to_5d']['prob_corr']:.4f}, 个股平均: {np.mean(prob_corrs):.4f}")

        print(f"\n  【5d→20d 链】")
        print(f"    恒指模型准确率: {hsi_baseline['5d_to_20d']['model_accuracy']:.2f}%")
        print(f"    个股平均准确率: {np.mean(model_accs2):.2f}%  (差: {hsi_baseline['5d_to_20d']['model_accuracy'] - np.mean(model_accs2):+.2f}%)")
        print(f"    恒指传导r: {hsi_baseline['5d_to_20d']['transmission_corr']:.4f}, 个股平均: {np.mean(trans_corrs2):.4f}")
        print(f"    恒指概率r: {hsi_baseline['5d_to_20d']['prob_corr']:.4f}, 个股平均: {np.mean(prob_corrs2):.4f}")

        # 一致性判断
        hsi_1d5d_gap = hsi_baseline['1d_to_5d']['model_accuracy'] - hsi_baseline['1d_to_5d']['majority_accuracy']
        stock_1d5d_gap = np.mean(model_accs) - np.mean(vote_accs)
        hsi_5d20d_gap = hsi_baseline['5d_to_20d']['model_accuracy'] - hsi_baseline['5d_to_20d']['majority_accuracy']
        stock_5d20d_gap = np.mean(model_accs2) - np.mean(vote_accs2)

        print(f"\n  【模型 vs 投票差距】")
        print(f"    1d→5d: 恒指 {hsi_1d5d_gap:+.2f}% vs 个股 {stock_1d5d_gap:+.2f}%")
        print(f"    5d→20d: 恒指 {hsi_5d20d_gap:+.2f}% vs 个股 {stock_5d20d_gap:+.2f}%")

        # 结论
        print(f"\n  【结论】")
        same_1d5d = (hsi_1d5d_gap > 5) == (stock_1d5d_gap > 5)
        same_5d20d = (hsi_5d20d_gap > 5) == (stock_5d20d_gap > 5)

        if same_1d5d and same_5d20d:
            print(f"    ✅ 恒指和个股的因果链模式一致：模型直接预测始终优于子预测积分")
        else:
            print(f"    ⚠️ 恒指和个股的因果链模式不完全一致")

        if np.mean(prob_corrs2) > 0.3:
            print(f"    ✅ 5d→20d 概率传导在个股中也较强（r={np.mean(prob_corrs2):.4f}）")
        else:
            print(f"    ⚠️ 5d→20d 概率传导在个股中较弱（r={np.mean(prob_corrs2):.4f}）")


def main():
    print("=" * 90)
    print("个股因果链分析（完整模型 918 特征）：与恒指对比")
    print("=" * 90)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 代表性股票（每板块1-2只）
    sample_stocks = {
        '0005.HK': '汇丰银行', '1288.HK': '农业银行',
        '0700.HK': '腾讯控股', '9988.HK': '阿里巴巴',
        '0981.HK': '中芯国际',
        '1211.HK': '比亚迪',
        '1299.HK': '友邦保险', '2318.HK': '中国平安',
        '0728.HK': '中国电信',
        '0388.HK': '香港交易所',
        '3690.HK': '美团',
        '0883.HK': '中海油',
    }

    feature_engineer = FullFeatureEngineer()
    all_stock_results = {}

    for i, (stock_code, stock_name) in enumerate(sample_stocks.items()):
        print(f"\n[{i+1}/{len(sample_stocks)}] 📊 分析 {stock_code} {stock_name}...")

        predictions = run_single_stock_walkforward(stock_code, stock_name, feature_engineer)
        if predictions is None:
            continue

        df_1d_5d = analyze_causal_chain(predictions, '1d_to_5d')
        df_5d_20d = analyze_causal_chain(predictions, '5d_to_20d')

        stock_result = {'stock_code': stock_code, 'stock_name': stock_name}

        if df_1d_5d is not None and len(df_1d_5d) > 30:
            stock_result['1d_to_5d'] = {
                'samples': len(df_1d_5d),
                'model_accuracy': float(df_1d_5d['long_correct'].mean() * 100),
                'majority_accuracy': float(df_1d_5d['majority_correct'].mean() * 100),
                'transmission_corr': float(df_1d_5d['correct_short_count'].corr(df_1d_5d['long_correct'].astype(float))),
                'prob_corr': float(df_1d_5d['avg_prob_short'].corr(df_1d_5d['actual_long'].astype(float))),
            }
            strong = df_1d_5d[(df_1d_5d['consensus_count'] >= 4) | (df_1d_5d['consensus_count'] <= 1)]
            mixed = df_1d_5d[(df_1d_5d['consensus_count'] >= 2) & (df_1d_5d['consensus_count'] <= 3)]
            if len(strong) > 0:
                stock_result['1d_to_5d']['strong_accuracy'] = float(strong['long_correct'].mean() * 100)
            if len(mixed) > 0:
                stock_result['1d_to_5d']['mixed_accuracy'] = float(mixed['long_correct'].mean() * 100)

        if df_5d_20d is not None and len(df_5d_20d) > 20:
            stock_result['5d_to_20d'] = {
                'samples': len(df_5d_20d),
                'model_accuracy': float(df_5d_20d['long_correct'].mean() * 100),
                'majority_accuracy': float(df_5d_20d['majority_correct'].mean() * 100),
                'transmission_corr': float(df_5d_20d['correct_short_count'].corr(df_5d_20d['long_correct'].astype(float))),
                'prob_corr': float(df_5d_20d['avg_prob_short'].corr(df_5d_20d['actual_long'].astype(float))),
            }
            strong = df_5d_20d[(df_5d_20d['consensus_count'] >= 3) | (df_5d_20d['consensus_count'] <= 1)]
            mixed = df_5d_20d[(df_5d_20d['consensus_count'] == 2)]
            if len(strong) > 0:
                stock_result['5d_to_20d']['strong_accuracy'] = float(strong['long_correct'].mean() * 100)
            if len(mixed) > 0:
                stock_result['5d_to_20d']['mixed_accuracy'] = float(mixed['long_correct'].mean() * 100)

        # 新增：三周期模式分析
        three_horizon_results = analyze_three_horizon_patterns(predictions)
        if three_horizon_results is not None:
            stock_result['three_horizon'] = three_horizon_results

        all_stock_results[stock_code] = stock_result
        res = stock_result
        print(f"  ✅ 1d→5d: {res.get('1d_to_5d',{}).get('model_accuracy','N/A')}%, "
              f"5d→20d: {res.get('5d_to_20d',{}).get('model_accuracy','N/A')}%")
        if 'three_horizon' in res:
            th = res['three_horizon']
            print(f"     三周期模式: {len(th.get('patterns', {}))}种, "
                  f"20d准确率: {th.get('independent_accuracy', {}).get('20d', 'N/A'):.2f}%")

    # 汇总对比
    print_summary(all_stock_results)

    # 打印三周期模式汇总
    print_three_horizon_summary(all_stock_results)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'stock_causal_chain_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_stock_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存到: {output_path}")

    print(f"\n{'='*90}")
    print(f"分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
