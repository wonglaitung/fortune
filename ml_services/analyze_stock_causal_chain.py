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

        all_stock_results[stock_code] = stock_result
        res = stock_result
        print(f"  ✅ 1d→5d: {res.get('1d_to_5d',{}).get('model_accuracy','N/A')}%, "
              f"5d→20d: {res.get('5d_to_20d',{}).get('model_accuracy','N/A')}%")

    # 汇总对比
    print_summary(all_stock_results)

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
