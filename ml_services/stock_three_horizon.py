#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股三周期预测验证脚本

支持两种模式：
1. 简化模型（--model simple）：约40个技术指标特征，快速验证
2. 完整模型（--model full）：918个特征（技术指标、基本面、情感指标等）

验证以下发现：
1. 传导律：短周期正确 → 长周期更可能正确
2. 过滤律：周期越长噪音越少
3. 一致律：三周期一致时准确率最高
4. 背离律：短涨长跌是逃顶信号（110模式）

使用方法：
  # 简化模型验证（默认）
  python3 ml_services/stock_three_horizon.py --stock 0700.HK
  python3 ml_services/stock_three_horizon.py --sector tech --model simple

  # 完整模型验证
  python3 ml_services/stock_three_horizon.py --stock 0700.HK --model full
  python3 ml_services/stock_three_horizon.py --quick --model full

  # 全量验证
  python3 ml_services/stock_three_horizon.py --all --model full
"""

import warnings
import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from config import STOCK_SECTOR_MAPPING, SECTOR_NAME_MAPPING

# ========== 配置 ==========
DATA_DIR = "data"
OUTPUT_DIR = "output"
RANDOM_SEED = 42

# 特征配置（简化模型，约40个技术指标）
SIMPLE_FEATURES = [
    # 移动平均线
    'MA20', 'MA60', 'MA120', 'MA250',
    # 均线斜率
    'MA250_Slope', 'MA250_Slope_5d', 'MA250_Slope_20d',
    # 价格距离
    'Price_Distance_MA250', 'Price_Distance_MA60', 'Price_Distance_MA20',
    # 均线排列
    'MA_Bullish_Alignment', 'MA_Bearish_Alignment',
    # 交叉信号
    'MA20_Golden_Cross_MA60', 'MA20_Death_Cross_MA60',
    'MA60_Golden_Cross_MA250', 'MA60_Death_Cross_MA250',
    # 波动率
    'Volatility_20d', 'Volatility_60d', 'Volatility_120d',
    # RSI
    'RSI', 'RSI_ROC',
    # MACD
    'MACD', 'MACD_Signal', 'MACD_Hist',
    # 布林带
    'BB_Width', 'BB_Position',
    # 收益率
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
    # ATR
    'ATR_Ratio',
    # 趋势强度
    'ADX',
    # 随机指标
    'Stoch_K', 'Stoch_D',
]

# CatBoost 参数（简化版）
SIMPLE_CATBOOST_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 3,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_SEED,
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'verbose': 0
}

# CatBoost 参数（完整版，按周期调整）
def get_full_catboost_params(horizon):
    """根据预测周期返回完整模型参数（优化版：减少迭代次数）"""
    if horizon == 1:
        return {
            'iterations': 100,
            'learning_rate': 0.15,
            'depth': 5,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_SEED,
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'verbose': 0
        }
    elif horizon == 5:
        return {
            'iterations': 100,
            'learning_rate': 0.15,
            'depth': 4,
            'l2_leaf_reg': 3,
            'subsample': 0.7,
            'random_seed': RANDOM_SEED,
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'verbose': 0
        }
    else:  # 20天
        return {
            'iterations': 100,
            'learning_rate': 0.15,
            'depth': 4,
            'l2_leaf_reg': 3,
            'subsample': 0.8,
            'random_seed': RANDOM_SEED,
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'verbose': 0
        }


class SimpleFeatureEngineer:
    """简化特征工程（约40个技术指标）"""

    def calculate_features(self, df):
        """计算技术特征"""
        df = df.copy()

        # 移动平均线
        for window in [20, 60, 120, 250]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

        # 均线斜率
        df['MA250_Slope'] = df['MA250'].diff()
        df['MA250_Slope_5d'] = df['MA250'].diff(5) / 5
        df['MA250_Slope_20d'] = df['MA250'].diff(20) / 20

        # 价格距离
        df['Price_Distance_MA250'] = (df['Close'] - df['MA250']) / df['MA250'] * 100
        df['Price_Distance_MA60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
        df['Price_Distance_MA20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100

        # 均线排列
        df['MA_Bullish_Alignment'] = ((df['MA20'] > df['MA60']) &
                                       (df['MA60'] > df['MA250'])).astype(int)
        df['MA_Bearish_Alignment'] = ((df['MA20'] < df['MA60']) &
                                       (df['MA60'] < df['MA250'])).astype(int)

        # 交叉信号
        df['MA20_Golden_Cross_MA60'] = ((df['MA20'] > df['MA60']) &
                                         (df['MA20'].shift(1) <= df['MA60'].shift(1))).astype(int)
        df['MA20_Death_Cross_MA60'] = ((df['MA20'] < df['MA60']) &
                                        (df['MA20'].shift(1) >= df['MA60'].shift(1))).astype(int)
        df['MA60_Golden_Cross_MA250'] = ((df['MA60'] > df['MA250']) &
                                          (df['MA60'].shift(1) <= df['MA250'].shift(1))).astype(int)
        df['MA60_Death_Cross_MA250'] = ((df['MA60'] < df['MA250']) &
                                         (df['MA60'].shift(1) >= df['MA250'].shift(1))).astype(int)

        # 收益率（使用昨日值避免数据泄漏）
        df['Return_1d'] = df['Close'].pct_change().shift(1)
        df['Return_5d'] = df['Close'].pct_change(5).shift(1)
        df['Return_10d'] = df['Close'].pct_change(10).shift(1)
        df['Return_20d'] = df['Close'].pct_change(20).shift(1)

        # 波动率
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()
        df['Volatility_120d'] = df['Return_1d'].rolling(window=120).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1)
        df['RSI_ROC'] = df['RSI'].pct_change()

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26).shift(1)
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = ((df['Close'].shift(1) - df['BB_Lower']) /
                              (df['BB_Upper'] - df['BB_Lower'] + 1e-10))

        # ATR
        high_prev = df['High'].shift(1)
        low_prev = df['Low'].shift(1)
        close_prev = df['Close'].shift(1)
        tr1 = high_prev - low_prev
        tr2 = abs(high_prev - close_prev)
        tr3 = abs(low_prev - close_prev)
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_MA'] = df['ATR'].rolling(window=10).mean().shift(1)
        df['ATR_Ratio'] = df['ATR'] / df['ATR_MA']

        # ADX
        up_move = df['High'].diff().shift(1)
        down_move = -df['Low'].diff().shift(1)
        df['Plus_DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['Minus_DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['Plus_DI'] = 100 * (df['Plus_DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        df['Minus_DI'] = 100 * (df['Minus_DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        dx = 100 * (np.abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'] + 1e-10))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # 随机指标
        df['Low_Min_14'] = df['Low'].rolling(window=14, min_periods=1).min().shift(1)
        df['High_Max_14'] = df['High'].rolling(window=14, min_periods=1).max().shift(1)
        df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min_14']) / (df['High_Max_14'] - df['Low_Min_14'] + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean().shift(1)

        return df


class FullFeatureEngineer:
    """完整特征工程（918个特征，调用 ml_trading_model.py）"""

    def __init__(self):
        from ml_services.ml_trading_model import FeatureEngineer
        self.feature_engineer = FeatureEngineer()

    def calculate_features(self, df, stock_code):
        """
        计算完整特征（918个）

        注意：完整模型需要额外数据（恒指、美股市场等）
        这里只计算股票自身特征，约200+个
        """
        df = df.copy()

        # 计算技术指标
        df = self.feature_engineer.calculate_technical_features(df)
        df = self.feature_engineer.calculate_multi_period_metrics(df)

        # 添加基本面特征
        fundamental_features = self.feature_engineer.create_fundamental_features(stock_code)
        for key, value in fundamental_features.items():
            df[key] = value

        # 添加股票类型特征
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        for key, value in stock_type_features.items():
            df[key] = value

        # 添加情感特征
        sentiment_features = self.feature_engineer.create_sentiment_features(stock_code, df)
        for key, value in sentiment_features.items():
            df[key] = value

        # 添加板块特征
        sector_features = self.feature_engineer.create_sector_features(stock_code, df)
        for key, value in sector_features.items():
            df[key] = value

        # 添加事件驱动特征
        df = self.feature_engineer.create_event_driven_features(stock_code, df)

        # 生成交互特征
        df = self.feature_engineer.create_technical_fundamental_interactions(df)
        df = self.feature_engineer.create_interaction_features(df)

        return df


class StockThreeHorizonAnalyzer:
    """个股三周期预测分析器"""

    def __init__(self, model_type='simple'):
        """
        参数:
        - model_type: 'simple'（简化模型，约40特征）或 'full'（完整模型，918特征）
        """
        self.model_type = model_type
        self.results = {}

        if model_type == 'simple':
            self.feature_engineer = SimpleFeatureEngineer()
            self.features = SIMPLE_FEATURES
            print(f"📊 使用简化模型（约{len(self.features)}个特征）")
        else:
            self.feature_engineer = FullFeatureEngineer()
            self.features = None  # 完整模型动态获取特征
            print(f"📊 使用完整模型（918个特征）")

    def fetch_stock_data(self, stock_code, start_date='2020-01-01', end_date='2025-12-31'):
        """获取个股历史数据"""
        try:
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty or len(df) < 300:
                return None

            return df
        except Exception as e:
            print(f"  ⚠️ 获取 {stock_code} 数据失败: {e}")
            return None

    def run_walkforward(self, df, stock_code, horizons=[1, 5, 20]):
        """
        Walk-forward 验证

        参数:
        - df: 特征数据
        - stock_code: 股票代码
        - horizons: 预测周期列表

        返回:
        - dict: 各周期的预测结果
        """
        all_predictions = {h: [] for h in horizons}

        # 确定特征
        if self.model_type == 'simple':
            available_features = [f for f in self.features if f in df.columns]
        else:
            # 完整模型：排除非特征列
            exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                              'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                              'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                              'BB_upper', 'BB_lower', 'BB_middle', 'Dividends', 'Stock Splits']
            available_features = [col for col in df.columns if col not in exclude_columns]

            # 过滤非数值特征（分类特征需要特殊处理，这里简化排除）
            numeric_features = [col for col in available_features
                               if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            if len(numeric_features) < len(available_features):
                removed = len(available_features) - len(numeric_features)
                print(f"  🔧 排除 {removed} 个非数值特征")
            available_features = numeric_features

            # 过滤高NaN率特征（>50% NaN）
            nan_ratio = df[available_features].isna().sum() / len(df)
            valid_features = nan_ratio[nan_ratio <= 0.5].index.tolist()
            if len(valid_features) < len(available_features):
                removed = len(available_features) - len(valid_features)
                print(f"  🔧 移除 {removed} 个高NaN率特征（>50% NaN）")
            available_features = valid_features

        if len(available_features) < 10:
            print(f"  ⚠️ {stock_code} 特征不足（{len(available_features)}），跳过")
            return None

        print(f"  📊 使用 {len(available_features)} 个特征")

        # 填充剩余NaN值
        for col in available_features:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Walk-forward 参数（优化版：减少folds数量）
        train_days = 252
        # 完整模型使用更大的step_days以加速（从5改为40）
        step_days = 40 if self.model_type == 'full' else 5
        total_days = len(df)
        num_folds = (total_days - train_days) // step_days

        if num_folds < 5:
            print(f"  ⚠️ {stock_code} 样本不足（{num_folds} folds），跳过")
            return None

        print(f"  📊 Walk-forward: {num_folds} folds, step={step_days}天")

        # 预先计算所有周期的目标
        for horizon in horizons:
            target_col = f'Target_{horizon}d'
            return_col = f'Future_Return_{horizon}d'
            df[return_col] = df['Close'].pct_change(horizon).shift(-horizon)
            df[target_col] = (df[return_col] > 0).astype(int)

        for fold in range(num_folds):
            train_start = fold * step_days
            train_end = train_start + train_days
            test_start = train_end
            test_end = min(test_start + step_days, total_days)

            if test_start >= total_days:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            for horizon in horizons:
                target_col = f'Target_{horizon}d'
                return_col = f'Future_Return_{horizon}d'

                # 移除 NaN
                train_clean = train_df.dropna(subset=available_features + [target_col])
                test_clean = test_df.dropna(subset=available_features + [target_col, return_col])

                if len(train_clean) < 50 or len(test_clean) < 1:
                    continue

                X_train = train_clean[available_features]
                y_train = train_clean[target_col]

                try:
                    # 根据模型类型选择参数
                    if self.model_type == 'simple':
                        model = CatBoostClassifier(**SIMPLE_CATBOOST_PARAMS)
                    else:
                        model = CatBoostClassifier(**get_full_catboost_params(horizon))

                    model.fit(X_train, y_train)

                    # 批量预测提高效率
                    X_test = test_clean[available_features]
                    probs = model.predict_proba(X_test)[:, 1]
                    preds = (probs > 0.5).astype(int)

                    for i, (idx, row) in enumerate(test_clean.iterrows()):
                        all_predictions[horizon].append({
                            'date': idx,
                            'prediction': preds[i],
                            'probability': probs[i],
                            'actual_direction': row[target_col],
                            'actual_return': row[return_col],
                            'close': row['Close']
                        })
                except Exception as e:
                    print(f"  ⚠️ Fold {fold} horizon {horizon} 错误: {e}")
                    continue

        return all_predictions

    def analyze_three_horizon_patterns(self, predictions, stock_code):
        """
        分析三周期模式

        返回:
        - dict: 模式分析结果
        """
        if not predictions or not predictions.get(1) or not predictions.get(5) or not predictions.get(20):
            return None

        # 构建 DataFrame
        df_1d = pd.DataFrame(predictions[1])
        df_5d = pd.DataFrame(predictions[5])
        df_20d = pd.DataFrame(predictions[20])

        if len(df_1d) < 50 or len(df_5d) < 50 or len(df_20d) < 50:
            return None

        df_1d.set_index('date', inplace=True)
        df_5d.set_index('date', inplace=True)
        df_20d.set_index('date', inplace=True)

        # 合并数据
        merged = pd.DataFrame({
            'pred_1d': df_1d['prediction'],
            'prob_1d': df_1d['probability'],
            'actual_1d': df_1d['actual_direction'],
            'return_1d': df_1d['actual_return'],
            'pred_5d': df_5d['prediction'],
            'prob_5d': df_5d['probability'],
            'actual_5d': df_5d['actual_direction'],
            'return_5d': df_5d['actual_return'],
            'pred_20d': df_20d['prediction'],
            'prob_20d': df_20d['probability'],
            'actual_20d': df_20d['actual_direction'],
            'return_20d': df_20d['actual_return'],
        }).dropna()

        if len(merged) < 30:
            return None

        results = {
            'stock_code': stock_code,
            'stock_name': STOCK_SECTOR_MAPPING.get(stock_code, {}).get('name', stock_code),
            'sector': STOCK_SECTOR_MAPPING.get(stock_code, {}).get('sector', 'unknown'),
            'sample_count': len(merged),
            'horizon_accuracy': {},
            'patterns': {},
            'transmission_effect': {},
        }

        # 1. 各周期独立准确率
        for horizon, col_prefix in [(1, '1d'), (5, '5d'), (20, '20d')]:
            actual_col = f'actual_{col_prefix}'
            pred_col = f'pred_{col_prefix}'
            results['horizon_accuracy'][horizon] = {
                'accuracy': (merged[pred_col] == merged[actual_col]).mean(),
                'sample_count': len(merged)
            }

        # 2. 八大模式分析
        merged['pattern'] = merged['pred_1d'].astype(str) + '-' + \
                            merged['pred_5d'].astype(str) + '-' + \
                            merged['pred_20d'].astype(str)

        pattern_names = {
            '1-1-1': '一致看涨',
            '1-1-0': '震荡回调',
            '1-0-1': '假突破',
            '1-0-0': '冲高回落',
            '0-1-1': '探底回升',
            '0-1-0': '反弹失败',
            '0-0-1': '下跌中继',
            '0-0-0': '一致看跌',
        }

        for pattern, name in pattern_names.items():
            pattern_df = merged[merged['pattern'] == pattern]
            if len(pattern_df) >= 5:
                accuracy_20d = (pattern_df['pred_20d'] == pattern_df['actual_20d']).mean()
                avg_return = pattern_df['return_20d'].mean()
                results['patterns'][pattern] = {
                    'name': name,
                    'count': len(pattern_df),
                    'accuracy_20d': accuracy_20d,
                    'avg_return': avg_return
                }

        # 3. 传导效应分析
        correct_1d = merged[merged['pred_1d'] == merged['actual_1d']]
        if len(correct_1d) >= 5:
            results['transmission_effect']['1d_correct_to_20d'] = {
                'accuracy': (correct_1d['pred_20d'] == correct_1d['actual_20d']).mean(),
                'sample_count': len(correct_1d)
            }

        correct_5d = merged[merged['pred_5d'] == merged['actual_5d']]
        if len(correct_5d) >= 5:
            results['transmission_effect']['5d_correct_to_20d'] = {
                'accuracy': (correct_5d['pred_20d'] == correct_5d['actual_20d']).mean(),
                'sample_count': len(correct_5d)
            }

        correct_both = merged[(merged['pred_1d'] == merged['actual_1d']) &
                              (merged['pred_5d'] == merged['actual_5d'])]
        if len(correct_both) >= 5:
            results['transmission_effect']['both_correct_to_20d'] = {
                'accuracy': (correct_both['pred_20d'] == correct_both['actual_20d']).mean(),
                'sample_count': len(correct_both)
            }

        return results

    def analyze_stock(self, stock_code, verbose=True):
        """分析单只股票"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 分析: {stock_code} ({STOCK_SECTOR_MAPPING.get(stock_code, {}).get('name', 'Unknown')})")
            print(f"{'='*60}")

        start_time = time.time()

        # 获取数据
        df = self.fetch_stock_data(stock_code)
        if df is None:
            return None

        if verbose:
            print(f"  ✅ 数据获取成功（{len(df)} 条记录）")

        # 计算特征
        if self.model_type == 'simple':
            df = self.feature_engineer.calculate_features(df)
        else:
            df = self.feature_engineer.calculate_features(df, stock_code)

        if verbose:
            print(f"  ✅ 特征计算完成")

        # Walk-forward 验证
        if verbose:
            print(f"  🔄 运行 Walk-forward 验证...")

        predictions = self.run_walkforward(df, stock_code)

        if predictions is None:
            return None

        # 分析三周期模式
        results = self.analyze_three_horizon_patterns(predictions, stock_code)

        if results is None:
            return None

        elapsed_time = time.time() - start_time

        # 打印结果
        if verbose:
            print(f"\n  📈 三周期准确率:")
            for h, acc in results['horizon_accuracy'].items():
                print(f"    {h}天: {acc['accuracy']:.2%} ({acc['sample_count']} 样本)")

            if results['transmission_effect']:
                print(f"\n  🔗 传导效应:")
                for effect, data in results['transmission_effect'].items():
                    print(f"    {effect}: {data['accuracy']:.2%} ({data['sample_count']} 样本)")

            if results['patterns']:
                print(f"\n  📋 模式分析（20天准确率）:")
                for pattern, data in sorted(results['patterns'].items(),
                                            key=lambda x: x[1]['accuracy_20d'], reverse=True)[:5]:
                    print(f"    {data['name']} ({pattern}): {data['accuracy_20d']:.2%} "
                          f"({data['count']} 样本, 收益: {data['avg_return']:.2%})")

            print(f"\n  ⏱️ 耗时: {elapsed_time:.1f}秒")

        self.results[stock_code] = results
        return results

    def analyze_multiple_stocks(self, stock_codes, verbose=True):
        """批量分析多只股票"""
        print(f"\n{'='*60}")
        print(f"🔬 个股三周期策略批量验证（{self.model_type}模型）")
        print(f"{'='*60}")
        print(f"股票数量: {len(stock_codes)}")

        success_count = 0
        for i, stock_code in enumerate(stock_codes):
            if verbose:
                print(f"\n[{i+1}/{len(stock_codes)}] 处理 {stock_code}...")

            result = self.analyze_stock(stock_code, verbose=verbose)
            if result:
                success_count += 1

        print(f"\n{'='*60}")
        print(f"✅ 分析完成: {success_count}/{len(stock_codes)} 只股票")
        print(f"{'='*60}")

        return self.results

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            print("⚠️ 没有分析结果")
            return None

        print(f"\n{'='*80}")
        print(f"📊 个股三周期策略验证汇总报告（{self.model_type}模型）")
        print(f"{'='*80}")

        # 汇总准确率
        all_1d = []
        all_5d = []
        all_20d = []

        for stock_code, result in self.results.items():
            if result.get('horizon_accuracy'):
                if 1 in result['horizon_accuracy']:
                    all_1d.append(result['horizon_accuracy'][1]['accuracy'])
                if 5 in result['horizon_accuracy']:
                    all_5d.append(result['horizon_accuracy'][5]['accuracy'])
                if 20 in result['horizon_accuracy']:
                    all_20d.append(result['horizon_accuracy'][20]['accuracy'])

        print(f"\n📈 各周期平均准确率:")
        print(f"  1天: {np.mean(all_1d):.2%} (范围: {np.min(all_1d):.2%} ~ {np.max(all_1d):.2%})")
        print(f"  5天: {np.mean(all_5d):.2%} (范围: {np.min(all_5d):.2%} ~ {np.max(all_5d):.2%})")
        print(f"  20天: {np.mean(all_20d):.2%} (范围: {np.min(all_20d):.2%} ~ {np.max(all_20d):.2%})")

        # 与恒指对比
        print(f"\n📊 与恒生指数对比:")
        hsi_benchmark = {1: 0.5131, 5: 0.5717, 20: 0.8073}
        for h, avg in [(1, np.mean(all_1d)), (5, np.mean(all_5d)), (20, np.mean(all_20d))]:
            diff = avg - hsi_benchmark[h]
            print(f"  {h}天: 个股 {avg:.2%} vs 恒指 {hsi_benchmark[h]:.2%} ({'+' if diff > 0 else ''}{diff:.2%})")

        # 模式汇总
        pattern_summary = {}
        for stock_code, result in self.results.items():
            for pattern, data in result.get('patterns', {}).items():
                if pattern not in pattern_summary:
                    pattern_summary[pattern] = {'count': 0, 'accuracy_sum': 0, 'return_sum': 0}
                pattern_summary[pattern]['count'] += data['count']
                pattern_summary[pattern]['accuracy_sum'] += data['accuracy_20d'] * data['count']
                pattern_summary[pattern]['return_sum'] += data['avg_return'] * data['count']

        print(f"\n📋 八大模式汇总（按20天准确率排序）:")
        print(f"{'模式':<12} {'样本数':>8} {'准确率':>10} {'平均收益':>10}")
        print("-" * 45)

        pattern_names = {
            '1-1-1': '一致看涨',
            '1-1-0': '震荡回调',
            '1-0-1': '假突破',
            '1-0-0': '冲高回落',
            '0-1-1': '探底回升',
            '0-1-0': '反弹失败',
            '0-0-1': '下跌中继',
            '0-0-0': '一致看跌',
        }

        sorted_patterns = sorted(pattern_summary.items(),
                                 key=lambda x: x[1]['accuracy_sum'] / x[1]['count'] if x[1]['count'] > 0 else 0,
                                 reverse=True)

        for pattern, data in sorted_patterns:
            if data['count'] > 0:
                accuracy = data['accuracy_sum'] / data['count']
                avg_return = data['return_sum'] / data['count']
                print(f"{pattern_names.get(pattern, pattern):<12} {data['count']:>8} {accuracy:>10.2%} {avg_return:>10.2%}")

        # 传导效应汇总
        transmission_1d = []
        transmission_5d = []
        transmission_both = []

        for stock_code, result in self.results.items():
            if result.get('transmission_effect'):
                if '1d_correct_to_20d' in result['transmission_effect']:
                    transmission_1d.append(result['transmission_effect']['1d_correct_to_20d']['accuracy'])
                if '5d_correct_to_20d' in result['transmission_effect']:
                    transmission_5d.append(result['transmission_effect']['5d_correct_to_20d']['accuracy'])
                if 'both_correct_to_20d' in result['transmission_effect']:
                    transmission_both.append(result['transmission_effect']['both_correct_to_20d']['accuracy'])

        print(f"\n🔗 传导效应汇总:")
        if transmission_1d:
            print(f"  1天正确→20天: {np.mean(transmission_1d):.2%}")
        if transmission_5d:
            print(f"  5天正确→20天: {np.mean(transmission_5d):.2%}")
        if transmission_both:
            print(f"  1+5天正确→20天: {np.mean(transmission_both):.2%} ⭐")

        # 关键发现
        print(f"\n💡 关键发现:")

        # 发现1：过滤律验证
        if np.mean(all_20d) > np.mean(all_1d):
            print(f"  ✅ 过滤律成立: 20天准确率({np.mean(all_20d):.2%}) > 1天({np.mean(all_1d):.2%})")
        else:
            print(f"  ⚠️ 过滤律不成立: 20天准确率({np.mean(all_20d):.2%}) <= 1天({np.mean(all_1d):.2%})")

        # 发现2：传导律验证
        if transmission_both and np.mean(transmission_both) > np.mean(all_20d):
            print(f"  ✅ 传导律成立: 1+5天正确后20天准确率({np.mean(transmission_both):.2%}) > 独立20天({np.mean(all_20d):.2%})")

        # 发现3：震荡回调模式（110）
        if '1-1-0' in pattern_summary and pattern_summary['1-1-0']['count'] > 0:
            acc_110 = pattern_summary['1-1-0']['accuracy_sum'] / pattern_summary['1-1-0']['count']
            print(f"  📌 震荡回调模式(110): 准确率 {acc_110:.2%} ({pattern_summary['1-1-0']['count']} 样本)")
            if acc_110 > 0.80:
                print(f"     ⭐ 该模式在个股中同样有效！")

        # 保存结果
        model_suffix = 'simple' if self.model_type == 'simple' else 'full'
        output_path = os.path.join(OUTPUT_DIR, f'stock_three_horizon_validation_{model_suffix}.json')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': self.model_type,
            'stock_count': len(self.results),
            'summary': {
                'avg_accuracy_1d': float(np.mean(all_1d)),
                'avg_accuracy_5d': float(np.mean(all_5d)),
                'avg_accuracy_20d': float(np.mean(all_20d)),
                'hsi_benchmark': hsi_benchmark,
            },
            'pattern_summary': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                    for kk, vv in v.items()}
                               for k, v in pattern_summary.items()},
            'stocks': {k: v for k, v in self.results.items()}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 结果已保存到: {output_path}")

        return output_data


def main():
    parser = argparse.ArgumentParser(description='个股三周期预测验证')
    parser.add_argument('--stock', type=str, help='单只股票代码（如 0700.HK）')
    parser.add_argument('--stocks', nargs='+', help='多只股票代码')
    parser.add_argument('--sector', type=str, help='板块类型（如 tech, bank）')
    parser.add_argument('--quick', action='store_true', help='快速验证（代表性股票）')
    parser.add_argument('--all', action='store_true', help='验证所有股票')
    parser.add_argument('--model', type=str, choices=['simple', 'full'], default='simple',
                        help='模型类型: simple（约40特征）或 full（918特征）')
    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')

    args = parser.parse_args()

    analyzer = StockThreeHorizonAnalyzer(model_type=args.model)

    # 确定要分析的股票
    stock_codes = []

    if args.stock:
        stock_codes = [args.stock]
    elif args.stocks:
        stock_codes = args.stocks
    elif args.sector:
        stock_codes = [code for code, info in STOCK_SECTOR_MAPPING.items()
                       if info.get('sector') == args.sector]
        print(f"📊 板块 '{args.sector}' 包含 {len(stock_codes)} 只股票")
    elif args.quick:
        quick_stocks = [
            '0700.HK',  # 腾讯
            '0005.HK',  # 汇丰
            '9988.HK',  # 阿里巴巴
            '2800.HK',  # 盈富基金
            '0941.HK',  # 中国移动
        ]
        stock_codes = [s for s in quick_stocks if s in STOCK_SECTOR_MAPPING]
        print(f"⚡ 快速验证模式：{len(stock_codes)} 只代表性股票")
    elif args.all:
        stock_codes = list(STOCK_SECTOR_MAPPING.keys())
        print(f"📊 全量验证：{len(stock_codes)} 只股票")

    if not stock_codes:
        print("⚠️ 请指定要分析的股票")
        print("使用方法:")
        print("  python3 ml_services/stock_three_horizon.py --stock 0700.HK")
        print("  python3 ml_services/stock_three_horizon.py --sector tech")
        print("  python3 ml_services/stock_three_horizon.py --quick")
        print("  python3 ml_services/stock_three_horizon.py --stock 0700.HK --model full")
        return

    # 执行分析
    analyzer.analyze_multiple_stocks(stock_codes, verbose=args.verbose)

    # 生成汇总报告
    analyzer.generate_summary_report()


if __name__ == '__main__':
    main()
