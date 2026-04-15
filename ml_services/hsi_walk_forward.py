#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数 CatBoost 模型 Walk-forward 验证

业界标准的时序验证方法，每个fold重新训练模型，评估真实预测能力

使用方法:
    python3 ml_services/hsi_walk_forward.py --horizon 20 --confidence-threshold 0.55
    python3 ml_services/hsi_walk_forward.py --train-window 12 --test-window 1
"""

import warnings
import os
import sys
import argparse
from datetime import datetime
import json
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import yfinance as yf

# 导入南向资金服务
from data_services.southbound_data import SouthboundDataService

# ========== 配置 ==========
HSI_SYMBOL = "^HSI"

# 特征配置（原始73特征集，经过验证效果最佳）
FEATURE_CONFIG = {
    'macro_features': ['US_10Y_Yield', 'US_10Y_Yield_Change_5d', 'VIX', 'VIX_Change_5d'],
    'southbound_features': ['Southbound_Net_Inflow', 'Southbound_Net_Buy'],
    'ma_features': ['MA20', 'MA60', 'MA120', 'MA250', 'Volume_MA250'],
    'slope_features': ['MA250_Slope', 'MA250_Slope_5d', 'MA250_Slope_20d', 'MA250_Slope_Direction'],
    'distance_features': ['Price_Distance_MA250', 'Price_Distance_MA60', 'Price_Distance_MA20'],
    'alignment_features': ['MA_Bullish_Alignment', 'MA_Bearish_Alignment'],
    'cross_features': ['MA20_Golden_Cross_MA60', 'MA20_Death_Cross_MA60',
                       'MA60_Golden_Cross_MA250', 'MA60_Death_Cross_MA250'],
    'volatility_features': ['Volatility_120d', 'Volatility_20d', 'Volatility_60d'],
    'rsi_features': ['RSI', 'RSI_ROC', 'RSI_Deviation'],
    'macd_features': ['MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_ROC'],
    'bb_features': ['BB_Width', 'BB_Position'],
    'atr_features': ['ATR', 'ATR_Ratio'],
    'adx_features': ['ADX', 'Plus_DI', 'Minus_DI'],
    'kdj_features': ['Stoch_K', 'Stoch_D'],
    'williams_features': ['Williams_R'],
    'cmf_features': ['CMF', 'CMF_Signal'],
    'rsi_divergence_features': ['RSI_Bullish_Divergence', 'RSI_Bearish_Divergence'],
    'macd_divergence_features': ['MACD_Bullish_Divergence', 'MACD_Bearish_Divergence'],
    'return_features': ['Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d', 'Return_60d'],
    'momentum_features': ['Momentum_Accel_5d', 'Momentum_Accel_10d'],
    'rs_signal_features': ['60d_RS_Signal_MA250', '60d_RS_Signal_Volume_MA250',
                           '20d_RS_Signal_MA250', '10d_RS_Signal_MA250',
                           '5d_RS_Signal_MA250', '3d_RS_Signal_MA250'],
    'trend_features': ['60d_Trend_MA250', '20d_Trend_MA250', '10d_Trend_MA250',
                       '5d_Trend_MA250', '3d_Trend_MA250', '60d_Trend_Volume_MA250',
                       '20d_Trend_Volume_MA250', '60d_Trend_MA120', '20d_RS_Signal_Volume_MA250']
}


class HSIWalkForwardValidator:
    """恒生指数 Walk-forward 验证器"""

    def __init__(self, horizon=20, train_window_months=12, test_window_months=1,
                 step_window_months=1, confidence_threshold=0.55):
        self.horizon = horizon
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_window_months = step_window_months
        self.confidence_threshold = confidence_threshold

        # 收集所有特征
        self.feature_names = []
        for category, features in FEATURE_CONFIG.items():
            self.feature_names.extend(features)

        print("=" * 80)
        print("🔬 恒生指数 Walk-forward 验证系统")
        print("=" * 80)
        print(f"预测周期: {horizon} 天")
        print(f"训练窗口: {train_window_months} 个月")
        print(f"测试窗口: {test_window_months} 个月")
        print(f"滚动步长: {step_window_months} 个月")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"特征数量: {len(self.feature_names)}")
        print("=" * 80)

    def fetch_data(self, start_date=None, end_date=None):
        """获取数据"""
        print("📊 正在获取数据...")

        # 获取恒指数据
        hsi = yf.Ticker(HSI_SYMBOL)
        hsi_df = hsi.history(period="5y", interval="1d")

        if hsi_df.empty:
            raise ValueError("恒指数据获取失败")

        # 获取美债收益率
        us_yield = yf.Ticker("^TNX")
        us_df = us_yield.history(period="5y", interval="1d")

        # 获取VIX
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(period="5y", interval="1d")

        print(f"  ✅ 数据获取完成（恒指：{len(hsi_df)} 条）")

        return hsi_df, us_df, vix_df

    def calculate_features(self, hsi_df, us_df, vix_df):
        """计算特征"""
        df = hsi_df.copy()

        # ========== 移动平均线 ==========
        for window in [20, 60, 120, 250]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()

        # ========== 均线斜率分析 ==========
        df['MA250_Slope'] = df['MA250'].diff()
        df['MA250_Slope_5d'] = df['MA250'].diff(5) / 5
        df['MA250_Slope_20d'] = df['MA250'].diff(20) / 20
        df['MA250_Slope_Direction'] = np.sign(df['MA250_Slope'])

        # ========== 价格与均线距离 ==========
        df['Price_Distance_MA250'] = (df['Close'] - df['MA250']) / df['MA250'] * 100
        df['Price_Distance_MA60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
        df['Price_Distance_MA20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100

        # ========== 均线排列信号 ==========
        df['MA_Bullish_Alignment'] = ((df['MA20'] > df['MA60']) &
                                      (df['MA60'] > df['MA250'])).astype(int)
        df['MA_Bearish_Alignment'] = ((df['MA20'] < df['MA60']) &
                                      (df['MA60'] < df['MA250'])).astype(int)

        # ========== 交叉信号 ==========
        df['MA20_Golden_Cross_MA60'] = ((df['MA20'] > df['MA60']) &
                                        (df['MA20'].shift(1) <= df['MA60'].shift(1))).astype(int)
        df['MA20_Death_Cross_MA60'] = ((df['MA20'] < df['MA60']) &
                                       (df['MA20'].shift(1) >= df['MA60'].shift(1))).astype(int)
        df['MA60_Golden_Cross_MA250'] = ((df['MA60'] > df['MA250']) &
                                         (df['MA60'].shift(1) <= df['MA250'].shift(1))).astype(int)
        df['MA60_Death_Cross_MA250'] = ((df['MA60'] < df['MA250']) &
                                        (df['MA60'].shift(1) >= df['MA250'].shift(1))).astype(int)

        # ========== 收益率 ==========
        df['Return_1d'] = df['Close'].pct_change().shift(1)
        for period in [3, 5, 10, 20, 60]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period).shift(1)

        # ========== 波动率 ==========
        df['Volatility_120d'] = df['Return_1d'].rolling(window=120).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()

        # ========== RSI ==========
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1)
        df['RSI_ROC'] = df['RSI'].pct_change()
        df['RSI_Deviation'] = abs(df['RSI'] - 50)

        # ========== MACD ==========
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26).shift(1)
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change()

        # ========== 布林带 ==========
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = ((df['Close'].shift(1) - df['BB_Lower']) /
                             (df['BB_Upper'] - df['BB_Lower'] + 1e-10))

        # ========== ATR ==========
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

        # ========== ADX ==========
        up_move = df['High'].diff().shift(1)
        down_move = -df['Low'].diff().shift(1)
        df['Plus_DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['Minus_DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['Plus_DI'] = 100 * (df['Plus_DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        df['Minus_DI'] = 100 * (df['Minus_DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        dx = 100 * (np.abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'] + 1e-10))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # ========== KDJ ==========
        k_period = 14
        d_period = 3
        df['Low_Min_14'] = df['Low'].rolling(window=k_period, min_periods=1).min().shift(1)
        df['High_Max_14'] = df['High'].rolling(window=k_period, min_periods=1).max().shift(1)
        df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min_14']) / (df['High_Max_14'] - df['Low_Min_14'] + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period, min_periods=1).mean().shift(1)

        # ========== Williams %R ==========
        df['Williams_R'] = (df['High_Max_14'] - df['Close']) / (df['High_Max_14'] - df['Low_Min_14'] + 1e-10) * -100

        # ========== CMF ==========
        df['MF_Multiplier'] = ((df['Close'] - df['Low'].shift(1)) - (df['High'].shift(1) - df['Close'])) / \
                              (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)
        df['MF_Volume'] = df['MF_Multiplier'] * df['Volume']
        df['CMF'] = df['MF_Volume'].rolling(20, min_periods=1).sum() / \
                    (df['Volume'].rolling(20, min_periods=1).sum() + 1e-10)
        df['CMF_Signal'] = df['CMF'].rolling(5, min_periods=1).mean().shift(1)

        # ========== RSI 背离 ==========
        lookback = 5
        df['Price_Low_5d'] = df['Close'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['Price_High_5d'] = df['Close'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['RSI_Low_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['RSI_High_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['RSI_Bullish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_Low_5d']) & (df['RSI'] > df['RSI_Low_5d_History'])
        ).astype(int)
        df['RSI_Bearish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_High_5d']) & (df['RSI'] < df['RSI_High_5d_History'])
        ).astype(int)

        # ========== MACD 背离 ==========
        df['MACD_Low_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['MACD_High_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['MACD_Bullish_Divergence'] = (
            (df['Close'] == df['Price_Low_5d']) & (df['MACD'] > df['MACD_Low_5d_History'])
        ).astype(int)
        df['MACD_Bearish_Divergence'] = (
            (df['Close'] == df['Price_High_5d']) & (df['MACD'] < df['MACD_High_5d_History'])
        ).astype(int)

        # ========== 动量加速度 ==========
        df['Momentum_Accel_5d'] = df['Return_5d'] - df['Return_5d'].shift(5)
        df['Momentum_Accel_10d'] = df['Return_10d'] - df['Return_10d'].shift(5)

        # ========== 多周期相对强度信号 ==========
        periods = [3, 5, 10, 20, 60]
        for period in periods:
            df[f'{period}d_RS_Signal_MA250'] = (df['Close'] > df['MA250']).astype(int)
            df[f'{period}d_RS_Signal_Volume_MA250'] = (df['Volume'] > df['Volume_MA250']).astype(int)

        # ========== 多周期趋势 ==========
        for period in periods:
            df[f'{period}d_Trend_MA250'] = (df['MA250'].diff(period) > 0).astype(int)
            df[f'{period}d_Trend_MA120'] = (df['MA120'].diff(period) > 0).astype(int)
            df[f'{period}d_Trend_Volume_MA250'] = (df['Volume_MA250'].diff(period) > 0).astype(int)

        # ========== 宏观因子 ==========
        if not us_df.empty:
            us_aligned = us_df.reindex(df.index, method='ffill')
            df['US_10Y_Yield'] = us_aligned['Close'] / 10
            df['US_10Y_Yield_Change_5d'] = df['US_10Y_Yield'].pct_change(5)

        if not vix_df.empty:
            vix_aligned = vix_df.reindex(df.index, method='ffill')
            df['VIX'] = vix_aligned['Close']
            df['VIX_Change_5d'] = df['VIX'].pct_change(5)

        # ========== 港股通特征 ==========
        df['Southbound_Net_Inflow'] = 0
        df['Southbound_Net_Buy'] = 0

        return df

    def create_target(self, df):
        """创建预测目标"""
        df = df.copy()
        df['Future_Return'] = df['Close'].pct_change(self.horizon).shift(-self.horizon)
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        return df

    def run_validation(self, start_date='2020-01-01', end_date='2025-12-31'):
        """执行 Walk-forward 验证"""

        # 获取数据
        hsi_df, us_df, vix_df = self.fetch_data()

        # 计算特征
        print("🔧 正在计算特征...")
        df = self.calculate_features(hsi_df, us_df, vix_df)
        df = self.create_target(df)

        # 筛选日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        print(f"  ✅ 数据准备完成：{len(df)} 条记录")
        print(f"  日期范围：{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

        # 生成月份列表
        all_months = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        while current <= end:
            all_months.append(current.strftime('%Y-%m'))
            current = current + pd.DateOffset(months=1)

        # 计算 fold 数量
        total_months = len(all_months)
        num_folds = (total_months - self.train_window_months - self.test_window_months) // self.step_window_months + 1

        if num_folds <= 0:
            raise ValueError(f"日期范围不足，需要至少 {self.train_window_months + self.test_window_months} 个月")

        print(f"\n📊 Fold 数量: {num_folds}")
        print("=" * 80)

        # 存储结果
        all_fold_results = []
        all_trades = []

        # 执行每个 fold
        for fold in range(num_folds):
            print(f"\n{'='*80}")
            print(f"📊 Fold {fold + 1}/{num_folds}")
            print(f"{'='*80}")

            # 计算训练和测试期间
            train_start_idx = fold * self.step_window_months
            train_end_idx = train_start_idx + self.train_window_months
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_window_months

            train_months = all_months[train_start_idx:train_end_idx]
            test_months = all_months[test_start_idx:test_end_idx]

            train_start = pd.to_datetime(train_months[0] + '-01').tz_localize('UTC')
            train_end = (pd.to_datetime(train_months[-1] + '-01') +
                        pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')
            test_start = pd.to_datetime(test_months[0] + '-01').tz_localize('UTC')
            test_end = (pd.to_datetime(test_months[-1] + '-01') +
                       pd.DateOffset(months=1) - pd.DateOffset(days=1)).tz_localize('UTC')

            print(f"训练期间: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
            print(f"测试期间: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

            # 筛选数据
            train_df = df[(df.index >= train_start) & (df.index <= train_end)]
            test_df = df[(df.index >= test_start) & (df.index <= test_end)]

            # 准备特征
            available_features = [f for f in self.feature_names if f in train_df.columns]

            train_clean = train_df[available_features + ['Target', 'Future_Return']].dropna()
            test_clean = test_df[available_features + ['Target', 'Future_Return']].dropna()

            if len(train_clean) < 50 or len(test_clean) < 5:
                print(f"  ⚠️ 样本不足，跳过此 fold")
                continue

            X_train = train_clean[available_features]
            y_train = train_clean['Target']
            X_test = test_clean[available_features]
            y_test = test_clean['Target']
            test_returns = test_clean['Future_Return']

            print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

            # 训练模型（原始参数配置）
            print(f"  🔄 训练模型...")
            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=5,
                min_data_in_leaf=5,
                random_seed=42,
                loss_function='Logloss',
                eval_metric='AUC',
                auto_class_weights='Balanced',
                verbose=0,
                task_type='CPU'
            )

            # 时序分割验证集
            val_idx = int(len(X_train) * 0.8)
            model.fit(
                X_train.iloc[:val_idx], y_train.iloc[:val_idx],
                eval_set=(X_train.iloc[val_idx:], y_train.iloc[val_idx:]),
                early_stopping_rounds=30,
                verbose=0
            )

            # 预测
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba > 0.5).astype(int)

            # 计算指标
            accuracy = accuracy_score(y_test, pred)
            try:
                auc = roc_auc_score(y_test, proba)
            except:
                auc = 0.5

            # 筛选高置信度信号
            high_conf_mask = (proba >= self.confidence_threshold) | (proba <= 1 - self.confidence_threshold)
            high_conf_signals = np.where(proba >= self.confidence_threshold, 1,
                                         np.where(proba <= 1 - self.confidence_threshold, 0, -1))

            # 计算交易收益
            trades = []
            for i, (signal, ret, prob) in enumerate(zip(high_conf_signals, test_returns, proba)):
                if signal != -1:  # 有信号
                    trade_return = ret if signal == 1 else -ret
                    trades.append({
                        'date': test_clean.index[i],
                        'signal': '买入' if signal == 1 else '卖出',
                        'probability': prob,
                        'actual_return': ret,
                        'trade_return': trade_return,
                        'correct': (signal == 1 and ret > 0) or (signal == 0 and ret < 0)
                    })

            # 计算策略指标
            if trades:
                trade_returns = [t['trade_return'] for t in trades]
                avg_return = np.mean(trade_returns)
                win_rate = sum(t['correct'] for t in trades) / len(trades)
                sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-10) * np.sqrt(12)

                # 计算最大回撤
                cumulative = np.cumsum(trade_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = cumulative - running_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            else:
                avg_return = 0
                win_rate = 0
                sharpe = 0
                max_drawdown = 0

            # 存储结果
            fold_result = {
                'fold': fold + 1,
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'auc': auc,
                'num_trades': len(trades),
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            }

            all_fold_results.append(fold_result)
            all_trades.extend(trades)

            # 打印结果
            print(f"\n  ✅ Fold {fold + 1} 结果:")
            print(f"     准确率: {accuracy:.2%}")
            print(f"     AUC: {auc:.4f}")
            print(f"     交易次数: {len(trades)}")
            print(f"     平均收益: {avg_return:.2%}")
            print(f"     胜率: {win_rate:.2%}")
            print(f"     夏普比率: {sharpe:.4f}")
            print(f"     最大回撤: {max_drawdown:.2%}")

        # 计算整体指标
        print(f"\n{'='*80}")
        print("📊 整体验证结果")
        print(f"{'='*80}")

        if all_fold_results:
            avg_accuracy = np.mean([r['accuracy'] for r in all_fold_results])
            avg_auc = np.mean([r['auc'] for r in all_fold_results])
            avg_return = np.mean([r['avg_return'] for r in all_fold_results])
            avg_win_rate = np.mean([r['win_rate'] for r in all_fold_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_fold_results])
            avg_max_drawdown = np.mean([r['max_drawdown'] for r in all_fold_results])
            total_trades = sum(r['num_trades'] for r in all_fold_results)

            print(f"\n📈 整体指标:")
            print(f"   平均准确率: {avg_accuracy:.2%}")
            print(f"   平均 AUC: {avg_auc:.4f}")
            print(f"   总交易次数: {total_trades}")
            print(f"   平均收益率: {avg_return:.2%}")
            print(f"   平均胜率: {avg_win_rate:.2%}")
            print(f"   平均夏普比率: {avg_sharpe:.4f}")
            print(f"   平均最大回撤: {avg_max_drawdown:.2%}")

            # 保存结果
            output = {
                'config': {
                    'horizon': self.horizon,
                    'train_window_months': self.train_window_months,
                    'test_window_months': self.test_window_months,
                    'step_window_months': self.step_window_months,
                    'confidence_threshold': self.confidence_threshold,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'fold_results': all_fold_results,
                'overall_metrics': {
                    'avg_accuracy': avg_accuracy,
                    'avg_auc': avg_auc,
                    'total_trades': total_trades,
                    'avg_return': avg_return,
                    'avg_win_rate': avg_win_rate,
                    'avg_sharpe_ratio': avg_sharpe,
                    'avg_max_drawdown': avg_max_drawdown
                }
            }

            # 保存到文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'output/hsi_walk_forward_{timestamp}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到: {output_file}")

            return output
        else:
            print("⚠️ 没有有效的 fold 结果")
            return None


def main():
    parser = argparse.ArgumentParser(description='恒生指数 Walk-forward 验证')
    parser.add_argument('--horizon', type=int, default=20, help='预测周期（天）')
    parser.add_argument('--train-window', type=int, default=12, help='训练窗口（月）')
    parser.add_argument('--test-window', type=int, default=1, help='测试窗口（月）')
    parser.add_argument('--step-window', type=int, default=1, help='滚动步长（月）')
    parser.add_argument('--confidence-threshold', type=float, default=0.55, help='置信度阈值')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='结束日期')

    args = parser.parse_args()

    validator = HSIWalkForwardValidator(
        horizon=args.horizon,
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_window_months=args.step_window,
        confidence_threshold=args.confidence_threshold
    )

    validator.run_validation(
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == '__main__':
    main()
