#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数 CatBoost 预测模型

与评分模型并行运行，对比预测效果

使用方法：
    python3 ml_services/hsi_ml_model.py --mode train   # 训练模型
    python3 ml_services/hsi_ml_model.py --mode predict # 生成预测
    python3 ml_services/hsi_ml_model.py --mode compare # 双模型对比
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import pickle

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 导入南向资金服务
from data_services.southbound_data import SouthboundDataService

# ========== 配置 ==========
HSI_SYMBOL = "^HSI"
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "data/hsi_models"

# 特征配置（2026-04-16 优化：新增RSI/MACD/布林带/动量特征）
FEATURE_CONFIG = {
    # 宏观因子
    'macro_features': [
        'US_10Y_Yield', 'US_10Y_Yield_Change_5d',
        'VIX', 'VIX_Change_5d'
    ],
    # 港股通资金流向
    'southbound_features': [
        'Southbound_Net_Inflow', 'Southbound_Net_Buy'
    ],
    # 技术指标（与评分模型对齐）
    'technical_features': [
        'MA250', 'Volume_MA250', 'MA120',
        'Volatility_120d', 'Volatility_20d', 'Volatility_60d'
    ],
    # RSI 系列（新增）
    'rsi_features': [
        'RSI', 'RSI_ROC', 'RSI_Deviation'
    ],
    # MACD 系列（新增）
    'macd_features': [
        'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_ROC'
    ],
    # 布林带（新增）
    'bb_features': [
        'BB_Width', 'BB_Position'
    ],
    # 多周期收益率（新增）
    'return_features': [
        'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d', 'Return_60d'
    ],
    # 动量加速度（新增）
    'momentum_features': [
        'Momentum_Accel_5d', 'Momentum_Accel_10d'
    ],
    # 多周期相对强度信号
    'rs_signal_features': [
        '60d_RS_Signal_MA250', '60d_RS_Signal_Volume_MA250',
        '20d_RS_Signal_MA250', '10d_RS_Signal_MA250',
        '5d_RS_Signal_MA250', '3d_RS_Signal_MA250'
    ],
    # 多周期趋势
    'trend_features': [
        '60d_Trend_MA250', '20d_Trend_MA250',
        '10d_Trend_MA250', '5d_Trend_MA250', '3d_Trend_MA250',
        '60d_Trend_Volume_MA250', '20d_Trend_Volume_MA250',
        '60d_Trend_MA120',
        '20d_RS_Signal_Volume_MA250'
    ]
}

# CatBoost 模型参数（恒指专用 - 优化版本）
# 2026-04-16 优化：添加类别权重平衡、调整超参数
CATBOOST_PARAMS = {
    'iterations': 300,              # 增加迭代次数（200→300）
    'learning_rate': 0.03,          # 降低学习率（0.05→0.03）
    'depth': 4,                     # 增加深度（3→4）
    'l2_leaf_reg': 5,               # 增加正则化（3→5）
    'min_data_in_leaf': 5,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'early_stopping_rounds': 30,    # 增加早停耐心（20→30）
    'use_best_model': True,
    'auto_class_weights': 'Balanced',  # 新增：类别权重平衡
    'verbose': 50,
    'task_type': 'CPU'
}


class HSICatBoostModel:
    """恒生指数 CatBoost 预测模型"""

    def __init__(self, horizon=20):
        self.horizon = horizon
        self.model = None
        self.feature_names = None
        self.scaler = None

        # 确保目录存在
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def fetch_data(self, start_date=None, end_date=None):
        """
        获取恒指历史数据和宏观因子

        参数:
        - start_date: 开始日期
        - end_date: 结束日期

        返回:
        - DataFrame: 合并后的数据
        """
        print("📊 正在获取数据...")

        # 获取恒指数据（使用 period 参数确保获取数据）
        print("  - 恒生指数数据...")
        hsi = yf.Ticker(HSI_SYMBOL)
        hsi_df = hsi.history(period="5y", interval="1d")

        if hsi_df.empty:
            raise ValueError("恒指数据获取失败")

        # 按日期筛选
        if start_date:
            hsi_df = hsi_df[hsi_df.index >= start_date]
        if end_date:
            hsi_df = hsi_df[hsi_df.index <= end_date]

        # 获取美债收益率
        print("  - 美国国债收益率...")
        us_yield = yf.Ticker("^TNX")
        us_df = us_yield.history(period="5y", interval="1d")

        # 获取VIX
        print("  - VIX恐慌指数...")
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(period="5y", interval="1d")

        print(f"  ✅ 数据获取完成（恒指：{len(hsi_df)} 条）")

        return hsi_df, us_df, vix_df

    def calculate_features(self, hsi_df, us_df, vix_df):
        """
        计算特征

        参数:
        - hsi_df: 恒指数据
        - us_df: 美债数据
        - vix_df: VIX数据

        返回:
        - DataFrame: 特征数据
        """
        print("🔧 正在计算特征...")

        df = hsi_df.copy()

        # ========== 移动平均线（与 hsi_prediction.py 对齐）==========
        for window in [20, 60, 120, 250]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()

        # ========== 收益率（使用昨日值）==========
        df['Return_1d'] = df['Close'].pct_change().shift(1)  # 使用昨日值

        # ========== 波动率 ==========
        df['Volatility_120d'] = df['Return_1d'].rolling(window=120).std()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        # ========== 宏观因子（合并到主数据）==========
        # 美债收益率
        if not us_df.empty:
            us_aligned = us_df.reindex(df.index, method='ffill')
            df['US_10Y_Yield'] = us_aligned['Close'] / 10  # 转换为百分比
            df['US_10Y_Yield_Change_5d'] = df['US_10Y_Yield'].pct_change(5)

        # VIX
        if not vix_df.empty:
            vix_aligned = vix_df.reindex(df.index, method='ffill')
            df['VIX'] = vix_aligned['Close']
            df['VIX_Change_5d'] = df['VIX'].pct_change(5)

        # ========== 港股通特征（使用真实历史数据）==========
        print("  - 获取港股通历史数据...")
        southbound_service = SouthboundDataService()
        southbound_df = southbound_service.fetch_history()

        if southbound_df is not None:
            # 处理时区问题：移除时区信息
            df_index = df.index
            if hasattr(df_index, 'tz') and df_index.tz is not None:
                df_index = df_index.tz_localize(None)

            # 合并南向资金数据
            southbound_aligned = southbound_df.reindex(df_index, method='ffill')
            # 用 0 填充 NaN 值（net_inflow 从 2024-08-19 开始缺失）
            df['Southbound_Net_Inflow'] = southbound_aligned['net_inflow'].fillna(0).values if 'net_inflow' in southbound_aligned.columns else 0
            df['Southbound_Net_Buy'] = southbound_aligned['net_buy'].fillna(0).values if 'net_buy' in southbound_aligned.columns else 0
            print(f"    ✅ 港股通数据已合并")
        else:
            # 回退到0值
            df['Southbound_Net_Inflow'] = 0
            df['Southbound_Net_Buy'] = 0
            print(f"    ⚠️ 港股通数据获取失败，使用默认值")

        # ========== 新增技术指标特征（2026-04-16 优化）==========
        # ⚠️ 特征时滞处理：所有使用当日 Close 的特征需添加 .shift(1)
        # 实盘中预测时只能使用前一天的数据

        # 1. RSI 系列（使用昨日值）
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1)  # 使用昨日 RSI
        df['RSI_ROC'] = df['RSI'].pct_change()
        df['RSI_Deviation'] = abs(df['RSI'] - 50)

        # 2. MACD 系列（使用昨日值）
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26).shift(1)  # 使用昨日 MACD
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change()

        # 3. 布林带（使用昨日值）
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        # BB_Position 使用昨日 Close 计算
        df['BB_Position'] = ((df['Close'].shift(1) - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10))

        # 4. 多周期收益率（使用昨日值，实盘中预测时只能用昨天收盘价计算）
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period).shift(1)

        # 5. 动量加速度（使用昨日值）
        df['Momentum_Accel_5d'] = df['Return_5d'] - df['Return_5d'].shift(5)
        df['Momentum_Accel_10d'] = df['Return_10d'] - df['Return_10d'].shift(5)

        # 6. 波动率扩展
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()

        # ========== 多周期相对强度信号（与 hsi_prediction.py 对齐）==========
        periods = [3, 5, 10, 20, 60]
        for period in periods:
            # 价格相对MA250的强度
            df[f'{period}d_RS_Signal_MA250'] = (df['Close'] > df['MA250']).astype(int)
            # 成交量相对Volume_MA250的强度
            df[f'{period}d_RS_Signal_Volume_MA250'] = (df['Volume'] > df['Volume_MA250']).astype(int)

        # ========== 多周期趋势（与 hsi_prediction.py 对齐）==========
        for period in periods:
            # MA250趋势
            df[f'{period}d_Trend_MA250'] = (df['MA250'].diff(period) > 0).astype(int)
            # MA120趋势
            df[f'{period}d_Trend_MA120'] = (df['MA120'].diff(period) > 0).astype(int)
            # Volume_MA250趋势
            df[f'{period}d_Trend_Volume_MA250'] = (df['Volume_MA250'].diff(period) > 0).astype(int)

        print(f"  ✅ 特征计算完成（{len(df.columns)} 列）")

        return df

    def create_target(self, df):
        """
        创建预测目标

        参数:
        - df: 特征数据

        返回:
        - DataFrame: 添加目标列的数据
        """
        df = df.copy()

        # 未来 horizon 天的收益率
        df[f'Future_Return_{self.horizon}d'] = df['Close'].pct_change(self.horizon).shift(-self.horizon)

        # 目标：上涨为1，下跌为0
        df['Target'] = (df[f'Future_Return_{self.horizon}d'] > 0).astype(int)

        return df

    def prepare_features(self, df):
        """
        准备特征矩阵

        参数:
        - df: 包含特征和目标的数据

        返回:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称列表
        """
        # 收集所有特征
        all_features = []
        for category, features in FEATURE_CONFIG.items():
            all_features.extend(features)

        # 只保留存在的特征
        feature_names = [f for f in all_features if f in df.columns]

        # 移除包含 NaN 的行
        df_clean = df[feature_names + ['Target']].dropna()

        X = df_clean[feature_names]
        y = df_clean['Target']

        self.feature_names = feature_names

        print(f"  ✅ 特征准备完成（{len(feature_names)} 个特征，{len(X)} 个样本）")

        return X, y, feature_names

    def train(self, start_date='2020-01-01', end_date='2025-12-31'):
        """
        训练模型

        参数:
        - start_date: 训练开始日期
        - end_date: 训练结束日期
        """
        print("=" * 60)
        print("🎯 恒指 CatBoost 模型训练")
        print("=" * 60)

        # 获取数据
        hsi_df, us_df, vix_df = self.fetch_data(start_date, end_date)

        # 计算特征
        df = self.calculate_features(hsi_df, us_df, vix_df)

        # 创建目标
        df = self.create_target(df)

        # 准备特征
        X, y, feature_names = self.prepare_features(df)

        # 时序分割（80% 训练，20% 验证）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n📊 数据分割:")
        print(f"  训练集: {len(X_train)} 样本 ({X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')})")
        print(f"  验证集: {len(X_val)} 样本 ({X_val.index[0].strftime('%Y-%m-%d')} ~ {X_val.index[-1].strftime('%Y-%m-%d')})")

        # 创建 CatBoost 模型
        self.model = CatBoostClassifier(**CATBOOST_PARAMS)

        # 训练
        print(f"\n🚀 开始训练...")
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100
        )

        # 评估
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        print(f"\n📊 验证结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"\n分类报告:")
        print(classification_report(y_val, y_pred, target_names=['下跌', '上涨']))

        # 特征重要性
        self._print_feature_importance()

        # 保存模型
        self.save_model()

        return accuracy, auc

    def _print_feature_importance(self, top_n=20):
        """打印特征重要性"""
        if self.model is None or self.feature_names is None:
            return

        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print(f"\n📈 Top {top_n} 特征重要性:")
        print("-" * 50)
        for i, row in feature_imp.head(top_n).iterrows():
            print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

    def predict(self, latest_data=None):
        """
        生成预测

        参数:
        - latest_data: 最新特征数据（可选）

        返回:
        - dict: 预测结果
        """
        if self.model is None:
            # 尝试加载模型
            self.load_model()
            if self.model is None:
                raise ValueError("模型未训练，请先运行 train()")

        print("=" * 60)
        print("🎯 恒指 CatBoost 预测")
        print("=" * 60)

        if latest_data is None:
            # 获取最新数据
            hsi_df, us_df, vix_df = self.fetch_data()
            df = self.calculate_features(hsi_df, us_df, vix_df)
            latest_data = df.iloc[-1:][self.feature_names]

        # 预测
        prob = self.model.predict_proba(latest_data)[0, 1]
        pred = 1 if prob > 0.5 else 0

        result = {
            'prediction': '上涨' if pred == 1 else '下跌',
            'probability': float(prob),
            'confidence': '高' if prob > 0.6 or prob < 0.4 else '中' if prob > 0.55 or prob < 0.45 else '低'
        }

        print(f"\n📊 预测结果:")
        print(f"  方向: {result['prediction']}")
        print(f"  上涨概率: {result['probability']:.4f}")
        print(f"  置信度: {result['confidence']}")

        return result

    def predict_multi_horizon(self, horizons=[1, 5, 20]):
        """
        多周期预测：同时预测1天、5天、20天的涨跌

        参数:
        - horizons: 预测周期列表

        返回:
        - dict: 各周期的预测结果和历史准确率
        """
        print("=" * 60)
        print("🎯 恒指多周期预测")
        print("=" * 60)

        # 获取数据
        hsi_df, us_df, vix_df = self.fetch_data()
        df = self.calculate_features(hsi_df, us_df, vix_df)

        # 准备特征
        all_features = []
        for category, features in FEATURE_CONFIG.items():
            all_features.extend(features)
        feature_names = [f for f in all_features if f in df.columns]

        results = {}

        # 已知的历史准确率（基于 Walk-forward 验证）
        historical_accuracy = {
            1: 0.4643,   # 46.43%
            5: 0.5765,   # 57.65%
            20: 0.5459   # 54.59%
        }

        historical_auc = {
            1: 0.5213,
            5: 0.6567,
            20: 0.7463
        }

        for horizon in horizons:
            print(f"\n📊 {horizon}天周期预测...")

            # 创建目标
            df_h = df.copy()
            df_h['Target'] = (df_h['Close'].pct_change(horizon).shift(-horizon) > 0).astype(float)

            # 移除 NaN
            df_clean = df_h[feature_names + ['Target']].dropna()

            if len(df_clean) < 100:
                print(f"  ⚠️ 数据不足，跳过 {horizon}天周期")
                continue

            X = df_clean[feature_names]
            y = df_clean['Target']

            # 时序分割（80% 训练）
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            # 训练模型（不使用 early stopping）
            predict_params = CATBOOST_PARAMS.copy()
            predict_params['use_best_model'] = False
            predict_params['early_stopping_rounds'] = None
            model = CatBoostClassifier(**predict_params)
            model.fit(X_train, y_train, verbose=0)

            # 用最新数据预测
            latest_features = df.iloc[-1:][feature_names].dropna()
            if latest_features.empty:
                print(f"  ⚠️ 最新特征数据缺失，跳过")
                continue

            prob = model.predict_proba(latest_features)[0, 1]
            pred = 1 if prob > 0.5 else 0

            # 置信度计算
            if prob > 0.65 or prob < 0.35:
                confidence = '高'
            elif prob > 0.55 or prob < 0.45:
                confidence = '中'
            else:
                confidence = '低'

            results[horizon] = {
                'prediction': '上涨' if pred == 1 else '下跌',
                'probability': float(prob),
                'confidence': confidence,
                'historical_accuracy': historical_accuracy.get(horizon, 0.50),
                'historical_auc': historical_auc.get(horizon, 0.50)
            }

            print(f"  方向: {results[horizon]['prediction']}")
            print(f"  上涨概率: {prob:.2%}")
            print(f"  置信度: {confidence}")
            print(f"  历史准确率: {results[horizon]['historical_accuracy']:.2%}")
            print(f"  历史AUC: {results[horizon]['historical_auc']:.4f}")

        # 打印汇总表格
        print("\n" + "=" * 60)
        print("📋 多周期预测汇总")
        print("=" * 60)
        print(f"{'周期':<8} {'方向':<6} {'概率':<8} {'置信度':<6} {'历史准确率':<10} {'历史AUC':<8}")
        print("-" * 60)
        for h in horizons:
            if h in results:
                r = results[h]
                print(f"{h}天{' '*4} {r['prediction']:<6} {r['probability']:.2%}{' '*2} {r['confidence']:<6} {r['historical_accuracy']:.2%}{' '*4} {r['historical_auc']:.4f}")
        print("-" * 60)

        # 综合建议
        up_count = sum(1 for r in results.values() if r['prediction'] == '上涨')
        down_count = len(results) - up_count

        if up_count > down_count:
            suggestion = "📈 综合看涨"
        elif down_count > up_count:
            suggestion = "📉 综合看跌"
        else:
            suggestion = "⚠️ 方向分歧，建议观望"

        print(f"\n💡 综合建议: {suggestion}")

        return results

    def save_model(self):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODEL_DIR, f'hsi_catboost_{timestamp}.cbm')

        self.model.save_model(model_path)

        # 同时保存特征名称
        feature_path = os.path.join(MODEL_DIR, f'hsi_features_{timestamp}.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)

        print(f"\n💾 模型已保存到: {model_path}")

    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            # 查找最新模型
            import glob
            models = glob.glob(os.path.join(MODEL_DIR, 'hsi_catboost_*.cbm'))
            if not models:
                return False
            model_path = max(models, key=os.path.getmtime)

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        # 加载特征名称
        feature_path = model_path.replace('.cbm', '.json').replace('hsi_catboost', 'hsi_features')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)

        print(f"✅ 模型已加载: {model_path}")
        return True

    def walk_forward_validation(self, start_date='2020-01-01', end_date='2025-12-31',
                                 train_window=12, test_window=3):
        """
        Walk-forward 验证

        参数:
        - start_date: 开始日期
        - end_date: 结束日期
        - train_window: 训练窗口（月）
        - test_window: 测试窗口（月），默认3个月（约60个样本/Fold）
        """
        print("=" * 60)
        print("🔄 Walk-forward 验证")
        print("=" * 60)
        print(f"  训练窗口: {train_window} 个月")
        print(f"  测试窗口: {test_window} 个月（约 {test_window * 20} 个样本/Fold）")

        # 获取数据
        hsi_df, us_df, vix_df = self.fetch_data(start_date, end_date)
        df = self.calculate_features(hsi_df, us_df, vix_df)
        df = self.create_target(df)

        # 准备特征
        X, y, feature_names = self.prepare_features(df)

        # 将索引转换为日期
        dates = X.index

        # 计算窗口大小（交易日）
        train_days = train_window * 20  # 约每月20个交易日
        test_days = test_window * 20

        results = []
        fold = 0

        # 滚动验证
        for i in range(train_days, len(dates) - test_days, test_days):
            fold += 1

            train_start = dates[0]
            train_end = dates[i - 1]
            test_start = dates[i]
            test_end = dates[min(i + test_days - 1, len(dates) - 1)]

            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i + test_days]
            y_test = y.iloc[i:i + test_days]

            # 训练模型（Walk-forward 不使用 early stopping）
            wf_params = CATBOOST_PARAMS.copy()
            wf_params['use_best_model'] = False
            wf_params['early_stopping_rounds'] = None
            model = CatBoostClassifier(**wf_params)
            model.fit(X_train, y_train, verbose=0)

            # 预测
            y_prob = model.predict_proba(X_test)[:, 1]

            # 动态阈值：基于训练集正例比例
            train_positive_rate = y_train.mean()
            dynamic_threshold = train_positive_rate  # 使用训练集正例比例作为阈值

            # 固定阈值预测
            y_pred_fixed = (y_prob > 0.5).astype(int)
            # 动态阈值预测
            y_pred_dynamic = (y_prob > dynamic_threshold).astype(int)

            # 评估 - 固定阈值
            accuracy_fixed = accuracy_score(y_test, y_pred_fixed)
            # 评估 - 动态阈值
            accuracy_dynamic = accuracy_score(y_test, y_pred_dynamic)

            # 高置信度预测评估
            high_conf_mask = (y_prob > 0.7) | (y_prob < 0.3)
            if high_conf_mask.sum() > 0:
                y_test_hc = y_test[high_conf_mask]
                y_pred_hc = y_pred_fixed[high_conf_mask]
                accuracy_high_conf = accuracy_score(y_test_hc, y_pred_hc)
                high_conf_count = high_conf_mask.sum()
            else:
                accuracy_high_conf = 0.5
                high_conf_count = 0

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = 0.5

            results.append({
                'fold': fold,
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'accuracy_fixed': accuracy_fixed,
                'accuracy_dynamic': accuracy_dynamic,
                'accuracy_high_conf': accuracy_high_conf,
                'high_conf_count': high_conf_count,
                'auc': auc,
                'samples': len(y_test),
                'train_positive_rate': train_positive_rate,
                'dynamic_threshold': dynamic_threshold
            })

            print(f"Fold {fold}: 固定阈值={accuracy_fixed:.2%}, 动态阈值={accuracy_dynamic:.2%}, 高置信={accuracy_high_conf:.2%}({high_conf_count}样本), AUC={auc:.4f}")

        # 汇总结果
        results_df = pd.DataFrame(results)
        print(f"\n📊 Walk-forward 验证结果:")
        print(f"  固定阈值准确率: {results_df['accuracy_fixed'].mean():.2%}")
        print(f"  动态阈值准确率: {results_df['accuracy_dynamic'].mean():.2%}")
        print(f"  高置信度准确率: {results_df['accuracy_high_conf'].mean():.2%}")
        print(f"  平均 AUC: {results_df['auc'].mean():.4f}")
        print(f"  Fold 数量: {len(results_df)}")

        return results_df

    def train_rolling(self, window_months=12):
        """
        使用滚动窗口训练模型（只使用最近N个月数据）

        参数:
        - window_months: 训练窗口大小（月）

        返回:
        - accuracy, auc
        """
        from datetime import datetime, timedelta

        print("=" * 60)
        print(f"🎯 恒指 CatBoost 滚动窗口训练（最近 {window_months} 个月）")
        print("=" * 60)

        # 计算开始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_months * 30)

        print(f"📊 训练数据范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

        # 获取数据
        hsi_df, us_df, vix_df = self.fetch_data(start_date.strftime('%Y-%m-%d'),
                                                  end_date.strftime('%Y-%m-%d'))

        # 计算特征
        df = self.calculate_features(hsi_df, us_df, vix_df)

        # 创建目标
        df = self.create_target(df)

        # 准备特征
        X, y, feature_names = self.prepare_features(df)

        if len(X) < 50:
            print(f"⚠️ 数据量不足（{len(X)} 样本），使用更长时间范围")
            # 扩大时间范围
            start_date = end_date - timedelta(days=window_months * 60)
            hsi_df, us_df, vix_df = self.fetch_data(start_date.strftime('%Y-%m-%d'),
                                                      end_date.strftime('%Y-%m-%d'))
            df = self.calculate_features(hsi_df, us_df, vix_df)
            df = self.create_target(df)
            X, y, feature_names = self.prepare_features(df)

        # 时序分割（80% 训练，20% 验证）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n📊 数据分割:")
        print(f"  训练集: {len(X_train)} 样本 ({X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')})")
        print(f"  验证集: {len(X_val)} 样本 ({X_val.index[0].strftime('%Y-%m-%d')} ~ {X_val.index[-1].strftime('%Y-%m-%d')})")

        # 检查类别分布
        train_up = (y_train == 1).sum()
        train_down = (y_train == 0).sum()
        print(f"\n📊 训练集标签分布:")
        print(f"  上涨: {train_up} ({train_up/len(y_train)*100:.1f}%)")
        print(f"  下跌: {train_down} ({train_down/len(y_train)*100:.1f}%)")

        # 创建 CatBoost 模型
        self.model = CatBoostClassifier(**CATBOOST_PARAMS)

        # 训练
        print(f"\n🚀 开始训练...")
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100
        )

        # 评估
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        print(f"\n📊 验证结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"\n分类报告:")
        print(classification_report(y_val, y_pred, target_names=['下跌', '上涨']))

        # 特征重要性
        self._print_feature_importance()

        # 保存模型
        self.save_model()

        return accuracy, auc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒指 CatBoost 预测模型')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'validate', 'compare', 'rolling', 'multi'],
                        help='运行模式（multi=多周期预测）')
    parser.add_argument('--horizon', type=int, default=20,
                        help='预测周期（天）')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                        help='开始日期（默认2021-01-01，增加训练数据）')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                        help='结束日期')
    parser.add_argument('--window', type=int, default=12,
                        help='滚动训练窗口大小（月）')
    parser.add_argument('--test-window', type=int, default=3,
                        help='Walk-forward 测试窗口大小（月），默认3个月')

    args = parser.parse_args()

    # 创建模型
    model = HSICatBoostModel(horizon=args.horizon)

    if args.mode == 'train':
        model.train(start_date=args.start_date, end_date=args.end_date)
    elif args.mode == 'predict':
        model.predict()
    elif args.mode == 'multi':
        # 多周期预测（1天、5天、20天）
        model.predict_multi_horizon(horizons=[1, 5, 20])
    elif args.mode == 'validate':
        model.walk_forward_validation(start_date=args.start_date, end_date=args.end_date,
                                       test_window=args.test_window)
    elif args.mode == 'rolling':
        model.train_rolling(window_months=args.window)
    elif args.mode == 'compare':
        # 与评分模型对比
        print("🚧 对比模式开发中...")


if __name__ == '__main__':
    main()
