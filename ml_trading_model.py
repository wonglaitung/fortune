#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习交易模型 - 二分类模型预测次日涨跌
整合技术指标、基本面、资金流向等特征，使用LightGBM进行训练
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 导入项目模块
from tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from technical_analysis import TechnicalAnalyzer
from fundamental_data import get_comprehensive_fundamental_data
from base.base_model_processor import BaseModelProcessor
from us_market_data import us_market_data

# 自选股列表
WATCHLIST = [
    "0005.HK", "0388.HK", "0700.HK", "0728.HK", "0883.HK",
    "0939.HK", "0941.HK", "0981.HK", "1088.HK", "1138.HK",
    "1211.HK", "1288.HK", "1299.HK", "1330.HK", "1347.HK",
    "1398.HK", "1810.HK", "2269.HK", "2533.HK", "3690.HK",
    "3968.HK", "6682.HK", "9660.HK", "9988.HK"
]

# 股票名称映射
STOCK_NAMES = {
    "0005.HK": "汇丰银行",
    "0388.HK": "香港交易所",
    "0700.HK": "腾讯控股",
    "0728.HK": "中国电信",
    "0883.HK": "中国海洋石油",
    "0939.HK": "建设银行",
    "0941.HK": "中国移动",
    "0981.HK": "中芯国际",
    "1088.HK": "中国神华",
    "1138.HK": "中远海能",
    "1288.HK": "农业银行",
    "1330.HK": "绿色动力环保",
    "1347.HK": "华虹半导体",
    "1398.HK": "工商银行",
    "1810.HK": "小米集团-W",
    "2269.HK": "药明生物",
    "2533.HK": "黑芝麻智能",
    "2800.HK": "盈富基金",
    "3690.HK": "美团-W",
    "3968.HK": "招商银行",
    "6682.HK": "第四范式",
    "9660.HK": "地平线机器人",
    "9988.HK": "阿里巴巴-SW",
    "1211.HK": "比亚迪股份",
    "1299.HK": "友邦保险"
}


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()

    def calculate_technical_features(self, df):
        """计算技术指标特征"""
        if df.empty or len(df) < 200:
            return df

        # 移动平均线
        df = self.tech_analyzer.calculate_moving_averages(df, periods=[5, 10, 20, 50, 100, 200])

        # RSI
        df = self.tech_analyzer.calculate_rsi(df, period=14)

        # MACD
        df = self.tech_analyzer.calculate_macd(df)

        # 布林带
        df = self.tech_analyzer.calculate_bollinger_bands(df, period=20, std_dev=2)

        # ATR
        df = self.tech_analyzer.calculate_atr(df, period=14)

        # 成交量比率
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

        # 价格位置（相对于均线）
        df['Price_Ratio_MA5'] = df['Close'] / df['MA5']
        df['Price_Ratio_MA20'] = df['Close'] / df['MA20']
        df['Price_Ratio_MA50'] = df['Close'] / df['MA50']

        # 布林带位置
        df['BB_Position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 涨跌幅
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)

        return df

    def create_fundamental_features(self, code):
        """创建基本面特征"""
        try:
            # 移除代码中的.HK后缀
            stock_code = code.replace('.HK', '')
            
            fundamental_data = get_comprehensive_fundamental_data(stock_code)
            if fundamental_data and 'financial_indicator' in fundamental_data:
                fi = fundamental_data['financial_indicator']
                return {
                    'PE': fi.get('市盈率', np.nan),
                    'PB': fi.get('市净率', np.nan),
                    'ROE': fi.get('净资产收益率', np.nan) / 100 if fi.get('净资产收益率') else np.nan,
                    'ROA': fi.get('总资产收益率', np.nan) / 100 if fi.get('总资产收益率') else np.nan,
                    'Dividend_Yield': fi.get('股息率', np.nan) / 100 if fi.get('股息率') else np.nan,
                    'EPS': fi.get('每股收益', np.nan),
                    'Net_Margin': fi.get('净利率', np.nan) / 100 if fi.get('净利率') else np.nan,
                    'Gross_Margin': fi.get('毛利率', np.nan) / 100 if fi.get('毛利率') else np.nan
                }
        except Exception as e:
            print(f"获取基本面数据失败 {code}: {e}")
        return {}

    def create_smart_money_features(self, df):
        """创建资金流向特征"""
        if df.empty or len(df) < 50:
            return df

        # 价格相对位置
        df['Price_Pct_20d'] = df['Close'].rolling(window=20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))

        # 放量上涨信号
        df['Strong_Volume_Up'] = (df['Close'] > df['Open']) & (df['Vol_Ratio'] > 1.5)

        # 缩量回调信号
        df['Prev_Close'] = df['Close'].shift(1)
        df['Weak_Volume_Down'] = (df['Close'] < df['Prev_Close']) & (df['Vol_Ratio'] < 1.0) & ((df['Prev_Close'] - df['Close']) / df['Prev_Close'] < 0.02)

        # 动量信号
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1

        return df

    def create_market_environment_features(self, stock_df, hsi_df, us_market_df=None):
        """创建市场环境特征（包含港股和美股）
        
        Args:
            stock_df: 股票数据
            hsi_df: 恒生指数数据
            us_market_df: 美股市场数据（可选）
        """
        if stock_df.empty or hsi_df.empty:
            return stock_df

        # 计算恒生指数收益率
        hsi_df['HSI_Return'] = hsi_df['Close'].pct_change()
        hsi_df['HSI_Return_5d'] = hsi_df['Close'].pct_change(5)

        # 合并恒生指数数据
        stock_df = stock_df.merge(hsi_df[['HSI_Return', 'HSI_Return_5d']], left_index=True, right_index=True, how='left')

        # 相对表现（相对于恒生指数）
        stock_df['Relative_Return'] = stock_df['Return_5d'] - stock_df['HSI_Return_5d']

        # 如果提供了美股数据，合并美股特征
        if us_market_df is not None and not us_market_df.empty:
            # 美股特征列
            us_features = [
                'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
                'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
                'VIX_Change', 'VIX_Ratio_MA20',
                'US_10Y_Yield', 'US_10Y_Yield_Change'
            ]

            # 只合并存在的特征
            existing_us_features = [f for f in us_features if f in us_market_df.columns]
            if existing_us_features:
                stock_df = stock_df.merge(
                    us_market_df[existing_us_features],
                    left_index=True, right_index=True, how='left'
                )

        return stock_df

    def create_label(self, df, horizon=1):
        """创建标签：次日涨跌"""
        if df.empty or len(df) < horizon + 1:
            return df

        # 计算未来收益率
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1

        # 二分类标签：1=上涨，0=下跌
        df['Label'] = (df['Future_Return'] > 0).astype(int)

        return df


class MLTradingModel:
    """机器学习交易模型"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.processor = BaseModelProcessor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.horizon = 1  # 默认预测周期

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1):
        """准备训练数据
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
        """
        self.horizon = horizon
        all_data = []

        # 获取美股市场数据（只获取一次）
        print("📊 获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=730)
        if us_market_df is not None:
            print(f"✅ 成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            print("⚠️ 无法获取美股市场数据，将只使用港股特征")

        for code in codes:
            try:
                print(f"处理股票: {code}")

                # 移除代码中的.HK后缀，腾讯财经接口不需要
                stock_code = code.replace('.HK', '')

                # 获取股票数据（2年约730天）
                stock_df = get_hk_stock_data_tencent(stock_code, period_days=730)
                if stock_df is None or stock_df.empty:
                    continue

                # 获取恒生指数数据（2年约730天）
                hsi_df = get_hsi_data_tencent(period_days=730)
                if hsi_df is None or hsi_df.empty:
                    continue

                # 计算技术指标
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon)

                # 添加基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 添加股票代码
                stock_df['Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                print(f"处理股票 {code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError("没有获取到任何数据")

        # 合并所有数据
        df = pd.concat(all_data, ignore_index=True)

        # 过滤日期范围
        # 确保索引是 datetime 类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 转换过滤日期为 datetime 类型
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1):
        """训练模型
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
        """
        print("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon)

        # 删除包含NaN的行
        df = df.dropna()

        if len(df) < 100:
            raise ValueError(f"数据量不足，只有 {len(df)} 条记录")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        print(f"使用 {len(self.feature_columns)} 个特征")

        # 准备特征和标签
        X = df[self.feature_columns].values
        y = df['Label'].values

        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)

        # 训练模型
        print("训练LightGBM模型...")
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        # 使用时间序列交叉验证
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
            print(f"验证准确率: {score:.4f}")

        # 使用全部数据重新训练
        self.model.fit(X, y)

        print(f"\n平均验证准确率: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

        # 特征重要性（使用 BaseModelProcessor 统一格式）
        feat_imp = self.processor.analyze_feature_importance(
            self.model.booster_,
            self.feature_columns
        )

        # 计算特征影响方向（如果可能）
        try:
            contrib_values = self.model.booster_.predict(X, pred_contrib=True)
            mean_contrib_values = np.mean(contrib_values[:, :-1], axis=0)
            feat_imp['Mean_Contrib_Value'] = mean_contrib_values
            feat_imp['Impact_Direction'] = feat_imp['Mean_Contrib_Value'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative'
            )
        except Exception as e:
            print(f"⚠️ 特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        print("\n特征重要性 Top 10:")
        print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(10))

        return feat_imp

    def predict(self, code, predict_date=None, horizon=None):
        """预测单只股票

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)，基于该日期的数据预测下一个交易日，默认使用最新交易日
            horizon: 预测周期（1=次日，5=一周，20=一个月），默认使用训练时的周期
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 移除代码中的.HK后缀
            stock_code = code.replace('.HK', '')

            # 获取股票数据（2年约730天）
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=730)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据（2年约730天）
            hsi_df = get_hsi_data_tencent(period_days=730)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=730)

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                # 转换为字符串格式进行比较
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                # 确保索引是 datetime 类型
                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)
                if not isinstance(hsi_df.index, pd.DatetimeIndex):
                    hsi_df.index = pd.to_datetime(hsi_df.index)
                if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                    us_market_df.index = pd.to_datetime(us_market_df.index)

                # 使用字符串比较避免时区问题
                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                if us_market_df is not None:
                    us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    print(f"⚠️ 股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算特征
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

            # 添加基本面特征
            fundamental_features = self.feature_engineer.create_fundamental_features(code)
            for key, value in fundamental_features.items():
                stock_df[key] = value

            # 获取最新数据（或指定日期的数据）
            latest_data = stock_df.iloc[-1:]

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            X = latest_data[self.feature_columns].values

            # 预测
            proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]

            return {
                'code': code,
                'name': STOCK_NAMES.get(code, code),
                'prediction': int(prediction),
                'probability': float(proba[1]),
                'current_price': float(latest_data['Close'].values[0]),
                'date': latest_data.index[0]
            }

        except Exception as e:
            print(f"预测失败 {code}: {e}")
            return None

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"模型已从 {filepath} 加载")


class GBDTLRModel:
    """GBDT + LR 两阶段模型 - 提高准确度和可解释性"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.processor = BaseModelProcessor()
        self.gbdt_model = None
        self.lr_model = None
        self.feature_columns = []
        self.actual_n_estimators = 0
        self.gbdt_leaf_names = []
        self.horizon = 1  # 默认预测周期

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1):
        """准备训练数据
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
        """
        self.horizon = horizon
        all_data = []

        # 获取美股市场数据（只获取一次）
        print("📊 获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=730)
        if us_market_df is not None:
            print(f"✅ 成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            print("⚠️ 无法获取美股市场数据，将只使用港股特征")

        for code in codes:
            try:
                print(f"处理股票: {code}")

                # 移除代码中的.HK后缀，腾讯财经接口不需要
                stock_code = code.replace('.HK', '')

                # 获取股票数据（2年约730天）
                stock_df = get_hk_stock_data_tencent(stock_code, period_days=730)
                if stock_df is None or stock_df.empty:
                    continue

                # 获取恒生指数数据（2年约730天）
                hsi_df = get_hsi_data_tencent(period_days=730)
                if hsi_df is None or hsi_df.empty:
                    continue

                # 计算技术指标
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon)

                # 添加基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 添加股票代码
                stock_df['Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                print(f"处理股票 {code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError("没有获取到任何数据")

        # 合并所有数据
        df = pd.concat(all_data, ignore_index=True)

        # 过滤日期范围
        # 确保索引是 datetime 类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 转换过滤日期为 datetime 类型
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1):
        """训练 GBDT + LR 模型
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
        """
        print("="*70)
        print("🚀 开始训练 GBDT + LR 模型")
        print("="*70)

        # 准备数据
        print("📊 准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon)

        # 删除包含NaN的行
        df = df.dropna()

        if len(df) < 100:
            raise ValueError(f"数据量不足，只有 {len(df)} 条记录")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        print(f"✅ 使用 {len(self.feature_columns)} 个特征")

        # 准备特征和标签
        X = df[self.feature_columns].values
        y = df['Label'].values

        # 创建输出目录
        os.makedirs('output', exist_ok=True)

        # ========== Step 1: 训练 GBDT ==========
        print("\n" + "="*70)
        print("🌲 Step 1: 训练 GBDT 模型（特征工程）")
        print("="*70)

        n_estimators = 32
        num_leaves = 64

        self.gbdt_model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            subsample=0.8,
            min_child_weight=0.1,
            min_child_samples=10,
            colsample_bytree=0.7,
            num_leaves=num_leaves,
            learning_rate=0.05,
            n_estimators=n_estimators,
            random_state=2020,
            n_jobs=-1,
            verbose=-1
        )

        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        gbdt_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            self.gbdt_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric='binary_logloss',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=5, verbose=False)
                ]
            )

            y_pred_fold = self.gbdt_model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred_fold)
            gbdt_scores.append(score)
            print(f"   Fold {fold} 验证准确率: {score:.4f}")

        # 使用全部数据重新训练
        self.gbdt_model.fit(X, y)

        # 获取实际训练的树数量
        # 注意：在使用全部数据重新训练时，如果没有使用早停，best_iteration_ 可能为 None
        # 这种情况下使用 n_estimators
        self.actual_n_estimators = self.gbdt_model.best_iteration_ if self.gbdt_model.best_iteration_ else n_estimators
        print(f"\n✅ GBDT 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators} (原计划: {n_estimators})")
        print(f"   平均验证准确率: {np.mean(gbdt_scores):.4f} (+/- {np.std(gbdt_scores):.4f})")

        # ========== Step 2: 输出 GBDT 特征重要性 ==========
        print("\n" + "="*70)
        print("📊 Step 2: 分析 GBDT 特征重要性")
        print("="*70)

        feat_imp = self.processor.analyze_feature_importance(
            self.gbdt_model.booster_,
            self.feature_columns
        )

        # 计算特征影响方向
        try:
            contrib_values = self.gbdt_model.booster_.predict(X, pred_contrib=True)
            mean_contrib_values = np.mean(contrib_values[:, :-1], axis=0)
            feat_imp['Mean_Contrib_Value'] = mean_contrib_values
            feat_imp['Impact_Direction'] = feat_imp['Mean_Contrib_Value'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative'
            )

            # 保存特征重要性
            feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False)
            print("✅ 已保存特征重要性至 output/gbdt_feature_importance.csv")

            # 显示前20个重要特征
            print("\n📊 GBDT Top 20 重要特征 (含影响方向):")
            print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(20))

        except Exception as e:
            print(f"⚠️ 特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        # ========== Step 3: 获取叶子节点索引 ==========
        print("\n" + "="*70)
        print("🍃 Step 3: 生成叶子节点特征")
        print("="*70)

        gbdt_leaf_features = self.gbdt_model.booster_.predict(X, pred_leaf=True)

        # 获取实际的树数量（基于叶子节点特征的实际形状）
        actual_trees = gbdt_leaf_features.shape[1]
        print(f"   实际叶子节点特征数量: {actual_trees}")

        # 生成叶子节点特征名称
        self.gbdt_leaf_names = [f'gbdt_leaf_{i}' for i in range(actual_trees)]
        df_gbdt_leaf = pd.DataFrame(gbdt_leaf_features, columns=self.gbdt_leaf_names)

        # ========== Step 4: 对叶子节点做 One-Hot 编码 ==========
        print("   对叶子节点进行 One-Hot 编码...")
        df_gbdt_onehot = pd.DataFrame()

        for col in self.gbdt_leaf_names:
            onehot_feats = pd.get_dummies(df_gbdt_leaf[col], prefix=col)
            df_gbdt_onehot = pd.concat([df_gbdt_onehot, onehot_feats], axis=1)

        print(f"   生成了 {df_gbdt_onehot.shape[1]} 个叶子节点特征")

        # ========== Step 5: 训练 LR 模型 ==========
        print("\n" + "="*70)
        print("📈 Step 5: 训练 LR 模型（最终分类器）")
        print("="*70)

        # 划分训练集和验证集
        X_train_lr, X_val_lr, y_train_lr, y_val_lr = train_test_split(
            df_gbdt_onehot, y, test_size=0.2, random_state=2020, stratify=y
        )

        self.lr_model = LogisticRegression(
            penalty='l2',
            C=0.1,
            solver='liblinear',
            random_state=2020,
            max_iter=1000
        )
        self.lr_model.fit(X_train_lr, y_train_lr)

        # 评估
        tr_pred_prob = self.lr_model.predict_proba(X_train_lr)[:, 1]
        val_pred_prob = self.lr_model.predict_proba(X_val_lr)[:, 1]

        tr_logloss = log_loss(y_train_lr, tr_pred_prob)
        val_logloss = log_loss(y_val_lr, val_pred_prob)

        tr_ks = self.processor.calculate_ks_statistic(y_train_lr, tr_pred_prob)
        val_ks = self.processor.calculate_ks_statistic(y_val_lr, val_pred_prob)

        tr_auc = roc_auc_score(y_train_lr, tr_pred_prob)
        val_auc = roc_auc_score(y_val_lr, val_pred_prob)

        print(f"\n✅ LR 训练完成")
        print(f"   Train LogLoss: {tr_logloss:.4f}")
        print(f"   Val LogLoss: {val_logloss:.4f}")
        print(f"   Train KS: {tr_ks:.4f}")
        print(f"   Val KS: {val_ks:.4f}")
        print(f"   Train AUC: {tr_auc:.4f}")
        print(f"   Val AUC: {val_auc:.4f}")

        # 绘制 ROC 曲线
        self.processor.plot_roc_curve(y_val_lr, val_pred_prob, "output/roc_curve.png")

        # ========== Step 6: 输出 LR 系数 ==========
        print("\n" + "="*70)
        print("🔍 Step 6: 分析 LR 系数")
        print("="*70)

        lr_coef = pd.DataFrame({
            'Leaf_Feature': X_train_lr.columns,
            'Coefficient': self.lr_model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)

        lr_coef.to_csv('output/lr_leaf_coefficients.csv', index=False)
        print("✅ 已保存 LR 系数至 output/lr_leaf_coefficients.csv")

        print("\n📊 LR Top 10 重要叶子特征（按系数绝对值排序）:")
        print(lr_coef.head(10))

        # ========== Step 7: 解析高权重叶子规则 ==========
        print("\n" + "="*70)
        print("🧠 Step 7: 解析高权重叶子节点规则")
        print("="*70)

        top_leaves = lr_coef.head(5)

        for idx, row in top_leaves.iterrows():
            leaf_feat = row['Leaf_Feature']
            coef = row['Coefficient']

            if leaf_feat.startswith('gbdt_leaf_'):
                parts = leaf_feat.split('_')
                if len(parts) >= 4:
                    tree_idx = int(parts[2])
                    leaf_idx = int(parts[3])

                    print(f"\n🔎 解析 {leaf_feat} (LR系数: {coef:.4f})")
                    try:
                        rule = self.processor.get_leaf_path_enhanced(
                            self.gbdt_model.booster_,
                            tree_index=tree_idx,
                            leaf_index=leaf_idx,
                            feature_names=self.feature_columns
                        )
                        if rule:
                            for i, r in enumerate(rule, 1):
                                print(f"   {i}. {r}")
                        else:
                            print("   ⚠️ 路径未找到")
                    except Exception as e:
                        print(f"   ⚠️ 解析失败: {e}")

        print("\n" + "="*70)
        print("✅ GBDT + LR 模型训练完成！")
        print("="*70)
        print("📊 所有可解释性报告已生成在 output/ 目录下：")
        print("   - gbdt_feature_importance.csv")
        print("   - lr_leaf_coefficients.csv")
        print("   - roc_curve.png")

        return feat_imp

    def predict(self, code, predict_date=None, horizon=None):
        """预测单只股票

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)，基于该日期的数据预测下一个交易日，默认使用最新交易日
            horizon: 预测周期（1=次日，5=一周，20=一个月），默认使用训练时的周期
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 移除代码中的.HK后缀
            stock_code = code.replace('.HK', '')

            # 获取股票数据
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=730)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据
            hsi_df = get_hsi_data_tencent(period_days=730)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=730)

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                # 转换为字符串格式进行比较
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                # 确保索引是 datetime 类型
                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)
                if not isinstance(hsi_df.index, pd.DatetimeIndex):
                    hsi_df.index = pd.to_datetime(hsi_df.index)
                if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                    us_market_df.index = pd.to_datetime(us_market_df.index)

                # 使用字符串比较避免时区问题
                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                if us_market_df is not None:
                    us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    print(f"⚠️ 股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算特征
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

            # 添加基本面特征
            fundamental_features = self.feature_engineer.create_fundamental_features(code)
            for key, value in fundamental_features.items():
                stock_df[key] = value

            # 获取最新数据
            latest_data = stock_df.iloc[-1:]

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            X = latest_data[self.feature_columns].values

            # Step 1: 使用 GBDT 获取叶子节点
            gbdt_leaf = self.gbdt_model.booster_.predict(X, pred_leaf=True)[0]
            df_gbdt_leaf = pd.DataFrame([gbdt_leaf], columns=self.gbdt_leaf_names)

            # Step 2: One-Hot 编码
            df_gbdt_onehot = pd.DataFrame()
            for col in self.gbdt_leaf_names:
                onehot_feats = pd.get_dummies(df_gbdt_leaf[col], prefix=col)
                df_gbdt_onehot = pd.concat([df_gbdt_onehot, onehot_feats], axis=1)

            # 确保特征列与训练时一致
            for col in self.lr_model.feature_names_in_:
                if col not in df_gbdt_onehot.columns:
                    df_gbdt_onehot[col] = 0

            df_gbdt_onehot = df_gbdt_onehot[self.lr_model.feature_names_in_]

            # Step 3: 使用 LR 预测
            proba = self.lr_model.predict_proba(df_gbdt_onehot)[0]
            prediction = self.lr_model.predict(df_gbdt_onehot)[0]

            return {
                'code': code,
                'name': STOCK_NAMES.get(code, code),
                'prediction': int(prediction),
                'probability': float(proba[1]),
                'current_price': float(latest_data['Close'].values[0]),
                'date': latest_data.index[0]
            }

        except Exception as e:
            print(f"预测失败 {code}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'gbdt_model': self.gbdt_model,
            'lr_model': self.lr_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'gbdt_leaf_names': self.gbdt_leaf_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"GBDT + LR 模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.gbdt_model = model_data['gbdt_model']
        self.lr_model = model_data['lr_model']
        self.feature_columns = model_data['feature_columns']
        self.actual_n_estimators = model_data['actual_n_estimators']
        self.gbdt_leaf_names = model_data['gbdt_leaf_names']
        print(f"GBDT + LR 模型已从 {filepath} 加载")


def main():
    parser = argparse.ArgumentParser(description='机器学习交易模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                       help='运行模式: train=训练, predict=预测, evaluate=评估')
    parser.add_argument('--model-type', type=str, default='both', choices=['lgbm', 'gbdt_lr', 'both'],
                       help='模型类型: lgbm=单一LightGBM模型, gbdt_lr=GBDT+LR两阶段模型, both=同时训练两种模型（默认）')
    parser.add_argument('--model-path', type=str, default='data/ml_trading_model.pkl',
                       help='模型保存/加载路径')
    parser.add_argument('--start-date', type=str, default=None,
                       help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--predict-date', type=str, default=None,
                       help='预测日期：基于该日期的数据预测下一个交易日 (YYYY-MM-DD)，默认使用最新交易日')
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 5, 20],
                       help='预测周期: 1=次日（默认）, 5=一周, 20=一个月')

    args = parser.parse_args()

    # 判断是否同时训练两种模型
    train_both = args.model_type == 'both'

    if train_both:
        print("=" * 70)
        print("🚀 同时训练两种模型进行对比")
        print("=" * 70)
        lgbm_model = MLTradingModel()
        gbdt_lr_model = GBDTLRModel()
    elif args.model_type == 'gbdt_lr':
        print("=" * 70)
        print("🚀 使用 GBDT + LR 两阶段模型")
        print("=" * 70)
        lgbm_model = None
        gbdt_lr_model = GBDTLRModel()
    else:
        print("=" * 70)
        print("🚀 使用单一 LightGBM 模型")
        print("=" * 70)
        lgbm_model = MLTradingModel()
        gbdt_lr_model = None

    if args.mode == 'train':
        print("=" * 50)
        print("训练模式")
        print("=" * 50)

        if train_both:
            # 训练 LGBM 模型
            print("\n" + "="*70)
            print("🌳 训练 LightGBM 模型")
            print("="*70)
            lgbm_feature_importance = lgbm_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon)
            # 添加周期后缀：_1d, _5d, _20d
            horizon_suffix = f'_{args.horizon}d'
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            lgbm_model.save_model(lgbm_model_path)
            lgbm_importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
            lgbm_feature_importance.to_csv(lgbm_importance_path, index=False)
            print(f"\nLightGBM 模型已保存到 {lgbm_model_path}")
            print(f"特征重要性已保存到 {lgbm_importance_path}")

            # 训练 GBDT + LR 模型
            print("\n" + "="*70)
            print("🌲 训练 GBDT + LR 模型")
            print("="*70)
            gbdt_lr_feature_importance = gbdt_lr_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon)
            gbdt_lr_model_path = args.model_path.replace('.pkl', f'_gbdt_lr{horizon_suffix}.pkl')
            gbdt_lr_model.save_model(gbdt_lr_model_path)
            gbdt_lr_importance_path = gbdt_lr_model_path.replace('.pkl', '_importance.csv')
            gbdt_lr_feature_importance.to_csv(gbdt_lr_importance_path, index=False)
            print(f"\nGBDT + LR 模型已保存到 {gbdt_lr_model_path}")
            print(f"特征重要性已保存到 {gbdt_lr_importance_path}")

            # 对比特征重要性
            print("\n" + "="*70)
            print("📊 特征重要性对比")
            print("="*70)

            # 确保 Impact_Direction 列存在
            if 'Impact_Direction' not in lgbm_feature_importance.columns:
                lgbm_feature_importance['Impact_Direction'] = 'Unknown'
            if 'Impact_Direction' not in gbdt_lr_feature_importance.columns:
                gbdt_lr_feature_importance['Impact_Direction'] = 'Unknown'

            # 合并特征重要性
            comparison = lgbm_feature_importance.merge(
                gbdt_lr_feature_importance[['Feature', 'Gain_Importance', 'Impact_Direction']],
                on='Feature',
                suffixes=('_LGBM', '_GBDT_LR')
            )

            # 计算重要性差异（使用 Gain_Importance）
            comparison['Importance_Diff'] = abs(comparison['Gain_Importance_LGBM'] - comparison['Gain_Importance_GBDT_LR'])
            comparison = comparison.sort_values('Importance_Diff', ascending=False)

            print("\nTop 10 特征重要性差异:")
            print(comparison[['Feature', 'Gain_Importance_LGBM', 'Gain_Importance_GBDT_LR', 'Impact_Direction_LGBM', 'Impact_Direction_GBDT_LR']].head(10))

        else:
            # 训练单个模型
            horizon_suffix = f'_{args.horizon}d'
            if lgbm_model:
                feature_importance = lgbm_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon)
                lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
                lgbm_model.save_model(lgbm_model_path)
                importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
                feature_importance.to_csv(importance_path, index=False)
                print(f"\n特征重要性已保存到 {importance_path}")
            else:
                feature_importance = gbdt_lr_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon)
                gbdt_lr_model_path = args.model_path.replace('.pkl', f'_gbdt_lr{horizon_suffix}.pkl')
                gbdt_lr_model.save_model(gbdt_lr_model_path)
                importance_path = gbdt_lr_model_path.replace('.pkl', '_importance.csv')
                feature_importance.to_csv(importance_path, index=False)
                print(f"\n特征重要性已保存到 {importance_path}")

    elif args.mode == 'predict':
        print("=" * 50)
        print("预测模式")
        print("=" * 50)

        # 辅助函数：计算指定交易日后的目标日期
        def get_target_date(date, horizon=1):
            """计算指定交易日后的目标日期，跳过周末
            
            Args:
                date: 起始日期
                horizon: 预测周期（1=次日，5=一周，20=一个月）
            
            Returns:
                目标日期字符串 (YYYY-MM-DD)
            """
            target_day = date + pd.Timedelta(days=horizon)
            # 跳过周末
            while target_day.weekday() >= 5:
                target_day += pd.Timedelta(days=1)
            return target_day.strftime('%Y-%m-%d')

        if train_both:
            # 加载两个模型
            print("\n加载模型...")
            # 添加周期后缀：_1d, _5d, _20d
            horizon_suffix = f'_{args.horizon}d'
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            gbdt_lr_model_path = args.model_path.replace('.pkl', f'_gbdt_lr{horizon_suffix}.pkl')
            
            lgbm_model.load_model(lgbm_model_path)
            gbdt_lr_model.load_model(gbdt_lr_model_path)

            # 预测所有股票
            print("\n开始预测...")
            if args.predict_date:
                print(f"基于日期: {args.predict_date}")
            lgbm_predictions = []
            gbdt_lr_predictions = []

            for code in WATCHLIST:
                lgbm_result = lgbm_model.predict(code, predict_date=args.predict_date)
                gbdt_lr_result = gbdt_lr_model.predict(code, predict_date=args.predict_date)
                
                if lgbm_result and gbdt_lr_result:
                    lgbm_predictions.append(lgbm_result)
                    gbdt_lr_predictions.append(gbdt_lr_result)

            # 合并预测结果
            lgbm_pred_df = pd.DataFrame(lgbm_predictions)
            gbdt_lr_pred_df = pd.DataFrame(gbdt_lr_predictions)

            # 添加数据日期和目标日期
            lgbm_pred_df['data_date'] = lgbm_pred_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            lgbm_pred_df['target_date'] = lgbm_pred_df['date'].apply(lambda x: get_target_date(x, horizon=args.horizon))

            gbdt_lr_pred_df['data_date'] = gbdt_lr_pred_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            gbdt_lr_pred_df['target_date'] = gbdt_lr_pred_df['date'].apply(lambda x: get_target_date(x, horizon=args.horizon))

            # 合并对比
            comparison = lgbm_pred_df.merge(
                gbdt_lr_pred_df,
                on='code',
                suffixes=('_LGBM', '_GBDT_LR')
            )

            # 计算预测一致性
            comparison['预测一致'] = comparison['prediction_LGBM'] == comparison['prediction_GBDT_LR']
            comparison['概率差异'] = abs(comparison['probability_LGBM'] - comparison['probability_GBDT_LR'])

            # 显示对比结果
            print("\n" + "=" * 140)
            print("📊 两种模型预测结果对比")
            print("=" * 140)
            print(f"\n{'代码':<10} {'股票名称':<12} {'LGBM预测':<10} {'LGBM概率':<10} {'GBDT+LR预测':<12} {'GBDT+LR概率':<12} {'是否一致':<8} {'概率差异':<10} {'当前价格':<10} {'预测目标':<12}")
            print("-" * 140)

            for _, row in comparison.iterrows():
                lgbm_pred_label = "上涨" if row['prediction_LGBM'] == 1 else "下跌"
                gbdt_lr_pred_label = "上涨" if row['prediction_GBDT_LR'] == 1 else "下跌"
                consistent = "✓" if row['预测一致'] else "✗"

                print(f"{row['code']:<10} {row['name_LGBM']:<12} {lgbm_pred_label:<10} {row['probability_LGBM']:<10.4f} {gbdt_lr_pred_label:<12} {row['probability_GBDT_LR']:<12.4f} {consistent:<8} {row['概率差异']:<10.4f} {row['current_price_LGBM']:<10.2f} {row['target_date_LGBM']:<12}")

            # 统计摘要
            print("\n" + "=" * 140)
            print("📈 统计摘要")
            print("=" * 140)

            consistent_count = comparison['预测一致'].sum()
            total_count = len(comparison)
            print(f"\n预测一致性: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)")

            lgbm_up = (comparison['prediction_LGBM'] == 1).sum()
            lgbm_down = (comparison['prediction_LGBM'] == 0).sum()
            print(f"\nLGBM 模型: 上涨 {lgbm_up} 只, 下跌 {lgbm_down} 只")

            gbdt_lr_up = (comparison['prediction_GBDT_LR'] == 1).sum()
            gbdt_lr_down = (comparison['prediction_GBDT_LR'] == 0).sum()
            print(f"GBDT+LR 模型: 上涨 {gbdt_lr_up} 只, 下跌 {gbdt_lr_down} 只")

            avg_prob_diff = comparison['概率差异'].mean()
            print(f"\n平均概率差异: {avg_prob_diff:.4f}")

            # 显示不一致的预测
            inconsistent = comparison[~comparison['预测一致']]
            if len(inconsistent) > 0:
                print("\n" + "=" * 140)
                print("⚠️  预测不一致的股票")
                print("=" * 140)
                for _, row in inconsistent.iterrows():
                    lgbm_pred_label = "上涨" if row['prediction_LGBM'] == 1 else "下跌"
                    gbdt_lr_pred_label = "上涨" if row['prediction_GBDT_LR'] == 1 else "下跌"
                    print(f"{row['code']:<10} {row['name_LGBM']:<12} LGBM: {lgbm_pred_label} ({row['probability_LGBM']:.4f})  vs  GBDT+LR: {gbdt_lr_pred_label} ({row['probability_GBDT_LR']:.4f})")

            # 保存对比结果
            comparison_export = comparison[[
                'code', 'name_LGBM', 'prediction_LGBM', 'probability_LGBM',
                'prediction_GBDT_LR', 'probability_GBDT_LR', '预测一致', '概率差异',
                'current_price_LGBM', 'data_date_LGBM', 'target_date_LGBM'
            ]]
            comparison_export.columns = [
                'code', 'name', 'prediction_LGBM', 'probability_LGBM',
                'prediction_GBDT_LR', 'probability_GBDT_LR', 'consistent', 'probability_diff',
                'current_price', 'data_date', 'target_date'
            ]
            
            comparison_path = args.model_path.replace('.pkl', '_comparison.csv')
            comparison_export.to_csv(comparison_path, index=False)
            print(f"\n对比结果已保存到 {comparison_path}")

            # 保存各自的预测结果
            horizon_suffix = f'_{args.horizon}d'
            lgbm_pred_path = args.model_path.replace('.pkl', f'_lgbm_predictions{horizon_suffix}.csv')
            lgbm_pred_df[['code', 'name', 'prediction', 'probability', 'current_price', 'data_date', 'target_date']].to_csv(lgbm_pred_path, index=False)
            print(f"LGBM 预测结果已保存到 {lgbm_pred_path}")

            gbdt_lr_pred_path = args.model_path.replace('.pkl', f'_gbdt_lr_predictions{horizon_suffix}.csv')
            gbdt_lr_pred_df[['code', 'name', 'prediction', 'probability', 'current_price', 'data_date', 'target_date']].to_csv(gbdt_lr_pred_path, index=False)
            print(f"GBDT+LR 预测结果已保存到 {gbdt_lr_pred_path}")

        else:
            # 单个模型预测
            model = lgbm_model if lgbm_model else gbdt_lr_model
            horizon_suffix = f'_{args.horizon}d'
            if lgbm_model:
                model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            else:
                model_path = args.model_path.replace('.pkl', f'_gbdt_lr{horizon_suffix}.pkl')
            model.load_model(model_path)

            # 预测所有股票
            predictions = []
            if args.predict_date:
                print(f"基于日期: {args.predict_date}")
            for code in WATCHLIST:
                result = model.predict(code, predict_date=args.predict_date)
                if result:
                    predictions.append(result)

            # 显示预测结果
            print("\n预测结果:")
            horizon_text = {1: "次日", 5: "一周", 20: "一个月"}.get(args.horizon, f"{args.horizon}天")
            if args.predict_date:
                print(f"说明: 基于 {args.predict_date} 的数据预测{horizon_text}后的涨跌")
            else:
                print(f"说明: 基于最新交易日的数据预测{horizon_text}后的涨跌")
            print("-" * 100)
            print(f"{'代码':<10} {'股票名称':<12} {'预测':<8} {'概率':<10} {'当前价格':<12} {'数据日期':<15} {'预测目标':<15}")
            print("-" * 100)

            for pred in predictions:
                pred_label = "上涨" if pred['prediction'] == 1 else "下跌"
                data_date = pred['date'].strftime('%Y-%m-%d')
                target_date = get_target_date(pred['date'], horizon=args.horizon)

                print(f"{pred['code']:<10} {pred['name']:<12} {pred_label:<8} {pred['probability']:.4f}    {pred['current_price']:.2f}        {data_date:<15} {target_date:<15}")

            # 保存预测结果
            pred_df = pd.DataFrame(predictions)
            pred_df['data_date'] = pred_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            pred_df['target_date'] = pred_df['date'].apply(lambda x: get_target_date(x, horizon=args.horizon))
            
            pred_df_export = pred_df[['code', 'name', 'prediction', 'probability', 'current_price', 'data_date', 'target_date']]
            
            horizon_suffix = f'_{args.horizon}d'
            pred_path = args.model_path.replace('.pkl', f'_predictions{horizon_suffix}.csv')
            pred_df_export.to_csv(pred_path, index=False)
            print(f"\n预测结果已保存到 {pred_path}")

    elif args.mode == 'evaluate':
        print("=" * 50)
        print("评估模式")
        print("=" * 50)

        if train_both:
            # 加载两个模型
            print("\n加载模型...")
            horizon_suffix = f'_{args.horizon}d'
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            gbdt_lr_model_path = args.model_path.replace('.pkl', f'_gbdt_lr{horizon_suffix}.pkl')
            
            lgbm_model.load_model(lgbm_model_path)
            gbdt_lr_model.load_model(gbdt_lr_model_path)

            # 准备测试数据
            print("准备测试数据...")
            test_df = lgbm_model.prepare_data(WATCHLIST)
            test_df = test_df.dropna()

            X_test = test_df[lgbm_model.feature_columns].values
            y_test = test_df['Label'].values

            # LGBM 模型评估
            print("\n" + "="*70)
            print("🌳 LightGBM 模型评估")
            print("="*70)
            y_pred_lgbm = lgbm_model.model.predict(X_test)
            print("\n分类报告:")
            print(classification_report(y_test, y_pred_lgbm))
            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred_lgbm))
            lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
            print(f"\n准确率: {lgbm_accuracy:.4f}")

            # GBDT + LR 模型评估
            print("\n" + "="*70)
            print("🌲 GBDT + LR 模型评估")
            print("="*70)
            gbdt_leaf_test = gbdt_lr_model.gbdt_model.booster_.predict(X_test, pred_leaf=True)
            df_gbdt_leaf_test = pd.DataFrame(gbdt_leaf_test, columns=gbdt_lr_model.gbdt_leaf_names)

            df_gbdt_onehot_test = pd.DataFrame()
            for col in gbdt_lr_model.gbdt_leaf_names:
                onehot_feats = pd.get_dummies(df_gbdt_leaf_test[col], prefix=col)
                df_gbdt_onehot_test = pd.concat([df_gbdt_onehot_test, onehot_feats], axis=1)

            for col in gbdt_lr_model.lr_model.feature_names_in_:
                if col not in df_gbdt_onehot_test.columns:
                    df_gbdt_onehot_test[col] = 0

            df_gbdt_onehot_test = df_gbdt_onehot_test[gbdt_lr_model.lr_model.feature_names_in_]
            y_pred_gbdt_lr = gbdt_lr_model.lr_model.predict(df_gbdt_onehot_test)

            print("\n分类报告:")
            print(classification_report(y_test, y_pred_gbdt_lr))
            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred_gbdt_lr))
            gbdt_lr_accuracy = accuracy_score(y_test, y_pred_gbdt_lr)
            print(f"\n准确率: {gbdt_lr_accuracy:.4f}")

            # 对比结果
            print("\n" + "="*70)
            print("📊 模型对比")
            print("="*70)
            print(f"LightGBM 准确率: {lgbm_accuracy:.4f}")
            print(f"GBDT + LR 准确率: {gbdt_lr_accuracy:.4f}")
            print(f"准确率差异: {abs(lgbm_accuracy - gbdt_lr_accuracy):.4f}")
            
            if gbdt_lr_accuracy > lgbm_accuracy:
                print(f"\n✅ GBDT + LR 模型表现更好，提升 {gbdt_lr_accuracy - lgbm_accuracy:.4f} ({(gbdt_lr_accuracy - lgbm_accuracy)/lgbm_accuracy*100:.2f}%)")
            elif lgbm_accuracy > gbdt_lr_accuracy:
                print(f"\n✅ LightGBM 模型表现更好，提升 {lgbm_accuracy - gbdt_lr_accuracy:.4f} ({(lgbm_accuracy - gbdt_lr_accuracy)/gbdt_lr_accuracy*100:.2f}%)")
            else:
                print(f"\n⚖️  两种模型表现相同")

        else:
            # 单个模型评估
            model = lgbm_model if lgbm_model else gbdt_lr_model
            model.load_model(args.model_path)

            # 准备测试数据
            print("准备测试数据...")
            test_df = model.prepare_data(WATCHLIST)
            test_df = test_df.dropna()

            X_test = test_df[model.feature_columns].values
            y_test = test_df['Label'].values

            # 根据模型类型进行预测
            if gbdt_lr_model:
                # GBDT + LR 模型需要先通过 GBDT 获取叶子节点特征
                gbdt_leaf_test = model.gbdt_model.booster_.predict(X_test, pred_leaf=True)
                df_gbdt_leaf_test = pd.DataFrame(gbdt_leaf_test, columns=model.gbdt_leaf_names)

                # One-Hot 编码
                df_gbdt_onehot_test = pd.DataFrame()
                for col in model.gbdt_leaf_names:
                    onehot_feats = pd.get_dummies(df_gbdt_leaf_test[col], prefix=col)
                    df_gbdt_onehot_test = pd.concat([df_gbdt_onehot_test, onehot_feats], axis=1)

                # 确保特征列与训练时一致
                for col in model.lr_model.feature_names_in_:
                    if col not in df_gbdt_onehot_test.columns:
                        df_gbdt_onehot_test[col] = 0

                df_gbdt_onehot_test = df_gbdt_onehot_test[model.lr_model.feature_names_in_]

                # 使用 LR 预测
                y_pred = model.lr_model.predict(df_gbdt_onehot_test)
            else:
                # 单一 LightGBM 模型
                y_pred = model.model.predict(X_test)

            # 评估
            print("\n分类报告:")
            print(classification_report(y_test, y_pred))

            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred))

            print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == '__main__':
    main()
