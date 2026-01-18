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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb

# 导入项目模块
from tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from technical_analysis import TechnicalAnalyzer
from fundamental_data import get_comprehensive_fundamental_data

# 自选股列表
WATCHLIST = [
    "0005.HK", "0388.HK", "0700.HK", "0728.HK", "0883.HK",
    "0939.HK", "0941.HK", "0981.HK", "1088.HK", "1138.HK",
    "1211.HK", "1288.HK", "1299.HK", "1330.HK", "1347.HK",
    "1398.HK", "1810.HK", "2269.HK", "2533.HK", "3690.HK",
    "3968.HK", "6682.HK", "9660.HK", "9988.HK"
]


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
            fundamental_data = get_comprehensive_fundamental_data(code)
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

    def create_market_environment_features(self, stock_df, hsi_df):
        """创建市场环境特征"""
        if stock_df.empty or hsi_df.empty:
            return stock_df

        # 计算恒生指数收益率
        hsi_df['HSI_Return'] = hsi_df['Close'].pct_change()
        hsi_df['HSI_Return_5d'] = hsi_df['Close'].pct_change(5)

        # 合并数据
        stock_df = stock_df.merge(hsi_df[['HSI_Return', 'HSI_Return_5d']], left_index=True, right_index=True, how='left')

        # 相对表现
        stock_df['Relative_Return'] = stock_df['Return_5d'] - stock_df['HSI_Return_5d']

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
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

    def prepare_data(self, codes, start_date=None, end_date=None):
        """准备训练数据"""
        all_data = []

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

                # 创建市场环境特征
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df)

                # 创建标签
                stock_df = self.feature_engineer.create_label(stock_df, horizon=1)

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
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
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

    def train(self, codes, start_date=None, end_date=None):
        """训练模型"""
        print("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date)

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

        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\n特征重要性 Top 10:")
        print(feature_importance.head(10))

        return feature_importance

    def predict(self, code):
        """预测单只股票"""
        try:
            # 移除代码中的.HK后缀，腾讯财经接口不需要
            stock_code = code.replace('.HK', '')

            # 获取股票数据（2年约730天）
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=730)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据（2年约730天）
            hsi_df = get_hsi_data_tencent(period_days=730)
            if hsi_df is None or hsi_df.empty:
                return None

            # 计算特征
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df)

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

            # 预测
            proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]

            return {
                'code': code,
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


def main():
    parser = argparse.ArgumentParser(description='机器学习交易模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                       help='运行模式: train=训练, predict=预测, evaluate=评估')
    parser.add_argument('--model-path', type=str, default='data/ml_trading_model.pkl',
                       help='模型保存/加载路径')
    parser.add_argument('--start-date', type=str, default=None,
                       help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='训练结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    model = MLTradingModel()

    if args.mode == 'train':
        print("=" * 50)
        print("训练模式")
        print("=" * 50)

        # 训练模型
        feature_importance = model.train(WATCHLIST, args.start_date, args.end_date)

        # 保存模型
        model.save_model(args.model_path)

        # 保存特征重要性
        importance_path = args.model_path.replace('.pkl', '_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        print(f"\n特征重要性已保存到 {importance_path}")

    elif args.mode == 'predict':
        print("=" * 50)
        print("预测模式")
        print("=" * 50)

        # 加载模型
        model.load_model(args.model_path)

        # 预测所有股票
        predictions = []
        for code in WATCHLIST:
            result = model.predict(code)
            if result:
                predictions.append(result)

        # 显示预测结果
        print("\n预测结果:")
        print("-" * 80)
        print(f"{'代码':<10} {'预测':<8} {'概率':<10} {'当前价格':<12} {'日期':<15}")
        print("-" * 80)

        for pred in predictions:
            pred_label = "上涨" if pred['prediction'] == 1 else "下跌"
            print(f"{pred['code']:<10} {pred_label:<8} {pred['probability']:.4f}    {pred['current_price']:.2f}        {pred['date'].strftime('%Y-%m-%d')}")

        # 保存预测结果
        pred_df = pd.DataFrame(predictions)
        pred_path = args.model_path.replace('.pkl', '_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"\n预测结果已保存到 {pred_path}")

    elif args.mode == 'evaluate':
        print("=" * 50)
        print("评估模式")
        print("=" * 50)

        # 加载模型
        model.load_model(args.model_path)

        # 准备测试数据
        print("准备测试数据...")
        test_df = model.prepare_data(WATCHLIST)

        # 删除包含NaN的行
        test_df = test_df.dropna()

        # 准备特征和标签
        X_test = test_df[model.feature_columns].values
        y_test = test_df['Label'].values

        # 预测
        y_pred = model.model.predict(X_test)

        # 评估
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))

        print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == '__main__':
    main()
