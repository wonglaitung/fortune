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
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 缓存配置
CACHE_DIR = 'data/stock_cache'
STOCK_DATA_CACHE_DAYS = 7  # 股票历史数据缓存7天
HSI_DATA_CACHE_HOURS = 1   # 恒生指数数据缓存1小时

# 导入项目模块
from data_services.tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from data_services.technical_analysis import TechnicalAnalyzer
from data_services.fundamental_data import get_comprehensive_fundamental_data
from ml_services.base_model_processor import BaseModelProcessor
from ml_services.us_market_data import us_market_data
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

# 股票名称映射
STOCK_NAMES = STOCK_LIST

# 自选股列表（转换为列表格式）
WATCHLIST = list(STOCK_NAMES.keys())

# 获取日志记录器
logger = get_logger('ml_trading_model')


# ========== 保存预测结果到文本文件 ==========
def save_predictions_to_text(predictions_df, predict_date=None):
    """
    保存预测结果到文本文件，方便后续提取和对比

    参数:
    - predictions_df: 预测结果DataFrame
    - predict_date: 预测日期
    """
    try:
        from datetime import datetime

        # 生成文件名（使用日期）
        if predict_date:
            date_str = predict_date
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # 创建data目录（如果不存在）
        if not os.path.exists('data'):
            os.makedirs('data')

        # 文件路径
        filepath = f'data/ml_predictions_20d_{date_str}.txt'

        # 构建内容
        content = f"{'=' * 80}\n"
        content += f"机器学习20天预测结果\n"
        content += f"预测日期: {date_str}\n"
        content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"{'=' * 80}\n\n"

        # 添加预测结果
        content += "【预测结果】\n"
        content += "-" * 80 + "\n"
        content += f"{'股票代码':<10} {'股票名称':<12} {'预测方向':<10} {'上涨概率':<12} {'当前价格':<12} {'数据日期':<15} {'预测目标日期':<15}\n"
        content += "-" * 80 + "\n"

        # 按一致性排序（如果有consistent列）
        if 'consistent' in predictions_df.columns:
            predictions_df_sorted = predictions_df.sort_values(by=['consistent', 'avg_probability'], ascending=[False, False])
        else:
            predictions_df_sorted = predictions_df.sort_values(by='probability', ascending=False)

        for _, row in predictions_df_sorted.iterrows():
            code = row.get('code', 'N/A')
            name = row.get('name', 'N/A')
            current_price = row.get('current_price', None)
            data_date = row.get('data_date', 'N/A')
            target_date = row.get('target_date', 'N/A')
            
            # 尝试获取预测和概率（支持多种列名格式）
            prediction = None
            probability = None
            
            # 优先使用平均概率和一致性判断
            if 'avg_probability' in row and 'consistent' in row:
                if row['consistent']:
                    # 两个模型一致，使用平均概率
                    probability = row['avg_probability']
                    prediction = 1 if probability >= 0.5 else 0
            elif 'prediction' in row:
                prediction = row.get('prediction', None)
                probability = row.get('probability', None)
            elif 'prediction_LGBM' in row:
                # 使用LGBM的预测
                prediction = row.get('prediction_LGBM', None)
                probability = row.get('probability_LGBM', None)

            if prediction is not None:
                pred_label = "上涨" if prediction == 1 else "下跌"
                prob_str = f"{probability:.4f}" if probability is not None else "N/A"
                price_str = f"{current_price:.2f}" if current_price is not None else "N/A"
            else:
                pred_label = "N/A"
                prob_str = "N/A"
                price_str = "N/A"

            content += f"{code:<10} {name:<12} {pred_label:<10} {prob_str:<12} {price_str:<12} {data_date:<15} {target_date:<15}\n"

        # 添加统计信息
        content += "\n" + "-" * 80 + "\n"
        content += "【统计信息】\n"
        content += "-" * 80 + "\n"

        # 初始化变量
        total_count = 0
        up_count = 0
        down_count = 0
        consistent_count = 0
        
        # 计算统计信息
        total_count = len(predictions_df)
        
        # 计算上涨和下跌数量
        if 'avg_probability' in predictions_df.columns:
            up_count = (predictions_df['avg_probability'] >= 0.5).sum()
            down_count = total_count - up_count
        elif 'prediction' in predictions_df.columns:
            up_count = (predictions_df['prediction'] == 1).sum()
            down_count = (predictions_df['prediction'] == 0).sum()
        elif 'prediction_LGBM' in predictions_df.columns:
            up_count = (predictions_df['prediction_LGBM'] == 1).sum()
            down_count = total_count - up_count
        
        if total_count > 0:
            content += f"预测上涨: {up_count} 只\n"
            content += f"预测下跌: {down_count} 只\n"
            content += f"总计: {total_count} 只\n"
            content += f"上涨比例: {up_count/total_count*100:.1f}%\n"

        if 'consistent' in predictions_df.columns:
            consistent_count = predictions_df['consistent'].sum()
            content += f"\n两个模型一致性: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)\n"

        if 'avg_probability' in predictions_df.columns:
            avg_prob = predictions_df['avg_probability'].mean()
            content += f"平均上涨概率: {avg_prob:.4f}\n"

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"20天预测结果已保存到 {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"保存预测结果失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_target_date(date, horizon):
    """计算目标日期（数据日期 + 预测周期）"""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    target_date = date + timedelta(days=horizon)
    return target_date.strftime('%Y-%m-%d')


# ========== 缓存辅助函数 ==========
def _get_cache_key(stock_code, period_days):
    """生成缓存键"""
    return f"{stock_code}_{period_days}d"

def _get_cache_file_path(cache_key):
    """获取缓存文件路径"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def _is_cache_valid(cache_file_path, cache_hours):
    """检查缓存是否有效"""
    if not os.path.exists(cache_file_path):
        return False
    cache_time = os.path.getmtime(cache_file_path)
    current_time = datetime.now().timestamp()
    age_hours = (current_time - cache_time) / 3600
    return age_hours < cache_hours

def _save_cache(cache_file_path, data):
    """保存缓存"""
    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump({
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, f)
    except Exception as e:
        logger.warning(f"保存缓存失败: {e}")

def _load_cache(cache_file_path):
    """加载缓存"""
    try:
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
            return cache['data']
    except Exception as e:
        logger.warning(f"加载缓存失败: {e}")
        return None

def get_stock_data_with_cache(stock_code, period_days=730):
    """获取股票数据（带缓存）"""
    cache_key = _get_cache_key(stock_code, period_days)
    cache_file_path = _get_cache_file_path(cache_key)
    
    # 检查缓存
    if _is_cache_valid(cache_file_path, STOCK_DATA_CACHE_DAYS * 24):
        logger.debug(f"使用缓存的股票数据 {stock_code}")
        cached_data = _load_cache(cache_file_path)
        if cached_data is not None:
            return cached_data

    # 从网络获取
        logger.debug(f"下载股票数据 {stock_code}")
    stock_df = get_hk_stock_data_tencent(stock_code, period_days)
    
    # 保存缓存
    if stock_df is not None and not stock_df.empty:
        _save_cache(cache_file_path, stock_df)
    
    return stock_df

def get_hsi_data_with_cache(period_days=730):
    """获取恒生指数数据（带缓存）"""
    cache_key = _get_cache_key("HSI", period_days)
    cache_file_path = _get_cache_file_path(cache_key)
    
    # 检查缓存
    if _is_cache_valid(cache_file_path, HSI_DATA_CACHE_HOURS):
        logger.debug("使用缓存的恒生指数数据")
        cached_data = _load_cache(cache_file_path)
        if cached_data is not None:
            return cached_data

    # 从网络获取
        logger.debug("下载恒生指数数据")
    hsi_df = get_hsi_data_tencent(period_days)
    
    # 保存缓存
    if hsi_df is not None and not hsi_df.empty:
        _save_cache(cache_file_path, hsi_df)
    
    return hsi_df


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        # 板块分析缓存（避免重复计算）
        self._sector_analyzer = None
        self._sector_performance_cache = {}
        # 新闻数据缓存（避免重复加载）
        self._news_data_cache = None
        self._news_data_days = 30

    def _get_sector_analyzer(self):
        """获取板块分析器（单例模式）"""
        if self._sector_analyzer is None:
            try:
                from data_services.hk_sector_analysis import SectorAnalyzer
                self._sector_analyzer = SectorAnalyzer()
                logger.debug("板块分析器初始化成功")
            except ImportError:
                logger.warning("板块分析模块不可用")
                return None
        return self._sector_analyzer

    def _get_sector_performance(self, period):
        """获取板块表现数据（带缓存）"""
        cache_key = f'period_{period}'
        
        if cache_key not in self._sector_performance_cache:
            analyzer = self._get_sector_analyzer()
            if analyzer is None:
                return None
            
            try:
                perf_df = analyzer.calculate_sector_performance(period)
                self._sector_performance_cache[cache_key] = perf_df
            except Exception as e:
                print(f"  ⚠️ 获取板块表现失败 (period={period}): {e}")
                return None
        
        return self._sector_performance_cache[cache_key]

    def calculate_technical_features(self, df):
        """计算技术指标特征（扩展版：80个指标）"""
        if df.empty or len(df) < 200:
            return df

        # ========== 基础移动平均线 ==========
        df = self.tech_analyzer.calculate_moving_averages(df, periods=[5, 10, 20, 50, 100, 200])

        # ========== RSI (Wilder 平滑) ==========
        df = self.tech_analyzer.calculate_rsi(df, period=14)
        # RSI 变化率
        df['RSI_ROC'] = df['RSI'].pct_change()

        # ========== MACD ==========
        df = self.tech_analyzer.calculate_macd(df)
        # MACD 柱状图
        df['MACD_Hist'] = df['MACD'] - df['MACD_signal']
        # MACD 柱状图变化率
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change()

        # ========== 布林带 ==========
        df = self.tech_analyzer.calculate_bollinger_bands(df, period=20, std_dev=2)
        # 布林带宽度
        df['BB_Width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        # 布林带突破
        df['BB_Breakout'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ========== ATR ==========
        df = self.tech_analyzer.calculate_atr(df, period=14)
        # ATR 比率（ATR相对于10日均线的比率）
        df['ATR_MA'] = df['ATR'].rolling(window=10, min_periods=1).mean()
        df['ATR_Ratio'] = df['ATR'] / df['ATR_MA']

        # ========== 成交量相关 ==========
        df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
        # 成交量 z-score
        df['Vol_Mean_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Vol_Std_20'] = df['Volume'].rolling(20, min_periods=1).std()
        df['Vol_Z_Score'] = (df['Volume'] - df['Vol_Mean_20']) / df['Vol_Std_20']
        # 成交额
        df['Turnover'] = df['Close'] * df['Volume']
        # 成交额 z-score
        df['Turnover_Mean_20'] = df['Turnover'].rolling(20, min_periods=1).mean()
        df['Turnover_Std_20'] = df['Turnover'].rolling(20, min_periods=1).std()
        df['Turnover_Z_Score'] = (df['Turnover'] - df['Turnover_Mean_20']) / df['Turnover_Std_20']
        # 成交额变化率（多周期）
        df['Turnover_Change_1d'] = df['Turnover'].pct_change()
        df['Turnover_Change_5d'] = df['Turnover'].pct_change(5)
        df['Turnover_Change_10d'] = df['Turnover'].pct_change(10)
        df['Turnover_Change_20d'] = df['Turnover'].pct_change(20)
        # 换手率（假设总股本为常数，这里使用成交额/价格作为近似）
        df['Turnover_Rate'] = (df['Turnover'] / (df['Close'] * 1000000)) * 100
        # 换手率变化率
        df['Turnover_Rate_Change_5d'] = df['Turnover_Rate'].pct_change(5)
        df['Turnover_Rate_Change_20d'] = df['Turnover_Rate'].pct_change(20)

        # ========== VWAP (成交量加权平均价) ==========
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['TP'] * df['Volume']).rolling(window=20, min_periods=1).sum() / df['Volume'].rolling(window=20, min_periods=1).sum()

        # ========== OBV (能量潮) ==========
        df['OBV'] = 0.0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]

        # ========== CMF (Chaikin Money Flow) ==========
        df['MF_Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['MF_Volume'] = df['MF_Multiplier'] * df['Volume']
        df['CMF'] = df['MF_Volume'].rolling(20, min_periods=1).sum() / df['Volume'].rolling(20, min_periods=1).sum()
        # CMF 信号线
        df['CMF_Signal'] = df['CMF'].rolling(5, min_periods=1).mean()

        # ========== ADX (平均趋向指数) ==========
        # +DM and -DM
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        # +DI and -DI
        df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        # ADX
        dx = 100 * (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # ========== 随机振荡器 (Stochastic Oscillator) ==========
        K_Period = 14
        D_Period = 3
        df['Low_Min'] = df['Low'].rolling(window=K_Period, min_periods=1).min()
        df['High_Max'] = df['High'].rolling(window=K_Period, min_periods=1).max()
        df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min'])
        df['Stoch_D'] = df['Stoch_K'].rolling(window=D_Period, min_periods=1).mean()

        # ========== Williams %R ==========
        df['Williams_R'] = (df['High_Max'] - df['Close']) / (df['High_Max'] - df['Low_Min']) * -100

        # ========== ROC (价格变化率) ==========
        df['ROC'] = df['Close'].pct_change(periods=12)

        # ========== 波动率（年化） ==========
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(20, min_periods=10).std() * np.sqrt(252)

        # ========== 价格位置特征 ==========
        # 价格相对于均线的偏离
        df['MA5_Deviation'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['MA10_Deviation'] = (df['Close'] - df['MA10']) / df['MA10'] * 100
        # 价格百分位（相对于60日窗口）
        df['Price_Percentile'] = df['Close'].rolling(window=60, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
        )
        # 布林带位置
        df['BB_Position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ========== 多周期收益率 ==========
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_3d'] = df['Close'].pct_change(3)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)
        df['Return_60d'] = df['Close'].pct_change(60)

        # ========== 价格相对于均线的比率 ==========
        df['Price_Ratio_MA5'] = df['Close'] / df['MA5']
        df['Price_Ratio_MA20'] = df['Close'] / df['MA20']
        df['Price_Ratio_MA50'] = df['Close'] / df['MA50']

        # ========== 高优先级：滚动统计特征 ==========
        # 均线偏离度（标准化）
        df['MA5_Deviation_Std'] = (df['Close'] - df['MA5']) / df['Close'].rolling(5).std()
        df['MA20_Deviation_Std'] = (df['Close'] - df['MA20']) / df['Close'].rolling(20).std()

        # 滚动波动率（多周期）
        df['Volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        df['Volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()

        # 滚动偏度/峰度（业界常用）
        df['Skewness_20d'] = df['Close'].pct_change().rolling(20).skew()
        df['Kurtosis_20d'] = df['Close'].pct_change().rolling(20).kurt()

        # 动量加速度（业界重要特征）
        df['Momentum_Accel_5d'] = df['Return_5d'] - df['Return_5d'].shift(5)
        df['Momentum_Accel_10d'] = df['Return_10d'] - df['Return_10d'].shift(5)

        # ========== 高优先级：价格形态特征 ==========
        # N日高低点位置（0-1之间，1表示在最高点）
        df['High_Position_20d'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        df['High_Position_60d'] = (df['Close'] - df['Low'].rolling(60).min()) / (df['High'].rolling(60).max() - df['Low'].rolling(60).min())

        # 距离近期高点/低点的天数（业界常用）
        df['Days_Since_High_20d'] = df['Close'].rolling(20).apply(lambda x: 20 - np.argmax(x), raw=False)
        df['Days_Since_Low_20d'] = df['Close'].rolling(20).apply(lambda x: 20 - np.argmin(x), raw=False)

        # 日内特征（业界核心信号）
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Intraday_Range_MA5'] = df['Intraday_Range'].rolling(5).mean()
        df['Intraday_Range_MA20'] = df['Intraday_Range'].rolling(20).mean()

        # 收盘位置（阳线/阴线强度，0-1之间）
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        # 上影线/下影线比例
        df['Upper_Shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)

        # 开盘缺口
        df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap_Size'] > 0.01).astype(int)  # 跳空高开 >1%
        df['Gap_Down'] = (df['Gap_Size'] < -0.01).astype(int)  # 跳空低开 >1%

        # ========== 中优先级：量价关系特征 ==========
        # 量价背离（业界重要信号）
        df['Price_Up_Volume_Down'] = ((df['Return_1d'] > 0) & (df['Turnover'].pct_change() < 0)).astype(int)
        df['Price_Down_Volume_Up'] = ((df['Return_1d'] < 0) & (df['Turnover'].pct_change() > 0)).astype(int)

        # OBV 趋势
        df['OBV_MA5'] = df['OBV'].rolling(5).mean()
        df['OBV_Trend'] = (df['OBV'] > df['OBV_MA5']).astype(int)

        # 成交量波动率
        df['Volume_Volatility'] = df['Turnover'].rolling(20).std() / (df['Turnover'].rolling(20).mean() + 1e-10)

        # 成交量比率（多周期）
        df['Volume_Ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['Volume_Ratio_20d'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # ========== 长期趋势特征（专门优化一个月模型） ==========
        # 长期均线（120日半年线、250日年线）
        df['MA120'] = df['Close'].rolling(window=120, min_periods=1).mean()
        df['MA250'] = df['Close'].rolling(window=250, min_periods=1).mean()

        # 价格相对长期均线的比率（业界长期趋势指标）
        df['Price_Ratio_MA120'] = df['Close'] / df['MA120']
        df['Price_Ratio_MA250'] = df['Close'] / df['MA250']

        # 长期收益率（业界核心长期特征）
        df['Return_120d'] = df['Close'].pct_change(120)
        df['Return_250d'] = df['Close'].pct_change(250)

        # 长期动量（Momentum = 当前价格 / N日前价格 - 1）
        df['Momentum_120d'] = df['Close'] / df['Close'].shift(120) - 1
        df['Momentum_250d'] = df['Close'] / df['Close'].shift(250) - 1

        # 长期动量加速度（趋势变化的二阶导数）
        df['Momentum_Accel_120d'] = df['Return_120d'] - df['Return_120d'].shift(30)

        # 长期均线斜率（趋势强度指标）
        df['MA120_Slope'] = (df['MA120'] - df['MA120'].shift(10)) / df['MA120'].shift(10)
        df['MA250_Slope'] = (df['MA250'] - df['MA250'].shift(20)) / df['MA250'].shift(20)

        # 长期均线排列（多头/空头/混乱）
        df['MA_Alignment_Long'] = np.where(
            (df['MA50'] > df['MA120']) & (df['MA120'] > df['MA250']), 1,  # 多头排列
            np.where(
                (df['MA50'] < df['MA120']) & (df['MA120'] < df['MA250']), -1,  # 空头排列
                0  # 混乱排列
            )
        )

        # 长期均线乖离率（价格偏离长期均线的程度）
        df['MA120_Deviation'] = (df['Close'] - df['MA120']) / df['MA120'] * 100
        df['MA250_Deviation'] = (df['Close'] - df['MA250']) / df['MA250'] * 100

        # 长期波动率（风险指标）
        df['Volatility_60d'] = df['Close'].pct_change().rolling(60).std()
        df['Volatility_120d'] = df['Close'].pct_change().rolling(120).std()

        # 长期ATR（长期风险）
        df['ATR_MA60'] = df['ATR'].rolling(60, min_periods=1).mean()
        df['ATR_MA120'] = df['ATR'].rolling(120, min_periods=1).mean()
        df['ATR_Ratio_60d'] = df['ATR'] / df['ATR_MA60']
        df['ATR_Ratio_120d'] = df['ATR'] / df['ATR_MA120']

        # 长期成交量趋势
        df['Volume_MA120'] = df['Volume'].rolling(120, min_periods=1).mean()
        df['Volume_MA250'] = df['Volume'].rolling(250, min_periods=1).mean()
        df['Volume_Ratio_120d'] = df['Volume'] / df['Volume_MA120']
        df['Volume_Trend_Long'] = np.where(
            df['Volume_MA120'] > df['Volume_MA250'], 1, -1
        )

        # 长期支撑阻力位（基于120日高低点）
        df['Support_120d'] = df['Low'].rolling(120, min_periods=1).min()
        df['Resistance_120d'] = df['High'].rolling(120, min_periods=1).max()
        df['Distance_Support_120d'] = (df['Close'] - df['Support_120d']) / df['Close']
        df['Distance_Resistance_120d'] = (df['Resistance_120d'] - df['Close']) / df['Close']

        # 长期RSI（基于120日）
        df['RSI_120'] = self.tech_analyzer.calculate_rsi(df.copy(), period=120)['RSI']

        return df

    def create_fundamental_features(self, code):
        """创建基本面特征（只使用实际可用的数据）"""
        try:
            # 移除代码中的.HK后缀
            stock_code = code.replace('.HK', '')

            fundamental_data = get_comprehensive_fundamental_data(stock_code)
            if fundamental_data:
                # 只使用实际可用的基本面数据
                return {
                    'PE': fundamental_data.get('fi_pe_ratio', np.nan),
                    'PB': fundamental_data.get('fi_pb_ratio', np.nan),
                    'Market_Cap': fundamental_data.get('fi_market_cap', np.nan),
                    'ROE': np.nan,  # 暂不可用
                    'ROA': np.nan,  # 暂不可用
                    'Dividend_Yield': np.nan,  # 暂不可用
                    'EPS': np.nan,  # 暂不可用
                    'Net_Margin': np.nan,  # 暂不可用
                    'Gross_Margin': np.nan  # 暂不可用
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

    def create_stock_type_features(self, code, df):
        """创建股票类型特征（基于业界惯例）

        Args:
            code: 股票代码
            df: 股票数据DataFrame（用于计算流动性等动态特征）

        Returns:
            dict: 股票类型特征字典
        """
        # 股票类型分类（基于不同股票类型分析框架对比.md）
        stock_type_mapping = {
            # 银行股
            '0005.HK': {'type': 'bank', 'name': '汇丰银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 70, 'risk': 20},
            '0939.HK': {'type': 'bank', 'name': '建设银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 80, 'risk': 20},
            '1288.HK': {'type': 'bank', 'name': '农业银行', 'defensive': 95, 'growth': 25, 'cyclical': 20, 'liquidity': 85, 'risk': 15},
            '1398.HK': {'type': 'bank', 'name': '工商银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20},
            '3968.HK': {'type': 'bank', 'name': '招商银行', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 25},

            # 公用事业股
            '0728.HK': {'type': 'utility', 'name': '中国电信', 'defensive': 90, 'growth': 25, 'cyclical': 15, 'liquidity': 70, 'risk': 20},
            '0941.HK': {'type': 'utility', 'name': '中国移动', 'defensive': 95, 'growth': 30, 'cyclical': 15, 'liquidity': 80, 'risk': 15},

            # 科技股
            '0700.HK': {'type': 'tech', 'name': '腾讯控股', 'defensive': 40, 'growth': 85, 'cyclical': 30, 'liquidity': 90, 'risk': 60},
            '9988.HK': {'type': 'tech', 'name': '阿里巴巴-SW', 'defensive': 35, 'growth': 85, 'cyclical': 35, 'liquidity': 85, 'risk': 65},
            '3690.HK': {'type': 'tech', 'name': '美团-W', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 85, 'risk': 70},
            '1810.HK': {'type': 'tech', 'name': '小米集团-W', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 80, 'risk': 65},

            # 半导体股
            '0981.HK': {'type': 'semiconductor', 'name': '中芯国际', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 75, 'risk': 75},
            '1347.HK': {'type': 'semiconductor', 'name': '华虹半导体', 'defensive': 20, 'growth': 85, 'cyclical': 75, 'liquidity': 70, 'risk': 80},

            # 人工智能股
            '6682.HK': {'type': 'ai', 'name': '第四范式', 'defensive': 20, 'growth': 90, 'cyclical': 50, 'liquidity': 60, 'risk': 85},
            '9660.HK': {'type': 'ai', 'name': '地平线机器人', 'defensive': 15, 'growth': 95, 'cyclical': 60, 'liquidity': 55, 'risk': 90},
            '2533.HK': {'type': 'ai', 'name': '黑芝麻智能', 'defensive': 15, 'growth': 95, 'cyclical': 65, 'liquidity': 50, 'risk': 90},

            # 新能源股
            '1211.HK': {'type': 'new_energy', 'name': '比亚迪股份', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 80, 'risk': 70},
            '1330.HK': {'type': 'environmental', 'name': '绿色动力环保', 'defensive': 25, 'growth': 75, 'cyclical': 80, 'liquidity': 60, 'risk': 80},

            # 能源/周期股
            '0883.HK': {'type': 'energy', 'name': '中国海洋石油', 'defensive': 30, 'growth': 50, 'cyclical': 90, 'liquidity': 75, 'risk': 75},
            '1088.HK': {'type': 'energy', 'name': '中国神华', 'defensive': 40, 'growth': 45, 'cyclical': 85, 'liquidity': 70, 'risk': 70},
            '1138.HK': {'type': 'shipping', 'name': '中远海能', 'defensive': 25, 'growth': 45, 'cyclical': 95, 'liquidity': 65, 'risk': 80},
            '0388.HK': {'type': 'exchange', 'name': '香港交易所', 'defensive': 25, 'growth': 50, 'cyclical': 90, 'liquidity': 70, 'risk': 75},

            # 保险股
            '1299.HK': {'type': 'insurance', 'name': '友邦保险', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 30},

            # 生物医药股
            '2269.HK': {'type': 'biotech', 'name': '药明生物', 'defensive': 30, 'growth': 80, 'cyclical': 55, 'liquidity': 70, 'risk': 70},

            # 房地产股
            '0012.HK': {'type': 'real_estate', 'name': '恒基地产', 'defensive': 20, 'growth': 30, 'cyclical': 95, 'liquidity': 50, 'risk': 85},
            '0016.HK': {'type': 'real_estate', 'name': '新鸿基地产', 'defensive': 25, 'growth': 35, 'cyclical': 90, 'liquidity': 55, 'risk': 80},
            '1109.HK': {'type': 'real_estate', 'name': '华润置地', 'defensive': 30, 'growth': 40, 'cyclical': 85, 'liquidity': 60, 'risk': 75},

            # 指数基金
            '2800.HK': {'type': 'index', 'name': '盈富基金', 'defensive': 80, 'growth': 40, 'cyclical': 30, 'liquidity': 90, 'risk': 25},
        }

        # 获取股票类型信息
        stock_info_mapping = {
            # 银行股
            '0005.HK': {'type': 'bank', 'name': '汇丰银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 70, 'risk': 20},
            '0388.HK': {'type': 'exchange', 'name': '香港交易所', 'defensive': 25, 'growth': 50, 'cyclical': 90, 'liquidity': 70, 'risk': 75},
            '0700.HK': {'type': 'tech', 'name': '腾讯控股', 'defensive': 40, 'growth': 85, 'cyclical': 30, 'liquidity': 90, 'risk': 60},
            '0728.HK': {'type': 'utility', 'name': '中国电信', 'defensive': 90, 'growth': 25, 'cyclical': 15, 'liquidity': 70, 'risk': 20},
            '0883.HK': {'type': 'energy', 'name': '中国海洋石油', 'defensive': 30, 'growth': 50, 'cyclical': 90, 'liquidity': 75, 'risk': 75},
            '0939.HK': {'type': 'bank', 'name': '建设银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 80, 'risk': 20},
            '0941.HK': {'type': 'utility', 'name': '中国移动', 'defensive': 95, 'growth': 30, 'cyclical': 15, 'liquidity': 80, 'risk': 15},
            '0981.HK': {'type': 'semiconductor', 'name': '中芯国际', 'defensive': 25, 'growth': 80, 'cyclical': 70, 'liquidity': 75, 'risk': 75},
            '1088.HK': {'type': 'energy', 'name': '中国神华', 'defensive': 40, 'growth': 45, 'cyclical': 85, 'liquidity': 70, 'risk': 70},
            '1138.HK': {'type': 'shipping', 'name': '中远海能', 'defensive': 25, 'growth': 45, 'cyclical': 95, 'liquidity': 65, 'risk': 80},
            '1211.HK': {'type': 'new_energy', 'name': '比亚迪股份', 'defensive': 30, 'growth': 85, 'cyclical': 60, 'liquidity': 80, 'risk': 70},
            '1288.HK': {'type': 'bank', 'name': '农业银行', 'defensive': 95, 'growth': 25, 'cyclical': 20, 'liquidity': 85, 'risk': 15},
            '1299.HK': {'type': 'insurance', 'name': '友邦保险', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 30},
            '1330.HK': {'type': 'environmental', 'name': '绿色动力环保', 'defensive': 25, 'growth': 75, 'cyclical': 80, 'liquidity': 60, 'risk': 80},
            '1347.HK': {'type': 'semiconductor', 'name': '华虹半导体', 'defensive': 20, 'growth': 85, 'cyclical': 75, 'liquidity': 70, 'risk': 80},
            '1398.HK': {'type': 'bank', 'name': '工商银行', 'defensive': 90, 'growth': 30, 'cyclical': 20, 'liquidity': 85, 'risk': 20},
            '1810.HK': {'type': 'tech', 'name': '小米集团-W', 'defensive': 35, 'growth': 75, 'cyclical': 45, 'liquidity': 80, 'risk': 65},
            '2269.HK': {'type': 'biotech', 'name': '药明生物', 'defensive': 30, 'growth': 80, 'cyclical': 55, 'liquidity': 70, 'risk': 70},
            '2533.HK': {'type': 'ai', 'name': '黑芝麻智能', 'defensive': 15, 'growth': 95, 'cyclical': 65, 'liquidity': 50, 'risk': 90},
            '2800.HK': {'type': 'index', 'name': '盈富基金', 'defensive': 80, 'growth': 40, 'cyclical': 30, 'liquidity': 90, 'risk': 25},
            '3690.HK': {'type': 'tech', 'name': '美团-W', 'defensive': 30, 'growth': 80, 'cyclical': 40, 'liquidity': 85, 'risk': 70},
            '3968.HK': {'type': 'bank', 'name': '招商银行', 'defensive': 85, 'growth': 40, 'cyclical': 25, 'liquidity': 75, 'risk': 25},
            '6682.HK': {'type': 'ai', 'name': '第四范式', 'defensive': 20, 'growth': 90, 'cyclical': 50, 'liquidity': 60, 'risk': 85},
            '9660.HK': {'type': 'ai', 'name': '地平线机器人', 'defensive': 15, 'growth': 95, 'cyclical': 60, 'liquidity': 55, 'risk': 90},
            '9988.HK': {'type': 'tech', 'name': '阿里巴巴-SW', 'defensive': 35, 'growth': 85, 'cyclical': 35, 'liquidity': 85, 'risk': 65},
            # 房地产股
            '0012.HK': {'type': 'real_estate', 'name': '恒基地产', 'defensive': 20, 'growth': 30, 'cyclical': 95, 'liquidity': 50, 'risk': 85},
            '0016.HK': {'type': 'real_estate', 'name': '新鸿基地产', 'defensive': 25, 'growth': 35, 'cyclical': 90, 'liquidity': 55, 'risk': 80},
            '1109.HK': {'type': 'real_estate', 'name': '华润置地', 'defensive': 30, 'growth': 40, 'cyclical': 85, 'liquidity': 60, 'risk': 75},
        }

        # 获取股票类型信息
        stock_info = stock_info_mapping.get(code, None)
        if not stock_info:
            logger.warning(f"未找到股票 {code} 的类型信息")
            return {}

        features = {
            # 股票类型特征（字符串类型）
            'Stock_Type': stock_info['type'],

            # 综合评分特征（基于业界惯例）
            'Stock_Defensive_Score': stock_info['defensive'] / 100.0,  # 防御性评分（0-1）
            'Stock_Growth_Score': stock_info['growth'] / 100.0,          # 成长性评分（0-1）
            'Stock_Cyclical_Score': stock_info['cyclical'] / 100.0,        # 周期性评分（0-1）
            'Stock_Liquidity_Score': stock_info['liquidity'] / 100.0,      # 流动性评分（0-1）
            'Stock_Risk_Score': stock_info['risk'] / 100.0,                # 风险评分（0-1）

            # 衍生特征（基于业界分析权重）
            # 银行股：基本面权重70%，技术分析权重30%
            'Bank_Style_Fundamental_Weight': 0.7 if stock_info['type'] == 'bank' else 0.0,
            'Bank_Style_Technical_Weight': 0.3 if stock_info['type'] == 'bank' else 0.0,

            # 科技股：基本面权重40%，技术分析权重60%
            'Tech_Style_Fundamental_Weight': 0.4 if stock_info['type'] == 'tech' else 0.0,
            'Tech_Style_Technical_Weight': 0.6 if stock_info['type'] == 'tech' else 0.0,

            # 周期股：基本面权重10%，技术分析权重70%，资金流向权重20%
            'Cyclical_Style_Fundamental_Weight': 0.1 if stock_info['type'] in ['energy', 'shipping', 'exchange'] else 0.0,
            'Cyclical_Style_Technical_Weight': 0.7 if stock_info['type'] in ['energy', 'shipping', 'exchange'] else 0.0,
            'Cyclical_Style_Flow_Weight': 0.2 if stock_info['type'] in ['energy', 'shipping', 'exchange'] else 0.0,

            # 房地产股：基本面权重20%，技术分析权重60%，资金流向权重20%
            'RealEstate_Style_Fundamental_Weight': 0.2 if stock_info['type'] == 'real_estate' else 0.0,
            'RealEstate_Style_Technical_Weight': 0.6 if stock_info['type'] == 'real_estate' else 0.0,
            'RealEstate_Style_Flow_Weight': 0.2 if stock_info['type'] == 'real_estate' else 0.0,
        }

        # 动态特征（基于历史数据计算）
        if df is not None and not df.empty and len(df) >= 60:
            # 历史波动率（基于60日数据）
            returns = df['Close'].pct_change().dropna()
            if len(returns) >= 30:
                historical_volatility = returns.rolling(window=30, min_periods=10).std().iloc[-1]
                features['Stock_Historical_Volatility'] = historical_volatility

                # 实际流动性评分（基于成交额波动）
                turnover_volatility = df['Turnover'].rolling(window=20, min_periods=10).std().iloc[-1] / df['Turnover'].rolling(window=20, min_periods=10).mean().iloc[-1]
                features['Stock_Actual_Liquidity_Score'] = max(0, min(1, 1 - turnover_volatility))

                # 价格稳定性评分（基于价格波动）
                price_volatility = df['Close'].rolling(window=20, min_periods=10).std().iloc[-1] / df['Close'].rolling(window=20, min_periods=10).mean().iloc[-1]
                features['Stock_Price_Stability_Score'] = max(0, min(1, 1 - price_volatility))

        return features

    def calculate_multi_period_metrics(self, df):
        """计算多周期指标（趋势和相对强度）"""
        if df.empty or len(df) < 60:
            return df

        periods = [3, 5, 10, 20, 60]

        for period in periods:
            if len(df) < period:
                continue

            # 计算收益率
            return_col = f'Return_{period}d'
            if return_col in df.columns:
                # 计算趋势方向（1=上涨，0=下跌）
                trend_col = f'{period}d_Trend'
                df[trend_col] = (df[return_col] > 0).astype(int)

                # 计算相对强度信号（基于收益率）
                rs_signal_col = f'{period}d_RS_Signal'
                df[rs_signal_col] = (df[return_col] > 0).astype(int)

        # 计算多周期趋势评分
        trend_cols = [f'{p}d_Trend' for p in periods]
        if all(col in df.columns for col in trend_cols):
            df['Multi_Period_Trend_Score'] = df[trend_cols].sum(axis=1)

        # 计算多周期相对强度评分
        rs_cols = [f'{p}d_RS_Signal' for p in periods]
        if all(col in df.columns for col in rs_cols):
            df['Multi_Period_RS_Score'] = df[rs_cols].sum(axis=1)

        return df

    def calculate_relative_strength(self, stock_df, hsi_df):
        """计算相对强度指标（相对于恒生指数）"""
        if stock_df.empty or hsi_df.empty:
            return stock_df

        # 确保索引对齐
        stock_df = stock_df.copy()
        hsi_df = hsi_df.copy()

        # 计算恒生指数收益率
        hsi_df['HSI_Return_1d'] = hsi_df['Close'].pct_change()
        hsi_df['HSI_Return_3d'] = hsi_df['Close'].pct_change(3)
        hsi_df['HSI_Return_5d'] = hsi_df['Close'].pct_change(5)
        hsi_df['HSI_Return_10d'] = hsi_df['Close'].pct_change(10)
        hsi_df['HSI_Return_20d'] = hsi_df['Close'].pct_change(20)
        hsi_df['HSI_Return_60d'] = hsi_df['Close'].pct_change(60)

        # 合并恒生指数数据
        hsi_cols = ['HSI_Return_1d', 'HSI_Return_3d', 'HSI_Return_5d', 'HSI_Return_10d', 'HSI_Return_20d', 'HSI_Return_60d']
        stock_df = stock_df.merge(hsi_df[hsi_cols], left_index=True, right_index=True, how='left')

        # 计算相对强度（RS_ratio = (1+stock_ret)/(1+hsi_ret)-1）
        periods = [1, 3, 5, 10, 20, 60]
        for period in periods:
            stock_ret_col = f'Return_{period}d'
            hsi_ret_col = f'HSI_Return_{period}d'

            if stock_ret_col in stock_df.columns and hsi_ret_col in stock_df.columns:
                # RS_ratio（复合收益比）
                rs_ratio_col = f'RS_Ratio_{period}d'
                stock_df[rs_ratio_col] = (1 + stock_df[stock_ret_col]) / (1 + stock_df[hsi_ret_col]) - 1

                # RS_diff（收益差值）
                rs_diff_col = f'RS_Diff_{period}d'
                stock_df[rs_diff_col] = stock_df[stock_ret_col] - stock_df[hsi_ret_col]

        # 跑赢恒指（基于5日相对强度）
        if 'RS_Ratio_5d' in stock_df.columns:
            stock_df['Outperforms_HSI'] = (stock_df['RS_Ratio_5d'] > 0).astype(int)

        return stock_df

    def create_market_environment_features(self, stock_df, hsi_df, us_market_df=None):
        """创建市场环境特征（包含港股和美股）

        Args:
            stock_df: 股票数据
            hsi_df: 恒生指数数据
            us_market_df: 美股市场数据（可选）
        """
        if stock_df.empty or hsi_df.empty:
            return stock_df

        # 检查是否已经存在 HSI_Return_5d 列（由 calculate_relative_strength 创建）
        if 'HSI_Return_5d' not in stock_df.columns:
            # 如果不存在，则创建并合并
            hsi_df = hsi_df.copy()
            hsi_df['HSI_Return'] = hsi_df['Close'].pct_change()
            hsi_df['HSI_Return_5d'] = hsi_df['Close'].pct_change(5)
            stock_df = stock_df.merge(hsi_df[['HSI_Return', 'HSI_Return_5d']], left_index=True, right_index=True, how='left')

        # 相对表现（相对于恒生指数）
        stock_df['Relative_Return'] = stock_df['Return_5d'] - stock_df['HSI_Return_5d']

        # 如果提供了美股数据，合并美股特征
        if us_market_df is not None and not us_market_df.empty:
            # 美股特征列
            us_features = [
                'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
                'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
                'VIX_Change', 'VIX_Ratio_MA20', 'VIX_Level',
                'US_10Y_Yield', 'US_10Y_Yield_Change'
            ]

            # 只合并存在的特征
            existing_us_features = [f for f in us_features if f in us_market_df.columns]
            if existing_us_features:
                # 对美股特征进行 shift(1)，确保不包含未来信息
                # 因为美股数据比港股晚15小时开盘，所以在预测港股 T+1 日涨跌时，
                # 只能使用 T 日及之前的美股数据
                us_market_df_shifted = us_market_df[existing_us_features].shift(1)

                stock_df = stock_df.merge(
                    us_market_df_shifted,
                    left_index=True, right_index=True, how='left'
                )

        return stock_df

    def create_label(self, df, horizon, for_backtest=False):
        """创建标签：次日涨跌
        
        Args:
            df: 股票数据
            horizon: 预测周期
            for_backtest: 是否为回测准备数据（True时不移除最后horizon行）
        """
        if df.empty or len(df) < horizon + 1:
            return df

        # 计算未来收益率
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1

        # 二分类标签：1=上涨，0=下跌
        df['Label'] = (df['Future_Return'] > 0).astype(int)

        # 如果不是回测模式，移除最后horizon行（没有标签的数据）
        if not for_backtest:
            df = df.iloc[:-horizon]

        return df

    def create_technical_fundamental_interactions(self, df):
        """创建技术指标与基本面的交互特征

        根据业界最佳实践，技术指标与基本面的交互能够捕捉非线性关系，
        提高模型预测准确率。参考：arXiv 2025论文、量化交易最佳实践。

        交互特征列表：
        1. RSI × PE：超卖+低估=强力买入，超买+高估=强力卖出
        2. RSI × PB：超卖+低估值=价值机会
        3. MACD × ROE：趋势向上+高盈利能力=强劲增长
        4. MACD_Hist × ROE：动能增强+盈利能力强=加速上涨
        5. BB_Position × Dividend_Yield：下轨附近+高股息=防守价值
        6. Price_Pct_20d × PE：低位+低估=超跌反弹
        7. Price_Pct_20d × PB：低位+低估值=价值修复
        8. Price_Pct_20d × ROE：低位+高盈利=错杀机会
        9. ATR × PE：高波动+低估=高风险高回报
        10. ATR × ROE：高波动+高盈利=成长潜力
        11. Vol_Ratio × PE：放量+低估=资金流入价值股
        12. OBV_Slope × ROE：资金流入+高盈利=基本面驱动上涨
        13. CMF × Dividend_Yield：资金流入+高股息=防御性买入
        14. Return_5d × PE：短期上涨+低估值=可持续上涨
        15. Return_5d × ROE：短期上涨+高盈利=盈利确认
        """
        if df.empty:
            return df

        # 基本面特征列表（只使用实际可用的）
        fundamental_features = ['PE', 'PB']  # 目前只支持PE和PB

        # 技术指标特征列表（使用实际存在的列名）
        technical_features = ['RSI', 'RSI_ROC', 'MACD', 'MACD_Hist', 'MACD_Hist_ROC',
                             'BB_Position', 'ATR', 'Vol_Ratio', 'CMF',
                             'Return_5d', 'Price_Pct_20d', 'Momentum_5d']

        # 预定义的高价值交互组合（基于业界实践，只使用实际可用的基本面特征）
        high_value_interactions = [
            # 超买超卖与估值的交互
            ('RSI', 'PE'),           # RSI × PE
            ('RSI', 'PB'),           # RSI × PB
            # 趋势与估值的交互
            ('MACD', 'PE'),         # MACD × PE
            ('MACD', 'PB'),         # MACD × PB
            ('MACD_Hist', 'PE'),    # MACD柱状图 × PE
            ('MACD_Hist', 'PB'),    # MACD柱状图 × PB
            # 位置与估值的交互
            ('Price_Pct_20d', 'PE'), # 价格位置 × PE
            ('Price_Pct_20d', 'PB'), # 价格位置 × PB
            # 波动与估值的交互
            ('ATR', 'PE'),           # ATR × PE
            ('ATR', 'PB'),           # ATR × PB
            # 成交量与估值的交互
            ('Vol_Ratio', 'PE'),     # 成交量比率 × PE
            ('Vol_Ratio', 'PB'),     # 成交量比率 × PB
            # 资金流与估值的交互
            ('CMF', 'PE'),           # CMF × PE
            ('CMF', 'PB'),           # CMF × PB
            # 收益与估值的交互
            ('Return_5d', 'PE'),     # 5日收益 × PE
            ('Return_5d', 'PB'),     # 5日收益 × PB
            # 动量与估值的交互
            ('Momentum_5d', 'PE'),   # 5日动量 × PE
            ('Momentum_5d', 'PB'),   # 5日动量 × PB
        ]

        print(f"🔗 生成技术指标与基本面交互特征...")

        interaction_count = 0
        for tech_feat, fund_feat in high_value_interactions:
            if tech_feat in df.columns and fund_feat in df.columns:
                # 交互特征命名：技术_基本面
                interaction_name = f"{tech_feat}_{fund_feat}"
                df[interaction_name] = df[tech_feat] * df[fund_feat]
                interaction_count += 1

        logger.info(f"成功生成 {interaction_count} 个技术指标与基本面交互特征")

        # 删除所有值全为NaN的交互特征（基本面数据不可用导致的）
        interaction_cols = [col for col in df.columns if any(sub in col for sub in ['_PE', '_PB', '_ROE', '_ROA', '_Dividend_Yield', '_EPS', '_Net_Margin', '_Gross_Margin'])]
        cols_to_drop = [col for col in interaction_cols if df[col].isnull().all()]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"🗑️  删除 {len(cols_to_drop)} 个全为NaN的交互特征")

        return df

    def create_sentiment_features(self, code, df):
        """创建情感指标特征（参考 hk_smart_money_tracker.py）

        从新闻数据中计算情感趋势特征：
        - sentiment_ma3: 3日情感移动平均（短期情绪）
        - sentiment_ma7: 7日情感移动平均（中期情绪）
        - sentiment_ma14: 14日情感移动平均（长期情绪）
        - sentiment_volatility: 情感波动率（情绪稳定性）
        - sentiment_change_rate: 情感变化率（情绪变化方向）

        Args:
            code: 股票代码
            df: 股票数据DataFrame（日期索引）

        Returns:
            dict: 包含情感特征的字典
        """
        try:
            # 读取新闻数据
            news_file_path = 'data/all_stock_news_records.csv'
            if not os.path.exists(news_file_path):
                # 没有新闻文件，返回默认值
                return {
                    'sentiment_ma3': 0.0,
                    'sentiment_ma7': 0.0,
                    'sentiment_ma14': 0.0,
                    'sentiment_volatility': 0.0,
                    'sentiment_change_rate': 0.0,
                    'sentiment_days': 0
                }

            # 使用缓存的新闻数据（如果存在）
            if self._news_data_cache is None:
                self._news_data_cache = pd.read_csv(news_file_path)
                # 创建'文本'列（合并标题和内容）
                self._news_data_cache['文本'] = self._news_data_cache['新闻标题'].astype(str) + ' ' + self._news_data_cache['简要内容'].astype(str)
            
            news_df = self._news_data_cache
            if news_df.empty:
                # 新闻文件为空，返回默认值
                return {
                    'sentiment_ma3': 0.0,
                    'sentiment_ma7': 0.0,
                    'sentiment_ma14': 0.0,
                    'sentiment_volatility': 0.0,
                    'sentiment_change_rate': 0.0,
                    'sentiment_days': 0
                }

            # 筛选该股票的新闻
            stock_news = news_df[news_df['股票代码'] == code].copy()
            if stock_news.empty:
                # 该股票没有新闻，返回默认值
                return {
                    'sentiment_ma3': 0.0,
                    'sentiment_ma7': 0.0,
                    'sentiment_ma14': 0.0,
                    'sentiment_volatility': 0.0,
                    'sentiment_change_rate': 0.0,
                    'sentiment_days': 0
                }

            # 转换日期格式
            stock_news['新闻时间'] = pd.to_datetime(stock_news['新闻时间'])

            # 只使用已分析情感分数的新闻
            stock_news = stock_news[stock_news['情感分数'].notna()].copy()
            if stock_news.empty:
                # 没有情感分数数据，返回默认值
                return {
                    'sentiment_ma3': 0.0,
                    'sentiment_ma7': 0.0,
                    'sentiment_ma14': 0.0,
                    'sentiment_volatility': 0.0,
                    'sentiment_change_rate': 0.0,
                    'sentiment_days': 0
                }

            # 确保按日期排序
            stock_news = stock_news.sort_values('新闻时间')

            # 按日期聚合情感分数（使用平均值）
            sentiment_by_date = stock_news.groupby('新闻时间')['情感分数'].mean()

            # 获取实际数据天数
            actual_days = len(sentiment_by_date)

            # 动态调整移动平均窗口
            window_ma3 = min(3, actual_days)
            window_ma7 = min(7, actual_days)
            window_ma14 = min(14, actual_days)
            window_volatility = min(14, actual_days)

            # 计算移动平均
            sentiment_ma3 = sentiment_by_date.rolling(window=window_ma3, min_periods=1).mean().iloc[-1]
            sentiment_ma7 = sentiment_by_date.rolling(window=window_ma7, min_periods=1).mean().iloc[-1]
            sentiment_ma14 = sentiment_by_date.rolling(window=window_ma14, min_periods=1).mean().iloc[-1]

            # 计算波动率
            sentiment_volatility = sentiment_by_date.rolling(window=window_volatility, min_periods=2).std().iloc[-1] if actual_days >= 2 else np.nan

            # 计算变化率
            if actual_days >= 2:
                latest_sentiment = sentiment_by_date.iloc[-1]
                prev_sentiment = sentiment_by_date.iloc[-2]
                sentiment_change_rate = (latest_sentiment - prev_sentiment) / abs(prev_sentiment) if prev_sentiment != 0 else np.nan
            else:
                sentiment_change_rate = np.nan

            return {
                'sentiment_ma3': sentiment_ma3,
                'sentiment_ma7': sentiment_ma7,
                'sentiment_ma14': sentiment_ma14,
                'sentiment_volatility': sentiment_volatility,
                'sentiment_change_rate': sentiment_change_rate,
                'sentiment_days': actual_days
            }

        except Exception as e:
            logger.warning(f"计算情感特征失败 {code}: {e}")
            # 异常情况返回默认值
            return {
                'sentiment_ma3': 0.0,
                'sentiment_ma7': 0.0,
                'sentiment_ma14': 0.0,
                'sentiment_volatility': 0.0,
                'sentiment_change_rate': 0.0,
                'sentiment_days': 0
            }

    def create_topic_features(self, code, df):
        """创建主题分布特征（LDA主题建模）

        从新闻数据中提取主题分布特征：
        - Topic_1 ~ Topic_10: 10个主题的概率分布（0-1之间，总和为1）

        Args:
            code: 股票代码
            df: 股票数据DataFrame（日期索引）

        Returns:
            dict: 包含主题特征的字典
        """
        try:
            from ml_services.topic_modeling import TopicModeler

            # 创建主题建模器
            topic_modeler = TopicModeler(n_topics=10, language='mixed')

            # 尝试加载已训练的模型
            model_path = 'data/lda_topic_model.pkl'

            if os.path.exists(model_path):
                topic_modeler.load_model(model_path)

                # 使用缓存的新闻数据（如果存在）
                if self._news_data_cache is None:
                    self._news_data_cache = topic_modeler.load_news_data(days=self._news_data_days)
                
                # 检查新闻数据是否有效
                if self._news_data_cache is None:
                    logger.warning(f" 新闻数据加载失败（返回None）")
                    return {f'Topic_{i+1}': 0.0 for i in range(10)}
                
                if len(self._news_data_cache) == 0:
                    logger.warning(f" 新闻数据为空")
                    return {f'Topic_{i+1}': 0.0 for i in range(10)}
                
                if '文本' not in self._news_data_cache.columns:
                    logger.warning(f" 新闻数据缺少'文本'列，可用列: {self._news_data_cache.columns.tolist()}")
                    return {f'Topic_{i+1}': 0.0 for i in range(10)}
                
                # 获取股票主题特征
                topic_features = topic_modeler.get_stock_topic_features(code, self._news_data_cache)

                if topic_features:
                    return topic_features
                else:
                    return {f'Topic_{i+1}': 0.0 for i in range(10)}
            else:
                logger.warning(f" 主题模型不存在，请先运行: python ml_services/topic_modeling.py")
                return {f'Topic_{i+1}': 0.0 for i in range(10)}

        except Exception as e:
            import traceback
            logger.error(f"创建主题特征失败 {code}: {e}")
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return {f'Topic_{i+1}': 0.0 for i in range(10)}

    def create_topic_sentiment_interaction_features(self, code, df):
        """创建主题与情感交互特征

        将主题分布与情感评分进行交互，捕捉"某个主题的新闻带有某种情感时"的特定效果：
        - Topic_1 × sentiment_ma3: 主题1与3日移动平均情感的交互
        - Topic_1 × sentiment_ma7: 主题1与7日移动平均情感的交互
        - Topic_1 × sentiment_ma14: 主题1与14日移动平均情感的交互
        - Topic_1 × sentiment_volatility: 主题1与情感波动率的交互
        - Topic_1 × sentiment_change_rate: 主题1与情感变化率的交互
        - ... 共10个主题 × 5个情感指标 = 50个交互特征

        Args:
            code: 股票代码
            df: 股票数据DataFrame（日期索引）

        Returns:
            dict: 包含主题情感交互特征的字典
        """
        try:
            # 获取主题特征
            topic_features = self.create_topic_features(code, df)

            # 获取情感特征
            sentiment_features = self.create_sentiment_features(code, df)

            # 创建交互特征
            interaction_features = {}

            # 情感指标列表
            sentiment_keys = ['sentiment_ma3', 'sentiment_ma7', 'sentiment_ma14',
                            'sentiment_volatility', 'sentiment_change_rate']

            # 为每个主题与每个情感指标创建交互特征
            for topic_idx in range(10):
                topic_key = f'Topic_{topic_idx + 1}'
                topic_prob = topic_features.get(topic_key, 0.0)

                for sentiment_key in sentiment_keys:
                    sentiment_value = sentiment_features.get(sentiment_key, 0.0)

                    # 交互特征 = 主题概率 × 情感值
                    interaction_key = f'{topic_key}_x_{sentiment_key}'
                    interaction_features[interaction_key] = topic_prob * sentiment_value

            if interaction_features:
                logger.info(f"获取主题情感交互特征: {code} (共{len(interaction_features)}个)")
                return interaction_features
            else:
                logger.warning(f" 无法创建主题情感交互特征: {code}")
                return {}

        except Exception as e:
            logger.error(f"创建主题情感交互特征失败 {code}: {e}")
            return {}

    def create_expectation_gap_features(self, code, df):
        """创建预期差距特征

        计算新闻情感相对于市场预期的差距：
        - Sentiment_Gap_MA7: 当前情感与7日移动平均的差距
        - Sentiment_Gap_MA14: 当前情感与14日移动平均的差距
        - Positive_Surprise: 正向意外（情感超过预期的程度）
        - Negative_Surprise: 负向意外（情感低于预期的程度）
        - Expectation_Change_Strength: 预期变化强度

        Args:
            code: 股票代码
            df: 股票数据DataFrame（日期索引）

        Returns:
            dict: 包含预期差距特征的字典
        """
        try:
            # 获取情感特征
            sentiment_features = self.create_sentiment_features(code, df)

            # 创建预期差距特征
            expectation_gap_features = {}

            # 获取当前情感值（使用最新的情感值）
            current_sentiment = sentiment_features.get('sentiment_ma3', 0.0)

            # 计算与不同周期移动平均的差距
            ma7 = sentiment_features.get('sentiment_ma7', 0.0)
            ma14 = sentiment_features.get('sentiment_ma14', 0.0)

            # 预期差距 = 当前情感 - 长期移动平均
            expectation_gap_features['Sentiment_Gap_MA7'] = current_sentiment - ma7
            expectation_gap_features['Sentiment_Gap_MA14'] = current_sentiment - ma14

            # 正向意外（情感超预期，差距为正）
            expectation_gap_features['Positive_Surprise'] = max(0, current_sentiment - ma14)

            # 负向意外（情感不及预期，差距为负，取绝对值）
            expectation_gap_features['Negative_Surprise'] = max(0, ma14 - current_sentiment)

            # 使用情感变化率来衡量预期差距的强度
            sentiment_change_rate = sentiment_features.get('sentiment_change_rate', 0.0)
            expectation_gap_features['Expectation_Change_Strength'] = abs(sentiment_change_rate)

            if expectation_gap_features:
                logger.info(f"获取预期差距特征: {code} (共{len(expectation_gap_features)}个)")
                return expectation_gap_features
            else:
                logger.warning(f" 无法创建预期差距特征: {code}")
                return {}

        except Exception as e:
            logger.error(f"创建预期差距特征失败 {code}: {e}")
            return {}

    def create_sector_features(self, code, df):
        """创建板块分析特征（优化版，使用缓存）

        从板块分析中提取板块涨跌幅、板块排名、板块趋势等特征：
        - sector_avg_change: 板块平均涨跌幅（1日/5日/20日）
        - sector_rank: 板块涨跌幅排名（1日/5日/20日）
        - sector_rising_ratio: 板块上涨股票比例
        - sector_total_volume: 板块总成交量
        - sector_stock_count: 板块股票数量
        - sector_trend: 板块趋势（量化为数值）
        - sector_flow_score: 板块资金流向评分
        - is_sector_leader: 是否为板块龙头
        - sector_best_stock_change: 板块最佳股票涨跌幅
        - sector_worst_stock_change: 板块最差股票涨跌幅

        Args:
            code: 股票代码
            df: 股票数据DataFrame（日期索引）

        Returns:
            dict: 包含板块特征的字典
        """
        try:
            # 获取板块分析器（单例）
            sector_analyzer = self._get_sector_analyzer()
            if sector_analyzer is None:
                # 模块不可用，返回默认值
                return {
                    'sector_avg_change_1d': 0.0,
                    'sector_avg_change_5d': 0.0,
                    'sector_avg_change_20d': 0.0,
                    'sector_rank_1d': 0,
                    'sector_rank_5d': 0,
                    'sector_rank_20d': 0,
                    'sector_rising_ratio_1d': 0.5,
                    'sector_rising_ratio_5d': 0.5,
                    'sector_rising_ratio_20d': 0.5,
                    'sector_total_volume': 0.0,
                    'sector_stock_count': 0,
                    'sector_trend_score': 0.0,
                    'sector_flow_score': 0.0,
                    'is_sector_leader': 0,
                    'sector_best_stock_change': 0.0,
                    'sector_worst_stock_change': 0.0,
                    'sector_outperform_hsi': 0
                }

            # 获取股票所属板块
            sector_info = sector_analyzer.stock_mapping.get(code)
            if not sector_info:
                # 未找到板块信息，返回默认值
                return {
                    'sector_avg_change_1d': 0.0,
                    'sector_avg_change_5d': 0.0,
                    'sector_avg_change_20d': 0.0,
                    'sector_rank_1d': 0,
                    'sector_rank_5d': 0,
                    'sector_rank_20d': 0,
                    'sector_rising_ratio_1d': 0.5,
                    'sector_rising_ratio_5d': 0.5,
                    'sector_rising_ratio_20d': 0.5,
                    'sector_total_volume': 0.0,
                    'sector_stock_count': 0,
                    'sector_trend_score': 0.0,
                    'sector_flow_score': 0.0,
                    'is_sector_leader': 0,
                    'sector_best_stock_change': 0.0,
                    'sector_worst_stock_change': 0.0,
                    'sector_outperform_hsi': 0
                }

            sector_code = sector_info['sector']

            features = {}

            # 计算不同周期的板块表现（使用缓存）
            for period in [1, 5, 20]:
                try:
                    perf_df = self._get_sector_performance(period)

                    if perf_df is not None and not perf_df.empty:
                        # 找到该板块的排名
                        sector_row = perf_df[perf_df['sector_code'] == sector_code]

                        if not sector_row.empty:
                            sector_data = sector_row.iloc[0]

                            # 板块平均涨跌幅
                            features[f'sector_avg_change_{period}d'] = sector_data['avg_change_pct']

                            # 板块排名
                            sector_rank = perf_df[perf_df['sector_code'] == sector_code].index[0] + 1
                            features[f'sector_rank_{period}d'] = sector_rank

                            # 板块上涨股票比例
                            rising_count = sum(1 for s in sector_data['stocks'] if s['change_pct'] > 0)
                            total_count = len(sector_data['stocks'])
                            features[f'sector_rising_ratio_{period}d'] = rising_count / total_count if total_count > 0 else 0.5

                            # 板块总成交量
                            features['sector_total_volume'] = sector_data['total_volume']

                            # 板块股票数量
                            features['sector_stock_count'] = sector_data['stock_count']

                            # 最佳和最差股票表现
                            if sector_data['best_stock']:
                                features['sector_best_stock_change'] = sector_data['best_stock']['change_pct']
                            if sector_data['worst_stock']:
                                features['sector_worst_stock_change'] = sector_data['worst_stock']['change_pct']

                            # 是否为板块龙头（前3名）
                            features['is_sector_leader'] = 1 if sector_rank <= 3 else 0
                        else:
                            # 板块未找到，使用默认值
                            features[f'sector_avg_change_{period}d'] = 0.0
                            features[f'sector_rank_{period}d'] = 0
                            features[f'sector_rising_ratio_{period}d'] = 0.5
                    else:
                        # 无法获取板块数据，使用默认值
                        features[f'sector_avg_change_{period}d'] = 0.0
                        features[f'sector_rank_{period}d'] = 0
                        features[f'sector_rising_ratio_{period}d'] = 0.5

                except Exception as e:
                    logger.warning(f"计算板块表现失败 (period={period}): {e}")
                    features[f'sector_avg_change_{period}d'] = 0.0
                    features[f'sector_rank_{period}d'] = 0
                    features[f'sector_rising_ratio_{period}d'] = 0.5

            # 计算板块趋势
            try:
                trend_result = sector_analyzer.analyze_sector_trend(sector_code, days=20)

                if 'trend' in trend_result:
                    # 将趋势量化为数值
                    trend_mapping = {
                        '强势上涨': 2.0,
                        '温和上涨': 1.0,
                        '震荡整理': 0.0,
                        '温和下跌': -1.0,
                        '强势下跌': -2.0
                    }
                    features['sector_trend_score'] = trend_mapping.get(trend_result['trend'], 0.0)
                else:
                    features['sector_trend_score'] = 0.0
            except Exception as e:
                logger.warning(f"计算板块趋势失败: {e}")
                features['sector_trend_score'] = 0.0

            # 计算板块资金流向
            try:
                flow_result = sector_analyzer.analyze_sector_fund_flow(sector_code, days=5)

                if 'avg_flow_score' in flow_result:
                    features['sector_flow_score'] = flow_result['avg_flow_score']
                else:
                    features['sector_flow_score'] = 0.0
            except Exception as e:
                logger.warning(f"计算板块资金流向失败: {e}")
                features['sector_flow_score'] = 0.0

            # 判断板块是否跑赢恒指（基于板块平均涨跌幅）
            if 'sector_avg_change_1d' in features and 'sector_avg_change_5d' in features:
                # 简化处理：假设恒指涨跌幅为0（实际应该从恒指数据中获取）
                # 这里使用板块自身的涨跌幅作为参考
                features['sector_outperform_hsi'] = 1 if features['sector_avg_change_5d'] > 0 else 0

            return features

        except Exception as e:
            logger.warning(f"计算板块特征失败 {code}: {e}")
            # 异常情况返回默认值
            return {
                'sector_avg_change_1d': 0.0,
                'sector_avg_change_5d': 0.0,
                'sector_avg_change_20d': 0.0,
                'sector_rank_1d': 0,
                'sector_rank_5d': 0,
                'sector_rank_20d': 0,
                'sector_rising_ratio_1d': 0.5,
                'sector_rising_ratio_5d': 0.5,
                'sector_rising_ratio_20d': 0.5,
                'sector_total_volume': 0.0,
                'sector_stock_count': 0,
                'sector_trend_score': 0.0,
                'sector_flow_score': 0.0,
                'is_sector_leader': 0,
                'sector_best_stock_change': 0.0,
                'sector_worst_stock_change': 0.0,
                'sector_outperform_hsi': 0
            }

    def create_interaction_features(self, df):
        """创建所有可能的交叉特征（类别型 × 数值型）

        生成策略：将所有类别型特征（13个）与所有数值型特征（90个）进行交叉，
        形成 1170 个交叉特征。GBDT+LR 算法会自动过滤无用特征。
        """
        if df.empty:
            return df

        # 类别型特征（13个）
        categorical_features = [
            'Outperforms_HSI',
            'Strong_Volume_Up',
            'Weak_Volume_Down',
            '3d_Trend', '5d_Trend', '10d_Trend', '20d_Trend', '60d_Trend',
            '3d_RS_Signal', '5d_RS_Signal', '10d_RS_Signal', '20d_RS_Signal', '60d_RS_Signal'
        ]

        # 数值型特征（排除类别型特征、标签和原始价格数据）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Returns', 'TP', 'MF_Multiplier', 'MF_Volume',
                          'High_Max', 'Low_Min'] + categorical_features

        numeric_features = [col for col in df.columns if col not in exclude_columns]

        print(f"生成交叉特征: {len(categorical_features)} 个类别 × {len(numeric_features)} 个数值 = {len(categorical_features) * len(numeric_features)} 个交叉特征")

        # 生成所有交叉特征
        interaction_count = 0
        for cat_feat in categorical_features:
            if cat_feat not in df.columns:
                continue

            for num_feat in numeric_features:
                if num_feat not in df.columns:
                    continue

                # 交叉特征命名：类别_数值
                interaction_name = f"{cat_feat}_{num_feat}"
                df[interaction_name] = df[cat_feat] * df[num_feat]
                interaction_count += 1

        logger.info(f"成功生成 {interaction_count} 个交叉特征")
        return df


class BaseTradingModel:
    """交易模型基类 - 提供公共方法和属性"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.processor = BaseModelProcessor()
        self.feature_columns = []
        self.horizon = 1  # 默认预测周期
        self.model_type = None  # 子类必须设置
        self.categorical_encoders = {}

    def load_selected_features(self, filepath=None, current_feature_names=None):
        """加载选择的特征列表（使用特征名称交集，确保特征存在）

        Args:
            filepath: 特征名称文件路径（可选，默认使用最新的）
            current_feature_names: 当前数据集的特征名称列表（可选）

        Returns:
            list: 特征名称列表（如果找到），否则返回None
        """
        import os
        import glob

        if filepath is None:
            # 查找最新的特征名称文件
            # 优先查找 model_importance_selected_*.csv（模型重要性法）
            pattern = 'output/model_importance_selected_*.csv'
            files = glob.glob(pattern)
            if not files:
                # 回退到 selected_features_*.csv（统计方法）
                pattern = 'output/selected_features_*.csv'
                files = glob.glob(pattern)
            
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            import pandas as pd
            # 读取特征名称
            df = pd.read_csv(filepath)
            selected_names = df['Feature_Name'].tolist()

            logger.debug(f"加载特征列表文件: {filepath}")
            logger.info(f"加载了 {len(selected_names)} 个选择的特征")

            # 如果提供了当前特征名称，使用交集
            if current_feature_names is not None:
                current_set = set(current_feature_names)
                selected_set = set(selected_names)
                available_set = current_set & selected_set
                
                available_names = list(available_set)
                logger.info(f"当前数据集特征数量: {len(current_feature_names)}")
                logger.info(f"选择的特征数量: {len(selected_names)}")
                logger.info(f"实际可用的特征数量: {len(available_names)}")
                logger.warning(f" {len(selected_set) - len(available_names)} 个特征在当前数据集中不存在")
                
                return available_names
            else:
                return selected_names

        except Exception as e:
            logger.warning(f"加载特征列表失败: {e}")
            return None

    def get_feature_columns(self, df):
        """获取特征列（排除中间计算列）"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns


class LightGBMModel(BaseTradingModel):
    """LightGBM 模型 - 基于 LightGBM 梯度提升算法的单一模型"""

    def __init__(self):
        super().__init__()  # 调用基类初始化
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = 'lgbm'  # 模型类型标识

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False):
        """准备训练数据（80个指标版本，优化版）
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
        """
        self.horizon = horizon
        all_data = []

        # ========== 步骤1：获取共享数据（只获取一次） ==========
        logger.info("获取共享数据...")
        
        # 获取美股市场数据（只获取一次）
        us_market_df = us_market_data.get_all_us_market_data(period_days=730)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

        # 获取恒生指数数据（只获取一次，所有股票共享）
        hsi_df = get_hsi_data_with_cache(period_days=730)
        if hsi_df is None or hsi_df.empty:
            raise ValueError("无法获取恒生指数数据")

        # ========== 步骤2：并行下载股票数据 ==========
        print(f"\n🚀 并行下载 {len(codes)} 只股票数据...")
        
        def fetch_single_stock_data(code):
            """获取单只股票数据"""
            try:
                stock_code = code.replace('.HK', '')
                stock_df = get_stock_data_with_cache(stock_code, period_days=730)
                if stock_df is not None and not stock_df.empty:
                    return (code, stock_df)
                return None
            except Exception as e:
                logger.warning(f"下载股票 {code} 失败: {e}")
                return None

        # 使用线程池并行下载（最多8个并发）
        stock_data_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_code = {executor.submit(fetch_single_stock_data, code): code for code in codes}
            
            for i, future in enumerate(as_completed(future_to_code), 1):
                result = future.result()
                if result is not None:
                    stock_data_list.append(result)
                    print(f"  ✅ [{i}/{len(codes)}] {result[0]}")

        logger.info(f"成功下载 {len(stock_data_list)} 只股票数据")

        # ========== 步骤3：计算特征 ==========
        print(f"\n🔧 计算特征...")
        
        for i, (code, stock_df) in enumerate(stock_data_list, 1):
            try:
                print(f"  [{i}/{len(stock_data_list)}] 处理股票: {code}")

                # 计算技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 计算多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 计算相对强度指标（使用共享的恒生指数数据）
                stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest)

                # 添加基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 添加股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value

                # 添加情感特征
                sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
                for key, value in sentiment_features.items():
                    stock_df[key] = value

                # 添加主题特征（LDA主题建模）
                topic_features = self.feature_engineer.create_topic_features(code, stock_df)
                for key, value in topic_features.items():
                    stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

                # 添加板块特征
                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                # 添加股票代码
                stock_df['Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                print(f"处理股票 {code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError("没有获取到任何数据")

        # 合并所有数据（保留日期索引，不重置索引）
        df = pd.concat(all_data, ignore_index=False)

        # 按日期索引排序，确保时间顺序正确
        df = df.sort_index()

        # 生成技术指标与基本面交互特征（先执行，因为这是高价值特征）
        print("\n🔗 生成技术指标与基本面交互特征...")
        df = self.feature_engineer.create_technical_fundamental_interactions(df)

        # 生成交叉特征（类别型 × 数值型）
        print("\n🔗 生成交叉特征...")
        df = self.feature_engineer.create_interaction_features(df)

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        """训练模型

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（默认False，使用全部特征）
        """
        print("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon)

        # 先删除全为NaN的列（避免dropna删除所有行）
        cols_all_nan = df.columns[df.isnull().all()].tolist()
        if cols_all_nan:
            print(f"🗑️  删除 {len(cols_all_nan)} 个全为NaN的列")
            df = df.drop(columns=cols_all_nan)

        # 删除包含NaN的行
        df = df.dropna()

        # 确保数据按日期索引排序（dropna 可能会改变顺序）
        df = df.sort_index()

        if len(df) < 100:
            raise ValueError(f"数据量不足，只有 {len(df)} 条记录")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        print(f"使用 {len(self.feature_columns)} 个特征")

        # 应用特征选择（可选）
        if use_feature_selection:
            print("\n🎯 应用特征选择（LightGBM）...")
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)
            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                logger.info(f"特征选择应用完成：使用 {len(self.feature_columns)} 个特征")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")

        # 处理分类特征（将字符串转换为整数编码）
        categorical_features = []
        self.categorical_encoders = {}  # 存储编码器，用于预测时解码

        for col in self.feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                print(f"  编码分类特征: {col}")
                categorical_features.append(col)
                # 使用LabelEncoder进行编码
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.categorical_encoders[col] = le

        # 准备特征和标签
        X = df[self.feature_columns].values
        y = df['Label'].values

        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)

        # 根据预测周期调整正则化参数（分周期优化策略）
        # 次日模型：最强的正则化防止过拟合
        # 一周模型：适度正则化保持学习能力
        # 一个月模型：增强正则化（特征数量多，需要更强的正则化）
        if horizon == 1:
            # 次日模型参数（最强正则化）
            print("使用次日模型参数（强正则化）...")
            lgb_params = {
                'n_estimators': 40,           # 减少树数量（50→40）
                'learning_rate': 0.02,         # 降低学习率（0.03→0.02）
                'max_depth': 3,                # 降低深度（4→3）
                'num_leaves': 12,              # 减少叶子节点（15→12）
                'min_child_samples': 40,       # 增加最小样本（30→40）
                'subsample': 0.65,             # 减少行采样（0.7→0.65）
                'colsample_bytree': 0.65,      # 减少列采样（0.7→0.65）
                'reg_alpha': 0.2,              # 增强L1正则（0.1→0.2）
                'reg_lambda': 0.2,             # 增强L2正则（0.1→0.2）
                'min_split_gain': 0.15,        # 增加分割增益（0.1→0.15）
                'feature_fraction': 0.65,      # 减少特征采样（0.7→0.65）
                'bagging_fraction': 0.65,      # 减少Bagging采样（0.7→0.65）
                'bagging_freq': 5,
                'random_state': 42,
                'verbose': -1
            }
        elif horizon == 5:
            # 一周模型参数（适度正则化）
            print("使用5天模型参数（适度正则化）...")
            lgb_params = {
                'n_estimators': 50,           # 保持50
                'learning_rate': 0.03,         # 保持0.03
                'max_depth': 4,                # 保持4
                'num_leaves': 15,              # 保持15
                'min_child_samples': 30,       # 保持30
                'subsample': 0.7,              # 保持0.7
                'colsample_bytree': 0.7,       # 保持0.7
                'reg_alpha': 0.1,              # 保持0.1
                'reg_lambda': 0.1,             # 保持0.1
                'min_split_gain': 0.1,         # 保持0.1
                'feature_fraction': 0.7,       # 保持0.7
                'bagging_fraction': 0.7,       # 保持0.7
                'bagging_freq': 5,
                'random_state': 42,
                'verbose': -1
            }
        else:  # horizon == 20
            # 一个月模型参数（超增强正则化 - 2026-02-16优化）
            # 原因：特征数量从2530增至2936（+16%），需要更强的正则化防止过拟合
            # 优化目标：将训练/验证差距从±7.07%降至<5%
            print("使用20天模型参数（超增强正则化，降低过拟合）...")
            lgb_params = {
                'n_estimators': 40,           # 进一步减少树数量（45→40）
                'learning_rate': 0.02,         # 进一步降低学习率（0.025→0.02）
                'max_depth': 3,                # 降低深度（4→3）减少过拟合
                'num_leaves': 11,              # 进一步减少叶子节点（13→11）
                'min_child_samples': 40,       # 进一步增加最小样本（35→40）
                'subsample': 0.6,              # 进一步减少行采样（0.65→0.6）
                'colsample_bytree': 0.6,       # 进一步减少列采样（0.65→0.6）
                'reg_alpha': 0.25,             # 超增强L1正则（0.18→0.25）
                'reg_lambda': 0.25,            # 超增强L2正则（0.18→0.25）
                'min_split_gain': 0.15,        # 进一步增加分割增益（0.12→0.15）
                'feature_fraction': 0.6,       # 进一步减少特征采样（0.65→0.6）
                'bagging_fraction': 0.6,       # 进一步减少Bagging采样（0.65→0.6）
                'bagging_freq': 5,
                'random_state': 42,
                'verbose': -1
            }

        # 训练模型（增加正则化以减少过拟合）
        print("训练LightGBM模型...")
        self.model = lgb.LGBMClassifier(**lgb_params)

        # 使用时间序列交叉验证
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 添加early_stopping以减少过拟合
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=15, verbose=False)  # 增加patience（10→15）
                ]
            )
            y_pred = self.model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
            print(f"验证准确率: {score:.4f}")

        # 使用全部数据重新训练
        self.model.fit(X, y)

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        print(f"\n平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'lgbm',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        import json
        accuracy_file = 'data/model_accuracy.json'
        try:
            # 读取现有数据
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            # 更新当前模型的准确率
            key = f'lgbm_{horizon}d'
            existing_data[key] = accuracy_info
            
            # 保存回文件
            with open(accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(f"准确率已保存到 {accuracy_file}")
        except Exception as e:
            logger.warning(f"保存准确率失败: {e}")

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
            logger.warning(f"特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        print("\n特征重要性 Top 10:")
        print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(10))

        return feat_imp

    def predict(self, code, predict_date=None, horizon=None):
        """预测单只股票（80个指标版本）

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
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算技术指标（80个指标）
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)

            # 计算多周期指标
            stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

            # 计算相对强度指标
            stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

            # 创建资金流向特征
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)

            # 创建市场环境特征（包含港股和美股）
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

            # 添加基本面特征
            fundamental_features = self.feature_engineer.create_fundamental_features(code)
            for key, value in fundamental_features.items():
                stock_df[key] = value

            # 添加股票类型特征
            stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
            for key, value in stock_type_features.items():
                stock_df[key] = value

            # 添加情感特征
            sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
            for key, value in sentiment_features.items():
                stock_df[key] = value

            # 添加主题特征（LDA主题建模）
            topic_features = self.feature_engineer.create_topic_features(code, stock_df)
            for key, value in topic_features.items():
                stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

            # 添加板块特征
            sector_features = self.feature_engineer.create_sector_features(code, stock_df)
            for key, value in sector_features.items():
                stock_df[key] = value

            # 生成技术指标与基本面交互特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

            # 生成交叉特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_interaction_features(stock_df)

            # 获取最新数据（或指定日期的数据）
            latest_data = stock_df.iloc[-1:]

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            # 处理分类特征（使用训练时的编码器）
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    # 如果遇到训练时未见过的类别，映射到0
                    try:
                        latest_data[col] = encoder.transform(latest_data[col].astype(str))
                    except ValueError:
                        # 处理未见过的类别
                        logger.warning(f"警告: 分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

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
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders
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
        self.categorical_encoders = model_data.get('categorical_encoders', {})
        print(f"模型已从 {filepath} 加载")


class GBDTModel(BaseTradingModel):
    """GBDT 模型 - 基于梯度提升决策树的单一模型"""

    def __init__(self):
        super().__init__()  # 调用基类初始化
        self.gbdt_model = None
        self.actual_n_estimators = 0
        self.model_type = 'gbdt'  # 模型类型标识

    def load_selected_features(self, filepath=None, current_feature_names=None):
        """加载选择的特征列表（使用特征名称交集，确保特征存在）

        Args:
            filepath: 特征名称文件路径（可选，默认使用最新的）
            current_feature_names: 当前数据集的特征名称列表（可选）

        Returns:
            list: 特征名称列表（如果找到），否则返回None
        """
        import os
        import glob

        if filepath is None:
            # 查找最新的特征名称文件
            pattern = 'output/selected_features_*.csv'
            files = glob.glob(pattern)
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            import pandas as pd
            # 读取特征名称
            df = pd.read_csv(filepath)
            selected_names = df['Feature_Name'].tolist()

            logger.debug(f"加载特征列表文件: {filepath}")
            logger.info(f"加载了 {len(selected_names)} 个选择的特征")

            # 如果提供了当前特征名称，使用交集
            if current_feature_names is not None:
                current_set = set(current_feature_names)
                selected_set = set(selected_names)
                available_set = current_set & selected_set
                
                available_names = list(available_set)
                logger.info(f"当前数据集特征数量: {len(current_feature_names)}")
                logger.info(f"选择的特征数量: {len(selected_names)}")
                logger.info(f"实际可用的特征数量: {len(available_names)}")
                logger.warning(f" {len(selected_set) - len(available_set)} 个特征在当前数据集中不存在")
                
                return available_names
            else:
                return selected_names

        except Exception as e:
            logger.warning(f"加载特征列表失败: {e}")
            return None

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False):
        """准备训练数据（80个指标版本）
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
        """
        self.horizon = horizon
        all_data = []

        # 获取美股市场数据（只获取一次）
        logger.info("获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=730)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

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

                # 计算技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 计算多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 计算相对强度指标
                stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                
                # 添加股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest)

                # 添加基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 添加股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value

                # 添加情感特征
                sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
                for key, value in sentiment_features.items():
                    stock_df[key] = value

                # 添加主题特征（LDA主题建模）
                topic_features = self.feature_engineer.create_topic_features(code, stock_df)
                for key, value in topic_features.items():
                    stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

                # 添加板块特征
                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                # 添加股票代码
                stock_df['Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                print(f"处理股票 {code} 失败: {e}")
                continue

        if not all_data:
            raise ValueError("没有获取到任何数据")

        # 合并所有数据（保留日期索引，不重置索引）
        df = pd.concat(all_data, ignore_index=False)

        # 按日期索引排序，确保时间顺序正确
        df = df.sort_index()

        # 生成技术指标与基本面交互特征（先执行，因为这是高价值特征）
        print("\n🔗 生成技术指标与基本面交互特征...")
        df = self.feature_engineer.create_technical_fundamental_interactions(df)

        # 生成交叉特征（类别型 × 数值型）
        print("\n🔗 生成交叉特征...")
        df = self.feature_engineer.create_interaction_features(df)

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        """训练 GBDT 模型

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（默认False，使用全部特征）
        """
        print("="*70)
        logger.info("开始训练 GBDT 模型")
        print("="*70)

        # 准备数据
        logger.info("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon)

        # 先删除全为NaN的列（避免dropna删除所有行）
        cols_all_nan = df.columns[df.isnull().all()].tolist()
        if cols_all_nan:
            print(f"🗑️  删除 {len(cols_all_nan)} 个全为NaN的列")
            df = df.drop(columns=cols_all_nan)

        # 删除包含NaN的行
        df = df.dropna()

        # 确保数据按日期索引排序（dropna 可能会改变顺序）
        df = df.sort_index()

        if len(df) < 100:
            raise ValueError(f"数据量不足，只有 {len(df)} 条记录")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"使用 {len(self.feature_columns)} 个特征")

        # 应用特征选择（可选）
        if use_feature_selection:
            print("\n🎯 应用特征选择（GBDT）...")
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)
            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                logger.info(f"特征选择应用完成：使用 {len(self.feature_columns)} 个特征")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")
        else:
            logger.info(f"使用全部 {len(self.feature_columns)} 个特征")

        # 处理分类特征（将字符串转换为整数编码）
        categorical_features = []
        self.categorical_encoders = {}  # 存储编码器，用于预测时解码

        for col in self.feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                print(f"  编码分类特征: {col}")
                categorical_features.append(col)
                # 使用LabelEncoder进行编码
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.categorical_encoders[col] = le

        # 准备特征和标签
        X = df[self.feature_columns].values
        y = df['Label'].values

        # 创建输出目录
        os.makedirs('output', exist_ok=True)

        # ========== 训练 GBDT 模型 ==========
        print("\n" + "="*70)
        print("🌲 训练 GBDT 模型")
        print("="*70)

        # 根据预测周期调整叶子节点数量和早停耐心
        # 次日模型：适度参数
        # 一周模型：减少叶子节点数量以防止过拟合，增加早停耐心
        # 一个月模型：增强正则化（特征数量增加，需要更强的正则化）
        if horizon == 5:
            # 一周模型参数（防过拟合）
            print("使用一周模型参数（减少叶子节点，增加早停耐心）...")
            n_estimators = 32
            num_leaves = 24  # 减少叶子节点（32→24）
            stopping_rounds = 15  # 增加早停耐心（10→15）
            min_child_samples = 30  # 增加最小样本（20→30）
            reg_alpha = 0.1     # 保持0.1
            reg_lambda = 0.1    # 保持0.1
            subsample = 0.7     # 保持0.7
            colsample_bytree = 0.6  # 保持0.6
        elif horizon == 1:
            # 次日模型参数（适度）
            print("使用次日模型参数...")
            n_estimators = 32
            num_leaves = 28  # 适度减少（32→28）
            stopping_rounds = 12  # 适度增加
            min_child_samples = 25
            reg_alpha = 0.15    # 增强L1正则（0.1→0.15）
            reg_lambda = 0.15   # 增强L2正则（0.1→0.15）
            subsample = 0.65    # 减少行采样（0.7→0.65）
            colsample_bytree = 0.65  # 减少列采样（0.6→0.65）
        else:  # horizon == 20
            # 一个月模型参数（超增强正则化 - 2026-02-16优化）
            # 原因：特征数量从2530增至2936（+16%），需要更强的正则化防止过拟合
            # 优化目标：将训练/验证差距从±7.07%降至<5%
            print("使用20天模型参数（超增强正则化，降低过拟合）...")
            n_estimators = 28           # 进一步减少树数量（32→28）
            num_leaves = 20              # 进一步减少叶子节点（24→20）
            stopping_rounds = 18         # 进一步增加早停耐心（12→18）
            min_child_samples = 35       # 进一步增加最小样本（30→35）
            reg_alpha = 0.22             # 增强L1正则（0.15→0.22）
            reg_lambda = 0.22            # 增强L2正则（0.15→0.22）
            subsample = 0.6              # 进一步减少行采样（0.65→0.6）
            colsample_bytree = 0.6       # 进一步减少列采样（0.65→0.6）

        self.gbdt_model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            subsample=subsample,            # 根据周期调整
            min_child_weight=0.1,
            min_child_samples=min_child_samples,  # 根据周期调整
            colsample_bytree=colsample_bytree,  # 根据周期调整
            num_leaves=num_leaves,      # 根据周期调整
            learning_rate=0.025,        # 进一步降低学习率（0.03→0.025）
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,        # 根据周期调整L1正则
            reg_lambda=reg_lambda,       # 根据周期调整L2正则
            min_split_gain=0.12,        # 进一步增加分割增益（0.1→0.12）
            feature_fraction=0.6,       # 进一步减少特征采样（0.7→0.6）
            bagging_fraction=0.6,       # 进一步减少Bagging采样（0.7→0.6）
            bagging_freq=5,             # Bagging频率（新增）
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
                    lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)  # 根据周期调整早停耐心
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
        mean_accuracy = np.mean(gbdt_scores)
        std_accuracy = np.std(gbdt_scores)
        print(f"\n✅ GBDT 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators} (原计划: {n_estimators})")
        print(f"   平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'gbdt',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        import json
        accuracy_file = 'data/model_accuracy.json'
        try:
            # 读取现有数据
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            # 更新当前模型的准确率
            key = f'gbdt_{horizon}d'
            existing_data[key] = accuracy_info
            
            # 保存回文件
            with open(accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(f"准确率已保存到 {accuracy_file}")
        except Exception as e:
            logger.warning(f"保存准确率失败: {e}")

        # ========== Step 2: 输出 GBDT 特征重要性 ==========
        print("\n" + "="*70)
        logger.info("Step 2: 分析 GBDT 特征重要性")
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
            logger.info(r"已保存特征重要性至 output/gbdt_feature_importance.csv")

            # 显示前20个重要特征
            print("\n📊 GBDT Top 20 重要特征 (含影响方向):")
            print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(20))

        except Exception as e:
            logger.warning(f"特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        print("\n" + "="*70)
        logger.info(r"GBDT 模型训练完成！")
        print("="*70)

        return feat_imp

    def predict(self, code, predict_date=None, horizon=None):
        """预测单只股票（80个指标版本）

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
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算技术指标（80个指标）
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)

            # 计算多周期指标
            stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

            # 计算相对强度指标
            stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

            # 创建资金流向特征
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)

            # 创建市场环境特征（包含港股和美股）
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

            # 添加基本面特征
            fundamental_features = self.feature_engineer.create_fundamental_features(code)
            for key, value in fundamental_features.items():
                stock_df[key] = value

            # 添加股票类型特征
            stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
            for key, value in stock_type_features.items():
                stock_df[key] = value

            # 添加情感特征
            sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
            for key, value in sentiment_features.items():
                stock_df[key] = value

            # 添加主题特征（LDA主题建模）
            topic_features = self.feature_engineer.create_topic_features(code, stock_df)
            for key, value in topic_features.items():
                stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

            # 添加板块特征
            sector_features = self.feature_engineer.create_sector_features(code, stock_df)
            for key, value in sector_features.items():
                stock_df[key] = value

            # 生成技术指标与基本面交互特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

            # 生成交叉特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_interaction_features(stock_df)

            # 获取最新数据
            latest_data = stock_df.iloc[-1:]

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            # 处理分类特征（使用训练时的编码器）
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    # 如果遇到训练时未见过的类别，映射到0
                    try:
                        latest_data[col] = encoder.transform(latest_data[col].astype(str))
                    except ValueError:
                        # 处理未见过的类别
                        logger.warning(f"警告: 分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

            X = latest_data[self.feature_columns].values

            # 使用GBDT模型直接预测
            proba = self.gbdt_model.predict_proba(X)[0]
            prediction = self.gbdt_model.predict(X)[0]

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

        except Exception as e:
            print(f"预测失败 {code}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'gbdt_model': self.gbdt_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'categorical_encoders': self.categorical_encoders
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"GBDT 模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.gbdt_model = model_data['gbdt_model']
        self.feature_columns = model_data['feature_columns']
        self.actual_n_estimators = model_data['actual_n_estimators']
        self.categorical_encoders = model_data.get('categorical_encoders', {})
        print(f"GBDT 模型已从 {filepath} 加载")


class CatBoostModel(BaseTradingModel):
    """CatBoost 模型 - 基于 CatBoost 梯度提升算法的单一模型
    
    CatBoost 是 Yandex 开发的梯度提升库，具有以下优势：
    1. 自动处理分类特征，无需手动编码
    2. 更好的默认参数，减少调参工作量
    3. 更快的训练速度（GPU 支持）
    4. 更好的泛化能力，减少过拟合
    """

    def __init__(self):
        super().__init__()  # 调用基类初始化
        self.catboost_model = None
        self.actual_n_estimators = 0
        self.model_type = 'catboost'  # 模型类型标识

    def load_selected_features(self, filepath=None, current_feature_names=None):
        """加载选择的特征列表（使用特征名称交集，确保特征存在）

        Args:
            filepath: 特征名称文件路径（可选，默认使用最新的）
            current_feature_names: 当前数据集的特征名称列表（可选）

        Returns:
            list: 特征名称列表（如果找到），否则返回None
        """
        import os
        import glob

        if filepath is None:
            # 查找最新的特征名称文件
            pattern = 'output/selected_features_*.csv'
            files = glob.glob(pattern)
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            import pandas as pd
            # 读取特征名称
            df = pd.read_csv(filepath)
            selected_names = df['Feature_Name'].tolist()

            logger.debug(f"加载特征列表文件: {filepath}")
            logger.info(f"加载了 {len(selected_names)} 个选择的特征")

            # 如果提供了当前特征名称，使用交集
            if current_feature_names is not None:
                current_set = set(current_feature_names)
                selected_set = set(selected_names)
                available_set = current_set & selected_set
                
                available_names = list(available_set)
                logger.info(f"当前数据集特征数量: {len(current_feature_names)}")
                logger.info(f"选择的特征数量: {len(selected_names)}")
                logger.info(f"实际可用的特征数量: {len(available_names)}")
                logger.warning(f" {len(selected_set) - len(available_set)} 个特征在当前数据集中不存在")
                
                return available_names
            else:
                return selected_names

        except Exception as e:
            logger.warning(f"加载特征列表失败: {e}")
            return None

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False):
        """准备训练数据
        
        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
        """
        self.horizon = horizon
        all_data = []

        # 获取美股市场数据（只获取一次）
        logger.info("获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=730)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

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

                # 计算技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 计算多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 计算相对强度指标
                stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest)

                # 添加基本面特征
                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                # 添加股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value

                # 添加情感特征
                sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
                for key, value in sentiment_features.items():
                    stock_df[key] = value

                # 添加主题特征（LDA主题建模）
                topic_features = self.feature_engineer.create_topic_features(code, stock_df)
                for key, value in topic_features.items():
                    stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

                # 添加板块特征
                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                # 生成技术指标与基本面交互特征（与训练时保持一致）
                stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

                # 生成交叉特征（与训练时保持一致）
                stock_df = self.feature_engineer.create_interaction_features(stock_df)

                # 添加股票代码
                stock_df['Code'] = code

                all_data.append(stock_df)

            except Exception as e:
                logger.warning(f"处理股票 {code} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_data) == 0:
            raise ValueError("没有可用的数据")

        # 合并所有数据
        df = pd.concat(all_data, ignore_index=False)

        # 转换索引为 datetime
        df.index = pd.to_datetime(df.index)

        # 过滤日期范围（如果指定）
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        logger.info(f"数据准备完成，共 {len(df)} 条记录")

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        """训练 CatBoost 模型

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（只使用500个选择的特征）

        Returns:
            DataFrame: 特征重要性数据
        """
        print("\n" + "="*70)
        logger.info("开始训练 CatBoost 模型")
        print("="*70)
        print(f"预测周期: {horizon} 天")
        print(f"股票数量: {len(codes)}")
        print(f"特征选择: {'是' if use_feature_selection else '否'}")

        # ========== 准备数据 ==========
        print("\n" + "="*70)
        logger.info("准备训练数据")
        print("="*70)

        df = self.prepare_data(codes, start_date, end_date, horizon)

        # 删除包含 NaN 的行
        df = df.dropna(subset=['Label'])
        print(f"删除 NaN 后: {len(df)} 条记录")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"使用 {len(self.feature_columns)} 个特征")

        # ========== 特征选择（可选）==========
        if use_feature_selection:
            print("\n" + "="*70)
            print("🔍 应用特征选择")
            print("="*70)

            # 加载选择的特征
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)

            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                logger.info(f"特征选择应用完成：使用 {len(self.feature_columns)} 个特征")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")
        else:
            logger.info(f"使用全部 {len(self.feature_columns)} 个特征")

        # 检查特征列是否存在
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"以下特征列不存在，将被跳过: {missing_features[:10]}")
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        logger.info(f"最终使用 {len(self.feature_columns)} 个特征")

        # 准备训练数据 - 先处理分类特征
        from sklearn.preprocessing import LabelEncoder
        
        # 识别分类特征（字符串类型）
        self.categorical_encoders = {}
        categorical_features = []
        
        for col in self.feature_columns:
            if df[col].dtype == 'object':
                print(f"  检测到分类特征: {col}")
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.categorical_encoders[col] = encoder
                categorical_features.append(self.feature_columns.index(col))
        
        X = df[self.feature_columns].values
        y = df['Label'].values

        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"分类特征数量: {len(categorical_features)}")

        # ========== 训练 CatBoost 模型 ==========
        print("\n" + "="*70)
        print("🐱 训练 CatBoost 模型")
        print("="*70)

        # 根据预测周期调整参数
        if horizon == 5:
            # 一周模型参数（防过拟合）
            print("使用一周模型参数（减少树深度，增加早停耐心）...")
            n_estimators = 500
            depth = 6  # 减少深度（7→6）
            learning_rate = 0.05
            stopping_rounds = 50  # 增加早停耐心（30→50）
            l2_leaf_reg = 3  # 增加L2正则（2→3）
            subsample = 0.7
            colsample_bylevel = 0.6
        elif horizon == 1:
            # 次日模型参数（适度）
            print("使用次日模型参数...")
            n_estimators = 500
            depth = 7
            learning_rate = 0.05
            stopping_rounds = 40
            l2_leaf_reg = 3
            subsample = 0.75
            colsample_bylevel = 0.7
        else:  # horizon == 20
            # 一个月模型参数（超增强正则化）
            print("使用20天模型参数（超增强正则化，降低过拟合）...")
            n_estimators = 400  # 减少树数量（500→400）
            depth = 5  # 减少深度（6→5）
            learning_rate = 0.04  # 降低学习率（0.05→0.04）
            stopping_rounds = 60  # 增加早停耐心（40→60）
            l2_leaf_reg = 5  # 增强L2正则（3→5）
            subsample = 0.6  # 减少行采样（0.75→0.6）
            colsample_bylevel = 0.6  # 减少列采样（0.7→0.6）

        from catboost import CatBoostClassifier, Pool

        self.catboost_model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='Accuracy',
            depth=depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            random_seed=2020,
            verbose=100,
            early_stopping_rounds=stopping_rounds,
            thread_count=-1,
            allow_writing_files=False,
            cat_features=categorical_features if categorical_features else None
        )

        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        catboost_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 创建 Pool 对象（CatBoost 推荐）
            train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features if categorical_features else None)
            val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features if categorical_features else None)

            self.catboost_model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False
            )

            y_pred_fold = self.catboost_model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred_fold)
            catboost_scores.append(score)
            print(f"   Fold {fold} 验证准确率: {score:.4f}")

        # 使用全部数据重新训练
        full_pool = Pool(data=X, label=y, cat_features=categorical_features if categorical_features else None)
        self.catboost_model.fit(full_pool, verbose=100)

        # 获取实际训练的树数量
        self.actual_n_estimators = self.catboost_model.tree_count_
        mean_accuracy = np.mean(catboost_scores)
        std_accuracy = np.std(catboost_scores)
        print(f"\n✅ CatBoost 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators} (原计划: {n_estimators})")
        print(f"   平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'catboost',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        accuracy_file = 'data/model_accuracy.json'
        try:
            # 读取现有数据
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            # 更新当前模型的准确率
            key = f'catboost_{horizon}d'
            existing_data[key] = accuracy_info
            
            # 保存回文件
            with open(accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(f"准确率已保存到 {accuracy_file}")
        except Exception as e:
            logger.warning(f"保存准确率失败: {e}")

        # ========== 输出 CatBoost 特征重要性 ==========
        print("\n" + "="*70)
        logger.info("分析 CatBoost 特征重要性")
        print("="*70)

        # CatBoost 提供多种特征重要性计算方法
        feature_importance = self.catboost_model.get_feature_importance(prettified=True)
        feat_imp = pd.DataFrame({
            'Feature': [self.feature_columns[i] for i in range(len(self.feature_columns))],
            'Importance': self.catboost_model.feature_importances_
        })
        feat_imp = feat_imp.sort_values('Importance', ascending=False)

        # 计算特征影响方向（使用 SHAP 值）
        try:
            # CatBoost 的 get_feature_importance 返回的是重要性排序
            # 使用 predict_contributions 获取特征贡献值
            contrib_values = self.catboost_model.predict(X, prediction_type='RawFormulaVal')
            # 计算每个特征的边际贡献
            # 对于二分类问题，CatBoost 的贡献值计算比较复杂
            # 这里使用特征重要性作为替代，并基于特征的重要性方向推断
            # 注意：CatBoost 的特征重要性都是正数，无法直接判断影响方向
            # 因此我们标记为 'Unknown'
            feat_imp['Impact_Direction'] = 'Unknown'
            logger.info("CatBoost 特征贡献分析：由于 CatBoost 特征重要性为正值，无法直接判断影响方向，标记为 Unknown")
        except Exception as e:
            logger.warning(f"CatBoost 特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        # 保存特征重要性
        feat_imp.to_csv('output/catboost_feature_importance.csv', index=False)
        logger.info(r"已保存特征重要性至 output/catboost_feature_importance.csv")

        # 显示前20个重要特征
        print("\n📊 CatBoost Top 20 重要特征:")
        print(feat_imp[['Feature', 'Importance', 'Impact_Direction']].head(20))

        print("\n" + "="*70)
        logger.info(r"CatBoost 模型训练完成！")
        print("="*70)

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
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算技术指标（80个指标）
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)

            # 计算多周期指标
            stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

            # 计算相对强度指标
            stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

            # 创建资金流向特征
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)

            # 创建市场环境特征（包含港股和美股）
            stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

            # 添加基本面特征
            fundamental_features = self.feature_engineer.create_fundamental_features(code)
            for key, value in fundamental_features.items():
                stock_df[key] = value

            # 添加股票类型特征
            stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
            for key, value in stock_type_features.items():
                stock_df[key] = value

            # 添加情感特征
            sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
            for key, value in sentiment_features.items():
                stock_df[key] = value

            # 添加主题特征（LDA主题建模）
            topic_features = self.feature_engineer.create_topic_features(code, stock_df)
            for key, value in topic_features.items():
                stock_df[key] = value
                # 添加主题情感交互特征
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value
                # 添加预期差距特征
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

            # 添加板块特征
            sector_features = self.feature_engineer.create_sector_features(code, stock_df)
            for key, value in sector_features.items():
                stock_df[key] = value

            # 生成技术指标与基本面交互特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

            # 生成交叉特征（与训练时保持一致）
            stock_df = self.feature_engineer.create_interaction_features(stock_df)

            # 获取最新数据
            latest_data = stock_df.iloc[-1:]

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            # 处理分类特征（使用训练时的编码器）
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        latest_data[col] = encoder.transform(latest_data[col].astype(str))
                    except ValueError:
                        # 处理未见过的类别，映射到0
                        logger.warning(f"警告: 分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

            X = latest_data[self.feature_columns].values

            # 使用 CatBoost 模型直接预测
            from catboost import Pool
            
            # 获取分类特征索引
            categorical_features = [self.feature_columns.index(col) for col in self.categorical_encoders.keys() if col in self.feature_columns]
            
            test_pool = Pool(data=X, cat_features=categorical_features if categorical_features else None)
            proba = self.catboost_model.predict_proba(test_pool)[0]
            prediction = self.catboost_model.predict(test_pool)[0]

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
            'catboost_model': self.catboost_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'horizon': self.horizon,
            'model_type': self.model_type,
            'categorical_encoders': self.categorical_encoders
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"CatBoost 模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.catboost_model = model_data['catboost_model']
        self.feature_columns = model_data['feature_columns']
        self.actual_n_estimators = model_data['actual_n_estimators']
        self.horizon = model_data.get('horizon', 1)
        self.model_type = model_data.get('model_type', 'catboost')
        self.categorical_encoders = model_data.get('categorical_encoders', {})
        print(f"CatBoost 模型已从 {filepath} 加载")

    def predict_proba(self, X):
        """
        预测概率（用于回测评估器）

        Args:
            X: 测试数据（DataFrame 或 numpy 数组）

        Returns:
            numpy.ndarray: 预测概率数组
        """
        from catboost import Pool
        import numpy as np

        # 确保 test_data 是 DataFrame
        if isinstance(X, pd.DataFrame):
            # 检查 X 是否包含所有需要的特征列
            if all(col in X.columns for col in self.feature_columns):
                # 情况1：X 包含所有需要的特征列（可能是原始 DataFrame）
                test_df = X[self.feature_columns].copy()
            elif len(X.columns) == len(self.feature_columns):
                # 情况2：X 已经是只包含特征列的 DataFrame（列数匹配）
                test_df = X.copy()
                test_df.columns = self.feature_columns  # 确保列名正确
            else:
                # 情况3：X 的列数不匹配，尝试提取存在的列
                available_cols = [col for col in self.feature_columns if col in X.columns]
                if available_cols:
                    test_df = X[available_cols].copy()
                else:
                    raise ValueError(f"无法从输入数据中提取特征列。需要的列：{self.feature_columns[:10]}...")
        else:
            # 如果是 numpy 数组，转换为 DataFrame
            test_df = pd.DataFrame(X, columns=self.feature_columns)

        # 获取分类特征索引
        categorical_features = [self.feature_columns.index(col) for col in self.categorical_encoders.keys() if col in self.feature_columns]

        # 创建 Pool 对象
        test_pool = Pool(data=test_df, cat_features=categorical_features if categorical_features else None)

        # 返回预测概率
        return self.catboost_model.predict_proba(test_pool)


class EnsembleModel:
    """融合模型 - 整合 LightGBM、GBDT、CatBoost 三个模型
    
    支持多种融合方法：
    1. 简单平均：三个模型的概率平均
    2. 加权平均：根据准确率加权
    3. 投票机制：多数投票
    """

    def __init__(self, fusion_method='weighted'):
        """
        Args:
            fusion_method: 融合方法 ('average'/'weighted'/'voting')
        """
        self.lgbm_model = LightGBMModel()
        self.gbdt_model = GBDTModel()
        self.catboost_model = CatBoostModel()
        self.fusion_method = fusion_method
        self.model_accuracies = {}
        self.horizon = 1

    def load_model_accuracy(self):
        """加载模型准确率"""
        accuracy_file = 'data/model_accuracy.json'
        try:
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.model_accuracies = {
                    'lgbm': data.get(f'lgbm_{self.horizon}d', {}).get('accuracy', 0.5),
                    'gbdt': data.get(f'gbdt_{self.horizon}d', {}).get('accuracy', 0.5),
                    'catboost': data.get(f'catboost_{self.horizon}d', {}).get('accuracy', 0.5)
                }
                logger.info(f"已加载模型准确率: {self.model_accuracies}")
            else:
                logger.warning(r"未找到准确率文件，使用默认值")
                self.model_accuracies = {'lgbm': 0.5, 'gbdt': 0.5, 'catboost': 0.5}
        except Exception as e:
            logger.warning(f"加载准确率失败: {e}")
            self.model_accuracies = {'lgbm': 0.5, 'gbdt': 0.5, 'catboost': 0.5}

    def load_models(self, horizon=1):
        """加载三个模型"""
        self.horizon = horizon
        horizon_suffix = f'_{horizon}d'
        
        print("\n" + "="*70)
        print("📦 加载融合模型")
        print("="*70)
        
        # 加载 LightGBM 模型
        lgbm_path = f'data/ml_trading_model_lgbm{horizon_suffix}.pkl'
        if os.path.exists(lgbm_path):
            self.lgbm_model.load_model(lgbm_path)
            logger.info(f"LightGBM 模型已加载")
        else:
            logger.warning(f"LightGBM 模型文件不存在: {lgbm_path}")
        
        # 加载 GBDT 模型
        gbdt_path = f'data/ml_trading_model_gbdt{horizon_suffix}.pkl'
        if os.path.exists(gbdt_path):
            self.gbdt_model.load_model(gbdt_path)
            logger.info(f"GBDT 模型已加载")
        else:
            logger.warning(f"GBDT 模型文件不存在: {gbdt_path}")
        
        # 加载 CatBoost 模型
        catboost_path = f'data/ml_trading_model_catboost{horizon_suffix}.pkl'
        if os.path.exists(catboost_path):
            self.catboost_model.load_model(catboost_path)
            logger.info(f"CatBoost 模型已加载")
        else:
            logger.warning(f"CatBoost 模型文件不存在: {catboost_path}")
        
        # 加载模型准确率
        self.load_model_accuracy()
        
        print("="*70)

    def predict(self, code, predict_date=None):
        """融合预测
        
        Args:
            code: 股票代码
            predict_date: 预测日期
            
        Returns:
            dict: 融合预测结果
        """
        # 获取三个模型的预测结果
        lgbm_result = self.lgbm_model.predict(code, predict_date, self.horizon)
        gbdt_result = self.gbdt_model.predict(code, predict_date, self.horizon)
        catboost_result = self.catboost_model.predict(code, predict_date, self.horizon)
        
        # 检查是否有模型预测失败
        results = {'lgbm': lgbm_result, 'gbdt': gbdt_result, 'catboost': catboost_result}
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) == 0:
            logger.error(f"所有模型预测失败: {code}")
            return None
        
        # 获取概率和预测
        probabilities = []
        predictions = []
        
        for model_name, result in valid_results.items():
            probabilities.append(result['probability'])
            predictions.append(result['prediction'])
        
        # 融合
        if self.fusion_method == 'average':
            # 简单平均
            fused_prob = np.mean(probabilities)
            fused_pred = 1 if fused_prob > 0.5 else 0
            method_name = "简单平均"
        elif self.fusion_method == 'weighted':
            # 加权平均（基于准确率）
            weights = []
            for model_name in valid_results.keys():
                weights.append(self.model_accuracies.get(model_name, 0.5))
            
            total_weight = sum(weights)
            if total_weight > 0:
                fused_prob = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
            else:
                fused_prob = np.mean(probabilities)
            
            fused_pred = 1 if fused_prob > 0.5 else 0
            method_name = "加权平均"
        else:  # voting
            # 投票机制
            fused_pred = 1 if sum(predictions) >= len(predictions) / 2 else 0
            fused_prob = sum(predictions) / len(predictions)
            method_name = "投票机制"
        
        # 计算一致性和置信度
        # 计算预测一致性比例
        if len(valid_results) == 3:
            # 三个模型，检查一致性
            if predictions.count(predictions[0]) == 3:
                consistency_pct = 100  # 三模型一致
            elif predictions.count(predictions[0]) == 2 or predictions.count(predictions[1]) == 2:
                consistency_pct = 67  # 两模型一致
            else:
                consistency_pct = 33  # 三模型不一致
        elif len(valid_results) == 2:
            # 两个模型，检查一致性
            if predictions.count(predictions[0]) == 2:
                consistency_pct = 100  # 两模型一致
            else:
                consistency_pct = 50  # 两模型不一致
        else:
            # 只有一个模型
            consistency_pct = 100
        
        # 计算置信度和预测方向（基于融合概率）
        # 三分类：上涨(1)、观望(0.5)、下跌(0)
        if fused_prob > 0.60:
            confidence = "高"
            fused_direction = 1  # 上涨
        elif fused_prob > 0.50:
            confidence = "中"
            fused_direction = 0.5  # 观望
        else:
            confidence = "低"
            fused_direction = 0  # 下跌
        
        # 构建结果
        result = {
            'code': code,
            'name': STOCK_NAMES.get(code, code),
            'fusion_method': method_name,
            'fused_prediction': fused_direction,  # 上涨=1, 观望=0.5, 下跌=0
            'fused_probability': float(fused_prob),
            'confidence': confidence,
            'consistency': f"{consistency_pct}%",
            'current_price': valid_results[list(valid_results.keys())[0]]['current_price'],
            'date': valid_results[list(valid_results.keys())[0]]['date'],
            'model_predictions': {}
        }
        
        # 添加各模型的预测结果
        for model_name, pred_result in valid_results.items():
            result['model_predictions'][model_name] = {
                'prediction': int(pred_result['prediction']),
                'probability': float(pred_result['probability'])
            }
        
        return result
    
    def predict_batch(self, codes, predict_date=None):
        """批量预测
        
        Args:
            codes: 股票代码列表
            predict_date: 预测日期
            
        Returns:
            list: 融合预测结果列表
        """
        results = []
        for code in codes:
            result = self.predict(code, predict_date=predict_date)
            if result:
                results.append(result)
        return results
    
    def predict_proba(self, X):
        """预测概率（用于回测评估器）

        Args:
            X: 特征数据（numpy array 或 DataFrame）

        Returns:
            numpy array: 概率数组，形状为 (n_samples, 2)
        """
        import numpy as np

        # 使用加权平均融合预测概率
        n_samples = len(X)
        probabilities = np.zeros((n_samples, 2))

        # 获取每个模型的预测概率
        # LightGBM 和 GBDT 可以直接使用 X
        lgbm_probs = self.lgbm_model.model.predict_proba(X)
        gbdt_probs = self.gbdt_model.gbdt_model.predict_proba(X)

        # CatBoost 需要特殊处理分类特征
        # 检查 X 是否包含所有需要的特征列
        if isinstance(X, pd.DataFrame):
            # 检查 X 是否包含所有特征列（按名称匹配）
            if all(col in X.columns for col in self.catboost_model.feature_columns):
                # X 包含所有特征列，按顺序提取
                test_df = X[self.catboost_model.feature_columns].copy()
            elif len(X.columns) == len(self.catboost_model.feature_columns):
                # X 的列数匹配但列名可能不同
                # 假设 X 的列顺序与训练时一致（这是回测评估器的默认行为）
                test_df = X.copy()
                test_df.columns = self.catboost_model.feature_columns
            else:
                # X 的列数不匹配，尝试提取存在的列
                available_cols = [col for col in self.catboost_model.feature_columns if col in X.columns]
                if len(available_cols) == len(self.catboost_model.feature_columns):
                    # 所有特征都存在，按顺序提取
                    test_df = X[self.catboost_model.feature_columns].copy()
                elif len(available_cols) > 0:
                    # 部分特征存在，补齐缺失的特征
                    test_df = X[available_cols].copy()
                    for col in self.catboost_model.feature_columns:
                        if col not in test_df.columns:
                            test_df[col] = 0.0
                    test_df = test_df[self.catboost_model.feature_columns]
                else:
                    # 无法提取特征列，假设列顺序一致
                    test_df = X.copy()
                    if len(test_df.columns) >= len(self.catboost_model.feature_columns):
                        test_df = test_df.iloc[:, :len(self.catboost_model.feature_columns)]
                        test_df.columns = self.catboost_model.feature_columns
                    else:
                        raise ValueError(f"无法从输入数据中提取特征列: X 有 {len(X.columns)} 列，需要 {len(self.catboost_model.feature_columns)} 列")
        else:
            # 如果是 numpy 数组，转换为 DataFrame
            test_df = pd.DataFrame(X, columns=self.catboost_model.feature_columns)

        # 获取分类特征索引
        categorical_features = [self.catboost_model.feature_columns.index(col) for col in self.catboost_model.categorical_encoders.keys() if col in self.catboost_model.feature_columns]

        # 确保分类特征列是整数类型
        for cat_idx in categorical_features:
            col_name = self.catboost_model.feature_columns[cat_idx]
            if col_name in test_df.columns:
                test_df[col_name] = test_df[col_name].astype(np.int32)

        # 使用 Pool 对象进行预测
        from catboost import Pool
        test_pool = Pool(data=test_df)
        catboost_probs = self.catboost_model.catboost_model.predict_proba(test_pool)

        # 计算权重
        if self.fusion_method == 'weighted':
            lgbm_weight = self.model_accuracies.get('lgbm', 0.5)
            gbdt_weight = self.model_accuracies.get('gbdt', 0.5)
            catboost_weight = self.model_accuracies.get('catboost', 0.5)
            total_weight = lgbm_weight + gbdt_weight + catboost_weight

            if total_weight > 0:
                lgbm_weight /= total_weight
                gbdt_weight /= total_weight
                catboost_weight /= total_weight
            else:
                lgbm_weight = gbdt_weight = catboost_weight = 1.0 / 3.0
        else:
            # 简单平均
            lgbm_weight = gbdt_weight = catboost_weight = 1.0 / 3.0

        # 加权融合
        probabilities = (
            lgbm_weight * lgbm_probs +
            gbdt_weight * gbdt_probs +
            catboost_weight * catboost_probs
        )

        return probabilities
    
    def predict_classes(self, X):
        """预测类别（用于回测评估器）
        
        Args:
            X: 特征数据
            
        Returns:
            numpy array: 预测类别（0或1）
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def save_predictions(self, predictions, filepath=None):
        """保存预测结果到 CSV
        
        Args:
            predictions: 预测结果列表
            filepath: 保存路径（可选）
        """
        if filepath is None:
            filepath = f'data/ml_trading_model_ensemble_predictions_{self.horizon}d.csv'
        
        # 转换为 DataFrame
        data = []
        for pred in predictions:
            row = {
                'code': pred['code'],
                'name': pred['name'],
                'fusion_method': pred['fusion_method'],
                'fused_prediction': pred['fused_prediction'],
                'fused_probability': pred['fused_probability'],
                'confidence': pred['confidence'],
                'consistency': pred['consistency'],
                'current_price': pred['current_price'],
                'date': pred['date'].strftime('%Y-%m-%d')
            }
            
            # 添加各模型的预测结果
            for model_name, model_pred in pred['model_predictions'].items():
                row[f'{model_name}_prediction'] = model_pred['prediction']
                row[f'{model_name}_probability'] = model_pred['probability']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"融合预测结果已保存到 {filepath}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='机器学习交易模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                       help='运行模式: train=训练, predict=预测, evaluate=评估')
    parser.add_argument('--model-type', type=str, default='lgbm', choices=['lgbm', 'gbdt', 'catboost', 'ensemble'],
                       help='模型类型: lgbm=单一LightGBM模型, gbdt=单一GBDT模型, catboost=单一CatBoost模型, ensemble=融合模型（默认lgbm）')
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
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择（只使用500个选择的特征，而不是全部2936个）')
    parser.add_argument('--skip-feature-selection', action='store_true',
                       help='跳过特征选择，直接使用已有的特征文件（适用于批量训练多个模型）')
    parser.add_argument('--fusion-method', type=str, default='weighted', choices=['average', 'weighted', 'voting'],
                       help='融合方法: average=简单平均, weighted=加权平均（基于准确率）, voting=投票机制（默认weighted）')

    args = parser.parse_args()

    # 初始化模型
    if args.model_type == 'ensemble':
        logger.info("=" * 70)
        print(f"🎭 使用融合模型（方法: {args.fusion_method}）")
        logger.info("=" * 70)
        lgbm_model = None
        gbdt_model = None
        catboost_model = None
        ensemble_model = EnsembleModel(fusion_method=args.fusion_method)
    elif args.model_type == 'gbdt':
        logger.info("=" * 70)
        logger.info("使用单一 GBDT 模型")
        logger.info("=" * 70)
        lgbm_model = None
        gbdt_model = GBDTModel()
        catboost_model = None
        ensemble_model = None
    elif args.model_type == 'catboost':
        logger.info("=" * 70)
        print("🐱 使用单一 CatBoost 模型")
        logger.info("=" * 70)
        lgbm_model = None
        gbdt_model = None
        catboost_model = CatBoostModel()
        ensemble_model = None
    else:
        logger.info("=" * 70)
        logger.info("使用单一 LightGBM 模型")
        logger.info("=" * 70)
        lgbm_model = MLTradingModel()
        gbdt_model = None
        catboost_model = None
        ensemble_model = None

    if args.mode == 'train':
        logger.info("=" * 50)
        print("训练模式")
        logger.info("=" * 50)

        # 训练模型
        horizon_suffix = f'_{args.horizon}d'
        
        # 检查是否应用特征选择
        # --use-feature-selection: 应用特征选择（加载已有特征文件或运行特征选择）
        # --skip-feature-selection: 跳过运行特征选择过程，但仍加载已有特征文件
        apply_feature_selection = args.use_feature_selection
        run_feature_selection = args.use_feature_selection and not args.skip_feature_selection
        
        if ensemble_model:
            # 融合模型需要训练三个子模型
            print("\n" + "=" * 70)
            print("🎭 训练融合模型的三个子模型")
            logger.info("=" * 70)
            
            # 训练 LightGBM 模型
            print("\n📊 训练 LightGBM 模型...")
            lgbm_model = LightGBMModel()
            lgbm_feature_importance = lgbm_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            lgbm_model.save_model(lgbm_model_path)
            lgbm_importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
            lgbm_feature_importance.to_csv(lgbm_importance_path, index=False)
            logger.info(f"LightGBM 模型已保存到 {lgbm_model_path}")
            logger.info(f"特征重要性已保存到 {lgbm_importance_path}")
            
            # 训练 GBDT 模型
            print("\n📊 训练 GBDT 模型...")
            gbdt_model = GBDTModel()
            gbdt_feature_importance = gbdt_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            gbdt_model_path = args.model_path.replace('.pkl', f'_gbdt{horizon_suffix}.pkl')
            gbdt_model.save_model(gbdt_model_path)
            gbdt_importance_path = gbdt_model_path.replace('.pkl', '_importance.csv')
            gbdt_feature_importance.to_csv(gbdt_importance_path, index=False)
            logger.info(f"GBDT 模型已保存到 {gbdt_model_path}")
            logger.info(f"特征重要性已保存到 {gbdt_importance_path}")
            
            # 训练 CatBoost 模型
            print("\n📊 训练 CatBoost 模型...")
            catboost_model = CatBoostModel()
            catboost_feature_importance = catboost_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            catboost_model_path = args.model_path.replace('.pkl', f'_catboost{horizon_suffix}.pkl')
            catboost_model.save_model(catboost_model_path)
            catboost_importance_path = catboost_model_path.replace('.pkl', '_importance.csv')
            catboost_feature_importance.to_csv(catboost_importance_path, index=False)
            logger.info(f"CatBoost 模型已保存到 {catboost_model_path}")
            logger.info(f"特征重要性已保存到 {catboost_importance_path}")
            
            print("\n" + "=" * 70)
            logger.info(r"融合模型的所有子模型训练完成！")
            logger.info("=" * 70)
        elif lgbm_model:
            feature_importance = lgbm_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            lgbm_model.save_model(lgbm_model_path)
            importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"\n特征重要性已保存到 {importance_path}")
        elif catboost_model:
            feature_importance = catboost_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            catboost_model_path = args.model_path.replace('.pkl', f'_catboost{horizon_suffix}.pkl')
            catboost_model.save_model(catboost_model_path)
            importance_path = catboost_model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"\n特征重要性已保存到 {importance_path}")
        else:
            feature_importance = gbdt_model.train(WATCHLIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            gbdt_model_path = args.model_path.replace('.pkl', f'_gbdt{horizon_suffix}.pkl')
            gbdt_model.save_model(gbdt_model_path)
            importance_path = gbdt_model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"\n特征重要性已保存到 {importance_path}")

    elif args.mode == 'predict':
        logger.info("=" * 50)
        print("预测模式")
        logger.info("=" * 50)

        # 加载模型
        horizon_suffix = f'_{args.horizon}d'
        if ensemble_model:
            # 加载融合模型
            ensemble_model.load_models(args.horizon)
            model_name = f"融合模型（{ensemble_model.fusion_method}）"
            model_file_suffix = "ensemble"
        elif lgbm_model:
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            lgbm_model.load_model(lgbm_model_path)
            model = lgbm_model
            model_name = "LightGBM"
            model_file_suffix = "lgbm"
        elif catboost_model:
            catboost_model_path = args.model_path.replace('.pkl', f'_catboost{horizon_suffix}.pkl')
            catboost_model.load_model(catboost_model_path)
            model = catboost_model
            model_name = "CatBoost"
            model_file_suffix = "catboost"
        else:
            gbdt_model_path = args.model_path.replace('.pkl', f'_gbdt{horizon_suffix}.pkl')
            gbdt_model.load_model(gbdt_model_path)
            model = gbdt_model
            model_name = "GBDT"
            model_file_suffix = "gbdt"

        print(f"已加载 {model_name} 模型")

        # 预测所有股票
        predictions = []
        if args.predict_date:
            print(f"基于日期: {args.predict_date}")
        
        if ensemble_model:
            # 使用融合模型预测
            predictions = ensemble_model.predict_batch(WATCHLIST, args.predict_date)
        else:
            # 使用单一模型预测
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
        
        if ensemble_model:
            # 融合模型输出格式
            print("-" * 140)
            print(f"{'代码':<10} {'股票名称':<12} {'融合预测':<10} {'融合概率':<12} {'置信度':<15} {'一致性':<10} {'当前价格':<12} {'数据日期':<15}")
            print("-" * 140)
            
            for pred in predictions:
                # 三分类：上涨=1, 观望=0.5, 下跌=0
                if pred['fused_prediction'] == 1:
                    pred_label = "上涨"
                elif pred['fused_prediction'] == 0.5:
                    pred_label = "观望"
                else:
                    pred_label = "下跌"
                data_date = pred['date'].strftime('%Y-%m-%d')
                
                print(f"{pred['code']:<10} {pred['name']:<12} {pred_label:<10} {pred['fused_probability']:.4f}   {pred['confidence']:<15} {pred['consistency']:<10} {pred['current_price']:.2f}        {data_date:<15}")
                
                # 显示各模型预测详情
                print(f"         各模型: ", end="")
                for model_name, model_pred in pred['model_predictions'].items():
                    model_pred_label = "上涨" if model_pred['prediction'] == 1 else "下跌"
                    print(f"{model_name}={model_pred_label}({model_pred['probability']:.4f}) ", end="")
                print()
        else:
            # 单一模型输出格式
            print("-" * 100)
            print(f"{'代码':<10} {'股票名称':<12} {'预测':<8} {'概率':<10} {'当前价格':<12} {'数据日期':<15} {'预测目标':<15}")
            print("-" * 100)

            for pred in predictions:
                pred_label = "上涨" if pred['prediction'] == 1 else "下跌"
                data_date = pred['date'].strftime('%Y-%m-%d')
                target_date = get_target_date(pred['date'], horizon=args.horizon)

                print(f"{pred['code']:<10} {pred['name']:<12} {pred_label:<8} {pred['probability']:.4f}    {pred['current_price']:.2f}        {data_date:<15} {target_date:<15}")

        # 保存预测结果
        if ensemble_model:
            # 保存融合预测结果
            ensemble_model.save_predictions(predictions)
            print(f"\n融合预测结果已保存到 data/ml_trading_model_ensemble_predictions_{args.horizon}d.csv")
        else:
            # 保存单一模型预测结果
            pred_df = pd.DataFrame(predictions)
            pred_df['data_date'] = pred_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            pred_df['target_date'] = pred_df['date'].apply(lambda x: get_target_date(x, horizon=args.horizon))

            pred_df_export = pred_df[['code', 'name', 'prediction', 'probability', 'current_price', 'data_date', 'target_date']]

            pred_path = args.model_path.replace('.pkl', f'_{model_file_suffix}_predictions{horizon_suffix}.csv')
            pred_df_export.to_csv(pred_path, index=False)
            print(f"\n预测结果已保存到 {pred_path}")

            # 保存20天预测结果到文本文件（便于后续提取和对比）
            if args.horizon == 20:
                save_predictions_to_text(pred_df_export, args.predict_date)

    elif args.mode == 'evaluate':
        logger.info("=" * 50)
        print("评估模式")
        logger.info("=" * 50)

        if args.model_type == 'both':
            # 加载两个模型
            print("\n加载模型...")
            horizon_suffix = f'_{args.horizon}d'
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            gbdt_model_path = args.model_path.replace('.pkl', f'_gbdt{horizon_suffix}.pkl')
            
            lgbm_model.load_model(lgbm_model_path)
            gbdt_model.load_model(gbdt_model_path)

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

            # GBDT 模型评估
            print("\n" + "="*70)
            print("🌲 GBDT 模型评估")
            print("="*70)
            y_pred_gbdt = gbdt_model.gbdt_model.predict(X_test)

            print("\n分类报告:")
            print(classification_report(y_test, y_pred_gbdt))
            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred_gbdt))
            gbdt_accuracy = accuracy_score(y_test, y_pred_gbdt)
            print(f"\n准确率: {gbdt_accuracy:.4f}")

            # 对比结果
            print("\n" + "="*70)
            logger.info("模型对比")
            print("="*70)
            print(f"LightGBM 准确率: {lgbm_accuracy:.4f}")
            print(f"GBDT 准确率: {gbdt_accuracy:.4f}")
            print(f"准确率差异: {abs(lgbm_accuracy - gbdt_accuracy):.4f}")
            
            if gbdt_accuracy > lgbm_accuracy:
                print(f"\n✅ GBDT 模型表现更好，提升 {gbdt_accuracy - lgbm_accuracy:.4f} ({(gbdt_accuracy - lgbm_accuracy)/lgbm_accuracy*100:.2f}%)")
            elif lgbm_accuracy > gbdt_accuracy:
                print(f"\n✅ LightGBM 模型表现更好，提升 {lgbm_accuracy - gbdt_accuracy:.4f} ({(lgbm_accuracy - gbdt_accuracy)/gbdt_accuracy*100:.2f}%)")
            else:
                print(f"\n⚖️  两种模型表现相同")

        else:
            # 单个模型评估
            model = lgbm_model if lgbm_model else gbdt_model
            model.load_model(args.model_path)

            # 准备测试数据
            print("准备测试数据...")
            test_df = model.prepare_data(WATCHLIST)
            test_df = test_df.dropna()

            X_test = test_df[model.feature_columns].values
            y_test = test_df['Label'].values

            # 使用模型直接预测
            y_pred = model.gbdt_model.predict(X_test)

            # 评估
            print("\n分类报告:")
            print(classification_report(y_test, y_pred))

            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred))

            print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")

    else:
        logger.error(f"不支持的运行模式: {args.mode}")
        print("请使用以下模式之一: train, evaluate, predict")
        sys.exit(1)


# 向后兼容别名
MLTradingModel = LightGBMModel


if __name__ == '__main__':
    main()
