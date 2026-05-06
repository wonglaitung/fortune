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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 缓存配置
CACHE_DIR = 'data/stock_cache'
FEATURE_CACHE_DIR = 'data/feature_cache'  # 特征缓存目录
STOCK_DATA_CACHE_DAYS = 7  # 股票历史数据缓存7天
FEATURE_CACHE_DAYS = 7     # 特征缓存7天（与数据缓存一致）
HSI_DATA_CACHE_HOURS = 1   # 恒生指数数据缓存1小时

# 导入项目模块
from data_services.tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from data_services.technical_analysis import TechnicalAnalyzer
from data_services.fundamental_data import get_comprehensive_fundamental_data
from data_services.volatility_model import GARCHVolatilityModel
from data_services.regime_detector import RegimeDetector
from ml_services.base_model_processor import BaseModelProcessor
from ml_services.us_market_data import us_market_data
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST, TRAINING_STOCKS, STOCK_SECTOR_MAPPING

# 股票名称映射（预测用核心28只）
STOCK_NAMES = STOCK_LIST

# 训练用股票列表（扩展59只，用于增加训练样本）
TRAINING_NAMES = TRAINING_STOCKS

# 股票板块映射（用于特征工程）
STOCK_TYPE_MAPPING = STOCK_SECTOR_MAPPING

# 自选股列表（转换为列表格式）
WATCHLIST = list(STOCK_NAMES.keys())

# 训练股票列表（转换为列表格式）
TRAINING_LIST = list(TRAINING_NAMES.keys())

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


def save_prediction_to_history(predictions, horizon=20, predict_date=None):
    """
    保存预测结果到历史记录文件，用于后续性能监控
    
    参数:
    - predictions: 预测结果列表（字典格式）
    - horizon: 预测周期（天）
    - predict_date: 预测日期字符串
    """
    try:
        # 历史文件路径
        history_file = 'data/prediction_history.json'
        
        # 加载现有历史
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {'predictions': [], 'metadata': {}}
        
        # 当前时间戳
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%dT%H:%M:%S')
        date_str = predict_date if predict_date else now.strftime('%Y-%m-%d')
        
        # 为每个预测创建记录
        new_records = []
        for pred in predictions:
            stock_code = pred.get('code', '')
            stock_name = pred.get('name', '')
            
            # 获取板块信息
            stock_info = STOCK_TYPE_MAPPING.get(stock_code, {})
            if isinstance(stock_info, dict):
                sector = stock_info.get('sector', 'unknown')
            else:
                sector = 'unknown'
            
            # 获取数据日期
            data_date = pred.get('data_date', date_str)

            # 计算 target_date（如果没有提供，使用交易日计算）
            target_date = pred.get('target_date', '')
            if not target_date and data_date:
                target_date = get_target_date_trading_days(data_date, horizon, stock_code)
            
            # 创建预测记录（添加周期标识，用于传导律检测）
            record = {
                'prediction_id': f"{date_str}_{stock_code}_{horizon}d",
                'timestamp': timestamp,
                'stock_code': stock_code,
                'stock_name': stock_name,
                'sector': sector,
                'horizon': horizon,
                'predicted_direction': 'up' if pred.get('prediction', 0) == 1 else 'down',
                'prediction_probability': float(pred.get('probability', 0.5)),
                'confidence_level': 'high' if pred.get('probability', 0) > 0.6 else ('medium' if pred.get('probability', 0) > 0.5 else 'low'),
                'entry_price': float(pred.get('current_price', 0)),
                'model_type': 'catboost',
                'data_date': data_date,
                'target_date': target_date,
                'outcome': None,
                'actual_return': None,
                'actual_direction': None,
                'evaluated_at': None
            }
            
            # 检查是否已存在相同 prediction_id 的记录
            existing_ids = {p['prediction_id'] for p in history['predictions']}
            if record['prediction_id'] not in existing_ids:
                new_records.append(record)
        
        # 添加新记录
        history['predictions'].extend(new_records)
        
        # 更新元数据
        history['metadata']['last_updated'] = timestamp
        history['metadata']['total_predictions'] = len(history['predictions'])
        
        # 保存历史
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(new_records)} 条预测记录到 {history_file}")
        return len(new_records)
        
    except Exception as e:
        logger.error(f"保存预测历史失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 0


def get_target_date(date, horizon):
    """计算目标日期（数据日期 + 预测周期，自然日，已弃用）

    注意：此函数使用自然日计算，与模型预测的交易日不一致。
    推荐使用 get_target_date_trading_days() 获取更准确的结果。
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    target_date = date + timedelta(days=horizon)
    return target_date.strftime('%Y-%m-%d')


def get_target_date_trading_days(date, horizon, stock_code='^HSI'):
    """计算目标日期（数据日期 + N个交易日）

    参数:
    - date: 数据日期（datetime 或 'YYYY-MM-DD' 字符串）
    - horizon: 交易日数量
    - stock_code: 股票代码（保留参数，兼容性）

    返回:
    - 目标日期字符串 (YYYY-MM-DD)
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    try:
        # 使用 akshare 获取交易日历（更可靠）
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        trading_dates = set(df['trade_date'].astype(str).tolist())

        # 找到第 horizon 个交易日后的日期
        count = 0
        current = date

        while count < horizon:
            current += timedelta(days=1)
            date_str = current.strftime('%Y-%m-%d')
            if date_str in trading_dates:
                count += 1

        return current.strftime('%Y-%m-%d')

    except Exception as e:
        # 回退到 pandas 工作日（不包含节假日，但比自然日准确）
        logger.warning(f"获取交易日历失败，回退到工作日计算: {e}")
        try:
            from pandas.tseries.offsets import BDay
            target = date + horizon * BDay()
            return target.strftime('%Y-%m-%d')
        except:
            # 最终回退到自然日
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

def get_stock_data_with_cache(stock_code, period_days=1460):
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

def get_hsi_data_with_cache(period_days=1460):
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


# ========== 特征缓存函数 ==========
def _get_feature_cache_key(stock_code, last_date):
    """生成特征缓存键

    参数:
    - stock_code: 股票代码（如 '0005'）
    - last_date: 数据最后日期（如 '20260418'）

    返回:
    - 缓存键字符串
    """
    return f"{stock_code}_{last_date}"


def _get_feature_cache_file_path(cache_key):
    """获取特征缓存文件路径"""
    if not os.path.exists(FEATURE_CACHE_DIR):
        os.makedirs(FEATURE_CACHE_DIR)
    return os.path.join(FEATURE_CACHE_DIR, f"{cache_key}.pkl")


def _is_feature_cache_valid(cache_file_path):
    """检查特征缓存是否有效"""
    if not os.path.exists(cache_file_path):
        return False
    cache_time = os.path.getmtime(cache_file_path)
    current_time = datetime.now().timestamp()
    age_days = (current_time - cache_time) / 86400  # 转换为天
    return age_days < FEATURE_CACHE_DAYS


def _save_feature_cache(cache_file_path, feature_data):
    """保存特征缓存

    参数:
    - cache_file_path: 缓存文件路径
    - feature_data: dict，包含 'stock_df', 'hsi_df', 'us_market_df', 'feature_df'
    """
    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump({
                'data': feature_data,
                'timestamp': datetime.now().isoformat()
            }, f)
        logger.debug(f"特征缓存已保存: {cache_file_path}")
    except Exception as e:
        logger.warning(f"保存特征缓存失败: {e}")


def _load_feature_cache(cache_file_path):
    """加载特征缓存"""
    try:
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
            return cache['data']
    except Exception as e:
        logger.warning(f"加载特征缓存失败: {e}")
        return None


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

    def detect_market_regime(self, df):
        """
        市场环境识别 - 基于ADX固定阈值
        
        使用传统ADX阈值识别市场环境：
        - ADX > 25：趋势市（严格过滤）
        - ADX < 20：震荡市（放宽过滤）
        - 20 ≤ ADX ≤ 25：正常市（标准过滤）
        
        参数:
            df: 包含ADX列的DataFrame
            
        返回:
            regime: 'trending' | 'ranging' | 'normal'
            
        注意：实验性方案，需通过Walk-forward验证
        """
        if len(df) < 14:  # ADX需要至少14个数据点
            return 'normal'
        
        adx_current = df['ADX'].iloc[-1]
        
        if pd.isna(adx_current):
            return 'normal'
        
        # 固定阈值方法（传统做法）
        if adx_current > 25:
            return 'trending'  # 趋势市：严格过滤
        elif adx_current < 20:
            return 'ranging'   # 震荡市：放宽过滤
        else:
            return 'normal'    # 正常市：标准过滤

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
        # RSI偏离度（震荡市超买超卖识别特征）
        df['RSI_Deviation'] = abs(df['RSI'] - 50)  # RSI偏离50的程度
        df['RSI_Deviation_MA20'] = df['RSI_Deviation'].rolling(window=20, min_periods=1).mean().shift(1)
        df['RSI_Deviation_Normalized'] = (df['RSI_Deviation'].shift(1) - df['RSI_Deviation_MA20']) / (df['RSI_Deviation'].rolling(20, min_periods=1).std().shift(1) + 1e-10)
        # 价格高低点定义（用于背离检测，使用滞后数据避免数据泄漏）
        lookback = 5
        df['Price_Low_5d'] = df['Close'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['Price_High_5d'] = df['Close'].rolling(window=lookback, min_periods=1).max().shift(1)
        # RSI背离检测（震荡市假突破识别特征，使用滞后数据避免数据泄漏）
        # 看涨背离：价格创新低，但RSI未创新低
        df['RSI_Low_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['RSI_Bullish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_Low_5d']) &  # 昨日价格创5日新低
            (df['RSI'] > df['RSI_Low_5d_History'])  # RSI未创5日新低（对比历史最低点）
        ).astype(int)
        # 看跌背离：价格创新高，但RSI未创新高
        df['RSI_High_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['RSI_Bearish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_High_5d']) &  # 昨日价格创5日新高
            (df['RSI'] < df['RSI_High_5d_History'])  # RSI未创5日新高（对比历史最高点）
        ).astype(int)

        # ========== MACD ==========
        df = self.tech_analyzer.calculate_macd(df)
        # MACD 柱状图
        df['MACD_Hist'] = df['MACD'] - df['MACD_signal']
        # MACD 柱状图变化率
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change()
        # MACD背离检测（震荡市假突破识别特征，使用滞后数据避免数据泄漏）
        # 使用5日窗口检测背离
        lookback = 5
        # 看涨背离：价格创新低，但MACD未创新低
        df['MACD_Low_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['MACD_Bullish_Divergence'] = (
            (df['Close'] == df['Price_Low_5d']) &  # 价格创5日新低
            (df['MACD'] > df['MACD_Low_5d_History'])  # MACD未创5日新低（对比历史最低点）
        ).astype(int)
        # 看跌背离：价格创新高，但MACD未创新高
        df['MACD_High_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['MACD_Bearish_Divergence'] = (
            (df['Close'] == df['Price_High_5d']) &  # 价格创5日新高
            (df['MACD'] < df['MACD_High_5d_History'])  # MACD未创5日新高（对比历史最高点）
        ).astype(int)

        # ========== 布林带 ==========
        df = self.tech_analyzer.calculate_bollinger_bands(df, period=20, std_dev=2)
        # 布林带宽度（震荡市识别特征）
        df['BB_Width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        # 布林带宽度归一化（相对于60日均值，使用滞后数据避免数据泄漏）
        df['BB_Width_MA60'] = df['BB_Width'].rolling(window=60, min_periods=1).mean().shift(1)
        df['BB_Width_Normalized'] = (df['BB_Width'].shift(1) - df['BB_Width_MA60']) / (df['BB_Width'].rolling(60, min_periods=1).std().shift(1) + 1e-10)
        # 布林带突破（已删除：与 BB_Position 公式相同，保留 BB_Position）

        # ========== ATR ==========
        df = self.tech_analyzer.calculate_atr(df, period=14)
        # ATR 比率（ATR相对于10日均线的比率，使用滞后数据避免数据泄漏）
        df['ATR_MA'] = df['ATR'].rolling(window=10, min_periods=1).mean().shift(1)
        df['ATR_Ratio'] = df['ATR'] / df['ATR_MA']

        # ========== 成交量相关（P0 修复：所有 Volume/Turnover 使用 shift(1) 避免数据泄漏）==========
        df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean().shift(1)
        df['Vol_Ratio'] = df['Volume'].shift(1) / df['Vol_MA20']  # P0 修复：Volume shift(1)
        # 成交量 z-score（使用滞后数据避免数据泄漏）
        df['Vol_Mean_20'] = df['Volume'].rolling(20, min_periods=1).mean().shift(1)
        df['Vol_Std_20'] = df['Volume'].rolling(20, min_periods=1).std().shift(1)
        df['Vol_Z_Score'] = (df['Volume'].shift(1) - df['Vol_Mean_20']) / df['Vol_Std_20']  # P0 修复
        # 成交额（使用滞后数据避免数据泄漏）
        df['Turnover'] = df['Close'].shift(1) * df['Volume'].shift(1)  # P0 修复
        # 成交额 z-score（使用滞后数据避免数据泄漏）
        df['Turnover_Mean_20'] = df['Turnover'].rolling(20, min_periods=1).mean().shift(1)
        df['Turnover_Std_20'] = df['Turnover'].rolling(20, min_periods=1).std().shift(1)
        df['Turnover_Z_Score'] = (df['Turnover'] - df['Turnover_Mean_20']) / df['Turnover_Std_20']
        # 成交额变化率（多周期，使用滞后数据）
        df['Turnover_Change_1d'] = df['Turnover'].pct_change()
        df['Turnover_Change_5d'] = df['Turnover'].pct_change(5)
        df['Turnover_Change_10d'] = df['Turnover'].pct_change(10)
        df['Turnover_Change_20d'] = df['Turnover'].pct_change(20)
        # 换手率（使用滞后数据避免数据泄漏）
        df['Turnover_Rate'] = (df['Turnover'] / (df['Close'].shift(1) * 1000000)) * 100  # P0 修复
        # 换手率变化率
        df['Turnover_Rate_Change_5d'] = df['Turnover_Rate'].pct_change(5)
        df['Turnover_Rate_Change_20d'] = df['Turnover_Rate'].pct_change(20)

        # ========== VWAP (成交量加权平均价，P0 修复：所有分量使用 shift(1) 避免数据泄漏) ==========
        df['TP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['Volume_Lagged'] = df['Volume'].shift(1)  # P0 修复：Volume 必须滞后
        df['VWAP'] = (df['TP'] * df['Volume_Lagged']).rolling(window=20, min_periods=1).sum() / df['Volume_Lagged'].rolling(window=20, min_periods=1).sum()

        # ========== OBV (能量潮，P0 修复：使用滞后数据避免数据泄漏) ==========
        # OBV 累加逻辑必须使用 T-1 日的 Volume，否则模型可通过当日 Volume 推断当日 Close 涨跌
        df['OBV'] = 0.0
        for i in range(1, len(df)):
            # P0 修复：使用 shift(1) 后的 Close 比较和 Volume
            if df['Close'].iloc[i-1] > df['Close'].iloc[i-2]:  # 昨日收盘价 > 前日
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i-1]  # 使用昨日 Volume
            elif df['Close'].iloc[i-1] < df['Close'].iloc[i-2]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i-1]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        df['OBV'] = df['OBV'].shift(1)  # P0 修复：整体滞后，确保预测时不使用当日数据

        # ========== CMF (Chaikin Money Flow，P0 修复：所有 Volume 使用 shift(1)) ==========
        # 使用滞后High/Low避免数据泄漏
        df['MF_Multiplier'] = ((df['Close'].shift(1) - df['Low'].shift(1)) - (df['High'].shift(1) - df['Close'].shift(1))) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)  # P0 修复：Close 也 shift
        df['Volume_Lagged'] = df['Volume'].shift(1)  # P0 修复
        df['MF_Volume'] = df['MF_Multiplier'] * df['Volume_Lagged']
        df['CMF'] = df['MF_Volume'].rolling(20, min_periods=1).sum() / df['Volume_Lagged'].rolling(20, min_periods=1).sum()
        # CMF 信号线（使用滞后数据避免数据泄漏）
        df['CMF_Signal'] = df['CMF'].rolling(5, min_periods=1).mean().shift(1)

        # ========== ADX (平均趋向指数) ==========
        # +DM and -DM (使用滞后数据避免数据泄漏)
        up_move = df['High'].diff().shift(1)
        down_move = -df['Low'].diff().shift(1)
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
        # 使用滞后数据避免数据泄漏（昨日的14日高低点）
        df['Low_Min'] = df['Low'].rolling(window=K_Period, min_periods=1).min().shift(1)
        df['High_Max'] = df['High'].rolling(window=K_Period, min_periods=1).max().shift(1)
        df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min'])
        df['Stoch_D'] = df['Stoch_K'].rolling(window=D_Period, min_periods=1).mean().shift(1)

        # ========== Williams %R ==========
        df['Williams_R'] = (df['High_Max'] - df['Close']) / (df['High_Max'] - df['Low_Min']) * -100

        # ========== ROC (价格变化率) ==========
        df['ROC'] = df['Close'].pct_change(periods=12)

        # ========== 波动率（年化） ==========
        # 已删除 Returns 和 Volatility：与 Return_1d 和 Volatility_20d 完全相同

        # ========== 价格位置特征 ==========
        # 已删除 MA5_Deviation 和 MA10_Deviation：与 BIAS6 和 BIAS12 公式相同
        # 价格百分位（相对于60日窗口，使用滞后数据避免数据泄漏）
        df['Price_Percentile'] = df['Close'].rolling(window=60, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
        ).shift(1)
        # 价格通道位置（震荡市识别特征，使用滞后数据避免数据泄漏）
        df['Channel_High_20d'] = df['High'].rolling(window=20, min_periods=1).max().shift(1)
        df['Channel_Low_20d'] = df['Low'].rolling(window=20, min_periods=1).min().shift(1)
        df['Price_Channel_Position_20d'] = (df['Close'] - df['Channel_Low_20d']) / (df['Channel_High_20d'] - df['Channel_Low_20d'] + 1e-10)
        # 价格在通道中的位置（靠近上轨/下轨/中轨）
        df['Price_Channel_Zone'] = np.where(
            df['Price_Channel_Position_20d'] > 0.7, 1,  # 靠近上轨
            np.where(
                df['Price_Channel_Position_20d'] < 0.3, -1,  # 靠近下轨
                0  # 中轨
            )
        )
        # 布林带位置（使用滞后数据避免数据泄漏）
        df['BB_Position'] = (df['Close'] - df['BB_lower'].shift(1)) / (df['BB_upper'].shift(1) - df['BB_lower'].shift(1) + 1e-10)

        # ========== 多周期收益率 ==========
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_3d'] = df['Close'].pct_change(3)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)
        df['Return_60d'] = df['Close'].pct_change(60)

        # ========== 价格相对于均线的比率（使用滞后Close避免数据泄漏） ==========
        df['Price_Ratio_MA5'] = df['Close'].shift(1) / df['MA5']
        df['Price_Ratio_MA20'] = df['Close'].shift(1) / df['MA20']
        df['Price_Ratio_MA50'] = df['Close'].shift(1) / df['MA50']

        # ========== 高优先级：滚动统计特征 ==========
        # 均线偏离度（标准化，使用滞后数据避免数据泄漏）
        df['MA5_Deviation_Std'] = (df['Close'] - df['MA5']) / df['Close'].rolling(5).std().shift(1)
        df['MA20_Deviation_Std'] = (df['Close'] - df['MA20']) / df['Close'].rolling(20).std().shift(1)

        # 滚动波动率（多周期，使用滞后数据避免数据泄漏）
        df['Volatility_5d'] = df['Close'].pct_change().rolling(5).std().shift(1)
        df['Volatility_10d'] = df['Close'].pct_change().rolling(10).std().shift(1)
        df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std().shift(1)

        # ========== GARCH 波动率特征（per-stock，2026-04-27 新增）==========
        # GARCH(1,1) 条件波动率，捕捉波动率聚类和持续性
        # GARCHVolatilityModel 内置 shift(1) 数据泄漏保护
        try:
            garch_model = GARCHVolatilityModel()
            df = garch_model.calculate_features(df, return_col='Return_1d')
            # 填充开头可能存在的 NaN（shift 导致）
            garch_defaults = {
                'GARCH_Conditional_Vol': 0.0,
                'GARCH_Vol_Ratio': 1.0,
                'GARCH_Vol_Change_5d': 0.0,
                'GARCH_Persistence': 0.8,
            }
            for col, default_val in garch_defaults.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default_val)
        except Exception as e:
            logger.warning(f"GARCH 特征计算失败，使用默认值: {e}")

        # 滚动偏度/峰度（业界常用，使用滞后数据避免数据泄漏）
        df['Skewness_20d'] = df['Close'].pct_change().rolling(20).skew().shift(1)
        df['Kurtosis_20d'] = df['Close'].pct_change().rolling(20).kurt().shift(1)

        # 动量加速度（业界重要特征）
        df['Momentum_Accel_5d'] = df['Return_5d'] - df['Return_5d'].shift(5)
        df['Momentum_Accel_10d'] = df['Return_10d'] - df['Return_10d'].shift(5)

        # ========== 高优先级：价格形态特征 ==========
        # N日高低点位置（0-1之间，1表示在最高点，使用滞后数据避免泄漏）
        # 已删除 High_Position_20d：与 Price_Channel_Position_20d 公式相同
        df['High_Position_60d'] = (df['Close'] - df['Low'].rolling(60).min().shift(1)) / (df['High'].rolling(60).max().shift(1) - df['Low'].rolling(60).min().shift(1))

        # 距离近期高点/低点的天数（业界常用，使用滞后数据避免数据泄漏）
        df['Days_Since_High_20d'] = df['Close'].shift(1).rolling(20).apply(lambda x: 20 - np.argmax(x), raw=False)
        df['Days_Since_Low_20d'] = df['Close'].shift(1).rolling(20).apply(lambda x: 20 - np.argmin(x), raw=False)

        # 日内特征（业界核心信号，使用滞后High/Low避免数据泄漏）
        df['Intraday_Range'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Close']
        df['Intraday_Range_MA5'] = df['Intraday_Range'].rolling(5).mean().shift(1)
        df['Intraday_Range_MA20'] = df['Intraday_Range'].rolling(20).mean().shift(1)

        # 收盘位置（阳线/阴线强度，0-1之间，使用滞后High/Low避免数据泄漏）
        df['Close_Position'] = (df['Close'] - df['Low'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1))
        # 上影线/下影线比例（使用滞后High/Low）
        df['Upper_Shadow'] = (df['High'].shift(1) - df[['Close', 'Open']].max(axis=1)) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)
        df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)

        # 开盘缺口
        df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap_Size'] > 0.01).astype(int)  # 跳空高开 >1%
        df['Gap_Down'] = (df['Gap_Size'] < -0.01).astype(int)  # 跳空低开 >1%

        # ========== 中优先级：量价关系特征（P0 修复：使用滞后数据）==========
        # 量价背离（业界重要信号，使用滞后数据）
        df['Price_Up_Volume_Down'] = ((df['Return_1d'].shift(1) > 0) & (df['Turnover'].pct_change() < 0)).astype(int)  # P0 修复
        df['Price_Down_Volume_Up'] = ((df['Return_1d'].shift(1) < 0) & (df['Turnover'].pct_change() > 0)).astype(int)  # P0 修复

        # OBV 趋势（使用滞后数据避免数据泄漏）
        df['OBV_MA5'] = df['OBV'].rolling(5).mean().shift(1)
        df['OBV_Trend'] = (df['OBV'] > df['OBV_MA5']).astype(int)

        # 成交量波动率（使用滞后数据避免数据泄漏）
        df['Volume_Volatility'] = df['Turnover'].shift(1).rolling(20).std() / (df['Turnover'].shift(1).rolling(20).mean() + 1e-10)

        # 成交量比率（多周期，P0 修复：使用滞后数据）
        df['Volume_Ratio_5d'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(5).mean()  # P0 修复
        df['Volume_Ratio_20d'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(20).mean()  # P0 修复

        # ========== 长期趋势特征（专门优化一个月模型） ==========
        # 长期均线（120日半年线、250日年线，使用滞后数据避免数据泄漏）
        df['MA120'] = df['Close'].rolling(window=120, min_periods=1).mean().shift(1)
        df['MA250'] = df['Close'].rolling(window=250, min_periods=1).mean().shift(1)

        # ========== 新增指标：趋势斜率 ==========
        # 计算趋势斜率（线性回归斜率）
        def calc_trend_slope(prices):
            if len(prices) < 2:
                return 0.0
            x = np.arange(len(prices))
            try:
                slope, _ = np.polyfit(x, prices, 1)
                # 标准化斜率（相对于平均价格）
                normalized_slope = slope / (np.mean(np.abs(prices)) + 1e-10) * 100
                return normalized_slope
            except:
                return 0.0

        df['Trend_Slope_5d'] = df['Close'].rolling(window=5, min_periods=2).apply(calc_trend_slope, raw=True)
        df['Trend_Slope_20d'] = df['Close'].rolling(window=20, min_periods=2).apply(calc_trend_slope, raw=True)
        df['Trend_Slope_60d'] = df['Close'].rolling(window=60, min_periods=2).apply(calc_trend_slope, raw=True)

        # ========== 新增指标：乖离率 ==========
        # 计算乖离率
        df['BIAS6'] = ((df['Close'] - df['MA5']) / (df['MA5'] + 1e-10)) * 100
        df['BIAS12'] = ((df['Close'] - df['MA10']) / (df['MA10'] + 1e-10)) * 100
        df['BIAS24'] = ((df['Close'] - df['MA20']) / (df['MA20'] + 1e-10)) * 100

        # ========== 新增指标：均线排列 ==========
        # 判断均线排列
        df['MA_Alignment_Bullish_20_50'] = (df['MA20'] > df['MA50']) & (df['MA50'] > df['MA200'])
        df['MA_Alignment_Bearish_20_50'] = (df['MA20'] < df['MA50']) & (df['MA50'] < df['MA200'])

        # 均线排列强度（多头排列的数量减去空头排列的数量）
        df['MA_Alignment_Strength'] = (
            (df['MA20'] > df['MA50']).astype(int) +
            (df['MA50'] > df['MA200']).astype(int) -
            (df['MA20'] < df['MA50']).astype(int) -
            (df['MA50'] < df['MA200']).astype(int)
        )

        # ========== 新增指标：日内振幅（更精确的计算） ==========
        # 计算日内振幅（相对于开盘价，使用滞后数据避免数据泄漏）
        df['Intraday_Amplitude'] = ((df['High'].shift(1) - df['Low'].shift(1)) / (df['Open'] + 1e-10)) * 100

        # ========== 新增指标：多周期波动率 ==========
        # 补充10日和60日波动率（使用滞后数据避免数据泄漏）
        df['Volatility_10d'] = df['Close'].pct_change().rolling(10).std().shift(1)
        df['Volatility_60d'] = df['Close'].pct_change().rolling(60).std().shift(1)

        # ========== 新增指标：多周期偏度和峰度 ==========
        # 补充多周期偏度和峰度
        df['Skewness_5d'] = df['Close'].pct_change().rolling(5).skew()
        df['Skewness_10d'] = df['Close'].pct_change().rolling(10).skew()
        df['Kurtosis_5d'] = df['Close'].pct_change().rolling(5).kurt()
        df['Kurtosis_10d'] = df['Close'].pct_change().rolling(10).kurt()

        # 价格相对长期均线的比率（业界长期趋势指标，使用滞后数据避免数据泄漏）
        df['Price_Ratio_MA120'] = df['Close'].shift(1) / df['MA120']
        df['Price_Ratio_MA250'] = df['Close'].shift(1) / df['MA250']

        # 长期收益率（业界核心长期特征）
        df['Return_120d'] = df['Close'].pct_change(120)
        df['Return_250d'] = df['Close'].pct_change(250)

        # 长期动量（已删除 Momentum_120d 和 Momentum_250d：与 Return_120d 和 Return_250d 公式相同）

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

        # 长期波动率（风险指标，使用滞后数据避免数据泄漏）
        df['Volatility_60d'] = df['Close'].pct_change().rolling(60).std().shift(1)
        df['Volatility_120d'] = df['Close'].pct_change().rolling(120).std().shift(1)

        # 长期ATR（长期风险，使用滞后数据避免数据泄漏）
        df['ATR_MA60'] = df['ATR'].rolling(60, min_periods=1).mean().shift(1)
        df['ATR_MA120'] = df['ATR'].rolling(120, min_periods=1).mean().shift(1)
        df['ATR_Ratio_60d'] = df['ATR'] / df['ATR_MA60']
        df['ATR_Ratio_120d'] = df['ATR'] / df['ATR_MA120']

        # 长期成交量趋势（使用滞后数据避免数据泄漏）
        df['Volume_MA120'] = df['Volume'].rolling(120, min_periods=1).mean().shift(1)
        df['Volume_MA250'] = df['Volume'].rolling(250, min_periods=1).mean().shift(1)
        df['Volume_Ratio_120d'] = df['Volume'].shift(1) / df['Volume_MA120']  # P0 修复
        df['Volume_Trend_Long'] = np.where(
            df['Volume_MA120'] > df['Volume_MA250'], 1, -1
        )

        # 长期支撑阻力位（基于120日高低点，使用滞后数据避免数据泄漏）
        df['Support_120d'] = df['Low'].rolling(120, min_periods=1).min().shift(1)
        df['Resistance_120d'] = df['High'].rolling(120, min_periods=1).max().shift(1)
        df['Distance_Support_120d'] = (df['Close'] - df['Support_120d']) / df['Close']
        df['Distance_Resistance_120d'] = (df['Resistance_120d'] - df['Close']) / df['Close']

        # 长期RSI（基于120日）
        df['RSI_120'] = self.tech_analyzer.calculate_rsi(df.copy(), period=120)['RSI']

        # ========== 自适应成交量确认过滤器（实验性方案，P0 修复）==========
        # 7日成交量均值（业界常用周期，使用滞后数据避免数据泄漏）
        df['Volume_MA7'] = df['Volume'].rolling(window=7, min_periods=1).mean().shift(1)
        # 成交量比率（当前成交量/7日均量，P0 修复：使用滞后数据）
        df['Volume_Ratio_7d'] = df['Volume'].shift(1) / df['Volume_MA7']
        
        # 市场环境识别（基于ADX）
        market_regime = self.detect_market_regime(df)
        
        # 根据市场环境动态调整阈值
        if market_regime == 'ranging':
            # 震荡市：放宽过滤（1.2倍 → 1.0倍）
            volume_threshold = 1.0
            df['Market_Regime'] = 2  # 标记为震荡市
        elif market_regime == 'trending':
            # 趋势市：严格过滤（1.2倍 → 1.4倍）
            volume_threshold = 1.4
            df['Market_Regime'] = 1  # 标记为趋势市
        else:
            # 正常市：标准过滤
            volume_threshold = 1.2
            df['Market_Regime'] = 0  # 标记为正常市
        
        # 成交量确认信号：根据市场环境动态调整阈值
        df['Volume_Confirmation'] = (df['Volume_Ratio_7d'] >= volume_threshold).astype(int)
        # 成交量确认强度（0-1标准化）
        df['Volume_Confirmation_Strength'] = np.minimum(df['Volume_Ratio_7d'] / 2.0, 1.0)

        # ========== 新增特征：假突破检测（符合Bookmap 3点检查清单）==========
        # 1. 价格突破但成交量萎缩检测
        df['Price_Breakout'] = (df['Close'] > df['BB_upper'].shift(1)).astype(int)
        df['False_Breakout_Volume'] = (
            (df['Price_Breakout'] == 1) & (df['Volume_Ratio_7d'] < 0.8)
        ).astype(int)

        # 2. MACD顶背离检测（价格新高但MACD未新高）
        df['Price_Higher_High'] = (df['Close'] > df['Close'].rolling(5, min_periods=1).max().shift(1)).astype(int)
        df['MACD_Higher_High'] = (df['MACD_Hist'] > df['MACD_Hist'].rolling(5, min_periods=1).max().shift(1)).astype(int)
        df['MACD_Top_Divergence'] = (
            (df['Price_Higher_High'] == 1) & (df['MACD_Higher_High'] == 0) &
            (df['MACD_Hist'] > 0)  # 只在MACD正值区域检测顶背离
        ).astype(int)

        # 3. RSI背离检测（价格新高但RSI未新高，或价格新低但RSI未新低）
        df['RSI_Higher_High'] = (df['RSI'] > df['RSI'].rolling(5, min_periods=1).max().shift(1)).astype(int)
        df['RSI_Lower_Low'] = (df['RSI'] < df['RSI'].rolling(5, min_periods=1).min().shift(1)).astype(int)
        df['Price_Lower_Low'] = (df['Close'] < df['Close'].rolling(5, min_periods=1).min().shift(1)).astype(int)

        df['RSI_Top_Divergence'] = (
            (df['Price_Higher_High'] == 1) & (df['RSI_Higher_High'] == 0) &
            (df['RSI'] > 50)  # 只在RSI>50时检测顶背离
        ).astype(int)

        df['RSI_Bottom_Divergence'] = (
            (df['Price_Lower_Low'] == 1) & (df['RSI_Lower_Low'] == 0) &
            (df['RSI'] < 50)  # 只在RSI<50时检测底背离
        ).astype(int)

        # 自适应假突破检测（根据市场环境动态调整阈值）
        if market_regime == 'ranging':
            # 震荡市：提高触发阈值（2点 → 3点），避免过度过滤
            breakout_threshold = 3
        else:
            # 趋势市/正常市：保持原阈值
            breakout_threshold = 2
        
        # 综合假突破信号（3点检查清单满足阈值即触发）
        df['False_Breakout_Signal'] = (
            (df['False_Breakout_Volume'] + df['MACD_Top_Divergence'] + df['RSI_Top_Divergence']) >= breakout_threshold
        ).astype(int)

        # ========== 新增特征：增强的MA排列（符合掘金量化多周期共振标准）==========
        # 三周期均线排列（5/20/60日MA，与业界标准一致）
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean().shift(1)
        df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean().shift(1)

        # 三周期多头排列（5>20>60）
        df['MA_Alignment_Bullish_5_20_60'] = (
            (df['MA5'] > df['MA20']) & (df['MA20'] > df['MA60'])
        ).astype(int)

        # 三周期空头排列（5<20<60）
        df['MA_Alignment_Bearish_5_20_60'] = (
            (df['MA5'] < df['MA20']) & (df['MA20'] < df['MA60'])
        ).astype(int)

        # 多周期共振得分（已删除 MA_Bullish_Resonance：与 MA_Trend_Consistency 完全相同）

        # 趋势一致性得分（-3到3分，多头排列减去空头排列）
        df['MA_Trend_Consistency'] = (
            (df['MA5'] > df['MA20']).astype(int) -
            (df['MA5'] < df['MA20']).astype(int) +
            (df['MA20'] > df['MA50']).astype(int) -
            (df['MA20'] < df['MA50']).astype(int) +
            (df['MA50'] > df['MA200']).astype(int) -
            (df['MA50'] < df['MA200']).astype(int)
        )

        # ========== 新增特征：市场环境自适应过滤（符合QuantInsti HMM标准）==========
        # 多维度市场状态识别（ADX + 波动率双因子）
        # 计算波动率分位数（基于60日滚动窗口，使用滞后数据避免数据泄漏）
        df['Volatility_60d'] = df['Close'].pct_change().rolling(60).std().shift(1) * np.sqrt(252)
        df['Volatility_30pct'] = df['Volatility_60d'].rolling(120, min_periods=60).quantile(0.3).shift(1)
        df['Volatility_70pct'] = df['Volatility_60d'].rolling(120, min_periods=60).quantile(0.7).shift(1)

        # 市场状态分类（ADX + 波动率）
        df['Market_Regime'] = np.where(
            (df['ADX'] < 20) & (df['Volatility_60d'] < df['Volatility_30pct']), 'ranging',
            np.where(
                (df['ADX'] > 30) & (df['Volatility_60d'] > df['Volatility_70pct']), 'trending',
                'normal'
            )
        )

        # 动态成交量阈值（根据市场状态调整）
        df['Volume_Threshold_Adaptive'] = np.where(
            df['Market_Regime'] == 'ranging', 1.0,      # 震荡市放宽至1.0倍
            np.where(
                df['Market_Regime'] == 'trending', 1.1,  # 趋势市略放宽至1.1倍（趋势已确认）
                1.2                                      # 正常市标准1.2倍
            )
        )

        # 自适应成交量确认信号
        df['Volume_Confirmation_Adaptive'] = (
            df['Volume_Ratio_7d'] >= df['Volume_Threshold_Adaptive']
        ).astype(int)

        # 自适应成交量确认强度（0-1标准化，考虑市场状态）
        df['Volume_Confirmation_Strength_Adaptive'] = np.where(
            df['Market_Regime'] == 'ranging',
            np.minimum(df['Volume_Ratio_7d'] / 1.5, 1.0),   # 震荡市更容易达到满强度
            np.where(
                df['Market_Regime'] == 'trending',
                np.minimum(df['Volume_Ratio_7d'] / 1.8, 1.0),  # 趋势市标准
                np.minimum(df['Volume_Ratio_7d'] / 2.0, 1.0)   # 正常市严格标准
            )
        )

        # 自适应假突破检测（震荡市放宽假突破检测）
        df['False_Breakout_Signal_Adaptive'] = np.where(
            df['Market_Regime'] == 'ranging',
            # 震荡市：更宽松的假突破检测（3点中满足1点即触发）
            (df['False_Breakout_Volume'] + df['MACD_Top_Divergence'] + df['RSI_Top_Divergence']) >= 1,
            np.where(
                df['Market_Regime'] == 'trending',
                # 趋势市：更严格的假突破检测（3点中满足2点才触发，但减少假信号）
                (df['False_Breakout_Volume'] + df['MACD_Top_Divergence'] + df['RSI_Top_Divergence']) >= 2,
                # 正常市：标准假突破检测
                (df['False_Breakout_Volume'] + df['MACD_Top_Divergence'] + df['RSI_Top_Divergence']) >= 2
            )
        ).astype(int)

        # 动态置信度阈值乘数（用于后续模型预测）
        df['Confidence_Threshold_Multiplier'] = np.where(
            df['Market_Regime'] == 'ranging', 1.09,   # 震荡市提高阈值（更严格）
            np.where(
                df['Market_Regime'] == 'trending', 0.91,  # 趋势市降低阈值（更宽松）
                1.0                                       # 正常市标准
            )
        )

        # 市场状态编码（数值型，用于机器学习）
        df['Market_Regime_Encoded'] = np.where(
            df['Market_Regime'] == 'ranging', 0,
            np.where(df['Market_Regime'] == 'normal', 1, 2)
        )

        # ========== 新增特征：ATR动态止损与风险管理（解决盈亏比问题）==========
        # ATR止损距离（基于2倍ATR的止损位与当前价格距离）
        df['ATR_Stop_Loss_Distance'] = (df['Close'] - (df['Close'] - 2 * df['ATR'])) / df['Close']
        
        # 近期ATR变化率（ATR趋势）
        df['ATR_Change_5d'] = df['ATR'].pct_change(5)
        df['ATR_Change_10d'] = df['ATR'].pct_change(10)
        
        # 波动率扩张/收缩信号（使用滞后数据避免数据泄漏）
        df['Volatility_Expansion'] = (df['ATR'] > df['ATR'].shift(1).rolling(20).mean() * 1.2).astype(int)
        df['Volatility_Contraction'] = (df['ATR'] < df['ATR'].shift(1).rolling(20).mean() * 0.8).astype(int)
        
        # 基于ATR的动态风险评分（0-1，越高风险越大，使用滞后数据避免数据泄漏）
        atr_percentile = df['ATR'].rolling(60, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
        ).shift(1)
        df['ATR_Risk_Score'] = atr_percentile.fillna(0.5)

        # ========== 新增特征：连续市场状态记忆（解决连续震荡市问题）==========
        # 连续震荡市天数（过去20天内）
        df['Consecutive_Ranging_Days'] = df['Market_Regime_Encoded'].rolling(20).apply(
            lambda x: (x == 0).sum(), raw=True
        )
        
        # 连续趋势市天数（过去20天内）
        df['Consecutive_Trending_Days'] = df['Market_Regime_Encoded'].rolling(20).apply(
            lambda x: (x == 2).sum(), raw=True
        )
        
        # 市场状态转换频率（过去20天内状态变化次数）
        df['Market_Regime_Change_Freq'] = df['Market_Regime_Encoded'].diff().rolling(20).apply(
            lambda x: (x != 0).sum(), raw=True
        )
        
        # 近期市场状态连续性评分（简化版：当前状态与前一日一致的比例）
        df['Market_Continuity_Score'] = (
            df['Market_Regime_Encoded'] == df['Market_Regime_Encoded'].shift(1)
        ).rolling(10).mean()
        
        # 震荡市疲劳指数（在震荡市中停留时间占比，0-1连续值，避免硬阈值）
        df['Ranging_Fatigue_Index'] = df['Consecutive_Ranging_Days'] / 20.0

        # ========== 新增特征：盈亏比与交易质量评估（解决高胜率低收益问题）==========
        # 基于支撑阻力位的潜在盈亏比（Support_120d和Resistance_120d已滞后，无需额外shift）
        potential_reward = df['Resistance_120d'] - df['Close']
        potential_risk = df['Close'] - df['Support_120d']
        df['Risk_Reward_Ratio'] = np.where(
            potential_risk > 0,
            potential_reward / potential_risk,
            0
        )
        
        # 盈亏比质量评分（0-1）
        df['RR_Quality_Score'] = np.where(
            df['Risk_Reward_Ratio'] >= 2.0, 1.0,
            np.where(df['Risk_Reward_Ratio'] >= 1.0, 0.5, 0.0)
        )
        
        # 价格位置风险评分（接近支撑位=低风险，接近阻力位=高风险）
        # Distance_Support_120d和Distance_Resistance_120d已基于滞后数据计算
        df['Price_Position_Risk'] = (
            df['Distance_Support_120d'] / 
            (df['Distance_Support_120d'] + df['Distance_Resistance_120d'] + 1e-10)
        )
        
        # 综合交易质量评分（结合胜率预期和盈亏比）
        # 假设模型准确率约60%，计算期望收益
        win_prob = 0.6
        df['Expected_Value_Score'] = (
            win_prob * df['Risk_Reward_Ratio'] - (1 - win_prob)
        ) * df['Volume_Confirmation_Adaptive']
        
        # 高潜力交易标记（盈亏比>2且成交量确认）
        df['High_Potential_Trade'] = (
            (df['Risk_Reward_Ratio'] >= 2.0) & 
            (df['Volume_Confirmation_Adaptive'] == 1)
        ).astype(int)
        
        # 趋势强度与风险匹配度（强趋势应配低波动，弱趋势应配高波动）
        trend_strength = np.abs(df['Trend_Slope_20d'])
        volatility_normalized = df['ATR_Risk_Score']
        df['Trend_Vol_Match'] = np.where(
            trend_strength > 0.01,
            1 - np.abs(volatility_normalized - 0.5) * 2,  # 趋势强时，中等波动最佳
            volatility_normalized  # 趋势弱时，高波动可能有机会
        )

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

        # 价格相对位置（使用滞后数据避免数据泄漏）
        df['Price_Pct_20d'] = df['Close'].shift(1).rolling(window=20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))

        # 放量上涨信号
        df['Strong_Volume_Up'] = (df['Close'] > df['Open']) & (df['Vol_Ratio'] > 1.5)

        # 缩量回调信号
        df['Prev_Close'] = df['Close'].shift(1)
        df['Weak_Volume_Down'] = (df['Close'] < df['Prev_Close']) & (df['Vol_Ratio'] < 1.0) & ((df['Prev_Close'] - df['Close']) / df['Prev_Close'] < 0.02)

        # 动量信号（已删除 Momentum_5d 和 Momentum_10d：与 Return_5d 和 Return_10d 公式相同）

        # ========== 异常检测特征（使用滞后数据避免数据泄漏）==========
        # 基于两年数据验证（2024-04-01 至 2026-04-01，938个异常）
        # 关键发现：价格异常+当日下跌，5天胜率71.7%，10天胜率72.8%
        
        # 1. 价格异常标记（昨日是否有价格异常）
        # 使用滞后数据计算涨跌幅，Z-Score > 3.0 视为异常
        df['Price_Return_1d'] = df['Close'].pct_change().shift(1)
        df['Price_Return_Mean_30d'] = df['Price_Return_1d'].rolling(30, min_periods=10).mean()
        df['Price_Return_Std_30d'] = df['Price_Return_1d'].rolling(30, min_periods=10).std()
        df['Price_Anomaly_ZScore'] = (
            (df['Price_Return_1d'] - df['Price_Return_Mean_30d']) / 
            (df['Price_Return_Std_30d'] + 1e-10)
        )
        df['Price_Anomaly_Flag'] = (df['Price_Anomaly_ZScore'].abs() > 3.0).astype(int)
        
        # 2. 成交量异常标记（昨日是否有成交量异常）
        df['Volume_Mean_30d'] = df['Volume'].shift(1).rolling(30, min_periods=10).mean()
        df['Volume_Std_30d'] = df['Volume'].shift(1).rolling(30, min_periods=10).std()
        df['Volume_Anomaly_ZScore'] = (
            (df['Volume'].shift(1) - df['Volume_Mean_30d']) / 
            (df['Volume_Std_30d'] + 1e-10)
        )
        df['Volume_Anomaly_Flag'] = (df['Volume_Anomaly_ZScore'] > 3.0).astype(int)
        
        # 3. 异常严重程度评分（0-1，越高越异常）
        df['Anomaly_Severity_Score'] = np.clip(
            (df['Price_Anomaly_ZScore'].abs() + df['Volume_Anomaly_ZScore'].abs()) / 10.0, 
            0, 1
        )
        
        # 4. 连续异常天数（连续出现异常的天数，使用滞后数据）
        df['Consecutive_Anomaly_Days'] = (
            df['Price_Anomaly_Flag'].rolling(5, min_periods=1).sum()
        ).astype(int)
        
        # 5. 抄底信号（价格异常+当日下跌）- 胜率71.7%
        # 基于两年数据验证：价格异常+当日下跌是均值回归信号
        df['Anomaly_Buy_Signal'] = (
            (df['Price_Anomaly_Flag'] == 1) & 
            (df['Price_Return_1d'] < -0.03)  # 昨日下跌超过3%
        ).astype(int)
        
        # 6. 观望信号（价格异常+当日上涨）- 胜率53.7%
        df['Anomaly_Wait_Signal'] = (
            (df['Price_Anomaly_Flag'] == 1) & 
            (df['Price_Return_1d'] > 0.03)  # 昨日上涨超过3%
        ).astype(int)
        
        # 7. 成交量异常谨慎信号（预测能力较弱）
        df['Volume_Anomaly_Caution'] = (
            (df['Volume_Anomaly_Flag'] == 1) & 
            (df['Price_Anomaly_Flag'] == 0)  # 仅成交量异常，价格正常
        ).astype(int)
        
        # 8. 波动率异常标记（昨日波动率是否异常）
        df['Volatility_30d'] = df['Close'].pct_change().shift(1).rolling(30, min_periods=10).std() * np.sqrt(252)
        df['Volatility_Mean_60d'] = df['Volatility_30d'].rolling(60, min_periods=30).mean()
        df['Volatility_Anomaly_Flag'] = (
            df['Volatility_30d'] > df['Volatility_Mean_60d'] * 1.5
        ).astype(int)

        return df

    def create_stock_type_features(self, code, df):
        """创建股票类型特征（基于业界惯例）

        Args:
            code: 股票代码
            df: 股票数据DataFrame（用于计算流动性等动态特征）

        Returns:
            dict: 股票类型特征字典
        """
        # 获取股票类型信息（从 config.py 导入）
        stock_info = STOCK_TYPE_MAPPING.get(code, None)
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
                if 'Turnover' in df.columns:
                    turnover_volatility = df['Turnover'].rolling(window=20, min_periods=10).std().iloc[-1] / df['Turnover'].rolling(window=20, min_periods=10).mean().iloc[-1]
                    features['Stock_Actual_Liquidity_Score'] = max(0, min(1, 1 - turnover_volatility))
                else:
                    features['Stock_Actual_Liquidity_Score'] = 0.5  # 默认值

                # 价格稳定性评分（基于价格波动）
                price_volatility = df['Close'].rolling(window=20, min_periods=10).std().iloc[-1] / df['Close'].rolling(window=20, min_periods=10).mean().iloc[-1]
                features['Stock_Price_Stability_Score'] = max(0, min(1, 1 - price_volatility))

        return features

    def calculate_multi_period_metrics(self, df):
        """计算多周期指标（趋势方向）

        注：RS_Signal 特征已删除，因其与 Trend 使用完全相同的公式
        (df[return_col] > 0).astype(int)，导致 251 个完全重复的交叉特征。
        """
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

        # 计算多周期趋势评分
        trend_cols = [f'{p}d_Trend' for p in periods]
        if all(col in df.columns for col in trend_cols):
            df['Multi_Period_Trend_Score'] = df[trend_cols].sum(axis=1)

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

    def calculate_hsi_regime_features(self, stock_df, hsi_regime_df):
        """将预计算的 HSI 市场状态特征合并到个股 DataFrame

        使用 HMM 隐马尔可夫模型识别恒指的市场状态（牛市/熊市/震荡），
        作为所有个股共享的市场环境特征。

        Args:
            stock_df: 个股 DataFrame（需有 datetime index）
            hsi_regime_df: 预计算的 HSI regime 特征 DataFrame
                           （列名带 HSI_ 前缀，datetime index）

        Returns:
            stock_df with HSI regime columns added
        """
        if stock_df.empty or hsi_regime_df is None or hsi_regime_df.empty:
            return stock_df

        stock_df = stock_df.copy()

        try:
            # 对齐索引：确保两边都是 datetime 类型
            hsi_regime_aligned = hsi_regime_df.copy()
            hsi_regime_aligned.index = pd.to_datetime(hsi_regime_aligned.index)
            stock_idx = pd.to_datetime(stock_df.index)

            # 时区处理：移除时区信息以避免对齐问题
            if hasattr(hsi_regime_aligned.index, 'tz') and hsi_regime_aligned.index.tz is not None:
                hsi_regime_aligned.index = hsi_regime_aligned.index.tz_localize(None)
            if hasattr(stock_idx, 'tz') and stock_idx.tz is not None:
                stock_idx = stock_idx.tz_localize(None)

            # Reindex HSI regime 到个股日期，forward-fill 填补非交易日
            hsi_regime_aligned = hsi_regime_aligned.reindex(stock_idx, method='ffill')

            # 合并到个股 DataFrame
            for col in hsi_regime_aligned.columns:
                stock_df[col] = hsi_regime_aligned[col].values

            # 填充开头可能存在的 NaN（reindex ffill 无法填充开头）
            regime_defaults = {
                'HSI_Market_Regime': 0,       # 默认：震荡
                'HSI_Regime_Prob_0': 0.5,     # 50% 震荡概率
                'HSI_Regime_Prob_1': 0.25,    # 25% 牛市概率
                'HSI_Regime_Prob_2': 0.25,    # 25% 熊市概率
                'HSI_Regime_Duration': 0.0,
                'HSI_Regime_Transition_Prob': 0.0,
            }
            for col, default_val in regime_defaults.items():
                if col in stock_df.columns:
                    stock_df[col] = stock_df[col].fillna(default_val)

            logger.debug(f"HSI 市场状态特征已合并: {list(hsi_regime_aligned.columns)}")

        except Exception as e:
            logger.warning(f"HSI 市场状态特征合并失败: {e}")
            # 填充安全默认值
            defaults = {
                'HSI_Market_Regime': 0,       # 默认：震荡
                'HSI_Regime_Prob_0': 0.5,     # 50% 震荡概率
                'HSI_Regime_Prob_1': 0.25,    # 25% 牛市概率
                'HSI_Regime_Prob_2': 0.25,    # 25% 熊市概率
                'HSI_Regime_Duration': 0.0,
                'HSI_Regime_Transition_Prob': 0.0,
            }
            for col, default_val in defaults.items():
                stock_df[col] = default_val

        return stock_df

    def create_market_environment_features(self, stock_df, hsi_df, us_market_df=None):
        """创建市场环境特征（包含港股和美股）

        P7 优化（2026-05-04）：
        - 添加环境状态特征（HSI_Volatility、Market_Activeness）
        - 环境特征作为"开关"而非"预测变量"，让模型知道当前市场状态

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

        # ========== P7 新增：环境状态特征 ==========
        # 这些特征描述"市场环境"，而非"预测大盘方向"
        # 让模型知道当前是"风大"还是"水静"，调整对微观特征的敏感度

        # 1. HSI 波动率（环境状态，非预测变量）
        # 描述当前市场的"波动基准"，帮助模型判断信号强弱
        hsi_df = hsi_df.copy()
        hsi_df['HSI_Volatility_5d'] = hsi_df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
        hsi_df['HSI_Volatility_20d'] = hsi_df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        hsi_df['HSI_Volatility_Ratio'] = hsi_df['HSI_Volatility_5d'] / hsi_df['HSI_Volatility_20d']

        # 2. 市场活跃度（成交量相对历史水平，P0 修复：使用滞后数据）
        # 高活跃度 = 资金涌入，信号更可靠
        hsi_df['HSI_Volume_MA20'] = hsi_df['Volume'].shift(1).rolling(20).mean() if 'Volume' in hsi_df.columns else 1  # P0 修复
        hsi_df['Market_Activeness'] = hsi_df['Volume'].shift(1) / hsi_df['HSI_Volume_MA20'] if 'Volume' in hsi_df.columns else 1.0  # P0 修复

        # 3. 市场宽度（涨跌比例，描述市场健康度）
        # 这个特征需要从所有股票计算，暂时用 HSI 动量作为代理
        hsi_df['HSI_Momentum_5d'] = hsi_df['Close'].pct_change(5)
        hsi_df['HSI_Momentum_20d'] = hsi_df['Close'].pct_change(20)

        # 4. VIX 水平分类（风险偏好状态）
        # VIX < 15: 低波动，风险偏好高
        # VIX 15-25: 正常
        # VIX > 25: 高波动，风险规避
        # 这个特征在 us_market_df 中处理

        # 合并环境状态特征（shift 1 避免数据泄漏）
        env_features = ['HSI_Volatility_5d', 'HSI_Volatility_20d', 'HSI_Volatility_Ratio',
                        'Market_Activeness', 'HSI_Momentum_5d', 'HSI_Momentum_20d']
        existing_env_features = [f for f in env_features if f in hsi_df.columns]
        if existing_env_features:
            # shift(1) 确保使用 T-1 日数据
            hsi_env_shifted = hsi_df[existing_env_features].shift(1)
            stock_df = stock_df.merge(
                hsi_env_shifted,
                left_index=True, right_index=True, how='left'
            )

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

            # ========== P7 新增：VIX 风险状态分类 ==========
            if 'VIX_Level' in us_market_df.columns:
                # VIX 风险状态（分类，非连续值）
                # 让模型学习"在 VIX 高时如何调整策略"
                vix = us_market_df['VIX_Level'].shift(1)
                stock_df['VIX_Risk_State_Low'] = (vix < 15).astype(int)  # 低波动
                stock_df['VIX_Risk_State_Normal'] = ((vix >= 15) & (vix < 25)).astype(int)  # 正常
                stock_df['VIX_Risk_State_High'] = (vix >= 25).astype(int)  # 高波动

        return stock_df

    def create_label(self, df, horizon, for_backtest=False, min_return_threshold=0.0, label_type='absolute', all_stocks_df=None):
        """创建标签：未来涨跌（支持绝对标签和相对标签）

        根据业界最佳实践，标签定义应考虑交易成本，避免"纸上盈利"变成实际亏损。
        只有当预期收益 > 交易成本 + 最小盈利目标时才标记为正例。

        Args:
            df: 股票数据
            horizon: 预测周期
            for_backtest: 是否为回测准备数据（True时不移除最后horizon行）
            min_return_threshold: 最小收益阈值（默认0%），用于过滤小额波动
                                 推荐值 = 双边交易成本(约0.5%) + 缓冲(0%)
                                 注意：当前设为0%以保持标签分布均衡，后续可调整
            label_type: 标签类型
                       - 'absolute': 绝对标签（收益 > 阈值）
                       - 'relative': 相对标签（收益 > 当日所有股票中位数）
            all_stocks_df: 所有股票的数据（用于计算相对标签的中位数），格式为 DataFrame，包含 Date 和 Future_Return 列
        """
        if df.empty or len(df) < horizon + 1:
            return df

        # 计算未来收益率（累积收益）
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1

        if label_type == 'absolute':
            # 绝对标签：收益 > 阈值
            # 阈值化标签：只有当收益超过阈值时才标记为正例
            # 原因：小幅波动（如0.5%）扣除交易成本后可能变成亏损
            # 业界标准：min_return_threshold = 交易成本 + 最小盈利目标
            df['Label'] = (df['Future_Return'] > min_return_threshold).astype(int)
            df['Label_Threshold'] = min_return_threshold

        elif label_type == 'relative':
            # 相对标签：收益 > 当日所有股票中位数
            # 这种标签方式迫使模型学习"个股为什么比别人强"，而非"大盘会不会涨"
            # 预期效果：全局特征（如美债利率）权重下降，个股特异性特征权重上升

            if all_stocks_df is not None and 'Daily_Median_Return' in all_stocks_df.columns:
                # 使用预先计算的中位数（推荐，避免重复计算）
                df = df.merge(
                    all_stocks_df[['Date', 'Daily_Median_Return']].drop_duplicates(),
                    on='Date',
                    how='left'
                )
            else:
                # 从当前数据计算（仅适用于单只股票数据）
                # 注意：这种方式下，相对标签等于绝对标签（因为只有一只股票）
                df['Daily_Median_Return'] = df['Future_Return'].expanding().median()

            df['Label'] = (df['Future_Return'] > df['Daily_Median_Return']).astype(int)
            df['Label_Threshold'] = df['Daily_Median_Return']  # 记录动态阈值

        else:
            raise ValueError(f"不支持的标签类型: {label_type}，支持: 'absolute', 'relative'")

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
        # 已删除 Momentum_5d：与 Return_5d 公式相同
        technical_features = ['RSI', 'RSI_ROC', 'MACD', 'MACD_Hist', 'MACD_Hist_ROC',
                             'BB_Position', 'ATR', 'Vol_Ratio', 'CMF',
                             'Return_5d', 'Price_Pct_20d']

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
            # 已删除 Momentum_5d 交互：与 Return_5d 公式相同
        ]

        # 检测并跳过常量基本面特征（PE/PB在缓存中可能只有单一值）
        # 常量特征会导致交互特征 = 技术指标 × 常数，与原特征 r=1.0
        constant_fund_features = []
        for fund_feat in fundamental_features[:]:  # 复制列表以安全删除
            if fund_feat in df.columns and df[fund_feat].nunique() <= 1:
                constant_fund_features.append(fund_feat)
                logger.info(f"基本面特征 {fund_feat} 为常量（nunique={df[fund_feat].nunique()}），跳过交互")

        # 从交互列表中移除常量基本面特征的组合
        filtered_interactions = [
            (tech, fund) for tech, fund in high_value_interactions
            if fund not in constant_fund_features
        ]

        print(f"🔗 生成技术指标与基本面交互特征...")

        interaction_count = 0
        for tech_feat, fund_feat in filtered_interactions:
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

    def create_event_driven_features(self, code, df):
        """
        创建事件驱动特征（9个）

        特征列表：
        1. 除净日和分红特征（3个）：
           - Ex_Dividend_In_7d: 未来7天内是否有除净日
           - Ex_Dividend_In_30d: 未来30天内是否有除净日
           - Dividend_Frequency_12m: 过去12个月分红次数

        2. 财报公告日特征（3个）：
           - Earnings_Announcement_In_7d: 未来7天内是否有财报公告
           - Earnings_Announcement_In_30d: 未来30天内是否有财报公告
           - Days_Since_Last_Earnings: 距离上次财报公告的天数

        3. 财报超预期特征（3个）：
           - Earnings_Surprise_Score: 最新财报超预期评分（基于Surprise(%)）
           - Earnings_Surprise_Avg_3: 过去3次财报超预期平均
           - Earnings_Surprise_Trend: 近期财报超预期趋势

        参数:
            code: 股票代码
            df: 股票数据DataFrame

        返回:
            df: 添加事件驱动特征的DataFrame
        """
        if len(df) < 30:  # 需要足够的历史数据
            return df

        # ========== 阶段3.1：除净日和分红特征（3个）==========
        try:
            dividend_info = self._get_dividend_calendar(code)
            if dividend_info is not None and not dividend_info.empty:
                df = self._add_dividend_features(df, dividend_info)
        except Exception as e:
            print(f"  ⚠️ 添加分红特征失败 {code}: {e}")

        # ========== 阶段3.2：财报公告日特征（3个）==========
        try:
            earnings_calendar = self._get_earnings_calendar(code)
            if earnings_calendar is not None:
                df = self._add_earnings_date_features(df, earnings_calendar)
        except Exception as e:
            print(f"  ⚠️ 添加财报公告日特征失败 {code}: {e}")

        # ========== 阶段3.2：财报超预期特征（3个）==========
        try:
            earnings_surprise = self._get_earnings_surprise(code)
            if earnings_surprise is not None and not earnings_surprise.empty:
                df = self._add_earnings_surprise_features(df, earnings_surprise)
        except Exception as e:
            print(f"  ⚠️ 添加财报超预期特征失败 {code}: {e}")

        return df

    def _get_dividend_calendar(self, code):
        """
        获取股息日历（使用AKShare）

        参数:
            code: 股票代码（格式：0700.HK）

        返回:
            DataFrame: 包含除净日等信息的DataFrame
        """
        try:
            import akshare as ak

            # 移除.HK后缀，转换为5位数字格式
            symbol = code.replace('.HK', '')
            if len(symbol) < 5:
                symbol = symbol.zfill(5)
            elif len(symbol) > 5:
                symbol = symbol[-5:]

            # 获取股息数据
            df_dividend = ak.stock_hk_dividend_payout_em(symbol=symbol)

            if df_dividend is None or df_dividend.empty:
                return None

            # 提取关键列
            result = []
            for _, row in df_dividend.iterrows():
                ex_date = row.get('除净日', None)
                if pd.notna(ex_date):
                    result.append({
                        '除净日': ex_date,
                        '分红方案': row.get('分红方案', None),
                        '财政年度': row.get('财政年度', None)
                    })

            if not result:
                return None

            return pd.DataFrame(result)
        except Exception as e:
            print(f"  ⚠️ 获取股息日历失败 {code}: {e}")
            return None

    def _add_dividend_features(self, df, dividend_info):
        """
        添加除净日和分红特征（3个）

        参数:
            df: 股票数据DataFrame
            dividend_info: 股息信息DataFrame

        返回:
            df: 添加除净日特征的DataFrame
        """
        # 确保日期索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除时区信息，统一为无时区datetime
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 转换除净日为datetime（无时区）
        dividend_info['除净日'] = pd.to_datetime(dividend_info['除净日'])
        if dividend_info['除净日'].dt.tz is not None:
            dividend_info['除净日'] = dividend_info['除净日'].dt.tz_localize(None)

        # 特征1：未来7天内是否有除净日
        df['Ex_Dividend_In_7d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 7 for date in dividend_info['除净日'])
        ).astype(int)

        # 特征2：未来30天内是否有除净日
        df['Ex_Dividend_In_30d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 30 for date in dividend_info['除净日'])
        ).astype(int)

        # 特征3：过去12个月分红次数
        df['Dividend_Frequency_12m'] = df.index.to_series().apply(
            lambda x: sum(1 for date in dividend_info['除净日'] if -365 <= (date - x).days <= 0)
        )

        # 使用滞后数据避免数据泄漏
        df['Ex_Dividend_In_7d'] = df['Ex_Dividend_In_7d'].shift(1)
        df['Ex_Dividend_In_30d'] = df['Ex_Dividend_In_30d'].shift(1)
        df['Dividend_Frequency_12m'] = df['Dividend_Frequency_12m'].shift(1)

        return df

    def _get_earnings_calendar(self, code):
        """
        获取财报公告日（使用雅虎财经）

        参数:
            code: 股票代码（格式：0700.HK）

        返回:
            dict: 包含财报公告日等信息的字典
        """
        try:
            import yfinance as yf

            # 雅虎财经需要完整的股票代码
            symbol = code
            if not symbol.endswith('.HK'):
                symbol = code.replace('.HK', '').lstrip('0') + '.HK'

            ticker = yf.Ticker(symbol)

            # 获取财报日历
            calendar = ticker.calendar

            if calendar is None or not calendar:
                return None

            return calendar
        except Exception as e:
            print(f"  ⚠️ 获取财报公告日失败 {code}: {e}")
            return None

    def _add_earnings_date_features(self, df, earnings_calendar):
        """
        添加财报公告日特征（3个）

        参数:
            df: 股票数据DataFrame
            earnings_calendar: 财报日历字典

        返回:
            df: 添加财报公告日特征的DataFrame
        """
        # 确保日期索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除时区信息，统一为无时区datetime
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 获取财报公告日列表
        earnings_dates = earnings_calendar.get('Earnings Date', [])
        if not earnings_dates:
            # 没有财报数据，添加默认值
            df['Earnings_Announcement_In_7d'] = 0
            df['Earnings_Announcement_In_30d'] = 0
            df['Days_Since_Last_Earnings'] = 120  # 默认120天
            return df

        # 转换财报日期为datetime（无时区）
        earnings_dates_clean = []
        for date in earnings_dates:
            if isinstance(date, datetime):
                dt = pd.to_datetime(date)
            else:
                dt = pd.to_datetime(date)
            if dt.tz is not None:
                dt = dt.tz_localize(None)
            earnings_dates_clean.append(dt)
        earnings_dates = earnings_dates_clean

        # 特征1：未来7天内是否有财报公告
        df['Earnings_Announcement_In_7d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 7 for date in earnings_dates)
        ).astype(int)

        # 特征2：未来30天内是否有财报公告
        df['Earnings_Announcement_In_30d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 30 for date in earnings_dates)
        ).astype(int)

        # 特征3：距离上次财报公告的天数
        df['Days_Since_Last_Earnings'] = df.index.to_series().apply(
            lambda x: min(
                [(x - date).days for date in earnings_dates if (x - date).days >= 0],
                default=120
            )
        )

        # 使用滞后数据避免数据泄漏
        df['Earnings_Announcement_In_7d'] = df['Earnings_Announcement_In_7d'].shift(1)
        df['Earnings_Announcement_In_30d'] = df['Earnings_Announcement_In_30d'].shift(1)
        df['Days_Since_Last_Earnings'] = df['Days_Since_Last_Earnings'].shift(1)

        return df

    def _get_earnings_surprise(self, code):
        """
        获取财报超预期数据（使用雅虎财经）

        参数:
            code: 股票代码（格式：0700.HK）

        返回:
            DataFrame: 包含EPS预期、实际EPS、超预期百分比的DataFrame
        """
        try:
            import yfinance as yf

            # 雅虎财经需要完整的股票代码
            symbol = code
            if not symbol.endswith('.HK'):
                symbol = code.replace('.HK', '').lstrip('0') + '.HK'

            ticker = yf.Ticker(symbol)

            # 获取财报超预期数据
            earnings_dates = ticker.earnings_dates

            if earnings_dates is None or earnings_dates.empty:
                return None

            return earnings_dates
        except Exception as e:
            print(f"  ⚠️ 获取财报超预期数据失败 {code}: {e}")
            return None

    def _add_earnings_surprise_features(self, df, earnings_surprise):
        """
        添加财报超预期特征（3个）

        参数:
            df: 股票数据DataFrame
            earnings_surprise: 财报超预期DataFrame（包含Surprise(%)列）

        返回:
            df: 添加财报超预期特征的DataFrame
        """
        # 确保日期索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除时区信息，统一为无时区datetime
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 确保earnings_surprise的索引也是datetime且无时区
        if not isinstance(earnings_surprise.index, pd.DatetimeIndex):
            earnings_surprise.index = pd.to_datetime(earnings_surprise.index)
        if earnings_surprise.index.tz is not None:
            earnings_surprise.index = earnings_surprise.index.tz_localize(None)

        # 检查是否有Surprise(%)列
        if 'Surprise(%)' not in earnings_surprise.columns:
            # 没有超预期数据，添加默认值
            df['Earnings_Surprise_Score'] = 0.0
            df['Earnings_Surprise_Avg_3'] = 0.0
            df['Earnings_Surprise_Trend'] = 0.0
            return df

        # 初始化特征列
        df['Earnings_Surprise_Score'] = 0.0
        df['Earnings_Surprise_Avg_3'] = 0.0
        df['Earnings_Surprise_Trend'] = 0.0

        # 为每个交易日分配超预期评分
        for idx, date in enumerate(df.index):
            # 特征1：最新财报超预期评分（相对于当前日期）
            # 查找距离当前日期最近的已发布财报
            past_earnings = earnings_surprise[earnings_surprise.index <= date]

            if not past_earnings.empty:
                # 取最近一次财报的超预期百分比
                latest_surprise = past_earnings.iloc[-1]['Surprise(%)']

                # 将超预期百分比转换为评分（-1到+1）
                # 超预期>10%为+1，不及预期<-10%为-1，中间按比例缩放
                if pd.notna(latest_surprise):
                    df.loc[date, 'Earnings_Surprise_Score'] = np.clip(latest_surprise / 10.0, -1.0, 1.0)

                    # 特征2：过去3次财报超预期平均
                    if len(past_earnings) >= 3:
                        avg_surprise = past_earnings.tail(3)['Surprise(%)'].mean()
                        if pd.notna(avg_surprise):
                            df.loc[date, 'Earnings_Surprise_Avg_3'] = np.clip(avg_surprise / 10.0, -1.0, 1.0)

                    # 特征3：近期财报超预期趋势
                    if len(past_earnings) >= 2:
                        recent_surprises = past_earnings.tail(2)['Surprise(%)'].tolist()
                        if all(pd.notna(s) for s in recent_surprises):
                            # 计算趋势（最近一次 - 上一次）
                            trend = recent_surprises[-1] - recent_surprises[0]
                            df.loc[date, 'Earnings_Surprise_Trend'] = np.clip(trend / 10.0, -1.0, 1.0)

        # 使用滞后数据避免数据泄漏
        df['Earnings_Surprise_Score'] = df['Earnings_Surprise_Score'].shift(1)
        df['Earnings_Surprise_Avg_3'] = df['Earnings_Surprise_Avg_3'].shift(1)
        df['Earnings_Surprise_Trend'] = df['Earnings_Surprise_Trend'].shift(1)

        return df

    def create_interaction_features(self, df, limit_interaction_features=True):
        """创建交叉特征（类别型 × 数值型）

        生成策略（优化版）：
        - 方案1：只对重要的数值型特征生成交叉特征（100-150个）
        - 方案3：通过特征选择筛选出最重要的交叉特征
        
        参数:
            df: 数据框
            limit_interaction_features: 是否限制交叉特征数量（默认True，启用优化）
        """
        if df.empty:
            return df

        # 类别型特征（8个，已删除5个RS_Signal重复特征）
        categorical_features = [
            'Outperforms_HSI',
            'Strong_Volume_Up',
            'Weak_Volume_Down',
            '3d_Trend', '5d_Trend', '10d_Trend', '20d_Trend', '60d_Trend'
        ]

        # 方案1：定义重要的数值型特征（120个精选特征）
        # 基于特征重要性实验和历史数据选择
        important_numeric_features = [
            # 技术指标（40个）
            'RSI_14d', 'MACD', 'MACD_Signal', 'ATR_14d', 'BB_Width', 'BB_Position',
            'Volume_MA20', 'Volume_Ratio_7d', 'Volume_Trend_5d', 'OBV',
            'MA_Slope_20d', 'MA_Slope_60d', 'Price_Ratio_MA5', 'Price_Ratio_MA20',
            'Distance_MA20', 'Distance_MA60', 'Above_MA20', 'Above_MA60',
            'Volatility_20d', 'Volatility_60d', 'Kurtosis_20d', 'Skewness_20d',
            'Price_Percentile_20d', 'Price_ZScore_20d', 'High_Low_Range_5d',
            'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio', 'Intraday_Amplitude',
            'Gap_Up', 'Gap_Down', 'Gap_Size', 'Gap_Sign',
            'RSI_Overbought', 'RSI_Oversold', 'MACD_Bullish', 'MACD_Bearish',
            
            # 市场环境特征（20个）
            'VIX', 'VIX_Change_5d', 'VIX_Level',
            'HSI_Return_5d', 'HSI_Return_20d', 'HSI_Return_60d',
            'SP500_Return_5d', 'SP500_Return_20d', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
            'US10Y_Yield', 'US10Y_Yield_Change_5d',
            'Market_Regime_Ranging', 'Market_Regime_Normal', 'Market_Regime_Trending',
            'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio', 'GARCH_Vol_Change_5d',
            # 已删除 GARCH_Persistence：交叉后与基础版 r=1.0
            'HSI_Regime_Prob_0', 'HSI_Regime_Prob_1', 'HSI_Regime_Prob_2',
            'HSI_Regime_Duration', 'HSI_Regime_Transition_Prob',
            'Volume_Confirmation_Adaptive',
            # 已删除 False_Breakout_Signal_Adaptive：跨周期完全相同
            # 已删除 Confidence_Threshold_Multiplier：与 Market_Regime_Encoded r=-1.0
            'ATR_Risk_Score',
            
            # 基本面特征（20个）
            'PE_Ratio', 'PB_Ratio', 'ROE', 'ROA', 'Net_Margin',
            'Gross_Margin', 'EPS_Growth', 'Revenue_Growth',
            'Dividend_Yield', 'Beta',
            'PE_vs_Mean', 'PB_vs_Mean', 'ROE_vs_Mean', 'ROA_vs_Mean',
            'PE_Ranking_Percentile', 'PB_Ranking_Percentile',
            'Fundamental_Score', 'Valuation_Score', 'Growth_Score', 'Quality_Score',
            
            # 资金流向特征（15个）
            'Net_Flow_Ratio_5d', 'Net_Flow_Ratio_20d', 'Smart_Money_Indicator',
            'Price_Position_5d', 'Price_Position_20d', 'Price_Position_60d',
            'Volume_Signal_5d', 'Volume_Signal_20d', 'Momentum_Signal_5d',
            'Institutional_Holding', 'Insider_Trading_Signal',
            'Short_Interest_Ratio', 'Margin_Debt_Ratio', 'Put_Call_Ratio',
            'Money_Flow_Index', 'Accumulation_Distribution',
            
            # 风险管理特征（14个，已删除 Consecutive_Ranging_Days：与 Ranging_Fatigue_Index r=1.0）
            'ATR_Stop_Loss_Distance', 'ATR_Change_5d', 'ATR_Change_10d',
            'Ranging_Fatigue_Index',
            'Consecutive_Trending_Days', 'Trending_Momentum_Index',
            'Risk_Reward_Ratio', 'Expected_Value_Score',
            'Win_Loss_Ratio_5d', 'Win_Loss_Ratio_20d',
            'Max_Drawdown_20d', 'Max_Drawdown_60d',
            'Value_at_Risk_5d', 'Value_at_Risk_20d',
            'Expected_Shortfall_5d', 'Expected_Shortfall_20d',
            
            # 股票类型特征（10个）
            'Stock_Type_Bank', 'Stock_Type_Tech', 'Stock_Type_Semiconductor',
            'Stock_Type_AI', 'Stock_Type_Energy', 'Stock_Type_Insurance',
            'Stock_Type_Biotech', 'Stock_Type_RealEstate', 'Stock_Type_Utility',
            'Sector_Leader'
        ]

        # 排除列表
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Returns', 'TP', 'MF_Multiplier', 'MF_Volume',
                          'High_Max', 'Low_Min'] + categorical_features

        if limit_interaction_features:
            # 方案1：只对重要的数值型特征生成交叉特征
            numeric_features = [col for col in important_numeric_features if col in df.columns and col not in exclude_columns]
        else:
            # 原始方法：对所有数值型特征生成交叉特征
            numeric_features = [col for col in df.columns if col not in exclude_columns]

        print(f"生成交叉特征: {len(categorical_features)} 个类别 × {len(numeric_features)} 个数值 = {len(categorical_features) * len(numeric_features)} 个交叉特征")

        if limit_interaction_features:
            print(f"  💡 优化模式：只对 {len(numeric_features)} 个重要数值型特征生成交叉特征")

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
            # 支持多种文件格式和命名
            patterns = [
                'output/model_importance_selected_*.csv',  # 模型重要性法（CSV格式）
                'output/model_importance_features_*.txt',  # 模型重要性法（TXT格式）
                'output/selected_features_*.csv',          # 统计方法（CSV格式）
                'output/statistical_features_*.txt',       # 统计方法（TXT格式）
                'output/model_importance_features_latest.txt',  # 最新模型重要性特征
                'output/statistical_features_latest.txt'   # 最新统计特征
            ]
            
            files = []
            for pattern in patterns:
                found_files = glob.glob(pattern)
                files.extend(found_files)
                if found_files:
                    break  # 找到文件就停止
            
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            selected_names = []
            
            # 根据文件扩展名选择不同的读取方式
            if filepath.endswith('.csv'):
                import pandas as pd
                # 读取特征名称
                df = pd.read_csv(filepath)
                selected_names = df['Feature_Name'].tolist()
            elif filepath.endswith('.txt'):
                # 读取TXT文件（每行一个特征名称）
                with open(filepath, 'r', encoding='utf-8') as f:
                    selected_names = [line.strip() for line in f if line.strip()]
            else:
                logger.error(f"不支持的文件格式: {filepath}")
                return None

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
                if len(selected_set) - len(available_names) > 0:
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
    _deprecation_warning_shown = False  # 类变量，控制弃用警告只显示一次

    # 截面百分位特征列表（P7 精简版，2026-05-04）
    # 从 P6 的 ~100 个精简回核心高质量特征，去除高相关性冗余
    # 原则：质量 > 数量，保留核心 Alpha 信号
    CROSS_SECTIONAL_PERCENTILE_FEATURES = [
        # ========== 波动率特征（保留核心，去除冗余）==========
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',  # 短中长期
        'GARCH_Conditional_Vol', 'Intraday_Range',  # 条件波动率、日内波幅
        # ========== ATR特征（保留核心）==========
        'ATR', 'ATR_Ratio', 'ATR_Risk_Score',
        # ========== 成交量特征（保留核心）==========
        'Volume_Ratio_5d', 'Volume_Ratio_20d', 'Volume_Volatility',
        'OBV', 'CMF',  # 资金流向指标
        # ========== 动量特征（保留核心，去除高相关变体）==========
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_20d',
        'MACD_histogram', 'Close_Position',
        # ========== RSI特征（保留核心）==========
        'RSI', 'RSI_Deviation',
        # ========== 相对强度特征（关键 Alpha 信号）==========
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return',
        # ========== 板块相对动能特征（P4 新增，2026-05-06）==========
        'Sector_Relative_Momentum_5d', 'Sector_Relative_Momentum_20d',
        # ========== 布林带/位置特征 ==========
        'BB_Position', 'BB_Width',
        # ========== 风险特征（保留核心）==========
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Skewness_20d',
        # ========== 资金流向特征 ==========
        'Smart_Money_Score', 'Accumulation_Score',
        # ========== 基本面特征 ==========
        'PE', 'PB', 'ROE', 'Market_Cap',
        # ========== 异常检测特征（关键 Alpha 信号）==========
        'Anomaly_Severity_Score', 'Anomaly_Buy_Signal',
        # ========== 网络增量特征（P7 新增）==========
        'net_node_deviation', 'net_node_deviation_delta_5d',
    ]

    # 截面 Z-Score 特征列表（2026-05-04 P6 扩展，与 CatBoost 保持一致）
    CROSS_SECTIONAL_ZSCORE_FEATURES = [
        # 成交量特征
        'Volume', 'Turnover', 'Turnover_Mean_20',
        'Volume_Std_30d', 'Volume_Z_Score',  # P6 新增
        # 波动率特征
        'Volatility_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_60d',
        'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
        # ATR特征
        'ATR', 'ATR_Ratio', 'ATR_Risk_Score',
        'ATR_MA', 'ATR_MA60',  # P6 新增
        # 成交量比率特征
        'Volume_Ratio_5d', 'Volume_Ratio_20d', 'Volume_Volatility',
        'OBV', 'CMF',
        # 动量特征（P6 扩展）
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_10d',
        'Momentum_Accel_20d', 'Momentum_Accel_60d',  # P6 新增
        'MACD_histogram', 'MACD_Hist_ROC', 'Price_Pct_20d',
        # RSI特征
        'RSI', 'RSI_Deviation', 'RSI_ROC',
        'RSI_Deviation_Normalized',  # P6 新增
        # 相对强度特征
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return', 'RS_Diff_1d',  # P6 新增
        # 布林带/位置特征
        'BB_Position', 'BB_Width', 'BB_Width_Normalized',
        'MA5_Deviation_Std', 'MA20_Deviation_Std',
        # 风险特征（P6 扩展）
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Kurtosis_5d', 'Kurtosis_10d',
        'Skewness_5d', 'Skewness_10d', 'Skewness_20d',  # P6 新增
        # 资金流向特征
        'Smart_Money_Score', 'Accumulation_Score',
        'Net_Flow_5d', 'Net_Flow_20d',
        # 基本面特征
        'PE', 'PB', 'ROE', 'Market_Cap',
        # 异常检测特征（P6 新增）
        'Anomaly_Severity_Score', 'Price_Anomaly_ZScore', 'Volume_Anomaly_ZScore',
    ]

    def __init__(self, use_cross_sectional_percentile=True, use_cross_sectional_zscore=True):
        super().__init__()  # 调用基类初始化
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = 'lgbm'  # 模型类型标识
        self.use_cross_sectional_percentile = use_cross_sectional_percentile
        self.use_cross_sectional_zscore = use_cross_sectional_zscore
        self.cs_feature_stats = {}  # 截面特征统计量

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False, min_return_threshold=0.0):
        """准备训练数据（80个指标版本，优化版）

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
            min_return_threshold: 最小收益阈值（默认0%）
        """
        self.horizon = horizon
        self.min_return_threshold = min_return_threshold
        all_data = []

        # ========== 步骤1：获取共享数据（只获取一次） ==========
        logger.info("获取共享数据...")
        
        # 获取美股市场数据（只获取一次）
        us_market_df = us_market_data.get_all_us_market_data(period_days=1460)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

        # 获取恒生指数数据（只获取一次，所有股票共享）
        hsi_df = get_hsi_data_with_cache(period_days=1460)
        if hsi_df is None or hsi_df.empty:
            raise ValueError("无法获取恒生指数数据")

        # 计算 HSI 市场状态特征（一次性，所有股票共享）
        hsi_regime_df = None
        if hsi_df is not None and not hsi_df.empty:
            try:
                print("  计算 HSI 市场状态特征...")
                regime_detector = RegimeDetector()
                hsi_with_regime = regime_detector.calculate_features(hsi_df.copy())
                rename_map = {c: f'HSI_{c}' for c in RegimeDetector.get_feature_names()}
                hsi_regime_df = hsi_with_regime[RegimeDetector.get_feature_names()].rename(columns=rename_map)
                print("  ✅ HSI 市场状态特征计算完成")
            except Exception as e:
                print(f"  ⚠️ HSI 市场状态特征计算失败: {e}")

        # ========== 步骤2：并行下载股票数据 ==========
        print(f"\n🚀 并行下载 {len(codes)} 只股票数据...")
        
        def fetch_single_stock_data(code):
            """获取单只股票数据"""
            try:
                stock_code = code.replace('.HK', '')
                stock_df = get_stock_data_with_cache(stock_code, period_days=1460)
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

                # 合并 HSI 市场状态特征
                if hsi_regime_df is not None:
                    stock_df = self.feature_engineer.calculate_hsi_regime_features(stock_df, hsi_regime_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon 和阈值）
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest, min_return_threshold=min_return_threshold)

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

        # 计算截面特征（必须在合并所有股票后计算）
        if self.use_cross_sectional_percentile:
            print("\n📊 计算截面百分位特征...")
            df = self._calculate_cross_sectional_percentile_features(df)

        if self.use_cross_sectional_zscore:
            print("\n📊 计算截面Z-Score特征...")
            df = self._calculate_cross_sectional_zscore_features(df)

        # 生成技术指标与基本面交互特征（先执行，因为这是高价值特征）
        print("\n🔗 生成技术指标与基本面交互特征...")
        df = self.feature_engineer.create_technical_fundamental_interactions(df)

        # 生成交叉特征（类别型 × 数值型）
        print("\n🔗 生成交叉特征...")
        df = self.feature_engineer.create_interaction_features(df)

        return df

    def _calculate_cross_sectional_percentile_features(self, df):
        """计算截面百分位特征（在所有股票上联合计算）

        Args:
            df: 包含多只股票数据的DataFrame（必须含'Code'列）

        Returns:
            DataFrame: 添加了截面百分位特征的DataFrame
        """
        if 'Code' not in df.columns:
            logger.warning("数据中没有'Code'列，无法计算截面百分位特征")
            return df

        df = df.copy()

        for feature in self.CROSS_SECTIONAL_PERCENTILE_FEATURES:
            if feature not in df.columns:
                continue

            cs_col = f'{feature}_CS_Pct'
            if cs_col in df.columns:
                continue

            try:
                df[cs_col] = df.groupby(df.index)[feature].transform(
                    lambda x: (x.rank(method='average') - 1) / max(len(x) - 1, 1) if len(x) > 1 else 0.5
                )
            except Exception as e:
                logger.debug(f"截面百分位特征计算失败 {feature}: {e}")

        return df

    def _calculate_cross_sectional_zscore_features(self, df):
        """计算截面Z-Score特征（在所有股票上联合计算）

        Args:
            df: 包含多只股票数据的DataFrame（必须含'Code'列）

        Returns:
            DataFrame: 添加了截面Z-Score特征的DataFrame
        """
        if 'Code' not in df.columns:
            logger.warning("数据中没有'Code'列，无法计算截面Z-Score特征")
            return df

        df = df.copy()

        for feature in self.CROSS_SECTIONAL_ZSCORE_FEATURES:
            if feature not in df.columns:
                continue

            cs_col = f'{feature}_CS_ZScore'
            if cs_col in df.columns:
                continue

            try:
                df[cs_col] = df.groupby(df.index)[feature].transform(
                    lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0.0
                )
            except Exception as e:
                logger.debug(f"截面Z-Score特征计算失败 {feature}: {e}")

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False, min_return_threshold=0.0):
        """训练模型（默认使用全量特征892个）

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（已弃用，默认False使用全量特征）
            min_return_threshold: 最小收益阈值（默认0%）

        Returns:
        # 设置固定随机种子（确保模型训练的可重现性）
        np.random.seed(42)
        random.seed(42)
            特征重要性
        """
        # 检查是否需要显示弃用警告
        if use_feature_selection and not LightGBMModel._deprecation_warning_shown:
            print("⚠️  警告：特征选择功能已弃用，建议使用全量特征（892个）。Walk-forward验证显示全量特征性能更好。")
            LightGBMModel._deprecation_warning_shown = True

        print("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon,
                               min_return_threshold=min_return_threshold,
                               label_type='relative')  # 个股选股模型使用相对标签

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
            print("\n🎯 应用特征选择（LightGBM）...（已弃用）")
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)
            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                print(f"✅ 特征数量: {len(self.feature_columns)}（特征选择 - 已弃用）")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")
        else:
            print(f"\n✅ 特征数量: {len(self.feature_columns)}（全量特征）")

        # 对Market_Regime进行One-Hot编码（LightGBM专用）
        if 'Market_Regime' in df.columns:
            print("  对Market_Regime进行One-Hot编码(LightGBM)...")
            df = pd.get_dummies(df, columns=['Market_Regime'], prefix='Market_Regime')
            # 更新feature_columns
            self.feature_columns = [col for col in self.feature_columns if col != 'Market_Regime']
            self.feature_columns.extend([col for col in df.columns if col.startswith('Market_Regime_')])
            print(f"  One-Hot编码后特征数量: {len(self.feature_columns)}")

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

        # 时间序列分割（添加 gap 参数避免短期依赖）
        tscv = TimeSeriesSplit(n_splits=5, gap=horizon)

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
            # 原因：特征数量约2426个（基础892 + 交叉特征1534），需要更强的正则化防止过拟合
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
        f1_scores = []
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
            f1 = f1_score(y_val, y_pred, zero_division=0)
            scores.append(score)
            f1_scores.append(f1)
            print(f"验证准确率: {score:.4f}, 验证F1分数: {f1:.4f}")

        # 使用全部数据重新训练
        self.model.fit(X, y)

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"\n平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"平均验证F1分数: {mean_f1:.4f} (+/- {std_f1:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'lgbm',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'f1_score': float(mean_f1),
            'f1_std': float(std_f1),
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

        # 保存截面特征的训练集统计量，供单只股票预测时回退使用
        self.cs_feature_stats = {}
        for col in df.columns:
            if col.endswith('_CS_Pct') or col.endswith('_CS_ZScore'):
                self.cs_feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }

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
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据（2年约730天）
            hsi_df = get_hsi_data_tencent(period_days=1460)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

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

            # 添加主题情感交互特征（移到循环外，避免重复调用）
            topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
            for key, value in topic_sentiment_interaction.items():
                stock_df[key] = value

            # 添加预期差距特征（移到循环外，避免重复调用）
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

    def _extract_raw_features_single(self, code, predict_date=None, horizon=None):
        """提取单只股票的原始特征（不含截面特征），供批量预测使用

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期

        Returns:
            DataFrame: 带特征的 stock_df（含 Code 列），或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            stock_code = code.replace('.HK', '')

            # 获取股票数据
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据
            hsi_df = get_hsi_data_tencent(period_days=1460)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)
                if not isinstance(hsi_df.index, pd.DatetimeIndex):
                    hsi_df.index = pd.to_datetime(hsi_df.index)
                if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                    us_market_df.index = pd.to_datetime(us_market_df.index)

                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                if us_market_df is not None:
                    us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算技术指标
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)
            stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)
            stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)
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

            # 添加主题特征
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

            # 生成交互特征
            stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)
            stock_df = self.feature_engineer.create_interaction_features(stock_df)

            # 添加股票代码列
            stock_df['Code'] = code

            return stock_df

        except Exception as e:
            logger.warning(f"提取特征失败 {code}: {e}")
            return None

    def _predict_from_features(self, code, latest_data, horizon=None):
        """从特征数据进行预测（供 predict() 和 predict_batch() 共用）

        Args:
            code: 股票代码
            latest_data: 单行 DataFrame，包含所有特征
            horizon: 预测周期

        Returns:
            dict: 预测结果，或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 处理缺失的截面特征（使用训练集统计量回退）
            cs_feature_suffixes = ['_CS_Pct', '_CS_ZScore']
            for suffix in cs_feature_suffixes:
                cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
                for cs_feat in cs_features:
                    if cs_feat not in latest_data.columns:
                        if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                            latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
                        else:
                            latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0

            # 处理分类特征
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        latest_data[col] = encoder.transform(latest_data[col].astype(str))
                    except ValueError:
                        logger.warning(f"分类特征 {col} 包含训练时未见过的类别，使用默认值")
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
            logger.warning(f"预测失败 {code}: {e}")
            return None

    def predict_batch(self, codes, predict_date=None, horizon=None):
        """批量预测：先提取所有股票特征，再统一计算截面特征，最后逐只预测

        核心改进：截面特征（_CS_Pct, _CS_ZScore）在所有股票数据上联合计算，
        确保训练/预测一致，而非单只股票时退化为 0.5/0.0。

        Args:
            codes: 股票代码列表
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期

        Returns:
            list: 预测结果列表
        """
        if horizon is None:
            horizon = self.horizon

        if len(self.feature_columns) == 0:
            raise ValueError("模型未训练，请先调用train()方法")

        logger.info(f"LightGBM 开始批量预测 {len(codes)} 只股票...")

        # 阶段1：逐只提取原始特征
        all_features = {}
        for code in codes:
            stock_df = self._extract_raw_features_single(code, predict_date, horizon)
            if stock_df is not None:
                all_features[code] = stock_df

        if not all_features:
            logger.warning("LightGBM 批量预测：没有成功提取任何股票的特征")
            return []

        logger.info(f"LightGBM 成功提取 {len(all_features)} 只股票的特征")

        # 阶段2：合并所有股票，计算截面特征
        combined = pd.concat(all_features.values())

        # 统一索引时区：将 tz-aware 时间戳转换为 tz-naive，避免 groupby 时比较错误
        if hasattr(combined.index, 'tz') and combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        elif hasattr(combined.index, 'tz_localize'):
            try:
                combined.index = combined.index.tz_convert('UTC').tz_localize(None)
            except Exception:
                combined.index = pd.to_datetime(combined.index).tz_localize(None)

        # 计算截面百分位特征
        if self.use_cross_sectional_percentile:
            combined = self._calculate_cross_sectional_percentile_features(combined)
            logger.info("LightGBM 批量预测：截面百分位特征计算完成")

        # 计算截面 Z-Score 特征
        if self.use_cross_sectional_zscore:
            combined = self._calculate_cross_sectional_zscore_features(combined)
            logger.info("LightGBM 批量预测：截面 Z-Score 特征计算完成")

        # 阶段3：逐只预测（使用正确的截面特征）
        results = []
        for code in all_features.keys():
            stock_data = combined[combined['Code'] == code]
            if stock_data.empty:
                continue

            latest = stock_data.iloc[-1:].copy()

            # 使用辅助方法进行预测
            result = self._predict_from_features(code, latest, horizon)
            if result:
                results.append(result)

        logger.info(f"LightGBM 批量预测完成：{len(results)}/{len(codes)} 只股票")
        return results

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders,
            'use_cross_sectional_percentile': self.use_cross_sectional_percentile,
            'use_cross_sectional_zscore': self.use_cross_sectional_zscore,
            'cs_feature_stats': getattr(self, 'cs_feature_stats', {})
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
        self.use_cross_sectional_percentile = model_data.get('use_cross_sectional_percentile', True)
        self.use_cross_sectional_zscore = model_data.get('use_cross_sectional_zscore', True)
        self.cs_feature_stats = model_data.get('cs_feature_stats', {})
        print(f"模型已从 {filepath} 加载")


class GBDTModel(BaseTradingModel):
    """GBDT 模型 - 基于梯度提升决策树的单一模型"""
    _deprecation_warning_shown = False  # 类变量，控制弃用警告只显示一次

    # 截面百分位特征列表（P7 精简版，2026-05-04）
    # 从 P6 的 ~100 个精简回核心高质量特征，去除高相关性冗余
    # 原则：质量 > 数量，保留核心 Alpha 信号
    CROSS_SECTIONAL_PERCENTILE_FEATURES = [
        # ========== 波动率特征（保留核心，去除冗余）==========
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',  # 短中长期
        'GARCH_Conditional_Vol', 'Intraday_Range',  # 条件波动率、日内波幅
        # ========== ATR特征（保留核心）==========
        'ATR', 'ATR_Ratio', 'ATR_Risk_Score',
        # ========== 成交量特征（保留核心）==========
        'Volume_Ratio_5d', 'Volume_Ratio_20d', 'Volume_Volatility',
        'OBV', 'CMF',  # 资金流向指标
        # ========== 动量特征（保留核心，去除高相关变体）==========
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_20d',
        'MACD_histogram', 'Close_Position',
        # ========== RSI特征（保留核心）==========
        'RSI', 'RSI_Deviation',
        # ========== 相对强度特征（关键 Alpha 信号）==========
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return',
        # ========== 板块相对动能特征（P4 新增，2026-05-06）==========
        'Sector_Relative_Momentum_5d', 'Sector_Relative_Momentum_20d',
        # ========== 布林带/位置特征 ==========
        'BB_Position', 'BB_Width',
        # ========== 风险特征（保留核心）==========
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Skewness_20d',
        # ========== 资金流向特征 ==========
        'Smart_Money_Score', 'Accumulation_Score',
        # ========== 基本面特征 ==========
        'PE', 'PB', 'ROE', 'Market_Cap',
        # ========== 异常检测特征（关键 Alpha 信号）==========
        'Anomaly_Severity_Score', 'Anomaly_Buy_Signal',
        # ========== 网络增量特征（P7 新增）==========
        'net_node_deviation', 'net_node_deviation_delta_5d',
    ]

    # 截面 Z-Score 特征列表（P7 精简版，2026-05-04）
    CROSS_SECTIONAL_ZSCORE_FEATURES = [
        # 成交量特征
        'Volume', 'Turnover', 'Turnover_Mean_20',
        # 波动率特征
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',
        'GARCH_Conditional_Vol',
        # ATR特征
        'ATR', 'ATR_Ratio',
        # 成交量比率特征
        'Volume_Ratio_5d', 'Volume_Ratio_20d',
        'OBV', 'CMF',
        # 动量特征
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_20d',
        'MACD_histogram',
        # RSI特征
        'RSI', 'RSI_Deviation',
        # 相对强度特征
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return',
        # 布林带/位置特征
        'BB_Position', 'BB_Width',
        # 风险特征
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Skewness_20d',
        # 资金流向特征
        'Smart_Money_Score', 'Accumulation_Score',
        # 基本面特征
        'PE', 'PB', 'ROE', 'Market_Cap',
        # 网络增量特征（P7 新增）
        'net_node_deviation', 'net_node_deviation_delta_5d',
    ]

    def __init__(self, use_cross_sectional_percentile=True, use_cross_sectional_zscore=True):
        super().__init__()  # 调用基类初始化
        self.gbdt_model = None
        self.actual_n_estimators = 0
        self.model_type = 'gbdt'  # 模型类型标识
        self.use_cross_sectional_percentile = use_cross_sectional_percentile
        self.use_cross_sectional_zscore = use_cross_sectional_zscore
        self.cs_feature_stats = {}  # 截面特征统计量

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
            # 支持多种文件格式和命名
            patterns = [
                'output/selected_features_*.csv',          # 统计方法（CSV格式）
                'output/statistical_features_*.txt',       # 统计方法（TXT格式）
                'output/model_importance_selected_*.csv',  # 模型重要性法（CSV格式）
                'output/model_importance_features_*.txt',  # 模型重要性法（TXT格式）
                'output/statistical_features_latest.txt',   # 最新统计特征
                'output/model_importance_features_latest.txt'  # 最新模型重要性特征
            ]
            
            files = []
            for pattern in patterns:
                found_files = glob.glob(pattern)
                files.extend(found_files)
                if found_files:
                    break  # 找到文件就停止
            
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            selected_names = []
            
            # 根据文件扩展名选择不同的读取方式
            if filepath.endswith('.csv'):
                import pandas as pd
                # 读取特征名称
                df = pd.read_csv(filepath)
                selected_names = df['Feature_Name'].tolist()
            elif filepath.endswith('.txt'):
                # 读取TXT文件（每行一个特征名称）
                with open(filepath, 'r', encoding='utf-8') as f:
                    selected_names = [line.strip() for line in f if line.strip()]
            else:
                logger.error(f"不支持的文件格式: {filepath}")
                return None

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
                if len(selected_set) - len(available_set) > 0:
                    logger.warning(f" {len(selected_set) - len(available_set)} 个特征在当前数据集中不存在")
                
                return available_names
            else:
                return selected_names

        except Exception as e:
            logger.warning(f"加载特征列表失败: {e}")
            return None

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False, min_return_threshold=0.0):
        """准备训练数据（80个指标版本）

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
            min_return_threshold: 最小收益阈值（默认0%）
        """
        self.horizon = horizon
        self.min_return_threshold = min_return_threshold
        all_data = []

        # 获取美股市场数据（只获取一次）
        logger.info("获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=1460)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

        # 获取恒生指数数据（移到循环外，避免重复获取）
        hsi_df = get_hsi_data_tencent(period_days=1460)

        # 计算 HSI 市场状态特征（一次性，所有股票共享）
        hsi_regime_df = None
        if hsi_df is not None and not hsi_df.empty:
            try:
                print("  计算 HSI 市场状态特征...")
                regime_detector = RegimeDetector()
                hsi_with_regime = regime_detector.calculate_features(hsi_df.copy())
                rename_map = {c: f'HSI_{c}' for c in RegimeDetector.get_feature_names()}
                hsi_regime_df = hsi_with_regime[RegimeDetector.get_feature_names()].rename(columns=rename_map)
                print("  ✅ HSI 市场状态特征计算完成")
            except Exception as e:
                print(f"  ⚠️ HSI 市场状态特征计算失败: {e}")

        for code in codes:
            try:
                print(f"处理股票: {code}")

                # 移除代码中的.HK后缀，腾讯财经接口不需要
                stock_code = code.replace('.HK', '')

                # 获取股票数据（2年约730天）
                stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
                if stock_df is None or stock_df.empty:
                    continue

                if hsi_df is None or hsi_df.empty:
                    continue

                # 计算技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 计算多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 计算相对强度指标
                stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                # 合并 HSI 市场状态特征
                if hsi_regime_df is not None:
                    stock_df = self.feature_engineer.calculate_hsi_regime_features(stock_df, hsi_regime_df)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                # 创建标签（使用指定的 horizon）
                
                # 添加股票类型特征
                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value
                stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest, min_return_threshold=min_return_threshold)

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

        # 计算截面特征（必须在合并所有股票后计算）
        if self.use_cross_sectional_percentile:
            print("\n📊 计算截面百分位特征...")
            df = self._calculate_cross_sectional_percentile_features(df)

        if self.use_cross_sectional_zscore:
            print("\n📊 计算截面Z-Score特征...")
            df = self._calculate_cross_sectional_zscore_features(df)

        # 生成技术指标与基本面交互特征（先执行，因为这是高价值特征）
        print("\n🔗 生成技术指标与基本面交互特征...")
        df = self.feature_engineer.create_technical_fundamental_interactions(df)

        # 生成交叉特征（类别型 × 数值型）
        print("\n🔗 生成交叉特征...")
        df = self.feature_engineer.create_interaction_features(df)

        return df

    def _calculate_cross_sectional_percentile_features(self, df):
        """计算截面百分位特征（在所有股票上联合计算）

        Args:
            df: 包含多只股票数据的DataFrame（必须含'Code'列）

        Returns:
            DataFrame: 添加了截面百分位特征的DataFrame
        """
        if 'Code' not in df.columns:
            logger.warning("数据中没有'Code'列，无法计算截面百分位特征")
            return df

        df = df.copy()

        for feature in self.CROSS_SECTIONAL_PERCENTILE_FEATURES:
            if feature not in df.columns:
                continue

            cs_col = f'{feature}_CS_Pct'
            if cs_col in df.columns:
                continue

            try:
                df[cs_col] = df.groupby(df.index)[feature].transform(
                    lambda x: (x.rank(method='average') - 1) / max(len(x) - 1, 1) if len(x) > 1 else 0.5
                )
            except Exception as e:
                logger.debug(f"截面百分位特征计算失败 {feature}: {e}")

        return df

    def _calculate_cross_sectional_zscore_features(self, df):
        """计算截面Z-Score特征（在所有股票上联合计算）

        Args:
            df: 包含多只股票数据的DataFrame（必须含'Code'列）

        Returns:
            DataFrame: 添加了截面Z-Score特征的DataFrame
        """
        if 'Code' not in df.columns:
            logger.warning("数据中没有'Code'列，无法计算截面Z-Score特征")
            return df

        df = df.copy()

        for feature in self.CROSS_SECTIONAL_ZSCORE_FEATURES:
            if feature not in df.columns:
                continue

            cs_col = f'{feature}_CS_ZScore'
            if cs_col in df.columns:
                continue

            try:
                df[cs_col] = df.groupby(df.index)[feature].transform(
                    lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0.0
                )
            except Exception as e:
                logger.debug(f"截面Z-Score特征计算失败 {feature}: {e}")

        return df

    def get_feature_columns(self, df):
        """获取特征列"""
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False, min_return_threshold=0.0):
        """训练 GBDT 模型（默认使用全量特征892个）

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（已弃用，默认False使用全量特征）
            min_return_threshold: 最小收益阈值（默认0%）
        """
        # 设置固定随机种子（确保模型训练的可重现性）
        np.random.seed(42)
        random.seed(42)
        # 检查是否需要显示弃用警告
        if use_feature_selection and not GBDTModel._deprecation_warning_shown:
            print("⚠️  警告：特征选择功能已弃用，建议使用全量特征（892个）。Walk-forward验证显示全量特征性能更好。")
            GBDTModel._deprecation_warning_shown = True

        print("="*70)
        logger.info("开始训练 GBDT 模型")
        print("="*70)

        # 准备数据
        logger.info("准备训练数据...")
        df = self.prepare_data(codes, start_date, end_date, horizon=horizon,
                               min_return_threshold=min_return_threshold,
                               label_type='relative')  # 个股选股模型使用相对标签

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
            print("\n🎯 应用特征选择（GBDT）...（已弃用）")
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)
            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                print(f"✅ 特征数量: {len(self.feature_columns)}（特征选择 - 已弃用）")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")
        else:
            print(f"\n✅ 特征数量: {len(self.feature_columns)}（全量特征）")

        # 对Market_Regime进行One-Hot编码（GBDT专用）
        if 'Market_Regime' in df.columns:
            print("  对Market_Regime进行One-Hot编码(GBDT)...")
            df = pd.get_dummies(df, columns=['Market_Regime'], prefix='Market_Regime')
            # 更新feature_columns
            self.feature_columns = [col for col in self.feature_columns if col != 'Market_Regime']
            self.feature_columns.extend([col for col in df.columns if col.startswith('Market_Regime_')])
            print(f"  One-Hot编码后特征数量: {len(self.feature_columns)}")

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
            # 原因：特征数量约2426个（基础892 + 交叉特征1534），需要更强的正则化防止过拟合
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

        # 使用时间序列交叉验证（添加 gap 参数避免短期依赖）
        tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
        gbdt_scores = []
        gbdt_f1_scores = []

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
            f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
            gbdt_scores.append(score)
            gbdt_f1_scores.append(f1)
            print(f"   Fold {fold} 验证准确率: {score:.4f}, 验证F1分数: {f1:.4f}")

        # 使用全部数据重新训练
        self.gbdt_model.fit(X, y)

        # 获取实际训练的树数量
        # 注意：在使用全部数据重新训练时，如果没有使用早停，best_iteration_ 可能为 None
        # 这种情况下使用 n_estimators
        self.actual_n_estimators = self.gbdt_model.best_iteration_ if self.gbdt_model.best_iteration_ else n_estimators
        mean_accuracy = np.mean(gbdt_scores)
        std_accuracy = np.std(gbdt_scores)
        mean_f1 = np.mean(gbdt_f1_scores)
        std_f1 = np.std(gbdt_f1_scores)
        print(f"\n✅ GBDT 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators} (原计划: {n_estimators})")
        print(f"   平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"   平均验证F1分数: {mean_f1:.4f} (+/- {std_f1:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'gbdt',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'f1_score': float(mean_f1),
            'f1_std': float(std_f1),
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
            feat_imp.to_csv('output/ml_trading_model_gbdt_20d_importance.csv', index=False)
            logger.info(r"已保存特征重要性至 output/ml_trading_model_gbdt_20d_importance.csv")

            # 显示前20个重要特征
            print("\n📊 GBDT Top 20 重要特征 (含影响方向):")
            print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(20))

        except Exception as e:
            logger.warning(f"特征贡献分析失败: {e}")
            feat_imp['Impact_Direction'] = 'Unknown'

        print("\n" + "="*70)
        logger.info(r"GBDT 模型训练完成！")
        print("="*70)

        # 保存截面特征的训练集统计量，供单只股票预测时回退使用
        self.cs_feature_stats = {}
        for col in df.columns:
            if col.endswith('_CS_Pct') or col.endswith('_CS_ZScore'):
                self.cs_feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }

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
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据
            hsi_df = get_hsi_data_tencent(period_days=1460)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

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

            # 添加主题情感交互特征（移到循环外，避免重复调用）
            topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
            for key, value in topic_sentiment_interaction.items():
                stock_df[key] = value

            # 添加预期差距特征（移到循环外，避免重复调用）
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

    def _extract_raw_features_single(self, code, predict_date=None, horizon=None):
        """提取单只股票的原始特征（不含截面特征），供批量预测使用

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期

        Returns:
            DataFrame: 带特征的 stock_df（含 Code 列），或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            stock_code = code.replace('.HK', '')

            # 获取股票数据
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 获取恒生指数数据
            hsi_df = get_hsi_data_tencent(period_days=1460)
            if hsi_df is None or hsi_df.empty:
                return None

            # 获取美股市场数据
            us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)
                if not isinstance(hsi_df.index, pd.DatetimeIndex):
                    hsi_df.index = pd.to_datetime(hsi_df.index)
                if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                    us_market_df.index = pd.to_datetime(us_market_df.index)

                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                if us_market_df is not None:
                    us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 计算技术指标
            stock_df = self.feature_engineer.calculate_technical_features(stock_df)
            stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)
            stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)
            stock_df = self.feature_engineer.create_smart_money_features(stock_df)
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

            # 添加主题特征
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

            # 生成交互特征
            stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)
            stock_df = self.feature_engineer.create_interaction_features(stock_df)

            # 添加股票代码列
            stock_df['Code'] = code

            return stock_df

        except Exception as e:
            logger.warning(f"提取特征失败 {code}: {e}")
            return None

    def _predict_from_features(self, code, latest_data, horizon=None):
        """从特征数据进行预测（供 predict() 和 predict_batch() 共用）

        Args:
            code: 股票代码
            latest_data: 单行 DataFrame，包含所有特征
            horizon: 预测周期

        Returns:
            dict: 预测结果，或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 处理缺失的截面特征（使用训练集统计量回退）
            cs_feature_suffixes = ['_CS_Pct', '_CS_ZScore']
            for suffix in cs_feature_suffixes:
                cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
                for cs_feat in cs_features:
                    if cs_feat not in latest_data.columns:
                        if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                            latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
                        else:
                            latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0

            # 处理分类特征
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        latest_data[col] = encoder.transform(latest_data[col].astype(str))
                    except ValueError:
                        logger.warning(f"分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

            X = latest_data[self.feature_columns].values

            # 预测
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
            logger.warning(f"预测失败 {code}: {e}")
            return None

    def predict_batch(self, codes, predict_date=None, horizon=None):
        """批量预测：先提取所有股票特征，再统一计算截面特征，最后逐只预测

        核心改进：截面特征（_CS_Pct, _CS_ZScore）在所有股票数据上联合计算，
        确保训练/预测一致，而非单只股票时退化为 0.5/0.0。

        Args:
            codes: 股票代码列表
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期

        Returns:
            list: 预测结果列表
        """
        if horizon is None:
            horizon = self.horizon

        if len(self.feature_columns) == 0:
            raise ValueError("模型未训练，请先调用train()方法")

        logger.info(f"GBDT 开始批量预测 {len(codes)} 只股票...")

        # 阶段1：逐只提取原始特征
        all_features = {}
        for code in codes:
            stock_df = self._extract_raw_features_single(code, predict_date, horizon)
            if stock_df is not None:
                all_features[code] = stock_df

        if not all_features:
            logger.warning("GBDT 批量预测：没有成功提取任何股票的特征")
            return []

        logger.info(f"GBDT 成功提取 {len(all_features)} 只股票的特征")

        # 阶段2：合并所有股票，计算截面特征
        combined = pd.concat(all_features.values())

        # 统一索引时区：将 tz-aware 时间戳转换为 tz-naive，避免 groupby 时比较错误
        if hasattr(combined.index, 'tz') and combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        elif hasattr(combined.index, 'tz_localize'):
            try:
                combined.index = combined.index.tz_convert('UTC').tz_localize(None)
            except Exception:
                combined.index = pd.to_datetime(combined.index).tz_localize(None)

        # 计算截面百分位特征
        if self.use_cross_sectional_percentile:
            combined = self._calculate_cross_sectional_percentile_features(combined)
            logger.info("GBDT 批量预测：截面百分位特征计算完成")

        # 计算截面 Z-Score 特征
        if self.use_cross_sectional_zscore:
            combined = self._calculate_cross_sectional_zscore_features(combined)
            logger.info("GBDT 批量预测：截面 Z-Score 特征计算完成")

        # 阶段3：逐只预测（使用正确的截面特征）
        results = []
        for code in all_features.keys():
            stock_data = combined[combined['Code'] == code]
            if stock_data.empty:
                continue

            latest = stock_data.iloc[-1:].copy()

            # 使用辅助方法进行预测
            result = self._predict_from_features(code, latest, horizon)
            if result:
                results.append(result)

        logger.info(f"GBDT 批量预测完成：{len(results)}/{len(codes)} 只股票")
        return results

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'gbdt_model': self.gbdt_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'categorical_encoders': self.categorical_encoders,
            'use_cross_sectional_percentile': self.use_cross_sectional_percentile,
            'use_cross_sectional_zscore': self.use_cross_sectional_zscore,
            'cs_feature_stats': getattr(self, 'cs_feature_stats', {})
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
        self.use_cross_sectional_percentile = model_data.get('use_cross_sectional_percentile', True)
        self.use_cross_sectional_zscore = model_data.get('use_cross_sectional_zscore', True)
        self.cs_feature_stats = model_data.get('cs_feature_stats', {})
        print(f"GBDT 模型已从 {filepath} 加载")


class CatBoostModel(BaseTradingModel):
    """CatBoost 模型 - 基于 CatBoost 梯度提升算法的单一模型

    CatBoost 是 Yandex 开发的梯度提升库，具有以下优势：
    1. 自动处理分类特征，无需手动编码
    2. 更好的默认参数，减少调参工作量
    3. 更快的训练速度（GPU 支持）
    4. 更好的泛化能力，减少过拟合
    """
    _deprecation_warning_shown = False  # 类变量，控制弃用警告只显示一次

    # 单调约束映射：特征名 → 约束方向
    # -1: 递减约束（特征增大 → 预测概率减小）
    # +1: 递增约束（特征增大 → 预测概率增大）
    # 0: 无约束（逻辑不明确或依赖市场环境）
    #
    # 优化原则（2026-05-03 基于业界实践）：
    # 1. 只对因果关系明确的特征使用约束
    # 2. 波动率约束移除：相对标签模型下方向不稳定（牛市+1，熊市-1）
    # 3. 情感约束移除：可能是反向指标（情感高 = 过度乐观）
    # 4. 股息约束移除：港股股息溢价不明显
    #
    MONOTONE_CONSTRAINT_MAP = {
        # === 相对强度特征：+1（RS↑ → 超额收益↑）===
        # RS_Ratio = (1+股票收益)/(1+恒指收益) - 1
        # 含义：股票相对恒指的强度，约束方向正确
        'RS_Ratio_5d': +1,
        'RS_Ratio_20d': +1,

        # === RSI特征：-1（超买超卖）===
        # RSI高 → 超买 → 可能下跌
        'RSI': -1,

        # === MACD特征：+1（趋势）===
        # MACD柱正 → 上涨趋势
        'MACD_histogram': +1,
    }

    # 市场级特征列表（2026-05-03 新增）
    # 这些特征在同日所有股票值相同，对选股（相对标签）无区分力，反而导致模型变成宏观择时器
    # 仅排除"裸"市场级特征，保留交叉特征（如 10d_Trend_HSI_Return_60d，分类乘数使不同股票值不同）
    MARKET_LEVEL_FEATURES = [
        # 恒指收益（同日所有股票值相同）
        'HSI_Return_1d', 'HSI_Return_3d', 'HSI_Return_5d',
        'HSI_Return_10d', 'HSI_Return_20d', 'HSI_Return_60d',
        # HSI 市场状态
        'HSI_Market_Regime', 'HSI_Regime_Prob_0', 'HSI_Regime_Prob_1',
        'HSI_Regime_Prob_2', 'HSI_Regime_Duration', 'HSI_Regime_Transition_Prob',
        # 美股
        'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
        'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
        # VIX
        'VIX_Level', 'VIX_Change', 'VIX_Ratio_MA20',
        # 美债
        'US_10Y_Yield', 'US_10Y_Yield_Change',
        'US10Y_Yield', 'US10Y_Yield_Change_5d',
        # 市场状态 one-hot
        'Market_Regime_Ranging', 'Market_Regime_Normal', 'Market_Regime_Trending',
    ]

    # P6: 宏交叉特征排除列表（2026-05-04 新增）
    # 这些特征虽然在不同股票间有微小差异（通过 Trend 分类乘数），但核心信息仍是宏观环境
    # 模型会"偷懒"用这些特征做宏观拟合，而非学习个股 Alpha
    # 排除后强迫模型通过 Momentum/Network 特征寻找个股差异
    MACRO_CROSS_FEATURES = [
        # 趋势 × 恒指收益（Top 2 特征，噪音霸主）
        '10d_Trend_HSI_Return_60d', '20d_Trend_HSI_Return_60d', '60d_Trend_HSI_Return_60d',
        '10d_Trend_HSI_Return_20d', '20d_Trend_HSI_Return_20d', '60d_Trend_HSI_Return_20d',
        '10d_Trend_HSI_Return_5d', '20d_Trend_HSI_Return_5d', '60d_Trend_HSI_Return_5d',
        # 趋势 × 市场状态（Top 5 特征）
        '10d_Trend_HSI_Regime_Prob_0', '10d_Trend_HSI_Regime_Prob_1', '10d_Trend_HSI_Regime_Prob_2',
        '20d_Trend_HSI_Regime_Prob_0', '20d_Trend_HSI_Regime_Prob_1', '20d_Trend_HSI_Regime_Prob_2',
        '60d_Trend_HSI_Regime_Prob_0', '60d_Trend_HSI_Regime_Prob_1', '60d_Trend_HSI_Regime_Prob_2',
        # 趋势 × 美股
        '10d_Trend_SP500_Return', '20d_Trend_SP500_Return', '60d_Trend_SP500_Return',
        '10d_Trend_SP500_Return_5d', '20d_Trend_SP500_Return_5d', '60d_Trend_SP500_Return_5d',
        '10d_Trend_NASDAQ_Return', '20d_Trend_NASDAQ_Return', '60d_Trend_NASDAQ_Return',
        '10d_Trend_NASDAQ_Return_5d', '20d_Trend_NASDAQ_Return_5d', '60d_Trend_NASDAQ_Return_5d',
    ]

    # 滚动百分位特征列表（原始特征 → 百分位特征）
    # 原则：波动率、成交量、ATR等绝对量级特征受益于百分位化
    # 宏观特征（VIX, US_10Y_Yield）和已相对化特征（RS_Ratio, BB_Position, sector_rank）不转换
    ROLLING_PERCENTILE_FEATURES = [
        # 波动率特征
        'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
        'Volatility_60d', 'Volatility_120d',
        'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio',
        # ATR特征
        'ATR', 'ATR_Ratio', 'ATR_MA60', 'ATR_MA120',
        # 日内波动特征
        'Intraday_Range', 'Intraday_Range_MA5', 'Intraday_Range_MA20',
        # 成交量特征
        'Volume_Volatility', 'Volume_Ratio_5d', 'Volume_Ratio_20d',
    ]

    # 截面百分位特征列表（P7 精简版，2026-05-04）
    # 从 P6 的 ~100 个精简回核心高质量特征，去除高相关性冗余
    # 原则：质量 > 数量，保留核心 Alpha 信号
    CROSS_SECTIONAL_PERCENTILE_FEATURES = [
        # ========== 波动率特征（保留核心，去除冗余）==========
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',  # 短中长期
        'GARCH_Conditional_Vol', 'Intraday_Range',  # 条件波动率、日内波幅
        # ========== ATR特征（保留核心）==========
        'ATR', 'ATR_Ratio', 'ATR_Risk_Score',
        # ========== 成交量特征（保留核心）==========
        'Volume_Ratio_5d', 'Volume_Ratio_20d', 'Volume_Volatility',
        'OBV', 'CMF',  # 资金流向指标
        # ========== 动量特征（保留核心，去除高相关变体）==========
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_20d',
        'MACD_histogram', 'Close_Position',
        # ========== RSI特征（保留核心）==========
        'RSI', 'RSI_Deviation',
        # ========== 相对强度特征（关键 Alpha 信号）==========
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return',
        # ========== 板块相对动能特征（P4 新增，2026-05-06）==========
        'Sector_Relative_Momentum_5d', 'Sector_Relative_Momentum_20d',
        # ========== 布林带/位置特征 ==========
        'BB_Position', 'BB_Width',
        # ========== 风险特征（保留核心）==========
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Skewness_20d',
        # ========== 资金流向特征 ==========
        'Smart_Money_Score', 'Accumulation_Score',
        # ========== 基本面特征 ==========
        'PE', 'PB', 'ROE', 'Market_Cap',
        # ========== 异常检测特征（关键 Alpha 信号）==========
        'Anomaly_Severity_Score', 'Anomaly_Buy_Signal',
        # ========== 网络增量特征（P7 新增）==========
        'net_node_deviation', 'net_node_deviation_delta_5d',
    ]

    # 截面 Z-Score 特征列表（P7 精简版，2026-05-04）
    # 对关键特征进行截面标准化，确保特征在截面内可比
    CROSS_SECTIONAL_ZSCORE_FEATURES = [
        # 成交量特征（时间序列 Z-Score 基准问题最严重）
        'Volume', 'Turnover', 'Turnover_Mean_20',
        # 波动率特征
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',
        'GARCH_Conditional_Vol',
        # ATR特征
        'ATR', 'ATR_Ratio',
        # 成交量比率特征
        'Volume_Ratio_5d', 'Volume_Ratio_20d',
        'OBV', 'CMF',
        # 动量特征
        'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_20d',
        'MACD_histogram',
        # RSI特征
        'RSI', 'RSI_Deviation',
        # 相对强度特征
        'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
        'Relative_Return',
        # 布林带/位置特征
        'BB_Position', 'BB_Width',
        # 风险特征
        'Max_Drawdown_20d', 'Max_Drawdown_60d',
        'Vol_Z_Score', 'Kurtosis_20d', 'Skewness_20d',
        # 资金流向特征
        'Smart_Money_Score', 'Accumulation_Score',
        # 基本面特征
        'PE', 'PB', 'ROE', 'Market_Cap',
        # 网络增量特征（P7 新增）
        'net_node_deviation', 'net_node_deviation_delta_5d',
    ]

    # ========== P9 阶段：特征分层策略 ==========
    # L1 核心截面特征：选股核心信号，与相对标签逻辑一致
    L1_CORE_CS_FEATURES = [
        # ========== 资金流向（选股核心）==========
        'CMF_CS_Pct', 'CMF_CS_ZScore',
        'OBV_CS_Pct', 'OBV_CS_ZScore',
        'Smart_Money_Score_CS_Pct', 'Smart_Money_Score_CS_ZScore',
        'Accumulation_Score_CS_Pct', 'Accumulation_Score_CS_ZScore',
        # ========== 动量（选股核心）==========
        'Momentum_20d_CS_Pct', 'Momentum_20d_CS_ZScore',
        'Momentum_Accel_5d_CS_Pct', 'Momentum_Accel_5d_CS_ZScore',
        'MACD_histogram_CS_Pct', 'MACD_histogram_CS_ZScore',
        # ========== 相对强度（关键 Alpha）==========
        'RS_Ratio_5d_CS_Pct', 'RS_Ratio_5d_CS_ZScore',
        'RS_Ratio_20d_CS_Pct', 'RS_Ratio_20d_CS_ZScore',
        'RS_Diff_5d_CS_Pct', 'RS_Diff_5d_CS_ZScore',
        'RS_Diff_20d_CS_Pct', 'RS_Diff_20d_CS_ZScore',
        'Relative_Return_CS_Pct', 'Relative_Return_CS_ZScore',
        # ========== 波动率（风险调整）==========
        'Volatility_20d_CS_Pct', 'Volatility_20d_CS_ZScore',
        'ATR_Ratio_CS_Pct', 'ATR_Ratio_CS_ZScore',
        # ========== 风险特征 ==========
        'Max_Drawdown_20d_CS_Pct', 'Max_Drawdown_20d_CS_ZScore',
        'Kurtosis_20d_CS_Pct', 'Kurtosis_20d_CS_ZScore',
        'Skewness_20d_CS_Pct', 'Skewness_20d_CS_ZScore',
        # ========== 基本面 ==========
        'PE_CS_Pct', 'PE_CS_ZScore',
        'PB_CS_Pct', 'PB_CS_ZScore',
        'ROE_CS_Pct', 'ROE_CS_ZScore',
    ]

    # L3 剔除特征：市场级 + 宏观交叉特征（干扰选股）
    L3_EXCLUDE_FEATURES = [
        # ========== 纯市场特征（同日所有股票值相同）==========
        'US_10Y_Yield', 'US_10Y_Yield_Change',
        'US10Y_Yield', 'US10Y_Yield_Change_5d',
        'VIX_Level', 'VIX_Change', 'VIX_Ratio_MA20',
        'HSI_Return_1d', 'HSI_Return_3d', 'HSI_Return_5d',
        'HSI_Return_10d', 'HSI_Return_20d', 'HSI_Return_60d',
        'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
        'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
        'HSI_Market_Regime', 'HSI_Regime_Prob_0', 'HSI_Regime_Prob_1',
        'HSI_Regime_Prob_2', 'HSI_Regime_Duration', 'HSI_Regime_Transition_Prob',
        'Market_Regime_Ranging', 'Market_Regime_Normal', 'Market_Regime_Trending',
        # ========== 宏观交叉特征（含市场成分）==========
        # Trend × 市场收益
        '5d_Trend_HSI_Return_5d', '10d_Trend_HSI_Return_10d',
        '20d_Trend_HSI_Return_20d', '60d_Trend_HSI_Return_60d',
        # Trend × 市场状态
        '5d_Trend_HSI_Regime_Prob_0', '5d_Trend_HSI_Regime_Prob_1',
        '10d_Trend_HSI_Regime_Prob_0', '10d_Trend_HSI_Regime_Prob_1',
        '20d_Trend_HSI_Regime_Prob_0', '20d_Trend_HSI_Regime_Prob_1',
        # Trend × 美股
        '5d_Trend_SP500_Return_5d', '10d_Trend_SP500_Return_10d',
        '20d_Trend_SP500_Return_20d',
        '5d_Trend_NASDAQ_Return_5d', '10d_Trend_NASDAQ_Return_10d',
        '20d_Trend_NASDAQ_Return_20d',
    ]

    def __init__(self, class_weight='balanced', use_dynamic_threshold=False,
                 use_monotone_constraints=True, time_decay_lambda=0.5,
                 use_rolling_percentile=False,  # 2026-05-02: 关闭滚动百分位（消融实验证明其降低IC）
                 use_cross_sectional_percentile=True,  # 2026-05-03: 截面百分位（与相对标签匹配）
                 use_cross_sectional_zscore=True,  # 2026-05-03: 截面 Z-Score（解决时间序列基准问题）
                 feature_importance_threshold=0.0):  # P3-8: 特征修剪阈值（0=不修剪）
        """初始化 CatBoost 模型

        Args:
            class_weight: 类别权重策略
                - 'balanced': 自动平衡类别权重（推荐，温和调整）
                - 'balanced_subsample': 每棵树的子样本中平衡
                - None: 不使用类别权重
                - dict: 手动指定权重，如 {0: 1.0, 1: 1.2}
            use_dynamic_threshold: 是否使用动态阈值策略
            use_monotone_constraints: 是否使用单调约束（防止特征方向翻转，推荐开启）
            time_decay_lambda: 时间衰减系数（0=无衰减，0.5=默认，1.0=强衰减）
            use_rolling_percentile: 是否使用滚动百分位特征（已关闭，消融实验证明降低IC）
            use_cross_sectional_percentile: 是否使用截面百分位特征（与相对标签匹配）
            use_cross_sectional_zscore: 是否使用截面 Z-Score 特征（解决时间序列基准问题）
            feature_importance_threshold: 特征重要性阈值，低于此值的特征将被移除（0=不修剪）
        """
        super().__init__()  # 调用基类初始化
        self.catboost_model = None
        self.actual_n_estimators = 0
        self.model_type = 'catboost'  # 模型类型标识
        self.class_weight = class_weight
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_monotone_constraints = use_monotone_constraints
        self.time_decay_lambda = time_decay_lambda
        self.use_rolling_percentile = use_rolling_percentile
        self.use_cross_sectional_percentile = use_cross_sectional_percentile
        self.use_cross_sectional_zscore = use_cross_sectional_zscore
        self.feature_importance_threshold = feature_importance_threshold  # P3-8
        self.monotone_constraints_list = None  # 训练时填充
        self.sample_weights = None  # 时间衰减权重

        logger.info(f"CatBoostModel 初始化: class_weight={class_weight}, "
                    f"use_monotone_constraints={use_monotone_constraints}, "
                    f"time_decay_lambda={time_decay_lambda}, "
                    f"use_rolling_percentile={use_rolling_percentile}, "
                    f"use_cross_sectional_percentile={use_cross_sectional_percentile}, "
                    f"use_cross_sectional_zscore={use_cross_sectional_zscore}, "
                    f"feature_importance_threshold={feature_importance_threshold}")

    def _build_monotone_constraints(self, feature_columns):
        """根据特征名称构建单调约束列表

        Args:
            feature_columns: 特征列名列表（动态生成）

        Returns:
            list: 单调约束列表（-1/0/+1），与 feature_columns 一一对应
        """
        if not self.use_monotone_constraints:
            return None

        constraints = []
        matched = 0
        for col in feature_columns:
            # 精确匹配
            if col in self.MONOTONE_CONSTRAINT_MAP:
                constraints.append(self.MONOTONE_CONSTRAINT_MAP[col])
                matched += 1
            else:
                constraints.append(0)  # 未匹配的默认无约束

        logger.info(f"单调约束：{matched}/{len(feature_columns)} 个特征被约束 "
                    f"(+1={constraints.count(1)}, -1={constraints.count(-1)}, "
                    f"0={constraints.count(0)})")

        self.monotone_constraints_list = constraints
        return constraints

    def _compute_time_decay_weights(self, df, lambda_decay=0.5):
        """计算时间衰减样本权重

        公式: W_i = exp(-lambda * delta_t_years)
        其中 delta_t_years = (最新日期 - 当前样本日期) / 252

        Args:
            df: 包含时间索引的 DataFrame
            lambda_decay: 衰减系数（0=无衰减，0.5=温和衰减，1.0=强衰减）

        Returns:
            numpy.ndarray: 样本权重数组（归一化至均值=1）
        """
        if lambda_decay <= 0 or df.empty:
            return None

        # 获取日期索引
        dates = df.index
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)

        # 计算距最新日期的年数
        max_date = dates.max()
        delta_days = (max_date - dates).total_seconds() / (24 * 3600)
        # 转换为 numpy 数组，避免 Index 对象的问题
        delta_days = np.array(delta_days)
        delta_years = delta_days / 252.0  # 交易日年化

        # 指数衰减
        weights = np.exp(-lambda_decay * delta_years)

        # 归一化至均值=1（保持与无权重时的期望损失等价）
        weights = weights / np.mean(weights)

        # 日志输出
        oldest_weight = weights.min()
        newest_weight = weights.max()
        logger.info(f"时间衰减权重: lambda={lambda_decay}, "
                    f"最旧样本权重={oldest_weight:.3f}, 最新样本权重={newest_weight:.3f}, "
                    f"比值={newest_weight/max(oldest_weight, 1e-10):.2f}x")

        return weights

    def _calculate_rolling_percentile_features(self, df, window=252):
        """计算滚动百分位特征（时间序列自归一化）

        将绝对量级特征转换为历史百分位，使特征在regime变化时保持稳定。
        公式: feature_pct = feature.rolling(window).rank(pct=True)

        Args:
            df: 股票数据 DataFrame
            window: 滚动窗口（默认252个交易日=1年）

        Returns:
            DataFrame: 包含百分位特征的 DataFrame（原始特征被替换）
        """
        if not self.use_rolling_percentile:
            return df

        features_to_convert = self.ROLLING_PERCENTILE_FEATURES
        converted = 0

        for feat in features_to_convert:
            if feat not in df.columns:
                continue

            # 滚动百分位：当前值在过去window天中的排名百分位
            # 使用 rank(pct=True) 返回 0-1 之间的值
            pct_values = df[feat].rolling(window, min_periods=20).rank(pct=True)

            # 填充开头NaN为0.5（中位数，保守假设）
            pct_values = pct_values.fillna(0.5)

            # 直接替换原始特征
            df[feat] = pct_values
            converted += 1

        if converted > 0:
            logger.info(f"滚动百分位特征：转换了 {converted} 个特征（窗口={window}天）")

        return df

    def _calculate_cross_sectional_percentile_features(self, df):
        """计算截面百分位特征（2026-05-03 新增）

        核心思想：
        - 每天对所有股票进行排名，计算当日排名百分位
        - 与相对标签逻辑一致：判断"这只股票今天在所有股票中排第几"
        - 适用场景：截面选股（与相对标签匹配）

        对比滚动百分位：
        - 滚动百分位：df[feat].rolling(252).rank(pct=True) — 比较历史值
        - 截面百分位：df.groupby('Date')[feat].rank(pct=True) — 比较当日其他股票

        Args:
            df: 数据 DataFrame，索引为日期，包含 'Code' 列

        Returns:
            DataFrame: 包含新增的截面百分位特征（_CS_Pct 后缀）
        """
        if not self.use_cross_sectional_percentile:
            return df

        features_to_convert = self.CROSS_SECTIONAL_PERCENTILE_FEATURES
        converted = 0

        for feat in features_to_convert:
            if feat not in df.columns:
                continue

            # 截面百分位：按日期分组，计算当日排名百分位
            # 结果范围：0-1，0.5 表示当日中位数
            cs_pct = df.groupby(df.index)[feat].rank(pct=True)

            # 新增特征列（不替换原始特征）
            df[f'{feat}_CS_Pct'] = cs_pct
            converted += 1

        if converted > 0:
            logger.info(f"截面百分位特征：新增 {converted} 个特征（_CS_Pct 后缀）")

        return df

    def _calculate_cross_sectional_zscore_features(self, df):
        """计算截面 Z-Score 特征（2026-05-03 新增）

        核心思想：
        - 每天对所有股票进行 Z-Score 标准化
        - 确保特征在截面内可比，避免时间序列标准化的基准问题
        - 与相对标签逻辑一致：在当天所有股票范围内比较

        为什么需要截面 Z-Score：
        - 时间序列 Z-Score 使用该股票历史均值/标准差，不同股票基准不同
        - 港股存在均值回归，时间序列标准化会认错高低位
        - 截面 Z-Score 确保模型学习的是"这只票今天比别的票强/弱多少个标准差"

        公式：
        Z_i = (X_i - mean(X_day)) / std(X_day)

        Args:
            df: 数据 DataFrame，索引为日期，包含 'Code' 列

        Returns:
            DataFrame: 包含新增的截面 Z-Score 特征（_CS_ZScore 后缀）
        """
        if not self.use_cross_sectional_zscore:
            return df

        features_to_convert = self.CROSS_SECTIONAL_ZSCORE_FEATURES
        converted = 0

        for feat in features_to_convert:
            if feat not in df.columns:
                continue

            # 截面 Z-Score：按日期分组，计算当天所有股票的 Z-Score
            # 结果范围：约 -3 到 +3，0 表示当天中位数
            def zscore_safe(x):
                std = x.std()
                if std > 1e-10:
                    return (x - x.mean()) / std
                else:
                    return 0.0  # 常量特征返回 0

            cs_zscore = df.groupby(df.index)[feat].transform(zscore_safe)

            # 新增特征列（不替换原始特征）
            df[f'{feat}_CS_ZScore'] = cs_zscore
            converted += 1

        if converted > 0:
            logger.info(f"截面 Z-Score 特征：新增 {converted} 个特征（_CS_ZScore 后缀）")

        return df

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
            # 支持多种文件格式和命名
            patterns = [
                'output/selected_features_*.csv',          # 统计方法（CSV格式）
                'output/statistical_features_*.txt',       # 统计方法（TXT格式）
                'output/model_importance_selected_*.csv',  # 模型重要性法（CSV格式）
                'output/model_importance_features_*.txt',  # 模型重要性法（TXT格式）
                'output/statistical_features_latest.txt',   # 最新统计特征
                'output/model_importance_features_latest.txt'  # 最新模型重要性特征
            ]
            
            files = []
            for pattern in patterns:
                found_files = glob.glob(pattern)
                files.extend(found_files)
                if found_files:
                    break  # 找到文件就停止
            
            if not files:
                return None
            # 按修改时间排序，取最新的
            filepath = max(files, key=os.path.getmtime)

        try:
            selected_names = []
            
            # 根据文件扩展名选择不同的读取方式
            if filepath.endswith('.csv'):
                import pandas as pd
                # 读取特征名称
                df = pd.read_csv(filepath)
                selected_names = df['Feature_Name'].tolist()
            elif filepath.endswith('.txt'):
                # 读取TXT文件（每行一个特征名称）
                with open(filepath, 'r', encoding='utf-8') as f:
                    selected_names = [line.strip() for line in f if line.strip()]
            else:
                logger.error(f"不支持的文件格式: {filepath}")
                return None

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
                if len(selected_set) - len(available_set) > 0:
                    logger.warning(f" {len(selected_set) - len(available_set)} 个特征在当前数据集中不存在")
                
                return available_names
            else:
                return selected_names

        except Exception as e:
            logger.warning(f"加载特征列表失败: {e}")
            return None

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False, min_return_threshold=0.0, use_feature_cache=True, label_type='relative'):
        """准备训练数据

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            for_backtest: 是否为回测准备数据（True时不应用horizon过滤）
            min_return_threshold: 最小收益阈值（默认0%），用于标签定义
            use_feature_cache: 是否使用特征缓存（默认True）
            label_type: 标签类型（'absolute'=绝对标签，'relative'=相对标签，默认'relative'）
        """
        self.horizon = horizon
        self.min_return_threshold = min_return_threshold
        self.label_type = label_type
        all_data = []

        # 获取美股市场数据（只获取一次）
        logger.info("获取美股市场数据...")
        us_market_df = us_market_data.get_all_us_market_data(period_days=1460)
        if us_market_df is not None:
            logger.info(f"成功获取 {len(us_market_df)} 天的美股市场数据")
        else:
            logger.warning(r"无法获取美股市场数据，将只使用港股特征")

        # 获取恒生指数数据（只获取一次，用于缓存键）
        logger.info("获取恒生指数数据...")
        hsi_df = get_hsi_data_tencent(period_days=1460)
        if hsi_df is None or hsi_df.empty:
            logger.warning("无法获取恒生指数数据")
            hsi_df = None

        # 计算 HSI 市场状态特征（一次性，所有股票共享）
        hsi_regime_df = None
        if hsi_df is not None and not hsi_df.empty:
            try:
                print("  计算 HSI 市场状态特征...")
                regime_detector = RegimeDetector()
                hsi_with_regime = regime_detector.calculate_features(hsi_df.copy())
                rename_map = {c: f'HSI_{c}' for c in RegimeDetector.get_feature_names()}
                hsi_regime_df = hsi_with_regime[RegimeDetector.get_feature_names()].rename(columns=rename_map)
                print("  ✅ HSI 市场状态特征计算完成")
            except Exception as e:
                print(f"  ⚠️ HSI 市场状态特征计算失败: {e}")

        # 加载网络特征（跨截面特征，所有股票共享）
        # 已移除：网络特征存在数据泄漏风险，作为独立分析工具使用
        # network_loader = NetworkFeatureLoader()
        # network_available = network_loader.is_available()
        # if network_available:
        #     network_loader.load_features()
        #     print("  ✅ 网络特征加载完成")
        # else:
        #     print("  ⚠️ 网络特征文件不可用，将使用默认值")
        network_available = False  # 禁用网络特征

        cache_hits = 0
        cache_misses = 0

        for code in codes:
            try:
                print(f"处理股票: {code}")

                # 移除代码中的.HK后缀，腾讯财经接口不需要
                stock_code = code.replace('.HK', '')

                # 获取股票数据（2年约730天）
                stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
                if stock_df is None or stock_df.empty:
                    continue

                # 获取数据最后日期作为缓存键
                last_date = stock_df.index[-1].strftime('%Y%m%d') if hasattr(stock_df.index[-1], 'strftime') else str(stock_df.index[-1])[:10].replace('-', '')

                # 尝试加载特征缓存
                cache_key = _get_feature_cache_key(stock_code, last_date)
                cache_file_path = _get_feature_cache_file_path(cache_key)

                use_cache = False
                if use_feature_cache and _is_feature_cache_valid(cache_file_path):
                    cached_data = _load_feature_cache(cache_file_path)
                    if cached_data is not None and 'stock_df' in cached_data:
                        cached_df = cached_data['stock_df']
                        # 检查新特征列是否存在（GARCH + HSI Regime）
                        required_new_cols = ['GARCH_Conditional_Vol', 'HSI_Market_Regime']
                        missing_cols = [c for c in required_new_cols if c not in cached_df.columns]
                        if missing_cols:
                            print(f"  ⚠️ 缓存缺少新特征: {missing_cols}，重新计算...")
                        else:
                            stock_df = cached_df
                            use_cache = True
                            cache_hits += 1
                            print(f"  ✅ 使用特征缓存")
                            logger.debug(f"特征缓存命中: {cache_key}")

                if not use_cache:
                    # 计算特征
                    cache_misses += 1

                    # 计算技术指标（80个指标）
                    stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                    # 计算多周期指标
                    stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                    # 计算相对强度指标
                    if hsi_df is not None:
                        stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                    # 合并 HSI 市场状态特征
                    if hsi_regime_df is not None:
                        stock_df = self.feature_engineer.calculate_hsi_regime_features(stock_df, hsi_regime_df)

                    # 创建资金流向特征
                    stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                    # 创建市场环境特征（包含港股和美股）
                    if hsi_df is not None:
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

                    # 网络特征已移除：存在数据泄漏风险，作为独立分析工具使用

                    # 添加事件驱动特征（9个）
                    stock_df = self.feature_engineer.create_event_driven_features(code, stock_df)

                    # 生成技术指标与基本面交互特征（与训练时保持一致）
                    stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

                    # 生成交叉特征（与训练时保持一致）
                    stock_df = self.feature_engineer.create_interaction_features(stock_df)

                    # 保存特征缓存（不含标签）
                    if use_feature_cache:
                        _save_feature_cache(cache_file_path, {'stock_df': stock_df})
                        print(f"  💾 特征已缓存")
                        logger.debug(f"特征缓存已保存: {cache_key}")

                # 创建标签（使用指定的 horizon 和阈值，不缓存）
                # 注意：相对标签需要在所有股票数据收集完成后统一计算
                if label_type == 'absolute':
                    stock_df = self.feature_engineer.create_label(stock_df, horizon=horizon, for_backtest=for_backtest, min_return_threshold=min_return_threshold, label_type='absolute')

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

        # 打印缓存统计
        if use_feature_cache:
            print(f"\n📊 特征缓存统计: 命中 {cache_hits}, 未命中 {cache_misses}")

        # 合并所有数据
        df = pd.concat(all_data, ignore_index=False)

        # 转换索引为 datetime（统一为UTC时区）
        df.index = pd.to_datetime(df.index, utc=True)

        # ========== 网络特征（跨截面特征，所有股票共享）==========
        # 使用实时计算，确保数据泄漏防护（for_prediction=True 排除当日数据）
        # P7 优化（2026-05-04）：添加增量网络特征
        try:
            from data_services.network_features import get_network_calculator
            import networkx as nx

            print("  📊 计算网络特征...")
            network_calc = get_network_calculator()

            # 获取所有股票代码
            unique_codes = df['Code'].unique().tolist()

            # 计算网络洞察（中心性、社区等）
            insights = network_calc.calculate_network_insights(unique_codes, force_refresh=True)

            # 计算节点偏离度（当前窗口）
            deviations = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=20)

            # P7 新增：计算不同窗口的节点偏离度，用于增量特征
            deviations_5d = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=5)

            # 将网络特征添加到每只股票的数据中
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                # P7 新增：增量网络特征
                'net_node_deviation_delta_5d',  # 节点偏离度变化率
                'net_node_deviation_accel',     # 节点偏离度加速度
            ]

            # 初始化网络特征列
            for feat in network_feature_names:
                df[feat] = 0.0 if feat != 'net_community_id' else -1

            # 填充网络特征
            for code in unique_codes:
                mask = df['Code'] == code

                # 中心性
                if code in insights:
                    df.loc[mask, 'net_composite_centrality'] = insights[code].get('composite_centrality', 0)

                # 社区ID（分类特征）
                if code in insights:
                    df.loc[mask, 'net_community_id'] = insights[code].get('community', -1)

                # 节点偏离度
                dev_20d = deviations.get(code, {}).get('node_deviation', 0) if code in deviations else 0
                dev_5d = deviations_5d.get(code, {}).get('node_deviation', 0) if code in deviations_5d else 0

                df.loc[mask, 'net_node_deviation'] = dev_20d

                # P7 新增：增量特征
                # 节点偏离度变化率 = 当前偏离度 - 5日前偏离度
                # 正值表示"正在变得更强"，负值表示"正在变弱"
                df.loc[mask, 'net_node_deviation_delta_5d'] = dev_20d - dev_5d

                # 节点偏离度加速度 = 变化率的变化率
                # 正值表示"加速变强"，负值表示"减速变强"
                # 这里用 20d 偏离度与 5d 偏离度的差值作为加速度的代理
                df.loc[mask, 'net_node_deviation_accel'] = (dev_20d - dev_5d) / (abs(dev_5d) + 0.001)  # 归一化

            print(f"  ✅ 网络特征计算完成（5个特征，含增量特征）")
            logger.info(f"网络特征计算完成: {len(unique_codes)} 只股票")

        except Exception as e:
            logger.warning(f"网络特征计算失败: {e}，将使用默认值")
            print(f"  ⚠️ 网络特征计算失败: {e}")
            # 使用默认值
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                'net_node_deviation_delta_5d',
                'net_node_deviation_accel',
            ]
            for feat in network_feature_names:
                if feat not in df.columns:
                    df[feat] = 0.0 if feat != 'net_community_id' else -1

        # 相对标签：在合并所有股票数据后统一计算
        if label_type == 'relative':
            print(f"  计算相对标签（每日中位数）...")
            # 保存原始索引（日期）
            original_dates = df.index.copy()

            # 计算未来收益率（在原 DataFrame 上操作，保持日期索引）
            df['Future_Return'] = df.groupby('Code')['Close'].transform(
                lambda x: x.shift(-horizon) / x - 1
            )

            # 按日期计算每日中位数（使用日期索引）
            daily_median = df.groupby(df.index)['Future_Return'].median()
            daily_median.name = 'Daily_Median_Return'

            # 将中位数映射回原数据
            df['Daily_Median_Return'] = df.index.map(daily_median)

            # 创建相对标签
            df['Label'] = (df['Future_Return'] > df['Daily_Median_Return']).astype(int)

            # 如果不是回测模式，移除最后horizon行（没有标签的数据）
            if not for_backtest:
                df = df.iloc[:-horizon]

            print(f"  相对标签正例比例: {df['Label'].mean():.2%}")

        # 过滤日期范围（如果指定）
        if start_date:
            start_date = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_date]

        # ========== 特征残差化（剔除宏观因子对微观特征的影响）==========
        # 对微观特征剔除宏观因子贡献，迫使模型学习个股特异性
        try:
            from data_services.feature_residualizer import FeatureResidualizer
            residualizer = FeatureResidualizer()
            df = residualizer.residualize(df, inplace=True, keep_original=False)
            # 保存残差化器到实例，用于预测时保持一致
            self.residualizer = residualizer
            residual_features = residualizer.get_residual_features()
            if residual_features:
                logger.info(f"特征残差化完成，生成 {len(residual_features)} 个残差特征")
        except Exception as e:
            logger.warning(f"特征残差化失败: {e}，将使用原始特征")
            self.residualizer = None

        # ========== 滚动百分位特征（已关闭，消融实验证明其降低IC）==========
        if self.use_rolling_percentile:
            df = self._calculate_rolling_percentile_features(df)

        # ========== 截面百分位特征（2026-05-03 新增，与相对标签匹配）==========
        if self.use_cross_sectional_percentile:
            df = self._calculate_cross_sectional_percentile_features(df)

        # ========== 截面 Z-Score 特征（2026-05-03 新增，解决时间序列基准问题）==========
        if self.use_cross_sectional_zscore:
            df = self._calculate_cross_sectional_zscore_features(df)

        # ========== P4: 板块相对动能特征（2026-05-06 新增）==========
        # 计算 Individual_Stock_Momentum - Sector_Momentum，提供个股 Alpha 信号
        # 强迫模型关注个股相对于板块的超额收益，而非市场 Beta
        try:
            from config import STOCK_SECTOR_MAPPING

            # 获取股票板块映射
            stock_to_sector = {code: info['sector'] for code, info in STOCK_SECTOR_MAPPING.items()}

            # 添加板块列
            df['Sector'] = df['Code'].map(stock_to_sector)

            # 保存原始索引名称
            original_index_name = df.index.name
            df.index.name = 'Date'

            # 计算板块动量（每日每板块的平均动量）
            if 'Momentum_20d' in df.columns:
                # 板块动量 = 板块内所有股票 Momentum_20d 的均值
                sector_momentum = df.groupby([df.index, 'Sector'])['Momentum_20d'].mean()
                sector_momentum.name = 'Sector_Momentum_20d'

                # 将板块动量映射回原数据
                sector_momentum_df = sector_momentum.reset_index()
                df = df.reset_index().merge(
                    sector_momentum_df,
                    on=['Date', 'Sector'],
                    how='left'
                ).set_index('Date')

                # 计算板块相对动能
                df['Sector_Relative_Momentum_20d'] = df['Momentum_20d'] - df['Sector_Momentum_20d']

                # 删除临时列
                df = df.drop(columns=['Sector_Momentum_20d'], errors='ignore')

                logger.info(f"板块相对动能特征计算完成: Sector_Relative_Momentum_20d")

            # 同样计算 5d 版本
            if 'Momentum_5d' in df.columns or 'Return_5d' in df.columns:
                # 使用 Return_5d 作为短期动量代理
                if 'Return_5d' in df.columns:
                    momentum_col = 'Return_5d'
                else:
                    momentum_col = 'Momentum_5d'

                sector_momentum_5d = df.groupby([df.index, 'Sector'])[momentum_col].mean()
                sector_momentum_5d.name = 'Sector_Momentum_5d'

                sector_momentum_5d_df = sector_momentum_5d.reset_index()
                df = df.reset_index().merge(
                    sector_momentum_5d_df,
                    on=['Date', 'Sector'],
                    how='left'
                ).set_index('Date')

                # 计算板块相对动能
                df['Sector_Relative_Momentum_5d'] = df[momentum_col] - df['Sector_Momentum_5d']

                # 删除临时列
                df = df.drop(columns=['Sector_Momentum_5d'], errors='ignore')

            # 恢复原始索引名称
            df.index.name = original_index_name

            # 删除临时 Sector 列
            df = df.drop(columns=['Sector'], errors='ignore')

        except Exception as e:
            logger.warning(f"板块相对动能特征计算失败: {e}")

        # 保存截面特征的训练集统计量，供单只股票预测时回退使用
        # 当单只股票预测无法计算截面特征时，使用训练集均值作为中性值
        self.cs_feature_stats = {}
        for col in df.columns:
            if col.endswith('_CS_Pct') or col.endswith('_CS_ZScore'):
                self.cs_feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }
        if self.cs_feature_stats:
            logger.info(f"保存 {len(self.cs_feature_stats)} 个截面特征的统计量（供单股预测回退）")

        logger.info(f"数据准备完成，共 {len(df)} 条记录")

        return df

    def get_feature_columns(self, df, dedup_threshold=None):
        """获取特征列

        Args:
            df: 数据 DataFrame
            dedup_threshold: Pearson 相关性去冗余阈值（默认 None 不启用）
                            设置为 0.95 可删除 |r| > 0.95 的高相关特征

        Returns:
            list: 特征列名列表
        """
        # 排除非特征列（包括中间计算列）
        exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                          'Future_Return', 'Label', 'Prev_Close', 'Label_Threshold',
                          'Daily_Median_Return',  # 相对标签计算的中间结果，预测时无法获得
                          'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                          'BB_upper', 'BB_lower', 'BB_middle',
                          'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                          'TP', 'MF_Multiplier', 'MF_Volume',
                          # 已删除的冗余特征（保留 DataFrame 列用于中间计算）
                          'Returns', 'Volatility', 'MA5_Deviation', 'MA10_Deviation',
                          'BB_Breakout', 'High_Position_20d', 'MA_Bullish_Resonance',
                          'Momentum_5d', 'Momentum_10d', 'Momentum_120d', 'Momentum_250d',
                          'Consecutive_Ranging_Days', 'Confidence_Threshold_Multiplier',
                          'Price_Return_Std_30d']

        # 排除市场级特征（对选股无区分力，防止模型变成宏观择时器）
        # 注意：市场级特征保留在 DataFrame 中（残差化需要用），只是不进入模型特征列表
        market_exclude = set(self.MARKET_LEVEL_FEATURES) if hasattr(self, 'MARKET_LEVEL_FEATURES') else set()
        feature_columns = [col for col in df.columns if col not in exclude_columns and col not in market_exclude]

        # 日志记录排除的市场级特征
        excluded_market_features = [col for col in df.columns if col in market_exclude]
        if excluded_market_features:
            logger.info(f"排除 {len(excluded_market_features)} 个市场级特征: {excluded_market_features[:5]}{'...' if len(excluded_market_features) > 5 else ''}")

        # P6: 排除宏交叉特征（趋势 × 宏观变量，模型会"偷懒"用这些做宏观拟合）
        macro_cross_exclude = set(self.MACRO_CROSS_FEATURES) if hasattr(self, 'MACRO_CROSS_FEATURES') else set()
        feature_columns = [col for col in feature_columns if col not in macro_cross_exclude]

        # 日志记录排除的宏交叉特征
        excluded_macro_cross = [col for col in df.columns if col in macro_cross_exclude]
        if excluded_macro_cross:
            logger.info(f"排除 {len(excluded_macro_cross)} 个宏交叉特征: {excluded_macro_cross[:5]}{'...' if len(excluded_macro_cross) > 5 else ''}")

        # P9: 排除 L3 特征（市场级 + 宏观交叉特征的完整列表）
        l3_exclude = set(self.L3_EXCLUDE_FEATURES) if hasattr(self, 'L3_EXCLUDE_FEATURES') else set()
        feature_columns = [col for col in feature_columns if col not in l3_exclude]

        # 日志记录排除的 L3 特征
        excluded_l3 = [col for col in df.columns if col in l3_exclude]
        if excluded_l3:
            logger.info(f"P9 排除 {len(excluded_l3)} 个 L3 特征: {excluded_l3[:5]}{'...' if len(excluded_l3) > 5 else ''}")

        # 可选：Pearson 去冗余（防止新增特征时引入高相关冗余）
        if dedup_threshold and len(feature_columns) > 0:
            numeric_cols = [c for c in feature_columns if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            if len(numeric_cols) > 1:
                # 使用最近 500 行数据计算相关性（减少计算量）
                sample_df = df[numeric_cols].tail(500)
                corr_matrix = sample_df.corr()
                to_remove = set()
                for i in range(len(numeric_cols)):
                    if numeric_cols[i] in to_remove:
                        continue
                    for j in range(i+1, len(numeric_cols)):
                        if numeric_cols[j] in to_remove:
                            continue
                        if abs(corr_matrix.iloc[i, j]) > dedup_threshold:
                            to_remove.add(numeric_cols[j])
                feature_columns = [c for c in feature_columns if c not in to_remove]
                if to_remove:
                    logger.info(f"Pearson 去冗余（阈值={dedup_threshold}）：删除 {len(to_remove)} 个高相关特征")

        return feature_columns

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False, min_return_threshold=0.0):
        """训练 CatBoost 模型（默认使用全量特征892个）

        Args:
            codes: 股票代码列表
            start_date: 训练开始日期
            end_date: 训练结束日期
            horizon: 预测周期（1=次日，5=一周，20=一个月）
            use_feature_selection: 是否使用特征选择（已弃用，默认False使用全量特征）
            min_return_threshold: 最小收益阈值（默认0%），用于标签定义
        # 设置固定随机种子（确保模型训练的可重现性）
        np.random.seed(42)
        random.seed(42)

        Returns:
            DataFrame: 特征重要性数据
        """
        # 检查是否需要显示弃用警告
        if use_feature_selection and not CatBoostModel._deprecation_warning_shown:
            print("⚠️  警告：特征选择功能已弃用，建议使用全量特征（892个）。Walk-forward验证显示全量特征性能更好。")
            CatBoostModel._deprecation_warning_shown = True

        print("\n" + "="*70)
        logger.info("开始训练 CatBoost 模型")
        print("="*70)
        print(f"预测周期: {horizon} 天")
        print(f"股票数量: {len(codes)}")
        print(f"特征选择: {'是' if use_feature_selection else '否'}")
        print(f"最小收益阈值: {min_return_threshold:.2%}")

        # ========== 准备数据 ==========
        print("\n" + "="*70)
        logger.info("准备训练数据")
        print("="*70)

        df = self.prepare_data(codes, start_date, end_date, horizon,
                               min_return_threshold=min_return_threshold,
                               label_type='relative')  # 个股选股模型使用相对标签

        # 删除包含 NaN 的行
        df = df.dropna(subset=['Label'])
        print(f"删除 NaN 后: {len(df)} 条记录")

        # 计算时间衰减权重（在dropna之后，确保索引对齐）
        self.sample_weights = None
        if self.time_decay_lambda > 0:
            self.sample_weights = self._compute_time_decay_weights(df, self.time_decay_lambda)

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"使用 {len(self.feature_columns)} 个特征")

        # ========== 特征选择（可选）==========
        if use_feature_selection:
            print("\n" + "="*70)
            print("🔍 应用特征选择（已弃用）")
            print("="*70)

            # 加载选择的特征
            selected_features = self.load_selected_features(current_feature_names=self.feature_columns)

            if selected_features:
                # 筛选特征列
                self.feature_columns = [col for col in self.feature_columns if col in selected_features]
                print(f"✅ 特征数量: {len(self.feature_columns)}（特征选择 - 已弃用）")
            else:
                logger.warning(r"未找到特征选择文件，使用全部特征")
        else:
            print(f"\n✅ 特征数量: {len(self.feature_columns)}（全量特征）")

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
                # 先填充NaN值为'unknown'，避免CatBoost分类特征NaN错误
                df[col] = df[col].fillna('unknown').astype(str)
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
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
            # 一个月模型参数（适配 730 特征版本）
            # 2026-04-29：手动调优最优参数（夏普0.9059，回撤-0.20%）
            # 注：系统调优参数在快速验证表现好但完整验证泛化不足
            print("使用20天模型参数（手动调优最优版，730特征）...")
            n_estimators = 600  # 树数量
            depth = 7  # 树深度
            learning_rate = 0.03  # 学习率（保守）
            stopping_rounds = 80  # 早停耐心
            l2_leaf_reg = 2  # L2正则化
            subsample = 0.75  # 行采样
            colsample_bylevel = 0.75  # 列采样

        from catboost import CatBoostClassifier, Pool

        # 准备类别权重参数
        catboost_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',  # P3-7: 改为 AUC，与排序目标对齐（原 Accuracy）
            'depth': depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'l2_leaf_reg': l2_leaf_reg,
            'subsample': subsample,
            'colsample_bylevel': colsample_bylevel,
            'random_seed': 2020,
            'verbose': 100,
            'early_stopping_rounds': stopping_rounds,
            'thread_count': -1,
            'allow_writing_files': False,
            'cat_features': categorical_features if categorical_features else None
        }
        
        # 添加类别权重（温和调整）
        if self.class_weight == 'balanced':
            # 自动平衡类别权重（温和）
            catboost_params['auto_class_weights'] = 'Balanced'
            logger.info("使用自动平衡类别权重 (Balanced)")
        elif self.class_weight == 'balanced_subsample':
            catboost_params['auto_class_weights'] = 'Balanced'
            logger.info("使用子样本平衡类别权重 (Balanced)")
        elif isinstance(self.class_weight, dict):
            # 手动指定权重
            catboost_params['class_weights'] = [self.class_weight.get(0, 1.0), self.class_weight.get(1, 1.0)]
            logger.info(f"使用手动类别权重: {self.class_weight}")
        else:
            logger.info("不使用类别权重")

        # 构建单调约束（防止特征方向翻转）
        monotone_constraints = self._build_monotone_constraints(self.feature_columns)
        if monotone_constraints is not None:
            catboost_params['monotone_constraints'] = monotone_constraints
            constrained_pos = sum(1 for c in monotone_constraints if c == 1)
            constrained_neg = sum(1 for c in monotone_constraints if c == -1)
            logger.info(f"已启用单调约束: +1={constrained_pos}, -1={constrained_neg}")
        else:
            logger.info("单调约束已禁用")

        self.catboost_model = CatBoostClassifier(**catboost_params)

        # 使用时间序列交叉验证（添加 gap 参数避免短期依赖）
        tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
        catboost_scores = []
        catboost_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 计算当前fold的权重子集
            fold_weights = None
            if self.sample_weights is not None:
                fold_weights = self.sample_weights[train_idx]

            # 创建 Pool 对象（CatBoost 推荐）
            train_pool = Pool(
                data=X_train_fold,
                label=y_train_fold,
                cat_features=categorical_features if categorical_features else None,
                weight=fold_weights  # 时间衰减权重
            )
            val_pool = Pool(
                data=X_val_fold,
                label=y_val_fold,
                cat_features=categorical_features if categorical_features else None
                # 验证集不使用权重（权重只影响训练）
            )

            self.catboost_model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False
            )

            y_pred_fold = self.catboost_model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
            catboost_scores.append(score)
            catboost_f1_scores.append(f1)
            print(f"   Fold {fold} 验证准确率: {score:.4f}, 验证F1分数: {f1:.4f}")

        # 使用全部数据重新训练（带时间衰减权重）
        full_pool = Pool(
            data=X,
            label=y,
            cat_features=categorical_features if categorical_features else None,
            weight=self.sample_weights  # 时间衰减权重
        )
        self.catboost_model.fit(full_pool, verbose=100)

        # 获取实际训练的树数量
        self.actual_n_estimators = self.catboost_model.tree_count_
        mean_accuracy = np.mean(catboost_scores)
        std_accuracy = np.std(catboost_scores)
        mean_f1 = np.mean(catboost_f1_scores)
        std_f1 = np.std(catboost_f1_scores)
        print(f"\n✅ CatBoost 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators} (原计划: {n_estimators})")
        print(f"   平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"   平均验证F1分数: {mean_f1:.4f} (+/- {std_f1:.4f})")

        # 保存准确率到文件（供综合分析使用）
        accuracy_info = {
            'model_type': 'catboost',
            'horizon': horizon,
            'accuracy': float(mean_accuracy),
            'std': float(std_accuracy),
            'f1_score': float(mean_f1),
            'f1_std': float(std_f1),
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
        feat_imp.to_csv('output/ml_trading_model_catboost_20d_importance.csv', index=False)
        logger.info(r"已保存特征重要性至 output/ml_trading_model_catboost_20d_importance.csv")

        # 显示前20个重要特征
        print("\n📊 CatBoost Top 20 重要特征:")
        print(feat_imp[['Feature', 'Importance', 'Impact_Direction']].head(20))

        print("\n" + "="*70)
        logger.info(r"CatBoost 模型训练完成！")
        print("="*70)

        return feat_imp

    def predict(self, code, predict_date=None, horizon=None, use_feature_cache=True):
        """预测单只股票

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)，基于该日期的数据预测下一个交易日，默认使用最新交易日
            horizon: 预测周期（1=次日，5=一周，20=一个月），默认使用训练时的周期
            use_feature_cache: 是否使用特征缓存（默认True）
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 移除代码中的.HK后缀
            stock_code = code.replace('.HK', '')

            # 获取股票数据
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                # 确保索引是 datetime 类型
                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)

                # 使用字符串比较避免时区问题
                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 获取数据最后日期作为缓存键
            last_date = stock_df.index[-1].strftime('%Y%m%d') if hasattr(stock_df.index[-1], 'strftime') else str(stock_df.index[-1])[:10].replace('-', '')

            # 尝试加载特征缓存
            cache_key = _get_feature_cache_key(stock_code, last_date)
            cache_file_path = _get_feature_cache_file_path(cache_key)

            use_cache_predict = False
            if use_feature_cache and _is_feature_cache_valid(cache_file_path):
                # 使用缓存
                cached_data = _load_feature_cache(cache_file_path)
                if cached_data is not None and 'stock_df' in cached_data:
                    cached_df = cached_data['stock_df']
                    # 检查新特征列是否存在（GARCH + HSI Regime）
                    required_new_cols = ['GARCH_Conditional_Vol', 'HSI_Market_Regime', 'net_composite_centrality']
                    missing_cols = [c for c in required_new_cols if c not in cached_df.columns]
                    if missing_cols:
                        logger.debug(f"预测缓存缺少新特征: {missing_cols}，重新计算...")
                    else:
                        stock_df = cached_df
                        logger.debug(f"预测使用特征缓存: {cache_key}")
                        use_cache_predict = True

            if not use_cache_predict:
                # 计算特征
                # 获取恒生指数数据
                hsi_df = get_hsi_data_tencent(period_days=1460)
                if hsi_df is None or hsi_df.empty:
                    hsi_df = None

                # 获取美股市场数据
                us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

                # 如果指定了预测日期，过滤数据到该日期
                if predict_date and hsi_df is not None:
                    if not isinstance(hsi_df.index, pd.DatetimeIndex):
                        hsi_df.index = pd.to_datetime(hsi_df.index)
                    hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                    if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                        us_market_df.index = pd.to_datetime(us_market_df.index)
                    if us_market_df is not None:
                        us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                # 计算技术指标（80个指标）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)

                # 计算多周期指标
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                # 计算相对强度指标
                if hsi_df is not None:
                    stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)

                    # 合并 HSI 市场状态特征
                    hsi_regime_df_predict = None
                    try:
                        regime_detector = RegimeDetector()
                        hsi_with_regime = regime_detector.calculate_features(hsi_df.copy())
                        rename_map = {c: f'HSI_{c}' for c in RegimeDetector.get_feature_names()}
                        hsi_regime_df_predict = hsi_with_regime[RegimeDetector.get_feature_names()].rename(columns=rename_map)
                    except Exception as e:
                        logger.warning(f"HSI 市场状态特征计算失败: {e}")
                    if hsi_regime_df_predict is not None:
                        stock_df = self.feature_engineer.calculate_hsi_regime_features(stock_df, hsi_regime_df_predict)

                # 创建资金流向特征
                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                # 创建市场环境特征（包含港股和美股）
                if hsi_df is not None:
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

                # 添加主题情感交互特征（移到循环外，避免重复调用）
                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value

                # 添加预期差距特征（移到循环外，避免重复调用）
                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

                # 添加板块特征
                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                # 网络特征已移除：存在数据泄漏风险，作为独立分析工具使用

                # 添加事件驱动特征（9个，与训练时保持一致）
                stock_df = self.feature_engineer.create_event_driven_features(code, stock_df)

                # 生成技术指标与基本面交互特征（与训练时保持一致）
                stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)

                # 生成交叉特征（与训练时保持一致）
                stock_df = self.feature_engineer.create_interaction_features(stock_df)

                # ========== 网络特征（与训练时保持一致）==========
                # 预测时使用缓存的网络特征或默认值
                # 网络特征需要跨截面计算，预测单只股票时使用默认值
                # 注意：net_sector_community_match 和 net_mst_neighbor_sectors 已移除
                network_feature_names = [
                    'net_composite_centrality',
                    'net_community_id',
                    'net_node_deviation'
                ]
                for feat in network_feature_names:
                    if feat not in stock_df.columns:
                        stock_df[feat] = 0.0 if feat != 'net_community_id' else -1

                # ========== 滚动百分位特征（与训练时保持一致）==========
                # 如果训练时使用了滚动百分位，预测时也需要使用
                if self.use_rolling_percentile:
                    stock_df = self._calculate_rolling_percentile_features(stock_df)

                # ========== 截面百分位特征（2026-05-03 新增）==========
                # 注意：截面百分位需要当日所有股票数据，单只股票预测时无法计算
                # 如果训练时使用了截面百分位，预测时需要确保特征列存在
                if self.use_cross_sectional_percentile:
                    # 检查是否有截面百分位特征，如果没有则跳过（使用原始特征）
                    cs_pct_features = [f'{feat}_CS_Pct' for feat in self.CROSS_SECTIONAL_PERCENTILE_FEATURES]
                    missing_features = [f for f in cs_pct_features if f not in stock_df.columns]
                    if missing_features:
                        logger.warning(f"截面百分位特征 {missing_features} 不存在（单只股票预测时无法计算），将使用原始特征")

                if self.use_cross_sectional_zscore:
                    # 检查是否有截面 Z-Score 特征，如果没有则跳过（使用原始特征）
                    cs_zscore_features = [f'{feat}_CS_ZScore' for feat in self.CROSS_SECTIONAL_ZSCORE_FEATURES]
                    missing_zscore_features = [f for f in cs_zscore_features if f not in stock_df.columns]
                    if missing_zscore_features:
                        logger.warning(f"截面 Z-Score 特征 {missing_zscore_features} 不存在（单只股票预测时无法计算），将使用原始特征")

                # ========== 特征残差化（与训练时保持一致）==========
                # 如果训练时使用了残差化，预测时也需要使用
                if hasattr(self, 'residualizer') and self.residualizer is not None:
                    try:
                        stock_df = self.residualizer.residualize(stock_df, inplace=True, keep_original=False)
                        logger.debug("预测时特征残差化完成")
                    except Exception as e:
                        logger.warning(f"预测时特征残差化失败: {e}")

                # 保存特征缓存
                if use_feature_cache:
                    _save_feature_cache(cache_file_path, {'stock_df': stock_df})
                    logger.debug(f"特征缓存已保存: {cache_key}")

            # 获取最新数据
            latest_data = stock_df.iloc[-1:]

            # 确保所有事件驱动特征都存在（容错处理）
            event_features = [
                'Ex_Dividend_In_7d', 'Ex_Dividend_In_30d', 'Dividend_Frequency_12m',
                'Earnings_Announcement_In_7d', 'Earnings_Announcement_In_30d', 'Days_Since_Last_Earnings',
                'Earnings_Surprise_Score', 'Earnings_Surprise_Avg_3', 'Earnings_Surprise_Trend'
            ]
            for feat in event_features:
                if feat not in latest_data.columns:
                    logger.warning(f"警告: 事件驱动特征 {feat} 不存在，使用默认值0")
                    latest_data[feat] = 0.0

            # 准备特征
            if len(self.feature_columns) == 0:
                raise ValueError("模型未训练，请先调用train()方法")

            # 处理缺失的截面特征（单只股票预测时无法计算）
            # 优先使用训练集统计量的均值，这是更合理的中性值
            cs_feature_suffixes = ['_CS_Pct', '_CS_ZScore']
            missing_cs_features = False
            for suffix in cs_feature_suffixes:
                cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
                for cs_feat in cs_features:
                    if cs_feat not in latest_data.columns:
                        missing_cs_features = True
                        # 优先使用训练集统计量的均值
                        if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                            latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
                            logger.debug(f"截面特征 {cs_feat} 使用训练集均值回退")
                        else:
                            # 最终回退：_CS_Pct 用 0.5（中位数），_CS_ZScore 用 0.0（均值）
                            latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0
                            logger.warning(f"截面特征 {cs_feat} 缺失统计量，使用默认值 {latest_data[cs_feat].values[0]}")

            # 单股预测降级警告
            if missing_cs_features and (self.use_cross_sectional_percentile or self.use_cross_sectional_zscore):
                logger.warning("predict() 单股预测时截面特征使用训练集均值回退，精度降低。"
                              "建议使用 predict_batch() 获取正确的截面特征。")

            # 处理分类特征（使用训练时的编码器）
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        # 先填充NaN值为'unknown'，避免CatBoost分类特征NaN错误
                        latest_data[col] = latest_data[col].fillna('unknown').astype(str)
                        latest_data[col] = encoder.transform(latest_data[col])
                    except ValueError:
                        # 处理未见过的类别，映射到0
                        logger.warning(f"警告: 分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

            X = latest_data[self.feature_columns].values
            # 确保X中没有NaN值（除了分类特征已处理，数值特征可能也有NaN）
            import numpy as np
            import pandas as pd

            # 使用 DataFrame 来安全处理混合类型数据
            df_temp = pd.DataFrame(X, columns=self.feature_columns)

            # 分别处理数值列和分类列
            categorical_cols = list(self.categorical_encoders.keys())
            numeric_cols = [col for col in self.feature_columns if col not in categorical_cols]

            # 填充数值列的 NaN
            for col in numeric_cols:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].fillna(0.0)

            # 分类列已经在前面处理过（用 LabelEncoder 转换），这里不需要再处理
            # 转换回 numpy 数组
            X = df_temp.values

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

    def _extract_raw_features_single(self, code, predict_date=None, horizon=None, use_feature_cache=True):
        """提取单只股票的原始特征（不含截面特征），供批量预测使用

        Args:
            code: 股票代码
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期
            use_feature_cache: 是否使用特征缓存

        Returns:
            DataFrame: 带特征的 stock_df（含 Code 列），或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            stock_code = code.replace('.HK', '')

            # 获取股票数据
            stock_df = get_hk_stock_data_tencent(stock_code, period_days=1460)
            if stock_df is None or stock_df.empty:
                return None

            # 如果指定了预测日期，过滤数据到该日期
            if predict_date:
                predict_date = pd.to_datetime(predict_date)
                predict_date_str = predict_date.strftime('%Y-%m-%d')

                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    stock_df.index = pd.to_datetime(stock_df.index)

                stock_df = stock_df[stock_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                if stock_df.empty:
                    logger.warning(f"股票 {code} 在日期 {predict_date_str} 之前没有数据")
                    return None

            # 获取数据最后日期作为缓存键
            last_date = stock_df.index[-1].strftime('%Y%m%d') if hasattr(stock_df.index[-1], 'strftime') else str(stock_df.index[-1])[:10].replace('-', '')

            # 尝试加载特征缓存
            cache_key = _get_feature_cache_key(stock_code, last_date)
            cache_file_path = _get_feature_cache_file_path(cache_key)

            use_cache_predict = False
            if use_feature_cache and _is_feature_cache_valid(cache_file_path):
                cached_data = _load_feature_cache(cache_file_path)
                if cached_data is not None and 'stock_df' in cached_data:
                    cached_df = cached_data['stock_df']
                    required_new_cols = ['GARCH_Conditional_Vol', 'HSI_Market_Regime', 'net_composite_centrality']
                    missing_cols = [c for c in required_new_cols if c not in cached_df.columns]
                    if missing_cols:
                        logger.debug(f"预测缓存缺少新特征: {missing_cols}，重新计算...")
                    else:
                        stock_df = cached_df
                        logger.debug(f"预测使用特征缓存: {cache_key}")
                        use_cache_predict = True

            if not use_cache_predict:
                # 计算特征
                hsi_df = get_hsi_data_tencent(period_days=1460)
                if hsi_df is None or hsi_df.empty:
                    hsi_df = None

                us_market_df = us_market_data.get_all_us_market_data(period_days=1460)

                if predict_date and hsi_df is not None:
                    if not isinstance(hsi_df.index, pd.DatetimeIndex):
                        hsi_df.index = pd.to_datetime(hsi_df.index)
                    hsi_df = hsi_df[hsi_df.index.strftime('%Y-%m-%d') <= predict_date_str]
                    if us_market_df is not None and not isinstance(us_market_df.index, pd.DatetimeIndex):
                        us_market_df.index = pd.to_datetime(us_market_df.index)
                    if us_market_df is not None:
                        us_market_df = us_market_df[us_market_df.index.strftime('%Y-%m-%d') <= predict_date_str]

                # 计算所有特征（与 predict() 保持一致）
                stock_df = self.feature_engineer.calculate_technical_features(stock_df)
                stock_df = self.feature_engineer.calculate_multi_period_metrics(stock_df)

                if hsi_df is not None:
                    stock_df = self.feature_engineer.calculate_relative_strength(stock_df, hsi_df)
                    hsi_regime_df_predict = None
                    try:
                        regime_detector = RegimeDetector()
                        hsi_with_regime = regime_detector.calculate_features(hsi_df.copy())
                        rename_map = {c: f'HSI_{c}' for c in RegimeDetector.get_feature_names()}
                        hsi_regime_df_predict = hsi_with_regime[RegimeDetector.get_feature_names()].rename(columns=rename_map)
                    except Exception as e:
                        logger.warning(f"HSI 市场状态特征计算失败: {e}")
                    if hsi_regime_df_predict is not None:
                        stock_df = self.feature_engineer.calculate_hsi_regime_features(stock_df, hsi_regime_df_predict)

                stock_df = self.feature_engineer.create_smart_money_features(stock_df)

                if hsi_df is not None:
                    stock_df = self.feature_engineer.create_market_environment_features(stock_df, hsi_df, us_market_df)

                fundamental_features = self.feature_engineer.create_fundamental_features(code)
                for key, value in fundamental_features.items():
                    stock_df[key] = value

                stock_type_features = self.feature_engineer.create_stock_type_features(code, stock_df)
                for key, value in stock_type_features.items():
                    stock_df[key] = value

                sentiment_features = self.feature_engineer.create_sentiment_features(code, stock_df)
                for key, value in sentiment_features.items():
                    stock_df[key] = value

                topic_features = self.feature_engineer.create_topic_features(code, stock_df)
                for key, value in topic_features.items():
                    stock_df[key] = value

                topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(code, stock_df)
                for key, value in topic_sentiment_interaction.items():
                    stock_df[key] = value

                expectation_gap = self.feature_engineer.create_expectation_gap_features(code, stock_df)
                for key, value in expectation_gap.items():
                    stock_df[key] = value

                sector_features = self.feature_engineer.create_sector_features(code, stock_df)
                for key, value in sector_features.items():
                    stock_df[key] = value

                stock_df = self.feature_engineer.create_event_driven_features(code, stock_df)
                stock_df = self.feature_engineer.create_technical_fundamental_interactions(stock_df)
                stock_df = self.feature_engineer.create_interaction_features(stock_df)

                # 网络特征默认值（注意：net_sector_community_match 和 net_mst_neighbor_sectors 已移除）
                network_feature_names = [
                    'net_composite_centrality', 'net_community_id', 'net_node_deviation'
                ]
                for feat in network_feature_names:
                    if feat not in stock_df.columns:
                        stock_df[feat] = 0.0 if feat != 'net_community_id' else -1

                # 滚动百分位特征
                if self.use_rolling_percentile:
                    stock_df = self._calculate_rolling_percentile_features(stock_df)

                # 特征残差化
                if hasattr(self, 'residualizer') and self.residualizer is not None:
                    try:
                        stock_df = self.residualizer.residualize(stock_df, inplace=True, keep_original=False)
                        logger.debug("预测时特征残差化完成")
                    except Exception as e:
                        logger.warning(f"预测时特征残差化失败: {e}")

                # 保存特征缓存
                if use_feature_cache:
                    _save_feature_cache(cache_file_path, {'stock_df': stock_df})
                    logger.debug(f"特征缓存已保存: {cache_key}")

            # 添加股票代码列（用于批量预测时合并）
            stock_df['Code'] = code

            return stock_df

        except Exception as e:
            logger.warning(f"提取特征失败 {code}: {e}")
            return None

    def predict_batch(self, codes, predict_date=None, horizon=None, use_feature_cache=True):
        """批量预测：先提取所有股票特征，再统一计算截面特征，最后逐只预测

        核心改进：截面特征（_CS_Pct, _CS_ZScore）在所有股票数据上联合计算，
        确保训练/预测一致，而非单只股票时退化为 0.5/0.0。

        Args:
            codes: 股票代码列表
            predict_date: 预测日期 (YYYY-MM-DD)
            horizon: 预测周期
            use_feature_cache: 是否使用特征缓存

        Returns:
            list: 预测结果列表，每个元素为字典或 None
        """
        if horizon is None:
            horizon = self.horizon

        if len(self.feature_columns) == 0:
            raise ValueError("模型未训练，请先调用train()方法")

        logger.info(f"开始批量预测 {len(codes)} 只股票...")

        # 阶段1：逐只提取原始特征
        all_features = {}
        for code in codes:
            stock_df = self._extract_raw_features_single(code, predict_date, horizon, use_feature_cache)
            if stock_df is not None:
                all_features[code] = stock_df

        if not all_features:
            logger.warning("批量预测：没有成功提取任何股票的特征")
            return []

        logger.info(f"成功提取 {len(all_features)} 只股票的特征")

        # 阶段2：合并所有股票，计算截面特征
        combined = pd.concat(all_features.values())

        # 统一索引时区：强制将所有时间戳转换为 tz-naive，避免 groupby 时比较错误
        # 问题根源：不同股票数据可能来自不同源，索引时区状态不一致
        # 当合并 tz-aware 和 tz-naive 索引时，pandas 会创建 object 类型 Index
        def normalize_timestamp(ts):
            '''将任意时间戳转换为 tz-naive'''
            try:
                if hasattr(ts, 'tz') and ts.tz is not None:
                    return ts.tz_localize(None)
                return ts
            except Exception:
                return pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tz is not None else pd.Timestamp(ts)

        # 检查索引是否为 object 类型（混合时区情况）
        if combined.index.dtype == 'object' or (hasattr(combined.index, 'tz') and combined.index.tz is not None):
            new_index = [normalize_timestamp(ts) for ts in combined.index]
            combined.index = pd.DatetimeIndex(new_index)
            logger.debug("批量预测：索引时区已统一为 tz-naive")

        # 计算截面百分位特征
        if self.use_cross_sectional_percentile:
            combined = self._calculate_cross_sectional_percentile_features(combined)
            logger.info("批量预测：截面百分位特征计算完成")

        # 计算截面 Z-Score 特征
        if self.use_cross_sectional_zscore:
            combined = self._calculate_cross_sectional_zscore_features(combined)
            logger.info("批量预测：截面 Z-Score 特征计算完成")

        # ========== 网络特征（跨截面特征，所有股票共享）==========
        # 批量预测时可以正确计算网络特征（需要所有股票数据）
        try:
            from data_services.network_features import get_network_calculator

            logger.info("批量预测：计算网络特征...")
            network_calc = get_network_calculator()

            # 获取所有股票代码
            unique_codes = combined['Code'].unique().tolist()

            # 计算网络洞察（中心性、社区等）
            insights = network_calc.calculate_network_insights(unique_codes, force_refresh=False)

            # 计算节点偏离度（动量网络特征）
            deviations = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=20)

            # P7 新增：计算不同窗口的节点偏离度，用于增量特征
            deviations_5d = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=5)

            # 将网络特征添加到每只股票的数据中
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                # P7 新增：增量网络特征
                'net_node_deviation_delta_5d',  # 节点偏离度变化率
                'net_node_deviation_accel',     # 节点偏离度加速度
            ]

            # 初始化网络特征列
            for feat in network_feature_names:
                combined[feat] = 0.0 if feat != 'net_community_id' else -1

            # 填充网络特征
            for code in unique_codes:
                mask = combined['Code'] == code

                # 中心性
                if code in insights:
                    combined.loc[mask, 'net_composite_centrality'] = insights[code].get('composite_centrality', 0)

                # 社区ID（分类特征）
                if code in insights:
                    combined.loc[mask, 'net_community_id'] = insights[code].get('community', -1)

                # 节点偏离度
                dev_20d = deviations.get(code, {}).get('node_deviation', 0) if code in deviations else 0
                dev_5d = deviations_5d.get(code, {}).get('node_deviation', 0) if code in deviations_5d else 0

                combined.loc[mask, 'net_node_deviation'] = dev_20d

                # P7 新增：增量特征
                combined.loc[mask, 'net_node_deviation_delta_5d'] = dev_20d - dev_5d
                combined.loc[mask, 'net_node_deviation_accel'] = (dev_20d - dev_5d) / (abs(dev_5d) + 0.001)  # 归一化

            logger.info(f"批量预测：网络特征计算完成（{len(unique_codes)} 只股票）")

        except Exception as e:
            logger.warning(f"批量预测：网络特征计算失败: {e}，将使用默认值")
            # 使用默认值
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                'net_node_deviation_delta_5d',
                'net_node_deviation_accel',
            ]
            for feat in network_feature_names:
                if feat not in combined.columns:
                    combined[feat] = 0.0 if feat != 'net_community_id' else -1

        # 阶段3：逐只预测（使用正确的截面特征）
        results = []
        for code in all_features.keys():
            stock_data = combined[combined['Code'] == code]
            if stock_data.empty:
                continue

            latest = stock_data.iloc[-1:].copy()

            # 使用辅助方法进行预测
            result = self._predict_from_features(code, latest, horizon)
            if result:
                results.append(result)

        logger.info(f"批量预测完成：{len(results)}/{len(codes)} 只股票")
        return results

    def _predict_from_features(self, code, latest_data, horizon=None):
        """从特征数据进行预测（供 predict() 和 predict_batch() 共用）

        Args:
            code: 股票代码
            latest_data: 单行 DataFrame，包含所有特征
            horizon: 预测周期

        Returns:
            dict: 预测结果，或 None
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 确保所有事件驱动特征都存在
            event_features = [
                'Ex_Dividend_In_7d', 'Ex_Dividend_In_30d', 'Dividend_Frequency_12m',
                'Earnings_Announcement_In_7d', 'Earnings_Announcement_In_30d', 'Days_Since_Last_Earnings',
                'Earnings_Surprise_Score', 'Earnings_Surprise_Avg_3', 'Earnings_Surprise_Trend'
            ]
            for feat in event_features:
                if feat not in latest_data.columns:
                    logger.warning(f"事件驱动特征 {feat} 不存在，使用默认值0")
                    latest_data[feat] = 0.0

            # 处理缺失的网络特征（P7 增量特征）
            network_features = [
                'net_composite_centrality', 'net_community_id', 'net_node_deviation',
                'net_node_deviation_delta_5d', 'net_node_deviation_accel'
            ]
            for feat in network_features:
                if feat not in latest_data.columns:
                    logger.warning(f"网络特征 {feat} 不存在，使用默认值")
                    latest_data[feat] = 0.0 if feat != 'net_community_id' else -1

            # 处理缺失的截面特征（使用训练集统计量回退）
            cs_feature_suffixes = ['_CS_Pct', '_CS_ZScore']
            for suffix in cs_feature_suffixes:
                cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
                for cs_feat in cs_features:
                    if cs_feat not in latest_data.columns:
                        # 优先使用训练集统计量的均值
                        if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                            latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
                        else:
                            # 最终回退：_CS_Pct 用 0.5（中位数），_CS_ZScore 用 0.0（均值）
                            latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0

            # 处理分类特征
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        latest_data[col] = latest_data[col].fillna('unknown').astype(str)
                        latest_data[col] = encoder.transform(latest_data[col])
                    except ValueError:
                        logger.warning(f"分类特征 {col} 包含训练时未见过的类别，使用默认值")
                        latest_data[col] = 0

            X = latest_data[self.feature_columns].values

            # 处理 NaN
            df_temp = pd.DataFrame(X, columns=self.feature_columns)
            categorical_cols = list(self.categorical_encoders.keys())
            numeric_cols = [col for col in self.feature_columns if col not in categorical_cols]

            for col in numeric_cols:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].fillna(0.0)

            X = df_temp.values

            # CatBoost 预测
            from catboost import Pool
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
            logger.warning(f"预测失败 {code}: {e}")
            return None

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'catboost_model': self.catboost_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'horizon': self.horizon,
            'model_type': self.model_type,
            'categorical_encoders': self.categorical_encoders,
            'residualizer': getattr(self, 'residualizer', None),  # 保存残差化器
            # 新增：Regime Shift 修复参数
            'use_monotone_constraints': self.use_monotone_constraints,
            'monotone_constraints_list': self.monotone_constraints_list,
            'time_decay_lambda': self.time_decay_lambda,
            'use_rolling_percentile': self.use_rolling_percentile,
            'use_cross_sectional_percentile': self.use_cross_sectional_percentile,  # 2026-05-03 新增
            'use_cross_sectional_zscore': self.use_cross_sectional_zscore,  # 2026-05-03 新增
            'cs_feature_stats': getattr(self, 'cs_feature_stats', {}),  # 2026-05-03 新增：截面特征统计量
            'feature_importance_threshold': self.feature_importance_threshold,  # P3-8 新增
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
        self.residualizer = model_data.get('residualizer', None)  # 加载残差化器
        # 新增：Regime Shift 修复参数（向后兼容）
        self.use_monotone_constraints = model_data.get('use_monotone_constraints', False)
        self.monotone_constraints_list = model_data.get('monotone_constraints_list', None)
        self.time_decay_lambda = model_data.get('time_decay_lambda', 0)  # 默认0=无衰减（向后兼容）
        self.use_rolling_percentile = model_data.get('use_rolling_percentile', False)
        self.use_cross_sectional_percentile = model_data.get('use_cross_sectional_percentile', False)  # 2026-05-03 新增
        self.use_cross_sectional_zscore = model_data.get('use_cross_sectional_zscore', False)  # 2026-05-03 新增
        self.cs_feature_stats = model_data.get('cs_feature_stats', {})  # 2026-05-03 新增：截面特征统计量
        self.feature_importance_threshold = model_data.get('feature_importance_threshold', 0.0)  # P3-8 新增
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

        # 处理分类特征：填充 NaN 并使用 LabelEncoder 转换
        for col in self.categorical_encoders.keys():
            if col in test_df.columns:
                # 先填充 NaN 值为 'unknown'，避免 CatBoost 分类特征 NaN 错误
                test_df[col] = test_df[col].fillna('unknown').astype(str)
                # 使用训练时保存的 encoder 转换（处理未见过的类别）
                encoder = self.categorical_encoders[col]
                # 对未见过的类别使用 -1 或第一个已知类别
                test_df[col] = test_df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        # 获取分类特征索引
        categorical_features = [self.feature_columns.index(col) for col in self.categorical_encoders.keys() if col in self.feature_columns]

        # 创建 Pool 对象
        test_pool = Pool(data=test_df, cat_features=categorical_features if categorical_features else None)

        # 返回预测概率
        return self.catboost_model.predict_proba(test_pool)

    def get_dynamic_threshold(self, market_regime=None, vix_level=None, base_threshold=0.55):
        """获取动态阈值（基于市场环境调整）
        
        这是业界推荐的温和方案：模型内部使用类别权重，预测时使用动态阈值
        
        Args:
            market_regime: 市场状态 ('bull', 'bear', 'normal')
                - bull: 牛市，降低阈值增加交易机会
                - bear: 熊市，提高阈值只抓最强信号
                - normal: 震荡市，使用中等阈值
            vix_level: VIX指数水平（波动率指标）
                - high (>25): 高波动，提高阈值
                - normal (15-25): 正常波动
                - low (<15): 低波动，可降低阈值
            base_threshold: 基础阈值（默认0.55）
            
        Returns:
            float: 动态调整后的阈值
            
        示例:
            >>> model = CatBoostModel(class_weight='balanced', use_dynamic_threshold=True)
            >>> threshold = model.get_dynamic_threshold(market_regime='bull')
            >>> print(f"牛市阈值: {threshold}")  # 0.52
            >>> threshold = model.get_dynamic_threshold(market_regime='bear')
            >>> print(f"熊市阈值: {threshold}")  # 0.65
        """
        if not self.use_dynamic_threshold:
            return base_threshold
            
        threshold = base_threshold
        
        # 基于市场状态调整
        if market_regime == 'bull':
            # 牛市：降低阈值，增加交易机会（更激进）
            threshold = max(0.50, base_threshold - 0.03)
            logger.debug(f"牛市模式: 阈值 {base_threshold} -> {threshold}")
        elif market_regime == 'bear':
            # 熊市：提高阈值，只抓最强信号（更保守）
            threshold = min(0.70, base_threshold + 0.10)
            logger.debug(f"熊市模式: 阈值 {base_threshold} -> {threshold}")
        else:
            # 震荡市：使用基础阈值，轻微调整
            threshold = base_threshold
            
        # 基于VIX（波动率）二次调整
        if vix_level is not None:
            if vix_level > 30:  # 极高波动
                threshold = min(0.70, threshold + 0.05)
                logger.debug(f"高波动(VIX={vix_level}): 阈值调整 -> {threshold}")
            elif vix_level > 25:  # 高波动
                threshold = min(0.68, threshold + 0.03)
                logger.debug(f"较高波动(VIX={vix_level}): 阈值调整 -> {threshold}")
            elif vix_level < 15:  # 低波动
                threshold = max(0.50, threshold - 0.02)
                logger.debug(f"低波动(VIX={vix_level}): 阈值调整 -> {threshold}")
                
        return round(threshold, 2)


class DynamicMarketStrategy:
    """动态市场策略 - 根据市场状态动态选择融合方法
    
    支持三种市场状态：
    1. 牛市 (bull)：激进融合，使用全部模型
    2. 熊市 (bear)：保守策略，只使用 CatBoost
    3. 震荡市 (normal)：智能融合，基于一致性
    
    特点：
    - 从 model_accuracy.json 动态读取模型稳定性（std）
    - 基于市场状态动态调整融合策略
    - 符合业界最佳实践
    """

    def __init__(self):
        """初始化动态市场策略"""
        self.current_regime = 'normal'  # 当前市场状态
        self.model_stds = {}  # 模型标准差（稳定性）
        self.horizon = 20  # 预测周期
        self.load_model_stability()

    def load_model_stability(self):
        """从 model_accuracy.json 加载模型稳定性数据"""
        accuracy_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'model_accuracy.json')
        
        try:
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 读取各模型的标准差（稳定性指标）
                self.model_stds = {
                    'lgbm': data.get(f'lgbm_{self.horizon}d', {}).get('std', 0.05),
                    'gbdt': data.get(f'gbdt_{self.horizon}d', {}).get('std', 0.05),
                    'catboost': data.get(f'catboost_{self.horizon}d', {}).get('std', 0.02)
                }
                logger.info(f"已加载模型稳定性数据: {self.model_stds}")
            else:
                logger.warning(f"未找到准确率文件: {accuracy_file}，使用默认值")
                self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}
        except Exception as e:
            logger.warning(f"加载模型稳定性数据失败: {e}，使用默认值")
            self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}

    def detect_market_regime(self, hsi_data):
        """
        检测市场状态
        
        标准：
        - 牛市 (bull)：HSI 20日收益率 > 5%
        - 熊市 (bear)：HSI 20日收益率 < -5%
        - 震荡市 (normal)：-5% ≤ HSI 20日收益率 ≤ 5%
        
        Args:
            hsi_data: 恒生指数数据 (DataFrame 或 dict)
        
        Returns:
            str: 市场状态 ('bull'/'bear'/'normal')
        """
        try:
            if isinstance(hsi_data, dict):
                # 从字典中获取收益率
                hsi_return_20d = hsi_data.get('return_20d', 0)
            elif isinstance(hsi_data, pd.DataFrame):
                # 从 DataFrame 中计算收益率
                if 'Close' in hsi_data.columns and len(hsi_data) >= 20:
                    hsi_return_20d = (hsi_data['Close'].iloc[-1] - hsi_data['Close'].iloc[-20]) / hsi_data['Close'].iloc[-20]
                else:
                    hsi_return_20d = 0
            else:
                hsi_return_20d = 0
            
            # 判断市场状态
            if hsi_return_20d > 0.05:
                self.current_regime = 'bull'
                logger.info(f"检测到牛市：HSI 20日收益率 = {hsi_return_20d:.2%}")
            elif hsi_return_20d < -0.05:
                self.current_regime = 'bear'
                logger.info(f"检测到熊市：HSI 20日收益率 = {hsi_return_20d:.2%}")
            else:
                self.current_regime = 'normal'
                logger.info(f"检测到震荡市：HSI 20日收益率 = {hsi_return_20d:.2%}")
            
            return self.current_regime
        except Exception as e:
            logger.warning(f"市场状态检测失败: {e}，使用默认状态 'normal'")
            self.current_regime = 'normal'
            return 'normal'

    def calculate_consistency(self, predictions):
        """
        计算模型一致性
        
        Args:
            predictions: 三个模型的预测概率列表 [lgbm_pred, gbdt_pred, catboost_pred]
        
        Returns:
            float: 一致性比例 (1.0/0.67/0.33)
        """
        if len(predictions) != 3:
            return 1.0
        
        # 将概率转换为二分类预测
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        
        # 判断一致性
        if pred_labels.count(1) == 3 or pred_labels.count(0) == 3:
            return 1.0  # 三模型一致
        elif pred_labels.count(1) == 2 or pred_labels.count(0) == 2:
            return 0.67  # 两模型一致
        else:
            return 0.33  # 三模型不一致

    def bull_market_ensemble(self, predictions, confidences):
        """
        牛市策略：激进融合
        
        特点：
        - 使用全部三个模型
        - 基于稳定性加权（标准差倒数）
        - 降低置信度阈值
        
        Args:
            predictions: 三个模型的预测概率 [lgbm_pred, gbdt_pred, catboost_pred]
            confidences: 三个模型的置信度 [lgbm_conf, gbdt_conf, catboost_conf]
        
        Returns:
            tuple: (融合概率, 策略名称)
        """
        # 基于稳定性加权（标准差倒数）
        stds = [self.model_stds.get('lgbm', 0.05), 
                self.model_stds.get('gbdt', 0.05), 
                self.model_stds.get('catboost', 0.02)]
        weights = [1/std for std in stds]
        weights = np.array(weights) / sum(weights)
        
        fused_prob = sum(pred * w for pred, w in zip(predictions, weights))
        return fused_prob, 'bull_market_ensemble'

    def bear_market_ensemble(self, predictions, confidences):
        """
        熊市策略：保守策略
        
        特点：
        - 只使用 CatBoost 预测
        - 提高置信度阈值
        - 观望优先
        
        Args:
            predictions: 三个模型的预测概率 [lgbm_pred, gbdt_pred, catboost_pred]
            confidences: 三个模型的置信度 [lgbm_conf, gbdt_conf, catboost_conf]
        
        Returns:
            tuple: (融合概率, 策略名称)
        """
        catboost_pred = predictions[2]  # CatBoost 预测
        catboost_conf = confidences[2]  # CatBoost 置信度
        
        # 提高置信度阈值到 0.65
        if catboost_conf > 0.65:
            return catboost_pred, 'bear_market_high_conf'
        else:
            # 低置信度：观望（返回 0.5）
            return 0.5, 'bear_market_wait'

    def normal_market_ensemble(self, predictions, confidences):
        """
        震荡市策略：智能融合
        
        特点：
        - 检查模型一致性
        - 高一致性时使用稳定性加权
        - 低一致性时使用 CatBoost 主导
        
        Args:
            predictions: 三个模型的预测概率 [lgbm_pred, gbdt_pred, catboost_pred]
            confidences: 三个模型的置信度 [lgbm_conf, gbdt_conf, catboost_conf]
        
        Returns:
            tuple: (融合概率, 策略名称)
        """
        # 计算一致性
        consistency = self.calculate_consistency(predictions)
        
        # 检查 CatBoost 置信度
        catboost_pred = predictions[2]
        catboost_conf = confidences[2]
        
        # 情况1：CatBoost 高置信度 → 直接使用
        if catboost_conf > 0.60:
            return catboost_pred, 'normal_market_catboost_high'
        
        # 情况2：高一致性 → 使用稳定性加权
        if consistency >= 0.67:
            stds = [self.model_stds.get('lgbm', 0.05), 
                    self.model_stds.get('gbdt', 0.05), 
                    self.model_stds.get('catboost', 0.02)]
            weights = [1/std for std in stds]
            weights = np.array(weights) / sum(weights)
            fused_prob = sum(pred * w for pred, w in zip(predictions, weights))
            return fused_prob, 'normal_market_high_consistency'
        
        # 情况3：低一致性 → 使用 CatBoost
        return catboost_pred, 'normal_market_catboost_dominant'

    def predict(self, predictions, confidences, hsi_data=None):
        """
        根据市场状态动态选择融合策略
        
        Args:
            predictions: 三个模型的预测概率 [lgbm_pred, gbdt_pred, catboost_pred]
            confidences: 三个模型的置信度 [lgbm_conf, gbdt_conf, catboost_conf]
            hsi_data: 恒生指数数据（可选）
        
        Returns:
            tuple: (融合概率, 策略名称)
        """
        # 检测市场状态
        if hsi_data is not None:
            regime = self.detect_market_regime(hsi_data)
        else:
            regime = 'normal'  # 默认状态
        
        # 根据市场状态选择策略
        if regime == 'bull':
            return self.bull_market_ensemble(predictions, confidences)
        elif regime == 'bear':
            return self.bear_market_ensemble(predictions, confidences)
        else:
            return self.normal_market_ensemble(predictions, confidences)

class AdvancedDynamicStrategy:
    """高级动态市场策略 - 业界顶级标准
    
    特点：
    1. 多维度市场状态检测（收益率、波动率、成交量、情绪）
    2. 5种市场状态（强牛市、中牛市、震荡市、中熊市、强熊市）
    3. CatBoost 主导（权重 75-100%）
    4. 动态置信度阈值
    5. 仓位管理
    
    符合业界最佳实践：Renaissance Technologies、Two Sigma、DE Shaw
    """
    
    def __init__(self):
        self.model_stds = {}
        self.load_model_stability()
        self.current_regime = 'normal'
    
    def load_model_stability(self):
        """从 model_accuracy.json 加载模型稳定性数据"""
        accuracy_file = 'data/model_accuracy.json'
        try:
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.model_stds = {
                    'lgbm': data.get(f'lgbm_20d', {}).get('std', 0.05),
                    'gbdt': data.get(f'gbdt_20d', {}).get('std', 0.05),
                    'catboost': data.get(f'catboost_20d', {}).get('std', 0.02)
                }
                logger.info(f"已加载模型稳定性数据: {self.model_stds}")
            else:
                logger.warning("未找到准确率文件，使用默认值")
                self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}
        except Exception as e:
            logger.warning(f"加载稳定性数据失败: {e}")
            self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}
    
    def calculate_consistency(self, predictions):
        """
        计算模型一致性
        
        Returns:
            float: 一致性比例 (1.0, 0.67, 0.33)
        """
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        
        if pred_labels.count(1) == 3 or pred_labels.count(0) == 3:
            return 1.0
        elif pred_labels.count(1) == 2 or pred_labels.count(0) == 2:
            return 0.67
        else:
            return 0.33
    
    def detect_advanced_regime(self, hsi_data):
        """
        多维度市场状态检测
        
        维度：
        1. 收益率趋势（5日、20日）
        2. 波动率水平（当前 vs 20日均值）
        3. 成交量变化
        4. 市场情绪（基于波动率）
        
        Returns:
            str: 市场状态 ('strong_bull', 'moderate_bull', 'normal', 'moderate_bear', 'strong_bear')
        """
        if hsi_data is None or len(hsi_data) < 20:
            return 'normal'
        
        # 维度1：收益率趋势
        prices = hsi_data['Close'].values
        return_5d = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        return_20d = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # 维度2：波动率水平
        returns = np.diff(np.log(prices))
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        vol_ma = np.std(returns[-40:]) if len(returns) >= 40 else volatility
        vol_ratio = volatility / vol_ma if vol_ma > 0 else 1.0
        
        # 维度3：成交量变化
        volumes = hsi_data['Volume'].values
        volume_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / volume_ma20 if volume_ma20 > 0 else 1.0
        
        # 维度4：市场情绪（简化版）
        sentiment = 1.0 / (1.0 + volatility * 10)
        
        # 综合判断
        if return_20d > 0.10 and vol_ratio < 1.2 and volume_ratio > 1.0 and sentiment > 0.7:
            return 'strong_bull'
        elif return_20d > 0.02 and vol_ratio < 1.5:
            return 'moderate_bull'
        elif return_20d < -0.10 and vol_ratio > 1.5 and volume_ratio < 1.0 and sentiment < 0.3:
            return 'strong_bear'
        elif return_20d < -0.02 and vol_ratio > 1.2:
            return 'moderate_bear'
        else:
            return 'normal'
    
    def get_strategy_config(self, regime):
        """
        根据市场状态获取策略配置
        
        Returns:
            dict: {
                'catboost_weight': CatBoost 权重,
                'confidence_threshold': 置信度阈值,
                'position_size': 仓位大小
            }
        """
        configs = {
            'strong_bull': {
                'catboost_weight': 0.75,
                'confidence_threshold': 0.50,
                'position_size': 1.2
            },
            'moderate_bull': {
                'catboost_weight': 0.85,
                'confidence_threshold': 0.55,
                'position_size': 1.0
            },
            'normal': {
                'catboost_weight': 0.90,
                'confidence_threshold': 0.55,
                'position_size': 0.9
            },
            'moderate_bear': {
                'catboost_weight': 0.95,
                'confidence_threshold': 0.60,
                'position_size': 0.7
            },
            'strong_bear': {
                'catboost_weight': 1.00,
                'confidence_threshold': 0.65,
                'position_size': 0.5
            }
        }
        
        return configs.get(regime, configs['normal'])
    
    def predict(self, predictions, confidences, hsi_data=None):
        """
        高级动态预测
        
        Args:
            predictions: 三个模型的预测概率 [lgbm_pred, gbdt_pred, catboost_pred]
            confidences: 三个模型的置信度 [lgbm_conf, gbdt_conf, catboost_conf]
            hsi_data: 恒生指数数据（可选）
        
        Returns:
            tuple: (融合概率, 策略名称)
        """
        # 检测市场状态
        regime = self.detect_advanced_regime(hsi_data)
        self.current_regime = regime
        
        # 获取策略配置
        config = self.get_strategy_config(regime)
        
        # 检查 CatBoost 置信度
        catboost_pred = predictions[2]
        catboost_conf = confidences[2]
        
        # 如果 CatBoost 置信度低于阈值，观望
        if catboost_conf < config['confidence_threshold']:
            return 0.5, f'advanced_{regime}_wait'
        
        # 使用 CatBoost 主导权重
        catboost_weight = config['catboost_weight']
        remaining_weight = 1.0 - catboost_weight
        weights = [remaining_weight/2, remaining_weight/2, catboost_weight]
        
        # 融合
        fused_prob = sum(pred * w for pred, w in zip(predictions, weights))

        return fused_prob, f'advanced_{regime}'


class CatBoostRankerModel(BaseTradingModel):
    """CatBoost 排序模型 - 直接优化股票排序，最大化 Rank IC

    P3-9: 与 CatBoostClassifier 的关键区别：
    1. 使用 CatBoostRanker（排序模型）而非 CatBoostClassifier（分类模型）
    2. 标签为连续 Future_Return（而非二元 0/1），保留收益幅度信息
    3. 使用 group_id=date 分组，每个日期形成一个排序组
    4. 损失函数为 YetiRank（直接优化排序），而非 Logloss（优化分类）
    5. 输出为排序分数（而非概率），分数越高表示预期收益越高
    """

    # 复用 CatBoostModel 的特征列表
    MONOTONE_CONSTRAINT_MAP = CatBoostModel.MONOTONE_CONSTRAINT_MAP
    MARKET_LEVEL_FEATURES = CatBoostModel.MARKET_LEVEL_FEATURES
    MACRO_CROSS_FEATURES = CatBoostModel.MACRO_CROSS_FEATURES  # P6: 宏交叉特征排除
    CROSS_SECTIONAL_PERCENTILE_FEATURES = CatBoostModel.CROSS_SECTIONAL_PERCENTILE_FEATURES
    CROSS_SECTIONAL_ZSCORE_FEATURES = CatBoostModel.CROSS_SECTIONAL_ZSCORE_FEATURES
    ROLLING_PERCENTILE_FEATURES = CatBoostModel.ROLLING_PERCENTILE_FEATURES

    def __init__(self, loss_function='QuerySoftMax',  # P3: 从 YetiRank 改为 QuerySoftMax
                 use_monotone_constraints=True,
                 time_decay_lambda=0.5,
                 use_rolling_percentile=False,
                 use_cross_sectional_percentile=True,
                 use_cross_sectional_zscore=True,
                 feature_importance_threshold=0.0,
                 use_soft_label=False):  # P5 失败，回退到原始收益率
        """初始化 CatBoost 排序模型

        Args:
            loss_function: 排序损失函数
                - 'QuerySoftMax'（P3 新默认）：对样本权重兼容性更好，强制在同一天内做差分竞争
                - 'YetiRank'：基于位置权重的全局排序
                - 'YetiRankPairwise'：成对比较式排序
            use_monotone_constraints: 是否使用单调约束
            time_decay_lambda: 时间衰减系数
            use_rolling_percentile: 是否使用滚动百分位（已关闭）
            use_cross_sectional_percentile: 是否使用截面百分位
            use_cross_sectional_zscore: 是否使用截面 Z-Score
            feature_importance_threshold: 特征重要性阈值
            use_soft_label: 是否使用软标签（截面排名百分位），默认 True
        """
        super().__init__()
        self.ranker_model = None
        self.actual_n_estimators = 0
        self.model_type = 'catboost_ranker'
        self.loss_function = loss_function
        self.use_monotone_constraints = use_monotone_constraints
        self.time_decay_lambda = time_decay_lambda
        self.use_rolling_percentile = use_rolling_percentile
        self.use_cross_sectional_percentile = use_cross_sectional_percentile
        self.use_cross_sectional_zscore = use_cross_sectional_zscore
        self.feature_importance_threshold = feature_importance_threshold
        self.use_soft_label = use_soft_label
        self.monotone_constraints_list = None
        self.sample_weights = None
        self.feature_columns = None
        self.categorical_encoders = {}
        self.horizon = 20
        self.residualizer = None
        self.cs_feature_stats = {}

        logger.info(f"CatBoostRankerModel 初始化: loss_function={loss_function}, "
                    f"use_monotone_constraints={use_monotone_constraints}, "
                    f"time_decay_lambda={time_decay_lambda}, "
                    f"use_cross_sectional_percentile={use_cross_sectional_percentile}, "
                    f"use_cross_sectional_zscore={use_cross_sectional_zscore}, "
                    f"feature_importance_threshold={feature_importance_threshold}, "
                    f"use_soft_label={use_soft_label}")

    # 复用 CatBoostModel 的方法
    _build_monotone_constraints = CatBoostModel._build_monotone_constraints
    _calculate_cross_sectional_percentile_features = CatBoostModel._calculate_cross_sectional_percentile_features
    _calculate_cross_sectional_zscore_features = CatBoostModel._calculate_cross_sectional_zscore_features
    get_feature_columns = CatBoostModel.get_feature_columns
    prepare_data = CatBoostModel.prepare_data

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False, min_return_threshold=0.0):
        """训练 CatBoost 排序模型

        关键区别：使用 Future_Return 作为连续标签，group_id=date 分组
        """
        import numpy as np
        import random
        from catboost import CatBoostRanker, Pool
        from sklearn.model_selection import TimeSeriesSplit

        # 设置随机种子
        np.random.seed(42)
        random.seed(42)

        self.horizon = horizon

        # 准备数据（复用 CatBoostModel 的 prepare_data）
        print(f"\n{'='*70}")
        logger.info(f"准备 CatBoostRanker 训练数据（horizon={horizon}）")
        print(f"{'='*70}")

        df = self.prepare_data(codes, start_date, end_date, horizon, for_backtest=False,
                               label_type='relative')  # 个股选股模型使用相对标签

        if df is None or len(df) == 0:
            raise ValueError("数据准备失败或数据为空")

        # 获取特征列
        self.feature_columns = self.get_feature_columns(df)

        # 处理分类特征
        categorical_features = []
        for col in ['Stock_Type', 'Sector', 'Market_Regime']:
            if col in self.feature_columns:
                categorical_features.append(self.feature_columns.index(col))
                if col in df.columns:
                    df[col] = df[col].fillna('unknown').astype(str)
                    from sklearn.preprocessing import LabelEncoder
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    self.categorical_encoders[col] = encoder

        # 排序模型关键：标签选择
        # P5: 软标签实验 - 使用截面排名百分位作为标签
        if self.use_soft_label:
            # 计算每日截面排名百分位（0 到 1 之间）
            # 这样 YetiRankPairwise 能学到"好多少"而非仅仅"谁更好"
            df['Return_Rank_Pct'] = df.groupby(df.index.normalize())['Future_Return'].transform(
                lambda x: x.rank(pct=True)
            )
            y = df['Return_Rank_Pct'].values
            logger.info(f"使用软标签（截面排名百分位）: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        else:
            # 使用原始收益率作为标签
            y = df['Future_Return'].values
            logger.info(f"使用原始收益率作为标签: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

        # 构建 group_id（每天所有股票为一个排序组）
        df = df.sort_index()  # 确保按日期排序
        unique_dates = sorted(df.index.normalize().unique())
        date_to_gid = {d: i for i, d in enumerate(unique_dates)}
        group_ids = df.index.normalize().map(date_to_gid).values.astype(int)

        # 特征矩阵
        X = df[self.feature_columns].values

        # 时间衰减权重
        if self.time_decay_lambda > 0:
            dates = df.index
            max_date = dates.max()
            days_diff = (max_date - dates).days.values  # 转换为 numpy array
            self.sample_weights = np.exp(-self.time_decay_lambda * days_diff / 365)
        else:
            self.sample_weights = None

        # 模型参数（复用 CatBoostModel 20d 参数）
        if horizon == 20:
            n_estimators = 600
            depth = 7
            learning_rate = 0.03
            stopping_rounds = 80
            l2_leaf_reg = 2
            subsample = 0.75
            colsample_bylevel = 0.75
        elif horizon == 5:
            n_estimators = 500
            depth = 6
            learning_rate = 0.04
            stopping_rounds = 60
            l2_leaf_reg = 3
            subsample = 0.7
            colsample_bylevel = 0.7
        else:  # horizon == 1
            n_estimators = 500
            depth = 7
            learning_rate = 0.05
            stopping_rounds = 40
            l2_leaf_reg = 3
            subsample = 0.75
            colsample_bylevel = 0.7

        ranker_params = {
            'loss_function': self.loss_function,
            'eval_metric': 'NDCG',
            'depth': depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'l2_leaf_reg': l2_leaf_reg,
            'subsample': subsample,
            'colsample_bylevel': colsample_bylevel,
            'random_seed': 2020,
            'verbose': 100,
            'early_stopping_rounds': stopping_rounds,
            'has_time': True,  # 数据是时间有序的
            'thread_count': -1,
            'allow_writing_files': False,
        }

        # 单调约束（YetiRankPairwise 不支持单调约束）
        if self.loss_function != 'YetiRankPairwise':
            monotone_constraints = self._build_monotone_constraints(self.feature_columns)
            if monotone_constraints is not None:
                ranker_params['monotone_constraints'] = monotone_constraints
                constrained_pos = sum(1 for c in monotone_constraints if c == 1)
                constrained_neg = sum(1 for c in monotone_constraints if c == -1)
                logger.info(f"已启用单调约束: +1={constrained_pos}, -1={constrained_neg}")
        else:
            logger.info("YetiRankPairwise 不支持单调约束，已跳过")

        self.ranker_model = CatBoostRanker(**ranker_params)

        # 时间序列 CV
        tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            group_ids_train = group_ids[train_idx]
            group_ids_val = group_ids[val_idx]

            # P10 修复：Pairwise 损失函数不支持 object weights，改用 group_weight
            # group_weight 长度必须等于数据长度（每个样本一个权重）
            # 但同一 group 内的样本权重相同（取该日第一个样本的权重）
            group_weights_train = None
            if self.sample_weights is not None:
                fold_weights = self.sample_weights[train_idx]
                # 直接使用 fold_weights（每个样本的权重）
                # 同一日的样本权重相同，所以这实际上就是 group_weight
                group_weights_train = fold_weights

            train_pool = Pool(
                data=X_train_fold,
                label=y_train_fold,
                group_id=group_ids_train,
                cat_features=categorical_features if categorical_features else None,
                group_weight=group_weights_train  # ✅ YetiRank/YetiRankPairwise 支持组权重
            )
            val_pool = Pool(
                data=X_val_fold,
                label=y_val_fold,
                group_id=group_ids_val,
                cat_features=categorical_features if categorical_features else None
            )

            self.ranker_model.fit(train_pool, eval_set=val_pool, verbose=False)

            # 计算 NDCG 作为 CV 分数
            val_scores = self.ranker_model.predict(val_pool)
            try:
                from sklearn.metrics import ndcg_score
                # NDCG 需要真实相关性分数
                ndcg = ndcg_score([y_val_fold], [val_scores])
                cv_scores.append(ndcg)
                print(f"   Fold {fold} NDCG: {ndcg:.4f}")
            except:
                cv_scores.append(0.0)
                print(f"   Fold {fold} 完成")

        # 全量数据重训练（P10 修复：使用 group_weight）
        # group_weight 长度必须等于数据长度
        group_weights_full = None
        if self.sample_weights is not None:
            # 直接使用 sample_weights（每个样本的权重）
            group_weights_full = self.sample_weights

        full_pool = Pool(
            data=X,
            label=y,
            group_id=group_ids,
            cat_features=categorical_features if categorical_features else None,
            group_weight=group_weights_full  # ✅ YetiRank/YetiRankPairwise 支持组权重
        )
        self.ranker_model.fit(full_pool, verbose=100)

        self.actual_n_estimators = self.ranker_model.tree_count_
        print(f"\n✅ CatBoostRanker 训练完成")
        print(f"   实际训练树数量: {self.actual_n_estimators}")
        if cv_scores:
            print(f"   平均 CV NDCG: {np.mean(cv_scores):.4f}")

        # 特征重要性
        feat_importance = self.ranker_model.get_feature_importance(full_pool, prettified=True)
        feat_imp = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.ranker_model.get_feature_importance(full_pool)
        })
        feat_imp = feat_imp.sort_values('Importance', ascending=False)
        feat_imp['Impact_Direction'] = 'Unknown'

        # 保存特征重要性
        feat_imp.to_csv(f'output/ml_trading_model_catboost_ranker_{horizon}d_importance.csv', index=False)
        logger.info(f"已保存特征重要性至 output/ml_trading_model_catboost_ranker_{horizon}d_importance.csv")

        print(f"\n📊 CatBoostRanker Top 20 重要特征:")
        print(feat_imp[['Feature', 'Importance']].head(20))

        return self

    def predict_proba(self, X, temperature=1.0):
        """预测排序分数（兼容 predict_proba 接口）

        返回格式与 CatBoostClassifier.predict_proba() 兼容：
        - 第一列：1 - sigmoid(score)
        - 第二列：sigmoid(score)

        参数：
            X: 特征数据
            temperature: 温度参数，控制概率分布的陡峭程度
                        - temperature=1.0: 默认，使用原始分数
                        - temperature>1.0: 概率分布更平坦（更保守）
                        - temperature<1.0: 概率分布更陡峭（更激进）
                        - temperature='auto': 自动根据分数标准差调整
        """
        from catboost import Pool
        from scipy.special import expit
        import numpy as np

        # 处理输入
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.feature_columns):
                test_df = X[self.feature_columns].copy()
            elif len(X.columns) == len(self.feature_columns):
                test_df = X.copy()
                test_df.columns = self.feature_columns
            else:
                available_cols = [col for col in self.feature_columns if col in X.columns]
                if available_cols:
                    test_df = X[available_cols].copy()
                else:
                    raise ValueError("无法从输入数据中提取特征列")
        else:
            test_df = pd.DataFrame(X, columns=self.feature_columns)

        # 处理分类特征
        for col in self.categorical_encoders.keys():
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna('unknown').astype(str)
                encoder = self.categorical_encoders[col]
                test_df[col] = test_df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        # 填充 NaN
        test_df = test_df.fillna(0)

        # 获取分类特征索引
        categorical_features = [self.feature_columns.index(col) for col in self.categorical_encoders.keys() if col in self.feature_columns]

        test_pool = Pool(data=test_df.values, cat_features=categorical_features if categorical_features else None)
        scores = self.ranker_model.predict(test_pool)

        # 自动温度调整：归一化分数，使标准差约为 1.0
        # 验证结论：MinMax 归一化导致 IC/Rank IC 下降，已放弃
        if temperature == 'auto':
            score_std = np.std(scores)
            if score_std > 0.001:
                temperature = score_std
                logger.info(f"Ranker 自动温度: {temperature:.4f} (分数标准差)")
            else:
                temperature = 1.0
                logger.warning(f"Ranker 分数标准差过小 ({score_std:.4f})，使用默认温度 1.0")

        scaled_scores = scores / temperature
        proba_col1 = expit(scaled_scores)

        # 调试日志
        logger.info(f"Ranker 概率分布: min={proba_col1.min():.4f}, max={proba_col1.max():.4f}, mean={proba_col1.mean():.4f}, std={proba_col1.std():.4f}")

        return np.column_stack([1 - proba_col1, proba_col1])

    def predict(self, code, predict_date=None, horizon=None, use_feature_cache=True):
        """预测单只股票的排序分数"""
        # 简化实现：调用 prepare_data 获取特征，然后 predict_proba
        # 完整实现可参考 CatBoostModel.predict()
        logger.warning("CatBoostRankerModel.predict() 单股预测建议使用 predict_batch() 获取正确的截面特征")
        # 这里返回一个简化版本
        return None

    def _predict_from_features(self, code, latest_data, horizon=None):
        """从特征数据进行预测（供 predict_batch() 共用）

        P2 新增：计算 Expected_Value = (2*Prob-1) × ATR_Ratio
        - 将胜率转化为期望收益
        - 自动避开"胜率虽高但波动极小"的僵尸股

        Args:
            code: 股票代码
            latest_data: 单行 DataFrame，包含所有特征
            horizon: 预测周期

        Returns:
            dict: 预测结果，包含 Expected_Value
        """
        if horizon is None:
            horizon = self.horizon

        try:
            # 处理缺失的截面特征（使用训练集统计量回退）
            cs_feature_suffixes = ['_CS_Pct', '_CS_ZScore']
            for suffix in cs_feature_suffixes:
                cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
                for cs_feat in cs_features:
                    if cs_feat not in latest_data.columns:
                        if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                            latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
                        else:
                            latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0

            # 处理分类特征
            for col, encoder in self.categorical_encoders.items():
                if col in latest_data.columns:
                    try:
                        latest_data[col] = latest_data[col].fillna('unknown').astype(str)
                        latest_data[col] = encoder.transform(latest_data[col])
                    except ValueError:
                        latest_data[col] = 0

            # 提取特征
            X = latest_data[self.feature_columns].values

            # 处理 NaN
            df_temp = pd.DataFrame(X, columns=self.feature_columns)
            categorical_cols = list(self.categorical_encoders.keys())
            numeric_cols = [col for col in self.feature_columns if col not in categorical_cols]

            for col in numeric_cols:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].fillna(0.0)

            X = df_temp.values

            # Ranker 预测
            from catboost import Pool
            from scipy.special import expit

            categorical_features = [self.feature_columns.index(col) for col in self.categorical_encoders.keys() if col in self.feature_columns]

            test_pool = Pool(data=X, cat_features=categorical_features if categorical_features else None)
            scores = self.ranker_model.predict(test_pool)

            # sigmoid 变换到概率（使用自动温度）
            score = scores[0]
            proba = expit(score)  # 默认温度 1.0

            # 方向判断：正分数 = 预测跑赢
            prediction = 1 if score > 0 else 0

            # P2: 计算 Expected_Value
            # 公式: Expected_Value = (2 * Prob - 1) * ATR_Ratio
            # - 2*Prob-1 将 0.5（中性）映射到 0
            # - ATR_Ratio 衡量波动率相对水平
            atr_ratio = 0.0
            if 'ATR_Ratio' in latest_data.columns:
                atr_ratio = float(latest_data['ATR_Ratio'].values[0])
            elif 'ATR' in latest_data.columns and 'ATR_MA' in latest_data.columns:
                atr = float(latest_data['ATR'].values[0])
                atr_ma = float(latest_data['ATR_MA'].values[0])
                atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0

            expected_value = (2 * proba - 1) * atr_ratio

            return {
                'code': code,
                'name': STOCK_NAMES.get(code, code),
                'prediction': int(prediction),
                'probability': float(proba),
                'rank_score': float(score),  # 原始排序分数
                'expected_value': float(expected_value),  # P2: 期望收益
                'atr_ratio': float(atr_ratio),
                'current_price': float(latest_data['Close'].values[0]),
                'date': latest_data.index[0]
            }

        except Exception as e:
            logger.warning(f"Ranker 预测失败 {code}: {e}")
            return None

    def predict_batch(self, codes, predict_date=None, horizon=None, use_feature_cache=True):
        """批量预测：先提取所有股票特征，再统一计算截面特征，最后逐只预测

        复用 CatBoostModel 的 3 阶段批量预测架构（含网络特征计算）
        """
        if horizon is None:
            horizon = self.horizon

        if len(self.feature_columns) == 0:
            raise ValueError("模型未训练，请先调用train()方法")

        logger.info(f"CatBoostRanker 批量预测 {len(codes)} 只股票...")

        # 阶段1：逐只提取原始特征
        all_features = {}
        for code in codes:
            stock_df = self._extract_raw_features_single(code, predict_date, horizon, use_feature_cache)
            if stock_df is not None:
                all_features[code] = stock_df

        if not all_features:
            logger.warning("批量预测：没有成功提取任何股票的特征")
            return []

        logger.info(f"成功提取 {len(all_features)} 只股票的特征")

        # 阶段2：合并所有股票，计算截面特征
        combined = pd.concat(all_features.values())

        # 统一索引时区：强制将所有时间戳转换为 tz-naive，避免 groupby 时比较错误
        # 问题根源：不同股票数据可能来自不同源，索引时区状态不一致
        # 当合并 tz-aware 和 tz-naive 索引时，pandas 会创建 object 类型 Index
        def normalize_timestamp(ts):
            '''将任意时间戳转换为 tz-naive'''
            try:
                if hasattr(ts, 'tz') and ts.tz is not None:
                    return ts.tz_localize(None)
                return ts
            except Exception:
                return pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tz is not None else pd.Timestamp(ts)

        # 检查索引是否为 object 类型（混合时区情况）
        if combined.index.dtype == 'object' or (hasattr(combined.index, 'tz') and combined.index.tz is not None):
            new_index = [normalize_timestamp(ts) for ts in combined.index]
            combined.index = pd.DatetimeIndex(new_index)
            logger.debug("批量预测：索引时区已统一为 tz-naive")

        # 计算截面百分位特征
        if self.use_cross_sectional_percentile:
            combined = self._calculate_cross_sectional_percentile_features(combined)
            logger.info("批量预测：截面百分位特征计算完成")

        # 计算截面 Z-Score 特征
        if self.use_cross_sectional_zscore:
            combined = self._calculate_cross_sectional_zscore_features(combined)
            logger.info("批量预测：截面 Z-Score 特征计算完成")

        # ========== 网络特征（跨截面特征，所有股票共享）==========
        # 批量预测时可以正确计算网络特征（需要所有股票数据）
        try:
            from data_services.network_features import get_network_calculator

            logger.info("批量预测：计算网络特征...")
            network_calc = get_network_calculator()

            # 获取所有股票代码
            unique_codes = combined['Code'].unique().tolist()

            # 计算网络洞察（中心性、社区等）
            insights = network_calc.calculate_network_insights(unique_codes, force_refresh=False)

            # 计算节点偏离度（动量网络特征）
            deviations = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=20)

            # P7 新增：计算不同窗口的节点偏离度，用于增量特征
            deviations_5d = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=5)

            # 将网络特征添加到每只股票的数据中
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                # P7 新增：增量网络特征
                'net_node_deviation_delta_5d',  # 节点偏离度变化率
                'net_node_deviation_accel',     # 节点偏离度加速度
            ]

            # 初始化网络特征列
            for feat in network_feature_names:
                combined[feat] = 0.0 if feat != 'net_community_id' else -1

            # 填充网络特征
            for code in unique_codes:
                mask = combined['Code'] == code

                # 中心性
                if code in insights:
                    combined.loc[mask, 'net_composite_centrality'] = insights[code].get('composite_centrality', 0)

                # 社区ID（分类特征）
                if code in insights:
                    combined.loc[mask, 'net_community_id'] = insights[code].get('community', -1)

                # 节点偏离度
                dev_20d = deviations.get(code, {}).get('node_deviation', 0) if code in deviations else 0
                dev_5d = deviations_5d.get(code, {}).get('node_deviation', 0) if code in deviations_5d else 0

                combined.loc[mask, 'net_node_deviation'] = dev_20d

                # P7 新增：增量特征
                combined.loc[mask, 'net_node_deviation_delta_5d'] = dev_20d - dev_5d
                combined.loc[mask, 'net_node_deviation_accel'] = (dev_20d - dev_5d) / (abs(dev_5d) + 0.001)  # 归一化

            logger.info(f"批量预测：网络特征计算完成（{len(unique_codes)} 只股票）")

        except Exception as e:
            logger.warning(f"批量预测：网络特征计算失败: {e}，将使用默认值")
            # 使用默认值
            network_feature_names = [
                'net_composite_centrality',
                'net_community_id',
                'net_node_deviation',
                'net_node_deviation_delta_5d',
                'net_node_deviation_accel',
            ]
            for feat in network_feature_names:
                if feat not in combined.columns:
                    combined[feat] = 0.0 if feat != 'net_community_id' else -1

        # 阶段3：逐只预测（使用正确的截面特征）
        results = []
        for code in all_features.keys():
            stock_data = combined[combined['Code'] == code]
            if stock_data.empty:
                continue

            latest = stock_data.iloc[-1:].copy()

            # 使用辅助方法进行预测
            result = self._predict_from_features(code, latest, horizon)
            if result:
                results.append(result)

        logger.info(f"CatBoostRanker 批量预测完成：{len(results)}/{len(codes)} 只股票")
        return results

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'ranker_model': self.ranker_model,
            'feature_columns': self.feature_columns,
            'actual_n_estimators': self.actual_n_estimators,
            'horizon': self.horizon,
            'model_type': self.model_type,
            'loss_function': self.loss_function,
            'categorical_encoders': self.categorical_encoders,
            'use_monotone_constraints': self.use_monotone_constraints,
            'time_decay_lambda': self.time_decay_lambda,
            'use_cross_sectional_percentile': self.use_cross_sectional_percentile,
            'use_cross_sectional_zscore': self.use_cross_sectional_zscore,
            'feature_importance_threshold': self.feature_importance_threshold,
            'cs_feature_stats': self.cs_feature_stats,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"CatBoostRanker 模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.ranker_model = model_data['ranker_model']
        self.feature_columns = model_data['feature_columns']
        self.actual_n_estimators = model_data['actual_n_estimators']
        self.horizon = model_data.get('horizon', 20)
        self.model_type = model_data.get('model_type', 'catboost_ranker')
        self.loss_function = model_data.get('loss_function', 'YetiRank')
        self.categorical_encoders = model_data.get('categorical_encoders', {})
        self.use_monotone_constraints = model_data.get('use_monotone_constraints', True)
        self.time_decay_lambda = model_data.get('time_decay_lambda', 0.5)
        self.use_cross_sectional_percentile = model_data.get('use_cross_sectional_percentile', True)
        self.use_cross_sectional_zscore = model_data.get('use_cross_sectional_zscore', True)
        self.feature_importance_threshold = model_data.get('feature_importance_threshold', 0.0)
        self.cs_feature_stats = model_data.get('cs_feature_stats', {})
        print(f"CatBoostRanker 模型已从 {filepath} 加载")


class EnsembleModel:
    """融合模型 - 整合 LightGBM、GBDT、CatBoost 三个模型
    
    支持多种融合方法：
    1. 简单平均：三个模型的概率平均
    2. 加权平均：根据准确率加权
    3. 投票机制：多数投票
    4. 动态市场：根据市场状态动态选择融合方法
    """

    def __init__(self, fusion_method='weighted'):
        """
        Args:
            fusion_method: 融合方法 ('average'/'weighted'/'voting'/'dynamic-market')
        """
        self.lgbm_model = LightGBMModel()
        self.gbdt_model = GBDTModel()
        self.catboost_model = CatBoostModel()
        self.fusion_method = fusion_method
        self.model_accuracies = {}
        self.model_stds = {}  # 模型标准差（稳定性）
        self.horizon = 1
        self.dynamic_strategy = DynamicMarketStrategy()  # 初始化动态市场策略

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

    def load_model_stds(self):
        """加载模型稳定性数据（标准差）"""
        accuracy_file = 'data/model_accuracy.json'
        try:
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.model_stds = {
                    'lgbm': data.get(f'lgbm_{self.horizon}d', {}).get('std', 0.05),
                    'gbdt': data.get(f'gbdt_{self.horizon}d', {}).get('std', 0.05),
                    'catboost': data.get(f'catboost_{self.horizon}d', {}).get('std', 0.02)
                }
                logger.info(f"已加载模型稳定性数据: {self.model_stds}")
            else:
                logger.warning(r"未找到稳定性数据文件，使用默认值")
                self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}
        except Exception as e:
            logger.warning(f"加载稳定性数据失败: {e}")
            self.model_stds = {'lgbm': 0.05, 'gbdt': 0.05, 'catboost': 0.02}
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
        
        # 加载模型准确率和稳定性数据
        self.load_model_accuracy()
        self.load_model_stds()

        print("="*70)
        logger.info("融合模型已加载（包含3个子模型和准确率）")

    def predict(self, code, predict_date=None):
        """
        ✖️  已废弃：单股票预测方法

        此方法已被废弃，因为单股票预测无法计算正确的截面特征。
        请改用 predict_batch() 方法进行批量预测。

        Args:
            code: 股票代码
            predict_date: 预测日期

        Returns:
            dict: 融合预测结果
        """
        logger.warning("predict() 单股票预测已废弃，请使用 predict_batch() 方法。")

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
        elif self.fusion_method == 'dynamic-market':
            # 动态市场策略
            # 计算各模型的置信度（基于概率）
            confidences = [p for p in probabilities]
            
            # 获取恒生指数数据（用于市场状态检测）
            try:
                hsi_data = get_hsi_data_tencent()
                hsi_return_20d = None
                if hsi_data is not None and len(hsi_data) >= 20:
                    hsi_return_20d = (hsi_data['Close'].iloc[-1] - hsi_data['Close'].iloc[-20]) / hsi_data['Close'].iloc[-20]
                
                hsi_data_dict = {'return_20d': hsi_return_20d} if hsi_return_20d is not None else None
            except Exception as e:
                logger.warning(f"获取恒生指数数据失败: {e}")
                hsi_data_dict = None
            
            # 使用动态市场策略进行融合
            fused_prob, strategy_name = self.dynamic_strategy.predict(probabilities, confidences, hsi_data_dict)
            fused_pred = 1 if fused_prob > 0.5 else 0
            method_name = f"动态市场 ({strategy_name})"
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
        """
        批量预测：使用正确的截面特征计算

        核心改进：
        - 所有模型都使用 predict_batch() 进行批量预测
        - CatBoost 批量预测确保截面特征正确计算
        - LightGBM 和 GBDT 批量预测提升效率
        - 最后按股票代码融合三个模型的结果

        Args:
            codes: 股票代码列表
            predict_date: 预测日期

        Returns:
            list: 融合预测结果列表
        """
        if not codes:
            return []

        logger.info(f"Ensemble 批量预测 {len(codes)} 只股票...")

        # 所有模型都使用批量预测
        catboost_results = self.catboost_model.predict_batch(codes, predict_date, self.horizon)
        catboost_dict = {r['code']: r for r in catboost_results if r}

        lgbm_results = self.lgbm_model.predict_batch(codes, predict_date, self.horizon)
        lgbm_dict = {r['code']: r for r in lgbm_results if r}

        gbdt_results = self.gbdt_model.predict_batch(codes, predict_date, self.horizon)
        gbdt_dict = {r['code']: r for r in gbdt_results if r}

        # 按股票代码融合结果
        results = []
        for code in codes:
            result = self._fuse_predictions(code, lgbm_dict.get(code), gbdt_dict.get(code), catboost_dict.get(code))
            if result:
                results.append(result)

        logger.info(f"Ensemble 批量预测完成：{len(results)}/{len(codes)} 只股票")
        return results

    def _fuse_predictions(self, code, lgbm_result, gbdt_result, catboost_result):
        """
        融合三个模型的预测结果

        Args:
            code: 股票代码
            lgbm_result: LightGBM 预测结果
            gbdt_result: GBDT 预测结果
            catboost_result: CatBoost 预测结果

        Returns:
            dict: 融合预测结果
        """
        # 收集有效结果
        results = {}
        if lgbm_result:
            results['lgbm'] = lgbm_result
        if gbdt_result:
            results['gbdt'] = gbdt_result
        if catboost_result:
            results['catboost'] = catboost_result

        if not results:
            logger.error(f"所有模型预测失败: {code}")
            return None

        # 获取概率和预测
        probabilities = [r['probability'] for r in results.values()]
        predictions = [r['prediction'] for r in results.values()]

        # 融合逻辑（与原来的 predict() 方法相同）
        if self.fusion_method == 'average':
            fused_prob = np.mean(probabilities)
            method_name = "简单平均"
        elif self.fusion_method == 'weighted':
            weights = [self.model_accuracies.get(name, 0.5) for name in results.keys()]
            total_weight = sum(weights)
            if total_weight > 0:
                fused_prob = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
            else:
                fused_prob = np.mean(probabilities)
            method_name = "加权平均"
        elif self.fusion_method == 'dynamic-market':
            confidences = probabilities
            try:
                hsi_data = get_hsi_data_tencent()
                hsi_return_20d = None
                if hsi_data is not None and len(hsi_data) >= 20:
                    hsi_return_20d = (hsi_data['Close'].iloc[-1] - hsi_data['Close'].iloc[-20]) / hsi_data['Close'].iloc[-20]
                hsi_data_dict = {'return_20d': hsi_return_20d} if hsi_return_20d is not None else None
            except Exception as e:
                logger.warning(f"获取恒生指数数据失败: {e}")
                hsi_data_dict = None
            fused_prob, strategy_name = self.dynamic_strategy.predict(probabilities, confidences, hsi_data_dict)
            method_name = f"动态市场 ({strategy_name})"
        else:  # voting
            fused_pred = 1 if sum(predictions) >= len(predictions) / 2 else 0
            fused_prob = sum(predictions) / len(predictions)
            method_name = "投票机制"

        fused_pred = 1 if fused_prob > 0.5 else 0

        # 计算一致性
        if len(results) == 3:
            consistency_pct = 100 if predictions.count(predictions[0]) == 3 else (67 if predictions.count(predictions[0]) == 2 or predictions.count(predictions[1]) == 2 else 33)
        elif len(results) == 2:
            consistency_pct = 100 if predictions.count(predictions[0]) == 2 else 50
        else:
            consistency_pct = 100

        # 计算置信度
        if fused_prob > 0.60:
            confidence = "高"
            fused_direction = 1
        elif fused_prob > 0.50:
            confidence = "中"
            fused_direction = 0.5
        else:
            confidence = "低"
            fused_direction = 0

        # 构建结果
        first_result = list(results.values())[0]
        result = {
            'code': code,
            'name': STOCK_NAMES.get(code, code),
            'fusion_method': method_name,
            'fused_prediction': fused_direction,
            'fused_probability': float(fused_prob),
            'confidence': confidence,
            'consistency': f"{consistency_pct}%",
            'current_price': first_result['current_price'],
            'date': first_result['date'],
            'model_predictions': {name: {'prediction': int(r['prediction']), 'probability': float(r['probability'])}
                                  for name, r in results.items()}
        }

        return result
    
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
        elif self.fusion_method == 'advanced-dynamic':
            # 高级动态策略：CatBoost 主导（90%权重）
            lgbm_weight = 0.05
            gbdt_weight = 0.05
            catboost_weight = 0.90
        elif self.fusion_method == 'dynamic-market':
            # 动态市场策略：基于稳定性加权（CatBoost 权重约 2.4倍）
            stds = [self.model_stds.get('lgbm', 0.05), 
                    self.model_stds.get('gbdt', 0.05), 
                    self.model_stds.get('catboost', 0.02)]
            weights = [1/std for std in stds]
            total = sum(weights)
            lgbm_weight = weights[0] / total
            gbdt_weight = weights[1] / total
            catboost_weight = weights[2] / total
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
                       help='使用特征选择（只使用500个选择的特征，而不是全部2426个）')
    parser.add_argument('--skip-feature-selection', action='store_true',
                       help='跳过特征选择，直接使用已有的特征文件（适用于批量训练多个模型）')
    parser.add_argument('--fusion-method', type=str, default='weighted', 
                       choices=['average', 'weighted', 'voting', 'dynamic-market', 'advanced-dynamic'],
                       help='融合方法: average=简单平均, weighted=加权平均（基于准确率）, voting=投票机制, dynamic-market=动态市场策略（默认weighted）')

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
            # 融合模型需要三个子模型
            print("\n" + "=" * 70)
            print("🎭 准备融合模型的三个子模型")
            logger.info("=" * 70)
            
            # 检查子模型文件是否存在
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            gbdt_model_path = args.model_path.replace('.pkl', f'_gbdt{horizon_suffix}.pkl')
            catboost_model_path = args.model_path.replace('.pkl', f'_catboost{horizon_suffix}.pkl')
            
            all_submodels_exist = os.path.exists(lgbm_model_path) and os.path.exists(gbdt_model_path) and os.path.exists(catboost_model_path)
            
            if all_submodels_exist:
                print("\n✅ 所有子模型已存在，直接加载")
                logger.info("所有子模型已存在，直接加载")
                
                # 加载 LightGBM 模型
                print("\n📊 加载 LightGBM 模型...")
                lgbm_model = LightGBMModel()
                lgbm_model.load_model(lgbm_model_path)
                logger.info(f"LightGBM 模型已从 {lgbm_model_path} 加载")
                
                # 加载 GBDT 模型
                print("\n📊 加载 GBDT 模型...")
                gbdt_model = GBDTModel()
                gbdt_model.load_model(gbdt_model_path)
                logger.info(f"GBDT 模型已从 {gbdt_model_path} 加载")
                
                # 加载 CatBoost 模型
                print("\n📊 加载 CatBoost 模型...")
                catboost_model = CatBoostModel()
                catboost_model.load_model(catboost_model_path)
                logger.info(f"CatBoost 模型已从 {catboost_model_path} 加载")
            else:
                print("\n⚠️ 部分子模型不存在，开始训练缺失的子模型")
                logger.info("部分子模型不存在，开始训练缺失的子模型")
                
                # 训练或加载 LightGBM 模型
                print("\n📊 处理 LightGBM 模型...")
                lgbm_model = LightGBMModel()
                if os.path.exists(lgbm_model_path):
                    print(f"  ✅ LightGBM 模型已存在，直接加载")
                    lgbm_model.load_model(lgbm_model_path)
                    logger.info(f"LightGBM 模型已从 {lgbm_model_path} 加载")
                else:
                    print(f"  ⚠️ LightGBM 模型不存在，开始训练")
                    lgbm_feature_importance = lgbm_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
                    lgbm_model.save_model(lgbm_model_path)
                    lgbm_importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
                    lgbm_feature_importance.to_csv(lgbm_importance_path, index=False)
                    logger.info(f"LightGBM 模型已保存到 {lgbm_model_path}")
                    logger.info(f"特征重要性已保存到 {lgbm_importance_path}")
                
                # 训练或加载 GBDT 模型
                print("\n📊 处理 GBDT 模型...")
                gbdt_model = GBDTModel()
                if os.path.exists(gbdt_model_path):
                    print(f"  ✅ GBDT 模型已存在，直接加载")
                    gbdt_model.load_model(gbdt_model_path)
                    logger.info(f"GBDT 模型已从 {gbdt_model_path} 加载")
                else:
                    print(f"  ⚠️ GBDT 模型不存在，开始训练")
                    gbdt_feature_importance = gbdt_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
                    gbdt_model.save_model(gbdt_model_path)
                    gbdt_importance_path = gbdt_model_path.replace('.pkl', '_importance.csv')
                    gbdt_feature_importance.to_csv(gbdt_importance_path, index=False)
                    logger.info(f"GBDT 模型已保存到 {gbdt_model_path}")
                    logger.info(f"特征重要性已保存到 {gbdt_importance_path}")
                
                # 训练或加载 CatBoost 模型
                print("\n📊 处理 CatBoost 模型...")
                catboost_model = CatBoostModel()
                if os.path.exists(catboost_model_path):
                    print(f"  ✅ CatBoost 模型已存在，直接加载")
                    catboost_model.load_model(catboost_model_path)
                    logger.info(f"CatBoost 模型已从 {catboost_model_path} 加载")
                else:
                    print(f"  ⚠️ CatBoost 模型不存在，开始训练")
                    catboost_feature_importance = catboost_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
                    catboost_model.save_model(catboost_model_path)
                    catboost_importance_path = catboost_model_path.replace('.pkl', '_importance.csv')
                    catboost_feature_importance.to_csv(catboost_importance_path, index=False)
                    logger.info(f"CatBoost 模型已保存到 {catboost_model_path}")
                    logger.info(f"特征重要性已保存到 {catboost_importance_path}")
            
            print("\n" + "=" * 70)
            logger.info(r"融合模型的所有子模型已就绪！")
            logger.info("=" * 70)
        elif lgbm_model:
            feature_importance = lgbm_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            lgbm_model_path = args.model_path.replace('.pkl', f'_lgbm{horizon_suffix}.pkl')
            lgbm_model.save_model(lgbm_model_path)
            importance_path = lgbm_model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"\n特征重要性已保存到 {importance_path}")
        elif catboost_model:
            feature_importance = catboost_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
            catboost_model_path = args.model_path.replace('.pkl', f'_catboost{horizon_suffix}.pkl')
            catboost_model.save_model(catboost_model_path)
            importance_path = catboost_model_path.replace('.pkl', '_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"\n特征重要性已保存到 {importance_path}")
        else:
            feature_importance = gbdt_model.train(TRAINING_LIST, args.start_date, args.end_date, horizon=args.horizon, use_feature_selection=apply_feature_selection)
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

        # 预测所有股票（所有模型都使用批量预测）
        predictions = []
        if args.predict_date:
            print(f"基于日期: {args.predict_date}")

        if ensemble_model:
            # 使用融合模型批量预测（关键：确保截面特征正确）
            predictions = ensemble_model.predict_batch(WATCHLIST, args.predict_date)
        elif catboost_model:
            # CatBoost 使用批量预测（关键：确保截面特征正确）
            predictions = catboost_model.predict_batch(WATCHLIST, args.predict_date, args.horizon)
        elif lgbm_model:
            # LightGBM 使用批量预测
            predictions = lgbm_model.predict_batch(WATCHLIST, args.predict_date, args.horizon)
        elif gbdt_model:
            # GBDT 使用批量预测
            predictions = gbdt_model.predict_batch(WATCHLIST, args.predict_date, args.horizon)

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
                target_date = get_target_date_trading_days(pred['date'], horizon=args.horizon, stock_code=pred['code'])

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
            pred_df['target_date'] = pred_df.apply(
                lambda row: get_target_date_trading_days(row['date'], horizon=args.horizon, stock_code=row['code']),
                axis=1
            )

            pred_df_export = pred_df[['code', 'name', 'prediction', 'probability', 'current_price', 'data_date', 'target_date']]

            pred_path = args.model_path.replace('.pkl', f'_{model_file_suffix}_predictions{horizon_suffix}.csv')
            pred_df_export.to_csv(pred_path, index=False)
            print(f"\n预测结果已保存到 {pred_path}")

            # 保存预测到历史记录（用于性能监控和传导律检测）
            # 注意：同时保存所有周期的预测，以便后续验证传导律
            save_prediction_to_history(predictions, horizon=args.horizon, predict_date=args.predict_date)
            
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
            lgbm_f1 = f1_score(y_test, y_pred_lgbm, zero_division=0)
            print(f"\n准确率: {lgbm_accuracy:.4f}")
            print(f"F1分数: {lgbm_f1:.4f}")

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
            gbdt_f1 = f1_score(y_test, y_pred_gbdt, zero_division=0)
            print(f"\n准确率: {gbdt_accuracy:.4f}")
            print(f"F1分数: {gbdt_f1:.4f}")

            # 对比结果
            print("\n" + "="*70)
            logger.info("模型对比")
            print("="*70)
            print(f"LightGBM 准确率: {lgbm_accuracy:.4f}, F1分数: {lgbm_f1:.4f}")
            print(f"GBDT 准确率: {gbdt_accuracy:.4f}, F1分数: {gbdt_f1:.4f}")
            print(f"准确率差异: {abs(lgbm_accuracy - gbdt_accuracy):.4f}")
            print(f"F1分数差异: {abs(lgbm_f1 - gbdt_f1):.4f}")
            
            if gbdt_accuracy > lgbm_accuracy and gbdt_f1 > lgbm_f1:
                print(f"\n✅ GBDT 模型在准确率和F1分数上都表现更好")
            elif lgbm_accuracy > gbdt_accuracy and lgbm_f1 > gbdt_f1:
                print(f"\n✅ LightGBM 模型在准确率和F1分数上都表现更好")
            elif gbdt_accuracy > lgbm_accuracy:
                print(f"\n✅ GBDT 模型准确率更高，但F1分数比较...")
            elif lgbm_accuracy > gbdt_accuracy:
                print(f"\n✅ LightGBM 模型准确率更高，但F1分数比较...")
            else:
                print(f"\n⚖️  两种模型准确率相同，比较F1分数...")

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

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            print(f"\n准确率: {accuracy:.4f}")
            print(f"F1分数: {f1:.4f}")

    else:
        logger.error(f"不支持的运行模式: {args.mode}")
        print("请使用以下模式之一: train, evaluate, predict")
        sys.exit(1)


# 向后兼容别名
MLTradingModel = LightGBMModel


if __name__ == '__main__':
    main()
