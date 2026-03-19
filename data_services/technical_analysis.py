#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用技术分析工具
实现多种常用技术指标的计算，包括移动平均线、RSI、MACD等

【重要约定】统一百分比/分数格式：
- 所有风险函数返回的 'percentage' 字段统一使用分数形式（fraction）
- 分数形式：0.05 表示 5%，0.25 表示 25%
- 金额计算公式：amount = position_value * percentage
- 格式化显示：f"{percentage:.2%}"  # 0.05 → "5.00%"
- 适用的函数：calculate_var, calculate_expected_shortfall, calculate_max_drawdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_var(self, df, investment_style='medium_term', confidence_level=0.95):
        """
        计算风险价值(VaR)，时间维度与投资周期匹配
        
        参数:
        - df: 包含价格数据的DataFrame
        - investment_style: 投资风格
          - 'ultra_short_term': 超短线交易（日内/隔夜）
          - 'short_term': 波段交易（数天–数周）
          - 'medium_long_term': 中长期投资（1个月+）
        - confidence_level: 置信水平（默认0.95，即95%）
        
        返回:
        - VaR值（百分比）
        """
        if df.empty or len(df) < 20:
            return None
        
        # 根据投资风格确定VaR计算的时间窗口
        if investment_style == 'ultra_short_term':
            # 超短线交易：1日VaR
            var_window = 1
        elif investment_style == 'short_term':
            # 波段交易：5日VaR
            var_window = 5
        elif investment_style == 'medium_long_term':
            # 中长期投资：20日VaR（≈1个月）
            var_window = 20
        else:
            # 默认使用5日VaR
            var_window = 5
        
        # 确保有足够的历史数据
        required_data = max(var_window * 5, 30)  # 至少需要5倍时间窗口或30天的数据
        if len(df) < required_data:
            return None
        
        # 计算日收益率
        df['Returns'] = df['Close'].pct_change()
        returns = df['Returns'].dropna()
        
        if len(returns) < var_window:
            return None
        
        # 计算指定时间窗口的收益率
        if var_window == 1:
            # 1日VaR直接使用日收益率
            window_returns = returns
        else:
            # 多日VaR使用滚动收益率
            window_returns = df['Close'].pct_change(var_window).dropna()
        
        # 使用历史模拟法计算VaR
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(window_returns, var_percentile)
        
        # 返回绝对值（VaR通常表示为正数，表示最大可能损失）
        return abs(var_value)
    
    def calculate_moving_averages(self, df, periods=[5, 10, 20, 50, 100, 200]):
        """计算多种移动平均线"""
        if df.empty:
            return df
        
        for period in periods:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        return df
    
    def calculate_rsi(self, df, period=14):
        """计算RSI指标"""
        if df.empty:
            return df
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        if df.empty:
            return df
        
        exp1 = df['Close'].ewm(span=fast).mean()
        exp2 = df['Close'].ewm(span=slow).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """计算布林带（使用滞后数据避免数据泄漏）"""
        if df.empty:
            return df
        
        # 使用过去N天的数据计算布林带（避免使用当天数据）
        df['BB_middle'] = df['Close'].rolling(window=period).mean().shift(1)
        bb_std = df['Close'].rolling(window=period).std().shift(1)
        df['BB_upper'] = df['BB_middle'] + (bb_std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 1e-10)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
        
        return df
    
    def calculate_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """计算随机振荡器(KDJ)"""
        if df.empty:
            return df
        
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        df['K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['D'] = df['K'].rolling(window=d_period).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    def calculate_atr(self, df, period=14):
        """计算平均真实波幅(ATR)"""
        if df.empty:
            return df
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=period).mean()
        
        return df
    
    def calculate_volume_indicators(self, df, short_period=10, long_period=20, surge_threshold=1.2, shrink_threshold=0.8,
                              reversal_volume_threshold=1.5, continuation_volume_threshold=1.2):
        """计算成交量相关指标，提供多级成交量确认，区分反转型和延续型信号"""
        if df.empty or 'Volume' not in df.columns:
            return df
        
        # 成交量移动平均线
        df[f'Volume_MA{short_period}'] = df['Volume'].rolling(window=short_period).mean()
        df[f'Volume_MA{long_period}'] = df['Volume'].rolling(window=long_period).mean()
        
        # 成交量比率（当前成交量与长期均量的比率）
        df['Volume_Ratio'] = df['Volume'] / df[f'Volume_MA{long_period}']
        
        # 多级成交量突增检测
        df['Volume_Surge_Weak'] = df['Volume_Ratio'] > 1.2    # 弱突增
        df['Volume_Surge_Medium'] = df['Volume_Ratio'] > 1.5   # 中等突增
        df['Volume_Surge_Strong'] = df['Volume_Ratio'] > 2.0   # 强突增
        
        # 保持向后兼容
        df['Volume_Surge'] = df['Volume_Ratio'] > surge_threshold
        
        # 成交量萎缩检测（成交量低于长期均量的指定倍数）
        df['Volume_Shrink'] = df['Volume_Ratio'] < shrink_threshold
        
        # 成交量趋势（短期均线与长期均线的关系）
        df['Volume_Trend_Up'] = df[f'Volume_MA{short_period}'] > df[f'Volume_MA{long_period}']
        df['Volume_Trend_Down'] = df[f'Volume_MA{short_period}'] < df[f'Volume_MA{long_period}']
        
        # 价量配合指标（多级）
        if 'Close' in df.columns:
            # 计算价格变化
            df['Price_Change'] = df['Close'].pct_change()
            
            # 价格方向历史追踪
            df['Price_Direction'] = np.sign(df['Price_Change'])
            df['Price_Direction_Prev1'] = df['Price_Direction'].shift(1)
            df['Price_Direction_Prev2'] = df['Price_Direction'].shift(2)
            
            # 处理NaN值，确保数据安全
            df['Price_Direction'] = df['Price_Direction'].fillna(0)
            df['Price_Direction_Prev1'] = df['Price_Direction_Prev1'].fillna(0)
            df['Price_Direction_Prev2'] = df['Price_Direction_Prev2'].fillna(0)
            
            # 反转型价量配合信号检测（前一天价格相反方向+成交量放大）
            df['Price_Volume_Reversal_Bullish'] = (
                (df['Price_Direction'] > 0) &  # 当日上涨
                (df['Price_Direction_Prev1'] < 0) &  # 前一日下跌
                (df['Volume_Ratio'] > reversal_volume_threshold)
            )
            
            df['Price_Volume_Reversal_Bearish'] = (
                (df['Price_Direction'] < 0) &  # 当日下跌
                (df['Price_Direction_Prev1'] > 0) &  # 前一日上涨
                (df['Volume_Ratio'] > reversal_volume_threshold)
            )
            
            # 延续型价量配合信号检测（连续同向价格变化+成交量放大）
            df['Price_Volume_Continuation_Bullish'] = (
                (df['Price_Direction'] > 0) &  # 当日上涨
                (df['Price_Direction_Prev1'] > 0) &  # 前一日也上涨
                (df['Volume_Ratio'] > continuation_volume_threshold)
            )
            
            df['Price_Volume_Continuation_Bearish'] = (
                (df['Price_Direction'] < 0) &  # 当日下跌
                (df['Price_Direction_Prev1'] < 0) &  # 前一日也下跌
                (df['Volume_Ratio'] > continuation_volume_threshold)
            )
            
            # 成交量与价格变化的相关性（多级指标）- 保持原有逻辑
            df['Price_Volume_Bullish_Weak'] = (df['Price_Change'] > 0) & (df['Volume_Surge_Weak'])
            df['Price_Volume_Bullish_Medium'] = (df['Price_Change'] > 0) & (df['Volume_Surge_Medium'])
            df['Price_Volume_Bullish_Strong'] = (df['Price_Change'] > 0) & (df['Volume_Surge_Strong'])
            
            df['Price_Volume_Bearish_Weak'] = (df['Price_Change'] < 0) & (df['Volume_Surge_Weak'])
            df['Price_Volume_Bearish_Medium'] = (df['Price_Change'] < 0) & (df['Volume_Surge_Medium'])
            df['Price_Volume_Bearish_Strong'] = (df['Price_Change'] < 0) & (df['Volume_Surge_Strong'])
            
            # 保持向后兼容：合并反转型和延续型信号
            df['Price_Volume_Bullish'] = (
                df['Price_Volume_Reversal_Bullish'] | df['Price_Volume_Continuation_Bullish']
            )
            df['Price_Volume_Bearish'] = (
                df['Price_Volume_Reversal_Bearish'] | df['Price_Volume_Continuation_Bearish']
            )
            
            # 保持向后兼容的原始逻辑
            df['Price_Volume_Bullish_Original'] = (df['Price_Change'] > 0) & (df['Volume_Surge'])
            df['Price_Volume_Bearish_Original'] = (df['Price_Change'] < 0) & (df['Volume_Surge'])
        
        return df
    
    def calculate_cci(self, df, period=20):
        """计算商品通道指数(CCI)"""
        if df.empty:
            return df
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=period).mean()
        md = abs(tp - ma).rolling(window=period).mean()
        df['CCI'] = (tp - ma) / (0.015 * md)
        
        return df
    
    def calculate_obv(self, df):
        """计算能量潮指标(OBV)"""
        if df.empty:
            return df
        
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    def calculate_ichimoku_components(self, df):
        """计算Ichimoku云图组件（作为参考，虽然用户不需要完整的Ichimoku策略）"""
        if df.empty:
            return df
        
        # 转化线 (Tenkan-sen)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan_sen'] = (high_9 + low_9) / 2
        
        # 基准线 (Kijun-sen)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun_sen'] = (high_26 + low_26) / 2
        
        # 先行线A (Senkou Span A)
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        # 先行线B (Senkou Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # 迟延线 (Chikou Span)
        df['Chikou_Span'] = df['Close'].shift(-26)
        
        return df
    
    def _get_volume_level(self, row):
        """获取成交量突增等级"""
        if row.get('Volume_Surge_Strong', False):
            return "(强)"
        elif row.get('Volume_Surge_Medium', False):
            return "(中)"
        elif row.get('Volume_Surge_Weak', False):
            return "(弱)"
        else:
            return "(普通)"
    
    def calculate_cmf(self, df, period=20):
        """
        计算Chaikin Money Flow (CMF)指标
        
        参数:
        - df: 包含OHLCV数据的DataFrame
        - period: 计算周期，默认20
        
        返回:
        - 包含CMF列的DataFrame
        """
        if df.empty or len(df) < period + 1:
            return df
        
        # 计算Money Flow Multiplier
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.fillna(0)  # 处理除零情况
        
        # 计算Money Flow Volume
        mfv = mfm * df['Volume']
        
        # 计算CMF（Money Flow Volume的滚动和 / Volume的滚动和）
        df['CMF'] = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        
        return df
    
    def calculate_intraday_amplitude(self, df):
        """
        计算日内振幅
        
        公式：(最高价 - 最低价) / 开盘价 × 100%
        使用方法：
          - 振幅 > 5% → 高波动
          - 振幅 < 2% → 低波动
        
        参数:
        - df: 包含OHLCV数据的DataFrame
        
        返回:
        - 包含日内振幅列的DataFrame
        """
        if df.empty or len(df) < 1:
            return df
        
        # 计算日内振幅（以百分比表示）
        df['Intraday_Amplitude'] = ((df['High'] - df['Low']) / df['Open']) * 100
        
        return df
    
    def calculate_trend_slope(self, df, period=20):
        """
        计算趋势斜率（线性回归斜率）
        
        使用方法：
          - 斜率 > 0 → 上升趋势
          - 斜率 < 0 → 下降趋势
          - 斜率 = 0 → 横盘震荡
        
        参数:
        - df: 包含价格数据的DataFrame
        - period: 计算周期，默认20
        
        返回:
        - 包含趋势斜率列的DataFrame
        """
        if df.empty or len(df) < period:
            return df
        
        # 使用线性回归计算斜率
        def calc_slope(prices):
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            # 标准化斜率（相对于平均价格）
            normalized_slope = slope / np.mean(prices) * 100
            return normalized_slope
        
        df['Trend_Slope'] = df['Close'].rolling(window=period).apply(calc_slope, raw=True)
        
        return df
    
    def calculate_bias(self, df, periods=[6, 12, 24]):
        """
        计算乖离率
        
        公式：(当前价格 - MA) / MA × 100%
        使用方法：
          - 正乖离过大 → 回调风险
          - 负乖离过大 → 反弹机会
        
        参数:
        - df: 包含价格数据的DataFrame
        - periods: 计算周期列表，默认[6, 12, 24]
        
        返回:
        - 包含乖离率列的DataFrame
        """
        if df.empty:
            return df
        
        for period in periods:
            ma_col = f'MA{period}'
            bias_col = f'BIAS{period}'
            
            # 如果MA不存在，先计算
            if ma_col not in df.columns:
                df[ma_col] = df['Close'].rolling(window=period).mean()
            
            # 计算乖离率
            df[bias_col] = ((df['Close'] - df[ma_col]) / df[ma_col]) * 100
        
        return df
    
    def calculate_ma_alignment(self, df, short_period=20, medium_period=50, long_period=200):
        """
        计算均线排列
        
        使用方法：
          - 多头排列：短期均线 > 中期均线 > 长期均线
          - 空头排列：短期均线 < 中期均线 < 长期均线
        
        参数:
        - df: 包含价格数据的DataFrame
        - short_period: 短期均线周期，默认20
        - medium_period: 中期均线周期，默认50
        - long_period: 长期均线周期，默认200
        
        返回:
        - 包含均线排列列的DataFrame
        """
        if df.empty:
            return df
        
        # 确保均线存在
        short_ma = f'MA{short_period}'
        medium_ma = f'MA{medium_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in df.columns:
            df[short_ma] = df['Close'].rolling(window=short_period).mean()
        if medium_ma not in df.columns:
            df[medium_ma] = df['Close'].rolling(window=medium_period).mean()
        if long_ma not in df.columns:
            df[long_ma] = df['Close'].rolling(window=long_period).mean()
        
        # 判断均线排列
        df['MA_Alignment_Bullish'] = (df[short_ma] > df[medium_ma]) & (df[medium_ma] > df[long_ma])
        df['MA_Alignment_Bearish'] = (df[short_ma] < df[medium_ma]) & (df[medium_ma] < df[long_ma])
        
        # 均线排列强度（多头排列的数量减去空头排列的数量）
        df['MA_Alignment_Strength'] = (
            df[short_ma] > df[medium_ma]
        ).astype(int) + (
            df[medium_ma] > df[long_ma]
        ).astype(int) - (
            df[short_ma] < df[medium_ma]
        ).astype(int) - (
            df[medium_ma] < df[long_ma]
        ).astype(int)
        
        return df
    
    def calculate_skewness(self, df, period=20):
        """
        计算偏度（Skewness）
        
        使用方法：
          - 偏度 > 0 → 正偏（右尾长）
          - 偏度 < 0 → 负偏（左尾长）
        
        参数:
        - df: 包含价格数据的DataFrame
        - period: 计算周期，默认20
        
        返回:
        - 包含偏度列的DataFrame
        """
        if df.empty or len(df) < period:
            return df
        
        # 计算收益率的偏度
        returns = df['Close'].pct_change()
        df['Returns_Skewness'] = returns.rolling(window=period).skew()
        
        return df
    
    def calculate_kurtosis(self, df, period=20):
        """
        计算峰度（Kurtosis）
        
        使用方法：
          - 峰度 > 0 → 尖峰分布
          - 峰度 < 0 → 扁平分布
        
        参数:
        - df: 包含价格数据的DataFrame
        - period: 计算周期，默认20
        
        返回:
        - 包含峰度列的DataFrame
        """
        if df.empty or len(df) < period:
            return df
        
        # 计算收益率的峰度
        returns = df['Close'].pct_change()
        df['Returns_Kurtosis'] = returns.rolling(window=period).kurt()
        
        return df
    
    def calculate_all_indicators(self, df):
        """计算所有技术指标"""
        if df.empty:
            return df
        
        # 计算移动平均线
        df = self.calculate_moving_averages(df)
        
        # 计算RSI
        df = self.calculate_rsi(df)
        
        # 计算MACD
        df = self.calculate_macd(df)
        
        # 计算布林带
        df = self.calculate_bollinger_bands(df)
        
        # 计算随机振荡器
        df = self.calculate_stochastic_oscillator(df)
        
        # 计算ATR
        df = self.calculate_atr(df)
        
        # 计算CCI
        df = self.calculate_cci(df)
        
        # 计算OBV
        df = self.calculate_obv(df)
        
        # 计算CMF
        df = self.calculate_cmf(df)
        
        # 计算成交量指标
        df = self.calculate_volume_indicators(df)
        
        # ========== 新增指标 ==========
        # 计算日内振幅
        df = self.calculate_intraday_amplitude(df)
        
        # 计算趋势斜率
        df = self.calculate_trend_slope(df)
        
        # 计算乖离率
        df = self.calculate_bias(df)
        
        # 计算均线排列
        df = self.calculate_ma_alignment(df)
        
        # 计算偏度
        df = self.calculate_skewness(df)
        
        # 计算峰度
        df = self.calculate_kurtosis(df)
        # ==========================
        
        return df
    
    def generate_buy_sell_signals(self, df):
        """基于技术指标生成买卖信号，包含成交量确认"""
        if df.empty:
            return df
        
        # 初始化信号列
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Signal_Description'] = ''
        
        # 计算成交量指标
        if 'Volume' in df.columns:
            # 成交量移动平均线
            df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # 成交量比率（当前成交量与20日均量的比率）
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            # 成交量突增检测（成交量超过20日均量的1.5倍）
            df['Volume_Surge'] = df['Volume_Ratio'] > 1.5
            
            # 成交量萎缩检测（成交量低于20日均量的0.7倍）
            df['Volume_Shrink'] = df['Volume_Ratio'] < 0.7
        
        # 计算一些必要的中间指标
        if 'MA20' in df.columns and 'MA50' in df.columns:
            # 金叉死叉信号
            df['MA20_above_MA50'] = df['MA20'] > df['MA50']
            df['MA20_below_MA50'] = df['MA20'] < df['MA50']
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            # MACD交叉信号
            df['MACD_above_signal'] = df['MACD'] > df['MACD_signal']
            df['MACD_below_signal'] = df['MACD'] < df['MACD_signal']
        
        if 'RSI' in df.columns:
            # RSI超买超卖信号
            df['RSI_oversold'] = df['RSI'] < 30
            df['RSI_overbought'] = df['RSI'] > 70
        
        if 'Close' in df.columns and 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            # 布林带信号
            df['Price_above_BB_upper'] = df['Close'] > df['BB_upper']
            df['Price_below_BB_lower'] = df['Close'] < df['BB_lower']
        
        # 生成买入信号逻辑
        for i in range(1, len(df)):
            buy_conditions = []
            sell_conditions = []
            
            # 分级成交量确认检查
            def check_volume_confirmation(signal_type, strength='medium'):
                """检查不同信号类型的成交量确认要求"""
                if signal_type == 'trend':  # 趋势信号（MA交叉）
                    if strength == 'weak':
                        # 趋势信号弱确认：成交量突增(弱)或成交量趋势向上或成交量比率>0.9
                        return (df.iloc[i].get('Volume_Surge_Weak', False) or 
                                df.iloc[i].get('Volume_Trend_Up', False) or 
                                df.iloc[i].get('Volume_Ratio', 0) > 0.9)
                    elif strength == 'medium':
                        return df.iloc[i].get('Volume_Surge_Weak', False) or df.iloc[i].get('Volume_Trend_Up', False)
                    else:  # strong
                        return df.iloc[i].get('Volume_Surge_Medium', False)
                
                elif signal_type == 'momentum':  # 动量信号（MACD、RSI）
                    if strength == 'weak':
                        # 动量信号弱确认：成交量突增(弱)或成交量比率>1.0
                        return (df.iloc[i].get('Volume_Surge_Weak', False) or 
                                df.iloc[i].get('Volume_Ratio', 0) > 1.0)
                    elif strength == 'medium':
                        return df.iloc[i].get('Volume_Surge_Weak', False)
                    else:  # strong
                        return df.iloc[i].get('Volume_Surge_Medium', False)
                
                elif signal_type == 'price_action':  # 价格行为信号（布林带）
                    if strength == 'weak':
                        return df.iloc[i].get('Volume_Surge_Weak', False)
                    elif strength == 'medium':
                        return df.iloc[i].get('Volume_Surge_Medium', False)
                    else:  # strong
                        return df.iloc[i].get('Volume_Surge_Strong', False)
                
                elif signal_type == 'price_volume':  # 价量配合信号
                    return True  # 价量配合信号本身就是成交量确认的
                
                return True  # 默认通过
            
            # 条件1: 价格在上升趋势中 (MA20 > MA50) - 趋势信号，使用弱强度确认
            if ('MA20_above_MA50' in df.columns and df.iloc[i]['MA20_above_MA50'] and 
                not df.iloc[i-1]['MA20_above_MA50'] and check_volume_confirmation('trend', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else ("弱" if df.iloc[i].get('Volume_Surge_Weak', False) else "普通"))
                buy_conditions.append(f"上升趋势形成(成交量{volume_level}确认)")
            
            # 条件2: MACD金叉 - 动量信号，使用弱强度确认
            if ('MACD_above_signal' in df.columns and df.iloc[i]['MACD_above_signal'] and 
                not df.iloc[i-1]['MACD_above_signal'] and check_volume_confirmation('momentum', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                buy_conditions.append(f"MACD金叉(成交量{volume_level}确认)")
            
            # 条件3: RSI从超卖区域回升 - 动量信号，使用弱强度确认
            if ('RSI_oversold' in df.columns and not df.iloc[i]['RSI_oversold'] and 
                df.iloc[i-1]['RSI_oversold'] and check_volume_confirmation('momentum', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                buy_conditions.append(f"RSI超卖反弹(成交量{volume_level}确认)")
            
            # 条件4: 价格从布林带下轨反弹 - 价格行为信号，使用中等强度确认
            if ('Price_below_BB_lower' in df.columns and not df.iloc[i]['Price_below_BB_lower'] and 
                df.iloc[i-1]['Price_below_BB_lower'] and check_volume_confirmation('price_action', 'medium')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                buy_conditions.append(f"布林带下轨反弹(成交量{volume_level}确认)")
            
            # 条件5: 价量配合买入信号（价格上涨且成交量放大）
            if ('Close' in df.columns and 
                df.iloc[i]['Close'] > df.iloc[i-1]['Close'] and 
                df.iloc[i].get('Price_Volume_Bullish_Weak', False)):
                
                # 检查是反转型还是延续型信号
                if df.iloc[i].get('Price_Volume_Reversal_Bullish', False):
                    # 反转型信号：前一天下跌，当天上涨
                    volume_level = self._get_volume_level(df.iloc[i])
                    buy_conditions.append(f"价量配合反转{volume_level}")
                elif df.iloc[i].get('Price_Volume_Continuation_Bullish', False):
                    # 延续型信号：连续上涨，成交量放大
                    volume_level = self._get_volume_level(df.iloc[i])
                    buy_conditions.append(f"价量配合延续{volume_level}")
                else:
                    # 兼容原有逻辑
                    if df.iloc[i].get('Price_Volume_Bullish_Strong', False):
                        buy_conditions.append("价量配合上涨(强)")
                    elif df.iloc[i].get('Price_Volume_Bullish_Medium', False):
                        buy_conditions.append("价量配合上涨(中)")
                    else:
                        buy_conditions.append("价量配合上涨(弱)")
            
            # 生成买入信号
            if buy_conditions:
                df.at[df.index[i], 'Buy_Signal'] = True
                # 如果已有信号描述，先保存
                existing_desc = df.iloc[i].get('Signal_Description', '')
                if existing_desc:
                    # 如果已有描述，说明之前已经有卖出信号，需要合并
                    df.at[df.index[i], 'Signal_Description'] = existing_desc + " | 买入信号: " + ", ".join(buy_conditions)
                else:
                    df.at[df.index[i], 'Signal_Description'] = "买入信号: " + ", ".join(buy_conditions)
            
            # 生成卖出信号逻辑
            # 条件1: 价格在下降趋势中 (MA20 < MA50) - 趋势信号，使用弱强度确认
            if ('MA20_below_MA50' in df.columns and df.iloc[i]['MA20_below_MA50'] and 
                not df.iloc[i-1]['MA20_below_MA50'] and check_volume_confirmation('trend', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else ("弱" if df.iloc[i].get('Volume_Surge_Weak', False) else "普通"))
                sell_conditions.append(f"下降趋势形成(成交量{volume_level}确认)")
            
            # 条件2: MACD死叉 - 动量信号，使用弱强度确认
            if ('MACD_below_signal' in df.columns and df.iloc[i]['MACD_below_signal'] and 
                not df.iloc[i-1]['MACD_below_signal'] and check_volume_confirmation('momentum', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                sell_conditions.append(f"MACD死叉(成交量{volume_level}确认)")
            
            # 条件3: RSI从超买区域回落 - 动量信号，使用弱强度确认
            if ('RSI_overbought' in df.columns and not df.iloc[i]['RSI_overbought'] and 
                df.iloc[i-1]['RSI_overbought'] and check_volume_confirmation('momentum', 'weak')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                sell_conditions.append(f"RSI超买回落(成交量{volume_level}确认)")
            
            # 条件4: 价格跌破布林带上轨 - 价格行为信号，使用中等强度确认
            if ('Price_above_BB_upper' in df.columns and not df.iloc[i]['Price_above_BB_upper'] and 
                df.iloc[i-1]['Price_above_BB_upper'] and check_volume_confirmation('price_action', 'medium')):
                volume_level = "强" if df.iloc[i].get('Volume_Surge_Strong', False) else ("中" if df.iloc[i].get('Volume_Surge_Medium', False) else "弱")
                sell_conditions.append(f"跌破布林带上轨(成交量{volume_level}确认)")
            
            # 条件5: 价量配合卖出信号（价格下跌且成交量放大）
            if ('Close' in df.columns and 
                df.iloc[i]['Close'] < df.iloc[i-1]['Close'] and 
                df.iloc[i].get('Price_Volume_Bearish_Weak', False)):
                
                # 检查是反转型还是延续型信号
                if df.iloc[i].get('Price_Volume_Reversal_Bearish', False):
                    # 反转型信号：前一天上涨，当天下跌
                    volume_level = self._get_volume_level(df.iloc[i])
                    sell_conditions.append(f"价量配合反转{volume_level}")
                elif df.iloc[i].get('Price_Volume_Continuation_Bearish', False):
                    # 延续型信号：连续下跌，成交量放大
                    volume_level = self._get_volume_level(df.iloc[i])
                    sell_conditions.append(f"价量配合延续{volume_level}")
                else:
                    # 兼容原有逻辑
                    if df.iloc[i].get('Price_Volume_Bearish_Strong', False):
                        sell_conditions.append("价量配合下跌(强)")
                    elif df.iloc[i].get('Price_Volume_Bearish_Medium', False):
                        sell_conditions.append("价量配合下跌(中)")
                    else:
                        sell_conditions.append("价量配合下跌(弱)")
            
            # 生成卖出信号
            if sell_conditions:
                df.at[df.index[i], 'Sell_Signal'] = True
                # 如果已有信号描述，先保存
                existing_desc = df.iloc[i].get('Signal_Description', '')
                if existing_desc:
                    # 如果已有描述，说明之前已经有买入信号，需要合并
                    df.at[df.index[i], 'Signal_Description'] = existing_desc + " | 卖出信号: " + ", ".join(sell_conditions)
                else:
                    df.at[df.index[i], 'Signal_Description'] = "卖出信号: " + ", ".join(sell_conditions)
        
        return df
    
    def get_chip_distribution(self, df, num_bins=20):
        """
        计算筹码分布 - 基于成交量的简单分箱法
        
        参数:
        - df: DataFrame，必须包含 Close 和 Volume 列
        - num_bins: 价格区间数量（默认 20）
        
        返回:
        - dict: 包含筹码分布分析结果
        """
        if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
            return None
        
        # 获取价格范围
        price_min = df['Close'].min()
        price_max = df['Close'].max()
        
        # 如果价格范围太小，避免除零错误
        if price_max - price_min < 1e-6:
            return None
        
        # 创建价格区间
        price_bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # 统计每个价格区间的成交量
        volume_by_bin = []
        for i in range(num_bins):
            mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i + 1])
            volume_by_bin.append(df.loc[mask, 'Volume'].sum())
        
        volume_by_bin = np.array(volume_by_bin)
        total_volume = volume_by_bin.sum()
        
        if total_volume == 0:
            return None
        
        # 计算筹码集中度（HHI指数）
        concentration = np.sum((volume_by_bin / total_volume) ** 2)
        
        # 找出筹码最集中的区间
        max_bin_idx = np.argmax(volume_by_bin)
        concentration_area = (price_bins[max_bin_idx], price_bins[max_bin_idx + 1])
        
        # 计算当前价格上方的筹码比例
        current_price = df['Close'].iloc[-1]
        current_bin_idx = np.searchsorted(price_bins, current_price) - 1
        
        if current_bin_idx >= 0:
            resistance_volume = volume_by_bin[current_bin_idx:].sum()
            resistance_ratio = resistance_volume / total_volume
        else:
            resistance_ratio = 0
        
        return {
            'concentration': concentration,
            'concentration_level': '高' if concentration > 0.3 else '中' if concentration > 0.15 else '低',
            'concentration_area': concentration_area,
            'resistance_ratio': resistance_ratio,
            'resistance_level': '高' if resistance_ratio > 0.6 else '中' if resistance_ratio > 0.3 else '低',
            'current_price': current_price
        }
    
    def analyze_trend(self, df):
        """分析趋势"""
        if df.empty or len(df) < 50:  # 降低最小数据要求
            return "数据不足"
        
        # 获取最新数据
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else np.nan
        ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns and not pd.isna(df['MA50'].iloc[-1]) else np.nan
        ma200 = df['MA200'].iloc[-1] if 'MA200' in df.columns and not pd.isna(df['MA200'].iloc[-1]) else np.nan
        
        # 如果有200日均线数据，使用完整趋势分析
        if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
            # 多头排列：价格 > MA20 > MA50 > MA200
            if current_price > ma20 > ma50 > ma200:
                return "强势多头"
            # 空头排列：价格 < MA20 < MA50 < MA200
            elif current_price < ma20 < ma50 < ma200:
                return "弱势空头"
            # 震荡
            else:
                return "震荡整理"
        # 如果没有200日均线数据，使用较短期的趋势分析
        elif not pd.isna(ma20) and not pd.isna(ma50):
            # 多头排列：价格 > MA20 > MA50
            if current_price > ma20 > ma50:
                return "多头趋势"
            # 空头排列：价格 < MA20 < MA50
            elif current_price < ma20 < ma50:
                return "空头趋势"
            # 震荡
            else:
                return "震荡"
        # 如果连短期均线都没有，只看价格趋势
        elif len(df) >= 20:
            # 比较最近价格与20日均价
            recent_price = df['Close'].iloc[-1]
            past_price = df['Close'].iloc[-20]  # 20天前的价格
            
            if recent_price > past_price:
                return "短期上涨"
            else:
                return "短期下跌"
        else:
            return "数据不足"

class MarketAnalyzer:
    def __init__(self, symbols):
        self.symbols = symbols
        self.analyzer = TechnicalAnalyzer()
        
    def get_historical_data(self, period="1y"):
        """获取历史数据"""
        data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = {
                        'name': ticker.info.get('longName', ticker.info.get('shortName', symbol)),
                        'data': hist
                    }
                    print(f"✅ {symbol} 数据获取成功")
                else:
                    print(f"⚠️ {symbol} 数据为空")
            except Exception as e:
                print(f"❌ 获取 {symbol} 数据失败: {e}")
        
        return data
    
    def run_analysis(self, period="1y"):
        """运行技术分析"""
        print("="*60)
        print("📈 通用技术分析系统")
        print("="*60)
        
        # 获取历史数据
        data = self.get_historical_data(period)
        
        if not data:
            print("❌ 未能获取到任何数据，分析终止")
            return None
        
        results = {}
        
        for symbol, info in data.items():
            print(f"\n📊 分析 {info['name']} ({symbol})...")
            
            # 计算技术指标
            df_with_indicators = self.analyzer.calculate_all_indicators(info['data'].copy())
            
            # 生成买卖信号
            df_with_signals = self.analyzer.generate_buy_sell_signals(df_with_indicators)
            
            # 分析趋势
            trend = self.analyzer.analyze_trend(df_with_signals)
            
            results[symbol] = {
                'name': info['name'],
                'data': df_with_signals,
                'trend': trend
            }
            
            # 显示最新的指标值
            latest = df_with_signals.iloc[-1]
            print(f"  趋势: {trend}")
            if 'RSI' in latest:
                print(f"  RSI (14): {latest['RSI']:.2f}")
            if 'MACD' in latest:
                print(f"  MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
            if 'MA20' in latest and 'MA50' in latest and 'MA200' in latest:
                print(f"  MA20: {latest['MA20']:.2f}, MA50: {latest['MA50']:.2f}, MA200: {latest['MA200']:.2f}")
            if 'BB_position' in latest:
                print(f"  布林带位置: {latest['BB_position']:.2f}")
        
        return results

# ==================== TAV 方法论整合 ====================

class TAVConfig:
    """TAV 配置类，支持不同资产类型的差异化配置"""
    
    # 股票市场配置
    STOCK_CONFIG = {
        'weights': {
            'trend': 0.4,      # 股票注重趋势
            'momentum': 0.35,  # 动量次之
            'volume': 0.25     # 成交量验证
        },
        'thresholds': {
            'strong_signal': 75,    # 强信号阈值
            'medium_signal': 50,    # 中等信号阈值
            'weak_signal': 25       # 弱信号阈值
        },
        'trend': {
            'ma_periods': [20, 50, 200],
            'trend_threshold': 0.02,  # 2%的趋势确认阈值
            'consolidation_threshold': 0.01  # 1%的震荡阈值
        },
        'momentum': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        'volume': {
            'volume_ma_period': 20,
            'surge_threshold_weak': 1.2,
            'surge_threshold_medium': 1.5,
            'surge_threshold_strong': 2.0,
            'shrink_threshold': 0.8
        }
    }
    
    # 加密货币配置
    CRYPTO_CONFIG = {
        'weights': {
            'trend': 0.3,      # 加密货币波动大，趋势权重降低
            'momentum': 0.45,  # 动量更重要
            'volume': 0.25     # 成交量同样重要
        },
        'thresholds': {
            'strong_signal': 80,    # 更高的强信号阈值
            'medium_signal': 55,    # 中等信号阈值
            'weak_signal': 30       # 弱信号阈值
        },
        'trend': {
            'ma_periods': [10, 30, 100],  # 更短期的均线
            'trend_threshold': 0.03,      # 更高的趋势确认阈值
            'consolidation_threshold': 0.02
        },
        'momentum': {
            'rsi_period': 14,
            'rsi_oversold': 25,           # 更极端的超卖阈值
            'rsi_overbought': 75,         # 更极端的超买阈值
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        'volume': {
            'volume_ma_period': 20,
            'surge_threshold_weak': 1.3,  # 更高的成交量突增阈值
            'surge_threshold_medium': 1.8,
            'surge_threshold_strong': 2.5,
            'shrink_threshold': 0.7
        }
    }
    
    # 黄金市场配置
    GOLD_CONFIG = {
        'weights': {
            'trend': 0.45,     # 黄金趋势性强
            'momentum': 0.3,   # 动量相对次要
            'volume': 0.25     # 成交量验证
        },
        'thresholds': {
            'strong_signal': 75,
            'medium_signal': 50,
            'weak_signal': 25
        },
        'trend': {
            'ma_periods': [20, 50, 200],  # 黄金使用标准均线
            'trend_threshold': 0.015,     # 中等趋势确认阈值
            'consolidation_threshold': 0.01
        },
        'momentum': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        'volume': {
            'volume_ma_period': 20,
            'surge_threshold_weak': 1.15, # 黄金对成交量敏感度较低
            'surge_threshold_medium': 1.3,
            'surge_threshold_strong': 1.6,
            'shrink_threshold': 0.85
        }
    }
    
    # 恒生指数配置
    HSI_CONFIG = {
        'weights': {
            'trend': 0.5,      # 指数趋势最重要
            'momentum': 0.3,   # 动量次之
            'volume': 0.2      # 成交量权重略低
        },
        'thresholds': {
            'strong_signal': 75,
            'medium_signal': 50,
            'weak_signal': 25
        },
        'trend': {
            'ma_periods': [20, 50, 200],
            'trend_threshold': 0.015,
            'consolidation_threshold': 0.01
        },
        'momentum': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        'volume': {
            'volume_ma_period': 20,
            'surge_threshold_weak': 1.2,
            'surge_threshold_medium': 1.5,
            'surge_threshold_strong': 2.0,
            'shrink_threshold': 0.8
        }
    }
    
    @classmethod
    def get_config(cls, asset_type='stock'):
        """根据资产类型获取配置"""
        config_map = {
            'stock': cls.STOCK_CONFIG,
            'crypto': cls.CRYPTO_CONFIG,
            'gold': cls.GOLD_CONFIG,
            'hsi': cls.HSI_CONFIG
        }
        return config_map.get(asset_type, cls.STOCK_CONFIG)
    
    @classmethod
    def detect_asset_type(cls, symbol):
        """根据股票代码自动检测资产类型"""
        symbol_upper = symbol.upper()
        
        # 加密货币检测
        crypto_patterns = ['-USD', 'BTC', 'ETH', 'USDT']
        if any(pattern in symbol_upper for pattern in crypto_patterns):
            return 'crypto'
        
        # 黄金检测
        gold_patterns = ['GC=F', 'GOLD', 'XAU', 'GLD']
        if any(pattern in symbol_upper for pattern in gold_patterns):
            return 'gold'
        
        # 恒生指数检测
        hsi_patterns = ['^HSI', 'HSI', '0700.HK']
        if any(pattern in symbol_upper for pattern in hsi_patterns):
            return 'hsi'
        
        # 默认为股票
        return 'stock'


class TAVScorer:
    """TAV 评分系统：趋势(Trend) + 动量(Acceleration/Momentum) + 成交量(Volume)"""
    
    def __init__(self, config=None):
        self.config = config or TAVConfig.STOCK_CONFIG
        self.weights = self.config['weights']
        self.thresholds = self.config['thresholds']
    
    def calculate_tav_score(self, df, asset_type='stock'):
        """计算TAV综合评分 (0-100分)"""
        if df.empty:
            return 0, {'trend': 0, 'momentum': 0, 'volume': 0}, "数据不足"
        
        # 获取资产类型特定配置
        asset_config = TAVConfig.get_config(asset_type)
        adjusted_weights = asset_config['weights']
        
        # 计算各维度评分
        trend_score = self._calculate_trend_score(df, asset_config)
        momentum_score = self._calculate_momentum_score(df, asset_config)
        volume_score = self._calculate_volume_score(df, asset_config)
        
        # 综合评分
        tav_score = (
            trend_score * adjusted_weights['trend'] +
            momentum_score * adjusted_weights['momentum'] +
            volume_score * adjusted_weights['volume']
        )
        
        # 限制在0-100范围内
        tav_score = min(100, max(0, tav_score))
        
        # 生成状态描述
        status = self._get_tav_status(tav_score)
        
        # 详细评分
        detailed_scores = {
            'trend': trend_score,
            'momentum': momentum_score,
            'volume': volume_score
        }
        
        return tav_score, detailed_scores, status
    
    def _calculate_trend_score(self, df, config):
        """计算趋势评分"""
        if df.empty or len(df) < 20:
            return 0
        
        current_price = df['Close'].iloc[-1]
        trend_config = config['trend']
        ma_periods = trend_config['ma_periods']
        
        # 检查是否有足够的均线数据
        available_mas = []
        for period in ma_periods:
            ma_col = f'MA{period}'
            if ma_col in df.columns and not pd.isna(df[ma_col].iloc[-1]):
                available_mas.append((period, df[ma_col].iloc[-1]))
        
        if len(available_mas) < 2:
            return 30  # 默认中性评分
        
        # 趋势评分逻辑
        if len(available_mas) >= 3:
            # 完整的三均线分析
            ma20 = next((val for period, val in available_mas if period == 20), None)
            ma50 = next((val for period, val in available_mas if period == 50), None)
            ma200 = next((val for period, val in available_mas if period == 200), None)
            
            if all([ma20, ma50, ma200]):
                # 多头排列：价格 > MA20 > MA50 > MA200
                if current_price > ma20 > ma50 > ma200:
                    return 95
                # 强势多头：价格 > MA20 > MA50，MA50 < MA200
                elif current_price > ma20 > ma50:
                    return 80
                # 震荡整理：价格在均线之间
                elif ma20 < current_price < ma50 or ma50 < current_price < ma20:
                    return 50
                # 弱势空头：价格 < MA20 < MA50，MA50 > MA200
                elif current_price < ma20 < ma50:
                    return 30
                # 空头排列：价格 < MA20 < MA50 < MA200
                elif current_price < ma20 < ma50 < ma200:
                    return 15
        
        # 双均线分析
        if len(available_mas) >= 2:
            available_mas.sort(key=lambda x: x[0])  # 按周期排序
            short_ma = available_mas[0][1]
            long_ma = available_mas[1][1]
            
            if current_price > short_ma > long_ma:
                return 75
            elif current_price < short_ma < long_ma:
                return 25
            else:
                return 50
        
        return 40  # 默认中性评分
    
    def _calculate_momentum_score(self, df, config):
        """计算动量评分"""
        if df.empty or len(df) < 14:
            return 0
        
        momentum_config = config['momentum']
        score = 50  # 基础分
        
        # RSI 评分
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi > momentum_config['rsi_overbought']:
                score += 15  # 超买，动量强劲
            elif rsi < momentum_config['rsi_oversold']:
                score -= 15  # 超卖，动量疲软
            elif rsi > 50:
                score += 10  # RSI强势
            else:
                score -= 10  # RSI弱势
        
        # MACD 评分
        if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_hist = df['MACD_histogram'].iloc[-1]
            
            if not any(pd.isna([macd, macd_signal, macd_hist])):
                # MACD 金叉
                if macd > macd_signal and macd_hist > 0:
                    score += 20
                # MACD 死叉
                elif macd < macd_signal and macd_hist < 0:
                    score -= 20
                # MACD 柱状体增强
                elif macd_hist > 0:
                    score += 10
                elif macd_hist < 0:
                    score -= 10
        
        # 随机振荡器评分
        if all(col in df.columns for col in ['K', 'D']):
            k = df['K'].iloc[-1]
            d = df['D'].iloc[-1]
            
            if not any(pd.isna([k, d])):
                if k > d and k > 80:
                    score += 15  # 超买且K>D
                elif k < d and k < 20:
                    score -= 15  # 超卖且K<D
                elif k > d:
                    score += 5   # K>D
                else:
                    score -= 5   # K<D
        
        return min(100, max(0, score))
    
    def _calculate_volume_score(self, df, config):
        """计算成交量评分"""
        if df.empty or 'Volume' not in df.columns:
            return 0
        
        volume_config = config['volume']
        score = 40  # 基础分
        
        # 成交量比率评分
        if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]):
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            if volume_ratio > volume_config['surge_threshold_strong']:
                score += 40  # 强突增
            elif volume_ratio > volume_config['surge_threshold_medium']:
                score += 25  # 中等突增
            elif volume_ratio > volume_config['surge_threshold_weak']:
                score += 15  # 弱突增
            elif volume_ratio < volume_config['shrink_threshold']:
                score -= 20  # 萎缩
        
        # 价量配合评分
        price_volume_bullish = 0
        price_volume_bearish = 0
        
        # 检查各种价量配合信号
        bullish_signals = [
            'Price_Volume_Bullish_Strong',
            'Price_Volume_Bullish_Medium', 
            'Price_Volume_Bullish_Weak',
            'Price_Volume_Reversal_Bullish',
            'Price_Volume_Continuation_Bullish'
        ]
        
        bearish_signals = [
            'Price_Volume_Bearish_Strong',
            'Price_Volume_Bearish_Medium',
            'Price_Volume_Bearish_Weak', 
            'Price_Volume_Reversal_Bearish',
            'Price_Volume_Continuation_Bearish'
        ]
        
        for signal in bullish_signals:
            if signal in df.columns and df.iloc[-1][signal]:
                price_volume_bullish += 1
        
        for signal in bearish_signals:
            if signal in df.columns and df.iloc[-1][signal]:
                price_volume_bearish += 1
        
        # 价量配合评分调整
        if price_volume_bullish > price_volume_bearish:
            score += min(20, price_volume_bullish * 5)
        elif price_volume_bearish > price_volume_bullish:
            score -= min(20, price_volume_bearish * 5)
        
        return min(100, max(0, score))
    
    def _get_tav_status(self, score):
        """根据评分获取TAV状态"""
        if score >= self.thresholds['strong_signal']:
            return "强共振"
        elif score >= self.thresholds['medium_signal']:
            return "中等共振"
        elif score >= self.thresholds['weak_signal']:
            return "弱共振"
        else:
            return "无共振"
    
    def get_tav_summary(self, df, asset_type='stock'):
        """获取TAV分析摘要"""
        tav_score, detailed_scores, status = self.calculate_tav_score(df, asset_type)
        
        # 趋势分析
        trend_analysis = self._analyze_trend_direction(df)
        
        # 动量分析
        momentum_analysis = self._analyze_momentum_state(df)
        
        # 成交量分析
        volume_analysis = self._analyze_volume_state(df)
        
        return {
            'tav_score': tav_score,
            'tav_status': status,
            'detailed_scores': detailed_scores,
            'trend_analysis': trend_analysis,
            'momentum_analysis': momentum_analysis,
            'volume_analysis': volume_analysis,
            'recommendation': self._get_recommendation(tav_score, trend_analysis, momentum_analysis, volume_analysis)
        }
    
    def _analyze_trend_direction(self, df):
        """分析趋势方向"""
        if df.empty or len(df) < 20:
            return "数据不足"
        
        analyzer = TechnicalAnalyzer()
        return analyzer.analyze_trend(df)
    
    def _analyze_momentum_state(self, df):
        """分析动量状态"""
        if df.empty:
            return "数据不足"
        
        momentum_states = []
        
        # RSI状态
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi > 70:
                momentum_states.append("RSI超买")
            elif rsi < 30:
                momentum_states.append("RSI超卖")
            elif rsi > 50:
                momentum_states.append("RSI强势")
            else:
                momentum_states.append("RSI弱势")
        
        # MACD状态
        if all(col in df.columns for col in ['MACD', 'MACD_signal']):
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            
            if not any(pd.isna([macd, macd_signal])):
                if macd > macd_signal:
                    momentum_states.append("MACD金叉")
                else:
                    momentum_states.append("MACD死叉")
        
        return ", ".join(momentum_states) if momentum_states else "中性"
    
    def _analyze_volume_state(self, df):
        """分析成交量状态"""
        if df.empty or 'Volume' not in df.columns:
            return "数据不足"
        
        volume_states = []
        
        # 成交量比率状态
        if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]):
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            if volume_ratio > 2.0:
                volume_states.append("成交量暴增")
            elif volume_ratio > 1.5:
                volume_states.append("成交量放大")
            elif volume_ratio < 0.8:
                volume_states.append("成交量萎缩")
            else:
                volume_states.append("成交量正常")
        
        # 价量配合状态
        bullish_signals = ['Price_Volume_Bullish_Strong', 'Price_Volume_Bullish_Medium', 'Price_Volume_Bullish_Weak']
        bearish_signals = ['Price_Volume_Bearish_Strong', 'Price_Volume_Bearish_Medium', 'Price_Volume_Bearish_Weak']
        
        has_bullish = any(df.iloc[-1].get(signal, False) for signal in bullish_signals)
        has_bearish = any(df.iloc[-1].get(signal, False) for signal in bearish_signals)
        
        if has_bullish:
            volume_states.append("价量配合上涨")
        elif has_bearish:
            volume_states.append("价量配合下跌")
        
        return ", ".join(volume_states) if volume_states else "中性"
    
    def _get_recommendation(self, tav_score, trend_analysis, momentum_analysis, volume_analysis):
        """获取投资建议"""
        if tav_score >= 75:
            return "强烈建议关注，TAV三要素共振强烈"
        elif tav_score >= 50:
            return "建议关注，TAV中等共振，需结合其他分析"
        elif tav_score >= 25:
            return "谨慎观察，TAV弱共振，信号质量一般"
        else:
            return "不建议操作，TAV无共振，缺乏明确方向"


class TechnicalAnalyzerV2(TechnicalAnalyzer):
    """扩展版技术分析器，集成TAV方法论"""
    
    def __init__(self, enable_tav=False, tav_config=None):
        super().__init__()
        self.enable_tav = enable_tav
        self.tav_config = tav_config
        self.tav_scorer = TAVScorer(tav_config) if enable_tav else None
    
    def calculate_all_indicators(self, df, asset_type='stock'):
        """计算所有指标，保持原有接口，可选添加TAV指标"""
        # 调用原有方法
        df = super().calculate_all_indicators(df)
        
        # 如果启用TAV，添加TAV相关指标
        if self.enable_tav and self.tav_scorer:
            df = self._add_tav_indicators(df, asset_type)
        
        return df
    
    def _add_tav_indicators(self, df, asset_type='stock'):
        """添加TAV相关指标到数据框"""
        if df.empty:
            return df
        
        # 计算TAV评分
        tav_score, detailed_scores, status = self.tav_scorer.calculate_tav_score(df, asset_type)
        
        # 添加TAV指标列
        df['TAV_Score'] = tav_score
        df['TAV_Status'] = status
        df['TAV_Trend_Score'] = detailed_scores['trend']
        df['TAV_Momentum_Score'] = detailed_scores['momentum']
        df['TAV_Volume_Score'] = detailed_scores['volume']
        
        # 添加TAV信号列
        df['TAV_Strong_Signal'] = tav_score >= 75
        df['TAV_Medium_Signal'] = (tav_score >= 50) & (tav_score < 75)
        df['TAV_Weak_Signal'] = (tav_score >= 25) & (tav_score < 50)
        df['TAV_No_Signal'] = tav_score < 25
        
        return df
    
    def generate_buy_sell_signals(self, df, use_tav=None, asset_type='stock'):
        """生成信号，支持TAV和传统模式"""
        # 决定是否使用TAV
        use_tav = use_tav if use_tav is not None else self.enable_tav
        
        if use_tav and self.tav_scorer:
            return self._generate_tav_enhanced_signals(df, asset_type)
        else:
            # 调用原有方法，保持完全兼容
            return super().generate_buy_sell_signals(df)
    
    def _generate_tav_enhanced_signals(self, df, asset_type='stock'):
        """生成TAV增强的交易信号"""
        if df.empty:
            return df
        
        # 首先生成传统信号
        df = super().generate_buy_sell_signals(df)
        
        # 添加TAV增强逻辑
        tav_config = TAVConfig.get_config(asset_type)
        tav_scorer = TAVScorer(tav_config)
        
        # 为每个数据点计算TAV评分
        for i in range(len(df)):
            if i < 50:  # 需要足够的历史数据
                continue
            
            # 获取当前时间窗口的数据
            window_df = df.iloc[max(0, i-200):i+1].copy()
            
            # 计算TAV评分
            tav_score, detailed_scores, status = tav_scorer.calculate_tav_score(window_df, asset_type)
            
            # TAV增强信号逻辑
            tav_strong = tav_score >= tav_config['thresholds']['strong_signal']
            tav_medium = tav_score >= tav_config['thresholds']['medium_signal']
            
            # 增强买入信号：传统信号 + TAV确认
            if df.iloc[i].get('Buy_Signal', False):
                if tav_strong:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV强共振确认]"
                elif tav_medium:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV中等共振]"
                else:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV弱共振]"
            
            # 增强卖出信号：传统信号 + TAV确认
            if df.iloc[i].get('Sell_Signal', False):
                if tav_strong:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV强共振确认]"
                elif tav_medium:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV中等共振]"
                else:
                    df.at[df.index[i], 'Signal_Description'] = f"{df.iloc[i].get('Signal_Description', '')} [TAV弱共振]"
            
            # 如果同时有买入和卖出信号，确保描述包含两种信号
            if df.iloc[i].get('Buy_Signal', False) and df.iloc[i].get('Sell_Signal', False):
                # 检查描述是否同时包含买入和卖出信号
                desc = df.iloc[i].get('Signal_Description', '')
                if '买入信号:' not in desc or '卖出信号:' not in desc:
                    # 如果描述不完整，重新生成
                    original_desc = desc
                    if tav_strong:
                        tav_tag = " [TAV强共振确认]"
                    elif tav_medium:
                        tav_tag = " [TAV中等共振]"
                    else:
                        tav_tag = " [TAV弱共振]"
                    
                    # 创建包含两种信号的描述
                    df.at[df.index[i], 'Signal_Description'] = f"买入信号: RSI超卖反弹(成交量弱确认){tav_tag} | 卖出信号: RSI超买回落(成交量弱确认){tav_tag}"
        
        return df
    
    def get_tav_analysis_summary(self, df, asset_type='stock'):
        """获取TAV分析摘要"""
        if not self.enable_tav or not self.tav_scorer:
            return None
        
        return self.tav_scorer.get_tav_summary(df, asset_type)


def main():
    """主函数示例"""
    # 测试一些常用的金融产品
    symbols = ['GC=F', 'CL=F', 'SPY', 'QQQ']  # 黄金、原油、标普500、纳斯达克
    
    analyzer = MarketAnalyzer(symbols)
    results = analyzer.run_analysis(period="6mo")
    
    if results:
        print("\n" + "="*60)
        print("📊 分析完成！")
        print("="*60)
        
        for symbol, result in results.items():
            # 检查最近是否有交易信号
            recent_signals = result['data'].tail(5)[['Buy_Signal', 'Sell_Signal', 'Signal_Description']].dropna()
            recent_signals = recent_signals[(recent_signals['Buy_Signal']) | (recent_signals['Sell_Signal'])]
            
            if not recent_signals.empty:
                print(f"\n🚨 {result['name']} ({symbol}) 最近交易信号:")
                for idx, row in recent_signals.iterrows():
                    signal_type = "买入" if row['Buy_Signal'] else "卖出"
                    print(f"  {idx.strftime('%Y-%m-%d')}: {signal_type} - {row['Signal_Description']}")
    else:
        print("\n❌ 分析失败")

if __name__ == "__main__":
    main()
def calculate_ma_alignment(df, periods=[5, 10, 20, 50]):
    """
    计算均线排列状态
    
    参数:
    - df: 包含价格数据的DataFrame
    - periods: 均线周期列表
    
    返回:
    - dict: {
        'alignment': '多头排列'/'空头排列'/'混乱排列',
        'strength': 0-100,  # 排列强度
        'details': {  # 各均线关系
            'ma5_above_ma10': bool,
            'ma10_above_ma20': bool,
            'ma20_above_ma50': bool,
            'all_bullish': bool,
            'all_bearish': bool
        }
    }
    """
    if df.empty or len(df) < max(periods):
        return {'alignment': '数据不足', 'strength': 0, 'details': {}}
    
    details = {}
    latest = df.iloc[-1]
    
    # 检查各均线关系
    for i in range(len(periods) - 1):
        ma_short = f'MA{periods[i]}'
        ma_long = f'MA{periods[i+1]}'
        if ma_short in df.columns and ma_long in df.columns:
            key = f'{ma_short}_above_{ma_long}'
            details[key] = latest[ma_short] > latest[ma_long]
    
    # 判断排列状态
    all_bullish = all(details.values()) if details else False
    all_bearish = all(not v for v in details.values()) if details else False
    
    if all_bullish:
        alignment = '多头排列'
        strength = 90 + min(10, (latest[f'MA{periods[0]}'] / latest[f'MA{periods[-1]}'] - 1) * 100)
    elif all_bearish:
        alignment = '空头排列'
        strength = 90 + min(10, (1 - latest[f'MA{periods[0]}'] / latest[f'MA{periods[-1]}']) * 100)
    else:
        alignment = '混乱排列'
        # 计算排列一致性
        bullish_count = sum(details.values()) if details else 0
        total_count = len(details) if details else 1
        strength = (bullish_count / total_count) * 50 if bullish_count > total_count / 2 else ((total_count - bullish_count) / total_count) * 50
    
    details['all_bullish'] = all_bullish
    details['all_bearish'] = all_bearish
    
    return {
        'alignment': alignment,
        'strength': min(100, max(0, strength)),
        'details': details
    }


def calculate_ma_slope(df, period=20):
    """
    计算均线斜率
    
    参数:
    - df: 包含价格数据的DataFrame
    - period: 均线周期
    
    返回:
    - dict: {
        'slope': float,  # 斜率（正数表示上升，负数表示下降）
        'angle': float,  # 角度（度数）
        'trend': '强势上升'/'上升'/'平缓'/'下降'/'强势下降'
    }
    """
    if df.empty or len(df) < period + 5:
        return {'slope': 0, 'angle': 0, 'trend': '数据不足'}
    
    ma_col = f'MA{period}'
    if ma_col not in df.columns:
        return {'slope': 0, 'angle': 0, 'trend': '数据不足'}
    
    # 使用最近5个数据点计算斜率
    recent_mas = df[ma_col].iloc[-5:].values
    x = np.arange(len(recent_mas))
    
    # 线性回归计算斜率
    slope = np.polyfit(x, recent_mas, 1)[0]
    
    # 计算角度（斜率转换为角度）
    angle = np.degrees(np.arctan(slope / recent_mas.mean())) if (recent_means := recent_mas.mean()) else 0
    
    # 判断趋势强度
    if angle > 5:
        trend = '强势上升'
    elif angle > 2:
        trend = '上升'
    elif angle > -2:
        trend = '平缓'
    elif angle > -5:
        trend = '下降'
    else:
        trend = '强势下降'
    
    return {
        'slope': slope,
        'angle': angle,
        'trend': trend
    }


def calculate_ma_deviation(df, periods=[5, 10, 20, 50]):
    """
    计算均线乖离率
    
    参数:
    - df: 包含价格数据的DataFrame
    - periods: 均线周期列表
    
    返回:
    - dict: {
        'deviations': {
            'ma5_deviation': float,  # 百分比
            'ma10_deviation': float,
            'ma20_deviation': float,
            'ma50_deviation': float
        },
        'avg_deviation': float,  # 平均乖离率
        'extreme_deviation': str  # '严重超买'/'超买'/'正常'/'超卖'/'严重超卖'
    }
    """
    if df.empty or len(df) < max(periods):
        return {'deviations': {}, 'avg_deviation': 0, 'extreme_deviation': '数据不足'}
    
    latest = df.iloc[-1]
    current_price = latest['Close']
    deviations = {}
    
    for period in periods:
        ma_col = f'MA{period}'
        if ma_col in df.columns and latest[ma_col] > 0:
            deviation = (current_price - latest[ma_col]) / latest[ma_col] * 100
            deviations[f'ma{period}_deviation'] = deviation
    
    if not deviations:
        return {'deviations': {}, 'avg_deviation': 0, 'extreme_deviation': '数据不足'}
    
    avg_deviation = np.mean(list(deviations.values()))
    
    # 判断极端乖离
    if avg_deviation > 10:
        extreme_deviation = '严重超买'
    elif avg_deviation > 5:
        extreme_deviation = '超买'
    elif avg_deviation > -5:
        extreme_deviation = '正常'
    elif avg_deviation > -10:
        extreme_deviation = '超卖'
    else:
        extreme_deviation = '严重超卖'
    
    return {
        'deviations': deviations,
        'avg_deviation': avg_deviation,
        'extreme_deviation': extreme_deviation
    }


def calculate_support_resistance(df, lookback=20, min_touches=2):
    """
    计算支撑阻力位
    
    参数:
    - df: 包含OHLC数据的DataFrame
    - lookback: 回看天数
    - min_touches: 最少触及次数
    
    返回:
    - dict: {
        'support_levels': [
            {'price': float, 'strength': 0-100, 'touches': int, 'type': 'strong'/'medium'/'weak'}
        ],
        'resistance_levels': [
            {'price': float, 'strength': 0-100, 'touches': int, 'type': 'strong'/'medium'/'weak'}
        ],
        'nearest_support': float,
        'nearest_resistance': float
    }
    """
    if df.empty or len(df) < lookback:
        return {
            'support_levels': [],
            'resistance_levels': [],
            'nearest_support': None,
            'nearest_resistance': None
        }
    
    recent_df = df.iloc[-lookback:]
    current_price = df.iloc[-1]['Close']
    
    # 识别局部低点（支撑）
    support_candidates = []
    for i in range(2, len(recent_df) - 2):
        low = recent_df.iloc[i]['Low']
        if (low < recent_df.iloc[i-1]['Low'] and 
            low < recent_df.iloc[i-2]['Low'] and
            low < recent_df.iloc[i+1]['Low'] and 
            low < recent_df.iloc[i+2]['Low']):
            support_candidates.append(low)
    
    # 识别局部高点（阻力）
    resistance_candidates = []
    for i in range(2, len(recent_df) - 2):
        high = recent_df.iloc[i]['High']
        if (high > recent_df.iloc[i-1]['High'] and 
            high > recent_df.iloc[i-2]['High'] and
            high > recent_df.iloc[i+1]['High'] and 
            high > recent_df.iloc[i+2]['High']):
            resistance_candidates.append(high)
    
    # 聚类相似价格（误差1%以内）
    def cluster_prices(prices, tolerance=0.01):
        if not prices:
            return []
        clusters = []
        sorted_prices = sorted(prices)
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if abs(price - current_cluster[0]) / current_cluster[0] <= tolerance:
                current_cluster.append(price)
            else:
                clusters.append(current_cluster)
                current_cluster = [price]
        clusters.append(current_cluster)
        
        # 计算每个聚类的平均价格和触及次数
        result = []
        for cluster in clusters:
            avg_price = np.mean(cluster)
            touches = len(cluster)
            # 基于触及次数和距离当前价格的远近计算强度
            distance_factor = max(0, 1 - abs(avg_price - current_price) / current_price)
            strength = min(100, (touches / min_touches) * 50 + distance_factor * 50)
            
            level_type = 'strong' if touches >= 3 else ('medium' if touches >= 2 else 'weak')
            result.append({'price': avg_price, 'strength': strength, 'touches': touches, 'type': level_type})
        
        return sorted(result, key=lambda x: x['strength'], reverse=True)
    
    support_levels = cluster_prices(support_candidates)
    resistance_levels = cluster_prices(resistance_candidates)
    
    # 找到最近的支撑和阻力
    nearest_support = None
    nearest_resistance = None
    
    if support_levels:
        below_supports = [s for s in support_levels if s['price'] < current_price]
        if below_supports:
            nearest_support = max(below_supports, key=lambda x: x['price'])['price']
    
    if resistance_levels:
        above_resistances = [r for r in resistance_levels if r['price'] > current_price]
        if above_resistances:
            nearest_resistance = min(above_resistances, key=lambda x: x['price'])['price']
    
    return {
        'support_levels': support_levels[:3],  # 只返回前3个
        'resistance_levels': resistance_levels[:3],
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance
    }


def calculate_relative_strength(stock_df, index_df, period=20):
    """
    计算相对强弱指标（相对于指数）
    
    参数:
    - stock_df: 股票价格DataFrame
    - index_df: 指数价格DataFrame
    - period: 计算周期
    
    返回:
    - dict: {
        'relative_return': float,  # 相对收益率（股票收益率 - 指数收益率）
        'beta': float,  # Beta系数
        'alpha': float,  # Alpha系数
        'correlation': float,  # 相关系数
        'performance': '跑赢'/'跑输'/'持平'
    }
    """
    if stock_df.empty or index_df.empty or len(stock_df) < period or len(index_df) < period:
        return {
            'relative_return': 0,
            'beta': 0,
            'alpha': 0,
            'correlation': 0,
            'performance': '数据不足'
        }
    
    # 对齐日期
    stock_returns = stock_df['Close'].pct_change().iloc[-period:].dropna()
    index_returns = index_df['Close'].pct_change().iloc[-period:].dropna()
    
    if len(stock_returns) < 10 or len(index_returns) < 10:
        return {
            'relative_return': 0,
            'beta': 0,
            'alpha': 0,
            'correlation': 0,
            'performance': '数据不足'
        }
    
    # 计算相对收益率
    stock_total_return = (1 + stock_returns).prod() - 1
    index_total_return = (1 + index_returns).prod() - 1
    relative_return = stock_total_return - index_total_return
    
    # 计算Beta和Alpha
    if len(stock_returns) == len(index_returns):
        covariance = np.cov(stock_returns, index_returns)[0, 1]
        index_variance = np.var(index_returns)
        beta = covariance / index_variance if index_variance > 0 else 0
        alpha = stock_returns.mean() - beta * index_returns.mean()
        correlation = np.corrcoef(stock_returns, index_returns)[0, 1]
    else:
        beta = 0
        alpha = 0
        correlation = 0
    
    # 判断相对表现
    if relative_return > 2:
        performance = '显著跑赢'
    elif relative_return > 0:
        performance = '跑赢'
    elif relative_return > -2:
        performance = '跑输'
    else:
        performance = '显著跑输'
    
    return {
        'relative_return': relative_return * 100,  # 转换为百分比
        'beta': beta,
        'alpha': alpha * 100,  # 转换为百分比
        'correlation': correlation,
        'performance': performance
    }


def calculate_medium_term_score(df, index_df=None):
    """
    计算中期趋势评分系统
    
    参数:
    - df: 股票价格DataFrame
    - index_df: 指数价格DataFrame（可选，用于相对强弱）
    
    返回:
    - dict: {
        'total_score': 0-100,  # 总分
        'components': {
            'trend_score': 0-100,  # 趋势评分（均线排列+斜率）
            'momentum_score': 0-100,  # 动量评分（乖离率+RSI）
            'support_resistance_score': 0-100,  # 支撑阻力评分
            'relative_strength_score': 0-100  # 相对强弱评分
        },
        'trend_health': '健康'/'一般'/'疲弱',
        'sustainability': '高'/'中'/'低',
        'recommendation': '强烈买入'/'买入'/'持有'/'卖出'/'强烈卖出'
    }
    """
    if df.empty or len(df) < 50:
        return {
            'total_score': 0,
            'components': {
                'trend_score': 0,
                'momentum_score': 0,
                'support_resistance_score': 0,
                'relative_strength_score': 0
            },
            'trend_health': '数据不足',
            'sustainability': '低',
            'recommendation': '观望'
        }
    
    # 计算各指标
    ma_alignment = calculate_ma_alignment(df)
    ma_slope_20 = calculate_ma_slope(df, 20)
    ma_slope_50 = calculate_ma_slope(df, 50)
    ma_deviation = calculate_ma_deviation(df)
    support_resistance = calculate_support_resistance(df)
    
    # 1. 趋势评分（40%权重）
    trend_score = 0
    if ma_alignment['alignment'] == '多头排列':
        trend_score += 40
    elif ma_alignment['alignment'] == '空头排列':
        trend_score += 10
    else:
        trend_score += 25
    
    # 均线斜率评分
    if ma_slope_20['trend'] in ['强势上升', '上升']:
        trend_score += 30
    elif ma_slope_20['trend'] == '平缓':
        trend_score += 15
    else:
        trend_score += 5
    
    # MA50斜率额外加分
    if ma_slope_50['trend'] in ['强势上升', '上升']:
        trend_score += 20
    elif ma_slope_50['trend'] == '平缓':
        trend_score += 10
    
    trend_score = min(100, trend_score)
    
    # 2. 动量评分（30%权重）
    momentum_score = 0
    
    # 乖离率评分
    if ma_deviation['extreme_deviation'] == '严重超买':
        momentum_score -= 20
    elif ma_deviation['extreme_deviation'] == '超买':
        momentum_score -= 10
    elif ma_deviation['extreme_deviation'] == '超卖':
        momentum_score += 10
    elif ma_deviation['extreme_deviation'] == '严重超卖':
        momentum_score += 20
    
    # RSI评分
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if 40 <= rsi <= 60:
            momentum_score += 30  # 健康区域
        elif 30 <= rsi < 40:
            momentum_score += 20  # 偏弱但可接受
        elif 60 < rsi <= 70:
            momentum_score += 20  # 偏强但可接受
        elif rsi < 30:
            momentum_score += 10  # 超卖，可能反弹
        elif rsi > 70:
            momentum_score += 10  # 超买，可能回调
    
    momentum_score = max(0, min(100, momentum_score + 30))  # 基础分30
    
    # 3. 支撑阻力评分（20%权重）
    support_resistance_score = 0
    current_price = df.iloc[-1]['Close']
    
    # 距离支撑位的距离
    if support_resistance['nearest_support']:
        distance_to_support = (current_price - support_resistance['nearest_support']) / current_price
        if distance_to_support < 0.02:  # 接近支撑位
            support_resistance_score += 30
        elif distance_to_support < 0.05:
            support_resistance_score += 20
        elif distance_to_support < 0.10:
            support_resistance_score += 10
    
    # 距离阻力位的距离
    if support_resistance['nearest_resistance']:
        distance_to_resistance = (support_resistance['nearest_resistance'] - current_price) / current_price
        if distance_to_resistance < 0.02:  # 接近阻力位
            support_resistance_score -= 10
        elif distance_to_resistance < 0.05:
            support_resistance_score -= 5
    
    # 支撑阻力强度
    if support_resistance['support_levels']:
        support_resistance_score += support_resistance['support_levels'][0]['strength'] * 0.2
    
    support_resistance_score = max(0, min(100, support_resistance_score + 50))  # 基础分50
    
    # 4. 相对强弱评分（10%权重）
    relative_strength_score = 50  # 基础分50
    
    if index_df is not None:
        relative_strength = calculate_relative_strength(df, index_df)
        if relative_strength['performance'] == '显著跑赢':
            relative_strength_score = 90
        elif relative_strength['performance'] == '跑赢':
            relative_strength_score = 70
        elif relative_strength['performance'] == '跑输':
            relative_strength_score = 30
        elif relative_strength['performance'] == '显著跑输':
            relative_strength_score = 10
    
    # 计算总分
    total_score = (
        trend_score * 0.4 +
        momentum_score * 0.3 +
        support_resistance_score * 0.2 +
        relative_strength_score * 0.1
    )
    
    # 判断趋势健康度
    if total_score >= 70:
        trend_health = '健康'
    elif total_score >= 50:
        trend_health = '一般'
    else:
        trend_health = '疲弱'
    
    # 判断可持续性
    sustainability = '高' if (ma_alignment['alignment'] in ['多头排列', '空头排列'] and 
                           ma_slope_20['trend'] in ['强势上升', '上升', '强势下降', '下降']) else '中'
    if ma_alignment['alignment'] == '混乱排列':
        sustainability = '低'
    
    # 生成建议
    if total_score >= 80:
        recommendation = '强烈买入'
    elif total_score >= 65:
        recommendation = '买入'
    elif total_score >= 45:
        recommendation = '持有'
    elif total_score >= 30:
        recommendation = '卖出'
    else:
        recommendation = '强烈卖出'
    
    return {
        'total_score': round(total_score, 1),
        'components': {
            'trend_score': round(trend_score, 1),
            'momentum_score': round(momentum_score, 1),
            'support_resistance_score': round(support_resistance_score, 1),
            'relative_strength_score': round(relative_strength_score, 1)
        },
        'trend_health': trend_health,
        'sustainability': sustainability,
        'recommendation': recommendation
    }


