#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用技术分析工具
实现多种常用技术指标的计算，包括移动平均线、RSI、MACD等
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
        """计算布林带"""
        if df.empty:
            return df
        
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
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
        
        # 计算成交量指标
        df = self.calculate_volume_indicators(df)
        
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
                df.at[df.index[i], 'Signal_Description'] = "卖出信号: " + ", ".join(sell_conditions)
        
        return df
    
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