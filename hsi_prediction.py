#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数涨跌预测脚本

基于模型特征重要性，使用加权评分模型预测恒生指数短期走势
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'output')

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


class HSI_Predictor:
    """恒生指数预测器"""

    # 特征重要性配置（权重、影响方向）
    # 基于2026-03-02 statistical特征选择结果，使用top 20特征
    # 2026-04-16: 新增宏观因子（美债、VIX）和港股通资金流向
    # 2026-04-16: 新增RSI/MACD/布林带/动量特征
    # 2026-04-16: 新增多周期均线组合、均线斜率分析、价格距离分析
    FEATURE_IMPORTANCE = {
        # ========== 宏观因子（新增，业界最重要的预测因子）==========
        'US_10Y_Yield': {'weight': 0.07, 'direction': -1},  # 美债10年期收益率，利率上升不利股市
        'US_10Y_Yield_Change_5d': {'weight': 0.05, 'direction': -1},  # 美债5日变化，快速上升不利
        'VIX': {'weight': 0.06, 'direction': -1},  # VIX恐慌指数，恐慌上升不利
        'VIX_Change_5d': {'weight': 0.04, 'direction': -1},  # VIX 5日变化

        # ========== 港股通资金流向（新增，恒指最重要预测因子）==========
        'Southbound_Net_Inflow': {'weight': 0.08, 'direction': 1},  # 南向资金净流入（亿），资金流入利好
        'Southbound_Net_Buy': {'weight': 0.06, 'direction': 1},  # 南向资金净买入（亿），净买入利好

        # ========== 长期移动平均线相关（权重调整）==========
        'MA250': {'weight': 0.08, 'direction': 1},  # 250日均线，长期趋势支撑
        'Volume_MA250': {'weight': 0.06, 'direction': 1},  # 250日成交量均线，长期流动性
        'MA120': {'weight': 0.05, 'direction': 1},  # 120日均线，中期趋势
        'MA60': {'weight': 0.04, 'direction': 1},  # 60日均线，季线
        'MA20': {'weight': 0.03, 'direction': 1},  # 20日均线，月线

        # ========== 均线斜率分析（新增）==========
        'MA250_Slope': {'weight': 0.03, 'direction': 1},  # MA250每日斜率
        'MA250_Slope_5d': {'weight': 0.025, 'direction': 1},  # MA250的5日斜率变化
        'MA250_Slope_20d': {'weight': 0.02, 'direction': 1},  # MA250的20日斜率变化
        'MA250_Slope_Direction': {'weight': 0.02, 'direction': 1},  # MA250斜率方向

        # ========== 价格与均线距离（新增，超买超卖指标）==========
        'Price_Distance_MA250': {'weight': 0.03, 'direction': 1},  # 价格与MA250距离%
        'Price_Distance_MA60': {'weight': 0.02, 'direction': 1},  # 价格与MA60距离%
        'Price_Distance_MA20': {'weight': 0.015, 'direction': 1},  # 价格与MA20距离%

        # ========== 均线排列信号（新增）==========
        'MA_Bullish_Alignment': {'weight': 0.04, 'direction': 1},  # 多头排列
        'MA_Bearish_Alignment': {'weight': 0.04, 'direction': -1},  # 空头排列

        # ========== 交叉信号（新增）==========
        'MA20_Golden_Cross_MA60': {'weight': 0.03, 'direction': 1},  # MA20金叉MA60
        'MA20_Death_Cross_MA60': {'weight': 0.03, 'direction': -1},  # MA20死叉MA60
        'MA60_Golden_Cross_MA250': {'weight': 0.04, 'direction': 1},  # MA60金叉MA250
        'MA60_Death_Cross_MA250': {'weight': 0.04, 'direction': -1},  # MA60死叉MA250

        # ========== RSI 系列（新增）==========
        'RSI': {'weight': 0.03, 'direction': -1},  # RSI > 70 超买，不利
        'RSI_ROC': {'weight': 0.015, 'direction': 1},  # RSI 变化率
        'RSI_Deviation': {'weight': 0.02, 'direction': -1},  # RSI 偏离度大，不利

        # ========== MACD 系列（新增）==========
        'MACD': {'weight': 0.015, 'direction': 1},  # MACD
        'MACD_Signal': {'weight': 0.01, 'direction': 1},  # MACD 信号线
        'MACD_Hist': {'weight': 0.03, 'direction': 1},  # MACD 柱状图正值利好
        'MACD_Hist_ROC': {'weight': 0.015, 'direction': 1},  # MACD 柱状图变化率

        # ========== 布林带（新增）==========
        'BB_Width': {'weight': 0.01, 'direction': 1},  # 布林带宽度
        'BB_Position': {'weight': 0.02, 'direction': 1},  # 布林带位置

        # ========== ATR（真实波幅，新增）==========
        'ATR': {'weight': 0.01, 'direction': -1},  # ATR，高波动不利
        'ATR_Ratio': {'weight': 0.015, 'direction': -1},  # ATR比率，高波动不利

        # ========== ADX（趋势强度，新增）==========
        'ADX': {'weight': 0.02, 'direction': 1},  # ADX>25表示趋势市场
        'Plus_DI': {'weight': 0.01, 'direction': 1},  # +DI，多头力量
        'Minus_DI': {'weight': 0.01, 'direction': -1},  # -DI，空头力量

        # ========== KDJ随机振荡器（新增）==========
        'Stoch_K': {'weight': 0.015, 'direction': -1},  # K值，超买区不利
        'Stoch_D': {'weight': 0.01, 'direction': -1},  # D值，超买区不利

        # ========== Williams %R（新增）==========
        'Williams_R': {'weight': 0.01, 'direction': 1},  # Williams %R，超卖区利好

        # ========== CMF资金流（新增）==========
        'CMF': {'weight': 0.02, 'direction': 1},  # CMF正值表示资金流入
        'CMF_Signal': {'weight': 0.01, 'direction': 1},  # CMF信号线

        # ========== RSI背离（新增）==========
        'RSI_Bullish_Divergence': {'weight': 0.03, 'direction': 1},  # RSI看涨背离
        'RSI_Bearish_Divergence': {'weight': 0.03, 'direction': -1},  # RSI看跌背离

        # ========== MACD背离（新增）==========
        'MACD_Bullish_Divergence': {'weight': 0.03, 'direction': 1},  # MACD看涨背离
        'MACD_Bearish_Divergence': {'weight': 0.03, 'direction': -1},  # MACD看跌背离

        # ========== 动量加速度（新增）==========
        'Momentum_Accel_5d': {'weight': 0.02, 'direction': 1},  # 5日动量加速度
        'Momentum_Accel_10d': {'weight': 0.015, 'direction': 1},  # 10日动量加速度

        # ========== 多周期收益率（扩展）==========
        'Return_1d': {'weight': 0.005, 'direction': 1},  # 1日收益率
        'Return_3d': {'weight': 0.01, 'direction': 1},  # 3日收益率
        'Return_5d': {'weight': 0.01, 'direction': 1},  # 5日收益率
        'Return_10d': {'weight': 0.01, 'direction': 1},  # 10日收益率
        'Return_20d': {'weight': 0.01, 'direction': 1},  # 20日收益率
        'Return_60d': {'weight': 0.01, 'direction': 1},  # 60日收益率

        # ========== 多周期相对强度信号（RS_Signal）==========
        '60d_RS_Signal_MA250': {'weight': 0.04, 'direction': 1},  # 60日相对强度信号
        '60d_RS_Signal_Volume_MA250': {'weight': 0.03, 'direction': 1},  # 成交量相对强度
        '20d_RS_Signal_MA250': {'weight': 0.02, 'direction': 1},  # 20日相对强度
        '10d_RS_Signal_MA250': {'weight': 0.015, 'direction': 1},  # 10日相对强度
        '5d_RS_Signal_MA250': {'weight': 0.01, 'direction': 1},  # 5日相对强度
        '3d_RS_Signal_MA250': {'weight': 0.01, 'direction': 1},  # 3日相对强度

        # ========== 多周期趋势（Trend）==========
        '60d_Trend_MA250': {'weight': 0.03, 'direction': 1},  # 60日趋势
        '20d_Trend_MA250': {'weight': 0.025, 'direction': 1},  # 20日趋势
        '10d_Trend_MA250': {'weight': 0.02, 'direction': 1},  # 10日趋势
        '5d_Trend_MA250': {'weight': 0.02, 'direction': 1},  # 5日趋势
        '3d_Trend_MA250': {'weight': 0.015, 'direction': 1},  # 3日趋势

        # ========== 成交量趋势 ==========
        '60d_Trend_Volume_MA250': {'weight': 0.03, 'direction': 1},  # 60日成交量趋势
        '20d_Trend_Volume_MA250': {'weight': 0.02, 'direction': 1},  # 20日成交量趋势

        # ========== 波动率 ==========
        'Volatility_120d': {'weight': 0.025, 'direction': -1},  # 120日波动率，高波动率不利
        'Volatility_20d': {'weight': 0.015, 'direction': -1},  # 20日波动率
        'Volatility_60d': {'weight': 0.02, 'direction': -1},  # 60日波动率

        # ========== 中期均线趋势 ==========
        '60d_Trend_MA120': {'weight': 0.02, 'direction': 1},  # 60日MA120趋势

        # ========== 成交量相对强度 ==========
        '20d_RS_Signal_Volume_MA250': {'weight': 0.015, 'direction': 1},  # 20日成交量相对强度
    }

    def __init__(self):
        self.hsi_data = None
        self.us_data = None
        self.vix_data = None
        self.features = {}
        self.tencent_hsi_data = None  # 腾讯财经的历史数据（包含准确的成交额）

    def fetch_data(self):
        """获取所需数据"""
        print("📊 正在获取数据...")

        # 获取恒生指数数据（2年数据以确保MA250等长期指标有足够数据）
        print("  - 恒生指数数据...")
        hsi = yf.Ticker("^HSI")
        self.hsi_data = hsi.history(period="2y", interval="1d")

        # 获取腾讯财经的历史数据（包含准确的成交额）
        print("  - 恒生指数成交额数据（腾讯财经）...")
        from data_services.tencent_finance import get_hsi_data_tencent
        self.tencent_hsi_data = get_hsi_data_tencent(period_days=400)
        if self.tencent_hsi_data is not None:
            print(f"    ✅ 获取到 {len(self.tencent_hsi_data)} 天成交额数据")
        else:
            print("    ⚠️ 腾讯财经成交额数据获取失败，将使用yfinance估算")

        # 获取美国10年期国债收益率
        print("  - 美国国债收益率...")
        us_yield = yf.Ticker("^TNX")
        self.us_data = us_yield.history(period="2y", interval="1d")

        # 获取VIX指数
        print("  - VIX恐慌指数...")
        vix = yf.Ticker("^VIX")
        self.vix_data = vix.history(period="2y", interval="1d")

        # 获取港股通南向资金数据
        print("  - 港股通资金流向...")
        self.southbound_data = self._fetch_southbound_data()

        # 获取腾讯财经实时成交额数据（更准确的成交额）
        print("  - 腾讯财经成交额...")
        self.tencent_amount = self._fetch_tencent_amount()

        if self.hsi_data.empty or self.us_data.empty or self.vix_data.empty:
            raise ValueError("数据获取失败")

        print(f"  ✅ 数据获取完成（恒指：{len(self.hsi_data)} 条，美债：{len(self.us_data)} 条，VIX：{len(self.vix_data)} 条）")

    def _fetch_tencent_amount(self):
        """
        从腾讯财经获取恒生指数实时成交额数据

        返回:
        - dict: 包含成交额数据，或 None（如果获取失败）
          {
            'amount': float,      # 成交额（亿港元）
            'amount_raw': float,  # 成交额原始值（万元）
            'price': float,       # 最新价
            'prev_close': float,  # 昨收
            'volume_ratio': float # 成交量比率
          }
        """
        try:
            import requests
            url = 'https://web.sqt.gtimg.cn/q=r_hkHSI'
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                print("    ⚠️ 腾讯财经API请求失败")
                return None

            # 解析响应数据
            data_str = resp.text.split('"')[1]
            fields = data_str.split('~')

            if len(fields) < 24:
                print("    ⚠️ 腾讯财经API返回数据格式异常")
                return None

            # 提取数据
            # [3] 最新价, [4] 昨收, [5] 今开, [6] 成交额(万元), [21] 涨跌点数, [22] 涨跌幅, [23] 最高
            amount_raw = float(fields[6])  # 万元
            amount = amount_raw / 10000     # 转换为亿港元
            price = float(fields[3])
            prev_close = float(fields[4])

            # 计算成交量比率（与yfinance的Volume_MA20对比）
            volume_ratio = None
            if hasattr(self, 'hsi_data') and self.hsi_data is not None and len(self.hsi_data) >= 20:
                avg_volume = self.hsi_data['Volume'].tail(20).mean()
                # 估算当日成交量（用成交额/价格）
                estimated_volume = (amount_raw * 10000) / price  # 转为股数
                if avg_volume > 0:
                    volume_ratio = estimated_volume / avg_volume

            result = {
                'amount': amount,           # 亿港元
                'amount_raw': amount_raw,   # 万元
                'price': price,
                'prev_close': prev_close,
                'volume_ratio': volume_ratio
            }

            print(f"    ✅ 腾讯财经：成交额 {amount:.1f} 亿港元")
            return result

        except Exception as e:
            print(f"    ⚠️ 腾讯财经数据获取失败: {e}")
            return None

    def _fetch_southbound_data(self):
        """
        获取港股通南向资金数据

        返回:
        - dict: 包含南向资金净流入和净买入数据
        """
        try:
            import akshare as ak

            # 获取港股通资金流向汇总
            df = ak.stock_hsgt_fund_flow_summary_em()

            # 筛选南向资金
            southbound = df[df['资金方向'] == '南向']

            if southbound.empty:
                print("    ⚠️ 未获取到南向资金数据，使用默认值")
                return {
                    'net_inflow': 0,  # 资金净流入（亿）
                    'net_buy': 0      # 成交净买额（亿）
                }

            # 计算汇总数据（港股通(沪) + 港股通(深)）
            total_net_inflow = southbound['资金净流入'].sum() if '资金净流入' in southbound.columns else 0
            total_net_buy = southbound['成交净买额'].sum() if '成交净买额' in southbound.columns else 0

            # akshare返回的单位可能是亿，需要确认
            # 根据 hk_smart_money_tracker.py 的说明，单位转换为万
            # 但 fund_flow_summary_em 返回的似乎是亿元

            result = {
                'net_inflow': float(total_net_inflow) if pd.notna(total_net_inflow) else 0,
                'net_buy': float(total_net_buy) if pd.notna(total_net_buy) else 0
            }

            print(f"    ✅ 南向资金: 净流入 {result['net_inflow']:.2f}亿, 净买入 {result['net_buy']:.2f}亿")
            return result

        except Exception as e:
            print(f"    ⚠️ 获取港股通数据失败: {e}，使用默认值")
            return {
                'net_inflow': 0,
                'net_buy': 0
            }

    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()

        # ========== 多周期移动平均线（业界标准组合）==========
        # 短期趋势：20日均线（月线）
        df['MA20'] = df['Close'].rolling(window=20).mean()
        # 中期趋势：60日均线（季线）
        df['MA60'] = df['Close'].rolling(window=60).mean()
        # 半年线：120日均线
        df['MA120'] = df['Close'].rolling(window=120).mean()
        # 长期趋势：250日均线（年线）
        df['MA250'] = df['Close'].rolling(window=250).mean()

        # ========== 均线斜率分析（趋势强度）==========
        # MA250斜率（每日变化）
        df['MA250_Slope'] = df['MA250'].diff()
        # MA250的5日斜率变化（短期趋势加速/减速）
        df['MA250_Slope_5d'] = df['MA250'].diff(5) / 5
        # MA250的20日斜率变化（中期趋势强度）
        df['MA250_Slope_20d'] = df['MA250'].diff(20) / 20
        # MA250斜率方向（1=上升，-1=下降，0=持平）
        df['MA250_Slope_Direction'] = np.sign(df['MA250_Slope'])

        # ========== 价格与均线距离分析（超买超卖）==========
        # 价格与MA250的距离百分比（正值=高于年线，负值=低于年线）
        df['Price_Distance_MA250'] = (df['Close'] - df['MA250']) / df['MA250'] * 100
        # 价格与MA60的距离百分比
        df['Price_Distance_MA60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
        # 价格与MA20的距离百分比
        df['Price_Distance_MA20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100

        # ========== 多周期均线排列信号 ==========
        # 多头排列：MA20 > MA60 > MA250（强势上涨趋势）
        df['MA_Bullish_Alignment'] = ((df['MA20'] > df['MA60']) &
                                      (df['MA60'] > df['MA250'])).astype(int)
        # 空头排列：MA20 < MA60 < MA250（强势下跌趋势）
        df['MA_Bearish_Alignment'] = ((df['MA20'] < df['MA60']) &
                                      (df['MA60'] < df['MA250'])).astype(int)

        # ========== 黄金交叉/死亡交叉信号 ==========
        # MA20上穿MA60（黄金交叉）
        df['MA20_Golden_Cross_MA60'] = ((df['MA20'] > df['MA60']) &
                                        (df['MA20'].shift(1) <= df['MA60'].shift(1))).astype(int)
        # MA20下穿MA60（死亡交叉）
        df['MA20_Death_Cross_MA60'] = ((df['MA20'] < df['MA60']) &
                                       (df['MA20'].shift(1) >= df['MA60'].shift(1))).astype(int)
        # MA60上穿MA250（黄金交叉）
        df['MA60_Golden_Cross_MA250'] = ((df['MA60'] > df['MA250']) &
                                         (df['MA60'].shift(1) <= df['MA250'].shift(1))).astype(int)
        # MA60下穿MA250（死亡交叉）
        df['MA60_Death_Cross_MA250'] = ((df['MA60'] < df['MA250']) &
                                        (df['MA60'].shift(1) >= df['MA250'].shift(1))).astype(int)

        # 收益率（使用昨日值，实盘中预测时只能用昨天收盘价计算）
        df['Return_1d'] = df['Close'].pct_change().shift(1)
        df['Return_5d'] = df['Close'].pct_change(5).shift(1)
        df['Return_20d'] = df['Close'].pct_change(20).shift(1)
        df['Return_60d'] = df['Close'].pct_change(60).shift(1)

        # 成交量相关
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA250'] = df['Volume'].rolling(window=250).mean()
        df['Turnover_Std_20'] = df['Volume'].rolling(window=20).std()

        # OBV（能量潮指标）
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # ATR（平均真实波幅）
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_MA'] = df['ATR'].rolling(window=20).mean()
        df['ATR_MA120'] = df['ATR'].rolling(window=120).mean()

        # 波动率
        df['Volatility'] = df['Return_1d'].rolling(window=20).std()
        df['Vol_Std_20'] = df['Volatility'].rolling(window=20).std()

        # VWAP（成交量加权平均价）
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

        # 支撑阻力位
        df['Resistance_120d'] = df['High'].rolling(window=120).max()
        df['Support_120d'] = df['Low'].rolling(window=120).min()
        df['Distance_Support_120d'] = (df['Close'] - df['Support_120d']) / df['Support_120d']

        # 相对强弱信号
        df['RS_Signal_MA250_Slope'] = df['Close'] / df['MA250'] - 1

        # ========== 新增技术指标（2026-04-16 优化）==========
        # ⚠️ 特征时滞处理：所有使用当日 Close 的特征需添加 .shift(1)
        # 实盘中预测时只能使用前一天的数据

        # RSI（使用昨日值）
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1)  # 使用昨日 RSI
        df['RSI_ROC'] = df['RSI'].pct_change()
        df['RSI_Deviation'] = abs(df['RSI'] - 50)

        # MACD（使用昨日值）
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26).shift(1)  # 使用昨日 MACD
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change()

        # 布林带（使用昨日值）
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        # BB_Position 使用昨日 Close 计算
        df['BB_Position'] = ((df['Close'].shift(1) - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10))

        # 动量加速度（使用昨日值）
        df['Momentum_Accel_5d'] = df['Return_5d'] - df['Return_5d'].shift(5)
        df['Momentum_Accel_10d'] = df['Return_20d'] - df['Return_20d'].shift(5)

        # 波动率扩展（基于昨日收益率）
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        df['Volatility_60d'] = df['Return_1d'].rolling(window=60).std()

        # ========== 新增技术指标（从 ml_trading_model.py 借鉴）==========

        # ATR比率（相对于10日均线）
        df['ATR_Ratio'] = df['ATR'] / (df['ATR_MA'] + 1e-10)

        # ADX（平均趋向指数，趋势强度识别）
        up_move = df['High'].diff().shift(1)
        down_move = -df['Low'].diff().shift(1)
        df['Plus_DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['Minus_DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['Plus_DI'] = 100 * (df['Plus_DM'].ewm(alpha=1/14, adjust=False).mean() / (df['ATR'] + 1e-10))
        df['Minus_DI'] = 100 * (df['Minus_DM'].ewm(alpha=1/14, adjust=False).mean() / (df['ATR'] + 1e-10))
        dx = 100 * (np.abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'] + 1e-10))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # KDJ随机振荡器（使用昨日高低价避免数据泄漏）
        k_period = 14
        d_period = 3
        df['Low_Min_14'] = df['Low'].rolling(window=k_period, min_periods=1).min().shift(1)
        df['High_Max_14'] = df['High'].rolling(window=k_period, min_periods=1).max().shift(1)
        df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min_14']) / (df['High_Max_14'] - df['Low_Min_14'] + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period, min_periods=1).mean().shift(1)

        # Williams %R（使用昨日高低价避免数据泄漏）
        df['Williams_R'] = (df['High_Max_14'] - df['Close']) / (df['High_Max_14'] - df['Low_Min_14'] + 1e-10) * -100

        # CMF（Chaikin资金流，使用昨日高低价避免数据泄漏）
        df['MF_Multiplier'] = ((df['Close'] - df['Low'].shift(1)) - (df['High'].shift(1) - df['Close'])) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)
        df['MF_Volume'] = df['MF_Multiplier'] * df['Volume']
        df['CMF'] = df['MF_Volume'].rolling(20, min_periods=1).sum() / (df['Volume'].rolling(20, min_periods=1).sum() + 1e-10)
        df['CMF_Signal'] = df['CMF'].rolling(5, min_periods=1).mean().shift(1)

        # RSI背离检测（价格创新低但RSI未创新低 = 看涨背离）
        lookback = 5
        df['Price_Low_5d'] = df['Close'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['Price_High_5d'] = df['Close'].rolling(window=lookback, min_periods=1).max().shift(1)
        df['RSI_Low_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['RSI_High_5d_History'] = df['RSI'].rolling(window=lookback, min_periods=1).max().shift(1)
        # 看涨背离：价格创新低，但RSI未创新低
        df['RSI_Bullish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_Low_5d']) &
            (df['RSI'] > df['RSI_Low_5d_History'])
        ).astype(int)
        # 看跌背离：价格创新高，但RSI未创新高
        df['RSI_Bearish_Divergence'] = (
            (df['Close'].shift(1) == df['Price_High_5d']) &
            (df['RSI'] < df['RSI_High_5d_History'])
        ).astype(int)

        # MACD背离检测（价格创新低但MACD未创新低 = 看涨背离）
        df['MACD_Low_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).min().shift(1)
        df['MACD_High_5d_History'] = df['MACD'].rolling(window=lookback, min_periods=1).max().shift(1)
        # 看涨背离：价格创新低，但MACD未创新低
        df['MACD_Bullish_Divergence'] = (
            (df['Close'] == df['Price_Low_5d']) &
            (df['MACD'] > df['MACD_Low_5d_History'])
        ).astype(int)
        # 看跌背离：价格创新高，但MACD未创新高
        df['MACD_Bearish_Divergence'] = (
            (df['Close'] == df['Price_High_5d']) &
            (df['MACD'] < df['MACD_High_5d_History'])
        ).astype(int)

        return df

    def calculate_features(self):
        """计算所有特征"""
        print("🔧 正在计算特征...")

        hsi_df = self.calculate_technical_indicators(self.hsi_data)
        
        # 计算多周期指标
        periods = [3, 5, 10, 20, 60]
        
        # 计算多周期收益率（使用昨日值）
        for period in periods:
            if len(hsi_df) >= period:
                return_col = f'Return_{period}d'
                hsi_df[return_col] = hsi_df['Close'].pct_change(period).shift(1)  # 使用昨日值

                # 计算趋势方向（1=上涨，0=下跌）
                trend_col = f'{period}d_Trend'
                hsi_df[trend_col] = (hsi_df[return_col] > 0).astype(int)

                # 计算相对强度信号（基于收益率和MA250）
                rs_signal_col = f'{period}d_RS_Signal'
                hsi_df[rs_signal_col] = (hsi_df[return_col] > 0).astype(int)
        
        # 计算多周期MA250趋势
        for period in periods:
            if len(hsi_df) >= period:
                trend_col = f'{period}d_Trend_MA250'
                hsi_df[trend_col] = (hsi_df['MA250'].diff(period) > 0).astype(int)
                
                # 计算MA250的相对强度信号
                rs_signal_col = f'{period}d_RS_Signal_MA250'
                hsi_df[rs_signal_col] = (hsi_df['Close'] > hsi_df['MA250']).astype(int)
        
        # 计算多周期Volume_MA250趋势
        for period in periods:
            if len(hsi_df) >= period:
                trend_col = f'{period}d_Trend_Volume_MA250'
                hsi_df[trend_col] = (hsi_df['Volume_MA250'].diff(period) > 0).astype(int)
                
                # 计算Volume_MA250的相对强度信号
                rs_signal_col = f'{period}d_RS_Signal_Volume_MA250'
                hsi_df[rs_signal_col] = (hsi_df['Volume'] > hsi_df['Volume_MA250']).astype(int)
        
        # 计算多周期MA120趋势
        for period in periods:
            if len(hsi_df) >= period:
                trend_col = f'{period}d_Trend_MA120'
                hsi_df[trend_col] = (hsi_df['MA120'].diff(period) > 0).astype(int)
        
        # 计算120日波动率
        hsi_df['Volatility_120d'] = hsi_df['Return_1d'].rolling(window=120).std()
        
        # 获取最新数据（最近一天）
        latest_hsi = hsi_df.iloc[-1]
        
        # 安全获取特征值，处理NaN情况
        def safe_get(series, default=0):
            if pd.isna(series):
                return default
            return series
        
        # 计算特征值
        self.features = {
            # ========== 宏观因子（新增，业界最重要的预测因子）==========
            'US_10Y_Yield': safe_get(self.us_data['Close'].iloc[-1] / 10 if not self.us_data.empty else 0),  # 美债10年期收益率（%）
            'US_10Y_Yield_Change_5d': safe_get(self.us_data['Close'].pct_change(5).iloc[-1] if len(self.us_data) >= 5 else 0),  # 5日变化率
            'VIX': safe_get(self.vix_data['Close'].iloc[-1] if not self.vix_data.empty else 20),  # VIX恐慌指数
            'VIX_Change_5d': safe_get(self.vix_data['Close'].pct_change(5).iloc[-1] if len(self.vix_data) >= 5 else 0),  # VIX 5日变化率

            # ========== 港股通资金流向（新增，恒指最重要预测因子）==========
            'Southbound_Net_Inflow': self.southbound_data.get('net_inflow', 0),  # 南向资金净流入（亿）
            'Southbound_Net_Buy': self.southbound_data.get('net_buy', 0),  # 南向资金净买入（亿）

            # ========== 多周期移动平均线（业界标准组合）==========
            'MA20': safe_get(latest_hsi.get('MA20', latest_hsi['Close'])),
            'MA60': safe_get(latest_hsi.get('MA60', latest_hsi['Close'])),
            'MA120': safe_get(latest_hsi.get('MA120', latest_hsi['Close'])),
            'MA250': safe_get(latest_hsi.get('MA250', latest_hsi['Close'])),
            'Volume_MA250': safe_get(latest_hsi.get('Volume_MA250', latest_hsi['Volume'])),

            # ========== 均线斜率分析（新增）==========
            'MA250_Slope': safe_get(latest_hsi.get('MA250_Slope', 0), 0),
            'MA250_Slope_5d': safe_get(latest_hsi.get('MA250_Slope_5d', 0), 0),
            'MA250_Slope_20d': safe_get(latest_hsi.get('MA250_Slope_20d', 0), 0),
            'MA250_Slope_Direction': safe_get(latest_hsi.get('MA250_Slope_Direction', 0), 0),

            # ========== 价格与均线距离（新增，超买超卖指标）==========
            'Price_Distance_MA250': safe_get(latest_hsi.get('Price_Distance_MA250', 0), 0),
            'Price_Distance_MA60': safe_get(latest_hsi.get('Price_Distance_MA60', 0), 0),
            'Price_Distance_MA20': safe_get(latest_hsi.get('Price_Distance_MA20', 0), 0),

            # ========== 均线排列信号（新增）==========
            'MA_Bullish_Alignment': safe_get(latest_hsi.get('MA_Bullish_Alignment', 0), 0),
            'MA_Bearish_Alignment': safe_get(latest_hsi.get('MA_Bearish_Alignment', 0), 0),

            # ========== 交叉信号（新增）==========
            'MA20_Golden_Cross_MA60': safe_get(latest_hsi.get('MA20_Golden_Cross_MA60', 0), 0),
            'MA20_Death_Cross_MA60': safe_get(latest_hsi.get('MA20_Death_Cross_MA60', 0), 0),
            'MA60_Golden_Cross_MA250': safe_get(latest_hsi.get('MA60_Golden_Cross_MA250', 0), 0),
            'MA60_Death_Cross_MA250': safe_get(latest_hsi.get('MA60_Death_Cross_MA250', 0), 0),

            # ========== 多周期相对强度信号（RS_Signal）==========
            '60d_RS_Signal_MA250': safe_get(latest_hsi.get('60d_RS_Signal_MA250', 0), 0),
            '60d_RS_Signal_Volume_MA250': safe_get(latest_hsi.get('60d_RS_Signal_Volume_MA250', 0), 0),
            '20d_RS_Signal_MA250': safe_get(latest_hsi.get('20d_RS_Signal_MA250', 0), 0),
            '10d_RS_Signal_MA250': safe_get(latest_hsi.get('10d_RS_Signal_MA250', 0), 0),
            '5d_RS_Signal_MA250': safe_get(latest_hsi.get('5d_RS_Signal_MA250', 0), 0),
            '3d_RS_Signal_MA250': safe_get(latest_hsi.get('3d_RS_Signal_MA250', 0), 0),

            # ========== 多周期趋势（Trend）==========
            '60d_Trend_MA250': safe_get(latest_hsi.get('60d_Trend_MA250', 0), 0),
            '20d_Trend_MA250': safe_get(latest_hsi.get('20d_Trend_MA250', 0), 0),
            '10d_Trend_MA250': safe_get(latest_hsi.get('10d_Trend_MA250', 0), 0),
            '5d_Trend_MA250': safe_get(latest_hsi.get('5d_Trend_MA250', 0), 0),
            '3d_Trend_MA250': safe_get(latest_hsi.get('3d_Trend_MA250', 0), 0),

            # ========== 成交量趋势 ==========
            '60d_Trend_Volume_MA250': safe_get(latest_hsi.get('60d_Trend_Volume_MA250', 0), 0),
            '20d_Trend_Volume_MA250': safe_get(latest_hsi.get('20d_Trend_Volume_MA250', 0), 0),

            # ========== 波动率 ==========
            'Volatility_120d': safe_get(latest_hsi.get('Volatility_120d', 0), 0),
            'Volatility_20d': safe_get(latest_hsi.get('Volatility_20d', 0), 0),
            'Volatility_60d': safe_get(latest_hsi.get('Volatility_60d', 0), 0),

            # ========== 中期均线趋势 ==========
            '60d_Trend_MA120': safe_get(latest_hsi.get('60d_Trend_MA120', 0), 0),

            # ========== 成交量相对强度 ==========
            '20d_RS_Signal_Volume_MA250': safe_get(latest_hsi.get('20d_RS_Signal_Volume_MA250', 0), 0),

            # ========== RSI 系列（新增）==========
            'RSI': safe_get(latest_hsi.get('RSI', 50), 50),
            'RSI_ROC': safe_get(latest_hsi.get('RSI_ROC', 0), 0),
            'RSI_Deviation': safe_get(latest_hsi.get('RSI_Deviation', 0), 0),

            # ========== MACD 系列（新增）==========
            'MACD': safe_get(latest_hsi.get('MACD', 0), 0),
            'MACD_Signal': safe_get(latest_hsi.get('MACD_Signal', 0), 0),
            'MACD_Hist': safe_get(latest_hsi.get('MACD_Hist', 0), 0),
            'MACD_Hist_ROC': safe_get(latest_hsi.get('MACD_Hist_ROC', 0), 0),

            # ========== 布林带（新增）==========
            'BB_Width': safe_get(latest_hsi.get('BB_Width', 0), 0),
            'BB_Position': safe_get(latest_hsi.get('BB_Position', 0.5), 0.5),

            # ========== ATR（真实波幅，新增）==========
            'ATR': safe_get(latest_hsi.get('ATR', 0), 0),
            'ATR_Ratio': safe_get(latest_hsi.get('ATR_Ratio', 1), 1),

            # ========== ADX（趋势强度，新增）==========
            'ADX': safe_get(latest_hsi.get('ADX', 25), 25),
            'Plus_DI': safe_get(latest_hsi.get('Plus_DI', 20), 20),
            'Minus_DI': safe_get(latest_hsi.get('Minus_DI', 20), 20),

            # ========== KDJ随机振荡器（新增）==========
            'Stoch_K': safe_get(latest_hsi.get('Stoch_K', 50), 50),
            'Stoch_D': safe_get(latest_hsi.get('Stoch_D', 50), 50),

            # ========== Williams %R（新增）==========
            'Williams_R': safe_get(latest_hsi.get('Williams_R', -50), -50),

            # ========== CMF资金流（新增）==========
            'CMF': safe_get(latest_hsi.get('CMF', 0), 0),
            'CMF_Signal': safe_get(latest_hsi.get('CMF_Signal', 0), 0),

            # ========== RSI背离（新增）==========
            'RSI_Bullish_Divergence': safe_get(latest_hsi.get('RSI_Bullish_Divergence', 0), 0),
            'RSI_Bearish_Divergence': safe_get(latest_hsi.get('RSI_Bearish_Divergence', 0), 0),

            # ========== MACD背离（新增）==========
            'MACD_Bullish_Divergence': safe_get(latest_hsi.get('MACD_Bullish_Divergence', 0), 0),
            'MACD_Bearish_Divergence': safe_get(latest_hsi.get('MACD_Bearish_Divergence', 0), 0),

            # ========== 动量加速度（新增）==========
            'Momentum_Accel_5d': safe_get(latest_hsi.get('Momentum_Accel_5d', 0), 0),
            'Momentum_Accel_10d': safe_get(latest_hsi.get('Momentum_Accel_10d', 0), 0),

            # ========== 多周期收益率（扩展）==========
            'Return_1d': safe_get(latest_hsi.get('Return_1d', 0), 0),
            'Return_3d': safe_get(latest_hsi.get('Return_3d', 0), 0),
            'Return_5d': safe_get(latest_hsi.get('Return_5d', 0), 0),
            'Return_10d': safe_get(latest_hsi.get('Return_10d', 0), 0),
            'Return_20d': safe_get(latest_hsi.get('Return_20d', 0), 0),
            'Return_60d': safe_get(latest_hsi.get('Return_60d', 0), 0),
        }

        print(f"  ✅ 特征计算完成（{len(self.features)} 个特征）")

    def normalize_feature(self, feature_name, value):
        """特征标准化（使用z-score标准化）"""
        # RS_Signal和Trend特征通常是0-1的二元值，直接映射到[-1, 1]
        if 'RS_Signal' in feature_name or 'Trend' in feature_name:
            # 将0-1映射到[-1, 1]：0 -> -1, 1 -> 1
            return value * 2 - 1

        # ========== 宏观因子特征标准化（新增）==========
        # 美债收益率（通常在 3-6% 范围）
        elif feature_name == 'US_10Y_Yield':
            if pd.isna(value):
                return 0
            # 标准化：4%为中性，每1%变化对应0.5的标准化值
            return np.clip((value - 4.0) / 2.0, -1, 1)

        # 美债收益率变化率
        elif feature_name == 'US_10Y_Yield_Change_5d':
            if pd.isna(value):
                return 0
            # 标准化：假设5日变化在 -20% 到 +20% 范围
            return np.clip(value / 0.2, -1, 1)

        # VIX恐慌指数（通常在 10-40 范围）
        elif feature_name == 'VIX':
            if pd.isna(value):
                return 0
            # 标准化：15为低恐慌，30为高恐慌
            return np.clip((value - 15) / 15, -1, 1)

        # VIX变化率
        elif feature_name == 'VIX_Change_5d':
            if pd.isna(value):
                return 0
            # 标准化：假设5日变化在 -50% 到 +50% 范围
            return np.clip(value / 0.5, -1, 1)

        # ========== 港股通资金流向特征标准化（新增）==========
        # 南向资金净流入（通常在 -100 到 +100 亿范围）
        elif feature_name == 'Southbound_Net_Inflow':
            if pd.isna(value):
                return 0
            # 标准化：50亿为强流入，-50亿为强流出
            return np.clip(value / 50, -1, 1)

        # 南向资金净买入（通常在 -50 到 +50 亿范围）
        elif feature_name == 'Southbound_Net_Buy':
            if pd.isna(value):
                return 0
            # 标准化：30亿为强买入，-30亿为强卖出
            return np.clip(value / 30, -1, 1)

        # ========== 新增：均线斜率特征标准化 ==========
        # MA250斜率（每日变化，通常在 -100 到 +100 点范围）
        elif feature_name == 'MA250_Slope':
            if pd.isna(value):
                return 0
            return np.clip(value / 50, -1, 1)

        # MA250的5日斜率变化
        elif feature_name == 'MA250_Slope_5d':
            if pd.isna(value):
                return 0
            return np.clip(value / 20, -1, 1)

        # MA250的20日斜率变化
        elif feature_name == 'MA250_Slope_20d':
            if pd.isna(value):
                return 0
            return np.clip(value / 10, -1, 1)

        # MA250斜率方向（已经是 -1, 0, 1）
        elif feature_name == 'MA250_Slope_Direction':
            if pd.isna(value):
                return 0
            return value

        # ========== 新增：价格与均线距离标准化 ==========
        # 价格与MA250距离（通常在 -20% 到 +20% 范围）
        elif feature_name == 'Price_Distance_MA250':
            if pd.isna(value):
                return 0
            return np.clip(value / 15, -1, 1)

        # 价格与MA60距离
        elif feature_name == 'Price_Distance_MA60':
            if pd.isna(value):
                return 0
            return np.clip(value / 10, -1, 1)

        # 价格与MA20距离
        elif feature_name == 'Price_Distance_MA20':
            if pd.isna(value):
                return 0
            return np.clip(value / 8, -1, 1)

        # ========== 新增：均线排列信号标准化 ==========
        # 多头/空头排列（已经是0或1）
        elif 'Alignment' in feature_name or 'Cross' in feature_name:
            if pd.isna(value):
                return 0
            return value * 2 - 1  # 0 -> -1, 1 -> 1

        # 如果是收益率类特征，使用固定范围标准化
        elif 'Return' in feature_name or 'Yield' in feature_name:
            # 标准化到[-1, 1]区间，假设收益率在[-0.2, 0.2]范围内
            return np.clip(value / 0.2, -1, 1)

        # MA相关特征，使用相对标准化
        elif 'MA' in feature_name:
            # MA值通常很大，使用相对标准化
            if pd.isna(value):
                return 0
            return np.tanh(value / 50000)  # 假设MA值在50000左右

        # 波动率特征
        elif 'Volatility' in feature_name:
            # 波动率通常在0.01-0.05之间
            if pd.isna(value):
                return 0
            return np.clip((value - 0.02) / 0.03, -1, 1)

        # ========== 新增：ATR特征标准化 ==========
        elif feature_name == 'ATR':
            if pd.isna(value):
                return 0
            # ATR通常在200-500点范围（恒指）
            return np.clip(value / 500, -1, 1)

        elif feature_name == 'ATR_Ratio':
            if pd.isna(value):
                return 0
            # ATR比率通常在0.5-1.5范围
            return np.clip((value - 1) / 0.5, -1, 1)

        # ========== 新增：ADX特征标准化 ==========
        elif feature_name == 'ADX':
            if pd.isna(value):
                return 0
            # ADX: <20无趋势，20-25趋势形成，25-50强趋势，50+极强趋势
            return np.clip((value - 25) / 25, -1, 1)

        elif feature_name in ['Plus_DI', 'Minus_DI']:
            if pd.isna(value):
                return 0
            # DI通常在0-50范围
            return np.clip(value / 25 - 1, -1, 1)

        # ========== 新增：KDJ特征标准化 ==========
        elif feature_name in ['Stoch_K', 'Stoch_D']:
            if pd.isna(value):
                return 0
            # 随机振荡器在0-100范围，50为中性
            return np.clip((value - 50) / 50, -1, 1)

        # ========== 新增：Williams %R标准化 ==========
        elif feature_name == 'Williams_R':
            if pd.isna(value):
                return 0
            # Williams %R在-100到0范围，-50为中性
            return np.clip((value + 50) / 50, -1, 1)

        # ========== 新增：CMF特征标准化 ==========
        elif feature_name in ['CMF', 'CMF_Signal']:
            if pd.isna(value):
                return 0
            # CMF在-1到1范围
            return np.clip(value, -1, 1)

        # ========== 新增：背离信号标准化 ==========
        elif 'Divergence' in feature_name:
            if pd.isna(value):
                return 0
            return value * 2 - 1  # 0 -> -1, 1 -> 1

        # Level特征
        elif 'Level' in feature_name:
            if pd.isna(value):
                return 0
            return (value - 20) / 30  # 20为中位数

        # Slope特征
        elif 'Slope' in feature_name:
            # 斜率通常很小，放大处理
            if pd.isna(value):
                return 0
            return np.clip(value * 100, -1, 1)
        
        else:
            # 其他特征使用简单的相对标准化
            if pd.isna(value):
                return 0
            return np.tanh(value / (abs(value) + 1))  # 使用tanh函数标准化

    def calculate_prediction_score(self):
        """计算预测得分"""
        print("📈 正在计算预测得分...")

        weighted_score = 0
        feature_details = []

        for feature_name, feature_value in self.features.items():
            if pd.isna(feature_value):
                continue

            # 获取特征配置
            config = self.FEATURE_IMPORTANCE[feature_name]
            weight = config['weight']
            direction = config['direction']

            # 标准化特征值
            normalized_value = self.normalize_feature(feature_name, feature_value)

            # 计算加权贡献
            contribution = normalized_value * weight * direction
            weighted_score += contribution

            feature_details.append({
                'feature': feature_name,
                'value': feature_value,
                'normalized': normalized_value,
                'weight': weight,
                'direction': direction,
                'contribution': contribution
            })

        # 标准化得分到[0, 1]区间
        # 得分 > 0.5 表示看涨，< 0.5 表示看跌
        prediction_score = (weighted_score + 1) / 2  # 映射到[0, 1]
        prediction_score = np.clip(prediction_score, 0, 1)

        print(f"  ✅ 预测得分计算完成：{prediction_score:.4f}")

        return prediction_score, feature_details

    def interpret_score(self, score):
        """解读预测得分"""
        if score >= 0.65:
            return "强烈看涨", "🟢"
        elif score >= 0.55:
            return "看涨", "🟢"
        elif score >= 0.50:
            return "中性偏涨", "🟡"
        elif score >= 0.45:
            return "中性偏跌", "🟡"
        elif score >= 0.35:
            return "看跌", "🔴"
        else:
            return "强烈看跌", "🔴"

    def detect_volume_anomalies(self, window_size=30, threshold=2.5):
        """
        检测成交量和成交额的异常

        使用现有的Z-Score和Isolation Forest检测器进行检测
        - 成交量：使用yfinance数据
        - 成交额：使用腾讯财经API数据（更准确）

        Args:
            window_size: 滚动窗口大小（默认30天）
            threshold: Z-Score阈值（默认2.5，即约99%置信区间）

        Returns:
            dict: 包含成交量和成交额异常检测结果的字典
        """
        from anomaly_detector.zscore_detector import ZScoreDetector
        from anomaly_detector.isolation_forest_detector import IsolationForestDetector
        from anomaly_detector.feature_extractor import FeatureExtractor

        if self.hsi_data is None or len(self.hsi_data) < window_size:
            return {
                'volume_anomaly': None,
                'amount_anomaly': None,
                'if_anomaly': None,
                'data_source': '腾讯财经API + yfinance'
            }

        # 获取历史数据
        df = self.hsi_data.copy()
        current_timestamp = df.index[-1]

        anomalies = {
            'data_source': '腾讯财经API + yfinance'
        }

        # ========== 成交额异常检测（使用腾讯财经数据）==========
        if hasattr(self, 'tencent_amount') and self.tencent_amount:
            current_amount = self.tencent_amount['amount']  # 亿港元

            # 获取历史成交额数据（从yfinance估算）
            # 使用 Volume * Close / 1亿 作为历史成交额估计
            df['Amount_Estimated'] = df['Volume'] * df['Close'] / 100000000
            amount_history = df['Amount_Estimated'].iloc[:-1]

            # 使用Z-Score检测
            zscore_detector = ZScoreDetector(
                window_size=window_size,
                threshold=threshold,
                time_interval='day'
            )

            amount_zscore_result = zscore_detector.detect_anomaly(
                metric_name='amount',
                current_value=current_amount,
                history=amount_history,
                timestamp=current_timestamp
            )

            if amount_zscore_result:
                anomalies['amount_anomaly'] = {
                    'type': '成交额异常',
                    'z_score': amount_zscore_result['z_score'],
                    'current_value': current_amount,
                    'mean': amount_zscore_result['mean'],
                    'std': amount_zscore_result['std'],
                    'severity': amount_zscore_result['severity'],
                    'direction': '放大' if amount_zscore_result['z_score'] > 0 else '萎缩',
                    'detection_method': 'zscore_tencent',
                    'data_source': '腾讯财经API'
                }
            else:
                anomalies['amount_anomaly'] = None

            # 记录成交额数据用于后续显示
            anomalies['amount_data'] = {
                'current': current_amount,
                'ma250': amount_history.tail(250).mean() if len(amount_history) >= 250 else amount_history.mean()
            }
        else:
            anomalies['amount_anomaly'] = None
            anomalies['amount_data'] = None

        # ========== 成交量异常检测（使用yfinance数据）==========
        current_volume = df['Volume'].iloc[-1]
        volume_history = df['Volume'].iloc[:-1]

        zscore_detector = ZScoreDetector(
            window_size=window_size,
            threshold=threshold,
            time_interval='day'
        )

        volume_zscore_result = zscore_detector.detect_anomaly(
            metric_name='volume',
            current_value=current_volume,
            history=volume_history,
            timestamp=current_timestamp
        )

        if volume_zscore_result:
            anomalies['volume_anomaly'] = {
                'type': '成交量异常',
                'z_score': volume_zscore_result['z_score'],
                'current_value': current_volume,
                'mean': volume_zscore_result['mean'],
                'std': volume_zscore_result['std'],
                'severity': volume_zscore_result['severity'],
                'direction': '放大' if volume_zscore_result['z_score'] > 0 else '萎缩',
                'detection_method': 'zscore'
            }
        else:
            anomalies['volume_anomaly'] = None

        # ========== Isolation Forest检测（多维特征）==========
        feature_df = df[['Close', 'Volume']].copy()
        feature_df['Volume_MA20'] = feature_df['Volume'].rolling(20).mean()
        feature_df['Volume_Ratio'] = feature_df['Volume'] / feature_df['Volume_MA20']

        # 如果有腾讯财经成交额数据，添加成交额特征
        if hasattr(self, 'tencent_amount') and self.tencent_amount and anomalies['amount_data']:
            feature_df['Amount'] = anomalies['amount_data']['current']
            feature_df['Amount_MA20'] = feature_df['Amount'].rolling(20).mean()
            feature_df['Amount_Ratio'] = feature_df['Amount'] / feature_df['Amount_MA20']

        extractor = FeatureExtractor()
        features, timestamps = extractor.extract_features(feature_df)

        if len(features) >= window_size:
            if_detector = IsolationForestDetector(
                contamination=0.05,
                random_state=42,
                anomaly_type='volume_features'
            )

            train_features = features.iloc[:-1].tail(window_size * 3)
            if_detector.train(train_features)

            if_anomalies = if_detector.detect_anomalies(
                features=features,
                timestamps=timestamps,
                lookback_days=7,
                time_interval='day'
            )

            today_anomalies = [a for a in if_anomalies if a['timestamp'] == current_timestamp]
            if today_anomalies:
                anomaly = today_anomalies[0]
                anomalies['if_anomaly'] = {
                    'type': '多维特征异常',
                    'severity': anomaly['severity'],
                    'anomaly_score': anomaly['anomaly_score'],
                    'features': anomaly['features'],
                    'detection_method': 'isolation_forest'
                }
            else:
                anomalies['if_anomaly'] = None
        else:
            anomalies['if_anomaly'] = None

        return anomalies

    def generate_email_content(self, score, trend, feature_details, multi_horizon_results=None):
        """生成邮件内容（HTML格式）- 精简版，专注于投资决策参考"""
        current_price = self.hsi_data['Close'].iloc[-1]
        previous_price = self.hsi_data['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        current_date = self.hsi_data.index[-1].strftime('%Y-%m-%d')
        current_time = self.hsi_data.index[-1].strftime('%H:%M:%S')

        # 检测成交量异常
        volume_anomalies = self.detect_volume_anomalies()
        volume_anomaly = volume_anomalies.get('volume_anomaly')
        amount_anomaly = volume_anomalies.get('amount_anomaly')
        if_anomaly = volume_anomalies.get('if_anomaly')
        data_warning = volume_anomalies.get('data_warning', '')

        # 趋势颜色
        trend_colors = {
            '强烈看涨': '#16a34a',
            '看涨': '#22c55e',
            '中性偏涨': '#84cc16',
            '中性偏跌': '#f59e0b',
            '看跌': '#f97316',
            '强烈看跌': '#dc2626'
        }
        trend_color = trend_colors.get(trend, '#6b7280')

        # 预计算核心指标
        ma20 = self.features.get('MA20', 0)
        ma60 = self.features.get('MA60', 0)
        ma120 = self.features.get('MA120', 0)
        ma250 = self.features.get('MA250', 0)
        rsi_val = self.features.get('RSI', 50)
        adx_val = self.features.get('ADX', 0)

        # 成交额数据
        if self.tencent_hsi_data is not None and 'Amount' in self.tencent_hsi_data.columns:
            current_amount = self.tencent_amount['amount'] if hasattr(self, 'tencent_amount') and self.tencent_amount else self.tencent_hsi_data['Amount'].iloc[-1]
            amount_ma20 = self.tencent_hsi_data['Amount'].tail(20).mean() if len(self.tencent_hsi_data) >= 20 else 0
            amount_ma60 = self.tencent_hsi_data['Amount'].tail(60).mean() if len(self.tencent_hsi_data) >= 60 else 0
            amount_ma120 = self.tencent_hsi_data['Amount'].tail(120).mean() if len(self.tencent_hsi_data) >= 120 else 0

            # 计算成交额对比
            if amount_ma20 > 0:
                amount_vs_ma20 = (current_amount - amount_ma20) / amount_ma20 * 100
                amount_vs_ma20_str = f"+{amount_vs_ma20:.1f}%" if amount_vs_ma20 > 0 else f"{amount_vs_ma20:.1f}%"
                amount_vs_ma20_color = '#22c55e' if amount_vs_ma20 > 0 else '#dc2626'
            else:
                amount_vs_ma20_str = "N/A"
                amount_vs_ma20_color = '#6b7280'

            if amount_ma60 > 0:
                amount_vs_ma60 = (current_amount - amount_ma60) / amount_ma60 * 100
                amount_vs_ma60_str = f"+{amount_vs_ma60:.1f}%" if amount_vs_ma60 > 0 else f"{amount_vs_ma60:.1f}%"
                amount_vs_ma60_color = '#22c55e' if amount_vs_ma60 > 0 else '#dc2626'
            else:
                amount_vs_ma60_str = "N/A"
                amount_vs_ma60_color = '#6b7280'

            if amount_ma120 > 0:
                amount_vs_ma120 = (current_amount - amount_ma120) / amount_ma120 * 100
                amount_vs_ma120_str = f"+{amount_vs_ma120:.1f}%" if amount_vs_ma120 > 0 else f"{amount_vs_ma120:.1f}%"
                amount_vs_ma120_color = '#22c55e' if amount_vs_ma120 > 0 else '#dc2626'
            else:
                amount_vs_ma120_str = "N/A"
                amount_vs_ma120_color = '#6b7280'
        else:
            current_amount = None
            amount_ma20 = amount_ma60 = amount_ma120 = 0
            amount_vs_ma20_str = amount_vs_ma60_str = amount_vs_ma120_str = "N/A"
            amount_vs_ma20_color = amount_vs_ma60_color = amount_vs_ma120_color = '#6b7280'

        # 价格位置计算
        ma20_pos = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0
        ma60_pos = (current_price - ma60) / ma60 * 100 if ma60 > 0 else 0
        ma120_pos = (current_price - ma120) / ma120 * 100 if ma120 > 0 else 0
        ma250_pos = (current_price - ma250) / ma250 * 100 if ma250 > 0 else 0

        # ADX趋势判断
        adx_trend = "趋势市场" if adx_val >= 25 else "震荡市场"
        adx_color = '#22c55e' if adx_val >= 25 else '#f59e0b'

        # RSI状态
        rsi_status = "超买" if rsi_val >= 70 else ("超卖" if rsi_val <= 30 else "中性")
        rsi_color = '#dc2626' if rsi_val >= 70 else ('#22c55e' if rsi_val <= 30 else '#6b7280')

        # 异常检测汇总
        anomaly_list = []
        if volume_anomaly:
            anomaly_list.append(f"成交量异常（{volume_anomaly['direction']}）")
        if amount_anomaly:
            anomaly_list.append(f"成交额异常（{amount_anomaly['direction']}）")
        if if_anomaly:
            anomaly_list.append("多维特征异常")

        # 多周期预测一致性判断
        consistency_signal = ""
        if multi_horizon_results:
            predictions = [multi_horizon_results[h]['prediction'] for h in [1, 5, 20] if h in multi_horizon_results]
            all_up = all(p == '上涨' for p in predictions)
            all_down = all(p == '下跌' for p in predictions)
            if all_up:
                consistency_signal = "📈 三周期一致看涨"
            elif all_down:
                consistency_signal = "📉 三周期一致看跌"

        # 统计正面和负面因素数量
        positive_count = sum(1 for f in feature_details if f['contribution'] > 0)
        negative_count = sum(1 for f in feature_details if f['contribution'] < 0)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 构建HTML邮件内容 - 精简版，专注于投资决策参考
        content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>恒生指数分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #1f2937;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: 700; }}
        .header .subtitle {{ margin-top: 8px; font-size: 13px; opacity: 0.9; }}
        .section {{ padding: 20px 25px; border-bottom: 1px solid #e5e7eb; }}
        .section:last-child {{ border-bottom: none; }}
        .section-title {{
            font-size: 16px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e5e7eb;
        }}
        .signal-box {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 15px;
        }}
        .signal-box.bullish {{ background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border: 2px solid #22c55e; }}
        .signal-box.bearish {{ background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 2px solid #dc2626; }}
        .signal-box.neutral {{ background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; }}
        .signal-box .signal-text {{ font-size: 28px; font-weight: 700; }}
        .signal-box.bullish .signal-text {{ color: #166534; }}
        .signal-box.bearish .signal-text {{ color: #991b1b; }}
        .signal-box.neutral .signal-text {{ color: #92400e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
        th {{ background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 10px 8px; text-align: left; font-weight: 600; font-size: 12px; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #e5e7eb; }}
        tr:nth-child(even) {{ background-color: #f9fafb; }}
        .data-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }}
        .data-card {{ background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #6366f1; }}
        .data-card h4 {{ margin: 0 0 8px 0; font-size: 12px; color: #6b7280; font-weight: 500; }}
        .data-card .value {{ font-size: 20px; font-weight: 700; color: #1f2937; }}
        .footer {{ background-color: #1f2937; color: #9ca3af; padding: 15px 25px; text-align: center; font-size: 11px; }}
        .consistency {{ padding: 12px; border-radius: 6px; margin-bottom: 15px; font-size: 14px; font-weight: 600; text-align: center; }}
        .consistency.up {{ background: #dcfce7; color: #166534; }}
        .consistency.down {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <h1>📊 恒生指数分析报告</h1>
            <div class="subtitle">{current_date} {current_time}</div>
        </div>

        <!-- 第一部分：预测信号 -->
        <div class="section">
            <div class="section-title">🎯 预测信号</div>

            <div class="signal-box {'bullish' if score >= 0.55 else 'bearish' if score < 0.45 else 'neutral'}">
                <div class="signal-text">{trend}</div>
                <div style="font-size: 14px; margin-top: 8px;">
                    得分 {score:.2%} | {'正面因素 ' + str(positive_count) + ' 个' if positive_count > negative_count else '负面因素 ' + str(negative_count) + ' 个'}
                </div>
            </div>

            {'<div class="consistency up">' + consistency_signal + ' - 强信号</div>' if consistency_signal and '看涨' in consistency_signal else '<div class="consistency down">' + consistency_signal + ' - 强信号</div>' if consistency_signal else ''}
"""

        # 多周期预测表格
        if multi_horizon_results:
            content += """
            <table>
                <thead>
                    <tr>
                        <th>预测周期</th>
                        <th>方向</th>
                        <th>上涨概率</th>
                        <th>置信度</th>
                        <th>历史准确率</th>
                    </tr>
                </thead>
                <tbody>
"""
            for horizon in [1, 5, 20]:
                if horizon in multi_horizon_results:
                    r = multi_horizon_results[horizon]
                    pred_color = '#22c55e' if r['prediction'] == '上涨' else '#dc2626'
                    content += f"""
                    <tr>
                        <td style="font-weight: 600;">{horizon}天</td>
                        <td style="color: {pred_color}; font-weight: 600;">{r['prediction']}</td>
                        <td>{r['probability']:.1%}</td>
                        <td>{r['confidence']}</td>
                        <td>{r['historical_accuracy']:.1%}</td>
                    </tr>
"""
            content += """
                </tbody>
            </table>
"""
        else:
            content += """<div style="padding: 15px; background: #f8fafc; border-radius: 8px; text-align: center; color: #6b7280;">多周期预测数据暂未获取</div>"""

        # 价格信息
        price_change_color = '#dc2626' if price_change < 0 else '#22c55e'
        content += f"""
        </div>

        <!-- 第二部分：市场数据 -->
        <div class="section">
            <div class="section-title">📊 市场数据</div>

            <div class="data-grid">
                <div class="data-card">
                    <h4>恒指收盘</h4>
                    <div class="value">{current_price:.2f}</div>
                    <div style="color: {price_change_color}; font-size: 14px; margin-top: 4px;">{price_change:+.2f}%</div>
                </div>
                <div class="data-card">
                    <h4>RSI</h4>
                    <div class="value">{rsi_val:.1f}</div>
                    <div style="color: {rsi_color}; font-size: 14px; margin-top: 4px;">{rsi_status}</div>
                </div>
            </div>

            <table style="margin-top: 15px;">
                <thead>
                    <tr>
                        <th>均线</th>
                        <th>价格</th>
                        <th>位置</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>MA20</strong> 月线</td>
                        <td>{ma20:.2f}</td>
                        <td style="color: {'#22c55e' if ma20_pos >= 0 else '#dc2626'}; font-weight: 600;">{'+' if ma20_pos >= 0 else ''}{ma20_pos:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>MA60</strong> 季线</td>
                        <td>{ma60:.2f}</td>
                        <td style="color: {'#22c55e' if ma60_pos >= 0 else '#dc2626'}; font-weight: 600;">{'+' if ma60_pos >= 0 else ''}{ma60_pos:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>MA120</strong> 半年线</td>
                        <td>{ma120:.2f}</td>
                        <td style="color: {'#22c55e' if ma120_pos >= 0 else '#dc2626'}; font-weight: 600;">{'+' if ma120_pos >= 0 else ''}{ma120_pos:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>MA250</strong> 年线</td>
                        <td>{ma250:.2f}</td>
                        <td style="color: {'#22c55e' if ma250_pos >= 0 else '#dc2626'}; font-weight: 600;">{'+' if ma250_pos >= 0 else ''}{ma250_pos:.1f}%</td>
                    </tr>
                </tbody>
            </table>
"""

        # 成交额对比
        if current_amount and amount_ma20 > 0:
            content += f"""
            <div style="margin-top: 20px;">
                <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #374151;">💰 成交额对比</h4>
                <table>
                    <thead>
                        <tr>
                            <th>周期</th>
                            <th>金额</th>
                            <th>对比</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>今日成交额</td>
                            <td style="font-weight: 600;">{current_amount:.0f} 亿港元</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>20日均值</td>
                            <td>{amount_ma20:.0f} 亿</td>
                            <td style="color: {amount_vs_ma20_color}; font-weight: 600;">{amount_vs_ma20_str}</td>
                        </tr>
                        <tr>
                            <td>60日均值</td>
                            <td>{amount_ma60:.0f} 亿</td>
                            <td style="color: {amount_vs_ma60_color}; font-weight: 600;">{amount_vs_ma60_str}</td>
                        </tr>
                        <tr>
                            <td>120日均值</td>
                            <td>{amount_ma120:.0f} 亿</td>
                            <td style="color: {amount_vs_ma120_color}; font-weight: 600;">{amount_vs_ma120_str}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
"""

        # 异常检测
        if anomaly_list:
            content += f"""
            <div style="margin-top: 20px; padding: 12px; background: #fee2e2; border-radius: 8px; border-left: 4px solid #dc2626;">
                <strong style="color: #991b1b;">⚠️ 检测到异常：</strong>
                <span style="color: #991b1b;">{', '.join(anomaly_list)}</span>
            </div>
"""

        content += f"""
        </div>

        <!-- 第三部分：风险提示 -->
        <div class="section">
            <div class="section-title">⚠️ 风险提示</div>
            <div style="font-size: 13px; color: #6b7280; line-height: 1.8;">
                本报告基于历史数据和统计模型，仅供参考，不构成投资建议。股市有风险，投资需谨慎。模型预测存在不确定性，请结合基本面分析、市场情绪、政策面等多方面因素综合判断。
            </div>
        </div>

        <!-- 页脚 -->
        <div class="footer">
            <p style="margin: 5px 0;">⏰ 报告生成时间：{timestamp}</p>
            <p style="margin: 5px 0; color: #6b7280;">本报告由 AI 智能分析系统自动生成 | 仅供参考</p>
        </div>
    </div>
</body>
</html>
"""

        return content

    def _get_feature_explanation(self, feature_name):
        """获取特征说明"""
        explanations = {
            # ========== 宏观因子（新增）==========
            'US_10Y_Yield': '美国10年期国债收益率，全球资产定价锚。利率上升不利港股估值，下降利好。',
            'US_10Y_Yield_Change_5d': '美债收益率5日变化率，快速上升反映紧缩预期，不利股市。',
            'VIX': 'VIX恐慌指数，反映市场风险偏好。VIX上升表示恐慌情绪蔓延，不利股市。',
            'VIX_Change_5d': 'VIX 5日变化率，快速上升表示风险情绪急剧恶化。',

            # ========== 港股通资金流向（新增）==========
            'Southbound_Net_Inflow': '港股通南向资金净流入，内地资金流入港股的最重要指标。净流入利好恒指。',
            'Southbound_Net_Buy': '港股通南向资金净买入，反映内地投资者的实际买入力度。净买入利好。',

            # ========== 多周期移动平均线（业界标准组合）==========
            'MA20': '20日移动平均线（月线），反映短期趋势。价格在MA20上方通常表示短期上涨趋势。',
            'MA60': '60日移动平均线（季线），反映中期趋势。价格在MA60上方通常表示中期上涨趋势。',
            'MA120': '120日移动平均线（半年线），反映中期趋势支撑。是重要的技术分析指标。',
            'MA250': '250日移动平均线（年线），反映长期趋势支撑。价格在MA250上方通常表示长期上涨趋势。',
            'Volume_MA250': '250日平均成交量，反映长期流动性水平。上升表示资金活跃度提高。',

            # ========== 均线斜率分析（新增）==========
            'MA250_Slope': 'MA250每日斜率，反映长期趋势变化速度。正值表示年线上升，负值表示下降。',
            'MA250_Slope_5d': 'MA250的5日斜率变化，反映短期趋势加速或减速。',
            'MA250_Slope_20d': 'MA250的20日斜率变化，反映中期趋势强度。',
            'MA250_Slope_Direction': 'MA250斜率方向，1=上升，-1=下降，0=持平。',

            # ========== 价格与均线距离（新增）==========
            'Price_Distance_MA250': '价格与年线的距离百分比。正值表示高于年线，负值表示低于年线。偏离过大可能回调。',
            'Price_Distance_MA60': '价格与季线的距离百分比。正值表示高于季线，负值表示低于季线。',
            'Price_Distance_MA20': '价格与月线的距离百分比。正值表示高于月线，负值表示低于月线。',

            # ========== 均线排列信号（新增）==========
            'MA_Bullish_Alignment': '多头排列信号。MA20>MA60>MA250，表示强势上涨趋势。',
            'MA_Bearish_Alignment': '空头排列信号。MA20<MA60<MA250，表示强势下跌趋势。',

            # ========== 交叉信号（新增）==========
            'MA20_Golden_Cross_MA60': 'MA20上穿MA60黄金交叉，短期趋势转强信号。',
            'MA20_Death_Cross_MA60': 'MA20下穿MA60死亡交叉，短期趋势转弱信号。',
            'MA60_Golden_Cross_MA250': 'MA60上穿MA250黄金交叉，中期趋势转强信号。',
            'MA60_Death_Cross_MA250': 'MA60下穿MA250死亡交叉，中期趋势转弱信号。',

            # ========== 多周期相对强度信号（RS_Signal）==========
            '60d_RS_Signal_MA250': '60日相对强度信号，价格相对MA250的强度。值为1表示强于长期趋势。',
            '60d_RS_Signal_Volume_MA250': '60日成交量相对强度，成交量相对长期均值的强度。活跃度高通常利好。',
            '20d_RS_Signal_MA250': '20日相对强度信号，反映中期相对强度。正值表示强势。',
            '10d_RS_Signal_MA250': '10日相对强度信号，反映短期相对强度。正值表示短期强势。',
            '5d_RS_Signal_MA250': '5日相对强度信号，反映超短期相对强度。正值表示超短期强势。',
            '3d_RS_Signal_MA250': '3日相对强度信号，反映日内相对强度。正值表示日内强势。',

            # ========== 多周期趋势（Trend）==========
            '60d_Trend_MA250': 'MA250的60日趋势，反映长期趋势变化。上升表示长期趋势转强。',
            '20d_Trend_MA250': 'MA250的20日趋势，反映中期趋势变化。上升表示中期趋势转强。',
            '10d_Trend_MA250': 'MA250的10日趋势，反映短期趋势变化。上升表示短期趋势转强。',
            '5d_Trend_MA250': 'MA250的5日趋势，反映超短期趋势变化。上升表示超短期趋势转强。',
            '3d_Trend_MA250': 'MA250的3日趋势，反映日内趋势变化。上升表示日内趋势转强。',

            # ========== 成交量趋势 ==========
            '60d_Trend_Volume_MA250': 'Volume_MA250的60日趋势，反映长期流动性变化。上升表示资金活跃度提高。',
            '20d_Trend_Volume_MA250': 'Volume_MA250的20日趋势，反映中期流动性变化。上升表示资金活跃度提高。',

            # ========== 波动率 ==========
            'Volatility_120d': '120日波动率，反映中长期市场稳定性。低波动率通常利于上涨。',
            'Volatility_20d': '20日波动率，反映短期市场稳定性。低波动率通常利于上涨。',
            'Volatility_60d': '60日波动率，反映中期市场稳定性。低波动率通常利于上涨。',

            # ========== 中期均线趋势 ==========
            '60d_Trend_MA120': 'MA120的60日趋势，反映中期趋势强度。上升表示中期趋势强化。',

            # ========== 成交量相对强度 ==========
            '20d_RS_Signal_Volume_MA250': '20日成交量相对强度，反映中期资金活跃度。活跃度高通常利好。',

            # ========== RSI 系列 ==========
            'RSI': 'RSI相对强弱指数，>70超买风险，<30超卖机会。中性区间50左右。',
            'RSI_ROC': 'RSI变化率，反映动能变化方向。',
            'RSI_Deviation': 'RSI偏离度，偏离50越大表示超买超卖程度越深。',

            # ========== MACD 系列 ==========
            'MACD': 'MACD指标，反映趋势动能。正值表示上涨动能，负值表示下跌动能。',
            'MACD_Signal': 'MACD信号线，用于判断买卖时机。',
            'MACD_Hist': 'MACD柱状图，正值表示动能增强，负值表示动能减弱。',
            'MACD_Hist_ROC': 'MACD柱状图变化率，反映动能变化速度。',

            # ========== 布林带 ==========
            'BB_Width': '布林带宽度，反映波动率。宽度收窄可能预示突破。',
            'BB_Position': '布林带位置，>0.8超买，<0.2超卖。',

            # ========== 动量加速度 ==========
            'Momentum_Accel_5d': '5日动量加速度，正值表示上涨加速，负值表示上涨减速。',
            'Momentum_Accel_10d': '10日动量加速度，反映中期动能变化。',

            # ========== 多周期收益率 ==========
            'Return_1d': '1日收益率，反映短期表现。',
            'Return_3d': '3日收益率，反映超短期表现。',
            'Return_5d': '5日收益率，反映短期表现。',
            'Return_10d': '10日收益率，反映短期表现。',
            'Return_20d': '20日收益率，反映中期表现。',
            'Return_60d': '60日收益率，反映中期表现。',
        }
        return explanations.get(feature_name, '暂无详细说明')

    def send_email_notification(self, score, trend, feature_details, multi_horizon_results=None):
        """发送邮件通知"""
        try:
            # 生成邮件内容
            content = self.generate_email_content(score, trend, feature_details, multi_horizon_results)

            # 邮件配置
            sender_email = os.environ.get("EMAIL_SENDER")
            email_password = os.environ.get("EMAIL_PASSWORD")
            smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
            recipient_email = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")

            if ',' in recipient_email:
                recipients = [r.strip() for r in recipient_email.split(',')]
            else:
                recipients = [recipient_email]

            if not sender_email or not email_password:
                print("❌ 邮件配置不完整，跳过邮件发送")
                print("   请设置环境变量：EMAIL_SENDER, EMAIL_PASSWORD, RECIPIENT_EMAIL")
                return False

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # 根据SMTP服务器类型选择端口和SSL
            if "163.com" in smtp_server:
                smtp_port = 465
                use_ssl = True
            elif "gmail.com" in smtp_server:
                smtp_port = 587
                use_ssl = False
            else:
                smtp_port = 587
                use_ssl = False

            # 创建邮件对象
            current_date = datetime.now().strftime('%Y-%m-%d')
            subject = f"恒生指数涨跌预测 {current_date} - {trend}（得分{score:.4f}）"

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)

            # 添加HTML版本
            html_part = MIMEText(content, 'html', 'utf-8')
            msg.attach(html_part)

            # 重试机制（3次）
            for attempt in range(3):
                try:
                    if use_ssl:
                        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                        server.login(sender_email, email_password)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()
                    else:
                        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                        server.starttls()
                        server.login(sender_email, email_password)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()

                    print(f"✅ 预测邮件已发送到: {', '.join(recipients)}")
                    return True
                except Exception as e:
                    print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:
                        import time
                        time.sleep(5)

            print("❌ 3次尝试后仍无法发送邮件")
            return False

        except Exception as e:
            print(f"❌ 发送邮件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_report(self, score, feature_details):
        """保存预测报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存为JSON格式（详细数据）
        report_data = {
            'timestamp': timestamp,
            'prediction_date': self.hsi_data.index[-1].strftime('%Y-%m-%d'),
            'current_price': float(self.hsi_data['Close'].iloc[-1]),
            'prediction_score': float(score),
            'features': {k: (float(v) if not pd.isna(v) else None) for k, v in self.features.items()},
            'feature_details': feature_details,
            'prediction_trend': self.interpret_score(score)[0]
        }

        json_file = os.path.join(output_dir, f'hsi_prediction_report_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 保存为CSV格式（特征值）
        features_df = pd.DataFrame([self.features])
        features_file = os.path.join(data_dir, f'hsi_prediction_features_{timestamp}.csv')
        features_df.to_csv(features_file, index=False)

        print(f"💾 报告已保存到：")
        print(f"   - {json_file}")
        print(f"   - {features_file}")

    def save_prediction_to_history(self, score, trend, feature_details, catboost_result=None):
        """
        保存预测到历史记录

        参数:
        - score: 预测得分
        - trend: 预测趋势
        - feature_details: 特征详情列表
        - catboost_result: CatBoost 预测结果（可选）
        """
        history_file = os.path.join(data_dir, 'hsi_prediction_history.json')

        # 计算目标日期（horizon 天后）
        prediction_date = self.hsi_data.index[-1]
        from datetime import timedelta
        target_date = prediction_date + timedelta(days=self.horizon if hasattr(self, 'horizon') else 20)

        # 创建预测记录
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'target_date': target_date.strftime('%Y-%m-%d'),
            'current_price': float(self.hsi_data['Close'].iloc[-1]),
            'horizon': self.horizon if hasattr(self, 'horizon') else 20,
            'score_model': {
                'score': float(score),
                'trend': trend
            },
            'catboost_model': catboost_result,
            'features': {k: (float(v) if not pd.isna(v) else None) for k, v in self.features.items()},
            'top_features': [
                {
                    'feature': f['feature'],
                    'value': f['value'],
                    'contribution': f['contribution']
                }
                for f in sorted(feature_details, key=lambda x: abs(x['contribution']), reverse=True)[:10]
            ],
            'verified': False,
            'actual_return': None,
            'actual_direction': None
        }

        # 加载现有历史记录
        history = {'predictions': [], 'metadata': {}}
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception as e:
                print(f"⚠️ 加载历史记录失败，创建新文件: {e}")

        # 添加新记录
        history['predictions'].append(record)

        # 更新元数据
        history['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        history['metadata']['total_predictions'] = len(history['predictions'])

        # 保存历史记录
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"💾 预测历史已保存到: {history_file}")
            print(f"   - 历史记录总数: {history['metadata']['total_predictions']}")
        except Exception as e:
            print(f"❌ 保存历史记录失败: {e}")

    def _run_catboost_comparison(self, score_model_score, score_model_trend):
        """
        运行 CatBoost 模型与评分模型对比

        参数:
        - score_model_score: 评分模型得分
        - score_model_trend: 评分模型趋势

        返回:
        - dict: CatBoost 预测结果
        """
        print("\n" + "=" * 80)
        print("🤖 CatBoost 模型对比".center(80))
        print("=" * 80)

        try:
            # 尝试加载已有的 CatBoost 模型
            import glob
            model_files = glob.glob(os.path.join(data_dir, 'hsi_models', 'hsi_catboost_*.cbm'))

            if not model_files:
                print("⚠️ 未找到 CatBoost 模型，跳过对比")
                print("   提示: 运行 python3 ml_services/hsi_ml_model.py --mode train 训练模型")
                return None

            # 加载最新模型
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"📂 加载模型: {latest_model}")

            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(latest_model)

            # 加载特征名称
            feature_file = latest_model.replace('.cbm', '.json').replace('hsi_catboost', 'hsi_features')
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    feature_names = json.load(f)
            else:
                print("⚠️ 未找到特征文件，跳过对比")
                return None

            # 准备特征数据
            # 从 self.features 构建特征向量
            feature_values = []
            missing_features = []
            for fname in feature_names:
                if fname in self.features:
                    feature_values.append(self.features[fname])
                else:
                    missing_features.append(fname)
                    feature_values.append(0)  # 缺失特征用0填充

            if missing_features:
                print(f"⚠️ 缺失特征: {missing_features[:5]}... (共{len(missing_features)}个)")

            # 预测
            X = np.array([feature_values])
            prob = model.predict_proba(X)[0, 1]
            pred = model.predict(X)[0]

            catboost_trend = "上涨" if pred == 1 else "下跌"

            # 置信度分级（更细粒度）
            if prob > 0.7 or prob < 0.3:
                catboost_confidence = "高"
                confidence_action = "可作为参考"
            elif prob > 0.6 or prob < 0.4:
                catboost_confidence = "中高"
                confidence_action = "有一定参考价值"
            elif prob > 0.55 or prob < 0.45:
                catboost_confidence = "中"
                confidence_action = "需谨慎参考"
            else:
                catboost_confidence = "低"
                confidence_action = "建议观望"

            # 动态阈值（基于概率偏离 0.5 的程度）
            dynamic_threshold = 0.5  # 可以根据训练数据调整
            prob_deviation = abs(prob - 0.5)

            # 输出对比结果
            print(f"\n{'='*50}")
            print("📊 双模型预测对比")
            print(f"{'='*50}")
            print(f"\n{'指标':<20} {'评分模型':<20} {'CatBoost模型':<20}")
            print("-" * 60)
            print(f"{'预测趋势':<20} {score_model_trend:<20} {catboost_trend:<20}")
            print(f"{'预测概率':<20} {score_model_score:.4f}              {prob:.4f}")
            print(f"{'置信度':<20} {'-':<20} {catboost_confidence:<20}")
            print(f"{'置信建议':<20} {'-':<20} {confidence_action:<20}")

            # 判断一致性
            score_direction = "上涨" if score_model_score > 0.5 else "下跌"
            consistency = "✅ 一致" if score_direction == catboost_trend else "⚠️ 不一致"
            print(f"\n{'一致性分析':<20} {consistency}")

            # 综合建议
            print(f"\n{'='*50}")
            print("📋 综合分析")
            print(f"{'='*50}")

            if consistency == "✅ 一致":
                if catboost_confidence in ["高", "中高"]:
                    print(f"  两个模型一致且置信度较高，{catboost_trend}信号较强")
                else:
                    print(f"  两个模型一致但置信度一般，建议结合其他因素判断")
            else:
                print(f"  ⚠️ 模型意见分歧，建议谨慎决策")
                print(f"     评分模型: {score_model_trend} ({score_model_score:.2%})")
                print(f"     CatBoost: {catboost_trend} ({prob:.2%}, {catboost_confidence}置信度)")

            print(f"{'='*50}\n")

            return {
                'trend': catboost_trend,
                'probability': float(prob),
                'confidence': catboost_confidence,
                'consistency': consistency,
                'model_path': latest_model
            }

        except Exception as e:
            print(f"⚠️ CatBoost 模型对比失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_multi_horizon_prediction(self):
        """
        运行多周期预测（1天、5天、20天）

        返回:
        - dict: 各周期的预测结果
        """
        print("\n" + "=" * 80)
        print("📊 多周期预测".center(80))
        print("=" * 80)

        try:
            from catboost import CatBoostClassifier

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

            results = {}

            # 获取恒指数据
            import yfinance as yf
            hsi = yf.Ticker("^HSI")
            df = hsi.history(period="5y", interval="1d")

            # 计算基础特征
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA120'] = df['Close'].rolling(120).mean()
            df['MA250'] = df['Close'].rolling(250).mean()
            df['Return_1d'] = df['Close'].pct_change().shift(1)
            df['Return_5d'] = df['Close'].pct_change(5).shift(1)
            df['Return_20d'] = df['Close'].pct_change(20).shift(1)
            df['Volatility_20d'] = df['Return_1d'].rolling(20).std()
            df['Volatility_60d'] = df['Return_1d'].rolling(60).std()
            df['Volatility_120d'] = df['Return_1d'].rolling(120).std()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = (100 - (100 / (1 + gain/(loss+1e-10)))).shift(1)

            features = ['MA20', 'MA60', 'MA120', 'MA250', 'Return_1d', 'Return_5d', 'Return_20d',
                       'Volatility_20d', 'Volatility_60d', 'Volatility_120d', 'RSI']

            for horizon in [1, 5, 20]:
                # 创建目标
                df[f'Target_{horizon}d'] = (df['Close'].pct_change(horizon).shift(-horizon) > 0).astype(float)

                # 准备数据
                df_clean = df[features + [f'Target_{horizon}d']].dropna()

                if len(df_clean) < 100:
                    continue

                X = df_clean[features]
                y = df_clean[f'Target_{horizon}d']

                # 时序分割
                split = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                # 训练模型
                model = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=3,
                    auto_class_weights='Balanced', verbose=0
                )
                model.fit(X_train, y_train)

                # 用最新数据预测
                latest = df.iloc[-1:][features].dropna()
                if latest.empty:
                    continue

                prob = model.predict_proba(latest)[0, 1]
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

            # 打印汇总表格
            print(f"\n{'周期':<8} {'方向':<6} {'概率':<8} {'置信度':<6} {'历史准确率':<10} {'历史AUC':<8}")
            print("-" * 60)
            for h in [1, 5, 20]:
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

            print(f"\n💡 {suggestion}")
            print("=" * 80)

            return results

        except Exception as e:
            print(f"⚠️ 多周期预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self, send_email_flag=True, run_catboost=True):
        """运行预测流程

        参数:
        - send_email_flag: 是否发送邮件，默认True
        - run_catboost: 是否同时运行 CatBoost 模型对比，默认True
        """
        try:
            # 1. 获取数据
            self.fetch_data()

            # 2. 计算特征
            self.calculate_features()

            # 3. 生成报告（控制台显示）
            score, feature_details = self.calculate_prediction_score()
            trend = self.interpret_score(score)[0]

            # 4. 生成控制台报告
            self._generate_console_report(score, trend, feature_details)

            # 5. CatBoost 模型对比（新增）
            catboost_result = None
            if run_catboost:
                catboost_result = self._run_catboost_comparison(score, trend)

            # 6. 多周期预测（1天、5天、20天）
            multi_horizon_results = self._run_multi_horizon_prediction()

            # 7. 保存报告
            self.save_report(score, feature_details)

            # 8. 保存预测历史记录
            self.save_prediction_to_history(score, trend, feature_details, catboost_result)

            # 8. 发送邮件
            if send_email_flag:
                print("\n" + "="*80)
                print("正在发送预测邮件...".center(80))
                print("="*80 + "\n")
                email_sent = self.send_email_notification(score, trend, feature_details, multi_horizon_results)
                if email_sent:
                    print("\n✅ 预测报告已通过邮件发送")
                else:
                    print("\n❌ 邮件发送失败，但预测报告已保存")
            else:
                print("\n⚠️ 已跳过邮件发送（--no-email 参数）")

            return score, trend

        except Exception as e:
            print(f"❌ 预测失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _generate_console_report(self, score, trend, feature_details):
        """生成控制台显示的报告"""
        print("\n" + "="*80)
        print("恒生指数涨跌预测报告".center(80))
        print("="*80)

        # 显示基本信息
        current_price = self.hsi_data['Close'].iloc[-1]
        previous_price = self.hsi_data['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        current_date = self.hsi_data.index[-1].strftime('%Y-%m-%d')

        print(f"\n📅 分析日期：{current_date}")
        print(f"📊 恒指收盘：{current_price:.2f}（{price_change:+.2f}%）")
        print(f"📈 预测得分：{score:.4f}")
        print(f"🎯 预测趋势：{trend}")

        # 分析关键因素
        print(f"\n{'='*80}")
        print("关键因素分析（按权重排序，仅显示控制台）".center(80))
        print(f"{'='*80}\n")

        # 按贡献度排序
        sorted_features = sorted(feature_details, key=lambda x: abs(x['contribution']), reverse=True)

        print(f"{'特征':<30} {'当前值':<12} {'标准化':<10} {'权重':<8} {'方向':<8} {'贡献度':<12}")
        print("-" * 100)

        for i, feature in enumerate(sorted_features[:10], 1):  # 显示前10个最重要特征
            direction_str = "正面" if feature['direction'] > 0 else "负面"
            contribution_str = f"{feature['contribution']:>+.6f}"

            print(f"{i:2}. {feature['feature']:<27} {feature['value']:>10.4f}   "
                  f"{feature['normalized']:>7.3f}   {feature['weight']:>6.2%}   "
                  f"{direction_str:<6}   {contribution_str:<12}")

        # 计算正面/负面因素
        positive_features = [f for f in feature_details if f['contribution'] > 0]
        negative_features = [f for f in feature_details if f['contribution'] < 0]

        positive_score = sum(f['contribution'] for f in positive_features)
        negative_score = sum(abs(f['contribution']) for f in negative_features)

        print(f"\n📊 因素汇总：")
        print(f"  - 正面因素贡献：{positive_score:+.6f}（{len(positive_features)} 个）")
        print(f"  - 负面因素贡献：{-negative_score:.6f}（{len(negative_features)} 个）")

        # 显示关键指标
        print(f"\n{'='*80}")
        print("关键市场指标".center(80))
        print(f"{'='*80}\n")

# 安全格式化数值
        def safe_format(value, format_str, default_str='N/A'):
            if pd.isna(value) or value == 0:
                return default_str
            return format_str.format(value)

        # ========== 多周期均线表格 ==========
        print("📊 多周期均线系统")
        print("┌───────┬────────┬──────────┬──────────────┬─────────────────────────────────┐")
        print("│ 均线  │  周期  │  价格    │  价格位置    │  用途                           │")
        print("├───────┼────────┼──────────┼──────────────┼─────────────────────────────────┤")

        ma20_val = self.features.get('MA20', 0)
        ma60_val = self.features.get('MA60', 0)
        ma120_val = self.features.get('MA120', 0)
        ma250_val = self.features.get('MA250', 0)

        # 计算价格与均线的距离
        if current_price and ma20_val > 0:
            ma20_dist = (current_price - ma20_val) / ma20_val * 100
            ma20_pos = f"+{ma20_dist:.1f}%" if ma20_dist > 0 else f"{ma20_dist:.1f}%"
        else:
            ma20_pos = "N/A"

        if current_price and ma60_val > 0:
            ma60_dist = (current_price - ma60_val) / ma60_val * 100
            ma60_pos = f"+{ma60_dist:.1f}%" if ma60_dist > 0 else f"{ma60_dist:.1f}%"
        else:
            ma60_pos = "N/A"

        if current_price and ma120_val > 0:
            ma120_dist = (current_price - ma120_val) / ma120_val * 100
            ma120_pos = f"+{ma120_dist:.1f}%" if ma120_dist > 0 else f"{ma120_dist:.1f}%"
        else:
            ma120_pos = "N/A"

        if current_price and ma250_val > 0:
            ma250_dist = (current_price - ma250_val) / ma250_val * 100
            ma250_pos = f"+{ma250_dist:.1f}%" if ma250_dist > 0 else f"{ma250_dist:.1f}%"
        else:
            ma250_pos = "N/A"

        print(f"│ MA20  │ 月线   │ {safe_format(ma20_val, '{:.2f}', 'N/A'):>8} │ {ma20_pos:>12} │ 短期趋势                       │")
        print(f"│ MA60  │ 季线   │ {safe_format(ma60_val, '{:.2f}', 'N/A'):>8} │ {ma60_pos:>12} │ 中期趋势                       │")
        print(f"│ MA120 │ 半年线 │ {safe_format(ma120_val, '{:.2f}', 'N/A'):>8} │ {ma120_pos:>12} │ 中期支撑                       │")
        print(f"│ MA250 │ 年线   │ {safe_format(ma250_val, '{:.2f}', 'N/A'):>8} │ {ma250_pos:>12} │ 长期趋势                       │")
        print("└───────┴────────┴──────────┴──────────────┴─────────────────────────────────┘")

        # 成交额数据
        volume_ma250_val = self.features.get('Volume_MA250', 0)
        # 优先使用腾讯财经成交额数据
        if self.tencent_hsi_data is not None and 'Amount' in self.tencent_hsi_data.columns:
            current_amount = self.tencent_amount['amount'] if hasattr(self, 'tencent_amount') and self.tencent_amount else self.tencent_hsi_data['Amount'].iloc[-1]
            # 计算250日成交额均值
            amount_data = self.tencent_hsi_data['Amount'].tail(250)
            amount_ma250_val = amount_data.mean() if len(amount_data) > 0 else 0
            print(f"\n💰 成交额：今日 {safe_format(current_amount, '{:.0f}', 'N/A')} 亿港元 | 250日均值 {safe_format(amount_ma250_val, '{:.0f}', 'N/A')} 亿港元")
        else:
            # fallback：使用成交量
            volume_ma250_val = self.features.get('Volume_MA250', 0)
            volume_ma250_yi_val = volume_ma250_val / 100000000
            print(f"\n💰 成交量：250日均值 {safe_format(volume_ma250_yi_val, '{:.1f}', 'N/A')} 亿股（yfinance数据）")

        # 波动率指标
        vol_20d = self.features.get('Volatility_20d', 0)
        vol_60d = self.features.get('Volatility_60d', 0)
        vol_120d = self.features.get('Volatility_120d', 0)
        print(f"\n📉 波动率：20日 {safe_format(vol_20d*100, '{:.2f}', 'N/A')}% | 60日 {safe_format(vol_60d*100, '{:.2f}', 'N/A')}% | 120日 {safe_format(vol_120d*100, '{:.2f}', 'N/A')}%")

        # 新增技术指标
        adx_val = self.features.get('ADX', 0)
        rsi_val = self.features.get('RSI', 50)
        stoch_k = self.features.get('Stoch_K', 50)
        cmf_val = self.features.get('CMF', 0)
        atr_val = self.features.get('ATR', 0)
        atr_ratio = self.features.get('ATR_Ratio', 1)

        # ADX趋势判断
        if adx_val >= 50:
            adx_trend = "极强趋势"
        elif adx_val >= 25:
            adx_trend = "趋势市场"
        else:
            adx_trend = "震荡市场"

        # RSI超买超卖判断
        if rsi_val >= 70:
            rsi_status = "超买"
        elif rsi_val <= 30:
            rsi_status = "超卖"
        else:
            rsi_status = "中性"

        print(f"\n📊 技术指标：")
        print(f"   ADX {safe_format(adx_val, '{:.1f}', 'N/A')} ({adx_trend}) | RSI {safe_format(rsi_val, '{:.1f}', 'N/A')} ({rsi_status}) | KDJ-K {safe_format(stoch_k, '{:.1f}', 'N/A')}")
        print(f"   CMF {safe_format(cmf_val, '{:.2f}', 'N/A')} | ATR {safe_format(atr_val, '{:.1f}', 'N/A')} (比率 {safe_format(atr_ratio, '{:.2f}', 'N/A')})")

        print(f"\n📍 60日相对强度信号（MA250）：{safe_format(self.features.get('60d_RS_Signal_MA250', 0), '{:.0f}', 'N/A')}")

        # 检测并显示成交异常
        volume_anomalies = self.detect_volume_anomalies()
        volume_anomaly = volume_anomalies.get('volume_anomaly')
        amount_anomaly = volume_anomalies.get('amount_anomaly')
        if_anomaly = volume_anomalies.get('if_anomaly')
        data_source = volume_anomalies.get('data_source', 'yfinance')

        has_any_anomaly = volume_anomaly or amount_anomaly or if_anomaly

        if has_any_anomaly:
            print(f"\n⚠️  成交数据异常检测（数据源：{data_source}）：")
            if amount_anomaly:
                severity_icon = "🔴" if amount_anomaly['severity'] == 'high' else ("🟠" if amount_anomaly['severity'] == 'medium' else "🟡")
                print(f"    {severity_icon} 【成交额异常】当日成交额{amount_anomaly['direction']}（Z-Score: {amount_anomaly['z_score']:.2f}，{amount_anomaly['severity']}级别）")
            if volume_anomaly:
                severity_icon = "🔴" if volume_anomaly['severity'] == 'high' else ("🟠" if volume_anomaly['severity'] == 'medium' else "🟡")
                print(f"    {severity_icon} 【成交量异常】当日成交量{volume_anomaly['direction']}（Z-Score: {volume_anomaly['z_score']:.2f}，{volume_anomaly['severity']}级别）")
            if if_anomaly:
                severity_icon = "🔴" if if_anomaly['severity'] == 'high' else ("🟠" if if_anomaly['severity'] == 'medium' else "🟡")
                print(f"    {severity_icon} 【Isolation Forest】多维特征异常（异常分数: {if_anomaly['anomaly_score']:.4f}，{if_anomaly['severity']}级别）")

            # 智能提示
            anomaly_count = sum([bool(volume_anomaly), bool(amount_anomaly), bool(if_anomaly)])
            if anomaly_count >= 2:
                print(f"    💡 多维度验证检测到异常，预示市场可能出现显著波动，建议密切关注")
            elif amount_anomaly:
                print(f"    💡 成交额异常通常伴随价格大幅波动")
            elif volume_anomaly:
                print(f"    💡 成交量异常通常预示重要变盘信号")
            else:
                print(f"    💡 多维特征异常表明成交模式发生变化，建议关注后续走势")
        else:
            print(f"\n✅ 成交数据正常（数据源：{data_source}）：当日成交额/成交量在正常范围内，无异常信号")

        print(f"\n{'='*80}\n")


def verify_predictions():
    """
    验证历史预测的准确性

    对比评分模型与CatBoost模型的预测效果
    """
    print("=" * 80)
    print("恒指预测验证系统".center(80))
    print("=" * 80)

    history_file = os.path.join(data_dir, 'hsi_prediction_history.json')

    # 加载预测历史
    if not os.path.exists(history_file):
        print("❌ 未找到预测历史记录文件")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    predictions = history.get('predictions', [])
    if not predictions:
        print("❌ 预测历史记录为空")
        return

    print(f"📊 预测历史记录: {len(predictions)} 条")

    # 获取恒指历史数据用于验证
    print("\n📊 获取恒指历史数据用于验证...")
    hsi = yf.Ticker("^HSI")
    hsi_data = hsi.history(period="3mo", interval="1d")  # 获取最近3个月数据

    if hsi_data.empty:
        print("❌ 无法获取恒指数据")
        return

    # 验证结果统计
    verified_count = 0
    score_model_correct = 0
    catboost_correct = 0
    score_model_total = 0
    catboost_total = 0

    updated_predictions = []

    for pred in predictions:
        target_date_str = pred.get('target_date')
        prediction_date_str = pred.get('prediction_date')

        # 如果已经验证过，跳过
        if pred.get('verified', False):
            updated_predictions.append(pred)
            continue

        # 兼容旧格式：如果没有 target_date，从 prediction_date + horizon 计算
        if not target_date_str and prediction_date_str:
            horizon = pred.get('horizon', 20)
            try:
                prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d')
                target_date = prediction_date + timedelta(days=horizon)
                target_date_str = target_date.strftime('%Y-%m-%d')
                pred['target_date'] = target_date_str
            except:
                pass

        # 检查目标日期是否已过
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            except:
                updated_predictions.append(pred)
                continue

            today = datetime.now()

            if target_date.date() <= today.date():
                # 目标日期已过，进行验证
                try:
                    # 查找目标日期的收盘价
                    target_date_pd = pd.to_datetime(target_date_str)
                    if target_date_pd in hsi_data.index:
                        actual_price = float(hsi_data.loc[target_date_pd, 'Close'])
                    else:
                        # 找最近的交易日
                        mask = hsi_data.index <= target_date_pd
                        if mask.any():
                            actual_price = float(hsi_data[mask].iloc[-1]['Close'])
                        else:
                            updated_predictions.append(pred)
                            continue

                    # 获取预测时的价格
                    prediction_price = pred.get('current_price', 0)

                    # 计算实际收益
                    actual_return = (actual_price - prediction_price) / prediction_price
                    actual_direction = 1 if actual_return > 0 else 0

                    # 更新预测记录
                    pred['verified'] = True
                    pred['actual_return'] = float(actual_return)
                    pred['actual_direction'] = actual_direction

                    # 验证评分模型（兼容新旧格式）
                    score_model = pred.get('score_model', {})
                    if score_model:
                        # 新格式
                        score = score_model.get('score', 0.5)
                        predicted_direction = 1 if score > 0.5 else 0
                        score_model_total += 1
                        if predicted_direction == actual_direction:
                            score_model_correct += 1
                    elif 'prediction_score' in pred:
                        # 旧格式
                        score = pred.get('prediction_score', 0.5)
                        predicted_direction = 1 if score > 0.5 else 0
                        score_model_total += 1
                        if predicted_direction == actual_direction:
                            score_model_correct += 1

                    # 验证CatBoost模型
                    catboost_model = pred.get('catboost_model', {})
                    if catboost_model and catboost_model.get('trend'):
                        catboost_pred = catboost_model.get('trend')
                        catboost_predicted_direction = 1 if catboost_pred == '上涨' else 0
                        catboost_total += 1
                        if catboost_predicted_direction == actual_direction:
                            catboost_correct += 1

                    verified_count += 1
                    print(f"✅ 验证: {prediction_date_str} → {target_date_str}")
                    print(f"   实际收益: {actual_return*100:.2f}%, 方向: {'上涨' if actual_direction == 1 else '下跌'}")

                except Exception as e:
                    print(f"⚠️ 验证失败 {prediction_date_str}: {e}")

        updated_predictions.append(pred)

    # 更新历史记录
    history['predictions'] = updated_predictions
    history['metadata']['last_verified'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 输出验证结果
    print("\n" + "=" * 80)
    print("验证结果汇总".center(80))
    print("=" * 80)

    print(f"\n📊 已验证预测: {verified_count} 条")

    if score_model_total > 0:
        score_accuracy = score_model_correct / score_model_total * 100
        print(f"\n📈 评分模型准确率: {score_accuracy:.2f}% ({score_model_correct}/{score_model_total})")

    if catboost_total > 0:
        catboost_accuracy = catboost_correct / catboost_total * 100
        print(f"🤖 CatBoost模型准确率: {catboost_accuracy:.2f}% ({catboost_correct}/{catboost_total})")

    if score_model_total > 0 and catboost_total > 0:
        print(f"\n📊 模型对比:")
        if score_accuracy > catboost_accuracy:
            print(f"   评分模型更优 (+{score_accuracy - catboost_accuracy:.2f}%)")
        elif catboost_accuracy > score_accuracy:
            print(f"   CatBoost模型更优 (+{catboost_accuracy - score_accuracy:.2f}%)")
        else:
            print(f"   两模型准确率相同")

    print(f"\n💾 验证结果已保存到: {history_file}")


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='恒生指数涨跌预测系统')
    parser.add_argument('--no-email', action='store_true', help='不发送邮件')
    parser.add_argument('--verify', action='store_true', help='验证历史预测准确率')
    args = parser.parse_args()

    # 验证模式
    if args.verify:
        verify_predictions()
        return

    print("="*80)
    print("恒生指数涨跌预测系统".center(80))
    print("基于特征重要性加权评分模型".center(80))
    print("="*80 + "\n")

    # 创建预测器
    predictor = HSI_Predictor()

    # 运行预测
    send_email_flag = not args.no_email
    score, trend = predictor.run(send_email_flag=send_email_flag)

    if score is not None:
        print(f"\n✅ 预测完成")
        print(f"   预测得分：{score:.4f}")
        print(f"   预测趋势：{trend}")
        if send_email_flag:
            print(f"   邮件状态：已发送")
        else:
            print(f"   邮件状态：已跳过（--no-email）")
    else:
        print(f"\n❌ 预测失败")


if __name__ == "__main__":
    main()
