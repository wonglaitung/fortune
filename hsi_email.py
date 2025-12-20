#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数及港股主力资金追踪器股票价格监控和交易信号邮件通知系统
基于技术分析指标生成买卖信号，只在有交易信号时发送邮件

此版本改进了止损/止盈计算：
- 使用真实历史数据计算 ATR（若可用）
- 若 ATR 无效则回退到百分比法
- 可选最大允许亏损百分比（通过环境变量 MAX_LOSS_PCT 设置，示例 0.2 表示 20%）
- 对止损/止盈按可配置或推断的最小变动单位（tick size）进行四舍五入
- 删除了重复函数定义并改进了异常处理
- 将交易记录的 CSV 解析改为 pandas.read_csv，提高健壮性并修复原先手写解析的 bug
- 修复 generate_report_content 中被截断的文本构造导致的语法错误
"""

import os
import smtplib
import json
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# 导入技术分析工具（可选）
try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用简化指标计算")

# 从港股主力资金追踪器导入股票列表（可选）
try:
    from hk_smart_money_tracker import WATCHLIST
    STOCK_LIST = WATCHLIST
except ImportError:
    print("⚠️ 无法导入 hk_smart_money_tracker.WATCHLIST，使用默认股票列表")
    STOCK_LIST = {
        "2800.HK": "盈富基金",
        "3968.HK": "招商银行",
        "0939.HK": "建设银行",
        "1398.HK": "工商银行",
        "1288.HK": "农业银行",
        "0005.HK": "汇丰银行",
        "0728.HK": "中国电信",
        "0941.HK": "中国移动",
        "6682.HK": "第四范式",
        "1347.HK": "华虹半导体",
        "1138.HK": "中远海能",
        "1088.HK": "中国神华",
        "0883.HK": "中国海洋石油",
        "0981.HK": "中芯国际",
        "0388.HK": "香港交易所",
        "0700.HK": "腾讯控股",
        "9988.HK": "阿里巴巴-SW",
        "3690.HK": "美团-W",
        "1810.HK": "小米集团-W",
        "9660.HK": "地平线机器人",
        "2533.HK": "黑芝麻智能",
        "1330.HK": "绿色动力环保",
        "1211.HK": "比亚迪股份",
        "2269.HK": "药明生物",
        "1299.HK": "友邦保险"
    }


class HSIEmailSystem:
    """恒生指数及港股主力资金追踪器邮件系统"""

    def __init__(self, stock_list=None):
        self.stock_list = stock_list or STOCK_LIST
        self.technical_analyzer = TechnicalAnalyzer() if TECHNICAL_ANALYSIS_AVAILABLE else None

        # 可通过环境变量设置默认最大亏损百分比（例如 0.2 表示 20%）
        max_loss_env = os.environ.get("MAX_LOSS_PCT", None)
        try:
            self.default_max_loss_pct = float(max_loss_env) if max_loss_env is not None else None
        except Exception:
            self.default_max_loss_pct = None

        # 可通过环境变量设置默认 tick size（例如 0.01）
        tick_env = os.environ.get("DEFAULT_TICK_SIZE", None)
        try:
            self.default_tick_size = float(tick_env) if tick_env is not None else None
        except Exception:
            self.default_tick_size = None

    def get_hsi_data(self, target_date=None):
        """获取恒生指数数据"""
        try:
            hsi_ticker = yf.Ticker("^HSI")
            hist = hsi_ticker.history(period="6mo")
            if hist.empty:
                print("❌ 无法获取恒生指数历史数据")
                return None

            # 根据target_date截断历史数据
            if target_date is not None:
                # 将target_date转换为pandas时间戳，用于与历史数据的索引比较
                target_timestamp = pd.Timestamp(target_date)
                # 确保target_timestamp是date类型
                target_date_only = target_timestamp.date()
                # 过滤出日期小于等于target_date的数据
                hist = hist[hist.index.date <= target_date_only]
                
                if hist.empty:
                    print(f"⚠️ 在 {target_date} 之前没有历史数据")
                    return None

            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest

            hsi_data = {
                'current_price': latest['Close'],
                'change_1d': (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0,
                'change_1d_points': latest['Close'] - prev['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'hist': hist
            }

            return hsi_data
        except Exception as e:
            print(f"❌ 获取恒生指数数据失败: {e}")
            return None

    def get_stock_data(self, symbol, target_date=None):
        """获取指定股票的数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if hist.empty:
                print(f"❌ 无法获取 {symbol} 的历史数据")
                return None

            # 根据target_date截断历史数据
            if target_date is not None:
                # 将target_date转换为pandas时间戳，用于与历史数据的索引比较
                target_timestamp = pd.Timestamp(target_date)
                # 确保target_timestamp是date类型
                target_date_only = target_timestamp.date()
                # 过滤出日期小于等于target_date的数据
                hist = hist[hist.index.date <= target_date_only]
                
                if hist.empty:
                    print(f"⚠️ 在 {target_date} 之前没有 {symbol} 的历史数据")
                    return None

            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest

            stock_data = {
                'symbol': symbol,
                'name': self.stock_list.get(symbol, symbol),
                'current_price': latest['Close'],
                'change_1d': (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0,
                'change_1d_points': latest['Close'] - prev['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'hist': hist
            }

            return stock_data
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据失败: {e}")
            return None

    def calculate_atr(self, df, period=14):
        """
        计算平均真实波幅(ATR)，返回最后一行的 ATR 值（float）
        使用 DataFrame 的副本以避免修改原始数据。
        """
        try:
            if df is None or df.empty:
                return 0.0
            # work on a copy
            dfc = df.copy()
            high = dfc['High'].astype(float)
            low = dfc['Low'].astype(float)
            close = dfc['Close'].astype(float)

            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 使用 Wilder 平滑（EWMA）更稳健
            atr = true_range.ewm(alpha=1/period, adjust=False).mean()

            last_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else 0.0
            return float(last_atr)
        except Exception as e:
            print(f"⚠️ 计算 ATR 失败: {e}")
            return 0.0

    def _round_to_tick(self, price, current_price=None, tick_size=None):
        """
        将 price 四舍五入到最接近的 tick。优先使用传入的 tick_size，
        否则使用实例默认 tick，若都没有则根据 current_price 做简单推断。
        """
        try:
            if price is None or not np.isfinite(price):
                return price
            if tick_size is None:
                tick_size = self.default_tick_size

            if tick_size is None:
                # 简单规则推断（这只是近似）
                if current_price is None:
                    current_price = price
                if current_price >= 100:
                    ts = 0.1
                elif current_price >= 1:
                    ts = 0.01
                else:
                    ts = 0.001
            else:
                ts = float(tick_size)

            # 使用 Decimal 精确四舍五入到最接近的 tick
            if ts <= 0:
                return float(round(price, 8))
            quant = Decimal(str(ts))
            dec_price = Decimal(str(price))
            rounded = (dec_price / quant).to_integral_value(rounding=ROUND_HALF_UP) * quant
            # 把结果转换回 float 并截断多余小数
            return float(rounded)
        except Exception:
            # 回退为普通四舍五入
            return float(round(price, 8))

    def calculate_stop_loss_take_profit(self, hist_df, current_price, signal_type='BUY',
                                       method='ATR', atr_period=14, atr_multiplier=1.5,
                                       risk_reward_ratio=2.0, percentage=0.05,
                                       max_loss_pct=None, tick_size=None):
        """
        更稳健的止损/止盈计算：
        - hist_df: 包含历史 OHLC 的 DataFrame（用于 ATR 计算）
        - current_price: 当前价格（float）
        - signal_type: 'BUY' 或 'SELL'
        - method: 'ATR' 或 'PERCENTAGE'
        - atr_period: ATR 周期
        - atr_multiplier: ATR 倍数
        - risk_reward_ratio: 风险收益比
        - percentage: 固定百分比（如 method == 'PERCENTAGE' 时使用）
        - max_loss_pct: 可选的最大允许亏损百分比（0.2 表示 20%），None 表示不强制
        - tick_size: 最小价格变动单位（如 0.01）
        返回 (stop_loss, take_profit)（float 或 None）
        """
        try:
            # 参数校验
            if current_price is None or not np.isfinite(current_price) or current_price <= 0:
                return None, None

            # 优先根据历史计算 ATR
            atr_value = None
            if method == 'ATR':
                try:
                    atr_value = self.calculate_atr(hist_df, period=atr_period)
                    if not np.isfinite(atr_value) or atr_value <= 0:
                        # 回退到百分比法
                        method = 'PERCENTAGE'
                    # else 使用 atr_value
                except Exception:
                    method = 'PERCENTAGE'

            if method == 'ATR' and atr_value is not None and atr_value > 0:
                if signal_type == 'BUY':
                    sl_raw = current_price - atr_value * atr_multiplier
                    potential_loss = current_price - sl_raw
                    tp_raw = current_price + potential_loss * risk_reward_ratio
                else:  # SELL
                    sl_raw = current_price + atr_value * atr_multiplier
                    potential_loss = sl_raw - current_price
                    tp_raw = current_price - potential_loss * risk_reward_ratio
            else:
                # 使用百分比方法
                if signal_type == 'BUY':
                    sl_raw = current_price * (1 - percentage)
                    tp_raw = current_price * (1 + percentage * risk_reward_ratio)
                else:
                    sl_raw = current_price * (1 + percentage)
                    tp_raw = current_price * (1 - percentage * risk_reward_ratio)

            # 应用最大允许亏损（如设置）
            if max_loss_pct is None:
                max_loss_pct = self.default_max_loss_pct

            if max_loss_pct is not None and max_loss_pct > 0:
                if signal_type == 'BUY':
                    max_allowed_sl = current_price * (1 - max_loss_pct)
                    # 不允许止损低于 max_allowed_sl（即亏损更大于允许值）
                    if sl_raw < max_allowed_sl:
                        sl_raw = max_allowed_sl
                        potential_loss = current_price - sl_raw
                        tp_raw = current_price + potential_loss * risk_reward_ratio
                else:
                    max_allowed_sl = current_price * (1 + max_loss_pct)
                    if sl_raw > max_allowed_sl:
                        sl_raw = max_allowed_sl
                        potential_loss = sl_raw - current_price
                        tp_raw = current_price - potential_loss * risk_reward_ratio

            # 保证止损/止盈方向正确（避免等于或反向）
            eps = 1e-12
            if signal_type == 'BUY':
                sl = min(sl_raw, current_price - eps)
                tp = max(tp_raw, current_price + eps)
            else:
                sl = max(sl_raw, current_price + eps)
                tp = min(tp_raw, current_price - eps)

            # 四舍五入到 tick
            sl = self._round_to_tick(sl, current_price=current_price, tick_size=tick_size)
            tp = self._round_to_tick(tp, current_price=current_price, tick_size=tick_size)

            # 最后校验合理性
            if not (np.isfinite(sl) and np.isfinite(tp)):
                return None, None

            return round(float(sl), 8), round(float(tp), 8)
        except Exception as e:
            print("⚠️ 计算止损止盈异常:", e)
            return None, None

    def calculate_technical_indicators(self, data):
        """
        计算技术指标（适用于恒生指数或个股）
        """
        if data is None:
            return None

        hist = data.get('hist')
        if hist is None or hist.empty:
            return None

        if not TECHNICAL_ANALYSIS_AVAILABLE:
            # 简化指标计算（当 technical_analysis 不可用时）
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest

            indicators = {
                'rsi': self.calculate_rsi((latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0),
                'macd': self.calculate_macd(latest['Close']),
                'price_position': self.calculate_price_position(latest['Close'], hist['Close'].min(), hist['Close'].max()),
            }

            # 使用真实 ATR 计算止损/止盈，若失败回退到百分比法
            try:
                current_price = float(latest['Close'])
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                    hist,
                    current_price,
                    signal_type='BUY',  # 默认为 BUY，用场景可以调整
                    method='ATR',
                    atr_period=14,
                    atr_multiplier=1.5,
                    risk_reward_ratio=2.0,
                    percentage=0.05,
                    max_loss_pct=None,
                    tick_size=None
                )
                indicators['atr'] = self.calculate_atr(hist)
                indicators['stop_loss'] = stop_loss
                indicators['take_profit'] = take_profit
            except Exception as e:
                print(f"⚠️ 计算 ATR 或 止损止盈 失败: {e}")
                indicators['atr'] = 0.0
                indicators['stop_loss'] = None
                indicators['take_profit'] = None

            return indicators

        # 如果 technical_analysis 可用，则使用其方法（保留兼容逻辑）
        try:
            indicators_df = self.technical_analyzer.calculate_all_indicators(hist.copy())
            indicators_with_signals = self.technical_analyzer.generate_buy_sell_signals(indicators_df.copy())
            trend = self.technical_analyzer.analyze_trend(indicators_with_signals)

            latest = indicators_with_signals.iloc[-1]
            rsi = latest.get('RSI', 50.0)
            macd = latest.get('MACD', 0.0)
            macd_signal = latest.get('MACD_signal', 0.0)
            bb_position = latest.get('BB_position', 0.5) if 'BB_position' in latest else 0.5

            # recent signals
            recent_signals = indicators_with_signals.tail(5)
            buy_signals = []
            sell_signals = []

            if 'Buy_Signal' in recent_signals.columns:
                buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                for idx, row in buy_signals_df.iterrows():
                    description = row.get('Signal_Description', '')
                    for prefix in ['买入信号:', '买入信号', 'Buy Signal:', 'Buy Signal']:
                        if description.startswith(prefix):
                            description = description[len(prefix):].strip()
                    buy_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': description})

            if 'Sell_Signal' in recent_signals.columns:
                sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                for idx, row in sell_signals_df.iterrows():
                    description = row.get('Signal_Description', '')
                    for prefix in ['卖出信号:', '卖出信号', 'Sell Signal:', 'Sell Signal']:
                        if description.startswith(prefix):
                            description = description[len(prefix):].strip()
                    sell_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': description})

            # ATR 和止损止盈
            current_price = float(latest.get('Close', hist['Close'].iloc[-1]))
            atr_value = self.calculate_atr(hist)
            # 根据最近信号确定类型，默认 BUY
            signal_type = 'BUY'
            if recent_signals is not None and len(recent_signals) > 0:
                latest_signal = recent_signals.iloc[-1]
                if 'Buy_Signal' in latest_signal and latest_signal['Buy_Signal'] == True:
                    signal_type = 'BUY'
                elif 'Sell_Signal' in latest_signal and latest_signal['Sell_Signal'] == True:
                    signal_type = 'SELL'

            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                hist,
                current_price,
                signal_type=signal_type,
                method='ATR',
                atr_period=14,
                atr_multiplier=1.5,
                risk_reward_ratio=2.0,
                percentage=0.05,
                max_loss_pct=None,
                tick_size=None
            )

            # 添加成交量指标
            volume_ratio = latest.get('Volume_Ratio', 0.0)
            volume_surge = latest.get('Volume_Surge', False)
            volume_shrink = latest.get('Volume_Shrink', False)
            volume_ma10 = latest.get('Volume_MA10', 0.0)
            volume_ma20 = latest.get('Volume_MA20', 0.0)

            return {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'price_position': self.calculate_price_position(latest.get('Close', 0), hist['Close'].min(), hist['Close'].max()),
                'bb_position': bb_position,
                'trend': trend,
                'recent_buy_signals': buy_signals,
                'recent_sell_signals': sell_signals,
                'current_price': latest.get('Close', 0),
                'ma20': latest.get('MA20', 0),
                'ma50': latest.get('MA50', 0),
                'ma200': latest.get('MA200', 0),
                'hist': hist,
                'atr': atr_value,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume_ratio': volume_ratio,
                'volume_surge': volume_surge,
                'volume_shrink': volume_shrink,
                'volume_ma10': volume_ma10,
                'volume_ma20': volume_ma20
            }
        except Exception as e:
            print(f"⚠️ 计算技术指标失败: {e}")
            # 降级为简化计算
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest

            try:
                atr_value = self.calculate_atr(hist)
                current_price = float(latest['Close'])
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                    hist,
                    current_price,
                    signal_type='BUY',
                    method='ATR',
                    atr_period=14,
                    atr_multiplier=1.5,
                    risk_reward_ratio=2.0,
                    percentage=0.05,
                    max_loss_pct=None,
                    tick_size=None
                )
            except Exception as e2:
                print(f"⚠️ 计算 ATR 或 止损止盈 失败: {e2}")
                atr_value = 0.0
                stop_loss = None
                take_profit = None

            return {
                'rsi': self.calculate_rsi((latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0),
                'macd': self.calculate_macd(latest['Close']),
                'price_position': self.calculate_price_position(latest['Close'], hist['Close'].min(), hist['Close'].max()),
                'atr': atr_value,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

    def calculate_rsi(self, change_pct):
        """
        简化RSI计算（基于24小时变化率），仅作指示用途
        """
        try:
            if change_pct > 0:
                return min(100.0, 50.0 + change_pct * 2.0)
            else:
                return max(0.0, 50.0 + change_pct * 2.0)
        except Exception:
            return 50.0

    def calculate_macd(self, price):
        """
        简化MACD计算（基于价格），仅作指示用途
        """
        try:
            return float(price) * 0.01
        except Exception:
            return 0.0

    def calculate_price_position(self, current_price, min_price, max_price):
        """
        计算价格位置（在近期高低点之间的百分位）
        """
        try:
            if max_price == min_price:
                return 50.0
            return (current_price - min_price) / (max_price - min_price) * 100.0
        except Exception:
            return 50.0

    # ---------- 以下为交易记录分析和邮件/报告生成函数 ----------
    def _read_transactions_df(self, path='data/simulation_transactions.csv'):
        """
        使用 pandas 读取交易记录 CSV，返回 DataFrame 并确保 timestamp 列为 UTC datetime。
        该函数尽量智能匹配常见列名（timestamp/time/date, type/trans_type, code/symbol, name）。
        """
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
            if df.empty:
                return pd.DataFrame()
            # 找到时间列
            cols_lower = [c.lower() for c in df.columns]
            timestamp_col = None
            for candidate in ['timestamp', 'time', 'datetime', 'date']:
                if candidate in cols_lower:
                    timestamp_col = df.columns[cols_lower.index(candidate)]
                    break
            if timestamp_col is None:
                # fallback to first column
                timestamp_col = df.columns[0]

            # parse timestamp to UTC
            df[timestamp_col] = pd.to_datetime(df[timestamp_col].astype(str), utc=True, errors='coerce')

            # normalize key columns names to common names
            def find_col(possibilities):
                for p in possibilities:
                    if p in cols_lower:
                        return df.columns[cols_lower.index(p)]
                return None

            type_col = find_col(['type', 'trans_type', 'action'])
            code_col = find_col(['code', 'symbol', 'ticker'])
            name_col = find_col(['name', 'stock_name'])
            reason_col = find_col(['reason', 'desc', 'description'])
            current_price_col = find_col(['current_price', 'price', 'currentprice', 'last_price'])
            stop_loss_col = find_col(['stop_loss', 'stoploss', 'stop_loss_price'])

            # rename to standard columns
            rename_map = {}
            if timestamp_col:
                rename_map[timestamp_col] = 'timestamp'
            if type_col:
                rename_map[type_col] = 'type'
            if code_col:
                rename_map[code_col] = 'code'
            if name_col:
                rename_map[name_col] = 'name'
            if reason_col:
                rename_map[reason_col] = 'reason'
            if current_price_col:
                rename_map[current_price_col] = 'current_price'
            if stop_loss_col:
                rename_map[stop_loss_col] = 'stop_loss_price'

            df = df.rename(columns=rename_map)

            # ensure required columns exist
            for c in ['type', 'code', 'name', 'reason', 'current_price', 'stop_loss_price']:
                if c not in df.columns:
                    df[c] = ''

            # normalize type column
            df['type'] = df['type'].fillna('').astype(str).str.upper()
            # coerce numeric price columns where possible
            df['current_price'] = pd.to_numeric(df['current_price'].replace('', np.nan), errors='coerce')
            df['stop_loss_price'] = pd.to_numeric(df['stop_loss_price'].replace('', np.nan), errors='coerce')

            # drop rows without timestamp
            df = df[~df['timestamp'].isna()].copy()

            return df
        except Exception as e:
            print(f"⚠️ 读取交易记录 CSV 失败: {e}")
            return pd.DataFrame()

    def detect_continuous_signals_in_history_from_transactions(self, stock_code, hours=48, min_signals=3):
        """
        基于交易历史记录检测连续买卖信号（使用 pandas 读取 CSV）
        - stock_code: 股票代码
        - hours: 检测的时间范围（小时）
        - min_signals: 判定为连续信号的最小信号数量
        返回: 连续信号状态字符串
        """
        try:
            df = self._read_transactions_df()
            if df.empty:
                return "无交易记录"

            now = pd.Timestamp.now(tz='UTC')
            threshold = now - pd.Timedelta(hours=hours)

            df_recent = df[(df['timestamp'] >= threshold) & (df['code'] == stock_code)]
            if df_recent.empty:
                return "无建议信号"

            buy_count = int((df_recent['type'].str.contains('BUY')).sum())
            sell_count = int((df_recent['type'].str.contains('SELL')).sum())

            if buy_count >= min_signals and sell_count == 0 and buy_count > 0:
                return f"连续买入({buy_count}次)"
            elif sell_count >= min_signals and buy_count == 0 and sell_count > 0:
                return f"连续卖出({sell_count}次)"
            elif buy_count > 0 and sell_count == 0:
                return f"买入({buy_count}次)"
            elif sell_count > 0 and buy_count == 0:
                return f"卖出({sell_count}次)"
            elif buy_count > 0 and sell_count > 0:
                return f"买入{buy_count}次,卖出{sell_count}次"
            else:
                return "无建议信号"

        except Exception as e:
            print(f"⚠️ 检测连续信号失败: {e}")
            return "检测失败"

    def detect_continuous_signals_in_history(self, indicators_df, hours=48, min_signals=3):
        """
        占位函数：保留原有接口（实际实现建议基于交易记录）
        """
        return "无交易记录"

    def analyze_continuous_signals(self):
        """
        分析最近48小时内的连续买卖信号（使用 pandas 读取 data/simulation_transactions.csv）
        返回: (buy_without_sell_after, sell_without_buy_after)
        每个元素为 (code, name, times_list, reasons_list)
        """
        df = self._read_transactions_df()
        if df.empty:
            return [], []

        now = pd.Timestamp.now(tz='UTC')
        time_48_hours_ago = now - pd.Timedelta(hours=48)
        df_recent = df[df['timestamp'] >= time_48_hours_ago].copy()
        if df_recent.empty:
            return [], []

        results_buy = []
        results_sell = []

        grouped = df_recent.groupby('code')
        for code, group in grouped:
            types = group['type'].fillna('').astype(str).str.upper()
            buy_rows = group[types.str.contains('BUY')]
            sell_rows = group[types.str.contains('SELL')]

            if len(buy_rows) >= 3 and len(sell_rows) == 0:
                name = buy_rows['name'].iloc[0] if 'name' in buy_rows.columns and len(buy_rows) > 0 else 'Unknown'
                times = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in buy_rows['timestamp'].tolist()]
                reasons = buy_rows['reason'].fillna('').tolist() if 'reason' in buy_rows.columns else [''] * len(times)
                results_buy.append((code, name, times, reasons))
            elif len(sell_rows) >= 3 and len(buy_rows) == 0:
                name = sell_rows['name'].iloc[0] if 'name' in sell_rows.columns and len(sell_rows) > 0 else 'Unknown'
                times = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in sell_rows['timestamp'].tolist()]
                reasons = sell_rows['reason'].fillna('').tolist() if 'reason' in sell_rows.columns else [''] * len(times)
                results_sell.append((code, name, times, reasons))

        return results_buy, results_sell

    def has_any_signals(self, hsi_indicators, stock_results, target_date=None):
        """检查是否有任何股票有指定日期的交易信号"""
        if target_date is None:
            target_date = datetime.now().date()

        if hsi_indicators:
            recent_buy_signals = hsi_indicators.get('recent_buy_signals', [])
            recent_sell_signals = hsi_indicators.get('recent_sell_signals', [])
            for signal in recent_buy_signals + recent_sell_signals:
                try:
                    signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
                    if signal_date == target_date:
                        return True
                except Exception:
                    continue

        for stock_result in stock_results:
            indicators = stock_result.get('indicators')
            if indicators:
                for signal in indicators.get('recent_buy_signals', []) + indicators.get('recent_sell_signals', []):
                    try:
                        signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
                        if signal_date == target_date:
                            return True
                    except Exception:
                        continue

        return False

    def generate_stock_analysis_html(self, stock_data, indicators, continuous_buy_signals=None, continuous_sell_signals=None):
        """为单只股票生成HTML分析部分"""
        if not indicators:
            return ""

        continuous_signal_info = None
        if continuous_buy_signals is not None:
            for code, name, times, reasons in continuous_buy_signals:
                if code == stock_data['symbol']:
                    continuous_signal_info = f"连续买入({len(times)}次)"
                    break
        if continuous_signal_info is None and continuous_sell_signals is not None:
            for code, name, times, reasons in continuous_sell_signals:
                if code == stock_data['symbol']:
                    continuous_signal_info = f"连续卖出({len(times)}次)"
                    break

        hist = stock_data['hist']
        recent_data = hist.sort_index()
        last_5_days = recent_data.tail(5)

        multi_day_html = ""
        if len(last_5_days) > 0:
            multi_day_html += """
            <div class="section">
                <h4>📈 五日数据对比</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th>指标</th>
            """
            for date in last_5_days.index:
                multi_day_html += f"<th>{date.strftime('%m-%d')}</th>"
            multi_day_html += "</tr>"

            indicators_list = ['Open', 'High', 'Low', 'Close', 'Volume']
            indicators_names = ['开盘价', '最高价', '最低价', '收盘价', '成交量']

            for i, ind in enumerate(indicators_list):
                multi_day_html += "<tr>"
                multi_day_html += f"<td>{indicators_names[i]}</td>"
                for date, row in last_5_days.iterrows():
                    if ind == 'Volume':
                        value = f"{row[ind]:,.0f}"
                    else:
                        value = f"{row[ind]:,.2f}"
                    multi_day_html += f"<td>{value}</td>"
                multi_day_html += "</tr>"

            multi_day_html += "</table></div>"

        html = f"""
        <div class="section">
            <h3>📊 {stock_data['name']} ({stock_data['symbol']}) 分析</h3>
            <table>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                </tr>
        """

        html += f"""
                <tr>
                    <td>当前价格</td>
                    <td>{stock_data['current_price']:,.2f}</td>
                </tr>
                <tr>
                    <td>24小时变化</td>
                    <td>{stock_data['change_1d']:+.2f}% ({stock_data['change_1d_points']:+.2f})</td>
                </tr>
                <tr>
                    <td>当日开盘</td>
                    <td>{stock_data['open']:,.2f}</td>
                </tr>
                <tr>
                    <td>当日最高</td>
                    <td>{stock_data['high']:,.2f}</td>
                </tr>
                <tr>
                    <td>当日最低</td>
                    <td>{stock_data['low']:,.2f}</td>
                </tr>
                <tr>
                    <td>成交量</td>
                    <td>{stock_data['volume']:,.0f}</td>
                </tr>
        """

        rsi = indicators.get('rsi', 0.0)
        macd = indicators.get('macd', 0.0)
        macd_signal = indicators.get('macd_signal', 0.0)
        bb_position = indicators.get('bb_position', 0.5)
        trend = indicators.get('trend', '未知')
        ma20 = indicators.get('ma20', 0)
        ma50 = indicators.get('ma50', 0)
        ma200 = indicators.get('ma200', 0)
        atr = indicators.get('atr', 0.0)
        stop_loss = indicators.get('stop_loss', None)
        take_profit = indicators.get('take_profit', None)

        # 为趋势设置颜色
        if "多头" in trend:
            trend_color_style = "color: green; font-weight: bold;"
        elif "空头" in trend:
            trend_color_style = "color: red; font-weight: bold;"
        elif "震荡" in trend:
            trend_color_style = "color: blue; font-weight: bold;"
        else:
            trend_color_style = ""

        html += f"""
                <tr>
                    <td>趋势(技术分析)</td>
                    <td><span style=\"{trend_color_style}\">{trend}</span></td>
                </tr>
                <tr>
                    <td>RSI (14日)</td>
                    <td>{rsi:.2f}</td>
                </tr>
                <tr>
                    <td>MACD</td>
                    <td>{macd:.4f}</td>
                </tr>
                <tr>
                    <td>MACD信号线</td>
                    <td>{macd_signal:.4f}</td>
                </tr>
                <tr>
                    <td>布林带位置</td>
                    <td>{bb_position:.2f}</td>
                </tr>
                <tr>
                    <td>MA20</td>
                    <td>{ma20:,.2f}</td>
                </tr>
                <tr>
                    <td>MA50</td>
                    <td>{ma50:,.2f}</td>
                </tr>
                <tr>
                    <td>MA200</td>
                    <td>{ma200:,.2f}</td>
                </tr>
                <tr>
                    <td>ATR (14日)</td>
                    <td>{atr:.2f}</td>
                </tr>
        """

        if stop_loss is not None:
            html += f"""
                <tr>
                    <td>建议止损价</td>
                    <td>{stop_loss:,.2f}</td>
                </tr>
            """

        if take_profit is not None:
            html += f"""
                <tr>
                    <td>建议止盈价</td>
                    <td>{take_profit:,.2f}</td>
                </tr>
            """

        recent_buy_signals = indicators.get('recent_buy_signals', [])
        recent_sell_signals = indicators.get('recent_sell_signals', [])

        if recent_buy_signals:
            html += f"""
                <tr>
                    <td colspan="2">
                        <div class="buy-signal">
                            <strong>🔔 最近买入信号(五天内):</strong><br>
            """
            for signal in recent_buy_signals:
                html += f"<span style='color: green;'>• {signal['date']}: {signal['description']}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """

        if recent_sell_signals:
            html += f"""
                <tr>
                    <td colspan="2">
                        <div class="sell-signal">
                            <strong>🔻 最近卖出信号(五天内):</strong><br>
            """
            for signal in recent_sell_signals:
                html += f"<span style='color: red;'>• {signal['date']}: {signal['description']}</span><br>"
            html += """
                        </div>
                    </td>
                </tr>
            """

        if continuous_signal_info:
            html += f"""
            <tr>
                <td colspan="2">
                    <div class="continuous-signal">
                        <strong>🤖 48小时智能建议:</strong><br>
                        <span style='color: blue;'>• {continuous_signal_info}</span>
                    </div>
                </td>
            </tr>
            """

        html += """
                </table>
        """

        html += multi_day_html
        html += """
            </div>
        """

        return html

    def send_email(self, to, subject, text, html):
        smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
        smtp_user = os.environ.get("YAHOO_EMAIL")
        smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
        sender_email = smtp_user

        if not smtp_user or not smtp_pass:
            print("❌ 缺少YAHOO_EMAIL或YAHOO_APP_PASSWORD环境变量")
            return False

        if isinstance(to, str):
            to = [to]

        msg = MIMEMultipart("alternative")
        msg['From'] = f'<{sender_email}>'
        msg['To'] = ", ".join(to)
        msg['Subject'] = subject

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        if "163.com" in smtp_server:
            smtp_port = 465
            use_ssl = True
        elif "gmail.com" in smtp_server:
            smtp_port = 587
            use_ssl = False
        else:
            smtp_port = 587
            use_ssl = False

        for attempt in range(3):
            try:
                if use_ssl:
                    server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(sender_email, to, msg.as_string())
                    server.quit()
                else:
                    server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(sender_email, to, msg.as_string())
                    server.quit()

                print("✅ 邮件发送成功!")
                return True
            except Exception as e:
                print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                if attempt < 2:
                    import time
                    time.sleep(5)

        print("❌ 3次尝试后仍无法发送邮件")
        return False

    def generate_report_content(self, target_date, hsi_data, hsi_indicators, stock_results):
        """生成报告的HTML和文本内容（此处保留原有结构，使用新的止损止盈结果）"""
        # 创建信号汇总
        all_signals = []

        if hsi_indicators:
            for signal in hsi_indicators.get('recent_buy_signals', []):
                all_signals.append(('恒生指数', 'HSI', signal, '买入'))
            for signal in hsi_indicators.get('recent_sell_signals', []):
                all_signals.append(('恒生指数', 'HSI', signal, '卖出'))

        stock_trends = {}
        for stock_result in stock_results:
            indicators = stock_result.get('indicators') or {}
            trend = indicators.get('trend', '未知')
            stock_trends[stock_result['code']] = trend

        for stock_result in stock_results:
            indicators = stock_result.get('indicators') or {}
            for signal in indicators.get('recent_buy_signals', []):
                all_signals.append((stock_result['name'], stock_result['code'], signal, '买入'))
            for signal in indicators.get('recent_sell_signals', []):
                all_signals.append((stock_result['name'], stock_result['code'], signal, '卖出'))

        target_date_signals = []
        for stock_name, stock_code, signal, signal_type in all_signals:
            try:
                signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
                if signal_date == target_date:
                    trend = stock_trends.get(stock_code, '未知')
                    target_date_signals.append((stock_name, stock_code, trend, signal, signal_type))
            except Exception:
                continue

        # 添加48小时有智能建议但当天无量价信号的股票
        for stock_code, stock_name in self.stock_list.items():
            # 检查是否已经在target_date_signals中
            already_included = any(code == stock_code for _, code, _, _, _ in target_date_signals)
            if not already_included:
                # 检查48小时智能建议
                continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(stock_code)
                if continuous_signal_status != "无建议信号":
                    trend = stock_trends.get(stock_code, '未知')
                    # 创建一个虚拟的信号对象
                    # 确保target_date是date对象
                    if isinstance(target_date, str):
                        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                    else:
                        target_date_obj = target_date
                    dummy_signal = {'description': '仅48小时智能建议', 'date': target_date_obj.strftime('%Y-%m-%d')}
                    target_date_signals.append((stock_name, stock_code, trend, dummy_signal, '无建议信号'))

        target_date_signals.sort(key=lambda x: x[0])

        # 文本版表头（修复原先被截断的 f-string）
        text_lines = []
        text_lines.append("🔔 交易信号总结:")
        header = f"{'股票名称':<15} {'股票代码':<10} {'趋势(技术分析)':<12} {'信号类型':<8} {'48小时智能建议':<20} {'信号描述'}"
        text_lines.append(header)

        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #333; }}
                h3 {{ color: #555; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ margin: 20px 0; }}
                .highlight {{ background-color: #ffffcc; }}
                .buy-signal {{ background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .sell-signal {{ background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h2>📈 恒生指数及港股主力资金追踪器股票交易信号提醒 - {target_date}</h2>
            <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>分析日期:</strong> {target_date}</p>
        """

        html += """
            <div class="section">
                <h3>🔔 交易信号总结</h3>
                <table>
                    <tr>
                        <th>股票名称</th>
                        <th>股票代码</th>
                        <th>趋势(技术分析)</th>
                        <th>信号类型(量价分析)</th>
                        <th>48小时智能建议</th>
                        <th>信号描述(量价分析)</th>
                    </tr>
        """

        for stock_name, stock_code, trend, signal, signal_type in target_date_signals:
            signal_display = f"{signal_type}信号"
            color_style = "color: green; font-weight: bold;" if signal_type == '买入' else "color: red; font-weight: bold;"
            continuous_signal_status = "无信号"
            if stock_code != 'HSI':
                continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(stock_code)

            # 智能过滤：保留有量价信号或有48小时智能建议的股票
            should_show = (signal_type in ['买入', '卖出']) or (continuous_signal_status != "无建议信号")
            
            if not should_show:
                continue
            
            # 为无量价信号但有48小时建议的股票创建特殊显示
            if signal_type not in ['买入', '卖出'] and continuous_signal_status != "无建议信号":
                signal_display = "无量价信号"
                color_style = "color: blue; font-weight: bold;"
                signal_description = f"仅48小时智能建议: {continuous_signal_status}"
            else:
                signal_description = signal.get('description', '') if isinstance(signal, dict) else str(signal)

            # 为48小时智能建议设置颜色
            if "买入" in continuous_signal_status:
                signal_color_style = "color: green; font-weight: bold;"
            elif "卖出" in continuous_signal_status:
                signal_color_style = "color: red; font-weight: bold;"
            elif "无建议信号" in continuous_signal_status:
                signal_color_style = "color: blue; font-weight: bold;"
            else:
                signal_color_style = ""
            
            # 为趋势设置颜色
            if "多头" in trend:
                trend_color_style = "color: green; font-weight: bold;"
            elif "空头" in trend:
                trend_color_style = "color: red; font-weight: bold;"
            elif "震荡" in trend:
                trend_color_style = "color: blue; font-weight: bold;"
            else:
                trend_color_style = ""
            
            html += f"""
                    <tr>
                        <td>{stock_name}</td>
                        <td>{stock_code}</td>
                        <td><span style=\"{trend_color_style}\">{trend}</span></td>
                        <td><span style=\"{color_style}\">{signal_display}</span></td>
                        <td><span style=\"{signal_color_style}\">{continuous_signal_status}</span></td>
                        <td>{signal_description}</td>
                    </tr>
            """

            # 文本版本追加
            text_lines.append(f"{stock_name:<15} {stock_code:<10} {trend:<12} {signal_display:<8} {continuous_signal_status:<20} {signal_description}")

        # 检查过滤后是否有信号（使用新的过滤逻辑）
        has_filtered_signals = any(True for stock_name, stock_code, trend, signal, signal_type in target_date_signals
                                   if (signal_type in ['买入', '卖出']) or (self.detect_continuous_signals_in_history_from_transactions(stock_code) != "无建议信号"))

        if not has_filtered_signals:
            html += """
                    <tr>
                        <td colspan="6">当前没有检测到任何有效的交易信号（已过滤无信号股票）</td>
                    </tr>
            """
            text_lines.append("当前没有检测到任何有效的交易信号（已过滤无信号股票）")

        html += """
                </table>
            </div>
        """

        text = "\n".join(text_lines) + "\n\n"

        # 连续信号分析
        print("🔍 正在分析最近48小时内的连续交易信号...")
        buy_without_sell_after, sell_without_buy_after = self.analyze_continuous_signals()
        has_continuous_signals = len(buy_without_sell_after) > 0 or len(sell_without_buy_after) > 0

        if has_continuous_signals:
            html += """
            <div class="section">
                <h3>🔔 48小时连续交易信号分析</h3>
            """
            if buy_without_sell_after:
                html += """
                <div class="section">
                    <h3>📈 最近48小时内连续3次或以上建议买入同一只股票（期间没有卖出建议）</h3>
                    <table>
                        <tr>
                            <th>股票代码</th>
                            <th>股票名称</th>
                            <th>建议次数</th>
                            <th>建议时间、现价、止损价</th>
                        </tr>
                """
                for code, name, times, reasons in buy_without_sell_after:
                    combined_str = ""
                    for i in range(len(times)):
                        time_info = f"{times[i]}"
                        reason = reasons[i] if i < len(reasons) else ''
                        price_info = ""
                        stop_loss_info = ""
                        if isinstance(reason, str) and '现价' in reason:
                            import re
                            price_match = re.search(r'现价[:：]?\s*([0-9.]+)', reason)
                            if price_match:
                                price_info = f"现价: {price_match.group(1)}"
                        if isinstance(reason, str) and '止损价' in reason:
                            import re
                            stop_loss_match = re.search(r'止损价[:：]?\s*([0-9.]+)', reason)
                            if stop_loss_match:
                                stop_loss_info = f"止损价: {stop_loss_match.group(1)}"
                        info_parts = [part for part in [price_info, stop_loss_info] if part]
                        reason_info = ", ".join(info_parts)
                        time_reason = f"{time_info} {reason_info}".strip()
                        combined_str += time_reason + ("<br>" if i < len(times) - 1 else "")
                    html += f"""
                    <tr>
                        <td>{code}</td>
                        <td>{name}</td>
                        <td>{len(times)}次</td>
                        <td>{combined_str}</td>
                    </tr>
                    """
                html += """
                    </table>
                </div>
                """

            if sell_without_buy_after:
                html += """
                <div class="section">
                    <h3>📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）</h3>
                    <table>
                        <tr>
                            <th>股票代码</th>
                            <th>股票名称</th>
                            <th>建议次数</th>
                            <th>建议时间、现价、止损价</th>
                        </tr>
                """
                for code, name, times, reasons in sell_without_buy_after:
                    combined_str = ""
                    for i in range(len(times)):
                        time_info = f"{times[i]}"
                        reason = reasons[i] if i < len(reasons) else ''
                        price_info = ""
                        stop_loss_info = ""
                        if isinstance(reason, str) and '现价' in reason:
                            import re
                            price_match = re.search(r'现价[:：]?\s*([0-9.]+)', reason)
                            if price_match:
                                price_info = f"现价: {price_match.group(1)}"
                        if isinstance(reason, str) and '止损价' in reason:
                            import re
                            stop_loss_match = re.search(r'止损价[:：]?\s*([0-9.]+)', reason)
                            if stop_loss_match:
                                stop_loss_info = f"止损价: {stop_loss_match.group(1)}"
                        info_parts = [part for part in [price_info, stop_loss_info] if part]
                        reason_info = ", ".join(info_parts)
                        time_reason = f"{time_info} {reason_info}".strip()
                        combined_str += time_reason + ("<br>" if i < len(times) - 1 else "")
                    html += f"""
                    <tr>
                        <td>{code}</td>
                        <td>{name}</td>
                        <td>{len(times)}次</td>
                        <td>{combined_str}</td>
                    </tr>
                    """
                html += """
                    </table>
                </div>
                """
            html += """
            </div>
            """

        if buy_without_sell_after:
            text += f"📈 最近48小时内连续3次或以上建议买入同一只股票（期间没有卖出建议）:\n"
            for code, name, times, reasons in buy_without_sell_after:
                combined_list = []
                for i in range(len(times)):
                    time_info = f"{times[i]}"
                    reason = reasons[i] if i < len(reasons) else ''
                    price_info = ""
                    stop_loss_info = ""
                    if isinstance(reason, str) and '现价' in reason:
                        import re
                        price_match = re.search(r'现价[:：]?\s*([0-9.]+)', reason)
                        if price_match:
                            price_info = f"现价: {price_match.group(1)}"
                    if isinstance(reason, str) and '止损价' in reason:
                        import re
                        stop_loss_match = re.search(r'止损价[:：]?\s*([0-9.]+)', reason)
                        if stop_loss_match:
                            stop_loss_info = f"止损价: {stop_loss_match.group(1)}"
                    info_parts = [part for part in [price_info, stop_loss_info] if part]
                    reason_info = ", ".join(info_parts)
                    combined_item = f"{time_info} {reason_info}".strip()
                    combined_list.append(combined_item)
                combined_str = "\n    ".join(combined_list)
                text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
            text += "\n"

        if sell_without_buy_after:
            text += f"📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）:\n"
            for code, name, times, reasons in sell_without_buy_after:
                combined_list = []
                for i in range(len(times)):
                    time_info = f"{times[i]}"
                    reason = reasons[i] if i < len(reasons) else ''
                    price_info = ""
                    stop_loss_info = ""
                    if isinstance(reason, str) and '现价' in reason:
                        import re
                        price_match = re.search(r'现价[:：]?\s*([0-9.]+)', reason)
                        if price_match:
                            price_info = f"现价: {price_match.group(1)}"
                    if isinstance(reason, str) and '止损价' in reason:
                        import re
                        stop_loss_match = re.search(r'止损价[:：]?\s*([0-9.]+)', reason)
                        if stop_loss_match:
                            stop_loss_info = f"止损价: {stop_loss_match.group(1)}"
                    info_parts = [part for part in [price_info, stop_loss_info] if part]
                    reason_info = ", ".join(info_parts)
                    combined_item = f"{time_info} {reason_info}".strip()
                    combined_list.append(combined_item)
                combined_str = "\n    ".join(combined_list)
                text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
            text += "\n"

        if has_continuous_signals:
            text += "📋 说明:\n"
            text += "连续买入：指在最近48小时内，某只股票收到3次或以上买入建议，且期间没有收到任何卖出建议。\n"
            text += "连续卖出：指在最近48小时内，某只股票收到3次或以上卖出建议，且期间没有收到任何买入建议。\n\n"

            html += """
            <div class="section">
                <h3>📋 说明</h3>
                <div style="font-size:0.9em; line-height:1.4;">
                <ul>
                  <li><b>连续买入</b>：指在最近48小时内，某只股票收到3次或以上买入建议，且期间没有收到任何卖出建议。</li>
                  <li><b>连续卖出</b>：指在最近48小时内，某只股票收到3次或以上卖出建议，且期间没有收到任何买入建议。</li>
                </ul>
                </div>
            </div>
            """

        text += "\n"

        # 添加最近48小时的模拟交易记录（使用 pandas）
        html += """
        <div class="section">
            <h3>💰 最近48小时模拟交易记录</h3>
        """
        
        try:
            df_all = self._read_transactions_df()
            if df_all.empty:
                html += "<p>未找到交易记录文件或文件为空</p>"
                text += "💰 最近48小时模拟交易记录:\n  未找到交易记录文件或文件为空\n"
            else:
                now = pd.Timestamp.now(tz='UTC')
                time_48_hours_ago = now - pd.Timedelta(hours=48)
                df_recent = df_all[df_all['timestamp'] >= time_48_hours_ago].copy()
                if df_recent.empty:
                    html += "<p>最近48小时内没有交易记录</p>"
                    text += "💰 最近48小时模拟交易记录:\n  最近48小时内没有交易记录\n"
                else:
                    # sort by name then time
                    df_recent.sort_values(by=['name', 'timestamp'], inplace=True)
                    html += """
                    <table>
                        <tr>
                            <th>股票名称</th>
                            <th>股票代码</th>
                            <th>时间</th>
                            <th>类型</th>
                            <th>价格</th>
                            <th>理由</th>
                        </tr>
                    """
                    for _, trans in df_recent.iterrows():
                        trans_type = trans.get('type', '')
                        row_style = "background-color: #e8f5e9;" if 'BUY' in str(trans_type).upper() else "background-color: #ffebee;"
                        price = trans.get('current_price', np.nan)
                        price_display = f"{price:,.2f}" if not pd.isna(price) else (trans.get('price', '') or '')
                        reason = trans.get('reason', '') or ''
                        html += f"""
                        <tr style="{row_style}">
                            <td>{trans.get('name','')}</td>
                            <td>{trans.get('code','')}</td>
                            <td>{pd.Timestamp(trans['timestamp']).strftime('%m-%d %H:%M:%S')}</td>
                            <td>{trans_type}</td>
                            <td>{price_display}</td>
                            <td>{reason}</td>
                        </tr>
                        """
                    html += "</table>"

                    # 文本版
                    text += "💰 最近48小时模拟交易记录:\n"
                    from collections import OrderedDict
                    grouped_transactions = OrderedDict()
                    for _, tr in df_recent.iterrows():
                        n = tr.get('name','')
                        if n not in grouped_transactions:
                            grouped_transactions[n] = []
                        grouped_transactions[n].append(tr)
                    for stock_name, trans_list in grouped_transactions.items():
                        code = trans_list[0].get('code','')
                        text += f"  {stock_name} ({code}):\n"
                        for tr in trans_list:
                            trans_type = tr.get('type','')
                            timestamp = pd.Timestamp(tr['timestamp']).strftime('%m-%d %H:%M:%S')
                            price = tr.get('current_price', np.nan)
                            price_display = f"{price:,.2f}" if not pd.isna(price) else ''
                            reason = tr.get('reason','') or ''
                            text += f"    {timestamp} {trans_type} @ {price_display} ({reason})\n"
        except Exception as e:
            html += f"<p>读取交易记录时出错: {str(e)}</p>"
            text += f"💰 最近48小时模拟交易记录:\n  读取交易记录时出错: {str(e)}\n"
        
        html += """
            </div>
        """

        text += "\n"

        if hsi_data:
            html += """
                <div class="section">
                    <h3>📈 恒生指数价格概览</h3>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                        </tr>
            """

            html += f"""
                    <tr>
                        <td>当前指数</td>
                        <td>{hsi_data['current_price']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>24小时变化</td>
                        <td>{hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)</td>
                    </tr>
                    <tr>
                        <td>当日开盘</td>
                        <td>{hsi_data['open']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>当日最高</td>
                        <td>{hsi_data['high']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>当日最低</td>
                        <td>{hsi_data['low']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>成交量</td>
                        <td>{hsi_data['volume']:,.0f}</td>
                    </tr>
            """

            if hsi_indicators:
                rsi = hsi_indicators.get('rsi', 0.0)
                macd = hsi_indicators.get('macd', 0.0)
                macd_signal = hsi_indicators.get('macd_signal', 0.0)
                bb_position = hsi_indicators.get('bb_position', 0.5)
                trend = hsi_indicators.get('trend', '未知')
                ma20 = hsi_indicators.get('ma20', 0)
                ma50 = hsi_indicators.get('ma50', 0)
                ma200 = hsi_indicators.get('ma200', 0)
                atr = hsi_indicators.get('atr', 0.0)
                stop_loss = hsi_indicators.get('stop_loss', None)
                take_profit = hsi_indicators.get('take_profit', None)

                # 为恒生指数趋势设置颜色
                if "多头" in trend:
                    hsi_trend_color_style = "color: green; font-weight: bold;"
                elif "空头" in trend:
                    hsi_trend_color_style = "color: red; font-weight: bold;"
                elif "震荡" in trend:
                    hsi_trend_color_style = "color: blue; font-weight: bold;"
                else:
                    hsi_trend_color_style = ""
                
                html += f"""
                    <tr>
                        <td>趋势(技术分析)</td>
                        <td><span style=\"{hsi_trend_color_style}\">{trend}</span></td>
                    </tr>
                    <tr>
                        <td>RSI (14日)</td>
                        <td>{rsi:.2f}</td>
                    </tr>
                    <tr>
                        <td>MACD</td>
                        <td>{macd:.4f}</td>
                    </tr>
                    <tr>
                        <td>MACD信号线</td>
                        <td>{macd_signal:.4f}</td>
                    </tr>
                    <tr>
                        <td>布林带位置</td>
                        <td>{bb_position:.2f}</td>
                    </tr>
                    <tr>
                        <td>MA20</td>
                        <td>{ma20:,.2f}</td>
                    </tr>
                    <tr>
                        <td>MA50</td>
                        <td>{ma50:,.2f}</td>
                    </tr>
                    <tr>
                        <td>MA200</td>
                        <td>{ma200:,.2f}</td>
                    </tr>
                    <tr>
                        <td>ATR (14日)</td>
                        <td>{atr:.2f}</td>
                    </tr>
                """

                if stop_loss is not None:
                    html += f"""
                        <tr>
                            <td>建议止损价</td>
                            <td>{stop_loss:,.2f}</td>
                        </tr>
                    """

                if take_profit is not None:
                    html += f"""
                        <tr>
                            <td>建议止盈价</td>
                            <td>{take_profit:,.2f}</td>
                        </tr>
                    """

                recent_buy_signals = hsi_indicators.get('recent_buy_signals', [])
                recent_sell_signals = hsi_indicators.get('recent_sell_signals', [])

                if recent_buy_signals:
                    html += f"""
                        <tr>
                            <td colspan="2">
                                <div class="buy-signal">
                                    <strong>🔔 恒生指数最近买入信号:</strong><br>
                        """
                    for signal in recent_buy_signals:
                        html += f"<span style='color: green;'>• {signal['date']}: {signal['description']}</span><br>"
                    html += """
                                </div>
                            </td>
                        </tr>
                    """

                if recent_sell_signals:
                    html += f"""
                        <tr>
                            <td colspan="2">
                                <div class="sell-signal">
                                    <strong>🔻 恒生指数最近卖出信号:</strong><br>
                        """
                    for signal in recent_sell_signals:
                        html += f"<span style='color: red;'>• {signal['date']}: {signal['description']}</span><br>"
                    html += """
                                </div>
                            </td>
                        </tr>
                    """

            html += """
                    </table>
                </div>
            """

            text += f"📈 恒生指数价格概览:\n"
            text += f"  当前指数: {hsi_data['current_price']:,.2f}\n"
            text += f"  24小时变化: {hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)\n"
            text += f"  当日开盘: {hsi_data['open']:,.2f}\n"
            text += f"  当日最高: {hsi_data['high']:,.2f}\n"
            text += f"  当日最低: {hsi_data['low']:,.2f}\n"
            text += f"  成交量: {hsi_data['volume']:,.0f}\n\n"

            if hsi_indicators:
                text += f"📊 恒生指数技术分析:\n"
                text += f"  趋势(技术分析): {trend}\n"
                text += f"  RSI: {rsi:.2f}\n"
                text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
                text += f"  布林带位置: {bb_position:.2f}\n"
                text += f"  MA20: {ma20:,.2f}\n"
                text += f"  MA50: {ma50:,.2f}\n"
                text += f"  MA200: {ma200:,.2f}\n"
                text += f"  ATR: {atr:.2f}\n"

                if stop_loss is not None:
                    text += f"  建议止损价: {stop_loss:,.2f}\n"
                if take_profit is not None:
                    text += f"  建议止盈价: {take_profit:,.2f}\n"

                if recent_buy_signals:
                    text += f"  🔔 最近买入信号(五天内) ({len(recent_buy_signals)} 个):\n"
                    for signal in recent_buy_signals:
                        text += f"    {signal['date']}: {signal['description']}\n"

                if recent_sell_signals:
                    text += f"  🔻 最近卖出信号(五天内) ({len(recent_sell_signals)} 个):\n"
                    for signal in recent_sell_signals:
                        text += f"    {signal['date']}: {signal['description']}\n"

            text += "\n"

        # 添加股票分析结果
        for stock_result in stock_results:
            stock_data = stock_result['data']
            indicators = stock_result.get('indicators') or {}

            if indicators:
                html += self.generate_stock_analysis_html(stock_data, indicators, buy_without_sell_after, sell_without_buy_after)

                text += f"📊 {stock_result['name']} ({stock_result['code']}) 分析:\n"
                text += f"  当前价格: {stock_data['current_price']:,.2f}\n"
                text += f"  24小时变化: {stock_data['change_1d']:+.2f}% ({stock_data['change_1d_points']:+.2f})\n"
                text += f"  当日开盘: {stock_data['open']:,.2f}\n"
                text += f"  当日最高: {stock_data['high']:,.2f}\n"
                text += f"  当日最低: {stock_data['low']:,.2f}\n"
                text += f"  成交量: {stock_data['volume']:,.0f}\n"

                hist = stock_data['hist']
                recent_data = hist.sort_index()
                last_5_days = recent_data.tail(5)

                if len(last_5_days) > 0:
                    text += f"  📈 五日数据对比:\n"
                    date_line = "    日期:     "
                    for date in last_5_days.index:
                        date_str = date.strftime('%m-%d')
                        date_line += f"{date_str:>10} "
                    text += date_line + "\n"

                    open_line = "    开盘价:   "
                    for date, row in last_5_days.iterrows():
                        open_str = f"{row['Open']:,.2f}"
                        open_line += f"{open_str:>10} "
                    text += open_line + "\n"

                    high_line = "    最高价:   "
                    for date, row in last_5_days.iterrows():
                        high_str = f"{row['High']:,.2f}"
                        high_line += f"{high_str:>10} "
                    text += high_line + "\n"

                    low_line = "    最低价:   "
                    for date, row in last_5_days.iterrows():
                        low_str = f"{row['Low']:,.2f}"
                        low_line += f"{low_str:>10} "
                    text += low_line + "\n"

                    close_line = "    收盘价:   "
                    for date, row in last_5_days.iterrows():
                        close_str = f"{row['Close']:,.2f}"
                        close_line += f"{close_str:>10} "
                    text += close_line + "\n"

                    volume_line = "    成交量:   "
                    for date, row in last_5_days.iterrows():
                        volume_str = f"{row['Volume']:,.0f}"
                        volume_line += f"{volume_str:>10} "
                    text += volume_line + "\n"

                rsi = indicators.get('rsi', 0.0)
                macd = indicators.get('macd', 0.0)
                macd_signal = indicators.get('macd_signal', 0.0)
                bb_position = indicators.get('bb_position', 0.5)
                trend = indicators.get('trend', '未知')
                ma20 = indicators.get('ma20', 0)
                ma50 = indicators.get('ma50', 0)
                ma200 = indicators.get('ma200', 0)
                atr = indicators.get('atr', 0.0)
                stop_loss = indicators.get('stop_loss', None)
                take_profit = indicators.get('take_profit', None)

                text += f"  趋势(技术分析): {trend}\n"
                text += f"  RSI: {rsi:.2f}\n"
                text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
                text += f"  布林带位置: {bb_position:.2f}\n"
                text += f"  MA20: {ma20:,.2f}\n"
                text += f"  MA50: {ma50:,.2f}\n"
                text += f"  MA200: {ma200:,.2f}\n"
                text += f"  ATR: {atr:.2f}\n"

                if stop_loss is not None:
                    text += f"  建议止损价: {stop_loss:,.2f}\n"
                if take_profit is not None:
                    text += f"  建议止盈价: {take_profit:,.2f}\n"

                recent_buy_signals = indicators.get('recent_buy_signals', [])
                recent_sell_signals = indicators.get('recent_sell_signals', [])

                if recent_buy_signals:
                    text += f"  🔔 最近买入信号(五天内) ({len(recent_buy_signals)} 个):\n"
                    for signal in recent_buy_signals:
                        text += f"    {signal['date']}: {signal['description']}\n"

                if recent_sell_signals:
                    text += f"  🔻 最近卖出信号(五天内) ({len(recent_sell_signals)} 个):\n"
                    for signal in recent_sell_signals:
                        text += f"    {signal['date']}: {signal['description']}\n"

                continuous_signal_info = None
                for code, name, times, reasons in buy_without_sell_after:
                    if code == stock_result['code']:
                        continuous_signal_info = f"连续买入({len(times)}次)"
                        break
                if continuous_signal_info is None:
                    for code, name, times, reasons in sell_without_buy_after:
                        if code == stock_result['code']:
                            continuous_signal_info = f"连续卖出({len(times)}次)"
                            break

                if continuous_signal_info:
                    text += f"  🤖 48小时智能建议: {continuous_signal_info}\n"

                text += "\n"

        html += """
        <div class="section">
            <h3>📋 指标说明</h3>
            <div style="font-size:0.9em; line-height:1.4;">
            <ul>
              <li><b>当前指数/价格</b>：恒生指数或股票的实时点位/价格。</li>
              <li><b>24小时变化</b>：过去24小时内指数或股价的变化百分比和点数/金额。</li>
              <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
              <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
              <li><b>MA20(20日移动平均线)</b>：过去20个交易日的平均指数/股价，反映短期趋势。</li>
              <li><b>MA50(50日移动平均线)</b>：过去50个交易日的平均指数/股价，反映中期趋势。</li>
              <li><b>MA200(200日移动平均线)</b>：过去200个交易日的平均指数/股价，反映长期趋势。</li>
              <li><b>布林带位置</b>：当前指数/股价在布林带中的相对位置，范围0-1。</li>
              <li><b>ATR(平均真实波幅)</b>：衡量市场波动性的技术指标，数值越高表示波动越大，常用于设置止损和止盈位。</li>
              <li><b>趋势(技术分析)</b>：市场当前的整体方向。</li>
              <li><b>信号描述(量价分析)</b>：基于价格和成交量关系的技术信号类型：
                <ul>
                  <li><b>上升趋势形成</b>：短期均线(MA20)上穿中期均线(MA50)，形成上升趋势</li>
                  <li><b>下降趋势形成</b>：短期均线(MA20)下穿中期均线(MA50)，形成下降趋势</li>
                  <li><b>MACD金叉</b>：MACD线上穿信号线，预示上涨动能增强</li>
                  <li><b>MACD死叉</b>：MACD线下穿信号线，预示下跌动能增强</li>
                  <li><b>RSI超卖反弹</b>：RSI从超卖区域(30以下)回升，预示价格可能反弹</li>
                  <li><b>RSI超买回落</b>：RSI从超买区域(70以上)回落，预示价格可能回调</li>
                  <li><b>布林带下轨反弹</b>：价格从布林带下轨反弹，预示支撑有效</li>
                  <li><b>跌破布林带上轨</b>：价格跌破布林带上轨，预示阻力有效</li>
                  <li><b>价量配合反转(强/中/弱)</b>：前一天价格相反方向+当天价格反转+成交量放大，预示趋势反转</li>
                  <li><b>价量配合延续(强/中/弱)</b>：连续同向价格变化+成交量放大，预示趋势延续</li>
                  <li><b>价量配合上涨/下跌</b>：价格上涨/下跌+成交量放大，价量同向配合</li>
                  <li><b>成交量确认</b>：括号内表示成交量放大程度，强(>2倍)、中(>1.5倍)、弱(>1.2倍)、普通(>0.9倍)</li>
                </ul>
              </li>
              <li><b>48小时内人工智能买卖建议</b>：基于大模型分析的智能交易建议：
                <ul>
                  <li><b>连续买入(N次)</b>：48小时内连续N次买入建议，无卖出建议，强烈看好</li>
                  <li><b>连续卖出(N次)</b>：48小时内连续N次卖出建议，无买入建议，强烈看空</li>
                  <li><b>买入(N次)</b>：48小时内N次买入建议，可能有卖出建议</li>
                  <li><b>卖出(N次)</b>：48小时内N次卖出建议，可能有买入建议</li>
                  <li><b>买入M次,卖出N次</b>：48小时内买卖建议混合，市场观点不明</li>
                  <li><b>无建议信号</b>：48小时内无任何买卖建议，缺乏明确信号</li>
                </ul>
              </li>
            </ul>
            </div>
        </div>
        """

        # 添加文本版本的指标说明
        text += "\n📋 指标说明:\n"
        text += "• 当前指数/价格：恒生指数或股票的实时点位/价格。\n"
        text += "• 24小时变化：过去24小时内指数或股价的变化百分比和点数/金额。\n"
        text += "• RSI(相对强弱指数)：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。\n"
        text += "• MACD(异同移动平均线)：判断价格趋势和动能的技术指标。\n"
        text += "• MA20(20日移动平均线)：过去20个交易日的平均指数/股价，反映短期趋势。\n"
        text += "• MA50(50日移动平均线)：过去50个交易日的平均指数/股价，反映中期趋势。\n"
        text += "• MA200(200日移动平均线)：过去200个交易日的平均指数/股价，反映长期趋势。\n"
        text += "• 布林带位置：当前指数/股价在布林带中的相对位置，范围0-1。\n"
        text += "• ATR(平均真实波幅)：衡量市场波动性的技术指标，数值越高表示波动越大，常用于设置止损和止盈位。\n"
        text += "• 趋势(技术分析)：市场当前的整体方向。\n"
        text += "• 信号描述(量价分析)：基于价格和成交量关系的技术信号类型：\n"
        text += "  - 上升趋势形成：短期均线(MA20)上穿中期均线(MA50)，形成上升趋势\n"
        text += "  - 下降趋势形成：短期均线(MA20)下穿中期均线(MA50)，形成下降趋势\n"
        text += "  - MACD金叉：MACD线上穿信号线，预示上涨动能增强\n"
        text += "  - MACD死叉：MACD线下穿信号线，预示下跌动能增强\n"
        text += "  - RSI超卖反弹：RSI从超卖区域(30以下)回升，预示价格可能反弹\n"
        text += "  - RSI超买回落：RSI从超买区域(70以上)回落，预示价格可能回调\n"
        text += "  - 布林带下轨反弹：价格从布林带下轨反弹，预示支撑有效\n"
        text += "  - 跌破布林带上轨：价格跌破布林带上轨，预示阻力有效\n"
        text += "  - 价量配合反转(强/中/弱)：前一天价格相反方向+当天价格反转+成交量放大，预示趋势反转\n"
        text += "  - 价量配合延续(强/中/弱)：连续同向价格变化+成交量放大，预示趋势延续\n"
        text += "  - 价量配合上涨/下跌：价格上涨/下跌+成交量放大，价量同向配合\n"
        text += "  - 成交量确认：括号内表示成交量放大程度，强(>2倍)、中(>1.5倍)、弱(>1.2倍)、普通(>0.9倍)\n"
        text += "• 48小时内人工智能买卖建议：基于大模型分析的智能交易建议：\n"
        text += "  - 连续买入(N次)：48小时内连续N次买入建议，无卖出建议，强烈看好\n"
        text += "  - 连续卖出(N次)：48小时内连续N次卖出建议，无买入建议，强烈看空\n"
        text += "  - 买入(N次)：48小时内N次买入建议，可能有卖出建议\n"
        text += "  - 卖出(N次)：48小时内N次卖出建议，可能有买入建议\n"
        text += "  - 买入M次,卖出N次：48小时内买卖建议混合，市场观点不明\n"
        text += "  - 无建议信号：48小时内无任何买卖建议，缺乏明确信号\n"

        html += "</body></html>"

        return text, html

    def run_analysis(self, target_date=None):
        """执行分析并发送邮件"""
        if target_date is None:
            target_date = datetime.now().date()

        print(f"📅 分析日期: {target_date} (默认为今天)")

        print("🔍 正在获取恒生指数数据...")
        hsi_data = self.get_hsi_data(target_date=target_date)
        if hsi_data is None:
            print("❌ 无法获取恒生指数数据")
            hsi_indicators = None
        else:
            print("📊 正在计算恒生指数技术指标...")
            hsi_indicators = self.calculate_technical_indicators(hsi_data)

        print(f"🔍 正在获取股票列表并分析 ({len(self.stock_list)} 只股票)...")
        stock_results = []
        for stock_code, stock_name in self.stock_list.items():
            print(f"🔍 正在分析 {stock_name} ({stock_code}) ...")
            stock_data = self.get_stock_data(stock_code, target_date=target_date)
            if stock_data:
                print(f"📊 正在计算 {stock_name} ({stock_code}) 技术指标...")
                indicators = self.calculate_technical_indicators(stock_data)
                stock_results.append({
                    'code': stock_code,
                    'name': stock_name,
                    'data': stock_data,
                    'indicators': indicators
                })

        if not self.has_any_signals(hsi_indicators, stock_results, target_date):
            print("⚠️ 没有检测到任何交易信号，跳过发送邮件。")
            return False

        subject = "恒生指数及港股交易信号提醒 - 包含最近48小时模拟交易记录"
        text, html = self.generate_report_content(target_date, hsi_data, hsi_indicators, stock_results)

        recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        if ',' in recipient_env:
            recipients = [recipient.strip() for recipient in recipient_env.split(',')]
        else:
            recipients = [recipient_env]

        print("🔔 检测到交易信号，发送邮件到:", ", ".join(recipients))
        print("📝 主题:", subject)
        print("📄 文本预览:\n", text)

        success = self.send_email(recipients, subject, text, html)
        return success


# === 主逻辑 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='恒生指数及港股主力资金追踪器股票交易信号邮件通知系统')
    parser.add_argument('--date', type=str, default=None, help='指定日期 (格式: YYYY-MM-DD)，默认为今天')
    args = parser.parse_args()

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            print(f"📅 指定分析日期: {target_date}")
        except ValueError:
            print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
            exit(1)
    else:
        target_date = datetime.now().date()

    email_system = HSIEmailSystem()
    success = email_system.run_analysis(target_date)

    if not success:
        exit(1)
