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
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import akshare as ak
from decimal import Decimal, ROUND_HALF_UP

# 导入技术分析工具（可选）
try:
    from technical_analysis import TechnicalAnalyzer, TechnicalAnalyzerV2, TAVScorer, TAVConfig
    TECHNICAL_ANALYSIS_AVAILABLE = True
    TAV_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    TAV_AVAILABLE = False
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

    # 根据投资风格和计算窗口确定历史数据长度
    DATA_PERIOD_CONFIG = {
        'ultra_short_term': '6mo',    # 超短线：6个月数据（约125个交易日）
        'short_term': '1y',           # 波段交易：1年数据（约247个交易日）
        'medium_long_term': '2y',      # 中长期投资：2年数据（约493个交易日）
    }

    def __init__(self, stock_list=None):
        self.stock_list = stock_list or STOCK_LIST
        # 添加数据缓存机制
        self._data_cache = {}  # 格式: {symbol_investment_style: DataFrame}
        self._cache_timestamp = {}  # 缓存时间戳
        self._cache_ttl = 3600  # 缓存1小时
        if TECHNICAL_ANALYSIS_AVAILABLE:
            if TAV_AVAILABLE:
                self.technical_analyzer = TechnicalAnalyzerV2(enable_tav=True)
                self.use_tav = True
            else:
                self.technical_analyzer = TechnicalAnalyzer()
                self.use_tav = False
        else:
            self.technical_analyzer = None
            self.use_tav = False

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

    def get_data_for_investment_style(self, symbol, investment_style='short_term'):
        """
        根据投资风格动态获取历史数据（带缓存）
        
        参数:
        - symbol: 股票代码
        - investment_style: 投资风格
        
        返回:
        - 历史数据DataFrame
        """
        try:
            import time
            
            # 生成缓存键
            cache_key = f"{symbol}_{investment_style}"
            current_time = time.time()
            
            # 检查缓存
            if cache_key in self._data_cache:
                # 检查缓存是否过期
                if current_time - self._cache_timestamp.get(cache_key, 0) < self._cache_ttl:
                    return self._data_cache[cache_key]
                else:
                    # 缓存过期，删除
                    del self._data_cache[cache_key]
                    del self._cache_timestamp[cache_key]
            
            # 根据投资风格获取对应的数据周期
            period = self.DATA_PERIOD_CONFIG.get(investment_style, '6mo')
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"⚠️ 无法获取 {symbol} 的历史数据 (period={period})")
                return None
            
            # 验证数据量是否足够
            if investment_style == 'medium_long_term' and len(hist) < 200:
                print(f"⚠️ {symbol} 20日ES计算需要至少200个交易日数据，当前只有{len(hist)}个")
            elif investment_style == 'short_term' and len(hist) < 50:
                print(f"⚠️ {symbol} 5日ES计算建议至少50个交易日数据，当前只有{len(hist)}个")
            
            # 缓存数据
            self._data_cache[cache_key] = hist
            self._cache_timestamp[cache_key] = current_time
            
            return hist
        except Exception as e:
            print(f"⚠️ 获取 {symbol} 数据失败: {e}")
            return None

    def calculate_max_drawdown(self, hist_df, position_value=None):
        """
        计算历史最大回撤
        
        参数:
        - hist_df: 包含历史价格数据的DataFrame
        - position_value: 头寸市值（用于计算回撤货币值）
        
        返回:
        - 字典，包含最大回撤百分比和货币值 {'percentage': float, 'amount': float}
        """
        try:
            if hist_df is None or hist_df.empty:
                return None
            
            # 计算累计收益
            cumulative = (1 + hist_df['Close'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            # 最大回撤（取绝对值，转换为正数）
            max_drawdown_percentage = abs(drawdown.min()) * 100
            
            # 计算回撤货币值
            max_drawdown_amount = None
            if position_value is not None and position_value > 0:
                max_drawdown_amount = position_value * (max_drawdown_percentage / 100)
            
            return {
                'percentage': max_drawdown_percentage,
                'amount': max_drawdown_amount
            }
        except Exception as e:
            print(f"⚠️ 计算最大回撤失败: {e}")
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

    def _get_tav_color(self, tav_score):
        """
        根据TAV评分返回对应的颜色样式
        """
        if tav_score is None:
            return "color: orange; font-weight: bold;"
        
        if tav_score >= 75:
            return "color: green; font-weight: bold;"
        elif tav_score >= 50:
            return "color: orange; font-weight: bold;"
        elif tav_score >= 25:
            return "color: red; font-weight: bold;"
        else:
            return "color: orange; font-weight: bold;"
    
    def _format_price_info(self, current_price=None, stop_loss_price=None, target_price=None, validity_period=None):
        """
        公用方法：格式化价格信息，确保数字类型正确转换
        
        Returns:
            dict: 包含格式化后的价格信息
        """
        price_info = ""
        stop_loss_info = ""
        target_price_info = ""
        validity_period_info = ""
        
        try:
            # 格式化当前价格
            if current_price is not None and pd.notna(current_price):
                price_info = f"现价: {float(current_price):.2f}"
            
            # 格式化止损价格
            if stop_loss_price is not None and pd.notna(stop_loss_price):
                stop_loss_info = f"止损价: {float(stop_loss_price):.2f}"
            
            # 格式化目标价格
            if target_price is not None and pd.notna(target_price):
                try:
                    # 确保target_price是数字类型
                    if isinstance(target_price, str) and target_price.strip():
                        target_price_float = float(target_price)
                        target_price_info = f"目标价: {target_price_float:.2f}"
                    else:
                        target_price_info = f"目标价: {float(target_price):.2f}"
                except (ValueError, TypeError):
                    target_price_info = f"目标价: {target_price}"
            
            # 格式化有效期
            if validity_period is not None and pd.notna(validity_period):
                try:
                    # 确保validity_period是数字类型
                    if isinstance(validity_period, str) and validity_period.strip():
                        validity_period_int = int(float(validity_period))
                        validity_period_info = f"有效期: {validity_period_int}天"
                    else:
                        validity_period_info = f"有效期: {int(validity_period)}天"
                except (ValueError, TypeError):
                    validity_period_info = f"有效期: {validity_period}"
            
        except Exception as e:
            print(f"⚠️ 格式化价格信息时出错: {e}")
        
        return {
            'price_info': price_info,
            'stop_loss_info': stop_loss_info,
            'target_price_info': target_price_info,
            'validity_period_info': validity_period_info
        }

    def _clean_signal_description(self, description):
        """
        清理信号描述，移除前缀
        """
        if not description:
            return description
        
        # 买入信号前缀
        buy_prefixes = ['买入信号:', '买入信号', 'Buy Signal:', 'Buy Signal']
        # 卖出信号前缀
        sell_prefixes = ['卖出信号:', '卖出信号', 'Sell Signal:', 'Sell Signal']
        
        cleaned = description
        for prefix in buy_prefixes + sell_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        return cleaned

    def _calculate_technical_indicators_core(self, data, asset_type='stock'):
        """
        计算技术指标的核心方法（支持不同资产类型）- 修复版本
        """
        try:
            if data is None:
                print("   ❌ data 是 None")
                return None

            hist = data.get('hist')
            if hist is None or hist.empty:
                print("   ❌ hist 是 None 或空的")
                return None

            from hsi_email import TECHNICAL_ANALYSIS_AVAILABLE
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
                # 使用TAV增强分析（如果可用）
                if self.use_tav:
                    indicators_df = self.technical_analyzer.calculate_all_indicators(hist.copy(), asset_type=asset_type)
                    indicators_with_signals = self.technical_analyzer.generate_buy_sell_signals(indicators_df.copy(), use_tav=True, asset_type=asset_type)
                else:
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
                        # 从描述中提取买入信号部分
                        desc = row.get('Signal_Description', '')
                        if '买入信号:' in desc and '卖出信号:' in desc:
                            # 如果同时有买入和卖出信号，只提取买入部分
                            buy_part = desc.split('买入信号:')[1].split('卖出信号:')[0].strip()
                            # 移除可能的结尾分隔符
                            if buy_part.endswith('|'):
                                buy_part = buy_part[:-1].strip()
                            buy_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': f"买入信号: {buy_part}"})
                        elif '买入信号:' in desc:
                            # 如果只有买入信号
                            description = self._clean_signal_description(desc)
                            buy_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': f"买入信号: {description}"})

                if 'Sell_Signal' in recent_signals.columns:
                    sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                    for idx, row in sell_signals_df.iterrows():
                        # 从描述中提取卖出信号部分
                        desc = row.get('Signal_Description', '')
                        if '买入信号:' in desc and '卖出信号:' in desc:
                            # 如果同时有买入和卖出信号，只提取卖出部分
                            sell_part = desc.split('卖出信号:')[1].strip()
                            sell_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': f"卖出信号: {sell_part}"})
                        elif '卖出信号:' in desc:
                            # 如果只有卖出信号
                            description = self._clean_signal_description(desc)
                            sell_signals.append({'date': idx.strftime('%Y-%m-%d'), 'description': f"卖出信号: {description}"})

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

                # 计算不同投资风格的VaR
                current_price = float(latest.get('Close', hist['Close'].iloc[-1]))
                var_ultra_short = self.calculate_var(hist, 'ultra_short_term', position_value=current_price)
                var_short = self.calculate_var(hist, 'short_term', position_value=current_price)
                var_medium_long = self.calculate_var(hist, 'medium_long_term', position_value=current_price)
                
                # 初始化指标字典
                indicators = {
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
                    'volume_ma20': volume_ma20,
                    'var_ultra_short_term': var_ultra_short['percentage'] if var_ultra_short else None,
                    'var_short_term': var_short['percentage'] if var_short else None,
                    'var_medium_long_term': var_medium_long['percentage'] if var_medium_long else None,
                    'var_ultra_short_term_amount': var_ultra_short['amount'] if var_ultra_short else None,
                    'var_short_term_amount': var_short['amount'] if var_short else None,
                    'var_medium_long_term_amount': var_medium_long['amount'] if var_medium_long else None
                }
                
                # 添加TAV分析信息（如果可用）
                if self.use_tav:
                    try:
                        tav_summary = self.technical_analyzer.get_tav_analysis_summary(indicators_with_signals, asset_type)
                        if tav_summary:
                            indicators['tav_score'] = tav_summary.get('tav_score', 0)
                            indicators['tav_status'] = tav_summary.get('tav_status', '无TAV')
                            indicators['tav_summary'] = tav_summary
                    except Exception as e:
                        print(f"⚠️ TAV分析失败: {e}")
                        indicators['tav_score'] = 0
                        indicators['tav_status'] = 'TAV分析失败'
                        indicators['tav_summary'] = None
                
                return indicators
                
            except Exception as e:
                print(f"⚠️ 计算技术指标失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 降级为简化计算
                if hist is not None and not hist.empty:
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

                    indicators = {
                        'rsi': self.calculate_rsi((latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0),
                        'macd': self.calculate_macd(latest['Close']),
                        'price_position': self.calculate_price_position(latest['Close'], hist['Close'].min(), hist['Close'].max()),
                        'atr': atr_value,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'recent_buy_signals': [],
                        'recent_sell_signals': [],
                        'trend': '数据不足',
                        'current_price': latest.get('Close', 0),
                        'ma20': 0,
                        'ma50': 0,
                        'ma200': 0,
                        'hist': hist
                    }
                    
                    # 添加TAV分析信息（降级模式）
                    if self.use_tav:
                        indicators['tav_score'] = 0
                        indicators['tav_status'] = 'TAV分析失败'
                        indicators['tav_summary'] = None
                    
                    return indicators
                else:
                    return None
        except Exception as e:
            print(f"❌ _calculate_technical_indicators_core 发生未捕获的异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_hsi_technical_indicators(self, data):
        """
        计算恒生指数技术指标（使用HSI专用配置）
        """
        return self._calculate_technical_indicators_core(data, asset_type='hsi')

    def calculate_technical_indicators(self, data):
        """
        计算技术指标（适用于个股）
        """
        return self._calculate_technical_indicators_core(data, asset_type='stock')

    def calculate_var(self, hist_df, investment_style='medium_term', confidence_level=0.95, position_value=None):
        """
        计算风险价值(VaR)，时间维度与投资周期匹配
        
        参数:
        - hist_df: 包含历史价格数据的DataFrame
        - investment_style: 投资风格
          - 'ultra_short_term': 超短线交易（日内/隔夜）
          - 'short_term': 波段交易（数天–数周）
          - 'medium_long_term': 中长期投资（1个月+）
        - confidence_level: 置信水平（默认0.95，即95%）
        - position_value: 头寸市值（用于计算VaR货币值）
        
        返回:
        - 字典，包含VaR百分比和货币值 {'percentage': float, 'amount': float}
        """
        try:
            if hist_df is None or hist_df.empty:
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
            if len(hist_df) < required_data:
                return None
            
            # 计算日收益率
            returns = hist_df['Close'].pct_change().dropna()
            
            if len(returns) < var_window:
                return None
            
            # 计算指定时间窗口的收益率
            if var_window == 1:
                # 1日VaR直接使用日收益率
                window_returns = returns
            else:
                # 多日VaR使用滚动收益率
                window_returns = hist_df['Close'].pct_change(var_window).dropna()
            
            # 使用历史模拟法计算VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(window_returns, var_percentile)
            
            # 返回绝对值（VaR通常表示为正数，表示最大可能损失）
            var_percentage = abs(var_value)
            
            # 计算VaR货币值
            var_amount = None
            if position_value is not None and position_value > 0:
                var_amount = position_value * var_percentage
            
            return {
                'percentage': var_percentage,
                'amount': var_amount
            }
        except Exception as e:
            print(f"⚠️ 计算VaR失败: {e}")
            return None

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

    def detect_continuous_signals_in_history_from_transactions(self, stock_code, hours=48, min_signals=3, target_date=None):
        """
        基于交易历史记录检测连续买卖信号（使用 pandas 读取 CSV）
        - stock_code: 股票代码
        - hours: 检测的时间范围（小时）
        - min_signals: 判定为连续信号的最小信号数量
        - target_date: 目标日期，如果为None则使用当前时间
        返回: 连续信号状态字符串
        """
        try:
            df = self._read_transactions_df()
            if df.empty:
                return "无交易记录"

            # 使用目标日期或当前时间
            if target_date is not None:
                # 将目标日期转换为带时区的时间戳
                if isinstance(target_date, str):
                    target_dt = pd.Timestamp(target_date).tz_localize('UTC')
                else:
                    target_dt = pd.Timestamp(target_date).tz_localize('UTC')
                # 设置为目标日期的收盘时间（16:00 UTC，对应香港时间24:00）
                reference_time = target_dt.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                reference_time = pd.Timestamp.now(tz='UTC')
            
            threshold = reference_time - pd.Timedelta(hours=hours)

            df_recent = df[(df['timestamp'] >= threshold) & (df['timestamp'] <= reference_time) & (df['code'] == stock_code)]
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

    def analyze_continuous_signals(self, target_date=None):
        """
        分析最近48小时内的连续买卖信号（使用 pandas 读取 data/simulation_transactions.csv）
        参数:
        - target_date: 目标日期，如果为None则使用当前时间
        返回: (buy_without_sell_after, sell_without_buy_after)
        每个元素为 (code, name, times_list, reasons_list, transactions_df)
        其中 transactions_df 是该股票的所有相关交易记录的DataFrame
        """
        df = self._read_transactions_df()
        if df.empty:
            return [], []

        # 使用目标日期或当前时间
        if target_date is not None:
            # 将目标日期转换为带时区的时间戳
            if isinstance(target_date, str):
                target_dt = pd.Timestamp(target_date).tz_localize('UTC')
            else:
                target_dt = pd.Timestamp(target_date).tz_localize('UTC')
            # 设置为目标日期的收盘时间（16:00 UTC，对应香港时间24:00）
            reference_time = target_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            reference_time = pd.Timestamp.now(tz='UTC')
        
        time_48_hours_ago = reference_time - pd.Timedelta(hours=48)
        df_recent = df[(df['timestamp'] >= time_48_hours_ago) & (df['timestamp'] <= reference_time)].copy()
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
                results_buy.append((code, name, times, reasons, buy_rows))
            elif len(sell_rows) >= 3 and len(buy_rows) == 0:
                name = sell_rows['name'].iloc[0] if 'name' in sell_rows.columns and len(sell_rows) > 0 else 'Unknown'
                times = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in sell_rows['timestamp'].tolist()]
                reasons = sell_rows['reason'].fillna('').tolist() if 'reason' in sell_rows.columns else [''] * len(times)
                results_sell.append((code, name, times, reasons, sell_rows))

        return results_buy, results_sell

    def calculate_expected_shortfall(self, hist_df, investment_style='short_term', confidence_level=0.95, position_value=None):
        """
        计算期望损失（Expected Shortfall, ES），用于评估极端风险和尾部风险
        
        ES是超过VaR阈值的所有损失的平均值，因此ES > VaR
        
        参数:
        - hist_df: 包含历史价格数据的DataFrame
        - investment_style: 投资风格
          - 'ultra_short_term': 超短线交易（日内/隔夜）
          - 'short_term': 波段交易（数天–数周）
          - 'medium_long_term': 中长期投资（1个月+）
        - confidence_level: 置信水平（默认0.95，即95%）
        - position_value: 头寸市值（用于计算ES货币值）
        
        返回:
        - 字典，包含ES百分比和货币值 {'percentage': float, 'amount': float}
        """
        try:
            if hist_df is None or hist_df.empty:
                return None
            
            # 根据投资风格确定ES计算的时间窗口
            if investment_style == 'ultra_short_term':
                # 超短线交易：1日ES
                es_window = 1
            elif investment_style == 'short_term':
                # 波段交易：5日ES
                es_window = 5
            elif investment_style == 'medium_long_term':
                # 中长期投资：20日ES（≈1个月）
                es_window = 20
            else:
                # 默认使用5日ES
                es_window = 5
            
            # 确保有足够的历史数据
            required_data = max(es_window * 5, 30)  # 至少需要5倍时间窗口或30天的数据
            if len(hist_df) < required_data:
                return None
            
            # 计算指定时间窗口的收益率
            if es_window == 1:
                # 1日ES直接使用日收益率
                window_returns = hist_df['Close'].pct_change().dropna()
            else:
                # 多日ES使用滚动收益率
                window_returns = hist_df['Close'].pct_change(es_window).dropna()
            
            if len(window_returns) == 0:
                return None
            
            # 计算VaR（作为ES的基准）
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(window_returns, var_percentile)
            
            # 计算ES：所有小于等于VaR的收益率的平均值
            tail_losses = window_returns[window_returns <= var_value]
            
            if len(tail_losses) == 0:
                return abs(var_value) * 100  # 如果没有尾部数据，返回VaR值
            
            es_value = tail_losses.mean()
            
            # 返回绝对值（ES通常表示为正数，表示损失）
            es_percentage = abs(es_value) * 100
            
            # 计算ES货币值
            es_amount = None
            if position_value is not None and position_value > 0:
                es_amount = position_value * (es_percentage / 100)
            
            return {
                'percentage': es_percentage,
                'amount': es_amount
            }
            
        except Exception as e:
            print(f"⚠️ 计算期望损失失败: {e}")
            return None

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

    def generate_stock_analysis_html(self, stock_data, indicators, continuous_buy_signals=None, continuous_sell_signals=None, target_date=None):
        """为单只股票生成HTML分析部分"""
        if not indicators:
            return ""
        
        # 获取历史数据
        hist_data = self.get_stock_data(stock_data['symbol'], target_date=target_date)

        continuous_signal_info = None
        transactions_df_for_stock = None
        if continuous_buy_signals is not None:
            for code, name, times, reasons, transactions_df in continuous_buy_signals:
                if code == stock_data['symbol']:
                    continuous_signal_info = f"连续买入({len(times)}次)"
                    transactions_df_for_stock = transactions_df
                    break
        if continuous_signal_info is None and continuous_sell_signals is not None:
            for code, name, times, reasons, transactions_df in continuous_sell_signals:
                if code == stock_data['symbol']:
                    continuous_signal_info = f"连续卖出({len(times)}次)"
                    transactions_df_for_stock = transactions_df
                    break

        # 从交易记录中获取最新的止损价和目标价
        latest_stop_loss = None
        latest_target_price = None
        if transactions_df_for_stock is not None and not transactions_df_for_stock.empty:
            # 按时间排序，获取最新的交易记录
            transactions_df_for_stock = transactions_df_for_stock.sort_values('timestamp')
            latest_transaction = transactions_df_for_stock.iloc[-1]
            latest_stop_loss = latest_transaction.get('stop_loss_price')
            latest_target_price = latest_transaction.get('target_price')

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
        
        # 从交易记录中获取最新的止损价和目标价
        try:
            df_transactions = self._read_transactions_df()
            if not df_transactions.empty:
                stock_transactions = df_transactions[df_transactions['code'] == stock_data['symbol']]
                if not stock_transactions.empty:
                    # 按时间排序，获取最新的交易记录
                    stock_transactions = stock_transactions.sort_values('timestamp')
                    latest_transaction = stock_transactions.iloc[-1]
                    latest_stop_loss = latest_transaction.get('stop_loss_price')
                    latest_target_price = latest_transaction.get('target_price')
                else:
                    latest_stop_loss = None
                    latest_target_price = None
            else:
                latest_stop_loss = None
                latest_target_price = None
        except Exception as e:
            print(f"⚠️ 获取交易记录失败: {e}")
            latest_stop_loss = None
            latest_target_price = None

        # 为趋势设置颜色
        if "多头" in trend:
            trend_color_style = "color: green; font-weight: bold;"
        elif "空头" in trend:
            trend_color_style = "color: red; font-weight: bold;"
        elif "震荡" in trend:
            trend_color_style = "color: orange; font-weight: bold;"
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

        # 添加VaR信息
        var_ultra_short = indicators.get('var_ultra_short_term')
        var_ultra_short_amount = indicators.get('var_ultra_short_term_amount')
        var_short = indicators.get('var_short_term')
        var_short_amount = indicators.get('var_short_term_amount')
        var_medium_long = indicators.get('var_medium_long_term')
        var_medium_long_amount = indicators.get('var_medium_long_term_amount')
        
        if var_ultra_short is not None:
            var_amount_display = f" (¥{var_ultra_short_amount:.2f})" if var_ultra_short_amount is not None else ""
            html += f"""
                <tr>
                    <td>1日VaR (95%)</td>
                    <td>{var_ultra_short:.2%}{var_amount_display}</td>
                </tr>
            """
        
        if var_short is not None:
            var_amount_display = f" (¥{var_short_amount:.2f})" if var_short_amount is not None else ""
            html += f"""
                <tr>
                    <td>5日VaR (95%)</td>
                    <td>{var_short:.2%}{var_amount_display}</td>
                </tr>
            """
        
        if var_medium_long is not None:
            var_amount_display = f" (¥{var_medium_long_amount:.2f})" if var_medium_long_amount is not None else ""
            html += f"""
                <tr>
                    <td>20日VaR (95%)</td>
                    <td>{var_medium_long:.2%}{var_amount_display}</td>
                </tr>
            """
        
        # 添加ES信息（如果可用）
        if stock_data['symbol'] != 'HSI':
            # 使用已经根据target_date过滤的历史数据计算ES
            if hist_data is not None and not hist_data.get('hist', pd.DataFrame()).empty:
                hist = hist_data['hist']
                # 计算各时间窗口的ES
                current_price = float(stock_data['current_price'])
                es_1d = self.calculate_expected_shortfall(hist, 'ultra_short_term', position_value=current_price)
                es_5d = self.calculate_expected_shortfall(hist, 'short_term', position_value=current_price)
                es_20d = self.calculate_expected_shortfall(hist, 'medium_long_term', position_value=current_price)
                
                if es_1d is not None:
                    es_1d_percentage = es_1d['percentage'] / 100 if es_1d else None
                    es_1d_amount = es_1d['amount'] if es_1d else None
                    es_amount_display = f" (¥{es_1d_amount:.2f})" if es_1d_amount is not None else ""
                    html += f"""
                        <tr>
                            <td>1日ES (95%)</td>
                            <td>{es_1d_percentage:.2%}{es_amount_display}</td>
                        </tr>
                    """
                
                if es_5d is not None:
                    es_5d_percentage = es_5d['percentage'] / 100 if es_5d else None
                    es_5d_amount = es_5d['amount'] if es_5d else None
                    es_amount_display = f" (¥{es_5d_amount:.2f})" if es_5d_amount is not None else ""
                    html += f"""
                        <tr>
                            <td>5日ES (95%)</td>
                            <td>{es_5d_percentage:.2%}{es_amount_display}</td>
                        </tr>
                    """
                
                if es_20d is not None:
                    es_20d_percentage = es_20d['percentage'] / 100 if es_20d else None
                    es_20d_amount = es_20d['amount'] if es_20d else None
                    es_amount_display = f" (¥{es_20d_amount:.2f})" if es_20d_amount is not None else ""
                    html += f"""
                        <tr>
                            <td>20日ES (95%)</td>
                            <td>{es_20d_percentage:.2%}{es_amount_display}</td>
                        </tr>
                    """

        # 添加TAV信息（如果可用）
        tav_score = indicators.get('tav_score', None)
        tav_status = indicators.get('tav_status', '无TAV')
        tav_summary = indicators.get('tav_summary', None)
        
        if tav_score is not None:
            # TAV评分颜色
            tav_color = self._get_tav_color(tav_score)
            
            html += f"""
                <tr>
                    <td>TAV评分</td>
                    <td><span style="{tav_color}">{tav_score:.1f}</span> <span style="font-size: 0.8em; color: #666;">({tav_status})</span></td>
                </tr>
            """
            
            # 如果有TAV详细分析，添加详细信息
            if tav_summary:
                trend_analysis = tav_summary.get('trend_analysis', 'N/A')
                momentum_analysis = tav_summary.get('momentum_analysis', 'N/A')
                volume_analysis = tav_summary.get('volume_analysis', 'N/A')
                recommendation = tav_summary.get('recommendation', 'N/A')
                
                # 直接显示TAV详细分析内容，兼容所有邮件客户端
                html += f"""
                <tr>
                    <td colspan="2">
                        <div style="margin-top: 15px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; font-size: 0.9em; border-left: 4px solid #ff9800; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="margin-bottom: 8px; font-weight: bold; color: #000; font-size: 1.1em;">📊 TAV详细分析</div>
                            <div style="margin-bottom: 8px;"><strong style="color: #333;">趋势分析:</strong> {trend_analysis}</div>
                            <div style="margin-bottom: 8px;"><strong style="color: #333;">动量分析:</strong> {momentum_analysis}</div>
                            <div style="margin-bottom: 8px;"><strong style="color: #333;">成交量分析:</strong> {volume_analysis}</div>
                            <div><strong style="color: #333;">TAV建议:</strong> {recommendation}</div>
                        </div>
                    </td>
                </tr>
                """
            else:
                # 调试信息
                print(f"⚠️ 股票 {stock_data['name']} ({stock_data['symbol']}) 没有TAV摘要")

        # 只使用交易记录中的止损价和目标价
        if latest_stop_loss is not None and pd.notna(latest_stop_loss):
            try:
                stop_loss_float = float(latest_stop_loss)
                html += f"""
                <tr>
                    <td>建议止损价</td>
                    <td>{stop_loss_float:,.2f}</td>
                </tr>
            """
            except (ValueError, TypeError):
                pass

        if latest_target_price is not None and pd.notna(latest_target_price):
            try:
                target_price_float = float(latest_target_price)
                html += f"""
                <tr>
                    <td>建议止盈价</td>
                    <td>{target_price_float:,.2f}</td>
                </tr>
            """
            except (ValueError, TypeError):
                pass

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
            # 根据连续信号内容设置颜色
            if "买入" in continuous_signal_info:
                signal_color = "green"
            elif "卖出" in continuous_signal_info:
                signal_color = "red"
            else:
                signal_color = "orange"
                
            html += f"""
            <tr>
                <td colspan="2">
                    <div class="continuous-signal">
                        <strong>🤖 48小时智能建议:</strong><br>
                        <span style='color: {signal_color};'>• {continuous_signal_info}</span>
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
        # 获取股息信息
        print("📊 获取即将除净的港股信息...")
        dividend_data = self.get_upcoming_dividends(days_ahead=90)
        
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
                            continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(stock_code, target_date=target_date)
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

        target_date_signals.sort(key=lambda x: x[1])

        # 文本版表头（修复原先被截断的 f-string）
        text_lines = []
        
        # 添加股息信息到文本
        dividend_text = self.format_dividend_table_text(dividend_data)
        if dividend_text:
            text_lines.append(dividend_text)
        
        text_lines.append("🔔 交易信号总结:")
        header = f"{'股票名称':<15} {'股票代码':<10} {'趋势(技术分析)':<12} {'信号类型':<8} {'48小时智能建议':<20} {'信号描述':<30} {'TAV评分':<8} {'股票现价':<10} {'5日VaR':<8} {'20日VaR':<8} {'5日ES':<8} {'20日ES':<8} {'历史回撤':<10} {'风险评估':<6}"
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

        # 添加股息信息到HTML
        dividend_html = self.format_dividend_table_html(dividend_data)
        if dividend_html:
            html += dividend_html

        html += f"""
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
                        <th>TAV评分</th>
                        <th>股票现价</th>
                        <th>5日VaR(95%)</th>
                        <th>20日VaR(95%)</th>
                        <th>5日ES(95%)</th>
                        <th>20日ES(95%)</th>
                        <th>历史回撤</th>
                        <th>风险评估</th>
                    </tr>
        """

        for stock_name, stock_code, trend, signal, signal_type in target_date_signals:
            signal_display = f"{signal_type}信号"
            color_style = "color: green; font-weight: bold;" if signal_type == '买入' else "color: red; font-weight: bold;"
            continuous_signal_status = "无信号"
            if stock_code != 'HSI':
                continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(stock_code, target_date=target_date)

            # 智能过滤：保留有量价信号或有48小时智能建议的股票
            should_show = (signal_type in ['买入', '卖出']) or (continuous_signal_status != "无建议信号")
            
            if not should_show:
                continue
            
            # 为无量价信号但有48小时建议的股票创建特殊显示
            if signal_type not in ['买入', '卖出'] and continuous_signal_status != "无建议信号":
                signal_display = "无量价信号"
                color_style = "color: orange; font-weight: bold;"
                signal_description = f"仅48小时智能建议: {continuous_signal_status}"
            else:
                signal_description = signal.get('description', '') if isinstance(signal, dict) else (str(signal) if signal is not None else '')

            # 为48小时智能建议设置颜色
            if "买入" in continuous_signal_status:
                signal_color_style = "color: green; font-weight: bold;"
            elif "卖出" in continuous_signal_status:
                signal_color_style = "color: red; font-weight: bold;"
            elif "无建议信号" in continuous_signal_status:
                signal_color_style = "color: orange; font-weight: bold;"
            else:
                signal_color_style = ""
            
            # 为趋势设置颜色
            if "多头" in trend:
                trend_color_style = "color: green; font-weight: bold;"
            elif "空头" in trend:
                trend_color_style = "color: red; font-weight: bold;"
            elif "震荡" in trend:
                trend_color_style = "color: orange; font-weight: bold;"
            else:
                trend_color_style = ""
            
            # 判断三列颜色是否相同，如果相同则股票名称也使用相同颜色
            name_color_style = ""
            if trend_color_style == color_style == signal_color_style and trend_color_style != "":
                name_color_style = trend_color_style
            
            # 获取TAV评分信息和VaR值
            tav_score = None
            tav_status = None
            tav_color = "color: orange; font-weight: bold;"  # 默认颜色
            var_ultra_short = None
            var_short = None
            var_medium_long = None
            es_short = None
            es_medium_long = None
            max_drawdown = None            
            if stock_code != 'HSI':
                # stock_results是列表，需要查找匹配的股票代码
                stock_indicators = None
                for stock_result in stock_results:
                    if stock_result.get('code') == stock_code:
                        stock_indicators = stock_result.get('indicators', {})
                        break
                
                if stock_indicators:
                    tav_score = stock_indicators.get('tav_score', 0)
                    tav_status = stock_indicators.get('tav_status', '无TAV')
                    var_ultra_short = stock_indicators.get('var_ultra_short_term')
                    var_short = stock_indicators.get('var_short_term')
                    var_medium_long = stock_indicators.get('var_medium_long_term')
                    
                    # 计算ES值和回撤
                    hist_data = self.get_stock_data(stock_code, target_date=target_date)
                    if hist_data is not None:
                        # 使用已经根据target_date过滤的历史数据
                        hist = hist_data['hist']
                        if not hist.empty:
                            current_price = float(hist_data['current_price'])
                            es_short = self.calculate_expected_shortfall(hist, 'short_term', position_value=current_price)
                            es_medium_long = self.calculate_expected_shortfall(hist, 'medium_long_term', position_value=current_price)
                            # 计算历史最大回撤
                            max_drawdown = self.calculate_max_drawdown(hist, position_value=current_price)
                            
                            # 风险评估
                            risk_assessment = "正常"
                            if max_drawdown is not None and es_medium_long is not None:
                                # 将ES和回撤转换为小数进行比较
                                es_decimal = es_medium_long['percentage'] / 100 if isinstance(es_medium_long, dict) else es_medium_long / 100
                                max_dd_decimal = max_drawdown['percentage'] / 100 if isinstance(max_drawdown, dict) else max_drawdown / 100
                                
                                if es_decimal < max_dd_decimal / 3:
                                    risk_assessment = "优秀"
                                elif es_decimal > max_dd_decimal / 2:
                                    risk_assessment = "警示"
                                else:
                                    risk_assessment = "合理"
                
                # TAV评分颜色
                tav_color = self._get_tav_color(tav_score)
            
            # 确保所有变量都有默认值，避免格式化错误
            safe_name = stock_name if stock_name is not None else 'N/A'
            safe_code = stock_code if stock_code is not None else 'N/A'
            safe_trend = trend if trend is not None else 'N/A'
            safe_signal_display = signal_display if signal_display is not None else 'N/A'
            safe_tav_score = tav_score if tav_score is not None else 'N/A'
            safe_tav_status = tav_status if tav_status is not None else '无TAV'
            safe_continuous_signal_status = continuous_signal_status if continuous_signal_status is not None else 'N/A'
            safe_signal_description = signal_description if signal_description is not None else 'N/A'
            
            # 格式化VaR值和ES值
            var_ultra_short_display = f"{var_ultra_short:.2%}" if var_ultra_short is not None else "N/A"
            var_short_display = f"{var_short:.2%}" if var_short is not None else "N/A"
            var_medium_long_display = f"{var_medium_long:.2%}" if var_medium_long is not None else "N/A"
            
            # 获取VaR货币值
            var_ultra_short_amount = stock_indicators.get('var_ultra_short_term_amount')
            var_short_amount = stock_indicators.get('var_short_term_amount')
            var_medium_long_amount = stock_indicators.get('var_medium_long_term_amount')
            
            # 添加货币值显示
            if var_ultra_short is not None and var_ultra_short_amount is not None:
                var_ultra_short_display += f" (¥{var_ultra_short_amount:.2f})"
            if var_short is not None and var_short_amount is not None:
                var_short_display += f" (¥{var_short_amount:.2f})"
            if var_medium_long is not None and var_medium_long_amount is not None:
                var_medium_long_display += f" (¥{var_medium_long_amount:.2f})"
            # 格式化ES值
            es_short_display = f"{es_short['percentage']/100:.2%}" if es_short is not None else "N/A"
            es_medium_long_display = f"{es_medium_long['percentage']/100:.2%}" if es_medium_long is not None else "N/A"
            
            # 添加ES货币值显示
            if es_short is not None and es_short.get('amount') is not None:
                es_short_display += f" (¥{es_short['amount']:.2f})"
            if es_medium_long is not None and es_medium_long.get('amount') is not None:
                es_medium_long_display += f" (¥{es_medium_long['amount']:.2f})"
            
            # 格式化回撤和风险评估
            max_drawdown_display = f"{max_drawdown['percentage']/100:.2%}" if max_drawdown is not None else "N/A"
            
            # 添加回撤货币值显示
            if max_drawdown is not None and max_drawdown.get('amount') is not None:
                max_drawdown_display += f" (¥{max_drawdown['amount']:.2f})"
            risk_color = ""
            if risk_assessment == "优秀":
                risk_color = "color: green; font-weight: bold;"
            elif risk_assessment == "警示":
                risk_color = "color: red; font-weight: bold;"
            else:
                risk_color = "color: orange; font-weight: bold;"
            
            # 准备价格显示和TAV评分显示
            price_display = hist_data['current_price'] if hist_data is not None else None
            tav_score_display = f"{safe_tav_score:.1f}" if isinstance(safe_tav_score, (int, float)) else "N/A"
            price_value_display = f"{price_display:.2f}" if price_display is not None else "N/A"
            
            html += f"""
                    <tr>
                        <td><span style=\"{name_color_style}\">{safe_name}</span></td>
                        <td>{safe_code}</td>
                        <td><span style=\"{trend_color_style}\">{safe_trend}</span></td>
                        <td><span style=\"{color_style}\">{safe_signal_display}</span></td>
                        <td><span style=\"{signal_color_style}\">{safe_continuous_signal_status}</span></td>
                        <td>{safe_signal_description}</td>
                        <td><span style=\"{tav_color}\">{tav_score_display}</span> <span style=\"font-size: 0.8em; color: #666;\">({safe_tav_status})</span></td>
                        <td>{price_value_display}</td>
                        <td>{var_short_display}</td>
                        <td>{var_medium_long_display}</td>
                        <td>{es_short_display}</td>
                        <td>{es_medium_long_display}</td>
                        <td>{max_drawdown_display}</td>
                        <td><span style=\"{risk_color}\">{risk_assessment}</span></td>
                    </tr>
            """

            # 文本版本追加
            tav_display = f"{tav_score:.1f}" if tav_score is not None else "N/A"
            var_ultra_short_display = f"{var_ultra_short:.2%}" if var_ultra_short is not None else "N/A"
            var_short_display = f"{var_short:.2%}" if var_short is not None else "N/A"
            var_medium_long_display = f"{var_medium_long:.2%}" if var_medium_long is not None else "N/A"
            
            # 添加货币值显示
            if var_ultra_short is not None and var_ultra_short_amount is not None:
                var_ultra_short_display += f" (¥{var_ultra_short_amount:.2f})"
            if var_short is not None and var_short_amount is not None:
                var_short_display += f" (¥{var_short_amount:.2f})"
            if var_medium_long is not None and var_medium_long_amount is not None:
                var_medium_long_display += f" (¥{var_medium_long_amount:.2f})"
            # 格式化ES值
            es_short_display = f"{es_short['percentage']/100:.2%}" if es_short is not None else "N/A"
            es_medium_long_display = f"{es_medium_long['percentage']/100:.2%}" if es_medium_long is not None else "N/A"
            
            # 添加ES货币值显示
            if es_short is not None and es_short.get('amount') is not None:
                es_short_display += f" (¥{es_short['amount']:.2f})"
            if es_medium_long is not None and es_medium_long.get('amount') is not None:
                es_medium_long_display += f" (¥{es_medium_long['amount']:.2f})"
            # 添加股票现价显示
            price_value = hist_data['current_price'] if hist_data is not None else None
            price_display = f"{price_value:.2f}" if price_value is not None else 'N/A'
            text_lines.append(f"{stock_name:<15} {stock_code:<10} {trend:<12} {signal_display:<8} {continuous_signal_status:<20} {signal_description:<30} {tav_display:<8} {price_display:<10} {var_short_display:<8} {var_medium_long_display:<8} {es_short_display:<8} {es_medium_long_display:<8} {max_drawdown_display:<10} {risk_assessment:<6}")

        # 检查过滤后是否有信号（使用新的过滤逻辑）
        has_filtered_signals = any(True for stock_name, stock_code, trend, signal, signal_type in target_date_signals
                                   if (signal_type in ['买入', '卖出']) or (self.detect_continuous_signals_in_history_from_transactions(stock_code, target_date=target_date) != "无建议信号"))

        if not has_filtered_signals:
            html += """
                    <tr>
                        <td colspan="15">当前没有检测到任何有效的交易信号（已过滤无信号股票）</td>
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
        buy_without_sell_after, sell_without_buy_after = self.analyze_continuous_signals(target_date)
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
                            <th>建议时间、现价、目标价、止损价、有效期</th>
                        </tr>
                """
                for code, name, times, reasons, transactions_df in buy_without_sell_after:
                    combined_str = ""
                    # 确保交易记录按时间排序
                    transactions_df = transactions_df.sort_values('timestamp')
                    for i in range(len(times)):
                        time_info = f"{times[i]}"
                        price_info = ""
                        stop_loss_info = ""
                        target_price_info = ""
                        validity_period_info = ""
                        
                        # 从交易记录中获取现价、止损价、目标价格和有效期
                        if i < len(transactions_df):
                            transaction = transactions_df.iloc[i]
                            current_price = transaction.get('current_price')
                            stop_loss_price = transaction.get('stop_loss_price')
                            target_price = transaction.get('target_price')
                            validity_period = transaction.get('validity_period')
                            
                            # 使用公用的格式化方法
                            price_data = self._format_price_info(current_price, stop_loss_price, target_price, validity_period)
                            price_info = price_data['price_info']
                            stop_loss_info = price_data['stop_loss_info']
                            target_price_info = price_data['target_price_info']
                            validity_period_info = price_data['validity_period_info']
                        
                        info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
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
                            <th>建议时间、现价、目标价、止损价、有效期</th>
                        </tr>
                """
                for code, name, times, reasons, transactions_df in sell_without_buy_after:
                    combined_str = ""
                    # 确保交易记录按时间排序
                    transactions_df = transactions_df.sort_values('timestamp')
                    for i in range(len(times)):
                        time_info = f"{times[i]}"
                        price_info = ""
                        stop_loss_info = ""
                        target_price_info = ""
                        validity_period_info = ""
                        
                        # 从交易记录中获取现价、止损价、目标价格和有效期
                        if i < len(transactions_df):
                            transaction = transactions_df.iloc[i]
                            current_price = transaction.get('current_price')
                            stop_loss_price = transaction.get('stop_loss_price')
                            target_price = transaction.get('target_price')
                            validity_period = transaction.get('validity_period')
                            
                            # 使用公用的格式化方法
                            price_data = self._format_price_info(current_price, stop_loss_price, target_price, validity_period)
                            price_info = price_data['price_info']
                            stop_loss_info = price_data['stop_loss_info']
                            target_price_info = price_data['target_price_info']
                            validity_period_info = price_data['validity_period_info']
                        
                        info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
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
            for code, name, times, reasons, transactions_df in buy_without_sell_after:
                combined_list = []
                # 确保交易记录按时间排序
                transactions_df = transactions_df.sort_values('timestamp')
                for i in range(len(times)):
                    time_info = f"{times[i]}"
                    price_info = ""
                    stop_loss_info = ""
                    target_price_info = ""
                    validity_period_info = ""
                    
                    # 从交易记录中获取现价、止损价、目标价格和有效期
                    if i < len(transactions_df):
                        transaction = transactions_df.iloc[i]
                        current_price = transaction.get('current_price')
                        stop_loss_price = transaction.get('stop_loss_price')
                        target_price = transaction.get('target_price')
                        validity_period = transaction.get('validity_period')
                        
                                                                # 使用公用的格式化方法
                        price_data = self._format_price_info(current_price, stop_loss_price, target_price, validity_period)
                        price_info = price_data['price_info']
                        stop_loss_info = price_data['stop_loss_info']
                        target_price_info = price_data['target_price_info']
                        validity_period_info = price_data['validity_period_info']
                    
                    info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
                    reason_info = ", ".join(info_parts)
                    combined_item = f"{time_info} {reason_info}".strip()
                    combined_list.append(combined_item)
                combined_str = "\n    ".join(combined_list)
                text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
            text += "\n"

        if sell_without_buy_after:
            text += f"📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）:\n"
            for code, name, times, reasons, transactions_df in sell_without_buy_after:
                combined_list = []
                # 确保交易记录按时间排序
                transactions_df = transactions_df.sort_values('timestamp')
                for i in range(len(times)):
                    time_info = f"{times[i]}"
                    price_info = ""
                    stop_loss_info = ""
                    target_price_info = ""
                    validity_period_info = ""
                    
                    # 从交易记录中获取现价、止损价、目标价格和有效期
                    if i < len(transactions_df):
                        transaction = transactions_df.iloc[i]
                        current_price = transaction.get('current_price')
                        stop_loss_price = transaction.get('stop_loss_price')
                        target_price = transaction.get('target_price')
                        validity_period = transaction.get('validity_period')
                        
                        # 使用公用的格式化方法
                        price_data = self._format_price_info(current_price, stop_loss_price, target_price, validity_period)
                        price_info = price_data['price_info']
                        stop_loss_info = price_data['stop_loss_info']
                        target_price_info = price_data['target_price_info']
                        validity_period_info = price_data['validity_period_info']                    
                    info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
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
                # 使用目标日期或当前时间
                if target_date is not None:
                    # 将目标日期转换为带时区的时间戳
                    if isinstance(target_date, str):
                        target_dt = pd.Timestamp(target_date).tz_localize('UTC')
                    else:
                        target_dt = pd.Timestamp(target_date).tz_localize('UTC')
                    # 设置为目标日期的收盘时间（16:00 UTC，对应香港时间24:00）
                    reference_time = target_dt.replace(hour=16, minute=0, second=0, microsecond=0)
                else:
                    reference_time = pd.Timestamp.now(tz='UTC')
                
                time_48_hours_ago = reference_time - pd.Timedelta(hours=48)
                df_recent = df_all[(df_all['timestamp'] >= time_48_hours_ago) & (df_all['timestamp'] <= reference_time)].copy()
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
                            <th>目标价</th>
                            <th>止损价</th>
                            <th>有效期</th>
                            <th>理由</th>
                        </tr>
                    """
                    for _, trans in df_recent.iterrows():
                        trans_type = trans.get('type', '')
                        row_style = "background-color: #e8f5e9;" if 'BUY' in str(trans_type).upper() else "background-color: #ffebee;"
                        # 设置交易类型的颜色
                        if 'BUY' in str(trans_type).upper():
                            trans_type_style = "color: green; font-weight: bold;"
                        elif 'SELL' in str(trans_type).upper():
                            trans_type_style = "color: red; font-weight: bold;"
                        else:
                            trans_type_style = ""
                        price = trans.get('current_price', np.nan)
                        price_display = f"{price:,.2f}" if not pd.isna(price) else (trans.get('price', '') or '')
                        reason = trans.get('reason', '') or ''
                        
                        # 使用公用的格式化方法获取价格信息
                        price_data = self._format_price_info(
                            trans.get('current_price', np.nan),
                            trans.get('stop_loss_price', np.nan),
                            trans.get('target_price', np.nan),
                            trans.get('validity_period', np.nan)
                        )
                        
                        # 格式化显示
                        stop_loss_display = price_data['stop_loss_info'].replace('止损价: ', '') if price_data['stop_loss_info'] else ''
                        target_price_display = price_data['target_price_info'].replace('目标价: ', '') if price_data['target_price_info'] else ''
                        validity_period_display = price_data['validity_period_info'].replace('有效期: ', '') if price_data['validity_period_info'] else ''
                        
                        html += f"""
                        <tr style="{row_style}">
                            <td>{trans.get('name','')}</td>
                            <td>{trans.get('code','')}</td>
                            <td>{pd.Timestamp(trans['timestamp']).strftime('%m-%d %H:%M:%S')}</td>
                            <td><span style="{trans_type_style}">{trans_type}</span></td>
                            <td>{price_display}</td>
                            <td>{target_price_display}</td>
                            <td>{stop_loss_display}</td>
                            <td>{validity_period_display}</td>
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
                            
                            # 使用公用的格式化方法获取价格信息
                            price_data = self._format_price_info(
                                tr.get('current_price', np.nan),
                                tr.get('stop_loss_price', np.nan),
                                tr.get('target_price', np.nan),
                                tr.get('validity_period', np.nan)
                            )
                            
                            # 格式化显示
                            stop_loss_display = price_data['stop_loss_info'].replace('止损价: ', '') if price_data['stop_loss_info'] else ''
                            target_price_display = price_data['target_price_info'].replace('目标价: ', '') if price_data['target_price_info'] else ''
                            validity_period_display = price_data['validity_period_info'].replace('有效期: ', '') if price_data['validity_period_info'] else ''
                            
                            
                            
                            # 构建额外的价格信息
                            price_info = []
                            if target_price_display:
                                price_info.append(f"目标:{target_price_display}")
                            if stop_loss_display:
                                price_info.append(f"止损:{stop_loss_display}")
                            if validity_period_display:
                                price_info.append(f"有效期:{validity_period_display}")
                            
                            
                            
                            price_info_str = " | ".join(price_info) if price_info else ""
                            
                            if price_info_str:
                                text += f"    {timestamp} {trans_type} @ {price_display} ({price_info_str}) ({reason})\n"
                            else:
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
                    hsi_trend_color_style = "color: orange; font-weight: bold;"
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
                html += self.generate_stock_analysis_html(stock_data, indicators, buy_without_sell_after, sell_without_buy_after, target_date)
                
                # HTML版本：添加分割线
                html += f"""
                <tr>
                    <td colspan=\"2\" style=\"padding: 0;\"><hr style=\"border: 1px solid #e0e0e0; margin: 15px 0;\"></td>
                </tr>
                """

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
                
                # 从交易记录中获取最新的止损价和目标价
                latest_stop_loss = None
                latest_target_price = None
                try:
                    df_transactions = self._read_transactions_df()
                    if not df_transactions.empty:
                        stock_transactions = df_transactions[df_transactions['code'] == stock_result['code']]
                        if not stock_transactions.empty:
                            # 按时间排序，获取最新的交易记录
                            stock_transactions = stock_transactions.sort_values('timestamp')
                            latest_transaction = stock_transactions.iloc[-1]
                            latest_stop_loss = latest_transaction.get('stop_loss_price')
                            latest_target_price = latest_transaction.get('target_price')
                except Exception as e:
                    print(f"⚠️ 获取交易记录失败: {e}")

                text += f"  趋势(技术分析): {trend}\n"
                text += f"  RSI: {rsi:.2f}\n"
                text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
                text += f"  布林带位置: {bb_position:.2f}\n"
                text += f"  MA20: {ma20:,.2f}\n"
                text += f"  MA50: {ma50:,.2f}\n"
                text += f"  MA200: {ma200:,.2f}\n"
                text += f"  ATR: {atr:.2f}\n"
                
                # 添加VaR信息
                var_ultra_short = indicators.get('var_ultra_short_term')
                var_ultra_short_amount = indicators.get('var_ultra_short_term_amount')
                var_short = indicators.get('var_short_term')
                var_short_amount = indicators.get('var_short_term_amount')
                var_medium_long = indicators.get('var_medium_long_term')
                var_medium_long_amount = indicators.get('var_medium_long_term_amount')
                
                if var_ultra_short is not None:
                    amount_display = f" (¥{var_ultra_short_amount:.2f})" if var_ultra_short_amount is not None else ""
                    text += f"  1日VaR (95%): {var_ultra_short:.2%}{amount_display}\n"
                
                if var_short is not None:
                    amount_display = f" (¥{var_short_amount:.2f})" if var_short_amount is not None else ""
                    text += f"  5日VaR (95%): {var_short:.2%}{amount_display}\n"
                
                if var_medium_long is not None:
                    amount_display = f" (¥{var_medium_long_amount:.2f})" if var_medium_long_amount is not None else ""
                    text += f"  20日VaR (95%): {var_medium_long:.2%}{amount_display}\n"
                
                # 计算并显示ES值
                if stock_result['code'] != 'HSI':
                    # 使用已经根据target_date过滤的历史数据计算ES
                    hist = stock_result.get('data', {}).get('hist', pd.DataFrame())
                    if not hist.empty:
                        # 计算各时间窗口的ES
                        indicators = stock_result.get('indicators', {})
                        current_price = float(indicators.get('current_price', 0))
                        es_1d = self.calculate_expected_shortfall(hist, 'ultra_short_term', position_value=current_price)
                        es_5d = self.calculate_expected_shortfall(hist, 'short_term', position_value=current_price)
                        es_20d = self.calculate_expected_shortfall(hist, 'medium_long_term', position_value=current_price)
                        
                        if es_1d is not None:
                            amount_display = f" (¥{es_1d['amount']:.2f})" if es_1d.get('amount') is not None else ""
                            text += f"  1日ES (95%): {es_1d['percentage']/100:.2%}{amount_display}\n"
                        if es_5d is not None:
                            amount_display = f" (¥{es_5d['amount']:.2f})" if es_5d.get('amount') is not None else ""
                            text += f"  5日ES (95%): {es_5d['percentage']/100:.2%}{amount_display}\n"
                        if es_20d is not None:
                            amount_display = f" (¥{es_20d['amount']:.2f})" if es_20d.get('amount') is not None else ""
                            text += f"  20日ES (95%): {es_20d['percentage']/100:.2%}{amount_display}\n"

                if latest_stop_loss is not None and pd.notna(latest_stop_loss):
                    try:
                        stop_loss_float = float(latest_stop_loss)
                        text += f"  建议止损价: {stop_loss_float:,.2f}\n"
                    except (ValueError, TypeError):
                        pass
                if latest_target_price is not None and pd.notna(latest_target_price):
                    try:
                        target_price_float = float(latest_target_price)
                        text += f"  建议止盈价: {target_price_float:,.2f}\n"
                    except (ValueError, TypeError):
                        pass

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
                for code, name, times, reasons, transactions_df in buy_without_sell_after:
                    if code == stock_result['code']:
                        continuous_signal_info = f"连续买入({len(times)}次)"
                        break
                if continuous_signal_info is None:
                    for code, name, times, reasons, transactions_df in sell_without_buy_after:
                        if code == stock_result['code']:
                            continuous_signal_info = f"连续卖出({len(times)}次)"
                            break

                if continuous_signal_info:
                    text += f"  🤖 48小时智能建议: {continuous_signal_info}\n"

                text += "\n"
                # 文本版本：添加分割线
                text += "────────────────────────────────────────\n\n"

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
              <li><b>VaR(风险价值)</b>：在给定置信水平下，投资组合在特定时间内可能面临的最大损失。时间维度与投资周期相匹配：
                <ul>
                  <li><b>1日VaR(95%)</b>：适用于超短线交易（日内/隔夜），匹配持仓周期，控制单日最大回撤</li>
                  <li><b>5日VaR(95%)</b>：适用于波段交易（数天–数周），覆盖典型持仓期</li>
                  <li><b>20日VaR(95%)</b>：适用于中长期投资（1个月+），用于评估月度波动风险</li>
                </ul>
              </li>
              <li><b>ES(期望损失/Expected Shortfall)</b>：超过VaR阈值的所有损失的平均值，提供更全面的尾部风险评估。ES总是大于VaR，能更好地评估极端风险：
                <ul>
                  <li><b>1日ES(95%)</b>：超短线交易的极端损失预期，使用6个月历史数据计算</li>
                  <li><b>5日ES(95%)</b>：波段交易的极端损失预期，使用1年历史数据计算</li>
                  <li><b>20日ES(95%)</b>：中长期投资的极端损失预期，使用2年历史数据计算</li>
                  <li><b>重要性</b>：ES考虑了"黑天鹅"事件的潜在影响，为仓位管理和风险控制提供更保守的估计</li>
                </ul>
              </li>
              <li><b>历史回撤</b>：基于2年历史数据计算的最大回撤，衡量资产从历史高点到低点的最大跌幅。用于评估股票的历史波动性和风险特征：
                <ul>
                  <li><b>计算方式</b>：追踪资产的累计收益，计算从历史最高点到最低点的最大跌幅</li>
                  <li><b>参考价值</b>：回撤越大，说明该股票历史上波动性越高，风险越大</li>
                  <li><b>应用场景</b>：结合ES指标进行风险评估，判断当前风险水平是否合理</li>
                </ul>
              </li>
              <li><b>风险评估</b>：基于<b>20日ES</b>与历史最大回撤的比值进行的风险等级评估：
                <ul>
                  <li><b>优秀</b>：20日ES < 最大回撤/3，当前风险控制在历史波动范围内</li>
                  <li><b>合理</b>：回撤/3 ≤ 20日ES ≤ 回撤/2，风险水平适中，符合历史表现</li>
                  <li><b>警示</b>：20日ES > 最大回撤/2，当前风险水平超过历史波动，需要谨慎</li>
                  <li><b>决策参考</b>：绿色(优秀)可考虑增加仓位，红色(警示)建议降低仓位或规避</li>
                  <li><b>说明</b>：选择20日ES是因为它匹配中长期投资周期，能更好地评估月度波动风险</li>
                </ul>
              </li>
              <li><b>TAV评分(趋势-动量-成交量综合评分)</b>：基于趋势(Trend)、动量(Momentum)、成交量(Volume)三个维度的综合评分系统，范围0-100分：
                <ul>
                  <li><b>计算方式</b>：TAV评分 = 趋势评分 × 40% + 动量评分 × 35% + 成交量评分 × 25%</li>
                  <li><b>趋势评分(40%权重)</b>：基于20日、50日、200日移动平均线的排列和价格位置计算，评估长期、中期、短期趋势的一致性</li>
                  <li><b>动量评分(35%权重)</b>：结合RSI(14日)和MACD(12,26,9)指标，评估价格变化的动能强度和方向</li>
                  <li><b>成交量评分(25%权重)</b>：基于20日成交量均线，分析成交量突增(>1.2倍为弱、>1.5倍为中、>2倍为强)或萎缩(<0.8倍)情况</li>
                  <li><b>评分等级</b>：
                    <ul>
                      <li>≥75分：<b>强共振</b> - 三个维度高度一致，强烈信号</li>
                      <li>50-74分：<b>中等共振</b> - 多数维度一致，中等信号</li>
                      <li>25-49分：<b>弱共振</b> - 部分维度一致，弱信号</li>
                      <li><25分：<b>无共振</b> - 各维度分歧，无明确信号</li>
                    </ul>
                  </li>
                  <li><b>资产类型差异</b>：不同资产类型使用不同权重配置，股票(40%/35%/25%)、加密货币(30%/45%/25%)、黄金(45%/30%/25%)</li>
                </ul>
              </li>
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
        text += "• VaR(风险价值)：在给定置信水平下，投资组合在特定时间内可能面临的最大损失。时间维度与投资周期相匹配：\n"
        text += "  - 1日VaR(95%)：适用于超短线交易（日内/隔夜），匹配持仓周期，控制单日最大回撤\n"
        text += "  - 5日VaR(95%)：适用于波段交易（数天–数周），覆盖典型持仓期\n"
        text += "  - 20日VaR(95%)：适用于中长期投资（1个月+），用于评估月度波动风险\n"
        text += "• ES(期望损失/Expected Shortfall)：超过VaR阈值的所有损失的平均值，提供更全面的尾部风险评估。ES总是大于VaR，能更好地评估极端风险：\n"
        text += "  - 1日ES(95%)：超短线交易的极端损失预期，使用6个月历史数据计算\n"
        text += "  - 5日ES(95%)：波段交易的极端损失预期，使用1年历史数据计算\n"
        text += "  - 20日ES(95%)：中长期投资的极端损失预期，使用2年历史数据计算\n"
        text += "  - 重要性：ES考虑了'黑天鹅'事件的潜在影响，为仓位管理和风险控制提供更保守的估计\n"
        text += "• 历史回撤：基于2年历史数据计算的最大回撤，衡量资产从历史高点到低点的最大跌幅。用于评估股票的历史波动性和风险特征：\n"
        text += "  - 计算方式：追踪资产的累计收益，计算从历史最高点到最低点的最大跌幅\n"
        text += "  - 参考价值：回撤越大，说明该股票历史上波动性越高，风险越大\n"
        text += "  - 应用场景：结合ES指标进行风险评估，判断当前风险水平是否合理\n"
        text += "• 风险评估：基于20日ES与历史最大回撤的比值进行的风险等级评估：\n"
        text += "  - 优秀：20日ES < 最大回撤/3，当前风险控制在历史波动范围内\n"
        text += "  - 合理：回撤/3 ≤ 20日ES ≤ 回撤/2，风险水平适中，符合历史表现\n"
        text += "  - 警示：20日ES > 最大回撤/2，当前风险水平超过历史波动，需要谨慎\n"
        text += "  - 决策参考：绿色(优秀)可考虑增加仓位，红色(警示)建议降低仓位或规避\n"
        text += "  - 说明：选择20日ES是因为它匹配中长期投资周期，能更好地评估月度波动风险\n"
        text += "• TAV评分(趋势-动量-成交量综合评分)：基于趋势(Trend)、动量(Momentum)、成交量(Volume)三个维度的综合评分系统，范围0-100分：\n"
        text += "  - 计算方式：TAV评分 = 趋势评分 × 40% + 动量评分 × 35% + 成交量评分 × 25%\n"
        text += "  - 趋势评分(40%权重)：基于20日、50日、200日移动平均线的排列和价格位置计算，评估长期、中期、短期趋势的一致性\n"
        text += "  - 动量评分(35%权重)：结合RSI(14日)和MACD(12,26,9)指标，评估价格变化的动能强度和方向\n"
        text += "  - 成交量评分(25%权重)：基于20日成交量均线，分析成交量突增(>1.2倍为弱、>1.5倍为中、>2倍为强)或萎缩(<0.8倍)情况\n"
        text += "  - 评分等级：\n"
        text += "    * ≥75分：强共振 - 三个维度高度一致，强烈信号\n"
        text += "    * 50-74分：中等共振 - 多数维度一致，中等信号\n"
        text += "    * 25-49分：弱共振 - 部分维度一致，弱信号\n"
        text += "    * <25分：无共振 - 各维度分歧，无明确信号\n"
        text += "  - 资产类型差异：不同资产类型使用不同权重配置，股票(40%/35%/25%)、加密货币(30%/45%/25%)、黄金(45%/30%/25%)\n"
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

    def get_dividend_info(self, stock_code, stock_name):
        """
        获取单只股票的股息和除净日信息
        """
        try:
            # 移除.HK后缀，akshare要求5位数字格式
            symbol = stock_code.replace('.HK', '')
            if len(symbol) < 5:
                symbol = symbol.zfill(5)
            elif len(symbol) > 5:
                symbol = symbol[-5:]
            
            print(f"正在获取 {stock_name} ({stock_code}) 的股息信息...")
            
            # 获取港股股息数据
            df_dividend = ak.stock_hk_dividend_payout_em(symbol=symbol)
            
            if df_dividend is None or df_dividend.empty:
                print(f"⚠️ 未找到 {stock_name} 的股息数据")
                return None
                
            # 检查数据列
            available_columns = df_dividend.columns.tolist()
            print(f"📋 {stock_name} 数据列: {available_columns}")
            
            # 创建结果DataFrame
            result_data = []
            
            for _, row in df_dividend.iterrows():
                try:
                    # 提取关键信息
                    ex_date = row.get('除净日', None)
                    dividend_plan = row.get('分红方案', None)
                    record_date = row.get('截至过户日', None)
                    announcement_date = row.get('最新公告日期', None)
                    fiscal_year = row.get('财政年度', None)
                    distribution_type = row.get('分配类型', None)
                    payment_date = row.get('发放日', None)
                    
                    # 只处理有除净日的记录
                    if pd.notna(ex_date):
                        result_data.append({
                            '股票代码': stock_code,
                            '股票名称': stock_name,
                            '除净日': ex_date,
                            '分红方案': dividend_plan,
                            '截至过户日': record_date,
                            '最新公告日期': announcement_date,
                            '财政年度': fiscal_year,
                            '分配类型': distribution_type,
                            '发放日': payment_date
                        })
                except Exception as e:
                    print(f"⚠️ 处理 {stock_name} 股息数据时出错: {e}")
                    continue
            
            if not result_data:
                print(f"⚠️ {stock_name} 没有有效的除净日数据")
                return None
                
            return pd.DataFrame(result_data)
            
        except Exception as e:
            print(f"⚠️ 获取 {stock_name} 股息信息失败: {e}")
            return None

    def get_upcoming_dividends(self, days_ahead=90):
        """
        获取未来指定天数内的即将除净的股票
        """
        all_dividends = []
        
        for stock_code, stock_name in self.stock_list.items():
            dividend_data = self.get_dividend_info(stock_code, stock_name)
            
            if dividend_data is not None and not dividend_data.empty:
                all_dividends.append(dividend_data)
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        if not all_dividends:
            print("⚠️ 未获取到任何股息数据")
            return None
        
        # 合并所有数据
        all_dividends_df = pd.concat(all_dividends, ignore_index=True)
        
        # 转换日期格式
        all_dividends_df['除净日'] = pd.to_datetime(all_dividends_df['除净日'])
        
        # 筛选未来指定天数内的除净日
        today = datetime.now()
        future_date = today + timedelta(days=days_ahead)
        
        upcoming_dividends = all_dividends_df[
            (all_dividends_df['除净日'] >= today) & 
            (all_dividends_df['除净日'] <= future_date)
        ].sort_values('除净日')
        
        # 筛选历史除净日（最近30天）
        past_date = today - timedelta(days=30)
        recent_dividends = all_dividends_df[
            (all_dividends_df['除净日'] >= past_date) & 
            (all_dividends_df['除净日'] < today)
        ].sort_values('除净日', ascending=False)
        
        return {
            'upcoming': upcoming_dividends,
            'recent': recent_dividends,
            'all': all_dividends_df.sort_values('除净日', ascending=False)
        }

    def format_dividend_table_html(self, dividend_data):
        """
        格式化股息信息为HTML表格
        """
        if dividend_data is None or dividend_data['upcoming'] is None or dividend_data['upcoming'].empty:
            return ""
        
        html = """
        <div class="section">
            <h3>📈 即将除净的港股信息</h3>
            <table>
                <tr>
                    <th>股票名称</th>
                    <th>股票代码</th>
                    <th>除净日</th>
                    <th>分红方案</th>
                    <th>截至过户日</th>
                    <th>发放日</th>
                    <th>财政年度</th>
                </tr>
        """
        
        for _, row in dividend_data['upcoming'].iterrows():
            ex_date = row['除净日'].strftime('%Y-%m-%d') if pd.notna(row['除净日']) else 'N/A'
            html += f"""
                <tr>
                    <td>{row['股票名称']}</td>
                    <td>{row['股票代码']}</td>
                    <td>{ex_date}</td>
                    <td>{row['分红方案']}</td>
                    <td>{row['截至过户日']}</td>
                    <td>{row['发放日']}</td>
                    <td>{row['财政年度']}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html

    def format_dividend_table_text(self, dividend_data):
        """
        格式化股息信息为文本
        """
        if dividend_data is None or dividend_data['upcoming'] is None or dividend_data['upcoming'].empty:
            return ""
        
        text = "📈 即将除净的港股信息:\n"
        text += "-" * 80 + "\n"
        text += f"{'股票名称':<15} {'股票代码':<10} {'除净日':<12} {'分红方案':<30} {'截至过户日':<12} {'发放日':<12} {'财政年度':<8}\n"
        text += "-" * 80 + "\n"
        
        for _, row in dividend_data['upcoming'].iterrows():
            ex_date = row['除净日'].strftime('%Y-%m-%d') if pd.notna(row['除净日']) else 'N/A'
            dividend_plan = row['分红方案'][:28] + '...' if len(row['分红方案']) > 28 else row['分红方案']
            # 格式化截至过户日和发放日
            record_date = row['截至过户日'] if pd.notna(row['截至过户日']) and row['截至过户日'] != '' else 'N/A'
            pay_date = row['发放日'] if pd.notna(row['发放日']) and row['发放日'] != '' else 'N/A'
            text += f"{row['股票名称']:<15} {row['股票代码']:<10} {ex_date:<12} {dividend_plan:<30} {record_date:<12} {pay_date:<12} {row['财政年度']:<8}\n"
        
        text += "-" * 80 + "\n\n"
        
        return text

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
            hsi_indicators = self.calculate_hsi_technical_indicators(hsi_data)

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
