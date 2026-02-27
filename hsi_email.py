#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数及港股主力资金追踪器股票价格监控和交易信号邮件通知系统
基于技术分析指标生成买卖信号，只在有交易信号时发送邮件

【重要约定】统一百分比/分数格式：
- 所有风险函数返回的 'percentage' 字段统一使用分数形式（fraction）
- 分数形式：0.05 表示 5%，0.25 表示 25%
- 金额计算公式：amount = position_value * percentage
- 格式化显示：f"{percentage:.2%}"  # 0.05 → "5.00%"
- 适用的函数：calculate_var, calculate_expected_shortfall, calculate_max_drawdown

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
import warnings

# 抑制 pkg_resources 弃用警告
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:py_mini_racer.*'
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module='py_mini_racer')

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
    from data_services.technical_analysis import TechnicalAnalyzer, TechnicalAnalyzerV2, TAVScorer, TAVConfig
    TECHNICAL_ANALYSIS_AVAILABLE = True
    TAV_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False

# 导入中期分析指标
try:
    from data_services.technical_analysis import (
        calculate_ma_alignment,
        calculate_ma_slope,
        calculate_ma_deviation,
        calculate_support_resistance,
        calculate_relative_strength,
        calculate_medium_term_score
    )
    MEDIUM_TERM_AVAILABLE = True
except ImportError:
    MEDIUM_TERM_AVAILABLE = False
    print("⚠️ 中期分析指标不可用")

# 导入基本面数据模块
try:
    from data_services.fundamental_data import get_comprehensive_fundamental_data
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False
    print("⚠️ 基本面数据模块不可用")

# 导入板块分析模块
try:
    from data_services.hk_sector_analysis import SectorAnalyzer
    SECTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    SECTOR_ANALYSIS_AVAILABLE = False
    print("⚠️ 板块分析模块不可用")

# 从全局配置导入股票列表
from config import WATCHLIST
STOCK_LIST = WATCHLIST


class HSIEmailSystem:
    """恒生指数及港股主力资金追踪器邮件系统"""

    # 根据投资风格和计算窗口确定历史数据长度
    DATA_PERIOD_CONFIG = {
        'ultra_short_term': '6mo',    # 超短线：6个月数据（约125个交易日）
        'short_term': '1y',           # 波段交易：1年数据（约247个交易日）
        'medium_long_term': '2y',      # 中长期投资：2年数据（约493个交易日）
    }

    # ==============================
    # 加权评分系统参数（新增）
    # ==============================

    # 是否启用加权评分系统（向后兼容）
    USE_SCORED_SIGNALS = True   # True=使用新的评分系统，False=使用原有的布尔逻辑

    # 建仓信号权重配置
    BUILDUP_WEIGHTS = {
        'price_low': 2.0,      # 价格处于低位
        'vol_ratio': 2.0,      # 成交量放大
        'vol_z': 1.0,          # 成交量z-score
        'macd_cross': 1.5,     # MACD金叉
        'rsi_oversold': 1.2,   # RSI超卖
        'obv_up': 1.0,         # OBV上升
        'vwap_vol': 1.2,       # 价格高于VWAP且放量
        'cmf_in': 1.2,         # CMF资金流入
        'price_above_vwap': 0.8,  # 价格高于VWAP
        'bb_oversold': 1.0,    # 布林带超卖
    }

    # 建仓信号阈值
    BUILDUP_THRESHOLD_STRONG = 5.0   # 强烈建仓信号阈值
    BUILDUP_THRESHOLD_PARTIAL = 3.0  # 部分建仓信号阈值

    # 出货信号权重配置
    DISTRIBUTION_WEIGHTS = {
        'price_high': 2.0,     # 价格处于高位
        'vol_ratio': 2.0,      # 成交量放大
        'vol_z': 1.5,          # 成交量z-score
        'macd_cross': 1.5,     # MACD死叉
        'rsi_high': 1.5,       # RSI超买
        'cmf_out': 1.5,        # CMF资金流出
        'obv_down': 1.0,       # OBV下降
        'vwap_vol': 1.5,       # 价格低于VWAP且放量
        'price_down': 1.0,     # 价格下跌
        'bb_overbought': 1.0,  # 布林带超买
    }

    # 出货信号阈值
    DISTRIBUTION_THRESHOLD_STRONG = 5.0   # 强烈出货信号阈值
    DISTRIBUTION_THRESHOLD_WEAK = 3.0     # 弱出货信号阈值

    # 价格位置阈值
    PRICE_LOW_PCT = 40.0   # 价格百分位低于该值视为"低位"
    PRICE_HIGH_PCT = 60.0  # 高于该值视为"高位"

    # 成交量阈值
    VOL_RATIO_BUILDUP = 1.3
    VOL_RATIO_DISTRIBUTION = 2.0

    # 板块分析配置
    SECTOR_ANALYSIS_PERIOD = 5  # 板块分析计算周期（交易日）
    SECTOR_ANALYSIS_PERIOD_NAME = "5日"  # 显示名称

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
            # 注意：这里返回分数形式（例如 0.25 表示 25%），与 VaR/ES 保持一致
            max_drawdown_percentage = abs(drawdown.min())
            
            # 计算回撤货币值
            max_drawdown_amount = None
            if position_value is not None and position_value > 0:
                max_drawdown_amount = position_value * max_drawdown_percentage
            
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

    def _get_latest_stop_loss_target(self, stock_code, target_date=None):
        """
        公用方法：从交易记录中获取指定股票的最新止损价和目标价
        
        参数:
        - stock_code: 股票代码
        - target_date: 目标日期，如果为None则使用当前时间
        
        返回:
        - tuple: (latest_stop_loss, latest_target_price)
        """
        try:
            df_transactions = self._read_transactions_df()
            if df_transactions.empty:
                return None, None
                
            # 如果指定了目标日期，过滤出该日期之前的交易记录
            if target_date is not None:
                # 将目标日期转换为带时区的时间戳
                if isinstance(target_date, str):
                    target_dt = pd.to_datetime(target_date, utc=True)
                else:
                    target_dt = pd.to_datetime(target_date, utc=True)
                # 设置为目标日期的收盘时间（16:00 UTC，对应香港时间24:00）
                reference_time = target_dt.replace(hour=16, minute=0, second=0, microsecond=0)
                df_transactions = df_transactions[df_transactions['timestamp'] <= reference_time]
            
            stock_transactions = df_transactions[df_transactions['code'] == stock_code]
            if stock_transactions.empty:
                return None, None
                
            # 按时间排序，获取最新的交易记录
            stock_transactions = stock_transactions.sort_values('timestamp')
            latest_transaction = stock_transactions.iloc[-1]
            latest_stop_loss = latest_transaction.get('stop_loss_price')
            latest_target_price = latest_transaction.get('target_price')
            
            return latest_stop_loss, latest_target_price
        except Exception as e:
            print(f"⚠️ 获取股票 {stock_code} 的止损价和目标价失败: {e}")
            return None, None

    def _get_trend_color_style(self, trend):
        """
        公用方法：根据趋势内容返回对应的颜色样式
        
        参数:
        - trend: 趋势字符串
        
        返回:
        - str: 颜色样式字符串
        """
        if "多头" in trend:
            return "color: green; font-weight: bold;"
        elif "空头" in trend:
            return "color: red; font-weight: bold;"
        elif "震荡" in trend:
            return "color: orange; font-weight: bold;"
        else:
            return ""

    def _get_signal_color_style(self, signal_type):
        """
        公用方法：根据信号类型返回对应的颜色样式
        
        参数:
        - signal_type: 信号类型字符串
        
        返回:
        - str: 颜色样式字符串
        """
        if "买入" in signal_type:
            return "color: green; font-weight: bold;"
        elif "卖出" in signal_type:
            return "color: red; font-weight: bold;"
        else:
            return "color: orange; font-weight: bold;"

    def _format_var_es_display(self, var_value, var_amount=None, es_value=None, es_amount=None):
        """
        公用方法：格式化VaR和ES值的显示
        
        参数:
        - var_value: VaR值（百分比形式，如0.05表示5%）
        - var_amount: VaR货币值
        - es_value: ES值（百分比形式，如0.05表示5%）
        - es_amount: ES货币值
        
        返回:
        - dict: 包含格式化后的VaR和ES显示字符串
        """
        result = {
            'var_display': 'N/A',
            'es_display': 'N/A'
        }
        
        try:
            # 格式化VaR值
            if var_value is not None:
                var_display = f"{var_value:.2%}"
                if var_amount is not None:
                    var_display += f" (HK${var_amount:.2f})"
                result['var_display'] = var_display
            
            # 格式化ES值
            if es_value is not None:
                es_display = f"{es_value:.2%}"
                if es_amount is not None:
                    es_display += f" (HK${es_amount:.2f})"
                result['es_display'] = es_display
                
        except Exception as e:
            print(f"⚠️ 格式化VaR/ES值时出错: {e}")
        
        return result

    def _get_trend_change_arrow(self, current_trend, previous_trend):
        """
        公用方法：返回趋势变化箭头符号
        
        参数:
        - current_trend: 当前趋势
        - previous_trend: 上个交易日趋势
        
        返回:
        - str: 箭头符号和颜色样式
        """
        if previous_trend is None or previous_trend == 'N/A' or current_trend is None or current_trend == 'N/A':
            return '<span style="color: #999;">→</span>'
        
        # 定义看涨趋势
        bullish_trends = ['强势多头', '多头趋势', '短期上涨']
        # 定义看跌趋势
        bearish_trends = ['弱势空头', '空头趋势', '短期下跌']
        # 定义震荡趋势
        consolidation_trends = ['震荡整理', '震荡']
        
        # 趋势改善：看跌/震荡 → 看涨
        if (previous_trend in bearish_trends + consolidation_trends) and current_trend in bullish_trends:
            return '<span style="color: green; font-weight: bold;">↑</span>'
        
        # 趋势恶化：看涨 → 看跌
        if previous_trend in bullish_trends and current_trend in bearish_trends:
            return '<span style="color: red; font-weight: bold;">↓</span>'
        
        # 震荡 → 看跌（恶化）
        if previous_trend in consolidation_trends and current_trend in bearish_trends:
            return '<span style="color: red; font-weight: bold;">↓</span>'
        
        # 看涨 → 震荡（改善）
        if previous_trend in bullish_trends and current_trend in consolidation_trends:
            return '<span style="color: orange; font-weight: bold;">↓</span>'
        
        # 看跌 → 震荡（改善）
        if previous_trend in bearish_trends and current_trend in consolidation_trends:
            return '<span style="color: orange; font-weight: bold;">↑</span>'
        
        # 无明显变化（同类型趋势）
        return '<span style="color: #999;">→</span>'
    def _get_score_change_arrow(self, current_score, previous_score):
        """
        公用方法：返回评分变化箭头符号
        
        参数:
        - current_score: 当前评分
        - previous_score: 上个交易日评分
        
        返回:
        - str: 箭头符号和颜色样式
        """
        if previous_score is None or current_score is None:
            return '<span style="color: #999;">→</span>'
        
        if current_score > previous_score:
            return '<span style="color: green; font-weight: bold;">↑</span>'
        elif current_score < previous_score:
            return '<span style="color: red; font-weight: bold;">↓</span>'
        else:
            return '<span style="color: #999;">→</span>'

    def _get_price_change_arrow(self, current_price_str, previous_price):
        """
        公用方法：返回价格变化箭头符号
        
        参数:
        - current_price_str: 当前价格字符串（格式化后的）
        - previous_price: 上个交易日价格（数值）
        
        返回:
        - str: 箭头符号和颜色样式
        """
        if previous_price is None or current_price_str is None or current_price_str == 'N/A':
            return '<span style="color: #999;">→</span>'
        
        try:
            current_price = float(current_price_str.replace(',', ''))
            if current_price > previous_price:
                return '<span style="color: green; font-weight: bold;">↑</span>'
            elif current_price < previous_price:
                return '<span style="color: red; font-weight: bold;">↓</span>'
            else:
                return '<span style="color: #999;">→</span>'
        except:
            return '<span style="color: #999;">→</span>'

    def _format_continuous_signal_details(self, transactions_df, times):
        """
        公用方法：格式化连续信号的详细信息（HTML版本）
        
        参数:
        - transactions_df: 交易记录DataFrame
        - times: 时间列表
        
        返回:
        - str: 格式化后的连续信号详细信息
        """
        try:
            combined_str = ""
            # 确保交易记录按时间排序
            transactions_df = transactions_df.sort_values('timestamp')
            
            for i in range(len(times)):
                time_info = f"{times[i]}"
                
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
                else:
                    price_info = ""
                    stop_loss_info = ""
                    target_price_info = ""
                    validity_period_info = ""
                
                info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
                reason_info = ", ".join(info_parts)
                time_reason = f"{time_info} {reason_info}".strip()
                combined_str += time_reason + ("<br>" if i < len(times) - 1 else "")
            
            return combined_str
        except Exception as e:
            print(f"⚠️ 格式化连续信号详细信息时出错: {e}")
            return ""

    def _format_continuous_signal_details_text(self, transactions_df, times):
        """
        公用方法：格式化连续信号的详细信息（文本版本）
        
        参数:
        - transactions_df: 交易记录DataFrame
        - times: 时间列表
        
        返回:
        - str: 格式化后的连续信号详细信息
        """
        try:
            combined_list = []
            # 确保交易记录按时间排序
            transactions_df = transactions_df.sort_values('timestamp')
            
            for i in range(len(times)):
                time_info = f"{times[i]}"
                
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
                else:
                    price_info = ""
                    stop_loss_info = ""
                    target_price_info = ""
                    validity_period_info = ""
                
                info_parts = [part for part in [price_info, target_price_info, stop_loss_info, validity_period_info] if part]
                reason_info = ", ".join(info_parts)
                combined_item = f"{time_info} {reason_info}".strip()
                combined_list.append(combined_item)
            
            return "\n    ".join(combined_list)
        except Exception as e:
            print(f"⚠️ 格式化连续信号详细信息时出错: {e}")
            return ""

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

    def _calculate_buildup_score(self, row, hist_df=None):
        """
        基于加权评分的建仓信号检测
        
        Args:
            row: 包含技术指标的数据行（Series）
            hist_df: 历史数据DataFrame（用于计算某些指标）
        
        Returns:
            dict: 包含评分、信号级别和触发原因
            - score: 建仓评分（0-10+）
            - signal: 信号级别 ('none', 'partial', 'strong')
            - reasons: 触发条件的列表
        """
        score = 0.0
        reasons = []

        # 价格位置：低位加分
        price_percentile = row.get('price_position', 50.0)
        if pd.notna(price_percentile) and price_percentile < self.PRICE_LOW_PCT:
            score += self.BUILDUP_WEIGHTS['price_low']
            reasons.append('price_low')

        # 成交量倍数
        vol_ratio = row.get('volume_ratio', 0.0)
        if pd.notna(vol_ratio) and vol_ratio > self.VOL_RATIO_BUILDUP:
            score += self.BUILDUP_WEIGHTS['vol_ratio']
            reasons.append('vol_ratio')

        # 成交量 z-score
        vol_z_score = row.get('vol_z_score', 0.0)
        if pd.notna(vol_z_score) and vol_z_score > 1.2:
            score += self.BUILDUP_WEIGHTS['vol_z']
            reasons.append('vol_z')

        # MACD 线上穿（金叉）
        macd = row.get('macd', 0.0)
        macd_signal = row.get('macd_signal', 0.0)
        if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
            score += self.BUILDUP_WEIGHTS['macd_cross']
            reasons.append('macd_cross')

        # RSI 超卖
        rsi = row.get('rsi', 50.0)
        if pd.notna(rsi) and rsi < 40:
            score += self.BUILDUP_WEIGHTS['rsi_oversold']
            reasons.append('rsi_oversold')

        # OBV 上升
        obv = row.get('obv', 0.0)
        if pd.notna(obv) and obv > 0:
            score += self.BUILDUP_WEIGHTS['obv_up']
            reasons.append('obv_up')

        # 收盘高于 VWAP 且放量
        vwap = row.get('vwap', 0.0)
        current_price = row.get('current_price', 0.0)
        if (pd.notna(vwap) and pd.notna(vol_ratio) and pd.notna(current_price) and 
            current_price > vwap and vol_ratio > 1.2):
            score += self.BUILDUP_WEIGHTS['vwap_vol']
            reasons.append('vwap_vol')

        # 价格高于 VWAP
        if pd.notna(vwap) and pd.notna(current_price) and current_price > vwap:
            score += self.BUILDUP_WEIGHTS['price_above_vwap']
            reasons.append('price_above_vwap')

        # 布林带超卖
        bb_position = row.get('bb_position', 0.5)
        if pd.notna(bb_position) and bb_position < 0.2:
            score += self.BUILDUP_WEIGHTS['bb_oversold']
            reasons.append('bb_oversold')

        # CMF资金流入
        cmf = row.get('cmf', 0.0)
        if pd.notna(cmf) and cmf > 0.03:
            score += self.BUILDUP_WEIGHTS['cmf_in']
            reasons.append('cmf_in')

        # 返回分数与分层建议
        signal = None
        if score >= self.BUILDUP_THRESHOLD_STRONG:
            signal = 'strong'    # 强烈建仓（建议较高比例或确认）
        elif score >= self.BUILDUP_THRESHOLD_PARTIAL:
            signal = 'partial'   # 部分建仓 / 分批入场
        else:
            signal = 'none'      # 无信号

        return {
            'score': score,
            'signal': signal,
            'reasons': ','.join(reasons) if reasons else ''
        }

    def _calculate_distribution_score(self, row, hist_df=None):
        """
        基于加权评分的出货信号检测
        
        Args:
            row: 包含技术指标的数据行（Series）
            hist_df: 历史数据DataFrame（用于计算某些指标）
        
        Returns:
            dict: 包含评分、信号级别和触发原因
            - score: 出货评分（0-10+）
            - signal: 信号级别 ('none', 'weak', 'strong')
            - reasons: 触发条件的列表
        """
        score = 0.0
        reasons = []

        # 价格位置：高位加分
        price_percentile = row.get('price_position', 50.0)
        if pd.notna(price_percentile) and price_percentile > self.PRICE_HIGH_PCT:
            score += self.DISTRIBUTION_WEIGHTS['price_high']
            reasons.append('price_high')

        # 成交量倍数
        vol_ratio = row.get('volume_ratio', 0.0)
        if pd.notna(vol_ratio) and vol_ratio > self.VOL_RATIO_DISTRIBUTION:
            score += self.DISTRIBUTION_WEIGHTS['vol_ratio']
            reasons.append('vol_ratio')

        # 成交量 z-score
        vol_z_score = row.get('vol_z_score', 0.0)
        if pd.notna(vol_z_score) and vol_z_score > 1.5:
            score += self.DISTRIBUTION_WEIGHTS['vol_z']
            reasons.append('vol_z')

        # MACD 线下穿（死叉）
        macd = row.get('macd', 0.0)
        macd_signal = row.get('macd_signal', 0.0)
        if pd.notna(macd) and pd.notna(macd_signal) and macd < macd_signal:
            score += self.DISTRIBUTION_WEIGHTS['macd_cross']
            reasons.append('macd_cross')

        # RSI 超买
        rsi = row.get('rsi', 50.0)
        if pd.notna(rsi) and rsi > 65:
            score += self.DISTRIBUTION_WEIGHTS['rsi_high']
            reasons.append('rsi_high')

        # OBV 下降
        obv = row.get('obv', 0.0)
        if pd.notna(obv) and obv < 0:
            score += self.DISTRIBUTION_WEIGHTS['obv_down']
            reasons.append('obv_down')

        # 收盘低于 VWAP 且放量
        vwap = row.get('vwap', 0.0)
        current_price = row.get('current_price', 0.0)
        if (pd.notna(vwap) and pd.notna(vol_ratio) and pd.notna(current_price) and 
            current_price < vwap and vol_ratio > 1.2):
            score += self.DISTRIBUTION_WEIGHTS['vwap_vol']
            reasons.append('vwap_vol')

        # 价格下跌
        change_1d = row.get('change_1d', 0.0)
        if pd.notna(change_1d) and change_1d < 0:
            score += self.DISTRIBUTION_WEIGHTS['price_down']
            reasons.append('price_down')

        # 布林带超买
        bb_position = row.get('bb_position', 0.5)
        if pd.notna(bb_position) and bb_position > 0.8:
            score += self.DISTRIBUTION_WEIGHTS['bb_overbought']
            reasons.append('bb_overbought')

        # CMF资金流出
        cmf = row.get('cmf', 0.0)
        if pd.notna(cmf) and cmf < -0.05:
            score += self.DISTRIBUTION_WEIGHTS['cmf_out']
            reasons.append('cmf_out')

        # 返回分数与分层建议
        signal = None
        if score >= self.DISTRIBUTION_THRESHOLD_STRONG:
            signal = 'strong'    # 强烈出货（建议较大比例卖出）
        elif score >= self.DISTRIBUTION_THRESHOLD_WEAK:
            signal = 'weak'      # 弱出货（建议部分减仓或观察）
        else:
            signal = 'none'      # 无信号

        return {
            'score': score,
            'signal': signal,
            'reasons': ','.join(reasons) if reasons else ''
        }

    def _calculate_technical_indicators_core(self, data, asset_type='stock', us_df=None):
        """
        计算技术指标的核心方法（支持不同资产类型）- 修复版本

        参数:
        - data: 股票数据
        - asset_type: 资产类型（'stock' 或 'hsi'）
        - us_df: 美股市场数据（可选，避免重复获取）
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
                    'rsi': None,
                    'macd': None,
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
                if self.use_tav and isinstance(self.technical_analyzer, TechnicalAnalyzerV2):
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
            except Exception as e:
                print(f"⚠️ 技术分析计算失败: {e}")
                import traceback
                traceback.print_exc()
                # 返回一个基本的指标字典
                indicators = {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'price_position': 50.0,
                    'bb_position': 0.5,
                    'trend': '数据不足',
                    'recent_buy_signals': [],
                    'recent_sell_signals': [],
                    'current_price': hist['Close'].iloc[-1] if hist is not None and not hist.empty else 0,
                    'ma20': 0,
                    'ma50': 0,
                    'ma200': 0,
                    'hist': hist,
                    'atr': 0.0,
                    'stop_loss': None,
                    'take_profit': None,
                    'volume_ratio': 0.0,
                    'volume_surge': False,
                    'volume_shrink': False,
                    'volume_ma10': 0.0,
                    'volume_ma20': 0.0,
                    'var_ultra_short_term': None,
                    'var_short_term': None,
                    'var_medium_long_term': None,
                    'var_ultra_short_term_amount': None,
                    'var_short_term_amount': None,
                    'var_medium_long_term_amount': None
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
                         # 添加评分系统信息（如果启用）
            if self.USE_SCORED_SIGNALS:
                try:
                        # 准备评分所需的数据行
                        obv_value = latest.get('OBV', 0.0) if 'OBV' in latest else 0.0
                        vwap_value = latest.get('VWAP', 0.0) if 'VWAP' in latest else 0.0
                        cmf_value = latest.get('CMF', 0.0) if 'CMF' in latest else 0.0
                        
                        score_row = pd.Series({
                            'price_position': indicators.get('price_position', 50.0),
                            'volume_ratio': volume_ratio,
                            'vol_z_score': latest.get('Vol_Z_Score', 0.0) if 'Vol_Z_Score' in latest else 0.0,
                            'macd': macd,
                            'macd_signal': macd_signal,
                            'rsi': rsi,
                            'obv': obv_value,
                            'vwap': vwap_value,
                            'current_price': current_price,
                            'change_1d': data.get('change_1d', 0.0),
                            'bb_position': bb_position,
                            'cmf': cmf_value
                        })
                        
                        # 将评分所需的指标添加到 indicators 字典中
                        indicators['obv'] = obv_value
                        indicators['vwap'] = vwap_value
                        indicators['cmf'] = cmf_value
                        
                        # 计算建仓评分
                        buildup_result = self._calculate_buildup_score(score_row, hist)
                        indicators['buildup_score'] = buildup_result['score']
                        indicators['buildup_level'] = buildup_result['signal']
                        indicators['buildup_reasons'] = buildup_result['reasons']
                        
                        # 计算出货评分
                        distribution_result = self._calculate_distribution_score(score_row, hist)
                        indicators['distribution_score'] = distribution_result['score']
                        indicators['distribution_level'] = distribution_result['signal']
                        indicators['distribution_reasons'] = distribution_result['reasons']
                        
                except Exception as e:
                        print(f"⚠️ 评分系统计算失败: {e}")
                        indicators['buildup_score'] = 0.0
                        indicators['buildup_level'] = 'none'
                        indicators['buildup_reasons'] = ''
                        indicators['distribution_score'] = 0.0
                        indicators['distribution_level'] = 'none'
                        indicators['distribution_reasons'] = ''
            else:
                # 评分系统未启用，设置为默认值
                indicators['buildup_score'] = None
                indicators['buildup_level'] = None
                indicators['buildup_reasons'] = None
                indicators['distribution_score'] = None
                indicators['distribution_level'] = None
                indicators['distribution_reasons'] = None
                         # 添加中期分析指标
            try:
                if MEDIUM_TERM_AVAILABLE:
                        # 计算均线排列
                        ma_alignment = calculate_ma_alignment(indicators_with_signals)
                        indicators['ma_alignment'] = ma_alignment['alignment']
                        indicators['ma_alignment_strength'] = ma_alignment['strength']
                        
                        # 计算均线斜率
                        ma_slope_20 = calculate_ma_slope(indicators_with_signals, 20)
                        ma_slope_50 = calculate_ma_slope(indicators_with_signals, 50)
                        indicators['ma20_slope'] = ma_slope_20['slope']
                        indicators['ma20_slope_angle'] = ma_slope_20['angle']
                        indicators['ma20_slope_trend'] = ma_slope_20['trend']
                        indicators['ma50_slope'] = ma_slope_50['slope']
                        indicators['ma50_slope_angle'] = ma_slope_50['angle']
                        indicators['ma50_slope_trend'] = ma_slope_50['trend']
                        
                        # 计算均线乖离率
                        ma_deviation = calculate_ma_deviation(indicators_with_signals)
                        indicators['ma_deviation'] = ma_deviation['deviations']
                        indicators['ma_deviation_avg'] = ma_deviation['avg_deviation']
                        indicators['ma_deviation_extreme'] = ma_deviation['extreme_deviation']
                        
                        # 计算支撑阻力位
                        support_resistance = calculate_support_resistance(indicators_with_signals)
                        indicators['support_levels'] = support_resistance['support_levels']
                        indicators['resistance_levels'] = support_resistance['resistance_levels']
                        indicators['nearest_support'] = support_resistance['nearest_support']
                        indicators['nearest_resistance'] = support_resistance['nearest_resistance']
                        
                        # 计算中期趋势评分
                        medium_term_score = calculate_medium_term_score(indicators_with_signals)
                        indicators['medium_term_score'] = medium_term_score['total_score']
                        indicators['medium_term_components'] = medium_term_score['components']
                        indicators['medium_term_trend_health'] = medium_term_score['trend_health']
                        indicators['medium_term_sustainability'] = medium_term_score['sustainability']
                        indicators['medium_term_recommendation'] = medium_term_score['recommendation']
                        
            except Exception as e:
                print(f"⚠️ 中期分析指标计算失败: {e}")
                indicators['ma_alignment'] = '数据不足'
                indicators['ma_alignment_strength'] = 0
                indicators['ma20_slope'] = 0
                indicators['ma20_slope_angle'] = 0
                indicators['ma20_slope_trend'] = '数据不足'
                indicators['ma50_slope'] = 0
                indicators['ma50_slope_angle'] = 0
                indicators['ma50_slope_trend'] = '数据不足'
                indicators['ma_deviation'] = {}
                indicators['ma_deviation_avg'] = 0
                indicators['ma_deviation_extreme'] = '数据不足'
                indicators['support_levels'] = []
                indicators['resistance_levels'] = []
                indicators['nearest_support'] = None
                indicators['nearest_resistance'] = None
                indicators['medium_term_score'] = 0
                indicators['medium_term_components'] = {}
                indicators['medium_term_trend_health'] = '数据不足'
                indicators['medium_term_sustainability'] = '低'
                indicators['medium_term_recommendation'] = '观望'
                         # 添加基本面数据
            try:
                if FUNDAMENTAL_AVAILABLE:
                        # 获取股票代码（去掉.HK后缀）
                        stock_code = data.get('symbol', '').replace('.HK', '')
                        if stock_code:
                            fundamental_data = get_comprehensive_fundamental_data(stock_code)
                            
                            if fundamental_data is not None:
                                # 计算基本面评分（与hk_smart_money_tracker.py相同的逻辑）
                                fundamental_score = 0
                                fundamental_details = {}
                                
                                pe = fundamental_data.get('fi_pe_ratio')
                                pb = fundamental_data.get('fi_pb_ratio')
                                
                                # PE评分（50分）
                                if pe is not None:
                                    if pe < 10:
                                        fundamental_score += 50
                                        fundamental_details['pe_score'] = "低估值 (PE<10)"
                                    elif pe < 15:
                                        fundamental_score += 40
                                        fundamental_details['pe_score'] = "合理估值 (10<PE<15)"
                                    elif pe < 20:
                                        fundamental_score += 30
                                        fundamental_details['pe_score'] = "偏高估值 (15<PE<20)"
                                    elif pe < 25:
                                        fundamental_score += 20
                                        fundamental_details['pe_score'] = "高估值 (20<PE<25)"
                                    else:
                                        fundamental_score += 10
                                        fundamental_details['pe_score'] = "极高估值 (PE>25)"
                                else:
                                    fundamental_score += 25
                                    fundamental_details['pe_score'] = "无PE数据"
                                
                                # PB评分（50分）
                                if pb is not None:
                                    if pb < 1:
                                        fundamental_score += 50
                                        fundamental_details['pb_score'] = "低市净率 (PB<1)"
                                    elif pb < 1.5:
                                        fundamental_score += 40
                                        fundamental_details['pb_score'] = "合理市净率 (1<PB<1.5)"
                                    elif pb < 2:
                                        fundamental_score += 30
                                        fundamental_details['pb_score'] = "偏高市净率 (1.5<PB<2)"
                                    elif pb < 3:
                                        fundamental_score += 20
                                        fundamental_details['pb_score'] = "高市净率 (2<PB<3)"
                                    else:
                                        fundamental_score += 10
                                        fundamental_details['pb_score'] = "极高市净率 (PB>3)"
                                else:
                                    fundamental_score += 25
                                    fundamental_details['pb_score'] = "无PB数据"
                                
                                # 添加基本面指标到indicators
                                indicators['fundamental_score'] = fundamental_score
                                indicators['fundamental_details'] = fundamental_details
                                indicators['pe_ratio'] = pe
                                indicators['pb_ratio'] = pb
                                
                                print(f"  📊 {data.get('symbol', '')} 基本面数据获取成功: PE={pe}, PB={pb}, 评分={fundamental_score}")
                            else:
                                print(f"  ⚠️ {data.get('symbol', '')} 无法获取基本面数据")
                                indicators['fundamental_score'] = 0
                                indicators['pe_ratio'] = None
                                indicators['pb_ratio'] = None
                        else:
                            print(f"  ⚠️ {data.get('symbol', '')} 股票代码为空，跳过基本面数据获取")
                            indicators['fundamental_score'] = 0
                            indicators['pe_ratio'] = None
                            indicators['pb_ratio'] = None
            except Exception as e:
                print(f"⚠️ 获取基本面数据失败: {e}")
                indicators['fundamental_score'] = 0
                indicators['pe_ratio'] = None
                indicators['pb_ratio'] = None
            
            # 添加市场情绪和流动性指标（新增）
            try:
                # 导入 us_market_data（移到作用域开始处，确保在整个 try 块内可用）
                from ml_services.us_market_data import us_market_data
                
                # 1. 获取VIX恐慌指数（使用传入的 us_df，避免重复获取）
                if us_df is None:
                    # 如果没有传入 us_df，则获取一次（向后兼容）
                    us_df = us_market_data.get_all_us_market_data(period_days=30)
                
                if us_df is not None and not us_df.empty and 'VIX_Level' in us_df.columns:
                    indicators['vix_level'] = us_df['VIX_Level'].iloc[-1]
                else:
                    indicators['vix_level'] = None
                
                # 2. 计算成交额变化率
                if hist is not None and not hist.empty:
                    # 计算成交额（价格 × 成交量）
                    turnover = hist['Close'] * hist['Volume']
                    
                    # 计算成交额变化率
                    turnover_change_1d = turnover.pct_change(1).iloc[-1] if len(turnover) > 1 else None
                    turnover_change_5d = turnover.pct_change(5).iloc[-1] if len(turnover) > 5 else None
                    turnover_change_10d = turnover.pct_change(10).iloc[-1] if len(turnover) > 10 else None
                    turnover_change_20d = turnover.pct_change(20).iloc[-1] if len(turnover) > 20 else None
                    
                    # 检查并处理 inf 和 -inf 值（分母为0的情况）
                    turnover_change_1d = turnover_change_1d if pd.notna(turnover_change_1d) and not pd.isinf(turnover_change_1d) else None
                    turnover_change_5d = turnover_change_5d if pd.notna(turnover_change_5d) and not pd.isinf(turnover_change_5d) else None
                    turnover_change_10d = turnover_change_10d if pd.notna(turnover_change_10d) and not pd.isinf(turnover_change_10d) else None
                    turnover_change_20d = turnover_change_20d if pd.notna(turnover_change_20d) and not pd.isinf(turnover_change_20d) else None
                    
                    # 转换为百分比
                    indicators['turnover_change_1d'] = turnover_change_1d * 100 if turnover_change_1d is not None else None
                    indicators['turnover_change_5d'] = turnover_change_5d * 100 if turnover_change_5d is not None else None
                    indicators['turnover_change_10d'] = turnover_change_10d * 100 if turnover_change_10d is not None else None
                    indicators['turnover_change_20d'] = turnover_change_20d * 100 if turnover_change_20d is not None else None
                else:
                    indicators['turnover_change_1d'] = None
                    indicators['turnover_change_5d'] = None
                    indicators['turnover_change_10d'] = None
                    indicators['turnover_change_20d'] = None
                
                # 3. 计算换手率变化率
                if hist is not None and not hist.empty and 'Volume' in hist.columns:
                    # 获取已发行股本（从基本面数据）
                    stock_code = data.get('symbol', '').replace('.HK', '')
                    float_shares = None
                    try:
                        if FUNDAMENTAL_AVAILABLE and stock_code:
                            fundamental_data = get_comprehensive_fundamental_data(stock_code)
                            if fundamental_data is not None:
                                # 优先使用已发行股本
                                issued_shares = fundamental_data.get('fi_issued_shares')
                                if issued_shares is not None and issued_shares > 0:
                                    float_shares = float(issued_shares)
                                # 如果没有已发行股本，使用市值推算
                                elif fundamental_data.get('fi_market_cap') is not None:
                                    market_cap = fundamental_data.get('fi_market_cap')
                                    current_price = hist['Close'].iloc[-1]
                                    if current_price is not None and current_price > 0:
                                        float_shares = market_cap / current_price
                    except Exception as e:
                        print(f"⚠️ 获取已发行股本失败: {e}")
                    
                    # 计算换手率
                    if float_shares is not None and float_shares > 0:
                        turnover_rate = (hist['Volume'] / float_shares) * 100
                        
                        # 计算换手率变化率（需要检查NaN值和inf值）
                        turnover_rate_change_5d = None
                        turnover_rate_change_20d = None
                        
                        if len(turnover_rate) > 5:
                            change_5d = turnover_rate.pct_change(5).iloc[-1]
                            # 检查并处理 inf 和 -inf 值（分母为0的情况）
                            turnover_rate_change_5d = change_5d if pd.notna(change_5d) and not pd.isinf(change_5d) else None
                        
                        if len(turnover_rate) > 20:
                            change_20d = turnover_rate.pct_change(20).iloc[-1]
                            # 检查并处理 inf 和 -inf 值（分母为0的情况）
                            turnover_rate_change_20d = change_20d if pd.notna(change_20d) and not pd.isinf(change_20d) else None
                        
                        indicators['turnover_rate'] = turnover_rate.iloc[-1] if len(turnover_rate) > 0 else None
                        indicators['turnover_rate_change_5d'] = turnover_rate_change_5d
                        indicators['turnover_rate_change_20d'] = turnover_rate_change_20d
                    else:
                        indicators['turnover_rate'] = None
                        indicators['turnover_rate_change_5d'] = None
                        indicators['turnover_rate_change_20d'] = None
                else:
                    indicators['turnover_rate'] = None
                    indicators['turnover_rate_change_5d'] = None
                    indicators['turnover_rate_change_20d'] = None
                
                # 4. 计算系统性崩盘风险评分
                try:
                    # 收集市场指标
                    crash_risk_indicators = {}
                    
                    # VIX恐慌指数
                    if indicators.get('vix_level') is not None:
                        crash_risk_indicators['VIX'] = indicators['vix_level']
                    
                    # 恒指收益率
                    if hist is not None and not hist.empty:
                        if len(hist) > 1:
                            hsi_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                            crash_risk_indicators['HSI_Return_1d'] = hsi_change
                    
                    # 平均成交量比率（从换手率变化率计算中获取）
                    if hist is not None and not hist.empty and 'Volume' in hist.columns:
                        vol_ma20 = hist['Volume'].rolling(20).mean()
                        if len(vol_ma20) > 0:
                            avg_vol_ratio = hist['Volume'].iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1.0
                            crash_risk_indicators['Avg_Vol_Ratio'] = avg_vol_ratio
                    
                    # 标普500收益率
                    if us_df is not None and not us_df.empty and 'SP500_Return' in us_df.columns:
                        crash_risk_indicators['SP500_Return_1d'] = us_df['SP500_Return'].iloc[-1] * 100 if pd.notna(us_df['SP500_Return'].iloc[-1]) else 0
                    
                    # 计算系统性崩盘风险评分
                    if crash_risk_indicators:
                        crash_risk_result = us_market_data.calculate_systemic_crash_risk(crash_risk_indicators)
                        indicators['crash_risk_score'] = crash_risk_result.get('risk_score')
                        indicators['crash_risk_level'] = crash_risk_result.get('risk_level')
                        indicators['crash_risk_factors'] = crash_risk_result.get('factors', [])
                        indicators['crash_risk_recommendations'] = crash_risk_result.get('recommendations', [])
                    else:
                        indicators['crash_risk_score'] = None
                        indicators['crash_risk_level'] = None
                        indicators['crash_risk_factors'] = []
                        indicators['crash_risk_recommendations'] = []
                except Exception as e:
                    print(f"⚠️ 计算系统性崩盘风险评分失败: {e}")
                    indicators['crash_risk_score'] = None
                    indicators['crash_risk_level'] = None
                    indicators['crash_risk_factors'] = []
                    indicators['crash_risk_recommendations'] = []
                    
            except Exception as e:
                print(f"⚠️ 计算市场情绪和流动性指标失败: {e}")
                indicators['vix_level'] = None
                indicators['turnover_change_1d'] = None
                indicators['turnover_change_5d'] = None
                indicators['turnover_change_10d'] = None
                indicators['turnover_change_20d'] = None
                indicators['turnover_rate'] = None
                indicators['turnover_rate_change_5d'] = None
                indicators['turnover_rate_change_20d'] = None
            
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
                    'rsi': None,
                    'macd': None,
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
                
                # 添加评分系统信息（降级模式）
                if self.USE_SCORED_SIGNALS:
                    indicators['buildup_score'] = 0.0
                    indicators['buildup_level'] = 'none'
                    indicators['buildup_reasons'] = ''
                    indicators['distribution_score'] = 0.0
                    indicators['distribution_level'] = 'none'
                    indicators['distribution_reasons'] = ''
                else:
                    indicators['buildup_score'] = None
                    indicators['buildup_level'] = None
                    indicators['buildup_reasons'] = None
                    indicators['distribution_score'] = None
                    indicators['distribution_level'] = None
                    indicators['distribution_reasons'] = None
                
                return indicators
            else:
                return None

    def calculate_hsi_technical_indicators(self, data, us_df=None):
        """
        计算恒生指数技术指标（使用HSI专用配置）

        参数:
        - data: 股票数据
        - us_df: 美股市场数据（可选，避免重复获取）
        """
        return self._calculate_technical_indicators_core(data, asset_type='hsi', us_df=us_df)

    def calculate_technical_indicators(self, data, us_df=None):
        """
        计算技术指标（适用于个股）

        参数:
        - data: 股票数据
        - us_df: 美股市场数据（可选，避免重复获取）
        """
        return self._calculate_technical_indicators_core(data, asset_type='stock', us_df=us_df)

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

    def _read_portfolio_data(self, path='data/actual_porfolio.csv'):
        """
        读取持仓数据 CSV 文件
        
        参数:
        - path: 持仓文件路径
        
        返回:
        - list: 持仓列表，每个元素为字典，包含股票代码、名称、数量、成本价等信息
        """
        if not os.path.exists(path):
            print(f"⚠️ 持仓文件不存在: {path}")
            return []
        
        try:
            df = pd.read_csv(path, encoding='utf-8')
            if df.empty:
                print("⚠️ 持仓文件为空")
                return []
            
            portfolio = []
            for _, row in df.iterrows():
                # 尝试识别列名（支持中英文）
                stock_code = None
                lot_size = None
                cost_price = None
                lot_count = None
                
                # 查找股票代码列
                for col in df.columns:
                    if '股票号码' in col or 'stock_code' in col.lower() or 'code' in col.lower():
                        stock_code = str(row[col]).strip()
                        break
                
                # 查找每手股数列
                for col in df.columns:
                    if '一手股数' in col or 'lot_size' in col.lower():
                        lot_size = float(row[col]) if pd.notna(row[col]) else None
                        break
                
                # 查找成本价列
                for col in df.columns:
                    if '成本价' in col or 'cost_price' in col.lower() or 'cost' in col.lower():
                        cost_price = float(row[col]) if pd.notna(row[col]) else None
                        break
                
                # 查找持有手数列
                for col in df.columns:
                    if '持有手数' in col or 'lot_count' in col.lower() or 'quantity' in col.lower():
                        lot_count = int(row[col]) if pd.notna(row[col]) else None
                        break
                
                # 如果所有必要字段都存在，添加到持仓列表
                if stock_code and lot_size and cost_price and lot_count:
                    total_shares = lot_size * lot_count
                    total_cost = cost_price * total_shares
                    
                    # 获取股票名称
                    stock_name = self.stock_list.get(stock_code, stock_code)
                    
                    portfolio.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'lot_size': lot_size,
                        'cost_price': cost_price,
                        'lot_count': lot_count,
                        'total_shares': total_shares,
                        'total_cost': total_cost
                    })
                else:
                    print(f"⚠️ 跳过不完整的持仓记录: {row.to_dict()}")
            
            print(f"✅ 成功读取 {len(portfolio)} 条持仓记录")
            return portfolio
            
        except Exception as e:
            print(f"❌ 读取持仓数据失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _get_market_context(self, hsi_data):
        """
        获取市场环境信息（恒生指数）
        
        参数:
        - hsi_data: 恒生指数数据
        
        返回:
        - str: 市场环境信息字符串
        """
        if not hsi_data:
            return ""
        
        hsi_price = hsi_data.get('current_price', 0)
        hsi_change = hsi_data.get('change_1d', 0)
        return f"""
## 市场环境
- 恒生指数: {hsi_price:,.2f} ({hsi_change:+.2f}%)
"""
    
    def _get_stock_data_from_results(self, stock_code, stock_results):
        """
        从 stock_results 中获取股票数据和技术指标
        
        参数:
        - stock_code: 股票代码
        - stock_results: 股票分析结果列表
        
        返回:
        - tuple: (current_price, indicators, stock_name)
        """
        for stock_result in stock_results:
            if stock_result['code'] == stock_code:
                stock_data = stock_result.get('data', {})
                current_price = stock_data.get('current_price')
                indicators = stock_result.get('indicators', {})
                stock_name = stock_result.get('name', stock_code)
                return current_price, indicators, stock_name
        
        return None, None, None
    
    def _format_tech_info(self, indicators, include_trend=True):
        """
        格式化技术指标信息
        
        参数:
        - indicators: 技术指标字典
        - include_trend: 是否包含趋势
        
        返回:
        - str: 格式化后的技术指标信息
        """
        tech_info = []
        if indicators:
            # 趋势信息
            if include_trend:
                trend = indicators.get('trend', '未知')
                tech_info.append(f"趋势: {trend}")
            
            # 基础技术指标
            rsi = indicators.get('rsi', 0)
            macd = indicators.get('macd', 0)
            bb_position = indicators.get('bb_position', 0.5)
            
            if rsi > 0:
                tech_info.append(f"RSI: {rsi:.2f}")
            if macd != 0:
                tech_info.append(f"MACD: {macd:.4f}")
            if bb_position > 0:
                tech_info.append(f"布林带位置: {bb_position:.2%}")
            
            # 均线信息
            current_price = indicators.get('current_price', 0)
            ma20 = indicators.get('ma20', 0)
            ma50 = indicators.get('ma50', 0)
            ma200 = indicators.get('ma200', 0)
            
            if ma20 > 0 and current_price > 0:
                ma20_pct = (current_price - ma20) / ma20 * 100
                tech_info.append(f"MA20: {ma20:.2f} ({ma20_pct:+.2f}%)")
            if ma50 > 0 and current_price > 0:
                ma50_pct = (current_price - ma50) / ma50 * 100
                tech_info.append(f"MA50: {ma50:.2f} ({ma50_pct:+.2f}%)")
            if ma200 > 0 and current_price > 0:
                ma200_pct = (current_price - ma200) / ma200 * 100
                tech_info.append(f"MA200: {ma200:.2f} ({ma200_pct:+.2f}%)")
            
            # 成交量指标
            volume_ratio = indicators.get('volume_ratio', 0)
            volume_surge = indicators.get('volume_surge', False)
            volume_shrink = indicators.get('volume_shrink', False)
            
            if volume_ratio > 0:
                vol_status = ""
                if volume_surge:
                    vol_status = " (放量)"
                elif volume_shrink:
                    vol_status = " (缩量)"
                tech_info.append(f"量比: {volume_ratio:.2f}{vol_status}")
            
            # 评分系统
            tav_score = indicators.get('tav_score', 0)
            buildup_score = indicators.get('buildup_score', 0)
            distribution_score = indicators.get('distribution_score', 0)
            
            if tav_score > 0:
                tech_info.append(f"TAV评分: {tav_score:.1f}")
            if buildup_score > 0:
                tech_info.append(f"建仓评分: {buildup_score:.2f}")
            if distribution_score > 0:
                tech_info.append(f"出货评分: {distribution_score:.2f}")
            
            # 止损止盈
            stop_loss = indicators.get('stop_loss')
            take_profit = indicators.get('take_profit')
            
            if stop_loss is not None and stop_loss > 0:
                sl_pct = (stop_loss - current_price) / current_price * 100 if current_price > 0 else 0
                tech_info.append(f"止损: {stop_loss:.2f} ({sl_pct:+.2f}%)")
            if take_profit is not None and take_profit > 0:
                tp_pct = (take_profit - current_price) / current_price * 100 if current_price > 0 else 0
                tech_info.append(f"止盈: {take_profit:.2f} ({tp_pct:+.2f}%)")
            
            # ATR
            atr = indicators.get('atr', 0)
            if atr > 0:
                atr_pct = (atr / current_price * 100) if current_price > 0 else 0
                tech_info.append(f"ATR: {atr:.2f} ({atr_pct:.2f}%)")
            
            # 中期分析指标
            medium_term_score = indicators.get('medium_term_score', 0)
            if medium_term_score > 0:
                tech_info.append(f"中期评分: {medium_term_score:.1f}")
            
            ma_alignment = indicators.get('ma_alignment', 'N/A')
            if ma_alignment != 'N/A':
                tech_info.append(f"均线排列: {ma_alignment}")
            
            # 基本面指标
            fundamental_score = indicators.get('fundamental_score', 0)
            if fundamental_score > 0:
                # 根据评分设置颜色
                if fundamental_score > 60:
                    fundamental_status = "优秀"
                elif fundamental_score >= 30:
                    fundamental_status = "一般"
                else:
                    fundamental_status = "较差"
                tech_info.append(f"基本面评分: {fundamental_score:.0f}({fundamental_status})")
            
            pe_ratio = indicators.get('pe_ratio')
            if pe_ratio is not None and pe_ratio > 0:
                tech_info.append(f"PE: {pe_ratio:.2f}")
            
            pb_ratio = indicators.get('pb_ratio')
            if pb_ratio is not None and pb_ratio > 0:
                tech_info.append(f"PB: {pb_ratio:.2f}")
            
            # 新增：市场情绪和流动性指标
            vix_level = indicators.get('vix_level')
            if vix_level is not None and vix_level > 0:
                tech_info.append(f"VIX: {vix_level:.2f}")
            
            turnover_change_1d = indicators.get('turnover_change_1d')
            if turnover_change_1d is not None:
                tech_info.append(f"成交额变化1日: {turnover_change_1d:+.2f}%")
            
            turnover_change_5d = indicators.get('turnover_change_5d')
            if turnover_change_5d is not None:
                tech_info.append(f"成交额变化5日: {turnover_change_5d:+.2f}%")
            
            turnover_change_20d = indicators.get('turnover_change_20d')
            if turnover_change_20d is not None:
                tech_info.append(f"成交额变化20日: {turnover_change_20d:+.2f}%")
            
            turnover_rate = indicators.get('turnover_rate')
            if turnover_rate is not None:
                tech_info.append(f"换手率: {turnover_rate:.2f}%")
            
            turnover_rate_change_5d = indicators.get('turnover_rate_change_5d')
            if turnover_rate_change_5d is not None:
                tech_info.append(f"换手率变化5日: {turnover_rate_change_5d:+.2f}%")
            
            turnover_rate_change_20d = indicators.get('turnover_rate_change_20d')
            if turnover_rate_change_20d is not None:
                tech_info.append(f"换手率变化20日: {turnover_rate_change_20d:+.2f}%")
        
        return ', '.join(tech_info) if tech_info else 'N/A'
    
    def _get_signal_strength(self, indicators):
        """
        根据建仓和出货评分判断信号强度
        
        参数:
        - indicators: 技术指标字典
        
        返回:
        - str: 信号强度
        """
        buildup_level = indicators.get('buildup_level', 'none')
        distribution_level = indicators.get('distribution_level', 'none')
        
        if buildup_level == 'strong' and distribution_level == 'none':
            return "强烈买入"
        elif buildup_level == 'partial' and distribution_level == 'none':
            return "温和买入"
        elif distribution_level == 'strong' and buildup_level == 'none':
            return "强烈卖出"
        elif distribution_level == 'weak' and buildup_level == 'none':
            return "温和卖出"
        elif buildup_level == 'strong' and distribution_level == 'strong':
            return "多空分歧"
        else:
            return "中性"
    
    def _add_technical_signals_summary(self, prompt, stock_list, stock_results):
        """
        添加技术面信号摘要到提示词
        
        参数:
        - prompt: 提示词字符串
        - stock_list: 股票列表 [(stock_name, stock_code, ...), ...]
        - stock_results: 股票分析结果列表
        
        返回:
        - str: 添加了技术面信号摘要的提示词
        """
        prompt += """
## 今日技术面信号摘要
"""
        
        for stock_name, stock_code, trend, signal, signal_type in stock_list:
            current_price, indicators, _ = self._get_stock_data_from_results(stock_code, stock_results)
            
            if indicators:
                buildup_score = indicators.get('buildup_score', 0)
                buildup_level = indicators.get('buildup_level', 'none')
                buildup_reasons = indicators.get('buildup_reasons', '')
                distribution_score = indicators.get('distribution_score', 0)
                distribution_level = indicators.get('distribution_level', 'none')
                distribution_reasons = indicators.get('distribution_reasons', '')
                trend = indicators.get('trend', '未知')
                
                # 获取48小时智能建议
                continuous_signal = self.detect_continuous_signals_in_history_from_transactions(
                    stock_code, hours=48, min_signals=3, target_date=None
                )
                
                signal_strength = self._get_signal_strength(indicators)
                
                prompt += f"""
- {stock_name} ({stock_code}):
  * 技术趋势: {trend}
  * 信号强度: {signal_strength}
  * 建仓评分: {buildup_score:.2f} ({buildup_level})
  * 建仓原因: {buildup_reasons if buildup_reasons else '无'}
  * 出货评分: {distribution_score:.2f} ({distribution_level})
  * 出货原因: {distribution_reasons if distribution_reasons else '无'}
  * 48小时连续信号: {continuous_signal}
"""
        
        return prompt
    
    def _add_recent_transactions(self, prompt, stock_codes, hours=48):
        """
        添加最近交易记录到提示词
        
        参数:
        - prompt: 提示词字符串
        - stock_codes: 股票代码列表
        - hours: 查询小时数
        
        返回:
        - str: 添加了交易记录的提示词
        """
        prompt += f"""
## 最近{hours}小时模拟交易记录
"""
        
        try:
            df_transactions = self._read_transactions_df()
            if not df_transactions.empty:
                # 获取最近N小时的交易记录
                reference_time = pd.Timestamp.now(tz='UTC')
                start_time = reference_time - pd.Timedelta(hours=hours)
                
                # 过滤指定股票的交易记录
                recent_transactions = df_transactions[
                    (df_transactions['timestamp'] >= start_time) &
                    (df_transactions['timestamp'] <= reference_time) &
                    (df_transactions['code'].isin(stock_codes))
                ].sort_values('timestamp', ascending=False)
                
                if not recent_transactions.empty:
                    # 按股票分组
                    for stock_code in stock_codes:
                        stock_transactions = recent_transactions[recent_transactions['code'] == stock_code]
                        if not stock_transactions.empty:
                            stock_name = self.stock_list.get(stock_code, stock_code)
                            prompt += f"\n{stock_name} ({stock_code}):\n"
                            
                            for _, trans in stock_transactions.iterrows():
                                trans_type = trans.get('type', '')
                                timestamp = pd.Timestamp(trans['timestamp']).strftime('%m-%d %H:%M:%S')
                                current_price = trans.get('current_price')
                                target_price = trans.get('target_price')
                                stop_loss_price = trans.get('stop_loss_price')
                                validity_period = trans.get('validity_period')
                                reason = trans.get('reason', '')
                                
                                # 格式化交易信息
                                price_info = []
                                if pd.notna(current_price):
                                    try:
                                        price_float = float(current_price)
                                        price_info.append(f"现价:HK${price_float:.2f}")
                                    except (ValueError, TypeError):
                                        price_info.append(f"现价:{current_price}")
                                if pd.notna(target_price):
                                    try:
                                        price_float = float(target_price)
                                        price_info.append(f"目标:HK${price_float:.2f}")
                                    except (ValueError, TypeError):
                                        price_info.append(f"目标:{target_price}")
                                if pd.notna(stop_loss_price):
                                    try:
                                        price_float = float(stop_loss_price)
                                        price_info.append(f"止损:HK${price_float:.2f}")
                                    except (ValueError, TypeError):
                                        price_info.append(f"止损:{stop_loss_price}")
                                if pd.notna(validity_period):
                                    try:
                                        validity_int = int(float(validity_period))
                                        price_info.append(f"有效期:{validity_int}天")
                                    except (ValueError, TypeError):
                                        price_info.append(f"有效期:{validity_period}")
                                
                                price_info_str = " | ".join(price_info) if price_info else ""
                                prompt += f"  {timestamp} {trans_type} @ {price_info_str} ({reason})\n"
                else:
                    prompt += f"  最近{hours}小时无相关交易记录\n"
        except Exception as e:
            print(f"⚠️ 获取交易记录失败: {e}")
            prompt += f"  获取交易记录失败\n"
        
        return prompt

    def _add_systemic_crash_risk_summary(self, prompt, stock_results):
        """
        添加系统性崩盘风险评分到提示词
        
        参数:
        - prompt: 提示词字符串
        - stock_results: 股票分析结果列表
        
        返回:
        - str: 添加了系统性崩盘风险评分的提示词
        """
        if not stock_results:
            return prompt
        
        # 从第一个股票的结果中获取风险评分信息
        first_stock_result = stock_results[0] if isinstance(stock_results, list) else stock_results
        
        # 尝试获取风险评分
        crash_risk_score = None
        crash_risk_level = None
        crash_risk_factors = []
        crash_risk_recommendations = []
        
        # 从 stock_results 中查找风险评分信息
        if isinstance(stock_results, list):
            for result in stock_results:
                if isinstance(result, dict):
                    if 'indicators' in result:
                        indicators = result['indicators']
                        crash_risk_score = indicators.get('crash_risk_score')
                        crash_risk_level = indicators.get('crash_risk_level')
                        crash_risk_factors = indicators.get('crash_risk_factors', [])
                        crash_risk_recommendations = indicators.get('crash_risk_recommendations', [])
                        if crash_risk_score is not None:
                            break
        elif isinstance(stock_results, dict):
            indicators = stock_results.get('indicators', {})
            crash_risk_score = indicators.get('crash_risk_score')
            crash_risk_level = indicators.get('crash_risk_level')
            crash_risk_factors = indicators.get('crash_risk_factors', [])
            crash_risk_recommendations = indicators.get('crash_risk_recommendations', [])
        
        # 如果没有找到风险评分，返回原提示词
        if crash_risk_score is None:
            return prompt
        
        # 添加风险评分信息
        prompt += f"""
## 系统性崩盘风险评分（市场环境评估）
"""
        
        # 风险等级颜色标记
        risk_level_colors = {
            '低': '🟢',
            '中': '🟡',
            '高': '🟠',
            '极高': '🔴'
        }
        risk_level_emoji = risk_level_colors.get(crash_risk_level, '⚪')
        
        prompt += f"""
- **风险评分**: {crash_risk_score:.0f}/100
- **风险等级**: {risk_level_emoji} {crash_risk_level}
"""
        
        # 添加风险因素
        if crash_risk_factors:
            prompt += f"""
- **风险因素**:
"""
            for factor in crash_risk_factors:
                prompt += f"  * {factor}\n"
        
        # 添加建议措施
        if crash_risk_recommendations:
            prompt += f"""
- **建议措施**:
"""
            for i, recommendation in enumerate(crash_risk_recommendations[:3], 1):  # 只显示前3条建议
                prompt += f"  {i}. {recommendation}\n"
        
        # 根据风险等级添加操作建议
        if crash_risk_level == '极高':
            prompt += f"""
⚠️ **重要提醒**: 市场风险极高，建议暂停所有买入操作，优先考虑止损减仓。
"""
        elif crash_risk_level == '高':
            prompt += f"""
⚠️ **重要提醒**: 市场风险较高，建议大幅降低仓位，谨慎选股。
"""
        elif crash_risk_level == '中':
            prompt += f"""
⚠️ **重要提醒**: 市场波动加大，建议降低仓位，谨慎交易。
"""
        
        return prompt

    def _generate_analysis_prompt(self, investment_style='balanced', investment_horizon='short_term', 
                                  data_type='portfolio', stock_data=None, market_context=None, 
                                  stock_results=None, additional_info=None):
        """
        生成不同投资风格和周期的分析提示词
        
        参数:
        - investment_style: 投资风格 ('aggressive'进取型, 'balanced'稳健型, 'conservative'保守型)
        - investment_horizon: 投资周期 ('short_term'短期, 'medium_term'中期)
        - data_type: 数据类型 ('portfolio'持仓分析, 'buy_signals'买入信号分析)
        - stock_data: 股票数据列表
        - market_context: 市场环境信息
        - stock_results: 股票分析结果
        - additional_info: 额外信息（如总成本、市值等）
        
        返回:
        - str: 生成的提示词
        """
        
        # 读取新闻数据（用于新闻分析）
        news_data = {}
        news_file_path = "data/all_stock_news_records.csv"
        try:
            if os.path.exists(news_file_path):
                news_df = pd.read_csv(news_file_path)
                if not news_df.empty:
                    # 按股票代码分组新闻
                    for code, group in news_df.groupby('股票代码'):
                        news_data[code] = group.to_dict('records')
        except Exception as e:
            print(f"⚠️ 读取新闻数据失败: {e}")
        
        # 定义不同风格和周期的分析重点
        style_focus = {
            'aggressive': {
                'short_term': {
                    'role': '你是一位专业的进取型短线交易分析师，擅长捕捉日内和数天内的价格波动机会。',
                    'focus': '重点关注短期动量、成交量变化、突破信号，追求快速收益。',
                    'risk_tolerance': '风险承受能力高，可以接受较大波动以换取更高收益。',
                    'indicators': '重点关注：RSI超买超卖、MACD金叉死叉、成交量突增、价格突破关键位、ATR波动率',
                    'stop_loss': '止损位设置较紧（通常3-5%），快速止损保护本金。',
                    'take_profit': '目标价设置较近（通常5-10%），快速兑现利润。',
                    'timing': '操作时机：立即或等待突破信号，不宜长时间等待。',
                    'risks': '主要风险：短期波动剧烈、止损可能被触发、需要密切监控。'
                },
                'medium_term': {
                    'role': '你是一位专业的进取型中线投资分析师，擅长捕捉数周到数月内的趋势机会。',
                    'focus': '重点关注趋势持续性、均线排列、资金流向，追求趋势性收益。',
                    'risk_tolerance': '风险承受能力高，可以承受中期波动以换取趋势收益。',
                    'indicators': '重点关注：均线排列、均线斜率、中期趋势评分、支撑阻力位、乖离状态',
                    'stop_loss': '止损位设置适中（通常5-8%），允许一定波动空间。',
                    'take_profit': '目标价设置较远（通常15-25%），追求趋势性收益。',
                    'timing': '操作时机：等待趋势确认或回调至支撑位，不宜追高。',
                    'risks': '主要风险：趋势反转、中期调整、需要耐心持有。'
                }
            },
            'balanced': {
                'short_term': {
                    'role': '你是一位专业的稳健型短线交易分析师，注重风险收益平衡。',
                    'focus': '重点关注技术指标确认、成交量配合，追求稳健收益。',
                    'risk_tolerance': '风险承受能力中等，在控制风险的前提下追求收益。',
                    'indicators': '重点关注：RSI、MACD、成交量、布林带、短期趋势评分',
                    'stop_loss': '止损位设置合理（通常5-7%），平衡风险和收益。',
                    'take_profit': '目标价设置适中（通常8-15%），稳健兑现利润。',
                    'timing': '操作时机：等待技术指标确认或价格回调，避免追涨杀跌。',
                    'risks': '主要风险：短期震荡、止损可能被触发、需要灵活调整。'
                },
                'medium_term': {
                    'role': '你是一位专业的稳健型中线投资分析师，注重中长期价值投资。',
                    'focus': '重点关注基本面和技术面结合，追求稳健的中期收益。',
                    'risk_tolerance': '风险承受能力中等，注重风险控制和资产配置。',
                    'indicators': '重点关注：中期趋势评分、趋势健康度、可持续性、支撑阻力位、乖离状态',
                    'stop_loss': '止损位设置较宽（通常8-12%），允许中期波动。',
                    'take_profit': '目标价设置合理（通常20-30%），追求稳健的中期收益。',
                    'timing': '操作时机：等待趋势确认或回调至支撑位，分批建仓降低成本。',
                    'risks': '主要风险：中期趋势变化、基本面恶化、需要定期评估。'
                }
            },
            'conservative': {
                'short_term': {
                    'role': '你是一位专业的保守型短线交易分析师，注重本金安全。',
                    'focus': '重点关注低风险机会、确定性高的信号，追求稳健收益。',
                    'risk_tolerance': '风险承受能力低，优先保护本金，追求稳健收益。',
                    'indicators': '重点关注：RSI超卖、强支撑位、成交量萎缩、低波动率',
                    'stop_loss': '止损位设置较紧（通常2-3%），严格控制风险。',
                    'take_profit': '目标价设置较近（通常3-5%），快速兑现利润。',
                    'timing': '操作时机：等待超卖反弹或支撑位确认，避免追高。',
                    'risks': '主要风险：收益较低、机会成本、可能错过上涨机会。'
                },
                'medium_term': {
                    'role': '你是一位专业的保守型中线投资分析师，注重长期价值投资。',
                    'focus': '重点关注基本面、估值水平、长期趋势，追求稳健的长期收益。',
                    'risk_tolerance': '风险承受能力低，注重资产保值和稳健增长。',
                    'indicators': '重点关注：基本面指标（PE、PB）、估值水平、长期趋势、风险指标',
                    'stop_loss': '止损位设置较宽（通常10-15%），允许较大波动空间。',
                    'take_profit': '目标价设置较远（通常30-50%），追求长期价值增长。',
                    'timing': '操作时机：等待估值合理或长期趋势确认，分批建仓长期持有。',
                    'risks': '主要风险：长期持有期间市场变化、基本面恶化、需要耐心。'
                }
            }
        }
        
        # 获取对应风格和周期的配置
        config = style_focus.get(investment_style, {}).get(investment_horizon, style_focus['balanced']['short_term'])
        
        # 构建基础提示词
        prompt = f"""{config['role']}
{config['focus']}
{config['risk_tolerance']}

{market_context if market_context else ''}
"""
        
        # 根据数据类型添加不同的内容
        if data_type == 'portfolio' and additional_info:
            prompt += f"""
## 持仓概览
- 总投资成本: HK${additional_info.get('total_cost', 0):,.2f}
- 当前市值: HK${additional_info.get('total_current_value', 0):,.2f}
- 浮动盈亏: HK${additional_info.get('total_profit_loss', 0):,.2f} ({additional_info.get('total_profit_loss_pct', 0):+.2f}%)
- 持仓股票数量: {len(stock_data) if stock_data else 0}只

## 持仓股票详情
"""
            for i, pos in enumerate(stock_data, 1):
                position_pct = pos.get('position_pct', 0)
                stock_code = pos['stock_code']
                
                # 获取新闻摘要
                stock_news = news_data.get(stock_code, [])
                news_summary_text = ""
                if stock_news:
                    news_summary_text = "   - 新闻摘要:\n"
                    for news in stock_news[:3]:  # 只展示最近3条
                        news_summary_text += f"     * {news.get('新闻时间', '')}: {news.get('新闻标题', '')} - {news.get('简要内容', '')}\n"
                else:
                    news_summary_text = "   - 新闻摘要: 暂无相关新闻\n"
                
                prompt += f"""
{i}. {pos['stock_name']} ({pos['stock_code']})
   - 持仓占比: {position_pct:.1f}%
   - 持仓数量: {pos['total_shares']:,}股
   - 成本价: HK${pos['cost_price']:.2f}
   - 当前价格: HK${pos['current_price']:.2f}
   - 浮动盈亏: HK${pos['profit_loss']:,.2f} ({pos['profit_loss_pct']:+.2f}%)
   - 技术指标: {pos['tech_info']}
{news_summary_text}"""
        
        elif data_type == 'buy_signals' and stock_data:
            prompt += f"""
## 买入信号股票概览
- 买入信号股票数量: {len(stock_data)}只

## 买入信号股票详情
"""
            for i, stock in enumerate(stock_data, 1):
                stock_code = stock['stock_code']
                
                # 获取新闻摘要
                stock_news = news_data.get(stock_code, [])
                news_summary_text = ""
                if stock_news:
                    news_summary_text = "   - 新闻摘要:\n"
                    for news in stock_news[:3]:  # 只展示最近3条
                        news_summary_text += f"     * {news.get('新闻时间', '')}: {news.get('新闻标题', '')} - {news.get('简要内容', '')}\n"
                else:
                    news_summary_text = "   - 新闻摘要: 暂无相关新闻\n"
                
                prompt += f"""
{i}. {stock['stock_name']} ({stock['stock_code']})
   - 当前价格: HK${stock['current_price']:.2f}
   - 技术趋势: {stock['trend']}
   - 技术指标: {stock['tech_info']}
   - 信号描述: {stock['signal_description']}
{news_summary_text}"""
        
        elif data_type == 'watchlist' and stock_data:
            prompt += f"""
## 自选股概览
- 自选股数量: {len(stock_data)}只
- 分析范围: 全部26只自选股

## 自选股详情
"""
            for i, stock in enumerate(stock_data, 1):
                stock_code = stock['stock_code']
                
                # 获取新闻摘要
                stock_news = news_data.get(stock_code, [])
                news_summary_text = ""
                if stock_news:
                    news_summary_text = "   - 新闻摘要:\n"
                    for news in stock_news[:3]:  # 只展示最近3条
                        news_summary_text += f"     * {news.get('新闻时间', '')}: {news.get('新闻标题', '')} - {news.get('简要内容', '')}\n"
                else:
                    news_summary_text = "   - 新闻摘要: 暂无相关新闻\n"
                
                prompt += f"""
{i}. {stock['stock_name']} ({stock['stock_code']})
   - 当前价格: HK${stock['current_price']:.2f}
   - 技术指标: {stock['tech_info']}
{news_summary_text}"""
        
        # 添加分析要求
        prompt += f"""
## 分析框架（业界惯例）
请按照以下六层分析框架进行系统性分析：

【第一层：风险控制检查（最高优先级）】
⚠️ 必须首先检查所有股票的风险控制信号：
- 止损触发：亏损≥15%，立即全部卖出，风险等级极高
- 止盈触发：盈利≥10%，建议卖出30%，风险等级高
- Trailing Stop触发：价格从高点回撤超过2.5倍ATR，建议卖出30%，风险等级高

【第二层：市场环境评估（宏观层面）】
🌍 评估整体市场环境，判断是否适合交易：
- VIX恐慌指数：评估市场整体情绪
  * VIX < 15：市场过度乐观，需警惕回调风险，降低仓位
  * VIX 15-20：正常波动，市场情绪平稳，正常交易
  * VIX 20-30：轻度恐慌，市场波动加大，谨慎交易
  * VIX > 30：严重恐慌，通常伴随大跌，但可能存在反弹机会
- 成交额变化率：评估市场流动性
  * 正向变化率（1日/5日/20日）：资金持续流入，市场活跃，支持交易
  * 负向变化率（1日/5日/20日）：资金持续流出，市场低迷，减少交易
  * 多周期一致：1日、5日、20日变化率同向，信号更可靠
- 换手率变化率：评估市场关注度
  * 换手率上升+换手率变化率正向：关注度提升，流动性增强，适合交易
  * 换手率下降+换手率变化率负向：关注度下降，流动性减弱，观望为主
  * 换手率异常波动：可能预示重大消息或趋势转折，提高警惕
- 系统性崩盘风险评分：综合评估市场整体风险
  * 评分范围：0-100分，分数越高风险越大
  * 风险等级：
    - 低风险（<40分）：市场环境良好，正常交易
    - 中风险（40-60分）：市场波动加大，谨慎交易
    - 高风险（60-80分）：市场风险较高，降低仓位
    - 极高风险（≥80分）：市场风险极高，暂停交易
  * 风险因素：包括VIX恐慌、指数跌幅、成交额萎缩、美股联动等
  * 建议措施：根据风险等级提供具体的操作建议

【第三层：基本面质量评估（长期价值）】
🔍 评估股票的长期投资价值：
- 基本面评分评估：
  * 基本面评分>60：优质股票，大幅提升建仓信号可靠性，优先配置
  * 基本面评分40-60：良好股票，提升建仓信号可靠性，正常配置
  * 基本面评分20-40：一般股票，建仓信号需谨慎，少量配置
  * 基本面评分<20：差股票，建仓信号不可靠，避免配置
- 估值水平（PE、PB）：
  * 低估值（PE<15, PB<1）：安全边际高，适合长期持有
  * 合理估值（PE 15-25, PB 1-2）：估值合理，正常交易
  * 高估值（PE>25, PB>2）：估值偏高，谨慎交易

【第四层：技术面分析（短期趋势）】
📈 评估股票的短期技术面：
- 多周期趋势验证：
  * 多周期趋势评分>20：趋势向上，建仓信号可靠
  * 多周期趋势评分<-20：趋势向下，建仓信号谨慎
  * 多周期趋势评分-20到20：震荡趋势，建议观望
- 相对强度验证：
  * 多周期相对强度评分>20：跑赢恒指，优先选择
  * 多周期相对强度评分<-20：跑输恒指，谨慎对待
- 技术指标协同：
  * RSI+MACD+布林带+成交量比率+CMF：至少3个指标同向才可靠
  * 关注MACD信号、布林带突破、OBV趋势等关键信号

【第五层：信号识别（交易时机）】
🟢 建仓信号筛选：
- 建仓级别=strong（评分≥5.0）：强烈建仓信号，结合基本面和技术面确认
- 建仓级别=partial（3.0≤评分<5.0）：部分建仓信号，谨慎观察
- 结合市场环境、基本面评分和技术面分析调整仓位

🔴 出货信号筛选：
- 出货级别=strong（评分≥5.0）：强烈出货信号，建议卖出60-100%
- 出货级别=weak（3.0≤评分<5.0）：弱出货信号，建议卖出30-60%
- 结合市场环境、基本面评分和技术面分析调整卖出比例

【第六层：综合评分与决策（最终判断）】
⭐ 综合所有维度进行最终决策：
- 综合评分>70分：强烈推荐，建议仓位50-70%
- 综合评分50-70分：推荐，建议仓位30-50%
- 综合评分30-50分：观望，建议仓位10-30%
- 综合评分<30分：不推荐，建议仓位0-10%
- 综合评分构成：建仓评分25% + 多周期趋势20% + 相对强度15% + 基本面15% + 新闻影响15% + 技术指标协同10%

【新闻分析（辅助）】
📰 评估新闻对股价的影响（仅供参考，不改变核心技术分析决策）：
- 新闻分析原则：
  * 新闻作为辅助参考，不改变核心技术分析决策
  * 如果出现重大负面新闻（如财务造假、监管处罚等），建议观望
  * 如果出现重大正面新闻（如重大并购、业绩超预期等），可适当增加仓位
  * 投资者类型权重：
    - 进取型投资者：新闻权重10%
    - 稳健型投资者：新闻权重20%
    - 保守型投资者：新闻权重30%

## 针对{'短期' if investment_horizon == 'short_term' else '中期'}投资者的重点调整：

"""
        
        # 根据投资周期添加不同的重点调整
        if investment_horizon == 'short_term':
            prompt += f"""
📊 短期投资者重点关注：
- **市场环境**：优先关注VIX短期变化、成交额1日/5日变化率，捕捉短期情绪和流动性变化
- **技术面**：重点关注RSI超买超卖、MACD金叉死叉、成交量突增、价格突破关键位、ATR波动率
- **信号识别**：重点关注短期建仓/出货信号，快速响应市场变化
- **止损策略**：止损位设置较紧（3-5%），快速止损保护本金
- **操作时机**：立即或等待突破信号，不宜长时间等待
- **风险提示**：短期波动剧烈、止损可能被触发、需要密切监控
"""
        else:  # medium_term
            prompt += f"""
📊 中期投资者重点关注：
- **市场环境**：优先关注VIX中期趋势、成交额5日/20日变化率，判断中期市场趋势
- **基本面**：重点关注基本面评分、估值水平（PE、PB）、长期价值，选择优质标的
- **技术面**：重点关注均线排列、均线斜率、中期趋势评分、支撑阻力位、乖离状态
- **信号识别**：重点关注中期建仓/出货信号，结合基本面和技术面确认
- **止损策略**：止损位设置较宽（8-12%），允许中期波动
- **操作时机**：等待趋势确认或回调至支撑位，分批建仓降低成本
- **风险提示**：中期趋势变化、基本面恶化、需要定期评估
"""
        
        prompt += f"""
## 分析重点
- {config['indicators']}

## 评分体系（业界标准）

**综合评分计算公式：**
```
综合评分 = 建仓评分×25% + 多周期趋势×20% + 相对强度×15% 
         + 基本面×15% + 新闻影响×15% + 技术指标协同×10%
```

**评分构成说明：**
- **建仓评分（0-100）**：基于价格位置、成交量比率、MACD信号、RSI指标等
- **多周期趋势（-100到+100）**：短期、中期、长期趋势的综合评分
- **相对强度（-100到+100）**：相对于恒生指数的表现
- **基本面（0-100）**：PE、PB、基本面评分等
- **新闻影响（-50到+50）**：新闻情感对股价的影响（根据投资者类型调整权重）
- **技术指标协同（0-100）**：多个技术指标的一致性

**分类标准：**
- **买入机会**：
  * 综合评分 > 70分：强烈推荐（建议仓位50-70%）
  * 综合评分 50-70分：推荐（建议仓位30-50%）
  * 综合评分 45-50分：观察列表（建议仓位10-30%）
- **卖出机会**：
  * 综合评分 < 30分：强烈推荐卖出（建议减仓50-100%）
  * 综合评分 30-40分：推荐卖出（建议减仓30-50%）
  * 综合评分 40-45分：观察列表（建议减仓10-30%）
- **观望机会**：
  * 综合评分 45-50分：震荡整理，建议观望
  * 综合评分 < 45分且无卖出信号：趋势不明，建议观望

## 组合约束（风险控制）

**仓位管理原则：**
1. **单只股票最大仓位**：不超过30%（强烈推荐）或20%（推荐）
2. **单一行业最大仓位**：不超过30%（避免过度集中）
3. **买入股票数量限制**：不超过5只（分散风险）
4. **总体仓位控制**：根据市场环境调整
   * VIX < 15：最大总仓位70%
   * VIX 15-20：最大总仓位60%
   * VIX 20-30：最大总仓位40%
   * VIX > 30：最大总仓位20%

**选股优先级：**
1. 优先选择基本面评分 > 60的股票
2. 优先选择相对强度评分 > 0的股票
3. 避免高相关性股票同时重仓

## 分析要求

**重要：必须对全部26只自选股逐一进行详细分析，不得遗漏任何一只股票！**

请基于以上信息，对全部26只自选股进行买卖建议分析：

**分析策略：**
1. **计算每只股票的综合评分**：按照评分体系计算
2. **逐个分析每只股票**：对每只股票进行独立分析，提供详细的投资建议
3. **应用组合约束**：考虑仓位和行业集中度限制
4. **分类输出**：按买入/卖出/观察/观望分类展示

**输出格式要求：**

### 🟢 买入机会推荐（强烈推荐/推荐）

对综合评分 > 50分的股票，请提供：
1. **股票名称与代码**：[股票名称] ([股票代码])
2. **综合评分**：XX分（建仓评分XX + 多周期趋势XX + 相对强度XX + 基本面XX + 新闻影响XX + 技术指标协同XX）
3. **推荐理由**：基于技术面、基本面、交易信号的综合分析
4. **操作建议**：买入（建议仓位比例，如：30%仓位）
5. **价格指引**：
   - 建议买入价：HK$XX.XX（或基于当前价格的百分比）
   - 止损位：HK$XX.XX（{config['stop_loss']}）
   - 目标价：HK$XX.XX（{config['take_profit']}）
6. **操作时机**：{config['timing']}
7. **风险提示**：{config['risks']}
8. **行业分类**：所属行业（用于评估行业集中度）

### 🔴 卖出机会推荐（强烈推荐/推荐）

对综合评分 < 40分的股票，请提供：
1. **股票名称与代码**：[股票名称] ([股票代码])
2. **综合评分**：XX分（建仓评分XX + 多周期趋势XX + 相对强度XX + 基本面XX + 新闻影响XX + 技术指标协同XX）
3. **推荐理由**：基于技术面、基本面、交易信号的综合分析
4. **操作建议**：卖出（建议卖出比例，如：清仓/减仓50%）
5. **价格指引**：
   - 建议卖出价：HK$XX.XX（或基于当前价格的百分比）
   - 止损位（如适用）：HK$XX.XX
6. **操作时机**：{config['timing']}
7. **风险提示**：{config['risks']}

### 🔶 观察列表（接近交易机会）

对综合评分 40-50分的股票，请提供：
1. **股票名称与代码**：[股票名称] ([股票代码])
2. **综合评分**：XX分（建仓评分XX + 多周期趋势XX + 相对强度XX + 基本面XX + 新闻影响XX + 技术指标协同XX）
3. **推荐理由**：基于技术面、基本面、交易信号的综合分析
4. **操作建议**：观察/观望
5. **观察要点**：需要关注的关键指标变化
6. **潜在机会**：可能变为买入/卖出的条件
7. **风险提示**：{config['risks']}
8. **行业分类**：所属行业

### 🟡 观望建议（无明确交易机会）

对综合评分 45-50分且无明确交易信号的股票，请提供：
1. **股票名称与代码**：[股票名称] ([股票代码])
2. **综合评分**：XX分（建仓评分XX + 多周期趋势XX + 相对强度XX + 基本面XX + 新闻影响XX + 技术指标协同XX）
3. **推荐理由**：为什么建议观望（技术面、基本面分析）
4. **操作建议**：观望
5. **关键指标监控**：需要关注哪些指标变化
6. **风险提示**：{config['risks']}
7. **行业分类**：所属行业

**分析原则：**
- **必须分析全部26只自选股，不得遗漏任何一只**
- 按照评分体系客观评估，避免主观偏见
- 即使是观望股票，也要提供详细的分析理由
- 考虑行业集中度，避免过度集中
- {config['focus']}
- 以简洁、专业的语言回答，重点突出可操作的建议，避免模糊表述

**输出示例：**
```
🟢 买入机会推荐（强烈推荐）

1. 腾讯控股 (0700.HK)
   - 综合评分：75分（建仓评分80 + 多周期趋势70 + 相对强度75 + 基本面65 + 新闻影响70 + 技术指标协同75）
   - 推荐理由：MACD金叉、RSI超卖反弹、成交量放大、基本面评分65、跑赢恒指
   - 操作建议：买入（建议30%仓位）
   - 价格指引：
     * 建议买入价：HK$320.00
     * 止损位：HK$300.00（-6.25%）
     * 目标价：HK$380.00（+18.75%）
   - 操作时机：立即或等待回调至320港元附近
   - 风险提示：短期波动较大，需密切关注VIX变化
   - 行业分类：科技

2. 美团-W (3690.HK)
   - 综合评分：68分（建仓评分70 + 多周期趋势65 + 相对强度70 + 基本面60 + 新闻影响65 + 技术指标协同70）
   - 推荐理由：技术面反弹、基本面良好、行业景气度提升
   - 操作建议：买入（建议25%仓位）
   - 价格指引：
     * 建议买入价：HK$150.00
     * 止损位：HK$135.00（-10.00%）
     * 目标价：HK$180.00（+20.00%）
   - 操作时机：等待回调至150港元附近
   - 风险提示：行业竞争加剧，需关注政策变化
   - 行业分类：科技

🔴 卖出机会推荐（推荐）

1. 建设银行 (0939.HK)
   - 综合评分：35分（建仓评分30 + 多周期趋势40 + 相对强度35 + 基本面45 + 新闻影响30 + 技术指标协同35）
   - 推荐理由：MACD死叉、RSI超买、获利回吐压力、相对表现较弱
   - 操作建议：减仓50%
   - 价格指引：
     * 建议卖出价：HK$5.80
   - 操作时机：立即执行
   - 风险提示：中期趋势可能反转，利率环境不利
   - 行业分类：银行

🔶 观察列表（接近交易机会）

1. 小米集团-W (1810.HK)
   - 综合评分：48分（建仓评分45 + 多周期趋势50 + 相对强度48 + 基本面50 + 新闻影响48 + 技术指标协同47）
   - 推荐理由：MACD即将金叉、RSI接近超卖区、成交量开始放大、基本面稳健
   - 操作建议：观察
   - 观察要点：等待MACD金叉确认，关注成交量是否持续放大
   - 潜在机会：若MACD金叉且成交量放大，综合评分可能突破50分，可考虑买入
   - 风险提示：需确认技术信号是否持续，避免追高
   - 行业分类：科技

🟡 观望建议（无明确交易机会）

1. 汇丰银行 (0005.HK)
   - 综合评分：46分（建仓评分45 + 多周期趋势48 + 相对强度45 + 基本面50 + 新闻影响42 + 技术指标协同46）
   - 推荐理由：技术指标中性，处于震荡区间，缺乏明确方向；基本面稳健但增长有限
   - 操作建议：观望
   - 关键指标监控：关注MACD是否出现金叉/死叉信号，观察突破关键阻力位
   - 风险提示：受利率环境影响较大，需关注美联储政策变化
   - 行业分类：银行

2. 中国移动 (0941.HK)
   - 综合评分：47分（建仓评分48 + 多周期趋势45 + 相对强度50 + 基本面55 + 新闻影响40 + 技术指标协同44）
   - 推荐理由：基本面稳健，高股息率，但技术面缺乏突破信号；相对恒指表现平稳
   - 操作建议：观望
   - 关键指标监控：关注成交量变化，观察是否突破长期阻力位
   - 风险提示：作为防守型股票，适合长期持有但短期机会有限
   - 行业分类：公用事业

3. 友邦保险 (1299.HK)
   - 综合评分：45分（建仓评分42 + 多周期趋势48 + 相对强度46 + 基本面48 + 新闻影响38 + 技术指标协同46）
   - 推荐理由：基本面良好，但受保险行业整体环境影响，技术面震荡整理
   - 操作建议：观望
   - 关键指标监控：关注行业政策变化，观察利率走势对估值的影响
   - 风险提示：保险行业受宏观经济影响较大，需关注市场风险偏好
   - 行业分类：保险

4. 中芯国际 (0981.HK)
   - 综合评分：43分（建仓评分40 + 多周期趋势42 + 相对强度40 + 基本面45 + 新闻影响35 + 技术指标协同46）
   - 推荐理由：半导体行业景气度波动，技术面缺乏明确方向；基本面良好但受行业周期影响
   - 操作建议：观望
   - 关键指标监控：关注半导体行业景气度指标，观察全球需求变化
   - 风险提示：行业周期性强，受全球供应链影响较大
   - 行业分类：半导体

5. 华虹半导体 (1347.HK)
   - 综合评分：42分（建仓评分38 + 多周期趋势40 + 相对强度38 + 基本面42 + 新闻影响35 + 技术指标协同48）
   - 推荐理由：与中芯国际类似，受行业周期影响，技术面震荡整理
   - 操作建议：观望
   - 关键指标监控：关注同行业公司表现，观察行业景气度变化
   - 风险提示：作为小市值半导体公司，波动性较大
   - 行业分类：半导体

组合约束检查：
- 单只股票最大仓位：30% ✅
- 单一行业最大仓位：55%（科技55% > 30%）⚠️ 超过限制，建议降低科技股仓位
- 买入股票数量：2只 ✅
- 总仓位：55% ✅（当前市场环境VIX 18，允许最大总仓位60%）
```"""
        
        return prompt

    def _analyze_portfolio_with_llm(self, stock_results, hsi_data=None):
        """
        使用大模型分析全部自选股，生成买入/卖出建议
        
        参数:
        - stock_results: 股票分析结果列表
        - hsi_data: 恒生指数数据（可选）
        
        返回:
        - str: 大模型生成的分析报告
        """
        try:
            # 导入大模型服务
            from llm_services.qwen_engine import chat_with_llm
            
            # 构建自选股分析数据
            stock_analysis = []
            
            for stock_code, stock_name in self.stock_list.items():
                # 从 stock_results 中获取当前价格和技术指标
                current_price, indicators, _ = self._get_stock_data_from_results(stock_code, stock_results)
                
                if current_price is None:
                    print(f"⚠️ 无法获取 {stock_name} ({stock_code}) 的当前价格")
                    continue
                
                stock_analysis.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'current_price': current_price,
                    'tech_info': self._format_tech_info(indicators, include_trend=True)
                })
            
            if not stock_analysis:
                return None
            
            # 获取市场环境
            market_context = self._get_market_context(hsi_data)
            
            # 准备股票列表，包含正确的趋势和信号信息
            stock_list = []
            for stock in stock_analysis:
                stock_code = stock['stock_code']
                # 从 stock_results 中获取趋势和信号信息
                _, indicators, _ = self._get_stock_data_from_results(stock_code, stock_results)
                if indicators:
                    trend = indicators.get('trend', '未知')
                    # 获取最近的一个信号
                    recent_buy_signals = indicators.get('recent_buy_signals', [])
                    recent_sell_signals = indicators.get('recent_sell_signals', [])
                    signal = recent_buy_signals[0] if recent_buy_signals else (recent_sell_signals[0] if recent_sell_signals else None)
                    signal_type = '买入' if recent_buy_signals else ('卖出' if recent_sell_signals else '无')
                else:
                    trend = '未知'
                    signal = None
                    signal_type = '无'
                stock_list.append((stock['stock_name'], stock['stock_code'], trend, signal, signal_type))
            stock_codes = [stock['stock_code'] for stock in stock_analysis]

            # 配置开关：是否生成所有四种分析风格
            # True = 生成全部四种（进取型短期、稳健型短期、稳健型中期、保守型中期）
            # False = 只生成两种（稳健型短期、稳健型中期）
            ENABLE_ALL_ANALYSIS_STYLES = False

            # 定义所有可用的分析风格
            all_analysis_styles = [
                ('aggressive', 'short_term', '🎯 进取型短期分析（日内/数天）'),
                ('balanced', 'short_term', '⚖️ 稳健型短期分析（日内/数天）'),
                ('balanced', 'medium_term', '📊 稳健型中期分析（数周-数月）'),
                ('conservative', 'medium_term', '🛡️ 保守型中期分析（数周-数月）')
            ]

            # 根据配置开关选择要生成的分析风格
            if ENABLE_ALL_ANALYSIS_STYLES:
                analysis_styles = all_analysis_styles
            else:
                # 只生成稳健型短期和稳健型中期
                analysis_styles = [
                    ('balanced', 'short_term', '⚖️ 稳健型短期分析（日内/数天）'),
                    ('balanced', 'medium_term', '📊 稳健型中期分析（数周-数月）')
                ]
            
            all_analysis = []
            
            for style, horizon, title in analysis_styles:
                print(f"🤖 正在生成{title}...")
                
                # 生成基础提示词
                prompt = self._generate_analysis_prompt(
                    investment_style=style,
                    investment_horizon=horizon,
                    data_type='watchlist',
                    stock_data=stock_analysis,
                    market_context=market_context,
                    additional_info={}
                )
                
                # 添加技术面信号摘要
                prompt = self._add_technical_signals_summary(prompt, stock_list, stock_results)
                
                # 添加系统性崩盘风险评分
                prompt = self._add_systemic_crash_risk_summary(prompt, stock_results)
                
                # 添加最近48小时模拟交易记录
                prompt = self._add_recent_transactions(prompt, stock_codes, hours=48)
                
                # 调用大模型
                style_analysis = chat_with_llm(prompt, enable_thinking=False)
                
                # 添加标题（使用简洁的Markdown格式）
                all_analysis.append(f"\n\n### 📊 {title}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{style_analysis}")
                
                print(f"✅ {title}完成")
            
            # 直接返回大模型分析内容
            return ''.join(all_analysis)
            
        except Exception as e:
            print(f"❌ 大模型自选股分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_buy_signals_with_llm(self, buy_signals, stock_results, hsi_data=None):
        """
        使用大模型分析买入信号股票
        
        参数:
        - buy_signals: 买入信号列表 [(stock_name, stock_code, trend, signal, signal_type), ...]
        - stock_results: 股票分析结果列表
        - hsi_data: 恒生指数数据（可选）
        
        返回:
        - str: 大模型生成的分析报告
        """
        if not buy_signals:
            return None
        
        try:
            # 导入大模型服务
            from llm_services.qwen_engine import chat_with_llm
            
            # 获取市场环境
            market_context = self._get_market_context(hsi_data)
            
            # 构建买入信号股票分析数据
            buy_signal_analysis = []
            
            for stock_name, stock_code, trend, signal, signal_type in buy_signals:
                # 从 stock_results 中获取当前价格和技术指标
                current_price, indicators, _ = self._get_stock_data_from_results(stock_code, stock_results)
                
                if current_price is None:
                    print(f"⚠️ 无法获取 {stock_name} ({stock_code}) 的当前价格")
                    continue
                
                # 获取信号描述
                signal_description = signal.get('description', '') if isinstance(signal, dict) else (str(signal) if signal is not None else '')
                
                buy_signal_analysis.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'current_price': current_price,
                    'trend': trend,
                    'tech_info': self._format_tech_info(indicators, include_trend=False),
                    'signal_description': signal_description
                })
            
            if not buy_signal_analysis:
                return None
            
            # 构建大模型提示词
            prompt = f"""你是一位专业的港股投资分析师。请根据以下买入信号股票信息、技术指标和交易记录，提供详细的投资分析和建议。

{market_context}
## 买入信号股票概览
- 买入信号股票数量: {len(buy_signal_analysis)}只

## 买入信号股票详情
"""
            for i, stock in enumerate(buy_signal_analysis, 1):
                prompt += f"""
{i}. {stock['stock_name']} ({stock['stock_code']})
   - 当前价格: HK${stock['current_price']:.2f}
   - 技术趋势: {stock['trend']}
   - 技术指标: {stock['tech_info']}
   - 信号描述: {stock['signal_description']}
"""
            
            # 添加技术面信号摘要
            prompt = self._add_technical_signals_summary(prompt, buy_signals, stock_results)
            
            # 添加系统性崩盘风险评分
            prompt = self._add_systemic_crash_risk_summary(prompt, stock_results)
            
            # 添加最近48小时模拟交易记录
            stock_codes = [stock['stock_code'] for stock in buy_signal_analysis]
            prompt = self._add_recent_transactions(prompt, stock_codes, hours=48)
            
            prompt += """
## 分析要求
请基于以上信息，对每只买入信号股票提供独立的投资分析和建议：

对于每只股票，请提供：

1. **操作建议**
   - 明确建议：买入/持有/观望
   - 具体的操作理由（基于技术面、基本面、交易信号）

2. **价格指引**
   - 建议的止损位（基于当前价格的百分比或具体价格）
   - 建议的目标价（基于当前价格的百分比或具体价格）

3. **操作时机**
   - 建议操作时机（立即/等待突破/等待回调）

4. **风险提示**
   - 该股票的主要风险点
   - 需要关注的关键指标

请以简洁、专业的语言回答，针对每只股票单独分析，重点突出可操作的建议，避免模糊表述。"""
            
            print("🤖 正在使用大模型分析买入信号股票...")
            analysis_result = chat_with_llm(prompt, enable_thinking=False)
            print("✅ 大模型分析完成")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 大模型买入信号分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _markdown_to_html(self, markdown_text):
        """
        将Markdown文本转换为HTML格式
        
        参数:
        - markdown_text: Markdown格式的文本
        
        返回:
        - str: HTML格式的文本
        """
        if not markdown_text:
            return ""
        
        try:
            # 尝试导入markdown库
            import markdown
            # 使用markdown库转换
            html = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
            return html
        except ImportError:
            # 如果没有markdown库，使用简单的转换
            return self._simple_markdown_to_html(markdown_text)
    
    def _simple_markdown_to_html(self, markdown_text):
        """
        简单的Markdown到HTML转换器（当markdown库不可用时使用）
        
        参数:
        - markdown_text: Markdown格式的文本
        
        返回:
        - str: HTML格式的文本
        """
        if not markdown_text:
            return ""
        
        lines = markdown_text.split('\n')
        html_lines = []
        in_list = False
        in_code_block = False
        
        for line in lines:
            # 代码块
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    html_lines.append('<pre><code>')
                else:
                    html_lines.append('</code></pre>')
                continue
            
            if in_code_block:
                html_lines.append(f'{line}\n')
                continue
            
            # 标题
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('#### '):
                html_lines.append(f'<h4>{line[5:]}</h4>')
            # 列表
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line.strip()[2:]}</li>')
            elif line.strip().startswith('1. ') or line.strip().startswith('2. ') or line.strip().startswith('3. ') or line.strip().startswith('4. ') or line.strip().startswith('5. '):
                if not in_list:
                    html_lines.append('<ol>')
                    in_list = True
                # 提取数字和内容
                parts = line.strip().split('. ', 1)
                if len(parts) == 2:
                    html_lines.append(f'<li>{parts[1]}</li>')
            elif line.strip() == '':
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('<br>')
            # 粗体
            else:
                processed_line = line.replace('**', '<strong>').replace('__', '<strong>')
                # 斜体
                processed_line = processed_line.replace('*', '<em>').replace('_', '<em>')
                html_lines.append(f'<p>{processed_line}</p>')
        
        if in_list:
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)

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
                    target_dt = pd.to_datetime(target_date, utc=True)
                else:
                    target_dt = pd.to_datetime(target_date, utc=True)
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
                target_dt = pd.to_datetime(target_date, utc=True)
            else:
                target_dt = pd.to_datetime(target_date, utc=True)
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
            # 注意：这里返回分数形式（例如 0.05 表示 5%），与 VaR 保持一致
            es_percentage = abs(es_value)
            
            # 计算ES货币值
            es_amount = None
            if position_value is not None and position_value > 0:
                es_amount = position_value * es_percentage
            
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

        # 使用公共方法获取最新的止损价和目标价
        latest_stop_loss, latest_target_price = self._get_latest_stop_loss_target(stock_data['symbol'], target_date)

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
        
        # 使用公共方法获取最新的止损价和目标价
        latest_stop_loss, latest_target_price = self._get_latest_stop_loss_target(stock_data['symbol'], target_date)

        # 使用公共方法获取趋势颜色样式
        trend_color_style = self._get_trend_color_style(trend)

        # 添加ATR信息
        html += f"""
                <tr>
                    <td>ATR (14日)</td>
                    <td>{atr:.2f}</td>
                </tr>
        """

        # 添加ATR计算的止损价和止盈价
        if atr > 0 and stock_data.get('current_price'):
            try:
                current_price = float(stock_data['current_price'])
                # 使用1.5倍ATR作为默认止损距离
                atr_stop_loss = current_price - (atr * 1.5)
                # 使用3倍ATR作为默认止盈距离（基于2:1的风险收益比）
                atr_take_profit = current_price + (atr * 3.0)
                html += f"""
                <tr>
                    <td>ATR止损价(1.5x)</td>
                    <td>{atr_stop_loss:,.2f}</td>
                </tr>
                <tr>
                    <td>ATR止盈价(3x)</td>
                    <td>{atr_take_profit:,.2f}</td>
                </tr>
            """
            except (ValueError, TypeError):
                pass

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

        html += f"""
                <tr>
                    <td>成交量</td>
                    <td>{stock_data['volume']:,.0f}</td>
                </tr>
        """

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
                """

        # 添加VaR信息
        var_ultra_short = indicators.get('var_ultra_short_term')
        var_ultra_short_amount = indicators.get('var_ultra_short_term_amount')
        var_short = indicators.get('var_short_term')
        var_short_amount = indicators.get('var_short_term_amount')
        var_medium_long = indicators.get('var_medium_long_term')
        var_medium_long_amount = indicators.get('var_medium_long_term_amount')
        
        if var_ultra_short is not None:
            var_amount_display = f" (HK${var_ultra_short_amount:.2f})" if var_ultra_short_amount is not None else ""
            html += f"""
                <tr>
                    <td>1日VaR (95%)</td>
                    <td>{var_ultra_short:.2%}{var_amount_display}</td>
                </tr>
            """
        
        if var_short is not None:
            var_amount_display = f" (HK${var_short_amount:.2f})" if var_short_amount is not None else ""
            html += f"""
                <tr>
                    <td>5日VaR (95%)</td>
                    <td>{var_short:.2%}{var_amount_display}</td>
                </tr>
            """
        
        if var_medium_long is not None:
            var_amount_display = f" (HK${var_medium_long_amount:.2f})" if var_medium_long_amount is not None else ""
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
                    es_1d_percentage = es_1d['percentage'] if es_1d else None
                    es_1d_amount = es_1d['amount'] if es_1d else None
                    es_amount_display = f" (HK${es_1d_amount:.2f})" if es_1d_amount is not None else ""
                    html += f"""
                        <tr>
                            <td>1日ES (95%)</td>
                            <td>{es_1d_percentage:.2%}{es_amount_display}</td>
                        </tr>
                    """
                
                if es_5d is not None:
                    es_5d_percentage = es_5d['percentage'] if es_5d else None
                    es_5d_amount = es_5d['amount'] if es_5d else None
                    es_amount_display = f" (HK${es_5d_amount:.2f})" if es_5d_amount is not None else ""
                    html += f"""
                        <tr>
                            <td>5日ES (95%)</td>
                            <td>{es_5d_percentage:.2%}{es_amount_display}</td>
                        </tr>
                    """
                
                if es_20d is not None:
                    es_20d_percentage = es_20d['percentage'] if es_20d else None
                    es_20d_amount = es_20d['amount'] if es_20d else None
                    es_amount_display = f" (HK${es_20d_amount:.2f})" if es_20d_amount is not None else ""
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
        
        # 添加基本面指标
        fundamental_score = indicators.get('fundamental_score', None)
        pe_ratio = indicators.get('pe_ratio', None)
        pb_ratio = indicators.get('pb_ratio', None)
        
        # 基本面评分
        if fundamental_score is not None:
            if fundamental_score > 60:
                fundamental_color = "color: green; font-weight: bold;"
                fundamental_status = "优秀"
            elif fundamental_score >= 30:
                fundamental_color = "color: orange; font-weight: bold;"
                fundamental_status = "一般"
            else:
                fundamental_color = "color: red; font-weight: bold;"
                fundamental_status = "较差"
            
            html += f"""
                <tr>
                    <td>基本面评分</td>
                    <td><span style="{fundamental_color}">{fundamental_score:.0f}</span> <span style="font-size: 0.8em; color: #666;">({fundamental_status})</span></td>
                </tr>
            """
        
        # PE（市盈率）
        if pe_ratio is not None and pe_ratio > 0:
            pe_color = "color: green;" if pe_ratio < 15 else "color: orange;" if pe_ratio < 25 else "color: red;"
            html += f"""
                <tr>
                    <td>PE（市盈率）</td>
                    <td><span style="{pe_color}">{pe_ratio:.2f}</span></td>
                </tr>
            """
        
        # PB（市净率）
        if pb_ratio is not None and pb_ratio > 0:
            pb_color = "color: green;" if pb_ratio < 1.5 else "color: orange;" if pb_ratio < 3 else "color: red;"
            html += f"""
                <tr>
                    <td>PB（市净率）</td>
                    <td><span style="{pb_color}">{pb_ratio:.2f}</span></td>
                </tr>
            """
        
        # 添加中期评估指标
        # 均线排列
        ma_alignment = indicators.get('ma_alignment', None)
        if ma_alignment is not None and ma_alignment != 'N/A' and ma_alignment != '数据不足':
            ma_alignment_color = "color: green; font-weight: bold;" if ma_alignment == '多头排列' else "color: red; font-weight: bold;" if ma_alignment == '空头排列' else "color: orange; font-weight: bold;"
            html += f"""
                <tr>
                    <td>均线排列</td>
                    <td><span style="{ma_alignment_color}">{ma_alignment}</span></td>
                </tr>
            """
        
        # 均线斜率
        ma20_slope = indicators.get('ma20_slope', None)
        ma20_slope_trend = indicators.get('ma20_slope_trend', None)
        if ma20_slope is not None and ma20_slope_trend is not None:
            ma20_slope_color = "color: green; font-weight: bold;" if ma20_slope_trend == '上升' else "color: red; font-weight: bold;" if ma20_slope_trend == '下降' else "color: #666;"
            html += f"""
                <tr>
                    <td>MA20斜率</td>
                    <td><span style="{ma20_slope_color}">{ma20_slope:.4f}</span> ({ma20_slope_trend})</td>
                </tr>
            """
        
        ma50_slope = indicators.get('ma50_slope', None)
        ma50_slope_trend = indicators.get('ma50_slope_trend', None)
        if ma50_slope is not None and ma50_slope_trend is not None:
            ma50_slope_color = "color: green; font-weight: bold;" if ma50_slope_trend == '上升' else "color: red; font-weight: bold;" if ma50_slope_trend == '下降' else "color: #666;"
            html += f"""
                <tr>
                    <td>MA50斜率</td>
                    <td><span style="{ma50_slope_color}">{ma50_slope:.4f}</span> ({ma50_slope_trend})</td>
                </tr>
            """
        
        # 乖离率
        ma_deviation_avg = indicators.get('ma_deviation_avg', None)
        if ma_deviation_avg is not None and ma_deviation_avg != 0:
            deviation_color = "color: red; font-weight: bold;" if abs(ma_deviation_avg) > 5 else "color: orange; font-weight: bold;" if abs(ma_deviation_avg) > 3 else "color: #666;"
            html += f"""
                <tr>
                    <td>均线乖离率</td>
                    <td><span style="{deviation_color}">{ma_deviation_avg:.2f}%</span></td>
                </tr>
            """
        
        # 支撑阻力位
        nearest_support = indicators.get('nearest_support', None)
        nearest_resistance = indicators.get('nearest_resistance', None)
        if nearest_support is not None:
            support_pct = ((current_price - nearest_support) / current_price * 100) if current_price > 0 else 0
            html += f"""
                <tr>
                    <td>最近支撑位</td>
                    <td>{nearest_support:.2f} (距离{support_pct:.2f}%)</td>
                </tr>
            """
        
        if nearest_resistance is not None:
            resistance_pct = ((nearest_resistance - current_price) / current_price * 100) if current_price > 0 else 0
            html += f"""
                <tr>
                    <td>最近阻力位</td>
                    <td>{nearest_resistance:.2f} (距离{resistance_pct:.2f}%)</td>
                </tr>
            """
        
        # 相对强弱
        relative_strength = indicators.get('relative_strength', None)
        if relative_strength is not None:
            rs_color = "color: green; font-weight: bold;" if relative_strength > 0 else "color: red; font-weight: bold;" if relative_strength < 0 else "color: #666;"
            html += f"""
                <tr>
                    <td>相对强度(相对恒指)</td>
                    <td><span style="{rs_color}">{relative_strength:.2%}</span></td>
                </tr>
            """
        
        # 中期趋势评分
        medium_term_score = indicators.get('medium_term_score', None)
        if medium_term_score is not None and medium_term_score > 0:
            if medium_term_score >= 80:
                mt_color = "color: green; font-weight: bold;"
                mt_status = "强烈买入"
            elif medium_term_score >= 65:
                mt_color = "color: green; font-weight: bold;"
                mt_status = "买入"
            elif medium_term_score >= 45:
                mt_color = "color: orange; font-weight: bold;"
                mt_status = "持有"
            elif medium_term_score >= 30:
                mt_color = "color: red; font-weight: bold;"
                mt_status = "卖出"
            else:
                mt_color = "color: red; font-weight: bold;"
                mt_status = "强烈卖出"
            
            html += f"""
                <tr>
                    <td>中期趋势评分</td>
                    <td><span style="{mt_color}">{medium_term_score:.1f}</span> <span style="font-size: 0.8em; color: #666;">({mt_status})</span></td>
                </tr>
            """
        
        # 中期趋势健康度
        medium_term_trend_health = indicators.get('medium_term_trend_health', None)
        if medium_term_trend_health is not None:
            html += f"""
                <tr>
                    <td>中期趋势健康度</td>
                    <td>{medium_term_trend_health}</td>
                </tr>
            """
        
        # 中期可持续性
        medium_term_sustainability = indicators.get('medium_term_sustainability', None)
        if medium_term_sustainability is not None:
            sustainability_color = "color: green; font-weight: bold;" if medium_term_sustainability == '高' else "color: orange; font-weight: bold;" if medium_term_sustainability == '中' else "color: red; font-weight: bold;"
            html += f"""
                <tr>
                    <td>中期可持续性</td>
                    <td><span style="{sustainability_color}">{medium_term_sustainability}</span></td>
                </tr>
            """
        
        # 中期建议
        medium_term_recommendation = indicators.get('medium_term_recommendation', None)
        if medium_term_recommendation is not None:
            html += f"""
                <tr>
                    <td>中期建议</td>
                    <td>{medium_term_recommendation}</td>
                </tr>
            """
        
        # 添加评分系统信息（如果启用）
        if self.USE_SCORED_SIGNALS:
            buildup_score = indicators.get('buildup_score', None)
            buildup_level = indicators.get('buildup_level', None)
            buildup_reasons = indicators.get('buildup_reasons', None)
            distribution_score = indicators.get('distribution_score', None)
            distribution_level = indicators.get('distribution_level', None)
            distribution_reasons = indicators.get('distribution_reasons', None)
            
            # 显示建仓评分
            if buildup_score is not None:
                buildup_color = "color: green; font-weight: bold;" if buildup_level == 'strong' else "color: orange; font-weight: bold;" if buildup_level == 'partial' else "color: #666;"
                html += f"""
                <tr>
                    <td>建仓评分</td>
                    <td><span style="{buildup_color}">{buildup_score:.2f}</span> <span style="font-size: 0.8em; color: #666;">({buildup_level})</span></td>
                </tr>
                """

                # 显示CMF资金流
                cmf = indicators.get('cmf', None)
                if cmf is not None:
                    cmf_color = "color: green; font-weight: bold;" if cmf > 0.03 else "color: red; font-weight: bold;" if cmf < -0.05 else "color: #666;"
                    cmf_text = f"+{cmf:.3f}" if cmf > 0 else f"{cmf:.3f}"
                    cmf_status = "流入" if cmf > 0.03 else "流出" if cmf < -0.05 else "中性"
                    html += f"""
                <tr>
                    <td>CMF资金流</td>
                    <td><span style="{cmf_color}">{cmf_text}</span> <span style="font-size: 0.8em; color: #666;">({cmf_status})</span></td>
                </tr>
                """
                if buildup_reasons:
                    html += f"""
                <tr>
                    <td>建仓原因</td>
                    <td style="font-size: 0.9em; color: #666;">{buildup_reasons}</td>
                </tr>
                """
            
            # 显示出货评分
            if distribution_score is not None:
                distribution_color = "color: red; font-weight: bold;" if distribution_level == 'strong' else "color: orange; font-weight: bold;" if distribution_level == 'weak' else "color: #666;"
                html += f"""
                <tr>
                    <td>出货评分</td>
                    <td><span style="{distribution_color}">{distribution_score:.2f}</span> <span style="font-size: 0.8em; color: #666;">({distribution_level})</span></td>
                </tr>
                """
                # 显示CMF资金流（如果建仓评分未显示CMF）
                cmf = indicators.get('cmf', None)
                if cmf is not None and buildup_score is None:
                    cmf_color = "color: green; font-weight: bold;" if cmf > 0.03 else "color: red; font-weight: bold;" if cmf < -0.05 else "color: #666;"
                    cmf_text = f"+{cmf:.3f}" if cmf > 0 else f"{cmf:.3f}"
                    cmf_status = "流入" if cmf > 0.03 else "流出" if cmf < -0.05 else "中性"
                    html += f"""
                <tr>
                    <td>CMF资金流</td>
                    <td><span style="{cmf_color}">{cmf_text}</span> <span style="font-size: 0.8em; color: #666;">({cmf_status})</span></td>
                </tr>
                """
                if distribution_reasons:
                    html += f"""
                <tr>
                    <td>出货原因</td>
                    <td style="font-size: 0.9em; color: #666;">{distribution_reasons}</td>
                </tr>
                """
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
        
        # 针对全部自选股进行买入/卖出分析
        print("📊 正在分析自选股买入/卖出建议...")
        portfolio_analysis = None
        try:
            portfolio_analysis = self._analyze_portfolio_with_llm(stock_results, hsi_data)
        except Exception as e:
            print(f"⚠️ 自选股分析失败: {e}")
        
        # 计算上个交易日的日期
        previous_trading_date = None
        if target_date:
            if isinstance(target_date, str):
                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
            else:
                target_date_obj = target_date
            
            # 计算上个交易日（排除周末）
            previous_trading_date = target_date_obj - timedelta(days=1)
            while previous_trading_date.weekday() >= 5:  # 5=周六, 6=周日
                previous_trading_date -= timedelta(days=1)
        
        # 获取上个交易日的指标数据
        previous_day_indicators = {}
        if previous_trading_date:
            print(f"📊 获取上个交易日 ({previous_trading_date}) 的指标数据...")
            # 获取美股市场数据（一次性获取，所有股票共享）
            previous_us_df = None
            try:
                from ml_services.us_market_data import us_market_data
                previous_us_df = us_market_data.get_all_us_market_data(period_days=30)
                if previous_us_df is not None and not previous_us_df.empty:
                    print(f"✅ 美股数据获取成功（VIX: {previous_us_df.get('VIX_Level', pd.Series([None])).iloc[-1] if 'VIX_Level' in previous_us_df.columns else 'N/A'}）")
                else:
                    print("⚠️ 美股数据为空")
            except Exception as e:
                print(f"⚠️ 获取美股数据失败: {e}")

            for stock_code, stock_name in self.stock_list.items():
                try:
                    stock_data = self.get_stock_data(stock_code, target_date=previous_trading_date.strftime('%Y-%m-%d'))
                    if stock_data:
                        indicators = self.calculate_technical_indicators(stock_data, us_df=previous_us_df)
                        if indicators:
                            previous_day_indicators[stock_code] = {
                                'trend': indicators.get('trend', '未知'),
                                'buildup_score': indicators.get('buildup_score', None),
                                'buildup_level': indicators.get('buildup_level', None),
                                'distribution_score': indicators.get('distribution_score', None),
                                'distribution_level': indicators.get('distribution_level', None),
                                'tav_score': indicators.get('tav_score', None),
                                'tav_status': indicators.get('tav_status', None),
                                'current_price': stock_data.get('current_price', None),
                                'change_pct': stock_data.get('change_1d', None)
                            }
                except Exception as e:
                    print(f"⚠️ 获取 {stock_code} 上个交易日指标失败: {e}")
        
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

        # 分析买入信号股票（需同时满足买入信号、多头趋势和48小时智能建议有买入）
        buy_signals = []
        bullish_trends = ['强势多头', '多头趋势', '短期上涨']
        for stock_name, stock_code, trend, signal, signal_type in target_date_signals:
            if signal_type == '买入' and trend in bullish_trends:
                # 检查48小时智能建议是否有买入
                continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(
                    stock_code, hours=48, min_signals=3, target_date=target_date
                )
                # 如果48小时智能建议包含买入（连续买入或买入）
                if '买入' in continuous_signal_status:
                    buy_signals.append((stock_name, stock_code, trend, signal, signal_type))
        
        buy_signals_analysis = None
        if buy_signals:
            print("🤖 使用大模型分析买入信号股票...")
            buy_signals_analysis = self._analyze_buy_signals_with_llm(buy_signals, stock_results, hsi_data)

        # 保存大模型建议到文本文件
        self.save_llm_recommendations(portfolio_analysis, buy_signals_analysis, target_date)

        # 文本版表头（修复原先被截断的 f-string）
        text_lines = []
        
        # 添加股息信息到文本
        dividend_text = self.format_dividend_table_text(dividend_data)
        if dividend_text:
            text_lines.append(dividend_text)
        
        # 添加板块轮动相关性分析结果
        text_lines.append("  3. Technology: 0.180")
        text_lines.append("  4. Exchange: 0.155")
        text_lines.append("  5. Banking: 0.105")
        text_lines.append("  6. New Energy: 0.086")
        text_lines.append("  7. Semiconductor: 0.006")
        text_lines.append("")
        text_lines.append("负相关板块 (6个):")
        text_lines.append("  1. Shipping: -0.365 (最负相关)")
        text_lines.append("  2. Energy: -0.143")
        text_lines.append("  3. Biotech: -0.137")
        text_lines.append("  4. Insurance: -0.110")
        text_lines.append("  5. AI: -0.109")
        text_lines.append("  6. Index Fund: -0.011")
        text_lines.append("")
        text_lines.append("📈 关键发现:")
        text_lines.append("  1. 航运板块与恒指负相关：航运板块表现与恒生指数走势相反，可能反映经济周期性特征")
        text_lines.append("  2. 科技板块正相关：科技板块与恒指同向波动，显示市场风险偏好")
        text_lines.append("  3. 环保板块最强正相关：环保板块与恒指正相关性最强，可能受益于政策支持")
        text_lines.append("  4. 指数基金最弱相关：指数基金与恒指相关性最弱，显示其分散化特性")
        text_lines.append("")
        
        text_lines.append("🔔 交易信号总结:")
        header = f"{'股票名称':<15} {'股票代码':<10} {'股票现价':<10} {'信号类型':<8} {'48小时智能建议':<20} {'信号描述':<30} {'趋势(技术分析)':<12} {'均线排列':<10} {'中期趋势评分':<12} {'TAV评分':<8} {'建仓评分':<10} {'出货评分':<10} {'基本面评分':<12} {'PE':<8} {'PB':<8} {'成交额变化1日':<12} {'换手率变化5日':<12} {'上个交易日趋势':<12} {'上个交易日TAV评分':<15} {'上个交易日建仓评分':<15} {'上个交易日出货评分':<15} {'上个交易日价格':<15}"
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

        # 添加板块轮动相关性分析
        html += """
        """

        # 添加板块分析
        try:
            print("📊 生成板块分析报告...")
            from data_services.hk_sector_analysis import SectorAnalyzer, DEFAULT_MIN_MARKET_CAP
            sector_analyzer = SectorAnalyzer()
            perf_df = sector_analyzer.calculate_sector_performance(self.SECTOR_ANALYSIS_PERIOD)

            # 使用业界标准的龙头股识别方法
            sector_leaders = {}
            sector_top3_leaders = {}  # 存储每个板块的前3只龙头股
            try:
                print("📊 识别板块龙头股（业界标准：稳健型风格、5日周期、最小市值100亿港币）...")
                top_sector_code = None
                bottom_sector_code = None

                if not perf_df.empty:
                    # 获取所有板块的前3只龙头股
                    for idx, row in perf_df.iterrows():
                        sector_code = row['sector_code']
                        # 获取前3只龙头股
                        leaders_df = sector_analyzer.identify_sector_leaders(
                            sector_code=sector_code,
                            top_n=3,
                            period=self.SECTOR_ANALYSIS_PERIOD,
                            min_market_cap=DEFAULT_MIN_MARKET_CAP,
                            style='moderate'  # 稳健型风格
                        )
                        if not leaders_df.empty:
                            # 存储前3只龙头股
                            sector_top3_leaders[sector_code] = []
                            for _, leader_row in leaders_df.iterrows():
                                sector_top3_leaders[sector_code].append({
                                    'name': leader_row['name'],
                                    'code': leader_row['code'],
                                    'change_pct': leader_row['change_pct'],
                                    'composite_score': leader_row['composite_score'],
                                })

                            # 存储第一只龙头股（用于表格显示）
                            sector_leaders[sector_code] = {
                                'name': leaders_df.iloc[0]['name'],
                                'code': leaders_df.iloc[0]['code'],
                                'change_pct': leaders_df.iloc[0]['change_pct'],
                                'composite_score': leaders_df.iloc[0]['composite_score'],
                                'investment_style': '稳健型',
                            }

                            if idx == 0:
                                top_sector_code = sector_code
                            if idx == len(perf_df) - 1:
                                bottom_sector_code = sector_code

                print(f"✅ 识别完成，共识别 {len(sector_leaders)} 个板块的龙头股，{len(sector_top3_leaders)} 个板块的前3名")
            except Exception as e:
                print(f"⚠️ 识别板块龙头股失败: {e}")
                sector_leaders = {}
                sector_top3_leaders = {}

            if not perf_df.empty:
                html += f"""
            <div class="section">
                <h3 style="color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px;">📊 板块分析（{self.SECTOR_ANALYSIS_PERIOD_NAME}涨跌幅排名）</h3>
                <p style="color: #666; font-size: 14px; margin-bottom: 15px;">
                    <em>💡 说明：基于最近{self.SECTOR_ANALYSIS_PERIOD}个交易日的板块平均涨跌幅进行排名，反映短期板块轮动趋势</em>
                </p>
                <p style="color: #666; font-size: 13px; margin-bottom: 15px;">
                    <em>🔍 龙头股识别：采用业界标准MVP模型（动量+成交量+基本面），稳健型风格，最小市值100亿港币，⭐表示使用专业方法识别的龙头股</em>
                </p>
                """
                
                # 板块详细排名
                html += """
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #666; font-size: 16px; margin-bottom: 10px;">📊 板块详细排名</h4>
                    <table style="border-collapse: collapse; width: 100%; background-color: #fff;">
                        <tr style="background-color: #666; color: white;">
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 8%;">排名</th>
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 18%;">趋势</th>
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 22%;">板块名称</th>
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 12%;">平均涨跌幅</th>
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 8%;">股票数量</th>
                            <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 32%;">龙头股TOP 3</th>
                        </tr>
                """

                for idx, row in perf_df.iterrows():
                    trend_icon = "🔥" if row['avg_change_pct'] > 2 else "📈" if row['avg_change_pct'] > 0 else "📉"
                    change_color = "#4CAF50" if row['avg_change_pct'] > 0 else "#f44336"

                    # 获取该板块的前3只龙头股
                    leaders_text = ""
                    if row['sector_code'] in sector_top3_leaders:
                        leaders = sector_top3_leaders[row['sector_code']]
                        leader_lines = []
                        for i, leader in enumerate(leaders, 1):
                            leader_lines.append(f"{i}. {leader['name']} ({leader['change_pct']:+.2f}%)")
                        leaders_text = "<br>".join(leader_lines)
                        leaders_text += " ⭐"  # 添加星号标记
                    elif 'stocks' in row and row['stocks']:
                        # 回退到原有逻辑（显示涨跌幅前3的股票）
                        top_3 = row['stocks'][:3]
                        leader_lines = []
                        for i, stock in enumerate(top_3, 1):
                            leader_lines.append(f"{i}. {stock['name']} ({stock['change_pct']:.2f}%)")
                        leaders_text = "<br>".join(leader_lines)

                    html += f"""
                        <tr style="background-color: {'#f9f9f9' if idx % 2 == 0 else '#fff'};">
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{idx+1}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; font-size: 18px;">{trend_icon}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{row['sector_name']}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {change_color}; font-weight: bold;">{row['avg_change_pct']:+.2f}%</td>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row['stock_count']}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; font-size: 12px; line-height: 1.5;">{leaders_text}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
                
                # 投资建议
                html += """
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3;">
                    <h4 style="color: #1976D2; font-size: 16px; margin-top: 0; margin-bottom: 10px;">💡 投资建议</h4>
                """
                
                top_sector = perf_df.iloc[0]
                bottom_sector = perf_df.iloc[-1]

                if top_sector['avg_change_pct'] > 1:
                    html += f"""
                    <p style="margin: 5px 0; color: #333;">• <strong>当前热点板块：</strong>{top_sector['sector_name']}，平均涨幅 <span style="color: #4CAF50; font-weight: bold;">{top_sector['avg_change_pct']:.2f}%</span></p>
                    """
                    # 使用业界标准识别的龙头股
                    if top_sector['sector_code'] in sector_leaders:
                        leader = sector_leaders[top_sector['sector_code']]
                        html += f"""
                        <p style="margin: 5px 0; color: #333;">• 建议关注该板块的龙头股：<span style="color: #4CAF50; font-weight: bold;">{leader['name']}</span> <span style="color: #666; font-size: 12px;">（基于MVP模型：动量+成交量+基本面，稳健型风格）</span></p>
                        """
                    elif top_sector['best_stock']:
                        # 回退到原有逻辑
                        html += f"""
                        <p style="margin: 5px 0; color: #333;">• 建议关注该板块的龙头股：<span style="color: #4CAF50; font-weight: bold;">{top_sector['best_stock']['name']}</span></p>
                        """
                
                if bottom_sector['avg_change_pct'] < -1:
                    html += f"""
                    <p style="margin: 5px 0; color: #333;">• <strong>当前弱势板块：</strong>{bottom_sector['sector_name']}，平均跌幅 <span style="color: #f44336; font-weight: bold;">{bottom_sector['avg_change_pct']:.2f}%</span></p>
                    <p style="margin: 5px 0; color: #333;">• 建议谨慎操作该板块，等待企稳信号</p>
                    """
                
                html += """
                </div>
            </div>
                """
                
                print("✅ 板块分析完成")
            else:
                html += """
            <div class="section">
                <h3>📊 板块分析</h3>
                <p style="color: #666;">⚠️ 暂无板块数据</p>
            </div>
                """
                print("⚠️ 板块数据为空")
        except Exception as e:
            print(f"⚠️ 生成板块分析失败: {e}")
            html += """
            <div class="section">
                <h3>📊 板块分析</h3>
                <p style="color: #666;">⚠️ 板块分析暂不可用</p>
            </div>
            """

        html += f"""
            <div class="section">
                <h3>🔔 交易信号总结</h3>
                <table>
                    <tr>
                        <th>股票名称</th>
                        <th>股票代码</th>
                        <th>股票现价</th>
                        <th>信号类型(量价分析)</th>
                        <th>48小时智能建议</th>
                        <th>信号描述(量价分析)</th>
                        <th>趋势(技术分析)</th>
                        <th>均线排列</th>
                        <th>中期趋势评分</th>
                        <th>TAV评分</th>
                        <th>建仓评分</th>
                        <th>出货评分</th>
                        <th>基本面评分</th>
                        <th>PE(市盈率)</th>
                        <th>PB(市净率)</th>
                        <th>成交额变化1日</th>
                        <th>换手率变化5日</th>
                        <th>上个交易日趋势</th>
                        <th>上个交易日TAV评分</th>
                        <th>上个交易日建仓评分</th>
                        <th>上个交易日出货评分</th>
                        <th>上个交易日价格</th>
                    </tr>
        """

        for stock_name, stock_code, trend, signal, signal_type in target_date_signals:
            signal_display = f"{signal_type}信号"
            continuous_signal_status = "无信号"
            if stock_code != 'HSI':
                continuous_signal_status = self.detect_continuous_signals_in_history_from_transactions(stock_code, target_date=target_date)
            
            # 判断是否满足高质量买入条件（与AI分析判断逻辑一致）
            bullish_trends = ['强势多头', '多头趋势', '短期上涨']
            is_high_quality_buy = (signal_type == '买入' and 
                                   trend in bullish_trends and 
                                   '买入' in continuous_signal_status)
            
            # 根据条件设置颜色样式
            if is_high_quality_buy:
                color_style = "color: green; font-weight: bold;"
            elif signal_type == '卖出':
                color_style = "color: red; font-weight: bold;"
            else:
                color_style = "color: black; font-weight: normal;"

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

            # 使用公共方法获取48小时智能建议颜色样式
            signal_color_style = self._get_signal_color_style(continuous_signal_status)
            
            # 使用公共方法获取趋势颜色样式
            trend_color_style = self._get_trend_color_style(trend)
            
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
                                es_decimal = es_medium_long['percentage'] if isinstance(es_medium_long, dict) else es_medium_long
                                max_dd_decimal = max_drawdown['percentage'] if isinstance(max_drawdown, dict) else max_drawdown
                                
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
            
            # 使用公共方法格式化VaR和ES值
            var_ultra_short_amount = stock_indicators.get('var_ultra_short_term_amount')
            var_short_amount = stock_indicators.get('var_short_term_amount')
            var_medium_long_amount = stock_indicators.get('var_medium_long_term_amount')
            
            # 格式化VaR值
            var_ultra_short_display = f"{var_ultra_short:.2%}" if var_ultra_short is not None else "N/A"
            var_short_display = f"{var_short:.2%}" if var_short is not None else "N/A"
            var_medium_long_display = f"{var_medium_long:.2%}" if var_medium_long is not None else "N/A"
            
            # 添加货币值显示
            if var_ultra_short is not None and var_ultra_short_amount is not None:
                var_ultra_short_display += f" (HK${var_ultra_short_amount:.2f})"
            if var_short is not None and var_short_amount is not None:
                var_short_display += f" (HK${var_short_amount:.2f})"
            if var_medium_long is not None and var_medium_long_amount is not None:
                var_medium_long_display += f" (HK${var_medium_long_amount:.2f})"
            
            # 格式化ES值
            es_short_display = f"{es_short['percentage']:.2%}" if es_short is not None else "N/A"
            es_medium_long_display = f"{es_medium_long['percentage']:.2%}" if es_medium_long is not None else "N/A"
            
            # 添加ES货币值显示
            if es_short is not None and es_short.get('amount') is not None:
                es_short_display += f" (HK${es_short['amount']:.2f})"
            if es_medium_long is not None and es_medium_long.get('amount') is not None:
                es_medium_long_display += f" (HK${es_medium_long['amount']:.2f})"
            
            # 格式化回撤和风险评估
            max_drawdown_display = f"{max_drawdown['percentage']:.2%}" if max_drawdown is not None else "N/A"
            
            # 添加回撤货币值显示
            if max_drawdown is not None and max_drawdown.get('amount') is not None:
                max_drawdown_display += f" (HK${max_drawdown['amount']:.2f})"
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
            
            # 获取建仓和出货评分
            buildup_score = stock_indicators.get('buildup_score', None) if stock_indicators else None
            buildup_level = stock_indicators.get('buildup_level', None) if stock_indicators else None
            distribution_score = stock_indicators.get('distribution_score', None) if stock_indicators else None
            distribution_level = stock_indicators.get('distribution_level', None) if stock_indicators else None
            
            # 格式化建仓评分显示
            buildup_display = "N/A"
            if buildup_score is not None:
                buildup_color = "color: green; font-weight: bold;" if buildup_level == 'strong' else "color: orange; font-weight: bold;" if buildup_level == 'partial' else "color: #666;"
                buildup_display = f"<span style=\"{buildup_color}\">{buildup_score:.2f}</span> <span style=\"font-size: 0.8em; color: #666;\">({buildup_level})</span>"
            
            # 格式化出货评分显示
            distribution_display = "N/A"
            if distribution_score is not None:
                distribution_color = "color: red; font-weight: bold;" if distribution_level == 'strong' else "color: orange; font-weight: bold;" if distribution_level == 'weak' else "color: #666;"
                distribution_display = f"<span style=\"{distribution_color}\">{distribution_score:.2f}</span> <span style=\"font-size: 0.8em; color: #666;\">({distribution_level})</span>"
            
            # 获取上个交易日的指标
            prev_day_data = previous_day_indicators.get(stock_code, {})
            prev_trend = prev_day_data.get('trend', 'N/A')
            prev_buildup_score = prev_day_data.get('buildup_score', None)
            prev_buildup_level = prev_day_data.get('buildup_level', None)
            prev_distribution_score = prev_day_data.get('distribution_score', None)
            prev_distribution_level = prev_day_data.get('distribution_level', None)
            prev_tav_score = prev_day_data.get('tav_score', None)
            prev_tav_status = prev_day_data.get('tav_status', None)
            prev_price = prev_day_data.get('current_price', None)
            
            # 计算今天价格相对于上个交易日的涨跌幅
            prev_change_pct = None
            if prev_price is not None and price_display is not None:
                try:
                    current_price = float(price_display)
                    prev_change_pct = (current_price - prev_price) / prev_price * 100
                except:
                    pass
            
            # 格式化上个交易日指标显示
            prev_trend_display = prev_trend if prev_trend is not None else 'N/A'
            prev_buildup_display = "N/A"
            if prev_buildup_score is not None:
                prev_buildup_display = f"{prev_buildup_score:.2f}({prev_buildup_level})"
            prev_distribution_display = "N/A"
            if prev_distribution_score is not None:
                prev_distribution_display = f"{prev_distribution_score:.2f}({prev_distribution_level})"
            prev_tav_display = "N/A"
            if prev_tav_score is not None:
                prev_tav_display = f"{prev_tav_score:.1f}"
            prev_price_display = f"{prev_price:.2f}" if prev_price is not None else "N/A"
            prev_change_display = f"{prev_change_pct:+.2f}%" if prev_change_pct is not None else 'N/A'
            
            # 获取基本面指标
            fundamental_score = stock_indicators.get('fundamental_score', None) if stock_indicators else None
            pe_ratio = stock_indicators.get('pe_ratio', None) if stock_indicators else None
            pb_ratio = stock_indicators.get('pb_ratio', None) if stock_indicators else None
            
            # 格式化基本面评分显示
            fundamental_display = "N/A"
            if fundamental_score is not None:
                if fundamental_score > 60:
                    fundamental_color = "color: green; font-weight: bold;"
                    fundamental_status = "优秀"
                elif fundamental_score >= 30:
                    fundamental_color = "color: orange; font-weight: bold;"
                    fundamental_status = "一般"
                else:
                    fundamental_color = "color: red; font-weight: bold;"
                    fundamental_status = "较差"
                fundamental_display = f"<span style=\"{fundamental_color}\">{fundamental_score:.0f}</span> <span style=\"font-size: 0.8em; color: #666;\">({fundamental_status})</span>"
            
            # 格式化PE显示
            pe_display = "N/A"
            if pe_ratio is not None and pe_ratio > 0:
                pe_color = "color: green;" if pe_ratio < 15 else "color: orange;" if pe_ratio < 25 else "color: red;"
                pe_display = f"<span style=\"{pe_color}\">{pe_ratio:.2f}</span>"
            
            # 格式化PB显示
            pb_display = "N/A"
            if pb_ratio is not None and pb_ratio > 0:
                pb_color = "color: green;" if pb_ratio < 1.5 else "color: orange;" if pb_ratio < 3 else "color: red;"
                pb_display = f"<span style=\"{pb_color}\">{pb_ratio:.2f}</span>"
            
            # 格式化新指标显示（HTML版本）
            turnover_change_1d = stock_indicators.get('turnover_change_1d', None) if stock_indicators else None
            turnover_change_1d_display = "N/A"
            if turnover_change_1d is not None:
                turnover_change_1d_color = "color: green;" if turnover_change_1d > 0 else "color: red;"
                turnover_change_1d_display = f"<span style=\"{turnover_change_1d_color}\">{turnover_change_1d:+.2f}%</span>"
            
            turnover_rate_change_5d = stock_indicators.get('turnover_rate_change_5d', None) if stock_indicators else None
            turnover_rate_change_5d_display = "N/A"
            if turnover_rate_change_5d is not None:
                turnover_rate_change_5d_color = "color: green;" if turnover_rate_change_5d > 0 else "color: red;"
                turnover_rate_change_5d_display = f"<span style=\"{turnover_rate_change_5d_color}\">{turnover_rate_change_5d:+.2f}%</span>"
            
            # 格式化中期趋势评分显示
            medium_term_score = stock_indicators.get('medium_term_score', None) if stock_indicators else None
            medium_term_display = "N/A"
            if medium_term_score is not None and medium_term_score > 0:
                if medium_term_score >= 80:
                    mt_color = "color: green; font-weight: bold;"
                    mt_status = "强烈买入"
                elif medium_term_score >= 65:
                    mt_color = "color: green; font-weight: bold;"
                    mt_status = "买入"
                elif medium_term_score >= 45:
                    mt_color = "color: orange; font-weight: bold;"
                    mt_status = "持有"
                elif medium_term_score >= 30:
                    mt_color = "color: red; font-weight: bold;"
                    mt_status = "卖出"
                else:
                    mt_color = "color: red; font-weight: bold;"
                    mt_status = "强烈卖出"
                medium_term_display = f"<span style=\"{mt_color}\">{medium_term_score:.1f}</span> <span style=\"font-size: 0.8em; color: #666;\">({mt_status})</span>"
            
            # 格式化均线排列显示
            ma_alignment = stock_indicators.get('ma_alignment', None) if stock_indicators else None
            ma_alignment_display = "N/A"
            if ma_alignment is not None and ma_alignment != 'N/A' and ma_alignment != '数据不足':
                ma_alignment_color = "color: green; font-weight: bold;" if ma_alignment == '多头排列' else "color: red; font-weight: bold;" if ma_alignment == '空头排列' else "color: orange; font-weight: bold;"
                ma_alignment_display = f"<span style=\"{ma_alignment_color}\">{ma_alignment}</span>"
            
            # 计算变化方向和箭头
            prev_trend_arrow = self._get_trend_change_arrow(safe_trend, prev_trend)
            prev_buildup_arrow = self._get_score_change_arrow(buildup_score, prev_buildup_score)
            prev_distribution_arrow = self._get_score_change_arrow(distribution_score, prev_distribution_score)
            prev_tav_arrow = self._get_score_change_arrow(tav_score, prev_tav_score)
            prev_price_arrow = self._get_price_change_arrow(price_value_display, prev_price)
            
            html += f"""
                    <tr>
                        <td><span style=\"{name_color_style}\">{safe_name}</span></td>
                        <td>{safe_code}</td>
                        <td>{price_value_display}</td>
                        <td><span style=\"{color_style}\">{safe_signal_display}</span></td>
                        <td><span style=\"{signal_color_style}\">{safe_continuous_signal_status}</span></td>
                        <td>{safe_signal_description}</td>
                        <td><span style=\"{trend_color_style}\">{safe_trend}</span></td>
                        <td>{ma_alignment_display}</td>
                        <td>{medium_term_display}</td>
                        <td><span style=\"{tav_color}\">{tav_score_display}</span> <span style=\"font-size: 0.8em; color: #666;\">({safe_tav_status})</span></td>
                        <td>{buildup_display}</td>
                        <td>{distribution_display}</td>
                        <td>{fundamental_display}</td>
                        <td>{pe_display}</td>
                        <td>{pb_display}</td>
                        <td>{turnover_change_1d_display}</td>
                        <td>{turnover_rate_change_5d_display}</td>
                        <td>{prev_trend_arrow} {prev_trend_display}</td>
                        <td>{prev_tav_arrow} {prev_tav_display}</td>
                        <td>{prev_buildup_arrow} {prev_buildup_display}</td>
                        <td>{prev_distribution_arrow} {prev_distribution_display}</td>
                        <td>{prev_price_arrow} {prev_price_display} ({prev_change_display})</td>
                    </tr>
            """

            # 文本版本追加
            tav_display = f"{tav_score:.1f}" if tav_score is not None else "N/A"
            var_ultra_short_display = f"{var_ultra_short:.2%}" if var_ultra_short is not None else "N/A"
            var_short_display = f"{var_short:.2%}" if var_short is not None else "N/A"
            var_medium_long_display = f"{var_medium_long:.2%}" if var_medium_long is not None else "N/A"
            
            # 添加货币值显示
            if var_ultra_short is not None and var_ultra_short_amount is not None:
                var_ultra_short_display += f" (HK${var_ultra_short_amount:.2f})"
            if var_short is not None and var_short_amount is not None:
                var_short_display += f" (HK${var_short_amount:.2f})"
            if var_medium_long is not None and var_medium_long_amount is not None:
                var_medium_long_display += f" (HK${var_medium_long_amount:.2f})"
            # 格式化ES值
            es_short_display = f"{es_short['percentage']:.2%}" if es_short is not None else "N/A"
            es_medium_long_display = f"{es_medium_long['percentage']:.2%}" if es_medium_long is not None else "N/A"
            
            # 添加ES货币值显示
            if es_short is not None and es_short.get('amount') is not None:
                es_short_display += f" (HK${es_short['amount']:.2f})"
            if es_medium_long is not None and es_medium_long.get('amount') is not None:
                es_medium_long_display += f" (HK${es_medium_long['amount']:.2f})"
            # 添加股票现价显示
            price_value = hist_data['current_price'] if hist_data is not None else None
            price_display = f"{price_value:.2f}" if price_value is not None else 'N/A'
            
            # 格式化建仓评分（文本版本）
            buildup_text = "N/A"
            if buildup_score is not None:
                buildup_text = f"{buildup_score:.2f}({buildup_level})"
            
            # 格式化出货评分（文本版本）
            distribution_text = "N/A"
            if distribution_score is not None:
                distribution_text = f"{distribution_score:.2f}({distribution_level})"
            
            # 格式化上个交易日指标（文本版本）
            prev_trend_display = prev_trend if prev_trend is not None else 'N/A'
            prev_buildup_display = "N/A"
            if prev_buildup_score is not None:
                prev_buildup_display = f"{prev_buildup_score:.2f}({prev_buildup_level})"
            prev_distribution_display = "N/A"
            if prev_distribution_score is not None:
                prev_distribution_display = f"{prev_distribution_score:.2f}({prev_distribution_level})"
            prev_tav_display = "N/A"
            if prev_tav_score is not None:
                prev_tav_display = f"{prev_tav_score:.1f}"
            prev_price_display = "N/A"
            if prev_price is not None:
                prev_price_display = f"{prev_price:.2f}"
            # 计算今天价格相对于上个交易日的涨跌幅（文本版本）
            prev_change_pct_text = None
            if prev_price is not None and price_value is not None:
                try:
                    prev_change_pct_text = (price_value - prev_price) / prev_price * 100
                except:
                    pass
            prev_change_display = f"{prev_change_pct_text:+.2f}%" if prev_change_pct_text is not None else 'N/A'
            
            # 获取基本面指标（文本版本）
            fundamental_score = stock_indicators.get('fundamental_score', None) if stock_indicators else None
            pe_ratio = stock_indicators.get('pe_ratio', None) if stock_indicators else None
            pb_ratio = stock_indicators.get('pb_ratio', None) if stock_indicators else None
            
            # 格式化基本面评分（文本版本）
            fundamental_text = "N/A"
            if fundamental_score is not None:
                fundamental_status = "优秀" if fundamental_score > 60 else "一般" if fundamental_score >= 30 else "较差"
                fundamental_text = f"{fundamental_score:.0f}({fundamental_status})"
            
            # 格式化PE（文本版本）
            pe_text = "N/A"
            if pe_ratio is not None and pe_ratio > 0:
                pe_text = f"{pe_ratio:.2f}"
            
            # 格式化PB（文本版本）
            pb_text = "N/A"
            if pb_ratio is not None and pb_ratio > 0:
                pb_text = f"{pb_ratio:.2f}"
            
            # 格式化中期趋势评分（文本版本）
            medium_term_score = stock_indicators.get('medium_term_score', None) if stock_indicators else None
            medium_term_text = "N/A"
            if medium_term_score is not None and medium_term_score > 0:
                mt_status = "强烈买入" if medium_term_score >= 80 else "买入" if medium_term_score >= 65 else "持有" if medium_term_score >= 45 else "卖出" if medium_term_score >= 30 else "强烈卖出"
                medium_term_text = f"{medium_term_score:.1f}({mt_status})"
            
            # 格式化均线排列（文本版本）
            ma_alignment = stock_indicators.get('ma_alignment', None) if stock_indicators else None
            ma_alignment_text = "N/A"
            if ma_alignment is not None and ma_alignment != 'N/A' and ma_alignment != '数据不足':
                ma_alignment_text = f"{ma_alignment}"
            
            # 格式化新指标（文本版本）
            turnover_change_1d = stock_indicators.get('turnover_change_1d', None) if stock_indicators else None
            turnover_change_1d_text = "N/A"
            if turnover_change_1d is not None:
                turnover_change_1d_text = f"{turnover_change_1d:+.2f}%"
            
            turnover_rate_change_5d = stock_indicators.get('turnover_rate_change_5d', None) if stock_indicators else None
            turnover_rate_change_5d_text = "N/A"
            if turnover_rate_change_5d is not None:
                turnover_rate_change_5d_text = f"{turnover_rate_change_5d:+.2f}%"
            
            text_lines.append(f"{stock_name:<15} {stock_code:<10} {price_display:<10} {signal_display:<8} {continuous_signal_status:<20} {signal_description:<30} {trend:<12} {ma_alignment_text:<10} {medium_term_text:<12} {tav_display:<8} {buildup_text:<10} {distribution_text:<10} {fundamental_text:<12} {pe_text:<8} {pb_text:<8} {turnover_change_1d_text:<12} {turnover_rate_change_5d_text:<12} {prev_trend_display:<12} {prev_tav_display:<15} {prev_buildup_display:<15} {prev_distribution_display:<15} {prev_price_display:<15}")

        # 检查过滤后是否有信号（使用新的过滤逻辑）
        has_filtered_signals = any(True for stock_name, stock_code, trend, signal, signal_type in target_date_signals
                                   if (signal_type in ['买入', '卖出']) or (self.detect_continuous_signals_in_history_from_transactions(stock_code, target_date=target_date) != "无建议信号"))

        if not has_filtered_signals:
            html += """
                    <tr>
                        <td colspan="22">当前没有检测到任何有效的交易信号（已过滤无信号股票）</td>
                    </tr>
            """
            text_lines.append("当前没有检测到任何有效的交易信号（已过滤无信号股票）")

        html += """
                </table>
            </div>
        """

        text = "\n".join(text_lines) + "\n\n"

        # 添加买入信号股票分析（如果有）
        if buy_signals_analysis:
            # 将markdown转换为HTML
            buy_signals_analysis_html = self._markdown_to_html(buy_signals_analysis)
            
            html += """
        <div class="section">
            <h3>🎯 买入信号股票分析（AI智能分析）</h3>
            <div style="background-color: #e8f5e9; padding: 15px; border-left: 4px solid #4CAF50; margin: 10px 0;">
                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; margin: 0;">""" + buy_signals_analysis_html + """</div>
            </div>
        </div>
            """
            
            text += f"\n🎯 买入信号股票分析（AI智能分析）:\n{buy_signals_analysis}\n\n"

        # 添加自选股买卖建议分析（如果有）
        if portfolio_analysis:
            # 将markdown转换为HTML
            portfolio_analysis_html = self._markdown_to_html(portfolio_analysis)
            
            html += """
        <div class="section">
            <h3>💼 自选股买卖建议分析（AI智能分析）</h3>
            <div style="background-color: #f0f8ff; padding: 15px; border-left: 4px solid #2196F3; margin: 10px 0;">
                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; margin: 0;">""" + portfolio_analysis_html + """</div>
            </div>
        </div>
            """
            
            text += f"\n💼 自选股买卖建议分析（AI智能分析）:\n{portfolio_analysis}\n\n"

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
                    combined_str = self._format_continuous_signal_details(transactions_df, times)
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
                    combined_str = self._format_continuous_signal_details(transactions_df, times)
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
                combined_str = self._format_continuous_signal_details_text(transactions_df, times)
                text += f"  {code} ({name}) - 建议{len(times)}次\n    {combined_str}\n"
            text += "\n"

        if sell_without_buy_after:
            text += f"📉 最近48小时内连续3次或以上建议卖出同一只股票（期间没有买入建议）:\n"
            for code, name, times, reasons, transactions_df in sell_without_buy_after:
                combined_str = self._format_continuous_signal_details_text(transactions_df, times)
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
                        target_dt = pd.to_datetime(target_date, utc=True)
                    else:
                        target_dt = pd.to_datetime(target_date, utc=True)
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
                    # sort by stock code then time
                    df_recent.sort_values(by=['code', 'timestamp'], inplace=True)
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
                        c = tr.get('code','')
                        if c not in grouped_transactions:
                            grouped_transactions[c] = []
                        grouped_transactions[c].append(tr)
                    # 按股票代码排序
                    for stock_code in sorted(grouped_transactions.keys()):
                        trans_list = grouped_transactions[stock_code]
                        stock_name = trans_list[0].get('name','')
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

                # 使用公共方法获取恒生指数趋势颜色样式
                hsi_trend_color_style = self._get_trend_color_style(trend)
                
                # 添加ATR信息
                html += f"""
                    <tr>
                        <td>ATR (14日)</td>
                        <td>{atr:.2f}</td>
                    </tr>
                """

                # 添加ATR计算的止损价和止盈价
                if atr > 0 and hsi_data.get('current_price'):
                    try:
                        current_price = float(hsi_data['current_price'])
                        # 使用1.5倍ATR作为默认止损距离
                        atr_stop_loss = current_price - (atr * 1.5)
                        # 使用3倍ATR作为默认止盈距离（基于2:1的风险收益比）
                        atr_take_profit = current_price + (atr * 3.0)
                        html += f"""
                    <tr>
                        <td>ATR止损价(1.5x)</td>
                        <td>{atr_stop_loss:,.2f}</td>
                    </tr>
                    <tr>
                        <td>ATR止盈价(3x)</td>
                        <td>{atr_take_profit:,.2f}</td>
                    </tr>
                """
                    except (ValueError, TypeError):
                        pass

                if stop_loss is not None and pd.notna(stop_loss):
                    try:
                        stop_loss_float = float(stop_loss)
                        html += f"""
                    <tr>
                        <td>建议止损价</td>
                        <td>{stop_loss_float:,.2f}</td>
                    </tr>
                """
                    except (ValueError, TypeError):
                        pass

                if take_profit is not None and pd.notna(take_profit):
                    try:
                        take_profit_float = float(take_profit)
                        html += f"""
                    <tr>
                        <td>建议止盈价</td>
                        <td>{take_profit_float:,.2f}</td>
                    </tr>
                """
                    except (ValueError, TypeError):
                        pass

                html += f"""
                    <tr>
                        <td>成交量</td>
                        <td>{hsi_data['volume']:,.0f}</td>
                    </tr>
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

            if hsi_indicators:
                text += f"📊 恒生指数技术分析:\n"
                text += f"  ATR: {atr:.2f}\n"
                
                # 添加ATR计算的止损价和止盈价
                if atr > 0 and hsi_data.get('current_price'):
                    try:
                        current_price = float(hsi_data['current_price'])
                        # 使用1.5倍ATR作为默认止损距离
                        atr_stop_loss = current_price - (atr * 1.5)
                        # 使用3倍ATR作为默认止盈距离（基于2:1的风险收益比）
                        atr_take_profit = current_price + (atr * 3.0)
                        text += f"  ATR止损价(1.5x): {atr_stop_loss:,.2f}\n"
                        text += f"  ATR止盈价(3x): {atr_take_profit:,.2f}\n"
                    except (ValueError, TypeError):
                        pass
                
                if stop_loss is not None:
                    text += f"  建议止损价: {stop_loss:,.2f}\n"
                if take_profit is not None:
                    text += f"  建议止盈价: {take_profit:,.2f}\n"
                
                text += f"  成交量: {hsi_data['volume']:,.0f}\n"
                text += f"  趋势(技术分析): {trend}\n"
                text += f"  RSI: {rsi:.2f}\n"
                text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
                text += f"  布林带位置: {bb_position:.2f}\n"
                text += f"  MA20: {ma20:,.2f}\n"
                text += f"  MA50: {ma50:,.2f}\n"
                text += f"  MA200: {ma200:,.2f}\n\n"

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
                
                # 使用公共方法获取最新的止损价和目标价
                latest_stop_loss, latest_target_price = self._get_latest_stop_loss_target(stock_result['code'], target_date)

                text += f"  ATR: {atr:.2f}\n"
                
                # 添加ATR计算的止损价和止盈价
                if atr > 0 and stock_data.get('current_price'):
                    try:
                        current_price = float(stock_data['current_price'])
                        # 使用1.5倍ATR作为默认止损距离
                        atr_stop_loss = current_price - (atr * 1.5)
                        # 使用3倍ATR作为默认止盈距离（基于2:1的风险收益比）
                        atr_take_profit = current_price + (atr * 3.0)
                        text += f"  ATR止损价(1.5x): {atr_stop_loss:,.2f}\n"
                        text += f"  ATR止盈价(3x): {atr_take_profit:,.2f}\n"
                    except (ValueError, TypeError):
                        pass
                
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
                
                text += f"  成交量: {stock_data['volume']:,.0f}\n"
                text += f"  趋势(技术分析): {trend}\n"
                text += f"  RSI: {rsi:.2f}\n"
                text += f"  MACD: {macd:.4f} (信号线: {macd_signal:.4f})\n"
                text += f"  布林带位置: {bb_position:.2f}\n"
                text += f"  MA20: {ma20:,.2f}\n"
                text += f"  MA50: {ma50:,.2f}\n"
                text += f"  MA200: {ma200:,.2f}\n"
                
                # 添加基本面指标
                fundamental_score = indicators.get('fundamental_score', None)
                pe_ratio = indicators.get('pe_ratio', None)
                pb_ratio = indicators.get('pb_ratio', None)
                
                if fundamental_score is not None:
                    fundamental_status = "优秀" if fundamental_score > 60 else "一般" if fundamental_score >= 30 else "较差"
                    text += f"  基本面评分: {fundamental_score:.0f}({fundamental_status})\n"
                
                if pe_ratio is not None and pe_ratio > 0:
                    text += f"  PE(市盈率): {pe_ratio:.2f}\n"
                
                if pb_ratio is not None and pb_ratio > 0:
                    text += f"  PB(市净率): {pb_ratio:.2f}\n"
                
                # 添加中期评估指标
                # 均线排列
                ma_alignment = indicators.get('ma_alignment', None)
                if ma_alignment is not None and ma_alignment != 'N/A' and ma_alignment != '数据不足':
                    text += f"  均线排列: {ma_alignment}\n"
                
                # 均线斜率
                ma20_slope = indicators.get('ma20_slope', None)
                ma20_slope_trend = indicators.get('ma20_slope_trend', None)
                if ma20_slope is not None and ma20_slope_trend is not None:
                    text += f"  MA20斜率: {ma20_slope:.4f}({ma20_slope_trend})\n"
                
                ma50_slope = indicators.get('ma50_slope', None)
                ma50_slope_trend = indicators.get('ma50_slope_trend', None)
                if ma50_slope is not None and ma50_slope_trend is not None:
                    text += f"  MA50斜率: {ma50_slope:.4f}({ma50_slope_trend})\n"
                
                # 乖离率
                ma_deviation_avg = indicators.get('ma_deviation_avg', None)
                if ma_deviation_avg is not None and ma_deviation_avg != 0:
                    text += f"  均线乖离率: {ma_deviation_avg:.2f}%\n"
                
                # 支撑阻力位
                nearest_support = indicators.get('nearest_support', None)
                nearest_resistance = indicators.get('nearest_resistance', None)
                if nearest_support is not None:
                    support_pct = ((current_price - nearest_support) / current_price * 100) if current_price > 0 else 0
                    text += f"  最近支撑位: {nearest_support:.2f}(距离{support_pct:.2f}%)\n"
                
                if nearest_resistance is not None:
                    resistance_pct = ((nearest_resistance - current_price) / current_price * 100) if current_price > 0 else 0
                    text += f"  最近阻力位: {nearest_resistance:.2f}(距离{resistance_pct:.2f}%)\n"
                
                # 相对强弱
                relative_strength = indicators.get('relative_strength', None)
                if relative_strength is not None:
                    text += f"  相对强度(相对恒指): {relative_strength:.2%}\n"
                
                # 中期趋势评分
                medium_term_score = indicators.get('medium_term_score', None)
                if medium_term_score is not None and medium_term_score > 0:
                    if medium_term_score >= 80:
                        mt_status = "强烈买入"
                    elif medium_term_score >= 65:
                        mt_status = "买入"
                    elif medium_term_score >= 45:
                        mt_status = "持有"
                    elif medium_term_score >= 30:
                        mt_status = "卖出"
                    else:
                        mt_status = "强烈卖出"
                    text += f"  中期趋势评分: {medium_term_score:.1f}({mt_status})\n"
                
                # 中期趋势健康度
                medium_term_trend_health = indicators.get('medium_term_trend_health', None)
                if medium_term_trend_health is not None:
                    text += f"  中期趋势健康度: {medium_term_trend_health}\n"
                
                # 中期可持续性
                medium_term_sustainability = indicators.get('medium_term_sustainability', None)
                if medium_term_sustainability is not None:
                    text += f"  中期可持续性: {medium_term_sustainability}\n"
                
                # 中期建议
                medium_term_recommendation = indicators.get('medium_term_recommendation', None)
                if medium_term_recommendation is not None:
                    text += f"  中期建议: {medium_term_recommendation}\n"
                
                # 添加VaR信息
                var_ultra_short = indicators.get('var_ultra_short_term')
                var_ultra_short_amount = indicators.get('var_ultra_short_term_amount')
                var_short = indicators.get('var_short_term')
                var_short_amount = indicators.get('var_short_term_amount')
                var_medium_long = indicators.get('var_medium_long_term')
                var_medium_long_amount = indicators.get('var_medium_long_term_amount')
                
                if var_ultra_short is not None:
                    amount_display = f" (HK${var_ultra_short_amount:.2f})" if var_ultra_short_amount is not None else ""
                    text += f"  1日VaR (95%): {var_ultra_short:.2%}{amount_display}\n"
                
                if var_short is not None:
                    amount_display = f" (HK${var_short_amount:.2f})" if var_short_amount is not None else ""
                    text += f"  5日VaR (95%): {var_short:.2%}{amount_display}\n"
                
                if var_medium_long is not None:
                    amount_display = f" (HK${var_medium_long_amount:.2f})" if var_medium_long_amount is not None else ""
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
                            amount_display = f" (HK${es_1d['amount']:.2f})" if es_1d.get('amount') is not None else ""
                            text += f"  1日ES (95%): {es_1d['percentage']:.2%}{amount_display}\n"
                        if es_5d is not None:
                            amount_display = f" (HK${es_5d['amount']:.2f})" if es_5d.get('amount') is not None else ""
                            text += f"  5日ES (95%): {es_5d['percentage']:.2%}{amount_display}\n"
                        if es_20d is not None:
                            amount_display = f" (HK${es_20d['amount']:.2f})" if es_20d.get('amount') is not None else ""
                            text += f"  20日ES (95%): {es_20d['percentage']:.2%}{amount_display}\n"

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
              <li><b>ATR(平均真实波幅)</b>：衡量市场波动性的技术指标，数值越高表示波动越大，常用于设置止损和止盈位。
                <ul>
                  <li><b>港股单位</b>：港元（HK$），表示股票的平均价格波动幅度</li>
                  <li><b>恒指单位</b>：点数，表示恒生指数的平均波动幅度</li>
                  <li><b>应用</b>：通常使用1.5-2倍ATR作为止损距离，例如当前价-1.5×ATR可作为止损参考</li>
                </ul>
              </li>
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
              <li><b>建仓评分(0-10+)</b>：基于9个技术指标的加权评分系统，用于识别主力资金建仓信号：
                <ul>
                  <li><b>评分范围</b>：0-10+分，分数越高建仓信号越强</li>
                  <li><b>信号级别</b>：
                    <ul>
                      <li>strong（强烈建仓）：评分≥5.0，建议较高比例买入或确认建仓</li>
                      <li>partial（部分建仓）：评分≥3.0，建议分批入场或小仓位试探</li>
                      <li>none（无信号）：评分<3.0，无明确建仓信号</li>
                    </ul>
                  </li>
                  <li><b>评估指标（共9个）</b>：
                    <ul>
                      <li>price_low（权重2.0）：价格处于低位（价格百分位<40%）</li>
                      <li>vol_ratio（权重2.0）：成交量放大（成交量比率>1.3）</li>
                      <li>vol_z（权重1.0）：成交量z-score>1.2，显著高于平均水平</li>
                      <li>macd_cross（权重1.5）：MACD线上穿信号线（金叉），上涨动能增强</li>
                      <li>rsi_oversold（权重1.2）：RSI<40，超卖区域，反弹概率高</li>
                      <li>obv_up（权重1.0）：OBV>0，资金净流入</li>
                      <li>vwap_vol（权重1.2）：价格高于VWAP且成交量比率>1.2，强势特征</li>
                      <li>price_above_vwap（权重0.8）：价格高于VWAP，当日表现强势</li>
                      <li>bb_oversold（权重1.0）：布林带位置<0.2，接近下轨，超卖信号</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>建仓评分持续上升：主力资金持续流入，可考虑加仓</li>
                      <li>建仓评分下降：建仓动能减弱，需谨慎</li>
                      <li>建仓评分与出货评分同时高：多空信号冲突，建议观望</li>
                    </ul>
                  </li>
                </ul>
              </li>
              <li><b>出货评分(0-10+)</b>：基于10个技术指标的加权评分系统，用于识别主力资金出货信号：
                <ul>
                  <li><b>评分范围</b>：0-10+分，分数越高出货信号越强</li>
                  <li><b>信号级别</b>：
                    <ul>
                      <li>strong（强烈出货）：评分≥5.0，建议较大比例卖出或清仓</li>
                      <li>weak（弱出货）：评分≥3.0，建议部分减仓或密切观察</li>
                      <li>none（无信号）：评分<3.0，无明确出货信号</li>
                    </ul>
                  </li>
                  <li><b>评估指标（共10个）</b>：
                    <ul>
                      <li>price_high（权重2.0）：价格处于高位（价格百分位>60%）</li>
                      <li>vol_ratio（权重2.0）：成交量放大（成交量比率>1.5）</li>
                      <li>vol_z（权重1.5）：成交量z-score>1.5，显著高于平均水平</li>
                      <li>macd_cross（权重1.5）：MACD线下穿信号线（死叉），下跌动能增强</li>
                      <li>rsi_high（权重1.5）：RSI>65，超买区域，回调风险高</li>
                      <li>obv_down（权重1.0）：OBV<0，资金净流出</li>
                      <li>vwap_vol（权重1.5）：价格低于VWAP且成交量比率>1.2，弱势特征</li>
                      <li>price_down（权重1.0）：日变化<0，价格下跌</li>
                      <li>bb_overbought（权重1.0）：布林带位置>0.8，接近上轨，超买信号</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>出货评分持续上升：主力资金持续流出，建议减仓或清仓</li>
                      <li>出货评分下降：出货动能减弱，可考虑观望</li>
                      <li>建仓评分与出货评分同时低：缺乏明确方向，建议观望</li>
                      <li>建仓评分高且出货评分低：建仓信号明确，可考虑买入</li>
                    </ul>
                  </li>
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
              <li><b>基本面评分(0-100)</b>：基于PE（市盈率）和PB（市净率）的综合评分，评估股票的基本面质量：
                <ul>
                  <li><b>评分范围</b>：0-100分，分数越高基本面质量越好</li>
                  <li><b>评分等级</b>：
                    <ul>
                      <li>优秀（>60分）：基本面质量高，估值合理或偏低，适合长期投资</li>
                      <li>一般（30-60分）：基本面质量中等，估值适中，需结合其他指标综合判断</li>
                      <li>较差（<30分）：基本面质量低，估值偏高，投资风险较大</li>
                    </ul>
                  </li>
                  <li><b>PE评分（50分权重）</b>：基于市盈率评估估值水平
                    <ul>
                      <li>PE<10：50分，低估值，投资价值高</li>
                      <li>10≤PE<15：40分，合理估值，投资价值良好</li>
                      <li>15≤PE<20：30分，偏高估值，投资价值一般</li>
                      <li>20≤PE<25：20分，高估值，投资价值较低</li>
                      <li>PE≥25：10分，极高估值，投资风险高</li>
                    </ul>
                  </li>
                  <li><b>PB评分（50分权重）</b>：基于市净率评估估值水平
                    <ul>
                      <li>PB<1：50分，低市净率，投资价值高</li>
                      <li>1≤PB<1.5：40分，合理市净率，投资价值良好</li>
                      <li>1.5≤PB<2：30分，偏高市净率，投资价值一般</li>
                      <li>2≤PB<3：20分，高市净率，投资价值较低</li>
                      <li>PB≥3：10分，极高市净率，投资风险高</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>基本面评分高：股票估值合理，盈利能力强，适合长期投资</li>
                      <li>基本面评分低：股票估值偏高，盈利能力弱，投资风险较大</li>
                      <li>与技术指标结合：基本面评分高+技术指标好=强烈买入信号</li>
                    </ul>
                  </li>
                </ul>
              </li>
              <li><b>PE（市盈率）</b>：股票价格与每股收益的比率，衡量股票估值水平：
                <ul>
                  <li><b>计算方式</b>：PE = 股票价格 / 每股收益</li>
                  <li><b>估值判断</b>：
                    <ul>
                      <li>PE<15：低估值，投资价值高</li>
                      <li>15≤PE<25：合理估值，投资价值良好</li>
                      <li>PE≥25：高估值，投资风险高</li>
                    </ul>
                  </li>
                  <li><b>行业差异</b>：不同行业的PE水平不同，需结合行业平均水平判断</li>
                  <li><b>局限性</b>：PE不适用于亏损公司，需结合PB等其他指标综合判断</li>
                </ul>
              </li>
              <li><b>PB（市净率）</b>：股票价格与每股净资产的比率，衡量股票估值水平：
                <ul>
                  <li><b>计算方式</b>：PB = 股票价格 / 每股净资产</li>
                  <li><b>估值判断</b>：
                    <ul>
                      <li>PB<1.5：低市净率，投资价值高</li>
                      <li>1.5≤PB<3：合理市净率，投资价值良好</li>
                      <li>PB≥3：高市净率，投资风险高</li>
                    </ul>
                  </li>
                  <li><b>适用性</b>：PB适用于周期性行业和亏损公司，比PE更稳健</li>
                  <li><b>行业差异</b>：不同行业的PB水平不同，需结合行业平均水平判断</li>
                </ul>
              </li>
              <li><b>VIX恐慌指数</b>：衡量市场恐慌程度的指标，反映市场情绪和波动预期：
                <ul>
                  <li><b>计算方式</b>：基于标普500指数期权价格计算，反映未来30天的市场波动预期</li>
                  <li><b>市场情绪判断</b>：
                    <ul>
                      <li>VIX<15：市场过度乐观，需警惕回调风险</li>
                      <li>15≤VIX<20：正常波动，市场情绪平稳</li>
                      <li>20≤VIX<30：轻度恐慌，市场波动加大</li>
                      <li>VIX≥30：严重恐慌，通常伴随大跌，但可能存在反弹机会</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>VIX低时：市场情绪乐观，适合谨慎交易，降低仓位</li>
                      <li>VIX高时：市场恐慌，可能存在反弹机会，适合分批建仓</li>
                      <li>VIX急剧上升：市场恐慌加剧，建议观望或减仓</li>
                    </ul>
                  </li>
                  <li><b>ML模型重要性</b>：VIX_Level在所有预测周期的Top 10特征中都出现，是重要的市场环境特征</li>
                </ul>
              </li>
              <li><b>成交额变化率</b>：衡量资金流入流出的直接指标，反映市场流动性变化：
                <ul>
                  <li><b>计算方式</b>：成交额变化率 = (当前成交额 - N日前成交额) / N日前成交额 × 100%</li>
                  <li><b>时间周期</b>：
                    <ul>
                      <li>1日变化率：反映短期资金流向，捕捉短期情绪变化</li>
                      <li>5日变化率：反映中期资金流向，判断中期趋势</li>
                      <li>20日变化率：反映长期资金流向，评估长期趋势</li>
                    </ul>
                  </li>
                  <li><b>资金流向判断</b>：
                    <ul>
                      <li>正向变化率：资金持续流入，市场活跃，支持交易</li>
                      <li>负向变化率：资金持续流出，市场低迷，减少交易</li>
                      <li>多周期一致：1日、5日、20日变化率同向，信号更可靠</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>成交额持续增长：资金流入，市场活跃，适合交易</li>
                      <li>成交额持续萎缩：资金流出，市场低迷，建议观望</li>
                      <li>突发性成交额变化：可能预示重大消息或趋势转折</li>
                    </ul>
                  </li>
                  <li><b>ML模型重要性</b>：成交额变化率在长期预测（20天）中显著提升准确率</li>
                </ul>
              </li>
              <li><b>换手率变化率</b>：衡量市场关注度变化的指标，反映流动性增强或减弱：
                <ul>
                  <li><b>计算方式</b>：换手率 = 成交量 / 流通股本 × 100%；换手率变化率 = (当前换手率 - N日前换手率) / N日前换手率 × 100%</li>
                  <li><b>时间周期</b>：
                    <ul>
                      <li>5日变化率：反映短期关注度变化，适合短期交易</li>
                      <li>20日变化率：反映中期关注度变化，适合中期投资</li>
                    </ul>
                  </li>
                  <li><b>关注度判断</b>：
                    <ul>
                      <li>换手率上升+换手率变化率正向：关注度提升，流动性增强，适合交易</li>
                      <li>换手率下降+换手率变化率负向：关注度下降，流动性减弱，观望为主</li>
                      <li>换手率异常波动：可能预示重大消息或趋势转折，提高警惕</li>
                    </ul>
                  </li>
                  <li><b>应用场景</b>：
                    <ul>
                      <li>换手率持续上升：市场关注度提升，流动性增强，适合交易</li>
                      <li>换手率持续下降：市场关注度下降，流动性减弱，建议观望</li>
                      <li>换手率异常波动：可能预示重大消息或趋势转折</li>
                    </ul>
                  </li>
                  <li><b>ML模型重要性</b>：换手率变化率在长期预测（20天）中显著提升准确率</li>
                </ul>
              </li>
              <li><b>成交量</b>：股票在特定时间段内的交易数量，反映市场活跃度和资金流向：
                <ul>
                  <li><b>成交量放大</b>：通常表示市场关注度提高，可能预示趋势加速或反转</li>
                  <li><b>成交量萎缩</b>：通常表示市场观望情绪浓厚，可能预示趋势减弱</li>
                  <li><b>量价配合</b>：价格上涨+成交量放大=上涨动能强；价格下跌+成交量放大=下跌动能强</li>
                  <li><b>量价背离</b>：价格上涨+成交量萎缩=上涨动能弱，可能回调；价格下跌+成交量萎缩=下跌动能弱，可能反弹</li>
                </ul>
              </li>
              <li><b>均线排列</b>：多条移动平均线的相对位置关系，判断市场趋势方向：
                <ul>
                  <li><b>多头排列</b>：短期均线(MA20) > 中期均线(MA50) > 长期均线(MA200)，且所有均线向上，强烈看涨信号</li>
                  <li><b>空头排列</b>：短期均线(MA20) < 中期均线(MA50) < 长期均线(MA200)，且所有均线向下，强烈看空信号</li>
                  <li><b>混乱排列</b>：均线交叉纠缠，没有明确方向，市场处于震荡整理阶段</li>
                  <li><b>应用</b>：多头排列适合持有或加仓，空头排列适合减仓或清仓，混乱排列建议观望</li>
                </ul>
              </li>
              <li><b>MA20斜率</b>：20日移动平均线的斜率和趋势方向，判断短期趋势强度：
                <ul>
                  <li><b>上升</b>：斜率为正，短期趋势向上，价格动能强劲</li>
                  <li><b>下降</b>：斜率为负，短期趋势向下，价格动能疲弱</li>
                  <li><b>水平</b>：斜率接近0，短期趋势平稳，市场处于整理状态</li>
                  <li><b>斜率大小</b>：斜率绝对值越大，趋势强度越大，变化速度越快</li>
                </ul>
              </li>
              <li><b>MA50斜率</b>：50日移动平均线的斜率和趋势方向，判断中期趋势强度：
                <ul>
                  <li><b>上升</b>：斜率为正，中期趋势向上，价格中期动能强劲</li>
                  <li><b>下降</b>：斜率为负，中期趋势向下，价格中期动能疲弱</li>
                  <li><b>水平</b>：斜率接近0，中期趋势平稳，市场处于中期整理状态</li>
                  <li><b>与MA20配合</b>：MA20和MA50同时上升=中期上涨趋势确立；同时下降=中期下跌趋势确立</li>
                </ul>
              </li>
              <li><b>均线乖离率</b>：当前价格与移动平均线的偏离程度，评估超买超卖状态：
                <ul>
                  <li><b>正乖离率</b>：价格高于均线，表示价格上涨，乖离率过大=超买，可能回调</li>
                  <li><b>负乖离率</b>：价格低于均线，表示价格下跌，乖离率过大=超卖，可能反弹</li>
                  <li><b>乖离率阈值</b>：±3%以内=正常；±3-5%=轻微偏离；±5%以上=严重偏离</li>
                  <li><b>应用</b>：乖离率过大时，考虑反向操作；乖离率回归0时，趋势可能延续</li>
                </ul>
              </li>
              <li><b>最近支撑位</b>：近期价格多次触及后反弹的价格水平，是重要的买入参考位：
                <ul>
                  <li><b>识别方法</b>：基于近期局部低点识别，价格多次在此位置获得支撑</li>
                  <li><b>支撑强度</b>：触及次数越多，支撑越强；距离当前价格越近，参考价值越高</li>
                  <li><b>应用</b>：价格接近支撑位时，可考虑买入或加仓；跌破支撑位时，可能开启新一轮下跌</li>
                  <li><b>止损设置</b>：可将支撑位下方作为止损参考位</li>
                </ul>
              </li>
              <li><b>最近阻力位</b>：近期价格多次触及后回落的价格水平，是重要的卖出参考位：
                <ul>
                  <li><b>识别方法</b>：基于近期局部高点识别，价格多次在此位置遇到阻力</li>
                  <li><b>阻力强度</b>：触及次数越多，阻力越强；距离当前价格越近，参考价值越高</li>
                  <li><b>应用</b>：价格接近阻力位时，可考虑卖出或减仓；突破阻力位时，可能开启新一轮上涨</li>
                  <li><b>止盈设置</b>：可将阻力位附近作为止盈参考位</li>
                </ul>
              </li>
              <li><b>相对强度(相对恒指)</b>：股票相对于恒生指数的表现，评估个股跑赢或跑输大盘：
                <ul>
                  <li><b>正值</b>：股票表现强于恒生指数，跑赢大盘，资金青睐</li>
                  <li><b>负值</b>：股票表现弱于恒生指数，跑输大盘，资金流出</li>
                  <li><b>计算方式</b>：相对强度 = (股票收益率 - 恒指收益率)</li>
                  <li><b>应用</b>：相对强度持续上升=强势股，可重点关注；相对强度持续下降=弱势股，建议规避</li>
                </ul>
              </li>
              <li><b>中期趋势评分(0-100)</b>：综合趋势、动量、支撑阻力、相对强弱四维度的中期趋势评分：
                <ul>
                  <li><b>评分范围</b>：0-100分，分数越高中期趋势越强</li>
                  <li><b>评分等级</b>：
                    <ul>
                      <li>≥80分：强烈买入，中期趋势强劲，建议重仓</li>
                      <li>65-79分：买入，中期趋势良好，建议加仓</li>
                      <li>45-64分：持有，中期趋势中性，建议观望</li>
                      <li>30-44分：卖出，中期趋势疲弱，建议减仓</li>
                      <li><30分：强烈卖出，中期趋势恶化，建议清仓</li>
                    </ul>
                  </li>
                  <li><b>评分维度（各25%权重）</b>：
                    <ul>
                      <li>趋势评分：基于均线排列和均线斜率</li>
                      <li>动量评分：基于价格变化率和MACD</li>
                      <li>支撑阻力评分：基于支撑阻力位距离和强度</li>
                      <li>相对强弱评分：基于相对恒指的表现</li>
                    </ul>
                  </li>
                </ul>
              </li>
              <li><b>中期趋势健康度</b>：评估中期趋势的稳定性和可持续性：
                <ul>
                  <li><b>健康</b>：趋势稳定，波动性低，可持续性强，适合中长期持有</li>
                  <li><b>一般</b>：趋势尚可，波动性中等，可持续性一般，需密切关注</li>
                  <li><b>较差</b>：趋势不稳定，波动性高，可持续性弱，建议谨慎</li>
                  <li><b>评估因素</b>：趋势稳定性、价格波动性、成交量一致性、均线发散程度</li>
                  <li><b>应用</b>：健康度高=可放心持有；健康度低=需警惕趋势反转</li>
                </ul>
              </li>
              <li><b>中期可持续性</b>：评估中期趋势能够持续的可能性：
                <ul>
                  <li><b>高</b>：趋势有基本面支撑，资金持续流入，可持续性强</li>
                  <li><b>中</b>：趋势有技术面支撑，资金流入稳定，可持续性中等</li>
                  <li><b>低</b>：趋势缺乏支撑，资金流入不稳定，可持续性弱</li>
                  <li><b>评估因素</b>：基本面质量、资金流向、市场情绪、行业景气度</li>
                  <li><b>应用</b>：可持续性高=可中长期持有；可持续性低=建议短线操作</li>
                </ul>
              </li>
              <li><b>中期建议</b>：基于中期趋势评分、健康度、可持续性的综合投资建议：
                <ul>
                  <li><b>强烈买入</b>：中期趋势强劲且健康，建议重仓或加仓</li>
                  <li><b>买入</b>：中期趋势良好，建议建仓或加仓</li>
                  <li><b>持有</b>：中期趋势中性，建议继续持有或观望</li>
                  <li><b>卖出</b>：中期趋势疲弱，建议减仓或清仓</li>
                  <li><b>强烈卖出</b>：中期趋势恶化，建议清仓或规避</li>
                  <li><b>决策参考</b>：结合基本面评分和技术指标综合判断</li>
                </ul>
              </li>
            </ul>
            </div><div class="section">
            <h3>📊 板块轮动与恒指相关性分析</h3>
            <p><strong>最佳贴合板块:</strong> Shipping (r=-0.365)</p>
            <ul>
                <li><strong>负相关：</strong>与恒生指数走势相反</li>
                <li><strong>相关性强度：</strong>中等偏弱</li>
            </ul>
            
            <h4>正相关板块 (7个):</h4>
            <ul>
                <li><strong>Environmental:</strong> 0.306</li>
                <li><strong>Utility:</strong> 0.207</li>
                <li><strong>Technology:</strong> 0.180</li>
                <li><strong>Exchange:</strong> 0.155</li>
                <li><strong>Banking:</strong> 0.105</li>
                <li><strong>New Energy:</strong> 0.086</li>
                <li><strong>Semiconductor:</strong> 0.006</li>
            </ul>
            
            <h4>负相关板块 (6个):</h4>
            <ul>
                <li><strong>Shipping:</strong> -0.365 (最负相关)</li>
                <li><strong>Energy:</strong> -0.143</li>
                <li><strong>Biotech:</strong> -0.137</li>
                <li><strong>Insurance:</strong> -0.110</li>
                <li><strong>AI:</strong> -0.109</li>
                <li><strong>Index Fund:</strong> -0.011</li>
            </ul>
            
            <h4>📈 关键发现:</h4>
            <ul>
                <li><strong>航运板块与恒指负相关：</strong>航运板块表现与恒生指数走势相反，可能反映经济周期性特征</li>
                <li><strong>科技板块正相关：</strong>科技板块与恒指同向波动，显示市场风险偏好</li>
                <li><strong>环保板块最强正相关：</strong>环保板块与恒指正相关性最强，可能受益于政策支持</li>
                <li><strong>指数基金最弱相关：</strong>指数基金与恒指相关性最弱，显示其分散化特性</li>
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
        text += "  - 港股单位：港元（HK$），表示股票的平均价格波动幅度\n"
        text += "  - 恒指单位：点数，表示恒生指数的平均波动幅度\n"
        text += "  - 应用：通常使用1.5-2倍ATR作为止损距离，例如当前价-1.5×ATR可作为止损参考\n"
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
        text += "• 建仓评分(0-10+)：基于9个技术指标的加权评分系统，用于识别主力资金建仓信号：\n"
        text += "  - 评分范围：0-10+分，分数越高建仓信号越强\n"
        text += "  - 信号级别：\n"
        text += "    * strong（强烈建仓）：评分≥5.0，建议较高比例买入或确认建仓\n"
        text += "    * partial（部分建仓）：评分≥3.0，建议分批入场或小仓位试探\n"
        text += "    * none（无信号）：评分<3.0，无明确建仓信号\n"
        text += "  - 评估指标（共9个）：\n"
        text += "    * price_low（权重2.0）：价格处于低位（价格百分位<40%）\n"
        text += "    * vol_ratio（权重2.0）：成交量放大（成交量比率>1.3）\n"
        text += "    * vol_z（权重1.0）：成交量z-score>1.2，显著高于平均水平\n"
        text += "    * macd_cross（权重1.5）：MACD线上穿信号线（金叉），上涨动能增强\n"
        text += "    * rsi_oversold（权重1.2）：RSI<40，超卖区域，反弹概率高\n"
        text += "    * obv_up（权重1.0）：OBV>0，资金净流入\n"
        text += "    * vwap_vol（权重1.2）：价格高于VWAP且成交量比率>1.2，强势特征\n"
        text += "    * price_above_vwap（权重0.8）：价格高于VWAP，当日表现强势\n"
        text += "    * bb_oversold（权重1.0）：布林带位置<0.2，接近下轨，超卖信号\n"
        text += "  - 应用场景：\n"
        text += "    * 建仓评分持续上升：主力资金持续流入，可考虑加仓\n"
        text += "    * 建仓评分下降：建仓动能减弱，需谨慎\n"
        text += "    * 建仓评分与出货评分同时高：多空信号冲突，建议观望\n"
        text += "• 出货评分(0-10+)：基于10个技术指标的加权评分系统，用于识别主力资金出货信号：\n"
        text += "  - 评分范围：0-10+分，分数越高出货信号越强\n"
        text += "  - 信号级别：\n"
        text += "    * strong（强烈出货）：评分≥5.0，建议较大比例卖出或清仓\n"
        text += "    * weak（弱出货）：评分≥3.0，建议部分减仓或密切观察\n"
        text += "    * none（无信号）：评分<3.0，无明确出货信号\n"
        text += "  - 评估指标（共10个）：\n"
        text += "    * price_high（权重2.0）：价格处于高位（价格百分位>60%）\n"
        text += "    * vol_ratio（权重2.0）：成交量放大（成交量比率>1.5）\n"
        text += "    * vol_z（权重1.5）：成交量z-score>1.5，显著高于平均水平\n"
        text += "    * macd_cross（权重1.5）：MACD线下穿信号线（死叉），下跌动能增强\n"
        text += "    * rsi_high（权重1.5）：RSI>65，超买区域，回调风险高\n"
        text += "    * obv_down（权重1.0）：OBV<0，资金净流出\n"
        text += "    * vwap_vol（权重1.5）：价格低于VWAP且成交量比率>1.2，弱势特征\n"
        text += "    * price_down（权重1.0）：日变化<0，价格下跌\n"
        text += "    * bb_overbought（权重1.0）：布林带位置>0.8，接近上轨，超买信号\n"
        text += "  - 应用场景：\n"
        text += "    * 出货评分持续上升：主力资金持续流出，建议减仓或清仓\n"
        text += "    * 出货评分下降：出货动能减弱，可考虑观望\n"
        text += "    * 建仓评分与出货评分同时低：缺乏明确方向，建议观望\n"
        text += "    * 建仓评分高且出货评分低：建仓信号明确，可考虑买入\n"
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
        text += "• 基本面评分(0-100)：基于PE（市盈率）和PB（市净率）的综合评分，评估股票的基本面质量：\n"
        text += "  - 评分范围：0-100分，分数越高基本面质量越好\n"
        text += "  - 评分等级：\n"
        text += "    * 优秀（>60分）：基本面质量高，估值合理或偏低，适合长期投资\n"
        text += "    * 一般（30-60分）：基本面质量中等，估值适中，需结合其他指标综合判断\n"
        text += "    * 较差（<30分）：基本面质量低，估值偏高，投资风险较大\n"
        text += "  - PE评分（50分权重）：基于市盈率评估估值水平\n"
        text += "    * PE<10：50分，低估值，投资价值高\n"
        text += "    * 10≤PE<15：40分，合理估值，投资价值良好\n"
        text += "    * 15≤PE<20：30分，偏高估值，投资价值一般\n"
        text += "    * 20≤PE<25：20分，高估值，投资价值较低\n"
        text += "    * PE≥25：10分，极高估值，投资风险高\n"
        text += "  - PB评分（50分权重）：基于市净率评估估值水平\n"
        text += "    * PB<1：50分，低市净率，投资价值高\n"
        text += "    * 1≤PB<1.5：40分，合理市净率，投资价值良好\n"
        text += "    * 1.5≤PB<2：30分，偏高市净率，投资价值一般\n"
        text += "    * 2≤PB<3：20分，高市净率，投资价值较低\n"
        text += "    * PB≥3：10分，极高市净率，投资风险高\n"
        text += "  - 应用场景：\n"
        text += "    * 基本面评分高：股票估值合理，盈利能力强，适合长期投资\n"
        text += "    * 基本面评分低：股票估值偏高，盈利能力弱，投资风险较大\n"
        text += "    * 与技术指标结合：基本面评分高+技术指标好=强烈买入信号\n"
        text += "• PE（市盈率）：股票价格与每股收益的比率，衡量股票估值水平：\n"
        text += "  - 计算方式：PE = 股票价格 / 每股收益\n"
        text += "  - 估值判断：\n"
        text += "    * PE<15：低估值，投资价值高\n"
        text += "    * 15≤PE<25：合理估值，投资价值良好\n"
        text += "    * PE≥25：高估值，投资风险高\n"
        text += "  - 行业差异：不同行业的PE水平不同，需结合行业平均水平判断\n"
        text += "  - 局限性：PE不适用于亏损公司，需结合PB等其他指标综合判断\n"
        text += "• PB（市净率）：股票价格与每股净资产的比率，衡量股票估值水平：\n"
        text += "  - 计算方式：PB = 股票价格 / 每股净资产\n"
        text += "  - 估值判断：\n"
        text += "    * PB<1.5：低市净率，投资价值高\n"
        text += "    * 1.5≤PB<3：合理市净率，投资价值良好\n"
        text += "    * PB≥3：高市净率，投资风险高\n"
        text += "  - 适用性：PB适用于周期性行业和亏损公司，比PE更稳健\n"
        text += "  - 行业差异：不同行业的PB水平不同，需结合行业平均水平判断\n"
        text += "• VIX恐慌指数：衡量市场恐慌程度的指标，反映市场情绪和波动预期：\n"
        text += "  - 计算方式：基于标普500指数期权价格计算，反映未来30天的市场波动预期\n"
        text += "  - 市场情绪判断：\n"
        text += "    * VIX<15：市场过度乐观，需警惕回调风险\n"
        text += "    * 15≤VIX<20：正常波动，市场情绪平稳\n"
        text += "    * 20≤VIX<30：轻度恐慌，市场波动加大\n"
        text += "    * VIX≥30：严重恐慌，通常伴随大跌，但可能存在反弹机会\n"
        text += "  - 应用场景：\n"
        text += "    * VIX低时：市场情绪乐观，适合谨慎交易，降低仓位\n"
        text += "    * VIX高时：市场恐慌，可能存在反弹机会，适合分批建仓\n"
        text += "    * VIX急剧上升：市场恐慌加剧，建议观望或减仓\n"
        text += "  - ML模型重要性：VIX_Level在所有预测周期的Top 10特征中都出现，是重要的市场环境特征\n"
        text += "• 成交额变化率：衡量资金流入流出的直接指标，反映市场流动性变化：\n"
        text += "  - 计算方式：成交额变化率 = (当前成交额 - N日前成交额) / N日前成交额 × 100%\n"
        text += "  - 时间周期：\n"
        text += "    * 1日变化率：反映短期资金流向，捕捉短期情绪变化\n"
        text += "    * 5日变化率：反映中期资金流向，判断中期趋势\n"
        text += "    * 20日变化率：反映长期资金流向，评估长期趋势\n"
        text += "  - 资金流向判断：\n"
        text += "    * 正向变化率：资金持续流入，市场活跃，支持交易\n"
        text += "    * 负向变化率：资金持续流出，市场低迷，减少交易\n"
        text += "    * 多周期一致：1日、5日、20日变化率同向，信号更可靠\n"
        text += "  - 应用场景：\n"
        text += "    * 成交额持续增长：资金流入，市场活跃，适合交易\n"
        text += "    * 成交额持续萎缩：资金流出，市场低迷，建议观望\n"
        text += "    * 突发性成交额变化：可能预示重大消息或趋势转折\n"
        text += "  - ML模型重要性：成交额变化率在长期预测（20天）中显著提升准确率\n"
        text += "• 换手率变化率：衡量市场关注度变化的指标，反映流动性增强或减弱：\n"
        text += "  - 计算方式：换手率 = 成交量 / 流通股本 × 100%；换手率变化率 = (当前换手率 - N日前换手率) / N日前换手率 × 100%\n"
        text += "  - 时间周期：\n"
        text += "    * 5日变化率：反映短期关注度变化，适合短期交易\n"
        text += "    * 20日变化率：反映中期关注度变化，适合中期投资\n"
        text += "  - 关注度判断：\n"
        text += "    * 换手率上升+换手率变化率正向：关注度提升，流动性增强，适合交易\n"
        text += "    * 换手率下降+换手率变化率负向：关注度下降，流动性减弱，观望为主\n"
        text += "    * 换手率异常波动：可能预示重大消息或趋势转折，提高警惕\n"
        text += "  - 应用场景：\n"
        text += "    * 换手率持续上升：市场关注度提升，流动性增强，适合交易\n"
        text += "    * 换手率持续下降：市场关注度下降，流动性减弱，建议观望\n"
        text += "    * 换手率异常波动：可能预示重大消息或趋势转折\n"
        text += "  - ML模型重要性：换手率变化率在长期预测（20天）中显著提升准确率\n"
        text += "• 成交量：股票在特定时间段内的交易数量，反映市场活跃度和资金流向：\n"
        text += "  - 成交量放大：通常表示市场关注度提高，可能预示趋势加速或反转\n"
        text += "  - 成交量萎缩：通常表示市场观望情绪浓厚，可能预示趋势减弱\n"
        text += "  - 量价配合：价格上涨+成交量放大=上涨动能强；价格下跌+成交量放大=下跌动能强\n"
        text += "  - 量价背离：价格上涨+成交量萎缩=上涨动能弱，可能回调；价格下跌+成交量萎缩=下跌动能弱，可能反弹\n"
        text += "• 均线排列：多条移动平均线的相对位置关系，判断市场趋势方向：\n"
        text += "  - 多头排列：短期均线(MA20) > 中期均线(MA50) > 长期均线(MA200)，且所有均线向上，强烈看涨信号\n"
        text += "  - 空头排列：短期均线(MA20) < 中期均线(MA50) < 长期均线(MA200)，且所有均线向下，强烈看空信号\n"
        text += "  - 混乱排列：均线交叉纠缠，没有明确方向，市场处于震荡整理阶段\n"
        text += "  - 应用：多头排列适合持有或加仓，空头排列适合减仓或清仓，混乱排列建议观望\n"
        text += "• MA20斜率：20日移动平均线的斜率和趋势方向，判断短期趋势强度：\n"
        text += "  - 上升：斜率为正，短期趋势向上，价格动能强劲\n"
        text += "  - 下降：斜率为负，短期趋势向下，价格动能疲弱\n"
        text += "  - 水平：斜率接近0，短期趋势平稳，市场处于整理状态\n"
        text += "  - 斜率大小：斜率绝对值越大，趋势强度越大，变化速度越快\n"
        text += "• MA50斜率：50日移动平均线的斜率和趋势方向，判断中期趋势强度：\n"
        text += "  - 上升：斜率为正，中期趋势向上，价格中期动能强劲\n"
        text += "  - 下降：斜率为负，中期趋势向下，价格中期动能疲弱\n"
        text += "  - 水平：斜率接近0，中期趋势平稳，市场处于中期整理状态\n"
        text += "  - 与MA20配合：MA20和MA50同时上升=中期上涨趋势确立；同时下降=中期下跌趋势确立\n"
        text += "• 均线乖离率：当前价格与移动平均线的偏离程度，评估超买超卖状态：\n"
        text += "  - 正乖离率：价格高于均线，表示价格上涨，乖离率过大=超买，可能回调\n"
        text += "  - 负乖离率：价格低于均线，表示价格下跌，乖离率过大=超卖，可能反弹\n"
        text += "  - 乖离率阈值：±3%以内=正常；±3-5%=轻微偏离；±5%以上=严重偏离\n"
        text += "  - 应用：乖离率过大时，考虑反向操作；乖离率回归0时，趋势可能延续\n"
        text += "• 最近支撑位：近期价格多次触及后反弹的价格水平，是重要的买入参考位：\n"
        text += "  - 识别方法：基于近期局部低点识别，价格多次在此位置获得支撑\n"
        text += "  - 支撑强度：触及次数越多，支撑越强；距离当前价格越近，参考价值越高\n"
        text += "  - 应用：价格接近支撑位时，可考虑买入或加仓；跌破支撑位时，可能开启新一轮下跌\n"
        text += "  - 止损设置：可将支撑位下方作为止损参考位\n"
        text += "• 最近阻力位：近期价格多次触及后回落的价格水平，是重要的卖出参考位：\n"
        text += "  - 识别方法：基于近期局部高点识别，价格多次在此位置遇到阻力\n"
        text += "  - 阻力强度：触及次数越多，阻力越强；距离当前价格越近，参考价值越高\n"
        text += "  - 应用：价格接近阻力位时，可考虑卖出或减仓；突破阻力位时，可能开启新一轮上涨\n"
        text += "  - 止盈设置：可将阻力位附近作为止盈参考位\n"
        text += "• 相对强度(相对恒指)：股票相对于恒生指数的表现，评估个股跑赢或跑输大盘：\n"
        text += "  - 正值：股票表现强于恒生指数，跑赢大盘，资金青睐\n"
        text += "  - 负值：股票表现弱于恒生指数，跑输大盘，资金流出\n"
        text += "  - 计算方式：相对强度 = (股票收益率 - 恒指收益率)\n"
        text += "  - 应用：相对强度持续上升=强势股，可重点关注；相对强度持续下降=弱势股，建议规避\n"
        text += "• 中期趋势评分(0-100)：综合趋势、动量、支撑阻力、相对强弱四维度的中期趋势评分：\n"
        text += "  - 评分范围：0-100分，分数越高中期趋势越强\n"
        text += "  - 评分等级：\n"
        text += "    * ≥80分：强烈买入，中期趋势强劲，建议重仓\n"
        text += "    * 65-79分：买入，中期趋势良好，建议加仓\n"
        text += "    * 45-64分：持有，中期趋势中性，建议观望\n"
        text += "    * 30-44分：卖出，中期趋势疲弱，建议减仓\n"
        text += "    * <30分：强烈卖出，中期趋势恶化，建议清仓\n"
        text += "  - 评分维度（各25%权重）：\n"
        text += "    * 趋势评分：基于均线排列和均线斜率\n"
        text += "    * 动量评分：基于价格变化率和MACD\n"
        text += "    * 支撑阻力评分：基于支撑阻力位距离和强度\n"
        text += "    * 相对强弱评分：基于相对恒指的表现\n"
        text += "• 中期趋势健康度：评估中期趋势的稳定性和可持续性：\n"
        text += "  - 健康：趋势稳定，波动性低，可持续性强，适合中长期持有\n"
        text += "  - 一般：趋势尚可，波动性中等，可持续性一般，需密切关注\n"
        text += "  - 较差：趋势不稳定，波动性高，可持续性弱，建议谨慎\n"
        text += "  - 评估因素：趋势稳定性、价格波动性、成交量一致性、均线发散程度\n"
        text += "  - 应用：健康度高=可放心持有；健康度低=需警惕趋势反转\n"
        text += "• 中期可持续性：评估中期趋势能够持续的可能性：\n"
        text += "  - 高：趋势有基本面支撑，资金持续流入，可持续性强\n"
        text += "  - 中：趋势有技术面支撑，资金流入稳定，可持续性中等\n"
        text += "  - 低：趋势缺乏支撑，资金流入不稳定，可持续性弱\n"
        text += "  - 评估因素：基本面质量、资金流向、市场情绪、行业景气度\n"
        text += "  - 应用：可持续性高=可中长期持有；可持续性低=建议短线操作\n"
        text += "• 中期建议：基于中期趋势评分、健康度、可持续性的综合投资建议：\n"
        text += "  - 强烈买入：中期趋势强劲且健康，建议重仓或加仓\n"
        text += "  - 买入：中期趋势良好，建议建仓或加仓\n"
        text += "  - 持有：中期趋势中性，建议继续持有或观望\n"
        text += "  - 卖出：中期趋势疲弱，建议减仓或清仓\n"
        text += "  - 强烈卖出：中期趋势恶化，建议清仓或规避\n"
        text += "  - 决策参考：结合基本面评分和技术指标综合判断\n"

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

    def run_analysis(self, target_date=None, force=False, send_email=True):
        """执行分析并发送邮件

        参数:
        - target_date: 分析日期，默认为今天
        - force: 是否强制发送邮件，即使没有交易信号，默认为 False
        - send_email: 是否发送邮件，默认为 True
        """
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

        # 获取美股市场数据（一次性获取，所有股票共享）
        print("📊 正在获取美股市场数据...")
        us_df = None
        try:
            from ml_services.us_market_data import us_market_data
            us_df = us_market_data.get_all_us_market_data(period_days=30)
            if us_df is not None and not us_df.empty:
                print(f"✅ 美股数据获取成功（VIX: {us_df.get('VIX_Level', pd.Series([None])).iloc[-1] if 'VIX_Level' in us_df.columns else 'N/A'}）")
            else:
                print("⚠️ 美股数据为空")
        except Exception as e:
            print(f"⚠️ 获取美股数据失败: {e}")

        print(f"🔍 正在获取股票列表并分析 ({len(self.stock_list)} 只股票)...")
        stock_results = []
        for stock_code, stock_name in self.stock_list.items():
            print(f"🔍 正在分析 {stock_name} ({stock_code}) ...")
            stock_data = self.get_stock_data(stock_code, target_date=target_date)
            if stock_data:
                print(f"📊 正在计算 {stock_name} ({stock_code}) 技术指标...")
                indicators = self.calculate_technical_indicators(stock_data, us_df=us_df)
                stock_results.append({
                    'code': stock_code,
                    'name': stock_name,
                    'data': stock_data,
                    'indicators': indicators
                })

        has_signals = self.has_any_signals(hsi_indicators, stock_results, target_date)

        if not has_signals:
            if not force:
                print("⚠️ 没有检测到任何交易信号，跳过发送邮件。")
                return False
            else:
                print("⚡ 强制模式：没有交易信号，但仍然发送邮件")

        # 根据是否有信号调整主题
        if has_signals:
            subject = "恒生指数及港股交易信号提醒 - 包含最近48小时模拟交易记录"
        else:
            subject = "恒生指数及港股市场分析报告 - 无交易信号"

        text, html = self.generate_report_content(target_date, hsi_data, hsi_indicators, stock_results)

        recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        if ',' in recipient_env:
            recipients = [recipient.strip() for recipient in recipient_env.split(',')]
        else:
            recipients = [recipient_env]

        if send_email:
            if has_signals:
                print("🔔 检测到交易信号，发送邮件到:", ", ".join(recipients))
            else:
                print("📊 发送市场分析报告到:", ", ".join(recipients))
            print("📝 主题:", subject)
            print("📄 文本预览:\n", text)

            success = self.send_email(recipients, subject, text, html)
            return success
        else:
            print("📄 仅生成模式：跳过邮件发送")
            print("📝 主题:", subject)
            print("📄 内容已生成，但未发送")
            return True

    def save_llm_recommendations(self, portfolio_analysis, buy_signals_analysis, target_date=None):
        """
        保存大模型建议到文本文件，方便后续提取和对比

        参数:
        - portfolio_analysis: 持仓分析结果（大模型建议）
        - buy_signals_analysis: 买入信号分析结果（大模型建议）
        - target_date: 分析日期
        """
        try:
            from datetime import datetime

            # 生成文件名（使用日期）
            if target_date:
                if isinstance(target_date, str):
                    date_str = target_date
                else:
                    date_str = target_date.strftime('%Y-%m-%d')
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')

            # 创建data目录（如果不存在）
            if not os.path.exists('data'):
                os.makedirs('data')

            # 文件路径
            filepath = f'data/llm_recommendations_{date_str}.txt'

            # 构建内容
            content = f"{'=' * 80}\n"
            content += f"大模型买卖建议报告\n"
            content += f"日期: {date_str}\n"
            content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"{'=' * 80}\n\n"

            # 添加持仓分析（中期建议）
            if portfolio_analysis:
                content += "【中期建议】持仓分析\n"
                content += "-" * 80 + "\n"
                content += portfolio_analysis + "\n\n"

            # 添加买入信号分析（短期建议）
            if buy_signals_analysis:
                content += "【短期建议】买入信号分析\n"
                content += "-" * 80 + "\n"
                content += buy_signals_analysis + "\n\n"

            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 大模型建议已保存到 {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ 保存大模型建议失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# === 主逻辑 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='恒生指数及港股主力资金追踪器股票交易信号邮件通知系统')
    parser.add_argument('--date', type=str, default=None, help='指定日期 (格式: YYYY-MM-DD)，默认为今天')
    parser.add_argument('--force', action='store_true', help='强制发送邮件，即使没有交易信号')
    parser.add_argument('--no-email', action='store_true', help='不发送邮件，只生成分析报告')
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

    if args.force:
        print("⚡ 强制模式：即使没有交易信号也会发送邮件")
    
    if args.no_email:
        print("📄 仅生成模式：不发送邮件，只生成分析报告")

    email_system = HSIEmailSystem()
    success = email_system.run_analysis(target_date, force=args.force, send_email=not args.no_email)

    if not success:
        exit(1)


# 为其他模块提供访问接口的函数
def get_hsi_and_stock_indicators(stock_list=None, target_date=None):
    """
    为其他模块提供获取恒生指数和股票指标的接口
    
    参数:
    - stock_list: 股票列表，默认使用全局配置
    - target_date: 目标日期，默认为今天
    
    返回:
    - dict: 包含恒生指数数据和股票分析结果
    """
    email_system = HSIEmailSystem(stock_list=stock_list)
    
    # 获取恒生指数数据
    hsi_data = email_system.get_hsi_data(target_date=target_date)
    
    # 获取美股市场数据（一次性获取，所有股票共享）
    us_df = None
    try:
        from ml_services.us_market_data import us_market_data
        us_df = us_market_data.get_all_us_market_data(period_days=30)
        if us_df is not None and not us_df.empty:
            print(f"✅ 美股数据获取成功（VIX: {us_df.get('VIX_Level', pd.Series([None])).iloc[-1] if 'VIX_Level' in us_df.columns else 'N/A'}）")
        else:
            print("⚠️ 美股数据为空")
    except Exception as e:
        print(f"⚠️ 获取美股数据失败: {e}")

    # 获取股票分析结果
    stock_results = []
    for stock_code, stock_name in email_system.stock_list.items():
        print(f"🔍 正在分析 {stock_name} ({stock_code}) ...")
        stock_data = email_system.get_stock_data(stock_code, target_date=target_date)
        if stock_data:
            print(f"📊 正在计算 {stock_name} ({stock_code}) 技术指标...")
            indicators = email_system.calculate_technical_indicators(stock_data, us_df=us_df)
            stock_results.append({
                'code': stock_code,
                'name': stock_name,
                'data': stock_data,
                'indicators': indicators
            })
    
    return {
        'hsi_data': hsi_data,
        'hsi_indicators': email_system.calculate_hsi_technical_indicators(hsi_data) if hsi_data else None,
        'stock_results': stock_results
    }


def get_stock_technical_indicators(stock_code, target_date=None):
    """
    获取单只股票的详细技术指标（与comprehensive_analysis.py兼容的函数）
    
    参数:
    - stock_code: 股票代码（如 "0700.HK"）
    - target_date: 目标日期
    
    返回:
    - dict: 包含详细技术指标的字典
    """
    email_system = HSIEmailSystem()
    stock_data = email_system.get_stock_data(stock_code, target_date=target_date)
    
    if stock_data:
        # 获取美股数据
        us_df = None
        try:
            from ml_services.us_market_data import us_market_data
            us_df = us_market_data.get_all_us_market_data(period_days=30)
        except Exception:
            pass
        
        indicators = email_system.calculate_technical_indicators(stock_data, us_df=us_df)
        return {
            'current_price': indicators.get('current_price', stock_data.get('current_price')),
            'change_pct': stock_data.get('change_1d'),
            'rsi': indicators.get('rsi'),
            'macd': indicators.get('macd'),
            'macd_signal': indicators.get('macd_signal'),
            'ma20': indicators.get('ma20'),
            'ma50': indicators.get('ma50'),
            'ma200': indicators.get('ma200'),
            'ma_alignment': indicators.get('ma_alignment'),
            'ma_slope_20': indicators.get('ma20_slope'),
            'ma_slope_50': indicators.get('ma50_slope'),
            'ma_deviation': indicators.get('ma_deviation_avg'),
            'bb_upper': indicators.get('bb_position'),  # 实际是布林带位置，不是上轨
            'bb_lower': indicators.get('bb_position'),
            'bb_position': indicators.get('bb_position'),
            'atr': indicators.get('atr'),
            'volume': stock_data.get('volume'),
            'trend': indicators.get('trend'),
            'support_level': indicators.get('nearest_support'),
            'resistance_level': indicators.get('nearest_resistance'),
            'fundamental_score': indicators.get('fundamental_score'),
            'pe_ratio': indicators.get('pe_ratio'),
            'pb_ratio': indicators.get('pb_ratio'),
            'medium_term_score': indicators.get('medium_term_score'),
            'vix_level': indicators.get('vix_level'),
            'turnover_change_1d': indicators.get('turnover_change_1d'),
            'turnover_rate_change_5d': indicators.get('turnover_rate_change_5d'),
            'buildup_score': indicators.get('buildup_score'),
            'distribution_score': indicators.get('distribution_score'),
            'tav_score': indicators.get('tav_score'),
        }
    return None
