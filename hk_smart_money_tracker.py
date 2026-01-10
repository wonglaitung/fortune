# -*- coding: utf-8 -*-
"""
港股主力资金追踪器（建仓 + 出货 双信号）- 完整改进版（含完整版指标说明）
作者：AI助手（修补与重构版）
说明（要点）：
- 所有关键阈值已集中到顶部配置区，便于调参。
- 相对强度 RS_ratio = (1+stock_ret)/(1+hsi_ret)-1（数据层为小数），RS_diff = stock_ret - hsi_ret（小数）。
  输出/展示统一以百分比显示（乘 100 并带 %）。
- outperforms 判定支持三种语义：绝对正收益并跑赢、相对跑赢（收益差值）、基于 RS_ratio（复合收益比）。
- RSI 使用 Wilder 平滑（更接近经典 RSI）。
- OBV 使用 full history 的累计值，避免短期截断。
- 南向资金（ak 返回）会被缓存并转换为"万"（可调整 SOUTHBOUND_UNIT_CONVERSION）。
- 连续天数判定（建仓/出货）采用显式的 run-length 标注整段满足条件的日期。
- 输出：DataFrame 中保留原始数值（小数），显示及邮件中对 RS2 指标以百分比展示，并在说明中明确单位。
"""

import warnings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import math
import time
import argparse
from datetime import datetime, timedelta
import re

warnings.filterwarnings("ignore")
os.environ['MPLBACKEND'] = 'Agg'

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入腾讯财经接口
from tencent_finance import get_hk_stock_data_tencent, get_hk_stock_info_tencent

# 导入大模型服务
from llm_services import qwen_engine

# 导入基本面数据模块
from fundamental_data import get_comprehensive_fundamental_data

# 导入技术分析工具和TAV系统
try:
    from technical_analysis import TechnicalAnalyzer, TechnicalAnalyzerV2, TAVScorer, TAVConfig
    TECHNICAL_ANALYSIS_AVAILABLE = True
    TAV_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    TAV_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用原有分析逻辑")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 用户设置区（所有重要阈值集中于此）
# ==============================
WATCHLIST = {
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

# 窗口与样本
DAYS_ANALYSIS = 12
VOL_WINDOW = 20
PRICE_WINDOW = 60
BUILDUP_MIN_DAYS = 3
DISTRIBUTION_MIN_DAYS = 2

# 阈值（可调）
PRICE_LOW_PCT = 40.0   # 价格百分位低于该值视为"低位"
PRICE_HIGH_PCT = 60.0  # 高于该值视为"高位"
VOL_RATIO_BUILDUP = 1.3
VOL_RATIO_DISTRIBUTION = 2.0

# 南向资金：ak 返回的单位可能是"元"，将其除以此因子转换为"万"
SOUTHBOUND_UNIT_CONVERSION = 10000.0
SOUTHBOUND_THRESHOLD = 3000.0  # 单位：万

# outperforms 判定：三种语义选择
# 默认行为保持向后兼容（要求正收益并高于恒指）
OUTPERFORMS_REQUIRE_POSITIVE = True
# 如果 True，则优先用 RS_ratio > 0 判定（相对跑赢）
OUTPERFORMS_USE_RS = False

# 展示与保存
SAVE_CHARTS = True
CHART_DIR = "hk_smart_charts"
if SAVE_CHARTS and not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# 其它
AK_CALL_SLEEP = 0.1  # 调用 ak 时的短暂停顿以避免限流

# ==============================
# 加权评分系统参数（新增）
# ==============================

# 建仓信号权重配置
BUILDUP_WEIGHTS = {
    'price_low': 2.0,      # 价格处于低位
    'vol_ratio': 2.0,      # 成交量放大
    'vol_z': 1.0,          # 成交量z-score
    'macd_cross': 1.5,     # MACD金叉
    'rsi_oversold': 1.2,   # RSI超卖
    'obv_up': 1.0,         # OBV上升
    'vwap_vol': 1.2,       # 价格高于VWAP且放量
    'southbound_in': 1.8,  # 南向资金流入
    'cmf_in': 1.2,         # CMF资金流入
    'price_above_vwap': 0.8,  # 价格高于VWAP
}

# 建仓信号阈值
BUILDUP_THRESHOLD_STRONG = 5.0   # 强烈建仓信号阈值
BUILDUP_THRESHOLD_PARTIAL = 3.0  # 部分建仓信号阈值
SOUTHBOUND_THRESHOLD_IN = 1000.0  # 南向资金流入阈值（万）

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
    'southbound_out': 2.0, # 南向资金流出
    'price_down': 1.0,     # 价格下跌
}

# 出货信号阈值
DISTRIBUTION_THRESHOLD_STRONG = 5.0   # 强烈出货信号阈值
DISTRIBUTION_THRESHOLD_WEAK = 3.0     # 弱出货信号阈值
SOUTHBOUND_THRESHOLD_OUT = 1000.0     # 南向资金流出阈值（万）

# 止盈和止损参数
TAKE_PROFIT_PCT = 0.10      # 止盈百分比（10%）
PARTIAL_SELL_PCT = 0.3      # 部分卖出比例（30%）
TRAILING_ATR_MULT = 2.5     # ATR trailing stop倍数
STOP_LOSS_PCT = 0.15        # 止损百分比（15%）

# 是否启用加权评分系统（向后兼容）
USE_SCORED_SIGNALS = True   # True=使用新的评分系统，False=使用原有的布尔逻辑

# ==============================
# 2. 获取恒生指数数据 (使用腾讯财经接口)
# ==============================
print("📈 获取恒生指数（HSI）用于对比...")
from tencent_finance import get_hsi_data_tencent
hsi_hist = get_hsi_data_tencent(period_days=PRICE_WINDOW + 30)  # 余量更大以防节假日
# 注意：如果无法获取恒生指数数据，hsi_hist 可能为 None
# 在这种情况下，相对强度计算将不可用

def get_hsi_return(start, end):
    """
    使用前向/后向填充获取与股票时间戳对齐的恒指价格，返回区间收益（小数）。
    start/end 为 Timestamp（来自股票索引）。
    若无法获取，则返回 np.nan。
    """
    # 如果恒生指数数据不可用，返回 np.nan
    if hsi_hist is None or hsi_hist.empty:
        return np.nan
        
    try:
        s = hsi_hist['Close'].reindex([start], method='ffill').iloc[0]
        e = hsi_hist['Close'].reindex([end], method='ffill').iloc[0]
        if pd.isna(s) or pd.isna(e) or s == 0:
            return np.nan
        return (e - s) / s
    except Exception:
        return np.nan

# ==============================
# 3. 辅助函数与缓存（包括南向资金缓存，避免重复调用 ak）
# ==============================
southbound_cache = {}  # cache[(code, date_str)] = DataFrame from ak or cache[code] = full DataFrame

def fetch_ggt_components(code, date_str):
    """
    从 ak 获取指定股票和日期的港股南向资金数据，并缓存。
    date_str 格式 YYYYMMDD
    返回 DataFrame 或 None
    """
    cache_key = (code, date_str)
    if cache_key in southbound_cache:
        return southbound_cache[cache_key]
    
    import threading
    
    def fetch_with_timeout(symbol, timeout=3):
        """带超时的数据获取函数"""
        result = None
        exception = None
        
        def worker():
            nonlocal result, exception
            try:
                result = ak.stock_hsgt_individual_em(symbol=symbol)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            # 超时了，线程还在运行
            return None, "timeout"
        
        return result, exception
    
    try:
        # 使用新的接口获取个股南向资金数据
        # akshare要求股票代码为5位数字格式，不足5位的需要在前面补0
        symbol = code.replace('.HK', '')
        if len(symbol) < 5:
            symbol = symbol.zfill(5)
        elif len(symbol) > 5:
            # 如果超过5位，取后5位（处理像 "00700.HK" 这样的格式）
            symbol = symbol[-5:]
        
        # 检查缓存中是否已有该股票的数据
        stock_cache_key = symbol
        if stock_cache_key in southbound_cache and southbound_cache[stock_cache_key] is not None:
            df_individual = southbound_cache[stock_cache_key]
        else:
            # 获取个股南向资金数据（带10秒超时）
            df_individual, exception = fetch_with_timeout(symbol, timeout=10)
            
            # 检查是否超时
            if exception == "timeout":
                print(f"⚠️ 获取南向资金数据超时 {code} {date_str}，跳过")
                southbound_cache[stock_cache_key] = None
                time.sleep(AK_CALL_SLEEP)
                return None
            
            # 检查是否有其他异常
            if exception is not None:
                print(f"⚠️ 获取南向资金数据失败 {code} {date_str}: {exception}")
                southbound_cache[stock_cache_key] = None
                time.sleep(AK_CALL_SLEEP)
                return None
            
            # 检查返回的数据是否有效
            if df_individual is None or not isinstance(df_individual, pd.DataFrame) or df_individual.empty:
                print(f"⚠️ 获取南向资金数据为空 {code}")
                southbound_cache[stock_cache_key] = None
                time.sleep(AK_CALL_SLEEP)
                return None
            
            # 缓存该股票的所有数据
            southbound_cache[stock_cache_key] = df_individual
        
        # 检查DataFrame是否有效
        if not isinstance(df_individual, pd.DataFrame) or df_individual.empty:
            print(f"⚠️ 南向资金数据无效 {code}")
            southbound_cache[cache_key] = None
            time.sleep(AK_CALL_SLEEP)
            return None
        
        # 确保持股日期列是datetime类型
        if '持股日期' in df_individual.columns:
            df_individual['持股日期'] = pd.to_datetime(df_individual['持股日期'])
        
        # 将日期字符串转换为pandas日期格式进行匹配
        target_date = pd.to_datetime(date_str, format='%Y%m%d')
        
        # 筛选指定日期的数据
        df_filtered = df_individual[df_individual['持股日期'] == target_date.date()]
        
        # 如果未找到指定日期的数据，使用最近的可用日期数据
        if df_filtered.empty:
            # 获取所有可用日期
            available_dates = df_individual['持股日期']
            # 找到最近的日期（小于或等于目标日期）
            closest_date = available_dates[available_dates <= target_date].max()
            
            if pd.notna(closest_date):
                # 使用最近日期的数据
                df_filtered = df_individual[df_individual['持股日期'] == closest_date]
                print(f"⚠️ 未找到指定日期的南向资金数据 {code} {date_str}，使用最近日期 {closest_date.strftime('%Y%m%d')} 的数据")
            else:
                print(f"⚠️ 未找到指定日期及之前的南向资金数据 {code} {date_str}")
                southbound_cache[cache_key] = None
                time.sleep(AK_CALL_SLEEP)
                return None
        
        if isinstance(df_filtered, pd.DataFrame) and not df_filtered.empty:
            # 只返回需要的列以减少内存占用
            result = df_filtered[['持股日期', '持股市值变化-1日']].copy()
            southbound_cache[cache_key] = result
            # 略微延时以防被限流
            time.sleep(AK_CALL_SLEEP)
            return result
        else:
            print(f"⚠️ 未找到指定日期的南向资金数据 {code} {date_str}")
            southbound_cache[cache_key] = None
            time.sleep(AK_CALL_SLEEP)
            return None
    except Exception as e:
        print(f"⚠️ 获取南向资金数据失败 {code} {date_str}: {e}")
        southbound_cache[cache_key] = None
        time.sleep(AK_CALL_SLEEP)
        return None

def mark_runs(signal_series, min_len):
    """
    将 signal_series 中所有连续 True 的段标注为 True（整段），仅当段长度 >= min_len
    返回与 signal_series 相同索引的布尔 Series
    """
    res = pd.Series(False, index=signal_series.index)
    s = signal_series.fillna(False).astype(bool).values
    n = len(s)
    i = 0
    while i < n:
        if s[i]:
            j = i
            while j < n and s[j]:
                j += 1
            if (j - i) >= min_len:
                res.iloc[i:j] = True
            i = j
        else:
            i += 1
    return res

def mark_scored_runs(signal_level_series, min_len, min_level='partial'):
    """
    将分级信号中连续满足条件的段标注为确认信号
    
    Args:
        signal_level_series: 信号级别Series ('none', 'partial', 'strong')
        min_len: 最小连续天数
        min_level: 最低确认级别 ('partial' 或 'strong')
    
    Returns:
        确认信号Series (布尔值)
    """
    # 将信号级别转换为布尔值
    if min_level == 'strong':
        signal_bool = signal_level_series.isin(['strong'])
    else:  # 'partial'
        signal_bool = signal_level_series.isin(['partial', 'strong'])
    
    res = pd.Series(False, index=signal_level_series.index)
    s = signal_bool.fillna(False).astype(bool).values
    n = len(s)
    i = 0
    while i < n:
        if s[i]:
            j = i
            while j < n and s[j]:
                j += 1
            if (j - i) >= min_len:
                res.iloc[i:j] = True
            i = j
        else:
            i += 1
    return res

def safe_round(v, ndigits=2):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float, np.floating, np.integer)):
            if not math.isfinite(float(v)):
                return v
            return round(float(v), ndigits)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return v

def build_llm_analysis_prompt(stock_data, run_date=None, market_metrics=None):
    """
    构建发送给大模型的股票数据分析提示词
    
    Args:
        stock_data (list): 股票数据分析结果列表
        run_date (str): 指定的运行日期
        market_metrics (dict): 市场整体指标
        
    Returns:
        str: 构建好的提示词
    """
    # 构建股票数据表格 (CSV格式) - 使用新的评分系统字段和多周期指标
    csv_header = "股票名称,代码,最新价,涨跌幅(%),位置(%),量比,成交量z-score,成交额z-score,成交金额(百万),换手率(%),VWAP,成交量比率,成交量比率信号,ATR,ATR比率,ATR比率信号,布林带宽度(%),布林带突破,波动率(%),5日均线偏离(%),10日均线偏离(%),RSI,RSI变化率,RSI背离信号,MACD,MACD柱状图,MACD柱状图变化率,MACD柱状图变化率信号,OBV,CMF,CMF信号线,CMF趋势信号,随机振荡器K,随机振荡器D,随机振荡器信号,Williams %R,Williams %R信号,布林带突破信号,价格变化率信号,南向资金(万),相对强度(RS_ratio_%),相对强度差值(RS_diff_%),跑赢恒指,基本面评分,市盈率,市净率,建仓评分,建仓级别,建仓原因,出货评分,出货级别,出货原因,止盈,止损,Trailing Stop,放量上涨,缩量回调,TAV评分,TAV状态,上个交易日RSI,上个交易日MACD,上个交易日价格,上个交易日建仓评分,上个交易日出货评分,上个交易日TAV评分,3日收益率(%),3日趋势,5日收益率(%),5日趋势,10日收益率(%),10日趋势,20日收益率(%),20日趋势,60日收益率(%),60日趋势,3日相对强度(%),3日相对强度信号,5日相对强度(%),5日相对强度信号,10日相对强度(%),10日相对强度信号,20日相对强度(%),20日相对强度信号,60日相对强度(%),60日相对强度信号,多周期趋势评分,多周期相对强度评分"    
    csv_rows = []
    for stock in stock_data:
        # 正确处理相对强度指标的转换
        rs_ratio_value = stock.get('relative_strength')
        rs_ratio_pct = round(rs_ratio_value * 100, 2) if rs_ratio_value is not None else 'N/A'
        
        rs_diff_value = stock.get('relative_strength_diff')
        rs_diff_pct = round(rs_diff_value * 100, 2) if rs_diff_value is not None else 'N/A'
        
        # 获取上个交易日指标
        prev_day_indicators = stock.get('prev_day_indicators', {})
        prev_rsi = prev_day_indicators.get('rsi', 'N/A') if prev_day_indicators else 'N/A'
        prev_macd = prev_day_indicators.get('macd', 'N/A') if prev_day_indicators else 'N/A'
        prev_price = prev_day_indicators.get('price', 'N/A') if prev_day_indicators else 'N/A'
        prev_buildup_score = prev_day_indicators.get('buildup_score', 'N/A') if prev_day_indicators else 'N/A'
        prev_distribution_score = prev_day_indicators.get('distribution_score', 'N/A') if prev_day_indicators else 'N/A'
        prev_tav_score = prev_day_indicators.get('tav_score', 'N/A') if prev_day_indicators else 'N/A'
        
        # 获取多周期指标
        multi_period_3d_return = stock.get('3d_return', 'N/A')
        multi_period_3d_trend = stock.get('3d_trend', 'N/A')
        multi_period_5d_return = stock.get('5d_return', 'N/A')
        multi_period_5d_trend = stock.get('5d_trend', 'N/A')
        multi_period_10d_return = stock.get('10d_return', 'N/A')
        multi_period_10d_trend = stock.get('10d_trend', 'N/A')
        multi_period_20d_return = stock.get('20d_return', 'N/A')
        multi_period_20d_trend = stock.get('20d_trend', 'N/A')
        multi_period_60d_return = stock.get('60d_return', 'N/A')
        multi_period_60d_trend = stock.get('60d_trend', 'N/A')
        
        # 获取多周期相对强度
        multi_period_3d_rs = stock.get('3d_rs', 'N/A')
        multi_period_3d_rs_signal = stock.get('3d_rs_signal', 'N/A')
        multi_period_5d_rs = stock.get('5d_rs', 'N/A')
        multi_period_5d_rs_signal = stock.get('5d_rs_signal', 'N/A')
        multi_period_10d_rs = stock.get('10d_rs', 'N/A')
        multi_period_10d_rs_signal = stock.get('10d_rs_signal', 'N/A')
        multi_period_20d_rs = stock.get('20d_rs', 'N/A')
        multi_period_20d_rs_signal = stock.get('20d_rs_signal', 'N/A')
        multi_period_60d_rs = stock.get('60d_rs', 'N/A')
        multi_period_60d_rs_signal = stock.get('60d_rs_signal', 'N/A')
        
        # 获取多周期综合评分
        multi_period_trend_score = stock.get('multi_period_trend_score', 'N/A')
        multi_period_rs_score = stock.get('multi_period_rs_score', 'N/A')
        
        # 使用新的评分系统字段和多周期指标
        row = f"{stock['name']},{stock['code']},{stock['last_close'] or 'N/A'},{stock['change_pct'] or 'N/A'},{stock['price_percentile'] or 'N/A'},{stock['vol_ratio'] or 'N/A'},{stock['vol_z_score'] or 'N/A'},{stock['turnover_z_score'] or 'N/A'},{stock['turnover'] or 'N/A'},{stock['turnover_rate'] or 'N/A'},{stock['vwap'] or 'N/A'},{stock['volume_ratio'] or 'N/A'},{int(stock['volume_ratio_signal'])},{stock['atr'] or 'N/A'},{stock['atr_ratio'] or 'N/A'},{int(stock['atr_ratio_signal'])},{stock['bb_width'] or 'N/A'},{stock['bb_breakout'] or 'N/A'},{stock['volatility'] or 'N/A'},{stock['ma5_deviation'] or 'N/A'},{stock['ma10_deviation'] or 'N/A'},{stock['rsi'] or 'N/A'},{stock['rsi_roc'] or 'N/A'},{int(stock['rsi_divergence'])},{stock['macd'] or 'N/A'},{stock['macd_hist'] or 'N/A'},{stock['macd_hist_roc'] or 'N/A'},{int(stock['macd_hist_roc_signal'])},{stock['obv'] or 'N/A'},{stock['cmf'] or 'N/A'},{stock['cmf_signal'] or 'N/A'},{int(stock['cmf_trend_signal'])},{stock['stoch_k'] or 'N/A'},{stock['stoch_d'] or 'N/A'},{int(stock['stoch_signal'])},{stock['williams_r'] or 'N/A'},{int(stock['williams_r_signal'])},{int(stock['bb_breakout_signal'])},{stock['roc_signal'] or 'N/A'},{stock['southbound'] or 'N/A'},{rs_ratio_pct},{rs_diff_pct},{int(stock['outperforms_hsi'])},{stock.get('fundamental_score', 'N/A')},{stock.get('pe_ratio', 'N/A')},{stock.get('pb_ratio', 'N/A')},{stock.get('buildup_score', 'N/A') or 'N/A'},{stock.get('buildup_level', 'N/A') or 'N/A'},{stock.get('buildup_reasons', 'N/A') or 'N/A'},{stock.get('distribution_score', 'N/A') or 'N/A'},{stock.get('distribution_level', 'N/A') or 'N/A'},{stock.get('distribution_reasons', 'N/A') or 'N/A'},{int(stock.get('take_profit', False))},{int(stock.get('stop_loss', False))},{int(stock.get('trailing_stop', False))},{int(stock['strong_volume_up'])},{int(stock['weak_volume_down'])},{stock['tav_score'] or 'N/A'},{stock['tav_status'] or 'N/A'},{prev_rsi},{prev_macd},{prev_price},{prev_buildup_score},{prev_distribution_score},{prev_tav_score},{multi_period_3d_return},{multi_period_3d_trend},{multi_period_5d_return},{multi_period_5d_trend},{multi_period_10d_return},{multi_period_10d_trend},{multi_period_20d_return},{multi_period_20d_trend},{multi_period_60d_return},{multi_period_60d_trend},{multi_period_3d_rs},{multi_period_3d_rs_signal},{multi_period_5d_rs},{multi_period_5d_rs_signal},{multi_period_10d_rs},{multi_period_10d_rs_signal},{multi_period_20d_rs},{multi_period_20d_rs_signal},{multi_period_60d_rs},{multi_period_60d_rs_signal},{multi_period_trend_score},{multi_period_rs_score}"
        csv_rows.append(row)
    
    stock_table = csv_header + "\n" + "\n".join(csv_rows)
    
    # 检查是否存在新闻数据文件
    news_content = ""
    news_file_path = "data/all_stock_news_records.csv"
    if os.path.exists(news_file_path):
        try:
            # 读取CSV文件
            news_df = pd.read_csv(news_file_path)
            
            # 只保留WATCHLIST中的股票新闻
            watchlist_codes = list(WATCHLIST.keys())
            news_df = news_df[news_df['股票代码'].isin(watchlist_codes)]
            
            if not news_df.empty:
                # 构建新闻数据表格
                news_table_header = "| 股票名称 | 股票代码 | 新闻时间 | 新闻标题 | 简要内容 |\n"
                news_table_separator = "|----------|----------|----------|----------|----------|\n"
                
                news_table_rows = []
                for _, row in news_df.iterrows():
                    news_row = f"| {row['股票名称']} | {row['股票代码']} | {row['新闻时间']} | {row['新闻标题']} | {row['简要内容']} |"
                    news_table_rows.append(news_row)
                
                news_table = news_table_header + news_table_separator + "\n".join(news_table_rows)
                
                news_content = f"""
 additionally, here is recent news data for the stocks in your WATCHLIST:

{news_table}
"""
            else:
                news_content = "\n additionally, there is currently no relevant news data for the stocks in your WATCHLIST."
        except Exception as e:
            print(f"⚠️ 读取新闻数据文件失败: {e}")
            news_content = "\n additionally, unable to access news data due to an error."
    
    # 获取当前日期和时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 获取当前恒生指数
    current_hsi = "未知"
    if hsi_hist is not None and not hsi_hist.empty:
        current_hsi = hsi_hist['Close'].iloc[-1]
    
    # 构建市场整体指标内容
    market_context = ""
    if market_metrics:
        market_context = f"""
                
                📊 市场整体指标：
                - 总股票数：{market_metrics['total_stocks']}
                - 建仓信号股票数：{market_metrics['buildup_stocks_count']}
                - 出货信号股票数：{market_metrics['distribution_stocks_count']}
                - 跑赢恒指股票数：{market_metrics['outperforming_stocks_count']}
                - 平均相对强度：{market_metrics['avg_relative_strength']:.4f}
                - 平均市场波动率：{market_metrics['avg_market_volatility']:.2f}%
                - 平均量比：{market_metrics['avg_vol_ratio']:.2f}
                - 市场情绪：{market_metrics['market_sentiment']}
                - 总南向资金净流入：{market_metrics['total_southbound_net']:.2f}万
                - 恒生指数当前值：{market_metrics['hsi_current']}
                - 市场活跃度：{market_metrics['market_activity_level']}
                """
    
    # 构建提示词
    prompt = f"""
你是一位专业的港股技术分析师，请基于以下技术指标数据和新闻信息进行综合分析：

📊 分析背景：
- 时间：{current_time}
- 数据日期：{run_date if run_date else '最新数据'}
- 恒生指数：{current_hsi}
- 分析股票：{len(stock_data)}只

{market_context}

📋 技术指标数据（CSV格式）：
{stock_table}

{news_content}

⚠️ 重要说明：

🔴 优先级说明（按重要性排序）：
1. **风险控制信号（最高优先级）**：止损(1) > 止盈(1) > Trailing Stop(1)
2. **建仓/出货评分**：strong > weak > none
3. **多周期趋势评分**：综合趋势评分 > 单个周期趋势
4. **基本面评分**：基本面质量支撑技术信号
5. **其他技术指标**：辅助验证

📊 关键指标速查表：

【风险控制指标（最高优先级）】
- **止损(1)**：亏损≥15%触发，立即全部卖出止损，风险等级：🔴极高
- **止盈(1)**：盈利≥10%且配合技术信号触发，建议卖出30%锁定利润，风险等级：🟡高
- **Trailing Stop(1)**：价格从20日高点回撤超过2.5倍ATR触发，建议卖出30%保护利润，风险等级：🟡高

【建仓/出货评分系统】
- **建仓评分分级**：
  - strong（评分≥5.0）：强烈建仓信号，建议仓位40-60%
  - partial（3.0≤评分<5.0）：部分建仓信号，建议仓位20-40%
  - none（评分<3.0）：无建仓信号，建议仓位0-10%
- **出货评分分级**：
  - strong（评分≥5.0）：强烈出货信号，建议卖出60-100%
  - weak（3.0≤评分<5.0）：弱出货信号，建议卖出30-60%
  - none（评分<3.0）：无出货信号，建议持有

【多周期指标（趋势判断核心）】
- **多周期趋势评分**（-100到100）：
  - >50：强势多头趋势，建议积极建仓
  - 20-50：多头趋势，建议正常建仓
  - -20-20：震荡趋势，建议观望
  - -50--20：空头趋势，建议减仓
  - <-50：强势空头趋势，建议清仓
- **多周期相对强度评分**（-100到100）：
  - >50：显著跑赢恒指，优先选择
  - 20-50：跑赢恒指，可以建仓
  - -20-20：持平恒指，中性对待
  - -50--20：跑输恒指，谨慎对待
  - <-50：显著跑输恒指，避免建仓
- **趋势一致性判断**：
  - 3/5/10/20/60日均为上涨：强势多头，建仓信心高
  - 短期上涨+中长期下跌：反弹而非反转，谨慎建仓
  - 短期下跌+中长期上涨：回调而非反转，加仓机会

【基本面指标】
- **基本面评分**（0-100）：
  - >60：优秀，大幅提升建仓信号可靠性
  - 40-60：良好，提升建仓信号可靠性
  - 20-40：一般，建仓信号需谨慎
  - <20：差，建仓信号不可靠
- **估值指标**：
  - PE<10且PB<1：低估值，安全边际高
  - PE 10-15且PB 1-1.5：合理估值
  - PE>20或PB>2：高估值，风险较高

【技术指标（辅助验证）】
- **RSI**：>70超买，<30超卖，40-60正常
- **MACD**：金叉买入，死叉卖出
- **布林带**：突破上轨超买，突破下轨超卖
- **成交量比率**：>1.5放量，<0.5缩量
- **ATR比率**：>1.5波动放大，<0.5波动收敛
- **TAV评分**：趋势-加速度-成交量三维分析的综合评分(0-100)，>70强势，<30弱势

🎯 核心分析框架（决策树）：

【第一步：风险控制检查（最高优先级）】
⚠️ 必须首先检查所有股票的风险控制信号：
- 对于止损(1)的股票：立即全部卖出，风险等级🔴极高，在风险警示中重点标注
- 对于止盈(1)的股票：建议卖出30%锁定利润，风险等级🟡高
- 对于Trailing Stop(1)的股票：建议卖出30%保护利润，风险等级🟡高
- 统计触发风险控制信号的股票数量和比例

【第二步：建仓/出货信号识别】
🟢 建仓信号筛选：
- 建仓级别=strong（评分≥5.0）：强烈建仓信号
  - 评分5.0-7.0：建议仓位40-50%
  - 评分7.0-10.0：建议仓位50-60%
  - 评分>10.0：建议仓位60-70%
- 建仓级别=partial（3.0≤评分<5.0）：部分建仓信号，建议仓位20-40%
- 结合基本面评分调整仓位：
  - 基本面评分>60：仓位可增加10-20%
  - 基本面评分<40：仓位可减少10-20%

🔴 出货信号筛选：
- 出货级别=strong（评分≥5.0）：强烈出货信号，建议卖出60-100%
- 出货级别=weak（3.0≤评分<5.0）：弱出货信号，建议卖出30-60%
- 结合基本面评分调整卖出比例：
  - 基本面评分>60：卖出比例可减少10-20%
  - 基本面评分<40：卖出比例可增加10-20%

【第三步：新闻信息影响评估】
📰 新闻影响评估：
- **新闻情绪分类**：
  - 正面新闻：财报超预期、业绩增长、政策利好、并购重组、大股东增持等
  - 负面新闻：财报不及预期、业绩下滑、政策不利、监管处罚、大股东减持等
  - 中性新闻：人事变动、业务调整、行业动态等
- **新闻影响程度**：
  - 重大影响：财报、政策、并购等，权重100%
  - 中等影响：业绩预告、业务调整等，权重60%
  - 轻微影响：人事变动、行业动态等，权重30%
- **新闻与技术指标协同**：
  - 正面新闻配合建仓评分高和基本面扎实：强烈推荐，新闻加分+15%
  - 负面新闻配合出货评分高或基本面薄弱：强烈规避，新闻扣分-15%
  - 新闻情绪与技术指标背离：谨慎对待，等待确认，新闻权重50%
  - 无相关新闻：按技术指标和基本面分析，新闻权重0%

【第四步：多周期趋势验证】
📈 趋势方向验证：
- 多周期趋势评分>20：趋势向上，建仓信号可靠
- 多周期趋势评分<-20：趋势向下，建仓信号谨慎
- 多周期趋势评分-20到20：震荡趋势，建议观望
- 趋势一致性分析：
  - 3/5/10/20/60日均为上涨：建仓信心高
  - 短期上涨+中长期下跌：反弹而非反转，谨慎建仓
  - 短期下跌+中长期上涨：回调而非反转，加仓机会

📊 相对强度验证：
- 多周期相对强度评分>20：跑赢恒指，优先选择
- 多周期相对强度评分<-20：跑输恒指，谨慎对待
- 相对强度改善信号：
  - 短期相对强度显著跑赢且中长期改善：强势股票
  - 短期相对强度跑输但中长期改善：潜力股票

【第五步：基本面质量评估】
🔍 基本面评分评估：
- 基本面评分>60：优质股票，大幅提升建仓信号可靠性
- 基本面评分40-60：良好股票，提升建仓信号可靠性
- 基本面评分20-40：一般股票，建仓信号需谨慎
- 基本面评分<20：差股票，建仓信号不可靠

💰 估值分析：
- PE<10且PB<1：低估值，安全边际高
- PE 10-15且PB 1-1.5：合理估值
- PE>20或PB>2：高估值，风险较高

【第六步：技术指标协同验证】
🔗 技术指标协同：
- RSI+MACD+布林带+成交量比率+CMF：至少3个指标同向才可靠
- RSI背离信号=1：潜在反转信号，重点关注
- MACD柱状图变化率信号=1：动能增强/减弱信号
- 布林带突破信号=1：价格突破重要支撑/阻力

【第七步：市场环境判断】
🌍 市场整体环境：
- 市场情绪：乐观/中性/悲观
- 建仓信号股票数 vs 出货信号股票数
- 跑赢恒指股票数 vs 跑输恒指股票数
- 平均相对强度：正值表示整体强势，负值表示整体弱势
- 总南向资金净流入：正值表示资金流入，负值表示资金流出

📊 策略调整：
- 市场情绪乐观+资金流入+整体强势：积极建仓，仓位可提高10-20%
- 市场情绪悲观+资金流出+整体弱势：谨慎建仓，仓位可降低10-20%
- 市场震荡+资金平衡+整体中性：正常建仓，保持标准仓位

【第八步：综合评估与操作建议】
⭐ 综合评分计算（满分100分）：
- 建仓评分权重：25%
- 多周期趋势评分权重：20%
- 多周期相对强度评分权重：15%
- 基本面评分权重：15%
- 新闻影响权重：15%
- 技术指标协同权重：10%

📊 推荐标准：
- 综合评分>70分：强烈推荐，建议仓位50-70%
- 综合评分50-70分：推荐，建议仓位30-50%
- 综合评分30-50分：观望，建议仓位10-30%
- 综合评分<30分：不推荐，建议仓位0-10%

🎯 操作建议决策树：
1. **风险控制优先（最高优先级）**：
   - 止损(1)：CLOSE（全部清仓），风险等级5级
   - 止盈(1)：SELL（卖出30%），风险等级3级
   - Trailing Stop(1)：SELL（卖出30%），风险等级3级

2. **净信号评估**：
   - 净信号 = 建仓评分 - 出货评分
   - 净信号 > 3.0：倾向买入
   - 净信号 < -3.0：倾向卖出
   - 净信号 -3.0到3.0：倾向观望

3. **综合评分验证**：
   - **买入条件**：净信号 > 3.0 且 综合评分 > 50
   - **观望条件**：
     - 净信号 > 3.0 且 综合评分 40-50：观望
     - 净信号 -3.0 到3.0 且 综合评分 > 40：观望
     - 净信号 > 0 但综合评分 < 30：观望
     - 净信号 < 0 但综合评分 > 70：观望
   - **卖出条件**：
     - 净信号 < -3.0 且 综合评分 < 30
     - 净信号 < -3.0 且 综合评分 30-40：观望
     - 综合评分 < 30：卖出

4. **目标价格计算**：
   - 买入目标价 = 当前价格 × (1 + 预期收益率)
   - 预期收益率 = 多周期趋势评分/10（如趋势评分40，预期收益率4%）
   - 止损价 = 当前价格 × (1 - ATR比率 × 2) 或 当前价格 × 0.95

5. **持仓时间建议**：
   - 多周期趋势评分>50：长期（>4周）
   - 多周期趋势评分20-50：中期（1-4周）
   - 多周期趋势评分-20-20：短期（<1周）
   - 多周期趋势评分<-20：不建仓

6. **风险等级评估**：
   - 1级（低风险）：综合评分>70，基本面评分>60，多周期趋势评分>20，无风险控制信号，净信号>3.0
   - 2级（中低风险）：综合评分50-70，基本面评分>40，多周期趋势评分>0，无风险控制信号，净信号>0
   - 3级（中风险）：综合评分30-50，基本面评分>20，多周期趋势评分>-20，无风险控制信号，净信号-3.0到3.0
   - 4级（中高风险）：净信号<-3.0或触发止盈/Trailing Stop，或综合评分<30
   - 5级（高风险）：触发止损，或净信号<-5.0且基本面评分<40

📈 输出格式：

【详细分析报告】

1. 🎯 买入建议（综合评分>50分，按评分排序）
   ⚠️ 重要：每只股票的分析必须以"股票代码 股票名称"开头
   
   格式要求：
   股票代码 股票名称
   综合评分：XX分（建仓评分XX分 + 多周期趋势评分XX分 + 多周期相对强度评分XX分 + 基本面评分XX分 + 新闻影响XX分 + 技术指标协同XX分）
   建议仓位：XX%
   目标价格：XX港元（基于技术分析）
   止损价格：XX港元（基于ATR或支撑位）
   持仓时间：短期(<1周)/中期(1-4周)/长期(>4周)
   风险等级：1级(低)/2级(中低)/3级(中)/4级(中高)/5级(高)
   买入理由：详细说明各维度得分和协同性
   风险因素：详细说明潜在风险

2. ⚠️ 卖出建议（出货评分≥3.0或触发风险控制信号）
   ⚠️ 重要：每只股票的分析必须以"股票代码 股票名称"开头
   
   格式要求：
   股票代码 股票名称
   卖出原因：出货评分XX分/止损触发/止盈触发/Trailing Stop触发
   建议卖出比例：XX%
   目标价格：XX港元（如适用）
   风险等级：X级
   风险因素：详细说明

3. 📊 持有建议（综合评分30-50分）
   ⚠️ 重要：每只股票的分析必须以"股票代码 股票名称"开头
   
   格式要求：
   股票代码 股票名称
   持有理由：详细说明
   建议仓位：XX%
   风险等级：X级

4. 🌍 市场环境与策略
   - 市场环境评估：情绪/资金流向/整体强度
   - 整体策略：积极/中性/谨慎
   - 建议整体仓位：XX%
   - 重点关注的信号：建仓信号X只，出货信号Y只，止损Z只

5. 📊 统计摘要
   - 买入建议：X只
   - 卖出建议：Y只
   - 持有建议：Z只
   - 平均综合评分：XX分
   - 平均风险等级：X级

报告生成时间：{current_time}，数据截至：{run_date if run_date else '最新数据'}

⚠️ 重要提醒：
1. 风险控制信号（止损/止盈/Trailing Stop）必须优先处理，不可忽视
2. **出货信号必须全部列出**：所有出货评分≥3.0的股票（weak和strong级别）必须在风险警示中明确列出，并给出具体的卖出建议
3. 综合评分是推荐的核心依据，必须详细说明各维度得分
4. 市场环境必须考虑，根据市场情绪和资金流向调整策略
4. 仓位管理必须严格执行，根据综合评分和市场环境动态调整
5. 技术指标必须协同验证，单一指标不可作为决策依据

请用中文回答，确保技术分析、基本面分析与新闻信息并重，严格按照优先级和决策树进行分析。
"""
    
    return prompt


# ==============================
# 4. 上个交易日对比辅助函数
# ==============================

def get_trend_change_arrow(current_trend, previous_trend):
    """
    返回趋势变化箭头符号
    
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

def get_score_change_arrow(current_score, previous_score):
    """
    返回评分变化箭头符号
    
    参数:
    - current_score: 当前评分
    - previous_score: 上个交易日评分
    
    返回:
    - str: 箭头符号和颜色样式
    """
    if previous_score is None or current_score is None:
        return '<span style="color: #999;">→</span>'
    
    # 尝试转换为数值类型进行比较
    try:
        current_val = float(current_score) if current_score != 'N/A' else None
        previous_val = float(previous_score) if previous_score != 'N/A' else None
        
        if current_val is None or previous_val is None:
            return '<span style="color: #999;">→</span>'
        
        if current_val > previous_val:
            return '<span style="color: green; font-weight: bold;">↑</span>'
        elif current_val < previous_val:
            return '<span style="color: red; font-weight: bold;">↓</span>'
        else:
            return '<span style="color: #999;">→</span>'
    except (ValueError, TypeError):
        return '<span style="color: #999;">→</span>'

def get_price_change_arrow(current_price_str, previous_price):
    """
    返回价格变化箭头符号
    
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

# ==============================
# 4.5. 多周期指标计算函数
# ==============================

def calculate_multi_period_metrics(hist_df, periods=[3, 5, 10, 20, 60]):
    """
    计算多周期价格变化率和趋势方向
    
    参数:
    - hist_df: 历史价格数据（DataFrame，包含Close列）
    - periods: 周期列表，默认为[3, 5, 10, 20, 60]
    
    返回:
    - dict: 包含各周期的价格变化率和趋势方向
    """
    metrics = {}
    
    for period in periods:
        if len(hist_df) < period:
            metrics[f'{period}d_return'] = None
            metrics[f'{period}d_trend'] = None
            continue
        
        # 计算价格变化率
        current_price = hist_df['Close'].iloc[-1]
        past_price = hist_df['Close'].iloc[-period]
        return_pct = ((current_price - past_price) / past_price) * 100
        
        metrics[f'{period}d_return'] = safe_round(return_pct, 2)
        
        # 判断趋势方向
        if return_pct > 2:
            metrics[f'{period}d_trend'] = '强势上涨'
        elif return_pct > 0:
            metrics[f'{period}d_trend'] = '上涨'
        elif return_pct > -2:
            metrics[f'{period}d_trend'] = '下跌'
        else:
            metrics[f'{period}d_trend'] = '强势下跌'
    
    return metrics

def calculate_relative_strength_multi_period(stock_hist, hsi_hist, periods=[3, 5, 10, 20, 60]):
    """
    计算多周期相对强度（股票 vs 恒生指数）
    
    参数:
    - stock_hist: 股票历史价格数据（DataFrame，包含Close列）
    - hsi_hist: 恒生指数历史价格数据（DataFrame，包含Close列）
    - periods: 周期列表，默认为[3, 5, 10, 20, 60]
    
    返回:
    - dict: 包含各周期的相对强度
    """
    rs_metrics = {}
    
    for period in periods:
        if len(stock_hist) < period or len(hsi_hist) < period:
            rs_metrics[f'{period}d_rs'] = None
            rs_metrics[f'{period}d_rs_signal'] = None
            continue
        
        # 计算股票收益
        stock_current = stock_hist['Close'].iloc[-1]
        stock_past = stock_hist['Close'].iloc[-period]
        stock_return = (stock_current - stock_past) / stock_past
        
        # 计算恒生指数收益
        hsi_current = hsi_hist['Close'].iloc[-1]
        hsi_past = hsi_hist['Close'].iloc[-period]
        hsi_return = (hsi_current - hsi_past) / hsi_past
        
        # 计算相对强度（股票收益 - 恒生指数收益）
        rs = stock_return - hsi_return
        rs_pct = rs * 100
        
        rs_metrics[f'{period}d_rs'] = safe_round(rs_pct, 2)
        
        # 判断相对强度信号
        if rs_pct > 5:
            rs_metrics[f'{period}d_rs_signal'] = '显著跑赢'
        elif rs_pct > 2:
            rs_metrics[f'{period}d_rs_signal'] = '跑赢'
        elif rs_pct > -2:
            rs_metrics[f'{period}d_rs_signal'] = '持平'
        elif rs_pct > -5:
            rs_metrics[f'{period}d_rs_signal'] = '跑输'
        else:
            rs_metrics[f'{period}d_rs_signal'] = '显著跑输'
    
    return rs_metrics

def get_multi_period_trend_score(metrics, periods=[3, 5, 10, 20, 60]):
    """
    计算多周期趋势综合评分
    
    参数:
    - metrics: 多周期指标字典
    - periods: 周期列表
    
    返回:
    - float: 综合趋势评分（-100到100）
    """
    if not metrics:
        return None
    
    score = 0
    weights = {3: 0.1, 5: 0.15, 10: 0.2, 20: 0.25, 60: 0.3}
    
    for period in periods:
        return_key = f'{period}d_return'
        if return_key in metrics and metrics[return_key] is not None:
            # 标准化收益：假设±10%为极限
            normalized_return = metrics[return_key] / 10.0 * 100
            score += normalized_return * weights.get(period, 0.2)
    
    return safe_round(score, 1)

def get_multi_period_rs_score(rs_metrics, periods=[3, 5, 10, 20, 60]):
    """
    计算多周期相对强度综合评分
    
    参数:
    - rs_metrics: 多周期相对强度字典
    - periods: 周期列表
    
    返回:
    - float: 综合相对强度评分（-100到100）
    """
    if not rs_metrics:
        return None
    
    score = 0
    weights = {3: 0.1, 5: 0.15, 10: 0.2, 20: 0.25, 60: 0.3}
    
    for period in periods:
        rs_key = f'{period}d_rs'
        if rs_key in rs_metrics and rs_metrics[rs_key] is not None:
            # 标准化相对强度：假设±10%为极限
            normalized_rs = rs_metrics[rs_key] / 10.0 * 100
            score += normalized_rs * weights.get(period, 0.2)
    
    return safe_round(score, 1)

# ==============================
# 5. 单股分析函数
# ==============================

def analyze_stock(code, name, run_date=None):
    try:
        print(f"\n🔍 分析 {name} ({code}) ...")
        # 移除代码中的.HK后缀，腾讯财经接口不需要
        stock_code = code.replace('.HK', '')
        
        # 获取基本面数据
        print(f"  📊 获取 {name} 基本面数据...")
        fundamental_data = get_comprehensive_fundamental_data(stock_code)
        if fundamental_data is None:
            print(f"  ⚠️ 无法获取 {name} 基本面数据，将仅使用技术面数据")
        else:
            print(f"  ✅ {name} 基本面数据获取成功")
        
        # 如果指定了运行日期，则获取该日期的历史数据
        if run_date:
            # 获取指定日期前 PRICE_WINDOW+30 天的数据
            target_date = pd.to_datetime(run_date)
            # 计算需要获取的天数
            days_diff = (datetime.now() - target_date).days + PRICE_WINDOW + 30
            full_hist = get_hk_stock_data_tencent(stock_code, period_days=days_diff)
        else:
            # 默认行为：获取最近 PRICE_WINDOW+30 天的数据
            full_hist = get_hk_stock_data_tencent(stock_code, period_days=PRICE_WINDOW + 30)
            
        if len(full_hist) < PRICE_WINDOW:
            print(f"⚠️  {name} 数据不足（需要至少 {PRICE_WINDOW} 日）")
            return None

        # 如果指定了运行日期，使用包含指定日期的数据窗口
        if run_date:
            # 筛选出指定日期及之前的数据
            target_date = pd.to_datetime(run_date)
            # 确保时区一致
            if full_hist.index.tz is not None:
                target_date = target_date.tz_localize(full_hist.index.tz)
            
            # 筛选指定日期及之前的数据
            filtered_hist = full_hist[full_hist.index <= target_date]
            
            # 如果没有数据，使用最接近的日期数据
            if len(filtered_hist) == 0:
                # 找到最接近指定日期的数据（包括之后的日期）
                filtered_hist = full_hist[full_hist.index >= target_date]
            
            main_hist = filtered_hist[['Open', 'Close', 'Volume']].tail(DAYS_ANALYSIS).copy()
            
            # 获取上个交易日的日期（排除周末）
            previous_trading_date = target_date.date() - timedelta(days=1)
            while previous_trading_date.weekday() >= 5:  # 5=周六, 6=周日
                previous_trading_date -= timedelta(days=1)
        else:
            main_hist = full_hist[['Open', 'Close', 'Volume']].tail(DAYS_ANALYSIS).copy()
            
            # 获取上个交易日的日期（排除周末）
            previous_trading_date = (datetime.now() - timedelta(days=1)).date()
            while previous_trading_date.weekday() >= 5:  # 5=周六, 6=周日
                previous_trading_date -= timedelta(days=1)
            
        if len(main_hist) < 5:
            print(f"⚠️  {name} 主分析窗口数据不足")
            return None

        # 获取上个交易日的指标数据（移到技术指标计算之后）

        # ====== 排除周六日（只保留交易日）======
        main_hist = main_hist[main_hist.index.weekday < 5]
        full_hist = full_hist[full_hist.index.weekday < 5]

        # 基础指标（在 full_hist 上计算）
        # 数据质量检查
        if full_hist.empty:
            print(f"⚠️  {name} 数据为空")
            return None
            
        # 检查是否有足够的数据点
        if len(full_hist) < 5:
            print(f"⚠️  {name} 数据不足")
            return None
            
        # 检查数据是否包含必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # 从腾讯财经获取的数据可能不包含High和Low，需要处理
        for col in required_columns:
            if col not in full_hist.columns:
                print(f"⚠️  {name} 缺少必要的列 {col}")
                # 如果缺少High或Low，使用Close作为近似值
                if col in ['High', 'Low']:
                    full_hist[col] = full_hist['Close']
                else:
                    return None
                
        # 检查数据是否包含有效的数值
        if full_hist['Close'].isna().all() or full_hist['Volume'].isna().all():
            print(f"⚠️  {name} 数据包含大量缺失值")
            return None
            
        # 移除包含异常值的行
        full_hist = full_hist.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        full_hist = full_hist[(full_hist['Close'] > 0) & (full_hist['Volume'] >= 0)]
        
        if len(full_hist) < 5:
            print(f"⚠️  {name} 清理异常值后数据不足")
            return None
            
        full_hist['Vol_MA20'] = full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).mean()
        full_hist['MA5'] = full_hist['Close'].rolling(5, min_periods=1).mean()
        full_hist['MA10'] = full_hist['Close'].rolling(10, min_periods=1).mean()
        full_hist['MA20'] = full_hist['Close'].rolling(20, min_periods=1).mean()

        # MACD
        full_hist['EMA12'] = full_hist['Close'].ewm(span=12, adjust=False).mean()
        full_hist['EMA26'] = full_hist['Close'].ewm(span=26, adjust=False).mean()
        full_hist['MACD'] = full_hist['EMA12'] - full_hist['EMA26']
        full_hist['MACD_Signal'] = full_hist['MACD'].ewm(span=9, adjust=False).mean()

        # RSI (Wilder)
        delta_full = full_hist['Close'].diff()
        gain = delta_full.clip(lower=0)
        loss = -delta_full.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        full_hist['RSI'] = 100 - (100 / (1 + rs))

        # Returns & Volatility (年化)
        full_hist['Returns'] = full_hist['Close'].pct_change()
        # 使用 min_periods=10 保证样本充足再年化
        full_hist['Volatility'] = full_hist['Returns'].rolling(20, min_periods=10).std() * math.sqrt(252)

        # VWAP (使用 (High+Low+Close)/3 * Volume 的加权近似)
        full_hist['TP'] = (full_hist['High'] + full_hist['Low'] + full_hist['Close']) / 3
        full_hist['VWAP'] = (full_hist['TP'] * full_hist['Volume']).rolling(VOL_WINDOW, min_periods=1).sum() / full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).sum()
        
        # ATR (Average True Range)
        full_hist['TR'] = np.maximum(
            np.maximum(
                full_hist['High'] - full_hist['Low'],
                np.abs(full_hist['High'] - full_hist['Close'].shift(1))
            ),
            np.abs(full_hist['Low'] - full_hist['Close'].shift(1))
        )
        full_hist['ATR'] = full_hist['TR'].rolling(14, min_periods=1).mean()
        
        # Chaikin Money Flow (CMF)
        full_hist['MF_Multiplier'] = ((full_hist['Close'] - full_hist['Low']) - (full_hist['High'] - full_hist['Close'])) / (full_hist['High'] - full_hist['Low'])
        full_hist['MF_Volume'] = full_hist['MF_Multiplier'] * full_hist['Volume']
        full_hist['CMF'] = full_hist['MF_Volume'].rolling(20, min_periods=1).sum() / full_hist['Volume'].rolling(20, min_periods=1).sum()
        
        # ADX (Average Directional Index)
        # +DI and -DI
        up_move = full_hist['High'].diff()
        down_move = -full_hist['Low'].diff()
        
        # +DM and -DM
        full_hist['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        full_hist['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed +DM, -DM and TR
        full_hist['+DI'] = 100 * (full_hist['+DM'].ewm(alpha=1/14, adjust=False).mean() / full_hist['ATR'])
        full_hist['-DI'] = 100 * (full_hist['-DM'].ewm(alpha=1/14, adjust=False).mean() / full_hist['ATR'])
        
        # ADX
        dx = 100 * (np.abs(full_hist['+DI'] - full_hist['-DI']) / (full_hist['+DI'] + full_hist['-DI']))
        full_hist['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # Bollinger Bands
        full_hist['BB_Mid'] = full_hist['Close'].rolling(20, min_periods=1).mean()
        full_hist['BB_Upper'] = full_hist['BB_Mid'] + 2 * full_hist['Close'].rolling(20, min_periods=1).std()
        full_hist['BB_Lower'] = full_hist['BB_Mid'] - 2 * full_hist['Close'].rolling(20, min_periods=1).std()
        full_hist['BB_Width'] = (full_hist['BB_Upper'] - full_hist['BB_Lower']) / full_hist['BB_Mid']
        
        # Bollinger Band Breakout
        full_hist['BB_Breakout'] = (full_hist['Close'] - full_hist['BB_Lower']) / (full_hist['BB_Upper'] - full_hist['BB_Lower'])
        
        # 成交量 z-score
        full_hist['Vol_Mean_20'] = full_hist['Volume'].rolling(20, min_periods=1).mean()
        full_hist['Vol_Std_20'] = full_hist['Volume'].rolling(20, min_periods=1).std()
        full_hist['Vol_Z_Score'] = (full_hist['Volume'] - full_hist['Vol_Mean_20']) / full_hist['Vol_Std_20']
        
        # 成交额 z-score
        full_hist['Turnover'] = full_hist['Close'] * full_hist['Volume']
        full_hist['Turnover_Mean_20'] = full_hist['Turnover'].rolling(20, min_periods=1).mean()
        full_hist['Turnover_Std_20'] = full_hist['Turnover'].rolling(20, min_periods=1).std()
        full_hist['Turnover_Z_Score'] = (full_hist['Turnover'] - full_hist['Turnover_Mean_20']) / full_hist['Turnover_Std_20']
        
        # VWAP (Volume Weighted Average Price)
        full_hist['VWAP'] = (full_hist['TP'] * full_hist['Volume']).rolling(VOL_WINDOW, min_periods=1).sum() / full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).sum()
        
        # MACD Histogram and its rate of change
        full_hist['MACD_Hist'] = full_hist['MACD'] - full_hist['MACD_Signal']
        full_hist['MACD_Hist_ROC'] = full_hist['MACD_Hist'].pct_change()
        
        # RSI Divergence (Comparing RSI with price movements)
        full_hist['RSI_ROC'] = full_hist['RSI'].pct_change()
        
        # CMF Trend
        full_hist['CMF_Signal'] = full_hist['CMF'].rolling(5, min_periods=1).mean()
        
        # Dynamic ATR Threshold
        full_hist['ATR_MA'] = full_hist['ATR'].rolling(10, min_periods=1).mean()
        full_hist['ATR_Ratio'] = full_hist['ATR'] / full_hist['ATR_MA']
        
        # Stochastic Oscillator
        K_Period = 14
        D_Period = 3
        full_hist['Low_Min'] = full_hist['Low'].rolling(window=K_Period, min_periods=1).min()
        full_hist['High_Max'] = full_hist['High'].rolling(window=K_Period, min_periods=1).max()
        full_hist['Stoch_K'] = 100 * (full_hist['Close'] - full_hist['Low_Min']) / (full_hist['High_Max'] - full_hist['Low_Min'])
        full_hist['Stoch_D'] = full_hist['Stoch_K'].rolling(window=D_Period, min_periods=1).mean()
        
        # Williams %R
        full_hist['Williams_R'] = (full_hist['High_Max'] - full_hist['Close']) / (full_hist['High_Max'] - full_hist['Low_Min']) * -100
        
        # Price Rate of Change
        full_hist['ROC'] = full_hist['Close'].pct_change(periods=12)
        
        # Average Volume
        full_hist['Avg_Vol_30'] = full_hist['Volume'].rolling(30, min_periods=1).mean()
        full_hist['Volume_Ratio'] = full_hist['Volume'] / full_hist['Avg_Vol_30']

        # price percentile 基于 PRICE_WINDOW
        low60 = full_hist['Close'].tail(PRICE_WINDOW).min()
        high60 = full_hist['Close'].tail(PRICE_WINDOW).max()

        # 把 full_hist 上的指标 reindex 到 main_hist
        main_hist['Vol_MA20'] = full_hist['Vol_MA20'].reindex(main_hist.index, method='ffill')
        main_hist['Vol_Ratio'] = main_hist['Volume'] / main_hist['Vol_MA20']
        if high60 == low60:
            main_hist['Price_Percentile'] = 50.0
        else:
            main_hist['Price_Percentile'] = ((main_hist['Close'] - low60) / (high60 - low60) * 100).clip(0, 100)

        main_hist['MA5'] = full_hist['MA5'].reindex(main_hist.index, method='ffill')
        main_hist['MA10'] = full_hist['MA10'].reindex(main_hist.index, method='ffill')
        main_hist['MA20'] = full_hist['MA20'].reindex(main_hist.index, method='ffill')
        main_hist['MACD'] = full_hist['MACD'].reindex(main_hist.index, method='ffill')
        main_hist['MACD_Signal'] = full_hist['MACD_Signal'].reindex(main_hist.index, method='ffill')
        main_hist['RSI'] = full_hist['RSI'].reindex(main_hist.index, method='ffill')
        main_hist['Volatility'] = full_hist['Volatility'].reindex(main_hist.index, method='ffill')
        main_hist['VWAP'] = full_hist['VWAP'].reindex(main_hist.index, method='ffill')
        main_hist['ATR'] = full_hist['ATR'].reindex(main_hist.index, method='ffill')
        main_hist['CMF'] = full_hist['CMF'].reindex(main_hist.index, method='ffill')
        main_hist['ADX'] = full_hist['ADX'].reindex(main_hist.index, method='ffill')
        main_hist['BB_Upper'] = full_hist['BB_Upper'].reindex(main_hist.index, method='ffill')
        main_hist['BB_Lower'] = full_hist['BB_Lower'].reindex(main_hist.index, method='ffill')
        main_hist['BB_Width'] = full_hist['BB_Width'].reindex(main_hist.index, method='ffill')
        main_hist['Vol_Z_Score'] = full_hist['Vol_Z_Score'].reindex(main_hist.index, method='ffill')
        main_hist['Turnover_Z_Score'] = full_hist['Turnover_Z_Score'].reindex(main_hist.index, method='ffill')

        # OBV 从 full_hist 累计后 reindex
        full_hist['OBV'] = 0.0
        for i in range(1, len(full_hist)):
            if full_hist['Close'].iat[i] > full_hist['Close'].iat[i-1]:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1] + full_hist['Volume'].iat[i]
            elif full_hist['Close'].iat[i] < full_hist['Close'].iat[i-1]:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1] - full_hist['Volume'].iat[i]
            else:
                full_hist['OBV'].iat[i] = full_hist['OBV'].iat[i-1]
        main_hist['OBV'] = full_hist['OBV'].reindex(main_hist.index, method='ffill').fillna(0.0)
        
        # 将新指标 reindex 到 main_hist
        main_hist['BB_Breakout'] = full_hist['BB_Breakout'].reindex(main_hist.index, method='ffill')
        main_hist['MACD_Hist'] = full_hist['MACD_Hist'].reindex(main_hist.index, method='ffill')
        main_hist['MACD_Hist_ROC'] = full_hist['MACD_Hist_ROC'].reindex(main_hist.index, method='ffill')
        main_hist['RSI_ROC'] = full_hist['RSI_ROC'].reindex(main_hist.index, method='ffill')
        main_hist['CMF_Signal'] = full_hist['CMF_Signal'].reindex(main_hist.index, method='ffill')
        main_hist['ATR_Ratio'] = full_hist['ATR_Ratio'].reindex(main_hist.index, method='ffill')
        main_hist['Stoch_K'] = full_hist['Stoch_K'].reindex(main_hist.index, method='ffill')
        main_hist['Stoch_D'] = full_hist['Stoch_D'].reindex(main_hist.index, method='ffill')
        main_hist['Williams_R'] = full_hist['Williams_R'].reindex(main_hist.index, method='ffill')
        main_hist['ROC'] = full_hist['ROC'].reindex(main_hist.index, method='ffill')
        main_hist['Volume_Ratio'] = full_hist['Volume_Ratio'].reindex(main_hist.index, method='ffill')

        # 获取上个交易日的指标数据（在所有技术指标计算完成后）
        previous_day_indicators = None
        try:
            # 计算上个交易日日期
            yesterday = datetime.now().date() - timedelta(days=1)
            while yesterday.weekday() >= 5:  # 5=周六, 6=周日
                yesterday -= timedelta(days=1)
            
            # 如果指定了运行日期，使用运行日期的前一天
            if run_date:
                target_date = pd.to_datetime(run_date).date()
                previous_trading_date = target_date - timedelta(days=1)
                while previous_trading_date.weekday() >= 5:
                    previous_trading_date -= timedelta(days=1)
            else:
                previous_trading_date = yesterday
            
            # 筛选出上个交易日及之前的数据
            previous_trading_date_timestamp = pd.Timestamp(previous_trading_date)
            if full_hist.index.tz is not None:
                previous_trading_date_timestamp = previous_trading_date_timestamp.tz_localize(full_hist.index.tz)
            
            prev_filtered_hist = full_hist[full_hist.index <= previous_trading_date_timestamp]
            
            if not prev_filtered_hist.empty:
                # 获取上个交易日的数据（最后一天）
                prev_day_data = prev_filtered_hist.iloc[-1]
                
                # 计算上个交易日的关键指标
                prev_rsi = prev_day_data.get('RSI') if pd.notna(prev_day_data.get('RSI')) else None
                prev_macd = prev_day_data.get('MACD') if pd.notna(prev_day_data.get('MACD')) else None
                prev_price = prev_day_data.get('Close') if pd.notna(prev_day_data.get('Close')) else None
                
                # 获取上个交易日的建仓和出货评分（如果可用）
                prev_buildup_score = None
                prev_distribution_score = None
                prev_tav_score = None
                
                # 尝试从历史数据中获取评分
                if 'Buildup_Score' in prev_filtered_hist.columns:
                    prev_buildup_score = prev_filtered_hist['Buildup_Score'].iloc[-1] if pd.notna(prev_filtered_hist['Buildup_Score'].iloc[-1]) else None
                if 'Distribution_Score' in prev_filtered_hist.columns:
                    prev_distribution_score = prev_filtered_hist['Distribution_Score'].iloc[-1] if pd.notna(prev_filtered_hist['Distribution_Score'].iloc[-1]) else None
                
                previous_day_indicators = {
                    'rsi': safe_round(prev_rsi, 2) if prev_rsi is not None else None,
                    'macd': safe_round(prev_macd, 4) if prev_macd is not None else None,
                    'price': safe_round(prev_price, 2) if prev_price is not None else None,
                    'buildup_score': safe_round(prev_buildup_score, 2) if prev_buildup_score is not None else None,
                    'distribution_score': safe_round(prev_distribution_score, 2) if prev_distribution_score is not None else None,
                    'tav_score': safe_round(prev_tav_score, 1) if prev_tav_score is not None else None,
                }
        except Exception as e:
            print(f"  ⚠️ 获取 {name} 上个交易日指标失败: {e}")

        # 南向资金：按日期获取并缓存，转换为"万"
        main_hist['Southbound_Net'] = 0.0
        for ts in main_hist.index:
            # ===== 排除周六日 =====
            if ts.weekday() >= 5:
                continue
            date_str = ts.strftime('%Y%m%d')
            df_ggt = fetch_ggt_components(code, date_str)
            if df_ggt is None:
                continue
            # 获取南向资金净买入数据（使用持股市值变化-1日作为近似值）
            try:
                # 取第一个匹配的记录
                if '持股市值变化-1日' in df_ggt.columns:
                    net_val = df_ggt['持股市值变化-1日'].iloc[0]
                    if pd.notna(net_val):
                        # 转换为万元（原始数据单位可能是元）
                        main_hist.at[ts, 'Southbound_Net'] = float(net_val) / SOUTHBOUND_UNIT_CONVERSION
                else:
                    print(f"⚠️ 南向资金数据缺少持股市值变化字段 {code} {date_str}")
            except Exception as e:
                # 忽略解析错误
                print(f"⚠️ 解析南向资金数据失败 {code} {date_str}: {e}")
                pass

        # 计算区间收益（main_hist 首尾）
        start_date, end_date = main_hist.index[0], main_hist.index[-1]
        stock_ret = (main_hist['Close'].iloc[-1] - main_hist['Close'].iloc[0]) / main_hist['Close'].iloc[0]
        hsi_ret = get_hsi_return(start_date, end_date)
        if pd.isna(hsi_ret):
            hsi_ret = 0.0  # 若无法获取恒指收益，降级为0（可调整）
        rs_diff = stock_ret - hsi_ret
        if (1.0 + hsi_ret) == 0:
            rs_ratio = float('inf') if (1.0 + stock_ret) > 0 else float('-inf')
        else:
            rs_ratio = (1.0 + stock_ret) / (1.0 + hsi_ret) - 1.0

        # outperforms 多种判定
        outperforms_by_ret = (stock_ret > 0) and (stock_ret > hsi_ret)
        outperforms_by_diff = stock_ret > hsi_ret
        outperforms_by_rs = rs_ratio > 0

        # 如果无法获取恒生指数数据，将 outperforms 设置为 False
        if hsi_hist is None or hsi_hist.empty:
            outperforms = False
        else:
            if OUTPERFORMS_USE_RS:
                outperforms = bool(outperforms_by_rs)
            else:
                if OUTPERFORMS_REQUIRE_POSITIVE:
                    outperforms = bool(outperforms_by_ret)
                else:
                    outperforms = bool(outperforms_by_diff)

        # === 基本面质量评估函数 ===
        def evaluate_fundamental_quality():
            """评估基本面质量（简化版：只基于PE和PB），返回评分和关键指标"""
            if not fundamental_data:
                return 0, {}  # 无基本面数据，评分为0

            score = 0
            details = {}

            # 估值指标评分（100分，满分）
            pe = fundamental_data.get('fi_pe_ratio')
            pb = fundamental_data.get('fi_pb_ratio')

            # PE评分（50分）
            if pe is not None:
                if pe < 10:
                    score += 50
                    details['pe_score'] = "低估值 (PE<10)"
                elif pe < 15:
                    score += 40
                    details['pe_score'] = "合理估值 (10<PE<15)"
                elif pe < 20:
                    score += 30
                    details['pe_score'] = "偏高估值 (15<PE<20)"
                elif pe < 25:
                    score += 20
                    details['pe_score'] = "高估值 (20<PE<25)"
                else:
                    score += 10
                    details['pe_score'] = "极高估值 (PE>25)"
            else:
                score += 25  # 无PE数据，给中等分
                details['pe_score'] = "无PE数据"

            # PB评分（50分）
            if pb is not None:
                if pb < 1:
                    score += 50
                    details['pb_score'] = "低市净率 (PB<1)"
                elif pb < 1.5:
                    score += 40
                    details['pb_score'] = "合理市净率 (1<PB<1.5)"
                elif pb < 2:
                    score += 30
                    details['pb_score'] = "偏高市净率 (1.5<PB<2)"
                elif pb < 3:
                    score += 20
                    details['pb_score'] = "高市净率 (2<PB<3)"
                else:
                    score += 10
                    details['pb_score'] = "极高市净率 (PB>3)"
            else:
                score += 25  # 无PB数据，给中等分
                details['pb_score'] = "无PB数据"

            return score, details
        
        # 评估基本面质量
        fundamental_score, fundamental_details = evaluate_fundamental_quality()
        
        # === 建仓信号 ===
        def is_buildup(row):
            # 基本条件
            price_cond = row['Price_Percentile'] < PRICE_LOW_PCT
            vol_cond = pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_BUILDUP
            sb_cond = pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] > SOUTHBOUND_THRESHOLD
            
            # 基本面条件 - 根据基本面评分调整
            # 如果基本面评分高（>60分），可以放宽其他条件
            # 如果基本面评分低（<30分），需要更严格的技术面条件
            fundamental_cond = True  # 默认通过
            if fundamental_data:
                if fundamental_score > 60:
                    # 基本面优秀，降低价格位置要求
                    price_cond = row['Price_Percentile'] < (PRICE_LOW_PCT + 20)
                elif fundamental_score < 30:
                    # 基本面较差，需要更严格的技术面条件
                    fundamental_cond = False
            
            # 增加的辅助条件（调整后的阈值和新增条件）
            # MACD线上穿信号线
            macd_cond = pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] > row['MACD_Signal']
            # RSI超卖（调整阈值从30到35）
            rsi_cond = pd.notna(row.get('RSI')) and row['RSI'] < 35
            # OBV上升
            obv_cond = pd.notna(row.get('OBV')) and row['OBV'] > 0
            # 价格相对于5日均线位置（价格低于5日均线）
            ma5_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA5')) and row['Close'] < row['MA5']
            # 价格相对于10日均线位置（价格低于10日均线）
            ma10_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA10')) and row['Close'] < row['MA10']
            # 收盘价高于VWAP且放量 (VWAP条件)
            vwap_cond = pd.notna(row.get('Close')) and pd.notna(row.get('VWAP')) and pd.notna(row.get('Vol_Ratio')) and row['Close'] > row['VWAP'] and row['Vol_Ratio'] > 1.5
            # ATR放大 (ATR条件)
            atr_cond = pd.notna(row.get('ATR')) and pd.notna(row.get('Close')) and row['ATR'] > full_hist['ATR'].rolling(14).mean().reindex([row.name], method='ffill').iloc[0] * 1.5
            # CMF > 0.05 (资金流入)
            cmf_cond = pd.notna(row.get('CMF')) and row['CMF'] > 0.05
            # ADX > 25 (趋势明确)
            adx_cond = pd.notna(row.get('ADX')) and row['ADX'] > 25
            # 成交量z-score > 1.5 (异常放量)
            vol_z_cond = pd.notna(row.get('Vol_Z_Score')) and row['Vol_Z_Score'] > 1.5
            # 成交额z-score > 1.5 (异常成交额)
            turnover_z_cond = pd.notna(row.get('Turnover_Z_Score')) and row['Turnover_Z_Score'] > 1.5
            
            # 计算满足的辅助条件数量
            aux_conditions = [macd_cond, rsi_cond, obv_cond, ma5_cond, ma10_cond, vwap_cond, atr_cond, cmf_cond, adx_cond, vol_z_cond, turnover_z_cond]
            satisfied_aux_count = sum(aux_conditions)
            
            # 如果满足至少2个辅助条件，或者满足多个条件中的部分条件（更宽松的策略）
            aux_cond = satisfied_aux_count >= 2
            
            return price_cond and vol_cond and sb_cond and aux_cond and fundamental_cond

        main_hist['Buildup_Signal'] = main_hist.apply(is_buildup, axis=1)
        main_hist['Buildup_Confirmed'] = mark_runs(main_hist['Buildup_Signal'], BUILDUP_MIN_DAYS)

        # === 加权评分的建仓信号（新增）===
        def is_buildup_scored(row, fundamental_score=None):
            """
            基于加权评分的建仓信号检测
            
            返回: (score, signal, reasons)
            - score: 建仓评分（0-10+）
            - signal: 信号级别 ('none', 'partial', 'strong')
            - reasons: 触发条件的列表
            """
            score = 0.0
            reasons = []

            # 价格位置：低位加分
            if pd.notna(row.get('Price_Percentile')) and row['Price_Percentile'] < PRICE_LOW_PCT:
                score += BUILDUP_WEIGHTS['price_low']
                reasons.append('price_low')

            # 成交量倍数
            if pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_BUILDUP:
                score += BUILDUP_WEIGHTS['vol_ratio']
                reasons.append('vol_ratio')

            # 成交量 z-score
            if pd.notna(row.get('Vol_Z_Score')) and row['Vol_Z_Score'] > 1.2:
                score += BUILDUP_WEIGHTS['vol_z']
                reasons.append('vol_z')

            # MACD 线上穿
            if pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] > row['MACD_Signal']:
                score += BUILDUP_WEIGHTS['macd_cross']
                reasons.append('macd_cross')

            # RSI 超卖 -> 加分（但不必为30以下才算）
            if pd.notna(row.get('RSI')) and row['RSI'] < 40:
                score += BUILDUP_WEIGHTS['rsi_oversold']
                reasons.append('rsi_oversold')

            # OBV 上升
            if pd.notna(row.get('OBV')) and row['OBV'] > 0:
                score += BUILDUP_WEIGHTS['obv_up']
                reasons.append('obv_up')

            # 收盘高于 VWAP 且放量（表明资金开始买入）
            if pd.notna(row.get('VWAP')) and pd.notna(row.get('Vol_Ratio')) and row['Close'] > row['VWAP'] and row['Vol_Ratio'] > 1.2:
                score += BUILDUP_WEIGHTS['vwap_vol']
                reasons.append('vwap_vol')

            # CMF > 0 表示资金流入
            if pd.notna(row.get('CMF')) and row['CMF'] > 0.03:
                score += BUILDUP_WEIGHTS['cmf_in']
                reasons.append('cmf_in')

            # 南向资金流入作为加分项（不是必须）
            if pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] > SOUTHBOUND_THRESHOLD_IN:
                score += BUILDUP_WEIGHTS['southbound_in']
                reasons.append('southbound_in')

            # 基本面调整（示例：基本面越差，更容易做短线建仓；基本面好时偏长期持有）
            if fundamental_score is not None:
                if fundamental_score > 60:
                    # 对于基本面优秀的股票，减少被噪声触发的概率（需要更高 score）
                    score -= 0.5
                elif fundamental_score < 30:
                    # 基本面差时，允许更容易形成短线建仓（加一点分）
                    score += 0.5

            # 强信号快捷通道：如果同时满足若干关键强条件（例如低位+放量+南向流入），允许单日确认
            strong_fastpath = (
                (pd.notna(row.get('Price_Percentile')) and row['Price_Percentile'] < (PRICE_LOW_PCT - 10)) and
                (pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > 1.8) and
                (pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] > (SOUTHBOUND_THRESHOLD_IN * 1.5))
            )
            if strong_fastpath:
                score += 2.0
                reasons.append('fastpath')

            # 返回分数与分层建议
            signal = None
            if score >= BUILDUP_THRESHOLD_STRONG:
                signal = 'strong'    # 强烈建仓（建议较高比例或确认）
            elif score >= BUILDUP_THRESHOLD_PARTIAL:
                signal = 'partial'   # 部分建仓 / 分批入场
            else:
                signal = 'none'      # 无信号

            return score, signal, reasons

        # === 出货信号 ===
        main_hist['Prev_Close'] = main_hist['Close'].shift(1)
        def is_distribution(row):
            # 基本条件
            price_cond = row['Price_Percentile'] > PRICE_HIGH_PCT
            vol_cond = (pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_DISTRIBUTION)
            sb_cond = (pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] < -SOUTHBOUND_THRESHOLD)
            price_down_cond = (pd.notna(row.get('Prev_Close')) and (row['Close'] < row['Prev_Close'])) or (row['Close'] < row['Open'])
            
            # 基本面条件 - 根据基本面评分调整
            # 如果基本面评分低（<30分），更容易触发出货信号
            # 如果基本面评分高（>60分），需要更严格的技术面条件
            fundamental_cond = True  # 默认通过
            if fundamental_data:
                if fundamental_score < 30:
                    # 基本面较差，降低价格位置要求
                    price_cond = row['Price_Percentile'] > (PRICE_HIGH_PCT - 20)
                elif fundamental_score > 60:
                    # 基本面优秀，需要更严格的技术面条件
                    fundamental_cond = False
                    # 只有在价格极高且技术面明显恶化时才触发出货
                    if row['Price_Percentile'] > 80 and price_down_cond:
                        fundamental_cond = True
            
            # 增加的辅助条件（调整后的阈值和新增条件）
            # MACD线下穿信号线
            macd_cond = pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] < row['MACD_Signal']
            # RSI超买（调整阈值从70到65）
            rsi_cond = pd.notna(row.get('RSI')) and row['RSI'] > 65
            # OBV下降
            obv_cond = pd.notna(row.get('OBV')) and row['OBV'] < 0
            # 价格相对于5日均线位置（价格高于5日均线）
            ma5_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA5')) and row['Close'] > row['MA5']
            # 价格相对于10日均线位置（价格高于10日均线）
            ma10_cond = pd.notna(row.get('Close')) and pd.notna(row.get('MA10')) and row['Close'] > row['MA10']
            # 收盘价低于VWAP且放量 (VWAP条件)
            vwap_cond = pd.notna(row.get('Close')) and pd.notna(row.get('VWAP')) and pd.notna(row.get('Vol_Ratio')) and row['Close'] < row['VWAP'] and row['Vol_Ratio'] > 1.5
            # ATR放大 (ATR条件)
            atr_cond = pd.notna(row.get('ATR')) and pd.notna(row.get('Close')) and row['ATR'] > full_hist['ATR'].rolling(14).mean().reindex([row.name], method='ffill').iloc[0] * 1.5
            # CMF < -0.05 (资金流出)
            cmf_cond = pd.notna(row.get('CMF')) and row['CMF'] < -0.05
            # ADX > 25 (趋势明确)
            adx_cond = pd.notna(row.get('ADX')) and row['ADX'] > 25
            # 成交量z-score > 1.5 (异常放量)
            vol_z_cond = pd.notna(row.get('Vol_Z_Score')) and row['Vol_Z_Score'] > 1.5
            # 成交额z-score > 1.5 (异常成交额)
            turnover_z_cond = pd.notna(row.get('Turnover_Z_Score')) and row['Turnover_Z_Score'] > 1.5
            
            # 计算满足的辅助条件数量
            aux_conditions = [macd_cond, rsi_cond, obv_cond, ma5_cond, ma10_cond, vwap_cond, atr_cond, cmf_cond, adx_cond, vol_z_cond, turnover_z_cond]
            satisfied_aux_count = sum(aux_conditions)
            
            # 如果满足至少2个辅助条件，或者满足多个条件中的部分条件（更宽松的策略）
            aux_cond = satisfied_aux_count >= 2
            
            return price_cond and vol_cond and sb_cond and price_down_cond and aux_cond and fundamental_cond

        main_hist['Distribution_Signal'] = main_hist.apply(is_distribution, axis=1)
        main_hist['Distribution_Confirmed'] = mark_runs(main_hist['Distribution_Signal'], DISTRIBUTION_MIN_DAYS)

        # === 加权评分的出货信号（新增）===
        def is_distribution_scored(row, fundamental_score=None):
            """
            基于加权评分的出货信号检测
            
            返回: (score, signal, reasons)
            - score: 出货评分（0-10+）
            - signal: 信号级别 ('none', 'weak', 'strong')
            - reasons: 触发条件的列表
            """
            score = 0.0
            reasons = []

            # 价格位置：高位加分
            if pd.notna(row.get('Price_Percentile')) and row['Price_Percentile'] > PRICE_HIGH_PCT:
                score += DISTRIBUTION_WEIGHTS['price_high']
                reasons.append('price_high')

            # 成交量倍数（降低阈值从2.0到1.5）
            if pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > 1.5:
                score += DISTRIBUTION_WEIGHTS['vol_ratio']
                reasons.append('vol_ratio')

            # 成交量 z-score
            if pd.notna(row.get('Vol_Z_Score')) and row['Vol_Z_Score'] > 1.5:
                score += DISTRIBUTION_WEIGHTS['vol_z']
                reasons.append('vol_z')

            # MACD 线下穿
            if pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] < row['MACD_Signal']:
                score += DISTRIBUTION_WEIGHTS['macd_cross']
                reasons.append('macd_cross')

            # RSI 超买
            if pd.notna(row.get('RSI')) and row['RSI'] > 65:
                score += DISTRIBUTION_WEIGHTS['rsi_high']
                reasons.append('rsi_high')

            # CMF < -0.05 表示资金流出
            if pd.notna(row.get('CMF')) and row['CMF'] < -0.05:
                score += DISTRIBUTION_WEIGHTS['cmf_out']
                reasons.append('cmf_out')

            # OBV 下降
            if pd.notna(row.get('OBV')) and row['OBV'] < 0:
                score += DISTRIBUTION_WEIGHTS['obv_down']
                reasons.append('obv_down')

            # 收盘低于 VWAP 且放量
            if pd.notna(row.get('VWAP')) and pd.notna(row.get('Vol_Ratio')) and row['Close'] < row['VWAP'] and row['Vol_Ratio'] > 1.2:
                score += DISTRIBUTION_WEIGHTS['vwap_vol']
                reasons.append('vwap_vol')

            # 南向资金流出作为加分项（不是必须）
            if pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] < -SOUTHBOUND_THRESHOLD_OUT:
                score += DISTRIBUTION_WEIGHTS['southbound_out']
                reasons.append('southbound_out')

            # 价格下跌
            if (pd.notna(row.get('Prev_Close')) and row['Close'] < row['Prev_Close']) or (row['Close'] < row['Open']):
                score += DISTRIBUTION_WEIGHTS['price_down']
                reasons.append('price_down')

            # 基本面调整（不要完全阻止出货，而是调整阈值）
            if fundamental_score is not None:
                if fundamental_score > 60:
                    # 基本面优秀，需要更高的得分才抛售（避免把好公司频繁卖出）
                    score -= 1.0
                elif fundamental_score < 30:
                    # 基本面差时，更容易触发出货
                    score += 0.5

            # 返回分数与分层建议
            signal = None
            if score >= DISTRIBUTION_THRESHOLD_STRONG:
                signal = 'strong'    # 强烈出货（建议较大比例卖出）
            elif score >= DISTRIBUTION_THRESHOLD_WEAK:
                signal = 'weak'      # 弱出货（建议部分减仓或观察）
            else:
                signal = 'none'      # 无信号

            return score, signal, reasons

        # === 获利了结和ATR trailing stop功能（新增）===
        def check_profit_take_and_stop_loss(row, position_entry_price=None, full_hist=None):
            """
            检查是否需要止盈或止损
            
            Args:
                row: 当日数据
                position_entry_price: 持仓成本价（可选）
                full_hist: 完整历史数据（用于ATR计算）
            
            Returns:
                dict: 包含止盈/止损建议的字典
            """
            result = {
                'take_profit': False,
                'stop_loss': False,
                'trailing_stop': False,
                'reason': None,
                'action': None  # 'partial_sell', 'full_sell', 'hold'
            }

            if position_entry_price is None or pd.isna(position_entry_price):
                return result

            current_price = row['Close']

            # 计算持仓盈亏
            pnl = (current_price / position_entry_price - 1)

            # 止盈检查
            if pnl >= TAKE_PROFIT_PCT:
                # 如果同时出现任一出货相关信号（比如 RSI>65 或 MACD下穿），则建议部分卖出
                if pd.notna(row.get('RSI')) and row['RSI'] > 60:
                    result['take_profit'] = True
                    result['reason'] = f'止盈触发：盈利{pnl*100:.2f}%，RSI={row["RSI"]:.2f}'
                    result['action'] = 'partial_sell'
                elif pd.notna(row.get('MACD')) and pd.notna(row.get('MACD_Signal')) and row['MACD'] < row['MACD_Signal']:
                    result['take_profit'] = True
                    result['reason'] = f'止盈触发：盈利{pnl*100:.2f}%，MACD死叉'
                    result['action'] = 'partial_sell'

            # 止损检查
            if pnl <= -STOP_LOSS_PCT:
                result['stop_loss'] = True
                result['reason'] = f'止损触发：亏损{abs(pnl)*100:.2f}%'
                result['action'] = 'full_sell'

            # ATR trailing stop（需要完整历史数据）
            if full_hist is not None and pd.notna(row.get('ATR')):
                # 计算最近N天的最高价
                peak_price = full_hist['Close'].tail(20).max()
                current_atr = row['ATR']

                # 如果价格从最高点回撤超过TRAILING_ATR_MULT倍ATR，触发trailing stop
                if current_price < (peak_price - TRAILING_ATR_MULT * current_atr):
                    result['trailing_stop'] = True
                    result['reason'] = f'ATR Trailing Stop触发：价格从高点{peak_price:.2f}回撤{((peak_price - current_price) / peak_price * 100):.2f}%'
                    result['action'] = 'partial_sell'

            return result

        # 是否存在信号
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()

        # === 加权评分系统集成（新增）===
        if USE_SCORED_SIGNALS:
            # 计算建仓评分
            buildup_scores = []
            buildup_signals = []
            buildup_reasons_list = []

            for _, row in main_hist.iterrows():
                score, signal, reasons = is_buildup_scored(row, fundamental_score)
                buildup_scores.append(score)
                buildup_signals.append(signal)
                buildup_reasons_list.append(','.join(reasons) if reasons else '')

            main_hist['Buildup_Score'] = buildup_scores
            main_hist['Buildup_Signal_Level'] = buildup_signals
            main_hist['Buildup_Reasons'] = buildup_reasons_list

            # 计算出货评分
            distribution_scores = []
            distribution_signals = []
            distribution_reasons_list = []

            for _, row in main_hist.iterrows():
                score, signal, reasons = is_distribution_scored(row, fundamental_score)
                distribution_scores.append(score)
                distribution_signals.append(signal)
                distribution_reasons_list.append(','.join(reasons) if reasons else '')

            main_hist['Distribution_Score'] = distribution_scores
            main_hist['Distribution_Signal_Level'] = distribution_signals
            main_hist['Distribution_Reasons'] = distribution_reasons_list

            # 获取最新的评分和信号级别
            latest_buildup_score = main_hist['Buildup_Score'].iloc[-1]
            latest_buildup_level = main_hist['Buildup_Signal_Level'].iloc[-1]
            latest_buildup_reasons = main_hist['Buildup_Reasons'].iloc[-1]

            latest_distribution_score = main_hist['Distribution_Score'].iloc[-1]
            latest_distribution_level = main_hist['Distribution_Signal_Level'].iloc[-1]
            latest_distribution_reasons = main_hist['Distribution_Reasons'].iloc[-1]

            # 检查止盈和止损（假设没有持仓成本价，这里只是示例）
            # 在实际使用时，需要传入position_entry_price参数
            profit_take_result = check_profit_take_and_stop_loss(
                main_hist.iloc[-1],
                position_entry_price=None,  # 需要从外部传入
                full_hist=full_hist
            )

            print(f"  📊 {name} 建仓评分: {latest_buildup_score:.2f}, 信号级别: {latest_buildup_level}")
            if latest_buildup_reasons:
                print(f"    触发原因: {latest_buildup_reasons}")

            print(f"  📊 {name} 出货评分: {latest_distribution_score:.2f}, 信号级别: {latest_distribution_level}")
            if latest_distribution_reasons:
                print(f"    触发原因: {latest_distribution_reasons}")

            if profit_take_result['take_profit']:
                print(f"  💰 {name} {profit_take_result['reason']}")
            if profit_take_result['stop_loss']:
                print(f"  ⛔ {name} {profit_take_result['reason']}")
            if profit_take_result['trailing_stop']:
                print(f"  📉 {name} {profit_take_result['reason']}")
        else:
            # 使用原有的布尔逻辑（向后兼容）
            latest_buildup_score = None
            latest_buildup_level = None
            latest_buildup_reasons = None
            latest_distribution_score = None
            latest_distribution_level = None
            latest_distribution_reasons = None
            profit_take_result = None

        # TAV信号质量过滤（如果可用）
        tav_quality_score = None
        tav_recommendation = None
        if TAV_AVAILABLE and TECHNICAL_ANALYSIS_AVAILABLE:
            try:
                # 使用TAV分析器评估信号质量
                tav_analyzer = TechnicalAnalyzerV2(enable_tav=True)
                
                # 为TAV分析准备数据（需要完整的OHLCV数据）
                tav_data = full_hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # 计算TAV指标
                tav_data = tav_analyzer.calculate_all_indicators(tav_data, asset_type='stock')
                
                # 获取TAV分析摘要
                tav_summary = tav_analyzer.get_tav_analysis_summary(tav_data, 'stock')
                
                if tav_summary:
                    tav_quality_score = tav_summary.get('tav_score', 0)
                    tav_recommendation = tav_summary.get('recommendation', '无建议')
                    
                    # TAV信号质量过滤逻辑
                    # 如果TAV评分较低，降低信号的可靠性
                    if tav_quality_score < 30:
                        print(f"  ⚠️ TAV评分较低({tav_quality_score:.1f})，信号质量可能不佳")
                        # 可以选择性地降低信号的权重或标记为低质量
                        if has_buildup:
                            print(f"  ⚠️ 建仓信号被TAV系统标记为低质量")
                        if has_distribution:
                            print(f"  ⚠️ 出货信号被TAV系统标记为低质量")
                    elif tav_quality_score >= 70:
                        print(f"  ✅ TAV评分较高({tav_quality_score:.1f})，信号质量良好")
                        if has_buildup:
                            print(f"  ✅ 建仓信号得到TAV系统确认")
                        if has_distribution:
                            print(f"  ✅ 出货信号得到TAV系统确认")
                    
                    print(f"  📊 TAV分析: {tav_recommendation}")
            except Exception as e:
                print(f"  ⚠️ TAV分析失败: {e}")
                tav_quality_score = None
                tav_recommendation = None

        # 保存图表
        if SAVE_CHARTS:
            # 如果有恒生指数数据，则绘制对比图
            if hsi_hist is not None and not hsi_hist.empty:
                hsi_plot = hsi_hist['Close'].reindex(main_hist.index, method='ffill')
                stock_plot = main_hist['Close']
                rs_ratio_display = safe_round(rs_ratio * 100, 2)
                rs_diff_display = safe_round(rs_diff * 100, 2)
                plt.figure(figsize=(10, 6))
                plt.plot(stock_plot.index, stock_plot / stock_plot.iloc[0], 'b-o', label=f'{code} {name}')
                if not hsi_plot.isna().all():
                    plt.plot(hsi_plot.index, hsi_plot / hsi_plot.iloc[0], 'orange', linestyle='--', label='恒生指数')
                title = f"{code} {name} vs 恒指 | RS_ratio: {rs_ratio_display if rs_ratio_display is not None else 'NA'}% | RS_diff: {rs_diff_display if rs_diff_display is not None else 'NA'}%"
                if has_buildup:
                    title += " [建仓]"
                if has_distribution:
                    title += " [出货]"
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                status = ("_buildup" if has_buildup else "") + ("_distribution" if has_distribution else "")
                safe_name = name.replace('/', '_').replace(' ', '_')
                plt.savefig(f"{CHART_DIR}/{code}_{safe_name}{status}.png")
                plt.close()
            else:
                # 如果没有恒生指数数据，只绘制股票价格图
                stock_plot = main_hist['Close']
                plt.figure(figsize=(10, 6))
                plt.plot(stock_plot.index, stock_plot / stock_plot.iloc[0], 'b-o', label=f'{code} {name}')
                title = f"{code} {name}"
                if has_buildup:
                    title += " [建仓]"
                if has_distribution:
                    title += " [出货]"
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                status = ("_buildup" if has_buildup else "") + ("_distribution" if has_distribution else "")
                safe_name = name.replace('/', '_').replace(' ', '_')
                plt.savefig(f"{CHART_DIR}/{code}_{safe_name}{status}.png")
                plt.close()

        # 计算换手率 (使用实际流通股本)
        # 换手率 = 成交量 / 流通股本 * 100%
        # 从 AKShare 获取已发行股本数据
        float_shares = None
        try:
            # 使用 AKShare 获取港股财务指标数据
            financial_df = ak.stock_hk_financial_indicator_em(symbol=stock_code)
            # 检查数据是否有效
            if financial_df is not None:
                # 检查是否是DataFrame且不为空
                if isinstance(financial_df, pd.DataFrame) and not financial_df.empty:
                    # 检查是否包含所需列
                    if '已发行股本(股)' in financial_df.columns:
                        # 获取已发行股本字段
                        issued_shares = financial_df['已发行股本(股)'].iloc[0]
                        # 检查值是否有效
                        if issued_shares is not None and not (isinstance(issued_shares, float) and pd.notna(issued_shares)) and issued_shares > 0:
                            float_shares = float(issued_shares)
                            if float_shares <= 0:
                                float_shares = None
                        else:
                            print(f"  ⚠️ {code} 的已发行股本数据无效")
                    else:
                        print(f"  ⚠️ {code} 的财务指标数据中未找到已发行股本字段")
                else:
                    print(f"  ⚠️ {code} 的财务指标数据为空或不是DataFrame")
            else:
                print(f"  ⚠️ 无法获取 {code} 的财务指标数据")
        except Exception as e:
            float_shares = None
            print(f"  ⚠️ 获取 {code} 已发行股本数据时出错: {e}")
        
        # 只有在有流通股本数据时才计算换手率
        turnover_rate = (main_hist['Volume'].iloc[-1] / float_shares) * 100 if len(main_hist) > 0 and float_shares is not None and float_shares > 0 else None
        
        # 如果成功获取到换手率，显示调试信息
        if turnover_rate is not None:
            print(f"  ℹ️ {code} 换手率计算: 成交量={main_hist['Volume'].iloc[-1]}, 已发行股本={float_shares}, 换手率={turnover_rate:.4f}%")

        # 返回结构（保留原始数值：RS 为小数，RS_diff 小数；展示时再乘100）
        last_close = main_hist['Close'].iloc[-1]
        prev_close = main_hist['Close'].iloc[-2] if len(main_hist) >= 2 else None
        change_pct = ((last_close / prev_close) - 1) * 100 if prev_close is not None and prev_close != 0 else None

        # 计算放量上涨和缩量回调信号
        # 放量上涨：收盘价 > 开盘价 且 Vol_Ratio > 1.5
        main_hist['Strong_Volume_Up'] = (main_hist['Close'] > main_hist['Open']) & (main_hist['Vol_Ratio'] > 1.5)
        # 缩量回调：收盘价 < 前一日收盘价 且 Vol_Ratio < 1.0 且跌幅 < 2%
        main_hist['Weak_Volume_Down'] = (main_hist['Close'] < main_hist['Prev_Close']) & (main_hist['Vol_Ratio'] < 1.0) & ((main_hist['Prev_Close'] - main_hist['Close']) / main_hist['Prev_Close'] < 0.02)
        
        # 计算新增的技术指标信号
        # 布林带突破信号
        main_hist['BB_Breakout_Signal'] = (main_hist['BB_Breakout'] > 1.0) | (main_hist['BB_Breakout'] < 0.0)
        # RSI背离信号
        main_hist['RSI_Divergence'] = (main_hist['RSI_ROC'] < 0) & (main_hist['Close'].pct_change() > 0)
        # MACD柱状图变化率信号
        main_hist['MACD_Hist_ROC_Signal'] = main_hist['MACD_Hist_ROC'] > 0.1
        # CMF趋势信号
        main_hist['CMF_Trend_Signal'] = main_hist['CMF'] > main_hist['CMF_Signal']
        # ATR动态阈值信号
        main_hist['ATR_Ratio_Signal'] = main_hist['ATR_Ratio'] > 1.5
        # 随机振荡器信号
        main_hist['Stoch_Signal'] = (main_hist['Stoch_K'] < 20) | (main_hist['Stoch_K'] > 80)
        # Williams %R信号
        main_hist['Williams_R_Signal'] = (main_hist['Williams_R'] < -80) | (main_hist['Williams_R'] > -20)
        # 价格变化率信号
        main_hist['ROC_Signal'] = main_hist['ROC'] > 0.05
        # 成交量比率信号
        main_hist['Volume_Ratio_Signal'] = main_hist['Volume_Ratio'] > 1.5

        # === 多周期指标计算（新增）===
        # 计算多周期价格变化率和趋势方向
        multi_period_metrics = calculate_multi_period_metrics(full_hist, periods=[3, 5, 10, 20, 60])
        
        # 计算多周期相对强度（股票 vs 恒生指数）
        multi_period_rs = calculate_relative_strength_multi_period(full_hist, hsi_hist, periods=[3, 5, 10, 20, 60])
        
        # 计算多周期趋势综合评分
        multi_period_trend_score = get_multi_period_trend_score(multi_period_metrics, periods=[3, 5, 10, 20, 60])
        
        # 计算多周期相对强度综合评分
        multi_period_rs_score = get_multi_period_rs_score(multi_period_rs, periods=[3, 5, 10, 20, 60])

        result = {
            'code': code,
            'name': name,
            'has_buildup': bool(has_buildup),
            'has_distribution': bool(has_distribution),
            'outperforms_hsi': bool(outperforms),
            'relative_strength': safe_round(rs_ratio, 4),         # 小数（如 0.05 表示 5%）
            'relative_strength_diff': safe_round(rs_diff, 4),     # 小数（如 0.05 表示 5%）
            'last_close': safe_round(last_close, 2),
            'prev_close': safe_round(prev_close, 2) if prev_close is not None else None,
            'change_pct': safe_round(change_pct, 2) if change_pct is not None else None,
            'price_percentile': safe_round(main_hist['Price_Percentile'].iloc[-1], 2),
            'vol_ratio': safe_round(main_hist['Vol_Ratio'].iloc[-1], 2) if pd.notna(main_hist['Vol_Ratio'].iloc[-1]) else None,
            'turnover': safe_round((last_close * main_hist['Volume'].iloc[-1]) / 1_000_000, 2),  # 百万
            'turnover_rate': safe_round(turnover_rate, 2) if turnover_rate is not None else None,  # 换手率 %
            'southbound': safe_round(main_hist['Southbound_Net'].iloc[-1], 2),  # 单位：万
            'ma5_deviation': safe_round(((last_close / main_hist['MA5'].iloc[-1]) - 1) * 100, 2) if pd.notna(main_hist['MA5'].iloc[-1]) and main_hist['MA5'].iloc[-1] > 0 else None,
            'ma10_deviation': safe_round(((last_close / main_hist['MA10'].iloc[-1]) - 1) * 100, 2) if pd.notna(main_hist['MA10'].iloc[-1]) and main_hist['MA10'].iloc[-1] > 0 else None,
            'macd': safe_round(main_hist['MACD'].iloc[-1], 4) if pd.notna(main_hist['MACD'].iloc[-1]) else None,
            'rsi': safe_round(main_hist['RSI'].iloc[-1], 2) if pd.notna(main_hist['RSI'].iloc[-1]) else None,
            'volatility': safe_round(main_hist['Volatility'].iloc[-1] * 100, 2) if pd.notna(main_hist['Volatility'].iloc[-1]) else None,  # 百分比
            'obv': safe_round(main_hist['OBV'].iloc[-1], 2) if pd.notna(main_hist['OBV'].iloc[-1]) else None,  # OBV指标
            'vwap': safe_round(main_hist['VWAP'].iloc[-1], 2) if pd.notna(main_hist['VWAP'].iloc[-1]) else None,  # VWAP
            'atr': safe_round(main_hist['ATR'].iloc[-1], 2) if pd.notna(main_hist['ATR'].iloc[-1]) else None,  # ATR
            'cmf': safe_round(main_hist['CMF'].iloc[-1], 4) if pd.notna(main_hist['CMF'].iloc[-1]) else None,  # CMF
            'adx': safe_round(main_hist['ADX'].iloc[-1], 2) if pd.notna(main_hist['ADX'].iloc[-1]) else None,  # ADX
            'bb_width': safe_round(main_hist['BB_Width'].iloc[-1] * 100, 2) if pd.notna(main_hist['BB_Width'].iloc[-1]) else None,  # 布林带宽度
            'bb_breakout': safe_round(main_hist['BB_Breakout'].iloc[-1], 2) if pd.notna(main_hist['BB_Breakout'].iloc[-1]) else None,  # 布林带突破
            'vol_z_score': safe_round(main_hist['Vol_Z_Score'].iloc[-1], 2) if pd.notna(main_hist['Vol_Z_Score'].iloc[-1]) else None,  # 成交量z-score
            'turnover_z_score': safe_round(main_hist['Turnover_Z_Score'].iloc[-1], 2) if pd.notna(main_hist['Turnover_Z_Score'].iloc[-1]) else None,  # 成交额z-score
            'macd_hist': safe_round(main_hist['MACD_Hist'].iloc[-1], 4) if pd.notna(main_hist['MACD_Hist'].iloc[-1]) else None,  # MACD柱状图
            'macd_hist_roc': safe_round(main_hist['MACD_Hist_ROC'].iloc[-1], 4) if pd.notna(main_hist['MACD_Hist_ROC'].iloc[-1]) else None,  # MACD柱状图变化率
            'rsi_roc': safe_round(main_hist['RSI_ROC'].iloc[-1], 4) if pd.notna(main_hist['RSI_ROC'].iloc[-1]) else None,  # RSI变化率
            'cmf_signal': safe_round(main_hist['CMF_Signal'].iloc[-1], 4) if pd.notna(main_hist['CMF_Signal'].iloc[-1]) else None,  # CMF信号线
            'atr_ratio': safe_round(main_hist['ATR_Ratio'].iloc[-1], 2) if pd.notna(main_hist['ATR_Ratio'].iloc[-1]) else None,  # ATR比率
            'stoch_k': safe_round(main_hist['Stoch_K'].iloc[-1], 2) if pd.notna(main_hist['Stoch_K'].iloc[-1]) else None,  # 随机振荡器K值
            'stoch_d': safe_round(main_hist['Stoch_D'].iloc[-1], 2) if pd.notna(main_hist['Stoch_D'].iloc[-1]) else None,  # 随机振荡器D值
            'williams_r': safe_round(main_hist['Williams_R'].iloc[-1], 2) if pd.notna(main_hist['Williams_R'].iloc[-1]) else None,  # Williams %R
            'roc': safe_round(main_hist['ROC'].iloc[-1], 4) if pd.notna(main_hist['ROC'].iloc[-1]) else None,  # 价格变化率
            'volume_ratio': safe_round(main_hist['Volume_Ratio'].iloc[-1], 2) if pd.notna(main_hist['Volume_Ratio'].iloc[-1]) else None,  # 成交量比率
            'strong_volume_up': bool(main_hist['Strong_Volume_Up'].iloc[-1]),  # 放量上涨
            'weak_volume_down': bool(main_hist['Weak_Volume_Down'].iloc[-1]),  # 缩量回调
            'bb_breakout_signal': bool(main_hist['BB_Breakout_Signal'].iloc[-1]),  # 布林带突破信号
            'rsi_divergence': bool(main_hist['RSI_Divergence'].iloc[-1]),  # RSI背离信号
            'macd_hist_roc_signal': bool(main_hist['MACD_Hist_ROC_Signal'].iloc[-1]),  # MACD柱状图变化率信号
            'cmf_trend_signal': bool(main_hist['CMF_Trend_Signal'].iloc[-1]),  # CMF趋势信号
            'atr_ratio_signal': bool(main_hist['ATR_Ratio_Signal'].iloc[-1]),  # ATR比率信号
            'stoch_signal': bool(main_hist['Stoch_Signal'].iloc[-1]),  # 随机振荡器信号
            'williams_r_signal': bool(main_hist['Williams_R_Signal'].iloc[-1]),  # Williams %R信号
            'roc_signal': bool(main_hist['ROC_Signal'].iloc[-1]),  # 价格变化率信号
            'volume_ratio_signal': bool(main_hist['Volume_Ratio_Signal'].iloc[-1]),  # 成交量比率信号
            'buildup_dates': main_hist[main_hist['Buildup_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
            'distribution_dates': main_hist[main_hist['Distribution_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
            # 加权评分系统信息（新增）
            'buildup_score': safe_round(latest_buildup_score, 2) if latest_buildup_score is not None else None,
            'buildup_level': latest_buildup_level,
            'buildup_reasons': latest_buildup_reasons,
            'distribution_score': safe_round(latest_distribution_score, 2) if latest_distribution_score is not None else None,
            'distribution_level': latest_distribution_level,
            'distribution_reasons': latest_distribution_reasons,
            # 止盈止损信息（新增）
            'take_profit': profit_take_result['take_profit'] if profit_take_result else False,
            'stop_loss': profit_take_result['stop_loss'] if profit_take_result else False,
            'trailing_stop': profit_take_result['trailing_stop'] if profit_take_result else False,
            'profit_loss_reason': profit_take_result['reason'] if profit_take_result else None,
            'profit_loss_action': profit_take_result['action'] if profit_take_result else None,
            # TAV信号质量信息
            'tav_quality_score': tav_quality_score,
            'tav_recommendation': tav_recommendation,
            'tav_score': tav_quality_score if tav_quality_score is not None else 0,
            'tav_status': tav_recommendation if tav_recommendation else '无TAV',
            # 多周期指标（新增）
            '3d_return': multi_period_metrics.get('3d_return'),
            '3d_trend': multi_period_metrics.get('3d_trend'),
            '5d_return': multi_period_metrics.get('5d_return'),
            '5d_trend': multi_period_metrics.get('5d_trend'),
            '10d_return': multi_period_metrics.get('10d_return'),
            '10d_trend': multi_period_metrics.get('10d_trend'),
            '20d_return': multi_period_metrics.get('20d_return'),
            '20d_trend': multi_period_metrics.get('20d_trend'),
            '60d_return': multi_period_metrics.get('60d_return'),
            '60d_trend': multi_period_metrics.get('60d_trend'),
            # 多周期相对强度（新增）
            '3d_rs': multi_period_rs.get('3d_rs'),
            '3d_rs_signal': multi_period_rs.get('3d_rs_signal'),
            '5d_rs': multi_period_rs.get('5d_rs'),
            '5d_rs_signal': multi_period_rs.get('5d_rs_signal'),
            '10d_rs': multi_period_rs.get('10d_rs'),
            '10d_rs_signal': multi_period_rs.get('10d_rs_signal'),
            '20d_rs': multi_period_rs.get('20d_rs'),
            '20d_rs_signal': multi_period_rs.get('20d_rs_signal'),
            '60d_rs': multi_period_rs.get('60d_rs'),
            '60d_rs_signal': multi_period_rs.get('60d_rs_signal'),
            # 多周期综合评分（新增）
            'multi_period_trend_score': multi_period_trend_score,
            'multi_period_rs_score': multi_period_rs_score,
        }
        
        # 重新获取上个交易日的评分数据（在所有评分计算完成后）
        if previous_day_indicators is not None:
            try:
                # 计算上个交易日的日期
                yesterday = datetime.now().date() - timedelta(days=1)
                while yesterday.weekday() >= 5:  # 5=周六, 6=周日
                    yesterday -= timedelta(days=1)
                
                # 如果指定了运行日期，使用运行日期的前一天
                if run_date:
                    target_date = pd.to_datetime(run_date).date()
                    previous_trading_date = target_date - timedelta(days=1)
                    while previous_trading_date.weekday() >= 5:
                        previous_trading_date -= timedelta(days=1)
                else:
                    previous_trading_date = yesterday
                
                # 筛选出上个交易日及之前的数据
                previous_trading_date_timestamp = pd.Timestamp(previous_trading_date)
                if main_hist.index.tz is not None:
                    previous_trading_date_timestamp = previous_trading_date_timestamp.tz_localize(main_hist.index.tz)
                
                # 从main_hist中查找上个交易日数据（因为评分在main_hist中）
                prev_filtered_hist = main_hist[main_hist.index <= previous_trading_date_timestamp]
                
                if not prev_filtered_hist.empty:
                    # 获取上个交易日的建仓和出货评分
                    if 'Buildup_Score' in prev_filtered_hist.columns:
                        prev_buildup_score = prev_filtered_hist['Buildup_Score'].iloc[-1] if pd.notna(prev_filtered_hist['Buildup_Score'].iloc[-1]) else None
                        previous_day_indicators['buildup_score'] = safe_round(prev_buildup_score, 2) if prev_buildup_score is not None else None
                    if 'Distribution_Score' in prev_filtered_hist.columns:
                        prev_distribution_score = prev_filtered_hist['Distribution_Score'].iloc[-1] if pd.notna(prev_filtered_hist['Distribution_Score'].iloc[-1]) else None
                        previous_day_indicators['distribution_score'] = safe_round(prev_distribution_score, 2) if prev_distribution_score is not None else None
                    
                    # TAV评分需要重新计算（使用full_hist，因为需要High和Low列）
                    if TAV_AVAILABLE and TECHNICAL_ANALYSIS_AVAILABLE:
                        try:
                            # 从full_hist中获取上个交易日数据（包含High和Low列）
                            prev_filtered_full_hist = full_hist[full_hist.index <= previous_trading_date_timestamp]
                            if not prev_filtered_full_hist.empty:
                                tav_analyzer = TechnicalAnalyzerV2(enable_tav=True)
                                tav_data = prev_filtered_full_hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                                tav_data = tav_analyzer.calculate_all_indicators(tav_data, asset_type='stock')
                                tav_summary = tav_analyzer.get_tav_analysis_summary(tav_data, 'stock')
                                if tav_summary:
                                    prev_tav_score = tav_summary.get('tav_score', 0)
                                    previous_day_indicators['tav_score'] = safe_round(prev_tav_score, 1) if prev_tav_score is not None else None
                        except Exception:
                            pass
            except Exception as e:
                print(f"  ⚠️ 重新获取上个交易日评分失败: {e}")
        
        # 添加上个交易日指标信息
        result['prev_day_indicators'] = previous_day_indicators
        
        # 添加基本面数据
        if fundamental_data:
            # 添加基本面评分和详细信息
            result['fundamental_score'] = fundamental_score
            result['fundamental_details'] = fundamental_details

            # 只添加PE和PB
            result['pe_ratio'] = fundamental_data.get('fi_pe_ratio')
            result['pb_ratio'] = fundamental_data.get('fi_pb_ratio')

            # 添加数据获取时间
            result['fundamental_data_time'] = fundamental_data.get('data_fetch_time')
        return result

    except Exception as e:
        print(f"❌ {name} 分析出错: {e}")
        return None

# Markdown到HTML的转换函数
def markdown_to_html(md_text):
    if not md_text:
        return md_text

    # 保存原始文本并逐行处理
    lines = md_text.split('\n')
    html_lines = []
    in_list = False
    list_type = None  # 'ul' for unordered, 'ol' for ordered
    in_table = False  # 标记是否在表格中
    table_header_processed = False  # 标记表格头部是否已处理

    for line in lines:
        stripped_line = line.strip()
        
        # 检查是否是表格分隔行（包含 | 和 - 用于定义表格结构）
        table_separator_match = re.match(r'^\s*\|?\s*[:\-\s\|]*\|\s*$', line)
        if table_separator_match and '|' in line and any(c in line for c in ['-', ':']):
            # 这是表格的分隔行，跳过处理
            continue

        # 检查是否是表格行（包含 | 分隔符）
        is_table_row = '|' in line and not stripped_line.startswith('```')
        
        if is_table_row and not table_separator_match:
            # 处理表格行
            if not in_table:
                # 开始新表格
                in_table = True
                table_header_processed = False
                html_lines.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
            
            # 分割单元格并去除空白
            cells = [cell.strip() for cell in line.split('|')]
            # 过滤掉首尾的空字符串（因为 | 开头或结尾会产生空字符串）
            # 但是要保留所有非空的单元格
            cells = [cell for cell in cells if cell.strip()]
            
            # 确定是表头还是数据行
            if not table_header_processed and any('---' in cell for cell in [c for c in cells if c.strip()]):
                # 如果这一行包含 ---，则认为是分隔行，跳过
                continue
            elif not table_header_processed:
                # 首次遇到非分隔行，作为表头处理
                html_lines.append('<thead><tr>')
                for cell in cells:
                    # 处理单元格内的粗体和斜体
                    cell_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    cell_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cell_content)
                    cell_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', cell_content)
                    cell_content = re.sub(r'_(.*?)_', r'<em>\1</em>', cell_content)
                    # 处理代码
                    cell_content = re.sub(r'`(.*?)`', r'<code>\1</code>', cell_content)
                    # 处理链接
                    cell_content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', cell_content)
                    html_lines.append(f'<th style="padding: 8px; text-align: left; border: 1px solid #ddd;">{cell_content}</th>')
                html_lines.append('</tr></thead><tbody>')
                table_header_processed = True
            else:
                # 数据行
                html_lines.append('<tr>')
                for cell in cells:
                    # 处理单元格内的粗体和斜体
                    cell_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    cell_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cell_content)
                    cell_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', cell_content)
                    cell_content = re.sub(r'_(.*?)_', r'<em>\1</em>', cell_content)
                    # 处理代码
                    cell_content = re.sub(r'`(.*?)`', r'<code>\1</code>', cell_content)
                    # 处理链接
                    cell_content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', cell_content)
                    html_lines.append(f'<td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{cell_content}</td>')
                html_lines.append('</tr>')
            continue

        # 如果当前行不是表格行，但之前在表格中，则关闭表格
        if in_table:
            html_lines.append('</tbody></table>')
            in_table = False
            table_header_processed = False

        # 检查是否是标题
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
            header_level = len(header_match.group(1))
            header_content = header_match.group(2)
            # 处理标题内的粗体和斜体
            header_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', header_content)
            header_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', header_content)
            header_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', header_content)
            header_content = re.sub(r'_(.*?)_', r'<em>\1</em>', header_content)
            html_lines.append(f'<h{header_level}>{header_content}</h{header_level}>')
            continue

        # 检查是否是列表项（无序）
        ul_match = re.match(r'^\s*[-*+]\s+(.*)', line)
        if ul_match:
            content = ul_match.group(1).strip()
            # 处理列表项内的粗体和斜体
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', content)
            content = re.sub(r'_(.*?)_', r'<em>\1</em>', content)
            
            if not in_list or list_type != 'ul':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            
            # 检查是否有嵌套
            indent_level = len(ul_match.group(0)) - len(ul_match.group(0).lstrip())
            if indent_level > 0:
                # 这里简单处理，实际可以更复杂
                html_lines.append(f'<li>{content}</li>')
            else:
                html_lines.append(f'<li>{content}</li>')
            continue

        # 检查是否是列表项（有序）
        ol_match = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if ol_match:
            content = ol_match.group(2).strip()
            # 处理列表项内的粗体和斜体
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', content)
            content = re.sub(r'_(.*?)_', r'<em>\1</em>', content)
            
            if not in_list or list_type != 'ol':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            
            html_lines.append(f'<li>{content}</li>')
            continue

        # 如果当前行不是列表项，但之前在列表中，则关闭列表
        if in_list:
            html_lines.append(f'</{list_type}>')
            in_list = False

        # 处理普通行
        if stripped_line:
            # 处理粗体和斜体
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            processed_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_line)
            processed_line = re.sub(r'__(.*?)__', r'<strong>\1</strong>', processed_line)
            processed_line = re.sub(r'_(.*?)_', r'<em>\1</em>', processed_line)
            # 处理代码
            processed_line = re.sub(r'`(.*?)`', r'<code>\1</code>', processed_line)
            # 处理链接
            processed_line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', processed_line)
            html_lines.append(processed_line)
        else:
            # 空行转为<br>
            html_lines.append('<br>')

    # 如果文档以列表结束，关闭列表
    if in_list:
        html_lines.append(f'</{list_type}>')

    # 如果文档以表格结束，关闭表格
    if in_table:
        html_lines.append('</tbody></table>')

    # 将所有行用<br>连接（但避免在已有HTML标签后添加额外的<br>）
    final_html = '<br>'.join(html_lines)
    # 修复多余的<br>标签
    final_html = re.sub(r'<br>(\s*<(ul|ol|h[1-6]|/ul|/ol|/h[1-6]|table|/table|/tbody|/thead|tr|/tr|td|/td|th|/th)>)', r'\1', final_html)
    final_html = re.sub(r'<br><br>', r'<br>', final_html)

    return final_html
# ==============================
# 5. 批量分析与报告生成
# ==============================
def main(run_date=None):
    print("="*80)
    print("🚀 港股主力资金追踪器（建仓 + 出货 双信号）")
    if run_date:
        print(f"分析日期: {run_date}")
    print(f"分析 {len(WATCHLIST)} 只股票 | 窗口: {DAYS_ANALYSIS} 日")
    print("="*80)

    results = []
    for code, name in WATCHLIST.items():
        res = analyze_stock(code, name, run_date)
        if res:
            results.append(res)

    if not results:
        print("❌ 无结果")
    else:
        df = pd.DataFrame(results)

        # 为展示方便，添加展示列（百分比形式）但保留原始数值列用于机器化处理
        df['RS_ratio_%'] = df['relative_strength'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else None)
        df['RS_diff_%'] = df['relative_strength_diff'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else None)

        # 选择并重命名列用于最终报告（保留 machine-friendly 列名以及展示列）
        df_report = df[[
            # 基本信息
            'name', 'code', 'last_close', 'change_pct',
            # 价格位置
            'price_percentile',
            # 成交量相关
            'vol_ratio', 'vol_z_score', 'turnover_z_score', 'turnover', 'turnover_rate', 'vwap', 'volume_ratio', 'volume_ratio_signal',
            # 波动性指标
            'atr', 'atr_ratio', 'atr_ratio_signal', 'bb_width', 'bb_breakout', 'volatility',
            # 均线偏离
            'ma5_deviation', 'ma10_deviation',
            # 技术指标
            'rsi', 'rsi_roc', 'rsi_divergence',
            'macd', 'macd_hist', 'macd_hist_roc', 'macd_hist_roc_signal',
            'obv',
            'cmf', 'cmf_signal', 'cmf_trend_signal',
            'stoch_k', 'stoch_d', 'stoch_signal',
            'williams_r', 'williams_r_signal',
            'bb_breakout_signal',
            'roc_signal',
            # 资金流向指标
            'southbound',
            # 相对表现
            'RS_ratio_%', 'RS_diff_%', 'outperforms_hsi',
            # 基本面数据（只保留PE和PB）
            'fundamental_score', 'pe_ratio', 'pb_ratio',
            # 新的评分系统字段
            'buildup_score', 'buildup_level', 'buildup_reasons',
            'distribution_score', 'distribution_level', 'distribution_reasons',
            'take_profit', 'stop_loss', 'trailing_stop',
            # 信号指标
            'strong_volume_up', 'weak_volume_down',
            # TAV评分
            'tav_score', 'tav_status',
            # 上个交易日指标（新增）
            'prev_day_indicators',
            # 多周期指标（新增）
            '3d_return', '3d_trend', '5d_return', '5d_trend',
            '10d_return', '10d_trend', '20d_return', '20d_trend',
            '60d_return', '60d_trend',
            '3d_rs', '3d_rs_signal', '5d_rs', '5d_rs_signal',
            '10d_rs', '10d_rs_signal', '20d_rs', '20d_rs_signal',
            '60d_rs', '60d_rs_signal',
            'multi_period_trend_score', 'multi_period_rs_score'
        ]]
        
        # 从prev_day_indicators中提取上个交易日指标作为独立列
        df_report['prev_rsi'] = df_report['prev_day_indicators'].apply(lambda x: x.get('rsi') if isinstance(x, dict) else None)
        df_report['prev_macd'] = df_report['prev_day_indicators'].apply(lambda x: x.get('macd') if isinstance(x, dict) else None)
        df_report['prev_price'] = df_report['prev_day_indicators'].apply(lambda x: x.get('price') if isinstance(x, dict) else None)
        df_report['prev_buildup_score'] = df_report['prev_day_indicators'].apply(lambda x: x.get('buildup_score') if isinstance(x, dict) else None)
        df_report['prev_distribution_score'] = df_report['prev_day_indicators'].apply(lambda x: x.get('distribution_score') if isinstance(x, dict) else None)
        df_report['prev_tav_score'] = df_report['prev_day_indicators'].apply(lambda x: x.get('tav_score') if isinstance(x, dict) else None)
        
        # 移除prev_day_indicators列（已提取为独立列）
        df_report = df_report.drop('prev_day_indicators', axis=1)
        
        df_report.columns = [
            # 基本信息
            '股票名称', '代码', '最新价', '涨跌幅(%)',
            # 价格位置
            '位置(%)',
            # 成交量相关
            '量比', '成交量z-score', '成交额z-score', '成交金额(百万)', '换手率(%)', 'VWAP', '成交量比率', '成交量比率信号',
            # 波动性指标
            'ATR', 'ATR比率', 'ATR比率信号', '布林带宽度(%)', '布林带突破', '波动率(%)',
            # 均线偏离
            '5日均线偏离(%)', '10日均线偏离(%)',
            # 技术指标
            'RSI', 'RSI变化率', 'RSI背离信号',
            'MACD', 'MACD柱状图', 'MACD柱状图变化率', 'MACD柱状图变化率信号',
            'OBV',
            'CMF', 'CMF信号线', 'CMF趋势信号',
            '随机振荡器K', '随机振荡器D', '随机振荡器信号',
            'Williams %R', 'Williams %R信号',
            '布林带突破信号',
            '价格变化率信号',
            # 资金流向指标
            '南向资金(万)',
            # 相对表现
            '相对强度(RS_ratio_%)', '相对强度差值(RS_diff_%)', '跑赢恒指',
            # 基本面数据（只保留PE和PB）
            '基本面评分', '市盈率', '市净率',
            # 新的评分系统字段
            '建仓评分', '建仓级别', '建仓原因',
            '出货评分', '出货级别', '出货原因',
            '止盈', '止损', 'Trailing Stop',
            # 信号指标
            '放量上涨', '缩量回调',
            # TAV评分
            'TAV评分', 'TAV状态',
            # 上个交易日指标（新增）
            '上个交易日RSI', '上个交易日MACD', '上个交易日价格', '上个交易日建仓评分', '上个交易日出货评分', '上个交易日TAV评分',
            # 多周期指标（新增）
            '3日收益率(%)', '3日趋势', '5日收益率(%)', '5日趋势',
            '10日收益率(%)', '10日趋势', '20日收益率(%)', '20日趋势',
            '60日收益率(%)', '60日趋势',
            '3日相对强度(%)', '3日相对强度信号', '5日相对强度(%)', '5日相对强度信号',
            '10日相对强度(%)', '10日相对强度信号', '20日相对强度(%)', '20日相对强度信号',
            '60日相对强度(%)', '60日相对强度信号',
            '多周期趋势评分', '多周期相对强度评分'
        ]

        # 按代码号码排序
        df_report = df_report.sort_values(['代码'], ascending=[True])

        # 确保数值列格式化为两位小数用于显示
        for col in df_report.select_dtypes(include=['float64', 'int64']).columns:
            df_report[col] = df_report[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

        print("\n" + "="*120)
        print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
        print("="*120)
        print(df_report.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

        # 高亮信号（使用新的评分系统）
        strong_distribution_stocks = [r for r in results if r.get('distribution_level') in ['weak', 'strong']]
        strong_buildup_stocks = [r for r in results if r.get('buildup_level') in ['partial', 'strong']]

        if strong_distribution_stocks:
            print("\n🔴 警惕！检测到大户出货信号：")
            for r in strong_distribution_stocks:
                dist_score = r.get('distribution_score', 0)
                dist_level = r.get('distribution_level', 'unknown')
                dist_reasons = r.get('distribution_reasons', '')
                fundamental_score = r.get('fundamental_score', 'N/A')
                print(f"  • {r['name']} | 出货评分={dist_score:.2f} | 出货级别={dist_level} | 原因={dist_reasons} | 基本面评分={fundamental_score}")

        if strong_buildup_stocks:
            print("\n🟢 检测到建仓信号：")
            for r in strong_buildup_stocks:
                build_score = r.get('buildup_score', 0)
                build_level = r.get('buildup_level', 'unknown')
                build_reasons = r.get('buildup_reasons', '')
                rs_disp = (round(r['relative_strength'] * 100, 2) if (r.get('relative_strength') is not None) else None)
                rsd_disp = (round(r['relative_strength_diff'] * 100, 2) if (r.get('relative_strength_diff') is not None) else None)
                fundamental_score = r.get('fundamental_score', 'N/A')
                print(f"  • {r['name']} | 建仓评分={build_score:.2f} | 建仓级别={build_level} | 原因={build_reasons} | RS_ratio={rs_disp}% | RS_diff={rsd_disp}% | 基本面评分={fundamental_score} | 跑赢恒指: {r['outperforms_hsi']}")

        # 检查止盈止损信号
        take_profit_stocks = [r for r in results if r.get('take_profit')]
        stop_loss_stocks = [r for r in results if r.get('stop_loss')]
        trailing_stop_stocks = [r for r in results if r.get('trailing_stop')]

        if take_profit_stocks:
            print("\n💰 触发止盈信号：")
            for r in take_profit_stocks:
                print(f"  • {r['name']} | 建议部分卖出锁定利润")

        if stop_loss_stocks:
            print("\n⛔ 触发止损信号：")
            for r in stop_loss_stocks:
                print(f"  • {r['name']} | 建议全部卖出止损")

        if trailing_stop_stocks:
            print("\n📉 触发ATR Trailing Stop信号：")
            for r in trailing_stop_stocks:
                print(f"  • {r['name']} | 建议部分卖出保护利润")

        # 显示相关新闻信息
        news_file_path = "data/all_stock_news_records.csv"
        if os.path.exists(news_file_path):
            try:
                news_df = pd.read_csv(news_file_path)
                if not news_df.empty:
                    print("\n" + "="*50)
                    print("📰 相关新闻摘要")
                    print("="*50)
                    for _, row in news_df.iterrows():
                        print(f"\n【{row['股票名称']} ({row['股票代码']})】")
                        print(f"时间: {row['新闻时间']}")
                        print(f"标题: {row['新闻标题']}")
                        print(f"内容: {row['简要内容']}")
                else:
                    print("\n⚠️ 新闻文件为空")
            except Exception as e:
                print(f"\n⚠️ 读取新闻数据失败: {e}")
        else:
            print("\nℹ️ 未找到新闻数据文件")

        # 获取当前恒生指数
        current_hsi = "未知"
        if hsi_hist is not None and not hsi_hist.empty:
            current_hsi = hsi_hist['Close'].iloc[-1]
        
        # 计算市场整体指标，为大模型提供更全面的市场状态
        market_metrics = {}

        if results:
            # 计算整体市场情绪指标（使用新的评分系统）
            total_stocks = len(results)
            buildup_stocks_count = sum(1 for r in results if r.get('buildup_level') in ['partial', 'strong'])
            strong_buildup_stocks_count = sum(1 for r in results if r.get('buildup_level') == 'strong')
            distribution_stocks_count = sum(1 for r in results if r.get('distribution_level') in ['weak', 'strong'])
            strong_distribution_stocks_count = sum(1 for r in results if r.get('distribution_level') == 'strong')
            outperforming_stocks_count = sum(1 for r in results if r['outperforms_hsi'])

            # 计算平均建仓和出货评分
            valid_buildup_scores = [r['buildup_score'] for r in results if r.get('buildup_score') is not None]
            avg_buildup_score = sum(valid_buildup_scores) / len(valid_buildup_scores) if valid_buildup_scores else 0

            valid_distribution_scores = [r['distribution_score'] for r in results if r.get('distribution_score') is not None]
            avg_distribution_score = sum(valid_distribution_scores) / len(valid_distribution_scores) if valid_distribution_scores else 0

            # 计算平均相对强度
            valid_rs = [r['relative_strength'] for r in results if r['relative_strength'] is not None]
            avg_relative_strength = sum(valid_rs) / len(valid_rs) if valid_rs else 0

            # 计算平均波动率
            valid_volatility = [r['volatility'] for r in results if r['volatility'] is not None]
            avg_market_volatility = sum(valid_volatility) / len(valid_volatility) if valid_volatility else 0

            # 计算平均成交量变化
            valid_vol_ratio = [r['vol_ratio'] for r in results if r['vol_ratio'] is not None]
            avg_vol_ratio = sum(valid_vol_ratio) / len(valid_vol_ratio) if valid_vol_ratio else 0

            # 计算市场情绪指标（基于新的评分系统）
            market_sentiment = 'neutral'
            strong_signal_ratio = (strong_buildup_stocks_count + strong_distribution_stocks_count) / total_stocks
            if strong_signal_ratio > 0.3:
                market_sentiment = 'active'
            elif strong_signal_ratio < 0.1:
                market_sentiment = 'quiet'

            # 计算资金流向指标
            total_southbound_net = sum(r['southbound'] or 0 for r in results)

            # 计算市场活跃度
            market_activity_level = 'normal'
            if avg_vol_ratio > 1.5:
                market_activity_level = 'high'
            elif avg_vol_ratio < 0.8:
                market_activity_level = 'low'

            market_metrics = {
                'total_stocks': total_stocks,
                'buildup_stocks_count': buildup_stocks_count,
                'strong_buildup_stocks_count': strong_buildup_stocks_count,
                'distribution_stocks_count': distribution_stocks_count,
                'strong_distribution_stocks_count': strong_distribution_stocks_count,
                'outperforming_stocks_count': outperforming_stocks_count,
                'avg_relative_strength': avg_relative_strength,
                'avg_market_volatility': avg_market_volatility,
                'avg_vol_ratio': avg_vol_ratio,
                'avg_buildup_score': avg_buildup_score,
                'avg_distribution_score': avg_distribution_score,
                'market_sentiment': market_sentiment,
                'market_activity_level': market_activity_level,
                'total_southbound_net': total_southbound_net,
                'hsi_current': current_hsi,
                'market_activity_level': 'high' if avg_vol_ratio > 1.5 else 'normal' if avg_vol_ratio > 0.8 else 'low'
            }
        
        # 调用大模型分析股票数据
        try:
            print("\n🤖 正在调用大模型分析股票数据...")
            llm_prompt = build_llm_analysis_prompt(results, run_date, market_metrics)
            llm_analysis = qwen_engine.chat_with_llm(llm_prompt, enable_thinking=False)
            print("✅ 大模型分析完成")
            # 将大模型分析结果打印到屏幕
            if llm_analysis:
                print("\n" + "="*50)
                print("🤖 大模型分析结果:")
                print("="*50)
                print(llm_analysis)
                print("="*50)
        except Exception as e:
            print(f"⚠️ 大模型分析失败: {e}")
            llm_analysis = None

        # 保存 Excel（包含 machine-friendly 原始列 + 展示列）
        try:
            # 创建用于Excel的报告数据框，按常见分类和次序排列
            df_excel = df[[
                # 基本信息
                'name', 'code', 'last_close', 'change_pct',
                # 价格位置
                'price_percentile',
                # 成交量相关
                'vol_ratio', 'vol_z_score', 'turnover_z_score', 'turnover', 'turnover_rate', 'vwap', 'volume_ratio', 'volume_ratio_signal',
                # 波动性指标
                'atr', 'atr_ratio', 'atr_ratio_signal', 'bb_width', 'bb_breakout', 'volatility',
                # 均线偏离
                'ma5_deviation', 'ma10_deviation',
                # 技术指标
                'rsi', 'rsi_roc', 'rsi_divergence', 
                'macd', 'macd_hist', 'macd_hist_roc', 'macd_hist_roc_signal',
                'obv', 
                'cmf', 'cmf_signal', 'cmf_trend_signal',
                'stoch_k', 'stoch_d', 'stoch_signal',
                'williams_r', 'williams_r_signal',
                'bb_breakout_signal',
                'roc_signal',
                # 资金流向指标
                'southbound',
                # 相对表现
                'RS_ratio_%', 'RS_diff_%', 'outperforms_hsi',
                # 基本面数据
                'fundamental_score', 'pe_ratio', 'pb_ratio',
                # 信号指标
                'has_buildup', 'has_distribution', 'strong_volume_up', 'weak_volume_down',
                # TAV评分
                'tav_score', 'tav_status'
            ]]
            
            df_excel.columns = [
                # 基本信息
                '股票名称', '代码', '最新价', '涨跌幅(%)',
                # 价格位置
                '位置(%)',
                # 成交量相关
                '量比', '成交量z-score', '成交额z-score', '成交金额(百万)', '换手率(%)', 'VWAP', '成交量比率', '成交量比率信号',
                # 波动性指标
                'ATR', 'ATR比率', 'ATR比率信号', '布林带宽度(%)', '布林带突破', '波动率(%)',
                # 均线偏离
                '5日均线偏离(%)', '10日均线偏离(%)',
                # 技术指标
                'RSI', 'RSI变化率', 'RSI背离信号',
                'MACD', 'MACD柱状图', 'MACD柱状图变化率', 'MACD柱状图变化率信号',
                'OBV',
                'CMF', 'CMF信号线', 'CMF趋势信号',
                '随机振荡器K', '随机振荡器D', '随机振荡器信号',
                'Williams %R', 'Williams %R信号',
                '布林带突破信号',
                '价格变化率信号',
                # 资金流向指标
                '南向资金(万)',
                # 相对表现
                '相对强度(RS_ratio_%)', '相对强度差值(RS_diff_%)', '跑赢恒指',
                # 基本面数据
                '基本面评分', '市盈率', '市净率',
                # 信号指标
                '建仓信号', '出货信号', '放量上涨', '缩量回调',
                # TAV评分
                'TAV评分', 'TAV状态'
            ]
            
            # 排序与邮件报告一致
            df_excel = df_excel.sort_values(['出货信号', '建仓信号'], ascending=[True, False])
            
            # 确保数值列格式化为两位小数用于显示
            for col in df_excel.select_dtypes(include=['float64', 'int64']).columns:
                df_excel[col] = df_excel[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)
            
            # 保存到Excel文件
            try:
                df_excel.to_excel("hk_smart_money_report.xlsx", index=False)
                print("\n💾 报告已保存: hk_smart_money_report.xlsx")
            except Exception as e:
                print(f"⚠️  Excel保存失败: {e}")
        except Exception as e:
            print(f"⚠️  Excel保存失败: {e}")

        # 发送邮件（将表格分段为多个 HTML 表格并包含说明）
        def send_email_with_report(df_report, to):
            smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
            smtp_user = os.environ.get("YAHOO_EMAIL")
            smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
            sender_email = smtp_user

            if not smtp_user or not smtp_pass:
                print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
                return False

            if isinstance(to, str):
                to = [to]

            subject = "港股主力资金追踪报告"

            # 获取当前时间用于报告生成时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取香港时间（UTC+8）
            from datetime import timedelta
            hk_time = datetime.now() + timedelta(hours=8)
            hk_time_str = hk_time.strftime("%Y-%m-%d %H:%M")
            
            text = "港股主力资金追踪报告\n\n"
            html = "<html><body><h2>港股主力资金追踪报告</h2>"
            
            # 添加报告生成时间、分析日期和基本信息
            html += f"<p><strong>报告生成时间:</strong> {current_time}</p>"
            if run_date:
                html += f"<p><strong>分析日期:</strong> {run_date}</p>"
                text += f"分析日期: {run_date}\n"
            else:
                html += f"<p><strong>分析日期:</strong> {hk_time_str} (香港时间)</p>"
                text += f"分析日期: {hk_time_str} (香港时间)\n"
            text += f"报告生成时间: {current_time}\n"
            html += f"<p><strong>分析 {len(WATCHLIST)} 只股票</strong> | <strong>窗口:</strong> {DAYS_ANALYSIS} 日</p>"

            # 添加表格（每 5 行分一页，分类行放在字段名称上面）
            for i in range(0, len(df_report), 5):
                # 获取数据块
                chunk = df_report.iloc[i:i+5]
                
                # 创建包含分类信息和字段名的完整表格
                # 分类行
                category_row = [
                    # 基本信息 (4列)
                    '基本信息', '', '', '',
                    # 价格位置 (1列)
                    '价格位置',
                    # 成交量相关 (8列)
                    '成交量相关', '', '', '', '', '', '', '',
                    # 波动性指标 (6列)
                    '波动性指标', '', '', '', '', '',
                    # 均线偏离 (2列)
                    '', '',
                    # 技术指标 (18列)
                    '技术指标', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                    # 资金流向指标 (1列)
                    '资金流向指标',
                    # 相对表现 (3列)
                    '相对表现', '', '',
                    # 基本面数据 (7列)
                    '基本面数据', '', '', '', '', '', '',
                    # 信号指标 (4列)
                    '', '信号指标', '', '',
                    # TAV评分 (2列)
                    'TAV评分', ''
                ]
                
                # 将分类行作为第一行，字段名作为第二行，数据作为后续行
                all_data = [category_row] + [chunk.columns.tolist()] + chunk.values.tolist()
                
                # 创建临时DataFrame用于显示，但需要正确处理表头
                temp_df = pd.DataFrame(all_data[2:])  # 数据部分
                temp_df.columns = all_data[1]  # 使用字段名作为列名
                
                # 生成HTML表格
                html_table = temp_df.to_html(index=False, escape=False)
                
                # 在HTML表格中插入分类行，将分类信息插入到<th>标签中
                # 首先提取表头部分（字段名称行）
                field_names = chunk.columns.tolist()
                
                # 手动构建HTML表格以添加分类行
                html += '<table border="1" class="dataframe">\n'
                html += '  <thead>\n'
                # 添加分类行
                html += '    <tr>\n'
                for cat in category_row:
                    html += f'      <th>{cat}</th>\n'
                html += '    </tr>\n'
                # 添加字段名称行
                html += '    <tr>\n'
                for field in field_names:
                    html += f'      <th>{field}</th>\n'
                html += '    </tr>\n'
                html += '  </thead>\n'
                html += '  <tbody>\n'
                # 添加数据行
                for idx, row in chunk.iterrows():
                    html += '    <tr>\n'
                    for i, (col_name, cell) in enumerate(row.items()):
                        if pd.isna(cell) or cell is None:
                            html += f'      <td>None</td>\n'
                        else:
                            # 为上个交易日指标添加变化箭头
                            cell_display = str(cell)
                            if col_name == '上个交易日RSI' and pd.notna(row.get('RSI')):
                                arrow = get_score_change_arrow(row['RSI'], cell)
                                cell_display = f"{arrow} {cell}"
                            elif col_name == '上个交易日MACD' and pd.notna(row.get('MACD')):
                                arrow = get_score_change_arrow(row['MACD'], cell)
                                cell_display = f"{arrow} {cell}"
                            elif col_name == '上个交易日价格' and pd.notna(row.get('最新价')):
                                arrow = get_price_change_arrow(row['最新价'], cell)
                                cell_display = f"{arrow} {cell}"
                            elif col_name == '上个交易日建仓评分' and pd.notna(row.get('建仓评分')):
                                arrow = get_score_change_arrow(row['建仓评分'], cell)
                                cell_display = f"{arrow} {cell}"
                            elif col_name == '上个交易日出货评分' and pd.notna(row.get('出货评分')):
                                arrow = get_score_change_arrow(row['出货评分'], cell)
                                cell_display = f"{arrow} {cell}"
                            elif col_name == '上个交易日TAV评分' and pd.notna(row.get('TAV评分')):
                                arrow = get_score_change_arrow(row['TAV评分'], cell)
                                cell_display = f"{arrow} {cell}"
                            html += f'      <td>{cell_display}</td>\n'
                    html += '    </tr>\n'
                html += '  </tbody>\n'
                html += '</table>\n'

            

            # 添加大模型分析结果
            if llm_analysis:
                html += "<h3>🤖 大模型分析结果：</h3>"
                html += "<div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px;'>"
                # 使用markdown到HTML转换函数
                llm_analysis_html = markdown_to_html(llm_analysis)
                html += f"<p>{llm_analysis_html}</p>"
                html += "</div>"
            else:
                html += "<h3>🤖 大模型分析结果：</h3>"
                html += "<p>大模型分析暂不可用</p>"

            FULL_INDICATOR_HTML = """
            <h3>📋 指标说明</h3>
            <div style="font-size:0.9em; line-height:1.4;">
            <h4>基础信息</h4>
            <ul>
              <li><b>最新价</b>：股票当前最新成交价格（港元）。若当日存在盘中变动，建议结合成交量与盘口观察。</li>
              <li><b>涨跌幅(%)</b>：按 (最新价 - 前收) / 前收 计算并乘以100表示百分比。</li>
            </ul>
            
            <h4>价格位置</h4>
            <ul>
              <li><b>位置(%)</b>：当前价格在最近 PRICE_WINDOW（默认 60 日）内的百分位位置。</li>
              <li>计算：(当前价 - 最近N日最低) / (最高 - 最低) * 100，取 [0, 100]。</li>
              <li>含义：接近 0 表示处于历史窗口低位，接近 100 表示高位。</li>
              <li>用途：判断是否处于\"相对低位\"或\"高位\"，用于建仓/出货信号的价格条件。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>位置 < 30%：相对低位，可能有支撑</li>
                  <li>位置 > 70%：相对高位，可能有阻力</li>
                  <li>位置在 30%-70%：震荡区间</li>
                </ul>
              </li>
            </ul>
            
            <h4>成交量相关</h4>
            <ul>
              <li><b>量比 (Vol_Ratio)</b>：当日成交量 / 20 日平均成交量（VOL_WINDOW）。</li>
              <li>含义：衡量当日成交是否显著放大。</li>
              <li>建议：放量配合价格运动（如放量上涨或放量下跌）比单纯放量更具信号含义。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>Vol_Ratio > 1.5：显著放量</li>
                  <li>Vol_Ratio < 0.5：显著缩量</li>
                  <li>Vol_Ratio 在 0.5-1.5：正常成交量</li>
                </ul>
              </li>
              
              <li><b>成交量z-score</b>：成交量相对于20日均值的标准差倍数。</li>
              <li>含义：衡量成交量异常程度，比量比更考虑波动性。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>Vol_Z_Score > 2.0：极端放量</li>
                  <li>Vol_Z_Score > 1.5：显著放量</li>
                  <li>Vol_Z_Score < -1.5：显著缩量</li>
                </ul>
              </li>
              
              <li><b>成交额z-score</b>：成交额相对于20日均值的标准差倍数。</li>
              <li>含义：衡量成交额异常程度，考虑了价格和成交量的综合影响。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>Turnover_Z_Score > 2.0：极端放量</li>
                  <li>Turnover_Z_Score > 1.5：显著放量</li>
                  <li>Turnover_Z_Score < -1.5：显著缩量</li>
                </ul>
              </li>
              
              <li><b>成交金额(百万)</b>：当日成交金额，单位为百万港元（近似计算：最新价 * 成交量 / 1e6）。</li>
              <li><b>成交量比率</b>：
                <ul>
                  <li>计算：当日成交量 / 30日平均成交量</li>
                  <li>含义：衡量当日成交量相对于历史平均成交量的倍数</li>
                  <li>评估方法：
                    <ul>
                      <li>成交量比率 > 2.0：显著放量</li>
                      <li>成交量比率 < 0.5：显著缩量</li>
                      <li>成交量比率在0.5-2.0之间：正常成交量</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>成交量比率信号</b>：
                <ul>
                  <li>含义：基于成交量比率的放量信号</li>
                  <li>条件：当成交量比率 > 1.5时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：出现显著放量，可能预示价格变动</li>
                      <li>False：成交量正常或缩量</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>换手率(%)</b>：当日成交量占总股本的比例。</li>
              <li>含义：衡量股票的流动性，换手率高的股票通常流动性更好。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>换手率 > 5%：高流动性</li>
                  <li>换手率 < 1%：低流动性</li>
                </ul>
              </li>
            </ul>
            
            <h4>价格指标</h4>
            <ul>
              <li><b>VWAP（成交量加权平均价格）</b>：
                <ul>
                  <li>计算：(High+Low+Close)/3 * Volume 的加权平均</li>
                  <li>含义：衡量当日资金的平均成本</li>
                  <li>评估方法：
                    <ul>
                      <li>收盘价 > VWAP：资金在高位买入</li>
                      <li>收盘价 < VWAP：资金在低位卖出</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>ATR（平均真实波幅）</b>：
                <ul>
                  <li>计算：14日真实波幅的平均值</li>
                  <li>含义：衡量市场波动性</li>
                  <li>评估方法：
                    <ul>
                      <li>ATR 升高：波动加剧，可能有趋势行情</li>
                      <li>ATR 降低：波动收敛，可能有盘整行情</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>ATR比率</b>：
                <ul>
                  <li>计算：ATR / ATR的移动平均值（默认10日）</li>
                  <li>含义：衡量当前波动性相对于历史平均水平的程度</li>
                  <li>评估方法：
                    <ul>
                      <li>ATR比率 > 1：当前波动性高于历史平均水平</li>
                      <li>ATR比率 < 1：当前波动性低于历史平均水平</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>ATR比率信号</b>：
                <ul>
                  <li>含义：基于ATR比率的波动性信号</li>
                  <li>条件：当ATR比率 > 1.5时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：波动性显著放大，可能预示趋势行情</li>
                      <li>False：波动性正常或收敛</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>ADX（平均趋向指数）</b>：
                <ul>
                  <li>计算：基于+DI和-DI计算的趋势强度指标</li>
                  <li>含义：衡量趋势强度</li>
                  <li>评估方法：
                    <ul>
                      <li>ADX > 25：趋势行情</li>
                      <li>ADX < 20：盘整行情</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>布林带宽度(%)</b>：
                <ul>
                  <li>计算：(布林带上轨-布林带下轨)/布林带中轨 * 100</li>
                  <li>含义：衡量布林带的收窄或扩张程度</li>
                  <li>评估方法：
                    <ul>
                      <li>宽度低：波动收敛，可能预示后续波动扩张</li>
                      <li>宽度高：波动扩张</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>布林带突破</b>：
                <ul>
                  <li>计算：(收盘价 - 布林带下轨) / (布林带上轨 - 布林带下轨)</li>
                  <li>含义：衡量价格相对于布林带的位置，判断是否突破布林带边界</li>
                  <li>评估方法：
                    <ul>
                      <li>布林带突破 > 1：价格突破布林带上轨</li>
                      <li>布林带突破 < 0：价格跌破布林带下轨</li>
                      <li>布林带突破在0-1之间：价格在布林带范围内</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>布林带突破信号</b>：
                <ul>
                  <li>含义：基于布林带突破的突破信号</li>
                  <li>条件：当布林带突破 > 1.0 或 布林带突破 < 0.0 时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：价格突破布林带边界，可能预示趋势延续或反转</li>
                      <li>False：价格在布林带范围内</li>
                    </ul>
                  </li>
                </ul>
              </li>
            </ul>
            
            <h4>均线偏离</h4>
            <ul>
              <li><b>5日/10日均线偏离(%)</b>：最新价相对于短期均线的偏离百分比（正值表示价高于均线）。</li>
              <li>用途：短期动力判断；但对高波动或宽幅震荡个股需谨慎解读。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>偏离 > 5%：显著偏离均线</li>
                  <li>偏离在 -5% 到 5%：正常范围</li>
                  <li>偏离 < -5%：显著低于均线</li>
                </ul>
              </li>
            </ul>
            
            <h4>技术指标</h4>
            <ul>
              <li><b>RSI（Wilder 平滑）</b>：
                <ul>
                  <li>计算：基于 14 日 Wilder 指数平滑的涨跌幅比例，结果在 0-100。</li>
                  <li>含义：常用于判断超买/超卖（例如 RSI > 70 可能偏超买，RSI < 30 可能偏超卖）。</li>
                  <li>注意：单独使用 RSI 可能产生误导，建议与成交量和趋势指标结合。</li>
                  <li><b>评估方法</b>：
                    <ul>
                      <li>RSI > 70：超买</li>
                      <li>RSI < 30：超卖</li>
                      <li>RSI 在 30-70：正常</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>MACD</b>：
                <ul>
                  <li>计算：EMA12 - EMA26（MACD 线），并计算 9 日 EMA 作为 MACD Signal。</li>
                  <li>含义：衡量中短期动量，MACD 线上穿信号线通常被视为动能改善（反之则疲弱）。</li>
                  <li>注意：对剧烈震荡或极端股价数据（如停牌后复牌）可能失真。</li>
                  <li><b>评估方法</b>：
                    <ul>
                      <li>MACD > MACD_Signal：动能增强</li>
                      <li>MACD < MACD_Signal：动能减弱</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>MACD柱状图</b>：
                <ul>
                  <li>计算：MACD线 - MACD信号线</li>
                  <li>含义：衡量MACD线与信号线之间的差距，反映动量的强弱</li>
                  <li>评估方法：
                    <ul>
                      <li>MACD柱状图 > 0：多头动能占优</li>
                      <li>MACD柱状图 < 0：空头动能占优</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>MACD柱状图变化率</b>：
                <ul>
                  <li>计算：(当前MACD柱状图 - 前一期MACD柱状图) / 前一期MACD柱状图</li>
                  <li>含义：衡量MACD柱状图的变化速度，反映动量变化的快慢</li>
                  <li>评估方法：
                    <ul>
                      <li>MACD柱状图变化率 > 0：动量加速</li>
                      <li>MACD柱状图变化率 < 0：动量减速</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>MACD柱状图变化率信号</b>：
                <ul>
                  <li>含义：基于MACD柱状图变化率的信号</li>
                  <li>条件：当MACD柱状图变化率 > 0.1时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：动量显著加速，可能预示趋势延续</li>
                      <li>False：动量未显著加速</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>RSI变化率</b>：
                <ul>
                  <li>计算：(当前RSI - 前一期RSI) / 前一期RSI</li>
                  <li>含义：衡量RSI的变化速度，反映超买超卖状态的变化快慢</li>
                  <li>评估方法：
                    <ul>
                      <li>RSI变化率 > 0：超买状态加剧或超卖状态缓解</li>
                      <li>RSI变化率 < 0：超卖状态加剧或超买状态缓解</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>RSI背离信号</b>：
                <ul>
                  <li>含义：价格与RSI指标之间的背离信号</li>
                  <li>条件：当RSI变化率 < 0 且价格涨幅 > 0时为True，表示顶背离；当RSI变化率 > 0 且价格跌幅 > 0时为True，表示底背离</li>
                  <li>评估方法：
                    <ul>
                      <li>True：出现价格与RSI背离，可能预示趋势反转</li>
                      <li>False：价格与RSI同向运动</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>随机振荡器K</b>：
                <ul>
                  <li>计算：100 * (收盘价 - 最近N日最低价) / (最近N日最高价 - 最近N日最低价)，默认N=14</li>
                  <li>含义：衡量收盘价在最近N日价格区间中的相对位置</li>
                  <li>评估方法：
                    <ul>
                      <li>随机振荡器K > 80：超买区域</li>
                      <li>随机振荡器K < 20：超卖区域</li>
                      <li>随机振荡器K在20-80之间：正常区域</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>随机振荡器D</b>：
                <ul>
                  <li>计算：随机振荡器K的移动平均线（默认3日）</li>
                  <li>含义：随机振荡器K的平滑线，用于识别K值的趋势</li>
                  <li>评估方法：
                    <ul>
                      <li>随机振荡器D > 80：超买区域</li>
                      <li>随机振荡器D < 20：超卖区域</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>随机振荡器信号</b>：
                <ul>
                  <li>含义：基于随机振荡器的超买超卖信号</li>
                  <li>条件：当随机振荡器K < 20 或 随机振荡器K > 80时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：进入超买或超卖区域，可能预示价格反转</li>
                      <li>False：未进入超买或超卖区域</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>Williams %R</b>：
                <ul>
                  <li>计算：(最近N日最高价 - 收盘价) / (最近N日最高价 - 最近N日最低价) * -100，默认N=14</li>
                  <li>含义：衡量收盘价在最近N日价格区间中的相对位置，与随机振荡器相反</li>
                  <li>评估方法：
                    <ul>
                      <li>Williams %R > -20：超买区域</li>
                      <li>Williams %R < -80：超卖区域</li>
                      <li>Williams %R在-80到-20之间：正常区域</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>Williams %R信号</b>：
                <ul>
                  <li>含义：基于Williams %R的超买超卖信号</li>
                  <li>条件：当Williams %R < -80 或 Williams %R > -20时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：进入超买或超卖区域，可能预示价格反转</li>
                      <li>False：未进入超买或超卖区域</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>波动率(%)</b>：基于 20 日收益率样本的样本标准差年化后以百分比表示（std * sqrt(252)）。</li>
              <li>含义：衡量历史波动幅度，用于风险评估和头寸大小控制。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>波动率 > 30%：高波动</li>
                  <li>波动率 < 15%：低波动</li>
                </ul>
              </li>
              
              <li><b>OBV（On-Balance Volume）</b>：按照日涨跌累计成交量的方向（涨则加，跌则减）来累计。</li>
              <li>含义：尝试用成交量的方向性累积来辅助判断资金是否在积累/分配。</li>
              <li>注意：OBV 是累积序列，适合观察中长期趋势而非短期信号。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>OBV 上升：资金流入</li>
                  <li>OBV 下降：资金流出</li>
                </ul>
              </li>
              
              <li><b>CMF（Chaikin Money Flow）</b>：
                <ul>
                  <li>计算：20日资金流量的累积</li>
                  <li>含义：衡量资金流向</li>
                  <li>评估方法：
                    <ul>
                      <li>CMF > 0.05：资金流入</li>
                      <li>CMF < -0.05：资金流出</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>CMF信号线</b>：
                <ul>
                  <li>计算：CMF的移动平均线（默认5日）</li>
                  <li>含义：CMF的平滑线，用于识别CMF的趋势</li>
                  <li>评估方法：
                    <ul>
                      <li>CMF > CMF信号线：资金流入加速</li>
                      <li>CMF < CMF信号线：资金流出加速</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>CMF趋势信号</b>：
                <ul>
                  <li>含义：基于CMF与CMF信号线关系的趋势信号</li>
                  <li>条件：当CMF > CMF信号线时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：资金流入趋势</li>
                      <li>False：资金流出趋势或趋势不明显</li>
                    </ul>
                  </li>
                </ul>
              </li>
            </ul>
            
            <h4>相对表现 / 跑赢恒指（用于衡量个股相对大盘的表现）</h4>
            <ul>
              <li><b>相对强度 (RS_ratio)</b>：
                <ul>
                  <li>计算：RS_ratio = (1 + stock_ret) / (1 + hsi_ret) - 1</li>
                  <li>含义：基于复合收益（即把两个收益都视为复利因子）来度量个股相对恒指的表现。</li>
                  <li>RS_ratio > 0 表示个股在该区间的复合收益率高于恒指；RS_ratio < 0 则表示跑输。</li>
                  <li>优点：在收益率接近 -1 或波动较大时，更稳健地反映\"相对复合回报\"。</li>
                  <li>报告显示：以百分比列 RS_ratio_% 呈现（例如 5 表示 +5%）。</li>
                  <li><b>评估方法</b>：
                    <ul>
                      <li>RS_ratio > 5%：显著跑赢</li>
                      <li>RS_ratio > 0%：跑赢</li>
                      <li>RS_ratio < 0%：跑输</li>
                    </ul>
                  </li>
                </ul>
              </li>
              <li><b>相对强度差值 (RS_diff)</b>：
                <ul>
                  <li>计算：RS_diff = stock_ret - hsi_ret（直接的收益差值）。</li>
                  <li>含义：更直观，表示绝对收益的差额（例如股票涨 6%，恒指涨 2%，则 RS_diff = 4%）。</li>
                  <li>报告显示：以百分比列 RS_diff_% 呈现。</li>
                  <li><b>评估方法</b>：
                    <ul>
                      <li>RS_diff > 3%：显著跑赢</li>
                      <li>RS_diff > 0%：跑赢</li>
                      <li>RS_diff < 0%：跑输</li>
                    </ul>
                  </li>
                </ul>
              </li>
              <li><b>价格变化率信号</b>：
                <ul>
                  <li>含义：基于价格变化率的动量信号</li>
                  <li>条件：当12日价格变化率ROC > 0.05时为True，否则为False</li>
                  <li>评估方法：
                    <ul>
                      <li>True：价格在中长期呈现上涨趋势，动量为正</li>
                      <li>False：价格在中长期未呈现明显上涨趋势</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>跑赢恒指 (outperforms_hsi)</b>：
                <ul>
                  <li>脚本支持三种语义：
                    <ol>
                      <li>要求股票为正收益并且收益 > 恒指（默认，保守）：OUTPERFORMS_REQUIRE_POSITIVE = True</li>
                      <li>仅比较收益差值（无需股票为正）：OUTPERFORMS_REQUIRE_POSITIVE = False</li>
                      <li>使用 RS_ratio（以复合收益判断）：OUTPERFORMS_USE_RS = True</li>
                    </ol>
                  </li>
                </ul>
              </li>
            </ul>
            
            <h4>资金流向</h4>
            <ul>
              <li><b>南向资金(万)</b>：通过沪港通/深港通流入该股的资金净额。</li>
              <li>数据来源：使用 akshare 的 stock_hk_ggt_components_em 获取\"净买入\"字段，脚本假设原始单位为\"元\"并除以 SOUTHBOUND_UNIT_CONVERSION 转为\"万\"。</li>
              <li>用途：当南向资金显著流入时，通常被解读为北向/南向机构资金的买入兴趣；显著流出则表示机构抛售或撤出。</li>
              <li><b>评估方法</b>：
                <ul>
                  <li>南向资金 > 3000万：显著流入</li>
                  <li>南向资金 > 1000万：流入</li>
                  <li>南向资金 < -3000万：显著流出</li>
                  <li>南向资金 < -1000万：流出</li>
                </ul>
              </li>
              <li>限制与谨慎：
                <ul>
                  <li>ak 数据延迟或字段命名可能变化（脚本已做基本容错，但仍需关注源数据格式）。</li>
                  <li>单日南向资金异常需结合量价关系与连续性判断，避免被一次性大额交易误导。</li>
                </ul>
              </li>
            </ul>
            
            <h4>信号定义（本脚本采用的简化规则）</h4>
            <ul>
              <li><b>建仓信号（Buildup）</b>：
                <ul>
                  <li>条件：位置 < PRICE_LOW_PCT（低位） AND 量比 > VOL_RATIO_BUILDUP（成交放大） AND 南向资金净流入 > SOUTHBOUND_THRESHOLD（万） AND (MACD线上穿信号线 OR RSI<30 OR OBV>0)。</li>
                  <li>连续性：要求连续或累计达到 BUILDUP_MIN_DAYS 才被标注为确认（避免孤立样本）。</li>
                  <li>语义：在低位出现放量且机构买入力度强时，可能代表主力建仓或底部吸筹。</li>
                </ul>
              </li>
              <li><b>出货信号（Distribution）</b>：
                <ul>
                  <li>条件：位置 > PRICE_HIGH_PCT（高位） AND 量比 > VOL_RATIO_DISTRIBUTION（剧烈放量） AND 南向资金净流出 < -SOUTHBOUND_THRESHOLD（万） AND 当日收盘下行（相对前一日收盘价或开盘价） AND (MACD线下穿信号线 OR RSI>70 OR OBV<0)。</li>
                  <li>连续性：要求连续达到 DISTRIBUTION_MIN_DAYS 才标注为确认。</li>
                  <li>语义：高位放量且机构撤出，伴随价格下行，可能代表主力在高位分批出货/派发。</li>
                </ul>
              </li>
              <li><b>重要提醒</b>：
                <ul>
                  <li>本脚本规则为经验性启发式筛选，不构成投资建议。建议将信号作为筛选或复核工具，结合持仓风险管理、基本面与订单簿/资金面深度判断。</li>
                  <li>对于停牌、派息、拆股或其他公司事件，指标需特殊处理；脚本未一一覆盖这些事件。</li>
                </ul>
              </li>
            </ul>
            <h4>TAV评分系统</h4>
            <ul>
              <li><b>TAV评分</b>：
                <ul>
                  <li>含义：趋势-加速度-成交量三维分析的综合评分（0-100分）</li>
                  <li>计算：基于价格趋势、趋势加速度和成交量变化的综合分析</li>
                  <li>评估方法：
                    <ul>
                      <li>TAV评分 > 70：强势状态，技术面强劲</li>
                      <li>TAV评分 30-70：中性状态，技术面平稳</li>
                      <li>TAV评分 < 30：弱势状态，技术面疲弱</li>
                    </ul>
                  </li>
                  <li>用途：用于评估股票技术面的整体强度，辅助判断建仓/出货信号的可靠性</li>
                </ul>
              </li>
              <li><b>TAV状态</b>：
                <ul>
                  <li>含义：基于TAV评分生成的文字描述状态</li>
                  <li>可能状态：强势、中性、弱势、无TAV等</li>
                  <li>用途：提供直观的技术面状态描述，便于快速理解股票当前的技术健康状况</li>
                </ul>
              </li>
            </ul>
            
            <h4>基本面指标</h4>
            <ul>
              <li><b>基本面评分</b>：
                <ul>
                  <li>含义：综合估值、盈利能力、成长性、财务健康和股息率的评分（0-100分）</li>
                  <li>计算：基于PE、PB、ROE、净利率、营收增长、利润增长、负债率、流动比率、股息率等指标加权计算</li>
                  <li>评估方法：
                    <ul>
                      <li>基本面评分 > 60：基本面优秀，投资价值高</li>
                      <li>基本面评分 30-60：基本面一般，中性评价</li>
                      <li>基本面评分 < 30：基本面较差，投资风险高</li>
                    </ul>
                  </li>
                  <li>用途：用于评估股票的内在价值和长期投资价值，辅助技术分析决策</li>
                </ul>
              </li>
              
              <li><b>市盈率(PE)</b>：
                <ul>
                  <li>计算：股价 / 每股收益</li>
                  <li>含义：衡量股价相对于每股收益的倍数，反映投资者对公司未来盈利能力的预期</li>
                  <li>评估方法：
                    <ul>
                      <li>PE < 10：低估，投资价值较高</li>
                      <li>PE 10-15：合理估值</li>
                      <li>PE 15-25：偏高估值</li>
                      <li>PE > 25：高估值，投资风险较高</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>市净率(PB)</b>：
                <ul>
                  <li>计算：股价 / 每股净资产</li>
                  <li>含义：衡量股价相对于每股净资产的倍数，反映市场对公司资产价值的评估</li>
                  <li>评估方法：
                    <ul>
                      <li>PB < 1：股价低于净资产，可能被低估</li>
                      <li>PB 1-1.5：合理估值</li>
                      <li>PB 1.5-3：偏高估值</li>
                      <li>PB > 3：高估值，投资风险较高</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>ROE(%)</b>：
                <ul>
                  <li>计算：净利润 / 股东权益 * 100%</li>
                  <li>含义：衡量公司利用股东资本创造利润的效率，反映公司的盈利能力</li>
                  <li>评估方法：
                    <ul>
                      <li>ROE > 15%：高盈利能力</li>
                      <li>ROE 10-15%：良好盈利能力</li>
                      <li>ROE 5-10%：一般盈利能力</li>
                      <li>ROE < 5%：低盈利能力</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>股息率(%)</b>：
                <ul>
                  <li>计算：年度每股股息 / 股价 * 100%</li>
                  <li>含义：衡量投资回报中来自股息的比例，反映公司的分红政策和股东回报</li>
                  <li>评估方法：
                    <ul>
                      <li>股息率 > 5%：高股息收益，适合价值投资</li>
                      <li>股息率 3-5%：良好股息收益</li>
                      <li>股息率 1-3%：一般股息收益</li>
                      <li>股息率 < 1%：低股息收益</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>营收增长(%)</b>：
                <ul>
                  <li>计算：(本期营收 - 上期营收) / 上期营收 * 100%</li>
                  <li>含义：衡量公司业务规模的扩张速度，反映公司的成长性</li>
                  <li>评估方法：
                    <ul>
                      <li>营收增长 > 20%：高增长</li>
                      <li>营收增长 10-20%：良好增长</li>
                      <li>营收增长 0-10%：低速增长</li>
                      <li>营收增长 < 0%：负增长</li>
                    </ul>
                  </li>
                </ul>
              </li>
              
              <li><b>利润增长(%)</b>：
                <ul>
                  <li>计算：(本期净利润 - 上期净利润) / 上期净利润 * 100%</li>
                  <li>含义：衡量公司盈利能力的增长速度，反映公司的盈利质量</li>
                  <li>评估方法：
                    <ul>
                      <li>利润增长 > 20%：高增长</li>
                      <li>利润增长 10-20%：良好增长</li>
                      <li>利润增长 0-10%：低速增长</li>
                      <li>利润增长 < 0%：负增长</li>
                    </ul>
                  </li>
                </ul>
              </li>
            </ul>
            
            <h4>其他说明与实践建议</h4>
            <ul>
              <li>时间窗口与阈值（如 PRICE_WINDOW、VOL_WINDOW、阈值等）可根据策略偏好调整。更短窗口更灵敏但噪声更多，反之亦然。</li>
              <li>建议：把信号与多因子（行业动量、估值、持仓集中度）结合，避免单一信号操作。</li>
              <li>数据来源与一致性：本脚本结合 yfinance（行情）与 akshare（南向资金），两者更新频率与字段定义可能不同，使用时请确认数据来源的可用性与一致性。</li>
            </ul>
            <p>注：以上为启发式规则，非交易建议。请结合基本面、盘口、资金面与风险管理。</p>
            </div>
            """.format(unit=int(SOUTHBOUND_UNIT_CONVERSION),
                       low=int(PRICE_LOW_PCT),
                       high=int(PRICE_HIGH_PCT),
                       vr_build=VOL_RATIO_BUILDUP,
                       vr_dist=VOL_RATIO_DISTRIBUTION,
                       sb=int(SOUTHBOUND_THRESHOLD),
                       bd=BUILDUP_MIN_DAYS,
                       dd=DISTRIBUTION_MIN_DAYS)

            html += FULL_INDICATOR_HTML

            # 添加相关新闻信息到邮件末尾
            news_file_path = "data/all_stock_news_records.csv"
            if os.path.exists(news_file_path):
                try:
                    news_df = pd.read_csv(news_file_path)
                    # 只保留WATCHLIST中的股票新闻
                    watchlist_codes = list(WATCHLIST.keys())
                    news_df = news_df[news_df['股票代码'].isin(watchlist_codes)]
                    
                    if not news_df.empty:
                        html += "<h3>📰 相关新闻摘要</h3>"
                        html += "<div style='background-color: #f9f9f9; padding: 15px; border-radius: 5px;'>"
                        
                        # 按股票分组显示新闻
                        for stock_name in news_df['股票名称'].unique():
                            stock_news = news_df[news_df['股票名称'] == stock_name]
                            html += f"<h4>{stock_name} ({stock_news.iloc[0]['股票代码']})</h4>"
                            html += "<ul>"
                            for _, row in stock_news.iterrows():
                                html += f"<li><strong>{row['新闻时间']}</strong>: {row['新闻标题']}<br/>{row['简要内容']}</li>"
                            html += "</ul>"
                        
                        html += "</div>"
                except Exception as e:
                    html += f"<p>⚠️ 读取新闻数据失败: {e}</p>"
            else:
                html += "<h3>ℹ️ 未找到新闻数据文件</h3>"

            html += "</body></html>"

            msg = MIMEMultipart("mixed")
            msg['From'] = f'<{sender_email}>'
            msg['To'] = ", ".join(to)
            msg['Subject'] = subject

            body = MIMEMultipart("alternative")
            body.attach(MIMEText(text, "plain", "utf-8"))
            body.attach(MIMEText(html, "html", "utf-8"))
            msg.attach(body)

            # 附件图表
            if os.path.exists(CHART_DIR):
                for filename in os.listdir(CHART_DIR):
                    if filename.endswith(".png"):
                        with open(os.path.join(CHART_DIR, filename), "rb") as f:
                            part = MIMEBase('image', 'png')
                            part.set_payload(f.read())
                        encoders.encode_base64(part)
                        # 使用更安全的文件名编码方式
                        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                        part.add_header('Content-Type', 'image/png')
                        msg.attach(part)

            # 根据SMTP服务器类型选择合适的端口和连接方式
            if "163.com" in smtp_server:
                # 163邮箱使用SSL连接，端口465
                smtp_port = 465
                use_ssl = True
            elif "gmail.com" in smtp_server:
                # Gmail使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False
            else:
                # 默认使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False

            # 发送邮件（增加重试机制）
            for attempt in range(3):
                try:
                    if use_ssl:
                        # 使用SSL连接
                        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, to, msg.as_string())
                        server.quit()
                    else:
                        # 使用TLS连接
                        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                        server.starttls()
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, to, msg.as_string())
                        server.quit()
                    
                    print("✅ 邮件发送成功")
                    return True
                except Exception as e:
                    print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试，等待后重试
                        time.sleep(5)
            
            print("❌ 发送邮件失败，已重试3次")
            return False

        recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        recipients = [r.strip() for r in recipient_env.split(',')] if ',' in recipient_env else [recipient_env]
        print("📧 发送邮件到:", ", ".join(recipients))
        send_email_with_report(df_report, recipients)

    print(f"\n✅ 分析完成！图表保存至: {CHART_DIR}/")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='港股主力资金追踪器')
    parser.add_argument('--date', type=str, help='运行日期 (YYYY-MM-DD 格式)')
    args = parser.parse_args()
    
    # 调用主函数
    main(args.date)
