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
- 南向资金（ak 返回）会被缓存并转换为“万”（可调整 SOUTHBOUND_UNIT_CONVERSION）。
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

warnings.filterwarnings("ignore")
os.environ['MPLBACKEND'] = 'Agg'

import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 用户设置区（所有重要阈值集中于此）
# ==============================
WATCHLIST = {
    "2800.HK": "盈富基金",
    "3968.HK": "招商银行",
    "0939.HK": "建设银行",
    "1398.HK": "工商银行",
    "1288.HK": "农业银行",
    "0005.HK": "汇丰银行",
    "6682.HK": "第四范式",
    "1347.HK": "华虹半导体",
    "0981.HK": "中芯国际",
    "0388.HK": "香港交易所",
    "0700.HK": "腾讯控股",
    "9988.HK": "阿里巴巴-SW",
    "3690.HK": "美团-W",
    "1810.HK": "小米集团-W",
    "9618.HK": "京东集团-SW",
    "9660.HK": "地平线机器人",
    "2533.HK": "黑芝麻智能",
}

# 窗口与样本
DAYS_ANALYSIS = 12
VOL_WINDOW = 20
PRICE_WINDOW = 60
BUILDUP_MIN_DAYS = 3
DISTRIBUTION_MIN_DAYS = 2

# 阈值（可调）
PRICE_LOW_PCT = 30.0   # 价格百分位低于该值视为“低位”
PRICE_HIGH_PCT = 70.0  # 高于该值视为“高位”
VOL_RATIO_BUILDUP = 1.5
VOL_RATIO_DISTRIBUTION = 2.5

# 南向资金：ak 返回的单位可能是“元”，将其除以此因子转换为“万”
SOUTHBOUND_UNIT_CONVERSION = 10000.0
SOUTHBOUND_THRESHOLD = 5000.0  # 单位：万

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
AK_CALL_SLEEP = 0.2  # 调用 ak 时的短暂停顿以避免限流

# ==============================
# 2. 获取恒生指数数据
# ==============================
print("📈 获取恒生指数（^HSI）用于对比...")
hsi_ticker = yf.Ticker("^HSI")
hsi_hist = hsi_ticker.history(period=f"{PRICE_WINDOW + 30}d")  # 余量更大以防节假日
if hsi_hist.empty:
    raise RuntimeError("无法获取恒生指数数据")

def get_hsi_return(start, end):
    """
    使用前向/后向填充获取与股票时间戳对齐的恒指价格，返回区间收益（小数）。
    start/end 为 Timestamp（来自股票索引）。
    若无法获取，则返回 np.nan。
    """
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
    
    try:
        # 使用新的接口获取个股南向资金数据
        # akshare要求股票代码为5位数字格式，不足5位的需要在前面补0
        symbol = code.replace('.HK', '')
        if len(symbol) < 5:
            symbol = symbol.zfill(5)
        
        # 检查缓存中是否已有该股票的数据
        stock_cache_key = symbol
        if stock_cache_key in southbound_cache and southbound_cache[stock_cache_key] is not None:
            df_individual = southbound_cache[stock_cache_key]
        else:
            # 获取个股南向资金数据
            df_individual = ak.stock_hsgt_individual_em(symbol=symbol)
            
            if df_individual is None or df_individual.empty:
                print(f"⚠️ 获取南向资金数据为空 {code}")
                southbound_cache[stock_cache_key] = None
                time.sleep(AK_CALL_SLEEP)
                return None
            
            # 缓存该股票的所有数据
            southbound_cache[stock_cache_key] = df_individual
        
        # 将日期字符串转换为pandas日期格式进行匹配
        target_date = pd.to_datetime(date_str, format='%Y%m%d')
        
        # 筛选指定日期的数据
        df_filtered = df_individual[df_individual['持股日期'] == target_date.date()]
        
        # 如果未找到指定日期的数据，使用最近的可用日期数据
        if df_filtered.empty:
            # 获取所有可用日期
            available_dates = df_individual['持股日期']
            # 找到最近的日期（小于或等于目标日期）
            closest_date = available_dates[available_dates.dt.date() <= target_date.date()].max()
            
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

# ==============================
# 4. 单股分析函数
# ==============================
def analyze_stock(code, name):
    try:
        print(f"\n🔍 分析 {name} ({code}) ...")
        ticker = yf.Ticker(code)
        full_hist = ticker.history(period=f"{PRICE_WINDOW + 30}d")
        if len(full_hist) < PRICE_WINDOW:
            print(f"⚠️  {name} 数据不足（需要至少 {PRICE_WINDOW} 日）")
            return None

        main_hist = full_hist[['Open', 'Close', 'Volume']].tail(DAYS_ANALYSIS).copy()
        if len(main_hist) < 5:
            print(f"⚠️  {name} 主分析窗口数据不足")
            return None

        # ====== 排除周六日（只保留交易日）======
        main_hist = main_hist[main_hist.index.weekday < 5]
        full_hist = full_hist[full_hist.index.weekday < 5]

        # 基础指标（在 full_hist 上计算）
        full_hist['Vol_MA20'] = full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).mean()
        full_hist['MA5'] = full_hist['Close'].rolling(5, min_periods=1).mean()
        full_hist['MA10'] = full_hist['Close'].rolling(10, min_periods=1).mean()

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
        main_hist['MACD'] = full_hist['MACD'].reindex(main_hist.index, method='ffill')
        main_hist['MACD_Signal'] = full_hist['MACD_Signal'].reindex(main_hist.index, method='ffill')
        main_hist['RSI'] = full_hist['RSI'].reindex(main_hist.index, method='ffill')
        main_hist['Volatility'] = full_hist['Volatility'].reindex(main_hist.index, method='ffill')

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

        # 南向资金：按日期获取并缓存，转换为“万”
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

        if OUTPERFORMS_USE_RS:
            outperforms = bool(outperforms_by_rs)
        else:
            if OUTPERFORMS_REQUIRE_POSITIVE:
                outperforms = bool(outperforms_by_ret)
            else:
                outperforms = bool(outperforms_by_diff)

        # === 建仓信号 ===
        def is_buildup(row):
            return ((row['Price_Percentile'] < PRICE_LOW_PCT) and
                    (pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_BUILDUP) and
                    (pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] > SOUTHBOUND_THRESHOLD))

        main_hist['Buildup_Signal'] = main_hist.apply(is_buildup, axis=1)
        main_hist['Buildup_Confirmed'] = mark_runs(main_hist['Buildup_Signal'], BUILDUP_MIN_DAYS)

        # === 出货信号 ===
        main_hist['Prev_Close'] = main_hist['Close'].shift(1)
        def is_distribution(row):
            cond1 = row['Price_Percentile'] > PRICE_HIGH_PCT
            cond2 = (pd.notna(row.get('Vol_Ratio')) and row['Vol_Ratio'] > VOL_RATIO_DISTRIBUTION)
            cond3 = (pd.notna(row.get('Southbound_Net')) and row['Southbound_Net'] < -SOUTHBOUND_THRESHOLD)
            cond4 = (pd.notna(row.get('Prev_Close')) and (row['Close'] < row['Prev_Close'])) or (row['Close'] < row['Open'])
            return cond1 and cond2 and cond3 and cond4

        main_hist['Distribution_Signal'] = main_hist.apply(is_distribution, axis=1)
        main_hist['Distribution_Confirmed'] = mark_runs(main_hist['Distribution_Signal'], DISTRIBUTION_MIN_DAYS)

        # 是否存在信号
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()

        # 保存图表
        if SAVE_CHARTS:
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

        # 返回结构（保留原始数值：RS 为小数，RS_diff 小数；展示时再乘100）
        last_close = main_hist['Close'].iloc[-1]
        prev_close = main_hist['Close'].iloc[-2] if len(main_hist) >= 2 else None
        change_pct = ((last_close / prev_close) - 1) * 100 if prev_close is not None and prev_close != 0 else None

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
            'southbound': safe_round(main_hist['Southbound_Net'].iloc[-1], 2),  # 单位：万
            'ma5_deviation': safe_round(((last_close / main_hist['MA5'].iloc[-1]) - 1) * 100, 2) if pd.notna(main_hist['MA5'].iloc[-1]) and main_hist['MA5'].iloc[-1] > 0 else None,
            'ma10_deviation': safe_round(((last_close / main_hist['MA10'].iloc[-1]) - 1) * 100, 2) if pd.notna(main_hist['MA10'].iloc[-1]) and main_hist['MA10'].iloc[-1] > 0 else None,
            'macd': safe_round(main_hist['MACD'].iloc[-1], 4) if pd.notna(main_hist['MACD'].iloc[-1]) else None,
            'rsi': safe_round(main_hist['RSI'].iloc[-1], 2) if pd.notna(main_hist['RSI'].iloc[-1]) else None,
            'volatility': safe_round(main_hist['Volatility'].iloc[-1] * 100, 2) if pd.notna(main_hist['Volatility'].iloc[-1]) else None,  # 百分比
            'obv': safe_round(main_hist['OBV'].iloc[-1], 2) if pd.notna(main_hist['OBV'].iloc[-1]) else None,  # OBV指标
            'buildup_dates': main_hist[main_hist['Buildup_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
            'distribution_dates': main_hist[main_hist['Distribution_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
        }
        return result

    except Exception as e:
        print(f"❌ {name} 分析出错: {e}")
        return None

# ==============================
# 5. 批量分析与报告生成
# ==============================
print("="*80)
print("🚀 港股主力资金追踪器（建仓 + 出货 双信号） - 改进版")
print(f"分析 {len(WATCHLIST)} 只股票 | 窗口: {DAYS_ANALYSIS} 日")
print("="*80)

results = []
for code, name in WATCHLIST.items():
    res = analyze_stock(code, name)
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
        'name', 'code', 'last_close', 'change_pct', 'turnover',
        'price_percentile', 'vol_ratio', 'ma5_deviation', 'ma10_deviation',
        'rsi', 'macd', 'volatility', 'obv',
        'RS_ratio_%', 'RS_diff_%', 'outperforms_hsi',
        'southbound', 'has_buildup', 'has_distribution'
    ]]
    df_report.columns = [
        '股票名称', '代码', '最新价', '涨跌幅(%)', '成交金额(百万)',
        '位置(%)', '量比', '5日均线偏离(%)', '10日均线偏离(%)',
        'RSI', 'MACD', '波动率(%)', 'OBV',
        '相对强度(RS_ratio_%)', '相对强度差值(RS_diff_%)', '跑赢恒指',
        '南向资金(万)', '建仓信号', '出货信号'
    ]

    df_report = df_report.sort_values(['出货信号', '建仓信号'], ascending=[True, False])

    # 确保数值列格式化为两位小数用于显示
    for col in df_report.select_dtypes(include=['float64', 'int64']).columns:
        df_report[col] = df_report[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

    print("\n" + "="*120)
    print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
    print("="*120)
    print(df_report.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

    # 指标说明（完整版，控制台与邮件中保持一致）
    FULL_INDICATOR_TEXT = """
================================================================================
📋 指标说明（完整版）-- 便于理解每个指标的含义、计算方法与局限性
================================================================================

【基础信息】
  • 最新价：股票当前最新成交价格（港元）。若当日存在盘中变动，建议结合成交量与盘口观察。
  • 前收市价：前一个交易日的收盘价格（港元）。
  • 涨跌幅(%)：按 (最新价 - 前收) / 前收 计算并乘以100表示百分比。

【相对表现 / 跑赢恒指（用于衡量个股相对大盘的表现）】
  • 相对强度 (RS_ratio)：
      - 计算：RS_ratio = (1 + stock_ret) / (1 + hsi_ret) - 1
        其中 stock_ret 和 hsi_ret 都是分析窗口内的区间收益（小数，如 0.05 表示 5%）。
      - 含义：基于复合收益（即把两个收益都视为复利因子）来度量个股相对恒指的表现。
        RS_ratio > 0 表示个股在该区间的复合收益率高于恒指；RS_ratio < 0 则表示跑输。
      - 优点：在收益率接近 -1 或波动较大时，更稳健地反映“相对复合回报”。
      - 报告显示：以百分比列 RS_ratio_% 呈现（例如 5 表示 +5%）。
  • 相对强度差值 (RS_diff)：
      - 计算：RS_diff = stock_ret - hsi_ret（直接的收益差值）。
      - 含义：更直观，表示绝对收益的差额（例如股票涨 6%，恒指涨 2%，则 RS_diff = 4%）。
      - 报告显示：以百分比列 RS_diff_% 呈现。
  • 跑赢恒指 (outperforms_hsi)：
      - 脚本支持三种语义：
        1) 要求股票为正收益并且收益 > 恒指（默认，保守）：OUTPERFORMS_REQUIRE_POSITIVE = True
        2) 仅比较收益差值（无需股票为正）：OUTPERFORMS_REQUIRE_POSITIVE = False
        3) 使用 RS_ratio（以复合收益判断）：OUTPERFORMS_USE_RS = True
      - 可根据需要在脚本顶部切换判定方式。

【价格与位置】
  • 位置(%)：当前价格在最近 PRICE_WINDOW（默认 60 日）内的百分位位置。
      - 计算：(当前价 - 最近N日最低) / (最高 - 最低) * 100，取 [0, 100]。
      - 含义：接近 0 表示处于历史窗口低位，接近 100 表示高位。
      - 用途：判断是否处于“相对低位”或“高位”，用于建仓/出货信号的价格条件。

【成交量相关】
  • 量比 (Vol_Ratio)：当日成交量 / 20 日平均成交量（VOL_WINDOW）。
      - 含义：衡量当日成交是否显著放大。
      - 建议：放量配合价格运动（如放量上涨或放量下跌）比单纯放量更具信号含义。
  • 成交金额(百万)：当日成交金额，单位为百万港元（近似计算：最新价 * 成交量 / 1e6）。

【均线偏离】
  • 5日/10日均线偏离(%)：最新价相对于短期均线的偏离百分比（正值表示价高于均线）。
      - 用途：短期动力判断；但对高波动或宽幅震荡个股需谨慎解读。

【动量与震荡指标】
  • MACD：
      - 计算：EMA12 - EMA26（MACD 线），并计算 9 日 EMA 作为 MACD Signal。
      - 含义：衡量中短期动量，MACD 线上穿信号线通常被视为动能改善（反之则疲弱）。
      - 注意：对剧烈震荡或极端股价数据（如停牌后复牌）可能失真。
  • RSI（Wilder 平滑）：
      - 计算：基于 14 日 Wilder 指数平滑的涨跌幅比例，结果在 0-100。
      - 含义：常用于判断超买/超卖（例如 RSI > 70 可能偏超买，RSI < 30 可能偏超卖）。
      - 注意：单独使用 RSI 可能产生误导，建议与成交量和趋势指标结合。

【波动率】
  • 波动率(%)：基于 20 日收益率样本的样本标准差年化后以百分比表示（std * sqrt(252)）。
      - 含义：衡量历史波动幅度，用于风险评估和头寸大小控制。

【OBV（On-Balance Volume）】
  • OBV：按照日涨跌累计成交量的方向（涨则加，跌则减）来累计。
      - 含义：尝试用成交量的方向性累积来辅助判断资金是否在积累/分配。
      - 注意：OBV 是累积序列，适合观察中长期趋势而非短期信号。

【南向资金（沪港通/深港通南向）】
  • 数据来源：使用 akshare 的 stock_hk_ggt_components_em 获取“净买入”字段，脚本假设原始单位为“元”并除以 SOUTHBOUND_UNIT_CONVERSION 转为“万”。
  • 报告字段：南向资金(万) 表示当日或该交易日的净买入额（万元）。
  • 用途：当南向资金显著流入时，通常被解读为北向/南向机构资金的买入兴趣；显著流出则表示机构抛售或撤出。
  • 限制与谨慎：
      - ak 数据延迟或字段命名可能变化（脚本已做基本容错，但仍需关注源数据格式）。
      - 单日南向资金异常需结合量价关系与连续性判断，避免被一次性大额交易误导。

【信号定义（本脚本采用的简化规则）】
  • 建仓信号（Buildup）：
      - 条件：位置 < PRICE_LOW_PCT（低位） AND 量比 > VOL_RATIO_BUILDUP（成交放大） AND 南向资金净流入 > SOUTHBOUND_THRESHOLD（万）。
      - 连续性：要求连续或累计达到 BUILDUP_MIN_DAYS 才被标注为确认（避免孤立样本）。
      - 语义：在低位出现放量且机构买入力度强时，可能代表主力建仓或底部吸筹。
  • 出货信号（Distribution）：
      - 条件：位置 > PRICE_HIGH_PCT（高位） AND 量比 > VOL_RATIO_DISTRIBUTION（剧烈放量） AND 南向资金净流出 < -SOUTHBOUND_THRESHOLD（万） AND 当日收盘下行（相对前[...]
      - 连续性：要求连续达到 DISTRIBUTION_MIN_DAYS 才标注为确认。
      - 语义：高位放量且机构撤出，伴随价格下行，可能代表主力在高位分批出货/派发。
  • 重要提醒：
      - 本脚本规则为经验性启发式筛选，不构成投资建议。建议将信号作为筛选或复核工具，结合持仓风险管理、基本面与订单簿/资金面深度判断。
      - 对于停牌、派息、拆股或其他公司事件，指标需特殊处理；脚本未一一覆盖这些事件。

【其他说明与实践建议】
  • 时间窗口与阈值（如 PRICE_WINDOW、VOL_WINDOW、阈值等）可根据策略偏好调整。更短窗口更灵敏但噪声更多，反之亦然。
  • 建议：把信号与多因子（行业动量、估值、持仓集中度）结合，避免单一信号操作。
  • 数据来源与一致性：本脚本结合 yfinance（行情）与 akshare（南向资金），两者更新频率与字段定义可能不同，使用时请确认数据来源的可用性与一致[...]
================================================================================
"""
    # 高亮信号
    distribution_stocks = [r for r in results if r['has_distribution']]
    buildup_stocks = [r for r in results if r['has_buildup']]

    if distribution_stocks:
        print("\n🔴 警惕！检测到大户出货信号：")
        for r in distribution_stocks:
            print(f"  • {r['name']} | 日期: {', '.join(r['distribution_dates'])}")

    if buildup_stocks:
        print("\n🟢 检测到建仓信号：")
        for r in buildup_stocks:
            rs_disp = (round(r['relative_strength'] * 100, 2) if (r.get('relative_strength') is not None) else None)
            rsd_disp = (round(r['relative_strength_diff'] * 100, 2) if (r.get('relative_strength_diff') is not None) else None)
            print(f"  • {r['name']} | RS_ratio={rs_disp}% | RS_diff={rsd_disp}% | 日期: {', '.join(r['buildup_dates'])} | 跑赢恒指: {r['outperforms_hsi']}")

    # 保存 Excel（包含 machine-friendly 原始列 + 展示列）
    try:
        df.to_excel("hk_smart_money_report.xlsx", index=False)
        print("\n💾 报告已保存: hk_smart_money_report.xlsx")
    except Exception as e:
        print(f"⚠️  Excel保存失败: {e}")

    # 发送邮件（将表格分段为多个 HTML 表格并包含说明）
    def send_email_with_report(df_report, to):
        smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
        smtp_port = 587
        smtp_user = os.environ.get("YAHOO_EMAIL")
        smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
        sender_email = smtp_user

        if not smtp_user or not smtp_pass:
            print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
            return False

        if isinstance(to, str):
            to = [to]

        subject = "港股主力资金追踪报告（改进版）"

        text = "港股主力资金追踪报告（改进版）\n\n"
        html = "<html><body><h2>港股主力资金追踪报告（改进版）</h2>"

        # 添加表格（每 5 行分一页）
        for i in range(0, len(df_report), 5):
            chunk = df_report.iloc[i:i+5]
            html += chunk.to_html(index=False, escape=False)

        # 添加简洁的信号摘要
        dist = df_report[df_report['出货信号'] == True]
        build = df_report[df_report['建仓信号'] == True]
        if not dist.empty:
            html += "<h3 style='color:red;'>🔴 出货信号：</h3><ul>"
            for _, row in dist.iterrows():
                html += f"<li>{row['股票名称']} ({row['代码']})</li>"
            html += "</ul>"
        if not build.empty:
            html += "<h3 style='color:green;'>🟢 建仓信号：</h3><ul>"
            for _, row in build.iterrows():
                html += f"<li>{row['股票名称']} ({row['代码']})</li>"
            html += "</ul>"

        # 指标说明（完整版 HTML 版本）
        FULL_INDICATOR_HTML = """
        <h3>📋 指标说明（完整版）</h3>
        <div style="font-size:0.9em; line-height:1.4;">
        <h4>基础信息</h4>
        <ul>
          <li><b>最新价</b>：股票当前最新成交价格（港元）。</li>
          <li><b>前收市价</b>：前一个交易日的收盘价格（港元）。</li>
          <li><b>涨跌幅(%)</b>：按 (最新价 - 前收) / 前收 计算并乘以100表示百分比。</li>
        </ul>
        <h4>相对表现 / 跑赢恒指</h4>
        <ul>
          <li><b>RS_ratio</b>：(1+stock_ret)/(1+hsi_ret)-1，表示基于复合收益率的相对表现，>0 表示跑赢恒指。报告以百分比显示（RS_ratio_%）。</li>
          <li><b>RS_diff</b>：stock_ret - hsi_ret，表示绝对收益差值，报告以百分比显示（RS_diff_%）。</li>
          <li><b>跑赢恒指 (outperforms)</b>：脚本支持多种判定语义（详见脚本顶部配置）。</li>
        </ul>
        <h4>价格与位置</h4>
        <ul>
          <li><b>位置(%)</b>：当前价格在最近 60 日区间的百分位（0-100）。</li>
        </ul>
        <h4>成交量相关</h4>
        <ul>
          <li><b>量比</b>：当日成交量 / 20 日平均成交量（VOL_WINDOW）。</li>
          <li><b>成交金额(百万)</b>：当日成交金额，单位为百万港元。</li>
        </ul>
        <h4>动量与震荡指标</h4>
        <ul>
          <li><b>MACD</b>：EMA12 - EMA26，配合 9 日信号线判断动能变化。</li>
          <li><b>RSI（Wilder）</b>：基于 14 日 Wilder 平滑，范围 0-100，常用于超买/超卖判断。</li>
          <li><b>波动率(%)</b>：基于 20 日收益率样本的年化波动率。</li>
        </ul>
        <h4>南向资金（万）</h4>
        <ul>
          <li>来自 akshare 的净买入字段，经除以 {unit} 转为“万”显示。单日流入/流出需结合量价与连续性判断。</li>
        </ul>
        <h4>信号定义（简述）</h4>
        <ul>
          <li><b>建仓信号</b>：位置 < {low}% 且量比 > {vr_build} 且南向资金净流入 > {sb}（万），连续至少 {bd} 日。</li>
          <li><b>出货信号</b>：位置 > {high}% 且量比 > {vr_dist} 且南向资金净流出 < -{sb}（万）且当日收盘下行，连续至少 {dd} 日。</li>
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

        html += "</body></html>"

        msg = MIMEMultipart("mixed")
        msg['From'] = f'"wonglaitung" <{sender_email}>'
        msg['To'] = ", ".join(to)
        msg['Subject'] = subject

        body = MIMEMultipart("alternative")
        body.attach(MIMEText(text, "plain"))
        body.attach(MIMEText(html, "html"))
        msg.attach(body)

        # 附件图表
        if os.path.exists(CHART_DIR):
            for filename in os.listdir(CHART_DIR):
                if filename.endswith(".png"):
                    with open(os.path.join(CHART_DIR, filename), "rb") as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    msg.attach(part)

        try:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender_email, to, msg.as_string())
            server.quit()
            print("✅ 邮件发送成功")
            return True
        except Exception as e:
            print(f"❌ 发送邮件失败: {e}")
            return False

    recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
    recipients = [r.strip() for r in recipient_env.split(',')] if ',' in recipient_env else [recipient_env]
    print("📧 发送邮件到:", ", ".join(recipients))
    send_email_with_report(df_report, recipients)

print(f"\n✅ 分析完成！图表保存至: {CHART_DIR}/")
