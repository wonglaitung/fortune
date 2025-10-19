# -*- coding: utf-8 -*-
"""
港股主力资金追踪器（建仓 + 出货 双信号）- 完整改进版
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
southbound_cache = {}  # cache[(date_str)] = DataFrame from ak

def fetch_ggt_components(date_str):
    """
    从 ak 获取当日的港股南向资金成分（整表），并缓存。
    date_str 格式 YYYYMMDD
    返回 DataFrame 或 None
    """
    if date_str in southbound_cache:
        return southbound_cache[date_str]
    try:
        df = ak.stock_hk_ggt_components_em(date=date_str)
        # 有时 ak 返回空表或异常格式，做基本校验
        if isinstance(df, pd.DataFrame) and not df.empty:
            southbound_cache[date_str] = df
            # 略微延时以防被限流
            time.sleep(AK_CALL_SLEEP)
            return df
        southbound_cache[date_str] = None
        time.sleep(AK_CALL_SLEEP)
        return None
    except Exception:
        southbound_cache[date_str] = None
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
            date_str = ts.strftime('%Y%m%d')
            df_ggt = fetch_ggt_components(date_str)
            if df_ggt is None:
                continue
            # 匹配代码（ak 返回 '代码' 可能没有后缀）
            match = df_ggt[df_ggt.get('代码', '').astype(str) == code.replace('.HK', '')]
            if not match.empty:
                # 取第一个匹配
                try:
                    net_raw = str(match['净买入'].values[0]).replace(',', '')
                    net_val = pd.to_numeric(net_raw, errors='coerce')
                    if pd.notna(net_val):
                        main_hist.at[ts, 'Southbound_Net'] = float(net_val) / SOUTHBOUND_UNIT_CONVERSION
                except Exception:
                    # 忽略解析错误
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
        'name', 'code', 'last_close', 'prev_close', 'change_pct',
        'has_buildup', 'has_distribution', 'outperforms_hsi',
        'RS_ratio_%', 'RS_diff_%', 'price_percentile', 'vol_ratio', 'turnover',
        'ma5_deviation', 'ma10_deviation', 'macd', 'rsi', 'volatility',
        'southbound'
    ]]
    df_report.columns = [
        '股票名称', '代码', '最新价', '前收市价', '涨跌幅(%)',
        '建仓信号', '出货信号', '跑赢恒指',
        '相对强度(RS_ratio_%)', '相对强度差值(RS_diff_%)', '位置(%)', '量比', '成交金额(百万)',
        '5日均线偏离(%)', '10日均线偏离(%)', 'MACD', 'RSI', '波动率(%)',
        '南向资金(万)'
    ]

    df_report = df_report.sort_values(['出货信号', '建仓信号'], ascending=[True, False])

    # 确保数值列格式化为两位小数用于显示
    for col in df_report.select_dtypes(include=['float64', 'int64']).columns:
        df_report[col] = df_report[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

    print("\n" + "="*120)
    print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
    print("="*120)
    print(df_report.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

    # 指标说明（控制台与邮件中保持一致）
    print("\n" + "="*120)
    print("📋 指标说明（已更新）：")
    print("="*120)
    print("【基础信息】")
    print("  • 最新价：股票当前最新成交价格（港元）")
    print("  • 前收市价：前一个交易日的收盘价格（港元）")
    print("  • 涨跌幅(%)：当前价格相对于前收市价的涨跌幅度（%）")
    print("\n【相对表现 / 跑赢恒指说明】")
    print("  • 相对强度(RS_ratio)：(1+股票收益)/(1+恒指收益)-1（小数），>0 表示按复合收益率跑赢恒指。")
    print("    在报告表中以百分比显示（RS_ratio_%），例如 5 表示 +5%。")
    print("  • 相对强度差值(RS_diff)：股票收益 - 恒指收益（小数），>0 表示股票收益高于恒指。")
    print("    在报告表中以百分比显示（RS_diff_%），例如 5 表示 +5%。")
    print("  • 跑赢恒指(outperforms)：可配置判定语义（脚本顶部 OUTPERFORMS_REQUIRE_POSITIVE / OUTPERFORMS_USE_RS）。")
    print("    默认要求股票为正收益并高于恒指（更保守）。可切换为只比收益差值或使用 RS_ratio。")
    print("\n【技术指标】")
    print("  • 位置(%)：当前价格在最近 60 日价格区间（最低-最高）中的百分位（0-100）。")
    print("  • 量比：当日成交量 / 20 日平均成交量（VOL_WINDOW）。")
    print("  • 成交金额(百万)：当日成交金额（以百万港元为单位）。")
    print("  • 5日/10日均线偏离(%)：当前价格相对于均线的偏离（%）。")
    print("  • MACD：基于 EMA12-EMA26。")
    print("  • RSI：使用 Wilder 平滑，范围 0-100。")
    print("  • 波动率(%)：基于 20 日收益率样本年化后以百分比显示。")
    print("\n【资金流向】")
    print(f"  • 南向资金(万)：沪港通/深港通南向资金净买入（万元）。脚本假设 ak 返回单位为“元”，并除以 {int(SOUTHBOUND_UNIT_CONVERSION)} 转为“万”。")
    print(f"    检测建仓阈值：南向资金 > {SOUTHBOUND_THRESHOLD} 万；出货阈值：南向资金 < -{SOUTHBOUND_THRESHOLD} 万。")
    print("\n【信号定义（简述）】")
    print(f"  • 建仓信号：位置 < {PRICE_LOW_PCT}%，量比 > {VOL_RATIO_BUILDUP}，且南向资金净流入超阈值（{SOUTHBOUND_THRESHOLD} 万）。")
    print(f"  • 出货信号：位置 > {PRICE_HIGH_PCT}%，量比 > {VOL_RATIO_DISTRIBUTION}，南向资金净流出超阈值且当日收盘下行。")
    print("\n备注：RS_ratio 与 RS_diff 都表示相对表现，但语义略有差别。RS_ratio 为复合收益比，RS_diff 为直观差值。")

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

        # 添加表格（每 8 行分一页）
        for i in range(0, len(df_report), 8):
            chunk = df_report.iloc[i:i+8]
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

        # 指标说明（简洁版本）
        html += "<h3>📋 指标说明</h3>"
        html += "<ul>"
        html += "<li>RS_ratio: (1+stock_ret)/(1+hsi_ret)-1（以百分比显示，>0 表示跑赢恒指）</li>"
        html += "<li>RS_diff: stock_ret - hsi_ret（以百分比显示）</li>"
        html += "<li>位置(%)：当前价格在最近60日区间的百分位（0-100）</li>"
        html += "<li>量比：当日成交量 / 20日均量</li>"
        html += "<li>南向资金(万)：ak 返回值转换为万元显示（阈值以万元计）</li>"
        html += "</ul>"

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
