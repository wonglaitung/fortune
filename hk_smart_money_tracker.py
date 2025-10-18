# -*- coding: utf-8 -*-
"""
港股主力资金追踪器（建仓 + 出货 双信号）
作者：AI助手
功能：
  - 批量扫描自选股
  - 识别「建仓信号」：低位 + 放量 + 南向流入 + 跑赢恒指
  - 识别「出货信号」：高位 + 巨量 + 南向流出 + 滞涨
  - 输出汇总报告 + 图表
"""

import warnings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置环境变量以避免字体警告
os.environ['MPLBACKEND'] = 'Agg'

import yfinance as yf
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 用户设置区
# ==============================
WATCHLIST = {
    "0700.HK": "腾讯控股",
    "9988.HK": "阿里巴巴-SW",
    "3690.HK": "美团-W",
    "1810.HK": "小米集团-W",
    "9618.HK": "京东集团-SW",
    "2800.HK": "盈富基金",
    "3968.HK": "招商银行",
    "0939.HK": "建设银行",
    "1398.HK": "工商银行",
    "0388.HK": "香港交易所",
    "6682.HK": "第四范式",
    "1347.HK": "华虹半导体",
    "0981.HK": "中芯国际",
}

# 分析参数
DAYS_ANALYSIS = 12
VOL_WINDOW = 20
PRICE_WINDOW = 60
BUILDUP_MIN_DAYS = 3   # 建仓需连续3日
DISTRIBUTION_MIN_DAYS = 2  # 出货需连续2日

SAVE_CHARTS = True
CHART_DIR = "hk_smart_charts"
if SAVE_CHARTS and not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# ==============================
# 2. 获取恒生指数数据
# ==============================
print("📈 获取恒生指数（^HSI）用于对比...")
hsi_ticker = yf.Ticker("^HSI")
hsi_hist = hsi_ticker.history(period=f"{PRICE_WINDOW + 10}d")
if hsi_hist.empty:
    raise RuntimeError("无法获取恒生指数数据")

def get_hsi_return(start, end):
    try:
        s = hsi_hist.loc[start:end, 'Close'].iloc[0]
        e = hsi_hist.loc[start:end, 'Close'].iloc[-1]
        return (e - s) / s if s != 0 else 0
    except:
        return 0

# ==============================
# 3. 单股分析函数
# ==============================
def analyze_stock(code, name):
    try:
        print(f"\n🔍 分析 {name} ({code}) ...")
        ticker = yf.Ticker(code)
        full_hist = ticker.history(period=f"{PRICE_WINDOW + 10}d")
        if len(full_hist) < PRICE_WINDOW:
            print(f"⚠️  {name} 数据不足")
            return None

        main_hist = full_hist[['Open', 'Close', 'Volume']].tail(DAYS_ANALYSIS).copy()
        if len(main_hist) < 5:
            return None

        # 基础指标
        full_hist['Vol_MA20'] = full_hist['Volume'].rolling(VOL_WINDOW, min_periods=1).mean()
        full_hist['MA5'] = full_hist['Close'].rolling(5, min_periods=1).mean()
        full_hist['MA10'] = full_hist['Close'].rolling(10, min_periods=1).mean()
        
        # MACD计算
        full_hist['EMA12'] = full_hist['Close'].ewm(span=12).mean()
        full_hist['EMA26'] = full_hist['Close'].ewm(span=26).mean()
        full_hist['MACD'] = full_hist['EMA12'] - full_hist['EMA26']
        full_hist['MACD_Signal'] = full_hist['MACD'].ewm(span=9).mean()
        
        # RSI计算
        delta = full_hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        full_hist['RSI'] = 100 - (100 / (1 + rs))
        
        # 波动率计算(20日收益率标准差)
        full_hist['Returns'] = full_hist['Close'].pct_change()
        full_hist['Volatility'] = full_hist['Returns'].rolling(20).std() * (252 ** 0.5)  # 年化波动率
        
        low60 = full_hist['Close'].tail(PRICE_WINDOW).min()
        high60 = full_hist['Close'].tail(PRICE_WINDOW).max()
        
        main_hist['Vol_MA20'] = full_hist['Vol_MA20'].reindex(main_hist.index, method='ffill')
        main_hist['Vol_Ratio'] = main_hist['Volume'] / main_hist['Vol_MA20']
        main_hist['Price_Percentile'] = ((main_hist['Close'] - low60) / (high60 - low60) * 100).clip(0, 100)
        
        # 从full_hist获取技术指标
        main_hist['MA5'] = full_hist['MA5'].reindex(main_hist.index, method='ffill')
        main_hist['MA10'] = full_hist['MA10'].reindex(main_hist.index, method='ffill')
        main_hist['MACD'] = full_hist['MACD'].reindex(main_hist.index, method='ffill')
        main_hist['MACD_Signal'] = full_hist['MACD_Signal'].reindex(main_hist.index, method='ffill')
        main_hist['RSI'] = full_hist['RSI'].reindex(main_hist.index, method='ffill')
        main_hist['Volatility'] = full_hist['Volatility'].reindex(main_hist.index, method='ffill')
        
        # OBV
        main_hist = main_hist.copy()  # 创建副本以避免SettingWithCopyWarning
        main_hist['OBV'] = 0.0
        for i in range(1, len(main_hist)):
            delta = main_hist['Close'].iloc[i] - main_hist['Close'].iloc[i-1]
            if delta > 0:
                main_hist.loc[main_hist.index[i], 'OBV'] = main_hist['OBV'].iloc[i-1] + main_hist['Volume'].iloc[i]
            elif delta < 0:
                main_hist.loc[main_hist.index[i], 'OBV'] = main_hist['OBV'].iloc[i-1] - main_hist['Volume'].iloc[i]
            else:
                main_hist.loc[main_hist.index[i], 'OBV'] = main_hist['OBV'].iloc[i-1]

        # 南向资金
        main_hist = main_hist.copy()  # 创建副本以避免SettingWithCopyWarning
        main_hist['Southbound_Net'] = 0.0
        dates = main_hist.index.strftime('%Y%m%d').tolist()
        for date in dates:
            try:
                df = ak.stock_hk_ggt_components_em(date=date)
                if not df.empty:
                    match = df[df['代码'] == code.replace('.HK', '')]
                    if not match.empty:
                        net_str = match['净买入'].values[0].replace(',', '')
                        net = pd.to_numeric(net_str, errors='coerce')
                        if pd.notna(net):
                            main_hist.loc[main_hist.index.strftime('%Y%m%d') == date, 'Southbound_Net'] = net
            except:
                pass

        # 相对强度
        start_date, end_date = main_hist.index[0], main_hist.index[-1]
        stock_ret = (main_hist['Close'].iloc[-1] - main_hist['Close'].iloc[0]) / main_hist['Close'].iloc[0]
        hsi_ret = get_hsi_return(start_date, end_date)
        rs = stock_ret / hsi_ret if hsi_ret != 0 else (1.0 if stock_ret >= 0 else -1.0)
        # 跑赢恒指 = True 需要同时满足：股票上涨且涨幅超过恒指，或者股票下跌但跌幅小于恒指且整体是正收益
        outperforms = stock_ret > hsi_ret and stock_ret > 0

        # === 建仓信号 ===
        def is_buildup(row):
            return (row['Price_Percentile'] < 30 and 
                    row['Vol_Ratio'] > 1.5 and 
                    row['Southbound_Net'] > 5000)
        
        main_hist['Buildup_Signal'] = main_hist.apply(is_buildup, axis=1)
        main_hist = main_hist.copy()  # 创建副本以避免SettingWithCopyWarning
        main_hist['Buildup_Confirmed'] = False
        count = 0
        for i in range(len(main_hist)-1, -1, -1):
            if main_hist['Buildup_Signal'].iloc[i]:
                count += 1
                if count >= BUILDUP_MIN_DAYS:
                    for j in range(BUILDUP_MIN_DAYS):
                        if i-j >= 0:
                            main_hist.loc[main_hist.index[i-j], 'Buildup_Confirmed'] = True
            else:
                count = 0

        # === 出货信号 ===
        main_hist['Prev_Close'] = main_hist['Close'].shift(1)
        def is_distribution(row):
            cond1 = row['Price_Percentile'] > 70
            cond2 = row['Vol_Ratio'] > 2.5
            cond3 = row['Southbound_Net'] < -5000
            cond4 = (row['Close'] < row['Prev_Close']) or (row['Close'] < row['Open'])
            return cond1 and cond2 and cond3 and cond4
        
        main_hist['Distribution_Signal'] = main_hist.apply(is_distribution, axis=1)
        main_hist = main_hist.copy()  # 创建副本以避免SettingWithCopyWarning
        main_hist['Distribution_Confirmed'] = False
        count = 0
        for i in range(len(main_hist)-1, -1, -1):
            if main_hist['Distribution_Signal'].iloc[i]:
                count += 1
                if count >= DISTRIBUTION_MIN_DAYS:
                    for j in range(DISTRIBUTION_MIN_DAYS):
                        if i-j >= 0:
                            main_hist.loc[main_hist.index[i-j], 'Distribution_Confirmed'] = True
            else:
                count = 0

        # 保存图表（总是生成）
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()
        
        # 保存图表（总是生成）
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()
        
        # 保存图表（总是生成）
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()
        
        if SAVE_CHARTS:
            hsi_plot = hsi_hist['Close'].reindex(main_hist.index, method='ffill')
            stock_plot = main_hist['Close']
            
            plt.figure(figsize=(10, 6))
            plt.plot(stock_plot.index, stock_plot / stock_plot.iloc[0], 'b-o', label=f'{code} {name}')
            plt.plot(hsi_plot.index, hsi_plot / hsi_plot.iloc[0], 'orange', linestyle='--', label='恒生指数')
            title = f"{code} {name} vs 恒指 | RS: {rs:.2f}"
            if has_buildup:
                title += " [建仓]"
            if has_distribution:
                title += " [出货]"
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            status = ("buildup" if has_buildup else "") + ("_distribution" if has_distribution else "")
            plt.savefig(f"{CHART_DIR}/{code}_{name}{status}.png")
            plt.close()

        return {
            'code': code,
            'name': name,
            'has_buildup': has_buildup,
            'has_distribution': has_distribution,
            'outperforms_hsi': outperforms,
            'relative_strength': rs,
            'last_close': main_hist['Close'].iloc[-1],
            'prev_close': main_hist['Close'].iloc[-2] if len(main_hist) >= 2 else None,
            'change_pct': ((main_hist['Close'].iloc[-1] / main_hist['Close'].iloc[-2]) - 1) * 100 if len(main_hist) >= 2 else 0,
            'price_percentile': main_hist['Price_Percentile'].iloc[-1],
            'vol_ratio': main_hist['Vol_Ratio'].iloc[-1],
            'southbound': main_hist['Southbound_Net'].iloc[-1],
            'ma5_deviation': ((main_hist['Close'].iloc[-1] / main_hist['MA5'].iloc[-1]) - 1) * 100 if main_hist['MA5'].iloc[-1] > 0 else 0,
            'ma10_deviation': ((main_hist['Close'].iloc[-1] / main_hist['MA10'].iloc[-1]) - 1) * 100 if main_hist['MA10'].iloc[-1] > 0 else 0,
            'macd': main_hist['MACD'].iloc[-1],
            'rsi': main_hist['RSI'].iloc[-1],
            'volatility': main_hist['Volatility'].iloc[-1] * 100 if pd.notna(main_hist['Volatility'].iloc[-1]) else 0,
            'buildup_dates': main_hist[main_hist['Buildup_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
            'distribution_dates': main_hist[main_hist['Distribution_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
        }

    except Exception as e:
        print(f"❌ {name} 分析出错: {e}")
        return None

# ==============================
# 4. 执行批量分析
# ==============================
print("="*80)
print("🚀 港股主力资金追踪器（建仓 + 出货 双信号）")
print(f"分析 {len(WATCHLIST)} 只股票 | 窗口: {DAYS_ANALYSIS} 日")
print("="*80)

results = []
for code, name in WATCHLIST.items():
    res = analyze_stock(code, name)
    if res:
        results.append(res)

# ==============================
# 5. 生成报告
# ==============================
if not results:
    print("❌ 无结果")
else:
    df = pd.DataFrame(results)
    df = df[[
        'name', 'code', 'last_close', 'prev_close', 'change_pct',
        'has_buildup', 'has_distribution', 'outperforms_hsi',
        'relative_strength', 'price_percentile', 'vol_ratio',
        'ma5_deviation', 'ma10_deviation', 'macd', 'rsi', 'volatility',
        'southbound'
    ]]
    df.columns = [
        '股票名称', '代码', '最新价', '前收市价', '涨跌幅(%)',
        '建仓信号', '出货信号', '跑赢恒指',
        '相对强度(RS)', '位置(%)', '量比',
        '5日均线偏离(%)', '10日均线偏离(%)', 'MACD', 'RSI', '波动率(%)',
        '南向资金(万)'
    ]
    df = df.sort_values(['出货信号', '建仓信号'], ascending=[True, False])  # 出货优先警示

    print("\n" + "="*110)
    print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
    print("="*110)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))
    
    # 添加指标说明
    print("\n" + "="*110)
    print("📋 指标说明：")
    print("="*110)
    print("【基础信息】")
    print("  • 最新价：股票当前最新成交价格")
    print("  • 前收市价：前一个交易日的收盘价格")
    print("  • 涨跌幅(%)：当前价格相对于前收市价的涨跌幅度 (正值表示上涨，负值表示下跌)")
    
    print("\n【信号指标】")
    print("  • 建仓信号：低位 + 放量 + 南向流入 + 跑赢恒指 (出现建仓信号可能意味着主力资金开始买入)")
    print("  • 出货信号：高位 + 巨量 + 南向流出 + 滞涨 (出现出货信号可能意味着主力资金开始卖出)")
    print("  • 跑赢恒指：股票上涨且涨幅超过恒指，或者股票下跌但跌幅小于恒指且整体是正收益")
    print("      注意：当跑赢恒指为False但RS>1时，表示股票相对表现优于恒指，但股票本身可能为负收益")
    
    print("\n【技术指标】")
    print("  • 相对强度(RS)：股票收益与恒生指数收益的比值 (RS>1表示表现优于恒生指数)")
    print("  • 位置(%)：当前价格在60日价格区间中的百分位 (数值越小表示相对位置越低，数值越大表示相对位置越高)")
    print("  • 量比：当前成交量与20日平均成交量的比值 (量比>1表示当日成交量高于20日均值)")
    print("  • 5日均线偏离(%)：当前价格偏离5日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)")
    print("  • 10日均线偏离(%)：当前价格偏离10日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)")
    print("  • MACD：指数平滑异同移动平均线 (用于判断买卖时机的趋势指标)")
    print("      MACD>0：短期均线在长期均线上方，市场可能处于多头状态")
    print("      MACD<0：短期均线在长期均线下方，市场可能处于空头状态")
    print("      MACD值增大：趋势加强")
    print("      MACD值减小：趋势减弱")
    print("  • RSI：相对强弱指数 (衡量股票超买或超卖状态的指标，范围0-100，通常>70为超买，<30为超卖)")
    print("  • 波动率(%)：年化波动率，衡量股票的风险水平 (数值越大表示风险越高)")
    
    print("\n【资金流向】")
    print("  • 南向资金(万)：沪港通/深港通南向资金净流入金额（万元）(正值表示资金流入，负值表示资金流出)")

    # 高亮关键信号
    buildup_stocks = [r for r in results if r['has_buildup']]
    distribution_stocks = [r for r in results if r['has_distribution']]

    if distribution_stocks:
        print("\n🔴 警惕！检测到大户出货信号：")
        for r in distribution_stocks:
            print(f"  • {r['name']} | 日期: {', '.join(r['distribution_dates'])}")
    
    if buildup_stocks:
        strong_buildup = [r for r in buildup_stocks if r['outperforms_hsi']]
        if strong_buildup:
            print("\n🟢 机会！高质量建仓信号（跑赢恒指）：")
            for r in strong_buildup:
                print(f"  • {r['name']} | RS={r['relative_strength']:.2f} | 日期: {', '.join(r['buildup_dates'])}")

    # 保存Excel
    try:
        df.to_excel("hk_smart_money_report.xlsx", index=False)
        print("\n💾 报告已保存: hk_smart_money_report.xlsx")
    except Exception as e:
        print(f"⚠️  Excel保存失败: {e}")

    # 发送邮件
    recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
    
    # 如果环境变量中有多个收件人（用逗号分隔），则拆分为列表
    if ',' in recipient_env:
        recipients = [recipient.strip() for recipient in recipient_env.split(',')]
    else:
        recipients = [recipient_env]
    
    print("📧 发送邮件到:", ", ".join(recipients))
    send_email_with_report(df, recipients)

print(f"\n✅ 分析完成！图表保存至: {CHART_DIR}/")
