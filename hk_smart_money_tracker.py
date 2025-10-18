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

import yfinance as yf
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. 用户设置区
# ==============================
WATCHLIST = {
    "0700.HK": "腾讯控股",
    "09988.HK": "阿里巴巴",
    "03690.HK": "美团-W",
    "01810.HK": "小米集团-W",
    "09618.HK": "京东集团",
    "02318.HK": "中国平安",
    "02800.HK": "盈富基金",
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
        low60 = full_hist['Close'].tail(PRICE_WINDOW).min()
        high60 = full_hist['Close'].tail(PRICE_WINDOW).max()
        
        main_hist['Vol_MA20'] = full_hist['Vol_MA20'].reindex(main_hist.index, method='ffill')
        main_hist['Vol_Ratio'] = main_hist['Volume'] / main_hist['Vol_MA20']
        main_hist['Price_Percentile'] = ((main_hist['Close'] - low60) / (high60 - low60) * 100).clip(0, 100)
        
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

        # 保存图表（如有信号）
        has_buildup = main_hist['Buildup_Confirmed'].any()
        has_distribution = main_hist['Distribution_Confirmed'].any()
        
        if SAVE_CHARTS and (has_buildup or has_distribution):
            hsi_plot = hsi_hist['Close'].reindex(main_hist.index, method='ffill')
            stock_plot = main_hist['Close']
            
            plt.figure(figsize=(10, 6))
            plt.plot(stock_plot.index, stock_plot / stock_plot.iloc[0], 'b-o', label=name)
            plt.plot(hsi_plot.index, hsi_plot / hsi_plot.iloc[0], 'orange--', label='恒生指数')
            title = f"{name} vs 恒指 | RS: {rs:.2f}"
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
            'price_percentile': main_hist['Price_Percentile'].iloc[-1],
            'vol_ratio': main_hist['Vol_Ratio'].iloc[-1],
            'southbound': main_hist['Southbound_Net'].iloc[-1],
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
        'name', 'code', 'has_buildup', 'has_distribution',
        'outperforms_hsi', 'relative_strength',
        'last_close', 'price_percentile', 'vol_ratio', 'southbound'
    ]]
    df.columns = [
        '股票名称', '代码', '建仓信号', '出货信号',
        '跑赢恒指', '相对强度(RS)',
        '最新价', '位置(%)', '量比', '南向资金(万)'
    ]
    df = df.sort_values(['出货信号', '建仓信号'], ascending=[True, False])  # 出货优先警示

    print("\n" + "="*110)
    print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
    print("="*110)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

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

print(f"\n✅ 分析完成！图表保存至: {CHART_DIR}/")
