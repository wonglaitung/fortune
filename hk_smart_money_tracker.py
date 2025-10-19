# -*- coding: utf-8 -*-
"""
港股主力资金追踪器（建仓 + 出货 双信号）
作者：AI助手（修补版）
说明：对所有计算结果统一保留小数点后两位
"""

import warnings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import math

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

# ==============================
# 1. 用户设置区
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

# 分析参数
DAYS_ANALYSIS = 12
VOL_WINDOW = 20
PRICE_WINDOW = 60
BUILDUP_MIN_DAYS = 3   # 建仓需连续3日
DISTRIBUTION_MIN_DAYS = 2  # 出货需连续2日

# 配置：是否在判断 "跑赢恒指" 时强制要求股票为正收益
# True => outperforms = (stock_ret > 0) and (stock_ret > hsi_ret)
# False => outperforms = stock_ret > hsi_ret
OUTPERFORMS_REQUIRE_POSITIVE = True

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

def send_email_with_report(df, to):
    """
    发送主力资金追踪报告邮件
    """
    smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
    smtp_port = 587
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
        return False

    # 如果to是字符串，转换为列表
    if isinstance(to, str):
        to = [to]

    subject = "港股主力资金追踪报告"
    
    # 生成文本和HTML内容
    text = "港股主力资金追踪报告:\n\n"
    html = "<html><body><h2>港股主力资金追踪报告</h2>"
    
    if df is not None and not df.empty:
        # 添加文本内容
        text += df.to_string(index=False) + "\n\n"
        
        # 添加HTML内容
        # 创建带样式交替颜色的HTML表格，按每5个股票分拆
        # 添加CSS样式
        html += '''
        <style>
        .stock-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            margin-bottom: 20px;
        }
        .stock-table th, .stock-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .stock-table th {
            background-color: #4CAF50;
            color: white;
        }
        .stock-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .stock-table tr:nth-child(odd) {
            background-color: #ffffff;
        }
        .stock-table tr:hover {
            background-color: #ddd;
        }
        </style>
        '''
        
        # 按每5个股票分拆成多个表格
        for i in range(0, len(df), 5):
            chunk = df.iloc[i:i+5]
            html_table = chunk.to_html(index=False, escape=False, table_id=f"stock-table-{i//5}")
            html += f"<h3>股票数据 (第{i//5+1}页)</h3>"
            html += html_table
        
        # 添加关键信号提醒
        buildup_stocks = df[df['建仓信号'] == True]
        distribution_stocks = df[df['出货信号'] == True]
        
        if not distribution_stocks.empty:
            html += "<h3 style='color: red;'>🔴 警惕！检测到大户出货信号：</h3><ul>"
            for _, stock in distribution_stocks.iterrows():
                html += f"<li>{stock['股票名称']}</li>"
            html += "</ul>"
        
        if not buildup_stocks.empty:
            html += "<h3 style='color: green;'>🟢 机会！检测到建仓信号：</h3><ul>"
            for _, stock in buildup_stocks.iterrows():
                html += f"<li>{stock['股票名称']}</li>"
            html += "</ul>"
            
        # 添加指标说明（已更新：使用 ratio 和 diff 两种 RS 表示）
        html += "<h3>📋 指标说明（更新）：</h3>"
        html += "<h4>【基础信息】</h4>"
        html += "<ul>"
        html += "<li><strong>最新价</strong>：股票当前最新成交价格</li>"
        html += "<li><strong>前收市价</strong>：前一个交易日的收盘价格</li>"
        html += "<li><strong>涨跌幅(%)</strong>：当前价格相对于前收市价的涨跌幅度 (正值表示上涨，负值表示下跌)</li>"
        html += "</ul>"
        
        html += "<h4>【信号指标】</h4>"
        html += "<ul>"
        html += "<li><strong>建仓信号</strong>：低位 + 放量 + 南向流入 + 跑赢恒指 (出现建仓信号可能意味着主力资金开始买入)</li>"
        html += "<li><strong>出货信号</strong>：高位 + 巨量 + 南向流出 + 滞涨 (出现出货信号可能意味着主力资金开始卖出)</li>"
        html += "</ul>"
        
        html += "<h4>【相对表现 (跑赢恒指) 说明】</h4>"
        html += "<ul>"
        html += "<li><strong>relative_strength_ratio (RS)</strong>：使用 (1+股票收益)/(1+恒指收益)-1 计算；当 RS &gt; 0 表示股票按复合收益率跑赢恒指；该定义在恒指波动较大时更稳健。</li>"
        html += "<li><strong>relative_strength_diff</strong>：股票收益 - 恒指收益；>0 表示股票收益高于恒指（直观差值）。</li>"
        html += "<li><strong>跑赢恒指 (outperforms)</strong>：脚本可配置为两种语义（顶部配置 OUTPERFORMS_REQUIRE_POSITIVE）：</li>"
        html += "<ul>"
        html += "<li>如果 OUTPERFORMS_REQUIRE_POSITIVE = True：要求股票为正收益且收益高于恒指（较为保守，等同于“正收益并跑赢”）</li>"
        html += "<li>如果 OUTPERFORMS_REQUIRE_POSITIVE = False：只要股票收益高于恒指即视为跑赢（不要求股票为正收益）</li>"
        html += "</ul>"
        html += "<li><strong>示例说明</strong>：当恒指下跌而个股也下跌但跌幅更小，RS_ratio > 0（或 RS_diff > 0），表示相对表现更好，但股票仍可能为负收益；</li>"
        html += "</ul>"
        
        html += "<h4>【技术指标】</h4>"
        html += "<ul>"
        html += "<li><strong>位置(%)</strong>：当前价格在60日价格区间中的百分位 (数值越小表示相对位置越低，数值越大表示相对位置越高)</li>"
        html += "<li><strong>量比</strong>：当前成交量与20日平均成交量的比值 (量比&gt;1表示当日成交量高于20日均值)</li>"
        html += "<li><strong>成交金额(百万)</strong>：股票当日的成交金额，以百万为单位显示</li>"
        html += "<li><strong>5日均线偏离(%)</strong>：当前价格偏离5日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)</li>"
        html += "<li><strong>10日均线偏离(%)</strong>：当前价格偏离10日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)</li>"
        html += "<li><strong>MACD</strong>：指数平滑异同移动平均线 (用于判断买卖时机的趋势指标)</li>"
        html += "<li><strong>RSI</strong>：相对强弱指数 (衡量股票超买或超卖状态的指标，范围0-100，通常&gt;70为超买，&lt;30为超卖)</li>"
        html += "<li><strong>波动率(%)</strong>：年化波动率，衡量股票的风险水平 (数值越大表示风险越高)</li>"
        html += "</ul>"
        
        html += "<h4>【资金流向】</h4>"
        html += "<ul>"
        html += "<li><strong>南向资金(万)</strong>：沪港通/深港通南向资金净流入金额（万元）(正值表示资金流入，负值表示资金流出)</li>"
        html += "</ul>"
    else:
        text += "未能获取到数据\n\n"
        html += "<p>未能获取到数据</p>"
    
    html += "</body></html>"

    msg = MIMEMultipart("mixed")
    msg['From'] = f'"wonglaitung" <{sender_email}>'
    msg['To'] = ", ".join(to)
    msg['Subject'] = subject

    # 创建邮件正文部分
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(text, "plain"))
    body.attach(MIMEText(html, "html"))
    msg.attach(body)

    # 添加图表附件
    if os.path.exists(CHART_DIR):
        print(f"🔍 检查附件目录: {CHART_DIR}")
        attachment_count = 0
        for filename in os.listdir(CHART_DIR):
            if filename.endswith(".png"):
                filepath = os.path.join(CHART_DIR, filename)
                print(f"📎 找到附件: {filename}")
                with open(filepath, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}',
                )
                msg.attach(part)
                attachment_count += 1
        print(f"📧 总共添加了 {attachment_count} 个附件")
    else:
        print(f"❌ 附件目录不存在: {CHART_DIR}")

    # 打印邮件内容长度用于调试
    print(f"✉️ 邮件内容长度: {len(msg.as_string())} 字符")
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(sender_email, to, msg.as_string())
        server.quit()
        print("✅ 主力资金追踪报告邮件发送成功!")
        return True
    except Exception as e:
        print(f"❌ 发送邮件时出错: {e}")
        return False

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
        # 避免除以0
        main_hist['Price_Percentile'] = ((main_hist['Close'] - low60) / (high60 - low60) * 100).clip(0, 100) if high60 != low60 else 50.0
        
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

        # 相对强度（改为 ratio 与 diff）
        start_date, end_date = main_hist.index[0], main_hist.index[-1]
        stock_ret = (main_hist['Close'].iloc[-1] - main_hist['Close'].iloc[0]) / main_hist['Close'].iloc[0]
        hsi_ret = get_hsi_return(start_date, end_date)
        rs_diff = stock_ret - hsi_ret
        # ratio 采用 (1+stock_ret)/(1+hsi_ret)-1（更稳健，不会因为 hsi_ret 为负而直接反转符号含义）
        if (1.0 + hsi_ret) == 0:
            rs_ratio = float('inf') if (1.0 + stock_ret) > 0 else float('-inf')
        else:
            rs_ratio = (1.0 + stock_ret) / (1.0 + hsi_ret) - 1.0

        # 跑赢恒指判定（可配置）
        if OUTPERFORMS_REQUIRE_POSITIVE:
            outperforms = (stock_ret > 0) and (stock_ret > hsi_ret)
        else:
            outperforms = stock_ret > hsi_ret

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
        
        # 统一保留两位小数的辅助函数
        def round2(v):
            try:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    if not math.isfinite(v):
                        return v
                    return round(float(v), 2)
                # pandas types
                if pd.isna(v):
                    return None
                return v
            except:
                return v

        if SAVE_CHARTS:
            # 画图时使用未被四舍五入用于计算的序列，但展示值使用两位小数
            hsi_plot = hsi_hist['Close'].reindex(main_hist.index, method='ffill')
            stock_plot = main_hist['Close']
            
            # 用于标题显示的值（四舍五入两位）
            rs_ratio_display = round2(rs_ratio)
            rs_diff_display = round2(rs_diff * 100)  # 用百分比表示
            plt.figure(figsize=(10, 6))
            plt.plot(stock_plot.index, stock_plot / stock_plot.iloc[0], 'b-o', label=f'{code} {name}')
            plt.plot(hsi_plot.index, hsi_plot / hsi_plot.iloc[0], 'orange', linestyle='--', label='恒生指数')
            title = f"{code} {name} vs 恒指 | RS_ratio: {rs_ratio_display if rs_ratio_display is not None else 'NA'} | RS_diff: {rs_diff_display if rs_diff_display is not None else 'NA'}%"
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
            safe_name = name.replace('/', '_').replace(' ', '_')
            plt.savefig(f"{CHART_DIR}/{code}_{safe_name}{status}.png")
            plt.close()

        # 准备返回值并统一保留两位小数
        last_close = main_hist['Close'].iloc[-1]
        prev_close = main_hist['Close'].iloc[-2] if len(main_hist) >= 2 else None
        change_pct = ((main_hist['Close'].iloc[-1] / main_hist['Close'].iloc[-2]) - 1) * 100 if len(main_hist) >= 2 else 0

        result = {
            'code': code,
            'name': name,
            'has_buildup': bool(has_buildup),
            'has_distribution': bool(has_distribution),
            'outperforms_hsi': bool(outperforms),
            'relative_strength': round2(rs_ratio),            # ratio 形式
            'relative_strength_diff': round2(rs_diff),        # 差值形式（小数，用户可乘100显示百分比）
            'last_close': round2(last_close),
            'prev_close': round2(prev_close),
            'change_pct': round2(change_pct),
            'price_percentile': round2(main_hist['Price_Percentile'].iloc[-1]),
            'vol_ratio': round2(main_hist['Vol_Ratio'].iloc[-1]),
            'turnover': round2((main_hist['Close'].iloc[-1] * main_hist['Volume'].iloc[-1]) / 1000000),  # 成交金额（以百万为单位）
            'southbound': round2(main_hist['Southbound_Net'].iloc[-1]),
            'ma5_deviation': round2(((main_hist['Close'].iloc[-1] / main_hist['MA5'].iloc[-1]) - 1) * 100) if main_hist['MA5'].iloc[-1] > 0 else 0,
            'ma10_deviation': round2(((main_hist['Close'].iloc[-1] / main_hist['MA10'].iloc[-1]) - 1) * 100) if main_hist['MA10'].iloc[-1] > 0 else 0,
            'macd': round2(main_hist['MACD'].iloc[-1]),
            'rsi': round2(main_hist['RSI'].iloc[-1]),
            'volatility': round2(main_hist['Volatility'].iloc[-1] * 100) if pd.notna(main_hist['Volatility'].iloc[-1]) else 0,
            'buildup_dates': main_hist[main_hist['Buildup_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
            'distribution_dates': main_hist[main_hist['Distribution_Confirmed']].index.strftime('%Y-%m-%d').tolist(),
        }

        return result

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
    # 保持向后兼容性：'relative_strength' 字段为 ratio（>0 表示跑赢），并新增 'relative_strength_diff' 用于直观差值
    df = df[[
        'name', 'code', 'last_close', 'prev_close', 'change_pct',
        'has_buildup', 'has_distribution', 'outperforms_hsi',
        'relative_strength', 'relative_strength_diff', 'price_percentile', 'vol_ratio', 'turnover',
        'ma5_deviation', 'ma10_deviation', 'macd', 'rsi', 'volatility',
        'southbound'
    ]]
    df.columns = [
        '股票名称', '代码', '最新价', '前收市价', '涨跌幅(%)',
        '建仓信号', '出货信号', '跑赢恒指',
        '相对强度(RS_ratio)', '相对强度差值(RS_diff)', '位置(%)', '量比', '成交金额(百万)',
        '5日均线偏离(%)', '10日均线偏离(%)', 'MACD', 'RSI', '波动率(%)',
        '南向资金(万)'
    ]
    df = df.sort_values(['出货信号', '建仓信号'], ascending=[True, False])  # 出货优先警示

    # 将所有数值列保留两位小数（再次确保）
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: round(float(x), 2) if (pd.notna(x) and isinstance(x, (int, float))) else x)

    print("\n" + "="*110)
    print("📊 主力资金信号汇总（🔴 出货 | 🟢 建仓）")
    print("="*110)
    # 控制台显示保证两位小数
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))
    
    # 添加指标说明（控制台版）
    print("\n" + "="*110)
    print("📋 指标说明（已更新）:")
    print("="*110)
    print("【基础信息】")
    print("  • 最新价：股票当前最新成交价格")
    print("  • 前收市价：前一个交易日的收盘价格")
    print("  • 涨跌幅(%)：当前价格相对于前收市价的涨跌幅度 (正值表示上涨，负值表示下跌)")
    
    print("\n【相对表现 / 跑赢恒指说明】")
    print("  • 相对强度(RS_ratio)：(1+股票收益)/(1+恒指收益)-1；>0 表示按复合收益率跑赢恒指（对恒指为负时更稳健）。")
    print("  • 相对强度差值(RS_diff)：股票收益 - 恒指收益；>0 表示股票收益高于恒指（直观差值）。")
    print("  • 跑赢恒指(outperforms)：可配置（脚本顶部 OUTPERFORMS_REQUIRE_POSITIVE），"
          "True 表示要求股票为正收益且收益高于恒指；False 表示只要股票收益高于恒指即视为跑赢。")
    print("  • 说明示例：当恒指下跌、个股下跌但跌幅更小，RS_ratio 与 RS_diff 可能为正，但股票仍是负收益。")

    print("\n【技术指标】")
    print("  • 位置(%)：当前价格在60日价格区间中的百分位 (数值越小表示相对位置越低，数值越大表示相对位置越高)")
    print("  • 量比：当前成交量与20日平均成交量的比值 (量比>1表示当日成交量高于20日均值)")
    print("  • 成交金额(百万)：股票当日的成交金额，以百万为单位显示")
    print("  • 5日均线偏离(%)：当前价格偏离5日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)")
    print("  • 10日均线偏离(%)：当前价格偏离10日均线的程度 (正值表示价格在均线上方，负值表示价格在均线下方)")
    print("  • MACD：指数平滑异同移动平均线 (用于判断买卖时机的趋势指标)")
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
                rs_ratio_display = round(r['relative_strength'], 2) if (r.get('relative_strength') is not None and isinstance(r.get('relative_strength'), (int, float))) else r.get('relative_strength')
                rs_diff_display = round(r['relative_strength_diff'] * 100, 2) if (r.get('relative_strength_diff') is not None and isinstance(r.get('relative_strength_diff'), (int, float))) else r.get('relative_strength_diff')
                print(f"  • {r['name']} | RS_ratio={rs_ratio_display} | RS_diff={rs_diff_display}% | 日期: {', '.join(r['buildup_dates'])}")

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
