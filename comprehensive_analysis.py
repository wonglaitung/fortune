#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议
"""

import os
import sys
import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入大模型服务
from llm_services.qwen_engine import chat_with_llm

# 导入必要的模块
try:
    from data_services.hk_sector_analysis import SectorAnalyzer
    SECTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    SECTOR_ANALYSIS_AVAILABLE = False
    print("⚠️ 板块分析模块不可用")

try:
    from akshare import stock_a_div_em
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️ AKShare模块不可用")


def load_model_accuracy(horizon=20):
    """
    从文件加载模型准确率信息
    
    参数:
    - horizon: 预测周期（默认20天）
    
    返回:
    - dict: 包含LightGBM和GBDT准确率的字典
      {
        'lgbm': {'accuracy': float, 'std': float},
        'gbdt': {'accuracy': float, 'std': float}
      }
    """
    # 默认准确率值（如果文件不存在）
    default_accuracy = {
        'lgbm': {'accuracy': 0.6015, 'std': 0.0518},
        'gbdt': {'accuracy': 0.6069, 'std': 0.0500}
    }
    
    accuracy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_accuracy.json')
    
    try:
        if os.path.exists(accuracy_file):
            import json
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = {}
            lgbm_key = f'lgbm_{horizon}d'
            gbdt_key = f'gbdt_{horizon}d'
            
            if lgbm_key in data:
                result['lgbm'] = {
                    'accuracy': data[lgbm_key].get('accuracy', default_accuracy['lgbm']['accuracy']),
                    'std': data[lgbm_key].get('std', default_accuracy['lgbm']['std'])
                }
            else:
                result['lgbm'] = default_accuracy['lgbm']
            
            if gbdt_key in data:
                result['gbdt'] = {
                    'accuracy': data[gbdt_key].get('accuracy', default_accuracy['gbdt']['accuracy']),
                    'std': data[gbdt_key].get('std', default_accuracy['gbdt']['std'])
                }
            else:
                result['gbdt'] = default_accuracy['gbdt']
            
            print(f"✅ 已加载模型准确率: {accuracy_file}")
            print(f"   LightGBM: {result['lgbm']['accuracy']:.2%} (±{result['lgbm']['std']:.2%})")
            print(f"   GBDT: {result['gbdt']['accuracy']:.2%} (±{result['gbdt']['std']:.2%})")
            return result
        else:
            print(f"⚠️ 准确率文件不存在: {accuracy_file}，使用默认值")
            return default_accuracy
    except Exception as e:
        print(f"⚠️ 读取准确率文件失败: {e}，使用默认值")
        return default_accuracy


def extract_llm_recommendations(filepath):
    """
    从大模型建议文件中提取买卖建议，分别提取短期和中期建议
    
    参数:
    - filepath: 文件路径
    
    返回:
    - dict: 包含短期和中期建议的字典
      {
        'short_term': str,  # 短期建议文本
        'medium_term': str  # 中期建议文本
      }
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        
        # 使用更精确的正则表达式提取短期建议
        # 匹配"### 稳健型短期分析"标题后到下一个"###"标题之前的内容
        short_term_match = re.search(
            r'^###.*稳健型短期分析.*?\n(.*?)(?=^###|\Z)',
            content,
            re.DOTALL | re.MULTILINE
        )
        
        # 使用更精确的正则表达式提取中期建议
        # 匹配"### 稳健型中期分析"标题后到文件末尾或下一个"###"标题之前的内容
        medium_term_match = re.search(
            r'^###.*稳健型中期分析.*?\n(.*?)(?=\Z|^###)',
            content,
            re.DOTALL | re.MULTILINE
        )
        
        result = {
            'short_term': short_term_match.group(1).strip() if short_term_match else '',
            'medium_term': medium_term_match.group(1).strip() if medium_term_match else ''
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return {'short_term': '', 'medium_term': ''}


def extract_ml_predictions(filepath):
    """
    从ML预测CSV文件中提取LightGBM和GBDT的预测结果
    
    参数:
    - filepath: 文本预测文件路径（用于获取日期）
    
    返回:
    - dict: 包含LightGBM和GBDT预测结果的字典
      {
        'lgbm': str,      # LightGBM预测结果
        'gbdt': str   # GBDT预测结果
      }
    """
    try:
        import pandas as pd
        from datetime import datetime
        import os
        
        # 从文件路径中提取日期
        date_str = filepath.split('_')[-1].replace('.txt', '')
        
        # 使用相对路径（从当前脚本位置推导data目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        lgbm_csv = os.path.join(data_dir, 'ml_trading_model_lgbm_predictions_20d.csv')
        gbdt_csv = os.path.join(data_dir, 'ml_trading_model_gbdt_predictions_20d.csv')
        
        result = {
            'lgbm': '',
            'gbdt': ''
        }
        
        # 读取LightGBM预测结果
        if os.path.exists(lgbm_csv):
            df_lgbm = pd.read_csv(lgbm_csv)
            # 提取预测上涨的股票
            up_stocks_lgbm = df_lgbm[df_lgbm['prediction'] == 1].sort_values('probability', ascending=False)
            
            lgbm_text = "【LightGBM模型预测结果】\n"
            lgbm_text += f"预测日期: {date_str}\n\n"
            lgbm_text += "预测上涨的股票（按概率排序）:\n\n"
            
            # 构建Markdown表格
            lgbm_text += "| 股票代码 | 股票名称 | 上涨概率 | 当前价格 |\n"
            lgbm_text += "|----------|----------|----------|----------|\n"
            
            for _, row in up_stocks_lgbm.iterrows():
                lgbm_text += f"| {row['code']} | {row['name']} | {row['probability']:.4f} | {row['current_price']:.2f} |\n"
            
            lgbm_text += f"\n**统计信息**：\n"
            lgbm_text += f"- 预测上涨: {len(up_stocks_lgbm)} 只\n"
            lgbm_text += f"- 预测下跌: {len(df_lgbm) - len(up_stocks_lgbm)} 只\n"
            lgbm_text += f"- 平均上涨概率: {up_stocks_lgbm['probability'].mean():.4f}\n"
            
            result['lgbm'] = lgbm_text
        
        # 读取GBDT预测结果
        if os.path.exists(gbdt_csv):
            df_gbdt = pd.read_csv(gbdt_csv)
            # 提取预测上涨的股票
            up_stocks_gbdt = df_gbdt[df_gbdt['prediction'] == 1].sort_values('probability', ascending=False)
            
            gbdt_text = "【GBDT模型预测结果】\n"
            gbdt_text += f"预测日期: {date_str}\n\n"
            gbdt_text += "预测上涨的股票（按概率排序）:\n\n"
            
            # 构建Markdown表格
            gbdt_text += "| 股票代码 | 股票名称 | 上涨概率 | 当前价格 |\n"
            gbdt_text += "|----------|----------|----------|----------|\n"
            
            for _, row in up_stocks_gbdt.iterrows():
                gbdt_text += f"| {row['code']} | {row['name']} | {row['probability']:.4f} | {row['current_price']:.2f} |\n"
            
            gbdt_text += f"\n**统计信息**：\n"
            gbdt_text += f"- 预测上涨: {len(up_stocks_gbdt)} 只\n"
            gbdt_text += f"- 预测下跌: {len(df_gbdt) - len(up_stocks_gbdt)} 只\n"
            gbdt_text += f"- 平均上涨概率: {up_stocks_gbdt['probability'].mean():.4f}\n"
            
            result['gbdt'] = gbdt_text
        
        return result
        
    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'lgbm': '', 'gbdt_lr': ''}


def generate_html_email(content, date_str):
    """
    生成HTML格式的邮件内容
    
    参数:
    - content: 综合分析文本内容（Markdown格式）
    - date_str: 分析日期
    
    返回:
    - str: HTML格式的邮件内容
    """
    try:
        import markdown
    except ImportError:
        print("⚠️ 警告：未安装markdown库，使用简单转换")
        # 如果没有安装markdown库，使用简单转换
        simple_html = content.replace('\n', '<br>')
        return simple_html
    
    # 配置markdown扩展，使用更多功能以支持嵌套列表
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
    
    # 将Markdown转换为HTML
    html_content = md.convert(content)
    
    # 组装完整的HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            background-color: #ffffff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 28px;
        }}
        h2 {{
            color: #3498db;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 35px;
            margin-bottom: 20px;
            font-size: 22px;
        }}
        h3 {{
            color: #8e44ad;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 20px;
        }}
        h4 {{
            color: #2c3e50;
            margin: 0 0 12px 0;
            font-size: 18px;
        }}
        p {{
            color: #34495e;
            line-height: 1.8;
            margin: 10px 0;
        }}
        ul, ol {{
            color: #34495e;
            line-height: 1.8;
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .reference-section {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 4px solid #95a5a6;
        }}
        .reference-title {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .reference-content {{
            background: #ffffff;
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .metric-section {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
        }}
        .metric-title {{
            color: #2c3e50;
            font-size: 16px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .metric-item {{
            margin: 8px 0;
            padding-left: 15px;
            border-left: 2px solid #ddd;
        }}
        .risk-section {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }}
        .data-source {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #6c757d;
            font-size: 13px;
            line-height: 1.6;
        }}
        .model-accuracy {{
            background: #d4edda;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
            font-size: 14px;
        }}
        .warning {{
            background: #fff3cd;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 港股综合买卖建议</h1>
        <p style="color: #7f8c8d; font-size: 14px;">📅 分析日期：{date_str}</p>
        
        <div class="content">
            {html_content}
        </div>
        
        <div class="footer">
            <p>📧 本邮件由港股综合分析系统自动生成</p>
            <p>⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def get_sector_analysis():
    """
    获取板块分析数据
    
    返回:
    - dict: 包含板块分析结果
    """
    if not SECTOR_ANALYSIS_AVAILABLE:
        return None
    
    try:
        sector_analyzer = SectorAnalyzer()
        perf_df = sector_analyzer.calculate_sector_performance(period=5)
        
        if perf_df is None or perf_df.empty:
            return None
        
        # 识别龙头股（前3名）
        sector_leaders = {}
        for idx, row in perf_df.iterrows():
            sector_code = row['sector_code']
            leaders_df = sector_analyzer.identify_sector_leaders(
                sector_code=sector_code,
                top_n=3,
                period=5,
                min_market_cap=100,
                style='moderate'
            )
            
            if not leaders_df.empty:
                sector_leaders[sector_code] = []
                for _, leader_row in leaders_df.iterrows():
                    sector_leaders[sector_code].append({
                        'name': leader_row['name'],
                        'code': leader_row['code'],
                        'change_pct': leader_row['change_pct'],
                    })
        
        return {
            'performance': perf_df,
            'leaders': sector_leaders
        }
    except Exception as e:
        print(f"⚠️ 获取板块分析失败: {e}")
        return None


def get_dividend_info():
    """
    获取股息信息
    
    返回:
    - dict: 包含即将除净的港股信息
    """
    if not AKSHARE_AVAILABLE:
        return None
    
    try:
        # 获取即将除净的港股
        df_dividend = stock_a_div_em(em="hk", start_date=datetime.now().strftime('%Y%m%d'), end_date=(datetime.now() + timedelta(days=90)).strftime('%Y%m%d'))
        
        if df_dividend is None or df_dividend.empty:
            return None
        
        # 只取前10个
        df_dividend = df_dividend.head(10)
        
        return df_dividend.to_dict('records')
    except Exception as e:
        print(f"⚠️ 获取股息信息失败: {e}")
        return None


def get_hsi_analysis():
    """
    获取恒生指数分析
    
    返回:
    - dict: 包含恒生指数技术分析结果
    """
    try:
        hsi_ticker = yf.Ticker("^HSI")
        hist = hsi_ticker.history(period="6mo")
        
        if hist.empty:
            return None
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # 计算基本指标
        current_price = latest['Close']
        change_pct = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] != 0 else 0
        
        # 计算RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # 计算移动平均线
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        
        # 趋势判断
        if current_price > ma20 > ma50:
            trend = "强势多头"
        elif current_price > ma20:
            trend = "短期上涨"
        elif current_price > ma50:
            trend = "震荡整理"
        else:
            trend = "弱势空头"
        
        return {
            'current_price': current_price,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'ma20': ma20,
            'ma50': ma50,
            'trend': trend
        }
    except Exception as e:
        print(f"⚠️ 获取恒生指数分析失败: {e}")
        return None


def get_ai_portfolio_analysis():
    """
    获取AI持仓分析
    
    返回:
    - dict: 包含AI持仓分析结果
    """
    try:
        # 读取大模型建议文件
        date_str = datetime.now().strftime('%Y-%m-%d')
        llm_file = f'data/llm_recommendations_{date_str}.txt'
        
        if not os.path.exists(llm_file):
            return None
        
        with open(llm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取AI持仓分析部分
        import re
        ai_analysis_match = re.search(r'【大模型持仓分析】(.*?)(?=\n\n【|$)', content, re.DOTALL)
        
        if ai_analysis_match:
            return ai_analysis_match.group(1).strip()
        
        return None
    except Exception as e:
        print(f"⚠️ 获取AI持仓分析失败: {e}")
        return None


def get_stock_technical_indicators(stock_code):
    """
    获取单只股票的详细技术指标
    
    参数:
    - stock_code: 股票代码（如 "0700.HK"）
    
    返回:
    - dict: 包含详细技术指标的字典
    """
    try:
        # 移除.HK后缀
        symbol = stock_code.replace('.HK', '')
        
        # 获取股票数据
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            return None
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # 基本指标
        current_price = latest['Close']
        change_pct = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] != 0 else 0
        
        # 技术指标
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]
        
        # 移动平均线
        ma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        ma10 = hist['Close'].rolling(window=10).mean().iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # 均线排列
        if ma5 > ma10 > ma20 > ma50:
            ma_alignment = "多头排列"
        elif ma5 < ma10 < ma20 < ma50:
            ma_alignment = "空头排列"
        else:
            ma_alignment = "震荡整理"
        
        # 均线斜率
        ma_slope_20 = (ma20 - hist['Close'].rolling(window=20).mean().iloc[-2]) / ma20 * 100 if len(hist) > 20 else 0
        ma_slope_50 = (ma50 - hist['Close'].rolling(window=50).mean().iloc[-2]) / ma50 * 100 if len(hist) > 50 else 0
        
        # 均线乖离率
        ma_deviation = ((current_price - ma20) / ma20 * 100) if ma20 > 0 else 0
        
        # 布林带
        bb_period = 20
        bb_std = 2
        bb_middle = hist['Close'].rolling(window=bb_period).mean()
        bb_std_dev = hist['Close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        
        # 布林带位置
        bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100
        
        # ATR
        high = hist['High'].astype(float)
        low = hist['Low'].astype(float)
        close = hist['Close'].astype(float)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean()
        current_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else 0
        
        # 成交量
        volume = latest['Volume']
        avg_volume_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # 趋势判断
        if current_price > ma20 > ma50:
            trend = "强势多头"
        elif current_price > ma20:
            trend = "短期上涨"
        elif current_price > ma50:
            trend = "震荡整理"
        else:
            trend = "弱势空头"
        
        # 支撑阻力位
        recent_highs = hist['High'].rolling(window=20).max()
        recent_lows = hist['Low'].rolling(window=20).min()
        support_level = recent_lows.iloc[-1]
        resistance_level = recent_highs.iloc[-1]
        support_distance = ((current_price - support_level) / current_price * 100) if current_price > 0 else 0
        resistance_distance = ((resistance_level - current_price) / current_price * 100) if current_price > 0 else 0
        
        # OBV（能量潮）
        obv_change = (latest['Close'] - prev['Close']) * latest['Volume']
        obv = (obv_change.cumsum() / 1e6).iloc[-1] if len(hist) > 0 else 0
        
        # 价格位置（基于20日区间）
        price_range_20d = hist['Close'].rolling(window=20).max() - hist['Close'].rolling(window=20).min()
        price_position = ((current_price - hist['Close'].rolling(window=20).min().iloc[-1]) / price_range_20d.iloc[-1] * 100) if price_range_20d.iloc[-1] > 0 else 50
        
        return {
            'current_price': current_price,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_hist': current_macd_hist,
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'ma_alignment': ma_alignment,
            'ma_slope_20': ma_slope_20,
            'ma_slope_50': ma_slope_50,
            'ma_deviation': ma_deviation,
            'bb_upper': current_bb_upper,
            'bb_lower': current_bb_lower,
            'bb_position': bb_position,
            'atr': current_atr,
            'volume': volume,
            'avg_volume_20': avg_volume_20,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'obv': obv,
            'price_position': price_position
        }
    except Exception as e:
        print(f"⚠️ 获取股票 {stock_code} 技术指标失败: {e}")
        return None


def generate_technical_indicators_table(stock_codes):
    """
    为推荐股票生成技术指标表格
    
    参数:
    - stock_codes: 股票代码列表（从推荐建议中提取）
    
    返回:
    - str: Markdown格式的技术指标表格
    """
    try:
        if not stock_codes:
            return ""
        
        # 按股票代码排序
        stock_codes_sorted = sorted(stock_codes)
        
        table = "\n## 六、推荐股票技术指标详情\n\n"
        table += "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | RSI | MACD | MA20 | MA50 | MA200 | 均线排列 | 均线斜率 | 乖离率 | 布林带位置 | ATR | 成交量比率 | 趋势 | 支撑位 | 阻力位 |\n"
        table += "|---------|---------|---------|--------|-----|------|-----|-----|------|---------|---------|-------|-----------|-----|-----------|------|--------|--------|\n"
        
        for stock_code in stock_codes_sorted:
            indicators = get_stock_technical_indicators(stock_code)
            
            if indicators:
                # 获取股票名称
                stock_name = STOCK_NAMES.get(stock_code, stock_code)
                
                # 格式化数据
                price = f"{indicators['current_price']:.2f}"
                change = f"{indicators['change_pct']:+.2f}%"
                rsi = f"{indicators['rsi']:.2f}"
                macd = f"{indicators['macd']:.2f}"
                ma20 = f"{indicators['ma20']:.2f}"
                ma50 = f"{indicators['ma50']:.2f}"
                ma200 = f"{indicators['ma200']:.2f}" if pd.notna(indicators['ma200']) else "N/A"
                ma_align = indicators['ma_alignment']
                ma_slope = f"{indicators['ma_slope_20']:.4f}"
                ma_dev = f"{indicators['ma_deviation']:.2f}%"
                bb_pos = f"{indicators['bb_position']:.1f}%"
                atr = f"{indicators['atr']:.2f}"
                vol_ratio = f"{indicators['volume_ratio']:.2f}x"
                trend = indicators['trend']
                support = f"{indicators['support_level']:.2f} ({indicators['support_distance']:.2f}%)"
                resistance = f"{indicators['resistance_level']:.2f} ({indicators['resistance_distance']:.2f}%)"
                
                # 根据数值添加颜色标记（文本用括号标注）
                if indicators['rsi'] > 70:
                    rsi += " (超买)"
                elif indicators['rsi'] < 30:
                    rsi += " (超卖)"
                
                if indicators['change_pct'] > 0:
                    change = f"📈 {change}"
                else:
                    change = f"📉 {change}"
                
                if indicators['trend'] == "强势多头":
                    trend = f"🟢 {trend}"
                elif indicators['trend'] == "弱势空头":
                    trend = f"🔴 {trend}"
                
                table += f"| {stock_code} | {stock_name} | {price} | {change} | {rsi} | {macd} | {ma20} | {ma50} | {ma200} | {ma_align} | {ma_slope} | {ma_dev} | {bb_pos} | {atr} | {vol_ratio} | {trend} | {support} | {resistance} |\n"
        
        return table
        
    except Exception as e:
        print(f"⚠️ 生成技术指标表格失败: {e}")
        return ""


def send_email(subject, content, html_content=None):
    """
    发送邮件通知
    
    参数:
    - subject: 邮件主题
    - content: 邮件文本内容
    - html_content: 邮件HTML内容（可选）
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # 从环境变量获取邮件配置
        sender_email = os.environ.get("YAHOO_EMAIL")
        email_password = os.environ.get("YAHOO_APP_PASSWORD")
        smtp_server = os.environ.get("YAHOO_SMTP", "smtp.163.com")
        recipient_email = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        
        if ',' in recipient_email:
            recipients = [recipient.strip() for recipient in recipient_email.split(',')]
        else:
            recipients = [recipient_email]
        
        if not sender_email or not email_password:
            print("❌ 邮件配置不完整，跳过邮件发送")
            return False
        
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
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)
        
        # 添加文本版本
        text_part = MIMEText(content, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # 如果有HTML版本，添加HTML版本
        if html_content:
            html_part = MIMEText(html_content, 'html', 'utf-8')
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
                
                print(f"✅ 邮件已发送到: {', '.join(recipients)}")
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


def run_comprehensive_analysis(llm_filepath, ml_filepath, output_filepath=None, send_email_flag=True):
    """
    运行综合分析
    
    参数:
    - llm_filepath: 大模型建议文件路径
    - ml_filepath: ML预测结果文件路径
    - output_filepath: 输出文件路径（可选）
    """
    try:
        print("=" * 80)
        print("🤖 综合分析开始")
        print("=" * 80)
        
        # 检查文件是否存在
        if not os.path.exists(llm_filepath):
            print(f"❌ 大模型建议文件不存在: {llm_filepath}")
            return None
        
        if not os.path.exists(ml_filepath):
            print(f"❌ ML预测结果文件不存在: {ml_filepath}")
            return None
        
        print(f"📊 大模型建议文件: {llm_filepath}")
        print(f"📊 ML预测结果文件: {ml_filepath}")
        print("")
        
        # 提取大模型建议
        print("📝 提取大模型买卖建议...")
        llm_recommendations = extract_llm_recommendations(llm_filepath)
        print(f"✅ 提取完成\n")
        print(f"   - 短期建议长度: {len(llm_recommendations['short_term'])} 字符")
        print(f"   - 中期建议长度: {len(llm_recommendations['medium_term'])} 字符\n")
        
        # 提取ML预测
        print("📝 提取ML预测结果...")
        ml_predictions = extract_ml_predictions(ml_filepath)
        print(f"✅ 提取完成\n")
        print(f"   - LightGBM预测长度: {len(ml_predictions['lgbm'])} 字符")
        print(f"   - GBDT预测长度: {len(ml_predictions['gbdt'])} 字符\n")
        
        # 加载模型准确率
        print("📝 加载模型准确率...")
        model_accuracy = load_model_accuracy(horizon=20)
        print(f"✅ 准确率加载完成\n")
        
        # 生成日期
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 构建综合分析提示词
        prompt = f"""你是一位专业的投资分析师。请根据以下四部分信息，进行综合分析，给出实质的买卖建议。

=== 信息来源 ===

【主要信息源 - 决策依据】

【1. 大模型中期买卖建议（数周-数月）】
{llm_recommendations['medium_term']}

【2. LightGBM模型20天预测结果】
{ml_predictions['lgbm']}

【3. GBDT模型20天预测结果】
{ml_predictions['gbdt']}

【辅助信息源 - 操作时机参考】

【4. 大模型短期买卖建议（日内/数天）】
{llm_recommendations['short_term']}

=== 综合分析规则 ===

**规则1：时间维度匹配（业界最佳实践）**
- **短期信号（触发器）**：负责"何时做"（Timing）
- **中期信号（确认器）**：负责"是否做"（Direction）
- 只有短期和中期方向一致时，才采取行动
- 短期和中期冲突时，选择观望（避免不确定性）

**决策逻辑（短期触发 + 中期确认）**：
- 短期建议买入 + 中期建议买入 → 强买入信号
- 短期建议买入 + 中期建议观望 → 观望（等待中期确认）
- 短期建议买入 + 中期建议卖出 → 不买入（冲突，信号无效）
- 短期建议卖出 + 中期建议卖出 → 强卖出信号
- 短期建议卖出 + 中期建议观望 → 观望
- 短期建议卖出 + 中期建议买入 → 不卖出（冲突，信号无效）

**规则2：一致性判断标准（基于业界最佳实践）**

**核心原则：短期触发 + 中期确认 + ML验证**

- **强买入信号**：短期建议买入 AND 中期建议买入 AND (至少一个ML模型预测上涨且probability>0.62)
- **买入信号**：短期建议买入 AND 中期建议买入 AND (至少一个ML模型预测上涨且probability>0.60)
- **观望信号**：
  - 短期建议买入 AND 中期建议观望（等待中期确认）
  - 短期建议卖出 AND 中期建议观望（等待中期确认）
  - 短期建议买入 AND 中期建议卖出（冲突）
  - 短期建议卖出 AND 中期建议买入（冲突）
  - ML模型probability在0.45-0.55之间（低置信度）
  - 两个ML模型预测冲突（信号不一致）
- **卖出信号**：短期建议卖出 AND 中期建议卖出 AND (至少一个ML模型预测下跌且probability<0.40)

- **阈值优化说明**：
- 当前20天模型准确率：LightGBM {model_accuracy['lgbm']['accuracy']:.2%}（标准差±{model_accuracy['lgbm']['std']:.2%}），GBDT {model_accuracy['gbdt']['accuracy']:.2%}（标准差±{model_accuracy['gbdt']['std']:.2%}）
- 强买入阈值0.62略高于准确率，确保高置信度
- 买入阈值0.60接近准确率，平衡召回率和精确率
- 卖出阈值0.40确保下跌概率>60%
- 观望区间0.45-0.55避免低置信度决策

**重要说明 - LR算法probability含义**：
- probability字段始终代表**上涨概率**P(y=1|x)
- 当prediction=1时：probability > 0.5（上涨概率高）
- 当prediction=0时：probability <= 0.5（上涨概率低，即下跌概率高）
- 强烈下跌信号：prediction=0且probability < 0.40（即下跌概率 > 60%）
- 中性信号：probability在0.40-0.60之间（上涨或下跌概率都不超过60%）

**重要说明 - 信号优先级（业界标准）**：
- **短期信号（触发器）**：负责"何时做"（Timing），权重100%（必须满足）
- **中期信号（确认器）**：负责"是否做"（Direction），权重100%（必须满足）
- **ML预测（验证器）**：负责提升置信度，权重50%（辅助验证）
- **关键原则**：短期和中期必须一致（方向相同），ML预测用于验证和提升置信度

**重要说明 - 模型不确定性**：
- ML 20天模型准确率：LightGBM {model_accuracy['lgbm']['accuracy']:.2%}（标准差±{model_accuracy['lgbm']['std']:.2%}），GBDT {model_accuracy['gbdt']['accuracy']:.2%}（标准差±{model_accuracy['gbdt']['std']:.2%}）
- 即使probability>0.62，实际准确率也可能在{model_accuracy['lgbm']['accuracy']-model_accuracy['lgbm']['std']:.2%} ~ {model_accuracy['lgbm']['accuracy']+model_accuracy['lgbm']['std']:.2%}（LightGBM）或{model_accuracy['gbdt']['accuracy']-model_accuracy['gbdt']['std']:.2%} ~ {model_accuracy['gbdt']['accuracy']+model_accuracy['gbdt']['std']:.2%}（GBDT）之间波动
- 建议：短期和中期一致是主要决策依据，ML预测用于验证和提升置信度
- 对于probability在0.55-0.65之间的股票，建议降低仓位控制风险

**重要说明 - 时间维度标准化**：
- 短期：1-5个交易日（日内到一周）
- 中期：10-20个交易日（2-4周）
- 长期：>20个交易日（超过1个月）
- 当前映射：大模型短期建议 ↔ ML次日模型（1天），大模型中期建议 ↔ ML 20天模型（20天）✅

**规则3：ML模型冲突处理**
- 如果LightGBM和GBDT预测冲突（一个上涨，一个下跌）：
  - 优先相信预测概率更高的模型
  - 如果概率相近（相差<0.10），则参考大模型中期建议
- 如果两个ML模型预测一致（都上涨或都下跌）：
  - 信号可靠性高，优先级提升

**规则4：推荐理由格式**
- 必须说明：短期建议+中期建议+哪个ML模型预测+短期中期一致性程度
- 例如："短期建议买入（触发器），中期建议买入（确认器），短期中期方向一致，LightGBM预测上涨概率0.72，GBDT预测上涨概率0.68，三重确认买入，综合置信度高"

请基于上述规则，完成以下任务：

1. **一致性分析**（方案A核心：短期触发 + 中期确认）：
   - **第一步（核心）**：分析短期建议与中期建议的一致性
     - 短期买入 + 中期买入 → 方向一致，考虑ML验证
     - 短期买入 + 中期观望 → 等待中期确认
     - 短期买入 + 中期卖出 → 冲突，观望
     - 短期卖出 + 中期卖出 → 方向一致，考虑ML验证
     - 短期卖出 + 中期观望 → 等待中期确认
     - 短期卖出 + 中期买入 → 冲突，观望
   - **第二步（验证）**：对短期中期一致的股票，分析ML预测验证
     - 如果ML模型预测支持（probability>0.60），提升为强信号
     - 如果ML模型预测冲突（probability<0.40），降低为弱信号或观望
     - 如果ML模型不确定（0.45-0.55），保持中等置信度
   - 标注符合"强买入信号"、"买入信号"、"观望信号"、"卖出信号"的股票

2. **个股建议排序**：
   - 优先级：强买入信号 > 买入信号 > 观望信号 > 卖出信号
   - 在相同优先级内，按ML预测概率排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 强烈买入信号（2-3只）：最高优先级
   - 买入信号（3-5只）：次优先级
   - 持有/观望（如有）：第三优先级
   - 卖出信号（如有）：最低优先级

4. **风险提示**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比）
   - 给出止损位建议（如果有的话）

请按照以下格式输出（不要添加任何额外说明文字）：

2. **个股建议排序**：
   - 按照"强买入信号 > 中等信号 > 观望"的优先级排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 推荐买入的股票清单（按优先级排序）
   - 推荐卖出的股票清单（如有）
   - 需要关注的股票清单（观望）

4. **风险控制建议**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比）
   - 给出止损位建议（如果有的话）
   
   **特别要求 - 考虑模型不确定性**：
   - ML 20天模型标准差为±{model_accuracy['lgbm']['std']:.2%}（LightGBM）/±{model_accuracy['gbdt']['std']:.2%}（GBDT）
   - 对于probability在0.55-0.65之间的股票，建议仓位不超过2-3%
   - 强买入信号（短期/中期一致买入且ML模型确认）建议仓位4-6%
   - 总仓位控制在45%-55%
   - 必须设置止损位，单只股票最大亏损不超过-8%
   - **严格遵循"短期触发 + 中期确认"原则**：只有短期和中期方向一致时才行动，冲突时选择观望
   - 如果短期和中期建议冲突，优先选择观望，不进行交易
   - 采用"三重确认"策略：短期、中期、ML模型三者一致时才重仓操作

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由，必须说明：短期建议+中期建议+ML预测+一致性程度。例如："短期建议买入（触发器），中期建议买入（确认器），LightGBM预测上涨概率0.72，GBDT预测上涨概率0.68，短中长期方向一致（短期/中期一致买入，ML模型验证上涨），综合置信度高。注意ML模型当前准确率{model_accuracy['lgbm']['accuracy']:.2%}（标准差±{model_accuracy['lgbm']['std']:.2%}），probability在0.72附近实际准确率可能在{model_accuracy['lgbm']['accuracy']-model_accuracy['lgbm']['std']:.2%} ~ {model_accuracy['lgbm']['accuracy']+model_accuracy['lgbm']['std']:.2%}之间"]
   - 操作建议：买入/卖出/持有/观望
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 买入信号（3-5只）
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由]
   - 操作建议：买入/持有
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 持有/观望
1. [股票代码] [股票名称] 
   - 推荐理由：[观望理由]
   - 操作建议：持有/观望
   - 关注要点：[需要关注的关键指标或事件]
   - 风险提示：[主要风险因素]

## 卖出信号（如有）
1. [股票代码] [股票名称] 
   - 推荐理由：[卖出理由]
   - 操作建议：卖出/减仓
   - 建议卖出价：HK$XX.XX
   - 止损位（如持有）：HK$XX.XX（-X.X%）
   - 风险提示：[主要风险因素]

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 建议仓位百分比：[X]%
- 止损位设置：[策略]
- 组合调整建议：[具体的组合调整建议]

---
分析日期：{date_str}
"""
        
        print("🤖 提交大模型进行综合分析...")
        print("")
        
        # 调用大模型（关闭思考模式，避免输出思考过程）
        response = chat_with_llm(prompt, enable_thinking=False)
        
        if response:
            print("✅ 综合分析完成\n")
            print("=" * 80)
            print("📊 综合买卖建议")
            print("=" * 80)
            print("")
            print(response)
            print("")
            print("=" * 80)
            
            # 保存到文件
            if output_filepath is None:
                output_filepath = f'data/comprehensive_recommendations_{date_str}.txt'
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(f"{'=' * 80}\n")
                f.write(f"综合买卖建议\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析日期: {date_str}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(response)
            
            print(f"✅ 综合建议已保存到 {output_filepath}")
            
            # 发送邮件通知
            if send_email_flag:
                print("\n📧 准备发送邮件通知...")
                email_subject = f"【综合分析】港股买卖建议 - {date_str}"
                email_content = response
                
                # 构建板块分析、股息信息、恒生指数分析
                print("📊 获取板块分析...")
                sector_data = get_sector_analysis()
                
                print("📊 获取股息信息...")
                dividend_data = get_dividend_info()
                
                print("📊 获取恒生指数分析...")
                hsi_data = get_hsi_analysis()
                
                # 构建板块分析文本
                sector_text = ""
                if sector_data and sector_data['performance'] is not None:
                    sector_text = "\n## 三、板块分析（5日涨跌幅排名）\n"
                    perf_df = sector_data['performance']
                    sector_leaders = sector_data['leaders']
                    
                    sector_text += "| 排名 | 板块名称 | 平均涨跌幅 | 龙头股TOP 3 |\n"
                    sector_text += "|------|---------|-----------|-------------|\n"
                    
                    for idx, row in perf_df.iterrows():
                        trend_icon = "🔥" if row['avg_change_pct'] > 2 else "📈" if row['avg_change_pct'] > 0 else "📉"
                        change_color = "+" if row['avg_change_pct'] > 0 else ""
                        
                        leaders_text = ""
                        if row['sector_code'] in sector_leaders:
                            leaders = sector_leaders[row['sector_code']]
                            # 显示所有3个龙头股，使用斜线分隔避免与Markdown表格冲突
                            leader_items = []
                            for i, leader in enumerate(leaders, 1):
                                leader_items.append(f"{leader['name']}({leader['change_pct']:+.1f}%)")
                            leaders_text = " / ".join(leader_items)
                        
                        sector_text += f"| {idx+1} | {trend_icon} {row['sector_name']} | {change_color}{row['avg_change_pct']:.2f}% | {leaders_text} |\n"
                    
                    # 添加投资建议
                    top_sector = perf_df.iloc[0]
                    bottom_sector = perf_df.iloc[-1]
                    
                    sector_text += "\n**投资建议**：\n"
                    if top_sector['avg_change_pct'] > 1:
                        sector_text += f"- 当前热点板块：{top_sector['sector_name']}，平均涨幅 {top_sector['avg_change_pct']:.2f}%\n"
                        if top_sector['sector_code'] in sector_leaders and sector_leaders[top_sector['sector_code']]:
                            leader = sector_leaders[top_sector['sector_code']][0]
                            sector_text += f"- 建议关注该板块的龙头股：{leader['name']} ⭐\n"
                    
                    if bottom_sector['avg_change_pct'] < -1:
                        sector_text += f"- 当前弱势板块：{bottom_sector['sector_name']}，平均跌幅 {bottom_sector['avg_change_pct']:.2f}%\n"
                        sector_text += "- 建议谨慎操作该板块，等待企稳信号\n"
                
                # 构建股息信息文本
                dividend_text = ""
                if dividend_data:
                    dividend_text = "\n## 四、股息信息（即将除净）\n"
                    dividend_text += "| 股票代码 | 股票名称 | 除净日 | 股息率 |\n"
                    dividend_text += "|---------|---------|-------|--------|\n"
                    
                    for stock in dividend_data[:10]:
                        code = stock.get('A股代码', 'N/A')
                        name = stock.get('A股简称', 'N/A')
                        ex_date = stock.get('除权除息日', 'N/A')
                        div_rate = stock.get('股息率', 'N/A')
                        dividend_text += f"| {code} | {name} | {ex_date} | {div_rate} |\n"
                
                # 构建恒生指数分析文本
                hsi_text = ""
                if hsi_data:
                    hsi_text = "\n## 五、恒生指数技术分析\n"
                    hsi_text += f"- 当前价格：{hsi_data['current_price']:.2f}\n"
                    hsi_text += f"- 日涨跌幅：{hsi_data['change_pct']:+.2f}%\n"
                    hsi_text += f"- RSI（14日）：{hsi_data['rsi']:.2f}\n"
                    hsi_text += f"- MA20：{hsi_data['ma20']:.2f}\n"
                    hsi_text += f"- MA50：{hsi_data['ma50']:.2f}\n"
                    hsi_text += f"- 趋势：{hsi_data['trend']}\n"
                
                # 从大模型响应中提取股票代码
                import re
                stock_codes = re.findall(r'\b\d{4}\.HK\b', response)
                # 去重
                stock_codes = list(set(stock_codes))
                
                # 生成技术指标表格
                print("📊 生成推荐股票技术指标表格...")
                technical_indicators_table = generate_technical_indicators_table(stock_codes)
                
                # 构建完整的邮件内容（综合买卖建议 + 信息参考）
                # 注意：不添加标题，因为HTML模板已经有了标题
                full_content = f"""{response}

---

# 信息参考

## 一、大模型建议

### 短期买卖建议（日内/数天）
{llm_recommendations['short_term']}

### 中期买卖建议（数周-数月）
{llm_recommendations['medium_term']}

## 二、机器学习预测结果（20天）

### LightGBM模型
准确率：{model_accuracy['lgbm']['accuracy']:.2%}（标准差±{model_accuracy['lgbm']['std']:.2%}）
{ml_predictions['lgbm']}

### GBDT模型
准确率：{model_accuracy['gbdt']['accuracy']:.2%}（标准差±{model_accuracy['gbdt']['std']:.2%}）
{ml_predictions['gbdt']}
{sector_text}
{dividend_text}
{hsi_text}
{technical_indicators_table}
## 六、技术指标说明

**短期技术指标（日内/数天）**：
- RSI（相对强弱指数）：超买>70，超卖<30
- MACD：金叉（上涨信号），死叉（下跌信号）
- 布林带：价格突破上下轨预示反转
- 成交量：放大配合价格上涨=买入信号
- OBV（能量潮）：反映资金流向

**中期技术指标（数周-数月）**：
- 均线排列：多头排列（MA5>MA10>MA20>MA50）= 上升趋势
- 均线斜率：上升=趋势向上，下降=趋势向下
- 乖离率：价格偏离均线的程度
- 支撑阻力位：重要价格支撑和阻力
- 相对强度：相对于恒生指数的表现
- 中期趋势评分：0-100分，≥80买入，30-45卖出

**重要说明**：
- 短期指标用于捕捉买卖时机（Timing）
- 中期指标用于确认趋势方向（Direction）
- 短期和中期方向一致时，信号最可靠
- 短期和中期冲突时，选择观望

## 七、风险提示

1. **模型不确定性**：
   - ML 20天模型标准差为±{model_accuracy['lgbm']['std']:.2%}（LightGBM）/±{model_accuracy['gbdt']['std']:.2%}（GBDT）
   - 即使probability>0.62，实际准确率也可能在{model_accuracy['lgbm']['accuracy']-model_accuracy['lgbm']['std']:.2%} ~ {model_accuracy['lgbm']['accuracy']+model_accuracy['lgbm']['std']:.2%}之间波动
   - 建议：短期和中期一致是主要决策依据，ML预测用于验证和提升置信度

2. **市场风险**：
   - 当前市场整体风险：[高/中/低]（需根据恒生指数技术指标判断）
   - 建议仓位：45%-55%
   - 必须设置止损位，单只股票最大亏损不超过-8%

3. **投资原则**：
   - 短期触发 + 中期确认 + ML验证 = 高置信度信号
   - 短期和中期冲突 = 观望（避免不确定性）
   - 概率在0.45-0.55之间 = 低置信度，不建议操作
   - 总仓位控制在45%-55%，分散风险

## 八、数据来源

- 大模型分析：Qwen大模型
- ML预测：LightGBM + GBDT（2991个特征，500个精选特征）
- 技术指标：RSI、MACD、布林带、ATR、均线、成交量等80+个指标
- 基本面数据：PE、PB、ROE、ROA、股息率等8个指标
- 美股市场：标普500、纳斯达克、VIX、美国国债收益率等11个指标
- 股票类型：18个行业分类及衍生评分
- 情感分析：四维情感评分（Relevance/Impact/Expectation_Gap/Sentiment）
- 板块分析：16个板块涨跌幅排名、技术趋势分析、龙头识别
- 主题建模：LDA主题建模（10个主题）
- 主题情感交互：10个主题 × 5个情感指标 = 50个交互特征
- 预期差距：新闻情感相对于市场预期的差距（5个特征）

---
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析日期：{date_str}
"""
                
                # 生成HTML格式邮件内容（将完整内容转换为HTML）
                html_content = generate_html_email(full_content, date_str)
                send_email(email_subject, full_content, html_content)
            
            return response
        else:
            print("❌ 大模型分析失败")
            return None
        
    except Exception as e:
        print(f"❌ 综合分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='综合分析脚本 - 整合大模型建议和ML预测结果')
    parser.add_argument('--llm-file', type=str, default=None, 
                       help='大模型建议文件路径 (默认使用今天的文件)')
    parser.add_argument('--ml-file', type=str, default=None,
                       help='ML预测结果文件路径 (默认使用今天的文件)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认保存到data/comprehensive_recommendations_YYYY-MM-DD.txt)')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件通知')
    
    args = parser.parse_args()
    
    # 生成日期
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # 默认文件路径
    if args.llm_file is None:
        args.llm_file = f'data/llm_recommendations_{date_str}.txt'
    
    if args.ml_file is None:
        args.ml_file = f'data/ml_predictions_20d_{date_str}.txt'
    
    # 运行综合分析
    result = run_comprehensive_analysis(args.llm_file, args.ml_file, args.output, 
                                       send_email_flag=not args.no_email)
    
    if result:
        print("\n✅ 综合分析完成！")
        sys.exit(0)
    else:
        print("\n❌ 综合分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()