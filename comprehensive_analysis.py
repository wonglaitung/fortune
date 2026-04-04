#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入大模型服务
from llm_services.qwen_engine import chat_with_llm

# 导入配置
from config import WATCHLIST

# 从WATCHLIST提取股票名称映射
STOCK_NAMES = WATCHLIST
STOCK_LIST = WATCHLIST  # 为兼容 hsi_email 模块添加别名

# 导入必要的模块
try:
    from data_services.hk_sector_analysis import SectorAnalyzer
    SECTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    SECTOR_ANALYSIS_AVAILABLE = False
    print("⚠️ 板块分析模块不可用")

try:
    from detect_stock_anomalies import StockAnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    print("⚠️ 异常检测模块不可用")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️ AKShare模块不可用")

# 导入技术分析工具（用于筹码分布分析）
try:
    from data_services.technical_analysis import TechnicalAnalyzer
    from data_services.tencent_finance import get_hk_stock_data_tencent
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析模块不可用")


def safe_float_format(value, format_spec='.2f', default=''):
    """
    安全地格式化浮点数值，处理可能的字符串或非数值类型
    
    参数:
    - value: 要格式化的值
    - format_spec: 格式化规格，默认为'.2f'（可以是 '+.2f' 等带符号的格式）
    - default: 格式化失败时的默认返回值，默认为空字符串
    
    返回:
    - 格式化后的字符串，或默认值
    """
    try:
        if pd.isna(value) or value is None or value == '':
            return default
        # 尝试转换为浮点数并格式化
        float_value = float(value)
        # 直接使用 format_spec 作为格式说明符
        format_str = f"{{:{format_spec}}}"
        return format_str.format(float_value)
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        return default


def load_model_accuracy(horizon=20):
    """
    从文件加载模型准确率信息
    
    参数:
    - horizon: 预测周期（默认20天）
    
    返回:
    - dict: 包含LightGBM、GBDT和CatBoost准确率的字典
      {
        'lgbm': {'accuracy': float, 'std': float},
        'gbdt': {'accuracy': float, 'std': float},
        'catboost': {'accuracy': float, 'std': float}
      }
    """
    # 默认准确率值（如果文件不存在）
    default_accuracy = {
        'catboost': {'accuracy': 0.6101, 'std': 0.0219}
    }
    
    accuracy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_accuracy.json')
    
    try:
        if os.path.exists(accuracy_file):
            import json
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = {}
            catboost_key = f'catboost_{horizon}d'
            
            if catboost_key in data:
                result['catboost'] = {
                    'accuracy': data[catboost_key].get('accuracy', default_accuracy['catboost']['accuracy']),
                    'std': data[catboost_key].get('std', default_accuracy['catboost']['std'])
                }
            else:
                result['catboost'] = default_accuracy['catboost']
            
            print(f"✅ 已加载模型准确率: {accuracy_file}")
            print(f"   CatBoost: {result['catboost']['accuracy']:.2%} (±{result['catboost']['std']:.2%})")
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
        
        # 查找"稳健型短期分析"和"稳健型中期分析"的位置
        short_start = content.find("### 📊 ⚖️ 稳健型短期分析（日内/数天）")
        medium_start = content.find("### 📊 📊 稳健型中期分析（数周-数月）")
        
        if short_start == -1:
            short_start = content.find("### 稳健型短期分析")
        
        if medium_start == -1:
            medium_start = content.find("### 稳健型中期分析")
        
        short_content = ""
        medium_content = ""
        
        if short_start != -1:
            if medium_start != -1:
                # 提取短期分析内容（从短期分析标题后到中期分析标题前）
                short_content = content[short_start:medium_start].split('\n', 1)[-1].strip()  # 去掉标题行
            else:
                # 如果没有中期分析，提取到文件末尾
                short_content = content[short_start:].split('\n', 1)[-1].strip()  # 去掉标题行
        
        if medium_start != -1:
            # 提取中期分析内容（从中期分析标题后到文件末尾）
            medium_content = content[medium_start:].split('\n', 1)[-1].strip()  # 去掉标题行
        
        result = {
            'short_term': short_content,
            'medium_term': medium_content
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return {'short_term': '', 'medium_term': ''}


def extract_ml_predictions(filepath):
    """
    从ML预测CSV文件中提取融合模型的预测结果
    
    参数:
    - filepath: 文本预测文件路径（用于获取日期）
    
    返回:
    - dict: 包含融合模型预测结果的字典
      {
        'ensemble': str,  # 融合模型预测结果
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
        
        # 读取 CatBoost 单模型预测结果
        catboost_csv = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_20d.csv')
        
        result = {
            'ensemble': ''
        }
        
        # 读取 CatBoost 预测结果
        if os.path.exists(catboost_csv):
            df_catboost = pd.read_csv(catboost_csv)
            df_catboost_sorted = df_catboost.sort_values('probability', ascending=False)

            # 计算筹码分布（如果技术分析模块可用）
            chip_data = {}
            if TECHNICAL_ANALYSIS_AVAILABLE:
                try:
                    analyzer = TechnicalAnalyzer()
                    for stock_code in df_catboost['code'].tolist():
                        try:
                            # 获取股票数据（60天）
                            stock_df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=60)
                            if not stock_df.empty and len(stock_df) >= 20:
                                chip_result = analyzer.get_chip_distribution(stock_df)
                                if chip_result:
                                    chip_data[stock_code] = chip_result
                        except Exception as e:
                            print(f"  ⚠️ 计算 {stock_code} 筹码分布失败: {e}")
                            chip_data[stock_code] = None
                except Exception as e:
                    print(f"  ⚠️ 筹码分布计算失败: {e}")

            catboost_text = "【CatBoost模型预测结果（20天）】\n"
            catboost_text += f"预测日期: {date_str}\n\n"
            catboost_text += "全部股票预测结果（按概率排序）:\n\n"

            # 构建Markdown表格（添加阻力标识列）
            catboost_text += "| 股票代码 | 股票名称 | 预测方向 | 上涨概率 | 当前价格 | 阻力标识 |\n"
            catboost_text += "|----------|----------|----------|----------|----------|----------|\n"

            # 统计筹码分布
            resistance_stats = {'low': 0, 'medium': 0, 'high': 0}
            high_resistance_stocks = []

            for _, row in df_catboost_sorted.iterrows():
                if row['probability'] > 0.60:
                    direction = "上涨"
                elif row['probability'] > 0.50:
                    direction = "观望"
                else:
                    direction = "下跌"

                # 计算阻力标识
                resistance_icon = 'N/A'
                if TECHNICAL_ANALYSIS_AVAILABLE and row['code'] in chip_data and chip_data[row['code']]:
                    chip_result = chip_data[row['code']]
                    resistance_ratio = chip_result.get('resistance_ratio', 0)
                    if resistance_ratio < 0.3:
                        resistance_stats['low'] += 1
                        resistance_icon = '✅'
                    elif resistance_ratio < 0.6:
                        resistance_stats['medium'] += 1
                        resistance_icon = '⚠️'
                    else:
                        resistance_stats['high'] += 1
                        resistance_icon = '🔴'
                        # 记录高阻力股票
                        high_resistance_stocks.append({
                            'code': row['code'],
                            'name': row['name'],
                            'resistance_ratio': resistance_ratio
                        })

                # 为上涨概率添加颜色标记
                probability = row['probability']
                probability_formatted = safe_float_format(probability, '4f')
                if probability > 0.60:
                    probability_colored = f'<span style="color: green; font-weight: bold;">{probability_formatted}</span>'
                elif probability > 0.55:
                    probability_colored = f'<span style="color: orange; font-weight: bold;">{probability_formatted}</span>'
                else:
                    probability_colored = f'<span style="color: red; font-weight: bold;">{probability_formatted}</span>'

                catboost_text += f"| {row['code']} | {row['name']} | {direction} | {probability_colored} | {safe_float_format(row['current_price'], '2f')} | {resistance_icon} |\n"

            catboost_text += f"\n**统计信息**：\n"
            catboost_text += f"- 高置信度上涨（概率 > 0.60）: {len(df_catboost[df_catboost['probability'] > 0.60])} 只\n"
            catboost_text += f"- 中等置信度观望（0.50 < 概率 ≤ 0.60）: {len(df_catboost[(df_catboost['probability'] > 0.50) & (df_catboost['probability'] <= 0.60)])} 只\n"
            catboost_text += f"- 预测下跌（概率 ≤ 0.50）: {len(df_catboost[df_catboost['probability'] <= 0.50])} 只\n"

            # 添加筹码分布摘要
            if TECHNICAL_ANALYSIS_AVAILABLE and resistance_stats['low'] + resistance_stats['medium'] + resistance_stats['high'] > 0:
                catboost_text += f"\n**筹码分布摘要**：\n"
                catboost_text += f"- 低阻力股票（上方筹码 < 30%）: {resistance_stats['low']} 只 ✅\n"
                catboost_text += f"- 中等阻力股票（30-60%）: {resistance_stats['medium']} 只 ⚠️\n"
                catboost_text += f"- 高阻力股票（上方筹码 > 60%）: {resistance_stats['high']} 只 🔴\n"

                # 列出高阻力股票（按上方筹码比例降序）
                if high_resistance_stocks:
                    # 按上方筹码比例降序排序
                    high_resistance_stocks_sorted = sorted(high_resistance_stocks, key=lambda x: x['resistance_ratio'], reverse=True)
                    catboost_text += f"\n**高阻力股票列表**（按上方筹码比例降序）：\n"
                    catboost_text += '<table>\n'
                    catboost_text += '<tr><th>股票代码</th><th>股票名称</th><th>上方筹码比例</th><th>拉升难度</th></tr>\n'
                    for stock in high_resistance_stocks_sorted:
                        difficulty = "困难" if stock['resistance_ratio'] > 0.6 else "中等" if stock['resistance_ratio'] > 0.3 else "容易"
                        catboost_text += f'<tr><td>{stock["code"]}</td><td>{stock["name"]}</td><td>{stock["resistance_ratio"]:.1%}</td><td>{difficulty}</td></tr>\n'
                    catboost_text += '</table>\n'

                catboost_text += f"\n**阻力标识说明**：\n"
                catboost_text += "- ✅：低阻力（< 30%），拉升容易\n"
                catboost_text += "- ⚠️：中等阻力（30-60%），注意风险\n"
                catboost_text += "- 🔴：高阻力（> 60%），拉升困难\n"

            result['ensemble'] = catboost_text
        else:
            print(f"⚠️ CatBoost 预测文件不存在: {catboost_csv}")
            
            result['ensemble'] = ''
        
        return result
        
    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'ensemble': ''}


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
            
            # 先尝试使用默认市值阈值
            leaders_df = sector_analyzer.identify_sector_leaders(
                sector_code=sector_code,
                top_n=3,
                period=5,
                min_market_cap=100,
                style='moderate'
            )
            
            # 如果没有找到龙头股，可能是市值太小，降低阈值重试
            if leaders_df.empty:
                print(f"  ⚠️ 板块 {row['sector_name']}({sector_code}) 首次查询未找到龙头股，尝试降低市值阈值")
                # 尝试降低市值阈值
                for min_cap in [50, 20, 10, 5, 1]:
                    leaders_df = sector_analyzer.identify_sector_leaders(
                        sector_code=sector_code,
                        top_n=3,
                        period=5,
                        min_market_cap=min_cap,
                        style='moderate'
                    )
                    if not leaders_df.empty:
                        print(f"    找到 {len(leaders_df)} 只龙头股（市值阈值 {min_cap}亿港币）")
                        break
            
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
        import time
        
        # 获取自选股列表
        stock_list = WATCHLIST
        all_dividends = []
        
        # 对每只自选股查询股息信息
        for stock_code, stock_name in stock_list.items():
            try:
                # 提取数字部分并格式化为5位（与hsi_email.py保持一致）
                symbol = stock_code.replace('.HK', '')
                if len(symbol) < 5:
                    symbol = symbol.zfill(5)
                elif len(symbol) > 5:
                    symbol = symbol[-5:]
                
                # 使用港股股息接口
                df_dividend = ak.stock_hk_dividend_payout_em(symbol=symbol)
                
                if df_dividend is not None and not df_dividend.empty:
                    # 检查数据列
                    available_columns = df_dividend.columns.tolist()
                    
                    # 创建结果列表
                    result_data = []
                    for _, row in df_dividend.iterrows():
                        try:
                            # 提取关键信息（与hsi_email.py保持一致）
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
                    
                    if result_data:
                        all_dividends.append(pd.DataFrame(result_data))
                
                # 避免请求过于频繁
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ 获取 {stock_name}({stock_code}) 股息信息失败: {e}")
                continue
        
        if not all_dividends:
            return None
        
        # 合并所有数据
        all_dividends_df = pd.concat(all_dividends, ignore_index=True)
        
        # 转换日期格式
        all_dividends_df['除净日'] = pd.to_datetime(all_dividends_df['除净日'])
        
        # 筛选未来90天内的除净日
        today = datetime.now()
        future_date = today + timedelta(days=90)
        
        upcoming_dividends = all_dividends_df[
            (all_dividends_df['除净日'] >= today) & 
            (all_dividends_df['除净日'] <= future_date)
        ].sort_values('除净日')
        
        if upcoming_dividends.empty:
            return None
        
        # 只取前10个，转换为字典列表
        return upcoming_dividends.head(10).to_dict('records')
        
    except Exception as e:
        print(f"⚠️ 获取股息信息失败: {e}")
        import traceback
        traceback.print_exc()
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
        change_points = latest['Close'] - prev['Close']
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
            'change_points': change_points,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'ma20': ma20,
            'ma50': ma50,
            'trend': trend
        }
    except Exception as e:
        print(f"⚠️ 获取恒生指数分析失败: {e}")
        return None


def get_current_market_state():
    """
    获取当前市场状态（实时）
    
    返回:
    dict: 当前市场状态信息
    """
    try:
        # 获取最近30天的恒生指数数据（日线数据用于计算20天收益率）
        hsi_ticker = yf.Ticker("^HSI")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hsi_df = hsi_ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                    end=end_date.strftime('%Y-%m-%d'))
        
        if len(hsi_df) < 10:
            return None
        
        # 获取实时数据（1分钟间隔，用于显示当前价格）
        real_time_df = hsi_ticker.history(period='1d', interval='1m')
        
        # 计算最近20天收益率（使用日线数据）
        if len(hsi_df) >= 20:
            recent_20d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[-20]) / hsi_df['Close'].iloc[-20]
        else:
            recent_20d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[0]) / hsi_df['Close'].iloc[0]
        
        # 计算最近5天收益率（使用日线数据）
        if len(hsi_df) >= 5:
            recent_5d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[-5]) / hsi_df['Close'].iloc[-5]
        else:
            recent_5d_return = (hsi_df['Close'].iloc[-1] - hsi_df['Close'].iloc[0]) / hsi_df['Close'].iloc[0]
        
        # 获取实时价格（优先使用分钟级数据）
        current_hsi = None
        current_time = None
        
        if not real_time_df.empty:
            current_hsi = real_time_df['Close'].iloc[-1]
            # 转换时区到香港时间
            current_time = real_time_df.index[-1].tz_convert('Asia/Hong_Kong')
        
        # 如果没有实时数据，使用日线数据
        if current_hsi is None and not hsi_df.empty:
            current_hsi = hsi_df['Close'].iloc[-1]
            current_time = hsi_df.index[-1].tz_convert('Asia/Hong_Kong')
        
        # 计算当前市场状态
        if recent_20d_return > 0.05:
            market_state = 'bull'
            market_state_cn = '牛市'
            market_signal = '📈 强烈看涨'
        elif recent_20d_return < -0.05:
            market_state = 'bear'
            market_state_cn = '熊市'
            market_signal = '📉 强烈看跌'
        elif recent_20d_return > 0.02:
            market_state = 'neutral_bull'
            market_state_cn = '震荡偏涨'
            market_signal = '⬆️ 温和上涨'
        elif recent_20d_return < -0.02:
            market_state = 'neutral_bear'
            market_state_cn = '震荡偏跌'
            market_signal = '⬇️ 温和下跌'
        else:
            market_state = 'neutral'
            market_state_cn = '震荡市'
            market_signal = '➡️ 横盘整理'
        
        # 格式化时间（如果存在）
        date_str = current_time.strftime('%Y-%m-%d %H:%M:%S HKT') if current_time else 'N/A'
        
        return {
            'market_state': market_state,
            'market_state_cn': market_state_cn,
            'market_signal': market_signal,
            'recent_20d_return': recent_20d_return,
            'recent_5d_return': recent_5d_return,
            'current_hsi': current_hsi,
            'date': date_str
        }
    except Exception as e:
        print(f"⚠️ 获取当前市场状态失败: {e}")
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


def get_bollinger_position_cn(position: str) -> str:
    """获取布林带位置描述（中文）"""
    if position == 'upper':
        return "上轨（强势）"
    elif position == 'lower':
        return "下轨（弱势）"
    else:
        return "中轨（正常）"


def get_stock_anomalies(use_deep_analysis=True):
    """
    获取股票异常检测结果（P2集成功能）
    
    参数:
    - use_deep_analysis: 是否使用深度分析模式（默认True）
    
    返回:
    - list: 异常列表，格式为：
        [
            {
                'stock': '0700.HK',
                'name': '腾讯控股',
                'type': 'price' or 'volume' or 'isolation_forest',  # 异常类型
                'severity': 'high' or 'medium',
                'z_score': 4.2,
                'value': 320.50,  # 异常值（价格或成交量）
                'current_price': 320.50,
                'change_1d': 3.2,
                'change_5d': -5.8,
                'rsi': 68.5,
                'bollinger_position': 'upper',
                'macd_signal': 'bullish',
                'timestamp': datetime object
            }
        ]
        或 None（如果模块不可用）
    """
    try:
        if not ANOMALY_DETECTION_AVAILABLE:
            print("⚠️ 异常检测模块不可用，跳过异常检测")
            return None
        
        # 创建检测器实例（支持深度分析模式）
        detector = StockAnomalyDetector(
            window_size=30,
            threshold_high=4.0,
            threshold_medium=3.0,
            use_deep_analysis=use_deep_analysis  # 新增参数
        )
        
        # 获取自选股列表
        from config import WATCHLIST
        stocks = list(WATCHLIST.keys())
        
        # 检测异常（使用较短的数据周期以提高速度）
        anomalies = detector.detect_anomalies(stocks, period='3mo')
        
        print(f"📊 异常检测完成（模式: {'深度' if use_deep_analysis else '快速'}）：检测到 {len(anomalies)} 个异常")
        
        # 按类型和严重程度统计
        if anomalies:
            type_stats = {}
            for anomaly in anomalies:
                anomaly_type = anomaly.get('type', 'unknown')
                if anomaly_type not in type_stats:
                    type_stats[anomaly_type] = {'high': 0, 'medium': 0}
                type_stats[anomaly_type][anomaly['severity']] += 1
            
            # 打印统计信息
            for anomaly_type, stats in type_stats.items():
                type_name = {
                    'price': '价格',
                    'volume': '成交量',
                    'isolation_forest': 'Isolation Forest'
                }.get(anomaly_type, anomaly_type)
                print(f"   {type_name}异常：高 {stats['high']} 个，中 {stats['medium']} 个")
        
        # 打印异常详情
        for anomaly in anomalies:
            severity_icon = '🔴' if anomaly['severity'] == 'high' else '⚠️'
            type_name = {
                'price': '价格',
                'volume': '成交量',
                'isolation_forest': 'IF'
            }.get(anomaly.get('type', 'unknown'), '未知')
            
            if anomaly.get('type') == 'isolation_forest':
                print(f"   {severity_icon} {anomaly['name']}（{anomaly['stock']}）：{type_name}{anomaly['severity']}（Z-Score: N/A）")
            else:
                print(f"   {severity_icon} {anomaly['name']}（{anomaly['stock']}）：{type_name}{anomaly['severity']}（Z-Score: {anomaly['z_score']:.2f}）")
        
        return anomalies
        
    except Exception as e:
        print(f"⚠️ 获取股票异常检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_hsi_email_indicators():
    """
    从 hsi_email.py 获取实时指标
    """
    try:
        from hsi_email import get_hsi_and_stock_indicators
        # 调用hsi_email模块的指标获取函数
        indicators = get_hsi_and_stock_indicators()
        return indicators
    except Exception as e:
        print(f"⚠️ 获取 hsi_email.py 实时指标失败: {e}")
        import traceback
        traceback.print_exc()
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
        
        # 获取股票数据 - 使用完整的股票代码（带.HK）
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            print(f"⚠️ 警告: 无法获取 {stock_code} 的历史数据")
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
        # OBV 需要对整个历史数据计算，而不是只计算最新一天
        obv_series = ((hist['Close'].diff() > 0).astype(int) * 2 - 1) * hist['Volume']
        obv = (obv_series.cumsum() / 1e6).iloc[-1] if len(hist) > 0 else 0
        
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


def get_recent_transactions(hours=48):
    """
    获取最近指定小时数的模拟交易记录
    
    参数:
    - hours: 查询的小时数，默认48小时
    
    返回:
    - DataFrame: 交易记录数据框
    """
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 交易记录文件路径
        transactions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'simulation_transactions.csv')
        
        if not os.path.exists(transactions_file):
            print(f"⚠️ 交易记录文件不存在: {transactions_file}")
            return pd.DataFrame()
        
        # 读取交易记录
        df = pd.read_csv(transactions_file, dtype=str, low_memory=False)
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

        # 筛选最近指定小时的交易记录
        reference_time = pd.Timestamp.now(tz='UTC')
        time_threshold = reference_time - pd.Timedelta(hours=hours)
        df_recent = df[df['timestamp'] >= time_threshold].copy()
        
        return df_recent
        
    except Exception as e:
        print(f"⚠️ 读取交易记录失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def format_recent_transactions(transactions_df):
    """
    格式化最近的交易记录为表格格式
    
    参数:
    - transactions_df: 交易记录数据框
    
    返回:
    - str: 格式化的交易记录文本（表格格式）
    """
    if transactions_df is None or transactions_df.empty:
        return "  最近48小时内没有交易记录\n"
    
    # 按股票代码和时间排序
    transactions_df = transactions_df.sort_values(by=['code', 'timestamp'])
     
    # 构建Markdown表格
    text = "| 股票名称 | 股票代码 | 时间 | 类型 | 价格 | 目标价 | 止损价 | 有效期 | 理由 |\n"
    text += "|---------|---------|------|------|------|--------|--------|--------|------|\n"
    
    for _, trans in transactions_df.iterrows():
        stock_name = trans.get('name', '')
        code = trans.get('code', '')
        trans_type = trans.get('type', '')
        timestamp = pd.Timestamp(trans['timestamp']).strftime('%m-%d %H:%M:%S')
        price = trans.get('current_price', np.nan)
        price_display = f"{price:,.2f}" if not pd.isna(price) and price is not None else ''
        reason = trans.get('reason', '') or ''
        
        # 格式化止损价和目标价
        stop_loss = trans.get('stop_loss_price', np.nan)
        stop_loss_display = safe_float_format(stop_loss, '2f') if safe_float_format(stop_loss, '2f') else ''
        
        # 获取目标价
        target_price = trans.get('target_price', np.nan)
        target_price_display = safe_float_format(target_price, '2f') if safe_float_format(target_price, '2f') else ''
        
        # 获取有效期
        validity_period = trans.get('validity_period', np.nan)
        validity_period_display = safe_float_format(validity_period, '0f') if safe_float_format(validity_period, '0f') else ''
        
        text += f"| {stock_name} | {code} | {timestamp} | {trans_type} | {price_display} | {target_price_display} | {stop_loss_display} | {validity_period_display} | {reason} |\n"
    
    return text


def format_hsi_email_indicators(hsi_email_data):
    """
    格式化 hsi_email.py 的指标为文本和表格格式
    
    参数:
    - hsi_email_data: get_hsi_email_indicators 函数返回的数据
    
    返回:
    - tuple: (text_format, table_format) 格式化的文本和表格
    """
    if not hsi_email_data:
        return "", ""
    
    text_format = ""
    table_format = ""
    
    # 格式化恒生指数数据
    hsi_data = hsi_email_data.get('hsi_data')
    hsi_indicators = hsi_email_data.get('hsi_indicators')
    
    if hsi_data:
        text_format += "## 恒生指数实时技术指标\n\n"
        text_format += f"- 当前指数：{hsi_data['current_price']:,.2f}\n"
        text_format += f"- 24小时变化：{hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)\n"
        text_format += f"- 当日开盘：{hsi_data['open']:,.2f}\n"
        text_format += f"- 当日最高：{hsi_data['high']:,.2f}\n"
        text_format += f"- 当日最低：{hsi_data['low']:,.2f}\n"
        text_format += f"- 成交量：{hsi_data['volume']:,.0f}\n\n"
        
        if hsi_indicators:
            text_format += f"- RSI（14日）：{safe_float_format(hsi_indicators.get('rsi', 0), '2f')}\n"
            text_format += f"- MACD：{safe_float_format(hsi_indicators.get('macd', 0), '4f')}\n"
            text_format += f"- MACD信号线：{safe_float_format(hsi_indicators.get('macd_signal', 0), '4f')}\n"
            text_format += f"- MA20：{safe_float_format(hsi_indicators.get('ma20', 0), ',.2f')}\n"
            text_format += f"- MA50：{safe_float_format(hsi_indicators.get('ma50', 0), ',.2f')}\n"
            text_format += f"- MA200：{safe_float_format(hsi_indicators.get('ma200', 0), ',.2f')}\n"
            text_format += f"- 布林带位置：{safe_float_format(hsi_indicators.get('bb_position', 0), '2f')}\n"
            text_format += f"- ATR（14日）：{safe_float_format(hsi_indicators.get('atr', 0), '2f')}\n"
            text_format += f"- 趋势：{hsi_indicators.get('trend', '未知')}\n\n"
    
    # 格式化自选股数据
    stock_results = hsi_email_data.get('stock_results', [])
    
    if stock_results:
        text_format += "## 自选股实时技术指标\n\n"
        text_format += "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | RSI | MACD | MA20 | MA50 | 趋势 | ATR | 成交量比率 |\n"
        text_format += "|---------|---------|---------|--------|-----|------|-----|-----|------|-----|-----------|\n"
        
        for stock_result in stock_results:
            code = stock_result.get('code', 'N/A')
            name = stock_result.get('name', 'N/A')
            data = stock_result.get('data', {})
            indicators = stock_result.get('indicators', {})
            
            current_price = data.get('current_price', 0)
            change_pct = data.get('change_1d', 0)
            rsi = indicators.get('rsi', 0)
            macd = indicators.get('macd', 0)
            ma20 = indicators.get('ma20', 0)
            ma50 = indicators.get('ma50', 0)
            trend = indicators.get('trend', '未知')
            atr = indicators.get('atr', 0)
            volume_ratio = indicators.get('volume_ratio', 0)
            
            text_format += f"| {code} | {name} | {safe_float_format(current_price, '2f')} | {safe_float_format(change_pct, '+.2f')}% | {safe_float_format(rsi, '2f')} | {safe_float_format(macd, '4f')} | {safe_float_format(ma20, '2f')} | {safe_float_format(ma50, '2f')} | {trend} | {safe_float_format(atr, '2f')} | {safe_float_format(volume_ratio, '2f')}x |\n"
    
    return text_format, table_format


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
        
        table = "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | RSI | MACD | MA20 | MA50 | MA200 | 均线排列 | 均线斜率 | 乖离率 | 布林带位置 | ATR | 成交量比率 | 趋势 | 支撑位 | 阻力位 |\n"
        table += "|---------|---------|---------|--------|-----|------|-----|-----|------|---------|---------|-------|-----------|-----|-----------|------|--------|--------|\n"
        
        success_count = 0
        for stock_code in stock_codes_sorted:
            indicators = get_stock_technical_indicators(stock_code)
            
            if indicators:
                # 获取股票名称
                stock_name = WATCHLIST.get(stock_code, stock_code)
                
                # 格式化数据
                price = safe_float_format(indicators['current_price'], '2f')
                change = safe_float_format(indicators['change_pct'], '+.2f') + "%"
                rsi = safe_float_format(indicators['rsi'], '2f')
                macd = safe_float_format(indicators['macd'], '2f')
                ma20 = safe_float_format(indicators['ma20'], '2f')
                ma50 = safe_float_format(indicators['ma50'], '2f')
                ma200 = safe_float_format(indicators['ma200'], '2f') if pd.notna(indicators['ma200']) else "N/A"
                ma_align = indicators['ma_alignment']
                ma_slope = safe_float_format(indicators['ma_slope_20'], '4f')
                ma_dev = safe_float_format(indicators['ma_deviation'], '2f') + "%"
                bb_pos = safe_float_format(indicators['bb_position'], '1f') + "%"
                atr = safe_float_format(indicators['atr'], '2f')
                vol_ratio = safe_float_format(indicators['volume_ratio'], '2f') + "x"
                trend = indicators['trend']
                support = f"{safe_float_format(indicators['support_level'], '2f')} ({safe_float_format(indicators['support_distance'], '2f')}%)"
                resistance = f"{safe_float_format(indicators['resistance_level'], '2f')} ({safe_float_format(indicators['resistance_distance'], '2f')}%)"
                
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
                success_count += 1
        
        print(f"📊 技术指标表格: 成功获取 {success_count}/{len(stock_codes)} 只股票的数据")
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
        sender_email = os.environ.get("EMAIL_SENDER")
        email_password = os.environ.get("EMAIL_PASSWORD")
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
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


def run_comprehensive_analysis(llm_filepath, ml_filepath, output_filepath=None, 
                             send_email_flag=True, use_deep_analysis=True):
    """
    运行综合分析
    
    参数:
    - llm_filepath: 大模型建议文件路径
    - ml_filepath: ML预测结果文件路径（已废弃，保留用于兼容性）
    - output_filepath: 输出文件路径（可选）
    - send_email_flag: 是否发送邮件（默认True）
    - use_deep_analysis: 是否使用深度分析模式进行异常检测（默认True）
    """
    try:
        print("=" * 80)
        print("🤖 综合分析开始")
        print("=" * 80)
        
        # 检查大模型建议文件是否存在
        if not os.path.exists(llm_filepath):
            print(f"❌ 大模型建议文件不存在: {llm_filepath}")
            return None
        
        # 检查CatBoost单模型预测文件是否存在
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        catboost_csv = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_20d.csv')
        
        if not os.path.exists(catboost_csv):
            print(f"❌ CatBoost预测文件不存在: {catboost_csv}")
            return None
        
        print(f"📊 大模型建议文件: {llm_filepath}")
        print(f"📊 CatBoost预测文件: {catboost_csv}")
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
        print(f"   - CatBoost模型预测长度: {len(ml_predictions['ensemble'])} 字符\n")
        
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

【2. CatBoost模型20天预测结果】
**重要：probability = 上涨概率（不是下跌概率）**
{ml_predictions['ensemble']}

【辅助信息源 - 操作时机参考】

【3. 大模型短期买卖建议（日内/数天）】
{llm_recommendations['short_term']}

**🔴 核心硬性约束（不可违反）**

⚠️ **CatBoost概率约束（最高优先级，无例外）**：
- CatBoost概率 ≤ 0.50 → **绝对禁止推荐买入或强烈买入**
- CatBoost概率 < 0.40 → **绝对禁止推荐持有或观望**
- CatBoost概率 ≥ 0.50 → 可以考虑买入
- CatBoost概率 ≥ 0.60 → 可以考虑强烈买入
- **即使短期和中期方向一致，也绝对不允许违反此约束**
- **违反此约束的建议将被视为错误**

🔥 **决策顺序（严格遵守）**：
第一步：检查CatBoost概率 → 不满足则直接排除
第二步：检查短期和中期一致性
第三步：评估技术面和基本面
第四步：生成综合建议

=== 综合分析规则 ===

**规则1：时间维度匹配（业界最佳实践）**
- **短期信号（触发器）**：负责"何时做"（Timing）
- **中期信号（确认器）**：负责"是否做"（Direction）
- **CatBoost模型（验证器）**：负责提升置信度
- 只有短期和中期方向一致时，才采取行动
- 短期和中期冲突时，选择观望（避免不确定性）

**决策逻辑（短期触发 + 中期确认 + CatBoost验证）**：
- **第一步：检查CatBoost概率（硬约束）**
  - probability ≤ 0.50 → 排除买入或强烈买入
  - probability ≥ 0.50 → 进入下一步
- **第二步：检查短期和中期一致性**
  - 短期看好，中期看好 → 进入下一步
  - 方向不一致 → 观望
- **第三步：生成建议**
  - 强烈买入：短期看好，中期看好，CatBoost probability ≥ 0.60
  - 买入：短期看好，中期看好，0.50 < CatBoost probability < 0.60
  - 持有/观望：CatBoost probability ≤ 0.50 或 方向不一致
  - 卖出：短期看跌，中期看跌

**硬约束检查清单（必须逐项核对）**：
- [ ] CatBoost probability ≤ 0.50 → 绝对禁止推荐买入或强烈买入
- [ ] CatBoost probability < 0.40 → 绝对禁止推荐持有或观望
- [ ] CatBoost probability ≥ 0.50 → 可以考虑买入
- [ ] CatBoost probability ≥ 0.60 → 可以考虑强烈买入
- [ ] 短期和中期方向是否一致？
- [ ] 三重确认是否全部满足？

**规则2：CatBoost概率评估**

**CatBoost概率阈值**：
- **高置信度上涨**：probability > 0.60
- **中等置信度观望**：0.50 < probability ≤ 0.60
- **预测下跌**：probability ≤ 0.50

**重要说明 - CatBoost probability 定义**：
- `probability` = **上涨概率**（模型预测股票上涨的概率）
- 下跌概率 = 1 - probability
- 例如：probability = 0.35 表示上涨概率35%，下跌概率65%
- 例如：probability = 0.68 表示上涨概率68%，下跌概率32%
- **切勿将 probability 误解为下跌概率**

**阈值优化说明**：
- 当前CatBoost模型20天准确率：约{model_accuracy['catboost']['accuracy']:.2%}（CatBoost 单模型）
- CatBoost模型准确率：{model_accuracy['catboost']['accuracy']:.2%}（±{model_accuracy['catboost']['std']:.2%}）
- 强买入阈值0.60略高于CatBoost准确率，确保高置信度
- 买入阈值0.50接近CatBoost准确率，平衡召回率和精确率
- 卖出阈值0.50确保下跌概率>50%
- 观望区间0.45-0.50避免低置信度决策

**重要说明 - CatBoost模型优势**：
- **单模型策略**：CatBoost 单模型表现最佳（回测收益率 276.74%）
- **自动分类特征处理**：无需手动编码，使用 LabelEncoder 自动处理
- **更好的默认参数**：减少调参工作量，开箱即用
- **稳定性优异**：标准差 ±{model_accuracy['catboost']['std']:.2%}，表现稳定
- **置信度评估**：通过预测概率评估预测可靠性

**重要说明 - 模型不确定性（风险提示）**：
- CatBoost模型存在标准差（±{model_accuracy['catboost']['std']:.2%}），实际准确率可能波动
- 但这**不能**作为降低CatBoost概率标准的理由
- 对于probability在0.50-0.60之间的股票，建议观望而非买入
- 对于probability在0.60-0.70之间的股票，建议降低仓位（2-3%）而非4-6%

**重要说明 - 信号协同（必须同时满足）**：
- **短期信号（触发器）**：负责"何时做"（Timing）→ 必须100%满足
- **中期信号（确认器）**：负责"是否做"（Direction）→ 必须100%满足
- **CatBoost概率（硬性约束）**：负责验证方向性→ 必须100%满足
- **三重确认：短期、中期、CatBoost三者必须同时满足，缺一不可**

**重要说明 - 时间维度标准化**：
- 短期：1-5个交易日（日内到一周）
- 中期：10-20个交易日（2-4周）
- 长期：>20个交易日（超过1个月）
- 当前映射：大模型短期建议 ↔ CatBoost模型预测（20天），大模型中期建议 ↔ 基本面分析（数周-数月）✅

**规则3：CatBoost概率评估**
- **高置信度上涨（probability > 0.60）**：信号可靠性最高，优先级提升
- **中等置信度观望（0.50 < probability ≤ 0.60）**：信号可靠性中等，需要短期中期一致支持
- **预测下跌（probability ≤ 0.50）**：信号可靠性低，建议观望，不进行交易
- 如果probability高（>0.60），综合置信度最高
- 如果probability低（≤0.50），降低为中等置信度

**规则4：推荐理由格式**
- 必须说明：短期建议+中期建议+CatBoost预测（probability）
- 例如："短期建议买入（触发器），中期建议买入（确认器），CatBoost预测上涨概率0.72（高置信度），综合置信度高。注意CatBoost模型当前准确率约{model_accuracy['catboost']['accuracy']:.2%}（标准差约±{model_accuracy['catboost']['std']:.2%}），probability在0.72附近实际准确率可能在{model_accuracy['catboost']['accuracy']-model_accuracy['catboost']['std']:.2%} ~ {model_accuracy['catboost']['accuracy']+model_accuracy['catboost']['std']:.2%}之间"

请基于上述规则，完成以下任务：

1. **一致性分析**（方案A核心：短期触发 + 中期确认 + CatBoost验证）：
   - **第一步（核心）**：分析短期建议与中期建议的一致性
     - 短期买入 + 中期买入 → 方向一致，考虑CatBoost验证
     - 短期买入 + 中期观望 → 等待中期确认
     - 短期买入 + 中期卖出 → 冲突，观望
     - 短期卖出 + 中期卖出 → 方向一致，考虑CatBoost验证
     - 短期卖出 + 中期观望 → 等待中期确认
     - 短期卖出 + 中期买入 → 冲突，观望
   - **第二步（验证）**：对短期中期一致的股票，分析CatBoost预测验证
     - 如果CatBoost高置信度支持（probability>0.60），提升为强信号
     - 如果CatBoost中等置信度支持（0.50<probability≤0.60），提升为中等信号
     - 如果CatBoost低置信度（probability≤0.50），降低为弱信号或观望
   - 标注符合"强买入信号"、"买入信号"、"观望信号"、"卖出信号"的股票

2. **个股建议排序**：
   - 优先级：强买入信号 > 买入信号 > 观望信号 > 卖出信号
   - 在相同优先级内，按probability排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 强烈买入信号（2-3只）：最高优先级，建议仓位4-6%
   - 买入信号（3-5只）：次优先级，建议仓位2-4%
   - 持有/观望（如有）：第三优先级
   - 卖出信号（如有）：最低优先级

3.1. **特殊处理（CatBoost probability ≤ 0.50的股票）**：
   - **绝对禁止**： probability ≤ 0.50 的股票不能出现在"强烈买入信号"或"买入信号"中
   - **正确处理**： probability ≤ 0.50 的股票应该出现在"持有/观望"或"卖出信号"中
   - **理由说明**：在"推荐理由"中必须明确说明"CatBoost probability ≤ 0.50，违反硬约束，建议观望"
   - **示例**："短期建议买入，中期建议买入，但CatBoost预测上涨概率0.48（≤0.50），违反硬约束，建议观望"

4. **风险提示**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比，总仓位45%-55%）
   - 给出止损位建议（单只股票最大亏损不超过-8%）
   
   **特别要求 - 考虑CatBoost模型不确定性**：
   - CatBoost模型20天标准差约±{model_accuracy['catboost']['std']:.2%}
   - 对于probability在0.55-0.65之间的股票，建议仓位不超过2-3%
   - 强买入信号（短期/中期一致且CatBoost高置信度）建议仓位4-6%
   - 总仓位控制在45%-55%
   - **必须设置止损位，单只股票最大亏损不超过-8%**
   - **严格遵循"短期触发 + 中期确认 + CatBoost验证"原则**：只有短期和中期方向一致且CatBoost验证时才行动
   - 如果短期和中期建议冲突，优先选择观望，不进行交易
   - 采用"三重确认"策略：短期、中期、CatBoost三者一致时才重仓操作

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] 
   - 推荐理由：[简短理由，例如：短期买入，中期买入，CatBoost预测上涨概率0.72（高置信度），方向一致]
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
   - 推荐理由：[简短理由]
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
                
                print("📊 获取 hsi_email.py 实时指标...")
                hsi_email_indicators = get_hsi_email_indicators()
                
                print("📊 获取股票异常检测...")
                # 使用快速模式（Z-Score only），提高综合分析速度
                # 如需使用深度分析模式，通过 use_deep_analysis=True 参数控制
                anomaly_data = get_stock_anomalies(use_deep_analysis=use_deep_analysis)
                
                # 构建板块分析文本
                sector_text = ""
                if sector_data and sector_data['performance'] is not None:
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
                        
                        sector_text += f"| {idx+1} | {trend_icon} {row['sector_name']} | {change_color}{safe_float_format(row['avg_change_pct'], '2f')}% | {leaders_text} |\n"
                    
                    # 添加投资建议
                    top_sector = perf_df.iloc[0]
                    bottom_sector = perf_df.iloc[-1]
                    
                    sector_text += "\n**投资建议**：\n"
                    if float(top_sector['avg_change_pct']) > 1 if not pd.isna(top_sector['avg_change_pct']) else False:
                        sector_text += f"- 当前热点板块：{top_sector['sector_name']}，平均涨幅 {safe_float_format(top_sector['avg_change_pct'], '2f')}%\n"
                        if top_sector['sector_code'] in sector_leaders and sector_leaders[top_sector['sector_code']]:
                            leader = sector_leaders[top_sector['sector_code']][0]
                            sector_text += f"- 建议关注该板块的龙头股：{leader['name']} ⭐\n"
                    
                    if float(bottom_sector['avg_change_pct']) < -1 if not pd.isna(bottom_sector['avg_change_pct']) else False:
                        sector_text += f"- 当前弱势板块：{bottom_sector['sector_name']}，平均跌幅 {safe_float_format(bottom_sector['avg_change_pct'], '2f')}%\n"
                        sector_text += "- 建议谨慎操作该板块，等待企稳信号\n"
                
                # 构建股息信息文本
                dividend_text = ""
                if dividend_data:
                    dividend_text += "| 股票代码 | 股票名称 | 除净日 | 分红方案 |\n"
                    dividend_text += "|---------|---------|-------|----------|\n"
                    
                    for stock in dividend_data[:10]:
                        code = stock.get('股票代码', 'N/A')
                        name = stock.get('股票名称', 'N/A')
                        ex_date = stock.get('除净日', 'N/A')
                        dividend_plan = stock.get('分红方案', 'N/A')
                        # 格式化除净日
                        if isinstance(ex_date, pd.Timestamp):
                            ex_date = ex_date.strftime('%Y-%m-%d')
                        # 截断过长的分红方案
                        if dividend_plan != 'N/A' and len(str(dividend_plan)) > 30:
                            dividend_plan = str(dividend_plan)[:28] + '...'
                        dividend_text += f"| {code} | {name} | {ex_date} | {dividend_plan} |\n"
                
                # 获取当前市场状态
                current_market = get_current_market_state()
                
                # 构建恒生指数分析文本
                hsi_text = ""
                if current_market:
                    hsi_text += f"**市场信号**: {current_market['market_signal']}\n\n"
                    hsi_text += f"**市场状态**: {current_market['market_state_cn']}\n\n"
                    hsi_text += f"**恒生指数**: {current_market['current_hsi']:.2f} (实时)\n\n"
                    hsi_text += f"**数据更新时间**: {current_market['date']}\n\n"
                    hsi_text += f"**最近20天收益率**: {current_market['recent_20d_return']:.2%}\n\n"
                    hsi_text += f"**最近5天收益率**: {current_market['recent_5d_return']:.2%}\n\n"
                    
                    # 添加市场状态说明表格
                    hsi_text += "### 市场状态说明\n\n"
                    hsi_text += "| 市场状态 | 20天收益率范围 | 说明 |\n"
                    hsi_text += "|---------|--------------|------|\n"
                    hsi_text += "| 📈 牛市 | > 5% | 市场强劲上涨，适合积极配置 |\n"
                    hsi_text += "| ⬆️ 震荡偏涨 | 2% - 5% | 市场温和上涨，可以谨慎配置 |\n"
                    hsi_text += "| ➡️ 震荡市 | -2% - 2% | 市场横盘整理，建议观望 |\n"
                    hsi_text += "| ⬇️ 震荡偏跌 | -5% - -2% | 市场温和下跌，建议减仓 |\n"
                    hsi_text += "| 📉 熊市 | < -5% | 市场强劲下跌，建议空仓 |\n\n"
                    
                    # 添加投资建议
                    hsi_text += "### 投资建议\n\n"
                    
                    if current_market['market_state'] == 'bull':
                        hsi_text += "**牛市策略**:\n\n"
                        hsi_text += "- ✅ **重仓高市场关联性股票**: 牛市中高关联性股票平均收益率可达 +9.35%\n"
                        hsi_text += "- ✅ **关注科技、半导体板块**: 这些板块通常在牛市中表现优异\n"
                        hsi_text += "- ✅ **使用100%仓位**: 市场信号强烈，可全仓操作\n\n"
                    elif current_market['market_state'] == 'bear':
                        hsi_text += "**熊市策略**:\n\n"
                        hsi_text += "- ⚠️ **重仓低市场关联性股票**: 熊市中低关联性股票平均收益率为 +4.15%\n"
                        hsi_text += "- ⚠️ **配置银行、公用事业**: 这些股票具有防御性\n"
                        hsi_text += "- ⚠️ **降低仓位至30%**: 市场风险较高，控制仓位\n\n"
                    elif current_market['market_state'] in ['neutral_bull', 'neutral_bear']:
                        hsi_text += "**震荡市策略**:\n\n"
                        hsi_text += "- 🔄 **均衡配置**: 高低关联性股票各占50%\n"
                        hsi_text += "- 🔄 **动态调整**: 根据市场信号及时调整仓位\n"
                        hsi_text += "- 🔄 **关注波段机会**: 震荡市适合波段操作\n\n"
                        
                        # 额外的风险提示
                        if current_market['market_state'] == 'neutral_bear':
                            hsi_text += "**风险提示**:\n\n"
                            hsi_text += "- ⚠️ 市场温和下跌，建议保持谨慎\n"
                            hsi_text += "- ⚠️ 可考虑降低仓位至70%\n"
                        else:
                            hsi_text += "**机会提示**:\n\n"
                            hsi_text += "- ✅ 市场温和上涨，可考虑逐步加仓\n"
                            hsi_text += "- ✅ 建议仓位可提升至80%\n\n"
                    else:  # neutral
                        hsi_text += "**横盘策略**:\n\n"
                        hsi_text += "- ⏸️ **观望为主**: 市场缺乏明确方向，建议保持观望\n"
                        hsi_text += "- ⏸️ **低仓位试探**: 可用30%仓位试探性配置\n"
                        hsi_text += "- ⏸️ **等待信号**: 等待市场明确方向后再做决策\n\n"
                    
                    hsi_text += "---\n\n"
                    hsi_text += "**技术指标**:\n\n"
                
                if hsi_data:
                    hsi_text += f"- 当前价格：{safe_float_format(hsi_data['current_price'], '2f')}\n"
                    hsi_text += f"- 日涨跌幅：{safe_float_format(hsi_data['change_pct'], '+.2f')}% ({safe_float_format(hsi_data['change_points'], '+.2f')} 点)\n"
                    hsi_text += f"- RSI（14日）：{safe_float_format(hsi_data['rsi'], '2f')}\n"
                    hsi_text += f"- MA20：{safe_float_format(hsi_data['ma20'], '2f')}\n"
                    hsi_text += f"- MA50：{safe_float_format(hsi_data['ma50'], '2f')}\n"
                    hsi_text += f"- 趋势：{hsi_data['trend']}\n"
                
                # 使用配置文件中的所有自选股
                stock_codes = list(WATCHLIST.keys())
                print(f"📊 使用配置文件中的 {len(stock_codes)} 只自选股生成技术指标表格")
                
                # 生成技术指标表格
                print("📊 生成推荐股票技术指标表格...")
                technical_indicators_table = generate_technical_indicators_table(stock_codes)
                if not technical_indicators_table:
                    print("⚠️ 技术指标表格为空，可能是股票数据获取失败")
                
                # 添加 hsi_email.py 的实时指标内容
                hsi_email_text = ""
                if hsi_email_indicators:
                    hsi_email_text, _ = format_hsi_email_indicators(hsi_email_indicators)
                
                # 添加最近48小时模拟交易记录
                recent_transactions_df = get_recent_transactions(hours=48)
                recent_transactions_text = format_recent_transactions(recent_transactions_df)
                
                # 构建完整的邮件内容（综合买卖建议 + 信息参考）
                # 注意：不添加标题，因为HTML模板已经有了标题
                full_content = f"""{response}

---

# 信息参考

## 一、恒生指数技术分析

{hsi_text}

## 二、机器学习预测结果（20天）

### CatBoost模型（20天预测）

**模型准确率**：

- CatBoost：{model_accuracy['catboost']['accuracy']:.2%}（标准差±{model_accuracy['catboost']['std']:.2%}）

{ml_predictions['ensemble']}

## 三、大模型建议

### 短期买卖建议（日内/数天）
{llm_recommendations['short_term']}

### 中期买卖建议（数周-数月）
{llm_recommendations['medium_term']}

## 四、板块分析（5日涨跌幅排名）

{sector_text}

## 五、股息信息（即将除净）

{dividend_text}

## 六、股票技术指标详情

{technical_indicators_table}

## 七、最近48小时模拟交易记录

{recent_transactions_text}

## 八、股票异常检测提醒

"""
                
                # 添加异常检测结果
                if anomaly_data and len(anomaly_data) > 0:
                    full_content += f"**检测到 {len(anomaly_data)} 个异常**\n\n"
                    
                    # 按类型和严重程度分类
                    high_price_anomalies = [a for a in anomaly_data if a['severity'] == 'high' and a.get('type') == 'price']
                    high_volume_anomalies = [a for a in anomaly_data if a['severity'] == 'high' and a.get('type') == 'volume']
                    medium_price_anomalies = [a for a in anomaly_data if a['severity'] == 'medium' and a.get('type') == 'price']
                    medium_volume_anomalies = [a for a in anomaly_data if a['severity'] == 'medium' and a.get('type') == 'volume']
                    if_anomalies = [a for a in anomaly_data if a.get('type') == 'isolation_forest']
                    
                    # 高异常（价格）
                    if high_price_anomalies:
                        full_content += "### 🔴 高异常（价格）\n\n"
                        full_content += "| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | RSI | 布林带 |\n"
                        full_content += "|---------|---------|---------|---------|---------|-----|--------|\n"
                        
                        for anomaly in high_price_anomalies:
                            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
                            full_content += f"| {anomaly['stock']} | {anomaly['name']} | **{anomaly['z_score']:.2f}** | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} | {get_bollinger_position_cn(anomaly['bollinger_position'])} |\n"
                        
                        full_content += "\n**风险提示**：\n"
                        full_content += "⚠️ 价格波动率异常，风险显著增加\n"
                        full_content += "⚠️ 可能是突发利空消息或市场恐慌\n"
                        full_content += "⚠️ 建议立即检查持仓，考虑减仓或止损\n\n"
                    
                    # 高异常（成交量）
                    if high_volume_anomalies:
                        full_content += "### 🔴 高异常（成交量）\n\n"
                        full_content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
                        full_content += "|---------|---------|---------|--------|---------|---------|\n"
                        
                        for anomaly in high_volume_anomalies:
                            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
                            volume_str = f"{anomaly.get('value', 0):,.0f}"
                            full_content += f"| {anomaly['stock']} | {anomaly['name']} | **{anomaly['z_score']:.2f}** | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"
                        
                        full_content += "\n**风险提示**：\n"
                        full_content += "⚠️ 成交量异常放大，可能有大额交易\n"
                        full_content += "⚠️ 可能是机构建仓或出货\n"
                        full_content += "⚠️ 建议关注价格方向和持续性\n\n"
                    
                    # 中异常（价格）
                    if medium_price_anomalies:
                        full_content += "### ⚠️ 中异常（价格）\n\n"
                        full_content += "| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | RSI |\n"
                        full_content += "|---------|---------|---------|---------|---------|-----|\n"
                        
                        for anomaly in medium_price_anomalies:
                            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
                            full_content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"
                        
                        full_content += "\n**操作建议**：\n"
                        full_content += "- 观察价格是否能突破阻力位\n"
                        full_content += "- 注意止损位设置\n"
                        full_content += "- 避免在异常期间加仓\n\n"
                    
                    # 中异常（成交量）
                    if medium_volume_anomalies:
                        full_content += "### ⚠️ 中异常（成交量）\n\n"
                        full_content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
                        full_content += "|---------|---------|---------|--------|---------|---------|\n"
                        
                        for anomaly in medium_volume_anomalies:
                            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
                            volume_str = f"{anomaly.get('value', 0):,.0f}"
                            full_content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"
                        
                        full_content += "\n**操作建议**：\n"
                        full_content += "- 关注成交量是否持续放大\n"
                        full_content += "- 结合价格方向判断趋势\n"
                        full_content += "- 谨慎操作，避免追涨杀跌\n\n"
                    
                    # Isolation Forest 异常（仅深度分析模式）
                    if if_anomalies:
                        full_content += "### 🔬 Isolation Forest 异常\n\n"
                        full_content += "| 股票代码 | 股票名称 | 异常得分 | 当前价格 | 当日变化 | RSI |\n"
                        full_content += "|---------|---------|---------|---------|---------|-----|\n"
                        
                        for anomaly in if_anomalies:
                            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
                            # Isolation Forest 异常的 z_score 设为 0，使用其他指标
                            full_content += f"| {anomaly['stock']} | {anomaly['name']} | **异常** | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"
                        
                        full_content += "\n**说明**：\n"
                        full_content += "- 基于机器学习的多维特征异常检测\n"
                        full_content += "- 可能检测到传统统计方法遗漏的异常\n"
                        full_content += "- 建议结合其他指标综合判断\n\n"
                    
                    full_content += "**重要提醒**：\n"
                    full_content += "- 异常是警告信号，不一定是交易信号\n"
                    full_content += "- 不要看到异常就立即卖出\n"
                    full_content += "- 综合考虑基本面和市场情绪\n"
                else:
                    full_content += "✅ 未检测到异常，市场波动正常\n\n"
                
                # 继续添加其他内容
                full_content += f"""## 九、技术指标说明

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

## 十、**决策框架**



### ✦ 买入策略

- **CatBoost 概率 ≥ 0.60** + **短期看好** + **中期看好** → 强烈买入

- **0.50 < CatBoost 概率 < 0.60** + **短期看好** + **中期看好** → 买入

- **CatBoost 概率 ≤ 0.50** → 禁止买入（硬约束）



### ✦ 持有策略

- **CatBoost 概率 > 0.60** + **大模型建议买入** → 强烈持有

- **0.50 < CatBoost 概率 ≤ 0.60** + **大模型建议买入** → 观望持有

- **CatBoost 概率 ≤ 0.50** + **大模型建议卖出** → 考虑卖出



### ✦ 卖出策略

- **CatBoost 概率 ≤ 0.50** + **短期看跌** + **中期看跌** → 卖出

- **CatBoost 概率 < 0.40** → 禁止持有或观望（硬约束）



### ✦ 决策顺序（严格遵守）

1. **第一步：检查 CatBoost 概率（硬约束）**

   - CatBoost 概率 ≤ 0.50 → 排除买入或强烈买入

   - CatBoost 概率 ≥ 0.50 → 进入下一步

2. **第二步：检查短期和中期一致性**

   - 短期看好 + 中期看好 → 进入下一步

   - 方向不一致 → 观望

3. **第三步：生成建议**

   - 强烈买入：短期看好 + 中期看好 + CatBoost 概率 ≥ 0.60

   - 买入：短期看好 + 中期看好 + 0.50 < CatBoost 概率 < 0.60

   - 持有/观望：CatBoost 概率 ≤ 0.50 或 方向不一致

   - 卖出：短期看跌 + 中期看跌 + CatBoost 概率 ≤ 0.50

### ✦ 动态置信度阈值策略（根据市场环境调整）

根据市场环境动态调整置信度阈值，可显著提升风险调整收益和过滤噪声：

| 市场环境 | 置信度阈值 | 调整幅度 | 说明 |
|---------|-----------|---------|------|
| 牛市 (bull) | 0.55 | -0.05 | 更激进，增加交易机会 |
| 震荡市 (normal/ranging) | 0.65 | +0.05 | 更严格过滤噪声 |
| 熊市 (bear) | 0.60 | 基准 | 中等保守 |

**市场环境识别方法**：
- **牛市**：恒生指数 20 天收益率 > +5%
- **熊市**：恒生指数 20 天收益率 < -5%
- **震荡市**：恒生指数 20 天收益率在 -5% 到 +5% 之间

**使用建议**：
- 震荡市使用阈值 0.65 可显著过滤噪声，减少亏损交易
- 牛市使用阈值 0.55 可捕捉更多机会，不影响收益率
- 熊市使用阈值 0.60 保持中等保守策略

### ✦ 强烈买入信号
**强烈买入信号**是在每日综合分析邮件中的第一部分，包含：
- **股票代码和名称**：如"0700.HK 腾讯控股"
- **推荐理由**：详细的分析，说明短期建议+中期建议+CatBoost预测
- **操作建议**：明确的买入/持有/卖出建议
- **价格指引**：建议买入价、止损位、目标价
- **风险提示**：主要风险因素

## 十一、风险提示

1. **模型不确定性**：
   - ML 20天 CatBoost模型标准差为±{model_accuracy['catboost']['std']:.2%}
   - 融合预测概率>0.60为高置信度上涨，0.50-0.60为中等置信度观望，≤0.50为预测下跌
   - 建议：短期和中期一致是主要决策依据，ML预测用于验证和提升置信度

2. **市场风险**：
   - 当前市场整体风险：[高/中/低]（需根据恒生指数技术指标判断）
   - 建议仓位：45%-55%
   - **必须设置止损位，单只股票最大亏损不超过-8%**

3. **投资原则**：
   - 短期触发 + 中期确认 + ML验证 = 高置信度信号
   - 短期和中期冲突 = 观望（避免不确定性）
   - CatBoost概率在0.50-0.60之间 = 中等置信度，建议观望或轻仓
   - 总仓位控制在45%-55%，分散风险

## 十二、数据来源

- 大模型分析：Qwen大模型
- ML预测：CatBoost（单模型）
- 特征工程：2991个原始特征，500个精选特征（F-test+互信息混合方法）
- 技术指标：RSI、MACD、布林带、ATR、均线、成交量等80+个指标
- 基本面数据：PE、PB、ROE、ROA、股息率等8个指标
- 美股市场：标普500、纳斯达克、VIX、美国国债收益率等11个指标
- 股票类型：18个行业分类及衍生评分
- 情感分析：四维情感评分（Relevance/Impact/Expectation_Gap/Sentiment）
- 板块分析：16个板块涨跌幅排名、技术趋势分析、龙头识别
- 主题建模：LDA主题建模（10个主题）
- 主题情感交互：10个主题 × 5个情感指标 = 50个交互特征
- 预期差距：新闻情感相对于市场预期的差距（5个特征）
- 模型策略：CatBoost 单模型
- 置信度评估：高（>0.60）、中（0.50-0.60）、低（≤0.50）

"""
                
                # 如果有hsi_email.py指标，添加到数据源部分
                if hsi_email_indicators:
                    full_content += f"""
- **实时指标**：恒生指数及自选股实时技术指标，包括TAV评分、建仓/出货评分、基本面评分等高级分析指标
"""
                
                full_content += f"""

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
    parser.add_argument('--no-deep-analysis', action='store_true',
                       help='不使用深度分析模式进行异常检测（默认使用深度分析）')
    
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
                                       send_email_flag=not args.no_email,
                                       use_deep_analysis=not args.no_deep_analysis)
    
    if result:
        print("\n✅ 综合分析完成！")
        sys.exit(0)
    else:
        print("\n❌ 综合分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
