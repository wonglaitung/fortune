#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股AI报告生成脚本

功能：
1. 获取A股实时行情和技术指标
2. 调用通义千问生成买卖建议
3. 保存报告到文件
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import A_STOCK_WATCHLIST, A_STOCK_TRAINING_LIST, get_limit_rate
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.main_fund_flow import MainFundFlowService

# 尝试导入技术分析模块（用于筹码阻力）
try:
    from data_services.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False


def get_stock_technical_data(stock_code, period_days=100):
    """
    获取股票技术数据

    Args:
        stock_code: 股票代码
        period_days: 数据天数

    Returns:
        dict: 技术数据
    """
    df = get_a_stock_data(stock_code, period_days=period_days, use_cache=True)
    if df is None or df.empty:
        return None

    # 计算技术指标
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    # 收益率
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_20d'] = df['Close'].pct_change(20)

    # 波动率
    df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 布林带
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # 成交量
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()

    # 支撑位/阻力位（20日）
    df['Support_20d'] = df['Low'].rolling(20).min()
    df['Resistance_20d'] = df['High'].rolling(20).max()

    # 趋势判断（基于MA排列）
    df['MA_Alignment'] = df.apply(
        lambda row: '多头排列' if row['MA5'] > row['MA10'] > row['MA20']
        else ('空头排列' if row['MA5'] < row['MA10'] < row['MA20'] else '震荡'),
        axis=1
    )

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 获取实时价格
    realtime = get_a_stock_info_tencent(stock_code)
    current_price = realtime['current_price'] if realtime else latest['Close']
    change_percent = realtime['change_percent'] if realtime else None

    # 计算筹码分布（中期分析）
    chip_data = None
    if TECHNICAL_ANALYSIS_AVAILABLE:
        try:
            analyzer = TechnicalAnalyzer()
            chip_result = analyzer.get_chip_distribution(df)
            if chip_result:
                resistance_ratio = chip_result.get('resistance_ratio', 0.5)
                concentration = chip_result.get('concentration', 0)
                chip_data = {
                    'resistance_ratio': resistance_ratio,
                    'resistance_level': chip_result.get('resistance_level', '中'),
                    'concentration': concentration,
                    'concentration_level': chip_result.get('concentration_level', '中'),
                }
        except Exception as e:
            pass

    # 支撑位/阻力位计算
    support_20d = latest['Support_20d']
    resistance_20d = latest['Resistance_20d']
    distance_to_support = (current_price - support_20d) / current_price * 100 if current_price else None
    distance_to_resistance = (resistance_20d - current_price) / current_price * 100 if current_price else None

    # 涨跌停状态判断
    limit_rate = get_limit_rate(stock_code)
    limit_pct = limit_rate * 100
    is_near_limit_up = False
    is_near_limit_down = False
    if change_percent is not None:
        is_near_limit_up = change_percent >= (limit_pct - 1)  # 距离涨停1%以内
        is_near_limit_down = change_percent <= -(limit_pct - 1)  # 距离跌停1%以内

    return {
        'stock_code': stock_code,
        'current_price': current_price,
        'change_percent': change_percent,
        'ma5': latest['MA5'],
        'ma10': latest['MA10'],
        'ma20': latest['MA20'],
        'ma60': latest.get('MA60'),
        'price_vs_ma5': (current_price / latest['MA5'] - 1) * 100 if latest['MA5'] else None,
        'price_vs_ma20': (current_price / latest['MA20'] - 1) * 100 if latest['MA20'] else None,
        'return_5d': latest['Return_5d'] * 100 if latest['Return_5d'] else None,
        'return_20d': latest['Return_20d'] * 100 if latest['Return_20d'] else None,
        'volatility_20d': latest['Volatility_20d'] * 100 if latest['Volatility_20d'] else None,
        'rsi_14': latest['RSI_14'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_Signal'],
        'macd_hist': latest['MACD'] - latest['MACD_Signal'],
        'bb_upper': latest['BB_Upper'],
        'bb_lower': latest['BB_Lower'],
        'bb_position': (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100 if latest['BB_Upper'] != latest['BB_Lower'] else 50,
        'volume': latest['Volume'],
        'volume_ratio': latest['Volume'] / latest['Volume_MA5'] if latest['Volume_MA5'] else None,
        'limit_rate': limit_rate,
        # 新增字段
        'ma_alignment': latest['MA_Alignment'],
        'support_20d': support_20d,
        'resistance_20d': resistance_20d,
        'distance_to_support': distance_to_support,
        'distance_to_resistance': distance_to_resistance,
        'is_near_limit_up': is_near_limit_up,
        'is_near_limit_down': is_near_limit_down,
        'chip_data': chip_data,
    }


def generate_llm_prompt(stock_data_list, market_data, main_fund_data):
    """
    生成大模型提示词（参考港股六层分析框架）

    Args:
        stock_data_list: 股票数据列表
        market_data: 市场数据
        main_fund_data: 主力资金数据

    Returns:
        str: 提示词
    """
    # 读取A股新闻数据
    news_data = {}
    news_file_path = "data/a_stock_news_records.csv"
    try:
        if os.path.exists(news_file_path):
            news_df = pd.read_csv(news_file_path, dtype={'股票代码': str})
            if not news_df.empty:
                for code, group in news_df.groupby('股票代码'):
                    news_data[code] = group.to_dict('records')
    except Exception as e:
        print(f"⚠️ 读取A股新闻数据失败: {e}")

    # 构建股票信息（包含新增字段）
    stock_info = ""
    for data in stock_data_list:
        stock_name = A_STOCK_WATCHLIST.get(data['stock_code'], data['stock_code'])

        # 涨跌停状态
        limit_status = ""
        if data.get('is_near_limit_up'):
            limit_status = "⚠️ 接近涨停"
        elif data.get('is_near_limit_down'):
            limit_status = "⚠️ 接近跌停"
        else:
            limit_status = "正常"

        # 筹码分布（中期分析）
        chip_info = ""
        if data.get('chip_data'):
            chip = data['chip_data']
            chip_info = f"""
- **筹码分布**：
  - 上方筹码比例：{chip.get('resistance_ratio', 0):.1%}（{chip.get('resistance_level', '中')}）
  - 筹码集中度：{chip.get('concentration', 0):.3f}（{chip.get('concentration_level', '中')}）
"""

        stock_info += f"""
### {stock_name} ({data['stock_code']})

**基本数据**：
- 当前价格：{data['current_price']:.2f} 元
- 今日涨跌：{data['change_percent']:+.2f}%
- 涨跌停限制：{data['limit_rate']*100:.0f}%（{limit_status}）

**技术指标**：
- 均线位置：
  - MA5: {data['ma5']:.2f} ({data['price_vs_ma5']:+.2f}%)
  - MA20: {data['ma20']:.2f} ({data['price_vs_ma20']:+.2f}%)
  - MA排列：{data.get('ma_alignment', '震荡')}
- 近期涨跌：
  - 5日涨跌：{data['return_5d']:+.2f}%
  - 20日涨跌：{data['return_20d']:+.2f}%
- RSI(14)：{data['rsi_14']:.1f}（{'超买' if data['rsi_14'] > 70 else '超卖' if data['rsi_14'] < 30 else '中性'}）
- MACD：{data['macd']:.4f}，信号线：{data['macd_signal']:.4f}，柱状：{data['macd_hist']:.4f}（{'金叉' if data['macd_hist'] > 0 else '死叉'}）
- 布林带位置：{data['bb_position']:.1f}%（0%=下轨，100%=上轨）
- 成交量比率：{data['volume_ratio']:.2f}x

**支撑阻力**：
- 支撑位（20日）：{data.get('support_20d', 'N/A'):.2f} 元（距离 {data.get('distance_to_support', 0):+.1f}%）
- 阻力位（20日）：{data.get('resistance_20d', 'N/A'):.2f} 元（距离 {data.get('distance_to_resistance', 0):+.1f}%）
{chip_info}
"""

        # 新闻摘要
        stock_code = data['stock_code']
        stock_news = news_data.get(stock_code, [])
        if stock_news:
            stock_info += "**新闻摘要**：\n"
            for news in stock_news[:3]:
                sentiment_str = ""
                if pd.notna(news.get('情感分数')):
                    sentiment_str = f"（情感分数: {news['情感分数']:.1f}）"
                stock_info += f"  - {news.get('新闻时间', '')}: {news.get('新闻标题', '')}{sentiment_str}\n"
        else:
            stock_info += "**新闻摘要**：暂无相关新闻\n"

    # 构建市场环境（包含市场情绪层级）
    sentiment = market_data.get('sentiment', {})
    sentiment_name = sentiment.get('name', '正常市场')
    sentiment_action = sentiment.get('action', '正常交易')
    up_ratio = sentiment.get('up_ratio', 0.5)

    # 主力资金趋势
    mf_5d = main_fund_data.get('net_flow_5d_sum', 0)
    mf_20d = main_fund_data.get('net_flow_20d_sum', 0)
    consecutive = main_fund_data.get('consecutive_inflow', 0)

    market_info = f"""
### 市场环境

**上证指数**：
- 收盘：{market_data.get('sh_close', 'N/A'):.2f}
- 涨跌：{market_data.get('sh_change', 0):+.2f}%
- MA20：{market_data.get('sh_ma20', 'N/A'):.2f}
- 相对MA20：{market_data.get('sh_vs_ma20', 0):+.2f}%

**市场情绪**：
- 情绪层级：{sentiment_name}（{sentiment_action}）
- 上涨比例：{up_ratio:.1%}（全量{sentiment.get('total_count', len(stock_data_list))}只股票）
- 动态阈值：买入需概率≥{sentiment.get('dynamic_threshold', 0.5):.0%}

**主力资金**：
- 今日净流入：{main_fund_data.get('main_net_flow', 0):.2f} 亿
- 5日累积：{mf_5d:.2f} 亿
- 20日累积：{mf_20d:.2f} 亿
- 连续流入：{consecutive} 天
"""

    # 完整提示词（参考港股六层分析框架）
    prompt = f"""你是一位专业的A股投资分析师。请按照以下六层分析框架进行系统性分析：

## 股票数据
{stock_info}

## 市场环境
{market_info}

## 分析框架（业界惯例）

请按照以下六层分析框架进行系统性分析：

【第一层：风险控制检查（最高优先级）】
⚠️ 必须首先检查所有股票的涨跌停风险：
- 接近涨停（涨幅≥9%）：追高风险大，不建议追高
- 接近跌停（跌幅≥9%）：可能继续下跌，不建议抄底

【第二层：市场环境评估】
- 市场情绪层级：极端熊市暂停交易，熊市需概率≥0.70，弱震荡需概率≥0.65
- 主力资金流向：持续流入利好，持续流出需警惕
- 上证指数MA20：站上MA20为多头，跌破MA20为空头

【第三层：技术面分析】
- RSI状态：>70超买（回调风险），<30超卖（反弹机会）
- MACD状态：金叉看涨，死叉看跌
- 均线排列：多头排列看涨，空头排列看跌，震荡观望
- 支撑位/阻力位：接近支撑位可买入，接近阻力位需谨慎

【第四层：筹码分布分析】（中期分析）
- 上方筹码比例：<30%拉升阻力小，>60%阻力大
- 筹码集中度：高集中度表示筹码锁定，利于拉升

【第五层：成交量分析】
- 放量（成交量比率>2）：资金关注度高
- 缩量（成交量比率<0.5）：资金关注度低

【第六层：综合建议】
基于以上分析，给出具体的操作建议

## 分析要求

请针对以上{len(stock_data_list)}只A股，给出：

1. **市场判断**：当前A股市场整体趋势如何？主力资金流向说明了什么？

2. **个股分析**：对每只股票给出：
   - 技术面分析（趋势、支撑位、压力位）
   - 买卖建议（买入/持有/卖出）
   - 建议仓位（三种风险偏好）：
     * 保守型仓位：风险厌恶，追求稳健
     * 适度型仓位：平衡风险与收益
     * 激进型仓位：风险偏好，追求高收益
   - 止损价位（建议-8%）
   - 目标价位（建议+10%）

3. **风险提示**：
   - 涨跌停限制对交易的影响
   - 近期需要注意的风险点

4. **操作策略**：
   - 短期（1-5天）操作建议
   - 中期（1-4周）操作建议

请用中文回复，格式清晰，建议具体可操作。
"""

    return prompt


def analyze_with_llm(stock_data_list, market_data, main_fund_data):
    """
    使用大模型分析股票

    Args:
        stock_data_list: 股票数据列表
        market_data: 市场数据
        main_fund_data: 主力资金数据

    Returns:
        str: 分析报告
    """
    try:
        from llm_services.qwen_engine import chat_with_llm

        print("🤖 正在调用通义千问分析A股...")

        prompt = generate_llm_prompt(stock_data_list, market_data, main_fund_data)

        # 打印提示词摘要（调试用）
        print(f"  提示词长度: {len(prompt)} 字符")

        # 调用大模型
        result = chat_with_llm(prompt, enable_thinking=False)

        print("✅ 大模型分析完成")
        return result

    except Exception as e:
        print(f"❌ 大模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_llm_report(report_content, stock_data_list):
    """
    保存大模型报告

    Args:
        report_content: 报告内容
        stock_data_list: 股票数据列表

    Returns:
        str: 文件路径
    """
    os.makedirs('data', exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')
    filepath = f'data/a_stock_llm_recommendations_{date_str}.txt'

    # 构建完整报告（标准 Markdown 格式）
    content = """## 自选股概览

"""
    # 添加股票概览
    for data in stock_data_list:
        stock_name = A_STOCK_WATCHLIST.get(data['stock_code'], data['stock_code'])
        content += f"- **{stock_name}** ({data['stock_code']}): {data['current_price']:.2f} 元 ({data['change_percent']:+.2f}%)\n"

    content += "\n---\n\n"
    content += "## AI分析报告\n\n"
    content += report_content

    # 保存文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 报告已保存到 {filepath}")
    return filepath


def generate_ai_report(stocks=None, save_file=True):
    """
    生成A股AI报告

    Args:
        stocks: 股票列表，默认使用自选股
        save_file: 是否保存文件

    Returns:
        str: 报告内容
    """
    if stocks is None:
        stocks = A_STOCK_WATCHLIST

    print(f"\n{'=' * 60}")
    print("📊 A股AI报告生成")
    print(f"{'=' * 60}")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 分析股票: {len(stocks)} 只")

    # 获取主力资金
    print("\n📊 获取主力资金数据...")
    main_fund_service = MainFundFlowService()
    main_fund_data = main_fund_service.get_latest()
    if not main_fund_data:
        main_fund_data = {}

    # 获取主力资金趋势（5日/20日累积）
    main_fund_history = main_fund_service.fetch_history()
    if main_fund_history is not None and not main_fund_history.empty:
        main_fund_data['net_flow_5d_sum'] = main_fund_history['main_net_flow'].tail(5).sum()
        main_fund_data['net_flow_20d_sum'] = main_fund_history['main_net_flow'].tail(20).sum()
        # 连续流入天数
        net_flow_series = main_fund_history['main_net_flow'].tail(20)
        consecutive_inflow = 0
        for val in net_flow_series.iloc[::-1]:
            if val > 0:
                consecutive_inflow += 1
            else:
                break
        main_fund_data['consecutive_inflow'] = consecutive_inflow

    # 获取上证指数
    print("📊 获取上证指数数据...")
    sh_df = get_index_data('sh', period_days=30)
    market_data = {}
    if sh_df is not None and not sh_df.empty:
        latest_sh = sh_df.iloc[-1]
        prev_sh = sh_df.iloc[-2] if len(sh_df) > 1 else latest_sh
        market_data['sh_close'] = latest_sh['Close']
        market_data['sh_change'] = (latest_sh['Close'] / prev_sh['Close'] - 1) * 100

        # 计算MA20
        sh_df['MA20'] = sh_df['Close'].rolling(20).mean()
        market_data['sh_ma20'] = sh_df['MA20'].iloc[-1]
        market_data['sh_vs_ma20'] = (latest_sh['Close'] / sh_df['MA20'].iloc[-1] - 1) * 100 if sh_df['MA20'].iloc[-1] else None

    # 计算市场情绪层级（基于全量53只股票涨跌）
    print("📊 计算市场情绪层级（全量股票）...")
    all_stock_changes = []
    for stock_code in A_STOCK_TRAINING_LIST.keys():
        realtime = get_a_stock_info_tencent(stock_code)
        if realtime and realtime.get('change_percent') is not None:
            all_stock_changes.append(realtime['change_percent'])

    up_count = sum(1 for c in all_stock_changes if c > 0)
    total_count = len(all_stock_changes)
    up_ratio = up_count / total_count if total_count > 0 else 0.5

    if up_ratio < 0.20:
        market_sentiment = {
            'layer': 'extreme_bear',
            'name': '极端熊市',
            'action': '暂停交易',
            'dynamic_threshold': 1.0,
        }
    elif up_ratio < 0.30:
        market_sentiment = {
            'layer': 'bear',
            'name': '熊市',
            'action': '需概率≥0.70',
            'dynamic_threshold': 0.70,
        }
    elif up_ratio < 0.40:
        market_sentiment = {
            'layer': 'weak',
            'name': '弱震荡',
            'action': '需概率≥0.65',
            'dynamic_threshold': 0.65,
        }
    else:
        market_sentiment = {
            'layer': 'normal',
            'name': '正常市场',
            'action': '正常交易',
            'dynamic_threshold': 0.50,
        }
    market_sentiment['up_ratio'] = up_ratio
    market_sentiment['total_count'] = total_count
    market_data['sentiment'] = market_sentiment
    print(f"  市场情绪: {market_sentiment['name']}（{total_count}只股票，上涨比例 {up_ratio:.1%}）")

    # 获取股票技术数据（仅核心持仓）
    print("\n📊 获取股票技术数据（核心持仓）...")
    stock_data_list = []
    for stock_code, stock_name in stocks.items():
        print(f"  分析 {stock_code} {stock_name}...")
        data = get_stock_technical_data(stock_code)
        if data:
            stock_data_list.append(data)
        else:
            print(f"  ⚠️ 无法获取 {stock_code} 数据")

    if not stock_data_list:
        print("❌ 无法获取任何股票数据")
        return None

    # 调用大模型分析
    print("\n🤖 调用大模型分析...")
    report = analyze_with_llm(stock_data_list, market_data, main_fund_data)

    if report and save_file:
        save_llm_report(report, stock_data_list)

    return report


def main():
    parser = argparse.ArgumentParser(description='A股AI报告生成')
    parser.add_argument('--stocks', type=str, default=None,
                       help='股票代码列表，逗号分隔，如 300440,002655')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存文件')
    parser.add_argument('--force', action='store_true',
                       help='强制生成报告')
    parser.add_argument('--no-email', action='store_true',
                       help='兼容参数（此脚本不发送邮件）')

    args = parser.parse_args()

    # 解析股票列表
    stocks = None
    if args.stocks:
        stock_codes = args.stocks.split(',')
        stocks = {code: A_STOCK_WATCHLIST.get(code, code) for code in stock_codes}

    # 生成报告
    generate_ai_report(stocks=stocks, save_file=not args.no_save)


if __name__ == '__main__':
    main()
