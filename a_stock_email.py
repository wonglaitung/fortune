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
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import A_STOCK_WATCHLIST, get_limit_rate
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.northbound_data import NorthboundDataService


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

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 获取实时价格
    realtime = get_a_stock_info_tencent(stock_code)
    current_price = realtime['current_price'] if realtime else latest['Close']

    return {
        'stock_code': stock_code,
        'current_price': current_price,
        'change_percent': realtime['change_percent'] if realtime else None,
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
        'limit_rate': get_limit_rate(stock_code),
    }


def generate_llm_prompt(stock_data_list, market_data, northbound_data):
    """
    生成大模型提示词

    Args:
        stock_data_list: 股票数据列表
        market_data: 市场数据
        northbound_data: 北向资金数据

    Returns:
        str: 提示词
    """
    # 构建股票信息
    stock_info = ""
    for data in stock_data_list:
        stock_name = A_STOCK_WATCHLIST.get(data['stock_code'], data['stock_code'])
        stock_info += f"""
### {stock_name} ({data['stock_code']})

**基本数据**：
- 当前价格：{data['current_price']:.2f} 元
- 今日涨跌：{data['change_percent']:+.2f}%
- 涨跌停限制：{data['limit_rate']*100:.0f}%

**技术指标**：
- 均线位置：
  - MA5: {data['ma5']:.2f} ({data['price_vs_ma5']:+.2f}%)
  - MA20: {data['ma20']:.2f} ({data['price_vs_ma20']:+.2f}%)
- 近期涨跌：
  - 5日涨跌：{data['return_5d']:+.2f}%
  - 20日涨跌：{data['return_20d']:+.2f}%
- RSI(14)：{data['rsi_14']:.1f}
- MACD：{data['macd']:.4f}，信号线：{data['macd_signal']:.4f}，柱状：{data['macd_hist']:.4f}
- 布林带位置：{data['bb_position']:.1f}%（0%=下轨，100%=上轨）
- 成交量比率：{data['volume_ratio']:.2f}x

**技术判断**：
- 趋势：{'上涨' if data['price_vs_ma20'] > 0 else '下跌'}
- RSI状态：{'超买' if data['rsi_14'] > 70 else '超卖' if data['rsi_14'] < 30 else '中性'}
- MACD状态：{'金叉' if data['macd_hist'] > 0 else '死叉'}
"""

    # 构建市场环境
    market_info = f"""
### 市场环境

**上证指数**：
- 收盘：{market_data.get('sh_close', 'N/A')}
- 涨跌：{market_data.get('sh_change', 'N/A')}%

**北向资金**：
- 净买入：{northbound_data.get('net_buy', 0):.2f} 亿
- 沪股通：{northbound_data.get('sh_net_buy', 0):.2f} 亿
- 深股通：{northbound_data.get('sz_net_buy', 0):.2f} 亿
"""

    # 完整提示词
    prompt = f"""你是一位专业的A股投资分析师。请根据以下股票数据和市场环境，给出专业的投资建议。

## 股票数据

{stock_info}

## 市场环境

{market_info}

## 分析要求

请针对以上{len(stock_data_list)}只A股，给出：

1. **市场判断**：当前A股市场整体趋势如何？北向资金流向说明了什么？

2. **个股分析**：对每只股票给出：
   - 技术面分析（趋势、支撑位、压力位）
   - 买卖建议（买入/持有/卖出）
   - 建议仓位（0-100%）
   - 止损价位
   - 目标价位

3. **风险提示**：
   - 涨跌停限制对交易的影响
   - 近期需要注意的风险点

4. **操作策略**：
   - 短期（1-5天）操作建议
   - 中期（1-4周）操作建议

请用中文回复，格式清晰，建议具体可操作。
"""

    return prompt


def analyze_with_llm(stock_data_list, market_data, northbound_data):
    """
    使用大模型分析股票

    Args:
        stock_data_list: 股票数据列表
        market_data: 市场数据
        northbound_data: 北向资金数据

    Returns:
        str: 分析报告
    """
    try:
        from llm_services.qwen_engine import chat_with_llm

        print("🤖 正在调用通义千问分析A股...")

        prompt = generate_llm_prompt(stock_data_list, market_data, northbound_data)

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

    # 构建完整报告
    content = f"""{'=' * 80}
A股大模型买卖建议报告
日期: {date_str}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

## 自选股概览

"""
    # 添加股票概览
    for data in stock_data_list:
        stock_name = A_STOCK_WATCHLIST.get(data['stock_code'], data['stock_code'])
        content += f"- {stock_name} ({data['stock_code']}): {data['current_price']:.2f} 元 ({data['change_percent']:+.2f}%)\n"

    content += f"\n{'=' * 80}\n\n"
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

    # 获取北向资金
    print("\n📊 获取北向资金数据...")
    northbound_service = NorthboundDataService()
    northbound_data = northbound_service.get_latest()
    if not northbound_data:
        northbound_data = {}

    # 获取上证指数
    print("📊 获取上证指数数据...")
    sh_df = get_index_data('sh', period_days=30)
    market_data = {}
    if sh_df is not None and not sh_df.empty:
        latest_sh = sh_df.iloc[-1]
        prev_sh = sh_df.iloc[-2] if len(sh_df) > 1 else latest_sh
        market_data['sh_close'] = latest_sh['Close']
        market_data['sh_change'] = (latest_sh['Close'] / prev_sh['Close'] - 1) * 100

    # 获取股票技术数据
    print("\n📊 获取股票技术数据...")
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
    report = analyze_with_llm(stock_data_list, market_data, northbound_data)

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
