#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议

⚠️ 运行时机：建议在A股收市后（15:00 CST）运行
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入A股配置
from a_stock_config import (
    A_STOCK_WATCHLIST,
    A_STOCK_SECTOR_MAPPING,
    get_limit_rate,
)

# 导入数据服务
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.northbound_data import NorthboundDataService

# 股票名称映射
STOCK_NAMES = A_STOCK_WATCHLIST
STOCK_LIST = list(A_STOCK_WATCHLIST.keys())


def read_llm_recommendations(llm_file):
    """
    读取大模型建议文件

    Args:
        llm_file: 大模型建议文件路径

    Returns:
        str: 大模型建议内容
    """
    if not os.path.exists(llm_file):
        print(f"⚠️ 大模型建议文件不存在: {llm_file}")
        return None

    with open(llm_file, 'r', encoding='utf-8') as f:
        return f.read()


def read_ml_predictions(horizon=20):
    """
    读取ML预测结果

    Args:
        horizon: 预测周期

    Returns:
        DataFrame: 预测结果
    """
    # 查找最新的预测文件
    import glob
    files = glob.glob(f'data/ml_trading_model_catboost_predictions_{horizon}d.csv')

    if not files:
        print(f"⚠️ 未找到 {horizon}d 预测文件")
        return None

    latest_file = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest_file)
    print(f"✅ 读取预测文件: {latest_file}")
    return df


def get_stock_analysis(stock_code):
    """
    获取单只股票的详细分析

    Args:
        stock_code: 股票代码

    Returns:
        dict: 股票分析结果
    """
    stock_name = A_STOCK_WATCHLIST.get(stock_code, stock_code)
    result = {
        'code': stock_code,
        'name': stock_name,
        'limit_rate': get_limit_rate(stock_code),
    }

    # 获取实时行情
    realtime = get_a_stock_info_tencent(stock_code)
    if realtime:
        result['current_price'] = realtime.get('current_price')
        result['change_percent'] = realtime.get('change_percent')
        result['prev_close'] = realtime.get('prev_close')

    # 获取历史数据
    df = get_a_stock_data(stock_code, period_days=100)
    if df is not None and not df.empty:
        # 计算技术指标
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        latest = df.iloc[-1]
        result['ma5'] = latest['MA5']
        result['ma10'] = latest['MA10']
        result['ma20'] = latest['MA20']
        result['ma60'] = latest.get('MA60')

        # 计算涨跌
        if len(df) >= 5:
            result['return_5d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        if len(df) >= 20:
            result['return_20d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi_14'] = 100 - (100 / (1 + rs.iloc[-1]))

        # 计算MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        result['macd'] = df['EMA12'].iloc[-1] - df['EMA26'].iloc[-1]

    return result


def analyze_market():
    """
    分析市场环境

    Returns:
        dict: 市场分析结果
    """
    result = {}

    # 上证指数
    sh_df = get_index_data('sh', period_days=30)
    if sh_df is not None and not sh_df.empty:
        latest = sh_df.iloc[-1]
        prev = sh_df.iloc[-2] if len(sh_df) > 1 else latest
        result['sh_close'] = latest['Close']
        result['sh_change'] = (latest['Close'] / prev['Close'] - 1) * 100

        # 计算MA
        sh_df['MA20'] = sh_df['Close'].rolling(20).mean()
        result['sh_ma20'] = sh_df['MA20'].iloc[-1]
        result['sh_vs_ma20'] = (latest['Close'] / sh_df['MA20'].iloc[-1] - 1) * 100

    # 北向资金
    northbound_service = NorthboundDataService()
    nb_data = northbound_service.get_latest()
    if nb_data:
        result['northbound_net_buy'] = nb_data.get('net_buy', 0)
        result['northbound_sh'] = nb_data.get('sh_net_buy', 0)
        result['northbound_sz'] = nb_data.get('sz_net_buy', 0)

    return result


def generate_comprehensive_report(llm_content, ml_predictions_20d, stock_analyses, market_data):
    """
    生成综合分析报告

    Args:
        llm_content: 大模型建议内容
        ml_predictions_20d: 20天ML预测
        stock_analyses: 股票分析结果
        market_data: 市场数据

    Returns:
        str: 综合报告
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    report = f"""{'=' * 80}
A股综合分析报告
日期: {date_str}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

## 一、市场概况

### 1.1 上证指数
- 收盘: {market_data.get('sh_close', 'N/A'):.2f}
- 涨跌: {market_data.get('sh_change', 0):+.2f}%
- MA20: {market_data.get('sh_ma20', 'N/A'):.2f}
- 相对MA20: {market_data.get('sh_vs_ma20', 0):+.2f}%

### 1.2 北向资金
- 净买入: {market_data.get('northbound_net_buy', 0):.2f} 亿
- 沪股通: {market_data.get('northbound_sh', 0):.2f} 亿
- 深股通: {market_data.get('northbound_sz', 0):.2f} 亿

---

## 二、自选股技术分析

"""

    # 添加每只股票的分析
    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        report += f"""### {name} ({code})

**基本数据**：
- 当前价格: {analysis.get('current_price', 'N/A')} 元
- 今日涨跌: {analysis.get('change_percent', 0):+.2f}%
- 涨跌停限制: {analysis.get('limit_rate', 0) * 100:.0f}%

**技术指标**：
- MA5: {analysis.get('ma5', 'N/A'):.2f}
- MA10: {analysis.get('ma10', 'N/A'):.2f}
- MA20: {analysis.get('ma20', 'N/A'):.2f}
- RSI(14): {analysis.get('rsi_14', 'N/A'):.1f}

**近期涨跌**：
- 5日: {analysis.get('return_5d', 0):+.2f}%
- 20日: {analysis.get('return_20d', 0):+.2f}%

"""

    # 添加ML预测结果
    report += """---

## 三、机器学习预测（20天周期）

"""
    if ml_predictions_20d is not None and not ml_predictions_20d.empty:
        for _, row in ml_predictions_20d.iterrows():
            code = row.get('Stock_Code', '')
            name = A_STOCK_WATCHLIST.get(code, code)
            pred_proba = row.get('Prediction_Proba', 0.5)
            pred_label = '上涨' if pred_proba >= 0.5 else '下跌'
            confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba

            report += f"""### {name} ({code})
- 预测方向: **{pred_label}**
- 置信度: {confidence:.1%}

"""
    else:
        report += "⚠️ 未找到ML预测结果\n\n"

    # 添加大模型建议
    report += f"""---

## 四、AI分析建议

{llm_content if llm_content else '⚠️ 未找到大模型建议'}

---

## 五、操作建议汇总

| 股票 | 代码 | 当前价 | 涨跌 | 建议 |
|------|------|--------|------|------|
"""

    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', 'N/A')
        change = analysis.get('change_percent', 0)

        # 简单建议逻辑
        if analysis.get('return_20d', 0) > 10 and analysis.get('rsi_14', 50) > 70:
            suggestion = '谨慎持有'
        elif analysis.get('return_20d', 0) < -20:
            suggestion = '观望'
        elif analysis.get('rsi_14', 50) < 30:
            suggestion = '关注'
        else:
            suggestion = '持有'

        report += f"| {name} | {code} | {price} | {change:+.2f}% | {suggestion} |\n"

    report += f"""
---

## 六、风险提示

1. **涨跌停风险**: 创业板/科创板涨跌停限制20%，主板10%
2. **北向资金**: 关注外资流向变化
3. **市场情绪**: 上证指数跌破MA20时需谨慎

---

*本报告仅供参考，不构成投资建议*
"""

    return report


def main():
    parser = argparse.ArgumentParser(description='A股综合分析')
    parser.add_argument('--llm-file', type=str, default=None,
                       help='大模型建议文件路径')
    parser.add_argument('--use-cached-predictions', action='store_true',
                       help='使用缓存的预测结果')
    parser.add_argument('--horizon', type=int, default=20,
                       help='预测周期（默认20天）')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📊 A股综合分析")
    print("=" * 60)
    print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 分析股票: {len(STOCK_LIST)} 只")
    print("=" * 60)

    # 1. 读取大模型建议
    print("\n📊 读取大模型建议...")
    llm_file = args.llm_file
    if llm_file is None:
        # 查找最新的建议文件
        import glob
        files = glob.glob('data/a_stock_llm_recommendations_*.txt')
        if files:
            llm_file = max(files, key=os.path.getmtime)
            print(f"  使用文件: {llm_file}")

    llm_content = None
    if llm_file:
        llm_content = read_llm_recommendations(llm_file)
        if llm_content:
            print("  ✅ 大模型建议读取成功")

    # 2. 读取ML预测结果
    print("\n📊 读取ML预测结果...")
    ml_predictions = read_ml_predictions(args.horizon)

    # 3. 分析市场
    print("\n📊 分析市场环境...")
    market_data = analyze_market()
    if market_data:
        print(f"  上证指数: {market_data.get('sh_close', 'N/A'):.2f} ({market_data.get('sh_change', 0):+.2f}%)")
        print(f"  北向资金: {market_data.get('northbound_net_buy', 0):.2f} 亿")

    # 4. 分析每只股票
    print("\n📊 分析自选股...")
    stock_analyses = {}
    for code in STOCK_LIST:
        print(f"  分析 {code}...")
        analysis = get_stock_analysis(code)
        stock_analyses[code] = analysis

    # 5. 生成综合报告
    print("\n📊 生成综合报告...")
    report = generate_comprehensive_report(llm_content, ml_predictions, stock_analyses, market_data)

    # 保存报告
    os.makedirs('data', exist_ok=True)
    report_file = f"data/a_stock_comprehensive_recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告已保存: {report_file}")

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("📊 报告摘要")
    print("=" * 60)
    print(f"  市场状态: 上证 {market_data.get('sh_change', 0):+.2f}%")
    print(f"  北向资金: {market_data.get('northbound_net_buy', 0):.2f} 亿")
    print(f"\n  个股建议:")
    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', 'N/A')
        change = analysis.get('change_percent', 0)
        print(f"    {name}: {price} 元 ({change:+.2f}%)")

    print("\n" + "=" * 60)
    print(f"📄 完整报告: {report_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
