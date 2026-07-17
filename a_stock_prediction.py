#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股预测脚本

功能：
1. 获取A股实时行情
2. 生成CatBoost预测
3. 综合分析报告
4. 发送邮件通知
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import (
    A_STOCK_WATCHLIST,
    get_limit_rate,
    get_market_code,
)
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.northbound_data import get_northbound_features, NorthboundDataService
from a_stock_ml_model import AStockTradingModel


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def get_stock_realtime_info(stock_code):
    """
    获取股票实时信息

    Args:
        stock_code: 股票代码

    Returns:
        dict: 股票信息
    """
    info = get_a_stock_info_tencent(stock_code)
    if info:
        # 添加涨跌停信息
        limit_rate = get_limit_rate(stock_code)
        info['limit_rate'] = f"{limit_rate * 100:.0f}%"
        info['market'] = get_market_code(stock_code).upper()
    return info


def analyze_stock(stock_code, stock_name, horizon=20):
    """
    分析单只股票

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        horizon: 预测周期

    Returns:
        dict: 分析结果
    """
    result = {
        'code': stock_code,
        'name': stock_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 获取实时行情
    realtime = get_stock_realtime_info(stock_code)
    if realtime:
        result['current_price'] = realtime.get('current_price')
        result['change_percent'] = realtime.get('change_percent')
        result['limit_rate'] = realtime.get('limit_rate')
        result['market'] = realtime.get('market')
    else:
        result['error'] = '无法获取实时行情'
        return result

    # 获取历史数据
    df = get_a_stock_data(stock_code, period_days=100)
    if df is None or df.empty:
        result['error'] = '无法获取历史数据'
        return result

    # 计算简单技术指标
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    latest = df.iloc[-1]
    result['ma5'] = latest['MA5']
    result['ma20'] = latest['MA20']
    result['ma60'] = latest.get('MA60')

    # 均线位置
    if result['current_price'] and result['ma20']:
        result['price_vs_ma20'] = (result['current_price'] / result['ma20'] - 1) * 100

    # 近期涨跌
    if len(df) >= 5:
        result['return_5d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
    if len(df) >= 20:
        result['return_20d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

    return result


def generate_prediction_report(stocks=None, horizon=20, send_email=False):
    """
    生成预测报告

    Args:
        stocks: 股票列表，默认使用自选股
        horizon: 预测周期
        send_email: 是否发送邮件
    """
    if stocks is None:
        stocks = A_STOCK_WATCHLIST

    print_header(f"A股预测报告 - {horizon}天周期")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 预测股票: {len(stocks)} 只")

    # 获取北向资金
    print("\n📊 北向资金数据:")
    northbound_service = NorthboundDataService()
    latest_nb = northbound_service.get_latest()
    if latest_nb:
        print(f"  日期: {latest_nb['date']}")
        print(f"  净买入: {latest_nb['net_buy']:.2f} 亿")
        print(f"  沪股通: {latest_nb['sh_net_buy']:.2f} 亿")
        print(f"  深股通: {latest_nb['sz_net_buy']:.2f} 亿")

    # 获取指数
    print("\n📈 上证指数:")
    sh_df = get_index_data('sh', period_days=30)
    if sh_df is not None and not sh_df.empty:
        latest_sh = sh_df.iloc[-1]
        prev_sh = sh_df.iloc[-2] if len(sh_df) > 1 else latest_sh
        sh_change = (latest_sh['Close'] / prev_sh['Close'] - 1) * 100
        print(f"  收盘: {latest_sh['Close']:.2f}")
        print(f"  涨跌: {sh_change:+.2f}%")

    # 分析每只股票
    results = []
    for stock_code, stock_name in stocks.items():
        print(f"\n📊 分析 {stock_code} {stock_name}:")
        result = analyze_stock(stock_code, stock_name, horizon)
        results.append(result)

        if 'error' in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  现价: {result['current_price']:.2f} ({result['change_percent']:+.2f}%)")
            print(f"  涨跌停: {result['limit_rate']}")
            print(f"  MA20: {result['ma20']:.2f} ({result.get('price_vs_ma20', 0):+.2f}%)")
            if result.get('return_5d'):
                print(f"  5日涨跌: {result['return_5d']:+.2f}%")
            if result.get('return_20d'):
                print(f"  20日涨跌: {result['return_20d']:+.2f}%")

    # 保存报告
    report_file = f"data/a_stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 报告已保存: {report_file}")

    # 发送邮件
    if send_email:
        try:
            from message_services.email_sender import send_email_with_html
            # TODO: 生成HTML邮件并发送
            print("📧 邮件发送功能待完善")
        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='A股预测脚本')
    parser.add_argument('--stocks', type=str, default=None,
                       help='股票代码列表，逗号分隔，如 300440,002655')
    parser.add_argument('--horizon', type=int, default=20, choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月（默认）')
    parser.add_argument('--email', action='store_true',
                       help='发送邮件通知')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件通知（默认）')
    parser.add_argument('--comprehensive', action='store_true',
                       help='运行综合分析（训练+预测+报告）')
    parser.add_argument('--ai', action='store_true',
                       help='生成AI买卖建议')

    args = parser.parse_args()

    # 解析股票列表
    stocks = None
    if args.stocks:
        stock_codes = args.stocks.split(',')
        stocks = {code: A_STOCK_WATCHLIST.get(code, code) for code in stock_codes}

    # 综合分析模式
    if args.comprehensive:
        print_header("A股综合分析")

        # 1. 训练模型
        print("\n📊 步骤1: 训练模型...")
        model = AStockTradingModel(horizon=args.horizon)
        try:
            model.train(use_feature_selection=True)
            print("✅ 模型训练完成")
        except Exception as e:
            print(f"⚠️ 模型训练失败: {e}")

        # 2. 生成预测
        print("\n📊 步骤2: 生成预测...")
        try:
            model.predict(use_feature_selection=True, mode='production')
            print("✅ 预测完成")
        except Exception as e:
            print(f"⚠️ 预测失败: {e}")

        # 3. 生成AI报告
        if args.ai:
            print("\n📊 步骤3: 生成AI买卖建议...")
            try:
                from a_stock_email import generate_ai_report
                generate_ai_report(stocks=stocks)
            except Exception as e:
                print(f"⚠️ AI报告生成失败: {e}")

        # 4. 生成报告
        print("\n📊 步骤4: 生成报告...")
        generate_prediction_report(stocks=stocks, horizon=args.horizon, send_email=args.email)

    elif args.ai:
        # 仅生成AI报告
        from a_stock_email import generate_ai_report
        generate_ai_report(stocks=stocks)

    else:
        # 仅生成报告
        generate_prediction_report(stocks=stocks, horizon=args.horizon, send_email=args.email)


if __name__ == '__main__':
    main()
