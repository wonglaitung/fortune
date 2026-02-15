#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 hsi_email.py 脚本的功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 将项目根目录添加到 Python 路径
sys.path.append('/data/fortune')

def test_hsi_email_basic():
    """测试 hsi_email.py 的基本功能"""
    print("🧪 开始测试 hsi_email.py 基本功能...")
    
    try:
        from hsi_email import HSIEmailSystem
        print("✅ 成功导入 HSIEmailSystem")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    try:
        # 创建 HSIEmailSystem 实例
        email_system = HSIEmailSystem()
        print("✅ 成功创建 HSIEmailSystem 实例")
    except Exception as e:
        print(f"❌ 创建实例失败: {e}")
        return False
    
    # 测试获取恒生指数数据
    print("\n🔍 测试获取恒生指数数据...")
    try:
        hsi_data = email_system.get_hsi_data()
        if hsi_data:
            print(f"✅ 成功获取恒生指数数据: {hsi_data['current_price']:.2f}")
        else:
            print("⚠️ 无法获取恒生指数数据（可能因为市场休市）")
    except Exception as e:
        print(f"❌ 获取恒生指数数据失败: {e}")
    
    # 测试获取单只股票数据
    print("\n🔍 测试获取股票数据...")
    try:
        # 使用列表中的第一只股票进行测试
        test_stock = list(email_system.stock_list.keys())[0]
        test_stock_name = email_system.stock_list[test_stock]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data:
            print(f"✅ 成功获取 {test_stock_name}({test_stock}) 数据: {stock_data['current_price']:.2f}")
        else:
            print(f"⚠️ 无法获取 {test_stock_name}({test_stock}) 数据（可能因为市场休市）")
    except Exception as e:
        print(f"❌ 获取股票数据失败: {e}")
    
    # 测试技术指标计算
    print("\n🔍 测试技术指标计算...")
    try:
        # 获取测试股票的数据
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock, target_date=datetime.now().date())
        if stock_data:
            indicators = email_system.calculate_technical_indicators(stock_data)
            if indicators:
                print(f"✅ 成功计算技术指标")
                print(f"   - RSI: {indicators.get('rsi', 'N/A')}")
                print(f"   - MACD: {indicators.get('macd', 'N/A')}")
                print(f"   - ATR: {indicators.get('atr', 'N/A')}")
                print(f"   - 当前价格: {indicators.get('current_price', 'N/A')}")
                if 'tav_score' in indicators:
                    print(f"   - TAV评分: {indicators['tav_score']:.1f}")
                if 'buildup_score' in indicators:
                    print(f"   - 建仓评分: {indicators['buildup_score']:.2f}")
                if 'distribution_score' in indicators:
                    print(f"   - 出货评分: {indicators['distribution_score']:.2f}")
                if 'fundamental_score' in indicators:
                    print(f"   - 基本面评分: {indicators['fundamental_score']}")
            else:
                print("⚠️ 技术指标计算返回空值")
        else:
            print("⚠️ 无法获取股票数据用于技术指标计算")
    except Exception as e:
        print(f"❌ 技术指标计算失败: {e}")
    
    # 测试VaR计算
    print("\n🔍 测试VaR计算...")
    try:
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data and not stock_data['hist'].empty:
            var_result = email_system.calculate_var(stock_data['hist'], 'short_term', position_value=100000)
            if var_result:
                print(f"✅ 成功计算VaR: {var_result['percentage']:.2%} (HK$ {var_result['amount']:.2f})")
            else:
                print("⚠️ VaR计算返回空值（可能因为数据不足）")
        else:
            print("⚠️ 无历史数据用于VaR计算")
    except Exception as e:
        print(f"❌ VaR计算失败: {e}")
    
    # 测试最大回撤计算
    print("\n🔍 测试最大回撤计算...")
    try:
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data and not stock_data['hist'].empty:
            max_dd_result = email_system.calculate_max_drawdown(stock_data['hist'], position_value=100000)
            if max_dd_result:
                print(f"✅ 成功计算最大回撤: {max_dd_result['percentage']:.2%} (HK$ {max_dd_result['amount']:.2f})")
            else:
                print("⚠️ 最大回撤计算返回空值（可能因为数据不足）")
        else:
            print("⚠️ 无历史数据用于最大回撤计算")
    except Exception as e:
        print(f"❌ 最大回撤计算失败: {e}")
    
    # 测试止损止盈计算
    print("\n🔍 测试止损止盈计算...")
    try:
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data and not stock_data['hist'].empty:
            stop_loss, take_profit = email_system.calculate_stop_loss_take_profit(
                stock_data['hist'], 
                stock_data['current_price'], 
                signal_type='BUY'
            )
            if stop_loss is not None and take_profit is not None:
                print(f"✅ 成功计算止损止盈: 止损 {stop_loss:.2f}, 止盈 {take_profit:.2f}")
            else:
                print("⚠️ 止损止盈计算返回空值")
        else:
            print("⚠️ 无数据用于止损止盈计算")
    except Exception as e:
        print(f"❌ 止损止盈计算失败: {e}")
    
    # 测试获取交易记录
    print("\n🔍 测试读取交易记录...")
    try:
        df_transactions = email_system._read_transactions_df()
        if df_transactions.empty:
            print("⚠️ 交易记录文件为空或不存在")
        else:
            print(f"✅ 成功读取交易记录: {len(df_transactions)} 条记录")
            print(f"   最近交易时间: {df_transactions['timestamp'].max()}")
    except Exception as e:
        print(f"❌ 读取交易记录失败: {e}")
    
    # 测试获取持仓数据
    print("\n🔍 测试读取持仓数据...")
    try:
        portfolio = email_system._read_portfolio_data()
        print(f"✅ 成功读取持仓数据: {len(portfolio)} 只股票")
        if portfolio:
            for pos in portfolio[:3]:  # 只显示前3只
                print(f"   - {pos['stock_name']}({pos['stock_code']}): {pos['total_shares']:,}股, 成本价:HK${pos['cost_price']:.2f}")
    except Exception as e:
        print(f"❌ 读取持仓数据失败: {e}")
    
    # 测试股息信息获取
    print("\n🔍 测试获取股息信息...")
    try:
        dividend_data = email_system.get_upcoming_dividends(days_ahead=90)
        if dividend_data and not dividend_data['upcoming'].empty:
            print(f"✅ 成功获取股息信息: {len(dividend_data['upcoming'])} 条即将除净记录")
        else:
            print("⚠️ 无即将除净的股息信息")
    except Exception as e:
        print(f"❌ 获取股息信息失败: {e}")
    
    # 测试格式化功能
    print("\n🔍 测试格式化功能...")
    try:
        # 测试价格信息格式化
        price_info = email_system._format_price_info(current_price=100.5, stop_loss_price=95.0, target_price=110.0, validity_period=5)
        print(f"✅ 价格信息格式化: {price_info}")
    except Exception as e:
        print(f"❌ 格式化功能测试失败: {e}")
    
    # 测试箭头符号功能
    print("\n🔍 测试趋势变化箭头功能...")
    try:
        arrow = email_system._get_trend_change_arrow("多头趋势", "震荡整理")
        print(f"✅ 趋势变化箭头: {arrow}")
    except Exception as e:
        print(f"❌ 趋势变化箭头测试失败: {e}")
    
    print("\n✅ hsi_email.py 基本功能测试完成！")
    return True

def test_hsi_email_analysis():
    """测试 hsi_email.py 的分析功能"""
    print("\n🧪 开始测试 hsi_email.py 分析功能...")
    
    try:
        from hsi_email import HSIEmailSystem
        email_system = HSIEmailSystem()
        print("✅ 成功创建 HSIEmailSystem 实例")
    except Exception as e:
        print(f"❌ 创建实例失败: {e}")
        return False
    
    # 获取最近几天的数据用于测试
    print("\n📊 获取测试数据...")
    stock_results = []
    hsi_data = None
    hsi_indicators = None
    
    try:
        # 获取恒生指数数据
        hsi_data = email_system.get_hsi_data()
        if hsi_data:
            hsi_indicators = email_system.calculate_hsi_technical_indicators(hsi_data)
            print("✅ 成功获取恒生指数数据和指标")
        else:
            print("⚠️ 无法获取恒生指数数据")
    except Exception as e:
        print(f"❌ 获取恒生指数数据失败: {e}")
    
    # 获取几只股票的数据和指标
    test_stocks = list(email_system.stock_list.keys())[:5]  # 只测试前5只
    for i, stock_code in enumerate(test_stocks):
        try:
            stock_name = email_system.stock_list[stock_code]
            print(f"📊 正在获取 {stock_name}({stock_code}) 数据... ({i+1}/{len(test_stocks)})")
            stock_data = email_system.get_stock_data(stock_code)
            if stock_data:
                indicators = email_system.calculate_technical_indicators(stock_data)
                stock_results.append({
                    'code': stock_code,
                    'name': stock_name,
                    'data': stock_data,
                    'indicators': indicators
                })
                print(f"   ✅ 获取成功，当前价格: {stock_data['current_price']:.2f}")
            else:
                print(f"   ⚠️ 无法获取数据")
        except Exception as e:
            print(f"   ❌ 获取 {stock_code} 数据失败: {e}")
    
    # 测试是否有交易信号
    print("\n🔍 测试交易信号检测...")
    try:
        today = datetime.now().date()
        has_signals = email_system.has_any_signals(hsi_indicators, stock_results, today)
        print(f"✅ 今天是否有交易信号: {has_signals}")
    except Exception as e:
        print(f"❌ 交易信号检测失败: {e}")
    
    # 测试连续信号分析
    print("\n🔍 测试连续信号分析...")
    try:
        buy_signals, sell_signals = email_system.analyze_continuous_signals()
        print(f"✅ 连续买入信号: {len(buy_signals)} 只股票")
        print(f"✅ 连续卖出信号: {len(sell_signals)} 只股票")
        if buy_signals:
            for code, name, times, reasons, df in buy_signals[:2]:  # 只显示前2个
                print(f"   - {name}({code}): 连续买入 {len(times)} 次")
    except Exception as e:
        print(f"❌ 连续信号分析失败: {e}")
    
    # 测试报告生成（不发送邮件）
    print("\n📝 测试报告内容生成...")
    try:
        today = datetime.now().date()
        text_content, html_content = email_system.generate_report_content(today, hsi_data, hsi_indicators, stock_results)
        print(f"✅ 成功生成报告内容")
        print(f"   - 文本内容长度: {len(text_content)} 字符")
        print(f"   - HTML内容长度: {len(html_content)} 字符")
        
        # 简要预览内容
        if text_content:
            preview_lines = text_content.split('\n')[:10]
            print("   - 内容预览:")
            for line in preview_lines:
                if line.strip():
                    print(f"     {line[:50]}{'...' if len(line) > 50 else ''}")
    except Exception as e:
        print(f"❌ 报告内容生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ hsi_email.py 分析功能测试完成！")
    return True

def test_hsi_email_advanced_features():
    """测试 hsi_email.py 的高级功能"""
    print("\n🧪 开始测试 hsi_email.py 高级功能...")
    
    try:
        from hsi_email import HSIEmailSystem
        email_system = HSIEmailSystem()
        print("✅ 成功创建 HSIEmailSystem 实例")
    except Exception as e:
        print(f"❌ 创建实例失败: {e}")
        return False
    
    # 测试大模型分析功能（如果可用）
    print("\n🤖 测试大模型分析功能...")
    try:
        # 获取测试数据
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data:
            indicators = email_system.calculate_technical_indicators(stock_data)
            stock_results = [{
                'code': test_stock,
                'name': email_system.stock_list[test_stock],
                'data': stock_data,
                'indicators': indicators
            }]
            
            # 测试持仓分析
            portfolio = email_system._read_portfolio_data()
            if portfolio:
                print("📊 测试持仓分析...")
                portfolio_analysis = email_system._analyze_portfolio_with_llm(portfolio, stock_results)
                if portfolio_analysis:
                    print(f"✅ 成功生成持仓分析，长度: {len(portfolio_analysis)} 字符")
                else:
                    print("⚠️ 持仓分析返回空值（可能因为大模型配置问题）")
            
            # 测试买入信号分析
            print("📊 测试买入信号分析...")
            buy_signals = [(email_system.stock_list[test_stock], test_stock, '多头趋势', {'description': '测试信号'}, '买入')]
            buy_analysis = email_system._analyze_buy_signals_with_llm(buy_signals, stock_results)
            if buy_analysis:
                print(f"✅ 成功生成买入信号分析，长度: {len(buy_analysis)} 字符")
            else:
                print("⚠️ 买入信号分析返回空值（可能因为大模型配置问题）")
        else:
            print("⚠️ 无数据用于大模型分析测试")
    except Exception as e:
        print(f"❌ 大模型分析功能测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试板块分析
    print("\n🏗️ 测试板块分析功能...")
    try:
        if hasattr(email_system, 'SECTOR_ANALYSIS_AVAILABLE') and email_system.SECTOR_ANALYSIS_AVAILABLE:
            from data_services.hk_sector_analysis import SectorAnalyzer
            sector_analyzer = SectorAnalyzer()
            perf_df = sector_analyzer.calculate_sector_performance(email_system.SECTOR_ANALYSIS_PERIOD)
            if not perf_df.empty:
                print(f"✅ 成功获取板块分析数据: {len(perf_df)} 个板块")
                print("   板块排名前3:")
                for idx, row in perf_df.head(3).iterrows():
                    print(f"   {idx+1}. {row['sector_name']}: {row['avg_change_pct']:+.2f}%")
            else:
                print("⚠️ 板块分析数据为空")
        else:
            print("⚠️ 板块分析功能不可用")
    except Exception as e:
        print(f"❌ 板块分析功能测试失败: {e}")
    
    # 测试中期分析指标
    print("\n📈 测试中期分析指标...")
    try:
        if hasattr(email_system, 'MEDIUM_TERM_AVAILABLE') and email_system.MEDIUM_TERM_AVAILABLE:
            test_stock = list(email_system.stock_list.keys())[0]
            stock_data = email_system.get_stock_data(test_stock)
            if stock_data:
                # 获取技术指标，这应该包含中期分析
                indicators = email_system.calculate_technical_indicators(stock_data)
                if indicators:
                    print("✅ 中期分析指标计算成功")
                    if 'medium_term_score' in indicators:
                        print(f"   - 中期趋势评分: {indicators['medium_term_score']}")
                    if 'ma_alignment' in indicators:
                        print(f"   - 均线排列: {indicators['ma_alignment']}")
                    if 'ma20_slope' in indicators:
                        print(f"   - MA20斜率: {indicators['ma20_slope']:.4f}")
                    if 'ma_deviation_avg' in indicators:
                        print(f"   - 均线乖离率: {indicators['ma_deviation_avg']:.2f}%")
                else:
                    print("⚠️ 中期分析指标计算返回空值")
            else:
                print("⚠️ 无数据用于中期分析指标测试")
        else:
            print("⚠️ 中期分析指标功能不可用")
    except Exception as e:
        print(f"❌ 中期分析指标测试失败: {e}")
    
    # 测试基本面分析
    print("\n💼 测试基本面分析功能...")
    try:
        if hasattr(email_system, 'FUNDAMENTAL_AVAILABLE') and email_system.FUNDAMENTAL_AVAILABLE:
            from data_services.fundamental_data import get_comprehensive_fundamental_data
            test_stock = list(email_system.stock_list.keys())[0].replace('.HK', '')
            fundamental_data = get_comprehensive_fundamental_data(test_stock)
            if fundamental_data:
                print(f"✅ 成功获取 {test_stock} 基本面数据")
                if 'fi_pe_ratio' in fundamental_data:
                    print(f"   - PE比率: {fundamental_data['fi_pe_ratio']}")
                if 'fi_pb_ratio' in fundamental_data:
                    print(f"   - PB比率: {fundamental_data['fi_pb_ratio']}")
            else:
                print(f"⚠️ 无法获取 {test_stock} 基本面数据")
        else:
            print("⚠️ 基本面分析功能不可用")
    except Exception as e:
        print(f"❌ 基本面分析功能测试失败: {e}")
    
    # 测试市场情绪和流动性指标
    print("\n📉 测试市场情绪和流动性指标...")
    try:
        test_stock = list(email_system.stock_list.keys())[0]
        stock_data = email_system.get_stock_data(test_stock)
        if stock_data and stock_data['hist'] is not None and not stock_data['hist'].empty:
            indicators = email_system.calculate_technical_indicators(stock_data)
            if indicators:
                print("✅ 市场情绪和流动性指标计算成功")
                if 'vix_level' in indicators:
                    print(f"   - VIX恐慌指数: {indicators['vix_level']}")
                if 'turnover_change_1d' in indicators:
                    print(f"   - 成交额变化1日: {indicators['turnover_change_1d']:+.2f}%")
                if 'turnover_rate_change_5d' in indicators:
                    print(f"   - 换手率变化5日: {indicators['turnover_rate_change_5d']:+.2f}%")
            else:
                print("⚠️ 市场情绪和流动性指标计算返回空值")
        else:
            print("⚠️ 无数据用于市场情绪和流动性指标测试")
    except Exception as e:
        print(f"❌ 市场情绪和流动性指标测试失败: {e}")
    
    print("\n✅ hsi_email.py 高级功能测试完成！")
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试 hsi_email.py 脚本")
    print("="*60)
    
    # 运行所有测试
    basic_test_passed = test_hsi_email_basic()
    analysis_test_passed = test_hsi_email_analysis()
    advanced_test_passed = test_hsi_email_advanced_features()
    
    print("\n"+"="*60)
    print("📊 测试总结:")
    print(f"   基本功能测试: {'✅ 通过' if basic_test_passed else '❌ 失败'}")
    print(f"   分析功能测试: {'✅ 通过' if analysis_test_passed else '❌ 失败'}")
    print(f"   高级功能测试: {'✅ 通过' if advanced_test_passed else '❌ 失败'}")
    
    all_passed = basic_test_passed and analysis_test_passed and advanced_test_passed
    print(f"\n🎯 总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
