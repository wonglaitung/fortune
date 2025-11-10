#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数(HSI)大模型策略分析器
此脚本用于获取当前恒生指数数据并调用大模型生成明确的交易策略建议
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入腾讯财经接口
from tencent_finance import get_hsi_data_tencent

# 导入技术分析工具
from technical_analysis import TechnicalAnalyzer

# 导入大模型服务
try:
    from llm_services.qwen_engine import chat_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("警告: 无法导入大模型服务，将跳过大模型分析功能")

warnings.filterwarnings('ignore')

def generate_hsi_llm_strategy():
    """
    生成恒生指数大模型策略分析
    """
    print("🚀 开始获取恒生指数数据...")
    
    # 获取最新数据
    period_days = 90
    data = get_hsi_data_tencent(period_days=period_days)
    
    if data is None or data.empty:
        print("❌ 无法获取恒生指数数据")
        return None
    
    print(f"✅ 成功获取 {len(data)} 天的恒生指数数据")
    
    # 创建技术分析器并计算指标
    technical_analyzer = TechnicalAnalyzer()
    indicators = technical_analyzer.calculate_all_indicators(data.copy())
    
    # 计算额外的恒生指数专用指标
    # 计算价格位置（在最近N日内的百分位位置）
    price_window = 60
    if len(indicators) >= price_window:
        rolling_low = indicators['Close'].rolling(window=price_window).min()
        rolling_high = indicators['Close'].rolling(window=price_window).max()
        indicators['Price_Percentile'] = ((indicators['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
    else:
        # 如果数据不足，使用全部可用数据
        rolling_low = indicators['Close'].rolling(window=len(indicators)).min()
        rolling_high = indicators['Close'].rolling(window=len(indicators)).max()
        indicators['Price_Percentile'] = ((indicators['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
    
    # 计算成交量比率（相对于20日均量）
    indicators['Vol_MA20'] = indicators['Volume'].rolling(window=20).mean()
    indicators['Vol_Ratio'] = indicators['Volume'] / indicators['Vol_MA20']
    
    # 计算波动率（20日年化波动率）
    indicators['Returns'] = indicators['Close'].pct_change()
    indicators['Volatility'] = indicators['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    # 获取最新数据
    latest = indicators.iloc[-1]
    
    print(f"📊 当前恒生指数: {latest['Close']:.2f}")
    print(f"📈 RSI: {latest['RSI']:.2f}")
    print(f"📊 MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
    print(f"均线: MA20: {latest['MA20']:.2f}, MA50: {latest['MA50']:.2f}")
    print(f"价格位置: {latest['Price_Percentile']:.2f}%")
    print(f"波动率: {latest['Volatility']:.2f}%")
    print(f"量比: {latest['Vol_Ratio']:.2f}")
    
    # 构建分析报告内容作为大模型输入
    analysis_summary = []
    analysis_summary.append("恒生指数(HSI)技术分析数据:")
    analysis_summary.append(f"当前指数: {latest['Close']:.2f}")
    analysis_summary.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_summary.append("")
    
    # 添加关键技术指标
    analysis_summary.append("关键技术指标:")
    if 'RSI' in indicators.columns:
        analysis_summary.append(f"RSI: {latest['RSI']:.2f}")
    if 'MACD' in indicators.columns and 'MACD_signal' in indicators.columns:
        analysis_summary.append(f"MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
    if 'MA20' in indicators.columns:
        analysis_summary.append(f"MA20: {latest['MA20']:.2f}")
    if 'MA50' in indicators.columns:
        analysis_summary.append(f"MA50: {latest['MA50']:.2f}")
    if 'MA200' in indicators.columns:
        analysis_summary.append(f"MA200: {latest['MA200']:.2f}")
    if 'Price_Percentile' in indicators.columns:
        analysis_summary.append(f"价格位置: {latest['Price_Percentile']:.2f}%")
    if 'Volatility' in indicators.columns:
        analysis_summary.append(f"波动率: {latest['Volatility']:.2f}%")
    if 'Vol_Ratio' in indicators.columns:
        analysis_summary.append(f"量比: {latest['Vol_Ratio']:.2f}")
    analysis_summary.append("")
    
    # 添加趋势分析
    current_price = latest['Close']
    ma20 = latest['MA20'] if 'MA20' in indicators.columns and not pd.isna(latest['MA20']) else np.nan
    ma50 = latest['MA50'] if 'MA50' in indicators.columns and not pd.isna(latest['MA50']) else np.nan
    ma200 = latest['MA200'] if 'MA200' in indicators.columns and not pd.isna(latest['MA200']) else np.nan
    
    trend = "未知"
    if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
        if current_price > ma20 > ma50 > ma200:
            trend = "强势多头"
        elif current_price < ma20 < ma50 < ma200:
            trend = "弱势空头"
        else:
            trend = "震荡整理"
    elif not pd.isna(ma20) and not pd.isna(ma50):
        if current_price > ma20 > ma50:
            trend = "多头趋势"
        elif current_price < ma20 < ma50:
            trend = "空头趋势"
        else:
            trend = "震荡"
    
    analysis_summary.append(f"当前趋势: {trend}")
    analysis_summary.append("")
    
    # 获取历史数据用于趋势分析
    historical_data = indicators.tail(20)  # 最近20天的数据
    analysis_summary.append("最近20天指数变化:")
    for idx, row in historical_data.iterrows():
        analysis_summary.append(f"  {idx.strftime('%Y-%m-%d')}: {row['Close']:.2f}")
    analysis_summary.append("")
    
    # 构建大模型提示
    prompt = f"""
请分析以下恒生指数(HSI)技术分析数据，并提供明确的交易策略建议：

{chr(10).join(analysis_summary)}

请根据以下原则提供具体的交易策略：
1. 基于趋势分析：如果指数处于上升趋势，考虑多头策略；如果处于下降趋势，考虑空头或谨慎策略
2. 基于技术指标：利用RSI、MACD、移动平均线等指标判断买卖时机
3. 基于市场状态：考虑当前市场是处于高位、中位还是低位
4. 风险管理：在建议中包含止损和风险控制策略
5. 资金管理：考虑适当的仓位管理原则

策略定义参考：
- 保守型：偏好低风险、稳定收益的投资策略，如高股息股票，注重资本保值
- 平衡型：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进取型：偏好高风险、高收益的投资策略，如科技成长股，追求资本增值

请提供具体的交易策略，包括：
- 当前市场观点
- 交易方向建议（做多/做空/观望）
- 明确推荐一个最适合当前市场状况的投资者类型（保守型/平衡型/进取型）
- 具体操作建议
- 风险控制措施
- 目标价位和止损位

请确保策略符合港股市场特点和恒生指数的特性。
"""
    
    if LLM_AVAILABLE:
        try:
            print("\n🤖 正在调用大模型分析恒生指数策略...")
            response = chat_with_llm(prompt)
            print("\n" + "="*60)
            print("🤖 大模型恒生指数交易策略分析")
            print("="*60)
            print(response)
            print("="*60)
            return response
        except Exception as e:
            print(f"❌ 调用大模型失败: {str(e)}")
            print("💡 请确保已设置 QWEN_API_KEY 环境变量")
            return None
    else:
        print("❌ 大模型服务不可用")
        return None

def main():
    """主函数"""
    print("📈 恒生指数(HSI)大模型策略分析器")
    print("="*50)
    
    # 生成策略分析
    strategy = generate_hsi_llm_strategy()
    
    if strategy:
        print("\n✅ 恒生指数大模型策略分析完成！")
    else:
        print("\n❌ 恒生指数大模型策略分析失败")

if __name__ == "__main__":
    main()
