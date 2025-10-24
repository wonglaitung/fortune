#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄金市场分析器
集成技术分析、宏观经济数据和大模型深度分析
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入大模型服务
from llm_services import qwen_engine
LLM_AVAILABLE = True

class GoldDataCollector:
    def __init__(self):
        # 黄金相关资产代码
        self.gold_assets = {
            'GC=F': 'COMEX黄金期货',
            'GLD': 'SPDR黄金ETF',
            'IAU': 'iShares黄金ETF',
            'SLV': 'iShares白银ETF'
        }
        
        # 宏观经济指标
        self.macro_indicators = {
            'DX-Y.NYB': '美元指数',
            '^TNX': '10年期美债收益率',
            'CL=F': 'WTI原油',
            '^VIX': '恐慌指数'
        }
        
    def get_gold_data(self, period="1y"):
        """获取黄金价格数据"""
        print("📈 获取黄金相关资产数据...")
        data = {}
        for symbol, name in self.gold_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    # 计算价格变化率
                    hist['Price_change_1d'] = hist['Close'].pct_change(1)
                    hist['Price_change_5d'] = hist['Close'].pct_change(5)
                    hist['Price_change_20d'] = hist['Close'].pct_change(20)
                    data[symbol] = {
                        'name': name,
                        'data': hist,
                        'info': ticker.info if hasattr(ticker, 'info') else {}
                    }
                    print(f"  ✅ {name} ({symbol}) 数据获取成功")
                else:
                    print(f"  ⚠️ {name} ({symbol}) 数据为空")
            except Exception as e:
                print(f"  ❌ 获取{name} ({symbol}) 数据失败: {e}")
        return data
    
    def get_macro_data(self, period="1y"):
        """获取宏观经济数据"""
        print("📊 获取宏观经济数据...")
        data = {}
        for symbol, name in self.macro_indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    # 计算价格变化率
                    hist['Price_change_1d'] = hist['Close'].pct_change(1)
                    hist['Price_change_5d'] = hist['Close'].pct_change(5)
                    hist['Price_change_20d'] = hist['Close'].pct_change(20)
                    data[symbol] = {
                        'name': name,
                        'data': hist
                    }
                    print(f"  ✅ {name} ({symbol}) 数据获取成功")
                else:
                    print(f"  ⚠️ {name} ({symbol}) 数据为空")
            except Exception as e:
                print(f"  ❌ 获取{name} ({symbol}) 数据失败: {e}")
        return data

class GoldTechnicalAnalyzer:
    def __init__(self):
        pass
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        if df.empty:
            return df
            
        # 确保必要的列存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"  ⚠️ 缺少必要的列: {col}")
                return df
        
        # 移动平均线
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
        # RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # 布林带
        df['BB_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # 成交量指标
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # 价格变化率
        df['Price_change_1d'] = df['Close'].pct_change(1)
        df['Price_change_5d'] = df['Close'].pct_change(5)
        df['Price_change_20d'] = df['Close'].pct_change(20)
        
        return df
    
    def identify_support_resistance(self, df, window=20):
        """识别支撑位和阻力位"""
        if df.empty or len(df) < window:
            return {'support': None, 'resistance': None}
            
        recent_data = df.tail(window)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return {
            'support': float(support) if not pd.isna(support) else None,
            'resistance': float(resistance) if not pd.isna(resistance) else None
        }
    
    def identify_trend(self, df):
        """识别趋势"""
        if df.empty or len(df) < 200:
            return "数据不足"
            
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        ma50 = df['MA50'].iloc[-1]
        ma200 = df['MA200'].iloc[-1]
        
        if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
            return "数据不足"
        
        # 多头排列：价格 > MA20 > MA50 > MA200
        if current_price > ma20 > ma50 > ma200:
            return "强势多头"
        # 空头排列：价格 < MA20 < MA50 < MA200
        elif current_price < ma20 < ma50 < ma200:
             return "弱势空头"
        # 震荡
        else:
            return "震荡整理"

class GoldLLMAnalyzer:
    def __init__(self):
        self.llm_available = LLM_AVAILABLE
        
    def build_analysis_prompt(self, gold_data, technical_data, macro_data):
        """构建大模型分析提示"""
        if not self.llm_available:
            return None
            
        # 构建黄金数据摘要
        gold_summary = self._format_gold_summary(gold_data)
        tech_summary = self._format_technical_summary(technical_data)
        macro_summary = self._format_macro_summary(macro_data)
        
        prompt = f"""
你是一位专业的黄金投资分析师，请根据以下数据对黄金市场进行全面分析：

【黄金市场概况】
{gold_summary}

【技术面分析】
{tech_summary}

【宏观经济环境】
{macro_summary}

请从以下几个维度进行专业分析：

1. **当前黄金价格趋势分析**
   - 短期（1-2周）、中期（1-3个月）、长期（6个月以上）趋势
   - 价格波动性和风险评估

2. **技术面信号解读**
   - 关键技术指标（RSI、MACD、均线）状态
   - 支撑位和阻力位分析
   - 买卖信号判断

3. **宏观经济因素影响**
   - 美元指数对黄金的影响
   - 美债收益率与黄金关系
   - 原油价格对通胀预期的影响
   - 市场恐慌情绪（VIX）对避险需求的影响

4. **投资建议**
   - 短期（1-4周）操作建议
   - 中期（1-3个月）策略建议
   - 长期（6个月以上）配置建议
   - 仓位管理和风险控制建议

5. **风险预警**
   - 需要关注的关键风险因素
   - 可能影响黄金价格的重大事件

请用中文回答，格式清晰易读，给出明确但谨慎的投资建议。避免过于绝对的预测，重点分析当前市场状况和可能的发展方向。

请严格按照以下JSON格式输出：
{{
    "trend_analysis": "趋势分析内容",
    "technical_signals": "技术信号解读",
    "macro_impact": "宏观经济影响分析",
    "investment_advice": {{
        "short_term": "短期建议",
        "medium_term": "中期建议",
        "long_term": "长期建议"
    }},
    "risk_warning": "风险预警"
}}
"""
        
        return prompt
    
    def _format_gold_summary(self, data):
        """格式化黄金数据摘要"""
        summary = ""
        for symbol, info in data.items():
            if not info['data'].empty:
                latest = info['data'].iloc[-1]
                name = info['name']
                # 检查是否存在价格变化列
                if 'Price_change_1d' in latest:
                    summary += f"- {name} ({symbol}): ${latest['Close']:.2f} (24h: {latest['Price_change_1d']*100:.2f}%)\n"
                else:
                    summary += f"- {name} ({symbol}): ${latest['Close']:.2f}\n"
        return summary or "暂无数据"
        
    def _format_technical_summary(self, data):
        """格式化技术数据摘要"""
        summary = ""
        for symbol, info in data.items():
            if 'indicators' in info and not info['indicators'].empty:
                latest = info['indicators'].iloc[-1]
                name = info.get('name', symbol)
                summary += f"- {name}: RSI={latest['RSI']:.1f}, MACD={latest['MACD']:.2f}, 20日均线=${latest['MA20']:.2f}\n"
        return summary or "暂无数据"
        
    def _format_macro_summary(self, data):
        """格式化宏观数据摘要"""
        summary = ""
        for symbol, info in data.items():
            if not info['data'].empty:
                latest = info['data'].iloc[-1]
                name = info['name']
                if 'Close' in latest:
                    summary += f"- {name} ({symbol}): {latest['Close']:.2f}\n"
        return summary or "暂无数据"
    
    def analyze_gold_market(self, prompt):
        """调用大模型进行黄金市场分析"""
        if not self.llm_available or not prompt:
            return None
            
        try:
            print("🤖 正在调用大模型进行深度分析...")
            analysis = qwen_engine.chat_with_llm(prompt)
            print("✅ 大模型分析完成")
            return analysis
        except Exception as e:
            print(f"❌ 大模型分析失败: {e}")
            return None

class GoldMarketAnalyzer:
    def __init__(self):
        self.collector = GoldDataCollector()
        self.tech_analyzer = GoldTechnicalAnalyzer()
        self.llm_analyzer = GoldLLMAnalyzer()
        
    def run_comprehensive_analysis(self, period="3mo"):
        """运行综合分析"""
        print("="*60)
        print("🥇 黄金市场综合分析系统")
        print("="*60)
        
        # 1. 获取数据
        gold_data = self.collector.get_gold_data(period=period)
        macro_data = self.collector.get_macro_data(period=period)
        
        if not gold_data:
            print("❌ 未能获取到黄金数据，分析终止")
            return None
        
        # 2. 技术分析
        print("\n🔬 进行技术分析...")
        technical_analysis = {}
        main_gold_symbol = 'GC=F'  # 主要分析COMEX黄金期货
        
        for symbol, data in gold_data.items():
            print(f"  分析 {data['name']} ({symbol})...")
            df = self.tech_analyzer.calculate_indicators(data['data'].copy())
            support_resistance = self.tech_analyzer.identify_support_resistance(df)
            trend = self.tech_analyzer.identify_trend(df)
            
            technical_analysis[symbol] = {
                'name': data['name'],
                'indicators': df,
                'support_resistance': support_resistance,
                'trend': trend
            }
        
        # 3. 大模型分析
        llm_analysis = None
        if self.llm_analyzer.llm_available:
            prompt = self.llm_analyzer.build_analysis_prompt(
                gold_data, technical_analysis, macro_data
            )
            if prompt:
                llm_analysis = self.llm_analyzer.analyze_gold_market(prompt)
        
        # 4. 生成报告
        self._generate_report(gold_data, technical_analysis, macro_data, llm_analysis)
        
        return {
            'gold_data': gold_data,
            'technical_analysis': technical_analysis,
            'macro_data': macro_data,
            'llm_analysis': llm_analysis
        }
    
    def _generate_report(self, gold_data, technical_analysis, macro_data, llm_analysis):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📈 黄金市场综合分析报告")
        print("="*60)
        
        # 1. 黄金价格概览
        print("\n💰 黄金价格概览:")
        print("-" * 30)
        for symbol, data in gold_data.items():
            if not data['data'].empty:
                df = data['data']
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                price = latest['Close']
                change_1d = (price - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0
                change_5d = latest['Price_change_5d'] * 100 if 'Price_change_5d' in latest else 0
                change_20d = latest['Price_change_20d'] * 100 if 'Price_change_20d' in latest else 0
                
                print(f"{data['name']} ({symbol}):")
                print(f"  最新价格: ${price:.2f}")
                print(f"  24小时变化: {change_1d:+.2f}%")
                print(f"  5日变化: {change_5d:+.2f}%")
                print(f"  20日变化: {change_20d:+.2f}%")
                print()
        
        # 2. 技术分析
        print("\n🔬 技术分析:")
        print("-" * 30)
        for symbol, data in technical_analysis.items():
            if not data['indicators'].empty:
                latest = data['indicators'].iloc[-1]
                print(f"{data['name']} ({symbol}):")
                print(f"  趋势: {data['trend']}")
                print(f"  RSI (14日): {latest['RSI']:.1f}")
                print(f"  MACD: {latest['MACD']:.2f} (信号线: {latest['MACD_signal']:.2f})")
                if data['support_resistance']['support']:
                    print(f"  支撑位: ${data['support_resistance']['support']:.2f}")
                if data['support_resistance']['resistance']:
                    print(f"  阻力位: ${data['support_resistance']['resistance']:.2f}")
                print(f"  20日均线: ${latest['MA20']:.2f}")
                print(f"  50日均线: ${latest['MA50']:.2f}")
                print()
        
        # 3. 宏观经济环境
        print("\n📊 宏观经济环境:")
        print("-" * 30)
        for symbol, data in macro_data.items():
            if not data['data'].empty:
                latest = data['data'].iloc[-1]
                if 'Close' in latest:
                    print(f"{data['name']} ({symbol}): {latest['Close']:.2f}")
        print()
        
        # 4. 大模型分析
        if llm_analysis:
            print("\n🤖 大模型深度分析:")
            print("-" * 30)
            try:
                # 尝试解析JSON格式的输出
                import json
                analysis_json = json.loads(llm_analysis)
                print(f"趋势分析: {analysis_json.get('trend_analysis', 'N/A')}")
                print(f"技术信号: {analysis_json.get('technical_signals', 'N/A')}")
                print(f"宏观影响: {analysis_json.get('macro_impact', 'N/A')}")
                print("投资建议:")
                advice = analysis_json.get('investment_advice', {})
                print(f"  短期: {advice.get('short_term', 'N/A')}")
                print(f"  中期: {advice.get('medium_term', 'N/A')}")
                print(f"  长期: {advice.get('long_term', 'N/A')}")
                print(f"风险预警: {analysis_json.get('risk_warning', 'N/A')}")
            except:
                # 如果不是JSON格式，直接输出
                print(llm_analysis)
        else:
            print("\n⚠️ 大模型分析暂不可用")
            print("请检查大模型服务配置或API密钥")
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='黄金市场分析系统')
    parser.add_argument('--period', type=str, default='3mo', 
                       help='分析周期 (1mo, 3mo, 6mo, 1y, 2y)')
    args = parser.parse_args()
    
    analyzer = GoldMarketAnalyzer()
    result = analyzer.run_comprehensive_analysis(period=args.period)
    
    if result:
        print(f"\n✅ 分析完成，数据已获取")
    else:
        print(f"\n❌ 分析失败")

if __name__ == "__main__":
    main()