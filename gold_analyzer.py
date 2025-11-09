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
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入技术分析工具
try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，将使用简化指标计算")

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
        
        # 如果技术分析工具可用，则使用它
        if TECHNICAL_ANALYSIS_AVAILABLE:
            analyzer = TechnicalAnalyzer()
            df = analyzer.calculate_all_indicators(df)
            df = analyzer.generate_buy_sell_signals(df)
            return df
        else:
            # 使用原始的计算方法
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
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']) if (df['BB_upper'] - df['BB_lower']).any() != 0 else 0.5
            
            # 成交量指标
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
            
            # 价格变化率
            df['Price_change_1d'] = df['Close'].pct_change(1)
            df['Price_change_5d'] = df['Close'].pct_change(5)
            df['Price_change_20d'] = df['Close'].pct_change(20)
            
            # 生成买卖信号
            df = self._generate_buy_sell_signals(df)
        
        return df
    
    def _generate_buy_sell_signals(self, df):
        """基于技术指标生成买卖信号"""
        if df.empty:
            return df
        
        # 初始化信号列
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Signal_Description'] = ''
        
        # 计算一些必要的中间指标
        if 'MA20' in df.columns and 'MA50' in df.columns:
            # 金叉死叉信号
            df['MA20_above_MA50'] = df['MA20'] > df['MA50']
            df['MA20_below_MA50'] = df['MA20'] < df['MA50']
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            # MACD交叉信号
            df['MACD_above_signal'] = df['MACD'] > df['MACD_signal']
            df['MACD_below_signal'] = df['MACD'] < df['MACD_signal']
        
        if 'RSI' in df.columns:
            # RSI超买超卖信号
            df['RSI_oversold'] = df['RSI'] < 30
            df['RSI_overbought'] = df['RSI'] > 70
        
        if 'Close' in df.columns and 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            # 布林带信号
            df['Price_above_BB_upper'] = df['Close'] > df['BB_upper']
            df['Price_below_BB_lower'] = df['Close'] < df['BB_lower']
        
        # 生成买入信号逻辑
        for i in range(1, len(df)):
            buy_conditions = []
            sell_conditions = []
            
            # 条件1: 价格在上升趋势中 (MA20 > MA50)
            if ('MA20_above_MA50' in df.columns and df.iloc[i]['MA20_above_MA50'] and 
                not df.iloc[i-1]['MA20_above_MA50']):
                buy_conditions.append("上升趋势形成")
            
            # 条件2: MACD金叉
            if ('MACD_above_signal' in df.columns and df.iloc[i]['MACD_above_signal'] and 
                not df.iloc[i-1]['MACD_above_signal']):
                buy_conditions.append("MACD金叉")
            
            # 条件3: RSI从超卖区域回升
            if ('RSI_oversold' in df.columns and not df.iloc[i]['RSI_oversold'] and 
                df.iloc[i-1]['RSI_oversold']):
                buy_conditions.append("RSI超卖反弹")
            
            # 条件4: 价格从布林带下轨反弹
            if ('Price_below_BB_lower' in df.columns and not df.iloc[i]['Price_below_BB_lower'] and 
                df.iloc[i-1]['Price_below_BB_lower']):
                buy_conditions.append("布林带下轨反弹")
            
            # 生成买入信号
            if buy_conditions:
                df.at[df.index[i], 'Buy_Signal'] = True
                df.at[df.index[i], 'Signal_Description'] = "买入信号: " + ", ".join(buy_conditions)
            
            # 生成卖出信号逻辑
            # 条件1: 价格在下降趋势中 (MA20 < MA50)
            if ('MA20_below_MA50' in df.columns and df.iloc[i]['MA20_below_MA50'] and 
                not df.iloc[i-1]['MA20_below_MA50']):
                sell_conditions.append("下降趋势形成")
            
            # 条件2: MACD死叉
            if ('MACD_below_signal' in df.columns and df.iloc[i]['MACD_below_signal'] and 
                not df.iloc[i-1]['MACD_below_signal']):
                sell_conditions.append("MACD死叉")
            
            # 条件3: RSI从超买区域回落
            if ('RSI_overbought' in df.columns and not df.iloc[i]['RSI_overbought'] and 
                df.iloc[i-1]['RSI_overbought']):
                sell_conditions.append("RSI超买回落")
            
            # 条件4: 价格跌破布林带上轨
            if ('Price_above_BB_upper' in df.columns and not df.iloc[i]['Price_above_BB_upper'] and 
                df.iloc[i-1]['Price_above_BB_upper']):
                sell_conditions.append("跌破布林带上轨")
            
            # 生成卖出信号
            if sell_conditions:
                df.at[df.index[i], 'Sell_Signal'] = True
                df.at[df.index[i], 'Signal_Description'] = "卖出信号: " + ", ".join(sell_conditions)
        
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
        if df.empty or len(df) < 50:  # 降低最小数据要求
            return "数据不足"
        
        # 获取最新数据
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else np.nan
        ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns and not pd.isna(df['MA50'].iloc[-1]) else np.nan
        ma200 = df['MA200'].iloc[-1] if 'MA200' in df.columns and not pd.isna(df['MA200'].iloc[-1]) else np.nan
        
        # 如果有200日均线数据，使用完整趋势分析
        if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
            # 多头排列：价格 > MA20 > MA50 > MA200
            if current_price > ma20 > ma50 > ma200:
                return "强势多头"
            # 空头排列：价格 < MA20 < MA50 < MA200
            elif current_price < ma20 < ma50 < ma200:
                return "弱势空头"
            # 震荡
            else:
                return "震荡整理"
        # 如果没有200日均线数据，使用较短期的趋势分析
        elif not pd.isna(ma20) and not pd.isna(ma50):
            # 多头排列：价格 > MA20 > MA50
            if current_price > ma20 > ma50:
                return "多头趋势"
            # 空头排列：价格 < MA20 < MA50
            elif current_price < ma20 < ma50:
                return "空头趋势"
            # 震荡
            else:
                return "震荡"
        # 如果连短期均线都没有，只看价格趋势
        elif len(df) >= 20:
            # 比较最近价格与20日均价
            recent_price = df['Close'].iloc[-1]
            past_price = df['Close'].iloc[-20]  # 20天前的价格
            
            if recent_price > past_price:
                return "短期上涨"
            else:
                return "短期下跌"
        else:
            return "数据不足"

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
        
        # 5. 发送邮件报告
        self.send_email_report(gold_data, technical_analysis, macro_data, llm_analysis)
        
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
                print(f"  布林带位置: {latest.get('BB_position', 0.5):.2f}")
                if data['support_resistance']['support']:
                    print(f"  支撑位: ${data['support_resistance']['support']:.2f}")
                if data['support_resistance']['resistance']:
                    print(f"  阻力位: ${data['support_resistance']['resistance']:.2f}")
                print(f"  20日均线: ${latest['MA20']:.2f}")
                print(f"  50日均线: ${latest['MA50']:.2f}")
                
                # 检查最近的交易信号
                recent_signals = data['indicators'].tail(5)
                buy_signals = []
                sell_signals = []
                
                if 'Buy_Signal' in recent_signals.columns:
                    buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                    for idx, row in buy_signals_df.iterrows():
                        buy_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': row['Signal_Description']
                        })
                
                if 'Sell_Signal' in recent_signals.columns:
                    sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                    for idx, row in sell_signals_df.iterrows():
                        sell_signals.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'description': row['Signal_Description']
                        })
                
                if buy_signals:
                    print(f"  🔔 最近买入信号 ({len(buy_signals)} 个):")
                    for signal in buy_signals:
                        print(f"    {signal['date']}: {signal['description']}")
                
                if sell_signals:
                    print(f"  🔻 最近卖出信号 ({len(sell_signals)} 个):")
                    for signal in sell_signals:
                        print(f"    {signal['date']}: {signal['description']}")
                
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
                
                # 趋势分析
                trend_analysis = analysis_json.get('trend_analysis', 'N/A')
                if trend_analysis != 'N/A':
                    print("📈 趋势分析:")
                    # 格式化输出，每行不超过80个字符
                    lines = []
                    for i in range(0, len(trend_analysis), 80):
                        lines.append(trend_analysis[i:i+80])
                    for line in lines:
                        print(f"   {line}")
                    print()
                
                # 技术信号
                technical_signals = analysis_json.get('technical_signals', 'N/A')
                if technical_signals != 'N/A':
                    print("📊 技术信号:")
                    # 格式化输出，每行不超过80个字符
                    lines = []
                    for i in range(0, len(technical_signals), 80):
                        lines.append(technical_signals[i:i+80])
                    for line in lines:
                        print(f"   {line}")
                    print()
                
                # 宏观影响
                macro_impact = analysis_json.get('macro_impact', 'N/A')
                if macro_impact != 'N/A':
                    print("💼 宏观影响:")
                    # 格式化输出，每行不超过80个字符
                    lines = []
                    for i in range(0, len(macro_impact), 80):
                        lines.append(macro_impact[i:i+80])
                    for line in lines:
                        print(f"   {line}")
                    print()
                
                # 投资建议
                advice = analysis_json.get('investment_advice', {})
                short_term = advice.get('short_term', 'N/A')
                medium_term = advice.get('medium_term', 'N/A')
                long_term = advice.get('long_term', 'N/A')
                
                if short_term != 'N/A' or medium_term != 'N/A' or long_term != 'N/A':
                    print("💡 投资建议:")
                    if short_term != 'N/A':
                        print("   📌 短期:")
                        # 格式化输出，每行不超过75个字符
                        lines = []
                        for i in range(0, len(short_term), 75):
                            lines.append(short_term[i:i+75])
                        for line in lines:
                            print(f"      {line}")
                    if medium_term != 'N/A':
                        print("   📌 中期:")
                        # 格式化输出，每行不超过75个字符
                        lines = []
                        for i in range(0, len(medium_term), 75):
                            lines.append(medium_term[i:i+75])
                        for line in lines:
                            print(f"      {line}")
                    if long_term != 'N/A':
                        print("   📌 长期:")
                        # 格式化输出，每行不超过75个字符
                        lines = []
                        for i in range(0, len(long_term), 75):
                            lines.append(long_term[i:i+75])
                        for line in lines:
                            print(f"      {line}")
                    print()
                
                # 风险预警
                risk_warning = analysis_json.get('risk_warning', 'N/A')
                if risk_warning != 'N/A':
                    print("⚠️ 风险预警:")
                    # 格式化输出，每行不超过80个字符
                    lines = []
                    for i in range(0, len(risk_warning), 80):
                        lines.append(risk_warning[i:i+80])
                    for line in lines:
                        print(f"   {line}")
                    print()
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接输出
                print("分析结果:")
                # 格式化输出，每行不超过80个字符
                lines = []
                for i in range(0, len(llm_analysis), 80):
                    lines.append(llm_analysis[i:i+80])
                for line in lines:
                    print(f"   {line}")
                print()
            except Exception as e:
                print(f"   分析内容解析失败: {e}")
                print(f"   原始内容: {llm_analysis}")
                print()
        else:
            print("\n⚠️ 大模型分析暂不可用")
            print("   请检查大模型服务配置或API密钥")
            print()
    
    def send_email_report(self, gold_data, technical_analysis, macro_data, llm_analysis):
        """发送邮件报告"""
        try:
            # 获取SMTP配置
            smtp_server = os.environ.get("YAHOO_SMTP", "smtp.mail.yahoo.com")
            smtp_user = os.environ.get("YAHOO_EMAIL")
            smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
            sender_email = smtp_user
            
            if not smtp_user or not smtp_pass:
                print("⚠️  邮件配置缺失，跳过发送邮件")
                return False
            
            # 获取收件人
            recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
            recipients = [r.strip() for r in recipient_env.split(",")] if "," in recipient_env else [recipient_env]
            
            print(f"📧 正在发送邮件到: {', '.join(recipients)}")
            
            # 创建邮件内容
            subject = "黄金市场分析报告"
            
            # 纯文本版本
            text_body = "黄金市场分析报告\n\n"
            
            # HTML版本
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_body = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    h2 {{ color: #333; }}
                    h3 {{ color: #555; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .section {{ margin: 20px 0; }}
                    .highlight {{ background-color: #ffffcc; }}
                    .buy-signal {{ background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                    .sell-signal {{ background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h2>🥇 黄金市场综合分析报告</h2>
                <p><strong>报告生成时间:</strong> {report_time}</p>
            """
            
            # 1. 黄金价格概览
            html_body += """
                <div class="section">
                    <h3>💰 黄金价格概览</h3>
                    <table>
                        <tr>
                            <th>资产名称</th>
                            <th>代码</th>
                            <th>最新价格</th>
                            <th>24小时变化</th>
                            <th>5日变化</th>
                            <th>20日变化</th>
                        </tr>
            """
            
            for symbol, data in gold_data.items():
                if not data['data'].empty:
                    df = data['data']
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    price = latest['Close']
                    change_1d = (price - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0
                    change_5d = latest['Price_change_5d'] * 100 if 'Price_change_5d' in latest else 0
                    change_20d = latest['Price_change_20d'] * 100 if 'Price_change_20d' in latest else 0
                    
                    html_body += f"""
                        <tr>
                            <td>{data['name']}</td>
                            <td>{symbol}</td>
                            <td>${price:.2f}</td>
                            <td>{change_1d:+.2f}%</td>
                            <td>{change_5d:+.2f}%</td>
                            <td>{change_20d:+.2f}%</td>
                        </tr>
                    """
            
            html_body += """
                    </table>
                </div>
            """
            
            # 2. 技术分析
            html_body += """
                <div class="section">
                    <h3>🔬 技术分析</h3>
                    <table>
                        <tr>
                            <th>资产名称</th>
                            <th>代码</th>
                            <th>趋势</th>
                            <th>RSI (14日)</th>
                            <th>MACD</th>
                            <th>MACD信号线</th>
                            <th>布林带位置</th>
                            <th>支撑位</th>
                            <th>阻力位</th>
                            <th>20日均线</th>
                            <th>50日均线</th>
                        </tr>
            """
            
            for symbol, data in technical_analysis.items():
                if not data['indicators'].empty:
                    latest = data['indicators'].iloc[-1]
                    support = data['support_resistance']['support'] if data['support_resistance']['support'] else 'N/A'
                    resistance = data['support_resistance']['resistance'] if data['support_resistance']['resistance'] else 'N/A'
                    bb_position = latest.get('BB_position', 0.5) if 'BB_position' in latest else 0.5
                    
                    # 检查最近的交易信号
                    recent_signals = data['indicators'].tail(5)
                    buy_signals = []
                    sell_signals = []
                    
                    if 'Buy_Signal' in recent_signals.columns:
                        buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                        for idx, row in buy_signals_df.iterrows():
                            buy_signals.append({
                                'date': idx.strftime('%Y-%m-%d'),
                                'description': row['Signal_Description']
                            })
                    
                    if 'Sell_Signal' in recent_signals.columns:
                        sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                        for idx, row in sell_signals_df.iterrows():
                            sell_signals.append({
                                'date': idx.strftime('%Y-%m-%d'),
                                'description': row['Signal_Description']
                            })
                    
                    html_body += f"""
                        <tr>
                            <td>{data['name']}</td>
                            <td>{symbol}</td>
                            <td>{data['trend']}</td>
                            <td>{latest['RSI']:.1f}</td>
                            <td>{latest['MACD']:.2f}</td>
                            <td>{latest['MACD_signal']:.2f}</td>
                            <td>{bb_position:.2f}</td>
                            <td>${f"{support:.2f}" if isinstance(support, (int, float)) else support}</td>
                            <td>${f"{resistance:.2f}" if isinstance(resistance, (int, float)) else resistance}</td>
                            <td>${latest['MA20']:.2f}</td>
                            <td>${latest['MA50']:.2f}</td>
                        </tr>
                    """
                    
                    # 添加交易信号到HTML
                    if buy_signals:
                        html_body += f"""
                        <tr>
                            <td colspan="11">
                                <div class="buy-signal">
                                    <strong>🔔 {data['name']} ({symbol}) 最近买入信号:</strong><br>
                        """
                        for signal in buy_signals:
                            html_body += f"<span style='color: green;'>• {signal['date']}: {signal['description']}</span><br>"
                        html_body += """
                                </div>
                            </td>
                        </tr>
                        """
                    
                    if sell_signals:
                        html_body += f"""
                        <tr>
                            <td colspan="11">
                                <div class="sell-signal">
                                    <strong>🔻 {data['name']} ({symbol}) 最近卖出信号:</strong><br>
                        """
                        for signal in sell_signals:
                            html_body += f"<span style='color: red;'>• {signal['date']}: {signal['description']}</span><br>"
                        html_body += """
                                </div>
                            </td>
                        </tr>
                        """
            
            html_body += """
                    </table>
                </div>
            """
            
            # 3. 宏观经济环境
            html_body += """
                <div class="section">
                    <h3>📊 宏观经济环境</h3>
                    <table>
                        <tr>
                            <th>指标名称</th>
                            <th>代码</th>
                            <th>最新值</th>
                        </tr>
            """
            
            for symbol, data in macro_data.items():
                if not data['data'].empty:
                    latest = data['data'].iloc[-1]
                    if 'Close' in latest:
                        html_body += f"""
                            <tr>
                                <td>{data['name']}</td>
                                <td>{symbol}</td>
                                <td>{latest['Close']:.2f}</td>
                            </tr>
                        """
            
            html_body += """
                    </table>
                </div>
            """
            
            # 4. 大模型分析
            if llm_analysis:
                html_body += """
                    <div class="section">
                        <h3>🤖 大模型深度分析</h3>
                        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px;">
                """
                
                try:
                    # 尝试解析JSON格式的输出
                    analysis_json = json.loads(llm_analysis)
                    
                    # 趋势分析
                    trend_analysis = analysis_json.get('trend_analysis', 'N/A')
                    if trend_analysis != 'N/A':
                        trend_analysis_html = trend_analysis.replace('\n', '<br>').replace('。', '。<br>')
                    html_body += f"""
                        <div style="margin-bottom: 15px;">
                            <h4 style="color: #333; margin-bottom: 5px;">📈 趋势分析</h4>
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 3px; margin: 5px 0;">
                                {trend_analysis_html}
                            </div>
                        </div>
                        """
                    
                    # 技术信号
                    technical_signals = analysis_json.get('technical_signals', 'N/A')
                    if technical_signals != 'N/A':
                        technical_signals_html = technical_signals.replace('\n', '<br>').replace('。', '。<br>')
                    html_body += f"""
                        <div style="margin-bottom: 15px;">
                            <h4 style="color: #333; margin-bottom: 5px;">📊 技术信号</h4>
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 3px; margin: 5px 0;">
                                {technical_signals_html}
                            </div>
                        </div>
                        """
                    
                    # 宏观影响
                    macro_impact = analysis_json.get('macro_impact', 'N/A')
                    if macro_impact != 'N/A':
                        macro_impact_html = macro_impact.replace('\n', '<br>').replace('。', '。<br>')
                    html_body += f"""
                        <div style="margin-bottom: 15px;">
                            <h4 style="color: #333; margin-bottom: 5px;">💼 宏观影响</h4>
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 3px; margin: 5px 0;">
                                {macro_impact_html}
                            </div>
                        </div>
                        """
                    
                    # 投资建议
                    advice = analysis_json.get('investment_advice', {})
                    short_term = advice.get('short_term', 'N/A')
                    medium_term = advice.get('medium_term', 'N/A')
                    long_term = advice.get('long_term', 'N/A')
                    
                    if short_term != 'N/A' or medium_term != 'N/A' or long_term != 'N/A':
                        html_body += """
                        <div style="margin-bottom: 15px;">
                            <h4 style="color: #333; margin-bottom: 5px;">💡 投资建议</h4>
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 3px; margin: 5px 0;">
                        """
                        if short_term != 'N/A':
                            short_term_html = short_term.replace('\n', '<br>').replace('。', '。<br>')
                            html_body += f"""
                            <div style="margin: 5px 0;">
                                <strong>📌 短期:</strong><br>
                                <span>{short_term_html}</span>
                            </div>
                            """
                        if medium_term != 'N/A':
                            medium_term_html = medium_term.replace('\n', '<br>').replace('。', '。<br>')
                            html_body += f"""
                            <div style="margin: 5px 0;">
                                <strong>📌 中期:</strong><br>
                                <span>{medium_term_html}</span>
                            </div>
                            """
                        if long_term != 'N/A':
                            long_term_html = long_term.replace('\n', '<br>').replace('。', '。<br>')
                            html_body += f"""
                            <div style="margin: 5px 0;">
                                <strong>📌 长期:</strong><br>
                                <span>{long_term_html}</span>
                            </div>
                            """
                        html_body += """
                            </div>
                        </div>
                        """
                    
                    # 风险预警
                    risk_warning = analysis_json.get('risk_warning', 'N/A')
                    if risk_warning != 'N/A':
                        risk_warning_html = risk_warning.replace('\n', '<br>').replace('。', '。<br>')
                    html_body += f"""
                        <div style="margin-bottom: 15px;">
                            <h4 style="color: #333; margin-bottom: 5px;">⚠️ 风险预警</h4>
                            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 3px; margin: 5px 0;">
                                {risk_warning_html}
                            </div>
                        </div>
                        """
                except json.JSONDecodeError:
                    # 如果不是JSON格式，直接输出并格式化
                    formatted_output = llm_analysis.replace('\n', '<br>').replace('\\n', '<br>')
                    html_body += """
                        <div style="background-color: #ffffff; padding: 10px; border-radius: 3px; margin: 5px 0;">
                            {}
                        </div>
                    """.format(formatted_output)
                except Exception as e:
                    html_body += f"<p>分析内容解析失败: {e}</p>"
                    html_body += f"<p>原始内容: {llm_analysis}</p>"
                
                html_body += """
                        </div>
                    </div>
                """
            else:
                html_body += """
                    <div class="section">
                        <h3>🤖 大模型深度分析</h3>
                        <p style="color: #888;">⚠️ 大模型分析暂不可用<br>请检查大模型服务配置或API密钥</p>
                    </div>
                """
            
            # 添加指标说明
            html_body += """
                <div class="section">
                    <h3>📋 指标说明</h3>
                    <div style="font-size:0.9em; line-height:1.4;">
                    <ul>
                      <li><b>价格(USD)</b>：黄金相关资产的当前价格，以美元计价。</li>
                      <li><b>24小时变化(%)</b>：过去24小时内价格的变化百分比。</li>
                      <li><b>5日变化(%)</b>：过去5个交易日内价格的变化百分比。</li>
                      <li><b>20日变化(%)</b>：过去20个交易日内价格的变化百分比。</li>
                      <li><b>RSI(相对强弱指数)</b>：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。</li>
                      <li><b>MACD(异同移动平均线)</b>：判断价格趋势和动能的技术指标。</li>
                      <li><b>MA20(20日移动平均线)</b>：过去20个交易日的平均价格，反映短期趋势。</li>
                      <li><b>MA50(50日移动平均线)</b>：过去50个交易日的平均价格，反映中期趋势。</li>
                      <li><b>布林带位置</b>：当前价格在布林带中的相对位置，范围0-1。接近0表示价格接近下轨（可能超卖），接近1表示价格接近上轨（可能超买）。</li>
                      <li><b>趋势</b>：市场当前的整体方向。
                        <ul>
                          <li><b>强势多头</b>：价格强劲上涨趋势，各周期均线呈多头排列（价格 > MA20 > MA50 > MA200）</li>
                          <li><b>多头趋势</b>：价格上涨趋势，中期均线呈多头排列（价格 > MA20 > MA50）</li>
                          <li><b>弱势空头</b>：价格持续下跌趋势，各周期均线呈空头排列（价格 < MA20 < MA50 < MA200）</li>
                          <li><b>空头趋势</b>：价格下跌趋势，中期均线呈空头排列（价格 < MA20 < MA50）</li>
                          <li><b>震荡整理</b>：价格在一定区间内波动，无明显趋势</li>
                          <li><b>短期上涨/下跌</b>：基于最近价格变化的短期趋势判断</li>
                        </ul>
                      </li>
                    </ul>
                    </div>
                </div>
            """
            
            # 结束HTML
            html_body += """
            </body>
            </html>
            """
            
            # 在文本版本中也添加交易信号
            for symbol, data in technical_analysis.items():
                if not data['indicators'].empty:
                    # 检查最近的交易信号
                    recent_signals = data['indicators'].tail(5)
                    buy_signals = []
                    sell_signals = []
                    
                    if 'Buy_Signal' in recent_signals.columns:
                        buy_signals_df = recent_signals[recent_signals['Buy_Signal'] == True]
                        for idx, row in buy_signals_df.iterrows():
                            buy_signals.append({
                                'date': idx.strftime('%Y-%m-%d'),
                                'description': row['Signal_Description']
                            })
                    
                    if 'Sell_Signal' in recent_signals.columns:
                        sell_signals_df = recent_signals[recent_signals['Sell_Signal'] == True]
                        for idx, row in sell_signals_df.iterrows():
                            sell_signals.append({
                                'date': idx.strftime('%Y-%m-%d'),
                                'description': row['Signal_Description']
                            })
                    
                    if buy_signals or sell_signals:
                        text_body += f"\n📊 {data['name']} ({symbol}) 交易信号:\n"
                        if buy_signals:
                            text_body += f"  🔔 最近买入信号 ({len(buy_signals)} 个):\n"
                            for signal in buy_signals:
                                text_body += f"    {signal['date']}: {signal['description']}\n"
                        if sell_signals:
                            text_body += f"  🔻 最近卖出信号 ({len(sell_signals)} 个):\n"
                            for signal in sell_signals:
                                text_body += f"    {signal['date']}: {signal['description']}\n"
            
            # 添加大模型分析到文本版本
            if llm_analysis:
                text_body += "\n🤖 大模型深度分析:\n"
                text_body += "=" * 30 + "\n"
                try:
                    # 尝试解析JSON格式的输出
                    analysis_json = json.loads(llm_analysis)
                    
                    # 趋势分析
                    trend_analysis = analysis_json.get('trend_analysis', 'N/A')
                    if trend_analysis != 'N/A':
                        text_body += "📈 趋势分析:\n"
                        # 格式化输出，每行不超过70个字符
                        lines = []
                        for i in range(0, len(trend_analysis), 70):
                            lines.append(trend_analysis[i:i+70])
                        for line in lines:
                            text_body += f"  {line}\n"
                        text_body += "\n"
                    
                    # 技术信号
                    technical_signals = analysis_json.get('technical_signals', 'N/A')
                    if technical_signals != 'N/A':
                        text_body += "📊 技术信号:\n"
                        # 格式化输出，每行不超过70个字符
                        lines = []
                        for i in range(0, len(technical_signals), 70):
                            lines.append(technical_signals[i:i+70])
                        for line in lines:
                            text_body += f"  {line}\n"
                        text_body += "\n"
                    
                    # 宏观影响
                    macro_impact = analysis_json.get('macro_impact', 'N/A')
                    if macro_impact != 'N/A':
                        text_body += "💼 宏观影响:\n"
                        # 格式化输出，每行不超过70个字符
                        lines = []
                        for i in range(0, len(macro_impact), 70):
                            lines.append(macro_impact[i:i+70])
                        for line in lines:
                            text_body += f"  {line}\n"
                        text_body += "\n"
                    
                    # 投资建议
                    advice = analysis_json.get('investment_advice', {})
                    short_term = advice.get('short_term', 'N/A')
                    medium_term = advice.get('medium_term', 'N/A')
                    long_term = advice.get('long_term', 'N/A')
                    
                    if short_term != 'N/A' or medium_term != 'N/A' or long_term != 'N/A':
                        text_body += "💡 投资建议:\n"
                        if short_term != 'N/A':
                            text_body += "  📌 短期:\n"
                            # 格式化输出，每行不超过65个字符
                            lines = []
                            for i in range(0, len(short_term), 65):
                                lines.append(short_term[i:i+65])
                            for line in lines:
                                text_body += f"    {line}\n"
                        if medium_term != 'N/A':
                            text_body += "  📌 中期:\n"
                            # 格式化输出，每行不超过65个字符
                            lines = []
                            for i in range(0, len(medium_term), 65):
                                lines.append(medium_term[i:i+65])
                            for line in lines:
                                text_body += f"    {line}\n"
                        if long_term != 'N/A':
                            text_body += "  📌 长期:\n"
                            # 格式化输出，每行不超过65个字符
                            lines = []
                            for i in range(0, len(long_term), 65):
                                lines.append(long_term[i:i+65])
                            for line in lines:
                                text_body += f"    {line}\n"
                        text_body += "\n"
                    
                    # 风险预警
                    risk_warning = analysis_json.get('risk_warning', 'N/A')
                    if risk_warning != 'N/A':
                        text_body += "⚠️ 风险预警:\n"
                        # 格式化输出，每行不超过70个字符
                        lines = []
                        for i in range(0, len(risk_warning), 70):
                            lines.append(risk_warning[i:i+70])
                        for line in lines:
                            text_body += f"  {line}\n"
                        text_body += "\n"
                except json.JSONDecodeError:
                    # 如果不是JSON格式，直接输出
                    text_body += "分析结果:\n"
                    # 格式化输出，每行不超过70个字符
                    lines = []
                    for i in range(0, len(llm_analysis), 70):
                        lines.append(llm_analysis[i:i+70])
                    for line in lines:
                        text_body += f"  {line}\n"
                    text_body += "\n"
                except Exception as e:
                    text_body += f"分析内容解析失败: {e}\n"
                    text_body += f"原始内容: {llm_analysis}\n\n"
            else:
                text_body += "\n⚠️ 大模型分析暂不可用\n"
                text_body += "请检查大模型服务配置或API密钥\n\n"
            
            # 添加指标说明到文本版本
            text_body += "\n📋 指标说明:\n"
            text_body += "价格(USD)：黄金相关资产的当前价格，以美元计价。\n"
            text_body += "24小时变化(%)：过去24小时内价格的变化百分比。\n"
            text_body += "5日变化(%)：过去5个交易日内价格的变化百分比。\n"
            text_body += "20日变化(%)：过去20个交易日内价格的变化百分比。\n"
            text_body += "RSI(相对强弱指数)：衡量价格变化速度和幅度的技术指标，范围0-100。超过70通常表示超买，低于30表示超卖。\n"
            text_body += "MACD(异同移动平均线)：判断价格趋势和动能的技术指标。\n"
            text_body += "MA20(20日移动平均线)：过去20个交易日的平均价格，反映短期趋势。\n"
            text_body += "MA50(50日移动平均线)：过去50个交易日的平均价格，反映中期趋势。\n"
            text_body += "布林带位置：当前价格在布林带中的相对位置，范围0-1。接近0表示价格接近下轨（可能超卖），接近1表示价格接近上轨（可能超买）。\n"
            text_body += "趋势：市场当前的整体方向。\n"
            text_body += "  强势多头：价格强劲上涨趋势，各周期均线呈多头排列（价格 > MA20 > MA50 > MA200）\n"
            text_body += "  多头趋势：价格上涨趋势，中期均线呈多头排列（价格 > MA20 > MA50）\n"
            text_body += "  弱势空头：价格持续下跌趋势，各周期均线呈空头排列（价格 < MA20 < MA50 < MA200）\n"
            text_body += "  空头趋势：价格下跌趋势，中期均线呈空头排列（价格 < MA20 < MA50）\n"
            text_body += "  震荡整理：价格在一定区间内波动，无明显趋势\n"
            text_body += "  短期上涨/下跌：基于最近价格变化的短期趋势判断\n"
            
            # 创建邮件消息
            msg = MIMEMultipart("mixed")
            msg['From'] = f'"Gold Analyzer" <{sender_email}>'
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            # 添加文本和HTML版本
            body = MIMEMultipart("alternative")
            body.attach(MIMEText(text_body, "plain", "utf-8"))
            body.attach(MIMEText(html_body, "html", "utf-8"))
            msg.attach(body)
            
            # 根据SMTP服务器类型选择合适的端口和连接方式
            if "163.com" in smtp_server:
                # 163邮箱使用SSL连接，端口465
                smtp_port = 465
                use_ssl = True
            elif "gmail.com" in smtp_server:
                # Gmail使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False
            else:
                # 默认使用TLS连接，端口587
                smtp_port = 587
                use_ssl = False
            
            # 发送邮件（增加重试机制）
            for attempt in range(3):
                try:
                    if use_ssl:
                        # 使用SSL连接
                        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()
                    else:
                        # 使用TLS连接
                        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                        server.starttls()
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(sender_email, recipients, msg.as_string())
                        server.quit()
                    
                    print("✅ 邮件发送成功")
                    return True
                except Exception as e:
                    print(f"❌ 发送邮件失败 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试，等待后重试
                        time.sleep(5)
            
            print("❌ 发送邮件失败，已重试3次")
            return False
            
        except Exception as e:
            print("❌ 邮件发送过程中出现错误: {}".format(e))
            return False

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