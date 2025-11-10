#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数(HSI)分析器
使用腾讯财经接口获取恒生指数数据，结合技术分析工具和hk_smart_money_tracker的分析方法，
对恒生指数进行全面的技术分析和趋势判断
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

class HSIAnalyzer:
    def __init__(self, period_days=90):
        """
        初始化恒生指数分析器
        
        Args:
            period_days (int): 获取数据的天数，默认90天
        """
        self.period_days = period_days
        self.technical_analyzer = TechnicalAnalyzer()
        self.data = None
        self.indicators = None
        
    def fetch_hsi_data(self):
        """
        获取恒生指数数据
        
        Returns:
            pandas.DataFrame: 恒生指数数据，包含Date, Open, High, Low, Close, Volume等列
        """
        print("📈 获取恒生指数（HSI）数据...")
        self.data = get_hsi_data_tencent(period_days=self.period_days)
        
        if self.data is None or self.data.empty:
            print("❌ 无法获取恒生指数数据")
            return None
        
        print(f"✅ 成功获取 {len(self.data)} 天的恒生指数数据")
        return self.data
    
    def calculate_technical_indicators(self):
        """
        计算技术指标
        
        Returns:
            pandas.DataFrame: 包含所有技术指标的数据框
        """
        if self.data is None or self.data.empty:
            print("❌ 无数据可计算技术指标")
            return None
            
        print("📊 计算技术指标...")
        
        # 复制数据以避免修改原始数据
        df = self.data.copy()
        
        # 使用TechnicalAnalyzer计算所有技术指标
        df = self.technical_analyzer.calculate_all_indicators(df)
        
        # 计算额外的恒生指数专用指标
        df = self._calculate_hsi_specific_indicators(df)
        
        self.indicators = df
        print("✅ 技术指标计算完成")
        
        return df
    
    def _calculate_hsi_specific_indicators(self, df):
        """
        计算恒生指数专用指标
        
        Args:
            df (pandas.DataFrame): 包含基础数据的数据框
            
        Returns:
            pandas.DataFrame: 更新后的数据框，包含专用指标
        """
        # 计算价格位置（在最近N日内的百分位位置）
        price_window = 60
        if len(df) >= price_window:
            rolling_low = df['Close'].rolling(window=price_window).min()
            rolling_high = df['Close'].rolling(window=price_window).max()
            df['Price_Percentile'] = ((df['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
        else:
            # 如果数据不足，使用全部可用数据
            rolling_low = df['Close'].rolling(window=len(df)).min()
            rolling_high = df['Close'].rolling(window=len(df)).max()
            df['Price_Percentile'] = ((df['Close'] - rolling_low) / (rolling_high - rolling_low)) * 100
        
        # 计算移动平均线偏离率
        if 'MA5' in df.columns:
            df['MA5_Deviation'] = ((df['Close'] - df['MA5']) / df['MA5']) * 100
        if 'MA10' in df.columns:
            df['MA10_Deviation'] = ((df['Close'] - df['MA10']) / df['MA10']) * 100
        if 'MA20' in df.columns:
            df['MA20_Deviation'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        if 'MA50' in df.columns:
            df['MA50_Deviation'] = ((df['Close'] - df['MA50']) / df['MA50']) * 100
        if 'MA200' in df.columns:
            df['MA200_Deviation'] = ((df['Close'] - df['MA200']) / df['MA200']) * 100
        
        # 计算成交量比率（相对于20日均量）
        if 'Volume' in df.columns:
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
        
        # 计算波动率（20日年化波动率）
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # 计算VWAP（成交量加权平均价格）
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['TP'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # 计算资金流量指标
        # 钱德动量摆动指标（Chaikin Money Flow）
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns and 'Volume' in df.columns:
            money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            money_flow_volume = money_flow_multiplier * df['Volume']
            df['CMF'] = money_flow_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # 计算随机指标（Stochastic Oscillator）
        k_period = 14
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            df['Low_Min'] = df['Low'].rolling(window=k_period).min()
            df['High_Max'] = df['High'].rolling(window=k_period).max()
            df['Stoch_K'] = 100 * (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min'])
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 计算威廉指标（Williams %R）
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            df['Williams_R'] = (df['High_Max'] - df['Close']) / (df['High_Max'] - df['Low_Min']) * -100
        
        return df
    
    def generate_signals(self):
        """
        基于技术指标生成买卖信号
        
        Returns:
            pandas.DataFrame: 包含买卖信号的数据框
        """
        if self.indicators is None:
            print("❌ 请先计算技术指标")
            return None
        
        print("🔔 生成交易信号...")
        
        df = self.indicators.copy()
        
        # 使用TechnicalAnalyzer生成基本信号
        df = self.technical_analyzer.generate_buy_sell_signals(df)
        
        # 添加恒生指数专用信号
        df = self._generate_hsi_specific_signals(df)
        
        print("✅ 交易信号生成完成")
        
        return df
    
    def _generate_hsi_specific_signals(self, df):
        """
        生成恒生指数专用信号
        
        Args:
            df (pandas.DataFrame): 包含技术指标的数据框
            
        Returns:
            pandas.DataFrame: 更新后的数据框，包含专用信号
        """
        # 初始化信号列
        if 'Buy_Signal' not in df.columns:
            df['Buy_Signal'] = False
        if 'Sell_Signal' not in df.columns:
            df['Sell_Signal'] = False
        if 'Signal_Description' not in df.columns:
            df['Signal_Description'] = ''
        
        # 计算趋势信号
        df['Trend'] = self._analyze_trend(df)
        
        # 添加更多高级信号
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # 添加更多信号判断逻辑
            additional_signal_parts = []
            
            # 价格位置信号
            if 'Price_Percentile' in df.columns:
                if current['Price_Percentile'] is not None:
                    if current['Price_Percentile'] < 20:  # 超卖
                        additional_signal_parts.append("超卖")
                    elif current['Price_Percentile'] > 80:  # 超买
                        additional_signal_parts.append("超买")
            
            # 成交量信号
            if 'Vol_Ratio' in df.columns:
                if current['Vol_Ratio'] is not None:
                    if current['Vol_Ratio'] > 2.0:  # 显著放量
                        additional_signal_parts.append("放量")
                    elif current['Vol_Ratio'] < 0.5:  # 显著缩量
                        additional_signal_parts.append("缩量")
            
            # 波动率信号
            if 'Volatility' in df.columns:
                if current['Volatility'] is not None:
                    if current['Volatility'] > 30:  # 高波动
                        additional_signal_parts.append("高波动")
                    elif current['Volatility'] < 15:  # 低波动
                        additional_signal_parts.append("低波动")
            
            # 更新信号描述
            if additional_signal_parts:
                if df.at[df.index[i], 'Signal_Description']:
                    df.at[df.index[i], 'Signal_Description'] += " | " + ", ".join(additional_signal_parts)
                else:
                    df.at[df.index[i], 'Signal_Description'] = ", ".join(additional_signal_parts)
        
        return df
    
    def _analyze_trend(self, df):
        """
        分析趋势
        
        Args:
            df (pandas.DataFrame): 包含技术指标的数据框
            
        Returns:
            str: 趋势状态
        """
        if df.empty or len(df) < 50:
            return ["数据不足"] * len(df)
        
        trends = []
        for i in range(len(df)):
            current = df.iloc[i]
            
            # 获取当前价格和均线值
            current_price = current['Close']
            ma20 = current['MA20'] if 'MA20' in df.columns and not pd.isna(current['MA20']) else np.nan
            ma50 = current['MA50'] if 'MA50' in df.columns and not pd.isna(current['MA50']) else np.nan
            ma200 = current['MA200'] if 'MA200' in df.columns and not pd.isna(current['MA200']) else np.nan
            
            # 趋势判断逻辑
            if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
                if current_price > ma20 > ma50 > ma200:
                    trends.append("强势多头")
                elif current_price < ma20 < ma50 < ma200:
                    trends.append("弱势空头")
                else:
                    trends.append("震荡整理")
            elif not pd.isna(ma20) and not pd.isna(ma50):
                if current_price > ma20 > ma50:
                    trends.append("多头趋势")
                elif current_price < ma20 < ma50:
                    trends.append("空头趋势")
                else:
                    trends.append("震荡")
            elif len(df) >= 20:
                # 使用短期趋势判断
                if i >= 20:
                    past_price = df.iloc[i-20]['Close']
                    if current_price > past_price:
                        trends.append("短期上涨")
                    else:
                        trends.append("短期下跌")
                else:
                    trends.append("数据不足")
            else:
                trends.append("数据不足")
        
        return trends
    
    def analyze_market_regime(self):
        """
        分析市场状态（牛熊震荡）
        
        Returns:
            dict: 市场状态分析结果
        """
        if self.indicators is None or self.indicators.empty:
            return {"error": "无数据可分析"}
        
        latest = self.indicators.iloc[-1]
        
        # 基于价格位置和趋势判断市场状态
        price_level = "未知"
        if 'Price_Percentile' in self.indicators.columns:
            pct = latest['Price_Percentile']
            if pct is not None:
                if pct > 70:
                    price_level = "高位"
                elif pct < 30:
                    price_level = "低位"
                else:
                    price_level = "中位"
        
        # 基于趋势判断
        trend = "未知"
        if 'Trend' in self.indicators.columns:
            trend = latest['Trend']
        
        # 基于技术指标判断市场强度
        market_strength = "中性"
        if 'RSI' in self.indicators.columns and 'MACD' in self.indicators.columns:
            rsi = latest['RSI']
            macd = latest['MACD']
            
            if rsi is not None and macd is not None:
                if rsi > 60 and macd > 0:
                    market_strength = "强势"
                elif rsi < 40 and macd < 0:
                    market_strength = "弱势"
                else:
                    market_strength = "中性"
        
        return {
            "current_level": price_level,
            "trend": trend,
            "strength": market_strength,
            "current_price": latest['Close'],
            "current_date": latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)
        }
    
    def generate_report(self, include_llm_analysis=False):
        """
        生成分析报告
        
        Args:
            include_llm_analysis (bool): 是否包含大模型分析
            
        Returns:
            str: 分析报告内容
        """
        if self.indicators is None:
            return "❌ 无数据可生成报告"
        
        latest = self.indicators.iloc[-1]
        
        report = []
        report.append("="*60)
        report.append("📊 恒生指数(HSI)技术分析报告")
        report.append("="*60)
        report.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据日期: {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)}")
        report.append(f"当前指数: {latest['Close']:.2f}")
        report.append("")
        
        # 市场状态分析
        regime = self.analyze_market_regime()
        report.append("📈 市场状态分析:")
        report.append(f"  当前位置: {regime['current_level']}")
        report.append(f"  当前趋势: {regime['trend']}")
        report.append(f"  市场强度: {regime['strength']}")
        report.append("")
        
        # 关键技术指标
        report.append("📊 关键技术指标:")
        if 'RSI' in self.indicators.columns:
            report.append(f"  RSI(14): {latest['RSI']:.2f}")
        if 'MACD' in self.indicators.columns:
            report.append(f"  MACD: {latest['MACD']:.4f}, 信号线: {latest['MACD_signal']:.4f}")
        if 'MA20' in self.indicators.columns:
            report.append(f"  MA20: {latest['MA20']:.2f}")
        if 'MA50' in self.indicators.columns:
            report.append(f"  MA50: {latest['MA50']:.2f}")
        if 'MA200' in self.indicators.columns:
            report.append(f"  MA200: {latest['MA200']:.2f}")
        if 'Price_Percentile' in self.indicators.columns:
            report.append(f"  价格位置: {latest['Price_Percentile']:.2f}%")
        if 'Volatility' in self.indicators.columns:
            report.append(f"  波动率: {latest['Volatility']:.2f}%")
        if 'Vol_Ratio' in self.indicators.columns:
            report.append(f"  量比: {latest['Vol_Ratio']:.2f}")
        report.append("")
        
        # 交易信号
        signal_columns = [col for col in ['Buy_Signal', 'Sell_Signal', 'Signal_Description'] if col in self.indicators.columns]
        if signal_columns:
            recent_signals = self.indicators.tail(5)[signal_columns].dropna()
            # 过滤出有信号的行
            if 'Buy_Signal' in signal_columns and 'Sell_Signal' in signal_columns:
                recent_signals = recent_signals[(recent_signals['Buy_Signal']) | (recent_signals['Sell_Signal'])]
            elif 'Buy_Signal' in signal_columns:
                recent_signals = recent_signals[recent_signals['Buy_Signal']]
            elif 'Sell_Signal' in signal_columns:
                recent_signals = recent_signals[recent_signals['Sell_Signal']]
            
            if not recent_signals.empty:
                report.append("🔔 最近交易信号:")
                for idx, row in recent_signals.iterrows():
                    if 'Buy_Signal' in signal_columns and 'Sell_Signal' in signal_columns:
                        signal_type = "买入" if row['Buy_Signal'] else "卖出"
                    elif 'Buy_Signal' in signal_columns:
                        signal_type = "买入" if row['Buy_Signal'] else ""
                    elif 'Sell_Signal' in signal_columns:
                        signal_type = "卖出" if row['Sell_Signal'] else ""
                    
                    description = row['Signal_Description'] if 'Signal_Description' in signal_columns else "未提供描述"
                    report.append(f"  {idx.strftime('%Y-%m-%d')}: {signal_type} - {description}")
            else:
                report.append("🔔 最近无明显交易信号")
        else:
            report.append("🔔 最近无明显交易信号")
        report.append("")
        
        # 趋势分析
        report.append("📈 趋势分析:")
        if 'Trend' in self.indicators.columns:
            trend = latest['Trend']
            report.append(f"  当前趋势: {trend}")
            
            # 提供趋势操作建议
            if trend in ["强势多头", "多头趋势"]:
                report.append("  建议: 保持多头思维，关注回调买入机会")
            elif trend in ["弱势空头", "空头趋势"]:
                report.append("  建议: 谨慎操作，关注反弹卖出机会")
            else:
                report.append("  建议: 震荡市中注意高抛低吸")
        report.append("")
        
        # 风险提示
        report.append("⚠️ 风险提示:")
        if 'RSI' in self.indicators.columns:
            rsi = latest['RSI']
            if rsi is not None:
                if rsi > 70:
                    report.append("  - RSI超买，注意回调风险")
                elif rsi < 30:
                    report.append("  - RSI超卖，注意反弹机会")
        if 'Volatility' in self.indicators.columns:
            vol = latest['Volatility']
            if vol is not None:
                if vol > 30:
                    report.append("  - 市场波动率较高，注意风险控制")
                elif vol < 10:
                    report.append("  - 市场波动率较低，关注突破机会")
        report.append("")
        
        # 大模型分析
        if include_llm_analysis and LLM_AVAILABLE:
            try:
                llm_analysis = self.generate_llm_trading_strategy()
                report.append("🤖 大模型交易策略分析:")
                report.append(llm_analysis)
                report.append("")
            except Exception as e:
                report.append("❌ 大模型分析失败:")
                report.append(f"  错误信息: {str(e)}")
                report.append("")
        elif include_llm_analysis and not LLM_AVAILABLE:
            report.append("❌ 大模型分析不可用:")
            report.append("  未找到大模型服务模块")
            report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    
    
    def run_analysis(self, show_charts=False, save_charts=False):
        """
        运行完整的恒生指数分析
        
        Args:
            show_charts (bool): 是否显示图表
            save_charts (bool): 是否保存图表
            
        Returns:
            dict: 分析结果
        """
        print("🚀 开始恒生指数分析...")
        
        # 获取数据
        data = self.fetch_hsi_data()
        if data is None:
            return {"error": "无法获取数据"}
        
        # 计算技术指标
        indicators = self.calculate_technical_indicators()
        if indicators is None:
            return {"error": "无法计算技术指标"}
        
        # 生成信号
        signals = self.generate_signals()
        if signals is None:
            return {"error": "无法生成信号"}
        
        # 生成报告
        report = self.generate_report()
        
        # 不再生成图表
        
        # 返回分析结果
        result = {
            "data": self.data,
            "indicators": self.indicators,
            "report": report,
            "signals": signals,
            "regime": self.analyze_market_regime()
        }
        
        print("\n" + report)
        
        return result

    def send_email_report(self, report_content):
        """
        发送邮件报告
        
        Args:
            report_content (str): 分析报告内容
        """
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
            subject = "恒生指数(HSI)分析报告"
            
            # 纯文本版本
            text_body = report_content
            
            # HTML版本
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
                    pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }}
                </style>
            </head>
            <body>
                <h2>📈 恒生指数(HSI)分析报告</h2>
                <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <pre>{report_content}</pre>
            </body>
            </html>
            """
            
            # 创建邮件消息
            msg = MIMEMultipart("mixed")
            msg['From'] = f'"HSI Analyzer" <{sender_email}>'
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

    def generate_llm_trading_strategy(self):
        """
        使用大模型分析恒生指数数据并生成交易策略
        
        Returns:
            str: 大模型生成的交易策略
        """
        if self.indicators is None or self.indicators.empty:
            return "❌ 无数据可分析"
        
        # 获取最新的技术指标数据
        latest_data = self.indicators.iloc[-1]
        
        # 构建分析报告内容作为大模型输入
        analysis_summary = []
        analysis_summary.append("恒生指数(HSI)技术分析数据:")
        analysis_summary.append(f"当前指数: {latest_data['Close']:.2f}")
        analysis_summary.append(f"市场状态: {self.analyze_market_regime()}")
        analysis_summary.append("")
        
        # 添加关键技术指标
        analysis_summary.append("关键技术指标:")
        if 'RSI' in self.indicators.columns:
            analysis_summary.append(f"RSI: {latest_data['RSI']:.2f}")
        if 'MACD' in self.indicators.columns and 'MACD_signal' in self.indicators.columns:
            analysis_summary.append(f"MACD: {latest_data['MACD']:.4f}, 信号线: {latest_data['MACD_signal']:.4f}")
        if 'MA20' in self.indicators.columns:
            analysis_summary.append(f"MA20: {latest_data['MA20']:.2f}")
        if 'MA50' in self.indicators.columns:
            analysis_summary.append(f"MA50: {latest_data['MA50']:.2f}")
        if 'MA200' in self.indicators.columns:
            analysis_summary.append(f"MA200: {latest_data['MA200']:.2f}")
        if 'Price_Percentile' in self.indicators.columns:
            analysis_summary.append(f"价格位置: {latest_data['Price_Percentile']:.2f}%")
        if 'Volatility' in self.indicators.columns:
            analysis_summary.append(f"波动率: {latest_data['Volatility']:.2f}%")
        if 'Vol_Ratio' in self.indicators.columns:
            analysis_summary.append(f"量比: {latest_data['Vol_Ratio']:.2f}")
        analysis_summary.append("")
        
        # 添加趋势分析
        if 'Trend' in self.indicators.columns:
            analysis_summary.append(f"当前趋势: {latest_data['Trend']}")
        analysis_summary.append("")
        
        # 添加最近的交易信号
        signal_columns = [col for col in ['Buy_Signal', 'Sell_Signal', 'Signal_Description'] if col in self.indicators.columns]
        if signal_columns:
            recent_signals = self.indicators.tail(5)[signal_columns].dropna()
            if 'Buy_Signal' in signal_columns and 'Sell_Signal' in signal_columns:
                recent_signals = recent_signals[(recent_signals['Buy_Signal']) | (recent_signals['Sell_Signal'])]
            elif 'Buy_Signal' in signal_columns:
                recent_signals = recent_signals[recent_signals['Buy_Signal']]
            elif 'Sell_Signal' in signal_columns:
                recent_signals = recent_signals[recent_signals['Sell_Signal']]
            
            if not recent_signals.empty:
                analysis_summary.append("最近交易信号:")
                for idx, row in recent_signals.iterrows():
                    if 'Buy_Signal' in signal_columns and 'Sell_Signal' in signal_columns:
                        signal_type = "买入" if row['Buy_Signal'] else "卖出"
                    elif 'Buy_Signal' in signal_columns:
                        signal_type = "买入" if row['Buy_Signal'] else ""
                    elif 'Sell_Signal' in signal_columns:
                        signal_type = "卖出" if row['Sell_Signal'] else ""
                    
                    description = row['Signal_Description'] if 'Signal_Description' in signal_columns else "未提供描述"
                    analysis_summary.append(f"  {idx.strftime('%Y-%m-%d')}: {signal_type} - {description}")
            else:
                analysis_summary.append("最近无明显交易信号")
        analysis_summary.append("")
        
        # 获取历史数据用于趋势分析
        historical_data = self.indicators.tail(20)  # 最近20天的数据
        analysis_summary.append("最近20天指数变化:")
        for idx, row in historical_data.iterrows():
            analysis_summary.append(f"  {idx.strftime('%Y-%m-%d')}: {row['Close']:.2f}")
        
        # 构建大模型提示
        prompt = f"""
请分析以下恒生指数(HSI)技术分析数据，并提供专业的交易策略建议：

{chr(10).join(analysis_summary)}

请根据以下原则提供交易策略：
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
- 具体操作建议
- 风险控制措施
- 目标价位和止损位

请确保策略符合港股市场特点和恒生指数的特性。
"""
        
        try:
            # 调用大模型
            response = chat_with_llm(prompt)
            return response
        except Exception as e:
            return f"调用大模型失败: {str(e)}"

def main():
    """主函数"""
    import argparse
    
    print("📈 恒生指数(HSI)分析器")
    print("="*50)
    
    # 创建分析器实例
    analyzer = HSIAnalyzer(period_days=90)
    
    # 运行分析
    result = analyzer.run_analysis(show_charts=False, save_charts=False)
    
    if result is not None and "error" not in result:
        # 生成报告，始终包含大模型分析
        report = analyzer.generate_report(include_llm_analysis=True)
        print("\n" + report)
        
        # 始终发送邮件报告
        analyzer.send_email_report(report)
        
        print("\n✅ 恒生指数分析完成！")
    else:
        if result and "error" in result:
            print(f"\n❌ 分析失败: {result['error']}")
        else:
            print("\n❌ 分析失败: 未知错误")

if __name__ == "__main__":
    main()
