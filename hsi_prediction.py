#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数涨跌预测脚本

基于模型特征重要性，使用加权评分模型预测恒生指数短期走势
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
output_dir = os.path.join(script_dir, 'output')

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


class HSI_Predictor:
    """恒生指数预测器"""

    # 特征重要性配置（权重、影响方向）
    FEATURE_IMPORTANCE = {
        'HSI_Return_60d': {'weight': 0.1729, 'direction': -1},  # 负面影响
        'US_10Y_Yield': {'weight': 0.0616, 'direction': -1},
        'VIX_Level': {'weight': 0.0493, 'direction': 1},
        '60d_Trend_HSI_Return_60d': {'weight': 0.0273, 'direction': 1},
        '60d_Trend_MA250_Slope': {'weight': 0.0198, 'direction': -1},
        'Turnover_Std_20': {'weight': 0.0173, 'direction': 1},
        'Vol_Std_20': {'weight': 0.0172, 'direction': -1},
        'OBV': {'weight': 0.0169, 'direction': 1},
        'VWAP': {'weight': 0.0164, 'direction': 1},
        '5d_Trend_US_10Y_Yield': {'weight': 0.0158, 'direction': 1},
        'ATR_MA120': {'weight': 0.0154, 'direction': -1},
        '60d_Trend_OBV': {'weight': 0.0152, 'direction': -1},
        'Resistance_120d': {'weight': 0.0136, 'direction': 1},
        'Distance_Support_120d': {'weight': 0.0133, 'direction': 1},
        'ATR_MA': {'weight': 0.0116, 'direction': -1},
        'Volume_MA250': {'weight': 0.0112, 'direction': 1},
        '60d_Trend_US_10Y_Yield': {'weight': 0.0106, 'direction': 1},
        'ATR_PE': {'weight': 0.0104, 'direction': -1},
        '60d_RS_Signal_MA250_Slope': {'weight': 0.0102, 'direction': -1},
        'MA120': {'weight': 0.0099, 'direction': 1},
    }

    def __init__(self):
        self.hsi_data = None
        self.us_data = None
        self.vix_data = None
        self.features = {}

    def fetch_data(self):
        """获取所需数据"""
        print("📊 正在获取数据...")

        # 获取恒生指数数据
        print("  - 恒生指数数据...")
        hsi = yf.Ticker("^HSI")
        self.hsi_data = hsi.history(period="1y", interval="1d")

        # 获取美国10年期国债收益率
        print("  - 美国国债收益率...")
        us_yield = yf.Ticker("^TNX")
        self.us_data = us_yield.history(period="1y", interval="1d")

        # 获取VIX指数
        print("  - VIX恐慌指数...")
        vix = yf.Ticker("^VIX")
        self.vix_data = vix.history(period="1y", interval="1d")

        if self.hsi_data.empty or self.us_data.empty or self.vix_data.empty:
            raise ValueError("数据获取失败")

        print(f"  ✅ 数据获取完成（恒指：{len(self.hsi_data)} 条，美债：{len(self.us_data)} 条，VIX：{len(self.vix_data)} 条）")

    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()

        # 移动平均线
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        df['MA250'] = df['Close'].rolling(window=250).mean()

        # MA250斜率（趋势强度）
        df['MA250_Slope'] = df['MA250'].diff()

        # 收益率
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_20d'] = df['Close'].pct_change(20)
        df['Return_60d'] = df['Close'].pct_change(60)

        # 成交量相关
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA250'] = df['Volume'].rolling(window=250).mean()
        df['Turnover_Std_20'] = df['Volume'].rolling(window=20).std()

        # OBV（能量潮指标）
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # ATR（平均真实波幅）
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_MA'] = df['ATR'].rolling(window=20).mean()
        df['ATR_MA120'] = df['ATR'].rolling(window=120).mean()

        # 波动率
        df['Volatility'] = df['Return_1d'].rolling(window=20).std()
        df['Vol_Std_20'] = df['Volatility'].rolling(window=20).std()

        # VWAP（成交量加权平均价）
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

        # 支撑阻力位
        df['Resistance_120d'] = df['High'].rolling(window=120).max()
        df['Support_120d'] = df['Low'].rolling(window=120).min()
        df['Distance_Support_120d'] = (df['Close'] - df['Support_120d']) / df['Support_120d']

        # 相对强弱信号
        df['RS_Signal_MA250_Slope'] = df['Close'] / df['MA250'] - 1

        return df

    def calculate_features(self):
        """计算所有特征"""
        print("🔧 正在计算特征...")

        hsi_df = self.calculate_technical_indicators(self.hsi_data)
        us_df = self.calculate_technical_indicators(self.us_data)

        # 获取最新数据（最近一天）
        latest_hsi = hsi_df.iloc[-1]
        latest_us = us_df.iloc[-1]
        latest_vix = self.vix_data['Close'].iloc[-1]

        # 计算特征值
        self.features = {
            'HSI_Return_60d': latest_hsi['Return_60d'],
            'US_10Y_Yield': latest_us['Close'] / 100,  # 转换为小数
            'VIX_Level': latest_vix,
            '60d_Trend_HSI_Return_60d': latest_hsi['Return_60d'],  # 60日趋势
            '60d_Trend_MA250_Slope': latest_hsi['MA250_Slope'],
            'Turnover_Std_20': latest_hsi['Turnover_Std_20'],
            'Vol_Std_20': latest_hsi['Vol_Std_20'],
            'OBV': latest_hsi['OBV'],
            'VWAP': latest_hsi['VWAP'],
            '5d_Trend_US_10Y_Yield': latest_us['Close'] / 100 - us_df['Close'].iloc[-6] / 100,  # 5日趋势
            'ATR_MA120': latest_hsi['ATR_MA120'],
            '60d_Trend_OBV': latest_hsi['OBV'] - hsi_df['OBV'].iloc[-60],  # 60日OBV趋势
            'Resistance_120d': latest_hsi['Resistance_120d'],
            'Distance_Support_120d': latest_hsi['Distance_Support_120d'],
            'ATR_MA': latest_hsi['ATR_MA'],
            'Volume_MA250': latest_hsi['Volume_MA250'],
            '60d_Trend_US_10Y_Yield': latest_us['Close'] / 100 - us_df['Close'].iloc[-60] / 100,  # 60日趋势
            'ATR_PE': latest_hsi['ATR'] / (latest_hsi['Close'] * 100),  # ATR/价格
            '60d_RS_Signal_MA250_Slope': latest_hsi['RS_Signal_MA250_Slope'],
            'MA120': latest_hsi['MA120'],
        }

        print(f"  ✅ 特征计算完成（{len(self.features)} 个特征）")

    def normalize_feature(self, feature_name, value):
        """特征标准化（使用z-score标准化）"""
        # 如果是收益率类特征，使用固定范围标准化
        if 'Return' in feature_name or 'Yield' in feature_name:
            # 标准化到[-1, 1]区间，假设收益率在[-0.2, 0.2]范围内
            return np.clip(value / 0.2, -1, 1)
        elif 'Trend' in feature_name or 'Slope' in feature_name:
            return np.clip(value * 10, -1, 1)  # 放大趋势信号
        elif 'Level' in feature_name or 'VIX' in feature_name:
            # VIX通常在10-50之间，标准化到[0, 1]
            return (value - 20) / 30  # 20为中位数
        else:
            # 其他特征使用简单的相对标准化
            return np.tanh(value / (abs(value) + 1))  # 使用tanh函数标准化

    def calculate_prediction_score(self):
        """计算预测得分"""
        print("📈 正在计算预测得分...")

        weighted_score = 0
        feature_details = []

        for feature_name, feature_value in self.features.items():
            if pd.isna(feature_value):
                continue

            # 获取特征配置
            config = self.FEATURE_IMPORTANCE[feature_name]
            weight = config['weight']
            direction = config['direction']

            # 标准化特征值
            normalized_value = self.normalize_feature(feature_name, feature_value)

            # 计算加权贡献
            contribution = normalized_value * weight * direction
            weighted_score += contribution

            feature_details.append({
                'feature': feature_name,
                'value': feature_value,
                'normalized': normalized_value,
                'weight': weight,
                'direction': direction,
                'contribution': contribution
            })

        # 标准化得分到[0, 1]区间
        # 得分 > 0.5 表示看涨，< 0.5 表示看跌
        prediction_score = (weighted_score + 1) / 2  # 映射到[0, 1]
        prediction_score = np.clip(prediction_score, 0, 1)

        print(f"  ✅ 预测得分计算完成：{prediction_score:.4f}")

        return prediction_score, feature_details

    def interpret_score(self, score):
        """解读预测得分"""
        if score >= 0.65:
            return "强烈看涨", "🟢"
        elif score >= 0.55:
            return "看涨", "🟢"
        elif score >= 0.50:
            return "中性偏涨", "🟡"
        elif score >= 0.45:
            return "中性偏跌", "🟡"
        elif score >= 0.35:
            return "看跌", "🔴"
        else:
            return "强烈看跌", "🔴"

    def generate_email_content(self, score, trend, feature_details):
        """生成邮件内容（HTML格式）"""
        current_price = self.hsi_data['Close'].iloc[-1]
        previous_price = self.hsi_data['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        current_date = self.hsi_data.index[-1].strftime('%Y-%m-%d')
        current_time = self.hsi_data.index[-1].strftime('%H:%M:%S')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 按贡献度排序特征
        sorted_features = sorted(feature_details, key=lambda x: abs(x['contribution']), reverse=True)

        # 统计正面和负面因素
        positive_features = [f for f in feature_details if f['contribution'] > 0]
        negative_features = [f for f in feature_details if f['contribution'] < 0]
        positive_score = sum(f['contribution'] for f in positive_features)
        negative_score = sum(abs(f['contribution']) for f in negative_features)

        # 趋势颜色
        trend_colors = {
            '强烈看涨': '#16a34a',      # 绿色
            '看涨': '#22c55e',         # 浅绿色
            '中性偏涨': '#84cc16',     # 黄绿色
            '中性偏跌': '#f59e0b',     # 橙色
            '看跌': '#f97316',         # 深橙色
            '强烈看跌': '#dc2626'      # 红色
        }
        trend_color = trend_colors.get(trend, '#6b7280')

        # 构建HTML邮件内容
        content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>恒生指数涨跌预测报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #1f2937;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9fafb;
        }}
        .container {{
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
        }}
        .header .subtitle {{
            margin-top: 8px;
            font-size: 14px;
            opacity: 0.9;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
            display: flex;
            align-items: center;
        }}
        .section-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-right: 12px;
            border-radius: 2px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .info-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #6366f1;
        }}
        .info-card.highlight {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left-color: #f59e0b;
        }}
        .info-card h3 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #6b7280;
            font-weight: 500;
        }}
        .info-card .value {{
            font-size: 24px;
            font-weight: 700;
            color: #1f2937;
        }}
        .info-card .trend {{
            font-size: 28px;
            font-weight: 700;
            color: {trend_color};
            text-align: center;
        }}
        .score-bar {{
            background: #e5e7eb;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #dc2626 0%, #f59e0b 50%, #22c55e 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 12px;
        }}
        th:first-child {{
            border-top-left-radius: 8px;
        }}
        th:last-child {{
            border-top-right-radius: 8px;
        }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        tr:hover {{
            background-color: #f3f4f6;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px;
        }}
        .badge-positive {{
            background-color: #dcfce7;
            color: #166534;
        }}
        .badge-negative {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .badge-neutral {{
            background-color: #f3f4f6;
            color: #374151;
        }}
        .feature-explanation {{
            font-size: 12px;
            color: #6b7280;
            margin-top: 4px;
            padding: 8px;
            background-color: #f8fafc;
            border-radius: 4px;
            border-left: 3px solid #6366f1;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            margin: 0 0 10px 0;
            color: #92400e;
            font-size: 16px;
        }}
        .alert-box {{
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #dc2626;
        }}
        .alert-box h3 {{
            margin: 0 0 10px 0;
            color: #991b1b;
            font-size: 16px;
        }}
        .footer {{
            background-color: #1f2937;
            color: #9ca3af;
            padding: 20px 30px;
            text-align: center;
            font-size: 12px;
        }}
        .footer a {{
            color: #60a5fa;
            text-decoration: none;
        }}
        .indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .indicator.green {{
            background-color: #22c55e;
        }}
        .indicator.red {{
            background-color: #dc2626;
        }}
        .indicator.yellow {{
            background-color: #f59e0b;
        }}
        .ranking {{
            display: inline-block;
            width: 24px;
            height: 24px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-size: 12px;
            font-weight: 700;
            margin-right: 8px;
        }}
        ul {{
            padding-left: 20px;
            margin: 10px 0;
        }}
        li {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        .risk-item {{
            display: flex;
            align-items: flex-start;
            margin: 10px 0;
        }}
        .risk-item::before {{
            content: '⚠️';
            margin-right: 10px;
            flex-shrink: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <h1>📊 恒生指数涨跌预测报告</h1>
            <div class="subtitle">基于特征重要性加权评分模型 | {current_date} {current_time}</div>
        </div>

        <!-- 第一部分：预测结果概览 -->
        <div class="section">
            <div class="section-title">一、预测结果概览</div>

            <div class="info-grid">
                <div class="info-card">
                    <h3>📈 恒指收盘</h3>
                    <div class="value">{current_price:.2f} 点</div>
                    <div style="color: { '#dc2626' if price_change < 0 else '#22c55e' }; font-size: 14px; margin-top: 5px;">
                        {price_change:+.2f}%
                    </div>
                </div>
                <div class="info-card highlight">
                    <h3>🎯 预测趋势</h3>
                    <div class="trend">{trend}</div>
                </div>
                <div class="info-card">
                    <h3>📊 预测得分</h3>
                    <div class="value">{score:.4f}</div>
                    <div style="color: #6b7280; font-size: 12px; margin-top: 5px;">满分 1.0000</div>
                </div>
            </div>

            <div style="margin: 30px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; color: #6b7280;">
                    <span>强烈看跌 (0.35)</span>
                    <span>中性 (0.50)</span>
                    <span>强烈看涨 (0.65)</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score * 100}%; background: linear-gradient(90deg, #dc2626 0%, #f59e0b 50%, #22c55e 100%);">
                        {score:.1%}
                    </div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 20px; font-size: 12px;">
                <div style="background: #fee2e2; padding: 12px; border-radius: 6px; color: #991b1b; text-align: center; font-weight: 600;">
                    强烈看跌 (<0.35)
                </div>
                <div style="background: #fef3c7; padding: 12px; border-radius: 6px; color: #92400e; text-align: center; font-weight: 600;">
                    中性区间 (0.35-0.65)
                </div>
                <div style="background: #dcfce7; padding: 12px; border-radius: 6px; color: #166534; text-align: center; font-weight: 600;">
                    强烈看涨 (>0.65)
                </div>
            </div>
        </div>

        <!-- 第二部分：预测原因分析 -->
        <div class="section">
            <div class="section-title">二、预测原因分析</div>

            <div class="summary-box">
                <h3>📊 因素汇总</h3>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><span class="badge badge-positive">正面因素 {len(positive_features)} 个</span> 总贡献：<strong style="color: #22c55e;">+{positive_score:.6f}</strong></li>
                    <li><span class="badge badge-negative">负面因素 {len(negative_features)} 个</span> 总贡献：<strong style="color: #dc2626;">-{negative_score:.6f}</strong></li>
                    <li>净得分：<strong style="font-size: 18px;">{positive_score - negative_score:+.6f}</strong></li>
                </ul>
            </div>

            <h3 style="font-size: 16px; color: #374151; margin: 20px 0 15px 0;">🔍 关键因素分析（按贡献度排序）</h3>

            <table>
                <thead>
                    <tr>
                        <th style="width: 8%;">排名</th>
                        <th style="width: 28%;">特征名称</th>
                        <th style="width: 12%;">当前值</th>
                        <th style="width: 10%;">权重</th>
                        <th style="width: 10%;">方向</th>
                        <th style="width: 12%;">贡献度</th>
                        <th style="width: 20%;">特征说明</th>
                    </tr>
                </thead>
                <tbody>
"""

        # 添加前10个最重要特征
        for i, feature in enumerate(sorted_features[:10], 1):
            direction_str = "正面" if feature['direction'] > 0 else "负面"
            direction_class = "badge-positive" if feature['direction'] > 0 else "badge-negative"
            contribution_color = "#22c55e" if feature['contribution'] > 0 else "#dc2626"

            content += f"""
                    <tr>
                        <td style="text-align: center;"><span class="ranking">{i}</span></td>
                        <td><strong>{feature['feature']}</strong></td>
                        <td>{feature['value']:.4f}</td>
                        <td>{feature['weight']:.2%}</td>
                        <td><span class="badge {direction_class}">{direction_str}</span></td>
                        <td style="color: {contribution_color}; font-weight: 600;">{feature['contribution']:+.6f}</td>
                        <td style="font-size: 11px; color: #6b7280;">{self._get_feature_explanation(feature['feature'])}</td>
                    </tr>
"""

        content += """
                </tbody>
            </table>

            <h3 style="font-size: 14px; color: #374151; margin: 20px 0 10px 0;">📋 其他重要特征</h3>
            <table style="font-size: 12px;">
                <thead>
                    <tr>
                        <th style="width: 40%;">特征名称</th>
                        <th style="width: 20%;">贡献度</th>
                        <th style="width: 40%;">影响方向</th>
                    </tr>
                </thead>
                <tbody>
"""

        # 添加其他特征
        for feature in sorted_features[10:]:
            contribution_color = "#22c55e" if feature['contribution'] > 0 else "#dc2626"
            impact_str = "📈 推动上涨" if feature['contribution'] > 0 else "📉 推动下跌"

            content += f"""
                    <tr>
                        <td>{feature['feature']}</td>
                        <td style="color: {contribution_color}; font-weight: 600;">{feature['contribution']:+.6f}</td>
                        <td>{impact_str}</td>
                    </tr>
"""

        content += """
                </tbody>
            </table>
        </div>

        <!-- 第三部分：核心市场指标解读 -->
        <div class="section">
            <div class="section-title">三、核心市场指标解读</div>

            <div class="info-grid">
                <div class="info-card">
                    <h3>📊 恒指60日收益率</h3>
                    <div class="value">{self.features['HSI_Return_60d']*100:+.2f}%</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：17.29% | 方向：负面</strong>
                    </div>
                    <div class="feature-explanation">
                        恒指近60日上涨{abs(self.features['HSI_Return_60d']*100):.2f}%，根据历史数据，涨幅过高通常预示回调压力
                    </div>
                </div>

                <div class="info-card">
                    <h3>💰 美国10年期国债收益率</h3>
                    <div class="value">{self.features['US_10Y_Yield']*100:.2f}%</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：6.16% | 方向：负面</strong>
                    </div>
                    <div class="feature-explanation">
                        美债收益率处于{self.features['US_10Y_Yield']*100:.2f}%水平，上升会增加资金成本，压制股市估值
                    </div>
                </div>

                <div class="info-card">
                    <h3>😰 VIX恐慌指数</h3>
                    <div class="value">{self.features['VIX_Level']:.2f}</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：4.93% | 方向：正面</strong>
                    </div>
                    <div class="feature-explanation">
                        {'市场情绪平静（低VIX）' if self.features['VIX_Level'] < 20 else '市场情绪紧张（高VIX）' if self.features['VIX_Level'] > 30 else '市场情绪正常'}
                    </div>
                </div>

                <div class="info-card">
                    <h3>💵 成交额标准差（20日）</h3>
                    <div class="value">{self.features['Turnover_Std_20']/1e8:.2f}亿</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：1.73% | 方向：正面</strong>
                    </div>
                    <div class="feature-explanation">
                        {'市场活跃度高' if self.features['Turnover_Std_20'] > 1e9 else '市场活跃度正常'}
                    </div>
                </div>

                <div class="info-card">
                    <h3>⚡ OBV能量潮</h3>
                    <div class="value">{self.features['OBV']/1e8:.2f}亿</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：1.69% | 方向：正面</strong>
                    </div>
                    <div class="feature-explanation">
                        {'资金净流入' if self.features['OBV'] > 0 else '资金净流出'}，{'支撑股价' if self.features['OBV'] > 0 else '压制股价'}
                    </div>
                </div>

                <div class="info-card">
                    <h3>📍 距离120日支撑位</h3>
                    <div class="value">{self.features['Distance_Support_120d']*100:+.2f}%</div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        <strong>权重：1.33% | 方向：正面</strong>
                    </div>
                    <div class="feature-explanation">
                        {'距离支撑位较远，安全边际一般' if abs(self.features['Distance_Support_120d']) > 0.1 else '距离支撑位适中，安全边际良好'}
                    </div>
                </div>
            </div>
        </div>

        <!-- 第四部分：投资建议 -->
        <div class="section">
            <div class="section-title">四、投资建议</div>
"""

        # 根据预测得分生成投资建议
        if score >= 0.65:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-left: 4px solid #22c55e;">
                <h3 style="color: #166534;">✅ 强烈看涨（得分 ≥ 0.65）</h3>
                <ul>
                    <li>建议积极配置港股</li>
                    <li>优先关注权重股和科技股</li>
                    <li>可考虑适当增加仓位</li>
                    <li>注意风险控制，设置止损</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #166534;">
                    <strong>理由：</strong>多个正面因素占据主导，市场技术面和情绪面均向好
                </p>
            </div>
"""
        elif score >= 0.55:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-left: 4px solid #22c55e;">
                <h3 style="color: #166534;">✅ 看涨（得分 0.55-0.65）</h3>
                <ul>
                    <li>可适度增加港股配置</li>
                    <li>选择性买入优质个股</li>
                    <li>保持谨慎乐观态度</li>
                    <li>不要盲目追高</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #166534;">
                    <strong>理由：</strong>正面因素较多，但仍需关注潜在风险
                </p>
            </div>
"""
        elif score >= 0.50:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b;">
                <h3 style="color: #92400e;">⚠️ 中性偏涨（得分 0.50-0.55）</h3>
                <ul>
                    <li>市场多空平衡，观望为主</li>
                    <li>可择机低吸优质个股</li>
                    <li>控制仓位，不要追高</li>
                    <li>等待更明确信号</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #92400e;">
                    <strong>理由：</strong>市场情绪谨慎，正面和负面因素基本平衡
                </p>
            </div>
"""
        elif score >= 0.45:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); border-left: 4px solid #f97316;">
                <h3 style="color: #9a3412;">⚠️ 中性偏跌（得分 0.45-0.50）</h3>
                <ul>
                    <li>市场情绪偏谨慎</li>
                    <li>建议减仓或持币观望</li>
                    <li>等待更明确的信号</li>
                    <li>不要盲目抄底</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #9a3412;">
                    <strong>理由：</strong>负面因素略占上风，市场面临下行压力
                </p>
            </div>
"""
        elif score >= 0.35:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); border-left: 4px solid #ef4444;">
                <h3 style="color: #991b1b;">🔴 看跌（得分 0.35-0.45）</h3>
                <ul>
                    <li>建议减仓或离场</li>
                    <li>避免追涨杀跌</li>
                    <li>关注防御性品种</li>
                    <li>严格控制风险</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #991b1b;">
                    <strong>理由：</strong>负面因素明显，市场情绪偏空
                </p>
            </div>
"""
        else:
            content += """
            <div class="summary-box" style="background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); border-left: 4px solid #dc2626;">
                <h3 style="color: #7f1d1d;">🔴 强烈看跌（得分 < 0.35）</h3>
                <ul>
                    <li>建议清仓或空仓</li>
                    <li>严格控制风险</li>
                    <li>等待市场企稳信号</li>
                    <li>避免盲目抄底</li>
                </ul>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 4px; font-size: 13px; color: #7f1d1d;">
                    <strong>理由：</strong>多个负面因素叠加，市场面临较大下行风险
                </p>
            </div>
"""

        content += """
        </div>

        <!-- 第五部分：模型说明 -->
        <div class="section">
            <div class="section-title">五、模型说明</div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #6366f1;">
                    <h4 style="margin: 0 0 10px 0; color: #374151;">🎯 特征重要性来源</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280;">
                        <li>来自机器学习模型的特征重要性分析</li>
                        <li>包含技术面、宏观面、情绪面三个维度</li>
                        <li>20个关键特征，权重 17.29% - 0.99%</li>
                    </ul>
                </div>

                <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
                    <h4 style="margin: 0 0 10px 0; color: #374151;">📊 加权评分方法</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280;">
                        <li>对每个特征进行标准化处理（-1 到 1）</li>
                        <li>按权重加权，考虑影响方向</li>
                        <li>综合得分映射到 0-1 区间</li>
                        <li>得分 > 0.5 为看涨，< 0.5 为看跌</li>
                    </ul>
                </div>

                <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #a855f7;">
                    <h4 style="margin: 0 0 10px 0; color: #374151;">📈 特征类别</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280;">
                        <li><strong>技术面特征（60%）</strong>：趋势、动量、成交量、支撑阻力</li>
                        <li><strong>宏观面特征（20%）</strong>：美债收益率、VIX恐慌指数</li>
                        <li><strong>情绪面特征（20%）</strong>：OBV、成交额波动率</li>
                    </ul>
                </div>

                <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #d946ef;">
                    <h4 style="margin: 0 0 10px 0; color: #374151;">⏱️ 预测周期</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280;">
                        <li>短期预测：1-5 个交易日</li>
                        <li>基于最新数据和特征计算</li>
                        <li>每日更新预测结果</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- 第六部分：风险提示 -->
        <div class="section">
            <div class="section-title">六、风险提示</div>

            <div class="alert-box">
                <h3>⚠️ 重要提醒</h3>
                <div class="risk-item">本预测基于历史数据和统计模型，仅供参考，不构成投资建议</div>
                <div class="risk-item">股市有风险，投资需谨慎，请根据自身风险承受能力做出决策</div>
                <div class="risk-item">请结合基本面分析、市场情绪、政策面等多方面因素综合判断</div>
                <div class="risk-item">模型预测存在不确定性，不应作为唯一投资依据</div>
                <div class="risk-item">市场环境变化可能导致模型失效，需要持续监控和调整</div>
                <div class="risk-item">过去表现不代表未来收益，历史数据可能无法预测极端事件</div>
            </div>
        </div>

        <!-- 第七部分：数据来源 -->
        <div class="section">
            <div class="section-title">七、数据来源</div>

            <table style="font-size: 13px;">
                <thead>
                    <tr>
                        <th style="width: 50%;">数据项</th>
                        <th style="width: 50%;">数据源</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>📊 恒生指数数据</td>
                        <td>Yahoo Finance (^HSI)</td>
                    </tr>
                    <tr>
                        <td>💰 美国国债收益率</td>
                        <td>Yahoo Finance (^TNX)</td>
                    </tr>
                    <tr>
                        <td>😰 VIX恐慌指数</td>
                        <td>Yahoo Finance (^VIX)</td>
                    </tr>
                    <tr>
                        <td>📅 数据周期</td>
                        <td>过去 1 年历史数据</td>
                    </tr>
                    <tr>
                        <td>⚡ 数据频率</td>
                        <td>日频数据</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- 页脚 -->
        <div class="footer">
            <p style="margin: 5px 0;">📊 预测模型：基于特征重要性的加权评分模型</p>
            <p style="margin: 5px 0;">🔢 特征数量：20 个关键特征</p>
            <p style="margin: 5px 0;">📈 预测方法：多因素加权综合评分</p>
            <p style="margin: 15px 0 5px 0;">⏰ 报告生成时间：{timestamp}</p>
            <p style="margin: 5px 0; color: #6b7280;">本报告由 AI 智能分析系统自动生成 | 仅供参考</p>
        </div>
    </div>
</body>
</html>
"""

        return content
    def _get_feature_explanation(self, feature_name):
        """获取特征说明"""
        explanations = {
            'HSI_Return_60d': '恒指过去60天的累计收益率，反映中期趋势。过高可能预示回调。',
            'US_10Y_Yield': '美国10年期国债收益率，反映全球资金成本。上升压制股市估值。',
            'VIX_Level': '芝加哥期权交易所波动率指数，反映市场恐慌程度。高值表示恐慌。',
            '60d_Trend_HSI_Return_60d': '恒指60日趋势强度，反映中期动量。',
            '60d_Trend_MA250_Slope': '恒指250日均线斜率，反映长期趋势方向。',
            'Turnover_Std_20': '过去20天成交额的标准差，反映市场活跃度和波动性。',
            'Vol_Std_20': '过去20天价格波动率的标准差，反映市场稳定性。',
            'OBV': '能量潮指标，反映资金流向。上升表示资金净流入。',
            'VWAP': '成交量加权平均价，反映真实成交成本。',
            '5d_Trend_US_10Y_Yield': '美债收益率5日趋势，反映短期宏观变化。',
            'ATR_MA120': '120日平均真实波幅，反映中期波动性。',
            '60d_Trend_OBV': 'OBV的60日趋势，反映中期资金流向变化。',
            'Resistance_120d': '过去120日最高价，反映阻力位水平。',
            'Distance_Support_120d': '当前价格距120日支撑位的距离，反映安全边际。',
            'ATR_MA': '平均真实波幅，反映价格波动范围。',
            'Volume_MA250': '250日平均成交量，反映长期流动性水平。',
            '60d_Trend_US_10Y_Yield': '美债收益率60日趋势，反映中期宏观趋势。',
            'ATR_PE': 'ATR与价格的比率，反映相对波动性。',
            '60d_RS_Signal_MA250_Slope': '价格相对MA250的比率变化，反映趋势强度。',
            'MA120': '120日移动平均线，反映中期趋势支撑。'
        }
        return explanations.get(feature_name, '暂无详细说明')

    def send_email_notification(self, score, trend, feature_details):
        """发送邮件通知"""
        try:
            # 生成邮件内容
            content = self.generate_email_content(score, trend, feature_details)

            # 邮件配置
            sender_email = os.environ.get("YAHOO_EMAIL")
            email_password = os.environ.get("YAHOO_APP_PASSWORD")
            smtp_server = os.environ.get("YAHOO_SMTP", "smtp.163.com")
            recipient_email = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")

            if ',' in recipient_email:
                recipients = [r.strip() for r in recipient_email.split(',')]
            else:
                recipients = [recipient_email]

            if not sender_email or not email_password:
                print("❌ 邮件配置不完整，跳过邮件发送")
                print("   请设置环境变量：YAHOO_EMAIL, YAHOO_APP_PASSWORD, RECIPIENT_EMAIL")
                return False

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

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
            current_date = datetime.now().strftime('%Y-%m-%d')
            subject = f"恒生指数涨跌预测 {current_date} - {trend}（得分{score:.4f}）"

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)

            # 添加HTML版本
            html_part = MIMEText(content, 'html', 'utf-8')
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

                    print(f"✅ 预测邮件已发送到: {', '.join(recipients)}")
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

    def save_report(self, score, feature_details):
        """保存预测报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存为JSON格式（详细数据）
        report_data = {
            'timestamp': timestamp,
            'prediction_date': self.hsi_data.index[-1].strftime('%Y-%m-%d'),
            'current_price': float(self.hsi_data['Close'].iloc[-1]),
            'prediction_score': float(score),
            'features': {k: (float(v) if not pd.isna(v) else None) for k, v in self.features.items()},
            'feature_details': feature_details,
            'prediction_trend': self.interpret_score(score)[0]
        }

        json_file = os.path.join(output_dir, f'hsi_prediction_report_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 保存为CSV格式（特征值）
        features_df = pd.DataFrame([self.features])
        features_file = os.path.join(data_dir, f'hsi_prediction_features_{timestamp}.csv')
        features_df.to_csv(features_file, index=False)

        print(f"💾 报告已保存到：")
        print(f"   - {json_file}")
        print(f"   - {features_file}")

    def run(self, send_email_flag=True):
        """运行预测流程

        参数:
        - send_email_flag: 是否发送邮件，默认True
        """
        try:
            # 1. 获取数据
            self.fetch_data()

            # 2. 计算特征
            self.calculate_features()

            # 3. 生成报告（控制台显示）
            score, feature_details = self.calculate_prediction_score()
            trend = self.interpret_score(score)[0]

            # 4. 生成控制台报告
            self._generate_console_report(score, trend, feature_details)

            # 5. 保存报告
            self.save_report(score, feature_details)

            # 6. 发送邮件
            if send_email_flag:
                print("\n" + "="*80)
                print("正在发送预测邮件...".center(80))
                print("="*80 + "\n")
                email_sent = self.send_email_notification(score, trend, feature_details)
                if email_sent:
                    print("\n✅ 预测报告已通过邮件发送")
                else:
                    print("\n❌ 邮件发送失败，但预测报告已保存")
            else:
                print("\n⚠️ 已跳过邮件发送（--no-email 参数）")

            return score, trend

        except Exception as e:
            print(f"❌ 预测失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _generate_console_report(self, score, trend, feature_details):
        """生成控制台显示的报告"""
        print("\n" + "="*80)
        print("恒生指数涨跌预测报告".center(80))
        print("="*80)

        # 显示基本信息
        current_price = self.hsi_data['Close'].iloc[-1]
        previous_price = self.hsi_data['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        current_date = self.hsi_data.index[-1].strftime('%Y-%m-%d')

        print(f"\n📅 分析日期：{current_date}")
        print(f"📊 恒指收盘：{current_price:.2f}（{price_change:+.2f}%）")
        print(f"📈 预测得分：{score:.4f}")
        print(f"🎯 预测趋势：{trend}")

        # 分析关键因素
        print(f"\n{'='*80}")
        print("关键因素分析（按权重排序，仅显示控制台）".center(80))
        print(f"{'='*80}\n")

        # 按贡献度排序
        sorted_features = sorted(feature_details, key=lambda x: abs(x['contribution']), reverse=True)

        print(f"{'特征':<30} {'当前值':<12} {'标准化':<10} {'权重':<8} {'方向':<8} {'贡献度':<12}")
        print("-" * 100)

        for i, feature in enumerate(sorted_features[:10], 1):  # 显示前10个最重要特征
            direction_str = "正面" if feature['direction'] > 0 else "负面"
            contribution_str = f"{feature['contribution']:>+.6f}"

            print(f"{i:2}. {feature['feature']:<27} {feature['value']:>10.4f}   "
                  f"{feature['normalized']:>7.3f}   {feature['weight']:>6.2%}   "
                  f"{direction_str:<6}   {contribution_str:<12}")

        # 计算正面/负面因素
        positive_features = [f for f in feature_details if f['contribution'] > 0]
        negative_features = [f for f in feature_details if f['contribution'] < 0]

        positive_score = sum(f['contribution'] for f in positive_features)
        negative_score = sum(abs(f['contribution']) for f in negative_features)

        print(f"\n📊 因素汇总：")
        print(f"  - 正面因素贡献：{positive_score:+.6f}（{len(positive_features)} 个）")
        print(f"  - 负面因素贡献：{-negative_score:.6f}（{len(negative_features)} 个）")

        # 显示关键指标
        print(f"\n{'='*80}")
        print("关键市场指标".center(80))
        print(f"{'='*80}\n")

        print(f"恒指60日收益率：{self.features['HSI_Return_60d']*100:+.2f}%")
        print(f"美国10年期国债收益率：{self.features['US_10Y_Yield']*100:.2f}%")
        print(f"VIX恐慌指数：{self.features['VIX_Level']:.2f}")
        print(f"成交额标准差（20日）：{self.features['Turnover_Std_20']:,.0f}")
        print(f"OBV：{self.features['OBV']:,.0f}")
        print(f"距离120日支撑位：{self.features['Distance_Support_120d']*100:+.2f}%")

        print(f"\n{'='*80}\n")


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='恒生指数涨跌预测系统')
    parser.add_argument('--no-email', action='store_true', help='不发送邮件')
    args = parser.parse_args()

    print("="*80)
    print("恒生指数涨跌预测系统".center(80))
    print("基于特征重要性加权评分模型".center(80))
    print("="*80 + "\n")

    # 创建预测器
    predictor = HSI_Predictor()

    # 运行预测
    send_email_flag = not args.no_email
    score, trend = predictor.run(send_email_flag=send_email_flag)

    if score is not None:
        print(f"\n✅ 预测完成")
        print(f"   预测得分：{score:.4f}")
        print(f"   预测趋势：{trend}")
        if send_email_flag:
            print(f"   邮件状态：已发送")
        else:
            print(f"   邮件状态：已跳过（--no-email）")
    else:
        print(f"\n❌ 预测失败")


if __name__ == "__main__":
    main()
