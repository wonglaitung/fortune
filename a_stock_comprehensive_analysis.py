#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议并发送邮件

⚠️ 运行时机：建议在A股收市后（15:00 CST）运行

版本：v3.0 (2026-07-18)
- 新增三周期预测表格（参考港股邮件格式）
- 新增市场情绪分析
- 增强ML预测表格（板块、涨跌幅、风险建议）
- 新增北向资金详细分析
- 新增涨跌停状态分析
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
    is_core_holding,
)

# 导入数据服务
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.northbound_data import NorthboundDataService

# 导入综合买卖建议生成器
from a_stock_recommendation_generator import AStockRecommendationGenerator

# 股票名称映射
STOCK_NAMES = A_STOCK_WATCHLIST
STOCK_LIST = list(A_STOCK_WATCHLIST.keys())

# 三周期模式定义（参考港股）
THREE_HORIZON_PATTERNS = {
    'AAA': {'name': '强势上涨', 'action': '积极买入', 'color': '#16a34a'},
    'AAB': {'name': '短期调整', 'action': '逢低买入', 'color': '#16a34a'},
    'ABA': {'name': '震荡上行', 'action': '适度买入', 'color': '#ea580c'},
    'ABB': {'name': '趋势转弱', 'action': '观望', 'color': '#ea580c'},
    'BAA': {'name': '底部反弹', 'action': '试探买入', 'color': '#ea580c'},
    'BAB': {'name': '震荡下行', 'action': '谨慎观望', 'color': '#dc2626'},
    'BBA': {'name': '下跌反弹', 'action': '观望', 'color': '#dc2626'},
    'BBB': {'name': '持续下跌', 'action': '回避', 'color': '#dc2626'},
}


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
    # 查找最新的预测文件（A股专用路径）
    import glob
    files = glob.glob(f'data/a_stock_models/ml_predictions_{horizon}d.csv')

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
            direction = '上涨' if pred_proba >= 0.5 else '下跌'
            confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba

            report += f"- **{name} ({code})**: {direction} (置信度: {confidence:.1%})\n"

    # 添加大模型建议
    if llm_content:
        report += f"""
---

## 四、AI 分析建议

{llm_content}
"""

    report += """
---

## 五、风险提示

1. **涨跌停限制**：创业板/科创板 20%，主板 10%，ST股 5%
2. **北向资金**：关注外资流向变化
3. **市场情绪**：上证指数跌破MA20时需谨慎

---

*本报告仅供参考，不构成投资建议*
"""

    return report


def generate_three_horizon_predictions(stock_list):
    """
    生成三周期预测（1天、5天、20天）

    Args:
        stock_list: 股票代码列表

    Returns:
        dict: {股票代码: {predictions: {1: {...}, 5: {...}, 20: {...}}, pattern: 'AAA', ...}}
    """
    from a_stock_ml_model import AStockTradingModel
    import pickle

    three_horizon_results = {}

    # 加载三个周期的模型
    for horizon in [1, 5, 20]:
        model_path = f'data/a_stock_models/trading_model_catboost_{horizon}d.pkl'
        if not os.path.exists(model_path):
            print(f"⚠️ 模型文件不存在: {model_path}")
            continue

        try:
            # 加载模型
            model = AStockTradingModel(horizon=horizon)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            model.catboost_model = model_data['catboost_model']
            model.feature_columns = model_data['feature_columns']
            model.categorical_encoders = model_data.get('categorical_encoders', {})
            model.community_ids = model_data.get('community_ids', [0, 1, 2, 3, 4, 5, 6])

            print(f"  ✅ 加载 {horizon}d 模型")

            # 预测每只股票
            for code in stock_list:
                try:
                    result = model.predict(code=code, mode='production')
                    if result:
                        if code not in three_horizon_results:
                            three_horizon_results[code] = {'predictions': {}}

                        prob = result.get('probability', 0.5)
                        direction = '↑' if prob >= 0.5 else '↓'

                        three_horizon_results[code]['predictions'][horizon] = {
                            'direction': direction,
                            'probability': prob,
                            'current_price': result.get('current_price'),
                            'date': result.get('date'),
                        }
                except Exception as e:
                    print(f"  ⚠️ 预测 {code} {horizon}d 失败: {e}")

        except Exception as e:
            print(f"  ⚠️ 加载模型 {model_path} 失败: {e}")

    # 计算三周期模式
    for code, data in three_horizon_results.items():
        preds = data.get('predictions', {})
        if len(preds) == 3:
            # 模式计算：A=上涨(概率>=0.5)，B=下跌(概率<0.5)
            pattern = ''
            for h in [1, 5, 20]:
                if h in preds:
                    pattern += 'A' if preds[h]['probability'] >= 0.5 else 'B'
                else:
                    pattern += 'B'

            data['pattern'] = pattern
            pattern_info = THREE_HORIZON_PATTERNS.get(pattern, {'name': '未知', 'action': '观望'})
            data['pattern_info'] = pattern_info

            # 计算历史胜率（简化版，基于概率）
            probs = [preds[h]['probability'] for h in [1, 5, 20] if h in preds]
            data['avg_probability'] = np.mean(probs) if probs else 0.5
            data['win_rate'] = data['avg_probability'] * 100  # 简化估算

    return three_horizon_results


def get_market_sentiment(stock_analyses):
    """
    计算市场情绪（上涨比例）

    Args:
        stock_analyses: 股票分析结果

    Returns:
        dict: {layer: 'normal'/'bear'/'extreme_bear', up_ratio: 0.5, dynamic_threshold: 0.5}
    """
    if not stock_analyses:
        return {'layer': 'normal', 'up_ratio': 0.5, 'dynamic_threshold': 0.5}

    # 计算上涨比例
    up_count = 0
    total = 0
    for code, analysis in stock_analyses.items():
        change = analysis.get('change_percent', 0)
        if change is not None:
            total += 1
            if change > 0:
                up_count += 1

    up_ratio = up_count / total if total > 0 else 0.5

    # 计算情绪层级
    if up_ratio < 0.20:
        layer = 'extreme_bear'
        dynamic_threshold = 1.0  # 暂停交易
    elif up_ratio < 0.30:
        layer = 'bear'
        dynamic_threshold = 0.70  # 高置信
    elif up_ratio < 0.40:
        layer = 'weak'
        dynamic_threshold = 0.65  # 谨慎
    else:
        layer = 'normal'
        dynamic_threshold = 0.50  # 标准

    return {
        'layer': layer,
        'up_ratio': up_ratio,
        'dynamic_threshold': dynamic_threshold,
    }


def get_northbound_trend():
    """
    获取北向资金趋势

    Returns:
        dict: 北向资金详细数据
    """
    result = {
        'net_buy': 0,
        'sh_net_buy': 0,
        'sz_net_buy': 0,
        'net_buy_5d_sum': 0,
        'net_buy_20d_sum': 0,
        'consecutive_inflow': 0,
    }

    try:
        service = NorthboundDataService()

        # 获取最新数据
        latest = service.get_latest()
        if latest:
            result['net_buy'] = latest.get('net_buy', 0)
            result['sh_net_buy'] = latest.get('sh_net_buy', 0)
            result['sz_net_buy'] = latest.get('sz_net_buy', 0)

        # 获取历史数据计算趋势
        history = service.fetch_history()
        if history is not None and not history.empty:
            # 累积流入（5日、20日）
            if 'net_buy' in history.columns:
                result['net_buy_5d_sum'] = history['net_buy'].tail(5).sum()
                result['net_buy_20d_sum'] = history['net_buy'].tail(20).sum()

                # 连续流入天数
                net_buy_series = history['net_buy'].tail(20)
                consecutive = 0
                for val in net_buy_series.iloc[::-1]:
                    if val > 0:
                        consecutive += 1
                    else:
                        break
                result['consecutive_inflow'] = consecutive

    except Exception as e:
        print(f"⚠️ 获取北向资金趋势失败: {e}")

    return result


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


def _format_recommendations_section(recommendations):
    """
    格式化综合买卖建议HTML部分

    Args:
        recommendations: 综合买卖建议字典

    Returns:
        str: HTML格式的建议
    """
    html = ""

    # 强烈买入
    if recommendations.get('strong_buy'):
        html += """
    <h3>🟢 强烈买入信号</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议仓位</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
        for rec in recommendations['strong_buy']:
            change_class = 'positive' if rec.get('change_percent', 0) >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{change_class}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #16a34a; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['position_pct']}%</td>
            <td>{rec['stop_loss']:.2f}</td>
            <td>{rec['target_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
        html += "    </table>\n"

    # 买入
    if recommendations.get('buy'):
        html += """
    <h3>🟡 买入信号</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议仓位</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
        for rec in recommendations['buy']:
            change_class = 'positive' if rec.get('change_percent', 0) >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{change_class}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #ea580c; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['position_pct']}%</td>
            <td>{rec['stop_loss']:.2f}</td>
            <td>{rec['target_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
        html += "    </table>\n"

    # 持有/观望
    if recommendations.get('hold'):
        html += """
    <h3>⚪ 持有/观望</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议</th><th>推荐理由</th>
        </tr>
"""
        for rec in recommendations['hold']:
            change_class = 'positive' if rec.get('change_percent', 0) >= 0 else 'negative'
            html += f"""        <tr>
            <td>{rec['stock_name']}</td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{change_class}">{rec['change_percent']:+.2f}%</td>
            <td>{rec['probability_20d']:.2f}</td>
            <td>观望</td>
            <td>{rec['reason']}</td>
        </tr>
"""
        html += "    </table>\n"

    # 卖出
    if recommendations.get('sell'):
        html += """
    <h3>🔴 卖出信号</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议卖出价</th><th>推荐理由</th>
        </tr>
"""
        for rec in recommendations['sell']:
            change_class = 'positive' if rec.get('change_percent', 0) >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{change_class}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #dc2626; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['current_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
        html += "    </table>\n"

    # 风险控制建议
    risk = recommendations.get('risk_control', {})
    if risk:
        html += f"""
    <h3>⚠️ 风险控制建议</h3>
    <table>
        <tr><th>指标</th><th>建议</th></tr>
        <tr><td>当前市场风险</td><td>{risk.get('market_risk', '中')}</td></tr>
        <tr><td>建议总仓位</td><td>{risk.get('total_position', 0)}%</td></tr>
        <tr><td>止损策略</td><td>{risk.get('stop_loss_strategy', '')}</td></tr>
        <tr><td>仓位策略</td><td>{risk.get('position_strategy', '')}</td></tr>
    </table>
"""

    return html


def generate_html_email(llm_content, ml_predictions_20d, stock_analyses, market_data,
                          three_horizon_results=None, market_sentiment=None, northbound_trend=None,
                          recommendations=None):
    """
    生成增强版HTML邮件（参考港股详细度）

    Args:
        llm_content: 大模型建议内容
        ml_predictions_20d: ML预测结果
        stock_analyses: 股票分析结果
        market_data: 市场数据
        three_horizon_results: 三周期预测结果（新增）
        market_sentiment: 市场情绪数据（新增）
        northbound_trend: 北向资金趋势（新增）
        recommendations: 综合买卖建议（新增）

    Returns:
        str: HTML格式邮件
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 市场涨跌颜色
    sh_change = market_data.get('sh_change', 0)
    sh_color = 'green' if sh_change >= 0 else 'red'

    # 北向资金颜色
    nb_buy = market_data.get('northbound_net_buy', 0)
    nb_color = 'green' if nb_buy >= 0 else 'red'

    # 市场情绪层级名称
    layer_names = {
        'extreme_bear': ('🔴 极端熊市 - 暂停交易', '#dc2626'),
        'bear': ('🟠 熊市 - 需概率≥0.70', '#ea580c'),
        'weak': ('🟡 弱震荡 - 需概率≥0.65', '#eab308'),
        'normal': ('🟢 正常市场', '#16a34a'),
    }
    sentiment_layer = market_sentiment.get('layer', 'normal') if market_sentiment else 'normal'
    sentiment_name, sentiment_color = layer_names.get(sentiment_layer, ('正常市场', '#16a34a'))
    up_ratio = market_sentiment.get('up_ratio', 0.5) if market_sentiment else 0.5
    dynamic_threshold = market_sentiment.get('dynamic_threshold', 0.5) if market_sentiment else 0.5

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #007bff; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .market-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .sentiment-box {{ background: {sentiment_color}15; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid {sentiment_color}; }}
        .stock-card {{ background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; }}
        .prediction-up {{ color: #16a34a; font-weight: bold; }}
        .prediction-down {{ color: #dc2626; font-weight: bold; }}
        .prediction-neutral {{ color: #ea580c; font-weight: bold; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
        .core-stock {{ background-color: #fff3cd; }}
        .pattern-AAA, .pattern-AAB {{ background-color: #dcfce7; }}
        .pattern-ABA, .pattern-ABB {{ background-color: #fef3c7; }}
        .pattern-BAA, .pattern-BAB {{ background-color: #fee2e2; }}
        .pattern-BBA, .pattern-BBB {{ background-color: #fecaca; }}
    </style>
</head>
<body>
    <h1>📊 A股综合分析报告</h1>
    <p>日期: {date_str} | 生成时间: {datetime.now().strftime('%H:%M:%S')}</p>

    <!-- 综合买卖建议（核心） -->
    <h2>📋 综合买卖建议</h2>
    <p style="color: #666; font-size: 12px;">基于大模型分析 + CatBoost预测 + 技术指标综合判断</p>
"""

    # 添加综合买卖建议
    if recommendations:
        html += _format_recommendations_section(recommendations)

    html += """

    <!-- 市场情绪分析 -->
    <div class="sentiment-box">
        <h2>📈 市场情绪分析</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>说明</th></tr>
            <tr>
                <td><strong>市场情绪</strong></td>
                <td style="color: {sentiment_color}; font-weight: bold;">{sentiment_name}</td>
                <td>{'暂停交易' if sentiment_layer == 'extreme_bear' else '正常交易' if sentiment_layer == 'normal' else '谨慎交易'}</td>
            </tr>
            <tr>
                <td>今日上涨比例</td>
                <td>{up_ratio:.1%}</td>
                <td>自选股上涨数量占比</td>
            </tr>
            <tr>
                <td>动态置信阈值</td>
                <td>{dynamic_threshold:.0%}</td>
                <td>买入信号需达到的概率</td>
            </tr>
        </table>
    </div>

    <!-- 市场概况 -->
    <div class="market-box">
        <h2>📊 市场概况</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>变化</th></tr>
            <tr>
                <td>上证指数</td>
                <td>{market_data.get('sh_close', 0):.2f}</td>
                <td class="{'positive' if sh_change >= 0 else 'negative'}">{sh_change:+.2f}%</td>
            </tr>
            <tr>
                <td>上证 MA20</td>
                <td>{market_data.get('sh_ma20', 0):.2f}</td>
                <td>{'📈 站上' if market_data.get('sh_close', 0) >= market_data.get('sh_ma20', 0) else '📉 跌破'} MA20</td>
            </tr>
        </table>

        <h3>💰 北向资金</h3>
        <table>
            <tr><th>指标</th><th>数值</th><th>趋势</th></tr>
            <tr>
                <td>今日净买入</td>
                <td class="{'positive' if nb_buy >= 0 else 'negative'}">{nb_buy:.2f} 亿</td>
                <td>{'流入' if nb_buy >= 0 else '流出'}</td>
            </tr>
            <tr>
                <td>沪股通</td>
                <td>{market_data.get('northbound_sh', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>深股通</td>
                <td>{market_data.get('northbound_sz', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
"""
    # 北向资金趋势
    if northbound_trend:
        html += f"""            <tr>
                <td>5日累积流入</td>
                <td class="{'positive' if northbound_trend.get('net_buy_5d_sum', 0) >= 0 else 'negative'}">{northbound_trend.get('net_buy_5d_sum', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>20日累积流入</td>
                <td class="{'positive' if northbound_trend.get('net_buy_20d_sum', 0) >= 0 else 'negative'}">{northbound_trend.get('net_buy_20d_sum', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>连续流入天数</td>
                <td>{northbound_trend.get('consecutive_inflow', 0)} 天</td>
                <td>-</td>
            </tr>
"""
    html += """        </table>
    </div>
"""

    # 三周期预测表格（核心新增）
    if three_horizon_results and len(three_horizon_results) > 0:
        html += """
    <h2>🔮 三周期预测结果（核心）</h2>
    <p style="color: #666; font-size: 12px;">按20天概率排序 | 三色系统：概率≥60%绿色，50-60%橙色，<50%红色</p>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th><th>板块</th>
            <th>1天预测</th><th>5天预测</th><th>20天预测</th>
            <th>模式</th><th>建议</th><th>胜率</th><th>核心</th>
        </tr>
"""
        # 按20天概率排序
        sorted_results = sorted(
            three_horizon_results.items(),
            key=lambda x: x[1].get('predictions', {}).get(20, {}).get('probability', 0.5),
            reverse=True
        )

        for code, data in sorted_results:
            name = A_STOCK_WATCHLIST.get(code, code)
            preds = data.get('predictions', {})
            pattern = data.get('pattern', '-')
            pattern_info = data.get('pattern_info', {})
            win_rate = data.get('win_rate', 0)

            # 获取股票分析数据
            analysis = stock_analyses.get(code, {})
            current_price = analysis.get('current_price', '-')
            change_pct = analysis.get('change_percent', 0)
            limit_rate = analysis.get('limit_rate', 0.1) * 100

            # 板块信息
            sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
            sector_name = sector_info.get('sector_name', '-')

            # 是否核心股
            is_core = is_core_holding(code) if 'is_core_holding' in dir() else False
            core_str = '⭐' if is_core else ''

            # 格式化预测（三色系统）
            def format_pred(h):
                if h not in preds:
                    return '-'
                p = preds[h]
                prob = p.get('probability', 0.5)
                direction = p.get('direction', '-')
                if direction == '↑':
                    if prob >= 0.60:
                        return f'<span style="color: #16a34a; font-weight: bold;">↑ {prob:.2f}</span>'
                    else:
                        return f'<span style="color: #ea580c; font-weight: bold;">↑ {prob:.2f}</span>'
                elif direction == '↓':
                    return f'<span style="color: #dc2626; font-weight: bold;">↓ {prob:.2f}</span>'
                return f'{direction} {prob:.2f}'

            pred_1d = format_pred(1)
            pred_5d = format_pred(5)
            pred_20d = format_pred(20)

            # 涨跌颜色
            change_str = f"{change_pct:+.2f}%" if change_pct else "-"
            change_class = 'positive' if change_pct and change_pct >= 0 else 'negative' if change_pct else ''

            # 模式颜色
            pattern_class = f'pattern-{pattern}' if pattern in THREE_HORIZON_PATTERNS else ''

            # 模式名称
            pattern_name = pattern_info.get('name', '-') if pattern_info else '-'
            action = pattern_info.get('action', '-') if pattern_info else '-'

            # 格式化价格
            price_str = f"{current_price:.2f}" if current_price else "-"

            html += f"""        <tr class="{pattern_class}">
            <td><strong>{name}</strong></td>
            <td>{code}</td>
            <td>{price_str}</td>
            <td class="{change_class}">{change_str}</td>
            <td>{sector_name}</td>
            <td>{pred_1d}</td>
            <td>{pred_5d}</td>
            <td>{pred_20d}</td>
            <td><strong>{pattern_name}</strong><br><small>{pattern}</small></td>
            <td>{action}</td>
            <td>{win_rate:.0f}%</td>
            <td>{core_str}</td>
        </tr>
"""

        html += """    </table>
"""

    # 自选股详细分析
    html += """
    <h2>📋 自选股技术分析</h2>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>涨跌停</th><th>RSI</th><th>MA5</th><th>MA20</th>
            <th>5日涨跌</th><th>20日涨跌</th>
        </tr>
"""
    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', '-')
        change = analysis.get('change_percent', 0)
        limit_rate = analysis.get('limit_rate', 0.1) * 100
        rsi = analysis.get('rsi_14', 50)
        ma5 = analysis.get('ma5', 0)
        ma20 = analysis.get('ma20', 0)
        return_5d = analysis.get('return_5d', 0)
        return_20d = analysis.get('return_20d', 0)

        change_class = 'positive' if change and change >= 0 else 'negative' if change else ''
        rsi_class = 'negative' if rsi > 70 else ('positive' if rsi < 30 else '')
        rsi_str = f"{rsi:.1f}" if rsi else "-"

        # 涨跌停状态
        limit_status = ""
        if change and abs(change) >= limit_rate - 0.1:
            limit_status = "🔴 涨停" if change > 0 else "🟢 跌停"

        # 格式化数值
        price_str = f"{price:.2f}" if price else "-"
        ma5_str = f"{ma5:.2f}" if ma5 else "-"
        ma20_str = f"{ma20:.2f}" if ma20 else "-"
        change_str = f"{change:+.2f}%" if change else "-"

        html += f"""        <tr>
            <td>{name}</td>
            <td>{code}</td>
            <td>{price_str}</td>
            <td class="{change_class}">{change_str}</td>
            <td>{limit_rate:.0f}% {limit_status}</td>
            <td class="{rsi_class}">{rsi_str}</td>
            <td>{ma5_str}</td>
            <td>{ma20_str}</td>
            <td>{return_5d:+.2f}%</td>
            <td>{return_20d:+.2f}%</td>
        </tr>
"""

    html += """    </table>
"""

    # AI 分析建议（完整版）
    html += """
    <h2>💡 AI 分析建议</h2>
    <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-size: 14px; line-height: 1.6;">
"""
    if llm_content:
        # 完整显示LLM内容
        html += llm_content.replace('\n', '<br>')
    else:
        html += '暂无AI建议'

    html += """
    </div>

    <h2>⚠️ 风险提示</h2>
    <ul>
        <li><strong>涨跌停限制</strong>：创业板/科创板 20%，主板 10%，ST股 5%</li>
        <li><strong>北向资金</strong>：关注外资流向变化，连续流出需警惕</li>
        <li><strong>市场情绪</strong>：极端熊市时暂停交易，保护本金</li>
        <li><strong>动态阈值</strong>：熊市需更高置信度才可买入</li>
    </ul>

    <div class="footer">
        <p>📧 本邮件由A股综合分析系统自动生成</p>
        <p>⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>⚠️ 本报告仅供参考，不构成投资建议</p>
    </div>
</body>
</html>"""

    return html


def send_email(subject, content, html_content=None):
    """
    发送邮件通知（使用统一消息服务模块）

    参数:
    - subject: 邮件主题
    - content: 邮件文本内容
    - html_content: 邮件HTML内容（可选）
    """
    try:
        from message_services import EmailSender
        sender = EmailSender()
        return sender.send_with_retry(subject, content, html_content)
    except ImportError:
        print("⚠️ 消息服务模块未安装，使用内置邮件发送")
        return _send_email_legacy(subject, content, html_content)


def _send_email_legacy(subject, content, html_content=None):
    """
    发送邮件通知（备用实现）

    参数:
    - subject: 邮件主题
    - content: 邮件文本内容
    - html_content: 邮件HTML内容（可选）
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # 从环境变量获取邮件配置
        sender_email = os.environ.get("EMAIL_SENDER")
        email_password = os.environ.get("EMAIL_PASSWORD")
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
        recipient_email = os.environ.get("RECIPIENT_EMAIL", "")

        if ',' in recipient_email:
            recipients = [recipient.strip() for recipient in recipient_email.split(',')]
        else:
            recipients = [recipient_email]

        if not sender_email or not email_password:
            print("❌ 邮件配置不完整，跳过邮件发送")
            return False

        # 根据SMTP服务器类型选择端口和SSL
        if "163.com" in smtp_server:
            smtp_port = 465
            use_ssl = True
        elif "qq.com" in smtp_server:
            smtp_port = 465
            use_ssl = True
        elif "gmail.com" in smtp_server:
            smtp_port = 587
            use_ssl = False
        else:
            smtp_port = 465
            use_ssl = True

        # 创建邮件对象
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)

        # 添加文本内容
        msg.attach(MIMEText(content, 'plain', 'utf-8'))

        # 添加HTML内容（如果有）
        if html_content:
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
            server.login(sender_email, email_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        print(f"✅ 邮件已发送到: {', '.join(recipients)}")
        return True

    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='A股综合分析')
    parser.add_argument('--llm-file', type=str, default=None,
                       help='大模型建议文件路径')
    parser.add_argument('--use-cached-predictions', action='store_true',
                       help='使用缓存的预测结果')
    parser.add_argument('--horizon', type=int, default=20,
                       help='预测周期（默认20天）')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件')
    parser.add_argument('--email', action='store_true',
                       help='发送邮件（默认行为）')

    args = parser.parse_args()

    # 确定是否发送邮件（默认发送，除非指定 --no-email）
    send_email_flag = not args.no_email

    print("\n" + "=" * 60)
    print("📊 A股综合分析")
    print("=" * 60)
    print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 分析股票: {len(STOCK_LIST)} 只")
    print(f"📧 发送邮件: {'是' if send_email_flag else '否'}")
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

    # 5. 生成三周期预测（核心新增）
    print("\n🔮 生成三周期预测（1d/5d/20d）...")
    three_horizon_results = None
    if not args.use_cached_predictions:
        three_horizon_results = generate_three_horizon_predictions(STOCK_LIST)
        if three_horizon_results:
            print(f"  ✅ 三周期预测完成: {len(three_horizon_results)} 只股票")
    else:
        print("  ⚠️ 使用缓存预测，跳过三周期预测")

    # 6. 计算市场情绪
    print("\n📈 计算市场情绪...")
    market_sentiment = get_market_sentiment(stock_analyses)
    layer_names = {
        'extreme_bear': '🔴 极端熊市',
        'bear': '🟠 熊市',
        'weak': '🟡 弱震荡',
        'normal': '🟢 正常市场',
    }
    print(f"  市场情绪: {layer_names.get(market_sentiment['layer'], market_sentiment['layer'])}")
    print(f"  上涨比例: {market_sentiment['up_ratio']:.1%}")
    print(f"  动态阈值: {market_sentiment['dynamic_threshold']:.0%}")

    # 7. 获取北向资金趋势
    print("\n💰 获取北向资金趋势...")
    northbound_trend = get_northbound_trend()
    print(f"  5日累积: {northbound_trend.get('net_buy_5d_sum', 0):.2f} 亿")
    print(f"  20日累积: {northbound_trend.get('net_buy_20d_sum', 0):.2f} 亿")
    print(f"  连续流入: {northbound_trend.get('consecutive_inflow', 0)} 天")

    # 8. 生成综合买卖建议
    print("\n📊 生成综合买卖建议...")
    recommendations = None
    try:
        from a_stock_recommendation_generator import AStockRecommendationGenerator
        rec_generator = AStockRecommendationGenerator()
        recommendations = rec_generator.generate_recommendations(
            llm_report=llm_content or '',
            ml_predictions=ml_predictions or {},
            stock_analyses=stock_analyses,
            market_data=market_data or {},
            northbound_data=northbound_trend or {}
        )
        print(f"  强烈买入: {len(recommendations.get('strong_buy', []))} 只")
        print(f"  买入: {len(recommendations.get('buy', []))} 只")
        print(f"  持有/观望: {len(recommendations.get('hold', []))} 只")
        print(f"  卖出: {len(recommendations.get('sell', []))} 只")
    except Exception as e:
        print(f"  ⚠️ 综合买卖建议生成失败: {e}")

    # 9. 生成综合报告
    print("\n📊 生成综合报告...")
    report = generate_comprehensive_report(llm_content, ml_predictions, stock_analyses, market_data)

    # 保存报告
    os.makedirs('data', exist_ok=True)
    report_file = f"data/a_stock_comprehensive_recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告已保存: {report_file}")

    # 10. 发送邮件（增强版）
    if send_email_flag:
        print("\n📊 发送增强版邮件...")
        html_content = generate_html_email(
            llm_content, ml_predictions, stock_analyses, market_data,
            three_horizon_results=three_horizon_results,
            market_sentiment=market_sentiment,
            northbound_trend=northbound_trend,
            recommendations=recommendations
        )

        date_str = datetime.now().strftime('%Y-%m-%d')
        email_subject = f"【综合分析】A股三周期预测 - {date_str}"

        if send_email(email_subject, report, html_content):
            print("  ✅ 邮件发送成功")
        else:
            print("  ⚠️ 邮件发送失败")

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