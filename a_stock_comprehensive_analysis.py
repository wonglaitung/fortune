#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议并发送邮件

⚠️ 运行时机：建议在A股收市后（15:00 CST）运行

版本：v4.0 (2026-07-19)
- 重构：综合买卖建议改由AI生成（替代硬编码逻辑）
- 新增：generate_comprehensive_recommendations_with_llm 函数
- 新增：数据格式化辅助函数（技术指标、ML预测、板块分析、异常检测）
- 删除：a_stock_recommendation_generator.py（硬编码逻辑）
- 优化：AI综合分析接收全部数据（技术指标+ML预测+市场情绪+主力资金+板块分析+异常检测）
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
    A_STOCK_TRAINING_LIST,
    A_STOCK_SECTOR_MAPPING,
    get_limit_rate,
    is_core_holding,
)

# 导入数据服务
from data_services.a_stock_data import get_a_stock_data, get_a_stock_info_tencent, get_index_data
from data_services.main_fund_flow import MainFundFlowService

# 导入LLM服务
from llm_services.qwen_engine import chat_with_llm

# 尝试导入技术分析模块（用于筹码阻力）
try:
    from data_services.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False

# 尝试导入A股网络分析模块（用于网络洞察）
try:
    from ml_services.a_stock_network_analysis import (
        fetch_stock_data,
        build_returns_dataframe,
        build_minimum_spanning_tree,
        calculate_centrality_metrics,
        detect_communities,
        identify_bridge_stocks,
        get_stock_name,
        get_stock_list,
    )
    A_STOCK_NETWORK_AVAILABLE = True
except ImportError:
    A_STOCK_NETWORK_AVAILABLE = False

# 尝试导入风险回报分析器
try:
    from ml_services.risk_reward_analyzer import RiskRewardAnalyzer
    RISK_REWARD_AVAILABLE = True
except ImportError:
    RISK_REWARD_AVAILABLE = False

# 股票名称映射
STOCK_NAMES = A_STOCK_WATCHLIST
STOCK_LIST = list(A_STOCK_WATCHLIST.keys())

# 板块分析使用全量股票（53只）
SECTOR_STOCK_LIST = list(A_STOCK_TRAINING_LIST.keys())

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

# A股传导模式准确率（参考港股，需要根据A股验证结果更新）
A_STOCK_TRANSMISSION_ACCURACY = {
    'AAA': {'win_rate': 58.0, 'description': '一致看涨'},
    'AAB': {'win_rate': 56.0, 'description': '短期调整'},
    'ABA': {'win_rate': 55.0, 'description': '震荡上行'},
    'ABB': {'win_rate': 54.0, 'description': '趋势转弱'},
    'BAA': {'win_rate': 56.0, 'description': '底部反弹'},
    'BAB': {'win_rate': 54.0, 'description': '震荡下行'},
    'BBA': {'win_rate': 53.0, 'description': '下跌反弹'},
    'BBB': {'win_rate': 52.0, 'description': '持续下跌'},
}


def extract_llm_recommendations(filepath):
    """
    从A股大模型建议文件中提取买卖建议，分别提取短期和中期建议

    参数:
    - filepath: 文件路径

    返回:
    - dict: 包含短期和中期建议的字典
      {
        'short_term': str,  # 短期建议文本
        'medium_term': str  # 中期建议文本
      }
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # A股文件格式：查找"短期操作建议"和"中期操作建议"的位置
        short_start = content.find("短期操作建议")
        medium_start = content.find("中期操作建议")

        # 备用格式：查找"短期（1-5天）"和"中期（1-4周）"
        if short_start == -1:
            short_start = content.find("短期（1-5天）")
        if medium_start == -1:
            medium_start = content.find("中期（1-4周）")

        # 再备用：查找"短期"和"中期"标题
        if short_start == -1:
            short_start = content.find("### 短期")
        if medium_start == -1:
            medium_start = content.find("### 中期")

        short_content = ""
        medium_content = ""

        if short_start != -1:
            if medium_start != -1 and medium_start > short_start:
                # 提取短期分析内容（从短期分析标题后到中期分析标题前）
                short_content = content[short_start:medium_start].strip()
            else:
                # 如果没有中期分析或中期在短期之前，提取到下一个主要标题
                next_section = content.find("\n## ", short_start + 10)
                if next_section != -1:
                    short_content = content[short_start:next_section].strip()
                else:
                    short_content = content[short_start:].strip()

        if medium_start != -1:
            # 提取中期分析内容（从中期分析标题后到文件末尾或下一个主要标题）
            next_section = content.find("\n## ", medium_start + 10)
            if next_section != -1:
                medium_content = content[medium_start:next_section].strip()
            else:
                medium_content = content[medium_start:].strip()

        # 如果都没找到，尝试从"AI分析报告"之后分割
        if not short_content and not medium_content:
            ai_report_start = content.find("## AI分析报告")
            if ai_report_start != -1:
                # 整个AI分析报告作为中期建议
                medium_content = content[ai_report_start:].strip()

        result = {
            'short_term': short_content if short_content else "暂无短期建议",
            'medium_term': medium_content if medium_content else "暂无中期建议"
        }

        return result

    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return {'short_term': '暂无短期建议', 'medium_term': '暂无中期建议'}


def load_a_stock_model_accuracy(horizon=20):
    """
    从Walk-forward验证结果或文件加载A股模型准确率信息

    参数:
    - horizon: 预测周期（默认20天）

    返回:
    - dict: 包含准确率信息
      {
        'catboost': {'accuracy': float, 'std': float},
        '1d': {'accuracy': float, 'std': float},
        '5d': {'accuracy': float, 'std': float},
        '20d': {'accuracy': float, 'std': float}
      }
    """
    import glob

    # 默认准确率值（A股个股预测准确率通常在50-60%）
    default_accuracy = {
        'catboost': {'accuracy': 0.55, 'std': 0.05},
        '1d': {'accuracy': 0.52, 'std': 0.08},
        '5d': {'accuracy': 0.54, 'std': 0.06},
        '20d': {'accuracy': 0.55, 'std': 0.05}
    }

    # 尝试从文件加载
    accuracy_file = 'data/a_stock_model_accuracy.json'
    if os.path.exists(accuracy_file):
        try:
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = {}
            catboost_key = f'catboost_{horizon}d'
            if catboost_key in data:
                result['catboost'] = {
                    'accuracy': data[catboost_key].get('accuracy', default_accuracy['catboost']['accuracy']),
                    'std': data[catboost_key].get('std', default_accuracy['catboost']['std'])
                }
            else:
                result['catboost'] = default_accuracy['catboost']

            for period in ['1d', '5d', '20d']:
                key = f'catboost_{period}'
                if key in data:
                    result[period] = {
                        'accuracy': data[key].get('accuracy', default_accuracy[period]['accuracy']),
                        'std': data[key].get('std', default_accuracy[period]['std'])
                    }
                else:
                    result[period] = default_accuracy[period]

            print(f"✅ 已加载模型准确率: {accuracy_file}")
            return result
        except Exception as e:
            print(f"⚠️ 加载模型准确率文件失败: {e}")

    # 尝试从Walk-forward验证结果计算
    search_patterns = [
        'output/*_a_stock_*_20d/validation_summary.json',
        'output/a_stock_walk_forward_*/validation_summary.json',
    ]

    for pattern in search_patterns:
        validation_files = glob.glob(pattern)
        if validation_files:
            try:
                latest_file = max(validation_files, key=os.path.getmtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    vf = json.load(f)

                if 'accuracy' in vf or 'avg_accuracy' in vf:
                    accuracy = vf.get('avg_accuracy', vf.get('accuracy', 0.55))
                    std = vf.get('accuracy_std', 0.05)

                    result = {
                        'catboost': {'accuracy': accuracy, 'std': std},
                        '1d': default_accuracy['1d'],
                        '5d': default_accuracy['5d'],
                        '20d': {'accuracy': accuracy, 'std': std}
                    }
                    print(f"✅ 从Walk-forward结果加载准确率: {latest_file}")
                    print(f"   CatBoost 20天: {accuracy:.2%} (±{std:.2%})")
                    return result
            except Exception as e:
                print(f"⚠️ 解析验证结果失败: {e}")

    print(f"⚠️ 使用默认模型准确率")
    return default_accuracy


def calculate_chip_resistance(stock_code):
    """
    计算筹码阻力（A股专用）

    Args:
        stock_code: 股票代码

    Returns:
        dict: {'resistance_ratio': 0.5, 'resistance_level': '中', 'resistance_icon': '⚠️中'}
    """
    if not TECHNICAL_ANALYSIS_AVAILABLE:
        return None

    try:
        analyzer = TechnicalAnalyzer()
        # 获取A股数据
        df = get_a_stock_data(stock_code, period_days=60)
        if df is None or df.empty or len(df) < 20:
            return None

        chip_result = analyzer.get_chip_distribution(df)
        if chip_result:
            resistance_ratio = chip_result.get('resistance_ratio', 0.5)
            if resistance_ratio < 0.3:
                resistance_level = '低'
                resistance_icon = '✅低'
            elif resistance_ratio < 0.6:
                resistance_level = '中'
                resistance_icon = '⚠️中'
            else:
                resistance_level = '高'
                resistance_icon = '🔴高'

            return {
                'resistance_ratio': resistance_ratio,
                'resistance_level': resistance_level,
                'resistance_icon': resistance_icon,
            }
    except Exception as e:
        pass

    return None


def calculate_risk_reward_scores(stock_code, stock_analyses):
    """
    计算风险得分和回报得分（A股专用）

    Args:
        stock_code: 股票代码
        stock_analyses: 股票分析结果

    Returns:
        dict: {'risk_score': 70, 'return_score': 60, 'comprehensive_score': 65, 'suggestion': '🟢 推荐'}
    """
    analysis = stock_analyses.get(stock_code, {})

    # 默认值
    result = {
        'risk_score': 50,
        'return_score': 50,
        'comprehensive_score': 50,
        'suggestion': '🟡 观察',
    }

    if not analysis:
        return result

    try:
        # 风险得分（基于波动率、涨跌幅偏离）
        volatility = analysis.get('volatility_20d', 0.03)
        change_pct = analysis.get('change_percent', 0)

        # 波动率风险（波动越大风险越高，得分越低）
        vol_risk = max(0, 100 - volatility * 1000)  # 波动率3% -> 70分

        # 涨跌风险（涨太多有回调风险，跌太多有反弹机会）
        if change_pct > 5:
            price_risk = 40  # 大涨后风险高
        elif change_pct < -5:
            price_risk = 60  # 大跌后有机会
        else:
            price_risk = 70  # 正常波动

        risk_score = (vol_risk + price_risk) / 2

        # 回报得分（基于RSI、趋势）
        rsi = analysis.get('rsi_14', 50)
        ma5 = analysis.get('ma5', 0)
        ma20 = analysis.get('ma20', 0)
        current_price = analysis.get('current_price', 0)

        # RSI回报（超卖有机会，超买有风险）
        if rsi < 30:
            rsi_return = 80  # 超卖，反弹机会大
        elif rsi > 70:
            rsi_return = 30  # 超买，回调风险大
        else:
            rsi_return = 50  # 中性

        # 趋势回报（多头排列回报高）
        if ma5 > 0 and ma20 > 0 and current_price > ma5 > ma20:
            trend_return = 75  # 多头排列
        elif current_price < ma5 < ma20:
            trend_return = 35  # 空头排列
        else:
            trend_return = 50  # 震荡

        return_score = (rsi_return + trend_return) / 2

        # 综合得分
        comprehensive_score = risk_score * 0.5 + return_score * 0.5

        # 建议等级
        if comprehensive_score >= 75:
            suggestion = '⭐ 优选'
        elif comprehensive_score >= 60:
            suggestion = '🟢 推荐'
        elif comprehensive_score >= 45:
            suggestion = '🟡 观察'
        else:
            suggestion = '🔴 暂缓'

        result = {
            'risk_score': int(risk_score),
            'return_score': int(return_score),
            'comprehensive_score': int(comprehensive_score),
            'suggestion': suggestion,
        }

    except Exception as e:
        pass

    return result


def calculate_network_insight(stock_code):
    """
    计算网络洞察（A股专用）

    从缓存文件读取网络特征，包括社区ID和枢纽等级

    Args:
        stock_code: 股票代码

    Returns:
        str: 网络洞察字符串（如 "社区1/高枢纽"）
    """
    # A股网络特征缓存文件路径
    cache_file = 'data/a_stock_network_features/network_features_for_ml.json'

    if not os.path.exists(cache_file):
        return '未知'

    try:
        with open(cache_file, 'r') as f:
            network_features = json.load(f)

        if stock_code not in network_features:
            return '未知'

        features = network_features[stock_code]
        community_id = features.get('net_community_id', -1)
        composite_centrality = features.get('net_composite_centrality', 0)
        community_rank = features.get('net_community_centrality_rank', 0)
        community_size = features.get('net_community_size', 1)

        # 计算枢纽等级（基于社区内排名）
        if community_size > 0:
            percentile = community_rank / community_size
            if percentile <= 0.3:  # Top 30% in community
                hub_level = '高'
            elif percentile <= 0.6:  # Middle 30%
                hub_level = '中'
            else:
                hub_level = '低'
        else:
            hub_level = '低'

        if community_id >= 0:
            return f"社区{community_id}/{hub_level}枢纽"
        else:
            return '未知'

    except Exception as e:
        return '未知'


def load_historical_profit_loss_ratio_a_stock():
    """
    从A股Walk-forward验证结果计算每只股票的历史盈亏比

    Returns:
        dict: {股票代码: {'profit_loss_ratio': 1.5, 'expected_return': 2.0, 'profit_loss_ratio_str': '1.50:1'}}
    """
    import glob

    # 查找最新的A股prediction_analysis.csv文件
    search_patterns = [
        'output/*_a_stock_*_20d/prediction_analysis.csv',
        'output/*_catboost_20d/prediction_analysis.csv',  # 兼容港股格式
    ]

    prediction_files = []
    for pattern in search_patterns:
        prediction_files.extend(glob.glob(os.path.join(os.getcwd(), pattern)))

    if not prediction_files:
        return {}

    # 使用最新的文件
    latest_file = max(prediction_files, key=lambda x: os.path.getmtime(x))

    try:
        df = pd.read_csv(latest_file)

        # 只分析预测UP的交易（实际买入场景）
        df_up = df[df['Predicted_Direction'] == 'UP'].copy()

        if df_up.empty:
            return {}

        results = {}
        for stock_code in df_up['code'].unique():
            stock_df = df_up[df_up['code'] == stock_code]
            n_samples = len(stock_df)

            if n_samples >= 3:  # 至少3次交易才计算
                correct_df = stock_df[stock_df['Is_Correct'] == True]
                incorrect_df = stock_df[stock_df['Is_Correct'] == False]

                n_correct = len(correct_df)
                avg_profit = correct_df['Actual_Return'].mean() * 100 if n_correct > 0 else 0
                avg_loss = abs(incorrect_df['Actual_Return'].mean() * 100) if len(incorrect_df) > 0 else 1
                win_rate = n_correct / n_samples

                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 999
                expected_return = win_rate * avg_profit - (1 - win_rate) * avg_loss

                # 盈亏比等级
                if profit_loss_ratio >= 3.0:
                    pl_grade = '⭐⭐⭐'
                elif profit_loss_ratio >= 2.0:
                    pl_grade = '⭐⭐'
                elif profit_loss_ratio >= 1.5:
                    pl_grade = '⭐'
                else:
                    pl_grade = '⚠️'

                results[stock_code] = {
                    'profit_loss_ratio': profit_loss_ratio,
                    'profit_loss_ratio_str': f'{profit_loss_ratio:.2f}:1',
                    'expected_return': expected_return,
                    'expected_return_str': f'{expected_return:+.2f}%',
                    'profit_loss_grade': pl_grade,
                    'win_rate': win_rate * 100,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'n_samples': n_samples
                }

        return results

    except Exception as e:
        return {}


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
    # 优先从全量股票列表获取名称，其次从核心持仓获取
    stock_name = A_STOCK_TRAINING_LIST.get(stock_code) or A_STOCK_WATCHLIST.get(stock_code, stock_code)
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

    # 主力资金
    main_fund_service = MainFundFlowService()
    mf_data = main_fund_service.get_latest()
    if mf_data:
        result['main_fund_net_flow'] = mf_data.get('main_net_flow', 0)
        result['main_fund_super_large'] = mf_data.get('super_large', 0)
        result['main_fund_large'] = mf_data.get('large', 0)

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
"""

    # 主力资金 - 仅当数据可用时显示
    mf_net = market_data.get('main_fund_net_flow', 0)
    mf_super = market_data.get('main_fund_super_large', 0)
    mf_large = market_data.get('main_fund_large', 0)
    if mf_net != 0 or mf_super != 0 or mf_large != 0:
        report += f"""
### 1.2 主力资金
- 净流入: {mf_net:.2f} 亿
- 超大单: {mf_super:.2f} 亿
- 大单: {mf_large:.2f} 亿
"""

    report += """
---

## 二、技术指标分析

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
            code = str(row.get('Stock_Code', '')).zfill(6)  # 标准化为6位代码
            name = A_STOCK_WATCHLIST.get(code, code)
            pred_proba = row.get('Prediction_Proba', 0.5)
            direction = '上涨' if pred_proba >= 0.5 else '下跌'
            confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba

            report += f"- **{name} ({code})**: {direction} (置信度: {confidence:.1%})\n"

    # 添加大模型建议
    if llm_content:
        report += f"""
---

## 四、大模型分析建议

{llm_content}
"""

    report += """
---

## 五、风险提示

1. **涨跌停限制**：创业板/科创板 20%，主板 10%，ST股 5%
2. **主力资金**：关注大资金流向变化
3. **市场情绪**：上证指数跌破MA20时需谨慎

---

*本报告仅供参考，不构成投资建议*
"""

    return report


def generate_three_horizon_predictions(stock_list, stock_analyses=None):
    """
    生成三周期预测（1天、5天、20天）- 增强版，包含筹码阻力、风险得分、网络洞察

    Args:
        stock_list: 股票代码列表
        stock_analyses: 股票分析结果（用于计算风险得分）

    Returns:
        dict: {股票代码: {
            predictions: {1: {...}, 5: {...}, 20: {...}},
            pattern: 'AAA',
            pattern_info: {...},
            chip_resistance: {...},
            risk_reward: {...},
            network_insight: '社区1/高枢纽',
            ...
        }}
    """
    from a_stock_ml_model import AStockTradingModel
    import pickle

    if stock_analyses is None:
        stock_analyses = {}

    three_horizon_results = {}

    # 预加载历史盈亏比数据
    historical_pl_data = load_historical_profit_loss_ratio_a_stock()

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

    # 计算三周期模式和附加指标
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

            # 计算历史胜率（基于传导模式准确率）
            transmission_info = A_STOCK_TRANSMISSION_ACCURACY.get(pattern, {'win_rate': 50, 'description': '未知'})
            data['win_rate'] = transmission_info['win_rate']

            # 计算筹码阻力
            chip_result = calculate_chip_resistance(code)
            if chip_result:
                data['chip_resistance'] = chip_result
            else:
                data['chip_resistance'] = {'resistance_icon': 'N/A'}

            # 计算风险回报得分
            risk_reward = calculate_risk_reward_scores(code, stock_analyses)
            data['risk_reward'] = risk_reward

            # 计算网络洞察
            data['network_insight'] = calculate_network_insight(code)

            # 获取历史盈亏比
            pl_info = historical_pl_data.get(code, {})
            if pl_info:
                data['profit_loss_ratio'] = pl_info.get('profit_loss_ratio_str', '-')
                data['expected_return'] = pl_info.get('expected_return_str', '-')
                data['profit_loss_grade'] = pl_info.get('profit_loss_grade', '')
            else:
                data['profit_loss_ratio'] = '-'
                data['expected_return'] = '-'
                data['profit_loss_grade'] = ''

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


def get_main_fund_trend():
    """
    获取主力资金趋势

    Returns:
        dict: 主力资金详细数据
    """
    result = {
        'net_flow': 0,
        'super_large': 0,
        'large': 0,
        'net_flow_5d_sum': 0,
        'net_flow_20d_sum': 0,
        'consecutive_inflow': 0,
    }

    try:
        service = MainFundFlowService()

        # 获取最新数据
        latest = service.get_latest()
        if latest:
            result['net_flow'] = latest.get('main_net_flow', 0)
            result['super_large'] = latest.get('super_large', 0)
            result['large'] = latest.get('large', 0)

        # 获取历史数据计算趋势
        history = service.fetch_history()
        if history is not None and not history.empty:
            # 累积流入（5日、20日）
            if 'main_net_flow' in history.columns:
                result['net_flow_5d_sum'] = history['main_net_flow'].tail(5).sum()
                result['net_flow_20d_sum'] = history['main_net_flow'].tail(20).sum()

                # 连续流入天数
                net_flow_series = history['main_net_flow'].tail(20)
                consecutive = 0
                for val in net_flow_series.iloc[::-1]:
                    if val > 0:
                        consecutive += 1
                    else:
                        break
                result['consecutive_inflow'] = consecutive

    except Exception as e:
        print(f"⚠️ 获取主力资金趋势失败: {e}")

    return result


def analyze_sectors(stock_analyses: dict) -> dict:
    """
    分析板块涨跌幅排名

    Args:
        stock_analyses: 股票分析结果字典

    Returns:
        dict: 板块分析结果
    """
    from a_stock_config import A_STOCK_SECTOR_MAPPING

    # 按板块聚合
    sector_data = {}
    for code, analysis in stock_analyses.items():
        sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
        sector_name = sector_info.get('sector', 'unknown')

        if sector_name not in sector_data:
            sector_data[sector_name] = {
                'stocks': [],
                'total_change': 0,
                'count': 0,
                'type': sector_info.get('type', 'unknown'),
            }

        sector_data[sector_name]['stocks'].append({
            'code': code,
            'name': analysis.get('name', code),
            'change': analysis.get('change_percent', 0),
        })
        sector_data[sector_name]['total_change'] += analysis.get('change_percent', 0)
        sector_data[sector_name]['count'] += 1

    # 计算平均涨跌幅并排序
    sector_ranking = []
    for sector_name, data in sector_data.items():
        avg_change = data['total_change'] / data['count'] if data['count'] > 0 else 0
        # 找出龙头股（涨幅最大的前3只）
        top_stocks = sorted(data['stocks'], key=lambda x: x['change'], reverse=True)[:3]
        sector_ranking.append({
            'sector': sector_name,
            'type': data['type'],
            'avg_change': avg_change,
            'stock_count': data['count'],
            'top_stocks': top_stocks,
        })

    # 按平均涨跌幅排序
    sector_ranking.sort(key=lambda x: x['avg_change'], reverse=True)

    return {
        'sector_ranking': sector_ranking,
        'total_sectors': len(sector_ranking),
    }


def detect_stock_anomalies(stock_analyses: dict, market_data: dict) -> dict:
    """
    检测股票异常（基于技术指标）

    Args:
        stock_analyses: 股票分析结果
        market_data: 市场数据

    Returns:
        dict: 异常检测结果
    """
    anomalies = {
        'high': [],    # 高严重度
        'medium': [],  # 中严重度
        'low': [],     # 低严重度
    }

    for code, analysis in stock_analyses.items():
        name = analysis.get('name', code)
        price = analysis.get('current_price', 0)
        change = analysis.get('change_percent', 0)
        rsi = analysis.get('rsi_14', 50)
        volume_ratio = analysis.get('volume_ratio', 1.0)

        # 异常原因列表
        reasons = []
        severity = 'low'

        # 1. RSI 超买/超卖
        if rsi > 80:
            reasons.append(f"RSI严重超买({rsi:.1f})")
            severity = 'high'
        elif rsi > 70:
            reasons.append(f"RSI超买({rsi:.1f})")
            severity = max(severity, 'medium')
        elif rsi < 20:
            reasons.append(f"RSI严重超卖({rsi:.1f})")
            severity = 'high'
        elif rsi < 30:
            reasons.append(f"RSI超卖({rsi:.1f})")
            severity = max(severity, 'medium')

        # 2. 大涨/大跌
        limit_rate = analysis.get('limit_rate', 0.1) * 100
        if abs(change) >= limit_rate - 0.5:
            if change > 0:
                reasons.append(f"接近涨停({change:+.2f}%)")
            else:
                reasons.append(f"接近跌停({change:+.2f}%)")
            severity = 'high'
        elif abs(change) >= 5:
            if change > 0:
                reasons.append(f"大幅上涨({change:+.2f}%)")
            else:
                reasons.append(f"大幅下跌({change:+.2f}%)")
            severity = max(severity, 'medium')

        # 3. 成交量异常
        if volume_ratio > 3:
            reasons.append(f"成交量放大{volume_ratio:.1f}倍")
            severity = max(severity, 'medium')
        elif volume_ratio > 2:
            reasons.append(f"成交量放大{volume_ratio:.1f}倍")
            severity = max(severity, 'low')

        # 4. 价格异常（与MA20偏离）
        ma20 = analysis.get('ma20', 0)
        if ma20 > 0 and price > 0:
            deviation = (price - ma20) / ma20 * 100
            if abs(deviation) > 15:
                reasons.append(f"偏离MA20 {deviation:+.1f}%")
                severity = max(severity, 'medium')

        # 如果有异常，添加到列表
        if reasons:
            anomaly_info = {
                'code': code,
                'name': name,
                'price': price,
                'change': change,
                'rsi': rsi,
                'reasons': reasons,
                'severity': severity,
            }

            if severity == 'high':
                anomalies['high'].append(anomaly_info)
            elif severity == 'medium':
                anomalies['medium'].append(anomaly_info)
            else:
                anomalies['low'].append(anomaly_info)

    # 统计
    total = len(anomalies['high']) + len(anomalies['medium']) + len(anomalies['low'])

    return {
        'anomalies': anomalies,
        'total_count': total,
        'high_count': len(anomalies['high']),
        'medium_count': len(anomalies['medium']),
        'low_count': len(anomalies['low']),
    }


def format_anomaly_summary_for_llm(anomaly_result: dict) -> str:
    """
    将异常数据格式化为大模型可理解的摘要

    Args:
        anomaly_result: 异常检测结果

    Returns:
        str: 格式化的异常摘要
    """
    if not anomaly_result or anomaly_result.get('total_count', 0) == 0:
        return "今日未检测到异常"

    anomalies = anomaly_result.get('anomalies', {})
    high_anomalies = anomalies.get('high', [])
    medium_anomalies = anomalies.get('medium', [])
    low_anomalies = anomalies.get('low', [])

    summary_lines = []

    # 高严重度异常
    if high_anomalies:
        summary_lines.append(f"## 高严重度异常（{len(high_anomalies)}只）")
        for a in high_anomalies:
            code = a['code']
            name = a['name']
            change = a.get('change', 0)
            rsi = a.get('rsi', 50)
            reasons = a.get('reasons', [])
            sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
            sector_name = sector_info.get('sector_name', '未知板块')

            summary_lines.append(f"- {code} {name}（{sector_name}）：当日{change:+.2f}%，RSI {rsi:.1f}，{', '.join(reasons)}")
        summary_lines.append("")

    # 中严重度异常
    if medium_anomalies:
        summary_lines.append(f"## 中严重度异常（{len(medium_anomalies)}只）")
        for a in medium_anomalies:
            code = a['code']
            name = a['name']
            change = a.get('change', 0)
            rsi = a.get('rsi', 50)
            reasons = a.get('reasons', [])
            sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
            sector_name = sector_info.get('sector_name', '未知板块')

            summary_lines.append(f"- {code} {name}（{sector_name}）：当日{change:+.2f}%，RSI {rsi:.1f}，{', '.join(reasons)}")
        summary_lines.append("")

    # 低严重度异常
    if low_anomalies:
        summary_lines.append(f"## 低严重度异常（{len(low_anomalies)}只）")
        for a in low_anomalies[:10]:  # 最多显示10只
            code = a['code']
            name = a['name']
            change = a.get('change', 0)
            rsi = a.get('rsi', 50)
            sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
            sector_name = sector_info.get('sector_name', '未知板块')

            summary_lines.append(f"- {code} {name}（{sector_name}）：当日{change:+.2f}%，RSI {rsi:.1f}")
        if len(low_anomalies) > 10:
            summary_lines.append(f"- ... 还有 {len(low_anomalies) - 10} 只低严重度异常")
        summary_lines.append("")

    # 添加统计摘要
    all_anomalies = high_anomalies + medium_anomalies + low_anomalies
    oversold_count = sum(1 for a in all_anomalies if a.get('rsi', 50) < 30)
    overbought_count = sum(1 for a in all_anomalies if a.get('rsi', 50) > 70)
    limit_up_count = sum(1 for a in all_anomalies if a.get('change', 0) > 9)
    limit_down_count = sum(1 for a in all_anomalies if a.get('change', 0) < -9)

    summary_lines.append("## 统计摘要")
    summary_lines.append(f"- 总异常数：{anomaly_result['total_count']}只")
    summary_lines.append(f"- RSI超卖（<30）：{oversold_count}只")
    summary_lines.append(f"- RSI超买（>70）：{overbought_count}只")
    summary_lines.append(f"- 接近涨停：{limit_up_count}只")
    summary_lines.append(f"- 接近跌停：{limit_down_count}只")

    return "\n".join(summary_lines)


def analyze_anomalies_with_llm(anomaly_result: dict) -> str:
    """
    使用大模型分析异常数据

    Args:
        anomaly_result: 异常检测结果

    Returns:
        str: 大模型生成的分析报告（Markdown格式）
    """
    if not anomaly_result or anomaly_result.get('total_count', 0) == 0:
        return "✅ 未检测到异常，市场波动正常"

    # 构建异常数据摘要
    anomaly_summary = format_anomaly_summary_for_llm(anomaly_result)

    # 构建提示词
    prompt = f"""你是A股量化分析师。请分析以下股票异常数据，提供深度洞察。

## 异常数据

{anomaly_summary}

## 分析要求

请从以下角度分析（但不限于）：
1. **整体市场状态**：超卖/超买比例、市场情绪判断
2. **板块异动**：哪些板块集体异动、板块轮动信号
3. **资金流向**：主力资金的流向判断
4. **个股亮点**：值得特别关注的个股及原因
5. **交易启示**：基于异常信号的操作建议（表格形式）
6. **风险提示**：需要警惕的风险点

输出格式要求：
- 使用Markdown格式
- 表格优先，简洁专业
- 每个分析板块用二级标题分隔
- 交易启示用表格展示（信号|解读|操作建议）
- 总字数控制在500字以内"""

    # 调用大模型
    try:
        response = chat_with_llm(prompt, enable_thinking=False)
        return response
    except Exception as e:
        print(f"⚠️ 大模型分析异常失败: {e}")
        return f"⚠️ 大模型分析失败: {e}"


# ============================================================
# AI综合买卖建议生成（替代硬编码逻辑）
# ============================================================

def format_stock_technical_data_for_llm(stock_analyses: dict) -> str:
    """
    格式化技术指标为LLM可读文本

    Args:
        stock_analyses: 股票分析结果

    Returns:
        str: 格式化的技术指标文本
    """
    lines = []
    # 按股票代码排序
    for code in sorted(stock_analyses.keys()):
        analysis = stock_analyses[code]
        name = analysis.get('name', code)
        price = analysis.get('current_price', '-')
        change = analysis.get('change_percent', 0)
        rsi = analysis.get('rsi_14', 50)
        ma5 = analysis.get('ma5', 0)
        ma20 = analysis.get('ma20', 0)
        ma60 = analysis.get('ma60', 0)
        return_5d = analysis.get('return_5d', 0)
        return_20d = analysis.get('return_20d', 0)
        limit_rate = analysis.get('limit_rate', 0.1) * 100

        # 均线位置判断
        ma_status = '中性'
        if price > ma5 > ma20 > ma60:
            ma_status = '多头排列'
        elif price < ma5 < ma20:
            ma_status = '空头排列'
        elif price > ma20:
            ma_status = '站上MA20'
        else:
            ma_status = '跌破MA20'

        # RSI状态
        if rsi > 70:
            rsi_status = '超买'
        elif rsi < 30:
            rsi_status = '超卖'
        else:
            rsi_status = '中性'

        lines.append(f"- {name}({code}): 现价{price:.2f}元, 今日{change:+.2f}%, RSI{rsi:.1f}({rsi_status}), MA{ma_status}, 5日{return_5d:+.1f}%, 20日{return_20d:+.1f}%, 涨跌停{limit_rate:.0f}%")

    return '\n'.join(lines)


def format_ml_predictions_for_llm(three_horizon_results: dict) -> str:
    """
    格式化三周期ML预测为LLM可读文本

    Args:
        three_horizon_results: 机器学习分析预测

    Returns:
        str: 格式化的预测文本
    """
    if not three_horizon_results:
        return "暂无ML预测数据"

    lines = []
    # 按股票代码排序
    sorted_results = sorted(
        three_horizon_results.items(),
        key=lambda x: x[0]
    )

    for code, data in sorted_results:
        name = A_STOCK_WATCHLIST.get(code, code)
        preds = data.get('predictions', {})
        pattern = data.get('pattern', '-')
        pattern_info = data.get('pattern_info', {})
        win_rate = data.get('win_rate', 0)

        # 获取概率
        prob_1d = preds.get(1, {}).get('probability', 0.5)
        prob_5d = preds.get(5, {}).get('probability', 0.5)
        prob_20d = preds.get(20, {}).get('probability', 0.5)

        # 格式化预测
        pred_str = f"1d={prob_1d:.2f}, 5d={prob_5d:.2f}, 20d={prob_20d:.2f}"
        pattern_name = pattern_info.get('name', '-')
        action = pattern_info.get('action', '-')

        lines.append(f"- {name}({code}): {pred_str}, 模式={pattern}({pattern_name}), 建议={action}, 胜率={win_rate:.0f}%")

    return '\n'.join(lines)


def format_sector_analysis_for_llm(sector_analysis: dict) -> str:
    """
    格式化板块分析为LLM可读文本

    Args:
        sector_analysis: 板块分析结果

    Returns:
        str: 格式化的板块分析文本
    """
    if not sector_analysis or not sector_analysis.get('sector_ranking'):
        return "暂无板块数据"

    lines = []
    for idx, sector in enumerate(sector_analysis['sector_ranking'][:10], 1):
        sector_name = sector.get('sector', '-')
        avg_change = sector.get('avg_change', 0)
        stock_count = sector.get('stock_count', 0)
        top_stocks = sector.get('top_stocks', [])

        # 龙头股
        top_stocks_str = ', '.join([f"{s['name']}({s['change']:+.1f}%)" for s in top_stocks[:3]])

        lines.append(f"{idx}. {sector_name}: 均涨{avg_change:+.2f}%, {stock_count}只, 龙头: {top_stocks_str}")

    return '\n'.join(lines)


def format_anomaly_for_llm(anomaly_result: dict) -> str:
    """
    格式化异常检测为LLM可读文本

    Args:
        anomaly_result: 异常检测结果

    Returns:
        str: 格式化的异常检测文本
    """
    if not anomaly_result or anomaly_result.get('total_count', 0) == 0:
        return "今日无异常"

    anomalies = anomaly_result.get('anomalies', {})
    lines = [f"共{anomaly_result['total_count']}个异常"]

    # 高严重度
    if anomalies.get('high'):
        lines.append(f"\n高严重度({len(anomalies['high'])}个):")
        for a in anomalies['high'][:5]:
            lines.append(f"  - {a['name']}({a['code']}): {a['change']:+.2f}%, RSI{a['rsi']:.1f}, {', '.join(a['reasons'])}")

    # 中严重度
    if anomalies.get('medium'):
        lines.append(f"\n中严重度({len(anomalies['medium'])}个):")
        for a in anomalies['medium'][:5]:
            lines.append(f"  - {a['name']}({a['code']}): {a['change']:+.2f}%, RSI{a['rsi']:.1f}")

    return '\n'.join(lines)


def parse_llm_json_response(response: str) -> dict:
    """
    解析AI返回的JSON响应

    Args:
        response: AI返回的文本

    Returns:
        dict: 解析后的字典
    """
    import re

    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取花括号内容
    brace_match = re.search(r'\{[\s\S]*\}', response)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 解析失败，返回空结构
    print("⚠️ 无法解析AI返回的JSON，使用默认结构")
    return {
        'strong_buy': [],
        'buy': [],
        'hold': [],
        'sell': [],
        'risk_control': {
            'market_risk': '中',
            'total_position': 0,
            'stop_loss_strategy': '单只股票最大亏损-8%',
            'position_strategy': '建议观望'
        }
    }


def generate_comprehensive_recommendations_with_llm(
    stock_analyses: dict,
    three_horizon_results: dict,
    market_sentiment: dict,
    main_fund_trend: dict,
    sector_analysis: dict,
    anomaly_result: dict,
    llm_recommendations: dict = None,
    model_accuracy: dict = None
) -> str:
    """
    使用AI生成综合买卖建议（参考港股实现）

    Args:
        stock_analyses: 股票技术分析结果
        three_horizon_results: 三周期ML预测结果
        market_sentiment: 市场情绪数据
        main_fund_trend: 主力资金趋势
        sector_analysis: 板块分析结果
        anomaly_result: 异常检测结果
        llm_recommendations: 大模型短期/中期建议（新增）
        model_accuracy: 模型准确率信息（新增）

    Returns:
        str: Markdown格式的综合买卖建议
    """
    print("\n🤖 正在调用AI生成综合买卖建议...")

    # 格式化ML预测数据（参考港股JSON格式）
    ml_stock_list = []
    for code, data in (three_horizon_results or {}).items():
        preds = data.get('predictions', {})
        pred_20d = preds.get(20, {})
        prob_20d = pred_20d.get('probability', 0.5)

        # 筹码阻力
        chip_data = data.get('chip_resistance', {})
        resistance_level = chip_data.get('resistance_level', 'N/A') if chip_data else 'N/A'

        # 风险回报
        risk_reward = data.get('risk_reward', {})

        ml_stock_list.append({
            'code': code,
            'name': A_STOCK_WATCHLIST.get(code, code),
            'probability_20d': round(prob_20d, 4),
            'prediction_20d': '上涨' if prob_20d >= 0.50 else '下跌',
            'chip_resistance': resistance_level,
            'risk_score': risk_reward.get('risk_score', '-'),
            'comprehensive_score': risk_reward.get('comprehensive_score', '-'),
        })

    # 按概率排序
    ml_stock_list.sort(key=lambda x: x['probability_20d'], reverse=True)

    # 构建ML预测JSON文本
    ml_predictions_text = "【CatBoost模型预测结果（20天）- JSON格式】\n"
    ml_predictions_text += "```json\n"
    ml_predictions_text += json.dumps(ml_stock_list, ensure_ascii=False, indent=2)
    ml_predictions_text += "\n```\n\n"
    ml_predictions_text += "**字段说明**：\n"
    ml_predictions_text += "- `probability_20d`: 20天上涨概率（>0.60=高置信度，0.50-0.60=中等，≤0.50=低）\n"
    ml_predictions_text += "- `chip_resistance`: 筹码阻力（低=拉升容易，中=注意风险，高=拉升困难）\n"
    ml_predictions_text += "- **使用建议**：probability_20d高 + chip_resistance低 = 更可靠信号\n"

    # 市场情绪
    sentiment_layer = market_sentiment.get('layer', 'normal')
    up_ratio = market_sentiment.get('up_ratio', 0.5)
    dynamic_threshold = market_sentiment.get('dynamic_threshold', 0.5)

    layer_names = {
        'extreme_bear': '极端熊市-暂停交易',
        'bear': '熊市-需概率≥0.70',
        'weak': '弱震荡-需概率≥0.65',
        'normal': '正常市场'
    }
    sentiment_name = layer_names.get(sentiment_layer, '正常市场')

    # 主力资金
    mf_flow = main_fund_trend.get('net_flow', 0)
    mf_5d = main_fund_trend.get('net_flow_5d_sum', 0)
    mf_20d = main_fund_trend.get('net_flow_20d_sum', 0)
    consecutive_inflow = main_fund_trend.get('consecutive_inflow', 0)
    main_fund_available = (mf_flow != 0 or mf_5d != 0 or mf_20d != 0)

    # 板块分析
    sector_data = format_sector_analysis_for_llm(sector_analysis)

    # 异常检测
    anomaly_data = format_anomaly_for_llm(anomaly_result)

    # 大模型建议
    llm_short_term = llm_recommendations.get('short_term', '暂无短期建议') if llm_recommendations else '暂无短期建议'
    llm_medium_term = llm_recommendations.get('medium_term', '暂无中期建议') if llm_recommendations else '暂无中期建议'

    # 模型准确率
    accuracy = model_accuracy.get('catboost', {}).get('accuracy', 0.55) if model_accuracy else 0.55
    std = model_accuracy.get('catboost', {}).get('std', 0.05) if model_accuracy else 0.05

    # 日期
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 构建Prompt（参考港股格式）
    prompt = f"""你是一位专业的A股投资分析师。请根据以下四部分信息，进行综合分析，给出实质的买卖建议。

=== 信息来源 ===

【主要信息源 - 决策依据】

【1. 大模型中期买卖建议（数周-数月）】
{llm_medium_term}

【2. CatBoost模型20天预测结果】
**重要：probability = 上涨概率（不是下跌概率）**
{ml_predictions_text}

【辅助信息源 - 操作时机参考】

【3. 大模型短期买卖建议（日内/数天）】
{llm_short_term}

【4. 市场情绪】
- 情绪层级：{sentiment_name}
- 上涨比例：{up_ratio:.1%}
- 动态阈值：买入需概率≥{dynamic_threshold:.0%}

【5. 主力资金趋势】
- 今日净流入：{mf_flow:.2f}亿
- 5日累积：{mf_5d:.2f}亿
- 20日累积：{mf_20d:.2f}亿
- 连续流入天数：{consecutive_inflow}天

【6. 板块分析（按涨幅排名）】
{sector_data}

【7. 异常检测】
{anomaly_data}

=== 核心硬性约束（不可违反）===

⚠️ **CatBoost概率约束（最高优先级，无例外）**：
- CatBoost概率 ≤ 0.50 → **绝对禁止推荐买入或强烈买入**
- CatBoost概率 < 0.40 → **绝对禁止推荐持有或观望**
- CatBoost概率 ≥ 0.50 → 可以考虑买入
- CatBoost概率 ≥ 0.60 → 可以考虑强烈买入
- **即使短期和中期方向一致，也绝对不允许违反此约束**
- **违反此约束的建议将被视为错误**

🔥 **决策顺序（严格遵守）**：
第一步：检查CatBoost概率 → 不满足则直接排除
第二步：检查市场情绪 → 极端熊市暂停交易，熊市需概率≥0.70，弱震荡需概率≥0.65
第三步：检查短期和中期一致性
第四步：生成综合建议

=== 综合分析规则 ===

**规则1：时间维度匹配（业界最佳实践）**
- **短期信号（触发器）**：负责"何时做"（Timing）
- **中期信号（确认器）**：负责"是否做"（Direction）
- **CatBoost模型（验证器）**：负责提升置信度
- 只有短期和中期方向一致时，才采取行动
- 短期和中期冲突时，选择观望（避免不确定性）

**决策逻辑（短期触发 + 中期确认 + CatBoost验证）**：
- **第一步：检查CatBoost概率（硬约束）**
  - probability ≤ 0.50 → 排除买入或强烈买入
  - probability ≥ 0.50 → 进入下一步
- **第二步：检查市场情绪**
  - 极端熊市 → 暂停所有买入
  - 熊市 → 需probability ≥ 0.70
  - 弱震荡 → 需probability ≥ 0.65
- **第三步：检查短期和中期一致性**
  - 短期看好，中期看好 → 进入下一步
  - 方向不一致 → 观望
- **第四步：生成建议**
  - 强烈买入：短期看好，中期看好，CatBoost probability ≥ 0.60，市场情绪正常
  - 买入：短期看好，中期看好，0.50 < CatBoost probability < 0.60
  - 持有/观望：CatBoost probability ≤ 0.50 或 方向不一致
  - 卖出：短期看跌，中期看跌

**规则2：CatBoost概率评估**

**CatBoost概率阈值**：
- **高置信度上涨**：probability > 0.60
- **中等置信度观望**：0.50 < probability ≤ 0.60
- **预测下跌**：probability ≤ 0.50

**阈值优化说明**：
- 当前CatBoost模型20天准确率：约{accuracy:.2%}（±{std:.2%}）
- 强买入阈值0.60略高于准确率，确保高置信度
- 买入阈值0.50接近准确率，平衡召回率和精确率

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（如有）
1. [股票代码] [股票名称]
   - 推荐理由：[包含短期建议、中期建议、CatBoost概率]
   - 操作建议：买入
   - 建议仓位：
     * 保守型：[X]%
     * 适度型：[X]%
     * 激进型：[X]%
   - 止损位：[现价-8%]
   - 目标价：[现价+10%]
   - 风险提示：[主要风险因素]

## 买入信号（如有）
...

## 持有/观望
...

## 卖出信号（如有）
...

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 保守型总仓位：[X]%（上限50%）
- 适度型总仓位：[X]%（上限70%）
- 激进型总仓位：[X]%（上限90%）
- 止损位设置：[策略]
- 组合调整建议：[具体的组合调整建议]

---
分析日期：{date_str}
"""

    # 调用大模型
    try:
        response = chat_with_llm(prompt, enable_thinking=False)

        print(f"  ✅ AI建议生成完成")

        return response

    except Exception as e:
        print(f"⚠️ AI综合建议生成失败: {e}")
        return f"""# 综合买卖建议

## 风险控制建议
- 当前市场整体风险：中
- 建议仓位百分比：0%
- 止损位设置：单只股票最大亏损-8%
- 组合调整建议：建议观望

---
分析日期：{date_str}
"""


def format_anomalies_html(anomaly_result: dict, anomaly_llm_analysis: str = None) -> str:
    """
    格式化异常检测为HTML

    Args:
        anomaly_result: 异常检测结果
        anomaly_llm_analysis: 大模型异常分析结果（新增）

    Returns:
        str: HTML格式的异常检测表格
    """
    if not anomaly_result or anomaly_result.get('total_count', 0) == 0:
        return ""

    anomalies = anomaly_result.get('anomalies', {})

    html = f"""
    <h2>🔴 股票异常检测提醒</h2>
    <p style="color: #666; font-size: 12px;">检测到 {anomaly_result['total_count']} 个异常（全量股票）</p>
"""

    # 高严重度
    if anomalies.get('high'):
        html += """
    <h3>🔴 高严重度异常</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th><th>RSI</th><th>异常原因</th>
        </tr>
"""
        # 按股票代码排序
        sorted_anomalies = sorted(anomalies['high'], key=lambda x: str(x.get('code', '')).zfill(6))
        for a in sorted_anomalies:
            change_class = 'positive' if a['change'] >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{a['name']}</strong></td>
            <td>{a['code']}</td>
            <td>{a['price']:.2f}</td>
            <td class="{change_class}">{a['change']:+.2f}%</td>
            <td>{a['rsi']:.1f}</td>
            <td>{', '.join(a['reasons'])}</td>
        </tr>
"""
        html += "    </table>\n"

    # 中严重度
    if anomalies.get('medium'):
        html += """
    <h3>⚠️ 中严重度异常</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th><th>RSI</th><th>异常原因</th>
        </tr>
"""
        # 按股票代码排序
        sorted_anomalies = sorted(anomalies['medium'], key=lambda x: str(x.get('code', '')).zfill(6))
        for a in sorted_anomalies:
            change_class = 'positive' if a['change'] >= 0 else 'negative'
            html += f"""        <tr>
            <td>{a['name']}</td>
            <td>{a['code']}</td>
            <td>{a['price']:.2f}</td>
            <td class="{change_class}">{a['change']:+.2f}%</td>
            <td>{a['rsi']:.1f}</td>
            <td>{', '.join(a['reasons'])}</td>
        </tr>
"""
        html += "    </table>\n"

    # 低严重度（最多显示10个）
    if anomalies.get('low'):
        # 先排序，再取前10个
        sorted_low = sorted(anomalies['low'], key=lambda x: str(x.get('code', '')).zfill(6))[:10]
        html += f"""
    <h3>📋 低严重度异常（显示前10个，共{len(anomalies['low'])}个）</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th><th>异常原因</th>
        </tr>
"""
        for a in sorted_low:
            change_class = 'positive' if a['change'] >= 0 else 'negative'
            html += f"""        <tr>
            <td>{a['name']}</td>
            <td>{a['code']}</td>
            <td>{a['price']:.2f}</td>
            <td class="{change_class}">{a['change']:+.2f}%</td>
            <td>{', '.join(a['reasons'])}</td>
        </tr>
"""
        html += "    </table>\n"

    # 大模型异常分析（新增）
    if anomaly_llm_analysis:
        # 将Markdown转换为HTML
        try:
            import markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
            llm_html = md.convert(anomaly_llm_analysis)
        except ImportError:
            # 如果没有安装markdown库，使用简单转换
            llm_html = anomaly_llm_analysis.replace('\n', '<br>')

        html += f"""
    <h3>🤖 大模型异常分析</h3>
    <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; font-size: 14px; line-height: 1.6;">
{llm_html}
    </div>
"""

    return html


def format_sectors_html(sector_analysis: dict) -> str:
    """
    格式化板块分析为HTML

    Args:
        sector_analysis: 板块分析结果

    Returns:
        str: HTML格式的板块分析表格
    """
    if not sector_analysis or not sector_analysis.get('sector_ranking'):
        return ""

    html = """
    <h2>📊 板块分析（5日涨跌幅排名）</h2>
    <table>
        <tr>
            <th>排名</th><th>板块名称</th><th>类型</th><th>平均涨跌幅</th><th>股票数</th><th>龙头股TOP 3</th>
        </tr>
"""

    for idx, sector in enumerate(sector_analysis['sector_ranking'], 1):
        # 涨跌标识
        if sector['avg_change'] > 3:
            icon = "🔥"
        elif sector['avg_change'] > 0:
            icon = "📈"
        else:
            icon = "📉"

        # 龙头股列表
        top_stocks_str = " / ".join([
            f"{s['name']}({s['change']:+.2f}%)"
            for s in sector['top_stocks']
        ]) if sector['top_stocks'] else "-"

        html += f"""        <tr>
            <td>{idx}</td>
            <td>{icon} {sector['sector']}</td>
            <td>{sector['type']}</td>
            <td class="{'positive' if sector['avg_change'] >= 0 else 'negative'}">{sector['avg_change']:+.2f}%</td>
            <td>{sector['stock_count']}</td>
            <td>{top_stocks_str}</td>
        </tr>
"""

    html += "    </table>\n"

    return html


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
"""

    # 主力资金 - 仅当数据可用时显示
    mf_net = market_data.get('main_fund_net_flow', 0)
    mf_super = market_data.get('main_fund_super_large', 0)
    mf_large = market_data.get('main_fund_large', 0)
    if mf_net != 0 or mf_super != 0 or mf_large != 0:
        report += f"""
### 1.2 主力资金
- 净流入: {mf_net:.2f} 亿
- 超大单: {mf_super:.2f} 亿
- 大单: {mf_large:.2f} 亿
"""

    report += """
---

## 二、技术指标分析

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
            code = str(row.get('Stock_Code', '')).zfill(6)  # 标准化为6位代码
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
2. **主力资金**: 关注大资金流向变化
3. **市场情绪**: 上证指数跌破MA20时需谨慎

---

*本报告仅供参考，不构成投资建议*
"""

    return report


def _format_recommendations_section(recommendations, stock_analyses=None):
    """
    格式化综合买卖建议HTML部分

    Args:
        recommendations: 综合买卖建议（Markdown字符串或字典）
        stock_analyses: 股票分析结果（用于补充现价和涨跌数据）

    Returns:
        str: HTML格式的建议
    """
    # 如果是字符串（Markdown格式），使用markdown库转换
    if isinstance(recommendations, str):
        try:
            import markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
            return md.convert(recommendations)
        except ImportError:
            # 如果没有安装markdown库，使用简单转换
            return recommendations.replace('\n', '<br>')

    # 如果是字典，按原有逻辑处理
    html = ""

    # 辅助函数：获取股票的现价和涨跌
    def get_stock_price_info(stock_code):
        if stock_analyses and stock_code:
            # 标准化股票代码为6位（补齐前导零）
            normalized_code = str(stock_code).zfill(6)
            if normalized_code in stock_analyses:
                analysis = stock_analyses[normalized_code]
                return {
                    'current_price': analysis.get('current_price', 0) or 0,
                    'change_percent': analysis.get('change_percent', 0) or 0
                }
        return {'current_price': 0, 'change_percent': 0}

    # 强烈买入
    if recommendations.get('strong_buy'):
        html += """
    <h3>🟢 强烈买入信号</h3>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>保守</th><th>适度</th><th>激进</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
        # 按股票代码排序
        sorted_recs = sorted(recommendations['strong_buy'], key=lambda x: str(x.get('stock_code', '')).zfill(6))
        for rec in sorted_recs:
            stock_code = rec.get('stock_code', '')
            price_info = get_stock_price_info(stock_code)
            current_price = rec.get('current_price') or price_info['current_price']
            change_percent = rec.get('change_percent', price_info['change_percent'])
            change_class = 'positive' if change_percent >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec.get('stock_name', stock_code)}</strong></td>
            <td>{stock_code}</td>
            <td>{current_price:.2f}</td>
            <td class="{change_class}">{change_percent:+.2f}%</td>
            <td>{rec.get('position_conservative', 0)}%</td>
            <td><strong>{rec.get('position_moderate', 0)}%</strong></td>
            <td>{rec.get('position_aggressive', 0)}%</td>
            <td>{rec.get('stop_loss', 0):.2f}</td>
            <td>{rec.get('target_price', 0):.2f}</td>
            <td>{rec.get('reason', '')}</td>
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
            <th>保守</th><th>适度</th><th>激进</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
        # 按股票代码排序
        sorted_recs = sorted(recommendations['buy'], key=lambda x: str(x.get('stock_code', '')).zfill(6))
        for rec in sorted_recs:
            stock_code = rec.get('stock_code', '')
            price_info = get_stock_price_info(stock_code)
            current_price = rec.get('current_price') or price_info['current_price']
            change_percent = rec.get('change_percent', price_info['change_percent'])
            change_class = 'positive' if change_percent >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec.get('stock_name', stock_code)}</strong></td>
            <td>{stock_code}</td>
            <td>{current_price:.2f}</td>
            <td class="{change_class}">{change_percent:+.2f}%</td>
            <td>{rec.get('position_conservative', 0)}%</td>
            <td><strong>{rec.get('position_moderate', 0)}%</strong></td>
            <td>{rec.get('position_aggressive', 0)}%</td>
            <td>{rec.get('stop_loss', 0):.2f}</td>
            <td>{rec.get('target_price', 0):.2f}</td>
            <td>{rec.get('reason', '')}</td>
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
            <th>建议</th><th>推荐理由</th>
        </tr>
"""
        # 按股票代码排序
        sorted_recs = sorted(recommendations['hold'], key=lambda x: str(x.get('stock_code', '')).zfill(6))
        for rec in sorted_recs:
            stock_code = rec.get('stock_code', '')
            price_info = get_stock_price_info(stock_code)
            current_price = rec.get('current_price') or price_info['current_price']
            change_percent = rec.get('change_percent', price_info['change_percent'])
            change_class = 'positive' if change_percent >= 0 else 'negative'
            html += f"""        <tr>
            <td>{rec.get('stock_name', stock_code)}</td>
            <td>{stock_code}</td>
            <td>{current_price:.2f}</td>
            <td class="{change_class}">{change_percent:+.2f}%</td>
            <td>观望</td>
            <td>{rec.get('reason', '')}</td>
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
            <th>建议卖出价</th><th>推荐理由</th>
        </tr>
"""
        # 按股票代码排序
        sorted_recs = sorted(recommendations['sell'], key=lambda x: str(x.get('stock_code', '')).zfill(6))
        for rec in sorted_recs:
            stock_code = rec.get('stock_code', '')
            price_info = get_stock_price_info(stock_code)
            current_price = rec.get('current_price') or price_info['current_price']
            change_percent = rec.get('change_percent', price_info['change_percent'])
            change_class = 'positive' if change_percent >= 0 else 'negative'
            html += f"""        <tr>
            <td><strong>{rec.get('stock_name', stock_code)}</strong></td>
            <td>{stock_code}</td>
            <td>{current_price:.2f}</td>
            <td class="{change_class}">{change_percent:+.2f}%</td>
            <td>{current_price:.2f}</td>
            <td>{rec.get('reason', '')}</td>
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
        <tr><td>保守型总仓位</td><td>{risk.get('total_position_conservative', 0)}%（上限50%）</td></tr>
        <tr><td>适度型总仓位</td><td>{risk.get('total_position_moderate', 0)}%（上限70%）</td></tr>
        <tr><td>激进型总仓位</td><td>{risk.get('total_position_aggressive', 0)}%（上限90%）</td></tr>
        <tr><td>止损策略</td><td>{risk.get('stop_loss_strategy', '')}</td></tr>
        <tr><td>仓位策略</td><td>{risk.get('position_strategy', '')}</td></tr>
    </table>
"""

    return html


def generate_html_email(llm_content, ml_predictions_20d, stock_analyses, market_data,
                          three_horizon_results=None, market_sentiment=None, main_fund_trend=None,
                          recommendations=None, sector_analysis=None, anomaly_result=None,
                          anomaly_llm_analysis=None):
    """
    生成增强版HTML邮件（参考港股详细度）

    Args:
        llm_content: 大模型建议内容
        ml_predictions_20d: ML预测结果
        stock_analyses: 股票分析结果
        market_data: 市场数据
        three_horizon_results: 机器学习分析预测（新增）
        market_sentiment: 市场情绪数据（新增）
        main_fund_trend: 主力资金趋势（新增）
        recommendations: 综合买卖建议（新增）
        sector_analysis: 板块分析结果（新增）
        anomaly_result: 异常检测结果（新增）
        anomaly_llm_analysis: 大模型异常分析结果（新增）

    Returns:
        str: HTML格式邮件
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 市场涨跌颜色
    sh_change = market_data.get('sh_change', 0)
    sh_color = 'green' if sh_change >= 0 else 'red'

    # 主力资金颜色
    mf_flow = market_data.get('main_fund_net_flow', 0)
    mf_color = 'green' if mf_flow >= 0 else 'red'

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
    <p style="margin: 5px 0 15px 0; font-size: 13px;">📖 <a href="https://github.com/wonglaitung/fortune/blob/main/docs/A_STOCK_DESIGN.md" style="color: #2563eb;">A股智能分析系统设计文档</a> 👆</p>
"""

    # ========== 1. 综合买卖建议（核心） ==========
    if recommendations:
        html += """
    <h2>📋 综合买卖建议</h2>
    <p style="color: #666; font-size: 12px;">综合ML预测概率、技术指标信号、大模型个股分析，由AI给出最终建议，仅供参考</p>
"""
        html += _format_recommendations_section(recommendations, stock_analyses)

    # ========== 2. 市场概况 ==========
    html += f"""
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
"""

    # 主力资金部分（仅当数据可用时显示）
    mf_super = market_data.get('main_fund_super_large', 0) or 0
    mf_large = market_data.get('main_fund_large', 0) or 0
    if mf_flow != 0 or mf_super != 0 or mf_large != 0:
        html += f"""
        <h3>💰 主力资金</h3>
        <table>
            <tr><th>指标</th><th>数值</th><th>趋势</th></tr>
            <tr>
                <td>今日净流入</td>
                <td class="{'positive' if mf_flow >= 0 else 'negative'}">{mf_flow:.2f} 亿</td>
                <td>{'流入' if mf_flow >= 0 else '流出'}</td>
            </tr>
            <tr>
                <td>超大单</td>
                <td>{mf_super:.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>大单</td>
                <td>{mf_large:.2f} 亿</td>
                <td>-</td>
            </tr>
"""
        # 主力资金趋势
        if main_fund_trend:
            html += f"""            <tr>
                <td>5日累积流入</td>
                <td class="{'positive' if main_fund_trend.get('net_flow_5d_sum', 0) >= 0 else 'negative'}">{main_fund_trend.get('net_flow_5d_sum', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>20日累积流入</td>
                <td class="{'positive' if main_fund_trend.get('net_flow_20d_sum', 0) >= 0 else 'negative'}">{main_fund_trend.get('net_flow_20d_sum', 0):.2f} 亿</td>
                <td>-</td>
            </tr>
            <tr>
                <td>连续流入天数</td>
                <td>{main_fund_trend.get('consecutive_inflow', 0)} 天</td>
                <td>-</td>
            </tr>
"""
        html += """        </table>
"""
    html += """    </div>
"""

    # ========== 3. 市场情绪分析 ==========
    trade_advice = '暂停交易' if sentiment_layer == 'extreme_bear' else ('正常交易' if sentiment_layer == 'normal' else '谨慎交易')
    html += f"""
    <div class="sentiment-box">
        <h2>📈 市场情绪分析</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>说明</th></tr>
            <tr>
                <td><strong>市场情绪</strong></td>
                <td style="color: {sentiment_color}; font-weight: bold;">{sentiment_name}</td>
                <td>{trade_advice}</td>
            </tr>
            <tr>
                <td>今日上涨比例</td>
                <td>{up_ratio:.1%}</td>
                <td>全量股票上涨数量占比（53只）</td>
            </tr>
            <tr>
                <td>动态置信阈值</td>
                <td>{dynamic_threshold:.0%}</td>
                <td>买入信号需达到的概率</td>
            </tr>
        </table>
    </div>
"""

    # ========== 4. 技术指标分析 ==========
    html += """
    <h2>📋 技术指标分析</h2>
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

    # ========== 5. 机器学习分析预测表格（核心，港股19列格式） ==========
    if three_horizon_results and len(three_horizon_results) > 0:
        html += """
    <h2>🔮 机器学习分析预测</h2>
    <p style="color: #666; font-size: 12px;">按股票代码排序 | 三色系统：概率≥60%绿色，50-60%橙色，<50%红色</p>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th><th>板块</th><th>类型</th>
            <th>1天</th><th>5天</th><th>20天</th><th>市场调整</th>
            <th>模式</th><th>量化建议</th><th>胜率</th>
            <th>筹码阻力</th><th>盈亏比</th><th>期望收益</th>
            <th>风险得分</th><th>综合得分</th><th>风险建议</th>
        </tr>
"""
        # 按股票代码排序
        sorted_results = sorted(
            three_horizon_results.items(),
            key=lambda x: x[0]
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

            # 板块信息
            sector_info = A_STOCK_SECTOR_MAPPING.get(code, {})
            sector_name = sector_info.get('sector', '-')
            sector_type = sector_info.get('type', '-')

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

            # 市场调整（基于市场情绪）
            market_layer = market_sentiment.get('layer', 'normal') if market_sentiment else 'normal'
            prob_20d = preds.get(20, {}).get('probability', 0.5) if 20 in preds else 0.5

            if market_layer == 'extreme_bear':
                market_adjust = '🔴暂停'
            elif market_layer == 'bear':
                market_adjust = '🟠高置信' if prob_20d >= 0.70 else '🟡降级'
            elif market_layer == 'weak':
                market_adjust = '🟡谨慎'
            else:
                market_adjust = '🟢正常'

            # 涨跌颜色
            change_str = f"{change_pct:+.2f}%" if change_pct else "-"
            change_class = 'positive' if change_pct and change_pct >= 0 else 'negative' if change_pct else ''

            # 模式名称
            pattern_name = pattern_info.get('name', '-') if pattern_info else '-'
            action = pattern_info.get('action', '-') if pattern_info else '-'

            # 筹码阻力
            chip_resistance = data.get('chip_resistance', {})
            resistance_icon = chip_resistance.get('resistance_icon', 'N/A')

            # 盈亏比
            pl_ratio = data.get('profit_loss_ratio', '-')
            pl_grade = data.get('profit_loss_grade', '')
            pl_display = f"{pl_ratio} {pl_grade}" if pl_ratio != '-' else '-'

            # 期望收益
            expected_return = data.get('expected_return', '-')

            # 风险回报得分
            risk_reward = data.get('risk_reward', {})
            risk_score = risk_reward.get('risk_score', '-')
            return_score = risk_reward.get('return_score', '-')
            comprehensive_score = risk_reward.get('comprehensive_score', '-')
            suggestion = risk_reward.get('suggestion', '🟡 观察')

            # 格式化价格
            price_str = f"{current_price:.2f}" if current_price else "-"

            html += f"""        <tr>
            <td><strong>{name}</strong></td>
            <td>{code}</td>
            <td>{price_str}</td>
            <td class="{change_class}">{change_str}</td>
            <td>{sector_name}</td>
            <td>{sector_type}</td>
            <td>{pred_1d}</td>
            <td>{pred_5d}</td>
            <td>{pred_20d}</td>
            <td>{market_adjust}</td>
            <td><strong>{pattern_name}</strong><br><small>{pattern}</small></td>
            <td>{action}</td>
            <td>{win_rate:.0f}%</td>
            <td>{resistance_icon}</td>
            <td>{pl_display}</td>
            <td>{expected_return}</td>
            <td>{risk_score}</td>
            <td>{comprehensive_score}</td>
            <td>{suggestion}</td>
        </tr>
"""

        html += """    </table>

    <div style="color: #888; font-size: 10px; margin-top: 8px; background: #fafafa; padding: 6px 8px; border-radius: 4px; line-height: 1.4;">
        <span style="margin-right: 12px;"><strong>颜色</strong>：<span style="color: #16a34a;">↑≥60%</span> 高置信 | <span style="color: #ea580c;">↑50-60%</span> 中等 | <span style="color: #dc2626;">↓&lt;50%</span> 看跌</span>
        <span style="margin-right: 12px;"><strong>市场</strong>：🟢正常 | 🟡谨慎 | 🟠高置信 | 🔴暂停</span>
        <span style="margin-right: 12px;"><strong>筹码</strong>：✅低 &lt;30% | ⚠️中 30-60% | 🔴高 &gt;60%</span>
        <span><strong>风险</strong>：⭐优选 ≥75 | 🟢推荐 60-75 | 🟡观察 45-60 | 🔴暂缓 &lt;45</span>
    </div>
"""

    # ========== 6. 大模型分析建议 ==========
    # 预计算生成时间
    generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html += """
    <h2>💡 大模型分析建议</h2>
    <p style="color: #666; font-size: 12px; margin-bottom: 10px;">
        <strong>说明</strong>：AI建议基于通义千问大模型，结合技术指标、市场环境、基本面等定性分析，与量化建议（基于CatBoost三周期预测）可能存在差异，请综合参考。
    </p>
    <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; font-size: 14px; line-height: 1.6;">
"""
    if llm_content:
        # 将Markdown转换为HTML
        try:
            import markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
            llm_html = md.convert(llm_content)
            html += llm_html
        except ImportError:
            # 如果没有安装markdown库，使用简单转换
            html += llm_content.replace('\n', '<br>')
    else:
        html += '暂无AI建议'

    html += """
    </div>
"""

    # ========== 7. 板块分析 ==========
    if sector_analysis and sector_analysis.get('sector_ranking'):
        html += format_sectors_html(sector_analysis)

    # ========== 8. 股票异常检测提醒 + 大模型异常分析 ==========
    if anomaly_result and anomaly_result.get('total_count', 0) > 0:
        html += format_anomalies_html(anomaly_result, anomaly_llm_analysis)

    # ========== 9. 风险提示 ==========
    html += f"""
    <h2>⚠️ 风险提示</h2>
    <ul>
        <li><strong>涨跌停限制</strong>：创业板/科创板 20%，主板 10%，ST股 5%</li>
        <li><strong>主力资金</strong>：关注大资金流向变化，连续流出需警惕</li>
        <li><strong>市场情绪</strong>：极端熊市时暂停交易，保护本金</li>
        <li><strong>动态阈值</strong>：熊市需更高置信度才可买入</li>
    </ul>

    <div class="footer">
        <p>📧 本邮件由A股综合分析系统自动生成</p>
        <p>⏰ 生成时间：{generation_time}</p>
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
    llm_recommendations = None
    if llm_file:
        llm_content = read_llm_recommendations(llm_file)
        if llm_content:
            print("  ✅ 大模型建议读取成功")
            # 提取短期和中期建议
            llm_recommendations = extract_llm_recommendations(llm_file)
            print(f"  ✅ 提取短期建议: {len(llm_recommendations.get('short_term', ''))} 字符")
            print(f"  ✅ 提取中期建议: {len(llm_recommendations.get('medium_term', ''))} 字符")

    # 1b. 加载模型准确率
    print("\n📊 加载模型准确率...")
    model_accuracy = load_a_stock_model_accuracy(horizon=args.horizon)

    # 2. 读取ML预测结果
    print("\n📊 读取ML预测结果...")
    ml_predictions = read_ml_predictions(args.horizon)

    # 3. 分析市场
    print("\n📊 分析市场环境...")
    market_data = analyze_market()
    if market_data:
        print(f"  上证指数: {market_data.get('sh_close', 'N/A'):.2f} ({market_data.get('sh_change', 0):+.2f}%)")
        print(f"  主力资金: {market_data.get('main_fund_net_flow', 0):.2f} 亿")

    # 4. 分析每只股票（核心股用于预测和交易）
    print("\n📊 分析自选股...")
    stock_analyses = {}
    for code in STOCK_LIST:
        print(f"  分析 {code}...")
        analysis = get_stock_analysis(code)
        stock_analyses[code] = analysis

    # 4b. 分析全量股票（用于市场情绪和板块分析）
    print(f"\n📊 分析全量股票（{len(SECTOR_STOCK_LIST)}只）...")
    sector_stock_analyses = {}
    for code in SECTOR_STOCK_LIST:
        analysis = get_stock_analysis(code)
        sector_stock_analyses[code] = analysis

    # 5. 生成三周期预测（核心新增，增强版）
    print("\n🔮 生成三周期预测（1d/5d/20d）...")
    three_horizon_results = None
    if not args.use_cached_predictions:
        three_horizon_results = generate_three_horizon_predictions(STOCK_LIST, sector_stock_analyses)
        if three_horizon_results:
            print(f"  ✅ 三周期预测完成: {len(three_horizon_results)} 只股票")
    else:
        print("  ⚠️ 使用缓存预测，尝试从缓存文件读取三周期数据...")
        # 从缓存文件构建机器学习分析预测
        three_horizon_results = {}
        for horizon in [1, 5, 20]:
            cache_file = f'data/a_stock_models/ml_predictions_{horizon}d.csv'
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, dtype={'Stock_Code': str})  # 确保股票代码为字符串
                    print(f"  ✅ 读取 {horizon}d 缓存: {len(df)} 条记录")
                    for _, row in df.iterrows():
                        code = str(row.get('Stock_Code', ''))
                        if not code:
                            continue
                        # 补齐前导零（A股代码为6位）
                        code = code.zfill(6)
                        if code not in three_horizon_results:
                            three_horizon_results[code] = {'predictions': {}}

                        prob = row.get('Prediction_Proba', 0.5)
                        direction = '↑' if prob >= 0.5 else '↓'

                        three_horizon_results[code]['predictions'][horizon] = {
                            'direction': direction,
                            'probability': prob,
                            'current_price': row.get('Current_Price'),
                            'date': row.get('Data_Date'),  # 修正列名
                        }
                except Exception as e:
                    print(f"  ⚠️ 读取 {horizon}d 缓存失败: {e}")

        # 如果读取到了数据，计算模式和附加指标
        if three_horizon_results:
            print(f"  ✅ 从缓存构建三周期预测: {len(three_horizon_results)} 只股票")
            # 计算三周期模式和附加指标
            for code, data in three_horizon_results.items():
                preds = data.get('predictions', {})
                if len(preds) >= 1:  # 至少有一个周期的预测
                    # 模式计算：A=上涨(概率>=0.5)，B=下跌(概率<0.5)
                    pattern = ''
                    for h in [1, 5, 20]:
                        if h in preds:
                            pattern += 'A' if preds[h]['probability'] >= 0.5 else 'B'
                        else:
                            pattern += 'B'  # 缺失的周期默认看跌

                    data['pattern'] = pattern
                    pattern_info = THREE_HORIZON_PATTERNS.get(pattern, {'name': '未知', 'action': '观望'})
                    data['pattern_info'] = pattern_info

                    # 计算历史胜率（基于传导模式准确率）
                    transmission_info = A_STOCK_TRANSMISSION_ACCURACY.get(pattern, {'win_rate': 50, 'description': '未知'})
                    data['win_rate'] = transmission_info['win_rate']

                    # 计算筹码阻力
                    chip_result = calculate_chip_resistance(code)
                    if chip_result:
                        data['chip_resistance'] = chip_result
                    else:
                        data['chip_resistance'] = {'resistance_icon': 'N/A'}

                    # 计算风险回报得分
                    risk_reward = calculate_risk_reward_scores(code, sector_stock_analyses)
                    data['risk_reward'] = risk_reward

                    # 计算网络洞察
                    data['network_insight'] = calculate_network_insight(code)

                    # 获取历史盈亏比
                    pl_info = load_historical_profit_loss_ratio_a_stock().get(code, {})
                    if pl_info:
                        data['profit_loss_ratio'] = pl_info.get('profit_loss_ratio_str', '-')
                        data['expected_return'] = pl_info.get('expected_return_str', '-')
                        data['profit_loss_grade'] = pl_info.get('profit_loss_grade', '')
                    else:
                        data['profit_loss_ratio'] = '-'
                        data['expected_return'] = '-'
                        data['profit_loss_grade'] = ''
        else:
            print("  ⚠️ 缓存文件不存在或读取失败，跳过三周期预测")

    # 6. 计算市场情绪（使用全量股票）
    print("\n📈 计算市场情绪...")
    market_sentiment = get_market_sentiment(sector_stock_analyses)
    layer_names = {
        'extreme_bear': '🔴 极端熊市',
        'bear': '🟠 熊市',
        'weak': '🟡 弱震荡',
        'normal': '🟢 正常市场',
    }
    print(f"  市场情绪: {layer_names.get(market_sentiment['layer'], market_sentiment['layer'])}")
    print(f"  上涨比例: {market_sentiment['up_ratio']:.1%}")
    print(f"  动态阈值: {market_sentiment['dynamic_threshold']:.0%}")

    # 7. 获取主力资金趋势
    print("\n💰 获取主力资金趋势...")
    main_fund_trend = get_main_fund_trend()
    print(f"  5日累积: {main_fund_trend.get('net_flow_5d_sum', 0):.2f} 亿")
    print(f"  20日累积: {main_fund_trend.get('net_flow_20d_sum', 0):.2f} 亿")
    print(f"  连续流入: {main_fund_trend.get('consecutive_inflow', 0)} 天")

    # 8. 分析板块涨跌幅（使用全量股票）
    print("\n📊 分析板块涨跌幅...")
    sector_analysis = analyze_sectors(sector_stock_analyses)
    if sector_analysis and sector_analysis.get('sector_ranking'):
        print(f"  板块数量: {sector_analysis['total_sectors']}")
        print(f"  分析股票: {len(sector_stock_analyses)} 只")
        top_sector = sector_analysis['sector_ranking'][0]
        print(f"  领涨板块: {top_sector['sector']} ({top_sector['avg_change']:+.2f}%)")

    # 9. 检测股票异常（使用全量股票）
    print("\n🔴 检测股票异常...")
    anomaly_result = detect_stock_anomalies(sector_stock_analyses, market_data or {})
    if anomaly_result and anomaly_result.get('total_count', 0) > 0:
        print(f"  异常总数: {anomaly_result['total_count']} 个")
        print(f"  高严重度: {anomaly_result['high_count']} 个")
        print(f"  中严重度: {anomaly_result['medium_count']} 个")
        print(f"  低严重度: {anomaly_result['low_count']} 个")

    # 9b. 使用LLM分析异常
    anomaly_llm_analysis = None
    if anomaly_result and anomaly_result.get('total_count', 0) > 0:
        print("\n🤖 使用LLM分析异常...")
        try:
            anomaly_llm_analysis = analyze_anomalies_with_llm(anomaly_result)
            print("  ✅ 大模型异常分析完成")
        except Exception as e:
            print(f"  ⚠️ 大模型异常分析失败: {e}")

    # 10. 生成综合买卖建议（使用AI综合分析）
    print("\n📊 生成综合买卖建议...")
    recommendations = generate_comprehensive_recommendations_with_llm(
        stock_analyses=stock_analyses,
        three_horizon_results=three_horizon_results,
        market_sentiment=market_sentiment,
        main_fund_trend=main_fund_trend,
        sector_analysis=sector_analysis,
        anomaly_result=anomaly_result,
        llm_recommendations=llm_recommendations,
        model_accuracy=model_accuracy
    )

    # 保存综合买卖建议到文件
    if recommendations:
        os.makedirs('data', exist_ok=True)
        recommendations_file = f"data/a_stock_comprehensive_recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(recommendations_file, 'w', encoding='utf-8') as f:
            f.write(recommendations)
        print(f"✅ 综合买卖建议已保存: {recommendations_file}")

    # 11. 生成综合报告
    print("\n📊 生成综合报告...")
    report = generate_comprehensive_report(llm_content, ml_predictions, stock_analyses, market_data)

    # 保存报告
    os.makedirs('data', exist_ok=True)
    report_file = f"data/a_stock_comprehensive_recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告已保存: {report_file}")

    # 12. 发送邮件（增强版）
    if send_email_flag:
        print("\n📊 发送增强版邮件...")
        html_content = generate_html_email(
            llm_content, ml_predictions, stock_analyses, market_data,
            three_horizon_results=three_horizon_results,
            market_sentiment=market_sentiment,
            main_fund_trend=main_fund_trend,
            recommendations=recommendations,
            sector_analysis=sector_analysis,
            anomaly_result=anomaly_result,
            anomaly_llm_analysis=anomaly_llm_analysis
        )

        date_str = datetime.now().strftime('%Y-%m-%d')
        email_subject = f"【综合分析】A股买卖建议 - {date_str}"

        if send_email(email_subject, report, html_content):
            print("  ✅ 邮件发送成功")
        else:
            print("  ⚠️ 邮件发送失败")

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("📊 报告摘要")
    print("=" * 60)
    print(f"  市场状态: 上证 {market_data.get('sh_change', 0):+.2f}%")
    print(f"  主力资金: {market_data.get('main_fund_net_flow', 0):.2f} 亿")
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