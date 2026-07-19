#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议

⚠️ 运行时机：建议在港股收市后（16:00 HKT）运行
   - 市场情绪过滤器依赖当日收市数据计算上涨比例
   - 盘中运行可能导致上涨比例不完整，影响阈值判断准确性
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入大模型服务
from llm_services.qwen_engine import chat_with_llm

# 导入配置
from config import WATCHLIST, STOCK_SECTOR_MAPPING, SECTOR_NAME_MAPPING

# 导入交易日历工具
from data_services.calendar_features import get_last_trading_day

# 从WATCHLIST提取股票名称映射
STOCK_NAMES = WATCHLIST
STOCK_LIST = WATCHLIST  # 为兼容 hsi_email 模块添加别名

# 导入必要的模块
try:
    from data_services.hk_sector_analysis import SectorAnalyzer
    SECTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    SECTOR_ANALYSIS_AVAILABLE = False
    print("⚠️ 板块分析模块不可用")

try:
    from detect_stock_anomalies import StockAnomalyDetector, run_stock_anomaly_detection
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    print("⚠️ 异常检测模块不可用")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️ AKShare模块不可用")

# 导入技术分析工具（用于筹码分布分析）
try:
    from data_services.technical_analysis import TechnicalAnalyzer
    from data_services.tencent_finance import get_hk_stock_data_tencent
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️ 技术分析模块不可用")

# 导入 ML 模型（用于三周期预测）
try:
    from ml_services.ml_trading_model import CatBoostModel
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False
    print("⚠️ ML模型模块不可用")

# 导入混合波动率模型
try:
    from ml_services.hybrid_volatility_model import HybridGARCHLSTM
    HYBRID_VOL_MODEL_AVAILABLE = True
except ImportError:
    HYBRID_VOL_MODEL_AVAILABLE = False

# 板块类型定义（周期性/防御性）
# 来源：SECTOR_ROTATION_ANALYSIS.md
SECTOR_TYPES = {
    # 周期性板块
    'semiconductor': {'name': '半导体', 'type': '周期'},
    'biotech': {'name': '生物医药', 'type': '周期'},
    'tech': {'name': '科技', 'type': '周期'},
    'consumer': {'name': '消费', 'type': '周期'},
    'real_estate': {'name': '房地产', 'type': '周期'},
    'energy': {'name': '能源', 'type': '周期'},
    'shipping': {'name': '航运', 'type': '周期'},
    'auto': {'name': '汽车', 'type': '周期'},
    'new_energy': {'name': '新能源', 'type': '周期'},
    'ai': {'name': '人工智能', 'type': '周期'},
    # 防御性板块
    'bank': {'name': '银行', 'type': '防御'},
    'insurance': {'name': '保险', 'type': '防御'},
    'utility': {'name': '公用事业', 'type': '防御'},
    'environmental': {'name': '环保', 'type': '防御'},
    'exchange': {'name': '交易所', 'type': '防御'},
    'index': {'name': '指数', 'type': '防御'},
}

# 三周期预测模式配置（基于个股完整模型验证结果）
# 来源：docs/THREE_HORIZON_ANALYSIS.md 第12章
# 验证数据：12只港股，Walk-forward 30 folds，Top 500特征（含利率特征）
# 更新日期：2026-05-23
THREE_HORIZON_PATTERNS = {
    '010': {'name': '反弹失败', 'action': '谨慎减仓', 'win_rate': '57.22%', 'avg_return': '-1.65%', 'confidence': '低'},
    '001': {'name': '下跌中继⭐', 'action': '谨慎做多', 'win_rate': '56.95%', 'avg_return': '+4.86%', 'confidence': '低'},
    '111': {'name': '一致看涨', 'action': '谨慎持有', 'win_rate': '56.80%', 'avg_return': '+0.14%', 'confidence': '低'},
    '011': {'name': '探底回升', 'action': '分批建仓', 'win_rate': '56.52%', 'avg_return': '+3.54%', 'confidence': '低'},
    '000': {'name': '一致看跌', 'action': '止损/减仓', 'win_rate': '56.28%', 'avg_return': '-2.54%', 'confidence': '低'},
    '100': {'name': '冲高回落', 'action': '获利了结', 'win_rate': '55.37%', 'avg_return': '-2.28%', 'confidence': '低'},
    '110': {'name': '震荡回调', 'action': '观望', 'win_rate': '54.62%', 'avg_return': '+2.34%', 'confidence': '低'},
    '101': {'name': '假突破', 'action': '持有观望', 'win_rate': '53.63%', 'avg_return': '+2.21%', 'confidence': '低'},
}

# 恒指三周期预测模式配置（基于恒指增强模型验证结果）
# 来源：docs/THREE_HORIZON_ANALYSIS.md 第一部分
# 验证数据：905个恒指样本，Walk-forward验证，增强模型（33特征）
# 更新日期：2026-05-18
# 注意：恒指准确率显著高于个股，最优模式为"假突破"(101)
HSI_THREE_HORIZON_PATTERNS = {
    '101': {'name': '假突破⭐⭐', 'action': '抄底买入', 'win_rate': '87.32%', 'avg_return': '高', 'confidence': '极高'},
    '111': {'name': '一致看涨⭐', 'action': '持有/买入', 'win_rate': '86.26%', 'avg_return': '+0.14%', 'confidence': '极高'},
    '011': {'name': '探底回升', 'action': '分批建仓', 'win_rate': '82.11%', 'avg_return': '+3.54%', 'confidence': '高'},
    '001': {'name': '下跌中继⭐', 'action': '谨慎做多', 'win_rate': '81.05%', 'avg_return': '+4.86%', 'confidence': '高'},
    '110': {'name': '震荡回调', 'action': '观望', 'win_rate': '80.77%', 'avg_return': '+2.34%', 'confidence': '中高'},
    '000': {'name': '一致看跌', 'action': '减仓/做空', 'win_rate': '79.80%', 'avg_return': '-2.54%', 'confidence': '中高'},
    '010': {'name': '反弹失败', 'action': '谨慎做多', 'win_rate': '77.78%', 'avg_return': '+3.54%', 'confidence': '中'},
    '100': {'name': '冲高回落', 'action': '获利了结', 'win_rate': '76.32%', 'avg_return': '-2.28%', 'confidence': '中'},
}

# 恒指传导律准确率数据（来源：docs/THREE_HORIZON_ANALYSIS.md）
# 更新日期：2026-05-18，905个恒指样本验证
HSI_TRANSMISSION_ACCURACY = {
    'both_correct_rate': 81.49,     # 1天+5天都正确时，20天准确率
    'independent_20d_rate': 81.22,   # 独立20天准确率
    'improvement': 0.27              # 传导效应略有提升
}

# 个股传导律准确率数据（来源：docs/THREE_HORIZON_ANALYSIS.md）
# 更新日期：2026-05-23，12只港股验证（Top 500特征，含利率特征）
TRANSMISSION_ACCURACY = {
    'both_correct_rate': 59.12,      # 1天+5天都正确时，20天准确率
    'independent_20d_rate': 56.91,   # 独立20天准确率
    'improvement': 2.21              # 提升幅度
}


def format_transmission_display(transmission_info):
    """
    格式化传导模式显示字符串

    参数:
    - transmission_info: check_transmission_mode() 返回的字典

    返回:
    - str: 格式化的显示字符串
    """
    pred_1d_dir = transmission_info.get('pred_1d_direction')
    pred_5d_dir = transmission_info.get('pred_5d_direction')
    pred_20d_dir = transmission_info.get('pred_20d_direction')
    pred_1d_correct = transmission_info.get('pred_1d_correct')
    pred_5d_correct = transmission_info.get('pred_5d_correct')
    pred_20d_correct = transmission_info.get('pred_20d_correct')
    transmission_mode = transmission_info.get('transmission_mode')
    prediction_date = transmission_info.get('prediction_date')

    # 如果没有预测数据
    if pred_1d_dir is None and pred_5d_dir is None and pred_20d_dir is None:
        return "暂无数据"

    # 方向符号
    def get_dir_symbol(direction):
        if direction == 'up':
            return "↑"
        elif direction == 'down':
            return "↓"
        else:
            return "?"

    dir_1d_symbol = get_dir_symbol(pred_1d_dir)
    dir_5d_symbol = get_dir_symbol(pred_5d_dir)
    dir_20d_symbol = get_dir_symbol(pred_20d_dir)

    # 结果符号
    def get_result_symbol(correct):
        if correct is None:
            return "⏳"  # 待验证
        elif correct:
            return "✓"   # 正确
        else:
            return "✗"   # 错误

    result_1d = get_result_symbol(pred_1d_correct)
    result_5d = get_result_symbol(pred_5d_correct)
    result_20d = get_result_symbol(pred_20d_correct)

    # 构建显示字符串
    parts = []

    if pred_1d_dir is not None:
        parts.append(f"1天{dir_1d_symbol}{result_1d}")

    if pred_5d_dir is not None:
        parts.append(f"5天{dir_5d_symbol}{result_5d}")

    if pred_20d_dir is not None:
        parts.append(f"20天{dir_20d_symbol}{result_20d}")

    display = " ".join(parts)

    # 如果传导模式激活，添加标记
    if transmission_mode:
        display = f"✅传导({display})"
    elif display:
        display = display
    else:
        display = "-"

    return display if display else "-"


def check_transmission_mode(stock_code, data_date):
    """
    检查传导模式：查看5个交易日前的预测验证情况

    逻辑：在4月19日时，查看4月10日做的预测：
    - 1天预测（目标4月11日）已到期，可验证
    - 5天预测（目标4月17日）已到期，可验证
    - 20天预测（目标5月8日）未到期

    传导律：当1天和5天预测都正确时，20天预测准确率提升5.53%

    参数:
    - stock_code: 股票代码（如 '0005.HK'）
    - data_date: 当前报告日期（如 '2026-04-19'）

    返回:
    - dict: {
        'transmission_mode': bool,      # 是否进入传导模式
        'pred_1d_correct': bool/None,   # 1天预测是否正确
        'pred_5d_correct': bool/None,   # 5天预测是否正确
        'pred_20d_correct': bool/None,  # 20天预测是否正确
        'pred_1d_direction': str/None,  # 1天预测方向
        'pred_5d_direction': str/None,  # 5天预测方向
        'pred_20d_direction': str/None, # 20天预测方向
        'prediction_date': str,         # 预测日期（5个交易日前）
        'both_correct_rate': 62.28,     # 传导模式准确率
        'independent_20d_rate': 56.75   # 独立20天准确率
      }
    """
    import json
    from datetime import datetime, timedelta

    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prediction_history.json')

    # 计算5个交易日前的日期
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        trading_dates = df['trade_date'].astype(str).tolist()

        # 找到当前日期在交易日列表中的位置，往前推5个交易日
        current_date_str = data_date if isinstance(data_date, str) else data_date.strftime('%Y-%m-%d')

        # 如果当前日期不在交易日列表中，找最近的交易日
        if current_date_str in trading_dates:
            current_idx = trading_dates.index(current_date_str)
        else:
            # 找最近的交易日
            for i, d in enumerate(trading_dates):
                if d >= current_date_str:
                    current_idx = i
                    break
            else:
                current_idx = len(trading_dates) - 1

        # 往前推5个交易日
        target_idx = max(0, current_idx - 5)
        prediction_date = trading_dates[target_idx]
    except Exception as e:
        # 回退到自然日计算
        if isinstance(data_date, str):
            data_date_obj = datetime.strptime(data_date, '%Y-%m-%d')
        else:
            data_date_obj = data_date
        prediction_date = (data_date_obj - timedelta(days=7)).strftime('%Y-%m-%d')

    if not os.path.exists(history_file):
        return {'transmission_mode': False, 'pred_1d_correct': None, 'pred_5d_correct': None, 'pred_20d_correct': None,
                'pred_1d_direction': None, 'pred_5d_direction': None, 'pred_20d_direction': None,
                'prediction_date': prediction_date}

    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)

        predictions = history.get('predictions', [])

        # 查找该股票在5个交易日前的1天、5天、20天预测
        pred_1d = None
        pred_5d = None
        pred_20d = None

        for pred in predictions:
            if pred['stock_code'] == stock_code and pred['data_date'] == prediction_date:
                if pred['horizon'] == 1:
                    pred_1d = pred
                elif pred['horizon'] == 5:
                    pred_5d = pred
                elif pred['horizon'] == 20:
                    pred_20d = pred

        # 检查是否都存在
        if not (pred_1d and pred_5d):
            return {'transmission_mode': False, 'pred_1d_correct': None, 'pred_5d_correct': None,
                    'pred_1d_direction': pred_1d.get('predicted_direction') if pred_1d else None,
                    'pred_5d_direction': pred_5d.get('predicted_direction') if pred_5d else None,
                    'pred_20d_correct': pred_20d.get('outcome') == 'correct' if pred_20d and pred_20d.get('outcome') else None,
                    'pred_20d_direction': pred_20d.get('predicted_direction') if pred_20d else None,
                    'prediction_date': prediction_date}

        # 如果outcome字段已存在，直接使用
        if pred_1d.get('outcome') is not None and pred_5d.get('outcome') is not None:
            pred_1d_correct = pred_1d.get('outcome') == 'correct'
            pred_5d_correct = pred_5d.get('outcome') == 'correct'
            pred_20d_correct = pred_20d.get('outcome') == 'correct' if pred_20d and pred_20d.get('outcome') else None
            return {
                'transmission_mode': pred_1d_correct and pred_5d_correct,
                'pred_1d_correct': pred_1d_correct,
                'pred_5d_correct': pred_5d_correct,
                'pred_20d_correct': pred_20d_correct,
                'pred_1d_direction': pred_1d.get('predicted_direction'),
                'pred_5d_direction': pred_5d.get('predicted_direction'),
                'pred_20d_direction': pred_20d.get('predicted_direction') if pred_20d else None,
                'prediction_date': prediction_date,
                'both_correct_rate': TRANSMISSION_ACCURACY['both_correct_rate'],
                'independent_20d_rate': TRANSMISSION_ACCURACY['independent_20d_rate']
            }
        
        # 动态计算：检查预测是否已到期并验证正确
        now = datetime.now()
        
        def verify_prediction(pred, horizon):
            """验证单个预测是否正确"""
            if not pred:
                return None
            
            # 检查是否有entry_price
            entry_price = pred.get('entry_price')
            predicted_direction = pred.get('predicted_direction')
            
            if not entry_price or entry_price <= 0:
                return None
            
            # 计算目标日期（data_date + horizon个交易日，近似为horizon天）
            try:
                data_date_obj = datetime.strptime(pred['data_date'], '%Y-%m-%d')
            except (ValueError, KeyError):
                return None
            
            # 检查是否已到期（当前日期 >= data_date + horizon天）
            target_date = data_date_obj + timedelta(days=horizon)
            if now < target_date:
                # 尚未到期，无法验证
                return None
            
            # 获取目标日期的收盘价
            try:
                ticker = yf.Ticker(stock_code)
                # 获取目标日期附近的数据
                hist = ticker.history(start=target_date - timedelta(days=3), end=target_date + timedelta(days=3))
                if hist.empty:
                    return None
                
                # 找到最接近目标日期的收盘价
                exit_price = hist['Close'].iloc[-1]
                
                # 计算实际收益
                actual_return = (exit_price - entry_price) / entry_price
                actual_direction = 'up' if actual_return > 0 else 'down'
                
                # 判断预测是否正确
                return predicted_direction == actual_direction
            except Exception:
                return None
        
        # 验证1天和5天预测
        pred_1d_correct = verify_prediction(pred_1d, 1)
        pred_5d_correct = verify_prediction(pred_5d, 5)
        pred_20d_correct = pred_20d.get('outcome') == 'correct' if pred_20d and pred_20d.get('outcome') else None

        # 如果任一预测尚未到期或无法验证，返回None
        if pred_1d_correct is None or pred_5d_correct is None:
            return {
                'transmission_mode': False,
                'pred_1d_correct': pred_1d_correct,
                'pred_5d_correct': pred_5d_correct,
                'pred_20d_correct': pred_20d_correct,
                'pred_1d_direction': pred_1d.get('predicted_direction'),
                'pred_5d_direction': pred_5d.get('predicted_direction'),
                'pred_20d_direction': pred_20d.get('predicted_direction') if pred_20d else None,
                'prediction_date': prediction_date,
                'both_correct_rate': TRANSMISSION_ACCURACY['both_correct_rate'],
                'independent_20d_rate': TRANSMISSION_ACCURACY['independent_20d_rate']
            }

        return {
            'transmission_mode': pred_1d_correct and pred_5d_correct,
            'pred_1d_correct': pred_1d_correct,
            'pred_5d_correct': pred_5d_correct,
            'pred_20d_correct': pred_20d_correct,
            'pred_1d_direction': pred_1d.get('predicted_direction'),
            'pred_5d_direction': pred_5d.get('predicted_direction'),
            'pred_20d_direction': pred_20d.get('predicted_direction') if pred_20d else None,
            'prediction_date': prediction_date,
            'both_correct_rate': TRANSMISSION_ACCURACY['both_correct_rate'],
            'independent_20d_rate': TRANSMISSION_ACCURACY['independent_20d_rate']
        }

    except Exception as e:
        print(f"⚠️ 检查传导模式失败: {e}")
        return {'transmission_mode': False, 'pred_1d_correct': None, 'pred_5d_correct': None, 'pred_20d_correct': None,
                'pred_1d_direction': None, 'pred_5d_direction': None, 'pred_20d_direction': None, 'prediction_date': None}


def get_sector_type(sector_code):
    """
    获取板块类型（周期/防御）

    参数:
    - sector_code: 板块代码

    返回:
    - str: '周期' 或 '防御'，未知则返回 '-'
    """
    if sector_code in SECTOR_TYPES:
        return SECTOR_TYPES[sector_code]['type']
    return '-'


def get_pattern_action(pattern, is_hsi=False):
    """
    根据三周期模式获取交易建议

    参数:
    - pattern: 三周期模式字符串（如 '111', '110' 等）
    - is_hsi: 是否为恒指预测（默认False，使用个股模式）

    返回:
    - dict: 包含模式名称、操作建议、胜率等信息
    """
    patterns = HSI_THREE_HORIZON_PATTERNS if is_hsi else THREE_HORIZON_PATTERNS
    if pattern in patterns:
        return patterns[pattern]
    return {'name': '未知', 'action': '观望', 'win_rate': '-', 'avg_return': '-', 'confidence': '低'}


def load_risk_reward_data(json_path='data/risk_reward_results.json'):
    """
    加载风险回报率分析结果

    参数:
    - json_path: JSON文件路径

    返回:
    - dict: {股票代码: 风险回报率数据} 的字典
    """
    import json

    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为字典 {股票代码: 数据}
        result = {}
        for item in data:
            result[item['code']] = item
        print(f"  ✅ 加载风险回报率数据: {len(result)} 只股票")
        return result
    except Exception as e:
        print(f"  ⚠️ 加载风险回报率数据失败: {e}")
        return {}


def load_historical_profit_loss_ratio(output_dir='output'):
    """
    从最新的 Walk-forward 验证结果计算每只股票的历史盈亏比和期望收益

    使用历史预测表现法：
    - 盈亏比 = 正确预测的平均盈利 / 错误预测的平均亏损
    - 期望收益 = 胜率 × 平均盈利 - (1-胜率) × 平均亏损

    参数:
    - output_dir: 输出目录，包含 walk-forward 验证结果

    返回:
    - dict: {股票代码: {'profit_loss_ratio': 盈亏比, 'expected_return': 期望收益, 'win_rate': 胜率, ...}}
    """
    import glob

    # 查找最新的 prediction_analysis.csv 文件
    search_patterns = [
        os.path.join(output_dir, '*_catboost_20d/prediction_analysis.csv'),
        os.path.join(output_dir, 'walk_forward_catboost_20d_*/prediction_analysis.csv'),
        os.path.join(output_dir, 'walk_forward_catboost_20d_*/*_catboost_20d/prediction_analysis.csv'),
    ]

    prediction_files = []
    for pattern in search_patterns:
        prediction_files.extend(glob.glob(pattern))

    if not prediction_files:
        print(f"  ⚠️ 未找到 Walk-forward 预测分析文件")
        return {}

    # 使用最新的文件
    latest_file = max(prediction_files, key=lambda x: os.path.getmtime(x))
    print(f"  📁 加载历史盈亏比数据: {os.path.basename(os.path.dirname(latest_file))}")

    try:
        df = pd.read_csv(latest_file)

        # 只分析预测UP的交易（实际买入场景）
        up_preds = df[df['Predict_Direction'] == 'UP'].copy()

        results = {}
        for stock_code in up_preds['Stock_Code'].unique():
            stock_data = up_preds[up_preds['Stock_Code'] == stock_code]

            if len(stock_data) < 10:  # 样本太少不统计
                continue

            # 正确预测（实际收益>0）
            correct = stock_data[stock_data['Actual_Return'] > 0]
            wrong = stock_data[stock_data['Actual_Return'] <= 0]

            n_samples = len(stock_data)
            n_correct = len(correct)
            n_wrong = len(wrong)

            if n_correct > 0 and n_wrong > 0:
                avg_profit = correct['Actual_Return'].mean()
                avg_loss = abs(wrong['Actual_Return'].mean())
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
                    'profit_loss_grade': pl_grade,
                    'expected_return': expected_return,
                    'expected_return_str': f'{expected_return:+.2%}',
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'n_samples': n_samples
                }

        print(f"  ✅ 加载历史盈亏比数据: {len(results)} 只股票")
        return results

    except Exception as e:
        print(f"  ⚠️ 加载历史盈亏比数据失败: {e}")
        return {}


# 全局模型缓存（避免重复加载）
_model_cache = {}


def load_multi_horizon_models():
    """
    加载三周期预测模型（1d, 5d, 20d）

    返回:
    - dict: 包含三个模型的字典，失败则为 None
    """
    global _model_cache

    if not ML_MODEL_AVAILABLE:
        print("⚠️ ML模型模块不可用，无法进行三周期预测")
        return None

    # 检查缓存
    if _model_cache.get('loaded'):
        return _model_cache

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    models = {}
    model_files = {
        1: os.path.join(data_dir, 'ml_trading_model_catboost_1d.pkl'),
        5: os.path.join(data_dir, 'ml_trading_model_catboost_5d.pkl'),
        20: os.path.join(data_dir, 'ml_trading_model_catboost_20d.pkl'),
    }

    missing_models = []
    for horizon, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                model = CatBoostModel()
                model.load_model(filepath)
                models[horizon] = model
                print(f"  ✅ 加载 {horizon}d 模型成功")
            except Exception as e:
                print(f"  ⚠️ 加载 {horizon}d 模型失败: {e}")
                missing_models.append(horizon)
        else:
            print(f"  ⚠️ {horizon}d 模型文件不存在: {filepath}")
            missing_models.append(horizon)

    if len(models) < 3:
        print(f"⚠️ 缺少模型: {missing_models}，将使用可用的模型进行预测")

    if len(models) == 0:
        return None

    # 缓存模型
    _model_cache = models
    _model_cache['loaded'] = True

    return models


def predict_three_horizons(stock_code, models=None):
    """
    对单只股票进行三周期预测

    参数:
    - stock_code: 股票代码（如 '0005.HK'）
    - models: 模型字典（如果为 None 则自动加载）

    返回:
    - dict: 包含三个周期预测结果和模式，失败返回 None
    """
    if models is None:
        models = load_multi_horizon_models()

    if models is None:
        return None

    result = {
        'code': stock_code,
        'predictions': {},
        'pattern': None,
        'pattern_info': None
    }

    # 对每个周期进行预测
    success_count = 0
    for horizon in [1, 5, 20]:
        if horizon in models:
            try:
                pred = models[horizon].predict(stock_code)
                if pred:
                    result['predictions'][horizon] = {
                        'prediction': pred.get('prediction', 0),
                        'probability': pred.get('probability', 0.5),
                        'direction': '↑' if pred.get('prediction') == 1 else '↓'
                    }
                    success_count += 1
            except KeyError as e:
                # 特征不匹配，跳过此周期
                print(f"  ⚠️ {stock_code} {horizon}d 模型特征不匹配，跳过")
                result['predictions'][horizon] = None
            except Exception as e:
                print(f"  ⚠️ 预测 {stock_code} {horizon}d 失败: {str(e)[:50]}")
                result['predictions'][horizon] = None
        else:
            result['predictions'][horizon] = None

    # 只有当所有三个周期都成功预测时才计算模式
    if success_count == 3 and all(result['predictions'].get(h) for h in [1, 5, 20]):
        pred_1d = result['predictions'][1]['prediction']
        pred_5d = result['predictions'][5]['prediction']
        pred_20d = result['predictions'][20]['prediction']

        pattern = f"{'1' if pred_1d == 1 else '0'}{'1' if pred_5d == 1 else '0'}{'1' if pred_20d == 1 else '0'}"
        result['pattern'] = pattern
        result['pattern_info'] = get_pattern_action(pattern)
        return result

    # 如果有任何预测失败，返回 None
    return None


def safe_float_format(value, format_spec='.2f', default=''):
    """
    安全地格式化浮点数值，处理可能的字符串或非数值类型
    
    参数:
    - value: 要格式化的值
    - format_spec: 格式化规格，默认为'.2f'（可以是 '+.2f' 等带符号的格式）
    - default: 格式化失败时的默认返回值，默认为空字符串
    
    返回:
    - 格式化后的字符串，或默认值
    """
    try:
        if pd.isna(value) or value is None or value == '':
            return default
        # 尝试转换为浮点数并格式化
        float_value = float(value)
        # 直接使用 format_spec 作为格式说明符
        format_str = f"{{:{format_spec}}}"
        return format_str.format(float_value)
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        return default


def load_model_accuracy(horizon=20):
    """
    从文件加载模型准确率信息

    参数:
    - horizon: 预测周期（默认20天）

    返回:
    - dict: 包含1天、5天、20天三个周期的CatBoost准确率
      {
        'catboost': {'accuracy': float, 'std': float},  # 指定horizon的准确率
        '1d': {'accuracy': float, 'std': float},
        '5d': {'accuracy': float, 'std': float},
        '20d': {'accuracy': float, 'std': float}
      }
    """
    # 默认准确率值（如果文件不存在）
    default_accuracy = {
        'catboost': {'accuracy': 0.6101, 'std': 0.0219},
        '1d': {'accuracy': 0.5100, 'std': 0.0500},
        '5d': {'accuracy': 0.5600, 'std': 0.0400},
        '20d': {'accuracy': 0.6101, 'std': 0.0219}
    }

    accuracy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_accuracy.json')

    try:
        if os.path.exists(accuracy_file):
            import json
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = {}

            # 加载指定horizon的准确率
            catboost_key = f'catboost_{horizon}d'
            if catboost_key in data:
                result['catboost'] = {
                    'accuracy': data[catboost_key].get('accuracy', default_accuracy['catboost']['accuracy']),
                    'std': data[catboost_key].get('std', default_accuracy['catboost']['std'])
                }
            else:
                result['catboost'] = default_accuracy['catboost']

            # 加载三个周期的准确率
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
            print(f"   CatBoost 1天: {result['1d']['accuracy']:.2%} (±{result['1d']['std']:.2%})")
            print(f"   CatBoost 5天: {result['5d']['accuracy']:.2%} (±{result['5d']['std']:.2%})")
            print(f"   CatBoost 20天: {result['20d']['accuracy']:.2%} (±{result['20d']['std']:.2%})")
            return result
        else:
            print(f"⚠️ 准确率文件不存在: {accuracy_file}，使用默认值")
            return default_accuracy
    except Exception as e:
        print(f"⚠️ 读取准确率文件失败: {e}，使用默认值")
        return default_accuracy


def extract_llm_recommendations(filepath):
    """
    从大模型建议文件中提取买卖建议，分别提取短期和中期建议
    
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
        
        # 查找"稳健型短期分析"和"稳健型中期分析"的位置
        short_start = content.find("### 📊 ⚖️ 稳健型短期分析（日内/数天）")
        medium_start = content.find("### 📊 📊 稳健型中期分析（数周-数月）")
        
        if short_start == -1:
            short_start = content.find("### 稳健型短期分析")
        
        if medium_start == -1:
            medium_start = content.find("### 稳健型中期分析")
        
        short_content = ""
        medium_content = ""
        
        if short_start != -1:
            if medium_start != -1:
                # 提取短期分析内容（从短期分析标题后到中期分析标题前）
                short_content = content[short_start:medium_start].split('\n', 1)[-1].strip()  # 去掉标题行
            else:
                # 如果没有中期分析，提取到文件末尾
                short_content = content[short_start:].split('\n', 1)[-1].strip()  # 去掉标题行
        
        if medium_start != -1:
            # 提取中期分析内容（从中期分析标题后到文件末尾）
            medium_content = content[medium_start:].split('\n', 1)[-1].strip()  # 去掉标题行
        
        result = {
            'short_term': short_content,
            'medium_term': medium_content
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 提取大模型建议失败: {e}")
        import traceback
        traceback.print_exc()
        return {'short_term': '', 'medium_term': ''}


def extract_ml_predictions(filepath, use_cached_predictions=False):
    """
    从ML预测CSV文件中提取融合模型的预测结果，并进行三周期预测

    参数:
    - filepath: 文本预测文件路径（用于获取日期）
    - use_cached_predictions: 是否使用已缓存的三周期预测CSV文件（跳过模型预测）

    返回:
    - dict: 包含融合模型预测结果的字典
      {
        'ensemble': str,       # 只包含20天概率（用于大模型决策）
        'ensemble_email': str, # 包含三周期预测（用于邮件显示）
      }
    """
    try:
        import pandas as pd
        from datetime import datetime
        import os

        # 从文件路径中提取日期
        date_str = filepath.split('_')[-1].replace('.txt', '')

        # 使用相对路径（从当前脚本位置推导data目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')

        # ========== 获取所有股票收益率数据用于市场情绪过滤 ==========
        # 参考 walk_forward_validation.py 的方案：使用所有股票的收益率计算上涨比例
        market_layer = 'normal'
        dynamic_threshold = 0.50
        up_ratio = 0.50

        try:
            from config import TRAINING_STOCKS as STOCK_LIST
            from data_services.tencent_finance import get_hk_stock_data_tencent
            from ml_services.market_regime import MarketSentimentFilter

            # STOCK_LIST 是字典，取 keys 作为股票代码列表
            stock_codes = list(STOCK_LIST.keys()) if isinstance(STOCK_LIST, dict) else STOCK_LIST
            print(f"  🔄 获取 {len(stock_codes)} 只股票数据用于市场情绪计算...")

            # 收集所有股票的收益率数据
            all_returns = []
            for stock_code in stock_codes:
                try:
                    stock_df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=90)
                    if not stock_df.empty and len(stock_df) >= 2:
                        stock_df['Return_1d'] = stock_df['Close'].pct_change()
                        stock_df['Date'] = pd.to_datetime(stock_df.index).tz_localize(None)
                        all_returns.append(stock_df[['Date', 'Return_1d']])
                except Exception:
                    pass

            if all_returns:
                returns_df = pd.concat(all_returns, ignore_index=True)
                print(f"  ✅ 收益率数据获取成功，共 {len(returns_df)} 条记录")

                # 初始化市场情绪过滤器
                # 收市后预测使用 lookback_days=0，因为当日收盘价已知，可计算当日市场上涨比例
                # Walk-forward 验证使用 lookback_days=1，模拟开盘前预测场景
                market_filter = MarketSentimentFilter(lookback_days=0)
                market_filter.prepare_market_schedule(returns_df, date_col='Date', ret_col='Return_1d')

                # 获取当日的市场情绪
                threshold, layer, ratio = market_filter.get_threshold(date_str)
                market_layer = layer
                dynamic_threshold = threshold
                up_ratio = ratio

                layer_names = {
                    'extreme_bear': '🔴 极端熊市',
                    'bear': '🟠 熊市',
                    'weak': '🟡 弱震荡',
                    'normal': '🟢 正常市场'
                }
                print(f"  ✅ 市场情绪: {layer_names.get(layer, layer)}, 上涨比例: {ratio:.1%}, 阈值: {threshold:.2f}")
            else:
                print(f"  ⚠️ 无法获取股票收益率数据，使用默认市场情绪")
        except Exception as e:
            print(f"  ⚠️ 市场情绪过滤器初始化失败: {e}")

        # 读取 CatBoost 单模型预测结果
        catboost_csv = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_20d.csv')

        result = {
            'ensemble': '',
            'ensemble_email': ''
        }

        # 读取 CatBoost 预测结果
        if os.path.exists(catboost_csv):
            df_catboost = pd.read_csv(catboost_csv)
            df_catboost_sorted = df_catboost.sort_values('probability', ascending=False)

            # 三周期预测结果
            three_horizon_results = {}

            # 如果指定使用缓存预测，尝试读取已有的三周期预测CSV文件
            csv_1d = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_1d.csv')
            csv_5d = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_5d.csv')
            csv_20d = catboost_csv  # 20d文件就是主文件

            if use_cached_predictions and all(os.path.exists(f) for f in [csv_1d, csv_5d]):
                print("  📁 读取已缓存的三周期预测文件...")
                try:
                    df_1d = pd.read_csv(csv_1d)
                    df_5d = pd.read_csv(csv_5d)

                    for stock_code in df_catboost['code'].tolist():
                        try:
                            row_1d = df_1d[df_1d['code'] == stock_code]
                            row_5d = df_5d[df_5d['code'] == stock_code]
                            row_20d = df_catboost[df_catboost['code'] == stock_code]

                            if len(row_1d) > 0 and len(row_5d) > 0 and len(row_20d) > 0:
                                pred_1d = int(row_1d.iloc[0]['prediction'])
                                prob_1d = float(row_1d.iloc[0]['probability'])
                                pred_5d = int(row_5d.iloc[0]['prediction'])
                                prob_5d = float(row_5d.iloc[0]['probability'])
                                pred_20d = int(row_20d.iloc[0]['prediction'])
                                prob_20d = float(row_20d.iloc[0]['probability'])

                                pattern = f"{'1' if pred_1d == 1 else '0'}{'1' if pred_5d == 1 else '0'}{'1' if pred_20d == 1 else '0'}"
                                pattern_info = get_pattern_action(pattern)

                                three_horizon_results[stock_code] = {
                                    'code': stock_code,
                                    'predictions': {
                                        1: {'prediction': pred_1d, 'probability': prob_1d, 'direction': '↑' if pred_1d == 1 else '↓'},
                                        5: {'prediction': pred_5d, 'probability': prob_5d, 'direction': '↑' if pred_5d == 1 else '↓'},
                                        20: {'prediction': pred_20d, 'probability': prob_20d, 'direction': '↑' if pred_20d == 1 else '↓'}
                                    },
                                    'pattern': pattern,
                                    'pattern_info': pattern_info
                                }
                        except Exception:
                            pass
                    print(f"  ✅ 成功读取三周期预测: {len(three_horizon_results)} 只股票")
                except Exception as e:
                    print(f"  ⚠️ 读取缓存预测文件失败: {e}，将使用模型预测")
                    three_horizon_results = {}

            # 如果没有使用缓存或缓存读取失败，使用模型预测
            if not three_horizon_results and ML_MODEL_AVAILABLE:
                print("  🔄 加载三周期预测模型...")
                three_horizon_models = load_multi_horizon_models()
                if three_horizon_models:
                    print(f"  ✅ 成功加载 {len([k for k in three_horizon_models.keys() if k != 'loaded'])} 个模型")
                    print("  🔄 进行三周期预测...")
                    for stock_code in df_catboost['code'].tolist():
                        try:
                            pred_result = predict_three_horizons(stock_code, three_horizon_models)
                            if pred_result:
                                three_horizon_results[stock_code] = pred_result
                        except Exception as e:
                            print(f"  ⚠️ 预测 {stock_code} 失败: {e}")
                    print(f"  ✅ 完成三周期预测: {len(three_horizon_results)} 只股票")
                else:
                    print("  ⚠️ 无法加载三周期模型，将仅显示20天预测")

            # ========== 计算筹码分布（用于邮件表格）==========
            chip_data = {}
            if TECHNICAL_ANALYSIS_AVAILABLE:
                try:
                    from data_services.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    for stock_code in df_catboost['code'].tolist():
                        try:
                            # 获取股票数据（60天）
                            stock_df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=60)
                            if not stock_df.empty and len(stock_df) >= 20:
                                chip_result = analyzer.get_chip_distribution(stock_df)
                                if chip_result:
                                    chip_data[stock_code] = chip_result
                        except Exception as e:
                            print(f"  ⚠️ 计算 {stock_code} 筹码分布失败: {e}")
                            chip_data[stock_code] = None
                    print(f"  ✅ 筹码分布计算完成: {len([k for k, v in chip_data.items() if v])} 只股票")
                except Exception as e:
                    print(f"  ⚠️ 筹码分布计算失败: {e}")

            # ========== 计算网络洞察（用于邮件展示）==========
            network_insights = {}
            stock_codes = df_catboost['code'].tolist()  # 提前定义，避免后续引用错误
            try:
                from data_services.network_features import get_network_calculator
                calculator = get_network_calculator()
                network_insights = calculator.calculate_network_insights(stock_codes)
                if network_insights and '_meta' in network_insights:
                    meta = network_insights['_meta']
                    print(f"  ✅ 网络洞察计算完成: {len(network_insights)-1} 只股票, {meta.get('community_count', 0)} 个社区")
            except Exception as e:
                print(f"  ⚠️ 网络洞察计算失败: {e}")

            # ========== 计算波动率网络密度预警 ==========
            volatility_density_info = {}
            try:
                from data_services.network_features import get_volatility_density_calculator
                vol_calculator = get_volatility_density_calculator()
                volatility_density_info = vol_calculator.calculate_volatility_network_density(stock_codes)
                if volatility_density_info and volatility_density_info.get('current_density', 0) > 0:
                    print(f"  ✅ 波动率网络密度: {volatility_density_info.get('current_density', 0):.4f}")
            except Exception as e:
                print(f"  ⚠️ 波动率网络密度计算失败: {e}")

            # ========== 1. 构建传给大模型的JSON数据（包含20天概率+筹码阻力）==========
            import json as json_module

            llm_stock_list = []
            for _, row in df_catboost_sorted.iterrows():
                stock_code = row['code']

                # 获取板块名称和类型
                sector_name = '-'
                sector_type = '-'
                if stock_code in STOCK_SECTOR_MAPPING:
                    sector_code = STOCK_SECTOR_MAPPING[stock_code].get('sector', '')
                    if sector_code and sector_code in SECTOR_NAME_MAPPING:
                        sector_name = SECTOR_NAME_MAPPING[sector_code]
                    if sector_code:
                        sector_type = get_sector_type(sector_code)

                # 20天预测概率
                probability = float(row['probability'])

                # 根据市场情绪和概率判断方向
                if market_layer == 'extreme_bear':
                    # 极端熊市：暂停所有看涨信号
                    direction = "暂停"
                    probability_display = f"{probability:.2f} (暂停)"
                    include_in_llm = False  # 不传给大模型
                elif market_layer == 'bear':
                    # 熊市：提高阈值到 0.70
                    if probability >= 0.70:
                        direction = "上涨"
                        probability_display = f"{probability:.2f} (高置信)"
                        include_in_llm = True
                    elif probability >= 0.50:
                        direction = "观望"
                        probability_display = f"{probability:.2f} (降级)"
                        include_in_llm = True  # 传给大模型但标注降级
                    else:
                        direction = "下跌"
                        probability_display = f"{probability:.2f}"
                        include_in_llm = True
                elif market_layer == 'weak':
                    # 弱震荡：提高阈值到 0.65
                    if probability >= 0.65:
                        direction = "上涨"
                        probability_display = f"{probability:.2f}"
                        include_in_llm = True
                    elif probability >= 0.50:
                        direction = "观望"
                        probability_display = f"{probability:.2f} (降级)"
                        include_in_llm = True
                    else:
                        direction = "下跌"
                        probability_display = f"{probability:.2f}"
                        include_in_llm = True
                else:
                    # 正常市场：标准阈值 0.50
                    if probability > 0.60:
                        direction = "上涨"
                    elif probability > 0.50:
                        direction = "观望"
                    else:
                        direction = "下跌"
                    probability_display = f"{probability:.2f}"
                    include_in_llm = True

                # 计算筹码阻力
                resistance_level = 'N/A'
                if stock_code in chip_data and chip_data[stock_code]:
                    chip_result = chip_data[stock_code]
                    resistance_ratio = chip_result.get('resistance_ratio', 0)
                    if resistance_ratio < 0.3:
                        resistance_level = '低'
                    elif resistance_ratio < 0.6:
                        resistance_level = '中'
                    else:
                        resistance_level = '高'

                # 根据市场情绪决定是否传给大模型
                if include_in_llm:
                    llm_stock_list.append({
                        'code': stock_code,
                        'name': row['name'],
                        'sector': sector_name,
                        'sector_type': sector_type,
                        'prediction_20d': direction,
                        'probability_20d': round(probability, 4),
                        'probability_display': probability_display,  # 市场情绪调整后的显示
                        'current_price': float(row['current_price']) if pd.notna(row.get('current_price')) else None,
                        'chip_resistance': resistance_level,
                        'market_layer': market_layer,
                        'dynamic_threshold': dynamic_threshold
                    })

            # 构建JSON格式文本
            catboost_text_llm = "【CatBoost模型预测结果（20天）- JSON格式】\n"
            catboost_text_llm += f"预测日期: {date_str}\n\n"
            catboost_text_llm += "```json\n"
            catboost_text_llm += json_module.dumps(llm_stock_list, ensure_ascii=False, indent=2)
            catboost_text_llm += "\n```\n\n"
            catboost_text_llm += "**字段说明**：\n"
            catboost_text_llm += "- `probability_20d`: 20天上涨概率（>0.60=高置信度，0.50-0.60=中等，≤0.50=低）\n"
            catboost_text_llm += "- `probability_display`: 市场情绪调整后的概率显示（含高置信/降级/暂停标注）\n"
            catboost_text_llm += "- `chip_resistance`: 筹码阻力（低=拉升容易，中=注意风险，高=拉升困难）\n"
            catboost_text_llm += "- `market_layer`: 市场情绪层级（extreme_bear/bear/weak/normal）\n"
            catboost_text_llm += "- `dynamic_threshold`: 动态阈值（根据市场情绪调整）\n"
            catboost_text_llm += "- **使用建议**：probability_20d高 + chip_resistance低 + market_layer=normal = 更可靠信号\n"

            # ========== 2. 构建邮件表格（保留三周期预测+筹码分布，用于用户查看）==========
            if three_horizon_results and len(three_horizon_results) > 0:
                # 加载风险回报率数据
                risk_reward_data = load_risk_reward_data()

                # 加载历史盈亏比数据（基于 Walk-forward 验证结果）
                historical_pl_data = load_historical_profit_loss_ratio()

                # 获取传导模式验证日期（取第一只股票的日期）
                first_stock = df_catboost_sorted.iloc[0]['code'] if len(df_catboost_sorted) > 0 else None
                transmission_date = None
                if first_stock:
                    first_transmission = check_transmission_mode(first_stock, date_str)
                    transmission_date = first_transmission.get('prediction_date')

                # 三周期预测表格（含筹码分布+风险回报率）
                # 在邮件开头显示市场情绪
                layer_names_email = {
                    'extreme_bear': '🔴 极端熊市 - 暂停交易',
                    'bear': '🟠 熊市 - 需概率≥0.70',
                    'weak': '🟡 弱震荡 - 需概率≥0.65',
                    'normal': '🟢 正常市场'
                }

                catboost_text_email = "【CatBoost模型三周期预测结果】\n"
                catboost_text_email += f"数据日期: {date_str}\n"
                catboost_text_email += f"**市场情绪**: {layer_names_email.get(market_layer, market_layer)}\n"
                catboost_text_email += f"**今日上涨比例**: {up_ratio:.1%}\n"
                catboost_text_email += f"**动态阈值**: {dynamic_threshold:.2f}\n"
                if transmission_date:
                    catboost_text_email += f"传导模式验证日期: {transmission_date}\n"
                catboost_text_email += "\n全部股票预测结果（按20天概率排序）:\n\n"
                catboost_text_email += "| 股票代码 | 股票名称 | 现价 | 涨跌幅 | 板块名称 | 类型 | 1天预测 | 5天预测 | 20天预测 | 市场调整 | 模式 | 交易建议 | 历史胜率 | 传导模式 | 筹码阻力 | 盈亏比 | 期望收益 | 风险得分 | 回报得分 | 综合得分 | 风险建议 | 网络洞察 |\n"
                catboost_text_email += "|----------|----------|------|--------|----------|------|--------|--------|---------|----------|------|---------|------|----------|----------|----------|----------|----------|----------|----------|----------|----------|\n"

                for _, row in df_catboost_sorted.iterrows():
                    stock_code = row['code']

                    # 获取涨跌幅
                    change_pct_str = '-'
                    try:
                        realtime_data = get_stock_realtime_data(stock_code)
                        if realtime_data and 'change_pct' in realtime_data:
                            change_pct = realtime_data['change_pct']
                            if change_pct > 0:
                                change_pct_str = f"📈 +{change_pct:.2f}%"
                            elif change_pct < 0:
                                change_pct_str = f"📉 {change_pct:.2f}%"
                            else:
                                change_pct_str = f"{change_pct:.2f}%"
                    except:
                        pass

                    # 获取板块名称和类型
                    sector_name = '-'
                    sector_type = '-'
                    if stock_code in STOCK_SECTOR_MAPPING:
                        sector_code = STOCK_SECTOR_MAPPING[stock_code].get('sector', '')
                        if sector_code and sector_code in SECTOR_NAME_MAPPING:
                            sector_name = SECTOR_NAME_MAPPING[sector_code]
                        if sector_code:
                            sector_type = get_sector_type(sector_code)

                    if stock_code in three_horizon_results:
                        pred = three_horizon_results[stock_code]
                        preds = pred['predictions']

                        # 1天预测（三色系统：概率>=60%绿色，50-60%橙色，<50%红色）
                        pred_1d = preds.get(1, {'direction': '-', 'probability': 0.5})
                        direction_1d = pred_1d['direction']
                        prob_1d = pred_1d['probability']
                        if direction_1d == '↑':
                            if prob_1d >= 0.60:
                                p1d_str = f'<span style="color: #16a34a; font-weight: bold;">↑</span> {prob_1d:.2f}'  # 亮绿色
                            else:
                                p1d_str = f'<span style="color: #ea580c; font-weight: bold;">↑</span> {prob_1d:.2f}'  # 亮橙色
                        elif direction_1d == '↓':
                            p1d_str = f'<span style="color: #dc2626; font-weight: bold;">↓</span> {prob_1d:.2f}'  # 亮红色
                        else:
                            p1d_str = f"{direction_1d} {prob_1d:.2f}"

                        # 5天预测（三色系统：概率>=60%绿色，50-60%橙色，<50%红色）
                        pred_5d = preds.get(5, {'direction': '-', 'probability': 0.5})
                        direction_5d = pred_5d['direction']
                        prob_5d = pred_5d['probability']
                        if direction_5d == '↑':
                            if prob_5d >= 0.60:
                                p5d_str = f'<span style="color: #16a34a; font-weight: bold;">↑</span> {prob_5d:.2f}'  # 亮绿色
                            else:
                                p5d_str = f'<span style="color: #ea580c; font-weight: bold;">↑</span> {prob_5d:.2f}'  # 亮橙色
                        elif direction_5d == '↓':
                            p5d_str = f'<span style="color: #dc2626; font-weight: bold;">↓</span> {prob_5d:.2f}'  # 亮红色
                        else:
                            p5d_str = f"{direction_5d} {prob_5d:.2f}"

                        # 20天预测（三色系统：概率>=60%绿色，50-60%橙色，<50%红色）
                        pred_20d = preds.get(20, {'direction': '-', 'probability': 0.5})
                        direction_20d = pred_20d['direction']
                        prob_20d = pred_20d['probability']
                        if direction_20d == '↑':
                            if prob_20d >= 0.60:
                                p20d_str = f'<span style="color: #16a34a; font-weight: bold;">↑</span> {prob_20d:.2f}'  # 亮绿色
                            else:
                                p20d_str = f'<span style="color: #ea580c; font-weight: bold;">↑</span> {prob_20d:.2f}'  # 亮橙色
                        elif direction_20d == '↓':
                            p20d_str = f'<span style="color: #dc2626; font-weight: bold;">↓</span> {prob_20d:.2f}'  # 亮红色
                        else:
                            p20d_str = f"{direction_20d} {prob_20d:.2f}"

                        # 模式和交易建议
                        pattern = pred.get('pattern', '-')
                        pattern_info = pred.get('pattern_info', {})
                        action = pattern_info.get('action', '观望')
                        win_rate = pattern_info.get('win_rate', '-')

                        # 格式化模式显示
                        pattern_display = pattern
                        if pattern != '-' and pattern in THREE_HORIZON_PATTERNS:
                            pattern_name = THREE_HORIZON_PATTERNS[pattern]['name']
                            pattern_display = f"{pattern_name}({pattern})"

                        # 计算筹码阻力标识
                        resistance_icon = 'N/A'
                        if stock_code in chip_data and chip_data[stock_code]:
                            chip_result = chip_data[stock_code]
                            resistance_ratio = chip_result.get('resistance_ratio', 0)
                            if resistance_ratio < 0.3:
                                resistance_icon = '✅低'
                            elif resistance_ratio < 0.6:
                                resistance_icon = '⚠️中'
                            else:
                                resistance_icon = '🔴高'

                        # 检查传导模式
                        transmission_info = check_transmission_mode(stock_code, date_str)
                        transmission_display = format_transmission_display(transmission_info)

                        # 获取风险回报率数据
                        rr_info = risk_reward_data.get(stock_code, {})
                        rr_comprehensive = rr_info.get('comprehensive_score', '-')
                        rr_risk = rr_info.get('risk_score', '-')
                        rr_return = rr_info.get('return_score', '-')
                        rr_suggestion = rr_info.get('suggestion', '-')

                        # 获取历史盈亏比数据
                        pl_info = historical_pl_data.get(stock_code, {})
                        profit_loss_ratio_str = pl_info.get('profit_loss_ratio_str', '-')
                        expected_return_str = pl_info.get('expected_return_str', '-')
                        pl_grade = pl_info.get('profit_loss_grade', '')

                        # 格式化盈亏比显示（带等级标识）
                        if profit_loss_ratio_str != '-':
                            pl_display = f"{profit_loss_ratio_str} {pl_grade}"
                        else:
                            pl_display = '-'

                        # 获取网络洞察
                        network_insight_str = network_insights.get(stock_code, {}).get('insight_str', '未知')

                        # 计算市场调整显示
                        probability_20d = float(row['probability'])
                        if market_layer == 'extreme_bear':
                            market_adjust_display = '🔴暂停'
                        elif market_layer == 'bear':
                            if probability_20d >= 0.70:
                                market_adjust_display = '🟠高置信'
                            elif probability_20d >= 0.50:
                                market_adjust_display = '🟠降级'
                            else:
                                market_adjust_display = '-'
                        elif market_layer == 'weak':
                            if probability_20d >= 0.65:
                                market_adjust_display = '🟡通过'
                            elif probability_20d >= 0.50:
                                market_adjust_display = '🟡降级'
                            else:
                                market_adjust_display = '-'
                        else:
                            market_adjust_display = '🟢正常'

                        price_str = f"{row['current_price']:.2f}" if pd.notna(row.get('current_price')) else '-'
                        catboost_text_email += f"| {stock_code} | {row['name']} | {price_str} | {change_pct_str} | {sector_name} | {sector_type} | {p1d_str} | {p5d_str} | {p20d_str} | {market_adjust_display} | {pattern_display} | {action} | {win_rate} | {transmission_display} | {resistance_icon} | {pl_display} | {expected_return_str} | {rr_risk} | {rr_return} | {rr_comprehensive} | {rr_suggestion} | {network_insight_str} |\n"

                # 添加三周期模式统计
                catboost_text_email += f"\n**三周期模式统计**：\n"
                pattern_counts = {}
                for pred in three_horizon_results.values():
                    pattern = pred.get('pattern')
                    if pattern:
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

                for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
                    pattern_info = THREE_HORIZON_PATTERNS.get(pattern, {})
                    pattern_name = pattern_info.get('name', '未知')
                    catboost_text_email += f"- {pattern_name}({pattern}): {count} 只\n"

                # 添加三色预测说明
                catboost_text_email += f"\n**三周期预测颜色说明**：\n"
                catboost_text_email += "- <span style=\"color: #16a34a; font-weight: bold;\">↑</span>（亮绿色）：概率 ≥ 60%，高置信度看涨\n"
                catboost_text_email += "- <span style=\"color: #ea580c; font-weight: bold;\">↑</span>（亮橙色）：概率 50-60%，中等置信度看涨\n"
                catboost_text_email += "- <span style=\"color: #dc2626; font-weight: bold;\">↓</span>（亮红色）：概率 < 50%，看跌\n"

                # 添加市场调整说明
                catboost_text_email += f"\n**市场调整说明**：\n"
                catboost_text_email += "- 🟢正常：正常市场环境，使用标准阈值\n"
                catboost_text_email += "- 🟡通过/降级：弱震荡市场，概率≥65%通过，否则降级为观望\n"
                catboost_text_email += "- 🟠高置信/降级：熊市环境，概率≥70%通过，否则降级为观望\n"
                catboost_text_email += "- 🔴暂停：极端熊市，暂停所有看涨信号\n"

                # 添加交易规则说明
                catboost_text_email += f"\n**三周期交易规则说明**：\n"
                catboost_text_email += "- 模式标注 = 1天预测 + 5天预测 + 20天预测（1=涨，0=跌）\n"
                catboost_text_email += f"- 个股传导模式：1天+5天都正确时，20天准确率({TRANSMISSION_ACCURACY['both_correct_rate']}%) > 独立20天({TRANSMISSION_ACCURACY['independent_20d_rate']}%)，提升 +{TRANSMISSION_ACCURACY['improvement']}%\n"
                catboost_text_email += f"- ⚠️ 注意：个股最优模式为\"反弹失败(010)\"，准确率66.32%；\"假突破(101)\"仅50%（随机水平）\n"
                catboost_text_email += f"- 💡 恒指模式：恒指最优模式为\"假突破(101)\"，准确率92.73%（远高于个股）\n"
                catboost_text_email += "- 策略含义：个股预测难度高，建议结合恒指趋势确认\n"

                # 添加筹码分布说明
                catboost_text_email += f"\n**筹码阻力说明**：\n"
                catboost_text_email += "- ✅低：上方筹码 < 30%，拉升容易\n"
                catboost_text_email += "- ⚠️中：上方筹码 30-60%，注意风险\n"
                catboost_text_email += "- 🔴高：上方筹码 > 60%，拉升困难\n"

                # 添加盈亏比说明
                catboost_text_email += f"\n**历史盈亏比说明**（基于Walk-forward验证）：\n"
                catboost_text_email += "- 盈亏比 = 正确预测平均盈利 / 错误预测平均亏损\n"
                catboost_text_email += "- 期望收益 = 胜率 × 平均盈利 - (1-胜率) × 平均亏损\n"
                catboost_text_email += "- ⭐⭐⭐：盈亏比 ≥ 3:1，优秀\n"
                catboost_text_email += "- ⭐⭐：盈亏比 2-3:1，良好\n"
                catboost_text_email += "- ⭐：盈亏比 1.5-2:1，一般\n"
                catboost_text_email += "- ⚠️：盈亏比 < 1.5:1，较差\n"
                catboost_text_email += "- 💡 使用建议：优先选择盈亏比 ≥ 2:1 且期望收益 > 0 的股票\n"

                # 添加风险回报率说明
                catboost_text_email += f"\n**风险回报率说明**（稳健型模式）：\n"
                catboost_text_email += "- 综合得分 = 风险得分 × 50% + 回报得分 × 50%\n"
                catboost_text_email += "- ⭐ 优选：综合得分 ≥ 75，风险回报率最佳\n"
                catboost_text_email += "- 🟢 推荐：综合得分 60-75，值得关注\n"
                catboost_text_email += "- 🟡 观察：综合得分 45-60，需谨慎\n"
                catboost_text_email += "- 🔴 暂缓：综合得分 < 45，暂不考虑\n"

                # 添加网络洞察统计表格
                if network_insights and '_meta' in network_insights:
                    from data_services.network_features import get_network_calculator
                    net_calc = get_network_calculator()
                    insights_table = net_calc.generate_insights_table(network_insights)
                    if insights_table:
                        catboost_text_email += insights_table

                # 添加波动率网络密度预警
                if volatility_density_info and volatility_density_info.get('current_density', 0) > 0:
                    from data_services.network_features import get_volatility_density_calculator
                    vol_calc = get_volatility_density_calculator()
                    density_table = vol_calc.generate_warning_table(volatility_density_info)
                    if density_table:
                        catboost_text_email += f"\n{density_table}\n"

                # ========== LSTM-GARCH 混合波动率分析 ==========
                if HYBRID_VOL_MODEL_AVAILABLE:
                    try:
                        catboost_text_email += "\n**📊 混合波动率预测**\n"
                        catboost_text_email += "| 股票代码 | 预测波动率 | 不确定性 | 趋势信号 | 风险建议 |\n"
                        catboost_text_email += "|----------|-----------|----------|---------|----------|\n"

                        # 获取波动率最高的前10只股票
                        vol_data = []
                        for _, row in df_catboost_sorted.head(20).iterrows():
                            stock_code = row['code']
                            try:
                                # 获取股票历史数据（需要足够数据训练LSTM）
                                tencent_code = stock_code.replace('.HK', '').zfill(5)
                                stock_data = get_hk_stock_data_tencent(tencent_code, period_days=250)
                                if stock_data is not None and len(stock_data) > 60:
                                    stock_data['Return_1d'] = stock_data['Close'].pct_change()

                                    # LSTM-GARCH 混合模型
                                    hybrid_model = HybridGARCHLSTM(
                                        lookback=60,
                                        fusion_weight=0.6,
                                        use_lstm=True
                                    )

                                    # 训练和预测
                                    hybrid_model.train(stock_data['Return_1d'], verbose=False)
                                    hybrid_result = hybrid_model.predict(stock_data['Return_1d'])

                                    # 获取最新预测
                                    latest = hybrid_model.get_latest_prediction()
                                    if latest:
                                        hybrid_vol = latest['hybrid_vol']
                                        vol_trend = hybrid_result['Hybrid_Vol_Trend'].iloc[-1]

                                        # 计算不确定性
                                        uncertainty = abs(latest['garch_vol'] - latest['lstm_vol']) if hybrid_model.use_lstm else 0

                                        vol_data.append({
                                            'code': stock_code,
                                            'vol': hybrid_vol,
                                            'uncertainty': uncertainty,
                                            'trend': vol_trend
                                        })
                            except Exception as e:
                                continue

                        # 按波动率排序，取前10
                        vol_data = sorted(vol_data, key=lambda x: x['vol'], reverse=True)[:10]

                        for item in vol_data:
                            # 风险建议
                            if item['vol'] > 0.04:  # 4% 日波动率
                                risk_advice = "🔴 高风险"
                            elif item['vol'] > 0.025:  # 2.5% 日波动率
                                risk_advice = "🟡 中等风险"
                            else:
                                risk_advice = "🟢 低风险"

                            # 趋势信号
                            if item['trend'] > 0.5:
                                trend_signal = "📈 波动上升"
                            elif item['trend'] < -0.5:
                                trend_signal = "📉 波动下降"
                            else:
                                trend_signal = "➡️ 波动稳定"

                            catboost_text_email += f"| {item['code']} | {item['vol']:.2%} | {item['uncertainty']:.4f} | {trend_signal} | {risk_advice} |\n"

                        catboost_text_email += "\n**波动率说明**：\n"
                        catboost_text_email += "- 预测波动率：LSTM-GARCH 混合模型预测的日波动率\n"
                        catboost_text_email += "- 趋势信号：波动率短期趋势，用于判断风险变化\n"
                        catboost_text_email += "- 🔴 高风险：日波动率 > 4%，建议减仓或观望\n"
                        catboost_text_email += "- 🟡 中等风险：日波动率 2.5%-4%，正常交易\n"
                        catboost_text_email += "- 🟢 低风险：日波动率 < 2.5%，可适当加仓\n"

                    except Exception as e:
                        print(f"  ⚠️ 混合波动率分析失败: {e}")
            else:
                # 无三周期预测时，邮件表格也使用简化版本
                catboost_text_email = catboost_text_llm

            # 极端熊市警告
            if market_layer == 'extreme_bear':
                catboost_text_email += """

⚠️ **极端熊市警告**
今日市场上涨比例极低（<20%），根据历史验证，此类市场环境下看涨信号准确率极低。
建议：暂停所有看涨操作，等待市场企稳。
"""

            result['ensemble'] = catboost_text_llm
            result['ensemble_email'] = catboost_text_email
        else:
            print(f"⚠️ CatBoost 预测文件不存在: {catboost_csv}")

            result['ensemble'] = ''
            result['ensemble_email'] = ''
            # 确保这些变量被定义，即使没有预测文件
            three_horizon_results = {}
            chip_data = {}
            network_insights = {}
            risk_reward_data = {}
            historical_pl_data = {}

        # 返回预测结果和额外数据（用于个股分析）
        return {
            'ensemble': result['ensemble'],
            'ensemble_email': result['ensemble_email'],
            'three_horizon_results': three_horizon_results,
            'chip_data': chip_data,
            'network_insights': network_insights,
            'risk_reward_data': risk_reward_data if 'risk_reward_data' in dir() else {},
            'historical_pl_data': historical_pl_data if 'historical_pl_data' in dir() else {},
        }

    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'ensemble': '',
            'ensemble_email': '',
            'three_horizon_results': {},
            'chip_data': {},
            'network_insights': {},
            'risk_reward_data': {},
            'historical_pl_data': {},
        }


def generate_html_email(content, date_str):
    """
    生成HTML格式的邮件内容
    
    参数:
    - content: 综合分析文本内容（Markdown格式）
    - date_str: 分析日期
    
    返回:
    - str: HTML格式的邮件内容
    """
    try:
        import markdown
    except ImportError:
        print("⚠️ 警告：未安装markdown库，使用简单转换")
        # 如果没有安装markdown库，使用简单转换
        simple_html = content.replace('\n', '<br>')
        return simple_html
    
    # 配置markdown扩展，使用更多功能以支持嵌套列表
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])

    # 将Markdown转换为HTML
    html_content = md.convert(content)

    # 为准确率添加颜色样式：三色系统
    import re
    def colorize_accuracy(percentage_str):
        """为准确率百分比添加颜色"""
        try:
            percentage = float(percentage_str)
            if percentage >= 60:
                return f'<span style="color: #16a34a; font-weight: bold;">{percentage_str}%</span>'  # 亮绿色
            elif percentage >= 50:
                return f'<span style="color: #ea580c; font-weight: bold;">{percentage_str}%</span>'  # 亮橙色
            else:
                return f'<span style="color: #dc2626; font-weight: bold;">{percentage_str}%</span>'  # 亮红色
        except ValueError:
            return f'{percentage_str}%'

    # 只匹配加粗的准确率百分比（<strong>XX.XX%</strong> 或 <b>XX.XX%</b>）
    html_content = re.sub(r'<(strong|b)>(\d+\.\d{2})%</(strong|b)>', lambda m: colorize_accuracy(m.group(2)), html_content)
    
    # 组装完整的HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            background-color: #ffffff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 28px;
        }}
        h2 {{
            color: #3498db;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 35px;
            margin-bottom: 20px;
            font-size: 22px;
        }}
        h3 {{
            color: #8e44ad;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 20px;
        }}
        h4 {{
            color: #2c3e50;
            margin: 0 0 12px 0;
            font-size: 18px;
        }}
        p {{
            color: #34495e;
            line-height: 1.8;
            margin: 10px 0;
        }}
        ul, ol {{
            color: #34495e;
            line-height: 1.8;
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .reference-section {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 4px solid #95a5a6;
        }}
        .reference-title {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .reference-content {{
            background: #ffffff;
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .metric-section {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
        }}
        .metric-title {{
            color: #2c3e50;
            font-size: 16px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .metric-item {{
            margin: 8px 0;
            padding-left: 15px;
            border-left: 2px solid #ddd;
        }}
        .risk-section {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }}
        .data-source {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #6c757d;
            font-size: 13px;
            line-height: 1.6;
        }}
        .model-accuracy {{
            background: #d4edda;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
            font-size: 14px;
        }}
        .warning {{
            background: #fff3cd;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 港股综合买卖建议</h1>
        <p style="color: #7f8c8d; font-size: 14px;">📅 分析日期：{date_str}</p>
        
        <div class="content">
            {html_content}
        </div>
        
        <div class="footer">
            <p>📧 本邮件由港股综合分析系统自动生成</p>
            <p>⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def get_sector_analysis():
    """
    获取板块分析数据
    
    返回:
    - dict: 包含板块分析结果
    """
    if not SECTOR_ANALYSIS_AVAILABLE:
        return None
    
    try:
        sector_analyzer = SectorAnalyzer()
        perf_df = sector_analyzer.calculate_sector_performance(period=5)
        
        if perf_df is None or perf_df.empty:
            return None
        
        # 识别龙头股（前3名）
        sector_leaders = {}
        for idx, row in perf_df.iterrows():
            sector_code = row['sector_code']
            
            # 先尝试使用默认市值阈值
            leaders_df = sector_analyzer.identify_sector_leaders(
                sector_code=sector_code,
                top_n=3,
                period=5,
                min_market_cap=100,
                style='moderate'
            )
            
            # 如果没有找到龙头股，可能是市值太小，降低阈值重试
            if leaders_df.empty:
                print(f"  ⚠️ 板块 {row['sector_name']}({sector_code}) 首次查询未找到龙头股，尝试降低市值阈值")
                # 尝试降低市值阈值
                for min_cap in [50, 20, 10, 5, 1]:
                    leaders_df = sector_analyzer.identify_sector_leaders(
                        sector_code=sector_code,
                        top_n=3,
                        period=5,
                        min_market_cap=min_cap,
                        style='moderate'
                    )
                    if not leaders_df.empty:
                        print(f"    找到 {len(leaders_df)} 只龙头股（市值阈值 {min_cap}亿港币）")
                        break
            
            if not leaders_df.empty:
                sector_leaders[sector_code] = []
                for _, leader_row in leaders_df.iterrows():
                    sector_leaders[sector_code].append({
                        'name': leader_row['name'],
                        'code': leader_row['code'],
                        'change_pct': leader_row['change_pct'],
                    })
        
        return {
            'performance': perf_df,
            'leaders': sector_leaders
        }
    except Exception as e:
        print(f"⚠️ 获取板块分析失败: {e}")
        return None


def get_dividend_info():
    """
    获取股息信息
    
    返回:
    - dict: 包含即将除净的港股信息
    """
    if not AKSHARE_AVAILABLE:
        return None
    
    try:
        import time
        
        # 获取自选股列表
        stock_list = WATCHLIST
        all_dividends = []
        
        # 对每只自选股查询股息信息
        for stock_code, stock_name in stock_list.items():
            try:
                # 提取数字部分并格式化为5位（与hsi_email.py保持一致）
                symbol = stock_code.replace('.HK', '')
                if len(symbol) < 5:
                    symbol = symbol.zfill(5)
                elif len(symbol) > 5:
                    symbol = symbol[-5:]
                
                # 使用港股股息接口
                df_dividend = ak.stock_hk_dividend_payout_em(symbol=symbol)
                
                if df_dividend is not None and not df_dividend.empty:
                    # 检查数据列
                    available_columns = df_dividend.columns.tolist()
                    
                    # 创建结果列表
                    result_data = []
                    for _, row in df_dividend.iterrows():
                        try:
                            # 提取关键信息（与hsi_email.py保持一致）
                            ex_date = row.get('除净日', None)
                            dividend_plan = row.get('分红方案', None)
                            record_date = row.get('截至过户日', None)
                            announcement_date = row.get('最新公告日期', None)
                            fiscal_year = row.get('财政年度', None)
                            distribution_type = row.get('分配类型', None)
                            payment_date = row.get('发放日', None)
                            
                            # 只处理有除净日的记录
                            if pd.notna(ex_date):
                                result_data.append({
                                    '股票代码': stock_code,
                                    '股票名称': stock_name,
                                    '除净日': ex_date,
                                    '分红方案': dividend_plan,
                                    '截至过户日': record_date,
                                    '最新公告日期': announcement_date,
                                    '财政年度': fiscal_year,
                                    '分配类型': distribution_type,
                                    '发放日': payment_date
                                })
                        except Exception as e:
                            print(f"⚠️ 处理 {stock_name} 股息数据时出错: {e}")
                            continue
                    
                    if result_data:
                        all_dividends.append(pd.DataFrame(result_data))
                
                # 避免请求过于频繁
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ 获取 {stock_name}({stock_code}) 股息信息失败: {e}")
                continue
        
        if not all_dividends:
            return None
        
        # 合并所有数据
        all_dividends_df = pd.concat(all_dividends, ignore_index=True)
        
        # 转换日期格式
        all_dividends_df['除净日'] = pd.to_datetime(all_dividends_df['除净日'])
        
        # 筛选未来90天内的除净日
        today = datetime.now()
        future_date = today + timedelta(days=90)
        
        upcoming_dividends = all_dividends_df[
            (all_dividends_df['除净日'] >= today) & 
            (all_dividends_df['除净日'] <= future_date)
        ].sort_values('除净日')
        
        if upcoming_dividends.empty:
            return None
        
        # 只取前10个，转换为字典列表
        return upcoming_dividends.head(10).to_dict('records')
        
    except Exception as e:
        print(f"⚠️ 获取股息信息失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_hsi_data_tencent():
    """
    从腾讯财经获取恒生指数实时数据（备用数据源）

    返回:
    - dict: 包含实时成交额数据，或 None（如果获取失败）
      {
        'amount': float,  # 成交额（亿元）
        'amount_raw': float,  # 成交额原始值（万元）
        'price': float,  # 最新价
        'prev_close': float,  # 昨收
        'open': float,  # 今开
        'high': float,  # 最高
        'change_points': float,  # 涨跌点数
        'change_pct': float,  # 涨跌幅
        'update_time': str  # 更新时间
      }
    """
    try:
        import requests
        url = 'https://web.sqt.gtimg.cn/q=r_hkHSI'
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            return None

        # 解析响应数据
        data_str = resp.text.split('\"')[1]
        fields = data_str.split('~')

        # 腾讯财经字段说明：
        # [3] 最新价, [4] 昨收, [5] 今开, [6] 成交额(万元), [21] 涨跌点数, [22] 涨跌幅, [23] 最高
        return {
            'amount': float(fields[6]) / 10000,  # 转换为亿元
            'amount_raw': float(fields[6]),  # 万元
            'price': float(fields[3]),
            'prev_close': float(fields[4]),
            'open': float(fields[5]),
            'high': float(fields[23]),
            'change_points': float(fields[21]),
            'change_pct': float(fields[22]),
            'update_time': fields[20] if len(fields) > 20 else ''
        }
    except Exception as e:
        print(f"⚠️ 腾讯财经获取恒指数据失败: {e}")
        return None


def get_stock_realtime_data(stock_code):
    """
    从腾讯财经获取个股实时数据

    参数:
    - stock_code: 股票代码（如 "0005.HK"）

    返回:
    - dict: 包含实时价格数据，或 None（如果获取失败）
      {
        'price': float,  # 最新价
        'prev_close': float,  # 昨收
        'open': float,  # 今开
        'high': float,  # 最高
        'low': float,  # 最低
        'change_points': float,  # 涨跌点数
        'change_pct': float,  # 涨跌幅
        'volume': float  # 成交量
      }
    """
    try:
        import requests
        # 移除 .HK 后缀，补零到5位
        symbol = stock_code.replace('.HK', '').zfill(5)
        url = f'https://web.sqt.gtimg.cn/q=r_hk{symbol}'
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            return None

        # 解析响应数据
        data_str = resp.text.split('\"')[1]
        fields = data_str.split('~')

        # 腾讯财经港股字段说明：
        # [3] 最新价, [4] 昨收, [5] 今开, [6] 成交量
        # [31] 涨跌点数, [32] 涨跌幅, [33] 最高, [34] 最低
        price = float(fields[3])
        prev_close = float(fields[4])
        return {
            'price': price,
            'prev_close': prev_close,
            'open': float(fields[5]),
            'high': float(fields[33]) if len(fields) > 33 else 0,
            'low': float(fields[34]) if len(fields) > 34 else 0,
            'change_points': float(fields[31]) if len(fields) > 31 else price - prev_close,
            'change_pct': float(fields[32]) if len(fields) > 32 else (price - prev_close) / prev_close * 100,
            'volume': float(fields[6]) if len(fields) > 6 else 0
        }
    except Exception as e:
        print(f"⚠️ 腾讯财经获取 {stock_code} 数据失败: {e}")
        return None


def get_hsi_analysis():
    """
    获取恒生指数分析

    返回:
    - dict: 包含恒生指数技术分析结果
    """
    try:
        hsi_ticker = yf.Ticker("^HSI")
        hist = hsi_ticker.history(period="6mo")

        if hist.empty:
            return None

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest

        # 成交额数据（优先使用腾讯财经实时数据）
        amount = None
        amount_ratio = None
        amount_ma20 = None
        tencent_data = get_hsi_data_tencent()
        if tencent_data:
            amount = tencent_data['amount']

        # 计算基本指标 - 优先使用腾讯财经实时数据
        if tencent_data and tencent_data.get('price', 0) > 0:
            # 使用腾讯财经实时数据
            current_price = tencent_data['price']
            prev_close = tencent_data['prev_close']
            # 腾讯财经接口的涨跌字段可能为0，手动计算
            change_points = current_price - prev_close
            change_pct = (change_points / prev_close * 100) if prev_close != 0 else 0
        else:
            # 回退到 yfinance 数据
            current_price = latest['Close']
            change_points = latest['Close'] - prev['Close']
            change_pct = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] != 0 else 0
            prev_close = prev['Close']

        # 计算RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # 计算移动平均线
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]

        # 计算成交量相关指标
        current_volume = latest['Volume']
        # 5日平均成交量
        volume_ma5 = hist['Volume'].rolling(window=5).mean().iloc[-1]
        # 20日平均成交量
        volume_ma20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
        # 成交量比率（当前成交量 / 20日平均成交量）
        volume_ratio = current_volume / volume_ma20 if volume_ma20 > 0 else 0
        # 成交量变化（相对前一日）
        prev_volume = prev['Volume']
        volume_change_pct = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0

        # 计算成交额比率
        if tencent_data:
            try:
                from data_services.tencent_finance import get_hsi_data_tencent as get_hsi_history
                hsi_history = get_hsi_history(period_days=30)
                if hsi_history is not None and 'Amount' in hsi_history.columns and len(hsi_history) >= 20:
                    amount_ma20 = hsi_history['Amount'].tail(20).mean()
                    amount_ratio = amount / amount_ma20 if amount_ma20 > 0 else 1.0
                    print(f"  📊 成交金额: {amount:.2f}亿, 20日均值: {amount_ma20:.2f}亿, 比率: {amount_ratio:.2f}x")
                else:
                    amount_ratio = None
                    print(f"  📊 成交金额: {amount:.2f}亿（无法计算比率）")
            except Exception as e:
                print(f"  ⚠️ 获取历史成交额失败: {e}")
                amount_ratio = None

        # 趋势判断
        if current_price > ma20 > ma50:
            trend = "强势多头"
        elif current_price > ma20:
            trend = "短期上涨"
        elif current_price > ma50:
            trend = "震荡整理"
        else:
            trend = "弱势空头"

        return {
            'current_price': current_price,
            'change_points': change_points,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'ma20': ma20,
            'ma50': ma50,
            'trend': trend,
            'volume': current_volume,
            'volume_ma5': volume_ma5,
            'volume_ma20': volume_ma20,
            'volume_ratio': volume_ratio,
            'volume_change_pct': volume_change_pct,
            'amount': amount,  # 成交额（亿元），从腾讯财经获取
            'amount_ratio': amount_ratio  # 成交额比率
        }
    except Exception as e:
        print(f"⚠️ 获取恒生指数分析失败: {e}")
        return None


def get_current_market_state():
    """
    获取当前市场状态（实时）- 使用腾讯财经API

    返回:
    dict: 当前市场状态信息，包含 Regime_Duration（市场状态持续时间）
    """
    try:
        from data_services.tencent_finance import get_hsi_data_tencent as get_hsi_history

        # 使用腾讯财经API获取恒指历史数据
        # 注意：需要至少 100 天数据才能计算 Regime_Duration（因为 HMM 需要 20 天滚动窗口）
        hsi_df = get_hsi_history(period_days=100)

        if hsi_df is None or len(hsi_df) < 10:
            # 回退到 yfinance
            print("⚠️ 腾讯财经获取恒指数据失败，回退到 yfinance")
            hsi_ticker = yf.Ticker("^HSI")
            hsi_df = hsi_ticker.history(period="5mo")

            if len(hsi_df) < 10:
                return None

        # 获取实时价格（使用最新收盘价）
        current_hsi = hsi_df['Close'].iloc[-1]
        current_time = hsi_df.index[-1]

        # 转换时区（如果需要）
        if current_time.tz is None:
            current_time = current_time.tz_localize('Asia/Hong_Kong')
        else:
            current_time = current_time.tz_convert('Asia/Hong_Kong')

        # 计算最近20天收益率（使用当前价格）
        if len(hsi_df) >= 20:
            close_20d_ago = hsi_df['Close'].iloc[-20]
            recent_20d_return = (current_hsi - close_20d_ago) / close_20d_ago
        else:
            recent_20d_return = 0

        # 计算最近5天收益率（使用当前价格）
        if len(hsi_df) >= 5:
            close_5d_ago = hsi_df['Close'].iloc[-5]
            recent_5d_return = (current_hsi - close_5d_ago) / close_5d_ago
        else:
            recent_5d_return = 0

        # 计算当前市场状态
        if recent_20d_return > 0.05:
            market_state = 'bull'
            market_state_cn = '牛市'
            market_signal = '📈 强烈看涨'
        elif recent_20d_return < -0.05:
            market_state = 'bear'
            market_state_cn = '熊市'
            market_signal = '📉 强烈看跌'
        elif recent_20d_return > 0.02:
            market_state = 'neutral_bull'
            market_state_cn = '震荡偏涨'
            market_signal = '⬆️ 温和上涨'
        elif recent_20d_return < -0.02:
            market_state = 'neutral_bear'
            market_state_cn = '震荡偏跌'
            market_signal = '⬇️ 温和下跌'
        else:
            market_state = 'neutral'
            market_state_cn = '震荡市'
            market_signal = '➡️ 横盘整理'

        # ========== 新增：获取 Regime_Duration ==========
        regime_duration = None
        regime_state = None
        regime_state_cn = None
        regime_stability = None

        try:
            from data_services.regime_detector import RegimeDetector

            detector = RegimeDetector()
            regime_result = detector.predict(hsi_df)

            if regime_result is not None and len(regime_result) > 0:
                # 获取最新的 Regime_Duration
                regime_duration = int(regime_result['Regime_Duration'].iloc[-1])
                regime_state = int(regime_result['Market_Regime'].iloc[-1])

                # 获取转换概率指标（动态计算）
                regime_transition_prob = float(regime_result['Regime_Transition_Prob'].iloc[-1])
                regime_switch_prob_5d = float(regime_result['Regime_Switch_Prob_5d'].iloc[-1])
                regime_expected_duration = float(regime_result['Regime_Expected_Duration'].iloc[-1])

                # 状态名称映射
                regime_labels = {0: '震荡', 1: '上涨', 2: '下跌'}
                regime_state_cn = regime_labels.get(regime_state, '未知')

                # 判断状态稳定性
                # Regime_Duration < 5: 状态不稳定，频繁转换
                # Regime_Duration 5-15: 中等稳定
                # Regime_Duration > 15: 状态稳定
                if regime_duration < 5:
                    regime_stability = '⚠️ 不稳定'
                elif regime_duration < 15:
                    regime_stability = '🟡 中等'
                else:
                    regime_stability = '✅ 稳定'

                print(f"  ✅ 市场状态检测: {regime_state_cn}，持续 {regime_duration} 天（{regime_stability}）")
                print(f"     转换概率: {regime_transition_prob:.2%}, 5日转换概率: {regime_switch_prob_5d:.2%}")
        except Exception as e:
            print(f"  ⚠️ 获取 Regime_Duration 失败: {e}")
            regime_transition_prob = None
            regime_switch_prob_5d = None
            regime_expected_duration = None

        # 格式化时间（如果存在）
        date_str = current_time.strftime('%Y-%m-%d %H:%M:%S HKT') if current_time else 'N/A'

        return {
            'market_state': market_state,
            'market_state_cn': market_state_cn,
            'market_signal': market_signal,
            'recent_20d_return': recent_20d_return,
            'recent_5d_return': recent_5d_return,
            'current_hsi': current_hsi,
            'date': date_str,
            'regime_duration': regime_duration,
            'regime_state': regime_state,
            'regime_state_cn': regime_state_cn,
            'regime_stability': regime_stability,
            'regime_transition_prob': regime_transition_prob,
            'regime_switch_prob_5d': regime_switch_prob_5d,
            'regime_expected_duration': regime_expected_duration,
        }
    except Exception as e:
        print(f"⚠️ 获取当前市场状态失败: {e}")
        return None


def get_ai_portfolio_analysis():
    """
    获取AI持仓分析
    
    返回:
    - dict: 包含AI持仓分析结果
    """
    try:
        # 读取大模型建议文件
        date_str = datetime.now().strftime('%Y-%m-%d')
        llm_file = f'data/llm_recommendations_{date_str}.txt'
        
        if not os.path.exists(llm_file):
            return None
        
        with open(llm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取AI持仓分析部分
        import re
        ai_analysis_match = re.search(r'【大模型持仓分析】(.*?)(?=\n\n【|$)', content, re.DOTALL)
        
        if ai_analysis_match:
            return ai_analysis_match.group(1).strip()
        
        return None
    except Exception as e:
        print(f"⚠️ 获取AI持仓分析失败: {e}")
        return None


def get_bollinger_position_cn(position: str) -> str:
    """获取布林带位置描述（中文）"""
    if position == 'upper':
        return "上轨（强势）"
    elif position == 'lower':
        return "下轨（弱势）"
    else:
        return "中轨（正常）"


def get_stock_anomalies(use_deep_analysis=True):
    """
    获取股票异常检测结果（P2集成功能）
    
    参数:
    - use_deep_analysis: 是否使用深度分析模式（默认True）
    
    返回:
    - list: 异常列表，格式为：
        [
            {
                'stock': '0700.HK',
                'name': '腾讯控股',
                'type': 'price' or 'volume' or 'isolation_forest',  # 异常类型
                'severity': 'high' or 'medium',
                'z_score': 4.2,
                'value': 320.50,  # 异常值（价格或成交量）
                'current_price': 320.50,
                'change_1d': 3.2,
                'change_5d': -5.8,
                'rsi': 68.5,
                'bollinger_position': 'upper',
                'macd_signal': 'bullish',
                'timestamp': datetime object
            }
        ]
        或 None（如果模块不可用）
    """
    try:
        if not ANOMALY_DETECTION_AVAILABLE:
            print("⚠️ 异常检测模块不可用，跳过异常检测")
            return None
        
        # 调用统一的异常检测入口函数
        result = run_stock_anomaly_detection(
            mode_type='deep' if use_deep_analysis else 'quick',
            time_interval='day',
            verbose=True
        )
        
        anomalies = result['anomalies']
        
        # 按类型和严重程度统计
        if anomalies:
            type_stats = {}
            for anomaly in anomalies:
                anomaly_type = anomaly.get('type', 'unknown')
                if anomaly_type not in type_stats:
                    type_stats[anomaly_type] = {'high': 0, 'medium': 0, 'low': 0}
                severity = anomaly.get('severity', 'low')
                if severity in type_stats[anomaly_type]:
                    type_stats[anomaly_type][severity] += 1
            
            # 打印统计信息
            for anomaly_type, stats in type_stats.items():
                type_name = {
                    'price': '价格',
                    'volume': '成交量',
                    'isolation_forest': 'Isolation Forest',
                    'stock': '多维特征',
                    'stock_hour': '多维特征(小时)'
                }.get(anomaly_type, anomaly_type)
                print(f"   {type_name}异常：高 {stats['high']} 个，中 {stats['medium']} 个，低 {stats['low']} 个")
        
        # 打印异常详情
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'low')
            severity_icon = '🔴' if severity == 'high' else ('⚠️' if severity == 'medium' else 'ℹ️')
            anomaly_type = anomaly.get('type', 'unknown')
            type_name = {
                'price': '价格',
                'volume': '成交量',
                'isolation_forest': 'IF',
                'stock': '多维特征',
                'stock_hour': '多维特征(小时)'
            }.get(anomaly_type, anomaly_type)
            
            if anomaly_type in ('isolation_forest', 'stock', 'stock_hour'):
                print(f"   {severity_icon} {anomaly['name']}（{anomaly['stock']}）：{type_name}{severity}级异常")
            else:
                print(f"   {severity_icon} {anomaly['name']}（{anomaly['stock']}）：{type_name}{severity}级异常（Z-Score: {anomaly['z_score']:.2f}）")
        
        return anomalies
        
    except Exception as e:
        print(f"⚠️ 获取股票异常检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_anomaly_summary_for_llm(anomaly_data):
    """
    将异常数据格式化为大模型可理解的摘要

    参数:
    - anomaly_data: 异常数据列表

    返回:
    - str: 格式化的异常摘要
    """
    if not anomaly_data or len(anomaly_data) == 0:
        return "今日未检测到异常"

    # 按严重程度分组
    high_anomalies = [a for a in anomaly_data if a['severity'] == 'high']
    medium_anomalies = [a for a in anomaly_data if a['severity'] == 'medium']
    low_anomalies = [a for a in anomaly_data if a['severity'] == 'low']

    summary_lines = []

    # 高严重度异常
    if high_anomalies:
        summary_lines.append(f"## 高严重度异常（{len(high_anomalies)}只）")
        for a in high_anomalies:
            stock_code = a['stock']
            stock_name = a['name']
            # 获取板块信息
            sector_info = STOCK_SECTOR_MAPPING.get(stock_code, {})
            sector_name = SECTOR_NAME_MAPPING.get(sector_info.get('sector', ''), '未知板块')

            change_1d = a.get('change_1d', 0)
            rsi = a.get('rsi', 50)
            anomaly_reason = a.get('anomaly_reason', '未知')

            summary_lines.append(f"- {stock_code} {stock_name}（{sector_name}）：当日{change_1d:+.2f}%，RSI {rsi:.1f}，{anomaly_reason}")
        summary_lines.append("")

    # 中严重度异常
    if medium_anomalies:
        summary_lines.append(f"## 中严重度异常（{len(medium_anomalies)}只）")
        for a in medium_anomalies:
            stock_code = a['stock']
            stock_name = a['name']
            sector_info = STOCK_SECTOR_MAPPING.get(stock_code, {})
            sector_name = SECTOR_NAME_MAPPING.get(sector_info.get('sector', ''), '未知板块')

            change_1d = a.get('change_1d', 0)
            change_5d = a.get('change_5d', 0)
            rsi = a.get('rsi', 50)
            anomaly_reason = a.get('anomaly_reason', '未知')

            summary_lines.append(f"- {stock_code} {stock_name}（{sector_name}）：当日{change_1d:+.2f}%，5日{change_5d:+.2f}%，RSI {rsi:.1f}，{anomaly_reason}")
        summary_lines.append("")

    # 低严重度异常
    if low_anomalies:
        summary_lines.append(f"## 低严重度异常（{len(low_anomalies)}只）")
        for a in low_anomalies:
            stock_code = a['stock']
            stock_name = a['name']
            sector_info = STOCK_SECTOR_MAPPING.get(stock_code, {})
            sector_name = SECTOR_NAME_MAPPING.get(sector_info.get('sector', ''), '未知板块')

            change_1d = a.get('change_1d', 0)
            rsi = a.get('rsi', 50)

            summary_lines.append(f"- {stock_code} {stock_name}（{sector_name}）：当日{change_1d:+.2f}%，RSI {rsi:.1f}")
        summary_lines.append("")

    # 添加统计摘要
    oversold_count = sum(1 for a in anomaly_data if a.get('rsi', 50) < 30)
    overbought_count = sum(1 for a in anomaly_data if a.get('rsi', 50) > 70)

    summary_lines.append("## 统计摘要")
    summary_lines.append(f"- 总异常数：{len(anomaly_data)}只")
    summary_lines.append(f"- RSI超卖（<30）：{oversold_count}只")
    summary_lines.append(f"- RSI超买（>70）：{overbought_count}只")

    return "\n".join(summary_lines)


def analyze_anomalies_with_llm(anomaly_data):
    """
    使用大模型分析异常数据

    参数:
    - anomaly_data: 异常数据列表

    返回:
    - str: 大模型生成的分析报告（Markdown格式）
    """
    if not anomaly_data or len(anomaly_data) == 0:
        return "✅ 未检测到异常，市场波动正常"

    # 构建异常数据摘要
    anomaly_summary = format_anomaly_summary_for_llm(anomaly_data)

    # 构建提示词
    prompt = f"""你是港股量化分析师。请分析以下股票异常数据，提供深度洞察。

## 异常数据

{anomaly_summary}

## 分析要求

请从以下角度分析（但不限于）：
1. **整体市场状态**：超卖/超买比例、市场情绪判断
2. **板块异动**：哪些板块集体异动、板块轮动信号
3. **资金流向**：防御性资金、成长性资金的流向
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
        response = chat_with_llm(prompt)
        return response
    except Exception as e:
        print(f"⚠️ 大模型分析异常失败: {e}")
        return f"⚠️ 大模型分析失败: {e}"


def generate_anomaly_report_content(anomaly_data):
    """
    生成异常检测报告内容
    
    参数:
    - anomaly_data: 异常数据列表
    
    返回:
    - str: 生成的 Markdown 格式报告内容
    """
    if not anomaly_data or len(anomaly_data) == 0:
        return "✅ 未检测到异常，市场波动正常\n\n"
    
    content = f"**检测到 {len(anomaly_data)} 个异常**\n\n"
    
    # 按类型和严重程度分类
    high_price_anomalies = [a for a in anomaly_data if a['severity'] == 'high' and a.get('type') == 'price']
    high_volume_anomalies = [a for a in anomaly_data if a['severity'] == 'high' and a.get('type') == 'volume']
    medium_price_anomalies = [a for a in anomaly_data if a['severity'] == 'medium' and a.get('type') == 'price']
    medium_volume_anomalies = [a for a in anomaly_data if a['severity'] == 'medium' and a.get('type') == 'volume']
    if_anomalies = [a for a in anomaly_data if a.get('type') in ('isolation_forest', 'stock_hour', 'stock')]
    
    # Isolation Forest 异常去重：每个股票只显示一次（选择最严重的）
    if_anomalies_dedup = {}
    for a in if_anomalies:
        stock = a['stock']
        if stock not in if_anomalies_dedup:
            if_anomalies_dedup[stock] = a
        else:
            # 如果已有记录，选择更严重的
            existing = if_anomalies_dedup[stock]
            if (existing['severity'] == 'low' and a['severity'] != 'low') or \
               (existing['severity'] == a['severity'] and abs(a.get('value', 0)) > abs(existing.get('value', 0))):
                if_anomalies_dedup[stock] = a
    if_anomalies = list(if_anomalies_dedup.values())
    
    # 按严重程度分组 IF 异常
    if_high_anomalies = [a for a in if_anomalies if a['severity'] == 'high']
    if_medium_anomalies = [a for a in if_anomalies if a['severity'] == 'medium']
    if_low_anomalies = [a for a in if_anomalies if a['severity'] == 'low']
    
    # 高异常（价格）
    if high_price_anomalies:
        content += "### 🔴 高异常（价格）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | RSI | 布林带 |\n"
        content += "|---------|---------|---------|---------|---------|-----|--------|\n"
        
        for anomaly in high_price_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            content += f"| {anomaly['stock']} | {anomaly['name']} | **{anomaly['z_score']:.2f}** | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} | {get_bollinger_position_cn(anomaly['bollinger_position'])} |\n"

        content += "\n"
    
    # 高异常（成交量）
    if high_volume_anomalies:
        content += "### 🔴 高异常（成交量）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
        content += "|---------|---------|---------|--------|---------|---------|\n"
        
        for anomaly in high_volume_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            volume_str = f"{anomaly.get('value', 0):,.0f}"
            content += f"| {anomaly['stock']} | {anomaly['name']} | **{anomaly['z_score']:.2f}** | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"

        content += "\n"
    
    # 中异常（价格）
    if medium_price_anomalies:
        content += "### ⚠️ 中异常（价格）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | RSI |\n"
        content += "|---------|---------|---------|---------|---------|-----|\n"
        
        for anomaly in medium_price_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"

        content += "\n"
    
    # 中异常（成交量）
    if medium_volume_anomalies:
        content += "### ⚠️ 中异常（成交量）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
        content += "|---------|---------|---------|--------|---------|---------|\n"
        
        for anomaly in medium_volume_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            volume_str = f"{anomaly.get('value', 0):,.0f}"
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"

        content += "\n"
    
    # Isolation Forest 异常（仅深度分析模式）- 按严重程度分组
    # 高严重度 IF 异常
    if if_high_anomalies:
        content += "### 🔴 Isolation Forest 异常（高）\n\n"
        content += "| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | RSI |\n"
        content += "|---------|---------|---------|---------|---------|---------|---------|-----|\n"
        
        for anomaly in if_high_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            anomaly_date = anomaly.get('anomaly_date', 'N/A')
            anomaly_reason = anomaly.get('anomaly_reason', '未知')
            anomaly_score = anomaly.get('value', 0)
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly_date} | {anomaly_reason} | **{anomaly_score:.3f}** | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"

        content += "\n"
    
    # 中等严重度 IF 异常
    if if_medium_anomalies:
        content += "### ⚠️ Isolation Forest 异常（中）\n\n"
        content += "| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | RSI |\n"
        content += "|---------|---------|---------|---------|---------|---------|---------|-----|\n"
        
        for anomaly in if_medium_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            anomaly_date = anomaly.get('anomaly_date', 'N/A')
            anomaly_reason = anomaly.get('anomaly_reason', '未知')
            anomaly_score = anomaly.get('value', 0)
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly_date} | {anomaly_reason} | {anomaly_score:.3f} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"

        content += "\n"
    
    # 低严重度 IF 异常
    if if_low_anomalies:
        content += "### 🔬 Isolation Forest 异常（低）\n\n"
        content += "| 股票代码 | 股票名称 | 异常日期 | 异常原因 | 异常评分 | 当前价格 | 当日变化 | RSI |\n"
        content += "|---------|---------|---------|---------|---------|---------|---------|-----|\n"
        
        for anomaly in if_low_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            anomaly_date = anomaly.get('anomaly_date', 'N/A')
            anomaly_reason = anomaly.get('anomaly_reason', '未知')
            anomaly_score = anomaly.get('value', 0)
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly_date} | {anomaly_reason} | {anomaly_score:.3f} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"

        content += "\n**说明**：\n"
        content += "- 基于多维特征检测的轻微异常\n"
        content += "- 可能是价格模式轻微变化\n\n"

    # 添加大模型分析
    content += "### 🤖 大模型异常分析\n\n"

    # 调用大模型分析异常
    llm_analysis = analyze_anomalies_with_llm(anomaly_data)
    content += llm_analysis

    return content


def get_hsi_email_indicators():
    """
    从 hsi_email.py 获取实时指标
    """
    try:
        from hsi_email import get_hsi_and_stock_indicators
        # 调用hsi_email模块的指标获取函数
        indicators = get_hsi_and_stock_indicators()
        return indicators
    except Exception as e:
        print(f"⚠️ 获取 hsi_email.py 实时指标失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_stock_technical_indicators(stock_code):
    """
    获取单只股票的详细技术指标

    参数:
    - stock_code: 股票代码（如 "0700.HK"）

    返回:
    - dict: 包含详细技术指标的字典
    """
    try:
        # 移除.HK后缀
        symbol = stock_code.replace('.HK', '')

        # 获取股票数据 - 使用完整的股票代码（带.HK）
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="6mo")

        if hist.empty:
            print(f"⚠️ 警告: 无法获取 {stock_code} 的历史数据")
            return None

        # 检查最后一天是否是 NaN（今日数据未更新）
        latest = hist.iloc[-1]
        if pd.isna(latest['Close']):
            # 使用倒数第二天作为最新数据
            if len(hist) > 1:
                latest = hist.iloc[-2]
                hist = hist.iloc[:-1]  # 移除最后一行 NaN 数据
            else:
                print(f"⚠️ 警告: {stock_code} 无有效数据")
                return None

        prev = hist.iloc[-2] if len(hist) > 1 else latest

        # 优先使用腾讯财经实时价格
        realtime_data = get_stock_realtime_data(stock_code)
        if realtime_data and realtime_data.get('price', 0) > 0:
            current_price = realtime_data['price']
            change_pct = realtime_data['change_pct']
        else:
            # 回退到 yfinance 数据
            current_price = latest['Close']
            change_pct = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] != 0 else 0

        # 技术指标
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]
        
        # 移动平均线
        ma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        ma10 = hist['Close'].rolling(window=10).mean().iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # 均线排列
        if ma5 > ma10 > ma20 > ma50:
            ma_alignment = "多头排列"
        elif ma5 < ma10 < ma20 < ma50:
            ma_alignment = "空头排列"
        else:
            ma_alignment = "震荡整理"
        
        # 均线斜率
        ma_slope_20 = (ma20 - hist['Close'].rolling(window=20).mean().iloc[-2]) / ma20 * 100 if len(hist) > 20 else 0
        ma_slope_50 = (ma50 - hist['Close'].rolling(window=50).mean().iloc[-2]) / ma50 * 100 if len(hist) > 50 else 0
        
        # 均线乖离率
        ma_deviation = ((current_price - ma20) / ma20 * 100) if ma20 > 0 else 0
        
        # 布林带
        bb_period = 20
        bb_std = 2
        bb_middle = hist['Close'].rolling(window=bb_period).mean()
        bb_std_dev = hist['Close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        
        # 布林带位置
        bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100
        
        # ATR
        high = hist['High'].astype(float)
        low = hist['Low'].astype(float)
        close = hist['Close'].astype(float)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean()
        current_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else 0
        
        # 成交量
        volume = latest['Volume']
        avg_volume_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # 趋势判断
        if current_price > ma20 > ma50:
            trend = "强势多头"
        elif current_price > ma20:
            trend = "短期上涨"
        elif current_price > ma50:
            trend = "震荡整理"
        else:
            trend = "弱势空头"
        
        # 支撑阻力位
        recent_highs = hist['High'].rolling(window=20).max()
        recent_lows = hist['Low'].rolling(window=20).min()
        support_level = recent_lows.iloc[-1]
        resistance_level = recent_highs.iloc[-1]
        support_distance = ((current_price - support_level) / current_price * 100) if current_price > 0 else 0
        resistance_distance = ((resistance_level - current_price) / current_price * 100) if current_price > 0 else 0
        
        # OBV（能量潮）
        # OBV 需要对整个历史数据计算，而不是只计算最新一天
        obv_series = ((hist['Close'].diff() > 0).astype(int) * 2 - 1) * hist['Volume']
        obv = (obv_series.cumsum() / 1e6).iloc[-1] if len(hist) > 0 else 0
        
        # 价格位置（基于20日区间）
        price_range_20d = hist['Close'].rolling(window=20).max() - hist['Close'].rolling(window=20).min()
        price_position = ((current_price - hist['Close'].rolling(window=20).min().iloc[-1]) / price_range_20d.iloc[-1] * 100) if price_range_20d.iloc[-1] > 0 else 50

        # 获取 PE/PB/ROE 基本面数据
        pe_ratio = None
        pb_ratio = None
        roe = None
        try:
            from data_services.fundamental_data import get_stock_financial_indicator
            financial_data = get_stock_financial_indicator(symbol)
            if financial_data:
                pe_ratio = financial_data.get('pe_ratio')
                pb_ratio = financial_data.get('pb_ratio')
                roe = financial_data.get('roe')
        except Exception as e:
            print(f"  ⚠️ 获取 {stock_code} PE/PB/ROE 数据失败: {e}")

        return {
            'current_price': current_price,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_hist': current_macd_hist,
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'ma_alignment': ma_alignment,
            'ma_slope_20': ma_slope_20,
            'ma_slope_50': ma_slope_50,
            'ma_deviation': ma_deviation,
            'bb_upper': current_bb_upper,
            'bb_lower': current_bb_lower,
            'bb_position': bb_position,
            'atr': current_atr,
            'volume': volume,
            'avg_volume_20': avg_volume_20,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'obv': obv,
            'price_position': price_position,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'roe': roe
        }
    except Exception as e:
        print(f"⚠️ 获取股票 {stock_code} 技术指标失败: {e}")
        return None


def get_recent_transactions(hours=48):
    """
    获取最近指定小时数的模拟交易记录
    
    参数:
    - hours: 查询的小时数，默认48小时
    
    返回:
    - DataFrame: 交易记录数据框
    """
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 交易记录文件路径
        transactions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'simulation_transactions.csv')
        
        if not os.path.exists(transactions_file):
            print(f"⚠️ 交易记录文件不存在: {transactions_file}")
            return pd.DataFrame()
        
        # 读取交易记录
        df = pd.read_csv(transactions_file, dtype=str, low_memory=False)
        if df.empty:
            return pd.DataFrame()
        
        # 找到时间列
        cols_lower = [c.lower() for c in df.columns]
        timestamp_col = None
        for candidate in ['timestamp', 'time', 'datetime', 'date']:
            if candidate in cols_lower:
                timestamp_col = df.columns[cols_lower.index(candidate)]
                break
        if timestamp_col is None:
            # fallback to first column
            timestamp_col = df.columns[0]

        # parse timestamp to UTC
        df[timestamp_col] = pd.to_datetime(df[timestamp_col].astype(str), utc=True, errors='coerce')

        # normalize key columns names to common names
        def find_col(possibilities):
            for p in possibilities:
                if p in cols_lower:
                    return df.columns[cols_lower.index(p)]
            return None

        type_col = find_col(['type', 'trans_type', 'action'])
        code_col = find_col(['code', 'symbol', 'ticker'])
        name_col = find_col(['name', 'stock_name'])
        reason_col = find_col(['reason', 'desc', 'description'])
        current_price_col = find_col(['current_price', 'price', 'currentprice', 'last_price'])
        stop_loss_col = find_col(['stop_loss', 'stoploss', 'stop_loss_price'])

        # rename to standard columns
        rename_map = {}
        if timestamp_col:
            rename_map[timestamp_col] = 'timestamp'
        if type_col:
            rename_map[type_col] = 'type'
        if code_col:
            rename_map[code_col] = 'code'
        if name_col:
            rename_map[name_col] = 'name'
        if reason_col:
            rename_map[reason_col] = 'reason'
        if current_price_col:
            rename_map[current_price_col] = 'current_price'
        if stop_loss_col:
            rename_map[stop_loss_col] = 'stop_loss_price'

        df = df.rename(columns=rename_map)

        # ensure required columns exist
        for c in ['type', 'code', 'name', 'reason', 'current_price', 'stop_loss_price']:
            if c not in df.columns:
                df[c] = ''

        # normalize type column
        df['type'] = df['type'].fillna('').astype(str).str.upper()
        # coerce numeric price columns where possible
        df['current_price'] = pd.to_numeric(df['current_price'].replace('', np.nan), errors='coerce')
        df['stop_loss_price'] = pd.to_numeric(df['stop_loss_price'].replace('', np.nan), errors='coerce')

        # drop rows without timestamp
        df = df[~df['timestamp'].isna()].copy()

        # 筛选最近指定小时的交易记录
        reference_time = pd.Timestamp.now(tz='UTC')
        time_threshold = reference_time - pd.Timedelta(hours=hours)
        df_recent = df[df['timestamp'] >= time_threshold].copy()
        
        return df_recent
        
    except Exception as e:
        print(f"⚠️ 读取交易记录失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def format_recent_transactions(transactions_df):
    """
    格式化最近的交易记录为表格格式

    参数:
    - transactions_df: 交易记录数据框

    返回:
    - str: 格式化的交易记录文本（表格格式）
    """
    if transactions_df is None or transactions_df.empty:
        return "  最近48小时内没有交易记录\n"

    # 按股票代码和时间排序
    transactions_df = transactions_df.sort_values(by=['code', 'timestamp'])

    # 构建Markdown表格
    text = "| 股票名称 | 股票代码 | 时间 | 类型 | 价格 | 目标价 | 止损价 | 有效期 | 理由 |\n"
    text += "|---------|---------|------|------|------|--------|--------|--------|------|\n"

    for _, trans in transactions_df.iterrows():
        stock_name = trans.get('name', '')
        code = trans.get('code', '')
        trans_type = trans.get('type', '').upper()

        # BUY用绿色，SELL用红色（使用 font 标签，邮件客户端兼容性更好）
        if trans_type == 'BUY':
            trans_type_display = '<font color="green"><b>BUY</b></font>'
        elif trans_type == 'SELL':
            trans_type_display = '<font color="red"><b>SELL</b></font>'
        else:
            trans_type_display = trans_type

        timestamp = pd.Timestamp(trans['timestamp']).strftime('%m-%d %H:%M:%S')
        price = trans.get('current_price', np.nan)
        price_display = f"{price:,.2f}" if not pd.isna(price) and price is not None else ''
        reason = trans.get('reason', '') or ''

        # 格式化止损价和目标价
        stop_loss = trans.get('stop_loss_price', np.nan)
        stop_loss_display = safe_float_format(stop_loss, '.2f') if safe_float_format(stop_loss, '.2f') else ''

        # 获取目标价
        target_price = trans.get('target_price', np.nan)
        target_price_display = safe_float_format(target_price, '.2f') if safe_float_format(target_price, '.2f') else ''

        # 获取有效期
        validity_period = trans.get('validity_period', np.nan)
        validity_period_display = safe_float_format(validity_period, '.0f') if safe_float_format(validity_period, '.0f') else ''

        text += f"| {stock_name} | {code} | {timestamp} | {trans_type_display} | {price_display} | {target_price_display} | {stop_loss_display} | {validity_period_display} | {reason} |\n"

    return text


def format_hsi_email_indicators(hsi_email_data):
    """
    格式化 hsi_email.py 的指标为文本和表格格式
    
    参数:
    - hsi_email_data: get_hsi_email_indicators 函数返回的数据
    
    返回:
    - tuple: (text_format, table_format) 格式化的文本和表格
    """
    if not hsi_email_data:
        return "", ""
    
    text_format = ""
    table_format = ""
    
    # 格式化恒生指数数据
    hsi_data = hsi_email_data.get('hsi_data')
    hsi_indicators = hsi_email_data.get('hsi_indicators')
    
    if hsi_data:
        text_format += "## 恒生指数实时技术指标\n\n"
        text_format += f"- 当前指数：{hsi_data['current_price']:,.2f}\n"
        text_format += f"- 24小时变化：{hsi_data['change_1d']:+.2f}% ({hsi_data['change_1d_points']:+.2f} 点)\n"
        text_format += f"- 当日开盘：{hsi_data['open']:,.2f}\n"
        text_format += f"- 当日最高：{hsi_data['high']:,.2f}\n"
        text_format += f"- 当日最低：{hsi_data['low']:,.2f}\n"
        text_format += f"- 成交量：{hsi_data['volume']:,.0f}\n\n"
        
        if hsi_indicators:
            text_format += f"- RSI（14日）：{safe_float_format(hsi_indicators.get('rsi', 0), '2f')}\n"
            text_format += f"- MACD：{safe_float_format(hsi_indicators.get('macd', 0), '4f')}\n"
            text_format += f"- MACD信号线：{safe_float_format(hsi_indicators.get('macd_signal', 0), '4f')}\n"
            text_format += f"- MA20：{safe_float_format(hsi_indicators.get('ma20', 0), ',.2f')}\n"
            text_format += f"- MA50：{safe_float_format(hsi_indicators.get('ma50', 0), ',.2f')}\n"
            text_format += f"- MA200：{safe_float_format(hsi_indicators.get('ma200', 0), ',.2f')}\n"
            text_format += f"- 布林带位置：{safe_float_format(hsi_indicators.get('bb_position', 0), '2f')}\n"
            text_format += f"- ATR（14日）：{safe_float_format(hsi_indicators.get('atr', 0), '2f')}\n"
            text_format += f"- 趋势：{hsi_indicators.get('trend', '未知')}\n\n"
    
    # 格式化自选股数据
    stock_results = hsi_email_data.get('stock_results', [])
    
    if stock_results:
        text_format += "## 自选股实时技术指标\n\n"
        text_format += "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | RSI | MACD | MA20 | MA50 | 趋势 | ATR | 成交量比率 |\n"
        text_format += "|---------|---------|---------|--------|-----|------|-----|-----|------|-----|-----------|\n"
        
        for stock_result in stock_results:
            code = stock_result.get('code', 'N/A')
            name = stock_result.get('name', 'N/A')
            data = stock_result.get('data', {})
            indicators = stock_result.get('indicators', {})
            
            current_price = data.get('current_price', 0)
            change_pct = data.get('change_1d', 0)
            rsi = indicators.get('rsi', 0)
            macd = indicators.get('macd', 0)
            ma20 = indicators.get('ma20', 0)
            ma50 = indicators.get('ma50', 0)
            trend = indicators.get('trend', '未知')
            atr = indicators.get('atr', 0)
            volume_ratio = indicators.get('volume_ratio', 0)
            
            text_format += f"| {code} | {name} | {safe_float_format(current_price, '2f')} | {safe_float_format(change_pct, '+.2f')}% | {safe_float_format(rsi, '2f')} | {safe_float_format(macd, '4f')} | {safe_float_format(ma20, '2f')} | {safe_float_format(ma50, '2f')} | {trend} | {safe_float_format(atr, '2f')} | {safe_float_format(volume_ratio, '2f')}x |\n"
    
    return text_format, table_format


def generate_technical_indicators_table(stock_codes):
    """
    为推荐股票生成技术指标表格
    
    参数:
    - stock_codes: 股票代码列表（从推荐建议中提取）
    
    返回:
    - str: Markdown格式的技术指标表格
    """
    try:
        if not stock_codes:
            return ""
        
        # 按股票代码排序
        stock_codes_sorted = sorted(stock_codes)
        
        table = "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | PE | PB | ROE | RSI | MACD | MA20 | MA50 | MA200 | 均线排列 | 均线斜率 | 乖离率 | 布林带位置 | ATR | 成交量比率 | 趋势 | 支撑位 | 阻力位 |\n"
        table += "|---------|---------|---------|--------|-----|-----|-----|-----|------|-----|-----|------|---------|---------|-------|-----------|-----|-----------|------|--------|--------|\n"
        
        success_count = 0
        for stock_code in stock_codes_sorted:
            indicators = get_stock_technical_indicators(stock_code)
            
            if indicators:
                # 获取股票名称
                stock_name = WATCHLIST.get(stock_code, stock_code)
                
                # 格式化数据
                price = safe_float_format(indicators['current_price'], '.2f')
                change = safe_float_format(indicators['change_pct'], '+.2f') + "%"

                # PE 市盈率格式化（带颜色）
                pe_value = indicators.get('pe_ratio')
                if pe_value is not None and pe_value > 0:
                    if pe_value < 15:
                        pe_display = f'<span style="color: #16a34a;">{pe_value:.2f}</span>'  # 绿色（低估）
                    elif pe_value < 25:
                        pe_display = f'<span style="color: #ea580c;">{pe_value:.2f}</span>'  # 橙色（正常）
                    else:
                        pe_display = f'<span style="color: #dc2626;">{pe_value:.2f}</span>'  # 红色（高估）
                else:
                    pe_display = "N/A"

                # PB 市净率格式化（带颜色）
                pb_value = indicators.get('pb_ratio')
                if pb_value is not None and pb_value > 0:
                    if pb_value < 1.5:
                        pb_display = f'<span style="color: #16a34a;">{pb_value:.2f}</span>'  # 绿色（低估）
                    elif pb_value < 3:
                        pb_display = f'<span style="color: #ea580c;">{pb_value:.2f}</span>'  # 橙色（正常）
                    else:
                        pb_display = f'<span style="color: #dc2626;">{pb_value:.2f}</span>'  # 红色（高估）
                else:
                    pb_display = "N/A"

                # ROE 净资产收益率格式化（带颜色）
                roe_value = indicators.get('roe')
                if roe_value is not None and roe_value > 0:
                    if roe_value >= 15:
                        roe_display = f'<span style="color: #16a34a;">{roe_value:.2f}%</span>'  # 绿色（优秀）
                    elif roe_value >= 10:
                        roe_display = f'<span style="color: #ea580c;">{roe_value:.2f}%</span>'  # 橙色（良好）
                    else:
                        roe_display = f'<span style="color: #dc2626;">{roe_value:.2f}%</span>'  # 红色（一般）
                else:
                    roe_display = "N/A"

                rsi = safe_float_format(indicators['rsi'], '.2f')
                macd = safe_float_format(indicators['macd'], '.2f')
                ma20 = safe_float_format(indicators['ma20'], '.2f')
                ma50 = safe_float_format(indicators['ma50'], '.2f')
                ma200 = safe_float_format(indicators['ma200'], '.2f') if pd.notna(indicators['ma200']) else "N/A"
                ma_align = indicators['ma_alignment']
                ma_slope = safe_float_format(indicators['ma_slope_20'], '.2f')
                ma_dev = safe_float_format(indicators['ma_deviation'], '.2f') + "%"

                # 布林带位置三色系统：<30%绿色（超卖），30-70%橙色（正常），>70%红色（超买）
                bb_value = indicators['bb_position']
                if bb_value < 30:
                    bb_pos = f'<span style="color: #16a34a; font-weight: bold;">{bb_value:.1f}%</span>'  # 亮绿色
                elif bb_value > 70:
                    bb_pos = f'<span style="color: #dc2626; font-weight: bold;">{bb_value:.1f}%</span>'  # 亮红色
                else:
                    bb_pos = f'<span style="color: #ea580c; font-weight: bold;">{bb_value:.1f}%</span>'  # 亮橙色

                atr = safe_float_format(indicators['atr'], '.2f')
                vol_ratio = safe_float_format(indicators['volume_ratio'], '.2f') + "x"
                trend = indicators['trend']
                support = f"{safe_float_format(indicators['support_level'], '.2f')} ({safe_float_format(indicators['support_distance'], '.2f')}%)"
                resistance = f"{safe_float_format(indicators['resistance_level'], '.2f')} ({safe_float_format(indicators['resistance_distance'], '.2f')}%)"
                
                # 根据数值添加颜色标记（文本用括号标注）
                if indicators['rsi'] > 70:
                    rsi += " (超买)"
                elif indicators['rsi'] < 30:
                    rsi += " (超卖)"
                
                if indicators['change_pct'] > 0:
                    change = f"📈 {change}"
                else:
                    change = f"📉 {change}"

                if indicators['trend'] == "强势多头":
                    trend = f"🟢 {trend}"
                elif indicators['trend'] == "弱势空头":
                    trend = f"🔴 {trend}"

                table += f"| {stock_code} | {stock_name} | {price} | {change} | {pe_display} | {pb_display} | {roe_display} | {rsi} | {macd} | {ma20} | {ma50} | {ma200} | {ma_align} | {ma_slope} | {ma_dev} | {bb_pos} | {atr} | {vol_ratio} | {trend} | {support} | {resistance} |\n"
                success_count += 1

        print(f"📊 技术指标表格: 成功获取 {success_count}/{len(stock_codes)} 只股票的数据")
        return table

    except Exception as e:
        print(f"⚠️ 生成技术指标表格失败: {e}")
        return ""


def save_comprehensive_report_md(content, date_str):
    """
    保存综合分析报告为 MD 文档（用于知识库）

    参数:
    - content: 报告内容
    - date_str: 日期字符串 (YYYY-MM-DD)
    """
    try:
        # 创建输出目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output', 'comprehensive_reports')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 创建目录: {output_dir}")

        # 生成文件路径
        md_filepath = os.path.join(output_dir, f'{date_str}.md')

        # 添加元数据头部
        md_content = f"""# 综合分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析日期**: {date_str}

---

{content}
"""

        # 写入文件
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✅ MD 报告已保存到 {md_filepath}")
        return md_filepath

    except Exception as e:
        print(f"⚠️ 保存 MD 报告失败: {e}")
        return None


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
        recipient_email = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")

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
        elif "gmail.com" in smtp_server:
            smtp_port = 587
            use_ssl = False
        else:
            smtp_port = 587
            use_ssl = False

        # 创建邮件对象
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)

        # 添加文本版本
        text_part = MIMEText(content, 'plain', 'utf-8')
        msg.attach(text_part)

        # 如果有HTML版本，添加HTML版本
        if html_content:
            html_part = MIMEText(html_content, 'html', 'utf-8')
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

                print(f"✅ 邮件已发送到: {', '.join(recipients)}")
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


# ============================================================================
# 详细个股分析模块（从 send_stock_analysis_email.py 合并）
# ============================================================================

# 大模型提取 Prompt 模板
STOCK_ANALYSIS_PROMPT = """你是一个股票分析专家。请从以下综合分析报告中提取指定股票的分析信息。

# 股票代码
{stock_code}

# 综合分析报告
{report_content}

# 提取要求

请提取以下信息，以 JSON 格式返回：

## 个股关键数据
1. **stock_name**: 股票名称
2. **current_price**: 当前价格（数值）
3. **price_change**: 涨跌幅（百分比数值，如 -2.5）
4. **catboost_prob_20d**: CatBoost 20天上涨概率（百分比数值，如 37.12）
5. **catboost_prob_5d**: CatBoost 5天上涨概率
6. **catboost_prob_1d**: CatBoost 1天上涨概率
7. **three_period_pattern**: 三周期模式（如"下跌中继⭐(001)"）
8. **transmission_mode**: 传导模式验证结果（如"✅传导(1天↓✓ 5天↓✓ 20天↑⏳)"）
9. **chip_resistance**: 筹码阻力（低/中/高）
10. **risk_score**: 风险得分（数值）
11. **return_score**: 回报得分（数值）
12. **total_score**: 综合得分（数值）
13. **risk_advice**: 风险建议（推荐/观察/暂缓/优选）

## 网络洞察
14. **community**: 网络社区归属（如"社区5"）
15. **hub_type**: 枢纽类型（低枢纽/中枢纽/高枢纽）
16. **is_bridge**: 是否为桥梁股（是/否）

## 大模型建议
17. **short_term_advice**: 大模型短期建议（买入/观察/卖出）
18. **mid_term_advice**: 大模型中期建议（买入/观察/卖出）
19. **position_advice**: 建议仓位（百分比数值）
20. **stop_loss**: 止损位（价格数值或 null）
21. **target_price**: 目标价（价格数值或 null）

## 技术指标
22. **rsi**: RSI值（数值）
23. **rsi_status**: RSI状态（超买/正常/超卖）
24. **macd_status**: MACD状态（金叉/死叉/弱金叉等）
25. **bb_position**: 布林带位置（百分比数值）
26. **ma_status**: 均线排列状态（多头排列/空头排列/震荡整理）

## 板块信息
27. **sector**: 所属板块名称
28. **sector_rank**: 板块排名（数值）
29. **sector_type**: 板块类型（周期/防御）
30. **sector_change**: 板块5日涨跌幅（百分比数值）

## 异常检测
31. **anomaly_alert**: 是否有异常检测警报（是/否）
32. **anomaly_reason**: 异常原因（如有）

## 股息信息
33. **dividend_info**: 股息信息（如"即将除净：2026-05-25，每股分红0.5元"或 null）

## 市场环境信息（所有股票共用，从报告开头提取）
34. **hsi_price**: 恒生指数价格（数值）
35. **hsi_change**: 恒生指数涨跌幅（百分比数值）
36. **market_status**: 市场状态（牛市/震荡偏涨/震荡市/震荡偏跌/熊市）
37. **market_duration**: 市场状态持续天数（数值）
38. **market_stability**: 市场状态稳定性（稳定/中等稳定/不稳定）
39. **vix**: VIX指数（数值）
40. **market_sentiment**: 市场情绪（正常/谨慎/暂停）
41. **modularity**: 模块度（数值）

# 返回格式

请以 JSON 格式返回，包含以上所有字段：
{{
    "stock_name": "...",
    "current_price": ...,
    "catboost_prob_20d": ...,
    ...
}}

**重要**：
1. 如果报告中没有某项信息，请填写 null
2. 数值字段请返回数值类型，不要返回字符串
3. 百分比字段请返回数值（如 37.12 表示 37.12%）
4. 确保返回的是纯 JSON，不要包含 Markdown 代码块标记
"""

# 大模型综合分析 Prompt 模板
COMPREHENSIVE_ANALYSIS_PROMPT = """你是一个专业的股票分析师。请基于以下股票数据，按照分析框架生成详细的综合分析。

# 股票数据
{stock_data}

# 分析框架

请按以下步骤进行综合分析：

## 第一步：硬约束检查
- CatBoost 20天上涨概率 ≤ 50% → **禁止买入**
- CatBoost 20天上涨概率 ≥ 60% → 高置信度，进入下一步
- CatBoost 20天上涨概率 50-60% → 中等置信度，需其他信号确认

## 第二步：方向一致性检查（关键）
- 短期建议 + 中期建议 + CatBoost 三者方向一致 → **可买入**
- 短期建议"观察" + 中期建议"买入" → **观望**（方向冲突）
- 短期建议"买入" + 中期建议"观察" → **观望**（方向冲突）
- 短期建议"卖出" + 中期建议任何 → **禁止买入**

**重要规则**：即使CatBoost概率>60%，如果短期/中期方向不一致，也应归入"观望"，而非"强烈买入"。

## 第三步：市场环境检查
- 市场状态为"熊市" → 提高阈值至0.70
- 市场状态为"震荡市" → 提高阈值至0.65
- 市场状态稳定性<5天 → 提示风险，建议降低仓位

## 第四步：异常检测检查
- 有超买异常（RSI>70）→ 提示短期回调风险
- 有超卖异常（RSI<30）→ 提示可能存在反弹机会
- 有成交量异常 → 提示资金异动

## 第五步：网络洞察检查
- 模块度<0.20 → 市场同涨同跌，降低仓位30%
- 为桥梁股 → 提示跨社区风险传导

## 分类标准

| 分类 | CatBoost概率 | 短期建议 | 中期建议 | 说明 |
|------|-------------|---------|---------|------|
| ⭐强烈买入 | ≥60% | 买入 | 买入 | 三重确认 |
| 🟢买入 | 50-60% | 买入 | 买入 | 需其他信号确认 |
| 🟡观望 | ≥50% | 观察 | 买入 | 方向冲突，等待确认 |
| 🟡观望 | ≥50% | 买入 | 观察 | 方向冲突，等待确认 |
| 🔴禁止买入 | ≤50% | 任何 | 任何 | 硬约束 |
| 🔴禁止买入 | 任何 | 卖出 | 任何 | 方向明确看跌 |

# 返回格式

请以 JSON 格式返回以下字段：
{{
    "recommendation": "综合建议（如：⭐强烈买入 / 🟢买入 / 🟡观望 / 🔴禁止买入）",
    "recommendation_class": "CSS类名（rec-strong-buy / rec-buy / rec-hold / rec-sell）",
    "consistency": "方向一致性分析（如：一致 / 方向冲突，短期中期不一致）",
    "analysis_summary": "分析摘要（一句话说明判断依据）",
    "operation_advice": "操作建议（详细说明，包括仓位、止损等）",
    "risk_warnings": ["风险提示1", "风险提示2", ...]
}}

**重要**：
1. risk_warnings 必须是数组，列出3-5个主要风险因素
2. operation_advice 要具体，包括仓位建议、止损设置等
3. 确保返回的是纯 JSON，不要包含 Markdown 代码块标记
"""

# 大模型持货人建议 Prompt 模板
HOLDER_ADVICE_PROMPT = """你是一个专业的股票分析师。请基于以下股票分析数据，为已持有该股票的投资者提供操作建议。

# 股票数据
{stock_data}

# 分析框架

## 第一步：趋势判断
- CatBoost 20天概率 > 60% 且短中期建议"买入" → 趋势向上，考虑持有或加仓
- CatBoost 20天概率 50-60% → 趋势不明，谨慎持有
- CatBoost 20天概率 < 50% → 趋势向下，考虑减仓或止损

## 第二步：止盈止损判断
- 当前价格接近目标价 → 考虑分批止盈
- RSI > 70 或布林带位置 > 80% → 短期超买，考虑减仓锁利
- RSI < 30 或布林带位置 < 20% → 短期超卖，若趋势仍向上可持有
- MACD 死叉 → 警惕回调

## 第三步：市场环境调整
- 熊市环境 → 降低持仓，严格止损
- 市场情绪"暂停" → 建议减仓避险
- 市场状态不稳定（<5天）→ 降低仓位

## 第四步：综合建议
基于以上分析，给出持货人的具体操作建议。

# 返回格式

请以 JSON 格式返回：
{{
    "holder_action": "继续持有 / 加仓 / 减仓 / 止盈离场 / 止损离场",
    "holder_reason": "操作理由（1-2句话）",
    "key_level": "关键价位说明（如：跌破XX止损，突破XX加仓）",
    "risk_level": "低 / 中 / 高",
    "time_horizon": "短期（1-5天）/ 中期（5-20天）"
}}

**重要**：
1. 确保返回纯 JSON，不要包含 Markdown 代码块标记
2. holder_action 只能是上述5种之一
3. holder_reason 要具体，引用具体指标数据
"""

# 持货人操作建议样式映射
HOLDER_ACTION_STYLES = {
    '继续持有': ('holder-action-hold', '#d4edda', '#155724'),
    '加仓': ('holder-action-add', '#cce5ff', '#004085'),
    '减仓': ('holder-action-reduce', '#fff3cd', '#856404'),
    '止盈离场': ('holder-action-exit-profit', '#f8d7da', '#721c24'),
    '止损离场': ('holder-action-exit-loss', '#f8d7da', '#721c24'),
}


def extract_json_from_response(response: str) -> dict:
    """从大模型响应中提取 JSON"""
    import re
    try:
        # 尝试提取 Markdown 代码块中的 JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接匹配 JSON 对象
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def build_stock_data_for_llm(stock_code: str, three_horizon_results: dict,
                               chip_data: dict, risk_reward_data: dict,
                               historical_pl_data: dict, network_insights: dict,
                               anomaly_data: dict, current_market: dict,
                               stock_realtime_data: dict, sector_data: dict) -> str:
    """
    从已计算数据构建精简文本，供大模型提取股票数据

    参数：
    - stock_code: 股票代码
    - three_horizon_results: 三周期预测结果
    - chip_data: 筹码阻力数据
    - risk_reward_data: 风险回报数据
    - historical_pl_data: 历史盈亏比数据
    - network_insights: 网络洞察数据
    - anomaly_data: 异常检测数据
    - current_market: 当前市场状态
    - stock_realtime_data: 股票实时数据
    - sector_data: 板块数据

    返回：
    - str: 精简文本（约2000字符）
    """
    lines = []

    # 股票基本信息
    stock_name = WATCHLIST.get(stock_code, stock_code)
    lines.append(f"# 股票基本信息")
    lines.append(f"股票代码: {stock_code}")
    lines.append(f"股票名称: {stock_name}")

    # 实时价格数据
    if stock_code in stock_realtime_data:
        rt = stock_realtime_data[stock_code]
        lines.append(f"当前价格: {rt.get('current_price', 'N/A')}")
        change_pct = rt.get('change_pct', 0)
        try:
            lines.append(f"涨跌幅: {float(change_pct):+.2f}%" if change_pct is not None else "涨跌幅: N/A")
        except (ValueError, TypeError):
            lines.append(f"涨跌幅: {change_pct}")

    lines.append("")

    # ML预测
    lines.append("# ML预测")
    if stock_code in three_horizon_results:
        pred = three_horizon_results[stock_code]
        preds = pred.get('predictions', {})

        pred_1d = preds.get(1, {})
        pred_5d = preds.get(5, {})
        pred_20d = preds.get(20, {})

        prob_1d = pred_1d.get('probability', 0)
        prob_5d = pred_5d.get('probability', 0)
        prob_20d = pred_20d.get('probability', 0)
        try:
            lines.append(f"CatBoost 1天预测: {pred_1d.get('direction', '-')} {float(prob_1d):.2f}")
            lines.append(f"CatBoost 5天预测: {pred_5d.get('direction', '-')} {float(prob_5d):.2f}")
            lines.append(f"CatBoost 20天预测: {pred_20d.get('direction', '-')} {float(prob_20d):.2f}")
        except (ValueError, TypeError):
            lines.append(f"CatBoost 1天预测: {pred_1d.get('direction', '-')} {prob_1d}")
            lines.append(f"CatBoost 5天预测: {pred_5d.get('direction', '-')} {prob_5d}")
            lines.append(f"CatBoost 20天预测: {pred_20d.get('direction', '-')} {prob_20d}")

        pattern = pred.get('pattern', '-')
        pattern_info = pred.get('pattern_info', {})
        pattern_name = pattern_info.get('name', '')
        if pattern_name:
            lines.append(f"三周期模式: {pattern_name}({pattern})")
        else:
            lines.append(f"三周期模式: {pattern}")

        action = pattern_info.get('action', '观望')
        win_rate = pattern_info.get('win_rate', 0)
        lines.append(f"交易建议: {action}")
        try:
            lines.append(f"历史胜率: {float(win_rate):.1f}%")
        except (ValueError, TypeError):
            lines.append(f"历史胜率: {win_rate}")
    else:
        lines.append("CatBoost预测: 数据缺失")

    lines.append("")

    # 风险评估
    lines.append("# 风险评估")

    # 筹码阻力
    if stock_code in chip_data and chip_data[stock_code]:
        chip_result = chip_data[stock_code]
        resistance_ratio = chip_result.get('resistance_ratio', 0)
        if resistance_ratio < 0.3:
            lines.append(f"筹码阻力: 低 ({resistance_ratio:.1%})")
        elif resistance_ratio < 0.6:
            lines.append(f"筹码阻力: 中 ({resistance_ratio:.1%})")
        else:
            lines.append(f"筹码阻力: 高 ({resistance_ratio:.1%})")
    else:
        lines.append("筹码阻力: N/A")

    # 风险回报数据
    if stock_code in risk_reward_data and risk_reward_data[stock_code]:
        rr = risk_reward_data[stock_code]
        lines.append(f"风险得分: {rr.get('risk_score', 'N/A')}")
        lines.append(f"回报得分: {rr.get('return_score', 'N/A')}")
        lines.append(f"综合得分: {rr.get('comprehensive_score', 'N/A')}")
        lines.append(f"风险建议: {rr.get('suggestion', 'N/A')}")

    # 历史盈亏比
    if stock_code in historical_pl_data and historical_pl_data[stock_code]:
        pl = historical_pl_data[stock_code]
        lines.append(f"盈亏比: {pl.get('profit_loss_ratio_str', 'N/A')}")
        lines.append(f"期望收益: {pl.get('expected_return_str', 'N/A')}")

    lines.append("")

    # 网络洞察
    lines.append("# 网络洞察")
    if stock_code in network_insights and network_insights[stock_code]:
        insight = network_insights[stock_code]
        lines.append(f"网络洞察: {insight.get('insight_str', '未知')}")
        lines.append(f"社区: {insight.get('community_id', 'N/A')}")
        lines.append(f"枢纽类型: {insight.get('hub_level', 'N/A')}")
        lines.append(f"是否桥梁股: {'是' if insight.get('is_bridge') else '否'}")
    else:
        lines.append("网络洞察: 数据缺失")

    lines.append("")

    # 技术指标（从实时数据获取）
    lines.append("# 技术指标")
    if stock_code in stock_realtime_data:
        rt = stock_realtime_data[stock_code]
        rsi = rt.get('rsi')
        if rsi is not None:
            try:
                rsi_val = float(rsi)
                rsi_status = "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "正常"
                lines.append(f"RSI: {rsi_val:.2f} ({rsi_status})")
            except (ValueError, TypeError):
                lines.append(f"RSI: {rsi}")
        macd = rt.get('macd')
        if macd is not None:
            try:
                lines.append(f"MACD: {float(macd):.2f}")
            except (ValueError, TypeError):
                lines.append(f"MACD: {macd}")
        ma_status = rt.get('ma_alignment')
        if ma_status:
            lines.append(f"MA状态: {ma_status}")

    lines.append("")

    # 板块信息
    lines.append("# 板块信息")
    if stock_code in STOCK_SECTOR_MAPPING:
        sector_code = STOCK_SECTOR_MAPPING[stock_code].get('sector', '')
        if sector_code and sector_code in SECTOR_NAME_MAPPING:
            sector_name = SECTOR_NAME_MAPPING[sector_code]
            lines.append(f"板块: {sector_name}")
            sector_type = get_sector_type(sector_code)
            lines.append(f"板块类型: {sector_type}")

            # 从板块数据获取排名
            if sector_data and sector_data.get('performance') is not None:
                perf_df = sector_data['performance']
                for idx, row in perf_df.iterrows():
                    if row.get('sector_code') == sector_code:
                        lines.append(f"板块排名: {idx + 1}")
                        avg_change = row.get('avg_change_pct', 0)
                        try:
                            lines.append(f"板块5日涨跌幅: {float(avg_change):+.2f}%")
                        except (ValueError, TypeError):
                            lines.append(f"板块5日涨跌幅: {avg_change}")
                        break

    lines.append("")

    # 异常检测
    lines.append("# 异常检测")
    if anomaly_data and stock_code in anomaly_data:
        anom = anomaly_data[stock_code]
        has_anomaly = anom.get('has_anomaly', False)
        lines.append(f"异常警报: {'是' if has_anomaly else '否'}")
        if has_anomaly:
            reasons = anom.get('reasons', [])
            if reasons:
                lines.append(f"异常原因: {', '.join(reasons)}")
    else:
        lines.append("异常警报: 否")

    lines.append("")

    # 市场环境
    lines.append("# 市场环境")
    if current_market:
        hsi_price = current_market.get('hsi_price', 'N/A')
        hsi_change = current_market.get('hsi_change', 0)
        try:
            lines.append(f"恒生指数: {hsi_price} ({float(hsi_change):+.2f}%)")
        except (ValueError, TypeError):
            lines.append(f"恒生指数: {hsi_price} ({hsi_change}%)")
        lines.append(f"市场状态: {current_market.get('market_state_cn', 'N/A')}")
        lines.append(f"市场状态持续: {current_market.get('duration', 'N/A')}天")
        lines.append(f"市场稳定性: {current_market.get('stability', 'N/A')}")
        lines.append(f"VIX: {current_market.get('vix', 'N/A')}")

    lines.append("")

    # 股息信息（如果有）
    if stock_code in stock_realtime_data:
        rt = stock_realtime_data[stock_code]
        dividend_info = rt.get('dividend_info')
        if dividend_info:
            lines.append("# 股息信息")
            lines.append(f"股息信息: {dividend_info}")

    return "\n".join(lines)


def extract_stock_data_with_llm(stock_code: str, report_content: str) -> dict:
    """
    使用大模型从综合报告中提取指定股票的分析数据

    参数：
    - stock_code: 股票代码（如 "2318.HK"）
    - report_content: 综合报告内容

    返回：
    - dict: 股票分析数据
    """
    print(f"📊 正在使用大模型提取 {stock_code} 的分析数据...")

    # 构建提取 Prompt
    prompt = STOCK_ANALYSIS_PROMPT.format(
        stock_code=stock_code,
        report_content=report_content
    )

    # 调用大模型提取
    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return None

    # 解析 JSON 响应
    stock_data = extract_json_from_response(response)
    if stock_data:
        stock_data['stock_code'] = stock_code
        print(f"✅ 成功提取 {stock_code} 的分析数据")
        return stock_data
    else:
        print(f"❌ 无法从大模型响应中提取 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None


def comprehensive_analyze_with_llm(stock_data: dict) -> dict:
    """
    使用大模型进行综合分析

    参数：
    - stock_data: 股票数据

    返回：
    - dict: 综合分析结果，包含 recommendation, operation_advice, risk_warnings 等
    """
    print(f"🤖 正在使用大模型进行综合分析...")

    # 构建分析 Prompt
    prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
        stock_data=json.dumps(stock_data, ensure_ascii=False, indent=2)
    )

    # 调用大模型分析
    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return None

    # 解析 JSON 响应
    analysis_result = extract_json_from_response(response)
    if analysis_result:
        print(f"✅ 成功生成综合分析")
        return analysis_result
    else:
        print(f"❌ 无法从大模型响应中提取 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None


def get_holder_advice_with_llm(stock_data: dict) -> dict:
    """
    使用大模型为已持货人生成操作建议

    参数：
    - stock_data: 股票数据

    返回：
    - dict: 持货人操作建议
    """
    print(f"📊 正在生成持货人操作建议...")

    prompt = HOLDER_ADVICE_PROMPT.format(
        stock_data=json.dumps(stock_data, ensure_ascii=False, indent=2)
    )

    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 持货人建议生成失败: {e}")
        return None

    result = extract_json_from_response(response)
    if result:
        print(f"✅ 成功生成持货人操作建议")
        return result
    else:
        print(f"❌ 无法从大模型响应中提取持货人建议 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None


def format_value_default(value, default="-"):
    """格式化数值，处理 None 值"""
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def generate_stock_section_html(stock_data: dict) -> str:
    """生成单只股票的 HTML 部分（详细个股分析）"""
    stock_code = stock_data.get('stock_code', '')
    stock_name = stock_data.get('stock_name', stock_code)

    # 获取大模型综合分析结果
    analysis = stock_data.get('analysis', {})

    # 综合建议（优先使用大模型分析结果）
    if analysis:
        recommendation = analysis.get('recommendation', '🟡 观望')
        rec_class = analysis.get('recommendation_class', 'rec-hold')
        consistency = analysis.get('consistency', '一致')
        operation_advice = analysis.get('operation_advice', '')
        risk_warnings = analysis.get('risk_warnings', [])
    else:
        # 如果没有大模型分析结果，使用默认逻辑
        recommendation = "🟡 观望"
        rec_class = "rec-hold"
        consistency = "一致"
        operation_advice = ""
        risk_warnings = []

    # CatBoost 概率
    prob_20d = stock_data.get('catboost_prob_20d', 0) or 0
    prob_5d = stock_data.get('catboost_prob_5d', 0) or 0
    prob_1d = stock_data.get('catboost_prob_1d', 0) or 0

    # 概率样式
    if prob_20d >= 60:
        prob_class = "metric-good"
        prob_desc = "高置信度，>60%"
    elif prob_20d >= 50:
        prob_class = "metric-neutral"
        prob_desc = "中等置信度，50-60%"
    else:
        prob_class = "metric-bad"
        prob_desc = "看跌，<50%硬约束禁止买入"

    # 方向判断
    dir_1d = "↑ 上涨" if prob_1d >= 50 else "↓ 下跌"
    dir_1d_class = "arrow-up" if prob_1d >= 50 else "arrow-down"
    dir_5d = "↑ 上涨" if prob_5d >= 50 else "↓ 下跌"
    dir_5d_class = "arrow-up" if prob_5d >= 50 else "arrow-down"
    dir_20d = "↑ 上涨" if prob_20d >= 50 else "↓ 下跌"
    dir_20d_class = "arrow-up" if prob_20d >= 50 else "arrow-down"

    # 涨跌幅显示
    price_change = stock_data.get('price_change', 0) or 0
    if price_change > 0:
        price_change_display = f"📈 +{price_change:.2f}%"
    elif price_change < 0:
        price_change_display = f"📉 {price_change:.2f}%"
    else:
        price_change_display = "-"

    # RSI 状态显示
    rsi = stock_data.get('rsi', 50) or 50
    rsi_status = stock_data.get('rsi_status', '正常') or '正常'
    if rsi > 70:
        rsi_status_display = f"超买（{rsi:.1f}）"
    elif rsi < 30:
        rsi_status_display = f"超卖（{rsi:.1f}）"
    else:
        rsi_status_display = f"{rsi_status}（{rsi:.1f}）"

    # 布林带状态
    bb_position = stock_data.get('bb_position', 50) or 50
    if bb_position > 80:
        bb_status = "接近上轨（超买）"
    elif bb_position < 20:
        bb_status = "接近下轨（超卖）"
    else:
        bb_status = "中性"

    # 止损位和目标价
    stop_loss = stock_data.get('stop_loss')
    if stop_loss:
        stop_loss = f"HK${stop_loss:.2f}"
    else:
        stop_loss = "-"

    target_price = stock_data.get('target_price')
    if target_price:
        target_price = f"HK${target_price:.2f}"
    else:
        target_price = "-"

    # 异常检测部分
    anomaly_alert = stock_data.get('anomaly_alert', '否') or '否'
    anomaly_reason = stock_data.get('anomaly_reason') or ''
    if anomaly_alert == '是':
        anomaly_section = f"""
            <h3>七、异常检测</h3>
            <div class="warning-box">
                ⚠️ 检测到异常：{anomaly_reason}
            </div>
        """
    else:
        anomaly_section = """
            <h3>七、异常检测</h3>
            <p>当日无异常检测警报</p>
        """

    # 股息信息
    dividend_info = stock_data.get('dividend_info')
    if dividend_info:
        dividend_section = f"""
            <h3>八、股息提醒</h3>
            <div class="info-box">
                📅 {dividend_info}
            </div>
        """
    else:
        dividend_section = ""

    # 操作建议框
    if operation_advice:
        if "强烈买入" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="success-box">
                    {operation_advice}
                </div>
            """
        elif "买入" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="info-box">
                    {operation_advice}
                </div>
            """
        elif "观望" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="info-box">
                    {operation_advice}
                </div>
            """
        else:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="warning-box">
                    {operation_advice}
                </div>
            """
    else:
        operation_box = """
            <h3>九、操作建议</h3>
            <div class="info-box">
                <strong>操作建议</strong>：暂时观望，等待更明确的信号。
            </div>
        """

    # 持货人操作建议部分
    holder_advice = stock_data.get('holder_advice')
    if holder_advice:
        holder_action = holder_advice.get('holder_action', '继续持有')
        holder_reason = holder_advice.get('holder_reason', '')
        key_level = holder_advice.get('key_level', '')
        risk_level = holder_advice.get('risk_level', '中')
        time_horizon = holder_advice.get('time_horizon', '中期（5-20天）')

        style_info = HOLDER_ACTION_STYLES.get(holder_action, HOLDER_ACTION_STYLES['继续持有'])
        holder_action_class = style_info[0]

        holder_advice_section = f"""
            <h3>十、持货人操作建议</h3>
            <div class="{holder_action_class}">
                <strong>建议操作</strong>：{holder_action}<br>
                <strong>风险等级</strong>：{risk_level} | <strong>适用周期</strong>：{time_horizon}
            </div>
            <p><strong>理由</strong>：{holder_reason}</p>
            <p><strong>关键价位</strong>：{key_level}</p>
        """
    else:
        holder_advice_section = ""

    # 风险提示部分
    if risk_warnings:
        risk_warnings_list = "".join([f"<li>{w}</li>" for w in risk_warnings])
        risk_warnings_section = f"""
            <h3>十一、风险提示</h3>
            <ul>
                {risk_warnings_list}
            </ul>
        """
    else:
        risk_warnings_section = ""

    return f"""
        <div class="stock-section">
            <h2>{stock_name}（{stock_code}）</h2>

            <div class="recommendation {rec_class}">
                综合建议：{recommendation}
            </div>

            <h3>一、核心指标</h3>
            <table>
                <tr><th>指标</th><th>数值</th><th>说明</th></tr>
                <tr><td>CatBoost 20天上涨概率</td><td class="{prob_class}">{format_value_default(prob_20d, "0")}%</td><td>{prob_desc}</td></tr>
                <tr><td>当前价格</td><td>HK${format_value_default(stock_data.get('current_price'), "-")}</td><td>{price_change_display}</td></tr>
                <tr><td>建议仓位</td><td>{format_value_default(stock_data.get('position_advice', 0), "0")}%</td><td></td></tr>
                <tr><td>止损位</td><td>{stop_loss}</td><td>最大亏损控制在-8%以内</td></tr>
                <tr><td>目标价</td><td>{target_price}</td><td></td></tr>
            </table>

            <h3>二、三周期预测</h3>
            <table>
                <tr><th>周期</th><th>预测概率</th><th>方向</th></tr>
                <tr><td>1天</td><td>{format_value_default(prob_1d, "0")}%</td><td class="{dir_1d_class}">{dir_1d}</td></tr>
                <tr><td>5天</td><td>{format_value_default(prob_5d, "0")}%</td><td class="{dir_5d_class}">{dir_5d}</td></tr>
                <tr><td>20天</td><td>{format_value_default(prob_20d, "0")}%</td><td class="{dir_20d_class}">{dir_20d}</td></tr>
            </table>
            <p><strong>三周期模式</strong>：{stock_data.get('three_period_pattern') or "-"}</p>
            <p><strong>传导模式</strong>：{stock_data.get('transmission_mode') or "-"}</p>

            <h3>三、大模型建议</h3>
            <ul>
                <li><strong>短期建议</strong>：{stock_data.get('short_term_advice') or "观察"}</li>
                <li><strong>中期建议</strong>：{stock_data.get('mid_term_advice') or "观察"}</li>
                <li><strong>一致性</strong>：{consistency}</li>
            </ul>

            <h3>四、技术指标</h3>
            <table>
                <tr><th>指标</th><th>数值</th><th>状态</th></tr>
                <tr><td>RSI（相对强弱指数）</td><td>{format_value_default(rsi, "-")}</td><td>{rsi_status_display}</td></tr>
                <tr><td>MACD</td><td>-</td><td>{stock_data.get('macd_status') or "-"}</td></tr>
                <tr><td>布林带位置</td><td>{format_value_default(bb_position, "-")}%</td><td>{bb_status}</td></tr>
                <tr><td>均线排列</td><td>-</td><td>{stock_data.get('ma_status') or "-"}</td></tr>
                <tr><td>筹码阻力</td><td>-</td><td>{stock_data.get('chip_resistance') or "-"}</td></tr>
            </table>

            <h3>五、风险评分</h3>
            <table>
                <tr><th>指标</th><th>得分</th></tr>
                <tr><td>风险得分</td><td>{format_value_default(stock_data.get('risk_score'), "-")}</td></tr>
                <tr><td>回报得分</td><td>{format_value_default(stock_data.get('return_score'), "-")}</td></tr>
                <tr><td>综合得分</td><td>{format_value_default(stock_data.get('total_score'), "-")}</td></tr>
                <tr><td>风险建议</td><td>{stock_data.get('risk_advice') or "-"}</td></tr>
            </table>

            <h3>六、网络洞察</h3>
            <ul>
                <li><strong>社区归属</strong>：{stock_data.get('community') or "-"}</li>
                <li><strong>枢纽类型</strong>：{stock_data.get('hub_type') or "-"}</li>
                <li><strong>是否桥梁股</strong>：{stock_data.get('is_bridge') or "-"}</li>
            </ul>

            <h3>七、板块表现</h3>
            <ul>
                <li><strong>所属板块</strong>：{stock_data.get('sector') or "-"}</li>
                <li><strong>板块排名</strong>：第{format_value_default(stock_data.get('sector_rank'), "-")}名</li>
                <li><strong>板块类型</strong>：{stock_data.get('sector_type') or "-"}</li>
            </ul>

            {anomaly_section}

            {dividend_section}

            {operation_box}

            {holder_advice_section}

            {risk_warnings_section}
        </div>
    """


def generate_detailed_stock_email(stock_results: list, market_info: dict, date_str: str) -> str:
    """生成完整的详细个股分析 HTML 邮件"""
    # 生成每只股票的分析部分
    stock_sections = ""
    for stock_data in stock_results:
        stock_sections += generate_stock_section_html(stock_data)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background-color: #ffffff;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 15px;
            margin-top: 0;
            font-size: 24px;
        }}
        .date-info {{
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 12px;
            font-size: 18px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
            font-size: 16px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-good {{
            color: #28a745;
            font-weight: bold;
        }}
        .metric-bad {{
            color: #dc3545;
            font-weight: bold;
        }}
        .metric-neutral {{
            color: #ffc107;
            font-weight: bold;
        }}
        .recommendation {{
            font-size: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: bold;
        }}
        .rec-strong-buy {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .rec-buy {{
            background-color: #cce5ff;
            color: #004085;
            border: 1px solid #b8daff;
        }}
        .rec-hold {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        .rec-sell {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .warning-box {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #dc3545;
        }}
        .info-box {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #17a2b8;
        }}
        .success-box {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 8px 0;
        }}
        .disclaimer {{
            font-size: 12px;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
            line-height: 1.8;
        }}
        .arrow-up {{
            color: #28a745;
            font-weight: bold;
        }}
        .arrow-down {{
            color: #dc3545;
            font-weight: bold;
        }}
        .stock-section {{
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .stock-section:last-child {{
            border-bottom: none;
        }}
        .holder-action-hold {{
            background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-add {{
            background-color: #cce5ff; color: #004085; border: 1px solid #b8daff;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-reduce {{
            background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-exit-profit, .holder-action-exit-loss {{
            background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>【港股个股分析报告】{date_str}</h1>
        <p class="date-info"><strong>分析日期</strong>：{date_str}</p>

        {stock_sections}

        <div class="disclaimer">
            <p><strong>免责声明</strong>：以上建议仅供参考，不构成投资建议，投资有风险，决策需谨慎。</p>
            <p>本报告由港股智能分析系统自动生成，分析日期：{date_str}</p>
        </div>
    </div>
</body>
</html>
"""


def run_detailed_stock_analysis(stock_codes: list, report_path: str, date_str: str, send_email_flag: bool = True) -> bool:
    """
    运行详细个股分析

    参数:
    - stock_codes: 股票代码列表
    - report_path: 综合报告路径
    - date_str: 分析日期
    - send_email_flag: 是否发送邮件

    返回:
    - bool: 是否成功
    """
    print("\n" + "="*50)
    print("🔍 详细个股分析模式")
    print("="*50)
    print(f"📊 分析股票: {', '.join(stock_codes)}")
    print(f"📅 分析日期: {date_str}")
    print(f"📄 报告路径: {report_path}")
    print("")

    results = []
    market_info = {}

    # 读取报告内容
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
    except FileNotFoundError:
        print(f"❌ 报告文件不存在: {report_path}")
        return False

    for stock_code in stock_codes:
        print(f"\n--- 分析 {stock_code} ---")

        # 第一步：提取股票数据
        stock_data = extract_stock_data_with_llm(stock_code, report_content)
        if not stock_data:
            print(f"⚠️ 跳过 {stock_code}")
            continue

        # 第二步：使用大模型进行综合分析
        analysis_result = comprehensive_analyze_with_llm(stock_data)
        if analysis_result:
            stock_data['analysis'] = analysis_result

        # 第三步：生成持货人操作建议
        holder_advice = get_holder_advice_with_llm(stock_data)
        if holder_advice:
            stock_data['holder_advice'] = holder_advice

        results.append(stock_data)

        # 提取市场环境信息
        if not market_info:
            market_info = {
                'hsi_price': stock_data.get('hsi_price'),
                'hsi_change': stock_data.get('hsi_change'),
                'market_status': stock_data.get('market_status'),
                'market_duration': stock_data.get('market_duration'),
                'market_stability': stock_data.get('market_stability'),
                'vix': stock_data.get('vix'),
                'market_sentiment': stock_data.get('market_sentiment'),
                'modularity': stock_data.get('modularity'),
            }

    if not results:
        print("❌ 未能提取任何股票的分析数据")
        return False

    print(f"\n✅ 成功分析 {len(results)} 只股票")

    # 生成 HTML 邮件
    html_content = generate_detailed_stock_email(results, market_info, date_str)

    # 发送邮件
    if send_email_flag:
        print("\n📧 发送邮件...")
        if len(results) == 1:
            stock_name = results[0].get('stock_name', results[0].get('stock_code'))
            subject = f"【个股分析】{stock_name} - {date_str}"
        else:
            subject = f"【个股分析报告】{len(results)}只股票 - {date_str}"

        send_email(subject, "请查看股票分析报告", html_content)
        print("✅ 邮件发送完成")
    else:
        # 保存到文件
        output_path = f'output/stock_analysis_email_{date_str}.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"📄 邮件内容已保存到: {output_path}")

    return True


def run_comprehensive_analysis(llm_filepath, ml_filepath, output_filepath=None,
                             send_email_flag=True, use_deep_analysis=True, use_cached_predictions=False,
                             stock_codes=None):
    """
    运行综合分析

    参数:
    - llm_filepath: 大模型建议文件路径
    - ml_filepath: ML预测结果文件路径（已废弃，保留用于兼容性）
    - output_filepath: 输出文件路径（可选）
    - send_email_flag: 是否发送邮件（默认True）
    - use_deep_analysis: 是否使用深度分析模式进行异常检测（默认True）
    - use_cached_predictions: 是否使用已缓存的三周期预测CSV文件（默认False）
    - stock_codes: 指定股票代码列表，在邮件最前面插入详细个股分析（默认None）
    """
    # 保存 stock_codes 供后续使用（在邮件发送前插入详细个股分析）
    _stock_codes_for_detail = stock_codes

    try:
        print("=" * 80)
        print("🤖 综合分析开始")
        print("=" * 80)
        
        # 检查大模型建议文件是否存在
        if not os.path.exists(llm_filepath):
            print(f"❌ 大模型建议文件不存在: {llm_filepath}")
            return None
        
        # 检查CatBoost单模型预测文件是否存在
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        catboost_csv = os.path.join(data_dir, 'ml_trading_model_catboost_predictions_20d.csv')
        
        if not os.path.exists(catboost_csv):
            print(f"❌ CatBoost预测文件不存在: {catboost_csv}")
            return None
        
        print(f"📊 大模型建议文件: {llm_filepath}")
        print(f"📊 CatBoost预测文件: {catboost_csv}")
        print("")
        
        # 提取大模型建议
        print("📝 提取大模型买卖建议...")
        llm_recommendations = extract_llm_recommendations(llm_filepath)
        print(f"✅ 提取完成\n")
        print(f"   - 短期建议长度: {len(llm_recommendations['short_term'])} 字符")
        print(f"   - 中期建议长度: {len(llm_recommendations['medium_term'])} 字符\n")
        
        # 提取ML预测
        print("📝 提取ML预测结果...")
        ml_predictions = extract_ml_predictions(ml_filepath, use_cached_predictions)
        print(f"✅ 提取完成\n")
        print(f"   - CatBoost模型预测长度: {len(ml_predictions['ensemble'])} 字符\n")

        # 提取用于个股分析的额外数据
        three_horizon_results = ml_predictions.get('three_horizon_results', {})
        chip_data = ml_predictions.get('chip_data', {})
        network_insights = ml_predictions.get('network_insights', {})
        risk_reward_data = ml_predictions.get('risk_reward_data', {})
        historical_pl_data = ml_predictions.get('historical_pl_data', {})

        # 加载模型准确率
        print("📝 加载模型准确率...")
        model_accuracy = load_model_accuracy(horizon=20)
        print(f"✅ 准确率加载完成\n")

        # 生成日期（使用最近交易日，而非当前日期）
        # 这样周末运行时，文件名会是周五的日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_str = get_last_trading_day()
        if date_str != current_date:
            print(f"📅 当前日期 {current_date} 非交易日，使用最近交易日: {date_str}\n")
        
        # 构建综合分析提示词
        prompt = f"""你是一位专业的投资分析师。请根据以下四部分信息，进行综合分析，给出实质的买卖建议。

=== 信息来源 ===

【主要信息源 - 决策依据】

【1. 大模型中期买卖建议（数周-数月）】
{llm_recommendations['medium_term']}

【2. CatBoost模型20天预测结果】
**重要：probability = 上涨概率（不是下跌概率）**
{ml_predictions['ensemble']}

【辅助信息源 - 操作时机参考】

【3. 大模型短期买卖建议（日内/数天）】
{llm_recommendations['short_term']}

**🔴 核心硬性约束（不可违反）**

⚠️ **CatBoost概率约束（最高优先级，无例外）**：
- CatBoost概率 ≤ 0.50 → **绝对禁止推荐买入或强烈买入**
- CatBoost概率 < 0.40 → **绝对禁止推荐持有或观望**
- CatBoost概率 ≥ 0.50 → 可以考虑买入
- CatBoost概率 ≥ 0.60 → 可以考虑强烈买入
- **即使短期和中期方向一致，也绝对不允许违反此约束**
- **违反此约束的建议将被视为错误**

🔥 **决策顺序（严格遵守）**：
第一步：检查CatBoost概率 → 不满足则直接排除
第二步：检查短期和中期一致性
第三步：评估技术面和基本面
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
- **第二步：检查短期和中期一致性**
  - 短期看好，中期看好 → 进入下一步
  - 方向不一致 → 观望
- **第三步：生成建议**
  - 强烈买入：短期看好，中期看好，CatBoost probability ≥ 0.60
  - 买入：短期看好，中期看好，0.50 < CatBoost probability < 0.60
  - 持有/观望：CatBoost probability ≤ 0.50 或 方向不一致
  - 卖出：短期看跌，中期看跌

**硬约束检查清单（必须逐项核对）**：
- [ ] CatBoost probability ≤ 0.50 → 绝对禁止推荐买入或强烈买入
- [ ] CatBoost probability < 0.40 → 绝对禁止推荐持有或观望
- [ ] CatBoost probability ≥ 0.50 → 可以考虑买入
- [ ] CatBoost probability ≥ 0.60 → 可以考虑强烈买入
- [ ] 短期和中期方向是否一致？
- [ ] 三重确认是否全部满足？

**规则2：CatBoost概率评估**

**CatBoost概率阈值**：
- **高置信度上涨**：probability > 0.60
- **中等置信度观望**：0.50 < probability ≤ 0.60
- **预测下跌**：probability ≤ 0.50

**重要说明 - CatBoost probability 定义**：
- `probability` = **上涨概率**（模型预测股票上涨的概率）
- 下跌概率 = 1 - probability
- 例如：probability = 0.35 表示上涨概率35%，下跌概率65%
- 例如：probability = 0.68 表示上涨概率68%，下跌概率32%
- **切勿将 probability 误解为下跌概率**

**阈值优化说明**：
- 当前CatBoost模型20天准确率：约{model_accuracy['catboost']['accuracy']:.2%}（CatBoost 单模型）
- CatBoost模型准确率：{model_accuracy['catboost']['accuracy']:.2%}（±{model_accuracy['catboost']['std']:.2%}）
- 强买入阈值0.60略高于CatBoost准确率，确保高置信度
- 买入阈值0.50接近CatBoost准确率，平衡召回率和精确率
- 卖出阈值0.50确保下跌概率>50%
- 观望区间0.45-0.50避免低置信度决策

**重要说明 - CatBoost模型优势**：
- **单模型策略**：CatBoost 单模型表现最佳（回测收益率 276.74%）
- **自动分类特征处理**：无需手动编码，使用 LabelEncoder 自动处理
- **更好的默认参数**：减少调参工作量，开箱即用
- **稳定性优异**：标准差 ±{model_accuracy['catboost']['std']:.2%}，表现稳定
- **置信度评估**：通过预测概率评估预测可靠性

**重要说明 - 模型不确定性（风险提示）**：
- CatBoost模型存在标准差（±{model_accuracy['catboost']['std']:.2%}），实际准确率可能波动
- 但这**不能**作为降低CatBoost概率标准的理由
- 对于probability在0.50-0.60之间的股票，建议观望而非买入
- 对于probability在0.60-0.70之间的股票，建议降低仓位（2-3%）而非4-6%

**重要说明 - 信号协同（必须同时满足）**：
- **短期信号（触发器）**：负责"何时做"（Timing）→ 必须100%满足
- **中期信号（确认器）**：负责"是否做"（Direction）→ 必须100%满足
- **CatBoost概率（硬性约束）**：负责验证方向性→ 必须100%满足
- **三重确认：短期、中期、CatBoost三者必须同时满足，缺一不可**

**重要说明 - 时间维度标准化**：
- 短期：1-5个交易日（日内到一周）
- 中期：10-20个交易日（2-4周）
- 长期：>20个交易日（超过1个月）
- 当前映射：大模型短期建议 ↔ CatBoost模型预测（20天），大模型中期建议 ↔ 基本面分析（数周-数月）✅

**规则3：CatBoost概率评估**
- **高置信度上涨（probability > 0.60）**：信号可靠性最高，优先级提升
- **中等置信度观望（0.50 < probability ≤ 0.60）**：信号可靠性中等，需要短期中期一致支持
- **预测下跌（probability ≤ 0.50）**：信号可靠性低，建议观望，不进行交易
- 如果probability高（>0.60），综合置信度最高
- 如果probability低（≤0.50），降低为中等置信度

**规则4：推荐理由格式**
- 必须说明：短期建议+中期建议+CatBoost预测（probability）
- 例如："短期建议买入（触发器），中期建议买入（确认器），CatBoost预测上涨概率0.72（高置信度），综合置信度高。注意CatBoost模型当前准确率约{model_accuracy['catboost']['accuracy']:.2%}（标准差约±{model_accuracy['catboost']['std']:.2%}），probability在0.72附近实际准确率可能在{model_accuracy['catboost']['accuracy']-model_accuracy['catboost']['std']:.2%} ~ {model_accuracy['catboost']['accuracy']+model_accuracy['catboost']['std']:.2%}之间"

请基于上述规则，完成以下任务：

1. **一致性分析**（方案A核心：短期触发 + 中期确认 + CatBoost验证）：
   - **第一步（核心）**：分析短期建议与中期建议的一致性
     - 短期买入 + 中期买入 → 方向一致，考虑CatBoost验证
     - 短期买入 + 中期观望 → 等待中期确认
     - 短期买入 + 中期卖出 → 冲突，观望
     - 短期卖出 + 中期卖出 → 方向一致，考虑CatBoost验证
     - 短期卖出 + 中期观望 → 等待中期确认
     - 短期卖出 + 中期买入 → 冲突，观望
   - **第二步（验证）**：对短期中期一致的股票，分析CatBoost预测验证
     - 如果CatBoost高置信度支持（probability>0.60），提升为强信号
     - 如果CatBoost中等置信度支持（0.50<probability≤0.60），提升为中等信号
     - 如果CatBoost低置信度（probability≤0.50），降低为弱信号或观望
   - 标注符合"强买入信号"、"买入信号"、"观望信号"、"卖出信号"的股票

2. **个股建议排序**：
   - 优先级：强买入信号 > 买入信号 > 观望信号 > 卖出信号
   - 在相同优先级内，按probability排序
   - 对每个股票给出明确的操作建议：强烈买入、买入、持有、卖出、强烈卖出

3. **综合推荐清单**：
   - 强烈买入信号（2-3只）：最高优先级，建议仓位4-6%
   - 买入信号（3-5只）：次优先级，建议仓位2-4%
   - 持有/观望（如有）：第三优先级
   - 卖出信号（如有）：最低优先级

3.1. **特殊处理（CatBoost probability ≤ 0.50的股票）**：
   - **绝对禁止**： probability ≤ 0.50 的股票不能出现在"强烈买入信号"或"买入信号"中
   - **正确处理**： probability ≤ 0.50 的股票应该出现在"持有/观望"或"卖出信号"中
   - **理由说明**：在"推荐理由"中必须明确说明"CatBoost probability ≤ 0.50，违反硬约束，建议观望"
   - **示例**："短期建议买入，中期建议买入，但CatBoost预测上涨概率0.48（≤0.50），违反硬约束，建议观望"

4. **风险提示**：
   - 分析当前市场整体风险
   - 给出仓位控制建议（建议仓位百分比，总仓位45%-55%）
   - 给出止损位建议（单只股票最大亏损不超过-8%）
   
   **特别要求 - 考虑CatBoost模型不确定性**：
   - CatBoost模型20天标准差约±{model_accuracy['catboost']['std']:.2%}
   - 对于probability在0.55-0.65之间的股票，建议仓位不超过2-3%
   - 强买入信号（短期/中期一致且CatBoost高置信度）建议仓位4-6%
   - 总仓位控制在45%-55%
   - **必须设置止损位，单只股票最大亏损不超过-8%**
   - **严格遵循"短期触发 + 中期确认 + CatBoost验证"原则**：只有短期和中期方向一致且CatBoost验证时才行动
   - 如果短期和中期建议冲突，优先选择观望，不进行交易
   - 采用"三重确认"策略：短期、中期、CatBoost三者一致时才重仓操作

请按照以下格式输出（不要添加任何额外说明文字）：

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] 
   - 推荐理由：[简短理由，例如：短期买入，中期买入，CatBoost预测上涨概率0.72（高置信度），方向一致]
   - 操作建议：买入/卖出/持有/观望
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 买入信号（3-5只）
1. [股票代码] [股票名称] 
   - 推荐理由：[简短理由]
   - 操作建议：买入/持有
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 持有/观望
1. [股票代码] [股票名称] 
   - 推荐理由：[观望理由]
   - 操作建议：持有/观望
   - 关注要点：[需要关注的关键指标或事件]
   - 风险提示：[主要风险因素]

## 卖出信号（如有）
1. [股票代码] [股票名称] 
   - 推荐理由：[卖出理由]
   - 操作建议：卖出/减仓
   - 建议卖出价：HK$XX.XX
   - 止损位（如持有）：HK$XX.XX（-X.X%）
   - 风险提示：[主要风险因素]

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 建议仓位百分比：[X]%
- 止损位设置：[策略]
- 组合调整建议：[具体的组合调整建议]

---
分析日期：{date_str}
"""
        
        print("🤖 提交大模型进行综合分析...")
        print("")
        
        # 调用大模型（关闭思考模式，避免输出思考过程）
        response = chat_with_llm(prompt, enable_thinking=False)
        
        if response:
            print("✅ 综合分析完成\n")
            print("=" * 80)
            print("📊 综合买卖建议")
            print("=" * 80)
            print("")
            print(response)
            print("")
            print("=" * 80)
            
            # 保存到文件
            if output_filepath is None:
                output_filepath = f'data/comprehensive_recommendations_{date_str}.txt'
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(f"{'=' * 80}\n")
                f.write(f"综合买卖建议\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析日期: {date_str}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(response)
            
            print(f"✅ 综合建议已保存到 {output_filepath}")
            
            # 发送邮件通知
            if send_email_flag:
                print("\n📧 准备发送邮件通知...")
                email_subject = f"【综合分析】港股买卖建议 - {date_str}"
                email_content = response
                
                # 构建板块分析、股息信息、恒生指数分析
                print("📊 获取板块分析...")
                sector_data = get_sector_analysis()
                
                print("📊 获取股息信息...")
                dividend_data = get_dividend_info()
                
                print("📊 获取恒生指数分析...")
                hsi_data = get_hsi_analysis()
                
                print("📊 获取 hsi_email.py 实时指标...")
                hsi_email_indicators = get_hsi_email_indicators()
                
                print("📊 获取股票异常检测...")
                # 使用快速模式（Z-Score only），提高综合分析速度
                # 如需使用深度分析模式，通过 use_deep_analysis=True 参数控制
                anomaly_data = get_stock_anomalies(use_deep_analysis=use_deep_analysis)
                
                # 构建板块分析文本
                sector_text = ""
                if sector_data and sector_data['performance'] is not None:
                    perf_df = sector_data['performance']
                    sector_leaders = sector_data['leaders']

                    sector_text += "| 排名 | 板块名称 | 类型 | 平均涨跌幅 | 龙头股TOP 3 |\n"
                    sector_text += "|------|---------|------|-----------|-------------|\n"

                    for idx, row in perf_df.iterrows():
                        trend_icon = "🔥" if row['avg_change_pct'] > 2 else "📈" if row['avg_change_pct'] > 0 else "📉"
                        change_color = "+" if row['avg_change_pct'] > 0 else ""
                        sector_type = get_sector_type(row['sector_code'])

                        leaders_text = ""
                        if row['sector_code'] in sector_leaders:
                            leaders = sector_leaders[row['sector_code']]
                            # 显示所有3个龙头股，使用斜线分隔避免与Markdown表格冲突
                            leader_items = []
                            for i, leader in enumerate(leaders, 1):
                                leader_items.append(f"{leader['name']}({leader['change_pct']:+.2f}%)")
                            leaders_text = " / ".join(leader_items)

                        sector_text += f"| {idx+1} | {trend_icon} {row['sector_name']} | {sector_type} | {change_color}{safe_float_format(row['avg_change_pct'], '.2f')}% | {leaders_text} |\n"
                    
                    # 添加投资建议（基于板块轮动规律）
                    sector_text += "\n**投资建议（基于板块轮动规律）**：\n\n"

                    # 获取当前市场状态
                    current_market = get_current_market_state()

                    if current_market:
                        market_state = current_market['market_state']
                        market_state_cn = current_market['market_state_cn']
                        recent_20d_return = current_market['recent_20d_return']

                        sector_text += f"**当前市场状态**：{market_state_cn}（HSI 20日收益：{recent_20d_return*100:+.2f}%）\n\n"

                        # 统计周期/防御板块表现
                        cyclical_avg = 0
                        defensive_avg = 0
                        cyclical_count = 0
                        defensive_count = 0

                        for idx, row in perf_df.iterrows():
                            sector_type = get_sector_type(row['sector_code'])
                            change = row['avg_change_pct'] if not pd.isna(row['avg_change_pct']) else 0
                            if sector_type == '周期':
                                cyclical_avg += change
                                cyclical_count += 1
                            elif sector_type == '防御':
                                defensive_avg += change
                                defensive_count += 1

                        if cyclical_count > 0:
                            cyclical_avg = cyclical_avg / cyclical_count
                        if defensive_count > 0:
                            defensive_avg = defensive_avg / defensive_count

                        sector_text += f"**板块表现对比**：\n"
                        sector_text += f"- 周期板块平均：{'+' if cyclical_avg > 0 else ''}{cyclical_avg:.2f}%\n"
                        sector_text += f"- 防御板块平均：{'+' if defensive_avg > 0 else ''}{defensive_avg:.2f}%\n\n"

                        # 根据市场状态给出配置建议
                        if market_state == 'bull':
                            sector_text += "**牛市配置建议**：\n"
                            sector_text += "- ✅ **重仓周期板块**：牛市中周期板块占优比例74.3%\n"
                            sector_text += "- ✅ **推荐配置**：半导体30%、生物医药25%、科技20%、消费15%\n"
                            sector_text += "- ✅ **总仓位建议**：90%\n"
                            if cyclical_avg > defensive_avg:
                                sector_text += f"- 📈 当前周期板块表现优于防御板块，符合牛市规律\n"
                        elif market_state == 'bear':
                            sector_text += "**熊市配置建议**：\n"
                            sector_text += "- 🛡️ **重仓防御板块**：熊市中防御板块占优比例65.9%\n"
                            sector_text += "- 🛡️ **推荐配置**：保险30%、公用事业25%、银行20%\n"
                            sector_text += "- 🛡️ **总仓位建议**：75%（保留25%现金）\n"
                            sector_text += "- ⚠️ **注意**：银行股Walk-forward夏普-0.053，不宜重仓\n"
                            if defensive_avg > cyclical_avg:
                                sector_text += f"- 📉 当前防御板块表现优于周期板块，符合熊市规律\n"
                        else:
                            sector_text += "**震荡市配置建议**：\n"
                            sector_text += "- 🔄 **均衡配置**：周期/防御各50%\n"
                            sector_text += "- 🔄 **动态调整**：根据相对强弱排名选择前3板块\n"
                            sector_text += "- 🔄 **总仓位建议**：60%（保留40%现金）\n"
                            sector_text += "- 📊 关注相对强势板块，每5日重新评估\n"

                        sector_text += "\n**⚠️ 重要提醒**：\n"
                        sector_text += "- 领涨频率高≠策略收益高，需以Walk-forward验证结果为准\n"
                        sector_text += "- 动量/反转策略效果有限，不建议单独使用\n"
                        sector_text += "- 板块轮动周期约28天，每2-3周评估轮动时机\n"

                    else:
                        # 无法获取市场状态时的默认建议
                        sector_text += "- 📊 请结合恒生指数20日收益判断市场状态\n"
                        sector_text += "- 牛市配置周期板块，熊市配置防御板块\n"
                        sector_text += "- 单股仓位不超过15%，单板块不超过30%\n"
                
                # 构建股息信息文本
                dividend_text = ""
                if dividend_data:
                    dividend_text += "| 股票代码 | 股票名称 | 除净日 | 分红方案 |\n"
                    dividend_text += "|---------|---------|-------|----------|\n"
                    
                    for stock in dividend_data[:10]:
                        code = stock.get('股票代码', 'N/A')
                        name = stock.get('股票名称', 'N/A')
                        ex_date = stock.get('除净日', 'N/A')
                        dividend_plan = stock.get('分红方案', 'N/A')
                        # 格式化除净日
                        if isinstance(ex_date, pd.Timestamp):
                            ex_date = ex_date.strftime('%Y-%m-%d')
                        # 截断过长的分红方案
                        if dividend_plan != 'N/A' and len(str(dividend_plan)) > 30:
                            dividend_plan = str(dividend_plan)[:28] + '...'
                        dividend_text += f"| {code} | {name} | {ex_date} | {dividend_plan} |\n"
                
                # 获取当前市场状态
                current_market = get_current_market_state()
                
                # 构建恒生指数分析文本
                hsi_text = "**技术指标**:\n\n"
                if hsi_data:
                    hsi_text += f"- 当前价格：{safe_float_format(hsi_data['current_price'], '.2f')}\n"
                    hsi_text += f"- 日涨跌幅：{safe_float_format(hsi_data['change_pct'], '+.2f')}% ({safe_float_format(hsi_data['change_points'], '+.2f')} 点)\n"
                    hsi_text += f"- RSI（14日）：{safe_float_format(hsi_data['rsi'], '.2f')}\n"
                    hsi_text += f"- MA20：{safe_float_format(hsi_data['ma20'], '.2f')}\n"
                    hsi_text += f"- MA50：{safe_float_format(hsi_data['ma50'], '.2f')}\n"
                    # 成交金额数据（优先显示）
                    if hsi_data.get('amount') is not None:
                        hsi_text += f"- 成交金额：{hsi_data['amount']:.2f}亿港元\n"
                        if hsi_data.get('amount_ratio') is not None:
                            hsi_text += f"- 成交金额比率：{hsi_data['amount_ratio']:.2f}x（相对20日均值）\n"
                        else:
                            hsi_text += "- 成交金额比率：N/A（无法获取历史数据）\n"
                    else:
                        hsi_text += "- 成交金额：N/A（数据暂未更新）\n"
                        hsi_text += "- 成交金额比率：N/A\n"

                if current_market:
                    hsi_text += f"- 市场信号: {current_market['market_signal']}\n"
                    hsi_text += f"- 市场状态: {current_market['market_state_cn']}\n"
                    hsi_text += f"- 最近20天收益率: {current_market['recent_20d_return']:.2%}\n"
                    hsi_text += f"- 最近5天收益率: {current_market['recent_5d_return']:.2%}\n"

                    # ========== 新增：显示 Regime_Duration && HMM 状态转换概率解读 ==========
                    if current_market.get('regime_duration') is not None and current_market.get('regime_transition_prob') is not None:
                        regime_duration = current_market['regime_duration']
                        regime_state_cn = current_market.get('regime_state_cn', '未知')
                        regime_stability = current_market.get('regime_stability', '')

                        transition_prob = current_market['regime_transition_prob']
                        switch_prob_5d = current_market['regime_switch_prob_5d']
                        expected_duration = current_market['regime_expected_duration']

                        # 动态判断当前状态稳定性
                        if switch_prob_5d < 0.23:
                            stability_desc = "处于实际范围的最低端，状态高度稳定，短期内几乎不会转换"
                        elif switch_prob_5d < 0.67:
                            stability_desc = "处于中等范围，状态有一定稳定性"
                        else:
                            stability_desc = "处于较高范围，状态不稳定，可能即将转换"

                        hsi_text += f"- 市场状态持续时间: {regime_state_cn} 持续 **{regime_duration}** 天 ({regime_stability})\n"
                        hsi_text += f"- HMM 状态转换概率: 当前5日转换概率为 {switch_prob_5d:.2%}，{stability_desc}。\n"

                        # 添加状态稳定性说明
                        hsi_text += "\n**状态持续性说明**:\n"
                        hsi_text += "- ⚠️ 不稳定（<5天）：市场状态频繁转换，建议降低仓位\n"
                        hsi_text += "- 🟡 中等（5-15天）：市场状态中等稳定，可正常交易\n"
                        hsi_text += "- ✅ 稳定（>15天）：市场状态稳定，趋势明确\n\n"

                        hsi_text += "**HMM 状态转换概率说明**:\n\n"
                        hsi_text += "| 指标 | 数值 | 含义 |\n"
                        hsi_text += "|------|------|------|\n"
                        hsi_text += f"| 转换概率 | {transition_prob:.2%} | 从当前状态转换到其他状态的概率 |\n"
                        hsi_text += f"| 5日转换概率 | {switch_prob_5d:.2%} | 5天内离开当前状态的概率 |\n"
                        hsi_text += f"| 期望剩余持续时间 | {expected_duration:.1f} 天 | 当前状态平均还能持续多久 |\n\n"

                    # 添加市场状态说明表格
                    hsi_text += "**市场状态说明**:\n\n"
                    hsi_text += "| 市场状态 | 20天收益率范围 | 说明 |\n"
                    hsi_text += "|---------|--------------|------|\n"
                    hsi_text += "| 📈 牛市 | > 5% | 市场强劲上涨，适合积极配置 |\n"
                    hsi_text += "| ⬆️ 震荡偏涨 | 2% - 5% | 市场温和上涨，可以谨慎配置 |\n"
                    hsi_text += "| ➡️ 震荡市 | -2% - 2% | 市场横盘整理，建议观望 |\n"
                    hsi_text += "| ⬇️ 震荡偏跌 | -5% - -2% | 市场温和下跌，建议减仓 |\n"
                    hsi_text += "| 📉 熊市 | < -5% | 市场强劲下跌，建议空仓 |\n\n"

                    # 添加投资建议
                    hsi_text += "### 投资建议\n\n"
                    
                    if current_market['market_state'] == 'bull':
                        hsi_text += "**牛市策略**:\n\n"
                        hsi_text += "- ✅ **重仓高市场关联性股票**: 牛市中高关联性股票平均收益率可达 +9.35%\n"
                        hsi_text += "- ✅ **关注科技、半导体板块**: 这些板块通常在牛市中表现优异\n"
                        hsi_text += "- ✅ **使用100%仓位**: 市场信号强烈，可全仓操作\n\n"
                    elif current_market['market_state'] == 'bear':
                        hsi_text += "**熊市策略**:\n\n"
                        hsi_text += "- ⚠️ **重仓低市场关联性股票**: 熊市中低关联性股票平均收益率为 +4.15%\n"
                        hsi_text += "- ⚠️ **配置银行、公用事业**: 这些股票具有防御性\n"
                        hsi_text += "- ⚠️ **降低仓位至30%**: 市场风险较高，控制仓位\n\n"
                    elif current_market['market_state'] in ['neutral_bull', 'neutral_bear']:
                        hsi_text += "**震荡市策略**:\n\n"
                        hsi_text += "- 🔄 **均衡配置**: 高低关联性股票各占50%\n"
                        hsi_text += "- 🔄 **动态调整**: 根据市场信号及时调整仓位\n"
                        hsi_text += "- 🔄 **关注波段机会**: 震荡市适合波段操作\n\n"
                        
                        # 额外的风险提示
                        if current_market['market_state'] == 'neutral_bear':
                            hsi_text += "**风险提示**:\n\n"
                            hsi_text += "- ⚠️ 市场温和下跌，建议保持谨慎\n"
                            hsi_text += "- ⚠️ 可考虑降低仓位至70%\n"
                        else:
                            hsi_text += "**机会提示**:\n\n"
                            hsi_text += "- ✅ 市场温和上涨，可考虑逐步加仓\n"
                            hsi_text += "- ✅ 建议仓位可提升至80%\n\n"
                    else:  # neutral
                        hsi_text += "**横盘策略**:\n\n"
                        hsi_text += "- ⏸️ **观望为主**: 市场缺乏明确方向，建议保持观望\n"
                        hsi_text += "- ⏸️ **低仓位试探**: 可用30%仓位试探性配置\n"
                        hsi_text += "- ⏸️ **等待信号**: 等待市场明确方向后再做决策\n\n"
                    
                
                # 使用配置文件中的所有自选股
                stock_codes = list(WATCHLIST.keys())
                print(f"📊 使用配置文件中的 {len(stock_codes)} 只自选股生成技术指标表格")
                
                # 生成技术指标表格
                print("📊 生成推荐股票技术指标表格...")
                technical_indicators_table = generate_technical_indicators_table(stock_codes)
                if not technical_indicators_table:
                    print("⚠️ 技术指标表格为空，可能是股票数据获取失败")
                
                # 添加 hsi_email.py 的实时指标内容
                hsi_email_text = ""
                if hsi_email_indicators:
                    hsi_email_text, _ = format_hsi_email_indicators(hsi_email_indicators)
                
                # 添加最近48小时模拟交易记录
                recent_transactions_df = get_recent_transactions(hours=48)
                recent_transactions_text = format_recent_transactions(recent_transactions_df)
                
                # 构建完整的邮件内容（综合买卖建议 + 信息参考）
                # 注意：不添加标题，因为HTML模板已经有了标题
                full_content = f"""{response}

---

# 信息参考

## 一、恒生指数技术分析

{hsi_text}

## 二、机器学习预测结果（20天）

### CatBoost模型（三周期准确率）

**模型准确率**：

| 周期 | 准确率 | 标准差 |
|------|--------|--------|
| 1天 | **{model_accuracy['1d']['accuracy']:.2%}** | ±{model_accuracy['1d']['std']:.2%} |
| 5天 | **{model_accuracy['5d']['accuracy']:.2%}** | ±{model_accuracy['5d']['std']:.2%} |
| 20天 | **{model_accuracy['20d']['accuracy']:.2%}** | ±{model_accuracy['20d']['std']:.2%} |

{ml_predictions.get('ensemble_email', ml_predictions.get('ensemble', ''))}

## 三、股票异常检测提醒

"""

                # 添加异常检测结果
                full_content += generate_anomaly_report_content(anomaly_data)
                
                full_content += f"""

## 四、板块分析（5日涨跌幅排名）

{sector_text}

## 五、大模型建议

### 短期买卖建议（日内/数天）
{llm_recommendations['short_term']}

### 中期买卖建议（数周-数月）
{llm_recommendations['medium_term']}

## 六、股息信息（即将除净）

{dividend_text}

## 七、股票技术指标详情

{technical_indicators_table}

## 八、最近48小时模拟交易记录

{recent_transactions_text}

"""
                
                # 继续添加其他内容
                full_content += f"""
## 九、技术指标说明

**短期技术指标（日内/数天）**：
- RSI（相对强弱指数）：超买>70，超卖<30
- MACD：金叉（上涨信号），死叉（下跌信号）
- 布林带：价格突破上下轨预示反转
- 成交量：放大配合价格上涨=买入信号
- OBV（能量潮）：反映资金流向

**中期技术指标（数周-数月）**：
- 均线排列：多头排列（MA5>MA10>MA20>MA50）= 上升趋势
- 均线斜率：上升=趋势向上，下降=趋势向下
- 乖离率：价格偏离均线的程度
- 支撑阻力位：重要价格支撑和阻力
- 相对强度：相对于恒生指数的表现
- 中期趋势评分：0-100分，≥80买入，30-45卖出

**重要说明**：
- 短期指标用于捕捉买卖时机（Timing）
- 中期指标用于确认趋势方向（Direction）
- 短期和中期方向一致时，信号最可靠
- 短期和中期冲突时，选择观望

## 十、**决策框架**



### ✦ 买入策略

- **CatBoost 概率 ≥ 0.60** + **短期看好** + **中期看好** → 强烈买入

- **0.50 < CatBoost 概率 < 0.60** + **短期看好** + **中期看好** → 买入

- **CatBoost 概率 ≤ 0.50** → 禁止买入（硬约束）



### ✦ 持有策略

- **CatBoost 概率 > 0.60** + **大模型建议买入** → 强烈持有

- **0.50 < CatBoost 概率 ≤ 0.60** + **大模型建议买入** → 观望持有

- **CatBoost 概率 ≤ 0.50** + **大模型建议卖出** → 考虑卖出



### ✦ 卖出策略

- **CatBoost 概率 ≤ 0.50** + **短期看跌** + **中期看跌** → 卖出

- **CatBoost 概率 < 0.40** → 禁止持有或观望（硬约束）



### ✦ 决策顺序（严格遵守）

1. **第一步：检查 CatBoost 概率（硬约束）**

   - CatBoost 概率 ≤ 0.50 → 排除买入或强烈买入

   - CatBoost 概率 ≥ 0.50 → 进入下一步

2. **第二步：检查短期和中期一致性**

   - 短期看好 + 中期看好 → 进入下一步

   - 方向不一致 → 观望

3. **第三步：生成建议**

   - 强烈买入：短期看好 + 中期看好 + CatBoost 概率 ≥ 0.60

   - 买入：短期看好 + 中期看好 + 0.50 < CatBoost 概率 < 0.60

   - 持有/观望：CatBoost 概率 ≤ 0.50 或 方向不一致

   - 卖出：短期看跌 + 中期看跌 + CatBoost 概率 ≤ 0.50

### ✦ 动态置信度阈值策略（根据市场环境调整）

根据市场环境动态调整置信度阈值，可显著提升风险调整收益和过滤噪声：

| 市场环境 | 置信度阈值 | 调整幅度 | 说明 |
|---------|-----------|---------|------|
| 牛市 (bull) | 0.55 | -0.05 | 更激进，增加交易机会 |
| 震荡市 (normal/ranging) | 0.65 | +0.05 | 更严格过滤噪声 |
| 熊市 (bear) | 0.60 | 基准 | 中等保守 |

**市场环境识别方法**：
- **牛市**：恒生指数 20 天收益率 > +5%
- **熊市**：恒生指数 20 天收益率 < -5%
- **震荡市**：恒生指数 20 天收益率在 -5% 到 +5% 之间

**使用建议**：
- 震荡市使用阈值 0.65 可显著过滤噪声，减少亏损交易
- 牛市使用阈值 0.55 可捕捉更多机会，不影响收益率
- 熊市使用阈值 0.60 保持中等保守策略

### ✦ 强烈买入信号
**强烈买入信号**是在每日综合分析邮件中的第一部分，包含：
- **股票代码和名称**：如"0700.HK 腾讯控股"
- **推荐理由**：详细的分析，说明短期建议+中期建议+CatBoost预测
- **操作建议**：明确的买入/持有/卖出建议
- **价格指引**：建议买入价、止损位、目标价
- **风险提示**：主要风险因素

## 十一、风险提示

1. **模型不确定性**：
   - ML 20天 CatBoost模型标准差为±{model_accuracy['catboost']['std']:.2%}
   - 融合预测概率>0.60为高置信度上涨，0.50-0.60为中等置信度观望，≤0.50为预测下跌
   - 建议：短期和中期一致是主要决策依据，ML预测用于验证和提升置信度

2. **市场风险**：
   - 当前市场整体风险：[高/中/低]（需根据恒生指数技术指标判断）
   - 建议仓位：45%-55%
   - **必须设置止损位，单只股票最大亏损不超过-8%**

3. **投资原则**：
   - 短期触发 + 中期确认 + ML验证 = 高置信度信号
   - 短期和中期冲突 = 观望（避免不确定性）
   - CatBoost概率在0.50-0.60之间 = 中等置信度，建议观望或轻仓
   - 总仓位控制在45%-55%，分散风险

## 十二、数据来源

- 大模型分析：Qwen大模型
- ML预测：CatBoost（单模型）
- 特征工程：2991个原始特征，500个精选特征（F-test+互信息混合方法）
- 技术指标：RSI、MACD、布林带、ATR、均线、成交量等80+个指标
- 基本面数据：PE、PB、ROE、ROA、股息率等8个指标
- 美股市场：标普500、纳斯达克、VIX、美国国债收益率等11个指标
- 股票类型：18个行业分类及衍生评分
- 情感分析：四维情感评分（Relevance/Impact/Expectation_Gap/Sentiment）
- 板块分析：16个板块涨跌幅排名、技术趋势分析、龙头识别
- 主题建模：LDA主题建模（10个主题）
- 主题情感交互：10个主题 × 5个情感指标 = 50个交互特征
- 预期差距：新闻情感相对于市场预期的差距（5个特征）
- 模型策略：CatBoost 单模型
- 置信度评估：高（>0.60）、中（0.50-0.60）、低（≤0.50）

"""

                # 如果有hsi_email.py指标，添加到数据源部分
                if hsi_email_indicators:
                    full_content += f"""
- **实时指标**：恒生指数及自选股实时技术指标，包括TAV评分、建仓/出货评分、基本面评分等高级分析指标
"""

                full_content += f"""

---
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析日期：{date_str}
"""

                # 生成HTML格式邮件内容（将完整内容转换为HTML）
                html_content = generate_html_email(full_content, date_str)

                # 如果指定了股票代码，在邮件最前面插入详细个股分析
                if _stock_codes_for_detail:
                    print(f"\n📊 生成详细个股分析: {', '.join(_stock_codes_for_detail)}")

                    # 确定报告路径
                    report_path = f'output/comprehensive_reports/{date_str}.md'

                    # 提取个股数据
                    stock_results = []
                    market_info = {}
                    for stock_code in _stock_codes_for_detail:
                        # 构建精简文本（从已计算数据提取，避免大模型超时）
                        stock_realtime = get_stock_realtime_data(stock_code) or {}
                        compact_text = build_stock_data_for_llm(
                            stock_code,
                            three_horizon_results,
                            chip_data,
                            risk_reward_data,
                            historical_pl_data,
                            network_insights,
                            anomaly_data,
                            current_market,
                            {stock_code: stock_realtime},  # 单只股票的实时数据
                            sector_data
                        )
                        stock_data = extract_stock_data_with_llm(stock_code, compact_text)
                        if stock_data:
                            # 综合分析
                            analysis_result = comprehensive_analyze_with_llm(stock_data)
                            if analysis_result:
                                stock_data['analysis'] = analysis_result
                            # 持货人建议
                            holder_advice = get_holder_advice_with_llm(stock_data)
                            if holder_advice:
                                stock_data['holder_advice'] = holder_advice
                            stock_results.append(stock_data)
                            # 提取市场环境信息
                            if not market_info:
                                market_info = {
                                    'hsi_price': stock_data.get('hsi_price'),
                                    'hsi_change': stock_data.get('hsi_change'),
                                    'market_status': stock_data.get('market_status'),
                                    'market_duration': stock_data.get('market_duration'),
                                    'market_stability': stock_data.get('market_stability'),
                                    'vix': stock_data.get('vix'),
                                    'market_sentiment': stock_data.get('market_sentiment'),
                                    'modularity': stock_data.get('modularity'),
                                }

                    if stock_results:
                        # 生成详细个股分析 HTML
                        detailed_html = generate_detailed_stock_email(stock_results, market_info, date_str)

                        # 从详细个股分析 HTML 中提取 body 内容
                        import re
                        body_match = re.search(r'<body[^>]*>(.*?)</body>', detailed_html, re.DOTALL)
                        if body_match:
                            detailed_body = body_match.group(1)
                            # 提取 container div
                            container_match = re.search(r'<div class="container">(.*?)</div>\s*</body>', detailed_html, re.DOTALL)
                            if container_match:
                                detailed_content = container_match.group(1)
                                # 在综合报告 HTML 的 container 开头插入详细个股分析
                                # 找到 <div class="container"> 的位置
                                container_start = html_content.find('<div class="container">')
                                if container_start != -1:
                                    # 插入详细个股分析
                                    insert_pos = container_start + len('<div class="container">')
                                    html_content = html_content[:insert_pos] + detailed_content + html_content[insert_pos:]

                        # 更新邮件主题
                        if len(stock_results) == 1:
                            stock_name = stock_results[0].get('stock_name', stock_results[0].get('stock_code'))
                            email_subject = f"【个股分析】{stock_name} - {date_str}"
                        else:
                            email_subject = f"【个股分析报告】{len(stock_results)}只股票 - {date_str}"

                        print(f"✅ 已在邮件最前面插入 {len(stock_results)} 只股票的详细分析")

                    send_email(email_subject, full_content, html_content)

            # 保存 MD 文档（用于知识库）- 无论是否发送邮件都保存
            save_comprehensive_report_md(response, date_str)

            return response
        else:
            print("❌ 大模型分析失败")
            return None
        
    except Exception as e:
        print(f"❌ 综合分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='综合分析脚本 - 整合大模型建议和ML预测结果\n'
                    '⚠️ 建议在港股收市后（16:00 HKT）运行'
    )
    parser.add_argument('--llm-file', type=str, default=None,
                       help='大模型建议文件路径 (默认使用今天的文件)')
    parser.add_argument('--ml-file', type=str, default=None,
                       help='ML预测结果文件路径 (默认使用今天的文件)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认保存到data/comprehensive_recommendations_YYYY-MM-DD.txt)')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件通知')
    parser.add_argument('--no-deep-analysis', action='store_true',
                       help='不使用深度分析模式进行异常检测（默认使用深度分析）')
    parser.add_argument('--use-cached-predictions', action='store_true',
                       help='使用已缓存的三周期预测CSV文件（跳过模型预测）')
    parser.add_argument('--stocks', type=str, default=None,
                       help='指定股票代码生成详细个股分析邮件（逗号分隔，如 2318.HK,0700.HK）')

    args = parser.parse_args()

    # 检查运行时机（只在交易日盘中显示警告）
    now = datetime.now()
    weekday = now.weekday()  # 0=周一, 6=周日
    current_hour = now.hour

    # 周末不显示警告（市场休市，使用最近交易日数据）
    if weekday < 5 and current_hour < 16:  # 周一到周五，且未到16:00
        print("⚠️ 警告: 当前时间未到港股收市时间（16:00 HKT）")
        print("   市场情绪过滤器可能使用不完整的当日数据")
        print("   建议在收市后运行以获得准确结果")
        print("")

    # 生成日期
    # 使用最近交易日作为日期（周末运行时使用周五日期）
    date_str = get_last_trading_day()

    # 默认文件路径
    if args.llm_file is None:
        args.llm_file = f'data/llm_recommendations_{date_str}.txt'

    if args.ml_file is None:
        args.ml_file = f'data/ml_predictions_20d_{date_str}.txt'

    # 解析股票代码（如果指定）
    stock_codes = None
    if args.stocks:
        stock_codes = [code.strip() for code in args.stocks.split(',') if code.strip()]

    # 运行综合分析
    result = run_comprehensive_analysis(args.llm_file, args.ml_file, args.output,
                                       send_email_flag=not args.no_email,
                                       use_deep_analysis=not args.no_deep_analysis,
                                       use_cached_predictions=args.use_cached_predictions,
                                       stock_codes=stock_codes)
    
    if result:
        print("\n✅ 综合分析完成！")
        sys.exit(0)
    else:
        print("\n❌ 综合分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
