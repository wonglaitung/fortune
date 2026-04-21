#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析脚本 - 整合大模型建议和ML预测结果
生成综合的买卖建议
"""

import os
import sys
import argparse
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
# 验证数据：56只港股，Walk-forward 30 folds，完整模型（545特征）
# 更新日期：2026-04-20
THREE_HORIZON_PATTERNS = {
    '010': {'name': '反弹失败⭐', 'action': '谨慎减仓', 'win_rate': '66.32%', 'avg_return': '-1.65%', 'confidence': '高'},
    '000': {'name': '一致看跌', 'action': '止损/减仓', 'win_rate': '63.14%', 'avg_return': '-2.54%', 'confidence': '中高'},
    '100': {'name': '冲高回落', 'action': '获利了结', 'win_rate': '62.56%', 'avg_return': '-2.28%', 'confidence': '中高'},
    '001': {'name': '下跌中继', 'action': '谨慎观望', 'win_rate': '61.45%', 'avg_return': '+4.86%', 'confidence': '中'},
    '011': {'name': '探底回升', 'action': '分批建仓', 'win_rate': '60.43%', 'avg_return': '+3.54%', 'confidence': '中'},
    '101': {'name': '假突破', 'action': '持有观望', 'win_rate': '50.00%', 'avg_return': '+2.21%', 'confidence': '低'},
    '110': {'name': '震荡回调', 'action': '观望', 'win_rate': '48.48%', 'avg_return': '+2.34%', 'confidence': '低'},
    '111': {'name': '一致看涨', 'action': '谨慎持有', 'win_rate': '47.99%', 'avg_return': '+0.14%', 'confidence': '低'},
}

# 传导律准确率数据（来源：docs/THREE_HORIZON_ANALYSIS.md）
# 更新日期：2026-04-20，56只港股验证
TRANSMISSION_ACCURACY = {
    'both_correct_rate': 62.24,      # 1天+5天都正确时，20天准确率
    'independent_20d_rate': 56.54,   # 独立20天准确率
    'improvement': 5.70               # 提升幅度
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
        return "-"

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

    # 添加预测日期
    date_prefix = f"[{prediction_date}] " if prediction_date else ""

    # 如果传导模式激活，添加标记
    if transmission_mode:
        display = f"✅传导({date_prefix}{display})"
    else:
        display = f"{date_prefix}{display}"

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


def get_pattern_action(pattern):
    """
    根据三周期模式获取交易建议

    参数:
    - pattern: 三周期模式字符串（如 '111', '110' 等）

    返回:
    - dict: 包含模式名称、操作建议、胜率等信息
    """
    if pattern in THREE_HORIZON_PATTERNS:
        return THREE_HORIZON_PATTERNS[pattern]
    return {'name': '未知', 'action': '观望', 'win_rate': '-', 'avg_return': '-', 'confidence': '低'}


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
                if probability > 0.60:
                    direction = "上涨"
                elif probability > 0.50:
                    direction = "观望"
                else:
                    direction = "下跌"

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

                llm_stock_list.append({
                    'code': stock_code,
                    'name': row['name'],
                    'sector': sector_name,
                    'sector_type': sector_type,
                    'prediction_20d': direction,
                    'probability_20d': round(probability, 4),
                    'current_price': float(row['current_price']) if pd.notna(row.get('current_price')) else None,
                    'chip_resistance': resistance_level
                })

            # 构建JSON格式文本
            catboost_text_llm = "【CatBoost模型预测结果（20天）- JSON格式】\n"
            catboost_text_llm += f"预测日期: {date_str}\n\n"
            catboost_text_llm += "```json\n"
            catboost_text_llm += json_module.dumps(llm_stock_list, ensure_ascii=False, indent=2)
            catboost_text_llm += "\n```\n\n"
            catboost_text_llm += "**字段说明**：\n"
            catboost_text_llm += "- `probability_20d`: 20天上涨概率（>0.60=高置信度，0.50-0.60=中等，≤0.50=低）\n"
            catboost_text_llm += "- `chip_resistance`: 筹码阻力（低=拉升容易，中=注意风险，高=拉升困难）\n"
            catboost_text_llm += "- **使用建议**：probability_20d高 + chip_resistance低 = 更可靠信号\n"

            # ========== 2. 构建邮件表格（保留三周期预测+筹码分布，用于用户查看）==========
            if three_horizon_results and len(three_horizon_results) > 0:
                # 三周期预测表格（含筹码分布）
                catboost_text_email = "【CatBoost模型三周期预测结果】\n"
                catboost_text_email += f"预测日期: {date_str}\n\n"
                catboost_text_email += "全部股票预测结果（按20天概率排序）:\n\n"
                catboost_text_email += "| 股票代码 | 股票名称 | 现价 | 板块名称 | 类型 | 1天预测 | 5天预测 | 20天预测 | 模式 | 交易建议 | 历史胜率 | 筹码阻力 | 传导模式 |\n"
                catboost_text_email += "|----------|----------|------|----------|------|--------|--------|---------|------|---------|------|----------|----------|\n"

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

                    if stock_code in three_horizon_results:
                        pred = three_horizon_results[stock_code]
                        preds = pred['predictions']

                        # 1天预测
                        pred_1d = preds.get(1, {'direction': '-', 'probability': 0.5})
                        p1d_str = f"{pred_1d['direction']} {pred_1d['probability']:.2f}"

                        # 5天预测
                        pred_5d = preds.get(5, {'direction': '-', 'probability': 0.5})
                        p5d_str = f"{pred_5d['direction']} {pred_5d['probability']:.2f}"

                        # 20天预测
                        pred_20d = preds.get(20, {'direction': '-', 'probability': 0.5})
                        p20d_str = f"{pred_20d['direction']} {pred_20d['probability']:.2f}"

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

                        price_str = f"{row['current_price']:.2f}" if pd.notna(row.get('current_price')) else '-'
                        catboost_text_email += f"| {stock_code} | {row['name']} | {price_str} | {sector_name} | {sector_type} | {p1d_str} | {p5d_str} | {p20d_str} | {pattern_display} | {action} | {win_rate} | {resistance_icon} | {transmission_display} |\n"

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

                # 添加交易规则说明
                catboost_text_email += f"\n**三周期交易规则说明**：\n"
                catboost_text_email += "- 模式标注 = 1天预测 + 5天预测 + 20天预测（1=涨，0=跌）\n"
                catboost_text_email += f"- 传导模式：当同一时间的1天+5天预测都正确时，20天准确率({TRANSMISSION_ACCURACY['both_correct_rate']}%) > 独立20天({TRANSMISSION_ACCURACY['independent_20d_rate']}%)，提升 +{TRANSMISSION_ACCURACY['improvement']}%\n"
                catboost_text_email += "- 策略含义：短期预测正确 → 中期预测更可靠，可增加仓位信心\n"

                # 添加筹码分布说明
                catboost_text_email += f"\n**筹码阻力说明**：\n"
                catboost_text_email += "- ✅低：上方筹码 < 30%，拉升容易\n"
                catboost_text_email += "- ⚠️中：上方筹码 30-60%，注意风险\n"
                catboost_text_email += "- 🔴高：上方筹码 > 60%，拉升困难\n"
            else:
                # 无三周期预测时，邮件表格也使用简化版本
                catboost_text_email = catboost_text_llm

            result['ensemble'] = catboost_text_llm
            result['ensemble_email'] = catboost_text_email
        else:
            print(f"⚠️ CatBoost 预测文件不存在: {catboost_csv}")

            result['ensemble'] = ''
            result['ensemble_email'] = ''

        return result

    except Exception as e:
        print(f"❌ 提取ML预测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'ensemble': '', 'ensemble_email': ''}


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
    dict: 当前市场状态信息
    """
    try:
        from data_services.tencent_finance import get_hsi_data_tencent as get_hsi_history

        # 使用腾讯财经API获取恒指历史数据（更稳定）
        hsi_df = get_hsi_history(period_days=60)

        if hsi_df is None or len(hsi_df) < 10:
            # 回退到 yfinance
            print("⚠️ 腾讯财经获取恒指数据失败，回退到 yfinance")
            hsi_ticker = yf.Ticker("^HSI")
            hsi_df = hsi_ticker.history(period="3mo")

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
        
        # 格式化时间（如果存在）
        date_str = current_time.strftime('%Y-%m-%d %H:%M:%S HKT') if current_time else 'N/A'
        
        return {
            'market_state': market_state,
            'market_state_cn': market_state_cn,
            'market_signal': market_signal,
            'recent_20d_return': recent_20d_return,
            'recent_5d_return': recent_5d_return,
            'current_hsi': current_hsi,
            'date': date_str
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
        
        content += "\n**📊 Z-Score 价格异常策略建议**：\n"
        content += "- **抄底信号**：价格异常 + 当日下跌 → 5日收益 +4.12%，胜率 71.7%\n"
        content += "- 价格大幅偏离均值，可能是短期超跌/超涨\n"
        content += "- **如果当日下跌**：考虑抄底（均值回归信号）\n"
        content += "- **如果当日上涨**：观望，等待确认\n\n"
    
    # 高异常（成交量）
    if high_volume_anomalies:
        content += "### 🔴 高异常（成交量）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
        content += "|---------|---------|---------|--------|---------|---------|\n"
        
        for anomaly in high_volume_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            volume_str = f"{anomaly.get('value', 0):,.0f}"
            content += f"| {anomaly['stock']} | {anomaly['name']} | **{anomaly['z_score']:.2f}** | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"
        
        content += "\n**风险提示**：\n"
        content += "⚠️ 成交量异常放大，可能有大额交易\n"
        content += "⚠️ 可能是机构建仓或出货\n"
        content += "⚠️ 建议关注价格方向和持续性\n\n"
    
    # 中异常（价格）
    if medium_price_anomalies:
        content += "### ⚠️ 中异常（价格）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 当前价格 | 当日变化 | RSI |\n"
        content += "|---------|---------|---------|---------|---------|-----|\n"
        
        for anomaly in medium_price_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% | {anomaly['rsi']:.1f} |\n"
        
        content += "\n**操作建议**：\n"
        content += "- 观察价格是否能突破阻力位\n"
        content += "- 注意止损位设置\n"
        content += "- 避免在异常期间加仓\n\n"
    
    # 中异常（成交量）
    if medium_volume_anomalies:
        content += "### ⚠️ 中异常（成交量）\n\n"
        content += "| 股票代码 | 股票名称 | Z-Score | 成交量 | 当前价格 | 当日变化 |\n"
        content += "|---------|---------|---------|--------|---------|---------|\n"
        
        for anomaly in medium_volume_anomalies:
            change_icon = "📈" if anomaly['change_1d'] > 0 else "📉"
            volume_str = f"{anomaly.get('value', 0):,.0f}"
            content += f"| {anomaly['stock']} | {anomaly['name']} | {anomaly['z_score']:.2f} | {volume_str} | {anomaly['current_price']:.2f} | {change_icon} {anomaly['change_1d']:+.2f}% |\n"
        
        content += "\n**操作建议**：\n"
        content += "- 关注成交量是否持续放大\n"
        content += "- 结合价格方向判断趋势\n"
        content += "- 谨慎操作，避免追涨杀跌\n\n"
    
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
        
        content += "\n**风险提示**：\n"
        content += "🔴 多维特征严重偏离正常模式\n"
        content += "🔴 可能存在价格操纵或重大消息\n"
        content += "🔴 建议立即检查持仓，考虑减仓\n\n"
    
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
        
        content += "\n**操作建议**：\n"
        content += "- 多维特征出现异常，值得关注\n"
        content += "- 观察后续价格走势确认\n"
        content += "- 避免在异常期间加仓\n\n"
    
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
        content += "- 可能是价格模式轻微变化\n"
        content += "- 建议结合基本面分析\n\n"
    
    content += "**重要提醒**：\n"
    content += "- 异常是警告信号，不一定是交易信号\n"
    content += "- 不要看到异常就立即卖出\n"
    content += "- 综合考虑基本面和市场情绪\n\n"
    
    # 添加基于两年数据验证的策略建议（精简版）
    content += "---\n\n"
    content += "### 📊 异常策略建议（2024-2026两年验证）\n\n"
    
    content += "| 异常类型 | 含义 | 5日收益 | 胜率 | 策略 |\n"
    content += "|---------|------|---------|------|------|\n"
    content += "| **IF high** | 多维异常（系统风险） | -3.04% | 43% | 🔴 减仓 |\n"
    content += "| **价格异常+当日下跌** | 超跌反弹 | +4.12% | 72% | 🟢 抄底 |\n"
    content += "| 价格异常+当日上涨 | 追涨风险 | +1.96% | 54% | ⚠️ 观望 |\n\n"
    
    content += "**核心逻辑**：IF high 预警风险，Z-Score 价格异常+当日下跌 = 抄底机会\n"
    content += "*基于86个IF样本、11个Z-Score样本回测，需结合其他指标*\n"
    
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
            'price_position': price_position
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
        
        table = "| 股票代码 | 股票名称 | 当前价格 | 涨跌幅 | RSI | MACD | MA20 | MA50 | MA200 | 均线排列 | 均线斜率 | 乖离率 | 布林带位置 | ATR | 成交量比率 | 趋势 | 支撑位 | 阻力位 |\n"
        table += "|---------|---------|---------|--------|-----|------|-----|-----|------|---------|---------|-------|-----------|-----|-----------|------|--------|--------|\n"
        
        success_count = 0
        for stock_code in stock_codes_sorted:
            indicators = get_stock_technical_indicators(stock_code)
            
            if indicators:
                # 获取股票名称
                stock_name = WATCHLIST.get(stock_code, stock_code)
                
                # 格式化数据
                price = safe_float_format(indicators['current_price'], '2f')
                change = safe_float_format(indicators['change_pct'], '+.2f') + "%"
                rsi = safe_float_format(indicators['rsi'], '2f')
                macd = safe_float_format(indicators['macd'], '2f')
                ma20 = safe_float_format(indicators['ma20'], '2f')
                ma50 = safe_float_format(indicators['ma50'], '2f')
                ma200 = safe_float_format(indicators['ma200'], '2f') if pd.notna(indicators['ma200']) else "N/A"
                ma_align = indicators['ma_alignment']
                ma_slope = safe_float_format(indicators['ma_slope_20'], '4f')
                ma_dev = safe_float_format(indicators['ma_deviation'], '2f') + "%"
                bb_pos = safe_float_format(indicators['bb_position'], '1f') + "%"
                atr = safe_float_format(indicators['atr'], '2f')
                vol_ratio = safe_float_format(indicators['volume_ratio'], '2f') + "x"
                trend = indicators['trend']
                support = f"{safe_float_format(indicators['support_level'], '2f')} ({safe_float_format(indicators['support_distance'], '2f')}%)"
                resistance = f"{safe_float_format(indicators['resistance_level'], '2f')} ({safe_float_format(indicators['resistance_distance'], '2f')}%)"
                
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
                
                table += f"| {stock_code} | {stock_name} | {price} | {change} | {rsi} | {macd} | {ma20} | {ma50} | {ma200} | {ma_align} | {ma_slope} | {ma_dev} | {bb_pos} | {atr} | {vol_ratio} | {trend} | {support} | {resistance} |\n"
                success_count += 1
        
        print(f"📊 技术指标表格: 成功获取 {success_count}/{len(stock_codes)} 只股票的数据")
        return table
        
    except Exception as e:
        print(f"⚠️ 生成技术指标表格失败: {e}")
        return ""


def send_email(subject, content, html_content=None):
    """
    发送邮件通知
    
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


def run_comprehensive_analysis(llm_filepath, ml_filepath, output_filepath=None,
                             send_email_flag=True, use_deep_analysis=True, use_cached_predictions=False):
    """
    运行综合分析

    参数:
    - llm_filepath: 大模型建议文件路径
    - ml_filepath: ML预测结果文件路径（已废弃，保留用于兼容性）
    - output_filepath: 输出文件路径（可选）
    - send_email_flag: 是否发送邮件（默认True）
    - use_deep_analysis: 是否使用深度分析模式进行异常检测（默认True）
    - use_cached_predictions: 是否使用已缓存的三周期预测CSV文件（默认False）
    """
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
        
        # 加载模型准确率
        print("📝 加载模型准确率...")
        model_accuracy = load_model_accuracy(horizon=20)
        print(f"✅ 准确率加载完成\n")
        
        # 生成日期
        date_str = datetime.now().strftime('%Y-%m-%d')
        
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
                                leader_items.append(f"{leader['name']}({leader['change_pct']:+.1f}%)")
                            leaders_text = " / ".join(leader_items)

                        sector_text += f"| {idx+1} | {trend_icon} {row['sector_name']} | {sector_type} | {change_color}{safe_float_format(row['avg_change_pct'], '2f')}% | {leaders_text} |\n"
                    
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
                    
                    # 添加市场状态说明表格
                    hsi_text += "### 市场状态说明\n\n"
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
| 1天 | {model_accuracy['1d']['accuracy']:.2%} | ±{model_accuracy['1d']['std']:.2%} |
| 5天 | {model_accuracy['5d']['accuracy']:.2%} | ±{model_accuracy['5d']['std']:.2%} |
| 20天 | {model_accuracy['20d']['accuracy']:.2%} | ±{model_accuracy['20d']['std']:.2%} |

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
                send_email(email_subject, full_content, html_content)
            
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
    parser = argparse.ArgumentParser(description='综合分析脚本 - 整合大模型建议和ML预测结果')
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
    
    args = parser.parse_args()
    
    # 生成日期
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # 默认文件路径
    if args.llm_file is None:
        args.llm_file = f'data/llm_recommendations_{date_str}.txt'
    
    if args.ml_file is None:
        args.ml_file = f'data/ml_predictions_20d_{date_str}.txt'
    
    # 运行综合分析
    result = run_comprehensive_analysis(args.llm_file, args.ml_file, args.output,
                                       send_email_flag=not args.no_email,
                                       use_deep_analysis=not args.no_deep_analysis,
                                       use_cached_predictions=args.use_cached_predictions)
    
    if result:
        print("\n✅ 综合分析完成！")
        sys.exit(0)
    else:
        print("\n❌ 综合分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
