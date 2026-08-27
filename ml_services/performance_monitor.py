# -*- coding: utf-8 -*-
"""
预测性能监控系统
追踪 ML 模型预测的实际表现，生成月度报告
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from config import STOCK_SECTOR_MAPPING

# 历史文件路径
HISTORY_FILE = 'data/prediction_history.json'
A_STOCK_HISTORY_FILE = 'data/a_stock_prediction_history.json'
REPORT_OUTPUT_DIR = 'output'


def load_prediction_history(path: str = None) -> Dict:
    """加载预测历史数据"""
    if path is None:
        path = HISTORY_FILE
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'predictions': [], 'metadata': {}}


def save_prediction_history(history: Dict, path: str = None):
    """保存预测历史数据"""
    if path is None:
        path = HISTORY_FILE
    history['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    history['metadata']['total_predictions'] = len(history['predictions'])
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def count_trading_days(start_date: str, end_date: str, stock_code: str) -> int:
    """
    计算两个日期之间的实际交易日数量
    
    参数:
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    - stock_code: 股票代码（用于获取交易日历）
    
    返回:
    - 交易日数量
    """
    try:
        # 使用 yfinance 获取股票数据
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date)
        return len(df)
    except Exception as e:
        print(f"⚠️ 获取交易日历失败: {e}")
        # 回退到估算（工作日）
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = 0
        current = start
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                days += 1
            current += timedelta(days=1)
        return days


def fetch_price(stock_code: str, date: str) -> Optional[float]:
    """
    获取指定日期的股票收盘价
    
    参数:
    - stock_code: 股票代码
    - date: 日期 (YYYY-MM-DD)
    
    返回:
    - 收盘价，如果获取失败返回 None
    """
    try:
        # 转换股票代码格式: 0700.HK -> 0700.HK (yfinance 格式)
        ticker = yf.Ticker(stock_code)
        
        # 获取日期前后的数据（处理非交易日）
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        start = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')
        
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            return None
        
        # 找到最接近目标日期的交易日
        df['date'] = df.index.strftime('%Y-%m-%d')
        target_date = date
        
        if target_date in df['date'].values:
            return float(df[df['date'] == target_date]['Close'].iloc[0])
        else:
            # 返回最接近的交易日收盘价
            closest_dates = df[df.index <= date_obj]
            if not closest_dates.empty:
                return float(closest_dates['Close'].iloc[-1])
            return None
            
    except Exception as e:
        print(f"⚠️ 获取 {stock_code} 价格失败: {e}")
        return None


def fetch_a_stock_price(stock_code: str, date: str) -> Optional[float]:
    """
    获取A股指定日期的收盘价（与 fetch_price 接口一致，仅数据源不同）

    参数:
    - stock_code: A股代码（6位，如 600000）
    - date: 日期 (YYYY-MM-DD)

    返回:
    - 收盘价，如果获取失败返回 None
    """
    try:
        from data_services.a_stock_data import get_a_stock_data

        date_obj = datetime.strptime(date, '%Y-%m-%d')
        start = (date_obj - timedelta(days=10)).strftime('%Y-%m-%d')
        end = (date_obj + timedelta(days=10)).strftime('%Y-%m-%d')

        df = get_a_stock_data(stock_code, period_days=30, use_cache=False)
        if df is None or df.empty:
            return None

        # 统一列名
        if 'Date' not in df.columns and df.index.name != 'Date':
            df = df.reset_index()
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        else:
            return None

        # 优先匹配目标日期，否则取目标日期之前最近的交易日
        exact = df[df['date'] == date]
        if not exact.empty:
            return float(exact['Close'].iloc[-1])

        before = df[df['date'] <= date]
        if not before.empty:
            return float(before['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"⚠️ 获取A股 {stock_code} 价格失败: {e}")
        return None


def evaluate_predictions(history: Dict, horizon: int = 20, force: bool = False,
                         save_path: str = None) -> Tuple[Dict, Dict]:
    """
    评估已到期的预测
    
    参数:
    - history: 预测历史数据
    - horizon: 预测周期
    - force: 是否强制重新评估
    
    返回:
    - 更新后的历史数据和统计信息
    """
    now = datetime.now()
    evaluated_count = 0
    stats = {
        'total': 0,
        'evaluated': 0,
        'pending': 0,
        'correct': 0,
        'wrong': 0
    }
    
    for pred in history['predictions']:
        # 只处理指定周期的预测
        if pred.get('horizon') != horizon:
            continue
            
        stats['total'] += 1
        
        # 跳过已评估的预测（除非强制重新评估）
        if pred.get('outcome') is not None and not force:
            stats['evaluated'] += 1
            if pred.get('outcome') == 'correct':
                stats['correct'] += 1
            else:
                stats['wrong'] += 1
            continue
        
        # 检查是否已到期
        # 步骤1：使用已存储的 target_date
        target_date = pred.get('target_date')
        
        # 步骤2：如果没有存储，则根据 data_date + horizon 计算
        if not target_date:
            data_date = pred.get('data_date', pred.get('timestamp', '').split('T')[0])
            if data_date:
                try:
                    data_date_obj = datetime.strptime(data_date, '%Y-%m-%d')
                    # 简单计算：data_date + horizon 天（近似交易日）
                    target_date_obj = data_date_obj + timedelta(days=horizon)
                    target_date = target_date_obj.strftime('%Y-%m-%d')
                    # 更新预测记录
                    pred['target_date'] = target_date
                except ValueError:
                    stats['pending'] += 1
                    continue
            else:
                stats['pending'] += 1
                continue

        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')

        # 检查目标日期是否已到达（不允许评估未来的预测）
        # 使用交易日判断：如果目标日期在今天之后，跳过评估
        if target_date_obj.date() > now.date():
            stats['pending'] += 1
            continue

        # 获取目标日期的收盘价（按市场选择数据源）
        market = pred.get('market', 'HK')
        if market == 'A':
            exit_price = fetch_a_stock_price(pred['stock_code'], target_date)
        else:
            exit_price = fetch_price(pred['stock_code'], target_date)

        # NaN 价格视同缺失：否则 NaN 收益会被误判为 'down' 方向，污染准确率统计
        if exit_price is None or np.isnan(exit_price):
            print(f"⚠️ 无法获取 {pred['stock_code']} 在 {target_date} 的有效价格")
            stats['pending'] += 1
            continue
        
        # 计算实际收益
        entry_price = pred.get('entry_price', 0)
        if entry_price <= 0:
            stats['pending'] += 1
            continue
            
        actual_return = (exit_price - entry_price) / entry_price
        actual_direction = 'up' if actual_return > 0 else 'down'
        
        # 判断预测是否正确
        predicted_direction = pred.get('predicted_direction')
        outcome = 'correct' if predicted_direction == actual_direction else 'wrong'
        
        # 更新预测记录
        pred['outcome'] = outcome
        pred['actual_return'] = round(actual_return, 4)
        pred['actual_direction'] = actual_direction
        pred['evaluated_at'] = now.strftime('%Y-%m-%dT%H:%M:%S')
        
        evaluated_count += 1
        stats['evaluated'] += 1
        if outcome == 'correct':
            stats['correct'] += 1
        else:
            stats['wrong'] += 1
    
    # 保存更新后的历史
    if evaluated_count > 0:
        save_prediction_history(history, save_path)
    
    return history, stats


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    计算性能指标
    
    参数:
    - predictions: 已评估的预测列表
    
    返回:
    - 性能指标字典
    """
    evaluated = [p for p in predictions if p.get('outcome') is not None]
    
    if not evaluated:
        return {}
    
    df = pd.DataFrame(evaluated)
    
    # 基础指标
    total = len(df)
    correct = len(df[df['outcome'] == 'correct'])
    accuracy = correct / total if total > 0 else 0

    # 收益指标（剔除 NaN，避免无效退出价格污染统计 —— 历史数据中存在 NaN actual_return）
    if 'actual_return' in df.columns:
        returns_series = pd.to_numeric(df['actual_return'], errors='coerce').dropna()
    else:
        returns_series = pd.Series(dtype=float)
    returns = returns_series.values
    avg_return = float(np.mean(returns)) if len(returns) > 0 else 0
    median_return = float(np.median(returns)) if len(returns) > 0 else 0

    # 风险指标
    std_return = float(np.std(returns)) if len(returns) > 1 else 0
    sharpe = avg_return / std_return if std_return > 0 else 0

    # 买入信号分析（只看预测上涨的）
    buy_signals = df[df['predicted_direction'] == 'up']
    if len(buy_signals) > 0:
        buy_wins = len(buy_signals[buy_signals['outcome'] == 'correct'])
        buy_win_rate = buy_wins / len(buy_signals)
        buy_returns = pd.to_numeric(buy_signals['actual_return'], errors='coerce').dropna()
        buy_avg_return = float(buy_returns.mean()) if len(buy_returns) > 0 else 0
    else:
        buy_win_rate = 0
        buy_avg_return = 0
    
    return {
        'total_predictions': total,
        'correct_predictions': correct,
        'accuracy': round(accuracy, 4),
        'avg_return': round(avg_return, 4),
        'median_return': round(median_return, 4),
        'std_return': round(std_return, 4),
        'sharpe_ratio': round(sharpe, 4),
        'buy_signal_count': len(buy_signals),
        'buy_win_rate': round(buy_win_rate, 4),
        'buy_avg_return': round(buy_avg_return, 4)
    }


# ── 共享指标计算（Markdown 报告与可视化报告复用，单一真相源） ──

# 时间窗口定义：(天数, 名称)
TIME_WINDOWS = [
    (30, '1个月'),
    (90, '3个月'),
    (180, '6个月'),
]

# 时间窗口排序顺序
WINDOW_ORDER = {'1个月': 1, '3个月': 2, '6个月': 3}


def _filter_evaluated(history: Dict, start_date_str: str, end_date_str: str,
                      horizon: Optional[int] = None) -> List[Dict]:
    """筛选日期范围 [start, end] 内已评估的预测（可按周期过滤）"""
    result = []
    for p in history['predictions']:
        if p.get('outcome') is None:
            continue
        d = p.get('target_date', p.get('timestamp', '').split('T')[0])
        if not (start_date_str <= d <= end_date_str):
            continue
        if horizon is not None and p.get('horizon') != horizon:
            continue
        result.append(p)
    return result


def compute_window_horizon_metrics(history: Dict, time_windows: Optional[List] = None,
                                   now: Optional[datetime] = None) -> Dict:
    """
    计算各时间窗口 × 各周期的指标

    返回: {窗口天数: {周期: calculate_metrics() 结果}}
    """
    if time_windows is None:
        time_windows = TIME_WINDOWS
    if now is None:
        now = datetime.now()
    end_str = now.strftime('%Y-%m-%d')

    result = {}
    for days, _ in time_windows:
        start_str = (now - timedelta(days=days)).strftime('%Y-%m-%d')
        preds = _filter_evaluated(history, start_str, end_str)

        by_horizon = {1: [], 5: [], 20: []}
        for p in preds:
            h = p.get('horizon', 20)
            if h in by_horizon:
                by_horizon[h].append(p)

        result[days] = {h: calculate_metrics(ps) for h, ps in by_horizon.items()}
    return result


def compute_grouped_metrics(history: Dict, days: int, horizon: int,
                            group_key: str, now: Optional[datetime] = None) -> Dict:
    """
    在指定窗口与周期内，按 group_key（'sector' / 'stock_code'）分组计算指标

    返回: {分组值: calculate_metrics() 结果}
    """
    if now is None:
        now = datetime.now()
    end_str = now.strftime('%Y-%m-%d')
    start_str = (now - timedelta(days=days)).strftime('%Y-%m-%d')

    preds = _filter_evaluated(history, start_str, end_str, horizon=horizon)
    groups = {}
    for p in preds:
        g = p.get(group_key, 'unknown')
        groups.setdefault(g, []).append(p)

    return {g: calculate_metrics(ps) for g, ps in groups.items() if ps}


def collect_group_detail(history: Dict, group_key: str,
                         time_windows: Optional[List] = None,
                         now: Optional[datetime] = None) -> List[Dict]:
    """
    收集 分组 × 周期 × 时间窗口 的明细行（用于表格）

    返回行列表: {'group', 'horizon', 'window', 'metrics', 'sample_pred'}，排序由调用方负责
    """
    if time_windows is None:
        time_windows = TIME_WINDOWS
    if now is None:
        now = datetime.now()
    end_str = now.strftime('%Y-%m-%d')

    rows = []
    for h in [1, 5, 20]:
        for days, window_name in time_windows:
            start_str = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            preds = _filter_evaluated(history, start_str, end_str, horizon=h)

            groups = {}
            for p in preds:
                g = p.get(group_key, 'unknown')
                groups.setdefault(g, []).append(p)

            for g, gps in groups.items():
                metrics = calculate_metrics(gps)
                if metrics.get('total_predictions', 0) > 0:
                    rows.append({
                        'group': g,
                        'horizon': h,
                        'window': window_name,
                        'metrics': metrics,
                        'sample_pred': gps[0],
                    })
    return rows


def get_sector_name(sector_code: str) -> str:
    """获取板块中文名称"""
    sector_names = {
        'bank': '银行股',
        'tech': '科技股',
        'semiconductor': '半导体股',
        'ai': '人工智能股',
        'new_energy': '新能源股',
        'environmental': '环保股',
        'energy': '能源股',
        'shipping': '航运股',
        'exchange': '交易所',
        'utility': '公用事业股',
        'insurance': '保险股',
        'biotech': '生物医药股',
        'index': '指数基金',
        'real_estate': '房地产股',
        'consumer': '消费股',
        'auto': '汽车股',
        'unknown': '未知'
    }
    return sector_names.get(sector_code, sector_code)


def get_sector_type(sector_code: str) -> str:
    """获取板块类型（周期/防御）"""
    sector_types = {
        # 周期性板块
        'semiconductor': '周期',
        'biotech': '周期',
        'tech': '周期',
        'consumer': '周期',
        'real_estate': '周期',
        'energy': '周期',
        'shipping': '周期',
        'auto': '周期',
        'new_energy': '周期',
        'ai': '周期',
        # 防御性板块
        'bank': '防御',
        'insurance': '防御',
        'utility': '防御',
        'environmental': '防御',
        'exchange': '防御',
        'index': '防御',
    }
    return sector_types.get(sector_code, '-')


def calculate_three_horizon_pattern_stats(history: Dict, start_date: Optional[str] = None) -> Dict:
    """
    从 prediction_history.json 实时计算三周期模式统计

    参数:
    - history: 预测历史数据
    - start_date: 起始日期 (YYYY-MM-DD)，None 表示不限制

    返回:
    - dict: {模式: {total, correct, avg_return, win_rate}}
    """
    from collections import defaultdict

    predictions = history.get('predictions', [])
    if not predictions:
        return {}

    # 按 data_date + stock_code 分组
    grouped = defaultdict(dict)
    for p in predictions:
        data_date = p.get('data_date', '')
        stock_code = p.get('stock_code', '')
        horizon = p.get('horizon')

        # 时间范围过滤
        if start_date and data_date < start_date:
            continue

        if data_date and stock_code and horizon:
            key = f"{data_date}_{stock_code}"
            grouped[key][horizon] = p

    # 找出有三周期预测的记录，计算各模式统计
    pattern_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'returns': []})

    for key, horizons in grouped.items():
        if 1 in horizons and 5 in horizons and 20 in horizons:
            p1 = horizons[1]
            p5 = horizons[5]
            p20 = horizons[20]

            # 只统计 20天已评估的
            if p20.get('outcome') is None:
                continue

            # 编码模式：up=1, down=0
            pattern = f"{'1' if p1.get('predicted_direction') == 'up' else '0'}{'1' if p5.get('predicted_direction') == 'up' else '0'}{'1' if p20.get('predicted_direction') == 'up' else '0'}"

            pattern_stats[pattern]['total'] += 1
            if p20.get('outcome') == 'correct':
                pattern_stats[pattern]['correct'] += 1

            ret = p20.get('actual_return')
            if ret is not None:
                pattern_stats[pattern]['returns'].append(ret)

    # 计算准确率和平均收益
    result = {}
    for pattern, stats in pattern_stats.items():
        total = stats['total']
        correct = stats['correct']
        returns = stats['returns']

        result[pattern] = {
            'total': total,
            'correct': correct,
            'win_rate': correct / total if total > 0 else 0,
            'avg_return': sum(returns) / len(returns) if returns else 0
        }

    return result


def generate_monthly_report(history: Dict, month: Optional[str] = None) -> str:
    """
    生成性能报告

    参数:
    - history: 预测历史数据
    - month: 保留参数（兼容性）

    返回:
    - Markdown 格式的报告
    """
    now = datetime.now()
    horizon_names = {1: '1天', 5: '5天', 20: '20天'}

    # 各时间窗口 × 各周期指标（与可视化报告共享 helper，单一真相源）
    window_horizon_metrics = compute_window_horizon_metrics(history, now=now)

    # 3个月窗口（用于模式验证与风险提示）
    detail_days = 90
    start_date_detail_str = (now - timedelta(days=detail_days)).strftime('%Y-%m-%d')
    detail_horizon_metrics = window_horizon_metrics.get(detail_days, {})

    # 生成报告
    report = f"""# 预测性能报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、各周期不同时间窗口表现

| 周期 | 时间窗口 | 预测数 | 准确率 | 平均收益 | 夏普比率 |
|------|----------|--------|--------|----------|----------|
"""

    # 各周期各时间窗口性能表格
    for h in [1, 5, 20]:
        for days, window_name in TIME_WINDOWS:
            m = window_horizon_metrics.get(days, {}).get(h, {})
            if m and m.get('total_predictions', 0) > 0:
                report += f"| {horizon_names[h]} | {window_name} | {m.get('total_predictions', 0)} | **{m.get('accuracy', 0):.2%}** | {m.get('avg_return', 0):.2%} | {m.get('sharpe_ratio', 0):.4f} |\n"
            else:
                report += f"| {horizon_names[h]} | {window_name} | 0 | - | - | - |\n"

    report += f"""
---

## 二、市场分布（港股 / A股）

| 市场 | 预测数 | 准确率 | 平均收益 | 夏普比率 |
|------|--------|--------|----------|----------|
"""

    _market_predictions = {}
    for _p in history.get('predictions', []):
        if _p.get('outcome') is not None:
            _market_predictions.setdefault(_p.get('market', 'HK'), []).append(_p)
    _market_label = {'HK': '港股', 'A': 'A股'}
    for _mk in ['HK', 'A']:
        _ps = _market_predictions.get(_mk, [])
        if _ps:
            _m = calculate_metrics(_ps)
            report += f"| {_market_label.get(_mk, _mk)} | {_m.get('total_predictions', 0)} | **{_m.get('accuracy', 0):.2%}** | {_m.get('avg_return', 0):.2%} | {_m.get('sharpe_ratio', 0):.4f} |\n"
    report += "\n"

    report += f"""
---

## 三、板块表现

| 板块 | 类型 | 周期 | 时间窗口 | 预测数 | 准确率 | 平均收益 | 夏普比率 |
|------|------|------|----------|--------|--------|----------|----------|
"""

    # 板块表现 - 收集所有数据后统一输出（复用共享 helper）
    all_sector_data = []
    for h in [1, 5, 20]:
        for days, window_name in TIME_WINDOWS:
            sector_metrics = compute_grouped_metrics(history, days, h, 'sector', now=now)
            for sector, metrics in sector_metrics.items():
                all_sector_data.append({
                    'sector': sector,
                    'horizon': h,
                    'window': window_name,
                    'metrics': metrics
                })

    # 按板块、周期、时间窗口排序
    all_sector_data.sort(key=lambda x: (
        get_sector_name(x['sector']),  # 板块名称
        x['horizon'],                   # 周期 (1, 5, 20)
        WINDOW_ORDER.get(x['window'], 99)  # 时间窗口
    ))

    for item in all_sector_data:
        sector_name = get_sector_name(item['sector'])
        sector_type = get_sector_type(item['sector'])
        m = item['metrics']
        # 准确率加粗，便于后续颜色标记
        accuracy_str = f"**{m.get('accuracy', 0):.2%}**"
        report += f"| {sector_name} | {sector_type} | {horizon_names[item['horizon']]} | {item['window']} | {m.get('total_predictions', 0)} | {accuracy_str} | {m.get('avg_return', 0):.2%} | {m.get('sharpe_ratio', 0):.4f} |\n"

    report += """
---

## 四、个股表现

| 股票代码 | 股票名称 | 板块 | 周期 | 时间窗口 | 预测数 | 准确率 | 平均收益 |
|----------|----------|------|------|----------|--------|--------|----------|
"""

    # 个股表现 - 收集所有数据后统一输出（复用共享 helper）
    all_stock_data = []
    for row in collect_group_detail(history, 'stock_code', now=now):
        all_stock_data.append({
            'stock': row['group'],
            'stock_name': row['sample_pred'].get('stock_name', row['group']),
            'sector': row['sample_pred'].get('sector', 'unknown'),
            'horizon': row['horizon'],
            'window': row['window'],
            'metrics': row['metrics']
        })

    # 按股票代码、周期、时间窗口排序
    all_stock_data.sort(key=lambda x: (
        x['stock'],                     # 股票代码
        x['horizon'],                   # 周期 (1, 5, 20)
        WINDOW_ORDER.get(x['window'], 99)  # 时间窗口
    ))

    # 显示所有股票
    for item in all_stock_data:
        sector_name = get_sector_name(item['sector'])
        m = item['metrics']
        # 准确率加粗，便于后续颜色标记
        accuracy_str = f"**{m.get('accuracy', 0):.2%}**"
        report += f"| {item['stock']} | {item['stock_name']} | {sector_name} | {horizon_names[item['horizon']]} | {item['window']} | {m.get('total_predictions', 0)} | {accuracy_str} | {m.get('avg_return', 0):.2%} |\n"

    # 三周期模式统计（3个月窗口）
    pattern_stats = calculate_three_horizon_pattern_stats(history, start_date_detail_str)

    # 模式名称映射（个股版本，参考 docs/THREE_HORIZON_ANALYSIS.md）
    pattern_names = {
        '010': '反弹失败⭐',
        '000': '一致看跌',
        '100': '冲高回落',
        '001': '下跌中继',
        '011': '探底回升',
        '101': '假突破',
        '110': '震荡回调',
        '111': '一致看涨',
    }

    # 模式建议映射（个股版本）
    pattern_actions = {
        '010': '谨慎减仓',
        '000': '止损/减仓',
        '100': '获利了结',
        '001': '谨慎观望',
        '011': '分批建仓',
        '101': '持有观望',
        '110': '观望',
        '111': '谨慎持有',
    }

    report += "\n---\n\n## 五、三周期模式验证（3个月窗口）\n\n"

    if pattern_stats:
        # 按准确率排序
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)

        report += "| 排名 | 模式 | 名称 | 样本数 | 准确率 | 平均收益 | 建议 |\n"
        report += "|------|------|------|--------|--------|----------|------|\n"

        for i, (pattern, stats) in enumerate(sorted_patterns, 1):
            name = pattern_names.get(pattern, '未知')
            action = pattern_actions.get(pattern, '观望')
            win_rate = stats['win_rate']
            avg_return = stats['avg_return']
            total = stats['total']

            ret_str = f"+{avg_return:.2%}" if avg_return >= 0 else f"{avg_return:.2%}"
            report += f"| {i} | {pattern} | {name} | {total} | {win_rate:.1%} | {ret_str} | {action} |\n"

        report += "\n**模式编码**：110 = 1天涨、5天涨、20天跌 | 统计时间：3个月窗口\n"
    else:
        report += "*样本量不足，暂无统计数据（需要同时有1天、5天、20天预测）*\n"

    # 计算3个月窗口的总预测数（用于风险提示）
    total_3m_predictions = sum(
        detail_horizon_metrics.get(h, {}).get('total_predictions', 0)
        for h in [1, 5, 20]
    )

    report += f"""
---

## 六、风险提示

1. **历史表现不代表未来收益**
2. 模型准确率统计基于 {total_3m_predictions} 个样本（3个月窗口），仅供参考
3. 投资有风险，请谨慎决策

---

**报告生成**: 金融资产智能分析系统 - 预测性能监控模块
"""

    return report


def generate_visual_html_report(history: Dict, plain_text: Optional[str] = None):
    """
    生成可视化性能报告（HTML + 内嵌图表附件）

    用图表替换主要汇总表格（整体雷达 / 窗口柱状图 / 板块雷达网格 / 模式柱状图），
    保留个股明细表与风险提示。

    参数:
    - history: 预测历史数据
    - plain_text: 纯文本正文（默认复用 Markdown 报告）

    返回: (html_content, plain_text, attachments)
    """
    from scripts.performance_charts import (
        generate_overall_radar_section,
        generate_window_bar_section,
        generate_sector_radar_section,
        generate_pattern_bar_section,
        generate_stock_section,
    )

    now = datetime.now()
    report_time = now.strftime('%Y-%m-%d %H:%M:%S')
    detail_days = 90
    start_date_detail_str = (now - timedelta(days=detail_days)).strftime('%Y-%m-%d')

    # 指标计算（与 Markdown 报告共享 helper）
    window_metrics = compute_window_horizon_metrics(history, now=now)
    horizon_metrics_3m = window_metrics.get(detail_days, {})
    sector_metrics_3m = compute_grouped_metrics(history, detail_days, 20, 'sector', now=now)
    pattern_stats = calculate_three_horizon_pattern_stats(history, start_date_detail_str)

    parts = []
    attachments = {}

    # 标题
    parts.append(f"""
    <h1 style="color:#333; margin-bottom:5px;">预测性能报告（港股 + A股）</h1>
    <p style="color:#666; font-size:13px; margin-top:0;">
        生成时间: {report_time} | 统计口径: 已到期且已评估的方向预测
    </p>
    <hr style="border:none; border-top:1px solid #ddd;">
""")

    # 一、整体性能雷达
    html, atts = generate_overall_radar_section(horizon_metrics_3m, window_name='3个月')
    parts.append(html)
    attachments.update(atts)

    # 二、时间窗口柱状图
    html, atts = generate_window_bar_section(window_metrics)
    parts.append(html)
    attachments.update(atts)

    # 三、板块雷达网格
    sector_bundle = {
        s: {'name': get_sector_name(s), 'metrics': m}
        for s, m in sector_metrics_3m.items()
    }
    html, atts = generate_sector_radar_section(sector_bundle)
    parts.append(html)
    attachments.update(atts)

    # 四、三周期模式柱状图（图表标签使用无 emoji 的名称）
    chart_pattern_names = {
        '010': '反弹失败', '000': '一致看跌', '100': '冲高回落', '001': '下跌中继',
        '011': '探底回升', '101': '假突破', '110': '震荡回调', '111': '一致看涨',
    }
    html, atts = generate_pattern_bar_section(pattern_stats, chart_pattern_names)
    parts.append(html)
    attachments.update(atts)

    # 五、个股表现（全部排名条形图 + Top 10 雷达网格，替换原明细表）
    # 复用 collect_group_detail 取「20天 / 3个月」口径，组装个股 bundle（含名称/板块）
    stock_bundle = {}
    for row in collect_group_detail(history, 'stock_code', now=now):
        if row['horizon'] != 20 or row['window'] != '3个月':
            continue
        stock_bundle[row['group']] = {
            'code': row['group'],
            'name': row['sample_pred'].get('stock_name', row['group']),
            'sector': row['sample_pred'].get('sector', 'unknown'),
            'metrics': row['metrics'],
        }
    html, atts = generate_stock_section(stock_bundle)
    parts.append(html)
    attachments.update(atts)

    # 六、风险提示
    total_3m = sum(
        horizon_metrics_3m.get(h, {}).get('total_predictions', 0) for h in [1, 5, 20]
    )
    parts.append(f"""
    <h2 style="color:#007bff; margin-top:30px; border-bottom:1px solid #ddd; padding-bottom:5px;">六、风险提示</h2>
    <ol style="color:#333; font-size:13px; line-height:1.8;">
        <li><b>历史表现不代表未来收益</b></li>
        <li>模型准确率统计基于 {total_3m} 个样本（3个月窗口），仅供参考</li>
        <li>投资有风险，请谨慎决策</li>
    </ol>
    <p style="color:#999; font-size:12px; margin-top:25px; border-top:1px solid #eee; padding-top:10px;">
        报告生成: 金融资产智能分析系统 - 预测性能监控模块
     </p>
""")

    body = '\n'.join(p for p in parts if p)

    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, "Microsoft YaHei", sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 12px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f7f7f7; }}
        h1, h2 {{ color: #333; }}
    </style>
    </head>
    <body>
    {body}
    </body>
    </html>
    """

    if plain_text is None:
        plain_text = generate_monthly_report(history)

    return html_content, plain_text, attachments


def send_email_report(report: str, subject: str) -> bool:
    """
    发送报告邮件（使用统一消息服务模块）

    参数:
    - report: Markdown 格式的报告内容
    - subject: 邮件主题

    返回:
    - 是否发送成功
    """
    try:
        from message_services import EmailSender
        import markdown
        import re

        # 转换 Markdown 为 HTML
        html_content = markdown.markdown(report, extensions=['tables'])

        # 为准确率添加颜色样式：三色系统
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

        # 只匹配加粗的准确率百分比
        html_content = re.sub(r'<(strong|b)>(\d+\.\d{2})%</(strong|b)>', lambda m: colorize_accuracy(m.group(2)), html_content)

        # 添加样式
        html_content = f"""
        <html>
        <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            h1, h2 {{ color: #333; }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; font-weight: bold; }}
        </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """

        # 使用统一消息服务模块
        sender = EmailSender()
        return sender.send_with_retry(subject, report, html_content)

    except ImportError:
        print("⚠️ 消息服务模块未安装，使用内置邮件发送")
        return _send_email_report_legacy(report, subject)


def _send_email_report_legacy(report: str, subject: str) -> bool:
    """
    发送报告邮件（备用实现）

    参数:
    - report: Markdown 格式的报告内容
    - subject: 邮件主题

    返回:
    - 是否发送成功
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import markdown
    import re

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
    smtp_user = os.environ.get("EMAIL_SENDER")
    smtp_pass = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL", "")

    if not smtp_user or not smtp_pass:
        print("❌ 缺少邮件配置环境变量")
        return False

    # 转换 Markdown 为 HTML
    html_content = markdown.markdown(report, extensions=['tables'])

    # 为准确率添加颜色样式：三色系统
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

    # 添加样式
    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1, h2 {{ color: #333; }}
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # 创建邮件
    msg = MIMEMultipart("alternative")
    msg['From'] = smtp_user
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(report, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    # 发送邮件
    try:
        if "163.com" in smtp_server:
            server = smtplib.SMTP_SSL(smtp_server, 465, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, 587, timeout=30)
            server.starttls()

        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient.split(","), msg.as_string())
        server.quit()

        print(f"✅ 报告邮件发送成功: {subject}")
        return True
    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预测性能监控')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'report', 'all'],
                       help='运行模式: evaluate=评估预测, report=生成报告, all=全部')
    parser.add_argument('--horizon', type=str, default='all',
                       help='预测周期: 1=1天, 5=5天, 20=20天, all=全部（默认）')
    parser.add_argument('--month', type=str, default=None,
                       help='报告月份 (YYYY-MM)，默认上个月')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件，仅生成报告')
    parser.add_argument('--force', action='store_true',
                       help='强制重新评估已评估的预测')

    args = parser.parse_args()

    # 解析 horizon 参数
    if args.horizon == 'all':
        horizons = [1, 5, 20]
    else:
        horizons = [int(args.horizon)]

    print("=" * 60)
    print("📊 预测性能监控系统")
    print("=" * 60)

    # 加载历史数据（港股 + A股，分别标注市场，互不干扰）
    print("\n📂 加载预测历史数据...")
    hk_history = load_prediction_history(HISTORY_FILE)
    for _p in hk_history.get('predictions', []):
        _p.setdefault('market', 'HK')
    a_history = load_prediction_history(A_STOCK_HISTORY_FILE)
    for _p in a_history.get('predictions', []):
        _p.setdefault('market', 'A')
    print(f"   港股: {len(hk_history.get('predictions', []))} 条 | A股: {len(a_history.get('predictions', []))} 条")

    if args.mode in ['evaluate', 'all']:
        # 评估各周期、各市场的预测（分别写回各自历史文件）
        total_stats = {'total': 0, 'evaluated': 0, 'correct': 0, 'wrong': 0}

        for label, hist, path in [('港股', hk_history, HISTORY_FILE), ('A股', a_history, A_STOCK_HISTORY_FILE)]:
            for h in horizons:
                print(f"\n📈 评估 {label} {h} 天周期的预测...")
                hist, stats = evaluate_predictions(hist, h, args.force, save_path=path)

                print(f"   总预测: {stats['total']}")
                print(f"   已评估: {stats['evaluated']}")
                print(f"   正确: {stats['correct']}")
                print(f"   错误: {stats['wrong']}")
                if stats['evaluated'] > 0:
                    print(f"   准确率: {stats['correct']/stats['evaluated']:.2%}")

                total_stats['total'] += stats['total']
                total_stats['evaluated'] += stats['evaluated']
                total_stats['correct'] += stats['correct']
                total_stats['wrong'] += stats['wrong']

        if len(horizons) > 1:
            print(f"\n📊 所有周期合计:")
            print(f"   总预测: {total_stats['total']}")
            print(f"   已评估: {total_stats['evaluated']}")
            print(f"   正确: {total_stats['correct']}")
            print(f"   错误: {total_stats['wrong']}")
            if total_stats['evaluated'] > 0:
                print(f"   准确率: {total_stats['correct']/total_stats['evaluated']:.2%}")

    # 合并两市场历史用于报告生成（只读）
    history = {
        'predictions': hk_history.get('predictions', []) + a_history.get('predictions', []),
        'metadata': {**hk_history.get('metadata', {}), **a_history.get('metadata', {})}
    }

    if args.mode in ['report', 'all']:
        # 生成 Markdown 报告（仍保存，供 CI 提交与纯文本正文使用）
        print(f"\n📝 生成性能报告...")
        report = generate_monthly_report(history, args.month)

        # 保存报告（使用当前日期命名）
        report_date = datetime.now().strftime('%Y-%m-%d')
        report_path = os.path.join(REPORT_OUTPUT_DIR, f'performance_report_{report_date}.md')

        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   报告已保存到: {report_path}")

        # 生成可视化报告（图表替换主要汇总表格）
        html_content, attachments = None, {}
        try:
            html_content, _, attachments = generate_visual_html_report(
                history, plain_text=report)
            print(f"   可视化报告已生成: {len(attachments)} 张图表")
        except Exception as e:
            print(f"⚠️ 可视化报告生成失败，将回退纯表格邮件: {e}")

        # 保存图表 PNG（便于本地预览）
        if attachments:
            charts_dir = os.path.join(REPORT_OUTPUT_DIR, 'performance_charts')
            os.makedirs(charts_dir, exist_ok=True)
            for cid, png in attachments.items():
                with open(os.path.join(charts_dir, f'{cid}.png'), 'wb') as f:
                    f.write(png)
            print(f"   图表已保存到: {charts_dir}")

        # 发送邮件（优先带内嵌图表的可视化版本，失败回退纯表格）
        if not args.no_email:
            subject = f"[金融资产智能分析] 预测性能报告（港股+A股） - {report_date}"
            if html_content and attachments:
                try:
                    from message_services.email_sender import send_email_with_images
                    send_email_with_images(
                        subject, report, html_content, inline_images=attachments)
                except Exception as e:
                    print(f"⚠️ 可视化邮件发送失败（{e}），回退纯表格邮件")
                    send_email_report(report, subject)
            else:
                send_email_report(report, subject)
        else:
            print("   (--no-email) 跳过邮件发送")

    print("\n" + "=" * 60)
    print("✅ 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()