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
REPORT_OUTPUT_DIR = 'output'


def load_prediction_history() -> Dict:
    """加载预测历史数据"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'predictions': [], 'metadata': {}}


def save_prediction_history(history: Dict):
    """保存预测历史数据"""
    history['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    history['metadata']['total_predictions'] = len(history['predictions'])
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
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


def evaluate_predictions(history: Dict, horizon: int = 20, force: bool = False) -> Tuple[Dict, Dict]:
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

        # 获取目标日期的收盘价（直接检查是否有价格数据）
        exit_price = fetch_price(pred['stock_code'], target_date)
        
        if exit_price is None:
            print(f"⚠️ 无法获取 {pred['stock_code']} 在 {target_date} 的价格")
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
        save_prediction_history(history)
    
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
    
    # 收益指标
    returns = df['actual_return'].values
    avg_return = np.mean(returns) if len(returns) > 0 else 0
    median_return = np.median(returns) if len(returns) > 0 else 0
    
    # 风险指标
    std_return = np.std(returns) if len(returns) > 1 else 0
    sharpe = avg_return / std_return if std_return > 0 else 0
    
    # 买入信号分析（只看预测上涨的）
    buy_signals = df[df['predicted_direction'] == 'up']
    if len(buy_signals) > 0:
        buy_wins = len(buy_signals[buy_signals['outcome'] == 'correct'])
        buy_win_rate = buy_wins / len(buy_signals)
        buy_avg_return = buy_signals['actual_return'].mean()
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
    # 时间窗口定义：天数和名称
    TIME_WINDOWS = [
        (30, '1个月'),
        (90, '3个月'),
        (180, '6个月')
    ]

    now = datetime.now()
    end_date_str = now.strftime('%Y-%m-%d')
    horizon_names = {1: '1天', 5: '5天', 20: '20天'}

    # 为每个时间窗口计算各周期的指标
    window_horizon_metrics = {}  # {窗口天数: {周期: 指标}}

    for days, _ in TIME_WINDOWS:
        start_date = now - timedelta(days=days)
        start_date_str = start_date.strftime('%Y-%m-%d')

        # 筛选时间范围内的已评估预测
        evaluated_predictions = [
            p for p in history['predictions']
            if p.get('outcome') is not None
            and start_date_str <= p.get('target_date', p.get('timestamp', '').split('T')[0]) <= end_date_str
        ]

        # 按周期分组
        horizon_predictions = {1: [], 5: [], 20: []}
        for pred in evaluated_predictions:
            h = pred.get('horizon', 20)
            if h in horizon_predictions:
                horizon_predictions[h].append(pred)

        # 计算各周期指标
        window_horizon_metrics[days] = {}
        for h, preds in horizon_predictions.items():
            window_horizon_metrics[days][h] = calculate_metrics(preds)

    # 使用3个月窗口的数据用于详细表现、板块和个股分析
    detail_days = 90
    start_date_detail = now - timedelta(days=detail_days)
    start_date_detail_str = start_date_detail.strftime('%Y-%m-%d')

    detail_predictions = [
        p for p in history['predictions']
        if p.get('outcome') is not None
        and start_date_detail_str <= p.get('target_date', p.get('timestamp', '').split('T')[0]) <= end_date_str
    ]

    # 按周期分组（详细）
    detail_horizon_predictions = {1: [], 5: [], 20: []}
    for pred in detail_predictions:
        h = pred.get('horizon', 20)
        if h in detail_horizon_predictions:
            detail_horizon_predictions[h].append(pred)

    # 计算各周期指标（详细）
    detail_horizon_metrics = {}
    for h, preds in detail_horizon_predictions.items():
        detail_horizon_metrics[h] = calculate_metrics(preds)

    # 按板块分组（仅20天，3个月窗口）
    sector_metrics = {}
    for pred in detail_horizon_predictions[20]:
        sector = pred.get('sector', 'unknown')
        if sector not in sector_metrics:
            sector_metrics[sector] = []
        sector_metrics[sector].append(pred)

    sector_results = {}
    for sector, preds in sector_metrics.items():
        sector_results[sector] = calculate_metrics(preds)

    # 按股票分组（仅20天，3个月窗口）
    stock_metrics = {}
    for pred in detail_horizon_predictions[20]:
        stock = pred.get('stock_code', 'unknown')
        if stock not in stock_metrics:
            stock_metrics[stock] = []
        stock_metrics[stock].append(pred)

    stock_results = {}
    for stock, preds in stock_metrics.items():
        stock_results[stock] = {
            **calculate_metrics(preds),
            'stock_name': preds[0].get('stock_name', stock),
            'sector': preds[0].get('sector', 'unknown')
        }

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

## 二、各周期详细表现（3个月窗口）

"""

    # 各周期详细表现
    for h in [1, 5, 20]:
        m = detail_horizon_metrics.get(h, {})
        if m:
            report += f"""### {horizon_names[h]}预测

| 指标 | 数值 |
|------|------|
| 总预测数 | {m.get('total_predictions', 0)} |
| 正确预测数 | {m.get('correct_predictions', 0)} |
| **准确率** | **{m.get('accuracy', 0):.2%}** |
| 平均收益率 | {m.get('avg_return', 0):.2%} |
| 收益率标准差 | {m.get('std_return', 0):.2%} |
| 夏普比率 | {m.get('sharpe_ratio', 0):.4f} |
| 买入胜率 | {m.get('buy_win_rate', 0):.2%} |

"""
        else:
            report += f"""### {horizon_names[h]}预测

| 指标 | 数值 |
|------|------|
| 总预测数 | 0 |
| 准确率 | - |

"""

    report += """---

## 三、板块表现（20天预测，3个月窗口）

"""

    # 板块表现表格
    if sector_results:
        report += "| 板块 | 类型 | 预测数 | 准确率 | 平均收益 | 买入胜率 |\n"
        report += "|------|------|--------|--------|----------|----------|\n"

        # 按准确率、平均收益排序
        sorted_sectors = sorted(
            sector_results.items(),
            key=lambda x: (x[1].get('accuracy', 0), x[1].get('avg_return', 0)),
            reverse=True
        )

        for sector, metrics in sorted_sectors:
            sector_name = get_sector_name(sector)
            sector_type = get_sector_type(sector)
            report += f"| {sector_name} | {sector_type} | {metrics.get('total_predictions', 0)} | {metrics.get('accuracy', 0):.2%} | {metrics.get('avg_return', 0):.2%} | {metrics.get('buy_win_rate', 0):.2%} |\n"
    else:
        report += "*暂无板块数据*\n"

    report += "\n---\n\n## 四、个股表现（20天预测，3个月窗口）\n\n"

    # 个股表现（按准确率、平均收益排序）
    if stock_results:
        report += "| 排名 | 股票代码 | 股票名称 | 板块 | 类型 | 预测数 | 准确率 | 平均收益 |\n"
        report += "|------|----------|----------|------|------|--------|--------|----------|\n"

        sorted_stocks = sorted(
            stock_results.items(),
            key=lambda x: (x[1].get('accuracy', 0), x[1].get('avg_return', 0)),
            reverse=True
        )

        for i, (stock, metrics) in enumerate(sorted_stocks, 1):
            sector = metrics.get('sector', 'unknown')
            sector_name = get_sector_name(sector)
            sector_type = get_sector_type(sector)
            report += f"| {i} | {stock} | {metrics.get('stock_name', stock)} | {sector_name} | {sector_type} | {metrics.get('total_predictions', 0)} | {metrics.get('accuracy', 0):.2%} | {metrics.get('avg_return', 0):.2%} |\n"
    else:
        report += "*暂无个股数据*\n"

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

**报告生成**: 港股智能分析系统 - 预测性能监控模块
"""

    return report


def send_email_report(report: str, subject: str) -> bool:
    """
    发送报告邮件
    
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
    
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
    smtp_user = os.environ.get("EMAIL_SENDER")
    smtp_pass = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL", "")
    
    if not smtp_user or not smtp_pass:
        print("❌ 缺少邮件配置环境变量")
        return False
    
    # 转换 Markdown 为 HTML
    html_content = markdown.markdown(report, extensions=['tables'])
    
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

    # 加载历史数据
    print("\n📂 加载预测历史数据...")
    history = load_prediction_history()
    print(f"   已加载 {len(history.get('predictions', []))} 条预测记录")

    if args.mode in ['evaluate', 'all']:
        # 评估各周期的预测
        total_stats = {'total': 0, 'evaluated': 0, 'correct': 0, 'wrong': 0}

        for h in horizons:
            print(f"\n📈 评估 {h} 天周期的预测...")
            history, stats = evaluate_predictions(history, h, args.force)

            # 显示统计
            print(f"   总预测: {stats['total']}")
            print(f"   已评估: {stats['evaluated']}")
            print(f"   正确: {stats['correct']}")
            print(f"   错误: {stats['wrong']}")
            if stats['evaluated'] > 0:
                print(f"   准确率: {stats['correct']/stats['evaluated']:.2%}")

            # 累计统计
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

    if args.mode in ['report', 'all']:
        # 生成报告
        print(f"\n📝 生成性能报告...")
        report = generate_monthly_report(history, args.month)

        # 保存报告（使用当前日期命名）
        report_date = datetime.now().strftime('%Y-%m-%d')
        report_path = os.path.join(REPORT_OUTPUT_DIR, f'performance_report_{report_date}.md')

        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   报告已保存到: {report_path}")

        # 发送邮件
        if not args.no_email:
            subject = f"[港股智能分析] 预测性能报告 - {report_date}"
            send_email_report(report, subject)
        else:
            print("   (--no-email) 跳过邮件发送")

    print("\n" + "=" * 60)
    print("✅ 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()