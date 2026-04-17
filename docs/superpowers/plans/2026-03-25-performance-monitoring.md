# 预测性能监控系统 - 实现计划

**创建日期**: 2026-03-25  
**设计文档**: [2026-03-25-performance-monitoring-design.md](../specs/2026-03-25-performance-monitoring-design.md)

---

## 概述

本计划将实现预测性能监控系统，用于追踪 ML 模型预测的实际表现，识别表现优异/欠佳的股票和板块，并生成月度报告。

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `ml_services/performance_monitor.py` | 创建 | 核心监控脚本 |
| `ml_services/ml_trading_model.py` | 修改 | 添加 `save_prediction_to_history()` 函数 |
| `data/prediction_history.json` | 创建 | 初始化空预测历史 |
| `.github/workflows/comprehensive-analysis.yml` | 修改 | 添加 git commit 步骤 |
| `.github/workflows/performance-monitor.yml` | 创建 | 月度工作流 |

---

## 任务分解

### 阶段 1: 数据层 (Infrastructure)

#### 任务 1.1: 创建预测历史数据文件

**文件**: `data/prediction_history.json`

**操作**: 创建空 JSON 文件

**代码**:
```json
{
  "predictions": [],
  "metadata": {
    "created_at": "2026-03-25T00:00:00",
    "last_updated": "2026-03-25T00:00:00",
    "version": "1.0"
  }
}
```

**验证命令**:
```bash
cat data/prediction_history.json | python3 -m json.tool > /dev/null && echo "JSON valid"
```

---

#### 任务 1.2: 在 ml_trading_model.py 中添加保存预测函数

**文件**: `ml_services/ml_trading_model.py`

**修改位置**: 在 `save_predictions_to_text()` 函数之后添加新函数（约第 150 行）

**新增函数**:
```python
def save_prediction_to_history(predictions, horizon=20, predict_date=None):
    """
    保存预测结果到历史记录文件，用于后续性能监控
    
    参数:
    - predictions: 预测结果列表
    - horizon: 预测周期（天）
    - predict_date: 预测日期字符串
    """
    import json
    from datetime import datetime
    
    # 历史文件路径
    history_file = 'data/prediction_history.json'
    
    # 加载现有历史
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {'predictions': [], 'metadata': {}}
    
    # 获取股票信息（板块）
    from config import STOCK_SECTOR_MAPPING
    
    # 当前时间戳
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S')
    date_str = predict_date if predict_date else now.strftime('%Y-%m-%d')
    
    # 为每个预测创建记录
    for pred in predictions:
        stock_code = pred.get('code', '')
        stock_name = pred.get('name', '')
        
        # 获取板块信息
        stock_info = STOCK_SECTOR_MAPPING.get(stock_code, {})
        sector = stock_info.get('sector', 'unknown') if isinstance(stock_info, dict) else 'unknown'
        
        # 创建预测记录
        record = {
            'prediction_id': f"{date_str}_{stock_code}",
            'timestamp': timestamp,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'sector': sector,
            'horizon': horizon,
            'predicted_direction': 'up' if pred.get('prediction', 0) == 1 else 'down',
            'prediction_probability': float(pred.get('probability', 0.5)),
            'confidence_level': 'high' if pred.get('probability', 0) > 0.6 else ('medium' if pred.get('probability', 0) > 0.5 else 'low'),
            'entry_price': float(pred.get('current_price', 0)),
            'model_type': 'catboost',
            'data_date': pred.get('data_date', date_str),
            'target_date': pred.get('target_date', ''),
            'outcome': None,
            'actual_return': None,
            'actual_direction': None,
            'evaluated_at': None
        }
        
        # 检查是否已存在相同 prediction_id 的记录
        existing_ids = [p['prediction_id'] for p in history['predictions']]
        if record['prediction_id'] not in existing_ids:
            history['predictions'].append(record)
    
    # 更新元数据
    history['metadata']['last_updated'] = timestamp
    history['metadata']['total_predictions'] = len(history['predictions'])
    
    # 保存历史
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存 {len(predictions)} 条预测记录到 {history_file}")
```

**修改位置 2**: 在 `main()` 函数的 predict 模式中调用新函数

在 `ml_trading_model.py` 约第 5210 行，保存预测结果之后添加调用：

```python
            # 保存20天预测结果到文本文件（便于后续提取和对比）
            if args.horizon == 20:
                save_predictions_to_text(pred_df_export, args.predict_date)
                # 保存预测到历史记录（用于性能监控）
                save_prediction_to_history(predictions, horizon=args.horizon, predict_date=args.predict_date)
```

**验证命令**:
```bash
python3 -m py_compile ml_services/ml_trading_model.py && echo "Syntax OK"
```

---

#### 任务 1.3: 修改 comprehensive-analysis.yml 添加 git commit 步骤

**文件**: `.github/workflows/comprehensive-analysis.yml`

**修改内容**: 在最后添加 git commit 步骤

**完整修改后的文件**:
```yaml
name: 综合分析邮件提醒

on:
  schedule:
    # 每天港股交易时段执行 (周一到周五，下午4:00 HK时间)
    - cron: '18 8 * * 1-5'
  # 可选：允许手动触发
  workflow_dispatch:

jobs:
  send-email:
    runs-on: ubuntu-latest
    env:
      TZ: Asia/Hong_Kong
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run comprehensive analysis script
        env:
          YAHOO_EMAIL: ${{ secrets.YAHOO_EMAIL }}
          YAHOO_APP_PASSWORD: ${{ secrets.YAHOO_APP_PASSWORD }}
          RECIPIENT_EMAIL: mall_cn@hotmail.com, wonglaitung@gmail.com
          YAHOO_SMTP: ${{ secrets.YAHOO_SMTP }}
          QWEN_API_KEY: ${{ secrets.QWEN_API_KEY }}
          QWEN_CHAT_URL: ${{ secrets.QWEN_CHAT_URL }}
          QWEN_CHAT_MODEL: ${{ secrets.QWEN_CHAT_MODEL }}
        run: |
          bash scripts/run_comprehensive_analysis.sh

      - name: Commit prediction history
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data/prediction_history.json
          git diff --quiet && git diff --staged --quiet || git commit -m "Update prediction history [skip ci]"
          git push
```

**验证命令**:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/comprehensive-analysis.yml'))" && echo "YAML valid"
```

---

### 阶段 2: 核心逻辑 (Core Logic)

#### 任务 2.1: 创建性能监控脚本

**文件**: `ml_services/performance_monitor.py`

**完整代码**:
```python
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


def evaluate_predictions(history: Dict, horizon: int = 20, force: bool = False) -> Dict:
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
        target_date = pred.get('target_date')
        if not target_date:
            stats['pending'] += 1
            continue
        
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        
        # 计算交易日
        entry_date = pred.get('data_date', pred.get('timestamp', '').split('T')[0])
        trading_days = count_trading_days(entry_date, now.strftime('%Y-%m-%d'), pred['stock_code'])
        
        # 如果交易日不足，跳过
        if trading_days < horizon:
            stats['pending'] += 1
            continue
        
        # 获取目标日期的收盘价
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


def generate_monthly_report(history: Dict, month: Optional[str] = None) -> str:
    """
    生成月度性能报告
    
    参数:
    - history: 预测历史数据
    - month: 月份 (YYYY-MM)，默认上个月
    
    返回:
    - Markdown 格式的报告
    """
    # 确定报告月份
    if month:
        report_month = month
    else:
        now = datetime.now()
        # 上个月
        first_of_this_month = now.replace(day=1)
        last_month = first_of_this_month - timedelta(days=1)
        report_month = last_month.strftime('%Y-%m')
    
    # 筛选该月份的预测
    month_predictions = [
        p for p in history['predictions']
        if p.get('timestamp', '').startswith(report_month)
    ]
    
    # 筛选已评估的预测
    evaluated_predictions = [
        p for p in month_predictions
        if p.get('outcome') is not None
    ]
    
    # 计算整体指标
    overall_metrics = calculate_metrics(evaluated_predictions)
    
    # 按板块分组
    sector_metrics = {}
    for pred in evaluated_predictions:
        sector = pred.get('sector', 'unknown')
        if sector not in sector_metrics:
            sector_metrics[sector] = []
        sector_metrics[sector].append(pred)
    
    sector_results = {}
    for sector, preds in sector_metrics.items():
        sector_results[sector] = calculate_metrics(preds)
    
    # 按股票分组
    stock_metrics = {}
    for pred in evaluated_predictions:
        stock = pred.get('stock_code', 'unknown')
        if stock not in stock_metrics:
            stock_metrics[stock] = []
        stock_metrics[stock].append(pred)
    
    stock_results = {}
    for stock, preds in stock_metrics.items():
        stock_results[stock] = {
            **calculate_metrics(preds),
            'stock_name': preds[0].get('stock_name', stock)
        }
    
    # 生成报告
    report = f"""# 预测性能月度报告

**报告月份**: {report_month}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、整体表现

| 指标 | 数值 |
|------|------|
| 总预测数 | {overall_metrics.get('total_predictions', 0)} |
| 正确预测数 | {overall_metrics.get('correct_predictions', 0)} |
| **准确率** | **{overall_metrics.get('accuracy', 0):.2%}** |
| 平均收益率 | {overall_metrics.get('avg_return', 0):.2%} |
| 收益率中位数 | {overall_metrics.get('median_return', 0):.2%} |
| 收益率标准差 | {overall_metrics.get('std_return', 0):.2%} |
| 夏普比率 | {overall_metrics.get('sharpe_ratio', 0):.4f} |

### 买入信号分析

| 指标 | 数值 |
|------|------|
| 买入信号数 | {overall_metrics.get('buy_signal_count', 0)} |
| 买入胜率 | {overall_metrics.get('buy_win_rate', 0):.2%} |
| 买入平均收益 | {overall_metrics.get('buy_avg_return', 0):.2%} |

---

## 二、板块表现

"""
    
    # 板块表现表格
    if sector_results:
        report += "| 板块 | 预测数 | 准确率 | 平均收益 | 买入胜率 |\n"
        report += "|------|--------|--------|----------|----------|\n"
        
        # 按准确率排序
        sorted_sectors = sorted(
            sector_results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        for sector, metrics in sorted_sectors:
            from config import STOCK_SECTOR_MAPPING
            sector_names = {v.get('sector'): v.get('sector_name', sector) for v in STOCK_SECTOR_MAPPING.values() if isinstance(v, dict)}
            sector_name = sector_names.get(sector, sector)
            report += f"| {sector_name} | {metrics.get('total_predictions', 0)} | {metrics.get('accuracy', 0):.2%} | {metrics.get('avg_return', 0):.2%} | {metrics.get('buy_win_rate', 0):.2%} |\n"
    else:
        report += "*暂无板块数据*\n"
    
    report += "\n---\n\n## 三、个股表现 TOP 10\n\n"
    
    # 个股表现 TOP 10（按准确率）
    if stock_results:
        report += "### 准确率 TOP 10\n\n"
        report += "| 排名 | 股票代码 | 股票名称 | 预测数 | 准确率 | 平均收益 |\n"
        report += "|------|----------|----------|--------|--------|----------|\n"
        
        sorted_stocks = sorted(
            stock_results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )[:10]
        
        for i, (stock, metrics) in enumerate(sorted_stocks, 1):
            report += f"| {i} | {stock} | {metrics.get('stock_name', stock)} | {metrics.get('total_predictions', 0)} | {metrics.get('accuracy', 0):.2%} | {metrics.get('avg_return', 0):.2%} |\n"
        
        report += "\n### 准确率 BOTTOM 10\n\n"
        report += "| 排名 | 股票代码 | 股票名称 | 预测数 | 准确率 | 平均收益 |\n"
        report += "|------|----------|----------|--------|--------|----------|\n"
        
        sorted_stocks_bottom = sorted(
            stock_results.items(),
            key=lambda x: x[1].get('accuracy', 0)
        )[:10]
        
        for i, (stock, metrics) in enumerate(sorted_stocks_bottom, 1):
            report += f"| {i} | {stock} | {metrics.get('stock_name', stock)} | {metrics.get('total_predictions', 0)} | {metrics.get('accuracy', 0):.2%} | {metrics.get('avg_return', 0):.2%} |\n"
    else:
        report += "*暂无个股数据*\n"
    
    report += f"""
---

## 四、风险提示

1. **历史表现不代表未来收益**
2. 模型准确率统计基于 {overall_metrics.get('total_predictions', 0)} 个样本，仅供参考
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
    
    smtp_server = os.environ.get("YAHOO_SMTP", "smtp.163.com")
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
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
    parser.add_argument('--horizon', type=int, default=20,
                       help='预测周期（默认20天）')
    parser.add_argument('--month', type=str, default=None,
                       help='报告月份 (YYYY-MM)，默认上个月')
    parser.add_argument('--no-email', action='store_true',
                       help='不发送邮件，仅生成报告')
    parser.add_argument('--force', action='store_true',
                       help='强制重新评估已评估的预测')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 预测性能监控系统")
    print("=" * 60)
    
    # 加载历史数据
    print("\n📂 加载预测历史数据...")
    history = load_prediction_history()
    print(f"   已加载 {len(history.get('predictions', []))} 条预测记录")
    
    if args.mode in ['evaluate', 'all']:
        # 评估已到期的预测
        print(f"\n📈 评估 {args.horizon} 天周期的预测...")
        history, stats = evaluate_predictions(history, args.horizon, args.force)
        print(f"   总预测: {stats['total']}")
        print(f"   已评估: {stats['evaluated']}")
        print(f"   正确: {stats['correct']}")
        print(f"   错误: {stats['wrong']}")
        if stats['evaluated'] > 0:
            print(f"   准确率: {stats['correct']/stats['evaluated']:.2%}")
    
    if args.mode in ['report', 'all']:
        # 生成月度报告
        print(f"\n📝 生成月度报告...")
        report = generate_monthly_report(history, args.month)
        
        # 保存报告
        report_month = args.month if args.month else (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
        report_path = os.path.join(REPORT_OUTPUT_DIR, f'performance_report_{report_month}.md')
        
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   报告已保存到: {report_path}")
        
        # 发送邮件
        if not args.no_email:
            subject = f"[港股智能分析] 预测性能月度报告 - {report_month}"
            send_email_report(report, subject)
        else:
            print("   (--no-email) 跳过邮件发送")
    
    print("\n" + "=" * 60)
    print("✅ 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

**验证命令**:
```bash
python3 -m py_compile ml_services/performance_monitor.py && echo "Syntax OK"
```

---

### 阶段 3: 工作流 (Workflow)

#### 任务 3.1: 创建月度性能监控工作流

**文件**: `.github/workflows/performance-monitor.yml`

**完整代码**:
```yaml
name: 预测性能月度报告

on:
  schedule:
    # 每月1号上午4:00 HK时间 (UTC 20:00 前一天)
    - cron: '0 20 28-31 * *'
  # 允许手动触发
  workflow_dispatch:

jobs:
  performance-report:
    runs-on: ubuntu-latest
    env:
      TZ: Asia/Hong_Kong
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install markdown

      - name: Run performance monitor
        env:
          YAHOO_EMAIL: ${{ secrets.YAHOO_EMAIL }}
          YAHOO_APP_PASSWORD: ${{ secrets.YAHOO_APP_PASSWORD }}
          RECIPIENT_EMAIL: mall_cn@hotmail.com, wonglaitung@gmail.com
          YAHOO_SMTP: ${{ secrets.YAHOO_SMTP }}
        run: |
          python3 ml_services/performance_monitor.py --mode all --horizon 20

      - name: Commit updated history
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data/prediction_history.json output/performance_report_*.md
          git diff --quiet && git diff --staged --quiet || git commit -m "Update performance monitoring data [skip ci]"
          git push
```

**注意**: cron 使用 `0 20 28-31 * *` 配合条件检查确保每月1号运行

**验证命令**:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/performance-monitor.yml'))" && echo "YAML valid"
```

---

## 验证清单

### 语法检查
```bash
# 检查 Python 语法
python3 -m py_compile ml_services/ml_trading_model.py
python3 -m py_compile ml_services/performance_monitor.py

# 检查 YAML 语法
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/comprehensive-analysis.yml'))"
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/performance-monitor.yml'))"

# 检查 JSON 语法
python3 -m json.tool data/prediction_history.json > /dev/null
```

### 功能测试
```bash
# 测试性能监控脚本（不发送邮件）
python3 ml_services/performance_monitor.py --mode evaluate --horizon 20 --no-email

# 测试报告生成（不发送邮件）
python3 ml_services/performance_monitor.py --mode report --no-email

# 完整测试
python3 ml_services/performance_monitor.py --mode all --horizon 20 --no-email
```

### 集成测试
```bash
# 模拟预测保存（需要先运行模型预测）
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost

# 验证预测历史文件更新
cat data/prediction_history.json | python3 -m json.tool | head -30
```

---

## 回滚方案

如果出现问题，可以通过以下方式回滚：

1. **回滚代码变更**:
   ```bash
   git checkout HEAD~1 -- ml_services/ml_trading_model.py
   git checkout HEAD~1 -- .github/workflows/comprehensive-analysis.yml
   ```

2. **删除新增文件**:
   ```bash
   rm ml_services/performance_monitor.py
   rm .github/workflows/performance-monitor.yml
   ```

3. **保留数据文件**:
   - `data/prediction_history.json` 可以保留，不影响其他功能

---

## 风险与注意事项

1. **API 限流**: yfinance 可能有 API 限流，大量获取价格时需要添加延迟
2. **非交易日**: 目标日期可能是非交易日，需要找最近的交易日
3. **数据缺失**: 部分股票可能无法获取价格数据，需要优雅处理
4. **时区问题**: 所有日期比较使用香港时区

---

## 后续优化

1. 添加趋势分析（月份对比）
2. 添加风险预警（连续预测错误）
3. 添加自定义报告周期
4. 集成到主界面（如果有 Web UI）